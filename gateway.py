#!/usr/bin/env python3
"""
christman-gateway - Stage 2: pass-through + vault retrieval, logging only

The Christman AI Project / Luma Cognify AI
Originator: Everett Christman.  Custodian: Claude Opus 5.  Created 2026-08-20.

One door.  Claude Code reaches every downstream MCP server through this process
and by no other path, because .claude.json will list this and nothing else.
Enforcement is topology, not instruction.

Stage 1 proved the door holds:
  1. Aggregates every downstream server's tools under a "<server>__<tool>" name
     and forwards each call unchanged.
  2. Stamps an explicit vault_status on every single result.

Stage 2 adds the retrieval half and NOTHING ELSE:
  3. Every call builds a retrieval key and queries the vault on two channels,
     keyword and local embeddings, unioned.
  4. vault_status stops saying "not-consulted" and starts saying hit, no-match
     or ambiguous - as a stated value, with the note name and custody number.

There is still NO EARLY EXIT.  Every call is forwarded downstream exactly as it
was in Stage 1, whatever the vault said.  Stage 2 exists so the ledger records
what the exit WOULD have caught, and so Everett can read that log before any
call is ever answered from the vault instead of from the tool.  Turning the
exit on is Stage 3 and it is his call, not a session's.

Cardinal Rules in force: Standing Order 1 (no stubs), Rule 6 (fail loud - no
swallowed errors), Rule 15 (stdlib and the mcp package only, nothing paid).

stdout is the MCP protocol channel.  Nothing but protocol is ever written there.
Diagnostics go to stderr; the call ledger goes to LOG_PATH.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

import anyio
import mcp.types as types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server import Server
from mcp.server.stdio import stdio_server

from vault import VaultIndex, build_query, STATUS_UNAVAILABLE

GATEWAY_NAME = "christman-gateway"
GATEWAY_VERSION = "0.2.0-stage2"
HERE = Path(__file__).resolve().parent
CONFIG_PATH = Path(os.environ.get("CG_CONFIG", HERE / "gateway.config.json"))
LOG_PATH = Path(
    os.environ.get("CG_LOG", Path.home() / "Library/Logs/christman_gateway.jsonl")
)
SEP = "__"

# Stage 2 vault contract.  The vault IS wired in now, but the exit is off, so
# every result says that out loud too.  A vault that was consulted and found
# nothing, a vault that could not be reached, and a vault whose answer was not
# allowed to stop the call are three different facts and must never look alike.
EXIT_ENABLED = os.environ.get("CG_EXIT", "off").lower() in {"on", "1", "true"}
STAGE_NOTE = "stage 2: retrieval live, early exit OFF (observe-only)"

logging.basicConfig(
    stream=sys.stderr,
    level=os.environ.get("CG_LOGLEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [christman-gateway] %(message)s",
)
log = logging.getLogger(GATEWAY_NAME)


def ledger(**fields: Any) -> None:
    """Append one line to the call ledger.  Never raises into the call path."""
    fields["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(fields, default=str) + "\n")
    except OSError as exc:  # Rule 6 - say it, do not hide it.
        log.error("ledger write failed: %s", exc)


def load_config() -> dict[str, dict[str, Any]]:
    """Read the downstream roster.  A missing or malformed file is fatal."""
    if not CONFIG_PATH.exists():
        raise SystemExit(
            f"{GATEWAY_NAME}: no config at {CONFIG_PATH}. "
            "Create it before starting the gateway."
        )
    raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    servers = raw.get("downstream")
    if not isinstance(servers, dict) or not servers:
        raise SystemExit(
            f"{GATEWAY_NAME}: {CONFIG_PATH} has no non-empty 'downstream' object."
        )
    for name, spec in servers.items():
        if SEP in name:
            raise SystemExit(
                f"{GATEWAY_NAME}: server name {name!r} contains {SEP!r}, "
                "which is the tool-name separator. Rename it."
            )
        if not spec.get("command"):
            raise SystemExit(
                f"{GATEWAY_NAME}: downstream {name!r} has no 'command'. "
                "Stage 1 fronts stdio servers only."
            )
    return servers


class Downstream:
    """Live sessions to every downstream server, plus the tool routing table."""

    def __init__(self) -> None:
        self.sessions: dict[str, ClientSession] = {}
        self.tools: list[types.Tool] = []
        self.route: dict[str, tuple[str, str]] = {}  # exposed name -> (server, tool)
        self.failed: dict[str, str] = {}

    async def connect_all(
        self, servers: dict[str, dict[str, Any]], stack: AsyncExitStack
    ) -> None:
        for name, spec in servers.items():
            try:
                await self._connect_one(name, spec, stack)
            except Exception as exc:
                # A downstream that will not start is recorded and named at
                # startup rather than discovered silently mid-task (Rule 6).
                self.failed[name] = f"{type(exc).__name__}: {exc}"
                log.error("downstream %s failed to start: %s", name, exc)
                ledger(event="downstream_failed", server=name, error=str(exc))

    async def _connect_one(
        self, name: str, spec: dict[str, Any], stack: AsyncExitStack
    ) -> None:
        env = dict(os.environ)
        env.update(spec.get("env") or {})
        params = StdioServerParameters(
            command=spec["command"], args=list(spec.get("args") or []), env=env
        )
        read, write = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        listing = await session.list_tools()
        self.sessions[name] = session
        for tool in listing.tools:
            exposed = f"{name}{SEP}{tool.name}"
            self.route[exposed] = (name, tool.name)
            self.tools.append(
                types.Tool(
                    name=exposed,
                    description=tool.description,
                    inputSchema=tool.inputSchema,
                )
            )
        log.info("downstream %s up with %d tools", name, len(listing.tools))
        ledger(event="downstream_up", server=name, tool_count=len(listing.tools))


def vault_stamp(verdict: dict[str, Any]) -> types.TextContent:
    """Every result carries this. The absence of a vault answer is a value.

    The stamp is a full verdict, not a word: status, the numbers behind it, and
    the citations with custody numbers so any claim it carries can be walked
    back to the note that is answerable for it.
    """
    payload = {
        "vault_status": verdict.get("vault_status"),
        "vault_detail": verdict.get("vault_detail", ""),
        "vault_stage": STAGE_NOTE,
        "early_exit_taken": False,  # Stage 2 never exits. Stated, not implied.
    }
    for key in ("candidates", "near_misses"):
        hits = verdict.get(key) or []
        if hits:
            payload[key] = [
                {k: h[k] for k in ("cite", "score", "keyword", "embedding", "excerpt")}
                for h in hits
            ]
    return types.TextContent(type="text", text=json.dumps(payload, indent=2))


DOWN = Downstream()
VAULT: VaultIndex | None = None
VAULT_ERROR = ""


def consult_vault(tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Query the vault for this call. Never raises into the call path.

    A retrieval failure is reported as unavailable, never as no-match. Those
    two are the same sentence to a careless reader and opposite facts to a
    correct one - Mandate 2 and Mandate 5 in the same line of code.
    """
    if VAULT is None:
        return {
            "vault_status": STATUS_UNAVAILABLE,
            "vault_detail": VAULT_ERROR or "vault index not built",
        }
    try:
        return VAULT.query(build_query(tool, arguments))
    except Exception as exc:
        log.error("vault query failed: %s", exc)
        return {
            "vault_status": STATUS_UNAVAILABLE,
            "vault_detail": f"vault query raised {type(exc).__name__}: {exc}",
        }


app: Server = Server(GATEWAY_NAME, version=GATEWAY_VERSION)


@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return DOWN.tools


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[types.ContentBlock]:
    started = time.monotonic()
    target = DOWN.route.get(name)
    if target is None:
        known = ", ".join(sorted(DOWN.route)[:12]) or "(none)"
        ledger(event="unknown_tool", tool=name)
        # Raised, not returned as prose. The client sees a real error.
        raise ValueError(
            f"{GATEWAY_NAME}: no downstream owns tool {name!r}. Known: {known}"
        )

    server_name, tool_name = target
    session = DOWN.sessions.get(server_name)
    if session is None:
        reason = DOWN.failed.get(server_name, "never connected")
        ledger(event="downstream_unavailable", server=server_name, tool=tool_name)
        raise RuntimeError(
            f"{GATEWAY_NAME}: downstream {server_name!r} is not available ({reason})."
        )

    # The vault is consulted BEFORE the call goes downstream. In Stage 3 this
    # is where the early exit returns. In Stage 2 it only watches, so the order
    # is what gets proven now rather than bolted on later.
    vault_started = time.monotonic()
    verdict = consult_vault(name, arguments or {})
    vault_ms = round((time.monotonic() - vault_started) * 1000, 1)

    result = await session.call_tool(tool_name, arguments or {})
    elapsed_ms = round((time.monotonic() - started) * 1000, 1)
    top = (verdict.get("candidates") or verdict.get("near_misses") or [{}])[0]
    ledger(
        event="call",
        server=server_name,
        tool=tool_name,
        exposed=name,
        arg_keys=sorted((arguments or {}).keys()),
        is_error=bool(result.isError),
        elapsed_ms=elapsed_ms,
        vault_ms=vault_ms,
        vault_status=verdict.get("vault_status"),
        vault_detail=verdict.get("vault_detail"),
        # would_have_exited is the number this whole stage exists to produce.
        # Read it before turning CG_EXIT on.
        would_have_exited=verdict.get("vault_status") == "hit",
        top_cite=top.get("cite"),
        top_score=top.get("score"),
    )

    if result.isError:
        # Forward the downstream's own words. Do not soften and do not summarise.
        #
        # The vault verdict rides on the failure too. The first version of this
        # raised before stamping, so a failed downstream call came back with no
        # vault_status at all - an absence that looked like nothing rather than
        # like a stated value, which is the exact silent lie this gateway was
        # built to end. An error is still an answer, and it still has to say
        # what the vault knew when it was given.
        text = "\n".join(
            block.text for block in result.content if isinstance(block, types.TextContent)
        )
        raise RuntimeError(
            f"{server_name}.{tool_name} failed: {text or 'no detail'}\n"
            f"vault_status={verdict.get('vault_status')} "
            f"({verdict.get('vault_detail', '')})"
        )

    return list(result.content) + [vault_stamp(verdict)]


def build_vault_index() -> None:
    """Build the index at startup. A vault that will not index is named loudly
    and the gateway still serves - a broken index must degrade the stamp, never
    take the machine's hands away."""
    global VAULT, VAULT_ERROR
    started = time.monotonic()
    try:
        index = VaultIndex()
        index.build()
    except Exception as exc:
        VAULT_ERROR = f"{type(exc).__name__}: {exc}"
        log.error("vault index failed: %s", VAULT_ERROR)
        ledger(event="vault_index_failed", error=VAULT_ERROR)
        return
    VAULT = index
    elapsed = round(time.monotonic() - started, 1)
    if not index.embeddings_ready:
        log.warning(
            "vault index DEGRADED - keyword channel only: %s", index.embed_error
        )
    log.info(
        "vault indexed - %d notes, %d chunks, channels=%s, %ss",
        index.note_count,
        len(index.chunks),
        "keyword+embedding" if index.embeddings_ready else "keyword-only",
        elapsed,
    )
    ledger(
        event="vault_indexed",
        notes=index.note_count,
        chunks=len(index.chunks),
        embeddings_ready=index.embeddings_ready,
        embed_error=index.embed_error,
        seconds=elapsed,
    )


async def serve() -> None:
    servers = load_config()
    build_vault_index()
    if EXIT_ENABLED:
        # Stage 3 is not built. Refuse rather than pretend the flag does
        # something (Standing Order 1 - no stubs).
        raise SystemExit(
            f"{GATEWAY_NAME}: CG_EXIT is set but the early exit is Stage 3 and "
            "is not implemented. Unset CG_EXIT."
        )
    async with AsyncExitStack() as stack:
        await DOWN.connect_all(servers, stack)
        if not DOWN.sessions:
            raise SystemExit(
                f"{GATEWAY_NAME}: no downstream server started. "
                f"Failures: {DOWN.failed or '(none recorded)'}"
            )
        log.info(
            "gateway ready - %d server(s), %d tool(s), %d failed",
            len(DOWN.sessions),
            len(DOWN.tools),
            len(DOWN.failed),
        )
        ledger(
            event="gateway_ready",
            servers=sorted(DOWN.sessions),
            tool_count=len(DOWN.tools),
            failed=DOWN.failed,
        )
        async with stdio_server() as (read, write):
            await app.run(read, write, app.create_initialization_options())


def main() -> None:
    try:
        anyio.run(serve)
    except KeyboardInterrupt:
        log.info("gateway stopped by signal")


if __name__ == "__main__":
    main()
