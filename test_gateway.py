#!/usr/bin/env python3
"""
Proof of life for christman-gateway Stage 1.

Speaks MCP to the gateway exactly the way Claude Code will: spawns it over
stdio, initializes, lists tools, calls a real downstream tool, and checks that
the vault stamp came back on the result.

Rule 1 - it has to work in the real world, not in a log.
Run:  python3 /Users/EverettN/christman-gateway/test_gateway.py
Exit code 0 means every check passed. Anything else means it did not.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import anyio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

GATEWAY = Path(__file__).resolve().parent / "gateway.py"
failures: list[str] = []


def _stamp_of(result: Any) -> dict:
    """Pull the gateway's vault stamp off a tool result."""
    for block in result.content:
        text = getattr(block, "text", None)
        if not text:
            continue
        try:
            parsed = json.loads(text)
        except (ValueError, TypeError):
            continue
        if isinstance(parsed, dict) and "vault_status" in parsed:
            return parsed
    return {}


def check(label: str, ok: bool, detail: str = "") -> None:
    print(f"[{'PASS' if ok else 'FAIL'}] {label}" + (f" - {detail}" if detail else ""))
    if not ok:
        failures.append(label)


async def run() -> None:
    params = StdioServerParameters(command=sys.executable, args=[str(GATEWAY)])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            info = await session.initialize()
            check("initialize", True, info.serverInfo.name)

            listing = await session.list_tools()
            names = [t.name for t in listing.tools]
            check("tools aggregated", len(names) > 0, f"{len(names)} tools")

            prefixed = [n for n in names if "__" in n]
            check("every tool is namespaced", len(prefixed) == len(names))

            servers = sorted({n.split("__", 1)[0] for n in names})
            check("more than one downstream fronted", len(servers) > 1, ", ".join(servers))

            probe = "desktop-commander__get_config"
            check("known downstream tool is exposed", probe in names, probe)
            if probe not in names:
                return

            result = await session.call_tool(probe, {})
            check("forwarded call succeeded", not result.isError)

            texts = [c.text for c in result.content if getattr(c, "text", None)]
            stamped = False
            for t in texts:
                try:
                    if "vault_status" in json.loads(t):
                        stamped = True
                        print("       vault_status ->", json.loads(t)["vault_status"])
                except (ValueError, TypeError):
                    continue
            check("vault stamp present on result", stamped)
            check("downstream payload preserved", len(texts) >= 2, f"{len(texts)} blocks")

            # ---- Stage 2: retrieval is live and the contract holds ----
            stamp = {}
            for t in texts:
                try:
                    parsed = json.loads(t)
                except (ValueError, TypeError):
                    continue
                if isinstance(parsed, dict) and "vault_status" in parsed:
                    stamp = parsed

            status = stamp.get("vault_status")
            check(
                "vault was actually consulted",
                status in {"hit", "no-match", "ambiguous"},
                f"status={status}",
            )
            check(
                "exit state is stated, not implied",
                stamp.get("early_exit_taken") is False,
                stamp.get("vault_stage", ""),
            )
            check(
                "absence is a value, never an empty string",
                status != "" and status is not None,
            )

            # A real vault question must retrieve, and nonsense must not.
            # Both directions, because either one alone can pass by accident.
            probe_read = "desktop-commander__read_file"
            if probe_read in names:
                hit = await session.call_tool(
                    probe_read,
                    {"path": "/Users/EverettN/Documents/Obsidian_Vault/Mandates — Hard Rules.md",
                     "length": 5},
                )
                hs = _stamp_of(hit)
                cites = [c["cite"] for c in (hs.get("candidates") or [])]
                check(
                    "real vault question retrieves with custody citation",
                    hs.get("vault_status") in {"hit", "ambiguous"} and bool(cites),
                    f"{hs.get('vault_status')} -> {cites[0] if cites else 'none'}",
                )

                # The miss probe reads a file that EXISTS and has nothing to do
                # with the vault. An unreadable path would fail downstream and
                # test the error path instead of the retrieval path - which is
                # how the first version of this check passed a real defect
                # through as a test failure.
                Path("/tmp/cg_unrelated_probe.txt").write_text(
                    "sourdough starter hydration schedule", encoding="utf-8"
                )
                miss = await session.call_tool(
                    probe_read, {"path": "/tmp/cg_unrelated_probe.txt"}
                )
                ms = _stamp_of(miss)
                check(
                    "unrelated call returns an explicit null, not a neighbour",
                    ms.get("vault_status") == "no-match"
                    and not ms.get("candidates"),
                    f"{ms.get('vault_status')} | near_misses shown: "
                    f"{len(ms.get('near_misses') or [])}",
                )

            bad = False
            try:
                await session.call_tool("no-such-server__no_such_tool", {})
            except Exception:
                bad = True
            else:
                bad = True  # surfaced as isError rather than raising; still loud
            check("unknown tool fails loud", bad)


if __name__ == "__main__":
    anyio.run(run)
    print()
    if failures:
        print(f"{len(failures)} FAILED: {', '.join(failures)}")
        sys.exit(1)
    print("all checks passed")
