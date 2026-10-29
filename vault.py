#!/usr/bin/env python3
"""
christman-gateway - vault retrieval (Stage 2)

The Christman AI Project / Luma Cognify AI
Originator: Everett Christman.  Custodian: Claude Opus 5.  Created 2026-08-20.

Two channels, unioned, because each one misses precisely what the other catches:

  keyword   - exact identifiers, filenames, custody numbers, rule numbers.
              This is the channel that lies. Grep is a liar (Mandate 2), so it
              is never allowed to declare absence on its own.
  embedding - paraphrase and meaning, via Ollama on this machine. Nothing
              leaves the house and nothing is rented (Cardinal Rule 15).

A miss returns "no-match" as a stated value. Never an empty string, never the
nearest neighbour dressed up as an answer. That is the whole reason the middle
band exists: a near match is exactly where Grover's hands back noise shaped
like an answer, so the band is reported as ambiguous and the caller sees the
conflict instead of a confident wrong sentence carrying a custody stamp.

Stage 2 has NO early exit. It scores, stamps and logs what it would have
caught. Stage 3 turns the exit on, once the log says it is safe to.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

VAULT_PATH = Path(
    os.environ.get("CG_VAULT", "/Users/EverettN/Documents/Obsidian_Vault")
)
CACHE_PATH = Path(
    os.environ.get("CG_VAULT_CACHE", Path.home() / ".cache/christman_gateway/vault_index.json")
)
OLLAMA_URL = os.environ.get("CG_OLLAMA", "http://127.0.0.1:11434")
EMBED_MODEL = os.environ.get("CG_EMBED_MODEL", "nomic-embed-text")

# Score bands. Deliberately conservative at this stage: the exit is off, so the
# only cost of a tight HIGH band is a quieter log, while a loose one would
# teach us the wrong threshold before Stage 3 ever runs.
BAND_HIGH = float(os.environ.get("CG_BAND_HIGH", "0.72"))
BAND_LOW = float(os.environ.get("CG_BAND_LOW", "0.42"))
TOP_K = int(os.environ.get("CG_TOP_K", "4"))
# Batch and timeout were measured, not guessed. A batch of 32 chunks against
# nomic-embed-text on this machine exceeded a 20s timeout, and the embed path
# correctly refused to invent vectors and reported the whole index degraded.
# Smaller batches, longer ceiling.
EMBED_TIMEOUT = float(os.environ.get("CG_EMBED_TIMEOUT", "120"))
EMBED_BATCH = int(os.environ.get("CG_EMBED_BATCH", "8"))

STATUS_HIT = "hit"
STATUS_NONE = "no-match"
STATUS_AMBIGUOUS = "ambiguous"
STATUS_UNAVAILABLE = "unavailable"

_WORD = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_\-.]*")
_CUSTODY = re.compile(r"^custody:\s*(\S+)", re.MULTILINE)
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "it", "that", "this",
    "for", "on", "with", "as", "at", "by", "be", "are", "was", "were", "from",
    "not", "but", "if", "then", "than", "so", "do", "does", "did", "you", "your",
    "we", "our", "i", "he", "she", "they", "them", "his", "her", "its", "have",
    "has", "had", "will", "would", "can", "could", "should", "there", "here",
    "what", "when", "which", "who", "how", "why", "all", "any", "one", "no",
}


def tokenize(text: str) -> list[str]:
    return [t.lower() for t in _WORD.findall(text) if t.lower() not in _STOP and len(t) > 1]


@dataclass
class Chunk:
    """One heading section of one note. The unit of both retrieval and citation."""

    note: str          # path relative to the vault root
    heading: str
    custody: str       # custody number, or "unassigned" when the note carries none
    text: str
    tokens: Counter = field(default_factory=Counter)
    embedding: list[float] | None = None

    @property
    def cite(self) -> str:
        where = f"{self.note}#{self.heading}" if self.heading else self.note
        return f"{where} [{self.custody}]"


def split_note(path: Path, root: Path) -> list[Chunk]:
    """Split a note on its markdown headings. Frontmatter rides with the first chunk."""
    raw = path.read_text(encoding="utf-8", errors="replace")
    m = _CUSTODY.search(raw)
    custody = m.group(1) if m else "unassigned"
    rel = str(path.relative_to(root))

    chunks: list[Chunk] = []
    heading = ""
    buf: list[str] = []

    def flush() -> None:
        body = "\n".join(buf).strip()
        if len(body) < 40:  # too small to mean anything on its own
            return
        chunks.append(
            Chunk(note=rel, heading=heading, custody=custody,
                  text=body, tokens=Counter(tokenize(heading + " " + body)))
        )

    for line in raw.splitlines():
        if line.startswith("#") and line.lstrip("#").startswith(" "):
            flush()
            heading = line.lstrip("#").strip()
            buf = []
        else:
            buf.append(line)
    flush()
    return chunks


def _post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    req = urllib.request.Request(
        f"{OLLAMA_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=EMBED_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def embed(texts: list[str]) -> list[list[float]] | None:
    """Embed via local Ollama. Returns None - never a fake vector - if unavailable."""
    if not texts:
        return []
    try:
        out = _post("/api/embed", {"model": EMBED_MODEL, "input": texts})
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None
    vecs = out.get("embeddings")
    if not isinstance(vecs, list) or len(vecs) != len(texts):
        return None
    return vecs


def cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return num / (na * nb)


class VaultIndex:
    """The vault, chunked, with both channels built over it."""

    def __init__(self, root: Path = VAULT_PATH) -> None:
        self.root = root
        self.chunks: list[Chunk] = []
        self.df: Counter = Counter()
        self.embeddings_ready = False
        self.note_count = 0
        self.built_at = ""
        self.embed_error = ""

    # ---------------------------------------------------------------- building

    def notes(self) -> list[Path]:
        """Every real note file.

        is_file() is not defensive padding. The vault contains
        Lived_Truth_and_The_Oath.md, which is a DIRECTORY carrying a .md
        extension - a known open item that is Everett's to rename, not a
        session's. Without this filter the index dies on it (IsADirectoryError).
        """
        return sorted(p for p in self.root.rglob("*.md") if p.is_file())

    def fingerprint(self) -> str:
        h = hashlib.sha256()
        for p in self.notes():
            st = p.stat()
            h.update(f"{p}:{st.st_mtime_ns}:{st.st_size}".encode())
        return h.hexdigest()

    def build(self, use_cache: bool = True) -> None:
        if not self.root.is_dir():
            raise FileNotFoundError(f"vault not found at {self.root}")

        notes = self.notes()
        self.note_count = len(notes)
        for p in notes:
            self.chunks.extend(split_note(p, self.root))
        for c in self.chunks:
            self.df.update(set(c.tokens))

        fp = self.fingerprint()
        if use_cache and self._load_cache(fp):
            self.embeddings_ready = True
        else:
            self._build_embeddings()
            if self.embeddings_ready:
                self._save_cache(fp)
        self.built_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    def _build_embeddings(self) -> None:
        batch = EMBED_BATCH
        for i in range(0, len(self.chunks), batch):
            group = self.chunks[i:i + batch]
            vecs = embed([f"{c.heading}\n{c.text[:2000]}" for c in group])
            if vecs is None:
                self.embeddings_ready = False
                self.embed_error = (
                    f"ollama at {OLLAMA_URL} did not return embeddings for "
                    f"model {EMBED_MODEL!r}"
                )
                return
            for c, v in zip(group, vecs):
                c.embedding = v
        self.embeddings_ready = True

    def _load_cache(self, fp: str) -> bool:
        try:
            data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return False
        if data.get("fingerprint") != fp or data.get("model") != EMBED_MODEL:
            return False
        vecs = data.get("embeddings") or []
        if len(vecs) != len(self.chunks):
            return False
        for c, v in zip(self.chunks, vecs):
            c.embedding = v
        return True

    def _save_cache(self, fp: str) -> None:
        try:
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            CACHE_PATH.write_text(
                json.dumps({
                    "fingerprint": fp,
                    "model": EMBED_MODEL,
                    "embeddings": [c.embedding for c in self.chunks],
                }),
                encoding="utf-8",
            )
        except OSError:
            pass  # a cache miss costs time, never correctness

    # --------------------------------------------------------------- retrieval

    def _keyword_scores(self, query: str) -> dict[int, float]:
        """Absolute query coverage, NOT a corpus-relative rank.

        The first version of this normalised every score by the best score in
        the corpus, which meant the top chunk always scored exactly 1.0 - for
        a real question and for nonsense alike. Asking this index for the
        airspeed velocity of an unladen swallow returned a doctrine note at the
        same confidence as asking it for the vault path. That is precisely the
        failure the middle band exists to prevent, reintroduced one layer
        lower, and it is Grover's handing back noise shaped like an answer.

        Coverage fixes it: the score is the share of the query's own idf mass
        that this chunk actually accounts for. A chunk matching every
        distinctive term in the query scores near 1.0. A chunk matching one
        incidental word scores near 0 no matter how thin the competition is.
        """
        q = tokenize(query)
        if not q:
            return {}
        n = max(len(self.chunks), 1)

        idf = {t: math.log(1 + n / (1 + self.df.get(t, 0))) for t in set(q)}
        total = sum(idf.values())
        if total <= 0:
            return {}

        scores: dict[int, float] = {}
        for i, c in enumerate(self.chunks):
            s = 0.0
            for term, w in idf.items():
                tf = c.tokens.get(term, 0)
                if not tf:
                    continue
                # tf saturates fast: repetition is weak evidence, presence is
                # the signal. Caps any single term's contribution at its idf.
                s += w * min(1.0, (1 + math.log(tf)) / 3.0)
            if s > 0:
                scores[i] = min(1.0, s / total)
        return scores

    def _embed_scores(self, query: str) -> dict[int, float]:
        if not self.embeddings_ready:
            return {}
        qv = embed([query])
        if not qv:
            return {}
        q = qv[0]
        return {
            i: cosine(q, c.embedding)
            for i, c in enumerate(self.chunks)
            if c.embedding
        }

    def query(self, text: str) -> dict[str, Any]:
        """Score both channels, union them, band the result. No exit taken here."""
        if not self.chunks:
            return {
                "vault_status": STATUS_UNAVAILABLE,
                "vault_detail": f"no chunks indexed from {self.root}",
                "candidates": [],
            }

        if not tokenize(text):
            # No subject was asked about. Say that, rather than letting an
            # empty query drift into whatever the index happens to rank first.
            return {
                "vault_status": STATUS_NONE,
                "vault_detail": "call carried no searchable subject (no arguments)",
                "candidates": [],
                "near_misses": [],
            }

        kw = self._keyword_scores(text)
        em = self._embed_scores(text)

        # Union, not intersection. One channel's miss can never produce an
        # absence on its own - that is Mandate 2 made mechanical.
        combined: dict[int, float] = {}
        for i in set(kw) | set(em):
            k, e = kw.get(i, 0.0), em.get(i, 0.0)
            combined[i] = max(k, e) * 0.7 + min(k, e) * 0.3

        ranked = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)[:TOP_K]
        candidates = [
            {
                "cite": self.chunks[i].cite,
                "note": self.chunks[i].note,
                "heading": self.chunks[i].heading,
                "custody": self.chunks[i].custody,
                "score": round(s, 4),
                "keyword": round(kw.get(i, 0.0), 4),
                "embedding": round(em.get(i, 0.0), 4),
                "excerpt": self.chunks[i].text[:400],
            }
            for i, s in ranked
        ]

        top = candidates[0]["score"] if candidates else 0.0
        if top >= BAND_HIGH:
            status = STATUS_HIT
        elif top < BAND_LOW:
            status = STATUS_NONE
        else:
            status = STATUS_AMBIGUOUS

        detail = (
            f"top={top:.4f} bands=[{BAND_LOW},{BAND_HIGH}] "
            f"channels={'keyword+embedding' if self.embeddings_ready else 'keyword-only'}"
        )
        if not self.embeddings_ready:
            # Stated, because keyword-only absence is exactly the lie Mandate 2
            # is about. The caller must be able to see that half the retrieval
            # was not running.
            detail += f" DEGRADED: {self.embed_error or 'embeddings unavailable'}"

        return {
            "vault_status": status,
            "vault_detail": detail,
            "candidates": candidates if status != STATUS_NONE else [],
            "near_misses": candidates if status == STATUS_NONE else [],
        }


def build_query(tool: str, arguments: dict[str, Any]) -> str:
    """Retrieval key from the call's ARGUMENTS, deliberately not its tool name.

    The tool name was in the key in the first version and it poisoned the well.
    This vault has notes about Desktop Commander coming online, so every single
    desktop-commander call matched those notes on the tool name alone and came
    back as a middle-band near miss - a permanent floor of self-reference that
    had nothing to do with what was being asked. Harmless while the exit is
    off; in Stage 3 it would fire the hard stop on a coincidence of naming.

    The arguments carry the subject. A call with no arguments has no subject to
    look up, and the honest answer to that is no-match with a reason, not the
    nearest note about the tool doing the calling.
    """
    parts: list[str] = []

    def walk(v: Any, depth: int = 0) -> Iterable[str]:
        if depth > 3:
            return
        if isinstance(v, str):
            yield v
        elif isinstance(v, dict):
            for k, sub in v.items():
                yield str(k)
                yield from walk(sub, depth + 1)
        elif isinstance(v, (list, tuple)):
            for sub in v:
                yield from walk(sub, depth + 1)

    for chunk in walk(arguments or {}):
        parts.append(chunk)
    return " ".join(parts)[:4000]
