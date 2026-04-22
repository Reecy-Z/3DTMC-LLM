#!/usr/bin/env python3
"""
Polish descriptions in-place in one LMDB: reads each value, calls the Chat Completions API,
and writes the model output to `enriched_description` (configurable with --output-key).

Each record dict must have non-empty `description` and non-empty `smiles`. LLM output is
written to `enriched_description` unless --output-key is set.

deps: lmdb, openai
auth: OPENAI_API_KEY or LLM_API_KEY

Example:
  OPENAI_API_KEY=sk-... python enrich_description.py /path/to/tmc.lmdb \\
    --base-url https://api.openai.com/v1 --model gpt-4o-mini
"""
import argparse
import json
import os
import pickle
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import lmdb

from openai import OpenAI

LLM_TIMEOUT = 300
# Large enough for growth when rewriting values (adjust if LMDB was created with a bigger map).
LMDB_MAP_SIZE = 1 << 40


def _decode_with_format(raw: bytes) -> Tuple[Any, str]:
    try:
        return pickle.loads(raw), "pickle"
    except Exception:
        pass
    try:
        return json.loads(raw.decode("utf-8")), "json"
    except Exception:
        pass
    return raw.decode("utf-8", errors="replace"), "text"


def _encode_value(record: Any, fmt: str) -> bytes:
    if fmt == "pickle":
        return pickle.dumps(record, protocol=pickle.HIGHEST_PROTOCOL)
    if fmt == "json":
        return json.dumps(record, ensure_ascii=False).encode("utf-8")
    if fmt == "text":
        return str(record).encode("utf-8")
    raise ValueError(f"unknown encoding format: {fmt}")


def _get_str(record: Any, keys: List[str]) -> Optional[str]:
    if not isinstance(record, dict) and not hasattr(record, "get"):
        return None
    get = getattr(record, "get", None) or (
        lambda k: record.get(k) if isinstance(record, dict) else None
    )
    for k in keys:
        v = get(k) if get else (record.get(k) if isinstance(record, dict) else None)
        if v is not None and str(v).strip():
            return v if isinstance(v, str) else str(v)
    return None


def _get_smiles(record: Any) -> Optional[str]:
    return _get_str(record, ["smiles"])


def _get_description(record: Any) -> Optional[str]:
    """Input body for the LLM: only the `description` field."""
    return _get_str(record, ["description"])


def _prompt_inputs(record: Any) -> Tuple[Optional[str], Optional[str]]:
    """Return (smiles, description_text) for POLISH_PROMPT. Skip if either field is missing/empty."""
    smiles = _get_smiles(record)
    text = _get_description(record)
    if not text or not text.strip():
        return (None, None)
    if not smiles or not str(smiles).strip():
        return (None, None)
    return (str(smiles).strip(), text.strip())


def _open_lmdb_rw(path: str) -> lmdb.Environment:
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"LMDB path not found: {path}")
    subdir = os.path.isdir(path)
    kwargs = dict(
        map_size=LMDB_MAP_SIZE,
        max_dbs=0,
        lock=True,
        sync=True,
        readahead=False,
        meminit=False,
    )
    if os.path.isfile(path):
        kwargs["subdir"] = False
    else:
        kwargs["subdir"] = True
    return lmdb.open(path, **kwargs)


POLISH_PROMPT = """You are given a combined description of a transition metal complex (SMILES: {smiles}). It is very template-like and formulaic.

Your task: Rewrite/polish the entire description so that it reads more naturally and less like a template. Requirements:
1. Keep all factual content (metal, oxidation state, d-electron count, charge, geometry, ligands, coordinating atoms, pi ligands, etc.). Do not add or remove facts.
2. Vary sentence structure and wording; avoid repetitive patterns (e.g. "The ... is ...", "It has ...").
3. Merge or reflow the reasoning and geometry analysis into a coherent narrative where appropriate; do not lose the reasoning approach.
4. For coordinating atoms, the text may refer to the '->' and '<-' coordination bonds in the SMILES (atom before '->' or after '<-' is the donor).
5. Output only the polished description. No preamble, no "Here is the polished version:", no QA.

Original description:
{description}

Polished description (output only the text):"""


def polish_completion(
    client: OpenAI,
    prompt: str,
    model: str,
    temperature: float = 0.3,
) -> Optional[str]:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        content = (resp.choices[0].message.content or "").strip()
        return content or None
    except Exception as e:
        err_str = str(e)
        print(f"[error] API call failed: {e}", file=sys.stderr)
        if "not found" in err_str.lower() and "model" in err_str.lower():
            print(
                "[hint] Model not found or unavailable. Check OPENAI_API_KEY, --model, and --base-url.",
                file=sys.stderr,
            )
        return None


def run(
    lmdb_path: str,
    base_url: str,
    model: str,
    limit: Optional[int] = None,
    verbose: bool = False,
    output_key: str = "enriched_description",
) -> None:
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY") or ""
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=LLM_TIMEOUT)

    if not (model or "").strip():
        raise ValueError("model name is required (use --model)")

    env = _open_lmdb_rw(lmdb_path)

    pending: List[Tuple[bytes, str, Dict[str, Any]]] = []
    with env.begin(write=False) as txn:
        for key, raw in txn.cursor():
            obj, fmt = _decode_with_format(raw)
            if fmt == "text" or not isinstance(obj, dict):
                continue
            desc_text = _prompt_inputs(obj)[1]
            if not desc_text:
                continue
            pending.append((key, fmt, obj))

    if limit is not None and limit > 0:
        pending = pending[:limit]

    total = len(pending)
    if total == 0:
        print(
            "[error] no updatable dict records (need non-empty `description` and `smiles`).",
            file=sys.stderr,
        )
        sys.exit(1)

    if verbose:
        print(f"[info] updating {total} record(s) in-place", flush=True)

    t0 = time.perf_counter()
    done = 0
    for idx, (key, fmt, record) in enumerate(pending):
        smiles, desc_text = _prompt_inputs(record)
        if not desc_text:
            continue
        prompt = POLISH_PROMPT.format(
            smiles=smiles,
            description=desc_text,
        )
        resp = polish_completion(client, prompt, model)
        if not resp:
            if verbose:
                print(
                    f"[warn] skip put for key (empty API response): {key[:32]!r}...",
                    file=sys.stderr,
                )
            continue
        record[output_key] = resp
        try:
            payload = _encode_value(record, fmt)
        except Exception as e:
            print(f"[error] serialize failed for key {key!r}: {e}", file=sys.stderr)
            continue
        try:
            with env.begin(write=True) as txn:
                txn.put(key, payload)
            done += 1
        except Exception as e:
            print(f"[error] LMDB put failed for key {key!r}: {e}", file=sys.stderr)

        if verbose:
            elapsed = time.perf_counter() - t0
            print(f"progress: {idx + 1}/{total} (committed {done}, {elapsed:.1f}s)", flush=True)

    env.close()
    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"done. committed {done}/{total} updates in {elapsed:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Polish LMDB records in-place: write LLM output to enriched_description (or --output-key)."
    )
    parser.add_argument(
        "lmdb",
        help="Path to LMDB file or directory",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
        help="Chat Completions API base URL",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("LLM_MODEL", ""),
        help="Model name (or env LLM_MODEL)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Only process the first N eligible records (debug)",
    )
    parser.add_argument(
        "--output-key",
        default="enriched_description",
        metavar="FIELD",
        help="Record field to write the polished text (default: enriched_description)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print progress and summary (default: silent except errors)",
    )
    args = parser.parse_args()
    if not (args.model or "").strip():
        parser.error("provide --model or set LLM_MODEL")
    out_key = (args.output_key or "").strip()
    if not out_key:
        parser.error("--output-key must be non-empty")
    run(
        lmdb_path=args.lmdb,
        base_url=args.base_url,
        model=args.model.strip(),
        limit=args.limit,
        verbose=args.verbose,
        output_key=out_key,
    )


if __name__ == "__main__":
    main()
