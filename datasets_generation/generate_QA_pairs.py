#!/usr/bin/env python3
"""
AutoQAG-style QA generation: no UI; writes results to a local JSON file.
Calls the OpenAI Chat Completions API sequentially (one chunk per request).

Example:
  OPENAI_API_KEY=sk-... python generate_QA_pairs.py -i Data/testdata.txt -o qa.json --model gpt-4o-mini
"""
import argparse
import json
import os
import sys
import time
from typing import List, Optional, Tuple

from openai import OpenAI
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    CSVLoader,
    EverNoteLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredEmailLoader,
    UnstructuredEPubLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredODTLoader,
    UnstructuredPowerPointLoader,
    UnstructuredWordDocumentLoader,
)
LLM_TIMEOUT = 300

LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
    ".eml": (UnstructuredEmailLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PyMuPDFLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
}


def get_completion(client: OpenAI, prompt: str, model: str) -> Optional[str]:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content
    except Exception as e:
        err_str = str(e)
        print(f"[error] API call failed: {e}", file=sys.stderr)
        if "not found" in err_str.lower() and "model" in err_str.lower():
            print(
                "[hint] Model not found or unavailable. Check OPENAI_API_KEY, --model, and --base-url (if using a custom endpoint).",
                file=sys.stderr,
            )
        return None


def _record_chunk_qa(
    idx: int,
    resp: Optional[str],
    chunk_texts: List[str],
    qa_by_index: List[Optional[dict]],
    total: int,
    completed_count: List[int],
    t0: float,
    output_path: Optional[str],
) -> None:
    chunk_text = chunk_texts[idx]
    qa = _parse_qa_response(resp, chunk_text)
    if qa:
        qa_by_index[idx] = qa
    else:
        reason = "empty API response" if not resp else "parse failed (expected Q: and A: sections)"
        print(f"[warn] chunk {idx + 1}/{total}: {reason}", file=sys.stderr)
        if resp and os.environ.get("DEBUG_QA_PARSE"):
            preview = (resp[:400] + "...") if len(resp) > 400 else resp
            print(f"[debug] response preview:\n{preview}", file=sys.stderr)
    completed_count[0] += 1
    qa_pairs_so_far = [x for x in qa_by_index if x is not None]
    elapsed = time.perf_counter() - t0
    print(
        f"progress: {completed_count[0]}/{total} ({len(qa_pairs_so_far)} QA pairs, {elapsed:.1f}s)",
        flush=True,
    )
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(qa_pairs_so_far, f, ensure_ascii=False, indent=2)


def _get_chunk_text(chunk) -> str:
    if hasattr(chunk, "page_content"):
        return chunk.page_content
    return str(chunk)


def _build_qa_prompt(chunk_text: str) -> str:
    return f"""Based on the text below, generate one high-quality Q&A pair. Follow these guidelines:

1. Questions:
- Generate 2-4 differently phrased questions on the same topic (e.g. What is…? Can we say…? Please explain…? Can you give an example of…?).
- Cover the key information and main concepts in the text.

2. Answer:
- After listing all questions, you must start the answer with "A:" and give a clear, informative answer based directly on the given text, with coherent logic.

3. Format (must be followed exactly for parsing):
- First write "Q:" then list all questions on the same line or on separate lines (separate questions by spaces or newlines).
- Then write "A:" followed by the full answer.
- Example: Q: Question 1? Question 2? Then on a new line: A: Here is the answer…

4. Content:
- Stay closely on the topic of the text; do not add information not mentioned. If something cannot be determined, say "Cannot be determined from the given information."

Given text:
{chunk_text}

Generate the Q&A pair from the text above. You must include both Q: and A: sections, and the answer after A: must be complete.
"""


def _parse_qa_response(response: Optional[str], chunk_text: str) -> Optional[dict]:
    """Parse model output with Q:/A: sections into {"question", "answer", "chunk"} or None."""
    if not response:
        return None
    try:
        parts = response.split("A:", 1)
        if len(parts) == 2:
            question = parts[0].replace("Q:", "").strip()
            answer = parts[1].strip()
            return {"question": question, "answer": answer, "chunk": chunk_text}
    except Exception:
        pass
    return None


def load_single_document(file_path: str) -> List[Document]:
    ext = "." + file_path.rsplit(".", 1)[-1].lower()
    if ext not in LOADER_MAPPING:
        raise ValueError(f"unsupported file extension: {ext}")
    loader_class, loader_args = LOADER_MAPPING[ext]
    loader = loader_class(file_path, **loader_args)
    return loader.load()


def file_to_chunks(file_path: str) -> List[Document]:
    """Load one file and split it into text chunks."""
    documents = load_single_document(file_path)
    if not documents:
        return []
    # Balance chunk size vs. prompt overhead and model context window
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    return text_splitter.split_documents(documents)


def generate_qa_pairs(
    text_chunks: List[Document],
    client: OpenAI,
    model: str,
    output_path: Optional[str] = None,
) -> Tuple[List[dict], float, List[float]]:
    """
    Generate QA pairs via sequential Chat Completions calls.
    If output_path is set, write JSON after each chunk completes.
    Returns (qa_pairs, total_elapsed_seconds, per-chunk elapsed times).
    """
    total = len(text_chunks)
    if total == 0:
        return [], 0.0, []

    if not model.strip():
        raise ValueError("model name is required (use --model)")

    chunk_texts = [_get_chunk_text(c) for c in text_chunks]
    prompts = [_build_qa_prompt(ct) for ct in chunk_texts]
    qa_by_index: List[Optional[dict]] = [None] * total
    completed_count: List[int] = [0]
    t0 = time.perf_counter()

    chunk_times: List[float] = []
    for idx, prompt in enumerate(prompts):
        c0 = time.perf_counter()
        resp = get_completion(client, prompt, model)
        chunk_times.append(time.perf_counter() - c0)
        _record_chunk_qa(idx, resp, chunk_texts, qa_by_index, total, completed_count, t0, output_path)

    total_elapsed = time.perf_counter() - t0
    qa_pairs = [x for x in qa_by_index if x is not None]
    return qa_pairs, total_elapsed, chunk_times


def run(input_paths: List[str], output_path: str, base_url: str, model: str) -> None:
    """Generate QA pairs from files and save JSON (sequential Chat Completions)."""
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY") or ""
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=LLM_TIMEOUT)

    all_chunks: List[Document] = []
    for path in input_paths:
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            print(f"[error] file not found: {path}", file=sys.stderr)
            sys.exit(1)
        try:
            chunks = file_to_chunks(path)
            all_chunks.extend(chunks)
            print(f"loaded {path} -> {len(chunks)} chunk(s)")
        except Exception as e:
            print(f"[error] failed to load {path}: {e}", file=sys.stderr)
            sys.exit(1)

    if not all_chunks:
        print("[error] no text chunks produced; check input files.", file=sys.stderr)
        sys.exit(1)

    out_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([], f, ensure_ascii=False, indent=2)
    print(f"[info] sequential OpenAI Chat Completions (model={model!r}, base_url={base_url!r})")
    print(f"{len(all_chunks)} chunk(s); generating QA -> {out_path}")
    qa_pairs, total_elapsed, chunk_times = generate_qa_pairs(
        all_chunks, client=client, model=model, output_path=out_path
    )

    n = len(chunk_times)
    if n > 0:
        avg_per_chunk = total_elapsed / n
        print()
        print("========== timing ==========")
        print(f"chunks: {n}")
        print(f"total time: {total_elapsed:.2f} s")
        print(f"avg per chunk: {avg_per_chunk:.2f} s")
        if chunk_times:
            print(f"min chunk: {min(chunk_times):.2f} s | max chunk: {max(chunk_times):.2f} s")
        print("============================")
        print()

    print(f"done. {len(qa_pairs)} QA pair(s) saved to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Build QA pairs from documents and save JSON (OpenAI Chat Completions, sequential)."
    )
    parser.add_argument(
        "-i", "--input",
        nargs="+",
        required=True,
        help="Input file path(s), e.g. -i a.txt b.pdf",
    )
    parser.add_argument(
        "-o", "--output",
        default="qa_output.json",
        help="Output JSON path (default: qa_output.json)",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
        help="Chat Completions API base URL (default: https://api.openai.com/v1 or env LLM_BASE_URL)",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("LLM_MODEL", ""),
        help="Model name (or set env LLM_MODEL)",
    )
    args = parser.parse_args()
    if not (args.model or "").strip():
        parser.error("provide --model or set LLM_MODEL")
    run(
        input_paths=args.input,
        output_path=args.output,
        base_url=args.base_url,
        model=args.model.strip(),
    )


if __name__ == "__main__":
    main()
