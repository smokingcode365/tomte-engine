"""Safe-by-default stage prompt evaluator. Real execution requires --live."""
import argparse
import json
import os
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

if __package__:
    from .prompt_eval_support import (
        ROOT, STAGES, VERSION, canonical, digest, evaluate, load_cases, local_path,
        manifest, select_cases, summarize,
    )
else:
    from prompt_eval_support import (
        ROOT, STAGES, VERSION, canonical, digest, evaluate, load_cases, local_path,
        manifest, select_cases, summarize,
    )


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return number


def create_provider():
    # This branch is never entered by validation/default mode.
    from dotenv import load_dotenv
    from tomte.engine.providers.openai_provider import OpenAILLMProvider
    load_dotenv(ROOT / ".env", override=False)
    if not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY is required for --live")
    return OpenAILLMProvider(os.environ["OPENAI_API_KEY"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", action="append", type=Path, help="Repository-local JSON file or directory; repeatable")
    parser.add_argument("--stage", action="append", choices=STAGES)
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--partition", choices=("dev", "validation"))
    parser.add_argument("--limit", type=positive)
    parser.add_argument("--repeat", type=positive, default=1)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--prompt-set", type=Path, help="Optional JSON map of stage to alternate prompt string; current prompts remain unchanged")
    args = parser.parse_args(argv)
    try:
        paths = []
        for value in args.cases or [ROOT / "evals/prompts/cases"]:
            path = local_path(value)
            if not path.exists():
                raise ValueError("Missing case path")
            paths.extend(sorted(path.rglob("*.json")) if path.is_dir() else [path])
        cases, hashes = load_cases(paths)  # Validate all before selecting a slice.
        prompts = None
        if args.prompt_set:
            prompts = json.loads(local_path(args.prompt_set).read_text(encoding="utf-8"))
            if not isinstance(prompts, dict) or any(s not in STAGES or not isinstance(p, str) or not p for s, p in prompts.items()):
                raise ValueError("Invalid alternate prompt set")
        # Default live use is dev; validation must be explicitly selected.
        partition = args.partition or ("dev" if args.live else None)
        chosen = select_cases(cases, args.stage, args.case_id, partition, args.limit)
    except (ValueError, OSError, SyntaxError):
        # Pydantic errors can contain full inputs: never print their bodies.
        parser.exit(2, "Invalid evaluation files or selection; check the versioned case contract.\n")
    print(canonical({"mode": "live" if args.live else "validate", "evaluator_version": VERSION,
        "loaded_cases": len(cases), "selected_cases": len(chosen), "repeat": args.repeat,
        "case_file_hashes": hashes, "selected_cases_digest": digest(canonical([c.model_dump() for c in chosen]))}))
    if not args.live:
        return 0
    if not chosen:
        parser.exit(2, "No cases selected; no provider created.\n")
    try:
        provider = create_provider()
    except Exception:
        parser.exit(2, "Unable to initialize live provider; check private OPENAI_API_KEY/configuration.\n")
    run_manifest = manifest(chosen, hashes, args.repeat, partition, prompts)
    run_manifest["prompt_set_file_hash"] = (
        sha256(local_path(args.prompt_set).read_bytes()).hexdigest() if args.prompt_set else None)
    run_id = run_manifest["timestamp"].replace(":", "").replace("+", "_") + "-" + uuid4().hex[:8]
    output = local_path(ROOT / "log/prompt_evals" / run_id)
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.json").write_text(canonical(run_manifest) + "\n", encoding="utf-8")
    records = []
    with (output / "results.jsonl").open("w", encoding="utf-8") as stream:
        for case in chosen:
            for repetition in range(1, args.repeat + 1):
                record = evaluate(provider, case, repetition, prompts)
                records.append(record)
                stream.write(canonical(record) + "\n")
                stream.flush()
    summary = summarize(records)
    usage = {"models": sorted({m for r in records for m in r["usage"]["models"]}),
             "request_count": sum(r["usage"]["request_count"] for r in records),
             "retry_count": sum(r["retry_count"] for r in records)}
    for field in ("input_tokens", "output_tokens"):
        values = [r["usage"][field] for r in records]
        usage[field] = sum(v for v in values if v is not None)
        usage[field + "_complete"] = all(v is not None for v in values)
    for filename, value in (("summary.json", summary), ("usage.json", usage)):
        (output / filename).write_text(canonical(value) + "\n", encoding="utf-8")
    print(canonical({"output": output.relative_to(ROOT).as_posix(), "summary": summary}))
    return int(summary["failed_cases"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
