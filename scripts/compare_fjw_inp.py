from __future__ import annotations

import argparse
import json
from pathlib import Path

from fem_analysis.fjw_validation import compare_text_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare generated FJW Abaqus input text with a golden file.")
    parser.add_argument("actual")
    parser.add_argument("expected")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    check = compare_text_files(Path(args.actual), Path(args.expected))
    payload = {
        "status": check.status,
        "check": {
            "name": check.name,
            "status": check.status,
            "message": check.message,
            "metadata": check.metadata,
        },
    }
    text = json.dumps(payload, indent=2)
    if args.output is None:
        print(text)
    else:
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    if check.status == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
