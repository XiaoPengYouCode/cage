from __future__ import annotations

import argparse
import json
from pathlib import Path

from fem_analysis.fjw_validation import validate_run_directory, write_validation_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate or compare a completed FJW optimization run.")
    parser.add_argument("--run-directory", default="runs/fjw_optimize")
    parser.add_argument("--golden-directory", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    report = validate_run_directory(
        Path(args.run_directory),
        golden_directory=None if args.golden_directory is None else Path(args.golden_directory),
    )
    output_path = write_validation_report(
        report,
        output_path=None if args.output is None else Path(args.output),
    )
    payload = report.as_jsonable()
    payload["report_path"] = str(output_path)
    print(json.dumps(payload, indent=2))
    if report.status == "fail":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
