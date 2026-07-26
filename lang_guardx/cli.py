"""LangGuardX CLI — verify, check, and validate from the command line."""

from __future__ import annotations

import argparse
import sys


def _get_version() -> str:
    try:
        from lang_guardx import __version__

        return __version__
    except ImportError:
        return "unknown"


def cmd_verify(args: argparse.Namespace) -> None:
    """Print version info."""
    print(f"LangGuardX {_get_version()} installed successfully.")


def cmd_check(args: argparse.Namespace) -> None:
    """Run Layer 1 detection on a text input."""
    from lang_guardx import LangGuardX

    guard = LangGuardX(args.config)
    text = args.text or sys.stdin.read().strip()
    if not text:
        print("No input provided.", file=sys.stderr)
        sys.exit(1)
    ctx = guard.protect(text)
    r = ctx.detection_result
    if r is None:
        print("No detection result returned.", file=sys.stderr)
        sys.exit(1)
    if r.blocked:
        print(f"BLOCKED  reason={r.reason}  detail={r.detail}  confidence={r.confidence:.3f}")
        sys.exit(1)
    else:
        print("PASS")
        sys.exit(0)


def cmd_validate(args: argparse.Namespace) -> None:
    """Run Layer 2 SQL policy validation on a SQL query."""
    from lang_guardx import LangGuardX

    guard = LangGuardX(args.config)
    sql = args.sql or sys.stdin.read().strip()
    if not sql:
        print("No SQL provided.", file=sys.stderr)
        sys.exit(1)
    verdict = guard.validate_sql(sql)
    print(f"{verdict.verdict.value}  violations={verdict.violations}")
    if verdict.safe_sql and verdict.safe_sql != sql:
        print(f"safe_sql: {verdict.safe_sql}")
    sys.exit(0 if verdict.verdict.value == "PASSED" else 1)


def cmd_scan(args: argparse.Namespace) -> None:
    """Run Layer 3 scanning on a JSON file of DB rows."""
    import json

    from lang_guardx import LangGuardX

    guard = LangGuardX(args.config)
    with open(args.file) as f:
        rows = json.load(f)
    sanitized, flags = guard.scan_results(rows)
    print(f"Scanned {len(rows)} rows, flagged {len(flags)}:")
    for f_ in flags:
        print(f"  reason={f_.reason}  original={f_.original[:100]!r}")
    if args.output:
        with open(args.output, "w") as f:
            json.dump(sanitized, f, indent=2)
        print(f"Sanitized output written to {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="langguardx", description="LangGuardX CLI")
    parser.add_argument("--config", "-c", help="Path to config file (YAML/JSON/TOML)", default=None)

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("verify", help="Print version info")

    check_p = sub.add_parser("check", help="Run detection on input text")
    check_p.add_argument("text", nargs="?", help="Text to check (reads stdin if omitted)")

    validate_p = sub.add_parser("validate", help="Run SQL policy validation")
    validate_p.add_argument("sql", nargs="?", help="SQL query to validate (reads stdin if omitted)")

    scan_p = sub.add_parser("scan", help="Scan DB results for indirect injection")
    scan_p.add_argument("file", help="JSON file containing list of row dicts")
    scan_p.add_argument("--output", "-o", help="Write sanitized rows to this file")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "verify":
        cmd_verify(args)
    elif args.command == "check":
        cmd_check(args)
    elif args.command == "validate":
        cmd_validate(args)
    elif args.command == "scan":
        cmd_scan(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
