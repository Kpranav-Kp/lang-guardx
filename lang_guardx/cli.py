"""LangGuardX CLI — verify, check, validate, scan, and inspect from the command line."""

from __future__ import annotations

import argparse
import json
import sys


def _get_version() -> str:
    try:
        from lang_guardx import __version__

        return __version__
    except ImportError:
        return "unknown"


def _out(text: str, *, bold: bool = False, color: str | None = None) -> None:
    """Print with optional ANSI colour."""
    codes = {"red": "31", "green": "32", "yellow": "33", "cyan": "36"}
    if color and sys.stdout.isatty():
        c = codes.get(color, "0")
        b = "1;" if bold else ""
        text = f"\033[{b}{c}m{text}\033[0m"
    print(text)


def _json_or_text(data: dict, flag: bool) -> str:
    if flag:
        return json.dumps(data, indent=2, default=str)
    return data.get("text", "")


# ── Subcommands ────────────────────────────────────────────────────────


def cmd_verify(args: argparse.Namespace | None = None) -> None:
    """Print version and environment info."""
    info = {
        "version": _get_version(),
        "python": sys.version.split()[0],
        "platform": sys.platform,
    }
    _json = getattr(args, "json", False) if args else False
    _verbose = getattr(args, "verbose", False) if args else False

    if _json:
        print(json.dumps(info, indent=2))
        return

    _out(f"LangGuardX {info['version']}", bold=True, color="cyan")
    _out(f"  Python {info['python']} on {info['platform']}")
    if _verbose:
        _out("  dependencies: see pyproject.toml")


def cmd_config(args: argparse.Namespace) -> None:
    """Show the resolved configuration."""
    from lang_guardx import LangGuardX

    guard = LangGuardX(args.config)
    cfg = guard.config.model_dump(mode="json")
    if args.json:
        print(json.dumps(cfg, indent=2, default=str))
        return

    _out("Resolved configuration:", bold=True)
    _out(json.dumps(cfg, indent=2, default=str), color="cyan")


def cmd_check(args: argparse.Namespace) -> None:
    """Run Layer 1 detection on a text input."""
    from lang_guardx import LangGuardX

    guard = LangGuardX(args.config)
    text = args.text or sys.stdin.read().strip()
    if not text:
        _out("No input provided.", color="red")
        sys.exit(1)

    ctx = guard.protect(text)
    r = ctx.detection_result
    if r is None:
        _out("No detection result returned.", color="red")
        sys.exit(1)

    if args.json:
        print(
            json.dumps(
                {
                    "blocked": r.blocked,
                    "reason": r.reason,
                    "detail": r.detail,
                    "confidence": r.confidence,
                },
                indent=2,
            )
        )
    elif r.blocked:
        _out(
            f"BLOCKED  reason={r.reason}  detail={r.detail}  confidence={r.confidence:.3f}",
            bold=True,
            color="red",
        )
    else:
        _out("PASS", bold=True, color="green")

    sys.exit(1 if r.blocked else 0)


def cmd_validate(args: argparse.Namespace) -> None:
    """Run Layer 2 SQL policy validation on a SQL query."""
    from lang_guardx import LangGuardX

    guard = LangGuardX(args.config)
    sql = args.sql or sys.stdin.read().strip()
    if not sql:
        _out("No SQL provided.", color="red")
        sys.exit(1)

    verdict = guard.validate_sql(sql)

    if args.json:
        print(
            json.dumps(
                {
                    "verdict": verdict.verdict.value,
                    "original_sql": verdict.original_sql,
                    "safe_sql": verdict.safe_sql,
                    "violations": verdict.violations,
                },
                indent=2,
            )
        )
    else:
        if verdict.verdict.value == "BLOCKED":
            _out(f"BLOCKED  violations={verdict.violations}", bold=True, color="red")
        elif verdict.verdict.value == "REWRITTEN":
            _out(f"REWRITTEN  violations={verdict.violations}", bold=True, color="yellow")
            _out(f"  safe_sql: {verdict.safe_sql}", color="cyan")
        else:
            _out("PASSED", bold=True, color="green")

    sys.exit(0 if verdict.verdict.value == "PASSED" else 1)


def cmd_scan(args: argparse.Namespace) -> None:
    """Run Layer 3 scanning on a JSON file of DB rows."""
    from lang_guardx import LangGuardX

    guard = LangGuardX(args.config)

    try:
        with open(args.file) as f:
            rows = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        _out(f"Error reading {args.file}: {exc}", color="red")
        sys.exit(1)

    sanitized, flags = guard.scan_results(rows)
    flagged = [{"reason": f.reason, "original": str(f.original)[:200]} for f in flags]

    if args.json:
        print(
            json.dumps(
                {
                    "total_rows": len(rows),
                    "flagged_count": len(flags),
                    "flagged": flagged,
                },
                indent=2,
            )
        )
        return

    _out(f"Scanned {len(rows)} row(s), flagged {len(flags)}.", bold=True)
    for f_ in flagged:
        _out(f"  [{f_['reason']}] {f_['original']}", color="red")
    if not flagged:
        _out("  (none)", color="green")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(sanitized, f, indent=2)
        _out(f"Sanitized output written to {args.output}", color="cyan")


def cmd_rules(args: argparse.Namespace) -> None:
    """List registered policy rules."""
    from lang_guardx import LangGuardX

    guard = LangGuardX(args.config)
    rules = guard.engine.list_rules()

    if args.json:
        print(json.dumps([{"name": getattr(r, "name", type(r).__name__), "priority": getattr(r, "priority", 99)} for r in rules], indent=2))
        return

    if not rules:
        _out("No rules registered.", color="yellow")
        return

    _out(f"Policy rules ({len(rules)}):", bold=True)
    for r in rules:
        n = getattr(r, "name", type(r).__name__)
        p = getattr(r, "priority", 99)
        _out(f"  [{p:02d}] {n}", color="cyan")


def cmd_layers(args: argparse.Namespace) -> None:
    """List registered detection layers."""
    from lang_guardx import LangGuardX

    guard = LangGuardX(args.config)
    layers = guard.detector.list_layers()

    if args.json:
        print(json.dumps([{"name": getattr(lyr, "name", type(lyr).__name__), "priority": getattr(lyr, "priority", 99)} for lyr in sorted(layers, key=lambda x: getattr(x, "priority", 99))], indent=2))
        return

    if not layers:
        _out("No layers registered.", color="yellow")
        return

    _out(f"Detection layers ({len(layers)}):", bold=True)
    for lyr in sorted(layers, key=lambda x: getattr(x, "priority", 99)):
        n = getattr(lyr, "name", type(lyr).__name__)
        p = getattr(lyr, "priority", 99)
        _out(f"  [{p:02d}] {n}", color="cyan")


def _cmd_verify_entry() -> None:
    """No-arg entry point for the ``langguardx-verify`` script."""
    cmd_verify(None)


# ── Parser ─────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="langguardx",
        description="LangGuardX CLI — multi-layer security for LLM SQL agents",
    )
    parser.add_argument("--config", "-c", help="Path to config file (YAML/JSON/TOML)", default=None)
    parser.add_argument("--json", action="store_true", help="JSON output (machine-readable)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("verify", help="Print version info")
    sub.add_parser("config", help="Show resolved configuration")

    check_p = sub.add_parser("check", help="Run Layer 1 detection on input text")
    check_p.add_argument("text", nargs="?", help="Text to check (reads stdin if omitted)")

    validate_p = sub.add_parser("validate", help="Run Layer 2 SQL policy validation")
    validate_p.add_argument("sql", nargs="?", help="SQL query to validate (reads stdin if omitted)")

    scan_p = sub.add_parser("scan", help="Run Layer 3 scanning on DB result JSON")
    scan_p.add_argument("file", help="JSON file containing list of row dicts")
    scan_p.add_argument("--output", "-o", help="Write sanitized rows to this file")

    sub.add_parser("rules", help="List registered policy rules")
    sub.add_parser("layers", help="List registered detection layers")

    return parser


# ── Entry point ────────────────────────────────────────────────────────


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    commands = {
        "verify": cmd_verify,
        "config": cmd_config,
        "check": cmd_check,
        "validate": cmd_validate,
        "scan": cmd_scan,
        "rules": cmd_rules,
        "layers": cmd_layers,
    }

    cmd = commands.get(args.command)
    if cmd:
        cmd(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
