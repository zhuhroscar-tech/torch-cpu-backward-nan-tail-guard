"""Command-line interface: run the from-scratch diagnosis of the seven
length-dependent NaN-gradient backward kernels against the currently
installed torch build, using the shared semantic-color design system."""
from __future__ import annotations

import argparse
import json
import sys

from .style import print_fields, resolve_style, section, status_headline


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="torch-cpu-backward-nan-tail-guard",
        description=(
            "Diagnose whether the currently installed torch build's CPU "
            "backward kernels for logit_backward/hardtanh/hardshrink/"
            "softshrink/elu/celu/hardswish return a DIFFERENT gradient at "
            "a NaN input element depending purely on tensor length versus "
            "the CPU's SIMD vector-block width (pytorch/pytorch#195075), "
            "and verify the guard functions are length-independent, on "
            "THIS host's actual installed torch version -- never trusts "
            "the upstream issue's reported version alone."
        ),
    )
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON instead of text")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI color even on a TTY")
    parser.add_argument("--version", action="store_true", help="print version and exit")
    args = parser.parse_args(argv)

    if args.version:
        from . import __version__

        print(f"torch-cpu-backward-nan-tail-guard {__version__}")
        return 0

    from .core import TorchUnavailableError, diagnose

    try:
        report = diagnose()
    except TorchUnavailableError as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}, indent=2))
        else:
            style = resolve_style(no_color_flag=args.no_color)
            print(status_headline(style, "fail", f"torch unavailable: {exc}"))
        return 2

    if args.json:
        print(json.dumps(report, indent=2))
        return 0 if report["guard_fully_effective"] else 1

    style = resolve_style(no_color_flag=args.no_color)
    print_fields(
        [
            ("torch version", report["torch_version"]),
            ("tracking issue", ", ".join(report["issue_urls"])),
        ]
    )

    if report["any_bug_present"]:
        print(status_headline(style, "warn", "length-dependent NaN-gradient divergence reproduced on this host's installed torch build"))
    else:
        print(status_headline(style, "info", "bug NOT reproduced on this host's installed torch build (fixed upstream)"))

    if report["guard_fully_effective"]:
        print(status_headline(style, "ok", "every safe_* guard is length-independent at every tested length/op"))
    else:
        print(status_headline(style, "fail", "at least one guard is still length-dependent"))

    section("per-case results (op x tensor length)")
    for c in report["cases"]:
        bug_flag = "LENGTH-DEP-BUG" if c["buggy_position_dependent"] else "consistent"
        guard_flag = "guard-ok" if c["guard_consistent"] else "GUARD-FAILED"
        print_fields(
            [
                (
                    f"{c['op']} n={c['length']}",
                    f"unguarded[0]={c['buggy_grad_first']!s:>6s} unguarded[-1]={c['buggy_grad_last']!s:>6s}  {bug_flag:15s}  {guard_flag}",
                )
            ]
        )

    return 0 if report["guard_fully_effective"] else 1


if __name__ == "__main__":
    sys.exit(main())
