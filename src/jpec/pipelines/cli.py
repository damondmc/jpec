"""The jpec command: one subcommand per pipeline step, then the tools."""

import argparse

from jpec.pipelines import check_sft
from jpec.pipelines.injection import injection_setup, sensitivity
from jpec.pipelines.search import search, search_setup, time_search

# in the order a run uses them; the tools after
COMMANDS = {
    "check-sft": check_sft,
    "search-setup": search_setup,
    "search": search,
    "injection-setup": injection_setup,
    "sensitivity": sensitivity,
    "time-search": time_search,
}


def main() -> None:
    """Parse the subcommand and its flags, and run it."""
    parser = argparse.ArgumentParser(
        prog="jpec", description=(__doc__ or "").splitlines()[0]
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    for name, module in COMMANDS.items():
        summary = (module.__doc__ or "").splitlines()[0]
        subparser = subparsers.add_parser(name, help=summary, description=summary)
        module.add_arguments(subparser)
        subparser.set_defaults(run=module.run)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
