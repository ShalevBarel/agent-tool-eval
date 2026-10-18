"""The agent-eval command. "run" calls the API, so it costs money.

    uv run agent-eval run configs/haiku-medium.toml --budget 2
    uv run agent-eval summary runs/haiku-medium.jsonl
    uv run agent-eval summary runs
    uv run agent-eval report runs

The API key is read from the ANTHROPIC_API_KEY environment variable.
"""

import argparse
from pathlib import Path

import anthropic

from agent_eval.agent.loop import REQUEST_TIMEOUT_SECONDS
from agent_eval.analysis.report import report
from agent_eval.analysis.tables import load_runs
from agent_eval.env.tasks import load_tasks
from agent_eval.experiment.configs import load_config
from agent_eval.experiment.run_log import load_records
from agent_eval.experiment.run_tasks import choose_tasks, run_tasks, show_summary


def main(argv: list[str] | None = None) -> None:
    """Run a configuration on the tasks, or summarize or report on run logs."""
    parser = argparse.ArgumentParser(prog="agent-eval", description="Evaluate a tool-using agent.")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="run a configuration on the tasks and log every run")
    run.add_argument("config", type=Path, help="a configuration file, such as configs/x.toml")
    run.add_argument(
        "--budget",
        type=float,
        required=True,
        help="stop once the log's runs cost this many dollars, earlier runs included",
    )
    run.add_argument("--repeats", type=int, default=1, help="runs of each task (default: 1)")
    run.add_argument("--sample", type=int, help="run on this many random tasks (default: all)")
    run.add_argument("--seed", type=int, default=0, help="seed of the sample (default: 0)")
    run.add_argument("--out", type=Path, help="the run log (default: runs/<config name>.jsonl)")
    run.add_argument("--workers", type=int, default=4, help="runs at a time (default: 4)")
    run.add_argument("--tasks-file", type=Path, default=Path("data/tasks.jsonl"))

    summary = commands.add_parser("summary", help="print the results of run logs")
    summary.add_argument("logs", type=Path, nargs="+", help="run logs, or folders of them")

    report_parser = commands.add_parser(
        "report", help="print every configuration's results with confidence intervals"
    )
    report_parser.add_argument("logs", type=Path, nargs="+", help="run logs, or folders of them")
    report_parser.add_argument(
        "--seed", type=int, default=0, help="seed of the resamples (default: 0)"
    )

    args = parser.parse_args(argv)
    if args.command == "run":
        run_command(parser, args)
    elif args.command == "report":
        try:
            print(report(load_runs(log_paths(args.logs)), seed=args.seed))
        except ValueError as error:
            parser.error(str(error))
    else:
        for path in log_paths(args.logs):
            print(f"{path}")
            show_summary(load_records(path))
            print()


def log_paths(paths: list[Path]) -> list[Path]:
    """Return the run logs named, with each folder replaced by the .jsonl files in it."""
    found = []
    for path in paths:
        if path.is_dir():
            found.extend(sorted(path.glob("*.jsonl")))
        else:
            found.append(path)
    return found


def run_command(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Check the arguments, then run the configuration and print a summary."""
    try:
        config = load_config(args.config)
    except ValueError as error:
        parser.error(str(error))
    tasks = load_tasks(args.tasks_file)
    if args.sample is not None and not 1 <= args.sample <= len(tasks):
        parser.error(f"--sample must be between 1 and {len(tasks)}")
    if args.repeats < 1 or args.workers < 1:
        parser.error("--repeats and --workers must be at least 1")
    tasks = choose_tasks(tasks, args.sample, args.seed)
    out = args.out or Path("runs") / f"{args.config.stem}.jsonl"

    # Many runs at once can hit the account's per-minute limits. On a "too many
    # requests" or "overloaded" error, the client waits and tries again.
    client = anthropic.Anthropic(max_retries=10, timeout=REQUEST_TIMEOUT_SECONDS)
    try:
        records = run_tasks(client, tasks, config, args.repeats, args.budget, out, args.workers)
    except ValueError as error:
        parser.error(str(error))
    print()
    show_summary(records)
