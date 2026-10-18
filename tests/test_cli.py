import copy
from pathlib import Path

import pytest

from agent_eval import cli
from agent_eval.agent.loop import REQUEST_TIMEOUT_SECONDS
from agent_eval.env.tasks import Task, save_tasks
from agent_eval.experiment.run_log import load_records
from agent_eval.solver import Solution
from fake_client import FakeClient, response, tool_use

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


@pytest.fixture
def tasks_file(city, tmp_path) -> Path:
    """Two tasks in the test city: task-000 has a solution, and task-001 has none."""
    no_solution = Task("task-001", "hard", copy.deepcopy(city), None)
    del city.orders["o4"]
    solvable = Task("task-000", "medium", city, Solution(["o1", "o2", "o3"], 90))
    path = tmp_path / "tasks.jsonl"
    save_tasks([no_solution, solvable], path)
    return path


def submit(solvable: bool, order_ids: list[str]):
    return response(tool_use("submit_answer", {"solvable": solvable, "order_ids": order_ids}))


def use_fake_client(responses, monkeypatch) -> FakeClient:
    """Make the command use a fake client with these responses instead of the real API."""
    client = FakeClient(responses)
    monkeypatch.setattr(cli.anthropic, "Anthropic", lambda **options: client)
    return client


def run(tasks_file: Path, *arguments: str) -> None:
    config = str(CONFIGS / "sonnet-low.toml")
    cli.main(["run", config, "--tasks-file", str(tasks_file), "--workers", "1", *arguments])


def test_run_logs_to_the_config_name_by_default(tasks_file, tmp_path, monkeypatch, capsys):
    client = use_fake_client([submit(True, ["o2", "o1", "o3"]), submit(False, [])], monkeypatch)
    monkeypatch.chdir(tmp_path)
    run(tasks_file, "--budget", "1")

    records = load_records(tmp_path / "runs" / "sonnet-low.jsonl")
    assert [record["task_id"] for record in records] == ["task-000", "task-001"]
    assert {(record["model"], record["effort"]) for record in records} == {
        ("claude-sonnet-5-5", "low")
    }
    assert client.messages.requests[0]["output_config"] == {"effort": "low"}
    assert "\nall          2        2   100%" in capsys.readouterr().out


def test_run_with_repeats_into_a_chosen_log(tasks_file, tmp_path, monkeypatch):
    use_fake_client([submit(True, ["o2", "o1", "o3"]), submit(False, [])] * 2, monkeypatch)
    out = tmp_path / "calibration" / "sonnet-low.jsonl"
    run(tasks_file, "--budget", "1", "--repeats", "2", "--out", str(out))
    assert len(load_records(out)) == 4


def test_run_a_sample(tasks_file, tmp_path, monkeypatch):
    use_fake_client([submit(False, [])], monkeypatch)
    out = tmp_path / "log.jsonl"
    run(tasks_file, "--budget", "1", "--sample", "1", "--seed", "0", "--out", str(out))
    assert len(load_records(out)) == 1


def test_run_gives_the_client_retries_and_a_timeout(tasks_file, tmp_path, monkeypatch):
    options = {}

    def make_client(**given):
        options.update(given)
        return FakeClient([submit(False, [])])

    monkeypatch.setattr(cli.anthropic, "Anthropic", make_client)
    run(tasks_file, "--budget", "1", "--sample", "1", "--out", str(tmp_path / "log.jsonl"))
    assert options == {"max_retries": 10, "timeout": REQUEST_TIMEOUT_SECONDS}


def test_the_budget_is_required(tasks_file, monkeypatch):
    client = use_fake_client([], monkeypatch)
    with pytest.raises(SystemExit):
        run(tasks_file)
    assert client.messages.requests == []


@pytest.mark.parametrize(
    "arguments", [["--sample", "3"], ["--sample", "0"], ["--repeats", "0"], ["--workers", "0"]]
)
def test_bad_numbers_stop_before_any_run(tasks_file, tmp_path, monkeypatch, arguments):
    client = use_fake_client([], monkeypatch)
    out = tmp_path / "log.jsonl"
    with pytest.raises(SystemExit):
        run(tasks_file, "--budget", "1", "--out", str(out), *arguments)
    assert client.messages.requests == []
    assert not out.exists()


def test_a_bad_config_stops_before_any_run(tasks_file, tmp_path, monkeypatch, capsys):
    client = use_fake_client([], monkeypatch)
    config = tmp_path / "typo.toml"
    config.write_text('model = "claude-haiku-5-5"\neffort = "hihg"\ntool_specs = "basic"\n')
    with pytest.raises(SystemExit):
        cli.main(["run", str(config), "--tasks-file", str(tasks_file), "--budget", "1"])
    assert client.messages.requests == []
    assert "effort must be one of" in capsys.readouterr().err


def test_a_log_of_another_config_stops_before_any_run(tasks_file, tmp_path, monkeypatch):
    use_fake_client([submit(True, ["o2", "o1", "o3"]), submit(False, [])], monkeypatch)
    out = tmp_path / "log.jsonl"
    run(tasks_file, "--budget", "1", "--out", str(out))
    client = use_fake_client([], monkeypatch)
    with pytest.raises(SystemExit):
        cli.main(
            [
                "run",
                str(CONFIGS / "haiku-high.toml"),
                "--tasks-file",
                str(tasks_file),
                "--budget",
                "1",
                "--out",
                str(out),
            ]
        )
    assert client.messages.requests == []


def test_summary_of_logs(tasks_file, tmp_path, monkeypatch, capsys):
    use_fake_client([submit(True, ["o2", "o1", "o3"]), submit(True, ["o1"])], monkeypatch)
    out = tmp_path / "log.jsonl"
    run(tasks_file, "--budget", "1", "--out", str(out))
    capsys.readouterr()
    cli.main(["summary", str(out), str(out)])
    printed = capsys.readouterr().out
    assert printed.count(f"{out}\n") == 2
    assert printed.count("\nall          2        1    50%") == 2


def test_summary_of_a_folder(tasks_file, tmp_path, monkeypatch, capsys):
    use_fake_client([submit(True, ["o2", "o1", "o3"]), submit(False, [])] * 2, monkeypatch)
    folder = tmp_path / "runs"
    run(tasks_file, "--budget", "1", "--out", str(folder / "b.jsonl"))
    run(tasks_file, "--budget", "1", "--out", str(folder / "a.jsonl"))
    (folder / "notes.txt").write_text("not a log", encoding="utf-8")
    capsys.readouterr()
    cli.main(["summary", str(folder)])
    printed = capsys.readouterr().out
    assert printed.index("a.jsonl") < printed.index("b.jsonl")
    assert "notes.txt" not in printed
    assert printed.count("\nall          2        2   100%") == 2


def test_report_of_a_folder(tasks_file, tmp_path, monkeypatch, capsys):
    folder = tmp_path / "runs"
    use_fake_client([submit(True, ["o2", "o1", "o3"]), submit(False, [])], monkeypatch)
    run(tasks_file, "--budget", "1", "--out", str(folder / "good.jsonl"))
    use_fake_client([submit(True, ["o1"]), submit(False, [])], monkeypatch)
    run(tasks_file, "--budget", "1", "--out", str(folder / "bad.jsonl"))
    capsys.readouterr()
    cli.main(["report", str(folder)])
    lines = capsys.readouterr().out.splitlines()
    assert lines[1].split()[:3] == ["good", "2", "100.0%"]
    assert lines[2].split()[:3] == ["bad", "2", "50.0%"]


def test_report_refuses_logs_of_different_tasks(tasks_file, tmp_path, monkeypatch, capsys):
    folder = tmp_path / "runs"
    use_fake_client([submit(True, ["o2", "o1", "o3"]), submit(False, [])], monkeypatch)
    run(tasks_file, "--budget", "1", "--out", str(folder / "good.jsonl"))
    use_fake_client([submit(False, [])], monkeypatch)
    run(tasks_file, "--budget", "1", "--sample", "1", "--out", str(folder / "bad.jsonl"))
    with pytest.raises(SystemExit):
        cli.main(["report", str(folder)])
    assert "ran different tasks" in capsys.readouterr().err
