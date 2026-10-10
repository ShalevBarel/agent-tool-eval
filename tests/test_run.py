from agent_eval.agent import run
from fake_client import FakeClient, response, text, tool_use


def use_fake_client(responses, monkeypatch):
    """Make the script use a fake client with these responses instead of the real API."""
    monkeypatch.setattr(run.anthropic, "Anthropic", lambda: FakeClient(responses))


def test_shows_the_calls_and_both_answers(monkeypatch, capsys):
    use_fake_client(
        [
            response(tool_use("check_schedule", {"order_ids": ["o1"]}, "toolu_a")),
            response(tool_use("submit_answer", {"solvable": True, "order_ids": ["o1"]}, "toolu_b")),
        ],
        monkeypatch,
    )
    run.main(["--task", "task-000"])
    printed = capsys.readouterr().out
    assert printed.startswith("task-000, easy, orders o1\n")
    assert 'turn  1  check_schedule {"order_ids": ["o1"]}' in printed
    assert "outcome  answered, after 2 turns" in printed
    assert "tokens   200 in, 20 out" in printed
    assert "agent    o1\n" in printed
    assert "exact    o1, last delivery at minute " in printed


def test_shows_a_run_without_an_answer(monkeypatch, capsys):
    use_fake_client([response(text("Hmm."), stop_reason="end_turn")], monkeypatch)
    run.main(["--task", "task-000"])
    printed = capsys.readouterr().out
    assert "outcome  end_turn, after 1 turns" in printed
    assert "agent    no answer" in printed
