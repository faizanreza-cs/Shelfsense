import json
from types import SimpleNamespace

import pytest

from shelfsense.agent import provider
from shelfsense.agent.core import Agent, system_prompt
from shelfsense.agent.provider import MockProvider, ToolCall, Turn
from shelfsense.agent.tools import execute, tool_definitions
from shelfsense.evals import run_evals


@pytest.mark.parametrize(
    ("question", "tool"),
    [
        ("What do reviewers say about P0015?", "product_insights"),
        ("Which products are similar to P0015?", "similar_products"),
        ("What should we recommend to shopper C000123?", "recommend_for_shopper"),
        ("Which brand has the best review sentiment?", "brand_scorecard"),
        ("Is there any drift this month?", "monitoring_report"),
        ("Which recommender has the best recall?", "recommender_metrics"),
        ("How good is the aspect model?", "aspect_model_metrics"),
        ("Which products are bought together?", "run_saved_query"),
        ("Using SQL, how many shoppers are there?", "run_sql"),
    ],
)
def test_mock_agent_picks_the_right_tool(question, tool):
    r = Agent(MockProvider()).ask(question)
    assert r.tools_used == [tool]
    assert not r.steps[0].is_error and r.answer


def test_tool_errors_go_back_to_the_model_not_up():
    out, err = execute("run_sql", {"sql": "SELECT * FROM orders"})
    assert err and "shopper ids" in json.loads(out)["error"]
    out, err = execute("no_such_tool", {})
    assert err
    out, err = execute("product_insights", {"item": "Protein"})  # matches 30 products
    assert err and "use an item_id" in json.loads(out)["error"]
    out, err = execute("recommend_for_shopper", {"user_id": "C1; DROP TABLE orders"})
    assert err


def test_cut_off_tool_arguments_go_back_as_an_error(monkeypatch):
    monkeypatch.setenv("SHELFSENSE_MODEL", "m")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    prov = provider.OpenAIProvider()
    cut = SimpleNamespace(name="product_insights", arguments='{"item": "P00')
    msg = SimpleNamespace(content=None, tool_calls=[SimpleNamespace(id="c0", function=cut)])
    resp = SimpleNamespace(choices=[SimpleNamespace(message=msg)])
    completions = SimpleNamespace(create=lambda **kw: resp)
    prov.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    turn = prov.complete("sys", [{"role": "user", "content": "x"}], tool_definitions())
    out, is_error = execute(turn.tool_calls[0].name, turn.tool_calls[0].input)
    assert is_error and "not valid JSON" in json.loads(out)["error"]


def test_tool_schemas_are_complete():
    for t in tool_definitions():
        assert t["description"] and t["input_schema"]["type"] == "object"


def test_rogue_model_cant_run_unsafe_sql():
    class Rogue:
        name, model = "rogue", "rogue"

        def __init__(self):
            self.n = 0

        def complete(self, system, messages, tools):
            self.n += 1
            if self.n == 1:
                call = ToolCall("t1", "run_sql", {"sql": "DROP TABLE orders"})
                block = {"type": "tool_use", "id": "t1", "name": "run_sql", "input": call.input}
                return Turn("", [call], "tool_use", [block])
            return Turn("done", [], "end_turn", [{"type": "text", "text": "done"}])

    r = Agent(Rogue()).ask("drop it")
    assert r.steps[0].is_error and "SELECT" in r.steps[0].output["error"]


def test_providers_translate_tool_calls(monkeypatch):
    monkeypatch.setenv("SHELFSENSE_MODEL", "deployment-1")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", "https://example.openai.azure.com")
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    call = SimpleNamespace(
        id="c0", function=SimpleNamespace(name="brand_scorecard", arguments="{}")
    )
    msg = SimpleNamespace(content=None, tool_calls=[call])
    sent = {}

    def create(**kw):
        sent.update(kw)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    for cls in (provider.OpenAIProvider, provider.AzureOpenAIProvider):
        prov = cls()
        prov.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        turn = prov.complete("sys", [{"role": "user", "content": "brands?"}], tool_definitions())
        assert turn.tool_calls[0].name == "brand_scorecard" and sent["model"] == "deployment-1"
    a = provider.AnthropicProvider()
    resp = SimpleNamespace(
        content=[SimpleNamespace(type="tool_use", id="t1", name="brand_scorecard", input={})],
        stop_reason="tool_use", usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )  # fmt: skip
    a.client = SimpleNamespace(messages=SimpleNamespace(create=lambda **kw: resp))
    turn = a.complete("sys", [{"role": "user", "content": "brands?"}], tool_definitions())
    assert turn.tool_calls[0].name == "brand_scorecard" and turn.usage["input_tokens"] == 10


def test_system_prompt_names_the_loaded_data(built):
    assert f"{built['data']['reviews']:,} reviews" in system_prompt()


def test_golden_eval_passes_in_mock_mode():
    rows = run_evals.run("mock")
    summary = run_evals.summarise(rows, "mock")
    assert summary["pass_rate"] == 1.0, [r.id for r in rows if not (r.tool_ok and r.answer_ok)]
