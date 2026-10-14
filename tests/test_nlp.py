import json
from types import SimpleNamespace

import pandas as pd
import pytest

from shelfsense import nlp
from shelfsense.agent import provider


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("No bloating at all.", {"digestion": "positive"}),
        ("The seal was broken when it arrived.", {"packaging": "negative"}),
        (
            "It tastes great but it gives me bloating.",
            {"taste": "positive", "digestion": "negative"},
        ),
        ("Didn't notice any difference.", {"effectiveness": "negative"}),
        ("Not worth the money.", {"value": "negative"}),
        ("I take it every morning with breakfast.", {}),
    ],
)
def test_keyword_labeller(text, expected):
    got = nlp.keyword_labels(text)
    assert {a: v for a, v in got.items() if v != "none"} == expected


def test_student_learns_and_handles_a_single_class():
    texts = ["tastes great", "tastes awful", "tastes great again", "awful taste really"]
    taste = ["positive", "negative", "positive", "negative"]
    others = dict.fromkeys(nlp.ASPECTS[1:], "none")
    labels = pd.DataFrame({"review_id": list("abcd"), "taste": taste, **others})
    m = nlp.AspectModel().fit(texts, labels)
    pred = m.predict(["tastes great", "awful"])
    assert list(pred["taste"]) == ["positive", "negative"]
    assert set(pred["packaging"]) == {"none"}  # only one class seen: predicted as is


def test_f1_is_one_against_itself():
    labels = pd.DataFrame(
        {"review_id": ["a", "b"], **{a: ["positive", "none"] for a in nlp.ASPECTS}}
    )
    assert nlp.f1_by_aspect(labels, labels)["mean"] == 1.0


def test_llm_labeller_batches_and_drops_unknown_ids():
    calls = []

    class Fake:
        name = "fake"

        def structured(self, system, user, schema):
            calls.append(user)
            sent = json.loads(user)
            got = [{"review_id": r["review_id"], **dict.fromkeys(nlp.ASPECTS, "positive")}
                   for r in sent]  # fmt: skip
            got.append({"review_id": "made-up", **{a: "negative" for a in nlp.ASPECTS}})
            return schema.model_validate({"reviews": got})

    reviews = pd.DataFrame({"review_id": [f"R{i}" for i in range(45)], "text": ["ok"] * 45})
    out = nlp.label_with_llm(reviews, Fake(), batch_size=20)
    assert len(calls) == 3 and len(out) == 45 and "made-up" not in set(out["review_id"])
    with pytest.raises(ValueError):
        nlp.label_with_llm(reviews, provider.MockProvider())


def test_providers_request_structured_output(monkeypatch):
    monkeypatch.setenv("SHELFSENSE_MODEL", "model-under-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    batch = nlp.LabelBatch(reviews=[])
    sent = {}

    def parse(**kw):
        sent.update(kw)
        return SimpleNamespace(parsed_output=batch, stop_reason="end_turn")

    a = provider.AnthropicProvider()
    a.client = SimpleNamespace(messages=SimpleNamespace(parse=parse))
    assert a.structured("sys", "[]", nlp.LabelBatch) is batch
    assert sent["output_format"] is nlp.LabelBatch and sent["model"] == "model-under-test"
    cut = SimpleNamespace(parsed_output=None, stop_reason="max_tokens")
    a.client = SimpleNamespace(messages=SimpleNamespace(parse=lambda **kw: cut))
    with pytest.raises(ValueError):
        a.structured("sys", "[]", nlp.LabelBatch)

    o = provider.OpenAIProvider()
    msg = SimpleNamespace(parsed=batch, refusal=None)
    o.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        parse=lambda **kw: SimpleNamespace(choices=[SimpleNamespace(message=msg)]))))  # fmt: skip
    assert o.structured("sys", "[]", nlp.LabelBatch) is batch


def test_aspect_model_quality_on_the_store(built):
    a = built["aspects"]
    assert a["student_f1"]["mean"] > 0.85
    curve = [p["f1"] for p in a["perfect_labels_curve"]]
    assert curve == sorted(curve) and curve[-1] > a["student_f1"]["mean"]
    assert min(a["quality_correlation"].values()) > 0.9
    rc = a["rating_check"]
    assert rc["mean_stars_with_a_negative_aspect"] < rc["mean_stars_positive_only"]
