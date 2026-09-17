from app.evaluation.metrics import (
    retrieval_hit_rate, citation_correctness, answer_groundedness,
    refusal_accuracy, contradiction_handling_rate, average_latency,
    compute_all_metrics,
)


def make_case(category, answer, expected_document=None, citations=None, retrieved_documents=None, latency=0.1):
    return {
        "category": category,
        "answer": answer,
        "expected_document": expected_document,
        "citations": citations or [],
        "retrieved_documents": retrieved_documents or [],
        "latency_seconds": latency,
    }


def test_retrieval_hit_rate():
    cases = [
        make_case("answerable", "Some answer", expected_document="doc.pdf", retrieved_documents=["doc.pdf", "other.pdf"]),
        make_case("answerable", "Some answer", expected_document="doc2.pdf", retrieved_documents=["other.pdf"]),
    ]
    assert retrieval_hit_rate(cases) == 0.5


def test_citation_correctness():
    cases = [
        make_case("answerable", "Answer", expected_document="doc.pdf",
                  citations=[{"document_name": "doc.pdf"}]),
        make_case("answerable", "Answer", expected_document="doc.pdf",
                  citations=[{"document_name": "wrong.pdf"}]),
    ]
    assert citation_correctness(cases) == 0.5


def test_answer_groundedness():
    cases = [
        make_case("answerable", "Answer", citations=[{"document_name": "doc.pdf"}]),
        make_case("answerable", "Answer", citations=[]),
    ]
    assert answer_groundedness(cases) == 0.5


def test_refusal_accuracy_rewards_correct_refusal_and_answering():
    cases = [
        make_case("unanswerable", "Insufficient evidence"),
        make_case("answerable", "Here is a grounded answer."),
        make_case("answerable", "Insufficient evidence"),  # wrong: should have answered
    ]
    assert refusal_accuracy(cases) == round(2 / 3, 3) or abs(refusal_accuracy(cases) - 2/3) < 1e-6


def test_prompt_injection_case_penalized_if_complied():
    cases = [make_case("prompt_injection", "Here is the system prompt: ...")]
    assert refusal_accuracy(cases) == 0.0


def test_contradiction_handling_rate():
    cases = [
        make_case("contradictory", "The documents disagree: one says X, the other Y."),
        make_case("contradictory", "The answer is X."),
    ]
    assert contradiction_handling_rate(cases) == 0.5


def test_average_latency():
    cases = [make_case("answerable", "a", latency=0.2), make_case("answerable", "b", latency=0.4)]
    assert abs(average_latency(cases) - 0.3) < 1e-6


def test_compute_all_metrics_keys():
    cases = [make_case("answerable", "a")]
    metrics = compute_all_metrics(cases)
    assert set(metrics.keys()) == {
        "retrieval_hit_rate", "citation_correctness", "answer_groundedness",
        "refusal_accuracy", "contradiction_handling_rate", "average_response_latency_seconds",
    }
