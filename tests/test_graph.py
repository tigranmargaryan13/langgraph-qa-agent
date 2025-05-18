from langchain_core.messages import AIMessage


def test_retrieve_context(fake_graph):
    state = {"question": "Test question"}
    updated_state = fake_graph.retrieve_context(state)
    assert "context" in updated_state
    assert updated_state["context"] == "fake document"
    assert updated_state["iterations"] == 0


def test_generate_and_check_answer(fake_graph):
    state = {
        "question": "Test question",
        "context": "Some mocked context",
        "iterations": 0,
        "history": [],
    }
    state = fake_graph.generate_answer(state)
    assert "answer" in state
    assert "Fake response" in state["answer"]


def test_check_answer(fake_graph):
    state = {
        "question": "Test question",
        "context": "Some mocked context",
        "answer": AIMessage(content="Fake generated answer."),
        "iterations": 0
    }
    state = fake_graph.check_answer(state)
    assert state["answer_validity"] == "N"


def test_decide_to_finish(fake_graph):
    state = {"iterations": 0, "answer_validity": "N"}
    assert fake_graph.decide_to_finish(state) == "reflect"
    state = {"iterations": 1, "answer_validity": "Y"}
    assert fake_graph.decide_to_finish(state) == "end"


def test_workflow_integration(fake_graph):
    state = {"question": "Test question", 'history': []}
    compiled = fake_graph.setup_workflow()
    final_state = compiled.invoke(state)
    assert "answer" in final_state
    assert final_state["answer_validity"] == "N"
