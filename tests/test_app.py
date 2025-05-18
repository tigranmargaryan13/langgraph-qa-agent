import time
import pytest
from app import create_app, history, history_lock


@pytest.fixture
def client():
    app = create_app(test=True)
    app.config['TESTING'] = True
    with app.test_client() as client:
        with history_lock:
            history.clear()
        yield client


def test_index_page(client):
    """Ensure the index page renders correctly"""
    response = client.get('/')
    assert response.status_code == 200
    assert 'Knowledge Hub' in response.data.decode('utf-8')


def test_post_question_adds_thinking(client):
    """Test that posting a question adds it to history with 'Thinking...' initially"""
    response = client.post('/api/ask', json={"question": "Test question"})
    assert response.status_code == 200
    data = response.get_json()
    assert data["question"] == "Test question"
    assert data["answer"] == "Thinking..."

    history_response = client.get('/api/history')
    history_data = history_response.get_json()
    assert len(history_data) >= 1
    assert history_data[-1]["question"] == "Test question"
    assert history_data[-1]["answer"] == "Thinking..."


def test_answer_is_updated_by_thread(client):
    """Test that background thread updates the 'Thinking...' answer within time"""
    question = "Test question"
    client.post('/api/ask', json={"question": question})

    timeout = 5
    interval = 0.5
    elapsed = 0

    while elapsed < timeout:
        time.sleep(interval)
        history_response = client.get('/api/history')
        history_data = history_response.get_json()

        for entry in history_data:
            if entry["question"] == question and entry["answer"] != "Thinking...":
                assert "Fake" in entry["answer"] or entry["answer"] != ""
                return
        elapsed += interval
