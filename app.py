import threading
import logging

from flask import Flask, request, jsonify, render_template

from config import settings
from graph import LangGraphClass


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

history = []

# Lock to safely update shared history across threads
history_lock = threading.Lock()


def create_app(test=False):
    """
    Initialize Flask app and LangGraph agent.

    Args:
        test (bool): Whether to enable test mode using fake LLMs.

    Returns:
        Flask: Configured Flask application.
    """

    app = Flask(__name__)

    if test:
        settings.TEST_TRUE = True

    # Initialize LangGraph pipeline
    graph = LangGraphClass(settings)
    agent_app = graph.setup_workflow()

    @app.route('/')
    def index():
        return render_template('index.html', history=history)

    @app.route('/api/ask', methods=['POST'])
    def ask():
        """
        Handle a POST request with a user question.

        Returns:
            JSON: Question and initial response with placeholder answer.
        """
        data = request.json
        question = data.get("question")

        entry = {"question": question, "answer": "Thinking..."}
        with history_lock:
            history.append(entry)

        def process_answer(question, entry):
            """
            Background task to generate and update answer using LangGraph.

            Args:
                question (str): The user's question.
                entry (dict): Reference to the entry in the shared history list.
            """
            try:
                state = {"question": question, "history": history}
                final_state = agent_app.invoke(state)
                answer = final_state["answer"]
                with history_lock:
                    entry["answer"] = answer  # Update answer in shared history
            except Exception as e:
                with history_lock:
                    entry["answer"] = f"Error: {str(e)}"

        # Launch processing in background thread (non-blocking)
        threading.Thread(target=process_answer, args=(question, entry)).start()

        return jsonify({"answer": "Thinking...", "question": question})

    @app.route('/api/history', methods=['GET'])
    def get_history():
        """Retrieve full conversation history"""
        with history_lock:
            return jsonify(history)

    return app


app = create_app()
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002, debug=True)
