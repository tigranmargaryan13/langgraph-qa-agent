# 🧠 LangGraph Q&A Assistant

A lightweight Flask-based Q&A application powered by **LangGraph**, **LangChain**, and **OpenAI**.  
This assistant is designed to help answer questions related to **legal, historical, economic, and political topics**.  
It retrieves relevant answers from a vector database using semantic search, validates them using a checking model, and iteratively improves responses using a reflection loop.

---

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/tigranmargaryan13/langgraph-qa-agent.git
cd langgraph-qa-agent
```

### 2. Create and Activate Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows use `.venv\Scripts\activate`
```

### 3. Install Dependencies

```bash
pip install -r requirements.dev.txt
```

### 4. Prepare Environment Variables

Create a `.env` file at the project root with the following content:

```env
OPENAI_API_KEY=your_openai_key
```

Or make sure it’s properly handled by your `config.py`.

---

## ⚙️ Execution Instructions

### 1. Vector Store

A prebuilt FAISS index already exists in the repository under `faiss_index/`.  
To rebuild the index from a CSV dataset (e.g., `data/rag_dataset.csv`), run:

```bash
python build_index.py
```

### 2. Start the Flask Server

```bash
python app.py
```

Visit: [http://localhost:5002](http://localhost:5002)

---

## 🐳 Run with Docker

### 1. Build and Start the App

```bash
docker-compose up --build
```

This will:
- Build the image using the `Dockerfile`
- Start the app on [http://localhost:5002](http://localhost:5002)
- Mount the current directory for live code updates
- Use environment variables from `.env`

### 2. File References

- 📦 [`Dockerfile`](./Dockerfile): Defines the Python base image and installs dependencies
- ⚙️ [`docker-compose.yml`](./docker-compose.yml): Sets up the service, ports, volumes, and env variables

---

## 🧪 Testing

### Run Tests with Pytest

```bash
pytest tests
```

### Test Files

- `tests/test_app.py`: Integration tests for Flask endpoints (`/`, `/api/ask`, `/api/history`).
- `tests/test_graph.py`: Unit tests for core LangGraph logic (`retrieve_context`, `generate_answer`, `check_answer`, etc.).
- `conftest.py`: Shared fixtures for fake models and settings used in testing.

---



## 📁 Project Structure

```
├── app.py                  # Flask application and routes
├── build_index.py          # Script to build vector index
├── config.py               # Loads app settings
├── graph.py                # Graph agents workflow
├── prompts.py
├── templates/index.html    # Web frontend (form UI)
├── static/style.css        # Page styling
├── data/rag_dataset.csv
├── faiss_index/            # FAISS index
├── tests/
│   ├── test_app.py         # Integration endpoint tests
│   └── test_graph.py       # Graph unit tests
├── requirements.txt
├── requirements.dev.txt
└── Dockerfile
└── docker-compose.yml
```

---

## 📚 Dataset
This project uses a question-answering open source dataset focused on **legal, historical, economic, and political domains**,  
taken from the [LegalQAEval dataset](https://huggingface.co/datasets/isaacus/LegalQAEval) on Hugging Face.

You can find a sample version in the `data/rag_dataset.csv` file, pre-processed for vector index building.

---

## 🧩 Approach Explanation

This project leverages a **multi-step LangGraph workflow** to generate and validate answers with the following structure:

1. **Retrieval**: Uses FAISS and OpenAI embeddings to fetch top-k relevant documents from a vector database.
2. **Generation**: A prompt-tuned OpenAI model generates answers based on retrieved context and conversation history.
3. **Answer Checking**: A structured LLM validates the answer and determines if it is grounded and sufficient.
4. **Reflection**: If the answer is invalid, a reflection mechanism re-evaluates and improves the answer, respecting a maximum retry count.
5. **Workflow Engine**: LangGraph powers the execution flow using `StateGraph`, allowing conditional edge transitions between nodes.

![LangGraph Workflow](image-1.png)
---

## ⚠️ Known Limitations / Trade-Offs

- **Reflection Depth**: Limited to `MAX_ITERATIONS` (default: 1). More iterations could improve quality but increase latency.
- **No Frontend Interaction**: The current `/` route renders basic HTML. Front-end could be enhanced for a better user experience.
- **No Persistence**: History is in-memory and resets on server restart. Consider database storage for persistence.

---

## 🏗️ Future Improvements

- Improve frontend interface
- Add persistent user sessions
- Fine-tune LLM prompts for domain-specific use cases
- Add streaming output support

---
