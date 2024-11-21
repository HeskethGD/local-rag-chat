# Offline Local AI Agent Chatbot with RAG

This project builds a chatbot connected to an AI Agent that provides retrieval-augmented generation (RAG), enabling users to ask questions of their docs. The backend is built with Python and uses FastAPI for the API layer. The application is designed to run offline and for the Large Language Model (LLM) it uses Ollama to run small 3B param models such as Llama 3.2 3B.

The frontend chat UI may eventually be built with Nextjs but for now there is a Python based Streamlit UI in the backend folder.

## Prerequisites

- Python 3.11 or higher
- Poetry (package manager)
- Git
- Ollama

To install Ollama, follow the instructions for your operating sytem here: https://ollama.com/download

## Using Llama 3.2 3B with Ollama

To view models available on Ollama see here: https://ollama.com/library.
This application uses the Llama 3.2 3B param model (about 2GB, approx 10 minutes download) for chat and summaries, and the nomic-embed-text model (274Mb approx 2 minutes download) for vectorisation. To pull them from Ollama run these commands:

```
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

Once you have done this once, you do not need to start Ollama again in a terminal.

## Run the script to embedd the pdfs

Create a folder called `pdfs` in the root of the `backend` folder then run the script to embed the pdfs and load them into the LanceDB vector database:

```
cd backend
poetry run python process_pdf_directory.py pdfs/ db_semantic/
```

## Run the FastAPI Python backend

To run the Python based FastAPI backend:

```
cd backend
poetry run uvicorn app.main:app --reload
```

## Run the Streamlit Python UI

There is a Python based UI built with Streamlit that is for debugging only. It is in the backend folder, LOL. To run it:

```
cd backend
poetry run streamlit run ui/streamlit_app.py
```
