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
After installing Ollama, running the following command in a terminal to use the Llama 3.2 3B param model. The first time you run this, it will download the model which is about 2GB and so this may take 10 minutes or so to download.

```
ollama run llama3.2:3b
```

This will then start the model in a terminal and you can test it by chatting to the model. To exit the chat type:

```
/bye
```

Once you have done this once, you do not need to start Ollama again in a terminal.

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
