import json
import os
from typing import List
from dotenv import load_dotenv
import requests
from ..semantic_db.semantic_db import SemanticDb

from ..models import ContentStream, Message
import logging

logger = logging.getLogger(__name__)

load_dotenv()

# from ..semantic_db.openai_vecs import OpenAIVecs
# openai_api_key = os.getenv("OPENAI_API_KEY")
# vecs = OpenAIVecs(api_key=openai_api_key)

from ..semantic_db.ollama_vecs import OllamaVecs
vecs = OllamaVecs()

semantic_db_path = os.getenv("SEMANTIC_DB_PATH")
llm_model = os.getenv("LLM_MODEL")
llm_generate_url = os.getenv("LLM_GENERATE_URL")

semantic_db = SemanticDb(
    embedding_function=vecs.get_embeddings,
    vec_dimension=vecs.dimensions,
    semantic_db_path=semantic_db_path
)
    
def run_agent(
    messages: List[Message],
    table_name="semantic-db-table"
    ) -> ContentStream:
    """
    Answers a user query with retrieval augmented generation (RAG) over documents in a LanceDB table in S3.

    Args
    - messages (List[Message]): The chat history as messages.
    - table_name (str): The name of the table in the semantic database.

    Yields
    - (ContentStream): The stream response from the LLM.
    """

    # Embedd the user query
    query_vector = []
    query = ""
    try:
        query = messages[-1].content
        query_vector = vecs.get_embedding(query)
    except Exception as e:
        err_message = f"There was an error processing the query: {e}"
        logger.error(err_message)
        yield err_message
        return
    if not query_vector:
        return

    # Retrieve relevant content from the vector db using semantic search
    results_text_array = []
    try:
        results_array = semantic_db.semantic_query(query_vector, table_name)
        sources = semantic_db.get_sources(results_array)
        results_text_array = [r.text for r in results_array]
        if not results_text_array:
            raise ValueError('results_text_array cannot be empty')
    except Exception as e:
        err_message = f"Error fetching sources: {e}"
        logger.error(err_message)
        yield err_message
        return


    # Cal LLM to summarise answer based on retrieved content
    try:
        # Define prompt for document summary
        all_text = ' '.join(results_text_array)
        prompt =  (
            'Answer this question:'
            f'\nUSER QUESTION\n{query}\n'
            'Use this content in your answer:'
            f'\n\nCONTENT_STARTS:\n"{all_text}"\nCONTENT_ENDS\n\n'
            'IMPORTANT - The content may be truncated so do not simply continue the sentence, '
            'always rephrase to give complete sentences when needed. '
            'Not everything will be relevant so do not comment on anything you do not use to answer the question.'
        )
        
        # Send the POST request with streaming enabled
        with requests.post(
                llm_generate_url, 
                json={
                    "model": llm_model, 
                    "prompt": prompt
                },
                stream=True
            ) as r:
            r.raise_for_status()
            # Process the streamed response
            for chunk in r.iter_lines(decode_unicode=True):
                if chunk:
                    try:
                        data = json.loads(chunk)
                        yield data.get("response", "")
                    except json.JSONDecodeError:
                        yield f"Error decoding response chunk: {chunk}"

        # Provide references
        if len(sources):
            yield '\n\n\n**References:**\n\n'
            for s in sources:
                yield s

    except Exception as e:
        err_message = f"Error drafting answer: {e}"
        logger.error(err_message)
        yield err_message
        return

    
        


