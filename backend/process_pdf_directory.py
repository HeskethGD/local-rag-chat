from app.semantic_db.semantic_db import SemanticDb
import os
from app.semantic_db.ollama_vecs import OllamaVecs
from app.semantic_db.pdf_utils import get_pdf_bytes
from dotenv import load_dotenv
from time import time
import logging
import sys
# from app.semantic_db.openai_vecs import OpenAIVecs
# openai_api_key = os.getenv("OPENAI_API_KEY")

if len(sys.argv) != 3:
    print("Usage: poetry run python process_pdf_directory.py pdfs/ db_semantic/")
    sys.exit(1)

load_dotenv()

def process_pdf_directory(
    pdf_dir_path: str,
    semantic_db_path: str,
) -> tuple[int, list[str]]:
    """
    Process all PDFs in a directory and add them to a semantic database.
    
    Args:
        pdf_dir_path (str): Path to directory containing PDF files
        semantic_db_path (str): Path where semantic database should be stored
        
    Returns:
        tuple[int, list[str]]: Number of files processed and list of any failed files
    """
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize embedding function and semantic DB
    ollama_vecs = OllamaVecs()
    if not ollama_vecs.dimensions:
        raise ValueError('Vector dimensions cannot be undefined')
    semantic_db = SemanticDb(
        embedding_function=ollama_vecs.get_embeddings,
        vec_dimension=ollama_vecs.dimensions,
        semantic_db_path=semantic_db_path
    )

    # openai_vecs = OpenAIVecs(api_key=openai_api_key)
    # semantic_db = SemanticDb(
    #     embedding_function=openai_vecs.get_embeddings,
    #     vec_dimension=openai_vecs.dimensions,
    #     semantic_db_path="test_db_semantic/"
    # )
    
    # Helper function to generate timestamp IDs
    def get_timestamp_id() -> str:
        return str(int(time() * 1000000))
    
    # Validate directory exists
    if not os.path.isdir(pdf_dir_path):
        raise ValueError(f"Directory not found: {pdf_dir_path}")
    
    # Get list of PDF files
    pdf_files = [f for f in os.listdir(pdf_dir_path) if f.lower().endswith('.pdf')]
    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_dir_path}")
        return 0, []
    
    processed_count = 0
    failed_files = []
    
    # Process each PDF file
    for pdf_file in pdf_files:
        full_path = os.path.join(pdf_dir_path, pdf_file)
        try:
                
            # Read and process the PDF
            logger.info(f"Processing: {pdf_file}")
            pdf_bytes = get_pdf_bytes(full_path)
            
            semantic_db.add_file_to_semantic_db(
                doc_as_bytes=pdf_bytes,
                file_name=pdf_file,
                file_id=get_timestamp_id()
            )
            
            processed_count += 1
            logger.info(f"Successfully processed: {pdf_file}")
            
        except Exception as e:
            logger.error(f"Failed to process {pdf_file}: {str(e)}")
            failed_files.append(pdf_file)
            
    logger.info(f"Processing complete. Successfully processed {processed_count} files.")
    if failed_files:
        logger.warning(f"Failed to process {len(failed_files)} files: {failed_files}")
        
    return processed_count, failed_files


process_pdf_directory(
    pdf_dir_path = sys.argv[1],
    semantic_db_path = sys.argv[2],
)