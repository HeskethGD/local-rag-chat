import io
from typing import List
import pymupdf
from pydantic import BaseModel

class TaggedChunk(BaseModel):
    text: str
    file_name: str
    file_id: str
    page_label: str
    page_index: int

def pdf_bytes_to_chunks(
    doc_as_bytes: bytes,
    file_name: str,
    file_id: str,
    min_chunk_length=10
    ) -> List[TaggedChunk]:
    """
    Processes a bytes representation of a pdf document.
    Returns an array of chunks from its pages tagged with meta data.

    Args
    - doc_as_bytes (bytes): Representation of a pdf document in bytes.
    - file_name (str): Name of document for use in meta data.
    - min_chunk_length (int): Drop any chunk that is smaller than this minimum length.

    Returns
    - text_chunks (List[TaggedChunk]): An array of text chunks with attached page and file meta data
    """

    # Check if the file type is PDF
    file_type =  file_name.split(".")[-1]
    if file_type.lower() != "pdf":
        error_msg = "File type is not a pdf"
        print(error_msg)
        raise ValueError(error_msg)
    
    # Open the PDF document from bytes
    pdf_document = pymupdf.open(stream = doc_as_bytes)
    # Treat every page as a chunk
    text_chunks = [
        TaggedChunk(
            text=p.get_text(), 
            file_name=file_name,
            file_id=file_id,
            page_label=p.get_label(),
            page_index=i + 1
        ) 
        for i, p in enumerate(pdf_document.pages()) if len(p.get_text()) > min_chunk_length
    ]

    return text_chunks

def get_pdf_bytes(file_path: str) -> bytes:
    """
    Opens a document using PyMuPDF and returns its bytes representation.
    
    Args:
        file_path (str): Path to the document file
        
    Returns:
        bytes: The document as bytes
    """
    try:
        # Open the document
        doc = pymupdf.open(file_path)
        
        # Get bytes representation
        doc_bytes = doc.tobytes()
        
        # Close the document
        doc.close()
        
        return doc_bytes
        
    except Exception as e:
        raise Exception(f"Error reading document: {str(e)}")