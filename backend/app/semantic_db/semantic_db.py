from typing import List, Callable
import lancedb
from lancedb.pydantic import Vector, LanceModel
from pydantic import BaseModel
from .pdf_utils import pdf_bytes_to_chunks

class RagSearchResult(BaseModel):
    text: str
    file_name: str
    file_id: str
    page_label: str
    page_index: int
    _distance: float

def create_embedded_chunk_type(dimension: int):
    """Create an EmbeddedChunk class with a specific vector dimension"""
    class EmbeddedChunk(LanceModel):
        text: str
        vector: Vector(dimension) # type: ignore
        file_name: str
        file_id: str
        page_label: str
        page_index: int
    
    return EmbeddedChunk

class SemanticDb:
    """
    Args
    - embedding_function (Callable[[List[str]], List[List[float]]]): Function for embedding strings as vectors with metadata
    - vec_dimension (int)
    - semantic_db_path (str): Folder path where the semantic db should be created. Examples:
        * A local folder path, e.g. 'test_semantic_db/'
        * An S3 folder path, e.g. 's3://my-bucket/my-folder/'
    """
    def __init__(
        self,
        embedding_function: Callable[[List[str]], List[List[float]]],
        vec_dimension: int,
        semantic_db_path: str
    ):
        self.embedding_function = embedding_function
        self.vec_dimension = vec_dimension
        self.semantic_db_path = semantic_db_path
        self.EmbeddedChunk = create_embedded_chunk_type(vec_dimension)

    def add_file_to_semantic_db(
        self,
        doc_as_bytes: bytes,
        file_name: str,
        file_id: str,
        table_name="semantic-db-table"
    ):
        """
        Embedds text_chunks using a vectoriser algorithm and writes it to a LanceDB table for semantic search.

        Args
        - doc_as_bytes (bytes): Representation of a pdf document in bytes.
        - file_name (str): Name of document for use in meta data.
        - file_id (str): Id of document for use in meta data.
        - table_name (str): The name of the table in the semantic database.
        """
        text_chunks = pdf_bytes_to_chunks(
            doc_as_bytes=doc_as_bytes,
            file_name=file_name,
            file_id=file_id
        )
        embeddings = self.embedding_function(text_chunks=[c.text for c in text_chunks])
        embedded_chunks = [self.EmbeddedChunk(**c.model_dump(), vector=v) for c, v in zip(text_chunks, embeddings)]
        (lancedb
            .connect(self.semantic_db_path)
            .create_table(
                table_name, 
                schema=self.EmbeddedChunk,
                exist_ok=True)
            .add(embedded_chunks)
        )

    def delete_file_from_semantic_db(
            self,
            file_id: str,
            table_name="semantic-db-table"
        ):
        """
        Delete rows from semantic db based on file_name.

        Args
        - file_name (str): The file name to be deleted from the semantic db (deletes all chunks from that file)
        - table_name (str): The name of the table in the semantic database.
        """
        
        (lancedb
            .connect(self.semantic_db_path)
            .open_table(table_name)
            .delete(f'file_id = "{file_id}"')
        )

    def semantic_query(self, 
                           query_vector: List[float],
                           table_name="semantic-db-table",
                           N_results = 4
                           ):
        """
        Query the vector database using semantic search.

        Args
        - query_vector (List[float]): The vectorised user query to query with.
        - table_name (str): The name of the table in the vector db to query.
        - N_results (int): The limit for the number of returned results.

        Return
        - search_results (List[RagSearchResult]): The semantic search results.
        """
        results = (lancedb
                .connect(self.semantic_db_path)
                .open_table(table_name)
                .search(query_vector)
                .limit(N_results))
        return  [RagSearchResult(**r) for r in results.to_list()]
    
    def get_sources(self, results: List[RagSearchResult]) -> List[str]:
        """
        Utility for providing the files and page numbers for the information used in RAG answers.

        Args
        - results (List[RagSearchResult]): The search results from the vector database query.

        Returns
        - sources (List[str]): The sources used to answer the question as reference style strings.
        """
        try:
            # Group sources by document
            grouped_sources = {}
            for r in results:
                doc = r.file_name
                page = r.page_index
                if doc in grouped_sources:
                    grouped_sources[doc].append(page)
                else:
                    grouped_sources[doc] = [page]
            # Generate formatted output
            sources = []
            for doc, pages in grouped_sources.items():
                if len(pages) > 1:
                    sources.append(f'*{doc}*, pp. {", ".join(map(str, sorted(list(set(pages)))))}\n') 
                else:
                    sources.append(f'*{doc}*, p. {pages[0]}\n')
            return sources
        
        except Exception as e:
            print("error fetching sources", e)
            return []