from typing import List, Optional
import ollama

class OllamaVecs:
    def __init__(self, 
                dimensions: Optional[int] = 768,
                batch_size=1000,
                embedding_model="nomic-embed-text"
                ):
        self.dimensions = dimensions
        self.batch_size = batch_size
        self.embedding_model = embedding_model
        
    def get_embeddings(
            self,
            text_chunks: List[str]
        ) -> List[List[float]]:
            """
            Generate embeddings for a list of text strings using a specified model in batches.

            Args:
                text_chunks (List[str]): A list of strings for which embeddings are to be generated.
                dimensions (Optional[int]): Integer for dimensions of embeddings. Trade off computationally efficiency for performance.
                batch_size (int): Size of batches for controlling parallel queries to Ollama
                embedding_model (str): Ollama model id for embeddings
            Returns:
                embeddings (List[List[float]]): A list of embeddings corresponding to each string in 'text_array'.
            """

            embeddings = []
            for batch_start in range(0, len(text_chunks), self.batch_size):
                response = ollama.embed(
                     model=self.embedding_model,
                     input=text_chunks[batch_start: batch_start + self.batch_size]
                )
                embeddings.extend(response["embeddings"])   
            return embeddings
    
    def get_embedding(self, text: str) -> List[float]:
        return self.get_embeddings([text])[0]