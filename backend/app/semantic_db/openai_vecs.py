from typing import Dict, List, Optional
from openai import OpenAI

class OpenAIVecs:
    def __init__(self, 
                api_key: str,
                dimensions: Optional[int] = 1536,
                batch_size=1000,
                embedding_model="text-embedding-3-small" 
                ):
        self.api_key = api_key
        self.openai_client = OpenAI(api_key=self.api_key)
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
                batch_size (int): Size of batches for controlling parallel queries to OpenAI
                embedding_model (str): OpenAI model id for embeddings
            Returns:
                embeddings (List[List[float]]): A list of embeddings corresponding to each string in 'text_array'.
            """
            params: Dict = {'model': self.embedding_model}
            if self.dimensions:
                params['dimensions'] = self.dimensions
            embeddings = []
            for batch_start in range(0, len(text_chunks), self.batch_size):
                params['input'] = text_chunks[batch_start: batch_start + self.batch_size]
                response = self.openai_client.embeddings.create(**params)
                embeddings.extend([e.embedding for e in response.data])   
            return embeddings
    
    def get_embedding(self, text: str) -> List[float]:
        return self.get_embeddings([text])[0]