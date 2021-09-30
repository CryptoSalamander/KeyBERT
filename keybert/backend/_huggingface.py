import numpy as np
from typing import List, Union
from transformers import AutoTokenizer, AutoModel
from keybert.backend import BaseEmbedder

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

class HuggingfaceBackend(BaseEmbedder):
    """ Huggingface embedding model
    The huggingface-bert embedding model used for generating document and
    word embeddings.
    Arguments:
        embedding_model: A huggingface embedding model / or path
        
    """
    def __init__(self, embedding_model: Union[str, AutoModel]):
        super().__init__()

        if isinstance(embedding_model, AutoModel):
            self.embedding_model = embedding_model
        elif isinstance(embedding_model, str):
            self.embedding_model = AutoModel.from_pretrained(embedding_model)
            self.embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model)
        else:
            raise ValueError("Please select a correct huggingface model: \n"
                             "`from transformers import AutoModel` \n"
                             "`model = AutoModel.from_pretrained('bert-base-uncased')`")
            


    def embed(self,
              documents: List[str],
              verbose: bool = False) -> np.ndarray:
        """ Embed a list of n documents/words into an n-dimensional
        matrix of embeddings
        Arguments:
            documents: A list of documents or words to be embedded
            verbose: Controls the verbosity of the process
        Returns:
            Document/words embeddings with shape (n, m) with `n` documents/words
            that each have an embeddings size of `m`
        """
        encoded_input = self.embedding_tokenizer(documents, padding=True, truncation=True, return_tensors='pt',max_length=512)
        with torch.no_grad():
            model_output = self.embedding_model(**encoded_input)
        embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        return embeddings