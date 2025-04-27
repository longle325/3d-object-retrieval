import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

class BGE_M3(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.device = "cuda:0"
        self.model = SentenceTransformer("BAAI/bge-large-en-v1.5", cache_folder="/root/train_caption/scripts/.cache")
        self.model.to(self.device).eval()

    def score_compare(self, sentence_1: list[str], sentence_2: list[str]) -> float:           
        embedding_1 = self.model.encode(sentence_1)
        embedding_2 = self.model.encode(sentence_2)
        return float(self.model.similarity(embedding_1, embedding_2)[0][0])
