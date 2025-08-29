import os, sys, uuid, hashlib, glob, orjson, math, time, argparse, typing
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Iterable, Sequence

from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

from openai import OpenAI

from sentence_transformers import SentenceTransformer


class JSON_to_vec():
    def __init__(self, path, model_id: str = "nomic-ai/nomic-embed-text-v2-moe", fields: Sequence[str] = ("title", "abstract", "body", "text", "content"), overlap: int = 40):
        self.model = SentenceTransformer(model_id, trust_remote_code=True) #HF wrapper for vector embedding model
        #self.tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v2-moe")
        self.out_dim = self.model.get_sentence_embedding_dimension() #dimension for output vectors, based on model
        self.max_words = self.model.get_max_seq_length() #maximum words that can be input, based on model
        self.overlap = overlap
        self.fields = tuple(fields) #categories for text, ex: data, title, abstract
        self.p = Path(path)

    def json_to_str(self) -> str: #method converts json to string first
        if self.p.suffix.lower() == ".json": #make sure file is json
            data = orjson.loads(str(self.p)) 
            if isinstance(data, dict):
                obj = data #checks if file contents is in dict format, then stores
            elif isinstance(data, list) and data and isinstance(data[0], dict):
                obj = data[0] #processes file if contents are list of dicts
            else:
                raise ValueError("Unsupported JSON: need object or non-empty array of objects")
        else:
            raise ValueError("invalid input")

        parts = []
        for key in self.fields: #iterate thru fields
            v = obj.get(key)
            if v is None: 
                continue #if no value found, skip
            if isinstance(v, (list, tuple)):
                v = " ".join(map(str, v)) #joins items together into strings
            elif not isinstance(v, str): 
                v = str(v)
            v = v.strip()
            if v:
                parts.append(v)
        text = " ".join(parts).strip()
        if not text:
            raise ValueError(f"No text found for fields {self.fields}")
        return text
    

    def str_to_vector(self, text: str, is_query: bool = False) -> np.ndarray:
        if is_query: #if string is a user input
            return self.model.encode(text, prompt_name="query", normalize_embeddings=True)
        else: #embedding of document (JSON)
            return self.model.encode(text, prompt_name="passage", normalize_embeddings=True)
        