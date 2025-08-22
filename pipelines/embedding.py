import os, sys, uuid, hashlib, glob, orjson, math, time, argparse
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm

from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

from openai import OpenAI

from transformers import AutoTokenizer, AutoModel


class JSON_to_vec():
    def __init__(self, model_id: str = "nomic-ai/nomic-embed-text-v1.5", max_words: int = 400, overlap: int = 40):
        self.model_id = AutoModel.from_pretrained(model_id, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v2-moe")
        self.out_dim = 768
        self.max_words = max_words
        self.overlap = overlap

    def load_single_json(path: Path) -> dict:
        if path.suffix.lower() == ".jsonl":
            with path.open("rb") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        obj = orjson.loads(line)
                        if isinstance(obj, dict):
                            return obj
                        raise ValueError("First JSONL line is not an object")
            raise ValueError("No non-empty lines found in JSONL")
        elif path.suffix.lower() == ".json":
            data = orjson.loads(path.read_bytes())
            if isinstance(data, dict):
                return data
            if isinstance(data, list) and data:
                if isinstance(data[0], dict):
                    return data[0]
                raise ValueError("First element of JSON array is not an object")
            raise ValueError("Unsupported JSON root; expected object or non-empty array")
        else:
            raise ValueError("invalid input")