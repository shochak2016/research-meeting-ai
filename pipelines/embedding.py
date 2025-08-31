import os, sys, uuid, hashlib, glob, orjson, math, time, argparse, typing
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import Iterable, Sequence

#from pinecone.grpc import PineconeGRPC as Pinecone
#from pinecone import ServerlessSpec

from openai import OpenAI

from sentence_transformers import SentenceTransformer

import xml.etree.ElementTree as ET

'''
document converts JSON files to vector embeddings. JSON files are used to store study contents, 
and through RAG pipeline, need to be converted into vector embeddings than can be fed to LLM, in this case ChatGPT
'''

class Embedder():
    def __init__(self, obj, model_id: str = "nomic-ai/nomic-embed-text-v2-moe", fields: Sequence[str] = ("title", "abstract", "body", "text", "content", "link"), overlap: int = 40):
        self.model = SentenceTransformer(model_id, trust_remote_code=True) #HF wrapper for vector embedding model
        #self.tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v2-moe")
        self.out_dim = self.model.get_sentence_embedding_dimension() #dimension for output vectors, based on model
        self.max_words = self.model.get_max_seq_length() #maximum tokens that can be input, based on model
        self.overlap = overlap
        self.fields = tuple(fields) #categories for text, ex: data, title, abstract
        self.obj = obj #JSON file from DB is dumped as a dict automatically
        

    def db_to_str(self) -> str: #method converts DB entry to string first
        '''
        Since DB handles JSON conversion and dumps as a python dict through package, 
        conversion below is redundant

        ######################

        if self.p.suffix.lower() == ".json": #make sure file is json
            data = orjson.loads(self.p.read_bytes()) #reads by bytes because data is stored as jsonb
            if isinstance(data, dict):
                obj = data #checks if file contents is in dict format, then stores
            elif isinstance(data, list) and data and isinstance(data[0], dict):
                obj = data[0] #processes file if contents are list of dicts
            else:
                raise ValueError("Unsupported JSON: need object or non-empty array of objects")
        else:
            raise ValueError("invalid input")
        '''

        parts = []
        for key in self.fields: #iterate thru fields
            v = self.obj.get(key)
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
    


    def xml_to_text(xml_str: str) -> str: #XML might be easier than JSON since pubmed API is buns
        root = ET.fromstring(xml_str)
        t = "".join((root.find(".//Article/ArticleTitle") or ET.Element("x")).itertext()).strip()
        parts = ["".join(n.itertext()).strip() for n in root.findall(".//Article/Abstract/AbstractText")]
        abstract = "\n".join(p for p in parts if p)
        return (t + ("\n\n" if t and abstract else "") + abstract).strip() or xml_str
        #vibe coded

    def str_to_vector(self, text: str, is_query: bool = False) -> np.ndarray:
        if is_query: #if string is a user input
            return self.model.encode(text, prompt_name="query", normalize_embeddings=True)
        else: #embedding of document (JSON)
            return self.model.encode(text, prompt_name="passage", normalize_embeddings=True)
        