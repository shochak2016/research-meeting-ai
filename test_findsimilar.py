import os
import sys
import argparse
from datetime import datetime
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

load_dotenv()

from pipelines.rag2 import FindSimilar

def main():
    pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(index_name="pubmed")

    obj = FindSimilar(
        idx = index,
        query = "CRISPR gene editing in cats"
        )
    
    
    
    


if __name__ == "main":
    main()