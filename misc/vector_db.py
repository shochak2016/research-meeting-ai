'''
File is meant to pull from DB, embed content, and store into vector DB. Should only be done once, daily_sync.py
will handle the updates.
'''
import os, sys, bcrypt, getpass
from supabase import create_client
from pinecone import Pinecone, ServerlessSpec
from pipelines.operations.embedding import Embedder #importing embedding class

from lxml import etree

HASH = os.environ.get("RAI_HASH")
if not HASH:
    sys.exit("Missing SCRIPT_PASS_HASH env var.")

pw = getpass.getpass("Password: ").encode()
if not bcrypt.checkpw(pw, HASH.encode()):
    sys.exit("Access denied.")


'''Helper functions'''
def batch_yield(obj, n):
    obj = list(obj)
    for i in range(0, len(obj), n):
        yield obj[i : i + n]

def fill_vector_db():
    pass