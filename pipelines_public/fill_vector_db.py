import os, time, datetime as dt, math, requests
from urllib.parse import urlencode
import os, sys, getpass, bcrypt
from datetime import date, timedelta
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

from lxml import etree
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
import gc  # For garbage collection

# Load environment variables from .env file
load_dotenv()

HASH = os.environ.get("RAI_HASH")

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
PMED_API_KEY = os.getenv("PMED_API_KEY")  
TOOL = "research-ai"                      # per NCBI guidance
EMAIL = "your email"             # per NCBI guidance

CUR_MONTH = "08"
MONTHS = {m:i for i,m in enumerate(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],1)}

PINECONE_KEY = os.environ.get("PINECONE_API_KEY")

# Don't initialize here - will initialize in main() function

'''
Function is used to make user input password to run this function.
OPTIONAL: This function provides password protection when running scripts from terminal.
For Streamlit or web apps, you may want to use different authentication methods.
'''
def password():
    if not HASH:
        sys.exit("Missing SCRIPT_PASS_HASH env var.")

    pw = getpass.getpass("Password: ").encode()
    if not bcrypt.checkpw(pw, HASH.encode()):
        sys.exit("Access denied.")

'''Function is used to return pointers to all the papers we want to add to vector DB'''
def search_papers(datetype="edat", days=30):
    """Find PMIDs by publication date (YYYY/MM/DD). Returns count + WebEnv/query_key."""
    params = {
    "db": "pubmed", 
    "term": "all[sb]", 
    "retmode": "json",
    "usehistory": "y", 
    "datetype": datetype, # publication date ("pdat") or by when it was added ("edat")
    "reldate":days,
    "tool": TOOL, 
    "email": EMAIL,
    "retmax": 0 
    }

    if PMED_API_KEY:
        params["api_key"] = PMED_API_KEY
    r = requests.get(f"{BASE}/esearch.fcgi", params=params, timeout=60)
    r.raise_for_status()
    js = r.json()["esearchresult"] #creates python dict from parse
    return int(js["count"]), js["webenv"], js["querykey"] #return values from corresponding keys for the dict

'''Function is used to process XML files'''
def clean_xml(elem) -> str: #takes tags out of an XML line/element
    if elem is not None:
        return "".join(elem.itertext()).strip()
    else: 
        return ""

'''Function is used to make process dates and make sure month is correct'''
def month_norm(m):  
    if m is None: 
        return CUR_MONTH
    m = m.strip()
    if not m:
        return CUR_MONTH
    if m in MONTHS:
        return f"{MONTHS[m]:02d}"
    try:
        return f"{int(m):02d}"
    except ValueError:
        return "01"

def convert_article(elem):
    pmid  = clean_xml(elem.find(".//MedlineCitation/PMID"))
    title = clean_xml(elem.find(".//Article/ArticleTitle"))

    parts = []
    for ab in elem.findall(".//Article/Abstract/AbstractText"):
        t = clean_xml(ab)
        if t:
            label = ab.get("Label")
            parts.append(f"{label}: {t}" if label else t)
    abstract = "\n".join(parts) or None

    y = clean_xml(elem.find(".//Article/ArticleDate/Year")) or clean_xml(elem.find(".//JournalIssue/PubDate/Year"))
    if not y:
        return None
    m = clean_xml(elem.find(".//Article/ArticleDate/Month")) or clean_xml(elem.find(".//JournalIssue/PubDate/Month"))
    d = clean_xml(elem.find(".//Article/ArticleDate/Day"))   or clean_xml(elem.find(".//JournalIssue/PubDate/Day"))
    pub_date = f"{y}-{month_norm(m)}-{(d or '01').zfill(2)}"

    authors = []
    for auth in elem.findall(".//Article/AuthorList/Author"):
        coll = clean_xml(auth.find("CollectiveName"))
        if coll:
            authors.append(coll); continue
        nm = " ".join(x for x in [clean_xml(auth.find("LastName")), clean_xml(auth.find("Initials"))] if x)
        if nm:
            authors.append(nm)

    if not (pmid and title):
        return None

    pmid_i = int(pmid)
    return {
        "pmid": pmid_i,
        "vector_id": str(pmid_i),  # optional but handy for Pinecone
        "contents": {
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "pub_date": pub_date
        },
        "pub_date": pub_date
    }

'''Helper function which creates embedder object and converts input to vector
def create_vec(obj):
    e = Embedder(obj)
    text = e.to_str()
    vector = e.str_to_vec(text)
    return vector
'''

'''Streams lines from a study after converting the article'''
def fetch_lines(webenv, query_key, retstart: int, retmax: int, api_key=PMED_API_KEY):
    p = {"db":"pubmed",
        "WebEnv":webenv,
        "query_key":query_key,
        "retstart":retstart,
        "retmax":retmax,
        "retmode":"xml",
        "tool":TOOL,
        "email":EMAIL}
    if api_key: 
        p["api_key"] = api_key
    with requests.get(f"{BASE}/efetch.fcgi", params=p, stream=True, timeout=120) as r:
        r.raise_for_status()
        r.raw.decode_content = True
        for _, elem in etree.iterparse(r.raw, events=("end",), tag="PubmedArticle"):
            row = convert_article(elem)
            elem.clear()
            while elem.getprevious() is not None:
                del elem.getparent()[0]
            if row:
                yield row

'''Converts pmid (int) to the actual link we will store'''
def pmid_link(pmid) -> str:
    return f"https://pubmed.ncbi.nlm.nih.gov/{int(pmid)}/"

'''Checkpoint management for resuming interrupted runs'''
def load_checkpoint(checkpoint_file="processed_pmids.txt"):
    """Load previously processed PMIDs from checkpoint file"""
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            processed = set(line.strip() for line in f)
        print(f"📌 Loaded checkpoint: {len(processed)} papers already processed")
        return processed
    return set()

def save_checkpoint(pmid, checkpoint_file="processed_pmids.txt"):
    """Append a processed PMID to checkpoint file"""
    with open(checkpoint_file, 'a') as f:
        f.write(f"{pmid}\n")


def push_to_pinecone(idx, namespace: str, model, api_key: str = PMED_API_KEY, retmax: int = 400, chunk: int = 200):
    
    print("🔍 Searching for papers in PubMed...")
    count, webenv, qk = search_papers()

    if not (count and webenv and qk):
        print(" No papers found or some other error")
        return
    
    print(f" Found {count} papers to process")
    print(f" Processing in batches of {retmax}, uploading in chunks of {chunk}")
    
    # Load checkpoint to skip already processed papers
    processed_pmids = load_checkpoint()
    
    retstart, batch = 0, []
    total_uploaded = 0
    
    # Progress bar for overall papers
    with tqdm(total=count, desc="Processing papers", unit="papers") as pbar:
        while retstart < count:
            try:
                print(f"\n Fetching papers {retstart+1} to {min(retstart+retmax, count)}...")
                papers_in_batch = 0
                
                for row in fetch_lines(webenv, qk, retstart, retmax, api_key):
                    pmid_str = str(row["pmid"])
                    
                    # Skip if already processed
                    if pmid_str in processed_pmids:
                        pbar.update(1)
                        continue
                    
                    content = row.get("contents") or {}
                    if not content.get("abstract"):
                        pbar.update(1)  # Still count it as processed
                        continue
                    
                    # Only embed title and abstract
                    title = content.get("title") or ""
                    abstract = content.get("abstract") or ""
                    text_to_embed = f"{title} {abstract}".strip()
                    
                    # Use the pre-initialized model instead of reinstatiating class
                    vec = model.encode(text_to_embed, normalize_embeddings=True)
                    
                    metadata = {
                        "pmid": row["pmid"], 
                        "title": title,
                        "pub_date": content.get("pub_date") or "",
                        "authors": content.get("authors") or []  # Store in metadata for filtering
                    }
                    batch.append({"id":pmid_str, "values": vec.tolist(), "metadata": metadata})
                    
                    # Save to checkpoint after adding to batch
                    save_checkpoint(pmid_str)
                    
                    # Clear the row from memory after processing
                    del row
                    papers_in_batch += 1
                    pbar.update(1)
                    
                    if len(batch) >= chunk:
                        print(f"Uploading {len(batch)} vectors to Pinecone...")
                        idx.upsert(vectors=batch, namespace=namespace)
                        total_uploaded += len(batch)
                        print(f"Total uploaded so far: {total_uploaded}")
                        batch.clear()
                        gc.collect()  # Force garbage collection to free memory
                
                if batch:
                    print(f"Uploading final {len(batch)} vectors to Pinecone...")
                    idx.upsert(vectors=batch, namespace=namespace)
                    total_uploaded += len(batch)
                    print(f"Total uploaded so far: {total_uploaded}")
                    batch.clear()
                    gc.collect()  # Force garbage collection to free memory
                    
                print(f"Processed {papers_in_batch} papers with abstracts from this batch")
                
            except Exception as e:
                print(f"\n Error at papers {retstart}:{retstart+retmax-1}: {e}")
                raise RuntimeError(f"page {retstart}:{retstart+retmax-1} failed: {e}") from e
            
            retstart += retmax
            time.sleep(0.11 if api_key else 0.34)
    
    print(f"\n Upload successful, {total_uploaded} papers to Pinecone index '{idx}' in namespace '{namespace}'")
    print(f"Summary: {total_uploaded}/{count} papers had abstracts and were indexed")

##################

def main():
    print("Starting PubMed to Pinecone pipeline...")
    
    # password()  # Commented out for Streamlit usage - uncomment if running from terminal with password protection
    
    print("Initializing embedding model (this takes a moment)...")
    model = SentenceTransformer(model_name_or_path="nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)
    dim = model.get_sentence_embedding_dimension()
    print(f"Model loaded with dimension: {dim}")
    
    print("Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_KEY)
    index_name = os.environ.get("DB_NAME", "pubmed")
    cloud = os.environ.get("PINECONE_CLOUD", "aws")
    region = os.environ.get("PINECONE_REGION", "us-east-1")

    if index_name not in [x["name"] for x in pc.list_indexes()]:
        print(f"Creating new index '{index_name}'...")
        pc.create_index(name=index_name, dimension=dim, metric="cosine",
                        spec=ServerlessSpec(cloud=cloud, region=region))
        print(f"Index created!")
    else:
        print(f"Using existing index '{index_name}'")
        
    idx = pc.Index(index_name)

    namespace = os.environ.get("PINECONE_NAMESPACE", None)  # Default to None (no namespace)
    if namespace:
        print(f"Using namespace: '{namespace}'")
    else:
        print("Using default namespace (no partition)")
    
    push_to_pinecone(idx, namespace, model)

if __name__ == "__main__":
    main()
