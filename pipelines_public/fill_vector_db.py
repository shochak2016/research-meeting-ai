import os, time, datetime as dt, math, requests
import os, sys, getpass, bcrypt
from datetime import date, timedelta
from pinecone import Pinecone, ServerlessSpec

from lxml import etree

from operations.embedding import Embedder

from sentence_transformers import SentenceTransformer

HASH = os.environ.get("RAI_HASH")

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
PMED_API_KEY = "PMED_API_KEY" 
TOOL = "project"                      # per NCBI guidance
EMAIL = "youremail@gmail.com"             # per NCBI guidance

CUR_MONTH = "08"
MONTHS = {m:i for i,m in enumerate(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],1)}

PINECONE_KEY = "INSERT PINECONE KEY HERE"

pc = Pinecone(api_key=PINECONE_KEY)

'''Function is used to make user input password to run this function'''
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

'''Helper function which creates embedder object and converts input to vector'''
def create_vec(obj):
    e = Embedder(obj)
    text = e.to_str()
    vector = e.str_to_vec(text)
    return vector

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


def push_to_pinecone(idx, namespace: str, api_key: str = PMED_API_KEY, retmax: int = 400, chunk: int = 200):
    
    count, webenv, qk = search_papers()

    if not (count and webenv and qk):
        print("No papers found or some other error")
        return
    
    retstart, batch = 0, []
    while retstart < count:
        try:
            for row in fetch_lines(webenv, qk, retstart, retmax, api_key):
                content = row.get("content") or {}
                if not content.get("abstract"):
                    continue
                obj = {
                    "title": content.get("title") or "",
                    "abstract": content.get("abstract") or "",
                    "pub_date": content.get("pub_date") or "",
                    "authors": content.get("authors")
                    #"link": pmid_link(row.get("pmid"))
                }
                e = Embedder(obj)
                vec = e.str_to_vec(e.to_str())
                metadata = {"pmid": row["pmid"], 
                        "title": obj["title"], 
                        "pub_date": obj["pub_date"]
                        }
                batch.append({"id":str(row["pmid"]), "vals": vec.tolist(), "metadata": metadata})
                if len(batch) >= chunk:
                    idx.upsert(vectors=batch, namespace=namespace)
                    batch.clear()
            if batch:
                idx.upsert(vectors=batch, namespace=namespace)
                batch.clear()
        except Exception as e:
            raise RuntimeError(f"page {retstart}:{retstart+retmax-1} failed: {e}") from e
        retstart += retmax
        time.sleep(0.11 if api_key else 0.34)

##################

def main():
    password()
    dim = SentenceTransformer(model_id="nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True).get_sentence_embedding_dimension()
    
    pc = Pinecone(api_key=os.environ[PINECONE_KEY])
    index_name = os.environ.get("DB_NAME", "pubmed") #incomplete
    cloud = os.environ.get("PINECONE_CLOUD", "aws") #incomplete
    region = os.environ.get("PINECONE_REGION", "us-west-2") #incomplete

    if index_name not in [x["name"] for x in pc.list_indexes()]:
        pc.create_index(name=index_name, dimension=dim, metric="cosine",
                        spec=ServerlessSpec(cloud=cloud, region=region))
    idx = pc.Index(index_name)

    namespace = os.environ.get("PINECONE_NAMESPACE", "pubmed")
    push_to_pinecone(idx, namespace)


if __name__ == "__main__":
    main()
