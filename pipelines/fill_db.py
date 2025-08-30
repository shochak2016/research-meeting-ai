import os, time, datetime as dt, math, requests, sqlite3, psycopg
from urllib.parse import urlencode
import os, sys, getpass, bcrypt

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

HASH = os.environ.get("RAI_HASH")
if not HASH:
    sys.exit("Missing SCRIPT_PASS_HASH env var.")

pw = getpass.getpass("Password: ").encode()
if not bcrypt.checkpw(pw, HASH.encode()):
    sys.exit("Access denied.")


''''
db password: Loated47!

-Purpose of this doc is to load our database with the pubmed publications, eventually update DB once per day
with new studies. 
- Eventually, this will be moved somewhere but for the sake of organization im keeping it in this repo for now.
-This file will also be in the .gitignore

'''

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
PMED_API_KEY = os.getenv("7efdbfda18b23dc38b740fee2393bc915c09")  # optional, raises limit to ~10 rps
TOOL = "research-ai"                      # per NCBI guidance
EMAIL = "shochak2016@gmail.com"             # per NCBI guidance

TABLE = "public.research"

def search_papers(mindate: str, maxdate: str, term="all[sb]"):
    """Find PMIDs by publication date (YYYY/MM/DD). Returns count + WebEnv/query_key."""
    params = {
    "db": "pubmed", 
    "term": term, 
    "retmode": "json",
    "usehistory": "y", 
    "datetype": "pdat", # publication date
    "mindate": mindate, 
    "maxdate": maxdate,
    "tool": TOOL, 
    "email": EMAIL,
    "retmax": 0 
    }

    if PMED_API_KEY:
        params["api_key"] = PMED_API_KEY
    r = requests.get(f"{BASE}/esearch.fcgi", params=p, timeout=60)
    r.raise_for_status()
    js = r.json()["esearchresult"] #creates python dict from parse
    return int(js["count"]), js["webenv"], js["querykey"] #return values from corresponding keys for the dict

def search_abstracts():
    pass   