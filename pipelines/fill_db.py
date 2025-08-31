import os, time, datetime as dt, math, requests, sqlite3, psycopg
from urllib.parse import urlencode
import os, sys, getpass, bcrypt
from datetime import date, timedelta
from supabase import create_client

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from lxml import etree

HASH = os.environ.get("RAI_HASH")
if not HASH:
    sys.exit("Missing SCRIPT_PASS_HASH env var.")

pw = getpass.getpass("Password: ").encode()
if not bcrypt.checkpw(pw, HASH.encode()):
    sys.exit("Access denied.")



BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
PMED_API_KEY = os.getenv("7efdbfda18b23dc38b740fee2393bc915c09")  # optional, raises limit to ~10 rps
TOOL = "research-ai"                      # per NCBI guidance
EMAIL = "shochak2016@gmail.com"             # per NCBI guidance

TABLE = "public.research"

CUR_MONTH = "08"
MONTHS = {m:i for i,m in enumerate(["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],1)}
SEASONS = {"Spring":"03","Summer":"06","Fall":"09","Autumn":"09","Winter":"12",
           "1st Quarter":"03","2nd Quarter":"06","3rd Quarter":"09","4th Quarter":"12"}
''''
db password: Loated47!

-Purpose of this doc is to load our database with the pubmed publications, eventually update DB once per day
with new studies. 
- Eventually, this will be moved somewhere but for the sake of organization im keeping it in this repo for now.
-This file will also be in the .gitignore

'''
def search_papers(datetype, days=30):
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


'''
Function that is used to convert XML format into dictionary
'''

def clean_xml(elem) -> str: #takes tags out of an XML line/element
    if elem is not None:
        return "".join(elem.itertext()).strip()
    else: 
        return ""

def month_norm(m):  #returns what the month is supposed to be 
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
    pmid = clean_xml(elem.find(".//MedlineCitation/PMID"))
    title = clean_xml(elem.find(".//Article/ArticleTitle"))

    parts = []
    for ab in elem.findall(".//Article/Abstract/AbstractText"):
        t = clean_xml(ab)
        if not t: 
            continue
        label = ab.get("Label")
        parts.append(f"{label}: {t}" if label else t)
    abstract = "\n".join(parts) or None

    y = clean_xml(elem.find(".//Article/ArticleDate/Year")) or clean_xml(elem.find(".//JournalIssue/PubDate/Year"))
    if not y:
        return None
    m = clean_xml(elem.find(".//Article/ArticleDate/Month")) or _tx(elem.find(".//JournalIssue/PubDate/Month"))
    d = clean_xml(elem.find(".//Article/ArticleDate/Day")) or _tx(elem.find(".//JournalIssue/PubDate/Day"))
    pub_date = f"{y}-{month_norm(m)}-{(d or '01').zfill(2)}"

    authors = []
    for auth in elem.findall(".//Article/AuthorList/Author"):
        coll = clean_xml(auth.find("CollectiveName"))
        if coll:
            authors.append(coll)
            continue
        nm = " ".join(x for x in [clean_xml(auth.find("LastName")), clean_xml(auth.find("Initials"))] if x)
        if nm:
            authors.append(nm)

    if not (pmid and title):
        return None

    return {
        "pmid": int(pmid),
        "doc": {
            "title": title,
            "abstract": abstract,
            "authors": authors,
            "pub_date": pub_date
        },
        "pub_date": pub_date
    }

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

'''
Function that actually feeds things into database as entries
'''

def add_past_month(sb, table: str, api_key: str = PMED_API_KEY, retmax: int = 400, chunk: int = 200):
    count, webenv, qk = search_papers("pdate")
    if not count or not webenv or not qk:
        return
    retstart, buf = 0, []
    while retstart < count:
        for row in fetch_lines(webenv, qk, retstart, retmax):
            if not row["doc"].get("abstract"):
                continue
            buf.append(row)
            if len(buf) >= chunk:
                sb.table(table).upsert(buf, on_conflict="pmid").execute()
                buf.clear()
        if buf:
            sb.table(table).upsert(buf, on_conflict="pmid").execute()
            buf.clear()
        retstart += retmax
        time.sleep(0.11 if api_key else 0.34)



def main():
    sb = create_client(os.environ["SUPABASE_URL"], os.environ[HASH])
    tool  = os.environ.get(BASE, "my_script")
    email = os.environ.get(EMAIL, "me@example.com")
    api_key = os.environ.get(PMED_API_KEY)
    table = "pubmed_articles"
    add_past_month(sb, table, tool, email, api_key, retmax=400, chunk=200, require_abstract=True)

if __name__ == "__main__":
    main()
