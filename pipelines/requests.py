import os, time, datetime as dt, math, requests, sqlite3
from urllib.parse import urlencode

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
API_KEY = os.getenv("NCBI_API_KEY")  # optional, raises limit to ~10 rps
TOOL = "yourapp"                      # per NCBI guidance
EMAIL = "you@example.com"             # per NCBI guidance