import os, time, datetime as dt, math, requests, sqlite3, psycopg
from urllib.parse import urlencode



''''
db password: Loated47!

-Purpose of this doc is to load our database with the pubmed publications, eventually update DB once per day
with new studies. 
- Eventually, this will be moved somewhere but for the sake of organization im keeping it in this repo for now.
-This file will also be in the .gitignore

'''

BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
API_KEY = os.getenv("7efdbfda18b23dc38b740fee2393bc915c09")  # optional, raises limit to ~10 rps
TOOL = "research-ai"                      # per NCBI guidance
EMAIL = "shochak2016@gmail.com"             # per NCBI guidance


def delete_entries(sb, table: str):
    pass


def main():
    pass

if __name__ == "__main__":
    main()



