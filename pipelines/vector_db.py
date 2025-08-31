'''
File is meant to pull from DB, embed content, and store into vector DB. Should only be done once, daily_sync.py
will handle the updates.
'''

import os, sys, bcrypt, getpass

HASH = os.environ.get("RAI_HASH")
if not HASH:
    sys.exit("Missing SCRIPT_PASS_HASH env var.")

pw = getpass.getpass("Password: ").encode()
if not bcrypt.checkpw(pw, HASH.encode()):
    sys.exit("Access denied.")