"""Shared SSH credentials for the windows/* helpers.
Set MSIGN_BOX_USER and MSIGN_BOX_PASSWORD in the environment (never in git)."""
import os
import sys

HOST = "172.20.10.11"
USER = os.environ.get("MSIGN_BOX_USER")
PW = os.environ.get("MSIGN_BOX_PASSWORD")

if not (USER and PW):
    sys.exit("Set MSIGN_BOX_USER and MSIGN_BOX_PASSWORD in your environment "
             "(credentials are intentionally not stored in this repo).")
