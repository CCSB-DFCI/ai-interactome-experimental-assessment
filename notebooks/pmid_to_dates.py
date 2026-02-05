#!/usr/bin/env python3

"""
TODO:
    - no need for title and journal in output
"""

import csv, sys, time
from xml.etree import ElementTree as ET

import requests


EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
HEADERS = {"User-Agent": "pmid-to-dates/1.0 (contact: your_email@example.com)"}


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]


def backoff_request(method, url, **kwargs):
    for i in range(6):
        try:
            r = requests.request(method, url, timeout=120, headers=HEADERS, **kwargs)
            if r.status_code == 200:
                return r
        except Exception:
            pass
        time.sleep(min(8.0, 0.5 * (2 ** i)))
    raise RuntimeError(f"Request failed after retries: {url}")

import random

RETRY_STATUS = {429, 500, 502, 503, 504}

def backoff_request_post(url, params=None, data=None, max_tries=10):
    """POST with exponential backoff + jitter, honoring Retry-After."""
    for i in range(max_tries):
        try:
            r = requests.post(url, params=params, data=data, headers=HEADERS, timeout=120)
            if r.status_code == 200:
                return r
            # If server asks us to slow down
            if r.status_code in RETRY_STATUS:
                retry_after = r.headers.get("Retry-After")
                if retry_after:
                    try:
                        time.sleep(float(retry_after))
                    except Exception:
                        pass
            # exponential backoff + jitter
            time.sleep(min(8.0, 0.5 * (2 ** i)) + random.uniform(0, 0.25))
        except Exception:
            time.sleep(min(8.0, 0.5 * (2 ** i)) + random.uniform(0, 0.25))
    raise RuntimeError(f"Request failed after retries: {url}")


def esummary_paged(webenv, query_key, total, page_size=10000, api_key=None):
    url = f"{EUTILS}/esummary.fcgi"
    ret = 0
    while ret < total:
        retmax = min(page_size, total - ret)
        params = {
            "db": "pubmed",
            "query_key": query_key,
            "WebEnv": webenv,
            "retstart": ret,
            "retmax": retmax,
            "retmode": "xml",
        }
        if api_key:
            params["api_key"] = api_key

        # retry with backoff
        ok = False
        for i in range(6):
            try:
                r = requests.get(url, params=params, headers=HEADERS, timeout=120)
                if r.status_code == 200:
                    root = ET.fromstring(r.text)
                    yield root
                    ok = True
                    break
                else:
                    backoff_sleep(i)
            except Exception:
                backoff_sleep(i)

        if not ok:
            raise RuntimeError(f"ESummary failed at retstart={ret} after retries")

        # rate limiting: with API key you can go up to ~10 req/s, without ~3 req/s.
        # We’re conservative to play nice.
        time.sleep(0.12 if api_key else 0.35)
        ret += retmax


def parse_docsums(xml_text):
    root = ET.fromstring(xml_text)
    rows = []
    for docsum in root.findall(".//DocSum"):
        pmid = docsum.findtext("Id") or ""
        fields = {}
        for item in docsum.findall("Item"):
            name = item.attrib.get("Name")
            if name in ("PubDate", "EPubDate", "SortPubDate", "Title", "FullJournalName"):
                fields[name] = (item.text or "").strip()
        pubdate = fields.get("PubDate", "")
        epubdate = fields.get("EPubDate", "")
        sortdate = fields.get("SortPubDate", "")
        title = fields.get("Title", "")
        journal = fields.get("FullJournalName", "")
        best_date = pubdate or epubdate or sortdate
        rows.append({
            "pmid": pmid,
            "pubdate": pubdate,
            "epubdate": epubdate,
            "sortpubdate": sortdate,
            "best_date": best_date,
            "title": title,
            "journal": journal,
        })
    return rows


def get_pubmed_to_date_mapping(pmids, api_key=None, chunk_size=10000, output_path="../data/processed/pmid_dates.csv"):
    #api_key = os.environ.get("NCBI_API_KEY")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["pmid","pubdate","epubdate","sortpubdate","best_date","title","journal"])
        w.writeheader()

        total = len(pmids)
        done = 0
        for block in chunks(pmids, chunk_size):
            params = {
                "db": "pubmed",
                "retmode": "xml",
            }
            if api_key:
                params["api_key"] = api_key

            # send PMIDs in the POST body (avoids long URLs)
            data = {"id": ",".join(block)}

            r = backoff_request_post(f"{EUTILS}/esummary.fcgi", params=params, data=data)
            rows = parse_docsums(r.text)
            # Write in the order of the input for this block
            order = {p:i for i,p in enumerate(block)}
            rows.sort(key=lambda r: order.get(r["pmid"], 10**12))
            for row in rows:
                w.writerow(row)

            done += len(block)
            print(f"Fetched {done}/{total}", file=sys.stderr)

            # Be polite to NCBI (even with an API key)
            time.sleep(0.12 if api_key else 0.35)