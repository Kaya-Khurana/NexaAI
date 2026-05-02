import re
from rag.doc_parser import ID_RE

text = "Bill No 1/2506. Consumer No 12345. Invoice: INV-2024-001."

print(f"ID Matches: {ID_RE.findall(text)}")

q = "bill no"
# Simple check for ID extraction near keyword
def extract_near(text, kw, pat):
    for m in re.finditer(re.escape(kw), text.lower()):
        snippet = text[max(0, m.start()-50):min(len(text), m.end()+50)]
        matches = pat.findall(snippet)
        if matches: return matches
    return []

print(f"Extraction near 'bill': {extract_near(text, 'bill', ID_RE)}")
