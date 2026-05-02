import re
from rag.doc_parser import DATE_RE, VALUE_RE, AMOUNT_RE

text = "This Agreement is made on 2nd day of May, 2026. House No. 1450. Rate is Rs. 85,00,000. Plot 7721."

print("Testing Patterns:")
print(f"Dates: {DATE_RE.findall(text)}")
print(f"Values: {VALUE_RE.findall(text)}")
print(f"Amounts: {AMOUNT_RE.findall(text)}")

q = "rate not date"
clean_q = re.sub(r'\bnot\s+\w+\b', '', q.lower())
print(f"Cleaned Q: '{clean_q}'")
