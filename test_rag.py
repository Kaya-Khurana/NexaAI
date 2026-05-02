from rag.rag_engine import RAGEngine

text = """AGREEMENT TO SELL / PROPERTY SALE AGREEMENT
This Agreement to Sell is made and executed on this 2nd day of May, 2026 at Ahmedabad, Gujarat.
BETWEEN
Mr. Rajesh Kumar Sharma, Son of Late Mahendra Sharma, Residing at 24, Shree Residency, Satellite Road, Ahmedabad, Gujarat,
Holding Aadhaar No. XXXX-XXXX-4587 / PAN No. AKTPS4587M,
Hereinafter referred to as the SELLER.
AND
Ms. Priya Mehta, Daughter of Suresh Mehta, Residing at 12, Green Park Society, Navrangpura, Ahmedabad, Gujarat,
Holding Aadhaar No. XXXX-XXXX-7721 / PAN No. BCDPM7721K,
Hereinafter referred to as the BUYER.
The Seller agrees to sell the property situated at Flat No. 302, Building Name: Shivalik Heights, Navrangpura, Ahmedabad for a total sale consideration of Rs. 85,00,000 (Rupees Eighty Five Lakhs Only).
The Buyer has paid an advance of Rs. 5,00,000 as token money on 2nd May 2026.
The Seller is the absolute and lawful owner and in peaceful possession of the property.
"""

r = RAGEngine()
r.load_data("test_sid", text, "Agreement.pdf")

tests = [
    ("who is seller",       "Rajesh"),
    ("who is buyer",        "Priya"),
    ("who is buying",       "Priya"),
    ("what is the price",   "85"),
    ("when was this signed","2026"),
    ("where is the property","Ahmedabad"),
    ("locatuo",             "Ahmedabad"),
]

passed = 0
for q, expected in tests:
    ans = r.generate_answer("test_sid", q)
    ok = expected.lower() in ans.lower()
    passed += ok
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] Q: {q:<30s}  A: {ans[:80]}")

print(f"\nAccuracy: {passed}/{len(tests)} = {passed*100//len(tests)}%")
