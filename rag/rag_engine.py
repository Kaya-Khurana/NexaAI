"""
NexaAI Universal RAG Engine v5
================================
Pure Python. No API keys. No model downloads.
Works for ANY document type — resume, legal, spec sheet,
medical, technical manual, contract, question paper, etc.

Query pipeline:
  1.  Typo correction  (Levenshtein, protected keywords never corrupted)
  2.  Context-value extraction  ("percentage of MCA" → 85.03%)
  3.  Named entity lookup  ("name" → primary person/entity name)
  4.  Definition lookup  ("what is X" → "X means...")
  5.  Legal role lookup  (for contracts/agreements)
  6.  Section direct lookup  (maps question intent → doc section)
  7.  BM25 retrieval + IDF-weighted sentence scoring  (universal fallback)
"""

import re, os, math, pickle
from collections import Counter, defaultdict
from rag.doc_parser import (
    DocumentParser, Q_TO_SECTION, LEGAL_ROLES, PRONOUN_WORDS,
    edit_distance, closest_word,
    TITLE_RE, EMAIL_RE, PHONE_RE, URL_RE,
    DATE_RE, VALUE_RE, AMOUNT_RE, PCT_RE, ID_RE,
)
from rag import openrouter


STOPWORDS = frozenset({
    "a","an","the","is","are","was","were","be","been","have","has","had",
    "do","does","did","will","would","could","should","may","might","to",
    "of","in","for","on","with","at","by","from","as","and","or","but",
    "not","this","that","it","its","i","me","my","you","your","he","she",
    "her","his","they","their","we","our","us","who","what","when","where",
    "how","which","there","here","any","all","some","get","tell","give",
    "make","let","please","can","about","just","so","than","very","their",
    "those","these","then","into","also","more","most","other","such",
    "each","been","than","too","very","just",
})

# Words whose meaning we must NEVER corrupt via typo correction
PROTECTED = frozenset({
    "name","date","price","cost","amount","rate","rent","address","location","contact",
    "skill","skills","project","projects","education","degree","collage",
    "locatuo","experience","internship","email","phone","linkedin","github",
    "seller","buyer","party","witness","agent","tenant","landlord","client",
    "employee","employer","contractor","borrower","lender","guarantor",
    "percentage","percent","salary","weight","height","speed","temperature",
    "voltage","current","power","frequency","capacity","version","model",
    "number","quantity","size","dimension","length","width","depth","area",
    "when","where","who","what","which","how","why","tell","list","show",
    "all","give","find","describe","explain","define","compare","summary",
})

# Universal synonym / expansion dictionary
SYNONYMS = {
    # Value types
    "percentage": ["percent","%","score","rate","cgpa","gpa","marks","grade"],
    "percent":    ["%","percentage","score","marks","grade"],
    "price":      ["cost","amount","rate","fee","charge","tariff","consideration","rupees","rs","inr","usd"],
    "cost":       ["price","amount","rate","fee","charge","consideration","rupees"],
    "salary":     ["ctc","package","compensation","remuneration","pay","stipend"],
    "quantity":   ["count","number","amount","total","units","pieces"],
    "weight":     ["mass","kg","grams","lbs","pounds","tonnes"],
    "dimension":  ["size","length","width","height","depth","diameter","radius"],
    "speed":      ["velocity","rpm","mph","kph","frequency","rate"],
    "temperature":["temp","celsius","fahrenheit","kelvin","degrees"],
    "voltage":    ["volts","v","potential","emf"],
    "power":      ["watts","watt","kw","mw","hp","horsepower"],
    # Person / identity
    "name":       ["called","titled","named","designation","known as","person","individual"],
    "who":        ["person","individual","party","name","candidate","employee","student"],
    "contact":    ["email","phone","mobile","linkedin","github","address","reach"],
    "email":      ["mail","contact","address","gmail","yahoo"],
    "phone":      ["mobile","cell","number","contact","tel","telephone"],
    # Resume / academic
    "education":  ["qualification","degree","university","college","studies","academic","studied"],
    "collage":    ["college","university","education","degree","studied","academic"],
    "degree":     ["mca","bca","btech","mtech","bachelor","master","graduation","pg","ug","phd","diploma"],
    "skill":      ["technology","framework","language","tools","expertise","proficiency","competency"],
    "project":    ["projects","built","developed","created","implemented","designed","work"],
    "experience": ["internship","work","company","role","position","job","employment","worked"],
    "internship": ["intern","training","apprenticeship","work experience"],
    "certification":["certificate","course","training","credential","license"],
    "achievement":["award","honor","accolade","recognition","accomplishment"],
    # Legal / contractual
    "buying":     ["buyer","purchaser","purchase"],
    "selling":    ["seller","vendor","sale"],
    "sold":       ["seller","vendor","transferred"],
    "purchased":  ["buyer","purchaser"],
    "agreement":  ["contract","deed","memorandum","mou","arrangement"],
    "clause":     ["term","condition","provision","article","section"],
    "payment":    ["amount","consideration","price","fee","compensation"],
    # Technical
    "specification":["spec","specs","parameter","feature","rating","datasheet"],
    "model":      ["version","type","variant","part number","product"],
    "installation":["setup","install","configure","deploy"],
    "requirement":["prerequisite","dependency","needs","must","should"],
    # Time / date
    "date":       ["day","month","year","when","signed","executed","issued","expiry"],
    "duration":   ["period","tenure","term","length","time","months","years"],
    # Location
    "location":   ["address","situated","place","city","state","country","region"],
    "locatuo":    ["location","address","situated","place","city","flat","property"],
    "address":    ["situated","place","city","state","flat","plot","building"],
}

# ── BM25 ──────────────────────────────────────────────────────────
class BM25:
    K1, B = 1.5, 0.75
    def __init__(self, chunks):
        self.chunks = chunks
        self.tok    = [re.findall(r'[a-z0-9]+', c.lower()) for c in chunks]
        N = len(chunks)
        self.avg_dl = sum(len(t) for t in self.tok) / max(N, 1)
        df = defaultdict(int)
        for t in self.tok:
            for w in set(t): df[w] += 1
        self.idf   = {w: math.log((N-f+.5)/(f+.5)+1) for w,f in df.items()}
        self.vocab = set(df)

    def query(self, tokens, top_k=10):
        k1,b,avg = self.K1, self.B, self.avg_dl
        scores = []
        for i, toks in enumerate(self.tok):
            dl = len(toks); tf = Counter(toks); s = 0.0
            for t in tokens:
                if t not in self.idf: continue
                f = tf.get(t,0)
                s += self.idf[t]*f*(k1+1)/(f+k1*(1-b+b*dl/max(avg,1)))
            scores.append((s, self.chunks[i]))
        scores.sort(reverse=True)
        return [(c,s) for s,c in scores[:top_k] if s > 0]


# ── Helpers ────────────────────────────────────────────────────────
def split_sentences(text: str) -> list:
    parts = re.split(r'(?<=[.!?])\s+|\n', text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 6]

def clean(t: str) -> str:
    t = re.sub(r'^[\s\-\u2022\u2013\*\u25ba]+', '', t, flags=re.MULTILINE)
    t = re.sub(r'\n+', ' ', t)
    t = re.sub(r'\s{2,}', ' ', t).strip()
    return t[0].upper() + t[1:] if t else t

def score_sentence(sent: str, tokens: list, idf: dict) -> float:
    toks = re.findall(r'[a-z0-9]+', sent.lower())
    if not toks: return 0.0
    tf = Counter(toks)
    s  = sum(idf.get(t, 1.0) * tf[t] for t in tokens if t in tf)
    s /= max(len(toks) ** 0.4, 1)
    if TITLE_RE.search(sent):  s += 2.5
    if AMOUNT_RE.search(sent): s += 1.5
    if PCT_RE.search(sent):    s += 1.5
    if DATE_RE.search(sent):   s += 1.0
    if ID_RE.search(sent):     s += 2.0
    return s

def expand_query(question: str) -> list:
    toks = [t for t in re.findall(r'[a-z0-9]+', question.lower()) if t not in STOPWORDS]
    out  = list(toks)
    for t in toks:
        out.extend(SYNONYMS.get(t, []))
    return list(dict.fromkeys(out))


# ══════════════════════════════════════════════════════════════════
#  MAIN ENGINE
# ══════════════════════════════════════════════════════════════════

class RAGEngine:
    SESSIONS_DIR = os.path.join("data", "sessions")

    def __init__(self):
        os.makedirs(self.SESSIONS_DIR, exist_ok=True)
        self.sessions = {}
        print("[RAGEngine] Universal QA Engine v5 ready.")

    # ── Persistence ───────────────────────────────────────────────
    def _pkl(self, sid):
        return os.path.join(self.SESSIONS_DIR, f"{sid}.pkl")
    def _save(self, sid):
        try:
            with open(self._pkl(sid), "wb") as f:
                pickle.dump(self.sessions[sid], f, protocol=4)
        except Exception as e: print(f"[RAGEngine] save warn: {e}")
    def _load(self, sid):
        p = self._pkl(sid)
        if not os.path.exists(p): return False
        try:
            with open(p, "rb") as f: self.sessions[sid] = pickle.load(f)
            return True
        except: return False
    def _sess(self, sid):
        if sid not in self.sessions: self._load(sid)
        return self.sessions.get(sid)

    # ── Chunking ──────────────────────────────────────────────────
    @staticmethod
    def _chunk(text, window=6, stride=3):
        paras = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        sents = []
        for p in paras:
            sents.extend(split_sentences(p))
        if not sents: return [text[:1500]]
        return [" ".join(sents[i:i+window]).strip()
                for i in range(0, len(sents), stride) if sents[i:i+window]]

    # ── Index ─────────────────────────────────────────────────────
    def load_data(self, sid, text, filename=""):
        chunks   = self._chunk(text)
        if not chunks: return False
        idx      = BM25(chunks)
        dp       = DocumentParser()
        doc_type = dp.detect_type(text)
        sections = dp.parse_sections(text)
        entities = dp.extract_all_entities(text)
        main_name= dp.find_main_name(text)
        if main_name and main_name not in entities["names"]:
            entities["names"].insert(0, main_name)
        if doc_type == "legal":
            entities["roles"] = dp.extract_role_map(text)

        self.sessions[sid] = dict(
            bm25=idx, chunks=chunks, text=text,
            filename=filename, doc_type=doc_type,
            sections=sections, entities=entities,
            main_name=main_name,
        )
        self._save(sid)
        print(f"[RAGEngine] {sid[:8]}: type={doc_type} | "
              f"sections={list(sections.keys())[:8]} | "
              f"name='{main_name}' | file='{filename}'")
        return True

    # ── Typo correction ───────────────────────────────────────────
    def _fix(self, question, vocab):
        return " ".join(
            w if w.lower() in STOPWORDS or len(w) < 4 or w.lower() in PROTECTED
            else (closest_word(w.lower(), vocab, 2) if closest_word(w.lower(), vocab, 2) != w.lower() else w)
            for w in question.split()
        )

    # ── STRATEGY 1: Context-value extraction ─────────────────────
    # "percentage of MCA"  → finds % near "MCA"
    # "price of model X"   → finds Rs/$ near "X"
    # "weight of product A"→ finds kg near "A"
    def _context_value(self, text, question):
        ql = question.lower()
        q_words = [w for w in re.findall(r'[a-z0-9]+', ql) if w not in STOPWORDS and len(w) > 2]

        # Detect value type requested
        value_pat, value_name = None, None
        if re.search(r'\b(percent(?:age)?|%|score|marks?|grade|cgpa|gpa)\b', ql):
            value_pat, value_name = PCT_RE, "Percentage"
        elif re.search(r'\b(price|cost|amount|fee|salary|ctc|pay|stipend|charge|consideration)\b', ql):
            value_pat, value_name = AMOUNT_RE, "Amount"
        elif re.search(r'\b(date|when|signed|executed|issued|born|joined)\b', ql):
            value_pat, value_name = DATE_RE, "Date"
        elif re.search(r'\b(no|number|id|code|ref|reference|folio|invoice|bill|receipt|consumer)\b', ql):
            value_pat, value_name = ID_RE, "Value"
        elif re.search(r'\b(number|count|total|quantity|how\s+many)\b', ql):
            value_pat = re.compile(r'\b\d+\b')
            value_name = "Count"

        if not value_pat:
            return None

        # Context words = non-stopword tokens (these are what we search near)
        context_words = [w for w in q_words
                         if not re.search(r'^(percent|percentage|price|cost|amount|date|when|total|count|number|no|id|code|how|many|marks|grade|score|fee|salary|bill|invoice|receipt)$', w)]

        if not context_words:
            # No context, return all matching values from doc
            vals = value_pat.findall(text)
            if vals:
                unique = list(dict.fromkeys(v.strip() for v in vals if v.strip()))[:5]
                return f"{value_name}: " + "; ".join(unique)
            return None

        # Search for value near each context word
        vals = DocumentParser.extract_value_in_context(text, context_words, value_pat, window=400)
        if vals:
            unique = list(dict.fromkeys(v.strip() for v in vals if v.strip()))[:5]
            return f"{value_name}: " + "; ".join(unique)
        return None

    # ── STRATEGY 2: Name lookup ───────────────────────────────────
    def _name_lookup(self, sess, question):
        ql = question.lower()
        q_words = set(re.findall(r'[a-z]+', ql)) - STOPWORDS

        # Direct "name" / "who" with no other context → main name
        intent_words = {"name","who","person","individual","candidate","author","owner","subject"}
        if q_words.issubset(intent_words | {"of","the","resume","document","this","pdf","file"}):
            name = sess.get("main_name", "")
            names = sess["entities"].get("names", [])
            if name: return name
            if names: return names[0]
            return None

        # "name of X" pattern — find name near context
        context = [w for w in q_words if w not in intent_words and len(w) > 2]
        if not context:
            names = sess["entities"].get("names", [])
            return ", ".join(names[:3]) if names else None

        # Find a name near the context word
        text = sess["text"]
        for cw in context:
            for m in re.finditer(re.escape(cw), text, re.I):
                snippet = text[max(0, m.start()-300):m.end()+300]
                found = TITLE_RE.findall(snippet)
                if found:
                    return f"{found[0][0]} {found[0][1]}".strip()

        names = sess["entities"].get("names", [])
        return names[0] if names else None

    # ── STRATEGY 3: Definition lookup ────────────────────────────
    def _definition_lookup(self, sess, question):
        ql = question.lower()
        if not re.search(r'\b(what\s+is|what\s+are|define|meaning|means|definition\s+of|explain)\b', ql):
            return None
        q_words = [w for w in re.findall(r'[a-z0-9]+', ql)
                   if w not in STOPWORDS and w not in {"what","is","are","define","meaning","means","definition","explain","of"}]
        if not q_words: return None
        defs = sess["entities"].get("definitions", {})
        for term, defn in defs.items():
            if any(qw in term for qw in q_words):
                return f"{term.title()}: {defn}"
        return None

    # ── STRATEGY 4: Legal role lookup ────────────────────────────
    def _role_lookup(self, sess, question):
        if sess.get("doc_type") not in ("legal","general"): return None
        ql = question.lower()
        roles = sess["entities"].get("roles", {})
        if not roles: return None
        verb_role = {"buying":"buyer","selling":"seller","sold":"seller","purchased":"buyer"}
        for verb, role in verb_role.items():
            if verb in ql:
                name = roles.get(role,"")
                if name: return f"{name} is the {role.title()}."
        for role, aliases in LEGAL_ROLES.items():
            if role in ql or any(a in ql for a in aliases):
                name = roles.get(role,"")
                if name: return f"{name} is the {role.title()}."
        return None

    # ── STRATEGY 5: Section content ───────────────────────────────
    def _section_lookup(self, sess, question):
        sections = sess.get("sections", {})
        if not sections: return None
        q_words = re.findall(r'[a-z]+', question.lower())

        # Direct section name match
        for word in q_words:
            # Check Q_TO_SECTION map
            targets = Q_TO_SECTION.get(word, [])
            for target in targets:
                for sec_name, sec_content in sections.items():
                    if target in sec_name.lower() and sec_content.strip():
                        return clean(sec_content)  # No truncation!
            # Also try edit-distance match against actual section names
            for sec_name in sections:
                if edit_distance(word, sec_name.lower()) <= 1 and sections[sec_name].strip():
                    return clean(sections[sec_name])

        return None

    # ── STRATEGY 6: BM25 + sentence scoring ───────────────────────
    def _bm25_answer(self, sess, q_tokens):
        ans, _, _ = self._bm25_answer_scored(sess, q_tokens)
        return ans

    def _bm25_answer_scored(self, sess, q_tokens) -> tuple:
        """Returns (answer_str, confidence_score, context_str)."""
        bm25: BM25 = sess["bm25"]
        results = bm25.query(q_tokens, top_k=10)
        if not results:
            return None, 0.0, ""
        context = " ".join(c for c, _ in results)
        sents   = split_sentences(context)
        scored  = sorted(
            [(score_sentence(s, q_tokens, bm25.idf), s) for s in sents],
            reverse=True,
        )
        if not scored or scored[0][0] <= 0:
            top = split_sentences(results[0][0])
            ans = clean(top[0]) if top else clean(results[0][0][:400])
            return ans, 0.1, context[:3000]
        best_score, best = scored[0]
        if len(best.split()) < 12 and len(scored) > 1:
            best = best + " " + scored[1][1]
        return clean(best), best_score, context[:3000]


    # ── Classify intent ───────────────────────────────────────────
    @staticmethod
    def _intent(q: str) -> str:
        ql = q.lower()
        # Handle negation (e.g. "rate not date") — remove negated words for intent
        clean_q = re.sub(r'\bnot\s+\w+\b', '', ql)
        
        if re.search(r'\b(who|whose|whom)\b', clean_q):                     return "who"
        if re.search(r'\b(how\s+many|how\s+much|count|number\s+of|total)\b', clean_q): return "count"
        if re.search(r'\b(where|address|location|situated|flat|plot|city)\b', clean_q): return "where"
        # Check Amount BEFORE When to handle "rate not date" or "price of 2026 model"
        if re.search(r'\b(price|cost|amount|fee|salary|pay|charge|consideration|rate|rent|value|rs|inr|usd)\b', clean_q): return "amount"
        if re.search(r'\b(when|date|year|time|signed|executed|born)\b', clean_q): return "when"
        if re.search(r'\b(percent(?:age)?|%|score|marks?|grade|cgpa|gpa)\b', clean_q): return "percent"
        if re.search(r'\b(list|all|every|enumerate|show\s+all)\b', clean_q): return "list"
        if re.search(r'\b(what\s+is|define|meaning|definition)\b', clean_q): return "definition"
        if re.search(r'\bname\b', clean_q):                                  return "name"
        if re.search(r'\b(no|number|id|code|ref|reference|folio|invoice|bill|receipt)\b', clean_q): return "id"
        return "general"

    # ── MASTER generate_answer ─────────────────────────────────────
    def generate_answer(self, sid: str, question: str,
                        openrouter_key: str = "") -> str:
        sess = self._sess(sid)
        if not sess:
            return "No document loaded. Please upload a PDF first."

        q_orig = question.strip()

        # Step 1: Force API (NVIDIA NIM / OpenRouter) if available
        # This provides maximum reasoning power and precision.
        if openrouter.is_available(openrouter_key):
            full_text = sess["text"]
            # For small documents (like bills/resumes), send the whole text.
            # For large docs, use BM25 to find the best 6000 chars.
            if len(full_text) < 10000:
                ctx = full_text
            else:
                tokens = [t for t in re.findall(r'[a-z0-9]+', q_orig.lower()) if t not in STOPWORDS]
                results = sess["bm25"].query(tokens, top_k=8)
                ctx = "\n\n".join(c for c, _ in results)

            prompt_q = (
                f"Question: {q_orig}\n"
                "Instruction: Provide a precise, direct answer. "
                "If the answer is a specific value (date, amount, ID, name), return ONLY that value. "
                "No conversational filler or markdown."
            )
            
            llm_ans = openrouter.ask_llm(
                context=ctx, question=prompt_q,
                api_key=openrouter_key,
                filename=sess.get("filename",""),
            )
            if llm_ans: return llm_ans

        # Step 2: Local RAG Fallback (if API fails or no key)
        bm25: BM25 = sess["bm25"]
        q = self._fix(q_orig, bm25.vocab)
        intent = self._intent(q)

        # Context-value extraction (Percentage, Amount, Date, Count, ID)
        if intent in ("percent","amount","when","count","id"):
            val = self._context_value(sess["text"], q)
            if val: return val

        # Name / Role lookup
        if intent in ("who", "name"):
            role_ans = self._role_lookup(sess, q)
            if role_ans: return role_ans
            ans = self._name_lookup(sess, q)
            if ans: return ans

        # Definition lookup
        if intent == "definition":
            ans = self._definition_lookup(sess, q)
            if ans: return ans

        # Section / Contact info
        ql = q.lower()
        if any(w in ql for w in ["contact","email","phone","mobile","linkedin","github"]):
            ents = sess["entities"]
            parts = []
            if ents.get("emails"): parts.append("Email: " + ents["emails"][0])
            if ents.get("phones"): parts.append("Phone: " + ents["phones"][0])
            if parts: return " | ".join(parts)

        # BM25 Fallback
        q_tokens = expand_query(q)
        bm25_ans, _, _ = self._bm25_answer_scored(sess, q_tokens)
        
        return bm25_ans if bm25_ans else "I couldn't find a precise answer in the document."
