"""
NexaAI Universal Document Parser
Detects structure in ANY document — resume, legal, spec sheet,
manual, contract, report, question paper, medical record, etc.
"""
import re


# ── Levenshtein edit distance ──────────────────────────────────────
def edit_distance(a: str, b: str) -> int:
    a, b = a.lower(), b.lower()
    if a == b: return 0
    if len(a) < len(b): a, b = b, a
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j]+1, row[j-1]+1, prev+(ca!=cb))
    return row[-1]

def closest_word(word: str, vocab: set, threshold: int = 2) -> str:
    w = word.lower()
    if w in vocab: return w
    if len(w) < 3: return w
    best, best_d = w, threshold + 1
    for v in vocab:
        if abs(len(v) - len(w)) > threshold + 1: continue
        d = edit_distance(w, v)
        if d < best_d: best_d, best = d, v
    return best


# ── Universal patterns ────────────────────────────────────────────

TITLE_RE = re.compile(
    r'\b(Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.|Shri|Smt\.|Er\.|Adv\.|CA|Col\.|Maj\.)\s+'
    r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,4})'
)
EMAIL_RE   = re.compile(r'[\w.+\-]+@[\w\-]+\.\w{2,}')
PHONE_RE   = re.compile(r'(?:\+\d{1,3}[\s\-]?)?\(?\d{3,5}\)?[\s\-]?\d{3,5}[\s\-]?\d{3,5}')
URL_RE     = re.compile(r'(?:https?://|www\.)\S+|linkedin\.com/\S+|github\.com/\S+', re.I)
DATE_RE    = re.compile(
    r'\b(?:\d{1,2}(?:st|nd|rd|th)?\s+(?:day\s+of\s+)?'
    r'(?:January|February|March|April|May|June|July|August|September|October|November|December)'
    r'(?:,?\s*(?:19|20)\d{2})?'
    r'|\d{1,2}[\/\-\.]\d{1,2}[\/\-\.](?:19|20)?\d{2}'
    r'|\b(?:19|20)\d{2}\b)\b', re.I
)
# Numbers with units or currency — covers %, Rs, $, kg, etc.
VALUE_RE   = re.compile(
    r'(?:Rs\.?|INR|USD|\$|€|£|¥)\s*[\d,]+(?:\.\d+)?(?:\s*(?:lakhs?|crores?|thousands?|millions?))?'
    r'|[\d,]+(?:\.\d+)?\s*(?:%|percent|kg|g|lb|km|m|cm|mm|MHz|GHz|V|A|W|kW|MW|lakhs?|crores?|thousands?|millions?)'
    r'|(?:\d{1,3}(?:,\d{2,3})+)(?:\.\d+)?'   # Comma separated large numbers (e.g. 85,00,000 or 1,23,456)
    r'|\b\d{5,}\b',                          # Or at least 5 digits (e.g. 50000)
    re.I
)
AMOUNT_RE  = re.compile(r'(?:Rs\.?|INR|USD|\$|€|£|¥)\s*[\d,]+(?:\.\d+)?(?:\s*(?:lakhs?|crores?))?', re.I)
PCT_RE     = re.compile(r'\d+(?:\.\d+)?\s*%')
ID_RE      = re.compile(r'\b(?:[A-Z0-9]{1,10}(?:[/\-][A-Z0-9]{1,10}){1,3}|[A-Z0-9]{5,25})\b')
DEFINITION_RE = re.compile(
    r'(["\']?[A-Z][A-Za-z\s]{1,40}["\']?)\s+'
    r'(?:means?|is\s+defined\s+as|shall\s+mean|refers?\s+to|is\s+known\s+as|stands?\s+for)\s+'
    r'([^.]{5,200})', re.I
)

# Section header: line that is ALL CAPS or Title Case followed by newline,
# possibly with numbering prefix like "1.", "1.1", "CHAPTER 1", etc.
SECTION_HDR_RE = re.compile(
    r'^(?:\d+(?:\.\d+)*\.?\s+)?'           # optional numbering
    r'([A-Z][A-Z\s\-]{2,50}|[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,5})'  # header text
    r'\s*:?\s*$',
    re.MULTILINE
)

# Legal role patterns
LEGAL_ROLES = {
    "seller":    ["seller","vendor","lessor","transferor","assignor"],
    "buyer":     ["buyer","purchaser","vendee","lessee","assignee","transferee"],
    "landlord":  ["landlord","lessor","licensor"],
    "tenant":    ["tenant","lessee","licensee"],
    "borrower":  ["borrower","debtor","mortgagor"],
    "lender":    ["lender","creditor","mortgagee","financier"],
    "witness":   ["witness","attesting witness"],
    "agent":     ["agent","broker","representative","attorney"],
    "guarantor": ["guarantor","surety"],
    "employer":  ["employer","company","organization","organisation"],
    "employee":  ["employee","worker","staff"],
    "contractor":["contractor","vendor","service provider"],
    "client":    ["client","customer","end user"],
    "party":     ["party","parties"],
}

# Q_TO_SECTION — maps question keywords to section name patterns
Q_TO_SECTION = {
    # Resume / academic
    "education":    ["education","qualification","academic","academics"],
    "collage":      ["education","qualification","academic"],
    "college":      ["education","qualification","academic"],
    "university":   ["education","qualification","academic"],
    "degree":       ["education","qualification","academic"],
    "skill":        ["skills","technical skills","competencies","technologies"],
    "skills":       ["skills","technical skills","competencies","technologies"],
    "project":      ["projects","project work","portfolio"],
    "projects":     ["projects","project work","portfolio"],
    "experience":   ["experience","work experience","employment history","professional experience"],
    "internship":   ["internship","internships","training","industrial training"],
    "certification":["certification","certifications","certificates","courses"],
    "certificate":  ["certification","certifications","certificates","courses"],
    "achievement":  ["achievements","awards","honors","accomplishments"],
    "language":     ["skills","technical skills","languages"],
    "contact":      ["contact","contact information","personal information","personal details"],
    "objective":    ["objective","career objective","summary","professional summary","profile"],
    "summary":      ["objective","summary","professional summary","about"],
    # Legal / contractual
    "clause":       ["terms","conditions","covenants","clauses","provisions"],
    "term":         ["terms","conditions","definitions"],
    "condition":    ["conditions","terms","covenants"],
    "payment":      ["payment","consideration","financial terms"],
    "schedule":     ["schedule","annexure","appendix","exhibit"],
    "definition":   ["definitions","interpretation","meaning"],
    "warranty":     ["warranties","representations","guarantees"],
    "obligation":   ["obligations","covenants","duties","responsibilities"],
    # Technical / spec sheets
    "specification":["specifications","technical specifications","spec","specs"],
    "feature":      ["features","key features","highlights"],
    "requirement":  ["requirements","system requirements","prerequisites"],
    "installation": ["installation","setup","getting started"],
    "usage":        ["usage","how to use","operation","operating instructions"],
    "parameter":    ["parameters","settings","configuration"],
    "dimension":    ["dimensions","physical dimensions","measurements","size"],
    "material":     ["materials","composition","construction"],
    "safety":       ["safety","warnings","cautions","precautions"],
    # General
    "introduction": ["introduction","overview","abstract","background","about"],
    "conclusion":   ["conclusion","summary","closing remarks"],
    "recommendation":["recommendations","suggestions"],
    "methodology":  ["methodology","methods","approach","procedure"],
    "result":       ["results","findings","outcomes","observations"],
    "reference":    ["references","bibliography","citations"],
}

PRONOUN_WORDS = frozenset({"she","he","they","her","him","them","his","hers","their","it","its"})


class DocumentParser:

    @staticmethod
    def detect_type(text: str) -> str:
        tl = text.lower()
        score = {
            "resume": sum([
                bool(re.search(r'\b(objective|career\s+objective)\b', tl)),
                bool(re.search(r'\b(bca|mca|btech|mtech|bsc|msc|be|ba|mba|phd)\b', tl)),
                bool(re.search(r'\b(internship|intern)\b', tl)),
                "github" in tl or "linkedin" in tl,
                bool(re.search(r'\bskills\b', tl)),
                bool(re.search(r'\b(experience|worked\s+at|working\s+at)\b', tl)),
            ]),
            "legal": sum([
                "hereinafter" in tl,
                "whereas" in tl,
                "agreement" in tl and "party" in tl,
                bool(re.search(r'\b(the\s+seller|the\s+buyer|the\s+lessee|the\s+lessor)\b', tl)),
                bool(re.search(r'referred\s+to\s+as', tl)),
                bool(re.search(r'\b(clause|section)\s+\d+', tl)),
            ]),
            "technical": sum([
                bool(re.search(r'\b(specification|datasheet|data\s+sheet)\b', tl)),
                bool(re.search(r'\b(voltage|current|power|frequency|resistance|temperature)\b', tl)),
                bool(re.search(r'\b(model\s+no|part\s+no|serial\s+no)\b', tl)),
                bool(re.search(r'\b(installation|configuration|setup)\b', tl)),
                bool(re.search(r'\b(mm|cm|kg|mhz|ghz|rpm|psi|bar)\b', tl)),
            ]),
            "academic": sum([
                bool(re.search(r'\b(question\s+paper|exam|examination)\b', tl)),
                bool(re.search(r'\b(marks?|score|grade)\b', tl)),
                bool(re.search(r'\b(chapter|unit|module|section)\s+\d+', tl)),
                bool(re.search(r'\b(hypothesis|methodology|literature\s+review)\b', tl)),
            ]),
        }
        best = max(score, key=score.get)
        return best if score[best] >= 2 else "general"

    @staticmethod
    def parse_sections(text: str) -> dict:
        """
        Universal section parser — works on any document.
        Detects headers as UPPERCASE or Title-Case-Only lines.
        """
        sections = {}
        current_header = "preamble"
        current_content = []

        for line in text.split('\n'):
            stripped = line.strip()
            if not stripped:
                continue

            # Check if this line looks like a section header
            is_header = False
            # Rule 1: All uppercase, 3–60 chars, no digits, not a full sentence
            if (stripped.isupper() and 3 <= len(stripped) <= 60
                    and not re.search(r'\d', stripped)
                    and not stripped.endswith('.')):
                is_header = True
            # Rule 2: Title case, short, no period, ends possibly with colon
            elif (re.match(r'^[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,5}:?\s*$', stripped)
                    and len(stripped) <= 50):
                is_header = True
            # Rule 3: Numbered section like "1.", "1.1", "Chapter 1:"
            elif re.match(r'^\d+(?:\.\d+)*\.?\s+[A-Z]', stripped) and len(stripped) <= 80:
                is_header = True

            if is_header:
                if current_content:
                    sections[current_header.lower().strip(':')] = '\n'.join(current_content)
                current_header = stripped.rstrip(':')
                current_content = []
            else:
                current_content.append(stripped)

        if current_content:
            sections[current_header.lower().strip(':')] = '\n'.join(current_content)

        return sections

    @staticmethod
    def find_main_name(text: str) -> str:
        """Find the primary person or entity name in any document."""
        lines = [l.strip() for l in text.split('\n') if l.strip()]

        # Strategy 1: Titled name anywhere in first 10 lines
        for line in lines[:10]:
            m = TITLE_RE.search(line)
            if m:
                return f"{m.group(1)} {m.group(2)}".strip()

        # Strategy 2: First short line that looks like a name (no digits/symbols)
        for line in lines[:5]:
            if (2 <= len(line.split()) <= 5
                    and line[0].isupper()
                    and not any(c in line for c in ['@','/',':','|','\\','.com'])
                    and not re.search(r'\d', line)):
                return line

        # Strategy 3: Titled names anywhere in first 500 chars
        matches = TITLE_RE.findall(text[:500])
        if matches:
            return f"{matches[0][0]} {matches[0][1]}".strip()

        return ""

    @staticmethod
    def extract_all_entities(text: str) -> dict:
        """Universal entity extraction — no domain assumptions."""
        entities = {
            "names":        DocumentParser._extract_names(text),
            "dates":        list(dict.fromkeys(DATE_RE.findall(text)))[:15],
            "amounts":      list(dict.fromkeys(AMOUNT_RE.findall(text)))[:15],
            "percentages":  list(dict.fromkeys(PCT_RE.findall(text)))[:20],
            "emails":       list(dict.fromkeys(EMAIL_RE.findall(text)))[:5],
            "phones":       list(dict.fromkeys(PHONE_RE.findall(text[:2000])))[:5],
            "urls":         list(dict.fromkeys(URL_RE.findall(text)))[:5],
            "definitions":  DocumentParser._extract_definitions(text),
            "roles":        {},   # populated for legal docs only
        }
        return entities

    @staticmethod
    def _extract_names(text: str) -> list:
        found = TITLE_RE.findall(text)
        seen, out = set(), []
        for t, n in found:
            full = f"{t} {n}".strip()
            if full not in seen:
                seen.add(full); out.append(full)
        # Also try all-caps name at top
        first_line = text.split('\n')[0].strip()
        if (first_line.isupper() and 2 <= len(first_line.split()) <= 4
                and not re.search(r'\d', first_line)):
            if first_line not in out:
                out.insert(0, first_line)
        return out

    @staticmethod
    def _extract_definitions(text: str) -> dict:
        defs = {}
        for m in DEFINITION_RE.finditer(text):
            key = m.group(1).strip().strip('"\'').lower()
            val = m.group(2).strip()
            if 2 <= len(key.split()) <= 5 and len(val) > 5:
                defs[key] = val
        return defs

    @staticmethod
    def extract_role_map(text: str) -> dict:
        """Legal document: map role → person name."""
        mapping = {}
        chunk = text[:10000]
        for role, aliases in LEGAL_ROLES.items():
            for alias in aliases:
                pat = re.compile(
                    r'(?:referred\s+to\s+as|hereinafter\s+called|hereinafter\s+referred\s+to\s+as)\s+'
                    r'["\u201c\u2018\']?\s*(?:the\s+)?' + re.escape(alias) + r'\b', re.I)
                for m in pat.finditer(chunk):
                    before = chunk[max(0, m.start()-900):m.start()]
                    names = TITLE_RE.findall(before)
                    if names:
                        t, n = names[-1]
                        mapping[role] = f"{t} {n}".strip()
                        break
                if role in mapping: break
        return mapping

    @staticmethod
    def extract_value_in_context(text: str, context_words: list,
                                  value_pattern: re.Pattern,
                                  window: int = 300) -> list:
        """
        Core universal extractor: find VALUE_PATTERN near CONTEXT_WORDS.
        E.g. 'percentage of MCA' → finds '%' near 'MCA'.
        Works for any domain: specs, legal, resumes, etc.
        """
        results = []
        tl = text.lower()
        for cw in context_words:
            for m in re.finditer(re.escape(cw.lower()), tl):
                start = max(0, m.start() - window)
                end   = min(len(text), m.end() + window)
                snippet = text[start:end]
                vals = value_pattern.findall(snippet)
                results.extend([v.strip() for v in vals if v.strip()])
        return list(dict.fromkeys(results))
