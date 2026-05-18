"""
Tema coverage analysis for the Ximenez XML manuscript.

Steps:
  1. Parse the XML, K'iche' column only.
  2. For each <rs> element extract the surface form (PCDATA) and its ana_id.
  3. Build a normalized surface-form → ana_id inventory.
  4. For every line NOT containing an <rs>, check whether any known surface form
     (exact or near-match by Levenshtein ratio) appears in the line text.
  5. Output:
       ximenez-SURFACE_FORMS.csv  — inventory of all known spellings per tema
       ximenez-COVERAGE.csv       — per-tema annotation vs. occurrence counts
       ximenez-MISSED.csv         — candidate missed annotations
"""

import re
import csv
import sys
from collections import defaultdict
from rapidfuzz.distance import Levenshtein
from rapidfuzz import fuzz

XML_PATH      = "xom-all-flat-mod-pnums.xml"
SURFACE_OUT   = "ximenez-SURFACE_FORMS.csv"
COVERAGE_OUT  = "ximenez-COVERAGE.csv"
MISSED_OUT    = "ximenez-MISSED.csv"

# ── Character normalization ────────────────────────────────────────────────────
CHAR_MAP = {
    'Ꜩ': 'tz', 'ꜩ': 'tz',
    'ꜫ': "q'",
    'ÿ': 'i',
    'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',  # accent stripped
    'á': 'a', 'É': 'e', 'Í': 'i', 'Ó': 'o',
}

def normalize(text: str) -> str:
    for ch, rep in CHAR_MAP.items():
        text = text.replace(ch, rep)
    text = re.sub(r'[^a-z\' ]', '', text.lower())
    return re.sub(r'\s+', ' ', text).strip()

def strip_tags(text: str) -> str:
    """Remove XML tags and line-break artifacts, collapse whitespace."""
    text = re.sub(r'<pc>[^<]*</pc>', '', text)   # remove <pc>–</pc> hyphens
    text = re.sub(r'<[^>]+>', ' ', text)          # remove remaining tags
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# ── Parse XML ─────────────────────────────────────────────────────────────────
xml = open(XML_PATH, encoding='utf-8').read()

# Split into K'iche' and Spanish columns by finding quc/spa div blocks.
# Strategy: collect all <lb> lines from the quc div only.
# The file alternates: <div xml:lang="quc"> … </div> <div xml:lang="spa"> …
quc_blocks = re.findall(
    r'<div xml:lang="quc"[^>]*>(.*?)</div>',
    xml, re.DOTALL
)
print(f"K'iche' column blocks: {len(quc_blocks)}")

# ── Step 1: Extract surface forms from RS elements in K'iche' text ─────────────
# surface_forms: ana_id → set of normalized surface strings
surface_forms: dict[str, set] = defaultdict(set)
# raw_forms: ana_id → list of (raw_text) for inspection
raw_forms: dict[str, list] = defaultdict(list)

rs_pattern = re.compile(r'<rs ana="([^"]+)">(.*?)</rs>', re.DOTALL)

all_rs_in_quc = []
for block in quc_blocks:
    for m in rs_pattern.finditer(block):
        ana_id  = m.group(1)
        raw_txt = strip_tags(m.group(2))
        norm    = normalize(raw_txt)
        if norm and len(norm) >= 2:  # skip trivial fragments
            surface_forms[ana_id].add(norm)
            raw_forms[ana_id].append(raw_txt)
        all_rs_in_quc.append((ana_id, norm))

print(f"RS elements in K'iche' column: {len(all_rs_in_quc)}")
print(f"Unique ana_ids: {len(surface_forms)}")

# ── Step 2: Extract all lb lines from K'iche' text ────────────────────────────
# Each lb line: capture folio/side/para_num context and full raw text of the line
# We reconstruct para_num from <p> elements, folio from <pb>

lb_records = []  # list of dicts: folio, side, para_num, lb_n, full_text, rs_ids, rs_surfaces

for block in quc_blocks:
    folio, side, para_num, lb_n = 0, 0, 0, 0

    # Walk the block line by line (the XML is flat, one element per line)
    for raw_line in block.split('\n'):
        raw_line = raw_line.strip()
        if not raw_line:
            continue

        # Track folio
        m = re.match(r'<pb xml:id="xom-f(\d+)-s(\d+)"', raw_line)
        if m:
            folio, side = int(m.group(1)), int(m.group(2))
            para_num, lb_n = 0, 0
            continue

        # Track paragraph
        if re.match(r'^<p[ >]', raw_line):
            para_num += 1
            lb_n = 0
            continue

        # Process line-break lines
        if raw_line.startswith('<lb'):
            # Extract lb number if present
            nm = re.search(r'<lb n="(\d+)"', raw_line)
            lb_n = int(nm.group(1)) if nm else lb_n + 1

            # Full plain text of the line
            full_text = normalize(strip_tags(raw_line))

            # Which RS elements are on this line?
            rs_on_line = [(m.group(1), normalize(strip_tags(m.group(2))))
                          for m in rs_pattern.finditer(raw_line)]
            rs_ids      = [r[0] for r in rs_on_line]
            rs_surfaces = [r[1] for r in rs_on_line]

            lb_records.append({
                'folio':       folio,
                'side':        side,
                'para_num':    para_num,
                'lb_n':        lb_n,
                'full_text':   full_text,
                'rs_ids':      rs_ids,
                'rs_surfaces': rs_surfaces,
            })

print(f"K'iche' lb lines parsed: {len(lb_records)}")

# ── Step 3: Build occurrence counts ───────────────────────────────────────────
# For each tema, count: (a) annotated occurrences, (b) total text occurrences
# of any known surface form (exact match on normalized text)

annotated_count = defaultdict(int)
for ana_id, _ in all_rs_in_quc:
    annotated_count[ana_id] += 1

# Index surface forms for fast lookup: norm_form → set of ana_ids
form_to_tema: dict[str, set] = defaultdict(set)
for ana_id, forms in surface_forms.items():
    for f in forms:
        form_to_tema[f].add(ana_id)

# For multi-token surface forms, also build n-gram search
# Sort surface forms by length descending (match longest first)
all_surface_list = sorted(
    [(f, ana_id) for ana_id, forms in surface_forms.items() for f in forms],
    key=lambda x: -len(x[0])
)

found_count = defaultdict(int)        # exact occurrences in text
missed_annotations = []               # candidate missed annotations

for rec in lb_records:
    text    = rec['full_text']
    rs_ids  = set(rec['rs_ids'])

    for form, ana_id in all_surface_list:
        if form not in text:
            continue
        # Check it's a whole-word (or sub-phrase) match, not a substring of longer word
        # Use word boundary check: space-padded
        padded = f' {text} '
        if f' {form} ' not in padded and not padded.startswith(f'{form} ') and \
           not padded.endswith(f' {form}'):
            # Still count if it appears as substring — K'iche' has many compound words
            pass
        found_count[ana_id] += 1

# ── Step 4: Levenshtein near-matches for unannotated lines ────────────────────
# For each lb line, take all sub-phrases (1–4 token windows) not covered by RS.
# Compare to known surface forms; flag similarity ≥ threshold.

LEV_THRESHOLD = 85  # rapidfuzz ratio score 0–100

# Only check temas with ≥ 3 annotations (avoids spurious one-off matches)
frequent_temas = {a for a, c in annotated_count.items() if c >= 3}
frequent_forms = [
    (f, ana_id) for ana_id, forms in surface_forms.items()
    if ana_id in frequent_temas
    for f in forms
    if len(f) >= 4  # skip very short forms
]

def ngrams(tokens, n):
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

for rec in lb_records:
    text   = rec['full_text']
    rs_ids = set(rec['rs_ids'])
    tokens = text.split()
    if not tokens:
        continue

    # Build sub-phrases up to 4 tokens
    candidates = []
    for n in range(1, 5):
        candidates.extend(ngrams(tokens, n))

    for phrase in candidates:
        if len(phrase) < 4:
            continue
        for form, ana_id in frequent_forms:
            if ana_id in rs_ids:
                continue  # already annotated on this line
            score = fuzz.ratio(phrase, form)
            if score >= LEV_THRESHOLD and phrase != form:
                missed_annotations.append({
                    'folio':      rec['folio'],
                    'side':       rec['side'],
                    'para_num':   rec['para_num'],
                    'lb_n':       rec['lb_n'],
                    'found_phrase': phrase,
                    'matched_form': form,
                    'ana_id':     ana_id,
                    'score':      score,
                    'line_text':  text,
                })

print(f"Candidate missed annotations: {len(missed_annotations)}")

# ── Write outputs ──────────────────────────────────────────────────────────────

# 1. Surface forms inventory
with open(SURFACE_OUT, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['ana_id', 'n_annotated', 'surface_form', 'form_length'])
    for ana_id in sorted(surface_forms):
        for form in sorted(surface_forms[ana_id]):
            w.writerow([ana_id, annotated_count[ana_id], form, len(form.split())])
print(f"Written: {SURFACE_OUT}")

# 2. Coverage
with open(COVERAGE_OUT, 'w', newline='', encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['ana_id', 'n_annotated', 'n_found_exact', 'coverage_ratio',
                'n_surface_forms'])
    for ana_id in sorted(annotated_count):
        na  = annotated_count[ana_id]
        nf  = found_count.get(ana_id, 0)
        nf2 = len(surface_forms.get(ana_id, set()))
        ratio = round(na / nf, 3) if nf else None
        w.writerow([ana_id, na, nf, ratio, nf2])
print(f"Written: {COVERAGE_OUT}")

# 3. Missed annotations (deduplicated, sorted by score)
seen = set()
deduped = []
for r in sorted(missed_annotations, key=lambda x: -x['score']):
    key = (r['para_num'], r['lb_n'], r['ana_id'], r['found_phrase'])
    if key not in seen:
        seen.add(key)
        deduped.append(r)

with open(MISSED_OUT, 'w', newline='', encoding='utf-8') as f:
    w = csv.DictWriter(f, fieldnames=deduped[0].keys() if deduped else [])
    w.writeheader()
    w.writerows(deduped)
print(f"Written: {MISSED_OUT}  ({len(deduped)} unique candidates)")
