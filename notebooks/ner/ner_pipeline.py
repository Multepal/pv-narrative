#!/usr/bin/env python3
"""
Cross-edition NER pipeline for the Popol Wuj project.

Strategy: gazetteer-based n-gram matching using the canonical entity list from
topics.xml + colop-glossary.txt + Ximénez surface-form variants.

K'iche' editions (5): paragraph (doc_id) is the granular location unit — the
manuscript has virtually no terminal punctuation as separate tokens.

Spanish (Recinos) and English (Tedlock): use existing sentence indices in TOKEN.csv.
Christenson English prose: uses chapter (chap_num) as location unit.

Outputs (all written to notebooks/ner/):
  GAZETTEER.csv          — canonical entity × language × surface-form table
  {edition}-NER.csv      — one row per entity mention with location + context
  ENTITY_CROSSMAP.csv    — aggregated mention counts per entity × edition
"""

import re
import json
import html as html_mod
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
ROOT  = _HERE.parent.parent          # notebooks/ner/ → project root

TEMAS_XML     = ROOT / 'temas' / 'topics.xml'
COLOP_GLOSS   = ROOT / 'textos' / 'colop-glossary.txt'
SURFACE_FORMS = ROOT / 'notebooks' / 'ximenez' / 'ximenez-SURFACE_FORMS.csv'

EDITIONS = {
    'ajtzibab':                  {'lang': 'quc'},
    'christenson':               {'lang': 'quc'},
    'colop':                     {'lang': 'quc'},
    'christenson_ximenez':       {'lang': 'quc'},
    'ximenez':                   {'lang': 'quc'},
    'recinos':                   {'lang': 'spa'},
    'tedlock':                   {'lang': 'eng'},
    'christenson_english_prose': {'lang': 'eng'},
}
for _name in EDITIONS:
    EDITIONS[_name]['token_path'] = (
        ROOT / 'notebooks' / _name / f'{_name}-TOKEN.csv'
    )

MAX_NGRAM = 4


# ── Text normalisation ────────────────────────────────────────────────────────

def _unify_apos(s: str) -> str:
    """Normalise Unicode apostrophe variants to plain ASCII apostrophe."""
    return s.replace('’', "'").replace('ʼ', "'").replace('‘', "'")


def normalize(s: str) -> str:
    """Canonical lookup form: uppercase, unified apostrophes, collapsed spaces."""
    return re.sub(r'\s+', ' ', _unify_apos(s).upper().strip())


def normalize_fuzzy(s: str) -> str:
    """Strip apostrophes + diacritics for cross-orthography matching."""
    s = normalize(s).replace("'", '')
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def clean_token(tok) -> str:
    """Prepare one token for n-gram assembly: uppercase, strip outer punctuation."""
    if not isinstance(tok, str):
        return ''
    t = normalize(tok)
    t = re.sub(r"^[^A-Z']+", '', t)
    t = re.sub(r"[^A-Z']+$", '', t)
    return t


# ── Gazetteer ─────────────────────────────────────────────────────────────────

def build_gazetteer() -> pd.DataFrame:
    """
    Assemble a long-format gazetteer: one row per (entity_id, lang, surface_form).

    Sources:
      1. temas/topics.xml  — K'iche' titles, English names, Spanish (Ximénez)
      2. textos/colop-glossary.txt — extra K'iche' surface forms
      3. ximenez-SURFACE_FORMS.csv — annotation-derived K'iche' variants
    """
    rows = []

    # ── 1. topics.xml ─────────────────────────────────────────────────────────
    tree = ET.parse(TEMAS_XML)
    for t in tree.getroot().findall('topic'):
        key   = (t.findtext('key') or '').strip()
        if not key:
            continue
        title = (t.findtext('title') or '').strip()
        ttype = (t.findtext('type') or '').strip()
        base  = dict(entity_id=key, label=title, type=ttype)

        # K'iche': title + key (underscores → spaces)
        if title:
            rows.append({**base, 'lang': 'quc', 'form': normalize(title)})
        key_form = normalize(key.replace('_', ' '))
        if key_form:
            rows.append({**base, 'lang': 'quc', 'form': key_form})

        # English: english_name (HTML-encoded, may have comma-separated variants)
        eng_el  = t.find('english_name')
        eng_raw = html_mod.unescape((eng_el.text or '') if eng_el is not None else '')
        eng_raw = re.sub(r'<[^>]+>', '', eng_raw)
        for part in re.split(r'[,;/\n]', eng_raw):
            part = part.strip()
            if len(part) > 2:
                rows.append({**base, 'lang': 'eng', 'form': normalize(part)})

        # Spanish: ximenez_spa field (Ximénez's 17th-c. Spanish rendering)
        spa = (t.findtext('ximenez_spa') or '').strip()
        if spa:
            rows.append({**base, 'lang': 'spa', 'form': normalize(spa)})

    # ── 2. Colop glossary — additional K'iche' surface forms ──────────────────
    # Only add forms for entities already in the list (matched by fuzzy key).
    existing_exact = {r['form']: r['entity_id'] for r in rows if r['lang'] == 'quc'}
    existing_fuzzy = {normalize_fuzzy(f): eid for f, eid in existing_exact.items()}
    id_to_meta = {
        r['entity_id']: (r['label'], r['type'])
        for r in rows if r.get('entity_id')
    }

    cdf = pd.read_csv(COLOP_GLOSS, sep='|', dtype=str).fillna('')
    for _, row in cdf.iterrows():
        term = normalize(row['term'])
        fuzz = normalize_fuzzy(term)
        eid  = existing_exact.get(term) or existing_fuzzy.get(fuzz)
        if eid:
            label, ttype = id_to_meta.get(eid, (row['term'].strip(), ''))
            rows.append({'entity_id': eid, 'label': label, 'type': ttype,
                         'lang': 'quc', 'form': term})

    # ── 3. Surface forms from Ximénez annotation coverage ────────────────────
    # Refresh id_to_meta with everything collected so far
    for r in rows:
        eid = r.get('entity_id')
        if eid and r.get('label') and eid not in id_to_meta:
            id_to_meta[eid] = (r['label'], r['type'])

    sf_df = pd.read_csv(SURFACE_FORMS)
    for _, row in sf_df.iterrows():
        eid = str(row['ana_id']).strip()
        sf  = normalize(str(row['surface_form']))
        if eid and sf:
            label, ttype = id_to_meta.get(eid, (eid, ''))
            rows.append({'entity_id': eid, 'label': label, 'type': ttype,
                         'lang': 'quc', 'form': sf})

    gaz = (pd.DataFrame(rows)
           .query("entity_id == entity_id and entity_id != '' and form != ''")
           .drop_duplicates(subset=['entity_id', 'lang', 'form'])
           .reset_index(drop=True))

    # Back-fill empty labels/types from authoritative rows
    fill = (gaz[gaz['label'] != '']
            .drop_duplicates('entity_id')
            .set_index('entity_id')[['label', 'type']])
    mask_lbl = (gaz['label'] == '') & gaz['entity_id'].isin(fill.index)
    mask_typ = (gaz['type']  == '') & gaz['entity_id'].isin(fill.index)
    gaz.loc[mask_lbl, 'label'] = gaz.loc[mask_lbl, 'entity_id'].map(fill['label'])
    gaz.loc[mask_typ, 'type']  = gaz.loc[mask_typ, 'entity_id'].map(fill['type'])

    return gaz


# ── Lookup dicts ──────────────────────────────────────────────────────────────

def build_lookups(gaz: pd.DataFrame) -> dict:
    """
    Returns {lang: (exact_dict, fuzzy_dict)}.
    exact_dict: normalized_form → entity_id
    fuzzy_dict: apostrophe+diacritic-stripped form → entity_id

    Spanish and English editions use K'iche' transliterations for proper names,
    so their lookups include quc forms as a fallback after the primary lang forms.
    quc forms are added second so that language-specific forms take priority.
    """
    result = {}
    # Build quc lookup once; reuse for spa/eng fallback
    quc_rows = gaz[gaz['lang'] == 'quc']
    quc_exact, quc_fuzzy = {}, {}
    for _, row in quc_rows.iterrows():
        f   = row['form']
        fuz = normalize_fuzzy(f)
        if f not in quc_exact:
            quc_exact[f] = row['entity_id']
        if fuz not in quc_fuzzy:
            quc_fuzzy[fuz] = row['entity_id']
    result['quc'] = (quc_exact, quc_fuzzy)

    for lang in ('spa', 'eng'):
        # Start with a copy of the quc dicts as fallback
        exact = dict(quc_exact)
        fuzzy = dict(quc_fuzzy)
        # Overwrite / add with language-specific forms (these take priority)
        for _, row in gaz[gaz['lang'] == lang].iterrows():
            f   = row['form']
            fuz = normalize_fuzzy(f)
            exact[f] = row['entity_id']
            fuzzy[fuz] = row['entity_id']
        result[lang] = (exact, fuzzy)
    return result


# ── Token loading ─────────────────────────────────────────────────────────────

def load_edition_tokens(name: str, info: dict) -> pd.DataFrame:
    """
    Load TOKEN.csv and return DataFrame with columns:
      ohco_path  — string key for the location (paragraph, sentence, or chapter)
      token_str  — raw token text

    Location granularity by edition:
      quc editions     → doc_id (paragraph; K'iche' text lacks sentence punctuation)
      spa (recinos)    → parte.capit.sent
      eng tedlock      → part.sent
      eng prose        → chap_num (no sentence index present)
    """
    df   = pd.read_csv(info['token_path'])
    lang = info['lang']

    if lang == 'spa':
        df['ohco_path'] = (df['parte'].astype(str) + '.' +
                           df['capit'].astype(str) + '.' +
                           df['sent'].astype(str))
    elif lang == 'eng' and 'sent' in df.columns:
        # tedlock: part, sent, token_num
        df['ohco_path'] = df['part'].astype(str) + '.' + df['sent'].astype(str)
    else:
        # K'iche' editions + christenson_english_prose
        grp_col = 'doc_id' if 'doc_id' in df.columns else 'chap_num'
        df['ohco_path'] = df[grp_col].astype(str)

    return df[['ohco_path', 'token_str']].copy()


# ── N-gram matching ───────────────────────────────────────────────────────────

def match_entities(tokens: list, exact: dict, fuzzy: dict) -> list:
    """
    Greedy longest-match entity lookup over a token sequence.

    Returns list of (raw_ngram, ngram_len, entity_id, match_type).
    Consumed positions are skipped so no token contributes to more than one match.
    """
    cleaned  = [clean_token(t) for t in tokens]
    n        = len(cleaned)
    consumed = set()
    results  = []

    for size in range(MAX_NGRAM, 0, -1):
        for i in range(n - size + 1):
            if any(p in consumed for p in range(i, i + size)):
                continue
            window = cleaned[i: i + size]
            if not any(window):
                continue
            ngram      = ' '.join(w for w in window if w)
            norm_ngram = normalize(ngram)
            entity_id  = exact.get(norm_ngram)
            mtype      = 'exact'
            if entity_id is None:
                entity_id = fuzzy.get(normalize_fuzzy(ngram))
                mtype     = 'fuzzy'
            if entity_id:
                consumed.update(range(i, i + size))
                raw_ngram = ' '.join(str(tokens[j]) for j in range(i, i + size))
                results.append((raw_ngram, size, entity_id, mtype))

    return results


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(out_dir: Path | None = None, verbose: bool = True) -> pd.DataFrame:
    """
    Build gazetteer, process all editions, write CSVs.
    Returns the concatenated NER DataFrame.
    """
    out = out_dir or _HERE

    if verbose:
        print('Building gazetteer...')
    gaz = build_gazetteer()
    gaz.to_csv(out / 'GAZETTEER.csv', index=False)
    if verbose:
        print(f'  {len(gaz)} form entries, {gaz["entity_id"].nunique()} unique entities')
        print(gaz.groupby('lang')['form'].count().to_string())

    lookups   = build_lookups(gaz)
    label_map = (gaz[['entity_id', 'label', 'type']]
                 .drop_duplicates('entity_id')
                 .set_index('entity_id'))

    all_ner = []

    for name, info in EDITIONS.items():
        lang        = info['lang']
        exact, fuzz = lookups[lang]

        if verbose:
            print(f'\nProcessing {name} ({lang})...')
        tok_df = load_edition_tokens(name, info)

        rows = []
        for ohco_path, grp in tok_df.groupby('ohco_path', sort=False):
            tokens = grp['token_str'].tolist()
            for raw_ngram, ngram_len, entity_id, match_type in \
                    match_entities(tokens, exact, fuzz):
                rows.append({
                    'entity_id':  entity_id,
                    'edition':    name,
                    'ohco_path':  ohco_path,
                    'ngram':      raw_ngram,
                    'ngram_len':  ngram_len,
                    'match_type': match_type,
                })

        NER = pd.DataFrame(rows)
        if len(NER):
            NER['label'] = NER['entity_id'].map(label_map['label'])
            NER['type']  = NER['entity_id'].map(label_map['type'])
        else:
            NER = pd.DataFrame(columns=[
                'entity_id', 'edition', 'ohco_path',
                'ngram', 'ngram_len', 'match_type', 'label', 'type'
            ])
        NER.to_csv(out / f'{name}-NER.csv', index=False)
        if verbose:
            n_unique = NER['entity_id'].nunique() if len(NER) else 0
            print(f'  → {len(NER)} mentions, {n_unique} unique entities')
        all_ner.append(NER)

    # ── Cross-edition map ─────────────────────────────────────────────────────
    ALL = pd.concat([df for df in all_ner if len(df)], ignore_index=True)

    xmap = (ALL
            .groupby(['entity_id', 'label', 'type', 'edition'], dropna=False)
            .apply(lambda g: pd.Series({
                'mention_count':  len(g),
                'locations': json.dumps(
                    sorted(g['ohco_path'].unique().tolist())
                ),
            }))
            .reset_index())
    xmap.to_csv(out / 'ENTITY_CROSSMAP.csv', index=False)

    if verbose:
        print(f'\nCrossmap: {len(xmap)} entity×edition rows')
        print(xmap.groupby('edition')['mention_count'].sum().sort_values(ascending=False).to_string())

    return ALL


if __name__ == '__main__':
    run()
