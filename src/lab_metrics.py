"""Transparent overlap and shot-structure proxies, not artistic quality judgments."""
from collections import Counter
import re
import statistics

SHOT = re.compile(r'\bSHOT\s+(\d+)\b', re.I)
FIELDS = {
    'time': r'\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?\s*s',
    'lens': r'\b\d+\s*mm\b',
    'framing': r'\b(?:wide|medium|close|master|two[- ]shot|over[- ]shoulder|insert|environmental)\b',
    'movement': r'\b(?:push|pull|track|pan|tilt|slider|static|still|handheld|crane|move|hold)\b',
}


def structure(text):
    matches = list(SHOT.finditer(text))
    shots = [text[m.end():matches[i + 1].start() if i + 1 < len(matches) else len(text)]
             for i, m in enumerate(matches)]
    return dict(shot_count=len(shots), shot_ids=[int(m[1]) for m in matches],
                fields={key: sum(bool(re.search(pattern, s, re.I)) for s in shots)
                        for key, pattern in FIELDS.items()})


def score(prediction, reference):
    def words(text):
        return re.findall(r'\w+', text.casefold())
    a, b = words(prediction), words(reference)
    common = sum((Counter(a) & Counter(b)).values())
    f1 = 2 * common / (len(a) + len(b)) if a or b else 1.0
    def bigrams(tokens):
        return Counter(zip(tokens, tokens[1:]))
    aa, bb = bigrams(a), bigrams(b)
    denom = sum(aa.values()) + sum(bb.values())
    bigram = 2 * sum((aa & bb).values()) / denom if denom else 0.0
    p, r = structure(prediction), structure(reference)
    count_score = max(0, 1 - abs(p['shot_count'] - r['shot_count']) / max(1, r['shot_count']))
    field_score = statistics.mean(min(p['fields'][k], r['fields'][k]) / r['fields'][k]
                                  if r['fields'][k] else float(p['fields'][k] == 0) for k in FIELDS)
    return dict(unigram_f1=f1, bigram_f1=bigram, shot_count_score=count_score,
                shot_field_recall=field_score, structure_score=(count_score + field_score) / 2,
                parsed=p, reference_parsed=r)


def aggregate(rows):
    if not rows:
        raise ValueError('Cannot summarize zero scenes')
    keys = ('unigram_f1', 'bigram_f1', 'shot_count_score', 'shot_field_recall', 'structure_score')
    return dict(n=len(rows), **{k: statistics.mean(r['scores'][k] for r in rows) for k in keys},
                truncation_rate=statistics.mean(r['truncated'] for r in rows),
                mean_seconds=statistics.mean(r['seconds'] for r in rows))
