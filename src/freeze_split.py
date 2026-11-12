"""Freeze input-only scene groups before case 2 training; never overwrite a split."""
import hashlib
import json
from pathlib import Path
import random
import re

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = '073bd194bdb79e3ecfcfad51e09f23510ca697431475221b785c369619d7b8cf'
# Conservative input-only review: shared scene, setting/action, or direct story beat.
GROUPS = {
    'bridge': [0, 26, 36, 49], 'ferry': [1, 24, 42],
    'school': [2, 3, 4, 7], 'markets': [5, 25, 41],
    'tea': [6, 46], 'rain_shelter': [9, 35],
    'music_rehearsal': [11, 12, 30, 31, 33, 45],
    'couch': [13, 40], 'ganpati': [14, 15, 16, 17, 18, 39],
    'festive_continuity': [20, 21, 22], 'audition': [27, 28],
    'backstage': [19, 29, 44],
    'owl_radio': [50, 72, 98], 'pollution': [51, 63, 70, 71, 79],
    'costume': [52, 67, 73, 74, 93, 94],
    'rooftop_movement': [53, 61, 82, 86],
    'hero_silhouette': [54, 62, 64, 96], 'cats': [57, 58, 90],
    'criminal_medicines': [85, 91], 'warehouse': [89, 92],
    'cosmic_separation': [100, 101, 102, 103, 104, 105],
    'kailash_meditation': [106, 107, 141, 142],
    'mansarovar': [108, 122], 'young_sati': [109, 110, 111, 112],
    'kankhal_geography': [113, 114, 115], 'daksha_rejection': [116, 134],
    'workshop': [117, 144], 'tapasya': [123, 124, 125],
    'wedding': [126, 127], 'cave_domestic': [128, 129, 130, 131, 133],
    'yagya': [135, 136, 137, 138, 139, 140, 143, 148, 149],
    'veerbhadra': [145, 146, 147],
}


def build():
    raw = (ROOT / 'data/dataset.json').read_bytes()
    assert hashlib.sha256(raw).hexdigest() == EXPECTED, 'Dataset changed'
    records = json.loads(raw)
    inputs = [re.findall(r'\w+', row["user's scene description"].casefold()) for row in records]
    parent = list(range(len(records)))

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i

    def join(a, b):
        parent[find(b)] = find(a)

    for indexes in GROUPS.values():
        for index in indexes[1:]:
            join(indexes[0], index)
    pairs = []
    normalized_duplicates = []
    for i in range(len(inputs)):
        for j in range(i):
            a, b = set(inputs[i]), set(inputs[j])
            score = len(a & b) / len(a | b)
            if inputs[i] == inputs[j]:
                normalized_duplicates.append([j, i])
                join(j, i)
            if score >= 0.45:
                pairs.append({'indexes': [j, i], 'word_set_jaccard': score})
                join(j, i)
    groups = {}
    for i in range(len(inputs)):
        groups.setdefault(find(i), []).append(i)
    # Previously inspected exploratory outputs cannot be an untouched test set.
    exploratory = [6, 22, 26, 28, 35, 57, 62, 70, 108, 139]
    forced = [g for g in groups.values() if set(g) & set(exploratory)]
    candidates = [g for g in groups.values() if g not in forced]
    random.Random(42).shuffle(candidates)
    splits = {'train': sum(forced, []), 'validation': [], 'test': []}
    for name in ('test', 'validation'):
        for group in candidates[:]:
            if len(splits[name]) + len(group) <= 15:
                splits[name].extend(group)
                candidates.remove(group)
    splits['train'].extend(sum(candidates, []))
    splits = {name: sorted(indexes) for name, indexes in splits.items()}
    assert sorted(sum(splits.values(), [])) == list(range(150))
    for group in groups.values():
        assert sum(bool(set(group) & set(s)) for s in splits.values()) == 1
    assert not set(exploratory) & set(splits['test'] + splits['validation'])
    canonical = json.dumps(splits, sort_keys=True, separators=(',', ':')).encode()
    return {'dataset_sha256': EXPECTED, 'seed': 42, 'split_sha256': hashlib.sha256(canonical).hexdigest(),
            'split_hash_encoding': 'UTF-8 json.dumps(indexes, sort_keys=True, separators=(comma,colon))',
            'indexes': splits, 'counts': {k: len(v) for k, v in splits.items()},
            'input_only_manual_groups': GROUPS, 'normalized_duplicate_pairs': normalized_duplicates,
            'lexical_group_threshold': 0.45, 'lexical_pairs': pairs,
            'connected_groups': sorted(groups.values()), 'exploratory_indexes_forced_train': exploratory,
            'limitations': 'Conservative manual input review and lexical grouping reduce leakage, but do not certify semantic independence. Characters and story worlds cross splits. No references used for grouping.'}


if __name__ == '__main__':
    path = ROOT / 'results/02/local-2026-11-11/case2-split.json'
    with path.open('x') as dest:
        dest.write(json.dumps(build(), indent=2) + '\n')
    print(path)
