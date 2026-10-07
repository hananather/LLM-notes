"""Freeze and fetch an ABO main-image/catalog association benchmark.

Only this preparation module reads publisher identity. Model-facing JSONL records
contain opaque record IDs and native descriptive fields. Images/models stay in an
external cache. No gold labels or model outcomes influence image selection.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import unicodedata

import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = Path(__file__).resolve().parent
DATA = HERE / 'data'
EVAL = HERE / 'evaluator'
CACHE = Path(os.environ.get('NSO_IMAGE_CACHE', Path.home() / '.cache/nso-adversarial-2026/abo'))
PROTOCOL = json.loads((HERE / 'protocol.json').read_text())
SEED = PROTOCOL['seed']
BASE = PROTOCOL['source_base']


def sha(value: str | bytes) -> str:
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, sort_keys=True) + '\n' for r in records))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text().splitlines() if s]


def norm(text: str) -> str:
    text = unicodedata.normalize('NFKD', str(text)).encode('ascii', 'ignore').decode().lower()
    return ' '.join(re.sub(r'[^a-z0-9]+', ' ', text).split())


def brand_norm(text: str) -> str:
    n = norm(text)
    return '365' if n in {'365', '365 everyday value', '365 by whole foods market', '365 by wfm'} else n


def values(row: dict, key: str) -> list:
    return [v['value'] for v in row.get(key, []) if isinstance(v, dict) and 'value' in v
            and (v.get('language_tag', '').startswith('en') or not v.get('language_tag'))]


def image_ids(row: dict) -> set[str]:
    return set(([row['main_image_id']] if row.get('main_image_id') else []) + row.get('other_image_id', []))


def opaque(kind: str, item: str) -> str:
    return kind + '_' + sha(SEED + '|' + kind + '|' + item)[:20]


def fetch_url(url: str, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    if path.exists():
        return {'status': 'cached', 'bytes': path.stat().st_size, 'sha256': sha(path.read_bytes()), 'seconds': 0}
    tmp = path.with_suffix(path.suffix + '.partial')
    errors = []
    for attempt in range(PROTOCOL['downloads']['attempts']):
        cp = subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error', '--max-time',
                             str(PROTOCOL['downloads']['timeout_seconds']), '--retry', '0', '-o', str(tmp), url],
                            capture_output=True, text=True)
        if cp.returncode == 0:
            tmp.replace(path)
            return {'status': 'downloaded', 'bytes': path.stat().st_size, 'sha256': sha(path.read_bytes()),
                    'seconds': round(time.monotonic() - start, 3), 'attempts': attempt + 1}
        errors.append(cp.stderr[:500])
    tmp.unlink(missing_ok=True)
    return {'status': 'failed', 'errors': errors, 'seconds': round(time.monotonic() - start, 3)}


def metadata(source_dir: Path | None) -> tuple[list[dict], dict, list[dict]]:
    files = [(f'abo-listings_{h}.json.gz', f'listings/metadata/listings_{h}.json.gz') for h in '0123456789abcdef']
    files += [('abo-images.csv.gz', 'images/metadata/images.csv.gz')]
    rows, image_meta, manifest = [], {}, []
    for local, remote in files:
        p = CACHE / 'source' / local
        if source_dir and (source_dir / local).exists() and not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_dir / local, p)
        result = fetch_url(BASE + remote, p)
        if result['status'] == 'failed':
            raise RuntimeError(f'Metadata unavailable: {remote}: {result}')
        manifest.append({'source_path': remote, 'url': BASE + remote, **result})
        if local.startswith('abo-listings'):
            with gzip.open(p, 'rt') as f:
                rows.extend(json.loads(s) for s in f)
        else:
            with gzip.open(p, 'rt') as f:
                image_meta = {r['image_id']: r for r in csv.DictReader(f)}
    return rows, image_meta, manifest


def frame(rows: list[dict], image_meta: dict) -> tuple[dict, dict, dict]:
    by_item, owners = collections.defaultdict(list), collections.defaultdict(set)
    for row in rows:
        by_item[row['item_id']].append(row)
        for image_id in image_ids(row):
            owners[image_id].add(row['item_id'])
    grocery = {}
    for item, records in by_item.items():
        for row in sorted(records, key=lambda r: (r['domain_name'] != 'amazon.com', r['domain_name'], json.dumps(r, sort_keys=True))):
            exclusive = {i for i in image_ids(row) if len(owners[i]) == 1 and i in image_meta}
            if values(row, 'item_name') and values(row, 'brand') and 'GROCERY' in values(row, 'product_type') and len(exclusive) >= 2:
                grocery[item] = row
                break
    eligible = {item: row for item, row in grocery.items() if row.get('main_image_id') in image_meta
                and len(owners[row['main_image_id']]) == 1}
    return grocery, eligible, owners


def families(grocery: dict) -> tuple[dict, list[dict], dict]:
    items = sorted(grocery)
    parent = {i: i for i in items}
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        a, b = root(a), root(b)
        if a != b:
            parent[max(a, b)] = min(a, b)
    exact = collections.defaultdict(list)
    titles, brands, numbers = [], [], []
    for item in items:
        row = grocery[item]
        title, brand = values(row, 'item_name')[0], values(row, 'brand')[0]
        exact[(norm(title), norm(brand))].append(item)
        b = brand_norm(brand)
        t = norm(title)
        t = re.sub(r'^amazon brand\s*', '', t)
        t = t.replace(norm(brand), ' ').strip()
        titles.append(t); brands.append(b); numbers.append(tuple(re.findall(r'\d+(?:\.\d+)?', norm(title))))
    edges = []
    for ids in exact.values():
        for item in ids[1:]:
            union(ids[0], item)
            edges.append({'a': ids[0], 'b': item, 'rule': 'normalized_name_brand'})
    mat = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), sublinear_tf=True).fit_transform(titles)
    sim = (mat @ mat.T).tocoo()
    for i, j, s in zip(sim.row, sim.col, sim.data):
        if i < j and s >= PROTOCOL['family_char_cosine_threshold'] and brands[i] == brands[j] and numbers[i] == numbers[j]:
            union(items[i], items[j])
            edges.append({'a': items[i], 'b': items[j], 'rule': 'same_brand_number_signature_char_cosine', 'cosine': round(float(s), 8)})
    group = {item: root(item) for item in items}
    counts = collections.Counter(group.values())
    exact_dup = [v for v in exact.values() if len(v) > 1]
    stats = {'grocery_items': len(items), 'families': len(counts), 'non_singleton_families': sum(n > 1 for n in counts.values()),
             'items_in_non_singleton_families': sum(n for n in counts.values() if n > 1),
             'normalized_name_brand_duplicate_groups': len(exact_dup), 'items_in_normalized_name_brand_duplicate_groups': sum(map(len, exact_dup))}
    return group, edges, stats


def select(eligible: dict, group: dict) -> tuple[dict, dict, dict]:
    members = collections.defaultdict(list)
    for item in eligible:
        members[group[item]].append(item)
    for family in members:
        members[family].sort(key=lambda i: sha(SEED + '|within|' + i))
    forced = set(PROTOCOL['forced_development_item_ids'])
    dev_families = {group[i] for i in forced if i in group}
    assigned = {f: 'dev' for f in dev_families}
    # All forced in-frame IDs precede optional siblings, so all available inspected images are retained.
    dev = sorted(forced & set(eligible), key=lambda i: sha(SEED + '|within|' + i))
    dev += sorted((i for f in dev_families for i in members[f] if i not in forced), key=lambda i: sha(SEED + '|within|' + i))
    ordered = sorted((f for f in members if f not in dev_families), key=lambda f: sha(SEED + '|family|' + f))
    pos = 0
    while len(dev) < PROTOCOL['development_target_queries']:
        f = ordered[pos]; pos += 1; assigned[f] = 'dev'; dev.extend(members[f])
    dev = dev[:PROTOCOL['development_target_queries']]
    test = []
    while len(test) < PROTOCOL['test_queries']:
        f = ordered[pos]; pos += 1; assigned[f] = 'test'; test.extend(members[f])
    test = test[:PROTOCOL['test_queries']]
    extra = []
    while len(extra) < PROTOCOL['test_nil_catalog_distractors']:
        f = ordered[pos]; pos += 1; assigned[f] = 'nil_catalog'; extra.extend(members[f])
    extra = extra[:PROTOCOL['test_nil_catalog_distractors']]
    test_groups = collections.defaultdict(list)
    for i in test:
        test_groups[group[i]].append(i)
    # Whole selected-query families leave the catalog together. Hash order plus a
    # deterministic subset sum gives exactly 200 without stranding an alias.
    target = PROTOCOL['test_nil_query_only']
    paths = {0: []}
    for f in sorted(test_groups, key=lambda f: sha(SEED + '|nil|' + f)):
        for total in sorted(list(paths), reverse=True):
            new = total + len(test_groups[f])
            if new <= target and new not in paths:
                paths[new] = paths[total] + [f]
        if target in paths:
            break
    if target not in paths:
        raise RuntimeError('Could not form exactly 200 whole-family query-only records.')
    removed = {i for f in paths[target] for i in test_groups[f]}
    selected = {'dev': dev, 'test': test, 'nil_catalog_extra': extra, 'nil_query_only': sorted(removed)}
    reserve = {'forced_development_item_ids': sorted(forced), 'forced_not_in_main_image_frame': sorted(forced - set(eligible)),
               'forced_not_in_grocery_frame': sorted(forced - set(group)),
               'reserved_family_members': {split: sorted(i for i, f in group.items() if assigned.get(f) == split) for split in ['dev', 'test', 'nil_catalog']}}
    return selected, assigned, reserve


def catalog(item: str, row: dict) -> dict:
    title = values(row, 'item_name')[0]
    brand = values(row, 'brand')[0]
    bullets = values(row, 'bullet_point')
    models = values(row, 'model_number')
    return {'record_id': opaque('c', item), 'title': title, 'brand': brand, 'bullet_points': bullets,
            'model_number': models[0] if models else None,
            'catalog_text': '\n'.join([title, 'Brand: ' + brand] + (['Model: ' + str(models[0])] if models else []) + [str(b) for b in bullets])}


def prepare(source_dir: Path | None) -> None:
    if (EVAL / 'freeze.json').exists():
        frozen = json.loads((EVAL / 'freeze.json').read_text())
        if frozen['protocol_sha256'] != sha((HERE / 'protocol.json').read_bytes()):
            raise RuntimeError('Protocol changed after selection freeze. Create a new explicitly versioned study.')
        print('Selection already frozen; not changing rows.'); return
    rows, imeta, source_manifest = metadata(source_dir)
    grocery, eligible, _ = frame(rows, imeta)
    group, edges, family_stats = families(grocery)
    selected, assigned, reserve = select(eligible, group)
    DATA.mkdir(exist_ok=True); EVAL.mkdir(exist_ok=True)
    jsonl(EVAL / 'source-files.jsonl', source_manifest)
    jsonl(EVAL / 'family-edges.jsonl', edges)
    (EVAL / 'development-reserve.json').write_text(json.dumps(reserve, indent=2) + '\n')
    jsonl(EVAL / 'ambiguity-panel.jsonl', [{'family_id': opaque('f', f), 'item_ids': sorted(i for i, ff in group.items() if ff == f)}
         for f, n in collections.Counter(group.values()).items() if n > 1])
    source_records, inputs = [], []
    n_by_family = collections.Counter(group.values())
    for split in ['dev', 'test']:
        query_items = sorted(selected[split], key=lambda i: sha(SEED + '|query_order|' + i))
        cat_items = sorted(selected[split], key=lambda i: sha(SEED + '|catalog_order|' + i))
        jsonl(DATA / f'queries-{split}.jsonl', [{'record_id': opaque('q', i), 'image_cache_key': 'prepared/' + opaque('q', i) + '.jpg'} for i in query_items])
        jsonl(DATA / f'catalog-{split}.jsonl', [catalog(i, eligible[i]) for i in cat_items])
        jsonl(EVAL / f'gold-{split}.jsonl', [{'query_id': opaque('q', i), 'catalog_id': opaque('c', i),
              'family_id': opaque('f', group[i]), 'input_ambiguous_family': n_by_family[group[i]] > 1} for i in query_items])
        for i in query_items:
            row = eligible[i]; m = imeta[row['main_image_id']]
            record = {'record_id': opaque('q', i), 'catalog_id': opaque('c', i), 'split': split,
                      'item_id': i, 'domain_name': row['domain_name'], 'image_id': row['main_image_id'], 'image_path': m['path'],
                      'original_url': BASE + 'images/original/' + m['path'], 'source_width': int(m['width']), 'source_height': int(m['height']),
                      'original_cache_key': 'original/' + opaque('q', i) + '.jpg', 'image_cache_key': 'prepared/' + opaque('q', i) + '.jpg',
                      'family_id': opaque('f', group[i]), 'input_ambiguous_family': n_by_family[group[i]] > 1}
            source_records.append(record)
    nil_items = [i for i in selected['test'] if i not in set(selected['nil_query_only'])] + selected['nil_catalog_extra']
    nil_items.sort(key=lambda i: sha(SEED + '|nil_catalog_order|' + i))
    jsonl(DATA / 'catalog-test-nil.jsonl', [catalog(i, eligible[i]) for i in nil_items])
    jsonl(EVAL / 'gold-test-nil.jsonl', [{'query_id': opaque('q', i), 'catalog_id': None if i in set(selected['nil_query_only']) else opaque('c', i),
          'family_id': opaque('f', group[i]), 'input_ambiguous_family': n_by_family[group[i]] > 1} for i in selected['test']])
    jsonl(EVAL / 'source-map.jsonl', source_records)
    jsonl(EVAL / 'catalog-source-map.jsonl', [{'record_id': opaque('c', i), 'item_id': i, 'domain_name': eligible[i]['domain_name'],
          'family_id': opaque('f', group[i])} for i in selected['dev'] + selected['test'] + selected['nil_catalog_extra']])
    stats = {'listing_rows': len(rows), 'unique_item_ids': len({r['item_id'] for r in rows}), 'grocery_eligible_two_exclusive_images': len(grocery),
             'main_image_frame': len(eligible), **family_stats, 'dev_queries': len(selected['dev']), 'test_queries': len(selected['test']),
             'closed_test_catalog': len(selected['test']), 'nil_test_catalog': len(nil_items), 'nil_queries': len(selected['nil_query_only']),
             'dev_families': len({group[i] for i in selected['dev']}), 'test_families': len({group[i] for i in selected['test']}),
             'forced_development_ids': len(PROTOCOL['forced_development_item_ids']), 'forced_outside_main_frame': reserve['forced_not_in_main_image_frame']}
    (EVAL / 'frame-summary.json').write_text(json.dumps(stats, indent=2) + '\n')
    frozen_paths = sorted(list(DATA.glob('*.jsonl')) + list(EVAL.glob('*.jsonl')) + [EVAL / 'development-reserve.json'])
    freeze = {'protocol_sha256': sha((HERE / 'protocol.json').read_bytes()), 'schema_sha256': sha((HERE / 'schema.json').read_bytes()),
              'prepare_sha256': sha(Path(__file__).read_bytes()), 'files': {str(p.relative_to(HERE)): sha(p.read_bytes()) for p in frozen_paths},
              'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'test_outcomes_opened': False}
    (EVAL / 'freeze.json').write_text(json.dumps(freeze, indent=2) + '\n')
    print(json.dumps(stats, indent=2), flush=True)


def prepare_image(row: dict) -> dict:
    record_path = CACHE / 'fetch-records' / (row['record_id'] + '.json')
    if record_path.exists():
        return json.loads(record_path.read_text())
    out = {**row, **fetch_url(row['original_url'], CACHE / row['original_cache_key'])}
    if out['status'] != 'failed':
        try:
            image = ImageOps.exif_transpose(Image.open(CACHE / row['original_cache_key'])).convert('RGB')
            out['original_width'], out['original_height'] = image.size
            out['original_pixel_sha256'] = sha(str(image.size).encode() + np.asarray(image).tobytes())
            scale = min(1.0, 2048 / max(image.size))
            w, h = max(1, int(image.width * scale)), max(1, int(image.height * scale))
            while math.ceil(w / 32) * math.ceil(h / 32) > 2500:
                scale *= 0.995
                w, h = max(1, int(image.width * scale)), max(1, int(image.height * scale))
            if image.size != (w, h):
                image = image.resize((w, h), Image.Resampling.LANCZOS)
            dest = CACHE / row['image_cache_key']; dest.parent.mkdir(parents=True, exist_ok=True)
            image.save(dest, 'JPEG', quality=95)
            decoded = Image.open(dest).convert('RGB')
            grey = np.asarray(decoded.convert('L').resize((32, 32), Image.Resampling.LANCZOS), float)
            low = dctn(grey, type=2, norm='ortho')[:8, :8].ravel()[1:]
            out.update(status='ready', width=w, height=h, patches_32=math.ceil(w/32)*math.ceil(h/32),
                       prepared_sha256=sha(dest.read_bytes()), prepared_bytes=dest.stat().st_size,
                       phash=''.join('1' if x else '0' for x in low > np.median(low)))
        except Exception as exc:
            out.update(status='decode_failed', error=repr(exc))
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(json.dumps(out, indent=2) + '\n')
    return out


def download(split: str) -> None:
    records = read_jsonl(EVAL / 'source-map.jsonl')
    if split != 'all':
        records = [r for r in records if r['split'] == split]
    with concurrent.futures.ThreadPoolExecutor(max_workers=PROTOCOL['downloads']['workers']) as pool:
        results = []
        for n, result in enumerate(pool.map(prepare_image, records), 1):
            results.append(result)
            if n % 50 == 0:
                print(f'Prepared {n}/{len(records)} images', flush=True)
    # Aggregate all available records, permitting dev to finish before test fetch.
    all_results = [json.loads(p.read_text()) for p in sorted((CACHE / 'fetch-records').glob('q_*.json'))]
    wanted = {r['record_id'] for r in read_jsonl(EVAL / 'source-map.jsonl')}
    all_results = [r for r in all_results if r['record_id'] in wanted]
    jsonl(EVAL / 'fetch-manifest.jsonl', all_results)
    jsonl(DATA / 'input-manifest.jsonl', [{k: r.get(k) for k in ['record_id', 'split', 'image_cache_key', 'status', 'width', 'height', 'patches_32', 'prepared_sha256', 'prepared_bytes']} for r in all_results])
    ready = [r for r in all_results if r['status'] == 'ready']
    flags = []
    for n, a in enumerate(ready):
        for b in ready[n + 1:]:
            byte_equal = a['sha256'] == b['sha256']
            pixel_equal = a['original_pixel_sha256'] == b['original_pixel_sha256']
            distance = sum(x != y for x, y in zip(a['phash'], b['phash']))
            if byte_equal or pixel_equal or distance <= 4:
                flags.append({'a': a['record_id'], 'b': b['record_id'], 'cross_split': a['split'] != b['split'],
                              'byte_equal': byte_equal, 'pixel_equal': pixel_equal, 'phash_distance': distance,
                              'same_predeclared_family': a['family_id'] == b['family_id']})
    jsonl(EVAL / 'duplicate-image-flags.jsonl', flags)
    summary = {'frozen_query_count': len(wanted), 'attempted': len(all_results), 'ready': len(ready),
               'failures': [{'record_id': r['record_id'], 'status': r['status']} for r in all_results if r['status'] != 'ready'],
               'original_bytes': sum(r.get('bytes', 0) for r in all_results), 'prepared_bytes': sum(r.get('prepared_bytes', 0) for r in ready),
               'max_patches': max((r['patches_32'] for r in ready), default=0), 'duplicate_flags': len(flags),
               'cross_split_exact_duplicate_flags': sum(r['cross_split'] and (r['byte_equal'] or r['pixel_equal']) for r in flags)}
    (EVAL / 'fetch-summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['prepare', 'download'])
    parser.add_argument('--source-dir', type=Path)
    parser.add_argument('--split', choices=['all', 'dev', 'test'], default='all')
    args = parser.parse_args()
    if args.command == 'prepare': prepare(args.source_dir)
    else: download(args.split)


if __name__ == '__main__':
    main()
