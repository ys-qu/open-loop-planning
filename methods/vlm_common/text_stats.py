#!/usr/bin/env python
"""Summarise the text side of VLM/VLA predictions: parse failures and vocabulary.

  python methods/vlm_common/text_stats.py --pred "outputs/*/predictions.json" [--examples 2] [--out x.json]

For every text field in `extra` (scene / objects / intent / text / motion_text), and for their
union, report the share of frames whose text contains
  * a parking term   (olp_inputs.PARKING_PATTERNS: park*, stall, spot, space, revers*, back up/out,
                      pull in/out, garage, parking lot, maneuver, aisle)
  * an on-road term  (olp_inputs.ONROAD_PATTERNS: lane, keep lane, traffic light, stop sign,
                      intersection, lane change, speed bump, crosswalk, highway)
overall and by ego direction (FWD / STP / REV). Regex word matches on the lower-cased text.
Note: the OpenEMMA prompts themselves ask about "lane markings" and "traffic lights", so
on-road words in their answers are partly prompted.
"""
import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from olp_inputs import DEFAULT_DATAROOT, ONROAD_PATTERNS, PARKING_PATTERNS, ROOT, ParkingDataset  # noqa: E402

TEXT_KEYS = ('scene', 'objects', 'intent', 'text', 'motion_text')
# parking terms that describe a parking manoeuvre / target rather than just naming the place
ACTION_KEYS = ('stall', 'spot', 'space', 'revers*', 'back up/out', 'pull in/out', 'maneuver', 'aisle')


def texts_of(extra):
    return ' '.join(str(extra[k]) for k in TEXT_KEYS if isinstance(extra.get(k), str))


def hits(txt, patterns):
    return [k for k, p in patterns.items() if re.search(p, txt)]


def analyse(P, db, examples=1):
    fields = [k for k in TEXT_KEYS if any(isinstance(e.get('extra', {}).get(k), str) for e in P.values())]
    res = {'n': len(P), 'parse_fail': sum(1 for e in P.values() if e.get('extra', {}).get('parse_fail')),
           'fields': {}}
    res['parse_fail_pct'] = 100.0 * res['parse_fail'] / max(res['n'], 1)
    for fld in fields + ['ALL']:
        n, pk, pka, onr, terms, oterms, by_dir, ex = 0, 0, 0, 0, {}, {}, {}, []
        for t, e in P.items():
            x = e.get('extra', {})
            txt = texts_of(x) if fld == 'ALL' else x.get(fld)
            if not isinstance(txt, str) or not txt.strip():
                continue
            txt = txt.lower()
            n += 1
            h, o = hits(txt, PARKING_PATTERNS), hits(txt, ONROAD_PATTERNS)
            pk += bool(h)
            pka += any(k in ACTION_KEYS for k in h)
            onr += bool(o)
            for k in h:
                terms[k] = terms.get(k, 0) + 1
            for k in o:
                oterms[k] = oterms.get(k, 0) + 1
            d = (db.direction(t) if db else None) or '?'
            a = by_dir.setdefault(d, [0, 0, 0])
            a[0] += bool(h)
            a[1] += bool(o)
            a[2] += 1
            if h and len(ex) < examples:
                m = re.search('|'.join(PARKING_PATTERNS[k] for k in h), txt)
                s0 = max(0, m.start() - 120)
                ex.append('...' + txt[s0:m.end() + 120].replace('\n', ' ') + '...')
        if not n:
            continue
        res['fields'][fld] = {
            'frames': n, 'parking_pct': 100.0 * pk / n, 'parking_action_pct': 100.0 * pka / n,
            'onroad_pct': 100.0 * onr / n,
            'parking_terms': dict(sorted(terms.items(), key=lambda kv: -kv[1])),
            'onroad_terms': dict(sorted(oterms.items(), key=lambda kv: -kv[1])),
            'by_direction': {k: {'n': v[2], 'parking_pct': round(100.0 * v[0] / v[2], 1),
                                 'onroad_pct': round(100.0 * v[1] / v[2], 1)} for k, v in sorted(by_dir.items())},
            'examples': ex}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', nargs='+', required=True)
    ap.add_argument('--examples', type=int, default=1)
    ap.add_argument('--scenes', default=os.environ.get('OLP_SCENES', os.path.join(ROOT, 'scenes_subset.txt')))
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    db = ParkingDataset(DEFAULT_DATAROOT, args.scenes, load_agents=False, verbose=False)
    files = sorted({f for p in args.pred for f in glob.glob(p)})
    summary = {}
    for f in files:
        d = json.load(open(f))
        name = d.get('method', os.path.basename(os.path.dirname(f)))
        r = analyse(d['predictions'], db, args.examples)
        summary[name] = r
        print('\n=== %s (%s): %d frames, parse failures %d (%.1f%%)' % (name, f, r['n'], r['parse_fail'], r['parse_fail_pct']))
        for fld, s in r['fields'].items():
            print('  [%-11s] %4d frames | parking terms %5.1f%% (manoeuvre terms %5.1f%%) | on-road terms %5.1f%% | by dir %s' % (
                fld, s['frames'], s['parking_pct'], s['parking_action_pct'], s['onroad_pct'],
                {k: (v['parking_pct'], v['onroad_pct']) for k, v in s['by_direction'].items()}))
            if fld == 'ALL':
                print('      parking:', s['parking_terms'])
                print('      on-road:', s['onroad_terms'])
                for e in s['examples']:
                    print('      e.g.', e)
    if args.out:
        json.dump(summary, open(args.out, 'w'), indent=1)
        print('written', args.out)


if __name__ == '__main__':
    main()
