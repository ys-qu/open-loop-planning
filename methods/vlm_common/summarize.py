#!/usr/bin/env python
"""Compact sanity table for VLM/VLA outputs (wraps tools/eval_planning.py, no new metric).

  python methods/vlm_common/summarize.py outputs/impromptu_7b_ad_ft outputs/openemma_qwen25vl7b ...
Prints, per variant: n, ADE/FDE@3s (all / FWD / STP / REV rows of eval_planning), revRec%, col%,
parse-fail %, % frames whose text uses parking terms, % with on-road terms, runtime.
Writes the markdown table to --md if given.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'tools'))
sys.path.insert(0, HERE)
import eval_planning as EP  # noqa: E402
from olpcommon import DEFAULT_DATAROOT, ParkingDataset, load_predictions  # noqa: E402
from text_stats import texts_of  # noqa: E402
from olp_inputs import ONROAD_PATTERNS, PARKING_PATTERNS  # noqa: E402
import re  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dirs', nargs='+')
    ap.add_argument('--scenes', default=os.environ.get('OLP_SCENES', os.path.join(ROOT, 'scenes_subset.txt')))
    ap.add_argument('--md', default=None)
    ap.add_argument('--json', default=None)
    args = ap.parse_args()
    db = ParkingDataset(DEFAULT_DATAROOT, args.scenes, verbose=False)
    hs = [1.0, 2.0, 3.0]
    keys = []
    for H in ['1', '2', '3', 'native']:
        keys += ['ADE@%s' % H, 'FDE@%s' % H, 'minADE@%s' % H, 'minFDE@%s' % H]
    rows_md, out = [], {}
    for d in args.dirs:
        f = os.path.join(d, 'predictions.json') if os.path.isdir(d) else d
        pred = load_predictions(f)
        rows = EP.evaluate(db, pred, hs)
        for r in rows:
            for k in list(r.keys()):
                for h in hs:
                    for pre in ('ADE@', 'FDE@', 'minADE@', 'minFDE@'):
                        if k == '%s%s' % (pre, h):
                            r['%s%g' % (pre, h)] = r.pop(k)
        S = {'all': EP.summarize(rows, keys)}
        for dr in ('FWD', 'STP', 'REV'):
            S[dr] = EP.summarize([r for r in rows if r['direction'] == dr], keys)
        raw = json.load(open(f))
        P = raw['predictions']
        ex = [e.get('extra', {}) for e in P.values()]
        n = len(ex)
        pf = 100.0 * sum(1 for e in ex if e.get('parse_fail')) / max(n, 1)
        txt = [texts_of(e).lower() for e in ex]
        txt = [t for t in txt if t.strip()]
        pk = 100.0 * sum(1 for t in txt if any(re.search(p, t) for p in PARKING_PATTERNS.values())) / max(len(txt), 1)
        onr = 100.0 * sum(1 for t in txt if any(re.search(p, t) for p in ONROAD_PATTERNS.values())) / max(len(txt), 1)
        info = raw.get('info', {})
        rt = info.get('runtime_s_per_frame', info.get('runtime_s_per_frame_all_variants'))
        name = raw.get('method', os.path.basename(os.path.dirname(f)))

        def af(s, h='3'):
            a, na = s['ADE@%s' % h]
            b, _ = s['FDE@%s' % h]
            return ('%.2f/%.2f' % (a, b)) if na else '-'
        rec = {'n': n, 'all@3': af(S['all']), 'FWD@3': af(S['FWD']), 'STP@3': af(S['STP']), 'REV@3': af(S['REV']),
               'all@1': af(S['all'], '1'), 'all@2': af(S['all'], '2'),
               'revRec%': round(S['all']['rev_recall%'], 1), 'col%': round(S['all']['col%'], 2),
               'parse_fail%': round(pf, 1), 'text_frames': len(txt),
               'parking_terms%': round(pk, 1) if txt else None, 'onroad_terms%': round(onr, 1) if txt else None,
               'native_h': rows[0]['native_h'] if rows else None, 'K': sorted({r['K'] for r in rows}),
               'runtime_s_per_frame': rt, 'peak_gpu_mem_gb': info.get('peak_gpu_mem_gb')}
        out[name] = rec
        print('%-26s n=%4d ADE/FDE@3 all %s FWD %s STP %s REV %s | @1 %s @2 %s | revRec %.1f%% col %.2f%% | '
              'parse-fail %.1f%% | parking-terms %s%% on-road %s%% | %s s/frame' % (
                  name, n, rec['all@3'], rec['FWD@3'], rec['STP@3'], rec['REV@3'], rec['all@1'], rec['all@2'],
                  rec['revRec%'], rec['col%'], pf, rec['parking_terms%'], rec['onroad_terms%'],
                  '%.2f' % rt if rt else '?'))
        rows_md.append('| %s | %d | %s | %s | %s | %s | %.1f | %.1f | %s | %s |' % (
            name, n, rec['all@3'], rec['FWD@3'], rec['STP@3'], rec['REV@3'], rec['revRec%'], pf,
            rec['parking_terms%'] if txt else '-', rec['onroad_terms%'] if txt else '-'))
    if args.md:
        with open(args.md, 'w') as fo:
            fo.write('| variant | n | ADE/FDE@3s all | FWD | STP | REV | revRec% | parse-fail% | parking terms% | on-road terms% |\n')
            fo.write('|---|---|---|---|---|---|---|---|---|---|\n')
            fo.write('\n'.join(rows_md) + '\n')
    if args.json:
        json.dump(out, open(args.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
