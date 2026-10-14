#!/usr/bin/env python
"""Markdown block with the sanity metrics of one or more variants for the method READMEs:
the `all`, `dir=FWD/STP/REV` rows of tools/eval_planning.py (horizons 1 2 3), parse failures,
parking / on-road vocabulary and runtime.

  python methods/vlm_common/readme_results.py outputs/impromptu_7b_ad_ft [...]
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from olp_inputs import ONROAD_PATTERNS, PARKING_PATTERNS  # noqa: E402
from text_stats import texts_of  # noqa: E402


def main():
    blocks = []
    for d in sys.argv[1:]:
        f = os.path.join(d, 'predictions.json') if os.path.isdir(d) else d
        out = subprocess.run([sys.executable, os.path.join(ROOT, 'tools', 'eval_planning.py'), '--pred', f,
                              '--horizons', '1', '2', '3'], cwd=ROOT, capture_output=True, text=True).stdout
        lines = out.splitlines()
        hdr = [l for l in lines if l.startswith('===')]
        tab = [l for l in lines if l.startswith('subset') or re.match(r'^(all|dir=)', l)]
        raw = json.load(open(f))
        P = raw['predictions']
        ex = [e.get('extra', {}) for e in P.values()]
        n = len(ex)
        nf = sum(1 for e in ex if e.get('parse_fail'))
        txt = [texts_of(e).lower() for e in ex]
        txt = [t for t in txt if t.strip()]
        pk = sum(1 for t in txt if any(re.search(p, t) for p in PARKING_PATTERNS.values()))
        onr = sum(1 for t in txt if any(re.search(p, t) for p in ONROAD_PATTERNS.values()))
        info = raw.get('info', {})
        rt = info.get('runtime_s_per_frame', info.get('runtime_s_per_frame_all_variants'))
        mem = info.get('peak_gpu_mem_gb')
        b = ['**%s**' % raw.get('method'), '```', hdr[0] if hdr else ''] + tab + ['```']
        b.append('- parse failures: %d / %d (%.1f %%) -> constant-velocity fallback' % (nf, n, 100.0 * nf / max(n, 1)))
        if txt:
            b.append('- model text mentions a parking term (park*, stall, spot, space, revers*, back up/out, pull in/out, '
                     'garage, parking lot, maneuver, aisle) in %d / %d frames (%.1f %%); on-road terms (lane, traffic light, '
                     'stop sign, intersection, ...) in %d (%.1f %%)' % (pk, len(txt), 100.0 * pk / len(txt), onr,
                                                                     100.0 * onr / len(txt)))
        if rt:
            b.append('- runtime %.2f s/frame%s' % (rt, ', peak GPU memory %.1f GB (PyTorch allocated)' % mem if mem else ''))
        blocks.append('\n'.join(b))
    print('\n\n'.join(blocks))


if __name__ == '__main__':
    main()
