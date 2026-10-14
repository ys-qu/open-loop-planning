#!/usr/bin/env python
"""Turn the json written by tools/eval_planning.py --out into a markdown table.

  python tools/eval_planning.py --pred outputs/*/predictions.json --by direction travel \
      --out outputs/summary_recommended.json
  python tools/summarize.py outputs/summary_recommended.json > outputs/summary_recommended.md
"""
import json
import sys


def fmt(v):
    m, n = v
    return '-' if n == 0 else '%.2f' % m


def main():
    d = json.load(open(sys.argv[1]))
    rows = []
    hdr = ('| method | frames | ADE@1s | ADE@2s | ADE@3s | FDE@3s | minADE@3s | FWD ADE/FDE@3s | REV ADE/FDE@3s '
           '| col% (GT floor) | revRec% |')
    rows.append(hdr)
    rows.append('|' + '---|' * (hdr.count('|') - 1))
    order = sorted(d, key=lambda k: d[k]['all']['ADE@3'][0] if d[k]['all'].get('ADE@3', [0, 0])[1] else 1e9)
    for name in order:
        a = d[name]['all']
        f = d[name].get('dir=FWD', {})
        r = d[name].get('dir=REV', {})
        rows.append('| %s | %d | %s | %s | %s | %s | %s | %s / %s | %s / %s | %.1f (%.1f) | %.0f |' % (
            name, a['n'], fmt(a['ADE@1']), fmt(a['ADE@2']), fmt(a['ADE@3']), fmt(a['FDE@3']), fmt(a['minADE@3']),
            fmt(f.get('ADE@3', (0, 0))), fmt(f.get('FDE@3', (0, 0))),
            fmt(r.get('ADE@3', (0, 0))), fmt(r.get('FDE@3', (0, 0))),
            a['col%'], a['col_gt%'], r.get('rev_recall%', float('nan'))))
    print('\n'.join(rows))


if __name__ == '__main__':
    main()
