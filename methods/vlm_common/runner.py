"""Small shared driver for the VLM adapters: CLI args, resumable prediction writing,
raw-text logging and summary statistics (parse failures, parking vocabulary)."""
import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from olp_inputs import DEFAULT_DATAROOT, ROOT, ParkingDataset, parking_vocab, select_tokens  # noqa: E402
from olpcommon.predictions import make_entry, save_predictions  # noqa: E402,F401


def base_argparser(desc, default_variant):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument('--dataroot', default=DEFAULT_DATAROOT)
    ap.add_argument('--scenes', default=os.environ.get('OLP_SCENES', os.path.join(ROOT, 'scenes_subset.txt')))
    ap.add_argument('--variant', default=default_variant, help='output name: outputs/<variant>/predictions.json')
    ap.add_argument('--out-dir', default=os.environ.get('OLP_OUTPUTS', os.path.join(ROOT, 'outputs')))
    ap.add_argument('--every', type=int, default=1, help='keep every N-th keyframe (time order)')
    ap.add_argument('--offset', type=int, default=0)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--tokens', default=None, help='comma separated sample tokens (debug)')
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--save-every', type=int, default=20, help='flush predictions every N frames')
    ap.add_argument('--no-resume', action='store_true', help='ignore an existing predictions.json')
    ap.add_argument('--reverse', action='store_true',
                    help='process frames in reverse time order (to let a 2nd worker on another GPU share a variant)')
    return ap


class PredWriter(object):
    """Accumulates entries; resumes from an existing predictions.json of the same variant."""

    def __init__(self, args, method, info):
        self.path = os.path.join(args.out_dir, args.variant, 'predictions.json')
        self.method, self.info = method, dict(info)
        self.preds = {}
        self.no_merge = args.no_resume  # --no-resume: start fresh and overwrite the file
        if os.path.exists(self.path) and not args.no_resume:
            with open(self.path) as f:
                self.preds = json.load(f)['predictions']
            print('[resume] %d entries already in %s' % (len(self.preds), self.path))
        self.raw_path = os.path.join(args.out_dir, args.variant, 'raw_outputs.jsonl')
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.raw = open(self.raw_path, 'a')
        self.save_every = args.save_every
        self.n_new = 0
        self.t_start = time.time()
        _WRITERS.append(self)
        _install_signal_handler()

    def done(self, tok):
        return tok in self.preds

    def add(self, tok, t, trajs, scores=None, extra=None, raw=None):
        self.preds[tok] = make_entry(t, trajs, scores, extra)
        if raw is not None:
            self.raw.write(json.dumps(dict(token=tok, **raw)) + '\n')
            self.raw.flush()
        self.n_new += 1
        if self.n_new % self.save_every == 0:
            self.save()

    def save(self):
        """Write predictions.json. Under an fcntl lock, entries written meanwhile by another worker
        of the same variant (e.g. a second GPU on a disjoint / reversed frame order) are merged in
        first, so concurrent workers never drop each other's frames; own entries take precedence."""
        import fcntl
        with open(self.path + '.lock', 'w') as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            if os.path.exists(self.path) and not getattr(self, 'no_merge', False):
                try:
                    with open(self.path) as f:
                        disk = json.load(f)['predictions']
                    disk.update(self.preds)
                    self.preds = disk
                except (ValueError, KeyError) as e:
                    print('[PredWriter] could not merge existing file:', e)
            info = dict(self.info)
            info['stats'] = self.stats()
            save_predictions(self.path, self.method, self.preds, info)

    def stats(self):
        ex = [e.get('extra', {}) for e in self.preds.values()]
        n = len(ex)
        nfail = sum(1 for e in ex if e.get('parse_fail'))
        vocab = {}
        for e in ex:
            for w in e.get('parking_vocab', []):
                vocab[w] = vocab.get(w, 0) + 1
        any_park = sum(1 for e in ex if e.get('parking_vocab'))
        return {'n': n, 'parse_fail': nfail, 'parse_fail_pct': 100.0 * nfail / max(n, 1),
                'frames_with_parking_vocab': any_park, 'parking_vocab_counts': vocab}


_WRITERS = []
_HANDLER = []


def _install_signal_handler():
    """On SIGTERM / SIGINT flush every open writer before exiting (runs are resumable)."""
    import signal
    if _HANDLER:
        return

    def handler(signum, frame):
        for w in _WRITERS:
            try:
                if w.n_new:
                    w.save()
            except Exception as e:  # pragma: no cover
                print('[PredWriter] save on signal failed:', e)
        sys.exit(128 + signum)
    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)
    _HANDLER.append(handler)


def iterate(args, db):
    toks = select_tokens(db, args.every, args.offset, args.limit,
                         args.tokens.split(',') if args.tokens else None)
    return toks[::-1] if getattr(args, 'reverse', False) else toks


def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def eta(i, n, t0):
    el = time.time() - t0
    rate = el / max(i, 1)
    return '%d/%d  %.1fs/frame  elapsed %.0f min  eta %.0f min' % (i, n, rate, el / 60, rate * (n - i) / 60)


__all__ = ['base_argparser', 'PredWriter', 'iterate', 'batched', 'eta', 'parking_vocab', 'ParkingDataset',
           'np', 'ROOT', 'make_entry']
