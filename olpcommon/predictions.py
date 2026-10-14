"""Common prediction file format used by every method adapter.

predictions.json:
{
  "method": "vad_base",
  "info": {...free-form: checkpoint, config, input adaptations...},
  "frame": "ego",            # x forward, y left (m), origin = ego origin at the keyframe
  "predictions": {
    "<sample_token>": {
      "t": [0.5, 1.0, ...],                 # seconds after the keyframe (T values)
      "trajs": [[[x, y], ...], ...],        # K x T x 2; trajs[0] is the method's top-1 plan
      "scores": [...],                      # optional, K values (higher = better)
      "extra": {...}                        # optional (e.g. VLM reasoning text, command used)
    }, ...
  }
}
"""
import json
import os

import numpy as np


def _tolist(x):
    return np.asarray(x, dtype=np.float64).round(4).tolist()


def make_entry(t, trajs, scores=None, extra=None):
    trajs = np.asarray(trajs, dtype=np.float64)
    if trajs.ndim == 2:
        trajs = trajs[None]
    e = {'t': _tolist(t), 'trajs': _tolist(trajs)}
    if scores is not None:
        e['scores'] = _tolist(scores)
    if extra:
        e['extra'] = extra
    return e


def save_predictions(path, method, predictions, info=None):
    d = os.path.dirname(os.path.abspath(path))
    if not os.path.isdir(d):
        os.makedirs(d)
    out = {'method': method, 'info': info or {}, 'frame': 'ego', 'predictions': predictions}
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(out, f)
    os.replace(tmp, path)
    print('[predictions] wrote %d entries to %s' % (len(predictions), path))


def load_predictions(path):
    with open(path) as f:
        d = json.load(f)
    for e in d['predictions'].values():
        e['t'] = np.asarray(e['t'], dtype=np.float64)
        e['trajs'] = np.asarray(e['trajs'], dtype=np.float64)
        if 'scores' in e:
            e['scores'] = np.asarray(e['scores'], dtype=np.float64)
    return d
