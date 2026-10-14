#!/usr/bin/env python
"""Pre-render virtual NAVSIM (nuPlan) camera images for the dataset keyframes.

See olp_navsim/virtual_cam.py for the camera model. Output:
  <out>/{distorted|pinhole}/<scene>/<CAM_X>/<sample_token>.jpg   (1920x1080, JPEG q95)

  python methods/navsim/make_navsim_virtual.py --cams CAM_F0 CAM_L0 CAM_R0 CAM_B0 --workers 8
  python methods/navsim/make_navsim_virtual.py --no-distort        # pinhole ablation
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from olp_navsim import DEFAULT_VIRTUAL_DIR, OLPNavsimInputs  # noqa: E402
from olp_navsim import ROOT  # noqa: E402
from olpcommon import DEFAULT_DATAROOT, ParkingDataset  # noqa: E402

G = {}


def work(job):
    tok, cams = job
    inp = G['inp']
    images = {}
    done = 0
    for cam in cams:
        p = inp.virtual_path(tok, cam)
        if os.path.exists(p):
            continue
        os.makedirs(os.path.dirname(p), exist_ok=True)
        from PIL import Image
        im = inp.renderer.render(tok, cam, images=images)
        Image.fromarray(im).save(p + '.tmp.jpg', quality=95)
        os.replace(p + '.tmp.jpg', p)
        done += 1
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataroot', default=DEFAULT_DATAROOT)
    ap.add_argument('--scenes', default=os.environ.get('OLP_SCENES', os.path.join(ROOT, 'scenes_subset.txt')))
    ap.add_argument('--out', default=DEFAULT_VIRTUAL_DIR)
    ap.add_argument('--cams', nargs='+', default=['CAM_F0', 'CAM_L0', 'CAM_R0', 'CAM_B0'])
    ap.add_argument('--no-distort', action='store_true', help='pinhole virtual cameras (ablation)')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()

    db = ParkingDataset(args.dataroot, args.scenes, load_agents=False)
    inp = OLPNavsimInputs(db, distort=not args.no_distort, virtual_dir=args.out)
    t0 = time.time()
    for cam in args.cams:  # build the remap tables once in the parent (shared after fork)
        _, cov = inp.renderer.maps(db.keyframes[0], cam)
        print('%s coverage %.3f' % (cam, cov))
    print('maps ready in %.1fs' % (time.time() - t0))
    G['inp'] = inp
    jobs = [(tok, args.cams) for tok in db.keyframes]
    n = 0
    with Pool(args.workers) as pool:
        for i, d in enumerate(pool.imap_unordered(work, jobs, chunksize=4)):
            n += d
            if i % 100 == 0:
                print('  %d/%d keyframes, %d images written, %.0fs' % (i, len(jobs), n, time.time() - t0))
    print('done: %d images in %.0fs -> %s' % (n, time.time() - t0, args.out))


if __name__ == '__main__':
    main()
