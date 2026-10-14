# Kinematic reference planners (no learning)

`run.py` writes three predictors, each 6.4 s at 10 Hz:

| name | plan |
|---|---|
| `stationary` | stay at the current pose |
| `const_vel` | keep the current velocity vector (ego frame) |
| `ctrv` | keep the current signed speed and yaw rate (yaw rate clipped to ±0.6 rad/s) |

Inputs come only from the past (`ParkingDataset.ego_state`): a least-squares fit over the last
0.6 s of (deduplicated) ego poses for velocity and yaw rate. Reversing gives a negative
speed, so CV and CTRV also extrapolate backwards.

```bash
python methods/kinematic/run.py --scenes scenes_subset.txt   # base python3 (numpy only), ~5 s
```

Outputs: `outputs/{stationary,const_vel,ctrv}/predictions.json`.

These give a floor that any learned planner should beat. On this subset, CTRV has
ADE/FDE@3s = 0.25 / 0.60 m (`docs/notes.md` §6).
