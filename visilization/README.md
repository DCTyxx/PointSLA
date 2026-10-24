# PointSLA qualitative visualization

This directory contains the Baseline-vs-Our qualitative evaluation pipeline.
The directory name intentionally follows the project convention requested for
this experiment: `visilization`.

## Default protocol

- datasets: `scannet`, `scannet200`, `scannetpp`
- K-shot: `K=1`
- Our checkpoint: choose the registration with the highest precise HM from
  `exp/<dataset>/split_phase3_k1/regis1..5/phase3_metrics_final.json`, then load
  `model/model_best.pth`
- Baseline checkpoint: official GFS-VL checkpoint from `pres_exp`, using the
  **same registration** as Our model
- inference: repository `GFSSemSegTester`, fragment inference on the validation set
- scene ranking: descending `Our scene mIoU - Baseline scene mIoU`
- showcase: top 20 scenes per dataset

Using the same registration is important: the Baseline and Our model then use the
same few-shot support split, so qualitative gains are not caused by different
registration samples.

## Run

From repository root:

```bash
chmod +x visilization/run_visualization.sh
GPU=4 bash visilization/run_visualization.sh
```

Only one dataset:

```bash
DATASETS="scannetpp" GPU=4 bash visilization/run_visualization.sh
```

K=5 is also supported if the corresponding split Phase-3 experiments exist:

```bash
K=5 TOPK=20 GPU=4 bash visilization/run_visualization.sh
```

Useful environment variables:

- `PYTHON_BIN`: Python executable, default `python`
- `GPU`: physical GPU id, default `0`
- `DATASETS`: space/comma-separated dataset list
- `K`: `1` or `5`, default `1`
- `TOPK`: number of showcase scenes per dataset, default `20`
- `PRETRAIN_ROOT`: official GFS-VL checkpoint root, default `./pres_exp`
- `VIS_ROOT`: output root, default `./visilization`
- `REUSE_EVAL`: reuse matching completed inference results, default `1`

## Output

```text
visilization/
├── evaluation/
│   └── <dataset>/
│       ├── baseline/
│       │   ├── test.log
│       │   ├── checkpoint.txt
│       │   └── scenes/<scene>/
│       │       ├── query.npy
│       │       ├── query_label.npy
│       │       └── query_pred.npy
│       └── ours/
│           └── ...
├── comparison/
│   └── <dataset>/
│       ├── evaluation_summary.json
│       ├── scene_metrics.csv
│       ├── scene_metrics.json
│       ├── showcase_selection.csv
│       └── showcase_selection.json
└── showcase/
    └── <dataset>/
        ├── 01_<scene>/
        │   ├── input.txt
        │   ├── GT.txt
        │   ├── baseline.txt
        │   └── Our.txt
        └── ... 20 scenes
```

## Showcase file format

Every text file uses the same seven columns:

```text
x y z label r g b
```

- `input.txt`: raw XYZ/RGB from the processed dataset source; `label=-1`
- `GT.txt`: unified GFS ground-truth label + semantic RGB
- `baseline.txt`: Baseline prediction + semantic RGB
- `Our.txt`: Our prediction + semantic RGB

The raw XYZ is reloaded from the original validation scene rather than from the
legacy `query.npy`. This is required because the tester inverse-maps predictions
to full point resolution, while `query.npy` can still contain GridSample-resolution
coordinates.

## Semantic colors

- ScanNet: repository benchmark `SCANNET_COLOR_MAP_20`
- ScanNet200: repository benchmark `SCANNET_COLOR_MAP_200`
- ScanNet++: the current repository does not provide an official semantic RGB
  table for the top-100 benchmark. A deterministic palette keyed by the official
  top-100 class index is therefore used and explicitly recorded as a fallback in
  `evaluation_summary.json`.

For ScanNet/ScanNet200, GFS output labels are first mapped by class name back to
the original benchmark class ID before the official color is assigned. This is
necessary because the GFS Base+Novel class order differs from the official class
order.

## Scene selection

For each scene, mIoU is computed over classes active in that scene (`union > 0`).
The primary ranking score is:

```text
delta_mIoU = ours_scene_mIoU - baseline_scene_mIoU
```

The top `TOPK` scenes are exported. `scene_metrics.csv` retains the full ranking,
including scene Novel-mIoU and point-accuracy differences, so the final paper
figures can be manually narrowed further if desired.
