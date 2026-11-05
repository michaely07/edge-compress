### CIFAR-100

Training: NVIDIA GeForce RTX 5070 Ti Laptop GPU. Latency (batch 1): Intel(R) Core(TM) Ultra 9 275HX, 4 threads, torch.compile on. CPU VNNI: yes. PyTorch 2.15.0+cu132, torchvision 0.29.2+cu132, torchao 0.18.0, Torch-Pruning 1.6.0, Triton 3.9.0.

Top-1 is the mean ± 95% CI over seeds (t-distribution; with 3 seeds t = 4.303, so intervals are wide). ECE is before → after temperature scaling.

| Config | Arch | Training | Compression | Top-1 (%) | Seeds | b1 median (ms) | b1 p90 (ms) | b32 (img/s) | MMACs | Params (M) | Size (MB) | ECE % → TS |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `r50_ce` | resnet50 | ce | none | 77.67 | 1 | 7.614 | 8.290 | 201 | 1308 | 23.71 | 90.73 | 9.13 → 2.01 |
| `r18_ce` | resnet18 | ce | none | 74.66 ± 0.49 | 3 | 2.815 | 3.285 | 520 | 557 | 11.22 | 42.88 | 9.40 → 2.37 |
| `r18_kd` | resnet18 | kd | none | 77.21 ± 0.34 | 3 | 3.007 | 3.655 | 514 | 557 | 11.22 | 42.88 | 11.52 → 2.71 |
| `mbv2_ce` | mobilenetv2 | ce | none | 67.24 ± 0.96 | 3 | 1.088 | 1.203 | 6113 | 26 | 2.35 | 9.20 | 5.16 → 1.21 |
| `mbv2_kd` | mobilenetv2 | kd | none | 69.63 ± 0.51 | 3 | 1.101 | 1.354 | 5869 | 26 | 2.35 | 9.20 | 17.09 → 3.30 |
| `r18_kd_ptq` | resnet18 | kd | int8-ptq | 77.19 ± 0.27 | 3 | 1.137 | 1.452 | 1853 | 557 | 11.22 | 11.11 | 11.57 → 2.74 |
| `mbv2_kd_ptq` | mobilenetv2 | kd | int8-ptq | 69.50 ± 0.39 | 3 | 1.256 | 1.524 | 10167 | 26 | 2.35 | 3.32 | 17.26 → 3.33 |
| `mbv2_kd_qat` | mobilenetv2 | kd | int8-qat | 69.60 ± 0.19 | 3 | 1.254 | 1.445 | 9610 | 26 | 2.35 | 3.32 | 16.96 → 3.70 |
| `r18_kd_prune50` | resnet18 | kd | prune50 | 75.00 ± 0.50 | 3 | 0.941 | 1.193 | 1999 | 140 | 2.82 | 10.82 | 13.23 → 2.85 |
| `r18_kd_prune50_ptq` | resnet18 | kd | prune50+int8-ptq | 74.96 ± 0.36 | 3 | 0.659 | 0.783 | 6567 | 140 | 2.82 | 3.07 | 13.26 → 2.74 |

**Paired comparisons** (difference on the same seed, which removes seed-to-seed variance):

| Comparison (same seed) | Δ top-1 (pts) | 95% CI | Seeds | Per-seed Δ |
|---|---|---|---|---|
| `r18_kd` − `r18_ce` | +2.55 | [+2.27, +2.83] | 3 | +2.45, +2.67, +2.53 |
| `mbv2_kd` − `mbv2_ce` | +2.38 | [+1.91, +2.85] | 3 | +2.30, +2.60, +2.25 |
| `r18_kd_ptq` − `r18_kd` | -0.02 | [-0.24, +0.20] | 3 | -0.07, +0.08, -0.07 |
| `mbv2_kd_ptq` − `mbv2_kd` | -0.13 | [-0.26, +0.00] | 3 | -0.15, -0.07, -0.17 |
| `mbv2_kd_qat` − `mbv2_kd_ptq` | +0.10 | [-0.11, +0.32] | 3 | +0.08, +0.20, +0.03 |
| `mbv2_kd_qat` − `mbv2_kd` | -0.03 | [-0.37, +0.32] | 3 | -0.07, +0.13, -0.14 |
| `r18_kd_prune50` − `r18_kd` | -2.21 | [-2.59, -1.83] | 3 | -2.05, -2.35, -2.23 |
| `r18_kd_prune50_ptq` − `r18_kd_prune50` | -0.04 | [-0.20, +0.12] | 3 | -0.09, +0.03, -0.06 |
