#!/usr/bin/env bash
# Dump every compiler stage (TTIR -> TTGIR -> LLIR -> PTX -> cubin) for the KD kernels.
# Run on a GPU machine from the repo root: bash triton_kd/dump_ir.sh
set -euo pipefail
OUT=${1:-triton_kd/ir_dump}
rm -rf "$OUT" && mkdir -p "$OUT"
TRITON_ALWAYS_COMPILE=1 TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR="$OUT" \
  python triton_kd/bench.py --B 256 --Vs 32768 --impls fused_v1 fused_v2 --out "$OUT/bench"
# Per-pass MLIR for the v2 forward kernel (large):
MLIR_ENABLE_DUMP=kd_fwd_kernel_v2 python triton_kd/bench.py --B 256 --Vs 32768 --impls fused_v2 \
  --out "$OUT/bench" 2> "$OUT/passes_kd_fwd_kernel_v2.mlir" || true

echo "stages written:"; find "$OUT" -name "*.ttir" -o -name "*.ttgir" -o -name "*.llir" -o -name "*.ptx" | sort
echo
echo "Where to look:"
echo " TTGIR layout:      grep -m3 '#ttg.blocked\|#triton_gpu.blocked' $OUT/*/*.ttgir"
echo " warp shuffles:     grep -c 'shfl.sync' $OUT/*/*kd_fwd_kernel_v2*.ptx"
echo " vectorized loads:  grep -m5 'ld.global.v4' $OUT/*/*kd_fwd_kernel_v2*.ptx"
