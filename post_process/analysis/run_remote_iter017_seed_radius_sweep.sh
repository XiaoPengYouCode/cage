#!/usr/bin/env bash
set -euo pipefail

REMOTE_ALIAS="${1:-wuyinyun}"
REMOTE_ROOT="${2:-/home/zhongjin_lu/project/cage}"
OUTPUT_DIR_REL="${OUTPUT_DIR_REL:-outputs/fjw_optimize_real_iter017/seed_radius_sweep_cvt500}"
SEED_COUNTS="${SEED_COUNTS:-500,750,1000,1250,1500}"
RADII_MM="${RADII_MM:-0.06,0.08,0.10,0.12,0.16,0.20,0.24,0.30}"
GAMMA="${GAMMA:-1.0}"
RNG_SEED="${RNG_SEED:-42}"
CVT_ITERS="${CVT_ITERS:-500}"
SUBDIVISION="${SUBDIVISION:-10}"
STAGES="${STAGES:-seeds,voronoi,skeleton}"
SKIP_MESH_EXPORT="${SKIP_MESH_EXPORT:-1}"

echo "remote=${REMOTE_ALIAS}"
echo "remote_root=${REMOTE_ROOT}"
echo "output_dir=${OUTPUT_DIR_REL}"
echo "seed_counts=${SEED_COUNTS}"
echo "radii_mm=${RADII_MM}"
echo "gamma=${GAMMA}"
echo "rng_seed=${RNG_SEED}"
echo "cvt_iters=${CVT_ITERS}"
echo "subdivision=${SUBDIVISION}"
echo "stages=${STAGES}"
echo "skip_mesh_export=${SKIP_MESH_EXPORT}"

mesh_arg=""
if [[ "${SKIP_MESH_EXPORT}" == "1" || "${SKIP_MESH_EXPORT}" == "true" ]]; then
  mesh_arg="--skip-mesh-export"
fi

REMOTE_CMD="
source ~/.local/bin/env 2>/dev/null || true
cd ${REMOTE_ROOT}
uv run python 'post_process/analysis/run_iter017_seed_radius_sweep.py' \
  --output-dir '${OUTPUT_DIR_REL}' \
  --seed-counts '${SEED_COUNTS}' \
  --radii-mm '${RADII_MM}' \
  --gamma '${GAMMA}' \
  --rng-seed '${RNG_SEED}' \
  --cvt-iters '${CVT_ITERS}' \
  --subdivision '${SUBDIVISION}' \
  --stages '${STAGES}' \
  ${mesh_arg}
"

if [[ "${REMOTE_ALIAS}" == "local" ]]; then
  bash -lc "${REMOTE_CMD}"
else
  ssh "${REMOTE_ALIAS}" "bash -lc $(printf '%q' "${REMOTE_CMD}")"
fi
