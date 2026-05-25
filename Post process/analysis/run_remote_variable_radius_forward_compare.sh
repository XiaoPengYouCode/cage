#!/usr/bin/env bash
set -euo pipefail

REMOTE_ALIAS="${1:-wuyinyun}"
REMOTE_ROOT="${2:-/home/zhongjin_lu/project/cage}"
DESIGN_MODE="${DESIGN_MODE:-modulus_weighted}"
LOAD_CASES="${LOAD_CASES:-force_1}"
REPLACEMENT_NPZ_REL="${REPLACEMENT_NPZ_REL:-outputs/fjw_optimize_real_iter017/fjw_iter017_replacement_design_variable_radius.npz}"

REMOTE_CMD="
source ~/.local/bin/env 2>/dev/null || true
cd ${REMOTE_ROOT}
uv run python 'Post process/analysis/compare_iter017_skeleton_vs_density.py' \
  --stage run_comparison \
  --design-mode '${DESIGN_MODE}' \
  --replacement-npz '${REPLACEMENT_NPZ_REL}'"

IFS=',' read -r -a LOAD_CASE_ARRAY <<< "${LOAD_CASES}"
for case_name in "${LOAD_CASE_ARRAY[@]}"; do
  REMOTE_CMD="${REMOTE_CMD} --load-case '${case_name}'"
done

REMOTE_CMD="${REMOTE_CMD}
"

echo "remote=${REMOTE_ALIAS}"
echo "remote_root=${REMOTE_ROOT}"
echo "design_mode=${DESIGN_MODE}"
echo "load_cases=${LOAD_CASES}"
echo "replacement_npz=${REPLACEMENT_NPZ_REL}"

ssh "${REMOTE_ALIAS}" "LOAD_CASES='${LOAD_CASES}' ${REMOTE_CMD}"
