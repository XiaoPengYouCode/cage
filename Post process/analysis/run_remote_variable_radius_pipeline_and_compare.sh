#!/usr/bin/env bash
set -euo pipefail

REMOTE_ALIAS="${1:-wuyinyun}"
REMOTE_ROOT="${2:-/home/zhongjin_lu/project/cage}"
SUBDIVISION="${SUBDIVISION:-6}"
DESIGN_MODE="${DESIGN_MODE:-modulus_weighted}"
LOAD_CASES="${LOAD_CASES:-force_1}"

REMOTE_CMD=$'source ~/.local/bin/env 2>/dev/null || true\n'
REMOTE_CMD+=$'cd '"${REMOTE_ROOT}"$'\n'
REMOTE_CMD+=$'uv run python '\''Post process/analysis/build_iter017_variable_radius_edges.py'\''\n'
REMOTE_CMD+=$'uv run python '\''Post process/analysis/build_iter017_variable_radius_skeleton.py'\'' --subdivision '\'''"${SUBDIVISION}"$'\'' --skip-mesh-export\n'
REMOTE_CMD+=$'uv run python '\''Post process/analysis/build_iter017_variable_radius_replacement_design.py'\''\n'
REMOTE_CMD+=$'uv run python '\''Post process/analysis/compare_iter017_skeleton_vs_density.py'\'' --stage run_comparison --design-mode '\'''"${DESIGN_MODE}"$'\'' --replacement-npz '\''outputs/fjw_optimize_real_iter017/fjw_iter017_replacement_design_variable_radius.npz'\'''

IFS=',' read -r -a LOAD_CASE_ARRAY <<< "${LOAD_CASES}"
for case_name in "${LOAD_CASE_ARRAY[@]}"; do
  REMOTE_CMD+=$' --load-case '\'''"${case_name}"$'\''
done

REMOTE_CMD+=$'\n'

echo "remote=${REMOTE_ALIAS}"
echo "remote_root=${REMOTE_ROOT}"
echo "subdivision=${SUBDIVISION}"
echo "design_mode=${DESIGN_MODE}"
echo "load_cases=${LOAD_CASES}"

if [[ "${REMOTE_ALIAS}" == "local" ]]; then
  bash -lc "${REMOTE_CMD}"
else
  ssh "${REMOTE_ALIAS}" "${REMOTE_CMD}"
fi
