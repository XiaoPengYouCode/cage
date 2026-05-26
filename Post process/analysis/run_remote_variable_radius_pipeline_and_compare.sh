#!/usr/bin/env bash
set -euo pipefail

REMOTE_ALIAS="${1:-wuyinyun}"
REMOTE_ROOT="${2:-/home/zhongjin_lu/project/cage}"
SUBDIVISION="${SUBDIVISION:-6}"
DESIGN_MODE="${DESIGN_MODE:-modulus_weighted}"
LOAD_CASES="${LOAD_CASES:-force_1}"
AGGREGATION_MODE="${AGGREGATION_MODE:-mean_only}"
LOCAL_SUPPORT_RADIUS_CELLS="${LOCAL_SUPPORT_RADIUS_CELLS:-1.5}"
COMPARISON_TAG="${COMPARISON_TAG:-${AGGREGATION_MODE}}"
LOOKUP_JSON_REL="${LOOKUP_JSON_REL:-Post process/analysis/output/iter017_band_radius_lookup_combined_seed55_plus_lowmid.json}"
EDGES_NPZ_REL="${EDGES_NPZ_REL:-outputs/fjw_optimize_real_iter017/fjw_iter017_voronoi_edges_variable_radius_seed55_plus_lowmid.npz}"
SKELETON_NPZ_REL="${SKELETON_NPZ_REL:-outputs/fjw_optimize_real_iter017/fjw_iter017_skeleton_voxels_variable_radius_seed55_plus_lowmid.npz}"
REPLACEMENT_NPZ_REL="${REPLACEMENT_NPZ_REL:-outputs/fjw_optimize_real_iter017/fjw_iter017_replacement_design_variable_radius_seed55_plus_lowmid.npz}"

IFS=',' read -r -a LOAD_CASE_ARRAY <<< "${LOAD_CASES}"

echo "remote=${REMOTE_ALIAS}"
echo "remote_root=${REMOTE_ROOT}"
echo "subdivision=${SUBDIVISION}"
echo "design_mode=${DESIGN_MODE}"
echo "load_cases=${LOAD_CASES}"
echo "aggregation_mode=${AGGREGATION_MODE}"
echo "local_support_radius_cells=${LOCAL_SUPPORT_RADIUS_CELLS}"
echo "comparison_tag=${COMPARISON_TAG}"
echo "lookup_json=${LOOKUP_JSON_REL}"
echo "edges_npz=${EDGES_NPZ_REL}"
echo "skeleton_npz=${SKELETON_NPZ_REL}"
echo "replacement_npz=${REPLACEMENT_NPZ_REL}"

if [[ "${REMOTE_ALIAS}" == "local" ]]; then
  source ~/.local/bin/env 2>/dev/null || true
  cd "${REMOTE_ROOT}"
  uv run python "Post process/analysis/build_iter017_variable_radius_edges.py" \
    --lookup-json "${LOOKUP_JSON_REL}" \
    --output-npz "${EDGES_NPZ_REL}"
  uv run python "Post process/analysis/build_iter017_variable_radius_skeleton.py" \
    --variable-radius-edges-npz "${EDGES_NPZ_REL}" \
    --output-npz "${SKELETON_NPZ_REL}" \
    --subdivision "${SUBDIVISION}" \
    --skip-mesh-export
  uv run python "Post process/analysis/build_iter017_variable_radius_replacement_design.py" \
    --skeleton-npz "${SKELETON_NPZ_REL}" \
    --lookup-json "${LOOKUP_JSON_REL}" \
    --output-npz "${REPLACEMENT_NPZ_REL}" \
    --aggregation-mode "${AGGREGATION_MODE}" \
    --local-support-radius-cells "${LOCAL_SUPPORT_RADIUS_CELLS}"
  compare_args=(
    uv run python "Post process/analysis/compare_iter017_skeleton_vs_density.py"
    --stage run_comparison
    --design-mode "${DESIGN_MODE}"
    --replacement-npz "${REPLACEMENT_NPZ_REL}"
    --comparison-tag "${COMPARISON_TAG}"
  )
  for case_name in "${LOAD_CASE_ARRAY[@]}"; do
    compare_args+=(--load-case "${case_name}")
  done
  "${compare_args[@]}"
else
  remote_script="source ~/.local/bin/env 2>/dev/null || true
cd ${REMOTE_ROOT}
uv run python 'Post process/analysis/build_iter017_variable_radius_edges.py' --lookup-json '${LOOKUP_JSON_REL}' --output-npz '${EDGES_NPZ_REL}'
uv run python 'Post process/analysis/build_iter017_variable_radius_skeleton.py' --variable-radius-edges-npz '${EDGES_NPZ_REL}' --output-npz '${SKELETON_NPZ_REL}' --subdivision '${SUBDIVISION}' --skip-mesh-export
uv run python 'Post process/analysis/build_iter017_variable_radius_replacement_design.py' --skeleton-npz '${SKELETON_NPZ_REL}' --lookup-json '${LOOKUP_JSON_REL}' --output-npz '${REPLACEMENT_NPZ_REL}' --aggregation-mode '${AGGREGATION_MODE}' --local-support-radius-cells '${LOCAL_SUPPORT_RADIUS_CELLS}'
uv run python 'Post process/analysis/compare_iter017_skeleton_vs_density.py' --stage run_comparison --design-mode '${DESIGN_MODE}' --replacement-npz '${REPLACEMENT_NPZ_REL}' --comparison-tag '${COMPARISON_TAG}'"
  for case_name in "${LOAD_CASE_ARRAY[@]}"; do
    remote_script="${remote_script} --load-case '${case_name}'"
  done
  ssh "${REMOTE_ALIAS}" "bash -lc $(printf '%q' "${remote_script}")"
fi
