from __future__ import annotations

"""Export a voxel occupancy grid as a GLB file for the Three.js viewer.

Uses Marching Cubes (skimage.measure.marching_cubes) to extract a smooth
isosurface from the voxel scalar field, then serializes it as a minimal
glTF 2.0 binary (GLB) that the existing viewer can load directly.

Why Marching Cubes over exposed-face meshing
--------------------------------------------
The previous approach emitted one quad per exposed voxel face, producing a
staircase surface with flat per-face normals.  Marching Cubes interpolates
the iso-surface crossing within each voxel cube, yielding a smooth triangulated
mesh with per-vertex normals.  The result has fewer triangles, no staircase
artefacts, and renders with smooth shading in Three.js.

The scalar field fed to MC is the float32 occupancy (0.0 or 1.0).  The
iso-level is 0.5 — exactly midway, so the surface is centred on the
voxel boundary between filled and empty cells, matching the old geometry's
position.  Padding with one layer of zeros on all sides ensures that closed
surfaces are generated at the domain boundary rather than open edge loops.
"""

import struct
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes


def multi_voxels_to_glb(
    meshes: list[dict],
    output_path: Path,
) -> Path:
    """Export multiple voxel occupancy grids as a single GLB with per-mesh colors.

    Parameters
    ----------
    meshes : list of dicts, each with keys:
        occupancy : (nx, ny, nz) bool array
        origin    : (3,) float, mm
        spacing   : (3,) float, mm
        color     : (r, g, b) float tuple
        name      : str (optional)
    output_path : destination .glb file
    """
    parts = []
    for m in meshes:
        v, n, f = _marching_cubes_surface(m["occupancy"], m["origin"], m["spacing"], m.get("smooth_sigma", 1.0))
        parts.append((v, n, f, m["color"], m.get("name", "mesh")))
    glb_bytes = _build_glb_multi(parts)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(glb_bytes)
    return output_path


def voxels_to_glb(
    occupancy: np.ndarray,
    origin: np.ndarray,
    spacing: np.ndarray,
    output_path: Path,
    color: tuple[float, float, float] = (0.76, 0.60, 0.42),
    smooth_sigma: float = 1.0,
) -> Path:
    """Convert a voxel occupancy grid to a GLB using Marching Cubes.

    Parameters
    ----------
    occupancy    : (nx, ny, nz) bool array
    origin       : (3,) float, mm coordinates of voxel [0,0,0] corner
    spacing      : (3,) float, mm per voxel on each axis
    output_path  : destination .glb file
    color        : RGB base color in linear space (default: bone/ivory)
    smooth_sigma : Gaussian pre-smoothing sigma in voxels (default 1.0)
    """
    verts, normals, faces = _marching_cubes_surface(occupancy, origin, spacing, smooth_sigma)
    glb_bytes = _build_glb(verts, normals, faces, color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(glb_bytes)
    return output_path


def _marching_cubes_surface(
    occupancy: np.ndarray,
    origin: np.ndarray,
    spacing: np.ndarray,
    smooth_sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract an isosurface via Marching Cubes and return (verts, normals, faces).

    The occupancy grid is padded with one layer of zeros on every side so that
    the MC algorithm always sees a closed boundary — without padding, voxels
    touching the grid edge produce open surface edges instead of a watertight cap.

    MC returns vertex coordinates in voxel index space (accounting for padding).
    We convert them to world coordinates using origin and spacing, then shift
    back by one voxel to undo the padding offset.

    Parameters
    ----------
    smooth_sigma : Gaussian blur sigma applied to the binary field before MC
                   (in voxels).  Smooths the 0/1 step into a gradient so MC
                   produces a smoother isosurface.  sigma=0 disables smoothing;
                   sigma=1.0 gives noticeably smoother rods with minimal shrinkage.
    """
    if not np.any(occupancy):
        raise ValueError("No occupied voxels — occupancy grid is empty.")

    # Pad with one empty layer on all sides to guarantee closed surfaces
    field = np.pad(occupancy.astype(np.float32), pad_width=1, mode="constant", constant_values=0.0)

    # Gaussian pre-smoothing: softens the hard 0/1 boundary so MC interpolates
    # a smoother isosurface.  The iso-level stays at 0.5 (field midpoint).
    if smooth_sigma > 0:
        field = gaussian_filter(field, sigma=smooth_sigma)

    mc_verts, mc_faces, mc_normals, _ = marching_cubes(
        field,
        level=0.5,
        spacing=(spacing[0], spacing[1], spacing[2]),
        allow_degenerate=False,
    )

    # MC vertex coordinates are in mm (spacing applied), but the origin is at
    # the pad=1 voxel, so shift back by one voxel in each axis.
    mc_verts = mc_verts - spacing  # undo pad offset
    mc_verts = mc_verts + origin   # translate to world origin

    # skimage marching_cubes uses the gradient direction to determine winding
    # order, which for a solid field (interior=1, exterior=0) produces
    # inward-facing normals.  Reversing the winding order of every triangle
    # (swap v1↔v2) flips the geometric normal to point outward without
    # touching the per-vertex normals skimage computed.
    verts   = mc_verts.astype(np.float32)
    normals = mc_normals.astype(np.float32)
    faces   = mc_faces[:, [0, 2, 1]].astype(np.uint32)  # swap v1↔v2 → outward winding

    return verts, normals, faces


# ---------------------------------------------------------------------------
# Minimal glTF 2.0 / GLB serialisation
# ---------------------------------------------------------------------------

def _pad4(data: bytes, pad_byte: bytes = b"\x00") -> bytes:
    rem = len(data) % 4
    return data + pad_byte * (4 - rem) if rem else data


def _build_glb(
    verts: np.ndarray,
    normals: np.ndarray,
    faces: np.ndarray,
    color: tuple[float, float, float],
) -> bytes:
    n_verts = len(verts)
    n_faces = len(faces)

    pos_bytes  = verts.tobytes()
    norm_bytes = normals.tobytes()
    idx_bytes  = faces.flatten().tobytes()

    # Bounding box for accessor
    pos_min = verts.min(axis=0).tolist()
    pos_max = verts.max(axis=0).tolist()

    # BIN chunk layout: [positions | normals | indices]
    pos_offset  = 0
    pos_len     = len(pos_bytes)
    norm_offset = pos_len
    norm_len    = len(norm_bytes)
    idx_offset  = norm_offset + norm_len
    idx_len     = len(idx_bytes)

    bin_data = _pad4(pos_bytes + norm_bytes + idx_bytes)
    bin_len  = len(bin_data)

    gltf = {
        "asset": {"version": "2.0", "generator": "cage ct-reconstruction"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "lumbar_vertebra"}],
        "meshes": [{
            "name": "lumbar_vertebra",
            "primitives": [{
                "attributes": {"POSITION": 0, "NORMAL": 1},
                "indices": 2,
                "material": 0,
                "mode": 4,
            }],
        }],
        "materials": [{
            "name": "bone",
            "pbrMetallicRoughness": {
                "baseColorFactor": [color[0], color[1], color[2], 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.65,
            },
            "doubleSided": True,
        }],
        "accessors": [
            {   # 0 — positions
                "bufferView": 0,
                "byteOffset": pos_offset,
                "componentType": 5126,  # FLOAT
                "count": n_verts,
                "type": "VEC3",
                "min": pos_min,
                "max": pos_max,
            },
            {   # 1 — normals
                "bufferView": 0,
                "byteOffset": norm_offset,
                "componentType": 5126,
                "count": n_verts,
                "type": "VEC3",
            },
            {   # 2 — indices
                "bufferView": 1,
                "byteOffset": 0,
                "componentType": 5125,  # UNSIGNED_INT
                "count": n_faces * 3,
                "type": "SCALAR",
            },
        ],
        "bufferViews": [
            {   # 0 — vertex data
                "buffer": 0,
                "byteOffset": 0,
                "byteLength": pos_len + norm_len,
                "target": 34962,  # ARRAY_BUFFER
            },
            {   # 1 — index data
                "buffer": 0,
                "byteOffset": idx_offset,
                "byteLength": idx_len,
                "target": 34963,  # ELEMENT_ARRAY_BUFFER
            },
        ],
        "buffers": [{"byteLength": bin_len}],
    }

    json_bytes = _pad4(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), b" ")

    # GLB header + JSON chunk + BIN chunk
    json_chunk = struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes
    bin_chunk  = struct.pack("<II", bin_len,          0x004E4942) + bin_data
    total_len  = 12 + len(json_chunk) + len(bin_chunk)
    header     = struct.pack("<III", 0x46546C67, 2, total_len)

    return header + json_chunk + bin_chunk


def _build_glb_multi(
    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray, tuple, str]],
) -> bytes:
    """Build a GLB with one node+mesh per part, each with its own material."""
    # Lay out BIN: for each part [positions | normals | indices]
    bin_segments: list[bytes] = []
    byte_offset = 0
    accessors = []
    buffer_views = []
    nodes = []
    meshes = []
    materials = []

    for i, (verts, normals, faces, color, name) in enumerate(parts):
        pos_bytes  = verts.tobytes()
        norm_bytes = normals.tobytes()
        idx_bytes  = faces.flatten().tobytes()
        n_verts = len(verts)
        n_faces = len(faces)

        pos_min = verts.min(axis=0).tolist()
        pos_max = verts.max(axis=0).tolist()

        # Pad each segment to 4-byte boundary
        pos_padded  = _pad4(pos_bytes)
        norm_padded = _pad4(norm_bytes)
        idx_padded  = _pad4(idx_bytes)

        pos_off  = byte_offset
        norm_off = pos_off  + len(pos_padded)
        idx_off  = norm_off + len(norm_padded)

        bin_segments += [pos_padded, norm_padded, idx_padded]
        byte_offset   = idx_off + len(idx_padded)

        acc_pos  = len(accessors)
        acc_norm = acc_pos + 1
        acc_idx  = acc_pos + 2

        accessors += [
            {"bufferView": len(buffer_views),     "byteOffset": 0, "componentType": 5126, "count": n_verts, "type": "VEC3", "min": pos_min,  "max": pos_max},
            {"bufferView": len(buffer_views) + 1, "byteOffset": 0, "componentType": 5126, "count": n_verts, "type": "VEC3"},
            {"bufferView": len(buffer_views) + 2, "byteOffset": 0, "componentType": 5125, "count": n_faces * 3, "type": "SCALAR"},
        ]
        buffer_views += [
            {"buffer": 0, "byteOffset": pos_off,  "byteLength": len(pos_bytes),  "target": 34962},
            {"buffer": 0, "byteOffset": norm_off, "byteLength": len(norm_bytes), "target": 34962},
            {"buffer": 0, "byteOffset": idx_off,  "byteLength": len(idx_bytes),  "target": 34963},
        ]
        mat_idx = len(materials)
        materials.append({
            "name": name,
            "pbrMetallicRoughness": {
                "baseColorFactor": [color[0], color[1], color[2], 1.0],
                "metallicFactor": 0.0,
                "roughnessFactor": 0.65,
            },
            "doubleSided": True,
        })
        mesh_idx = len(meshes)
        meshes.append({
            "name": name,
            "primitives": [{"attributes": {"POSITION": acc_pos, "NORMAL": acc_norm}, "indices": acc_idx, "material": mat_idx, "mode": 4}],
        })
        nodes.append({"mesh": mesh_idx, "name": name})

    bin_data = b"".join(bin_segments)
    bin_len  = len(bin_data)

    gltf = {
        "asset": {"version": "2.0", "generator": "cage ct-reconstruction"},
        "scene": 0,
        "scenes": [{"nodes": list(range(len(nodes)))}],
        "nodes": nodes,
        "meshes": meshes,
        "materials": materials,
        "accessors": accessors,
        "bufferViews": buffer_views,
        "buffers": [{"byteLength": bin_len}],
    }

    json_bytes = _pad4(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), b" ")
    json_chunk = struct.pack("<II", len(json_bytes), 0x4E4F534A) + json_bytes
    bin_chunk  = struct.pack("<II", bin_len,          0x004E4942) + bin_data
    total_len  = 12 + len(json_chunk) + len(bin_chunk)
    header     = struct.pack("<III", 0x46546C67, 2, total_len)
    return header + json_chunk + bin_chunk
