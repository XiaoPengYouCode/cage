from __future__ import annotations

"""Export a voxel occupancy grid as a GLB file for the Three.js viewer.

Uses Marching Cubes to extract the surface mesh, then serializes it as a
minimal glTF 2.0 binary (GLB) that the existing viewer can load directly.
"""

import struct
import json
from pathlib import Path

import numpy as np


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
        v, n, f = _exposed_faces(m["occupancy"], m["origin"], m["spacing"])
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
) -> Path:
    """Convert a voxel occupancy grid to a GLB by emitting only exposed faces.

    For each filled voxel, only the 6 faces that border an empty neighbour (or
    the grid boundary) are emitted.  Each face is two triangles with a flat
    outward normal.  This produces a clean, watertight-looking shell with no
    internal geometry and no winding ambiguity.

    Parameters
    ----------
    occupancy : (nx, ny, nz) bool array
    origin    : (3,) float, mm coordinates of voxel [0,0,0] corner
    spacing   : (3,) float, mm per voxel on each axis
    output_path : destination .glb file
    color     : RGB base color in linear space (default: bone/ivory)
    """
    verts, normals, faces = _exposed_faces(occupancy, origin, spacing)
    glb_bytes = _build_glb(verts, normals, faces, color)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(glb_bytes)
    return output_path


# Six face directions: (axis, sign, normal_vector)
# For each, we define 4 corner offsets in voxel units and 2 triangles.
_FACES = [
    # axis, step,  normal,           quad corners (in voxel-unit offsets, relative to voxel min corner)
    (0, -1, (-1, 0, 0), [(0,0,0),(0,0,1),(0,1,1),(0,1,0)]),  # -X
    (0, +1, (+1, 0, 0), [(1,0,0),(1,1,0),(1,1,1),(1,0,1)]),  # +X
    (1, -1, (0,-1, 0),  [(0,0,0),(1,0,0),(1,0,1),(0,0,1)]),  # -Y
    (1, +1, (0,+1, 0),  [(0,1,0),(0,1,1),(1,1,1),(1,1,0)]),  # +Y
    (2, -1, (0, 0,-1),  [(0,0,0),(0,1,0),(1,1,0),(1,0,0)]),  # -Z
    (2, +1, (0, 0,+1),  [(0,0,1),(1,0,1),(1,1,1),(0,1,1)]),  # +Z
]


def _exposed_faces(
    occupancy: np.ndarray,
    origin: np.ndarray,
    spacing: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (verts, normals, faces) for all exposed voxel faces."""
    nx, ny, nz = occupancy.shape
    sx, sy, sz = spacing

    all_verts: list[np.ndarray] = []
    all_norms: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    base_idx = 0

    # Pad with False so boundary voxels always have an empty neighbour
    pad = np.zeros((nx+2, ny+2, nz+2), dtype=bool)
    pad[1:-1, 1:-1, 1:-1] = occupancy

    filled_coords = np.argwhere(occupancy)  # (N, 3) in ix,iy,iz

    for ix, iy, iz in filled_coords:
        ox = origin[0] + ix * sx
        oy = origin[1] + iy * sy
        oz = origin[2] + iz * sz

        for axis, step, normal, corners in _FACES:
            # Check neighbour in padded array (pad offset = +1)
            ni = [ix+1, iy+1, iz+1]
            ni[axis] += step
            if pad[ni[0], ni[1], ni[2]]:
                continue  # neighbour is filled → internal face, skip

            # 4 vertices of this quad
            quad = np.array([
                [ox + c[0]*sx, oy + c[1]*sy, oz + c[2]*sz]
                for c in corners
            ], dtype=np.float32)

            all_verts.append(quad)
            all_norms.append(np.tile(normal, (4, 1)).astype(np.float32))
            # Two triangles: 0-1-2 and 0-2-3
            all_faces.append(np.array([
                [base_idx, base_idx+1, base_idx+2],
                [base_idx, base_idx+2, base_idx+3],
            ], dtype=np.uint32))
            base_idx += 4

    if not all_verts:
        raise ValueError("No exposed faces found — occupancy grid may be empty.")

    verts   = np.concatenate(all_verts,  axis=0)
    normals = np.concatenate(all_norms,  axis=0)
    faces   = np.concatenate(all_faces,  axis=0)
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
            "doubleSided": False,
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
            "doubleSided": False,
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
