"""独立核验 mesh .ply（阶段2 验收条目 B3）。用法: python scripts/verify_mesh.py <mesh.ply> [...]"""
import sys, os, json
import numpy as np
import open3d as o3d

def check(path):
    m = o3d.io.read_triangle_mesh(path)
    V = np.asarray(m.vertices); F = np.asarray(m.triangles)
    info = {
        "path": path,
        "file_bytes": os.path.getsize(path),
        "n_vertices": int(V.shape[0]),
        "n_triangles": int(F.shape[0]),
        "has_vertex_colors": bool(m.has_vertex_colors()),
        "has_vertex_normals": bool(m.has_vertex_normals()),
        "vertices_finite": bool(np.isfinite(V).all()),
        "any_nan_vertices": bool(np.isnan(V).any()),
        "bbox_min": V.min(0).tolist() if V.size else None,
        "bbox_max": V.max(0).tolist() if V.size else None,
        "edge_manifold": bool(m.is_edge_manifold()),
        "vertex_manifold": bool(m.is_vertex_manifold()),
    }
    if F.size:
        labels, cluster_n_tri, _ = m.cluster_connected_triangles()
        cn = np.asarray(cluster_n_tri)
        info["n_connected_components"] = int(cn.shape[0])
        info["largest_component_tri"] = int(cn.max())
        info["largest_component_frac"] = float(cn.max() / cn.sum())
        info["n_components_lt100_tri"] = int((cn < 100).sum())
    return info

if __name__ == "__main__":
    out = [check(p) for p in sys.argv[1:]]
    print(json.dumps(out, indent=2, ensure_ascii=False))
