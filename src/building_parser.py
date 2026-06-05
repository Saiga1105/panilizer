import json
from pathlib import Path

import numpy as np
import open3d as o3d

try:
    import mapbox_earcut as earcut
except ImportError:
    earcut = None

try:
    from shapely.geometry import Polygon
    from shapely.ops import triangulate
except ImportError:
    Polygon = None
    triangulate = None


CATEGORY_COLORS = {
    "roof": [0.75, 0.22, 0.17],
    "wall": [0.78, 0.66, 0.51],
    "reveal": [0.60, 0.48, 0.36],
    "balcony": [0.56, 0.27, 0.68],
    "other": [0.45, 0.62, 0.80],
}

SEMANTIC_COLORS = {
    "RoofSurface": [0.75, 0.22, 0.17],
    "RoofSurface:overhang": [0.95, 0.55, 0.15],
    "WallSurface": [0.78, 0.66, 0.51],
    "WallSurface:reveal": [0.60, 0.48, 0.36],
    "WallSurface:balkon": [0.56, 0.27, 0.68],
    "GroundSurface": [0.50, 0.55, 0.55],
    "GroundSurface:balkon": [0.72, 0.39, 0.80],
    "Window": [0.36, 0.68, 0.89],
    "Door": [0.90, 0.49, 0.13],
    "Unknown": [0.74, 0.76, 0.78],
}


def load_json_file(path: Path | str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def semantic_key(semantic_type: str) -> str:
    key = str(semantic_type or "Unknown").lower()
    for old, new in [(":", "_"), (" ", "_"), ("-", "_"), ("/", "_")]:
        key = key.replace(old, new)
    return "".join(char for char in key if char.isalnum() or char == "_") or "unknown"


def _vertices_world(cityjson: dict) -> np.ndarray:
    vertices = np.asarray(cityjson.get("vertices", []), dtype=float)
    transform = cityjson.get("transform", {}) or {}
    scale = np.asarray(transform.get("scale", [1.0, 1.0, 1.0]), dtype=float)
    translate = np.asarray(transform.get("translate", [0.0, 0.0, 0.0]), dtype=float)
    return vertices * scale + translate


def _iter_surfaces(geometry: dict):
    boundaries = geometry.get("boundaries", []) or []
    geom_type = geometry.get("type")

    if geom_type in {"MultiSurface", "CompositeSurface"}:
        for surface_index, surface in enumerate(boundaries):
            yield surface_index, surface
    elif geom_type == "Solid":
        for shell in boundaries:
            for surface_index, surface in enumerate(shell):
                yield surface_index, surface


def _surface_vertex_indices(surface: list) -> list[int]:
    return [
        vertex_index
        for ring in surface or []
        for vertex_index in ring or []
        if isinstance(vertex_index, int)
    ]


def _geometry_vertex_indices(geometry: dict) -> list[int]:
    indices = []
    for _, surface in _iter_surfaces(geometry):
        indices.extend(_surface_vertex_indices(surface))
    return indices


def _semantic_info(geometry: dict, surface_index: int) -> dict:
    semantics = geometry.get("semantics", {}) or {}
    surfaces = semantics.get("surfaces", []) or []
    values = semantics.get("values", []) or []

    if geometry.get("type") == "Solid" and values and isinstance(values[0], list):
        values = values[0]

    sem_idx = values[surface_index] if surface_index < len(values) else None
    if isinstance(sem_idx, int) and 0 <= sem_idx < len(surfaces):
        surface = dict(surfaces[sem_idx])
        base_type = surface.get("type", "Unknown")
        qualifiers = []
        if surface.get("is_balkon"):
            qualifiers.append("balkon")
        if surface.get("is_overhang"):
            qualifiers.append("overhang")
        if surface.get("is_reveal"):
            qualifiers.append("reveal")
        semantic_type = f"{base_type}:{':'.join(qualifiers)}" if qualifiers else base_type
        return {"semantic_type": semantic_type, "semantic_index": sem_idx}

    return {"semantic_type": "Unknown", "semantic_index": None}


def _category(semantic_type: str) -> str:
    semantic_type = str(semantic_type).lower()
    if "balkon" in semantic_type or "balcony" in semantic_type:
        return "balcony"
    if "reveal" in semantic_type:
        return "reveal"
    if "roof" in semantic_type:
        return "roof"
    if "wall" in semantic_type:
        return "wall"
    return "other"


def _color(semantic_type: str) -> list[float]:
    return SEMANTIC_COLORS.get(semantic_type, CATEGORY_COLORS.get(_category(semantic_type), SEMANTIC_COLORS["Unknown"]))


def _orient_triangles_outward(vertices: list[list[float]], triangles: list[list[int]], object_center: np.ndarray):
    pts = np.asarray(vertices, dtype=float)
    centered = pts - pts.mean(axis=0)
    normal = np.zeros(3, dtype=float)
    for i0, i1, i2 in triangles:
        normal += np.cross(centered[i1] - centered[i0], centered[i2] - centered[i0])
    if np.linalg.norm(normal) < 1e-12:
        return triangles
    if np.dot(normal, pts.mean(axis=0) - object_center) < 0:
        return [[i0, i2, i1] for i0, i1, i2 in triangles]
    return triangles


def _clean_ring_indices(ring: list) -> list[int]:
    cleaned = []
    for vertex_index in ring or []:
        if not isinstance(vertex_index, int):
            continue
        if cleaned and cleaned[-1] == vertex_index:
            continue
        cleaned.append(vertex_index)
    if len(cleaned) > 1 and cleaned[0] == cleaned[-1]:
        cleaned.pop()
    return cleaned


def _surface_rings_xyz(surface: list, vertices_world: np.ndarray) -> list[np.ndarray]:
    rings = []
    for ring in surface or []:
        cleaned = _clean_ring_indices(ring)
        if len(cleaned) >= 3:
            rings.append(np.asarray([vertices_world[vertex_index] for vertex_index in cleaned], dtype=float))
    return rings


def _ring_normal(ring: np.ndarray) -> np.ndarray:
    centered = ring - ring.mean(axis=0)
    normal = np.zeros(3, dtype=float)
    for index, point in enumerate(centered):
        next_point = centered[(index + 1) % len(centered)]
        normal += np.cross(point, next_point)
    norm = np.linalg.norm(normal)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 1.0])
    return normal / norm


def _ring_projection_axes(ring: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    origin = ring.mean(axis=0)
    normal = _ring_normal(ring)

    edge_vectors = [
        ring[(index + 1) % len(ring)] - ring[index]
        for index in range(len(ring))
    ]
    axis_u = max(edge_vectors, key=lambda edge: np.linalg.norm(edge))
    axis_u = axis_u - np.dot(axis_u, normal) * normal
    if np.linalg.norm(axis_u) < 1e-12:
        centered = ring - origin
        _, eigvecs = np.linalg.eigh(np.cov(centered.T))
        axis_u = eigvecs[:, -1] - np.dot(eigvecs[:, -1], normal) * normal
    if np.linalg.norm(axis_u) < 1e-12:
        fallback = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(fallback, normal)) > 0.95:
            fallback = np.array([0.0, 1.0, 0.0])
        axis_u = fallback - np.dot(fallback, normal) * normal
    axis_u = axis_u / np.linalg.norm(axis_u)
    axis_v = np.cross(normal, axis_u)
    axis_v = axis_v / np.linalg.norm(axis_v)
    return origin, axis_u, axis_v


def _project_rings_2d(
    rings: list[np.ndarray],
    origin: np.ndarray,
    axis_u: np.ndarray,
    axis_v: np.ndarray,
) -> list[np.ndarray]:
    return [
        np.asarray([
            [np.dot(point - origin, axis_u), np.dot(point - origin, axis_v)]
            for point in ring
        ], dtype=np.float64)
        for ring in rings
    ]


def _signed_area_2d(ring: np.ndarray) -> float:
    x = ring[:, 0]
    y = ring[:, 1]
    return float(0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _orient_projected_rings(projected_rings: list[np.ndarray]) -> list[np.ndarray]:
    oriented = []
    for index, ring in enumerate(projected_rings):
        area = _signed_area_2d(ring)
        if index == 0 and area < 0:
            ring = ring[::-1]
        elif index > 0 and area > 0:
            ring = ring[::-1]
        oriented.append(ring)
    return oriented


def _rebuild_vertices_from_2d(
    projected_rings: list[np.ndarray],
    origin: np.ndarray,
    axis_u: np.ndarray,
    axis_v: np.ndarray,
) -> np.ndarray:
    return np.vstack([
        np.asarray([origin + point[0] * axis_u + point[1] * axis_v for point in ring], dtype=float)
        for ring in projected_rings
    ])


def _fallback_triangulate_with_shapely(
    projected_rings: list[np.ndarray],
    origin: np.ndarray,
    axis_u: np.ndarray,
    axis_v: np.ndarray,
) -> tuple[list[list[float]], list[list[int]]]:
    if Polygon is None or triangulate is None:
        return [], []

    polygon = Polygon(projected_rings[0].tolist(), [ring.tolist() for ring in projected_rings[1:]])
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty:
        return [], []

    vertices = []
    triangles = []
    vertex_index = {}

    def add_point(point):
        key = (round(float(point[0]), 9), round(float(point[1]), 9))
        if key not in vertex_index:
            vertex_index[key] = len(vertices)
            vertices.append((origin + key[0] * axis_u + key[1] * axis_v).tolist())
        return vertex_index[key]

    for triangle in triangulate(polygon):
        if triangle.intersection(polygon).area < triangle.area - 1e-8:
            continue
        coords = list(triangle.exterior.coords)[:-1]
        if len(coords) == 3:
            triangles.append([add_point(point) for point in coords])
    return vertices, triangles


def _triangulate_surface_rings(rings: list[np.ndarray]) -> tuple[list[list[float]], list[list[int]]]:
    if not rings:
        return [], []

    origin, axis_u, axis_v = _ring_projection_axes(rings[0])
    projected_rings = _orient_projected_rings(_project_rings_2d(rings, origin, axis_u, axis_v))
    mesh_vertices = _rebuild_vertices_from_2d(projected_rings, origin, axis_u, axis_v)

    if earcut is not None:
        vertices_2d = np.vstack(projected_rings).astype(np.float64)
        ring_ends = np.cumsum([len(ring) for ring in projected_rings]).astype(np.uint32)
        triangles = earcut.triangulate_float64(vertices_2d, ring_ends).reshape((-1, 3)).tolist()
        if triangles:
            return mesh_vertices.tolist(), triangles

    return _fallback_triangulate_with_shapely(projected_rings, origin, axis_u, axis_v)


def _surface_mesh(surface: list, vertices_world: np.ndarray, object_center: np.ndarray, color: list[float]):
    if not surface or not surface[0] or len(surface[0]) < 3:
        return None, []

    rings = _surface_rings_xyz(surface, vertices_world)
    mesh_vertices, triangles = _triangulate_surface_rings(rings)
    if not mesh_vertices or not triangles:
        return None, rings
    triangles = _orient_triangles_outward(mesh_vertices, triangles, object_center)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh_vertices, dtype=float))
    mesh.triangles = o3d.utility.Vector3iVector(np.asarray(triangles, dtype=int))
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return mesh, rings


def _normalize_input(cityjson_input):
    if isinstance(cityjson_input, (str, Path)):
        path = Path(cityjson_input)
        return {path: load_json_file(path)}, True
    if isinstance(cityjson_input, dict) and "CityObjects" in cityjson_input:
        return {Path("<in-memory-cityjson>"): cityjson_input}, True
    if isinstance(cityjson_input, dict):
        return {Path(path): data for path, data in cityjson_input.items()}, False
    raise TypeError("Expected a CityJSON dict, a path, or a {path: cityjson} mapping")


def build_lod3_building_dictionaries(cityjson_input):
    cityjson_by_path, return_single = _normalize_input(cityjson_input)
    buildings = []

    for source_path, cityjson in cityjson_by_path.items():
        vertices_world = _vertices_world(cityjson)

        for object_id, city_object in cityjson.get("CityObjects", {}).items():
            if "building" not in str(city_object.get("type", "")).lower():
                continue

            lod3_geometries = [
                geometry
                for geometry in city_object.get("geometry", [])
                if str(geometry.get("lod")) == "3"
            ]
            if not lod3_geometries:
                continue

            object_vertex_indices = sorted(set(
                vertex_index
                for geometry in lod3_geometries
                for vertex_index in _geometry_vertex_indices(geometry)
            ))
            object_center = vertices_world[object_vertex_indices].mean(axis=0)

            meshes = {key: [] for key in ["roof", "wall", "reveal", "balcony", "other"]}
            for semantic_type in SEMANTIC_COLORS:
                meshes.setdefault(semantic_key(semantic_type), [])

            parts = []
            semantic_labels = {}
            surfaces = {key: [] for key in ["roof", "wall", "reveal", "balcony", "other"]}
            for semantic_type in SEMANTIC_COLORS:
                surfaces.setdefault(semantic_key(semantic_type), [])

            for geometry in lod3_geometries:
                for surface_index, surface in _iter_surfaces(geometry):
                    info = _semantic_info(geometry, surface_index)
                    sem_type = info["semantic_type"]
                    sem_key = semantic_key(sem_type)
                    mesh, rings = _surface_mesh(surface, vertices_world, object_center, _color(sem_type))
                    if mesh is None:
                        continue

                    surface_record = {
                        "mesh": mesh,
                        "rings": [ring.tolist() for ring in rings],
                        "semantic_type": sem_type,
                        "semantic_key": sem_key,
                        "semantic_index": info["semantic_index"],
                        "surface_index": surface_index,
                    }
                    parts.append(mesh)
                    meshes[_category(sem_type)].append(mesh)
                    meshes.setdefault(sem_key, []).append(mesh)
                    surfaces[_category(sem_type)].append(surface_record)
                    surfaces.setdefault(sem_key, []).append(surface_record)
                    semantic_labels[sem_key] = sem_type

            combined_mesh = o3d.geometry.TriangleMesh()
            for mesh in parts:
                combined_mesh += mesh
            if len(combined_mesh.triangles) > 0:
                combined_mesh.compute_vertex_normals()

            buildings.append({
                "id": object_id,
                "name": city_object.get("attributes", {}).get("name", object_id),
                "type": city_object.get("type"),
                "source_path": source_path,
                "lod": "3",
                "parts": parts,
                "meshes": meshes,
                "surfaces": surfaces,
                "semantic_labels": semantic_labels,
                "mesh": combined_mesh,
            })

    if return_single:
        if not buildings:
            raise ValueError("No LoD3 BuildingPart geometries found")
        return buildings[0]
    return buildings
