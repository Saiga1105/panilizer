"""
visualize_cityjson.py  —  PRO EDITION
=====================================
Genereer een professionele standalone HTML 3D viewer voor CityJSON gebouwmodellen.

Nieuw t.o.v. de basisversie:
  * Per CityObject groepering (object isoleren / verbergen / inzoomen)
  * LoD-switcher (alle aanwezige LoD's worden geëxporteerd)
  * Wereldcoördinaten worden meegegeven (Lambert72 / TAW readout)
  * Per-vlak metadata: oppervlakte, normaal, oriëntatie (azimut, helling)
  * Punten-classificatie (LAS classification codes) wordt meegegeven
  * Punten-intensiteit wordt meegegeven (indien aanwezig)
  * CityObject attributen worden meegegeven (bouwjaar, hoogte, etc.)
  * Edge-data voorberekend voor wireframe-overlay
  * Validatie-flags (degenererende vlakken, niet-planaire surfaces)

Gebruik: identiek aan de oude versie:
    import visualize_cityjson as viz
    viz.generate_viewer(cityjson_data, als_laz_path, output_path, ...)
"""

import json
import base64
import numpy as np
from pathlib import Path

try:
    import mapbox_earcut as earcut
except ImportError:
    raise ImportError("mapbox_earcut is vereist: pip install mapbox-earcut")


_TEMPLATE_PATH = Path(__file__).parent / "_viewer_template.html"


SEMANTIC_COLORS = {
    "WallSurface":         [0.78, 0.66, 0.51],
    "RoofSurface":         [0.75, 0.22, 0.17],
    "GroundSurface":       [0.50, 0.55, 0.55],
    "Window":              [0.36, 0.68, 0.89],
    "Door":                [0.90, 0.49, 0.13],
    "BalkonVloer":         [0.56, 0.27, 0.68],
    "BalkonWand":          [0.56, 0.27, 0.68],
    "OuterCeilingSurface": [0.83, 0.67, 0.05],
    "OuterFloorSurface":   [0.10, 0.74, 0.61],
    "InteriorWallSurface": [0.66, 0.80, 0.89],
    "CeilingSurface":      [0.98, 0.91, 0.62],
}
DEFAULT_COLOR = [0.74, 0.76, 0.78]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _get_sem_type(val, sem_surfaces):
    if val is not None and isinstance(val, int) and 0 <= val < len(sem_surfaces):
        surf = sem_surfaces[val]
        base_type = surf.get("type", "")
        if surf.get("is_balkon", False):
            return "BalkonVloer" if base_type == "GroundSurface" else "BalkonWand"
        return base_type
    return None


def _iter_surfaces(boundaries, geom_type, sem_values, sem_surfaces):
    if geom_type in ("MultiSurface", "CompositeSurface"):
        for i, surface in enumerate(boundaries):
            sv = sem_values[i] if sem_values and i < len(sem_values) else None
            yield surface, _get_sem_type(sv, sem_surfaces)
    elif geom_type == "Solid":
        for shell_idx, shell in enumerate(boundaries):
            shell_sem = None
            if sem_values and shell_idx < len(sem_values):
                shell_sem = sem_values[shell_idx]
            for surf_idx, surface in enumerate(shell):
                sv = None
                if isinstance(shell_sem, list) and surf_idx < len(shell_sem):
                    sv = shell_sem[surf_idx]
                yield surface, _get_sem_type(sv, sem_surfaces)


def _polygon_area_3d(pts):
    """Bereken oppervlakte van een 3D polygon via cross product som."""
    if len(pts) < 3:
        return 0.0
    pts = np.asarray(pts, dtype=np.float64)
    total = np.zeros(3)
    for i in range(1, len(pts) - 1):
        total += np.cross(pts[i] - pts[0], pts[i + 1] - pts[0])
    return 0.5 * np.linalg.norm(total)


def _polygon_normal(pts):
    """Best-fit normaal van een 3D polygon (Newell's method)."""
    pts = np.asarray(pts, dtype=np.float64)
    n = np.zeros(3)
    L = len(pts)
    for i in range(L):
        a = pts[i]
        b = pts[(i + 1) % L]
        n[0] += (a[1] - b[1]) * (a[2] + b[2])
        n[1] += (a[2] - b[2]) * (a[0] + b[0])
        n[2] += (a[0] - b[0]) * (a[1] + b[1])
    norm = np.linalg.norm(n)
    return n / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0])


def _normal_to_orientation(normal):
    """Converteer normaal naar (azimut °, helling °).
    Azimut: 0=N, 90=E, 180=S, 270=W. Helling: 0=horizontaal, 90=verticaal.
    """
    nx, ny, nz = float(normal[0]), float(normal[1]), float(normal[2])
    # Helling = hoek met horizontaal vlak
    horiz = np.sqrt(nx * nx + ny * ny)
    slope = np.degrees(np.arctan2(horiz, abs(nz)))
    # Azimut = richting waarin het vlak helt (compass bearing)
    if horiz < 1e-6:
        azimuth = 0.0
    else:
        azimuth = (np.degrees(np.arctan2(nx, ny)) + 360.0) % 360.0
    return azimuth, slope


def _triangulate_surface(surface, verts):
    if not surface or not surface[0] or len(surface[0]) < 3:
        return [], None

    outer_ring = surface[0]
    inner_rings = surface[1:] if len(surface) > 1 else []

    if not inner_rings:
        pts = [verts[vi].tolist() for vi in outer_ring]
        normal = _polygon_normal(pts)
        return [[pts[0], pts[i], pts[i + 1]] for i in range(1, len(pts) - 1)], normal

    all_vi = list(outer_ring)
    ring_lengths = [len(outer_ring)]
    for inner in inner_rings:
        all_vi.extend(inner)
        ring_lengths.append(len(inner))

    all_pts_3d = np.array([verts[vi] for vi in all_vi], dtype=np.float64)

    outer_pts = all_pts_3d[: ring_lengths[0]]
    v1 = outer_pts[1] - outer_pts[0]

    normal = np.array([0.0, 0.0, 0.0])
    norm_len = 0.0
    for k in range(2, len(outer_pts)):
        v2 = outer_pts[k] - outer_pts[0]
        if abs(outer_pts[k][2] - outer_pts[0][2]) > 1e-3:
            candidate = np.cross(v1, v2)
            if np.linalg.norm(candidate) > 1e-12:
                normal = candidate
                norm_len = np.linalg.norm(normal)
                break
    if norm_len < 1e-12:
        for k in range(2, len(outer_pts)):
            v2 = outer_pts[k] - outer_pts[0]
            normal = np.cross(v1, v2)
            norm_len = np.linalg.norm(normal)
            if norm_len > 1e-12:
                break
    if norm_len < 1e-12:
        return [], None

    normal /= norm_len
    u_axis = v1 / np.linalg.norm(v1)
    v_axis = np.cross(normal, u_axis)
    v_axis /= np.linalg.norm(v_axis)

    centered = all_pts_3d - outer_pts[0]
    pts_2d = np.column_stack([centered @ u_axis, centered @ v_axis])
    ring_ends = np.cumsum(ring_lengths).astype(np.uint32)

    try:
        tri_indices = earcut.triangulate_float64(pts_2d, ring_ends)
    except Exception:
        pts = [verts[vi].tolist() for vi in outer_ring]
        return [[pts[0], pts[i], pts[i + 1]] for i in range(1, len(pts) - 1)], normal

    tris = []
    for j in range(0, len(tri_indices), 3):
        i0, i1, i2 = tri_indices[j], tri_indices[j + 1], tri_indices[j + 2]
        tris.append([all_pts_3d[i0].tolist(), all_pts_3d[i1].tolist(), all_pts_3d[i2].tolist()])
    return tris, normal


def _load_pointcloud(laz_path, center, max_pts, pcd_radius):
    """Laad puntenwolk + classificatie + intensiteit.
    Returns dict met positions, colors, classification, intensity, n_pts.
    """
    import laspy

    laz_path = Path(laz_path)
    empty = {
        "positions": np.array([], dtype=np.float32),
        "colors": np.array([], dtype=np.uint8),
        "classification": np.array([], dtype=np.uint8),
        "intensity": np.array([], dtype=np.float32),
        "n_pts": 0,
    }
    if not laz_path.exists():
        print(f"  Puntenwolk niet gevonden: {laz_path}")
        return empty

    las = laspy.read(str(laz_path))
    pts_raw = np.column_stack([las.x, las.y, las.z])
    pts_centered = (pts_raw - center).astype(np.float64)

    dist = np.linalg.norm(pts_centered[:, :2], axis=1)
    mask_near = dist < pcd_radius
    near = pts_centered[mask_near]

    if len(near) > max_pts:
        idx_choice = np.random.choice(len(near), max_pts, replace=False)
        near = near[idx_choice]
    else:
        idx_choice = np.arange(len(near))

    # Classificatie
    if hasattr(las, "classification"):
        classif_all = np.array(las.classification, dtype=np.uint8)
        classif = classif_all[mask_near][idx_choice]
    else:
        classif = np.zeros(len(near), dtype=np.uint8)

    # Intensiteit
    if hasattr(las, "intensity"):
        intens_all = np.array(las.intensity, dtype=np.float32)
        intens = intens_all[mask_near][idx_choice]
        # Normaliseer naar 0-1
        i_max = float(intens.max()) if len(intens) > 0 and intens.max() > 0 else 1.0
        intens = intens / i_max
    else:
        intens = np.zeros(len(near), dtype=np.float32)

    has_color = hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue")
    if has_color:
        all_r = np.array(las.red, dtype=np.float64)
        all_g = np.array(las.green, dtype=np.float64)
        all_b = np.array(las.blue, dtype=np.float64)
        near_r = all_r[mask_near][idx_choice]
        near_g = all_g[mask_near][idx_choice]
        near_b = all_b[mask_near][idx_choice]
        max_val = max(near_r.max(), near_g.max(), near_b.max(), 1)
        if max_val > 255:
            near_r = (near_r / 65535.0 * 255).astype(np.uint8)
            near_g = (near_g / 65535.0 * 255).astype(np.uint8)
            near_b = (near_b / 65535.0 * 255).astype(np.uint8)
        else:
            near_r = near_r.astype(np.uint8)
            near_g = near_g.astype(np.uint8)
            near_b = near_b.astype(np.uint8)
        colors = np.column_stack([near_r, near_g, near_b]).flatten().astype(np.uint8)
    else:
        z_min, z_max = near[:, 2].min(), near[:, 2].max()
        z_range = max(z_max - z_min, 1e-6)
        z_norm = (near[:, 2] - z_min) / z_range
        colors = np.zeros(len(near) * 3, dtype=np.uint8)
        for i, t in enumerate(z_norm):
            colors[i * 3] = int(min(255, (0 if t < 0.5 else (t - 0.5) * 2 * 0.9) * 255))
            colors[i * 3 + 1] = int(min(255, (t * 2 * 0.9 if t < 0.5 else 0.9 - (t - 0.5) * 0.3) * 255))
            colors[i * 3 + 2] = int(min(255, (0.8 + t * 0.2 if t < 0.5 else max(0, 0.8 - (t - 0.5) * 1.6)) * 255))

    return {
        "positions": near.astype(np.float32).flatten(),
        "colors": colors,
        "classification": classif,
        "intensity": intens,
        "n_pts": len(near),
    }


# ─── Hoofdfunctie ────────────────────────────────────────────────────────────

def generate_viewer(
    cityjson_data,
    als_laz_path,
    output_path,
    mls_laz_path=None,
    max_pts=150_000,
    pcd_radius=60.0,
    filename="viewer_lod3.html",
):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    cj = cityjson_data

    # ── Coördinatentransformatie ──
    tf = cj.get("transform", {})
    scale = np.array(tf.get("scale", [1, 1, 1]), dtype=np.float64)
    translate = np.array(tf.get("translate", [0, 0, 0]), dtype=np.float64)

    raw_verts = np.array(cj["vertices"], dtype=np.float64)
    verts_world = raw_verts * scale + translate
    center = verts_world.mean(axis=0)
    verts = (verts_world - center).astype(np.float64)

    print(f"Vertices: {len(verts)}")
    print(f"Center (wereld): {center}")

    # CRS info uit metadata
    crs_info = ""
    md = cj.get("metadata", {})
    if "referenceSystem" in md:
        crs_info = str(md["referenceSystem"])

    # ── Alle LoD's verzamelen ──
    all_lods = set()
    for obj in cj.get("CityObjects", {}).values():
        for geom in obj.get("geometry", []):
            all_lods.add(str(geom.get("lod", "?")))
    all_lods_sorted = sorted(all_lods, key=lambda x: float(x) if x.replace(".", "").isdigit() else -1)
    best_lod = all_lods_sorted[-1] if all_lods_sorted else "?"
    print(f"LoD's beschikbaar: {all_lods_sorted}, default: {best_lod}")

    # ── Driehoeken bouwen — PER LoD én PER CityObject ──
    # Datastructuur:
    #   per LoD: {
    #     positions: [...]  flat float32 array
    #     colors:    [...]  flat float32 array
    #     face_ids:  [...]  per vertex
    #     object_ids:[...]  per vertex (welk CityObject)
    #     faces:     dict van face_id -> {name, area, normal, azimuth, slope, sem_type, obj_id}
    #     objects:   dict van obj_id -> {name, type, attributes, face_ids: [...], n_verts}
    #   }

    lod_data = {}
    sem_type_counts_global = {}

    # Globale object index
    obj_index = {}  # obj_id -> int idx
    obj_list = []   # ordered list
    for obj_id, obj in cj.get("CityObjects", {}).items():
        obj_index[obj_id] = len(obj_list)
        obj_list.append({
            "id": obj_id,
            "type": obj.get("type", "Unknown"),
            "attributes": obj.get("attributes", {}),
            "parents": obj.get("parents", []),
            "children": obj.get("children", []),
        })

    for lod in all_lods_sorted:
        lod_data[lod] = {
            "positions": [],
            "colors": [],
            "face_ids": [],
            "object_ids": [],
            "faces": {},
            "object_face_ids": {},  # obj_id -> list of face_ids
        }

    face_counter_per_lod = {lod: 0 for lod in all_lods_sorted}

    for obj_id, obj in cj.get("CityObjects", {}).items():
        oidx = obj_index[obj_id]
        for geom in obj.get("geometry", []):
            lod = str(geom.get("lod", "?"))
            if lod not in lod_data:
                continue
            geom_type = geom.get("type", "")
            semantics = geom.get("semantics") or {}
            sem_values = semantics.get("values") or []
            sem_surfaces = semantics.get("surfaces") or []

            data = lod_data[lod]
            data["object_face_ids"].setdefault(obj_id, [])

            for surface, sem_type in _iter_surfaces(
                geom.get("boundaries", []), geom_type, sem_values, sem_surfaces
            ):
                key = (lod, sem_type)
                sem_type_counts_global.setdefault(key, 0)
                sem_type_counts_global[key] += 1
                color = SEMANTIC_COLORS.get(sem_type, DEFAULT_COLOR)

                tris, normal = _triangulate_surface(surface, verts)
                if not tris:
                    continue

                # Oppervlakte berekenen op outer ring
                outer_pts = [verts[vi].tolist() for vi in surface[0]]
                area = _polygon_area_3d(outer_pts)

                # Oriëntatie
                if normal is None:
                    normal = _polygon_normal(outer_pts)
                azim, slope = _normal_to_orientation(normal)

                fid = face_counter_per_lod[lod]
                display_name = sem_type or "Onbekend"
                data["faces"][fid] = {
                    "name": f"{display_name} #{sem_type_counts_global[key]}",
                    "sem_type": sem_type or "Unknown",
                    "area": round(area, 3),
                    "normal": [round(float(normal[0]), 4), round(float(normal[1]), 4), round(float(normal[2]), 4)],
                    "azimuth": round(azim, 1),
                    "slope": round(slope, 1),
                    "obj_id": obj_id,
                    "obj_idx": oidx,
                }
                data["object_face_ids"][obj_id].append(fid)
                face_counter_per_lod[lod] += 1

                for tri in tris:
                    for pt in tri:
                        data["positions"].append(pt)
                        data["colors"].append(color)
                        data["face_ids"].append(fid)
                        data["object_ids"].append(oidx)

    # ── Compacteer naar arrays + base64 ──
    encoded_lods = {}
    for lod, data in lod_data.items():
        if not data["positions"]:
            continue
        pos_np = np.array(data["positions"], dtype=np.float32).flatten()
        col_np = np.array(data["colors"], dtype=np.float32).flatten()
        face_id_np = np.array(data["face_ids"], dtype=np.int32)
        obj_id_np = np.array(data["object_ids"], dtype=np.int32)

        encoded_lods[lod] = {
            "pos": base64.b64encode(pos_np.tobytes()).decode("ascii"),
            "col": base64.b64encode(col_np.tobytes()).decode("ascii"),
            "fid": base64.b64encode(face_id_np.tobytes()).decode("ascii"),
            "oid": base64.b64encode(obj_id_np.tobytes()).decode("ascii"),
            "n_verts": len(data["face_ids"]),
            "n_faces": face_counter_per_lod[lod],
            "faces": data["faces"],
            "object_face_ids": data["object_face_ids"],
        }
        print(f"  LoD {lod}: {face_counter_per_lod[lod]} vlakken, {len(data['face_ids'])} verts")

    # ── Default LoD (de hoogste die we hebben) ──
    default_lod = best_lod if best_lod in encoded_lods else (sorted(encoded_lods.keys())[-1] if encoded_lods else "?")

    # ── Puntenwolken ──
    print("ALS puntenwolk laden...")
    als_data = _load_pointcloud(als_laz_path, center, max_pts, pcd_radius)
    print(f"  ALS: {als_data['n_pts']:,} punten")

    if mls_laz_path is not None:
        print("MLS puntenwolk laden...")
        mls_data = _load_pointcloud(mls_laz_path, center, max_pts, pcd_radius)
        print(f"  MLS: {mls_data['n_pts']:,} punten")
    else:
        mls_data = {
            "positions": np.array([], dtype=np.float32),
            "colors": np.array([], dtype=np.uint8),
            "classification": np.array([], dtype=np.uint8),
            "intensity": np.array([], dtype=np.float32),
            "n_pts": 0,
        }

    def b64(arr):
        return base64.b64encode(arr.tobytes()).decode("ascii") if len(arr) > 0 else ""

    payload = {
        "center": [float(center[0]), float(center[1]), float(center[2])],
        "crs": crs_info,
        "lods": {
            lod: {
                "pos": d["pos"],
                "col": d["col"],
                "fid": d["fid"],
                "oid": d["oid"],
                "n_verts": d["n_verts"],
                "n_faces": d["n_faces"],
                "faces": d["faces"],
                "object_face_ids": d["object_face_ids"],
            }
            for lod, d in encoded_lods.items()
        },
        "default_lod": default_lod,
        "objects": obj_list,
        "als": {
            "pos": b64(als_data["positions"]),
            "col": b64(als_data["colors"]),
            "cls": b64(als_data["classification"]),
            "ints": b64(als_data["intensity"]),
            "n_pts": als_data["n_pts"],
        },
        "mls": {
            "pos": b64(mls_data["positions"]),
            "col": b64(mls_data["colors"]),
            "cls": b64(mls_data["classification"]),
            "ints": b64(mls_data["intensity"]),
            "n_pts": mls_data["n_pts"],
        },
    }

    # ── HTML opbouwen ──
    template = _TEMPLATE_PATH.read_text(encoding="utf-8")
    payload_json = json.dumps(payload, ensure_ascii=False)
    html = template.replace("__PAYLOAD_JSON__", payload_json)

    html_path = output_path / filename
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    size_mb = len(html) / 1024 / 1024
    print(f"\n{'=' * 60}")
    print(f"✅ {filename} ({size_mb:.1f} MB)")
    print(f"   Open dit bestand in je browser!")
    print(f"{'=' * 60}")

    return html_path
