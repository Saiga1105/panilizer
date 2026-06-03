import json
from pathlib import Path

import pandas as pd


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame."""
    return pd.read_csv(path)


def save_csv(df: pd.DataFrame, path: str) -> None:
    """Save a DataFrame to CSV."""
    df.to_csv(path, index=False)


def preview_dataframe(df: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Return the first n rows of a DataFrame."""
    return df.head(n)


def load_json_file(path: Path) -> dict:
    """Load a single JSON file from disk."""
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def load_json_files(input_dir: Path, recursive: bool = True) -> list[dict]:
    """Load all JSON files from the input folder or its subfolders."""
    input_dir = Path(input_dir)
    pattern = '**/*.json' if recursive else '*.json'
    json_files = sorted(input_dir.glob(pattern))
    return [load_json_file(path) for path in json_files]


def load_buildings_from_input(input_dir: Path, lod: str | None = None, recursive: bool = True) -> list[dict]:
    """Create one building dictionary per JSON file entry with name and geometry."""
    input_dir = Path(input_dir)
    buildings = []
    pattern = '**/*.json' if recursive else '*.json'
    for path in sorted(input_dir.glob(pattern)):
        data = load_json_file(path)
        buildings.extend(_extract_buildings(data, lod))
    return buildings


def _extract_buildings(data: dict, lod: str | None = None) -> list[dict]:
    buildings = []
    if 'CityObjects' in data:
        for object_id, obj in data.get('CityObjects', {}).items():
            if str(obj.get('type', '')).lower().endswith('building'):
                buildings.extend(_parse_cityjson_building(object_id, obj, data, lod))
    elif isinstance(data, dict) and data.get('type', '').lower() == 'featurecollection':
        for feature in data.get('features', []):
            geometry_type = str(feature.get('geometry', {}).get('type', '')).lower()
            if geometry_type in {'multisurface', 'polygon', 'multipolygon', 'building', 'buildings'}:
                buildings.append(_parse_geojson_building(feature, lod))
    return [b for b in buildings if b.get('geometry', {}).get('surfaces')]


def _parse_cityjson_building(object_id: str, obj: dict, data: dict, lod: str | None = None) -> dict:
    vertices = data.get('vertices', [])
    surfaces = []
    geometries = obj.get('geometry', [])
    selected_geometries = []

    if lod is None:
        selected_geometries = geometries
    else:
        selected_geometries = [g for g in geometries if str(g.get('lod', '')).lower() == str(lod).lower()]
        if not selected_geometries and geometries:
            selected_geometries = [geometries[0]]

    for geometry in selected_geometries:
        raw_surfaces = geometry.get('boundaries', [])
        semantics = geometry.get('semantics', {})
        surface_defs = semantics.get('surfaces', []) if isinstance(semantics.get('surfaces', []), list) else []
        values = semantics.get('values', []) if isinstance(semantics.get('values', []), list) else []

        for index, raw_surface in enumerate(raw_surfaces):
            polygons = _boundaries_to_polygons(raw_surface, vertices)
            semantic_type = None
            if index < len(values) and 0 <= values[index] < len(surface_defs):
                semantic_type = surface_defs[values[index]].get('type')
            surfaces.append({
                'surface_id': index,
                'surface_type': semantic_type or '',
                'polygons': polygons,
            })

    return {
        'id': object_id,
        'name': obj.get('attributes', {}).get('name', object_id),
        'type': obj.get('type', 'Building'),
        'geometry': {
            'lod': lod or 'auto',
            'surfaces': surfaces,
        },
    }


def _parse_geojson_building(feature: dict, lod: str | None = None) -> dict:
    geom = feature.get('geometry', {})
    polygons = []
    if geom.get('type', '').lower() in {'multisurface', 'polygon', 'multipolygon'}:
        polygons = _geojson_to_polygons(geom)
    return {
        'id': feature.get('id') or feature.get('properties', {}).get('name', ''),
        'name': feature.get('properties', {}).get('name', feature.get('id', 'building')),
        'type': feature.get('properties', {}).get('type', 'Building'),
        'geometry': {
            'lod': lod or 'auto',
            'surfaces': [{'surface_id': 0, 'surface_type': 'other', 'polygons': polygons}],
        },
    }


def _boundaries_to_polygons(boundary, vertices):
    polygons = []
    if not boundary:
        return polygons
    if isinstance(boundary[0][0], (int, float)):
        boundary = [boundary]
    for shell in boundary:
        for ring in shell:
            if not ring:
                continue
            if isinstance(ring[0], int):
                polygons.append([tuple(vertices[i]) for i in ring])
            else:
                polygons.append([tuple(point) for point in ring])
    return polygons


def _geojson_to_polygons(geom):
    polygons = []
    coords = geom.get('coordinates', [])
    if geom.get('type', '').lower() == 'polygon':
        polygons.append([tuple(point) for point in coords[0]])
    elif geom.get('type', '').lower() == 'multipolygon':
        for poly in coords:
            polygons.append([tuple(point) for point in poly[0]])
    elif geom.get('type', '').lower() == 'multisurface':
        for surface in coords:
            polygons.append([tuple(point) for point in surface[0]])
    return polygons


def _surface_category(surface_type: str) -> str:
    surface_type = str(surface_type or '').lower()
    if 'roof' in surface_type:
        return 'roof'
    if 'wall' in surface_type:
        return 'wall'
    if 'door' in surface_type:
        return 'door'
    if 'window' in surface_type:
        return 'window'
    return 'other'
