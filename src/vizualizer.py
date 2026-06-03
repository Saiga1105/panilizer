import matplotlib.pyplot as plt


def show_building(building: dict, figsize: tuple[int, int] = (12, 10), alpha: float = 0.5) -> None:
    """Display a single building in a 2D top-down view."""
    surfaces = building.get('geometry', {}).get('surfaces', [])
    if not surfaces:
        print('No surfaces to display for this building.')
        return

    fig, ax = plt.subplots(figsize=figsize)
    palette = {
        'roof': '#b5651d',
        'wall': '#4b77be',
        'door': '#f4a261',
        'window': '#89c2d9',
        'other': '#a8a8a8',
    }

    for surface in surfaces:
        surface_type = surface.get('surface_type', 'other')
        color = palette.get(_surface_category(surface_type), palette['other'])
        for polygon in surface.get('polygons', []):
            if len(polygon) < 3:
                continue
            xs = [coord[0] for coord in polygon]
            ys = [coord[1] for coord in polygon]
            ax.fill(xs, ys, color=color, alpha=alpha, edgecolor='black', linewidth=0.5)

    title = f"Building: {building.get('name', 'Unnamed')} ({building.get('id', '')})"
    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':', alpha=0.4)
    plt.show()


def show_buildings(buildings: list[dict], figsize: tuple[int, int] = (12, 10), alpha: float = 0.4) -> None:
    """Display multiple buildings in a 2D top-down view."""
    if not buildings:
        print('No buildings to display.')
        return

    fig, ax = plt.subplots(figsize=figsize)
    palette = {
        'roof': '#b5651d',
        'wall': '#4b77be',
        'door': '#f4a261',
        'window': '#89c2d9',
        'other': '#a8a8a8',
    }

    for building in buildings:
        for surface in building.get('geometry', {}).get('surfaces', []):
            surface_type = surface.get('surface_type', 'other')
            color = palette.get(_surface_category(surface_type), palette['other'])
            for polygon in surface.get('polygons', []):
                if len(polygon) < 3:
                    continue
                xs = [coord[0] for coord in polygon]
                ys = [coord[1] for coord in polygon]
                ax.fill(xs, ys, color=color, alpha=alpha, edgecolor='black', linewidth=0.5)

    ax.set_title('Buildings surfaces: walls, roofs, doors, windows')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, linestyle=':', alpha=0.4)
    plt.show()


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
