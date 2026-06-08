import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const canvas = document.querySelector("#scene");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75));

const scene = new THREE.Scene();
scene.background = new THREE.Color("#f5f7f9");

const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 1000);
camera.position.set(0, -28, 18);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

scene.add(new THREE.AmbientLight("#ffffff", 0.75));
const sun = new THREE.DirectionalLight("#ffffff", 1.2);
sun.position.set(18, -30, 38);
scene.add(sun);
scene.add(new THREE.GridHelper(42, 42, "#9aa5b1", "#d4dae2"));

const root = new THREE.Group();
root.rotation.x = -Math.PI / 2;
scene.add(root);

const groups = {
  building: new THREE.Group(),
  panels: new THREE.Group(),
  roofs: new THREE.Group(),
  balconies: new THREE.Group(),
};
Object.values(groups).forEach((group) => root.add(group));

const colors = {
  roof: "#b9473d",
  wall: "#c7ac86",
  reveal: "#8b735a",
  balcony: "#8b55a1",
  other: "#6f94bc",
  panel: "#39a96b",
  unique: "#e0bd3d",
  specialized: "#f06f2f",
};

const [building, panelization] = await Promise.all([
  fetch("demo_building.json").then((response) => response.json()),
  fetch("demo_panels.json").then((response) => response.json()),
]);

const allRings = [
  ...building.parts.flatMap((part) => part.surfaces.flatMap((surface) => surface.rings)),
  ...panelization.parts.flatMap((part) =>
    part.walls.flatMap((wall) => wall.panels.flatMap((panel) => panel.polygons_xyz.flatMap((piece) => piece))),
  ),
];
const offset = sceneGroundOffset(allRings);

for (const part of building.parts) {
  for (const surface of part.surfaces) {
    const target =
      surface.category === "roof"
        ? groups.roofs
        : surface.category === "balcony"
          ? groups.balconies
          : groups.building;
    target.add(surfaceObject(surface.rings, colors[surface.category] ?? colors.other, 0.48, offset, false));
  }
}

for (const part of panelization.parts) {
  for (const wall of part.walls) {
    for (const panel of wall.panels) {
      for (const piece of panel.polygons_xyz) {
        groups.panels.add(
          surfaceObject(
            piece,
            panel.is_specialized ? colors.specialized : panel.is_unique ? colors.unique : colors.panel,
            0.88,
            offset,
            true,
            panel.is_specialized,
          ),
        );
      }
    }
  }
}

document.querySelector("#stat-walls").textContent = panelization.summary.n_walls.toLocaleString();
document.querySelector("#stat-panels").textContent = panelization.summary.total_panels.toLocaleString();
document.querySelector("#stat-specialized").textContent = panelization.summary.total_specialized_panels.toLocaleString();
document.querySelector("#stat-cost").textContent = `EUR ${Math.round(panelization.summary.cost_total).toLocaleString()}`;

wireToggle("#layer-building", groups.building);
wireToggle("#layer-panels", groups.panels);
wireToggle("#layer-roofs", groups.roofs);
wireToggle("#layer-balconies", groups.balconies);
document.querySelector("#layer-specialized").addEventListener("change", (event) => {
  groups.panels.traverse((object) => {
    if (object.userData.specialized) object.visible = event.target.checked;
  });
});

resize();
renderer.setAnimationLoop(() => {
  controls.update();
  renderer.render(scene, camera);
});
window.addEventListener("resize", resize);

function wireToggle(selector, group) {
  document.querySelector(selector).addEventListener("change", (event) => {
    group.visible = event.target.checked;
  });
}

function resize() {
  const { clientWidth, clientHeight } = canvas.parentElement;
  camera.aspect = clientWidth / clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(clientWidth, clientHeight, false);
}

function surfaceObject(rings, color, opacity, offset, outline, specialized = false) {
  const group = new THREE.Group();
  const geometry = polygonGeometry(rings, offset);
  const material = new THREE.MeshStandardMaterial({
    color,
    side: THREE.DoubleSide,
    transparent: true,
    opacity,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.userData.specialized = specialized;
  group.add(mesh);

  if (outline || rings.length > 1) {
    group.add(lineLoop(rings[0], offset, specialized));
  }
  for (const ring of rings.slice(1)) {
    group.add(lineLoop(ring, offset, specialized));
  }
  return group;
}

function lineLoop(ring, offset, specialized) {
  const points = [...ring, ring[0]].map((point) => vector(point).sub(offset));
  const line = new THREE.LineLoop(
    new THREE.BufferGeometry().setFromPoints(points),
    new THREE.LineBasicMaterial({ color: "#1f2933", transparent: true, opacity: 0.62 }),
  );
  line.userData.specialized = specialized;
  return line;
}

function polygonGeometry(rings, offset) {
  const exterior = rings[0];
  const points3 = rings.flatMap((ring) => ring.map(vector));
  const vertices = [];
  for (const point of points3) {
    const shifted = point.clone().sub(offset);
    vertices.push(shifted.x, shifted.y, shifted.z);
  }

  const { origin, axisU, axisV } = polygonBasis(exterior);
  const points2 = points3.map((point) => {
    const relative = point.clone().sub(origin);
    return new THREE.Vector2(relative.dot(axisU), relative.dot(axisV));
  });

  const contour = points2.slice(0, exterior.length);
  const holes = [];
  let cursor = exterior.length;
  for (const ring of rings.slice(1)) {
    holes.push(points2.slice(cursor, cursor + ring.length));
    cursor += ring.length;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
  geometry.setIndex(THREE.ShapeUtils.triangulateShape(contour, holes).flat());
  geometry.computeVertexNormals();
  return geometry;
}

function polygonBasis(ring) {
  const origin = centerOfRings([ring]);
  const normal = new THREE.Vector3();
  const points = ring.map(vector);
  points.forEach((point, index) => {
    const next = points[(index + 1) % points.length];
    normal.x += (point.y - next.y) * (point.z + next.z);
    normal.y += (point.z - next.z) * (point.x + next.x);
    normal.z += (point.x - next.x) * (point.y + next.y);
  });
  normal.normalize();

  let axisU = new THREE.Vector3(1, 0, 0);
  let longest = 0;
  points.forEach((point, index) => {
    const edge = points[(index + 1) % points.length].clone().sub(point);
    if (edge.length() > longest) {
      longest = edge.length();
      axisU = edge;
    }
  });
  axisU.addScaledVector(normal, -axisU.dot(normal)).normalize();
  return { origin, axisU, axisV: new THREE.Vector3().crossVectors(normal, axisU).normalize() };
}

function sceneGroundOffset(rings) {
  const center = centerOfRings(rings);
  let minZ = Number.POSITIVE_INFINITY;
  for (const ring of rings) {
    for (const point of ring) minZ = Math.min(minZ, point[2]);
  }
  return new THREE.Vector3(center.x, center.y, Number.isFinite(minZ) ? minZ : center.z);
}

function centerOfRings(rings) {
  const center = new THREE.Vector3();
  let count = 0;
  for (const ring of rings) {
    for (const point of ring) {
      center.add(vector(point));
      count += 1;
    }
  }
  return count ? center.divideScalar(count) : center;
}

function vector(point) {
  return new THREE.Vector3(point[0], point[1], point[2]);
}
