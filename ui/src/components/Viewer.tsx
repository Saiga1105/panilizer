import { OrbitControls } from "@react-three/drei";
import { Canvas } from "@react-three/fiber";
import * as THREE from "three";
import type { BuildingPayload, Layers, PanelizationPayload, Ring3 } from "../types";
import {
  collectRingsFromBuilding,
  collectRingsFromPanels,
  lineGeometry,
  polygonGeometry,
  sceneGroundOffset,
} from "../lib/geometry";

type ViewerProps = {
  building?: BuildingPayload;
  panelization?: PanelizationPayload;
  layers: Layers;
};

const surfaceColors: Record<string, string> = {
  roof: "#b9473d",
  wall: "#c7ac86",
  reveal: "#8b735a",
  balcony: "#8b55a1",
  other: "#6f94bc",
};

export function Viewer({ building, panelization, layers }: ViewerProps) {
  const buildingRings = building ? collectRingsFromBuilding(building.parts) : [];
  const panelRings = panelization ? collectRingsFromPanels(panelization.parts) : [];
  const sceneOffset = sceneGroundOffset([...buildingRings, ...panelRings]);

  return (
    <Canvas camera={{ position: [0, -28, 18], fov: 45 }} dpr={[1, 1.5]}>
      <color attach="background" args={["#f5f7f9"]} />
      <ambientLight intensity={0.75} />
      <directionalLight position={[20, -30, 40]} intensity={1.2} />
      <group rotation={[-Math.PI / 2, 0, 0]}>
        {building?.parts.map((part) =>
          part.surfaces.map((surface, surfaceIndex) => {
            if (!layers[surface.category as keyof Layers]) return null;
            return (
              <SurfaceMesh
                key={`${part.building_id}-${surfaceIndex}`}
                rings={surface.rings}
                color={surfaceColors[surface.category] ?? surfaceColors.other}
                center={sceneOffset}
                opacity={0.48}
              />
            );
          }),
        )}
        {layers.panels &&
          panelization?.parts.map((part) =>
            part.walls.map((wall) =>
              wall.panels.map((panel) => {
                if (panel.is_specialized && !layers.specialized) return null;
                return panel.polygons_xyz.map((piece, pieceIndex) => (
                  <SurfaceMesh
                    key={`${part.building_id}-${wall.wall_id}-${panel.name}-${pieceIndex}`}
                    rings={piece}
                    color={panel.is_specialized ? "#f06f2f" : panel.is_unique ? "#e0bd3d" : "#39a96b"}
                    center={sceneOffset}
                    opacity={0.86}
                    outline
                  />
                ));
              }),
            ),
          )}
      </group>
      <gridHelper args={[40, 40, "#9aa5b1", "#d4dae2"]} />
      <OrbitControls makeDefault enableDamping />
    </Canvas>
  );
}

type SurfaceMeshProps = {
  rings: Ring3[];
  color: string;
  center: THREE.Vector3;
  opacity: number;
  outline?: boolean;
};

function SurfaceMesh({ rings, color, center, opacity, outline = false }: SurfaceMeshProps) {
  const exterior = rings[0];
  if (!exterior || exterior.length < 3) return null;
  const geometry = polygonGeometry(rings, center);
  const line = lineGeometry(exterior, center);

  return (
    <group>
      <mesh geometry={geometry}>
        <meshStandardMaterial color={color} side={THREE.DoubleSide} transparent opacity={opacity} />
      </mesh>
      {(outline || rings.length > 1) && (
        <primitive object={new THREE.LineLoop(line, new THREE.LineBasicMaterial({ color: "#1f2933", transparent: true, opacity: 0.55 }))} />
      )}
      {rings.slice(1).map((ring, index) => (
        <primitive
          key={index}
          object={new THREE.LineLoop(
            lineGeometry(ring, center),
            new THREE.LineBasicMaterial({ color: "#1f2933", transparent: true, opacity: 0.75 }),
          )}
        />
      ))}
    </group>
  );
}
