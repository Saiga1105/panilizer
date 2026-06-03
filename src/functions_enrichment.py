import json
from PIL import Image, ImageDraw, ImageFont
import open3d as o3d
import numpy as np
import os
import torch
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
import math
from collections import defaultdict
from shapely.geometry import Polygon
from itertools import combinations
from collections import Counter
import copy
import laspy
from collections import defaultdict


import geomapi
import geomapi.utils as ut
from geomapi.utils import geometryutils as gmu


# Afbeeldingen en data inladen
def load_image(image_path):
    return Image.open(image_path)

def load_json(json_path):
    with open(json_path) as f:
        return json.load(f)
    
def load_lidar_data(las_path):
    las = laspy.read(las_path)
    return gmu.las_to_pcd(las)

def load_mesh(mesh_path):
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    return mesh

# Cameraparameters verwerken
def generate_cartesian_transforms(camJson, img_size_lookup):
    """
    Genereert een lijst van cartesische transformaties voor zowel de camera-assen als de beeldcentrumrichting.
    
    Parameters:
        camJson (dict): JSON met panoramagegevens.
    
    Returns:
        List[dict]: Lijst met beide cartesische transformaties per panorama.
    """
    pano_transforms = []

    for imgId, properties in camJson.items():
        # camera-oriëntatie
        R_camera = R.from_euler('z', -properties["Heading"] - properties["Yaw"], degrees=True).as_matrix()
        # R_camera = R_heading  # Aangezien pitch en roll nul zijn, is enkel heading+yaw van belang
        # R_heading = R.from_euler('z', -properties["Heading"] - properties["Yaw"], degrees=True).as_matrix()
        # R_pitch = R.from_euler('x', 0, degrees=True).as_matrix()
        # R_roll = R.from_euler('y', 0, degrees=True).as_matrix()
        # R_camera = R_heading @ R_roll @ R_pitch

        # Beeldcentrum-oriëntatie (alleen yaw)
        R_image = R.from_euler('z', -properties["Yaw"], degrees=True).as_matrix()

        # Translatie
        translation = [properties["X"], properties["Y"], properties["Z"]]

        # Transformatie-matrices
        T_camera = gmu.get_cartesian_transform(translation, R_camera)
        T_image = gmu.get_cartesian_transform(translation, R_image)

        # Structuur opslaan
        width, height = img_size_lookup.get(imgId, (None, None))
        pano_data = {
            "id": imgId,
            "cartesianTransform_camera": T_camera,
            "cartesianTransform_image": T_image,
            "coordinaatSysteem": properties["Coordinaatsysteem"],
            "hoogteSysteem": properties["Hoogtesysteem"],
            "cameraHoogte": properties["Camerahoogte (m)"],
            "imageWidth": width,
            "imageHeight": height,
            "yawDeg": properties["Yaw"]
        }

        pano_transforms.append(pano_data)

    return pano_transforms

def visualize_camera_axes(camera_data, mesh=None, point_cloud=None, axis_size=3.0):
    """
    Visualiseer camera-assen:
    - Cameracoördinaten (met heading)
    - Beeldcentrumrichting (alleen yaw)

    Parameters:
        camera_data: Lijst van dicts met 'cartesianTransform_camera' en 'cartesianTransform_image'.
        mesh: Optionele mesh.
        point_cloud: Optionele point cloud.
        axis_size: Lengte van de assen.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    for camera in camera_data:
        T_cam = camera["cartesianTransform_camera"]
        T_img = camera["cartesianTransform_image"]

        # Lokale assen vanuit oorsprong in homogene coördinaten
        origin_h = np.array([0, 0, 0, 1])
        x_axis_h = np.array([axis_size, 0, 0, 1])
        y_axis_h = np.array([0, axis_size, 0, 1])
        z_axis_h = np.array([0, 0, axis_size, 1])

        # Camera-assen
        cam_origin = (T_cam @ origin_h)[:3]
        cam_x = (T_cam @ x_axis_h)[:3]
        cam_y = (T_cam @ y_axis_h)[:3]
        cam_z = (T_cam @ z_axis_h)[:3]

        cam_points = [cam_origin, cam_x, cam_y, cam_z]
        cam_lines = [[0, 1], [0, 2], [0, 3]]
        cam_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # R, G, B

        cam_lineset = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(cam_points),
            lines=o3d.utility.Vector2iVector(cam_lines)
        )
        cam_lineset.colors = o3d.utility.Vector3dVector(cam_colors)
        vis.add_geometry(cam_lineset)

        # Beeldcentrum-assen (lichtrood, lichtgroen, lichtblauw)
        img_origin = (T_img @ origin_h)[:3]
        img_x = (T_img @ x_axis_h)[:3]
        img_y = (T_img @ y_axis_h)[:3]
        img_z = (T_img @ z_axis_h)[:3]

        img_points = [img_origin, img_x, img_y, img_z]
        img_lines = [[0, 1], [0, 2], [0, 3]]
        img_colors = [[1, 0.6, 0.6], [0.5, 1, 0.5], [0.5, 0.5, 1]]

        img_lineset = o3d.geometry.LineSet(
            points=o3d.utility.Vector3dVector(img_points),
            lines=o3d.utility.Vector2iVector(img_lines)
        )
        img_lineset.colors = o3d.utility.Vector3dVector(img_colors)
        vis.add_geometry(img_lineset)

        # Camera-centrum bol
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.2)
        sphere.paint_uniform_color([0.7, 0.7, 0.7])
        sphere.translate(cam_origin)
        vis.add_geometry(sphere)

    # Extra geometrieën toevoegen
    if mesh:
        vis.add_geometry(mesh)
    if point_cloud:
        vis.add_geometry(point_cloud)

    vis.run()
    vis.destroy_window()

# Objectdetectie
#box_th=0.4, text_th=0.2 zijn default waarden die enkel werken indien er geen andere waarden worden doorgegeven
# def detect_objects(image_folder, model, processor, device, extension, box_th=0.4, text_th=0.2):
#     detection_results = {}

#     # Loop door alle bestanden in de opgegeven map en filter op bestandsextentie
#     for file_name in os.listdir(image_folder):
#         if file_name.endswith(extension):
#             img_id = os.path.splitext(file_name)[0]
#             pano_path = image_folder / file_name

#             if pano_path.exists():
#                 image = Image.open(pano_path)
#                 inputs = processor(images=image, text="window. door.", return_tensors="pt").to(device)
#                 with torch.no_grad():
#                     outputs = model(**inputs)
#                 results = processor.post_process_grounded_object_detection(
#                     outputs,
#                     inputs.input_ids,
#                     threshold=box_th,
#                     text_threshold=text_th,
#                     target_sizes=[image.size[::-1]]
#                 )
#                 detection_results[img_id] = results
#                 #print(f"Resultaten voor {img_id}: {results}")
#             else:
#                 print(f"Afbeelding niet gevonden: {pano_path}")
    
#     return detection_results

#nieuwe voor balcons
def detect_objects(image_folder, model, processor, device, extension, box_th=0.4, text_th=0.2, text="window. door. balcony"):
    detection_results = {}
 
    # Loop door alle bestanden in de opgegeven map en filter op bestandsextentie
    for file_name in os.listdir(image_folder):
        if file_name.endswith(extension):
            img_id = os.path.splitext(file_name)[0]
            pano_path = image_folder / file_name
 
            if pano_path.exists():
                image = Image.open(pano_path)
                inputs = processor(images=image, text=text, return_tensors="pt").to(device)
                with torch.no_grad():
                    outputs = model(**inputs)
                results = processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    threshold=box_th,
                    text_threshold=text_th,
                    target_sizes=[image.size[::-1]]
                )
                detection_results[img_id] = results
                #print(f"Resultaten voor {img_id}: {results}")
            else:
                print(f"Afbeelding niet gevonden: {pano_path}")
    
    return detection_results

def save_detectionboxes_images(detection_results, image_folder, path_image_detections, image_ext):
    path_image_detections.mkdir(parents=True, exist_ok=True)
    
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except IOError:
        font = ImageFont.load_default()

    for img_id, results in detection_results.items():
        pano_path = image_folder / f"{img_id}{image_ext}"
        
        if pano_path.exists():
            pano = Image.open(pano_path)
            image = pano.copy()
            draw = ImageDraw.Draw(image)
            
            for result in results:
                scores = result['scores'] if isinstance(result['scores'], list) else result['scores'].tolist()
                labels = result['labels'] if isinstance(result['labels'], list) else result['labels'].tolist()
                boxes = result['boxes'] if isinstance(result['boxes'], list) else result['boxes'].tolist()

                for score, label, box in zip(scores, labels, boxes):
                    box = [int(coord) for coord in box]
                    draw.rectangle(box, outline="red", width=5)

                    label_text = f"{label} {score:.2f}"
                    text_position = (box[0], box[1] - 30)

                    draw.text(text_position, label_text, font=font, fill="red")
            
            output_path = path_image_detections / f"{img_id}.jpg"
            image.save(output_path, quality=100, subsampling=0, optimize=True)
            print(f"Geannoteerde afbeelding opgeslagen: {output_path}")
        else:
            print(f"Afbeelding niet gevonden: {pano_path}")

# Rays maken (uit corners ipv boxes)
def convert_boxes_to_corners(detections):
    """
    Zet een dictionary met bounding boxes in (xmin, ymin, xmax, ymax) formaat om
    naar een lijst van hoekpunten per box: (linksboven, rechtsboven, linksonder, rechtsonder)

    :param detections: dictionary met detecties per image_id
    :return: dictionary met per image_id een lijst van detecties met hoekpunten
    """
    converted = {}

    for image_id, image_detections in detections.items():
        new_detections = []

        for detection in image_detections:
            boxes = detection['boxes']
            scores = detection['scores']
            labels = detection['labels']

            # Zet om naar lijst van hoeken per box
            box_corners = []
            for box in boxes:
                xmin, ymin, xmax, ymax = box.tolist()
                corners = [
                    (xmin, ymin),  # linksboven
                    (xmax, ymin),  # rechtsboven
                    (xmin, ymax),  # linksonder
                    (xmax, ymax),  # rechtsonder
                ]
                box_corners.append(corners)

            # Voeg nieuwe detectie toe
            new_detections.append({
                'scores': scores,
                'labels': labels,
                'corners': box_corners  # 'corners' ipv 'boxes'
            })

        converted[image_id] = new_detections

    return converted

#wordt aangeroepen in detections_to_rays
def corners_to_rays(image_width, image_height, corner_points, camera_transform):
    """
    Zet de opgegeven hoekpunten van objecten om naar rays in de wereldruimte.
    
    :param image_width: Breedte van het beeld
    :param image_height: Hoogte van het beeld
    :param corner_points: Lijst van 4 (x, y) hoeken per object
    :param camera_transform: 4x4 cartesische transformatie matrix
    :return: Lijst van rays (4x6) voor het object
    """
    fov_horizontal_rad = 2 * np.pi  # 360 graden
    fov_vertical_rad = np.pi        # 180 graden

    object_rays = []

    for u, v in corner_points:
        # Omzetten naar sferische coördinaten
        theta = (u / (image_width - 1)) * fov_horizontal_rad - np.pi
        phi = (1 - v / (image_height - 1)) * fov_vertical_rad - (np.pi / 2)

        # Sferisch naar cartesisch
        x = np.cos(phi) * np.sin(theta)
        y = np.cos(phi) * np.cos(theta)
        z = np.sin(phi)

        direction_local = np.array([x, y, z])
        direction_local /= np.linalg.norm(direction_local)

        # Omzetten naar wereldcoördinaten
        direction_world = camera_transform[:3, :3] @ direction_local
        origin_world = camera_transform[:3, 3]

        # Combineer oorsprong en richting tot één ray
        object_rays.append({
            '2d_point': (u, v),
            'ray': np.hstack((origin_world, direction_world))
        })

    return object_rays

def detections_to_rays(camera_data, detections):

    rays_per_image = {}

    for camera in camera_data:
        image_id = camera['id']
        if image_id not in detections:
            continue

        camera_transform = camera['cartesianTransform_image']
        image_width = camera['imageWidth']
        image_height = camera['imageHeight']

        image_detections = detections[image_id]
        rays_for_image = []

        for detection in image_detections:
            corners_list = detection['corners']  # Lijst van hoeken per object
            labels = detection['labels']
            scores = detection['scores']

            # Rays per object
            rays_per_object = []
            for corners in corners_list:
                object_rays = corners_to_rays(image_width, image_height, corners, camera_transform)
                rays_per_object.append(object_rays)

            rays_for_image.append({
                'rays_per_object': rays_per_object,
                'labels': labels,
                'scores': scores
            })

        rays_per_image[image_id] = rays_for_image

    return rays_per_image

def visualize_rays(rays_per_image, mesh=None, point_cloud=None, ray_length=15):

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Colormap voor verschillende camera's
    cmap = plt.cm.get_cmap("tab10", len(rays_per_image))

    for i, (image_id, data_list) in enumerate(rays_per_image.items()):
        color = list(map(float, cmap(i)[:3]))  # RGB kleur voor deze camera

        for data in data_list:
            rays_per_object = data['rays_per_object']
            labels = data.get('labels', [])

            for j, object_rays in enumerate(rays_per_object):
                object_label = labels[j] if j < len(labels) else "Unknown"
                # print(f"Visualizing rays for object {j+1} (Label: {object_label}) in Image ID {image_id}")

                for ray_entry in object_rays:
                    ray_vec = ray_entry['ray']
                    origin = ray_vec[:3]
                    direction = ray_vec[3:]
                    endpoint = origin + direction * ray_length

                    line = o3d.geometry.LineSet(
                        points=o3d.utility.Vector3dVector([origin, endpoint]),
                        lines=o3d.utility.Vector2iVector([[0, 1]])
                    )
                    line.colors = o3d.utility.Vector3dVector([color])
                    vis.add_geometry(line)

    if mesh:
        vis.add_geometry(mesh)

    if point_cloud:
        vis.add_geometry(point_cloud)

    vis.run()
    vis.destroy_window()

# 3D punten bepalen
def calculate_intersections(mesh, rays_dict):
    if not mesh.has_triangles():
        print("WAARSCHUWING: lege mesh meegegeven aan calculate_intersections — geen intersecties berekend.")
        return {}
    mesh_ray_tracer = o3d.t.geometry.RaycastingScene()
    mesh_id = mesh_ray_tracer.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))

    intersections_with_labels = {}

    for image_id, data_list in rays_dict.items():
        for entry in data_list:
            rays_per_object = entry['rays_per_object']
            labels = entry.get('labels', [])
            scores = entry.get('scores', [])

            ray_vectors = []
            ray_2d_points = []
            ray_counts_per_object = []

            # Verzamel alle rays + 2D-punten
            for object_rays in rays_per_object:
                count = 0
                for ray_entry in object_rays:
                    ray = ray_entry.get('ray')
                    point2d = ray_entry.get('2d_point')
                    if ray is not None and len(ray) == 6:
                        ray_vectors.append(ray)
                        ray_2d_points.append(point2d)
                        count += 1
                    else:
                        print(f"Skipping invalid ray in image {image_id}: {ray_entry}")
                ray_counts_per_object.append(count)

            if not ray_vectors:
                print(f"No valid rays found for image {image_id}")
                continue

            try:
                ray_array = np.array(ray_vectors, dtype=np.float32)
            except Exception as e:
                print(f"Failed converting rays to numpy for {image_id}: {e}")
                continue

            rays_o3d = o3d.core.Tensor(ray_array)
            results = mesh_ray_tracer.cast_rays(rays_o3d)

            t_hit = results['t_hit'].numpy()
            valid_hits = np.isfinite(t_hit)

            intersections = np.zeros((len(t_hit), 3))
            intersections[valid_hits] = ray_array[valid_hits, :3] + ray_array[valid_hits, 3:] * t_hit[valid_hits, None]
            
            print(f"Valid intersections for {image_id}: {np.sum(valid_hits)} / {len(t_hit)}")

            object_intersections = []
            object_labels = []
            object_scores = []
            index = 0

            for obj_idx, ray_count in enumerate(ray_counts_per_object):
                object_data = []
                for _ in range(ray_count):
                    intersection = intersections[index]
                    point2d = ray_2d_points[index]
                    index += 1

                    intersection_output = {
                        '2d_point': tuple(point2d),
                        'intersection': intersection.tolist() if np.all(np.isfinite(intersection)) else None
                    }
                    object_data.append(intersection_output)

                if object_data:
                    object_intersections.append(object_data)
                    label = labels[obj_idx] if obj_idx < len(labels) else "Unknown"
                    object_labels.append(label)
                    score = scores[obj_idx] if obj_idx < len(labels) else "Unknown"
                    object_scores.append(score)

            if image_id not in intersections_with_labels:
                intersections_with_labels[image_id] = []

            intersections_with_labels[image_id].append({
                'intersections': object_intersections,
                'labels': object_labels,
                'scores': object_scores
            })

    return intersections_with_labels

def remove_zero_pointsets(data):
    # (0,0,0) betekent dat geen intersectiepunt is gevonden en 4 intersectiepunten per objectdetectie zijn nodig
    filtered_data = {}

    for image_id, image_data in data.items():
        new_obj_data = []

        for intersection_group in image_data:
            new_intersections = []
            new_labels = []
            new_scores = []

            for intersections_set, label, score in zip(intersection_group['intersections'], intersection_group['labels'], intersection_group['scores']):
                # 3D punten extracten
                points_3d = [point['intersection'] for point in intersections_set if isinstance(point, dict)]
                
                # Controleren of er een punt [0, 0, 0] is --> deze weghalen
                if any(np.allclose(point, [0, 0, 0]) for point in points_3d):
                    continue  # Skip this set

                new_intersections.append(intersections_set)
                new_labels.append(label)
                new_scores.append(score)

            new_obj_data.append({
                'intersections': new_intersections,
                'labels': new_labels,
                'scores': new_scores
            })

        if new_obj_data:
            filtered_data[image_id] = new_obj_data

    return filtered_data

def are_points_coplanar(points, epsilon=1e-1):
    if len(points) < 4:
        return True  # Minder dan 4 punten zijn altijd coplanar

    points = np.array(points)
    p0, p1, p2 = points[:3]
    v1 = p1 - p0
    v2 = p2 - p0
    normal = np.cross(v1, v2)

    for i in range(3, len(points)):
        vi = points[i] - p0
        volume = abs(np.dot(normal, vi))
        if volume > epsilon:  # tolerantie
            return False
    return True

def filter_coplanar_detections(data):
    filtered_data = {}

    for image_id, image_data in data.items():
        new_obj_data = []

        for intersection_group in image_data:
            new_intersections = []
            new_labels = []
            new_scores = []

            for intersections_set, label, score in zip(intersection_group['intersections'], intersection_group['labels'], intersection_group['scores']):
                points_3d = [point['intersection'] for point in intersections_set]

                if are_points_coplanar(points_3d):
                    new_intersections.append(intersections_set)
                    new_labels.append(label)
                    new_scores.append(score)

            new_obj_data.append({
                'intersections': new_intersections,
                'labels': new_labels,
                'scores': new_scores
            })

        if new_obj_data:
            filtered_data[image_id] = new_obj_data

    return filtered_data

def save_2d_points_with_labels(detection_results, image_folder, path_image_detections, image_ext):
    path_image_detections.mkdir(parents=True, exist_ok=True)
    
    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()

    for img_id, results in detection_results.items():
        pano_path = image_folder / f"{img_id}{image_ext}"
        
        if pano_path.exists():
            pano = Image.open(pano_path)
            image = pano.copy()
            draw = ImageDraw.Draw(image)

            for result in results:
                intersections_list = result['intersections']
                scores = result['scores']
                labels = result['labels']

                for quad, label, score in zip(intersections_list, labels, scores):
                    if len(quad) == 4:
                        p0 = tuple(quad[0]['2d_point'])  # linksboven
                        p1 = tuple(quad[1]['2d_point'])  # rechtsboven
                        p2 = tuple(quad[3]['2d_point'])  # rechtsonder
                        p3 = tuple(quad[2]['2d_point'])  # linksonder

                        points = [p0, p1, p2, p3, p0]
                        draw.line(points, fill="red", width=4)

                        text_position = (p0[0], p0[1] - 25)
                        label_text = f"{label} {score:.2f}"
                        draw.text(text_position, label_text, font=font, fill="red")
                            
            output_path = path_image_detections / f"{img_id}.jpg"
            image.save(output_path, quality=100, subsampling=0, optimize=True)
            print(f"Geannoteerde afbeelding opgeslagen: {output_path}")
        else:
            print(f"Afbeelding niet gevonden: {pano_path}")

def visualize_3d_points(image_data, mesh=None, pointcloud=None):

    geometries = []

    for image_id, data_list in image_data.items():
        for entry in data_list:
            intersections_per_object = entry.get('intersections', [])
            labels = entry.get('labels', [])

            valid_points = []
            valid_colors = []

            for obj_idx, object_points in enumerate(intersections_per_object):
                label = labels[obj_idx] if obj_idx < len(labels) else "unknown"
                
                object_valid_points = []
                for point_entry in object_points:
                    intersection = point_entry.get('intersection')
                    if intersection is not None and all(np.isfinite(intersection)) and not np.allclose(intersection, [0.0, 0.0, 0.0]):
                        object_valid_points.append(intersection)

                if not object_valid_points:
                    continue

                # Assign color based on label
                if label == 'window':
                    color = [0, 0, 1]  # Blauw
                elif label == 'door':
                    color = [1, 0, 0]  # Rood
                else:
                    color = [0.5, 0.5, 0.5]  # Grijs indien ander label

                valid_points.extend(object_valid_points)
                valid_colors.extend([color] * len(object_valid_points))

            if valid_points:
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(np.array(valid_points))
                pcd.colors = o3d.utility.Vector3dVector(np.array(valid_colors))

                if mesh:
                    geometries.append(mesh)
                if pointcloud:
                    geometries.append(pointcloud)

                geometries.append(pcd)

            else:
                print(f"No valid 3D points found in {image_id}. Skipping.")

    if geometries:
        o3d.visualization.draw_geometries(geometries)
    else:
        print("Nothing to visualize.")

def convert_intersections_to_detection_corners(data):
    result = {}

    for image_id, image_data in data.items():
        transformed_data = []
        for item in image_data:
            labels = item['labels']
            scores = item['scores']
            corners = []

            for intersection_group in item['intersections']:
                group_corners = [intersection['2d_point'] for intersection in intersection_group]
                corners.append(group_corners)

            transformed_data.append({
                'labels': labels,
                'scores': scores,
                'corners': corners
            })

        result[image_id] = transformed_data

    return result

def convert_grouped_detections_to_visualization(format1_data):
    format2_data = {}

    for entry in format1_data:
        raamgroep_id = entry['raamgroep']
        label = entry['label']
        mean_corners = entry['gemiddelde_hoekpunten']

        intersection_entry = [
            {'intersection': point}
            for point in mean_corners
        ]

        format2_data[raamgroep_id] = [{
            'intersections': [intersection_entry],
            'labels': [label]
        }]

    return format2_data

def convert_visualization_to_points_labels(data):
    points = []
    labels = []

    for value in data.values():
        for item in value:
            intersections = item['intersections'][0]
            point_set = [i['intersection'] for i in intersections]
            points.append(point_set)
            labels.append(item['labels'][0])

    return {'3d_points': points, 'labels': labels}

def get_bounding_box(pts_2d):
    x_min, y_min = np.min(pts_2d, axis=0).astype(int)
    x_max, y_max = np.max(pts_2d, axis=0).astype(int)
    return x_min, y_min, x_max, y_max

def compute_homography(pts_2d, pts_3d, idx=None):
    centroid = np.mean(pts_3d, axis=0)
    translated_points = pts_3d - centroid

    # normaalvector
    v1 = translated_points[1] - translated_points[0]
    v2 = translated_points[2] - translated_points[0]
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)
    nx, ny, nz = normal

    # alligneren met YZ-vlak
    alpha = np.arctan2(nz, ny)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(alpha), -np.sin(alpha)],
        [0, np.sin(alpha),  np.cos(alpha)]
    ])

    # alligneren met y-as
    beta = np.arctan2(nx, ny)
    Rz = np.array([
        [np.cos(beta), -np.sin(beta), 0],
        [np.sin(beta),  np.cos(beta), 0],
        [0, 0, 1]
    ])

    R = Rz @ Rx
    rotated_pts_3d = (R @ translated_points.T).T

    # Check oriëntatie en flip indien nodig
    top_center = (rotated_pts_3d[0] + rotated_pts_3d[1]) / 2
    bottom_center = (rotated_pts_3d[2] + rotated_pts_3d[3]) / 2

    if top_center[2] < bottom_center[2]:
        rotated_pts_3d[:, 2] *= -1

    # Projecteer op XZ-vlak
    pts_3d_2d = rotated_pts_3d[:, [0, 2]]

    # Homografie berekenen
    H, _ = cv2.findHomography(pts_2d, pts_3d_2d)
    return H

def crop_and_save_image(pano_array, pts_2d, image_id, idx, cropped_dir):
    x_min, y_min, x_max, y_max = get_bounding_box(pts_2d)
    cropped_image = pano_array[y_min:y_max, x_min:x_max]
    save_path = cropped_dir / f"cropped_image_{image_id}_{idx}.jpg"
    Image.fromarray(cropped_image).save(save_path)
    return cropped_image

def apply_homography_and_save(pano_array, H, pts_2d, image_id, idx, transformed_bboxes, rectified_dir, margin=500):
    h, w = pano_array.shape[:2]

    pts_2d_h = np.hstack((pts_2d, np.ones((pts_2d.shape[0], 1)))).T
    transformed_pts = H @ pts_2d_h
    transformed_pts /= transformed_pts[2]

    x_min, y_min = np.min(transformed_pts[:2], axis=1)
    x_max, y_max = np.max(transformed_pts[:2], axis=1)

    x_min_margin = x_min - margin
    y_min_margin = y_min - margin
    x_max_margin = x_max + margin
    y_max_margin = y_max + margin

    offset_x = -x_min_margin if x_min_margin < 0 else 0
    offset_y = -y_min_margin if y_min_margin < 0 else 0

    new_width = int(x_max_margin + offset_x)
    new_height = int(y_max_margin + offset_y)

    T_offset = np.array([
        [1, 0, offset_x],
        [0, 1, offset_y],
        [0, 0, 1]
    ])
    H_offset = T_offset @ H

    warped = cv2.warpPerspective(pano_array, H_offset, (new_width, new_height))

    transformed_pts_offset = H_offset @ pts_2d_h
    transformed_pts_offset /= transformed_pts_offset[2]

    x_min_crop, y_min_crop = np.min(transformed_pts_offset[:2], axis=1)
    x_max_crop, y_max_crop = np.max(transformed_pts_offset[:2], axis=1)

    x_min_crop = max(0, int(x_min_crop - margin))
    y_min_crop = max(0, int(y_min_crop - margin))
    x_max_crop = min(new_width, int(x_max_crop + margin))
    y_max_crop = min(new_height, int(y_max_crop + margin))

    shown_image = warped[y_min_crop:y_max_crop, x_min_crop:x_max_crop]
    # Flip image
    shown_image = np.flipud(shown_image)
    shown_image = np.ascontiguousarray(shown_image).astype(np.uint8)
    
    image_height = shown_image.shape[0]

    # Originele bounding box op rectified image laten zien --> crop en flip toepassen
    # Crop and flip coordinates
    x_coords = transformed_pts_offset[0] - x_min_crop
    y_coords = transformed_pts_offset[1] - y_min_crop
    y_coords_flipped = image_height - y_coords  # Vertical flip

    flipped_points = np.stack([x_coords, y_coords_flipped], axis=1).astype(int)

    # Originele BBox visualiseren op rectified image
    # # Draw bounding box with flipped points
    # top_left = tuple(flipped_points[0])
    # top_right = tuple(flipped_points[1])
    # bottom_left = tuple(flipped_points[2])
    # bottom_right = tuple(flipped_points[3])

    # # Draw bounding box
    # cv2.line(shown_image, top_left, top_right, (255, 0, 0), 2)
    # cv2.line(shown_image, top_right, bottom_right, (255, 0, 0), 2)
    # cv2.line(shown_image, bottom_right, bottom_left, (255, 0, 0), 2)
    # cv2.line(shown_image, bottom_left, top_left, (255, 0, 0), 2)

    save_path = rectified_dir / f"{image_id}_{idx}.jpg"
    Image.fromarray(shown_image).save(save_path)


    transformed_bboxes[f"{image_id}_{idx}"] = {
        'transformed_pts': flipped_points.tolist(),
        'H_offset': H_offset.tolist(),
        'crop_offset': [x_min_crop, y_min_crop],
        'image_shape': shown_image.shape[:2]
    }

def calculate_homographies_and_crop(pano_array, image_id, idx, points_2d, points_3d, transformed_bboxes, rectified_dir, cropped_dir=None):
    homographies = {}
    if len(points_2d) != 4 or len(points_3d) != 4:
        print(f"Niet genoeg punten voor {image_id}_{idx}")
        return homographies

    pts_2d = np.array(points_2d, dtype=np.float32)
    pts_3d = np.array(points_3d, dtype=np.float32) * 100

    if cropped_dir is not None:
        crop_and_save_image(pano_array, pts_2d, image_id, idx, cropped_dir)

    H = compute_homography(pts_2d, pts_3d, idx)

    if H is not None:
        homographies[f"{image_id}_{idx}"] = H.tolist()
        apply_homography_and_save(pano_array, H, pts_2d, image_id, idx, transformed_bboxes, rectified_dir)

    return homographies

def homographies_imageProcess(folder_path, points_dict, rectified_dir, cropped_dir=None):
    if cropped_dir is not None:
        cropped_dir.mkdir(parents=True, exist_ok=True)
    rectified_dir.mkdir(parents=True, exist_ok=True)

    pano_folder = Path(folder_path)
    all_homographies = {}
    transformed_bboxes = {}

    for image_id, image_data in points_dict.items():
        pano_path = pano_folder / f"{image_id}.jpg"
        if not pano_path.exists():
            print(f"Afbeelding niet gevonden: {pano_path}")
            continue

        pano = Image.open(pano_path)
        pano_array = np.array(pano)

        try:
            intersections_sets = image_data[0]['intersections']
        except KeyError as e:
            print(f"Fout bij uitlezen van punten voor {image_id}: {e}")
            continue

        for idx, intersection_set in enumerate(intersections_sets):
            points_2d = [point['2d_point'] for point in intersection_set]
            points_3d = [point['intersection'] for point in intersection_set if point['intersection'] != [0.0, 0.0, 0.0]]

            if len(points_3d) < 4:
                print(f"Niet genoeg 3D punten voor {image_id}_{idx}")
                continue

            print(f"Verwerken: {image_id}_{idx}")
            homographies = calculate_homographies_and_crop(
                pano_array, image_id, idx, points_2d, points_3d, transformed_bboxes, rectified_dir, cropped_dir=cropped_dir
            )
            all_homographies.update(homographies)

    return transformed_bboxes

def filter_bounding_boxes(transformed_bboxes, detection_results2_boxes, offset=100):

    filtered_boxes = {}

    for image_id_idx, bbox_data in transformed_bboxes.items():
        transformed_pts = np.array(bbox_data['transformed_pts'])

        # Omzetten naar polygon of min-max box
        x_min, y_min = np.min(transformed_pts, axis=0)
        x_max, y_max = np.max(transformed_pts, axis=0)

        detection_data = detection_results2_boxes.get(image_id_idx, [{}])[0]
        new_bboxes = detection_data.get('boxes', [])
        new_labels = detection_data.get('labels', [])
        new_scores = detection_data.get('scores', [])

        filtered_boxes[image_id_idx] = []

        for new_box, label, score in zip(new_bboxes, new_labels, new_scores):
            if isinstance(new_box, torch.Tensor):
                new_box = new_box.tolist()
            if isinstance(score, torch.Tensor):
                score = score.item()

            if len(new_box) != 4:
                continue  # Sla ongeldige boxen over

            nb_xmin, nb_ymin, nb_xmax, nb_ymax = new_box

            # Check of de nieuwe box binnen de oude box valt (met marge)
            if (nb_xmin >= x_min - offset and nb_ymin >= y_min - offset and
                nb_xmax <= x_max + offset and nb_ymax <= y_max + offset):
                filtered_boxes[image_id_idx].append({
                    'box': new_box,
                    'label': label,
                    'score': score
                })

    return filtered_boxes

def map_rectified_point_to_original(rectified_point, H_offset, crop_offset, rectified_image_height):
    x_crop, y_crop = rectified_point
    x_offset, y_offset = crop_offset

    # Flip y back because the saved rectified image is flipped
    y_crop = rectified_image_height - y_crop

    x_global = x_crop + x_offset
    y_global = y_crop + y_offset

    point_h = np.array([x_global, y_global, 1.0])
    H_inv = np.linalg.inv(H_offset)
    orig_point_h = H_inv @ point_h
    orig_point_h /= orig_point_h[2]

    return (float(orig_point_h[0]), float(orig_point_h[1]))

def map_rectified_bboxes_to_original(transformed_bboxes):
    original_newbboxes = {}

    for image_id_idx, bbox_data in transformed_bboxes.items():
        if '_' in image_id_idx:
            image_id, idx = image_id_idx.rsplit('_', 1)
        else:
            print(f"Ongeldig ID-formaat: {image_id_idx}")
            continue

        H_offset = np.array(bbox_data['H_offset'])
        crop_offset = bbox_data['crop_offset']
        rectified_height = bbox_data.get('image_shape', [None])[0]

        for bbox_info in bbox_data['bounding_boxes']:
            bbox = bbox_info['box']
            label = bbox_info['label']
            score = bbox_info['score']

            xmin, ymin, xmax, ymax = bbox
            rectified_points = [
                (xmin, ymax),  # bottom-left
                (xmax, ymax),  # bottom-right
                (xmin, ymin),  # top-left
                (xmax, ymin)   # top-right
            ]

            original_newbbox = [
                map_rectified_point_to_original(point, H_offset, crop_offset, rectified_image_height=rectified_height)
                for point in rectified_points
            ]

            original_newbboxes.setdefault(image_id, {}).setdefault(idx, []).append({
                'box': original_newbbox,
                'label': label,
                'score': score
            })

    return original_newbboxes

def extract_boxes_and_labels(annotation_dict):
    boxes, labels, scores = [], [], []

    for idx in sorted(annotation_dict.keys(), key=int):
        for item in annotation_dict[idx]:
            corners = item["box"]
            label = item["label"]
            score = item["score"]
            # x_coords = [pt[0] for pt in corners]
            # y_coords = [pt[1] for pt in corners]

            boxes.append(corners)
            labels.append(label)
            scores.append(score)

    return boxes, labels, scores

def convert_annotations(raw_data):
    formatted_data = defaultdict(list)

    for image_id, annotations in raw_data.items():
        boxes, labels,scores = extract_boxes_and_labels(annotations)
        formatted_data[image_id].append({
            "labels": labels,
            "corners": boxes,
            "scores": scores
        })

    return dict(formatted_data)

def sort_points_clockwise(data):
    def angle_from_centroid(centroid, point):
        return math.atan2(point[1] - centroid[1], point[0] - centroid[0])

    for key in data:
        for item in data[key]:
            new_corners = []
            for group in item['corners']:
                # Bereken het centroid
                cx = sum(p[0] for p in group) / len(group)
                cy = sum(p[1] for p in group) / len(group)
                centroid = (cx, cy)
                # Sorteer de punten op hoek met centroid
                sorted_group = sorted(group, key=lambda pt: angle_from_centroid(centroid, pt))
                new_corners.append(sorted_group)
            item['corners'] = new_corners
    return data

def save_detectioncorners_images(detection_dict, image_folder, path_image_detections, image_ext=".jpg"):
    path_image_detections.mkdir(parents=True, exist_ok=True)
    detection_dict = sort_points_clockwise(detection_dict)
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except IOError:
        font = ImageFont.load_default()

    for img_id, detections_list in detection_dict.items():
        pano_path = image_folder / f"{img_id}{image_ext}"

        if pano_path.exists():
            pano = Image.open(pano_path)
            image = pano.copy()
            draw = ImageDraw.Draw(image)

            for detection_block in detections_list:
                labels = detection_block['labels']
                corners_list = detection_block['corners']
                scores = detection_block['scores']  # Assuming scores are included in the detection block

                for label, corners, score in zip(labels, corners_list, scores):
                    # Convert to integers
                    box = [(int(x), int(y)) for x, y in corners]

                    # Draw the bounding box (polygon or rectangle)
                    if len(box) == 4:
                        ordered_box = [box[0], box[1], box[2], box[3], box[0]]
                        draw.line(ordered_box, fill="red", width=6)
                    else:
                        draw.polygon(box, outline="red", width=6)

                    # Draw label and score near the first corner
                    label_position = box[0]
                    label_text = f"{label} {score:.2f}"  # Include score with 2 decimal places
                    draw.text((label_position[0], label_position[1] - 25), label_text, font=font, fill="red")

            output_path = path_image_detections / f"{img_id}_annotated.jpg"
            image.save(output_path, quality=100, subsampling=0, optimize=True)
            print(f"Annotated image saved: {output_path}")
        else:
            print(f"Image not found: {pano_path}")

def angle_between_vectors(v1, v2):
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot_product = np.clip(np.dot(unit_v1, unit_v2), -1.0, 1.0)
    angle_rad = np.arccos(dot_product)
    return np.degrees(angle_rad)

def is_almost_right_angle(points, tolerance_deg):
    pts = list(map(np.array, points))
    angles = [
        angle_between_vectors(pts[0] - pts[1], pts[2] - pts[1]),
        angle_between_vectors(pts[1] - pts[2], pts[3] - pts[2]),
        angle_between_vectors(pts[2] - pts[3], pts[0] - pts[3]),
        angle_between_vectors(pts[3] - pts[0], pts[1] - pts[0])
    ]
    return all(abs(angle - 90) <= tolerance_deg for angle in angles)

def filter_right_angle_detections(data, tolerance=3):
    for key in data:
        for item in data[key]:
            new_intersections = []
            new_labels = []
            for i, group in enumerate(item['intersections']):
                coords = [pt['intersection'] for pt in group]
                if is_almost_right_angle(coords, tolerance):
                    new_intersections.append(group)
                    new_labels.append(item['labels'][i])
            item['intersections'] = new_intersections
            item['labels'] = new_labels
    return data
def filter_raamgroepen_by_angle(raamgroepen, tolerance_deg = 3):
    return [
        groep for groep in raamgroepen 
        if is_almost_right_angle(groep['gemiddelde_hoekpunten'], tolerance_deg)
    ]

def distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def is_same_window(w1, w2, threshold, normal_dot_threshold=0.98):
    """Twee ramen zijn gelijk als alle hoekpunten binnen threshold liggen
    EN hun vlak-normalen voldoende parallel zijn (voorkomt groepering van
    ramen op verschillende gevels via transitieve union-find ketens)."""
    # Afstandscheck per hoekpunt
    if not all(distance(p1['intersection'], p2['intersection']) <= threshold
               for p1, p2 in zip(w1, w2)):
        return False
    # Normaalcheck: bereken normaal van elk raamquad en vergelijk
    import numpy as np
    def quad_normal(w):
        pts = [np.asarray(p['intersection'], dtype=float) for p in w]
        if len(pts) < 3:
            return None
        v1 = pts[1] - pts[0]
        for p in pts[2:]:
            v2 = p - pts[0]
            c = np.cross(v1, v2)
            if np.linalg.norm(c) > 1e-6:
                return c / np.linalg.norm(c)
        return None
    n1 = quad_normal(w1)
    n2 = quad_normal(w2)
    if n1 is None or n2 is None:
        return True  # kan normaal niet berekenen, val terug op afstandscheck
    return abs(np.dot(n1, n2)) >= normal_dot_threshold

def split_intersections_by_label(intersections, balkon_labels=None):
    """
    Splits een intersections-dict op in twee delen op basis van het label.
 
    Na de gedeelde detectie + raytracing + filtering pipeline roep je deze
    functie aan om ramen/deuren te scheiden van balkons, zodat elke groep
    daarna met eigen parameters (grouping threshold, sigma, hoektolerantie)
    verder verwerkt kan worden.
 
    Parameters
    ----------
    intersections : dict
        Output van filter_coplanar_detections (of remove_zero_pointsets).
        Structuur: {image_id: [{'labels': [...], 'points_3d': [...], ...}, ...]}
    balkon_labels : set of str, optional
        Labels die als balkon beschouwd worden.
        Default: {"balcony", "balkon"}
 
    Returns
    -------
    windows_doors : dict
        Intersecties met window/door labels.
    balkons : dict
        Intersecties met balkon labels.
    """
    if balkon_labels is None:
        balkon_labels = {"balcony", "balkon"}
 
    windows_doors = {}
    balkons = {}
 
    for img_id, detections in intersections.items():
        wd_list = []
        bk_list = []
 
        for det in detections:
            labels = det.get("labels", [])
            label = labels[0].lower().strip() if labels else ""
 
            if label in balkon_labels:
                bk_list.append(det)
            else:
                wd_list.append(det)
 
        if wd_list:
            windows_doors[img_id] = wd_list
        if bk_list:
            balkons[img_id] = bk_list
 
    n_wd = sum(len(v) for v in windows_doors.values())
    n_bk = sum(len(v) for v in balkons.values())
    print(f"[split_intersections_by_label] {n_wd} ramen/deuren, {n_bk} balkons")
    return windows_doors, balkons

def group_similar_windows(intersections_data, threshold=0.40):
    all_windows = []
    index_map = []

    for image_id, image_entries in intersections_data.items():
        for entry_idx, entry in enumerate(image_entries):
            for win_idx, window in enumerate(entry['intersections']):
                all_windows.append(window)
                index_map.append((image_id, entry_idx, win_idx))

    parent = list(range(len(all_windows)))

    def find(x):
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x, y):
        parent[find(x)] = find(y)

    for i in range(len(all_windows)):
        for j in range(i + 1, len(all_windows)):
            if is_same_window(all_windows[i], all_windows[j], threshold):
                union(i, j)

    groups = defaultdict(list)
    for i in range(len(all_windows)):
        root = find(i)
        image_id, entry_idx, win_idx = index_map[i]
        label = intersections_data[image_id][entry_idx]['labels'][win_idx]
        coords = [pt['intersection'] for pt in all_windows[i]]
        groups[root].append({
            'image_id': image_id,
            'coords': coords,
            'label': label
        })

    return list(groups.values())

# def mean_and_std_iteratief_filtered(grouped_intersections2, drempel=2.0):
#     grouped_intersections2_mean = []

#     for i, group in enumerate(grouped_intersections2):
#         werkset = group.copy()
#         if len(werkset) < 2:
#             continue  # minimaal 2 detecties nodig

#         labels = [d['label'] for d in werkset]
#         most_common_label, _ = Counter(labels).most_common(1)[0]
#         iteratie = 0

#         while True:
#             iteratie += 1
#             grouped_points = [[] for _ in range(4)]

#             for detection in werkset:
#                 for j in range(4):
#                     grouped_points[j].append(detection['coords'][j]['intersection'])

#             mean_per_corner = []
#             std_per_corner = []
#             for points in grouped_points:
#                 arr = np.array(points)
#                 mean_xy = np.mean(arr[:, :2], axis=0)
#                 mean_z = np.mean(arr[:, 2])
#                 std_xy = np.std(arr[:, :2], axis=0)
#                 std_z = np.std(arr[:, 2])
#                 planimetric_std = np.sqrt(std_xy[0]**2 + std_xy[1]**2)
#                 mean_per_corner.append([mean_xy[0], mean_xy[1], mean_z])
#                 std_per_corner.append([planimetric_std, std_z])

#             grootste_afwijking = -1
#             slechtste_idx = None

#             for idx, detection in enumerate(werkset):
#                 max_afwijking = 0

#                 for j in range(4):
#                     hoekpunt = np.array(detection['coords'][j]['intersection'])
#                     mean = np.array(mean_per_corner[j])
#                     std_xy, std_z = std_per_corner[j]

#                     dx, dy = hoekpunt[0] - mean[0], hoekpunt[1] - mean[1]
#                     dz = hoekpunt[2] - mean[2]
#                     planimetric_afstand = np.sqrt(dx**2 + dy**2)
#                     z_afstand = abs(dz)

#                     xy_sigma = planimetric_afstand / std_xy if std_xy > 0 else 0
#                     z_sigma = z_afstand / std_z if std_z > 0 else 0

#                     max_afwijking = max(max_afwijking, xy_sigma, z_sigma)

#                 if max_afwijking > grootste_afwijking:
#                     grootste_afwijking = max_afwijking
#                     slechtste_idx = idx

#             if grootste_afwijking > drempel and len(werkset) > 1:
#                 werkset.pop(slechtste_idx)
#             else:
#                 break

#         if not werkset:
#             continue

#         grouped_points_final = [[] for _ in range(4)]
#         for detection in werkset:
#             for j in range(4):
#                 grouped_points_final[j].append(detection['coords'][j]['intersection'])

#         mean_per_corner = []
#         std_per_corner = []
#         for points in grouped_points_final:
#             arr = np.array(points)
#             mean_xy = np.mean(arr[:, :2], axis=0)
#             mean_z = np.mean(arr[:, 2])
#             std_xy = np.std(arr[:, :2], axis=0)
#             std_z = np.std(arr[:, 2])
#             planimetric_std = np.sqrt(std_xy[0]**2 + std_xy[1]**2)
#             mean_per_corner.append([mean_xy[0], mean_xy[1], mean_z])
#             std_per_corner.append([planimetric_std, std_z])

#         std_xy_gem = np.mean([std[0] for std in std_per_corner])
#         std_z_gem = np.mean([std[1] for std in std_per_corner])
#         gemiddelde_stdev = [std_xy_gem, std_z_gem]

#         first_detection = werkset[0]
#         example_image_id = first_detection['image_id']
#         example_2d_points = [pt['2d_point'] for pt in first_detection['coords']]

#         group_dict = {
#             'raamgroep': i + 1,
#             'aantal_detecties': len(werkset),
#             'label': most_common_label,
#             'gemiddelde_hoekpunten': mean_per_corner,
#             'standaardafwijking_per_hoekpunt': std_per_corner,
#             'gemiddelde_standaardafwijking_xy_z': gemiddelde_stdev,
#             'voorbeeld_image_id': example_image_id,
#             'voorbeeld_2d_hoekpunten': example_2d_points
#         }

#         grouped_intersections2_mean.append(group_dict)

#     return grouped_intersections2_mean

import numpy as np
from collections import Counter

#toevoeging voor verschillende coord structuren probleem met chatgpt ignace
# def mean_and_std_iteratief_filtered(grouped_intersections2, drempel=2.5):
#     def _to_xyz(item):
#         """
#         Accepteert:
#           - dict met 'intersection' of '3d_point' of 'x','y','z'
#           - list/tuple/np.array van lengte >= 3
#         Geeft: np.ndarray shape (3,)
#         """
#         if isinstance(item, dict):
#             if 'intersection' in item:
#                 return np.asarray(item['intersection'], dtype=float)[:3]
#             if '3d_point' in item:
#                 return np.asarray(item['3d_point'], dtype=float)[:3]
#             if all(k in item for k in ('x', 'y', 'z')):
#                 return np.array([item['x'], item['y'], item['z']], dtype=float)
#             # fallback: pak eerste array-achtige waarde met lengte ≥ 3
#             for v in item.values():
#                 if isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 3:
#                     return np.asarray(v, dtype=float)[:3]
#             raise TypeError(f"Onbekende coords-structuur: {item}")
#         elif isinstance(item, (list, tuple, np.ndarray)):
#             arr = np.asarray(item, dtype=float)
#             if arr.ndim == 1 and arr.size >= 3:
#                 return arr[:3]
#             raise TypeError(f"Verwacht 1D xyz, kreeg shape {arr.shape}")
#         else:
#             raise TypeError(f"Type {type(item)} niet ondersteund voor coords")

#     grouped_intersections2_mean = []

#     for i, group in enumerate(grouped_intersections2):
#         werkset = list(group)  # shallow copy
#         if len(werkset) < 2:
#             continue

#         labels = [d.get('label') for d in werkset if 'label' in d]
#         most_common_label = Counter(labels).most_common(1)[0][0] if labels else None

#         while True:
#             grouped_points = [[] for _ in range(4)]
#             # ---- VERANDERD: gebruik _to_xyz in plaats van ['intersection'] ----
#             for detection in werkset:
#                 for j in range(4):
#                     grouped_points[j].append(_to_xyz(detection['coords'][j]))

#             mean_per_corner, std_per_corner = [], []
#             for points in grouped_points:
#                 arr = np.vstack(points)  # N x 3
#                 mean_xy = np.mean(arr[:, :2], axis=0)
#                 mean_z = np.mean(arr[:, 2])
#                 std_xy = np.std(arr[:, :2], axis=0)
#                 std_z = np.std(arr[:, 2])
#                 planimetric_std = float(np.hypot(std_xy[0], std_xy[1]))
#                 mean_per_corner.append([float(mean_xy[0]), float(mean_xy[1]), float(mean_z)])
#                 std_per_corner.append([planimetric_std, float(std_z)])

#             grootste_afwijking = -1.0
#             slechtste_idx = None

#             for idx, detection in enumerate(werkset):
#                 max_afwijking = 0.0
#                 for j in range(4):
#                     hoekpunt = _to_xyz(detection['coords'][j])
#                     mean = np.array(mean_per_corner[j])
#                     std_xy, std_z = std_per_corner[j]

#                     dx, dy, dz = hoekpunt - mean
#                     planimetric_afstand = float(np.hypot(dx, dy))
#                     z_afstand = float(abs(dz))

#                     xy_sigma = planimetric_afstand / std_xy if std_xy > 0 else 0.0
#                     z_sigma = z_afstand / std_z if std_z > 0 else 0.0

#                     if xy_sigma > max_afwijking: max_afwijking = xy_sigma
#                     if z_sigma > max_afwijking: max_afwijking = z_sigma

#                 if max_afwijking > grootste_afwijking:
#                     grootste_afwijking = max_afwijking
#                     slechtste_idx = idx

#             if grootste_afwijking > drempel and len(werkset) > 1:
#                 werkset.pop(slechtste_idx)
#             else:
#                 break

#         if not werkset:
#             continue

#         # Finale statistieken op de gefilterde set (zelfde wijziging: _to_xyz)
#         grouped_points_final = [[] for _ in range(4)]
#         for detection in werkset:
#             for j in range(4):
#                 grouped_points_final[j].append(_to_xyz(detection['coords'][j]))

#         mean_per_corner, std_per_corner = [], []
#         for points in grouped_points_final:
#             arr = np.vstack(points)
#             mean_xy = np.mean(arr[:, :2], axis=0)
#             mean_z = np.mean(arr[:, 2])
#             std_xy = np.std(arr[:, :2], axis=0)
#             std_z = np.std(arr[:, 2])
#             planimetric_std = float(np.hypot(std_xy[0], std_xy[1]))
#             mean_per_corner.append([float(mean_xy[0]), float(mean_xy[1]), float(mean_z)])
#             std_per_corner.append([planimetric_std, float(std_z)])

#         std_xy_gem = float(np.mean([std[0] for std in std_per_corner]))
#         std_z_gem = float(np.mean([std[1] for std in std_per_corner]))
#         gemiddelde_stdev = [std_xy_gem, std_z_gem]

#         first_detection = werkset[0]
#         example_image_id = first_detection.get('image_id')

#         # 2D-voorbeelden alleen toevoegen als die key bestaat
#         example_2d_points = []
#         for pt in first_detection.get('coords', []):
#             if isinstance(pt, dict) and '2d_point' in pt:
#                 example_2d_points.append(pt['2d_point'])

#         group_dict = {
#             'raamgroep': i + 1,
#             'aantal_detecties': len(werkset),
#             'label': most_common_label,
#             'gemiddelde_hoekpunten': mean_per_corner,
#             'standaardafwijking_per_hoekpunt': std_per_corner,
#             'gemiddelde_standaardafwijking_xy_z': gemiddelde_stdev,
#             'voorbeeld_image_id': example_image_id,
#             'voorbeeld_2d_hoekpunten': example_2d_points or None,
#         }

#         grouped_intersections2_mean.append(group_dict)

#     return grouped_intersections2_mean

#toevoeging voor verschillende coord structuren probleem met chatgpt ignace
def mean_and_std_iteratief_filtered(grouped_intersections2, drempel=2.5, include_single_detections=False):
    """
    Parameters:
        grouped_intersections2: lijst van groepen detecties
        drempel: sigma-drempel voor outlier filtering
        include_single_detections: indien True worden groepen met slechts 1 detectie
            ook meegenomen (zonder outlier-filtering, std = 0.0)
    """
    def _to_xyz(item):
        """
        Accepteert:
          - dict met 'intersection' of '3d_point' of 'x','y','z'
          - list/tuple/np.array van lengte >= 3
        Geeft: np.ndarray shape (3,)
        """
        if isinstance(item, dict):
            if 'intersection' in item:
                return np.asarray(item['intersection'], dtype=float)[:3]
            if '3d_point' in item:
                return np.asarray(item['3d_point'], dtype=float)[:3]
            if all(k in item for k in ('x', 'y', 'z')):
                return np.array([item['x'], item['y'], item['z']], dtype=float)
            # fallback: pak eerste array-achtige waarde met lengte ≥ 3
            for v in item.values():
                if isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 3:
                    return np.asarray(v, dtype=float)[:3]
            raise TypeError(f"Onbekende coords-structuur: {item}")
        elif isinstance(item, (list, tuple, np.ndarray)):
            arr = np.asarray(item, dtype=float)
            if arr.ndim == 1 and arr.size >= 3:
                return arr[:3]
            raise TypeError(f"Verwacht 1D xyz, kreeg shape {arr.shape}")
        else:
            raise TypeError(f"Type {type(item)} niet ondersteund voor coords")

    grouped_intersections2_mean = []

    for i, group in enumerate(grouped_intersections2):
        werkset = list(group)  # shallow copy
        if len(werkset) < 2:
            if not include_single_detections:
                continue
            # Enkele detectie: neem coördinaten direct over, std = 0.0
            detection = werkset[0]
            labels = [d.get('label') for d in werkset if 'label' in d]
            most_common_label = Counter(labels).most_common(1)[0][0] if labels else None
            mean_per_corner = [list(_to_xyz(detection['coords'][j])) for j in range(4)]
            std_per_corner = [[0.0, 0.0]] * 4

            example_2d_points = []
            for pt in detection.get('coords', []):
                if isinstance(pt, dict) and '2d_point' in pt:
                    example_2d_points.append(pt['2d_point'])

            grouped_intersections2_mean.append({
                'raamgroep': i + 1,
                'aantal_detecties': 1,
                'label': most_common_label,
                'gemiddelde_hoekpunten': mean_per_corner,
                'standaardafwijking_per_hoekpunt': std_per_corner,
                'gemiddelde_standaardafwijking_xy_z': [0.0, 0.0],
                'voorbeeld_image_id': detection.get('image_id'),
                'voorbeeld_2d_hoekpunten': example_2d_points or None,
            })
            continue

        labels = [d.get('label') for d in werkset if 'label' in d]
        most_common_label = Counter(labels).most_common(1)[0][0] if labels else None

        while True:
            grouped_points = [[] for _ in range(4)]
            # ---- VERANDERD: gebruik _to_xyz in plaats van ['intersection'] ----
            for detection in werkset:
                for j in range(4):
                    grouped_points[j].append(_to_xyz(detection['coords'][j]))

            mean_per_corner, std_per_corner = [], []
            for points in grouped_points:
                arr = np.vstack(points)  # N x 3
                mean_xy = np.mean(arr[:, :2], axis=0)
                mean_z = np.mean(arr[:, 2])
                std_xy = np.std(arr[:, :2], axis=0)
                std_z = np.std(arr[:, 2])
                planimetric_std = float(np.hypot(std_xy[0], std_xy[1]))
                mean_per_corner.append([float(mean_xy[0]), float(mean_xy[1]), float(mean_z)])
                std_per_corner.append([planimetric_std, float(std_z)])

            grootste_afwijking = -1.0
            slechtste_idx = None

            for idx, detection in enumerate(werkset):
                max_afwijking = 0.0
                for j in range(4):
                    hoekpunt = _to_xyz(detection['coords'][j])
                    mean = np.array(mean_per_corner[j])
                    std_xy, std_z = std_per_corner[j]

                    dx, dy, dz = hoekpunt - mean
                    planimetric_afstand = float(np.hypot(dx, dy))
                    z_afstand = float(abs(dz))

                    xy_sigma = planimetric_afstand / std_xy if std_xy > 0 else 0.0
                    z_sigma = z_afstand / std_z if std_z > 0 else 0.0

                    if xy_sigma > max_afwijking: max_afwijking = xy_sigma
                    if z_sigma > max_afwijking: max_afwijking = z_sigma

                if max_afwijking > grootste_afwijking:
                    grootste_afwijking = max_afwijking
                    slechtste_idx = idx

            if grootste_afwijking > drempel and len(werkset) > 1:
                werkset.pop(slechtste_idx)
            else:
                break

        if not werkset:
            continue

        # Finale statistieken op de gefilterde set (zelfde wijziging: _to_xyz)
        grouped_points_final = [[] for _ in range(4)]
        for detection in werkset:
            for j in range(4):
                grouped_points_final[j].append(_to_xyz(detection['coords'][j]))

        mean_per_corner, std_per_corner = [], []
        for points in grouped_points_final:
            arr = np.vstack(points)
            mean_xy = np.mean(arr[:, :2], axis=0)
            mean_z = np.mean(arr[:, 2])
            std_xy = np.std(arr[:, :2], axis=0)
            std_z = np.std(arr[:, 2])
            planimetric_std = float(np.hypot(std_xy[0], std_xy[1]))
            mean_per_corner.append([float(mean_xy[0]), float(mean_xy[1]), float(mean_z)])
            std_per_corner.append([planimetric_std, float(std_z)])

        std_xy_gem = float(np.mean([std[0] for std in std_per_corner]))
        std_z_gem = float(np.mean([std[1] for std in std_per_corner]))
        gemiddelde_stdev = [std_xy_gem, std_z_gem]

        first_detection = werkset[0]
        example_image_id = first_detection.get('image_id')

        # 2D-voorbeelden alleen toevoegen als die key bestaat
        example_2d_points = []
        for pt in first_detection.get('coords', []):
            if isinstance(pt, dict) and '2d_point' in pt:
                example_2d_points.append(pt['2d_point'])

        group_dict = {
            'raamgroep': i + 1,
            'aantal_detecties': len(werkset),
            'label': most_common_label,
            'gemiddelde_hoekpunten': mean_per_corner,
            'standaardafwijking_per_hoekpunt': std_per_corner,
            'gemiddelde_standaardafwijking_xy_z': gemiddelde_stdev,
            'voorbeeld_image_id': example_image_id,
            'voorbeeld_2d_hoekpunten': example_2d_points or None,
        }

        grouped_intersections2_mean.append(group_dict)

    return grouped_intersections2_mean

def mean_and_std(grouped_intersections2):
    grouped_intersections2_mean = []

    for i, group in enumerate(grouped_intersections2):
        grouped_points = [[] for _ in range(4)]  # Eén lijst per hoekpunt

        # Meest voorkomende label bepalen
        labels = [detection['label'] for detection in group]
        label_counter = Counter(labels)
        most_common_label, _ = label_counter.most_common(1)[0]

        # Hoekpunten per hoek groeperen
        for detection in group:
            coords = detection['coords']  # lijst van 4 [x, y, z]-punten
            for j in range(4):
                grouped_points[j].append(coords[j])

        mean_per_corner = []
        std_per_corner = []

        for points in grouped_points:
            arr = np.array(points)

            mean_xy = np.mean(arr[:, :2], axis=0)
            mean_z = np.mean(arr[:, 2])
            std_xy = np.std(arr[:, :2], axis=0)
            std_z = np.std(arr[:, 2])

            planimetric_std = np.sqrt(std_xy[0]**2 + std_xy[1]**2)

            mean_per_corner.append([mean_xy[0], mean_xy[1], mean_z])
            std_per_corner.append([planimetric_std, std_z])

        # Gemiddelde standaardafwijkingen over de 4 hoeken
        std_xy_gem = np.mean([std[0] for std in std_per_corner])
        std_z_gem = np.mean([std[1] for std in std_per_corner])
        gemiddelde_stdev = [std_xy_gem, std_z_gem]

        group_dict = {
            'raamgroep': i + 1,
            'aantal_detecties': len(group),
            'label': most_common_label,
            'gemiddelde_hoekpunten': mean_per_corner,  # [x, y, z] per hoekpunt
            'standaardafwijking_per_hoekpunt': std_per_corner,  # [xy_std, z_std]
            'gemiddelde_standaardafwijking_xy_z': gemiddelde_stdev
        }

        grouped_intersections2_mean.append(group_dict)

    return grouped_intersections2_mean

def transform_to_vertices(pts_labels, city_json):
    transform = city_json.get('transform', {})
    scale = transform.get('scale', [])
    translation = transform.get('translate', [])
    sx, sy, sz = scale
    tx, ty, tz = translation

    vertices = []
    for quad in pts_labels['3d_points']:
        transformed_quad = []
        for point in quad:
            x, y, z = point
            x_new = ((x - tx)/sx)           # round
            y_new = ((y - ty)/sy)           # round
            z_new = ((z - tz)/sz)           # round
            transformed_quad.append([x_new, y_new, z_new])
        vertices.append(transformed_quad)

    return {
        'vertices': vertices,
        'labels': pts_labels['labels']}

def extract_all_lod2_boundaries(city_json, lod="2.2"):
    """
    Haalt LOD 2.2 Solid boundaries op van ALLE CityObjects.
    Returns:
        flat_boundaries: platte lijst van alle face-polygonen (elke face = [outer_ring, inner_ring, ...])
        face_map: per flat index → (obj_id, geom_index, lokale face-index)
    """
    flat_boundaries = []
    face_map = []
    for obj_id, obj in city_json.get("CityObjects", {}).items():
        for geom_idx, geom in enumerate(obj.get("geometry", [])):
            if geom.get("lod") == lod and geom.get("type") == "Solid":
                shell = geom.get("boundaries", [[]])[0]
                for local_idx, face in enumerate(shell):
                    flat_boundaries.append(face)
                    face_map.append((obj_id, geom_idx, local_idx))
    return flat_boundaries, face_map

def extract_vertices(city_json):
    return city_json.get("vertices", [])

def convert_boundaries_to_vertices(boundaries, vertices):
    """
    Solid: boundaries[0] = outer shell, dan [face][ring][vertex_idx].
    Geeft terug: lijst van faces, elke face = lijst van rings, elke ring = lijst van vertices (raw integers).
    """
    shell = boundaries[0]  # sla de shell-laag over
    result = []
    for face in shell:
        rings = []
        for ring in face:
            ring_verts = [vertices[idx] for idx in ring]  # raw integers, geen transform
            rings.append(ring_verts)
        result.append(rings)
    return result

def compute_plane(polygon):
    """Zoekt 3 niet-collineaire punten voor een stabiele normaalberekening.

    FIX: geeft voorrang aan een p3 dat op een andere hoogte (Z) ligt dan p1,
    zodat samengevoegde WallSurfaces waarvan de onderrand als eerste punten
    staan (allemaal zelfde Z) een correcte horizontale normaal krijgen in
    plaats van (0, 0, ±1).
    """
    pts = [np.array(p, dtype=float) for p in polygon]
    p1 = pts[0]

    # Zoek p2: eerste punt dat niet samenvalt met p1
    p2 = None
    for p in pts[1:]:
        if np.linalg.norm(p - p1) > 1e-6:
            p2 = p
            break
    if p2 is None:
        return None

    v1 = p2 - p1

    # Zoek p3: geef voorrang aan een punt op andere hoogte (Z) zodat de
    # normaal niet puur verticaal wordt bij horizontale onderranden.
    p3 = None
    for p in pts[1:]:
        v2 = p - p1
        if np.linalg.norm(np.cross(v1, v2)) > 1e-6 and abs(p[2] - p1[2]) > 1e-3:
            p3 = p
            break
    # Fallback: elk niet-collineair punt (origineel gedrag)
    if p3 is None:
        for p in pts[1:]:
            v2 = p - p1
            if np.linalg.norm(np.cross(v1, v2)) > 1e-6:
                p3 = p
                break
    if p3 is None:
        return None

    normal = np.cross(v1, p3 - p1)
    normal = normal / np.linalg.norm(normal)
    return normal, p1

def compute_planes(boundaries_as_vertices):
    """
    boundaries_as_vertices = [face][ring][vertex]
    Geeft één plane per face (berekend op de outer ring).
    """
    planes = []
    for face in boundaries_as_vertices:
        outer_ring = face[0]  # eerste ring = buitenrand
        if len(outer_ring) >= 3:
            planes.append(compute_plane(outer_ring))
        else:
            planes.append(None)
    return planes


def point_to_plane_distance(point, plane_normal, plane_point):
    return abs(np.dot(plane_normal, np.array(point) - plane_point))

def project_to_plane_tuples(polygon3d, origin, normal):
    """Projecteer 3D polygon naar 2D vlak, retourneert lijst van (x,y) tuples."""
    normal = normal / np.linalg.norm(normal)
    x_axis = np.cross([1, 0, 0], normal)
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.cross([0, 1, 0], normal)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(normal, x_axis)
    projected = []
    for p in polygon3d:
        vec = np.array(p) - np.array(origin)
        projected.append((np.dot(vec, x_axis), np.dot(vec, y_axis)))
    return projected

def point_in_polygon_2d(point_2d, polygon_2d):
    """Ray casting algoritme (geen Shapely nodig)"""
    x, y = point_2d
    n = len(polygon_2d)
    inside = False
    px, py = polygon_2d[0]
    for i in range(1, n + 1):
        cx, cy = polygon_2d[i % n]
        if ((py > y) != (cy > y)) and (x < (cx - px) * (y - py) / (cy - py) + px):
            inside = not inside
        px, py = cx, cy
    return inside

def _points_in_polygon_2d(points, polygon):
    """
    Gevectoriseerde ray-casting point-in-polygon test.

    Parameters
    ----------
    points : np.array(M, 2)
    polygon : np.array(N, 2)

    Returns
    -------
    np.array(M,) bool
    """
    n = len(polygon)
    m = len(points)
    inside = np.zeros(m, dtype=bool)

    px = points[:, 0]
    py = points[:, 1]

    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        # Voorwaarde: ray kruist de rand
        cond = ((yi > py) != (yj > py))

        # X-coördinaat van het snijpunt
        slope = (xj - xi) / (yj - yi + 1e-30)
        x_intersect = xi + slope * (py - yi)

        # Flip inside waar het snijpunt rechts van het punt ligt
        inside ^= (cond & (px < x_intersect))

        j = i

    return inside

def vertex_in_polygon(vertex, polygon_3d, plane_normal, plane_point):
    """Controleer of een vertex IN een polygoon ligt via 3D->2D projectie"""
    try:
        polygon_2d = project_to_plane_tuples(polygon_3d, plane_point, plane_normal)
        vertex_2d = project_to_plane_tuples([vertex], plane_point, plane_normal)[0]
        if len(polygon_2d) < 3:
            return False
        return point_in_polygon_2d(vertex_2d, polygon_2d)
    except:
        return False

def find_closest_face_indices_improved(object_vertices, planes, boundaries_as_vertices, distance_tolerance, camera_positions=None):
    """
    Zoekt per vertex het dichtstbijzijnde muurvlak.
    boundaries_as_vertices = [face][ring][vertex] (output van convert_boundaries_to_vertices)
    distance_tolerance in dezelfde eenheden als de raw vertices (mm als scale=0.001)

    camera_positions: optionele lijst van 3D camera-posities (in dezelfde coordinaten als vertices).
        Als opgegeven, worden gevels waarvan de normaal van ALLE camera-posities wegwijst
        uitgesloten (back-face culling). Dit voorkomt dat partijmuren (zijgevels tegen
        aanpalende gebouwen) ramen toegewezen krijgen.
    """
    flat_polygons = [face[0] for face in boundaries_as_vertices]  # outer ring per face

    def face_visible_from_any_camera(normal, point_on_plane, cam_positions):
        """True als minstens één camera aan de voorkant van het vlak staat."""
        if not cam_positions:
            return True
        for cam_pos in cam_positions:
            cam_arr = np.asarray(cam_pos, dtype=float)
            pt_arr = np.asarray(point_on_plane, dtype=float)
            if np.dot(normal, cam_arr - pt_arr) > 0:
                return True
        return False

    object_face_indices = []
    for vertices in object_vertices:
        face_indices = []
        for vertex in vertices:
            # Stap 1: kandidaatvlakken binnen tolerantie
            candidates = []
            for idx, plane in enumerate(planes):
                if plane is None:
                    continue
                normal, point_on_plane = plane
                dist = point_to_plane_distance(vertex, normal, point_on_plane)
                if dist <= distance_tolerance:
                    candidates.append((idx, dist, normal, point_on_plane))

            # Stap 1b: back-face culling — verwijder gevels die van alle camera's wegkijken
            if camera_positions is not None:
                visible_candidates = [
                    c for c in candidates
                    if face_visible_from_any_camera(c[2], c[3], camera_positions)
                ]
                # Gebruik visible_candidates als er overblijven, anders val terug op alle
                if visible_candidates:
                    candidates = visible_candidates

            # Stap 2: vertex moet ook binnen het polygoon liggen
            best_idx = None
            for idx, dist, normal, point_on_plane in sorted(candidates, key=lambda x: x[1]):
                if vertex_in_polygon(vertex, flat_polygons[idx], normal, point_on_plane):
                    best_idx = idx
                    break

            # Stap 3: fallback naar dichtstbijzijnde als niet in polygoon
            if best_idx is None and candidates:
                best_idx = min(candidates, key=lambda x: x[1])[0]

            face_indices.append(best_idx)
        object_face_indices.append(face_indices)
    return object_face_indices


def get_normals_for_faces(face_indices, planes):
    return [
        [planes[idx][0].tolist() if idx is not None and planes[idx] else None for idx in indices]
        for indices in face_indices
    ]

def align_vertex_with_normal(vertex, normal, point_on_plane, offset_distance=0):
    distance = np.dot(normal, np.array(vertex) - point_on_plane)
    return (np.array(vertex) - distance * normal + offset_distance * normal).tolist()

def align_vertices_with_plane(object_vertices, face_indices, planes, offset_distance=0):
    aligned_objects = []
    for obj_idx, vertices in enumerate(object_vertices):
        ref_idx = face_indices[obj_idx][0]
        if ref_idx is None or planes[ref_idx] is None:
            aligned_objects.append(vertices)
            continue
        normal, point_on_plane = planes[ref_idx]
        aligned_objects.append([
            align_vertex_with_normal(v, normal, point_on_plane, offset_distance)
            for v in vertices
        ])
    return aligned_objects

def normal_allignment_to_faces(city_json, vertices_labels, offset_distance=0, camera_positions=None):
    """Werkt nu over ALLE BuildingParts heen (niet enkel het eerste).

    camera_positions: optionele lijst van [x, y, z] camera-posities in wereldcoördinaten
        (zelfde stelsel als city_json vertices na transform).
        Als opgegeven, worden gevels die van alle camera's wegkijken uitgesloten
        bij het toewijzen van ramen → voorkomt ramen op partijmuren.
        Geef door als: [cam['cartesianTransform_image'][:3, 3].tolist() for cam in panorama_parameters]
    """
    flat_boundaries, face_map = extract_all_lod2_boundaries(city_json)
    vertices = extract_vertices(city_json)
    if not flat_boundaries or not vertices:
        raise ValueError("Ontbrekende LOD2 boundaries of vertices.")

    boundaries_as_vertices = convert_boundaries_to_vertices(
        [flat_boundaries], vertices
    )
    planes = compute_planes(boundaries_as_vertices)

    # Zet camera_positions om naar raw vertex-coordinaten (CityJSON gebruikt integer indices
    # met scale/translate). We moeten de wereldcoördinaten omzetten naar CityJSON-ruimte.
    raw_camera_positions = None
    if camera_positions is not None:
        transform = city_json.get('transform', {})
        scale = transform.get('scale', [1, 1, 1])
        translation = transform.get('translate', [0, 0, 0])
        sx, sy, sz = scale
        tx, ty, tz = translation
        raw_camera_positions = [
            [(cx - tx) / sx, (cy - ty) / sy, (cz - tz) / sz]
            for cx, cy, cz in camera_positions
        ]

    face_indices = find_closest_face_indices_improved(
        vertices_labels['vertices'],
        planes,
        boundaries_as_vertices,
        distance_tolerance=50,
        camera_positions=raw_camera_positions
    )

    normals = get_normals_for_faces(face_indices, planes)
    aligned_vertices = align_vertices_with_plane(
        vertices_labels['vertices'], face_indices, planes, offset_distance
    )

    return {
        'vertices': aligned_vertices,
        'labels': vertices_labels['labels'],
        'face_indices': face_indices,
        'normals': normals
    }

def are_planes_parallel(n1, n2, tol=1e-3):
    return np.isclose((np.dot(n1, n2)), 1.0, atol=tol)

def project_to_plane(polygon3d, origin, normal):
    normal = normal / np.linalg.norm(normal)
    x_axis = np.cross([1, 0, 0], normal)
    if np.linalg.norm(x_axis) < 1e-6:
        x_axis = np.cross([0, 1, 0], normal)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(normal, x_axis)

    projected = []
    for p in polygon3d:
        vec = np.array(p) - np.array(origin)
        x = np.dot(vec, x_axis)
        y = np.dot(vec, y_axis)
        projected.append((x, y))
    return Polygon(projected)

def polygon_area_2d(polygon):
    return polygon.area

def filter_overlapping_planes(output):
    vertices = output["vertices"]
    labels = output["labels"]
    face_indices = output["face_indices"]   # toevoegen
    normals = output["normals"]             # toevoegen
    keep_indices = set(range(len(vertices)))

    for i, j in combinations(range(len(vertices)), 2):
        if i not in keep_indices or j not in keep_indices:
            continue

        poly_i = vertices[i]
        poly_j = vertices[j]

        normal_i, _ = compute_plane(poly_i)
        normal_j, _ = compute_plane(poly_j)

        if are_planes_parallel(normal_i, normal_j):
            poly_i_2d = project_to_plane(poly_i, poly_i[0], normal_i)
            poly_j_2d = project_to_plane(poly_j, poly_i[0], normal_i)

            if poly_i_2d.intersects(poly_j_2d):
                area_i = polygon_area_2d(poly_i_2d)
                area_j = polygon_area_2d(poly_j_2d)

                if area_i >= area_j:
                    keep_indices.discard(j)
                else:
                    keep_indices.discard(i)

    # Behoud ook face_indices en normals
    keep = sorted(keep_indices)
    return {
        "vertices":    [vertices[i]    for i in keep],
        "labels":      [labels[i]      for i in keep],
        "face_indices": [face_indices[i] for i in keep],   # ← dit ontbrak
        "normals":     [normals[i]     for i in keep],     # ← dit ontbrak
    }

def duplicate_lod22_geometry(cityjson_data):
    """
    Duplicate LOD 2.2 geometries and convert them to LOD 3.0 MultiSurface format.
    
    Simple conversion:
    - Solid:        boundaries = [[[ring1], [ring2], ...]]
    - MultiSurface: boundaries = [[ring1], [ring2], ...]
    
    Just use boundaries[0]!
    """
    for obj_id, obj in cityjson_data.get("CityObjects", {}).items():
        if "geometry" not in obj:
            continue

        new_geometries = []
        for geom in obj["geometry"]:
            if geom.get("lod") == "2.2":
                geom_copy = copy.deepcopy(geom)
                geom_copy["lod"] = "3"
                geom_copy["type"] = "MultiSurface"
                
                # ✅ SIMPLE: Just use boundaries[0]
                # Solid: boundaries[0] = [[ring1], [ring2], ...]
                # This becomes MultiSurface boundaries directly!
                if geom_copy.get("boundaries") and len(geom_copy["boundaries"]) > 0:
                    geom_copy["boundaries"] = geom_copy["boundaries"][0]
                
                # ✅ Convert semantics values from 2D to 1D
                # Solid: values = [[0, 1, 2, ...]]
                # MultiSurface: values = [0, 1, 2, ...]
                if geom_copy.get("semantics") and geom_copy["semantics"].get("values"):
                    values = geom_copy["semantics"]["values"]
                    if isinstance(values, list) and len(values) > 0:
                        if isinstance(values[0], list):
                            geom_copy["semantics"]["values"] = values[0]
                
                new_geometries.append(geom_copy)

        obj["geometry"].extend(new_geometries)

    return cityjson_data

def _best_face_for_window(face_indices_per_corner, window_vertices, flat_boundaries, vertices, planes_cache=None):
    """
    Bepaalt het beste vlak voor een raam door:
    1. Meerderheidsregel over de 4 hoekpunten
    2. Containment-check: alle hoekpunten moeten binnen de outer ring vallen
    3. Fallback naar volgende kandidaat als containment faalt
    """
    from collections import Counter

    # Tel hoe vaak elke face voorkomt
    counts = Counter(f for f in face_indices_per_corner if f is not None)
    if not counts:
        return None

    # Probeer faces in volgorde van populariteit
    for face_idx, _ in counts.most_common():
        outer_ring_indices = flat_boundaries[face_idx][0]
        outer_ring_verts = [vertices[i] for i in outer_ring_indices]

        # Bereken vlak-normaal van de outer ring
        plane = compute_plane(outer_ring_verts)
        if plane is None:
            continue
        normal, pt = plane

        # Projecteer outer ring en window naar 2D
        outer_2d = project_to_plane_tuples(outer_ring_verts, pt, normal)
        window_2d = [project_to_plane_tuples([v], pt, normal)[0] for v in window_vertices]

        # Check of alle raamhoeken binnen de outer ring liggen
        all_inside = all(point_in_polygon_2d(wp, outer_2d) for wp in window_2d)
        if all_inside:
            return face_idx

    # Geen enkel vlak bevat het volledige raam → gebruik meest populaire
    best = counts.most_common(1)[0][0]
    print(f"  Waarschuwing: raam past niet volledig in face {best}, toch toegewezen")
    return best


def add_cutouts_to_cityjson(city_json, aligned_result, lod="2.2"):
    """
    Voegt ramen/deuren als inner rings (cutouts) toe aan de juiste
    muurvlakken in CityJSON.

    FIX t.o.v. vorige versie:
    - Meerderheidsregel over 4 hoeken (niet alleen face_indices[0])
    - Containment-check: inner ring moet BINNEN outer ring vallen
    - Vertices worden afgerond naar integers (CityJSON spec)
    """
    flat_boundaries, face_map = extract_all_lod2_boundaries(city_json, lod)
    raw_verts = city_json["vertices"]

    cutout_index_quads = []

    for window_vertices, face_indices in zip(
        aligned_result['vertices'], aligned_result['face_indices']
    ):
        # Raam-vertices afronden naar integers (CityJSON vereist integers)
        window_vertices_int = [[round(c) for c in v] for v in window_vertices]

        # Bepaal best passend vlak via meerderheid + containment
        face_idx = _best_face_for_window(
            face_indices, window_vertices_int, flat_boundaries, raw_verts
        )

        if face_idx is None or face_idx >= len(flat_boundaries):
            print(f"Geen geldig vlak gevonden, overgeslagen")
            cutout_index_quads.append(None)
            continue

        new_indices = add_vertices_to_cityjson(city_json, window_vertices_int)
        cutout_index_quads.append(new_indices)

        # Inner ring met omgekeerde winding voor het gat
        flat_boundaries[face_idx].append(list(reversed(new_indices)))

    # Terugschrijven per CityObject
    obj_faces = defaultdict(list)
    for flat_idx, (obj_id, geom_idx, local_idx) in enumerate(face_map):
        obj_faces[(obj_id, geom_idx)].append((local_idx, flat_boundaries[flat_idx]))

    for (obj_id, geom_idx), faces in obj_faces.items():
        geom = city_json["CityObjects"][obj_id]["geometry"][geom_idx]
        new_shell = [face_data for _, face_data in sorted(faces, key=lambda x: x[0])]
        geom["boundaries"] = [new_shell]

    city_json["_cutout_index_quads"] = cutout_index_quads
    return city_json

def add_vertices(global_vertices, detection_vertices):
    """
    Adds new vertices to the global vertex list and returns a list of index quads.
    """
    index_quads = []
    base_index = len(global_vertices)

    for quad in detection_vertices:
        for vertex in quad:
            global_vertices.append(vertex)
        indices = [base_index, base_index + 1, base_index + 2, base_index + 3]
        index_quads.append(indices)
        base_index += 4

    return index_quads

def add_boundary(boundaries, quad_indices):

    boundaries[0].append(quad_indices)

def add_semantic_value(semantics, label):

    surfaces = semantics.setdefault("surfaces", [])

    type_name = label.capitalize()

    type_to_index = {s["type"]: i for i, s in enumerate(surfaces)}
    if type_name not in type_to_index:
        surfaces.append({"type": type_name})
        type_to_index[type_name] = len(surfaces) - 1

    return type_to_index[type_name]

# def add_windows_doors(cityjson_data, detection_data, lod_source="2.2"):
#     """
#     Voegt gedetecteerde ramen en deuren toe als APARTE surfaces aan de
#     LOD 3 MultiSurface geometrie van het juiste BuildingPart.

#     Hergebruikt de vertex-indices die add_cutouts_to_cityjson al heeft
#     toegevoegd (opgeslagen in cityjson_data["_cutout_index_quads"]),
#     zodat er geen dubbele vertices ontstaan.
#     """
#     labels = detection_data['labels']
#     face_indices_list = detection_data['face_indices']

#     # Hergebruik de indices die add_cutouts al heeft aangemaakt
#     cutout_index_quads = cityjson_data.pop("_cutout_index_quads", None)

#     if cutout_index_quads is None:
#         # Fallback: als add_cutouts niet is aangeroepen, voeg vertices zelf toe
#         print("WAARSCHUWING: _cutout_index_quads niet gevonden, vertices worden opnieuw toegevoegd")
#         new_vertices = detection_data['vertices']
#         cutout_index_quads = []
#         base_index = len(cityjson_data.get("vertices", []))
#         for quad in new_vertices:
#             quad_indices = []
#             for vertex in quad:
#                 cityjson_data["vertices"].append(vertex)
#                 quad_indices.append(base_index)
#                 base_index += 1
#             cutout_index_quads.append(quad_indices)

#     # Haal face_map op om te bepalen bij welk CityObject elke detectie hoort
#     _, face_map = extract_all_lod2_boundaries(cityjson_data, lod_source)

#     # Groepeer detecties per obj_id
#     detections_per_obj = defaultdict(list)
#     for i, (label, face_indices) in enumerate(zip(labels, face_indices_list)):
#         if cutout_index_quads[i] is None:
#             continue
#         face_idx = face_indices[0]
#         if face_idx is not None and face_idx < len(face_map):
#             obj_id = face_map[face_idx][0]
#         else:
#             print(f"Waarschuwing: face_idx {face_idx} ongeldig, detectie overgeslagen")
#             continue
#         detections_per_obj[obj_id].append((label, cutout_index_quads[i]))

#     # Per object toevoegen aan LOD 3 MultiSurface
#     for obj_id, obj in cityjson_data.get("CityObjects", {}).items():
#         if obj_id not in detections_per_obj:
#             continue
#         for geom in obj.get("geometry", []):
#             if geom.get("lod") != "3" or geom.get("type") != "MultiSurface":
#                 continue

#             semantics = geom.setdefault("semantics", {})
#             if "surfaces" not in semantics:
#                 semantics["surfaces"] = []
#             if "values" not in semantics or not semantics["values"]:
#                 num_surfaces = len(geom.get("boundaries", []))
#                 semantics["values"] = [0] * num_surfaces if num_surfaces > 0 else []

#             for label, quad_indices in detections_per_obj[obj_id]:
#                 geom["boundaries"].append([quad_indices])
#                 semantic_index = add_semantic_value(semantics, label)
#                 semantics["values"].append(semantic_index)

#             geom["semantics"] = semantics

#     return cityjson_data

def add_windows_doors(cityjson_data, detection_data, inset_distance=100, lod_source="2.2"):
    """
    Voegt ramen/deuren toe met een dagkant (reveal/insprong).
 
    Per raam/deur worden 5 surfaces toegevoegd aan de LOD 3 MultiSurface:
      1. Het raam-/deurvlak zelf, verschoven met inset_distance naar BINNEN
      2. 4 dagkantvlakken (boven, onder, links, rechts) die de cutout
         op het muurvlak verbinden met het verzonken raam
 
    BELANGRIJK: deze functie vervangt add_windows_doors(). De cutouts
    moeten EERST zijn toegevoegd via add_cutouts_to_cityjson() met
    vertices die OP het muurvlak liggen (offset_distance=0).
 
    Parameters
    ----------
    cityjson_data : dict
        CityJSON dict met reeds toegevoegde cutouts en LOD3 geometrie.
        Moet _cutout_index_quads bevatten (aangemaakt door add_cutouts).
    detection_data : dict
        Output van filter_overlapping_planes / normal_allignment_to_faces:
        - 'vertices': lijst van quads (4 hoekpunten per raam, op muurvlak)
        - 'labels':   lijst van labels ('window' / 'door')
        - 'face_indices': lijst van face-indices per raam
        - 'normals':  lijst van normaalvectoren per raam
                      (per raam een lijst van 4 normalen, 1 per hoek)
    inset_distance : float
        Insprong in CityJSON integer-eenheden.
        Bij scale=0.001 is 100 = 10 cm.
        Standaard: 100 (= 10 cm naar binnen).
    lod_source : str
        LOD van de bron-geometrie voor face_map lookup.
 
    Returns
    -------
    dict
        Aangepaste CityJSON dictionary.
    """
    labels = detection_data['labels']
    face_indices_list = detection_data['face_indices']
    normals_list = detection_data['normals']
 
    # Haal de cutout-indices op die add_cutouts_to_cityjson heeft opgeslagen
    cutout_index_quads = cityjson_data.pop("_cutout_index_quads", None)
    if cutout_index_quads is None:
        raise ValueError(
            "_cutout_index_quads niet gevonden in cityjson_data. "
            "Roep eerst add_cutouts_to_cityjson() aan."
        )
 
    # Face map om te bepalen bij welk CityObject elke detectie hoort
    from functions_enrichment import extract_all_lod2_boundaries, add_semantic_value
    _, face_map = extract_all_lod2_boundaries(cityjson_data, lod_source)
 
    raw_verts = cityjson_data["vertices"]
 
    # ── Per raam: maak inset vertices + groepeer per CityObject ──
    detections_per_obj = defaultdict(list)
 
    for i, (label, face_indices, normals_per_corner) in enumerate(
            zip(labels, face_indices_list, normals_list)):
 
        cutout_indices = cutout_index_quads[i]
        if cutout_indices is None:
            continue
 
        # Pak de normaal van het eerste hoekpunt (alle 4 liggen op
        # hetzelfde vlak na alignment, dus de normaal is identiek)
        normal = np.array(normals_per_corner[0], dtype=float)
        n_len = np.linalg.norm(normal)
        if n_len < 1e-12:
            print(f"  WARN: nul-normaal voor detectie {i}, overgeslagen")
            continue
        normal = normal / n_len  # zeker unit vector
 
        # Maak 4 inset vertices: cutout_vertex − inset_distance * normaal
        inset_indices = []
        for ci in cutout_indices:
            cutout_v = np.array(raw_verts[ci], dtype=float)
            inset_v = cutout_v - inset_distance * normal
            inset_v_int = [round(c) for c in inset_v]
            new_idx = len(raw_verts)
            raw_verts.append(inset_v_int)
            inset_indices.append(new_idx)
 
        # Bepaal bij welk CityObject dit raam hoort
        face_idx = face_indices[0]
        if face_idx is not None and face_idx < len(face_map):
            obj_id = face_map[face_idx][0]
        else:
            print(f"  WARN: ongeldige face_idx {face_idx} voor detectie {i}")
            continue
 
        detections_per_obj[obj_id].append({
            'label': label,
            'cutout_indices': cutout_indices,   # C0, C1, C2, C3
            'inset_indices': inset_indices,     # W0, W1, W2, W3
        })
 
    # ── Toevoegen aan LOD 3 geometrie ──
    n_windows = 0
    n_reveals = 0
 
    for obj_id, obj in cityjson_data.get("CityObjects", {}).items():
        if obj_id not in detections_per_obj:
            continue
 
        for geom in obj.get("geometry", []):
            if geom.get("lod") != "3" or geom.get("type") != "MultiSurface":
                continue
 
            semantics = geom.setdefault("semantics", {})
            surfaces = semantics.setdefault("surfaces", [])
            values = semantics.setdefault("values", [])
 
            # Zoek of maak het reveal surface type
            reveal_sem_idx = _get_or_create_surface_type(
                surfaces, "WallSurface", is_reveal=True
            )
 
            for det in detections_per_obj[obj_id]:
                label = det['label']
                ci = det['cutout_indices']   # [C0, C1, C2, C3]
                wi = det['inset_indices']    # [W0, W1, W2, W3]
 
                # ─── 1. Raam/deur-vlak (op inset positie) ───
                window_sem_idx = add_semantic_value(semantics, label)
                geom["boundaries"].append([list(wi)])
                values.append(window_sem_idx)
                n_windows += 1
 
                # ─── 2. Dagkantvlakken (4 quads) ───
                #
                # Hoekpunt-volgorde uit convert_boxes_to_corners:
                #   0 = linksboven  (TL)
                #   1 = rechtsboven (TR)
                #   2 = linksonder  (BL)
                #   3 = rechtsonder (BR)
                #
                #   C0(TL) ─── C1(TR)
                #     │            │
                #     │   muur     │
                #     │            │
                #   C2(BL) ─── C3(BR)
                #
                #   W0(TL) ─── W1(TR)     (10cm naar binnen)
                #     │  raam    │
                #   W2(BL) ─── W3(BR)
                #
                # De 4 dagkant-quads:
                reveal_quads = [
                    [ci[0], ci[1], wi[1], wi[0]],  # rand 0→1
                    [ci[1], ci[2], wi[2], wi[1]],  # rand 1→2
                    [ci[2], ci[3], wi[3], wi[2]],  # rand 2→3
                    [ci[3], ci[0], wi[0], wi[3]],  # rand 3→0
                ]
 
                for quad in reveal_quads:
                    geom["boundaries"].append([quad])
                    values.append(reveal_sem_idx)
                    n_reveals += 1
 
            geom["semantics"] = semantics
 
    print(f"[add_windows_doors_with_reveal]")
    print(f"  {n_windows} raam/deur-vlakken toegevoegd (inset={inset_distance} eenheden)")
    print(f"  {n_reveals} dagkantvlakken toegevoegd ({n_reveals // 4} ramen × 4 zijden)")
    print(f"  {n_windows * 4} nieuwe inset-vertices aangemaakt")
 
    return cityjson_data
 
 
def _get_or_create_surface_type(surfaces, type_name, **attrs):
    """
    Zoekt een bestaand surface type met exact dezelfde attributen,
    of maakt een nieuw type aan als het nog niet bestaat.
    """
    for i, s in enumerate(surfaces):
        if s.get("type") == type_name and all(
                s.get(k) == v for k, v in attrs.items()):
            return i
    new_surf = {"type": type_name}
    new_surf.update(attrs)
    surfaces.append(new_surf)
    return len(surfaces) - 1

def compute_plane_robust(polygon):
    """
    Berekent het vlak van een polygoon door 3 niet-collineaire punten te zoeken.
    Geeft (normaal, punt_op_vlak) terug, of None als het polygoon degeneraat is.

    FIX: geeft voorrang aan een p3 dat op een andere hoogte (Z) ligt dan p1,
    zodat samengevoegde WallSurfaces waarvan de onderrand als eerste punten
    staan (allemaal zelfde Z) een correcte horizontale normaal krijgen in
    plaats van (0, 0, ±1).
    """
    pts = [np.array(p, dtype=float) for p in polygon]
    p1 = pts[0]
    p2 = next((p for p in pts[1:] if np.linalg.norm(p - p1) > 1e-6), None)
    if p2 is None:
        return None
    v1 = p2 - p1
    # Geef voorrang aan p3 op andere hoogte (Z) dan p1
    p3 = next(
        (p for p in pts[1:]
         if np.linalg.norm(np.cross(v1, p - p1)) > 1e-6 and abs(p[2] - p1[2]) > 1e-3),
        None
    )
    # Fallback: elk niet-collineair punt (origineel gedrag)
    if p3 is None:
        p3 = next((p for p in pts[1:] if np.linalg.norm(np.cross(v1, p - p1)) > 1e-6), None)
    if p3 is None:
        return None
    n = np.cross(v1, p3 - p1)
    n = n / np.linalg.norm(n)
    return n, p1


def merge_two_rings(ring_a, ring_b):
    """
    Voegt twee aansluitende polygoonringen samen door de gedeelde edge te verwijderen.
    Beide ringen zijn lijsten van vertex-indices.
    Geeft de samengevoegde ring terug, of None als er geen gedeelde edge is.
    """
    shared = set(ring_a) & set(ring_b)
    if len(shared) < 2:
        return None

    n_a, n_b = len(ring_a), len(ring_b)

    # Zoek de gedeelde edge in ring_a (twee opeenvolgende vertices, beide in shared)
    edge_start_a = next(
        (i for i in range(n_a)
         if ring_a[i] in shared and ring_a[(i + 1) % n_a] in shared),
        None
    )
    if edge_start_a is None:
        return None

    v1 = ring_a[edge_start_a]
    v2 = ring_a[(edge_start_a + 1) % n_a]

    # Zoek de omgekeerde edge in ring_b
    edge_start_b = next(
        (i for i in range(n_b)
         if ring_b[i] == v2 and ring_b[(i + 1) % n_b] == v1),
        None
    )
    if edge_start_b is None:
        return None

    # Bouw de samengevoegde ring:
    # ring_a (zonder de gedeelde edge) + ring_b (zonder de gedeelde edge)
    merged = []
    for i in range(n_a - 1):
        merged.append(ring_a[(edge_start_a + 1 + i) % n_a])
    for i in range(n_b - 1):
        merged.append(ring_b[(edge_start_b + 1 + i) % n_b])
    return merged


def _merge_single_solid(geom, raw_verts, normal_tol, dist_tol):
    """
    Voert de coplanaire merge uit op één Solid-geometrie (in-place).
    """
    shell = geom["boundaries"][0]
    sem_values = geom.get("semantics", {}).get("values", [[]])[0]
    n = len(shell)

    # Bereken vlakken
    planes = [
        compute_plane_robust([raw_verts[i] for i in shell[fi][0]])
        for fi in range(n)
    ]

    # Coplanair + aansluitend?
    def coplanar(i, j):
        pi, pj = planes[i], planes[j]
        if pi is None or pj is None:
            return False
        ni, pti = pi
        nj, ptj = pj
        if abs(np.dot(ni, nj)) < 1 - normal_tol:
            return False
        dist = abs(np.dot(ni, np.array(ptj, dtype=float) - np.array(pti, dtype=float)))
        return dist < dist_tol

    def adjacent(i, j):
        si = set(v for ring in shell[i] for v in ring)
        sj = set(v for ring in shell[j] for v in ring)
        return bool(si & sj)

    # Union-Find
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        parent[find(x)] = find(y)

    for i in range(n):
        for j in range(i + 1, n):
            if coplanar(i, j) and adjacent(i, j):
                union(i, j)

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)

    # Samenvoegen
    new_shell = []
    new_sem_values = []

    for root in sorted(groups):
        members = groups[root]

        if len(members) == 1:
            new_shell.append(shell[members[0]])
            new_sem_values.append(sem_values[members[0]] if sem_values else None)
            continue

        # Merge iteratief
        current_ring = list(shell[members[0]][0])
        sem_val = sem_values[members[0]] if sem_values else None
        remaining = list(members[1:])

        changed = True
        while remaining and changed:
            changed = False
            for idx in remaining[:]:
                other_ring = list(shell[idx][0])
                merged = merge_two_rings(current_ring, other_ring)
                if merged is not None:
                    current_ring = merged
                    remaining.remove(idx)
                    changed = True
                    break

        if remaining:
            print(f"  [merge_coplanar] Waarschuwing: faces {remaining} konden niet "
                  f"samengevoegd worden met {members[0]}, worden apart gehouden.")
            new_shell.append([current_ring])
            new_sem_values.append(sem_val)
            for r in remaining:
                new_shell.append(shell[r])
                new_sem_values.append(sem_values[r] if sem_values else None)
        else:
            new_shell.append([current_ring])
            new_sem_values.append(sem_val)
            print(f"  [merge_coplanar] Faces {members} samengevoegd "
                  f"→ 1 vlak met {len(current_ring)} vertices")

    # Terugschrijven
    geom["boundaries"][0] = new_shell
    if sem_values:
        geom["semantics"]["values"][0] = new_sem_values

    print(f"  → {n} faces → {len(new_shell)} faces")


def merge_coplanar_adjacent_faces(city_json, lod="2.2", normal_tol=1e-2, dist_tol=50):
    """
    Voegt coplanaire én aansluitende vlakken samen in een CityJSON Solid vóór
    de raam/deur-toewijzing, zodat een raam nooit op een naad tussen twee
    coplanaire vlakken kan vallen.

    Twee vlakken worden samengevoegd als:
      1. Hun normaalvectoren (bijna) parallel zijn  [normal_tol]
      2. Ze op hetzelfde vlak liggen (afstand < dist_tol, in raw integer mm)  [dist_tol]
      3. Ze minstens één gedeelde vertex-index hebben (aansluitend)

    FIX: de oorspronkelijke versie had een 'break' die enkel de binnenste
    for-loop (over geometry) verbrak. De buitenste loop (over CityObjects)
    liep gewoon door, waardoor 'geom' telkens werd overschreven door de
    laatst gevonden Solid. Hierdoor werd alleen het laatste BuildingPart
    verwerkt (2-0) en werd het eerste (1-0, met de meeste merges) overgeslagen.

    Nu wordt _merge_single_solid aangeroepen voor ELKE Solid-geometrie.
    """
    raw_verts = city_json["vertices"]

    for obj_id, obj in city_json.get("CityObjects", {}).items():
        for g in obj.get("geometry", []):
            if g.get("lod") == lod and g.get("type") == "Solid":
                print(f"\n[merge_coplanar] Verwerken: {obj_id} (LOD {lod})")
                _merge_single_solid(g, raw_verts, normal_tol, dist_tol)

    return city_json

def add_vertices_to_cityjson(city_json, new_vertices):
    """
    Voegt nieuwe vertices toe aan de globale vertices lijst.
    Geeft de startindex terug van de toegevoegde vertices.
    """
    existing_vertices = city_json.get("vertices", [])
    start_index = len(existing_vertices)
    existing_vertices.extend(new_vertices)
    city_json["vertices"] = existing_vertices
    return list(range(start_index, start_index + len(new_vertices)))

def extract_roof_surfaces(city_json, lod="2.2"):
    """
    Extraheert alle dakvlakken (RoofSurface) uit een CityJSON bestand,
    puur in-memory – schrijft GEEN bestanden weg.
 
    Parameters
    ----------
    city_json : dict
        Ingeladen CityJSON dictionary.
    lod : str
        Het level of detail om te gebruiken (standaard "2.2").
 
    Returns
    -------
    list[dict]
        Lijst met info per geëxtraheerd dakvlak:
        {
            'obj_id': str,
            'surface_idx': int,
            'roof_index': int,
            'semantic_info': dict,
            'faces': list[list],
            'face_indices_in_shell': list[int],
        }
    """
    transform = city_json.get("transform", {})
    scale = transform.get("scale", [1, 1, 1])
    translate = transform.get("translate", [0, 0, 0])
    sx, sy, sz = scale
    tx, ty, tz = translate
 
    raw_vertices = city_json.get("vertices", [])
 
    results = []
    roof_counter = 0
 
    for obj_id, obj in city_json.get("CityObjects", {}).items():
        for geom in obj.get("geometry", []):
            if geom.get("lod") != lod or geom.get("type") != "Solid":
                continue
 
            semantics = geom.get("semantics", {})
            surfaces = semantics.get("surfaces", [])
            values = semantics.get("values", [[]])[0]
            shell = geom.get("boundaries", [[]])[0]
 
            roof_surface_defs = {}
            for surf_idx, surf in enumerate(surfaces):
                if surf.get("type") == "RoofSurface":
                    roof_surface_defs[surf_idx] = surf
 
            roof_face_groups = {}
            for face_idx, face in enumerate(shell):
                if face_idx >= len(values):
                    continue
                sem_value = values[face_idx]
                if sem_value not in roof_surface_defs:
                    continue
                outer_ring = face[0]
                if sem_value not in roof_face_groups:
                    roof_face_groups[sem_value] = []
                roof_face_groups[sem_value].append((face_idx, outer_ring))
 
            for surf_idx in sorted(roof_face_groups.keys()):
                face_entries = roof_face_groups[surf_idx]
                semantic_info = roof_surface_defs[surf_idx]
 
                all_faces_world = []
                face_indices_in_shell = []
 
                for face_idx, outer_ring in face_entries:
                    world_verts = [
                        [raw_vertices[vi][0] * sx + tx,
                         raw_vertices[vi][1] * sy + ty,
                         raw_vertices[vi][2] * sz + tz]
                        for vi in outer_ring
                    ]
                    all_faces_world.append(world_verts)
                    face_indices_in_shell.append(face_idx)
 
                results.append({
                    "obj_id": obj_id,
                    "surface_idx": surf_idx,
                    "roof_index": roof_counter,
                    "semantic_info": semantic_info,
                    "faces": all_faces_world,
                    "face_indices_in_shell": face_indices_in_shell,
                })
 
                roof_counter += 1
 
    print(f"[extract_roof_surfaces] {roof_counter} dakvlakken geëxtraheerd")
    return results



def _compute_plane_axes(vertices, normal):
    """
    Bereken twee robuuste orthogonale assen (u, v) in een vlak.
    Kiest u langs de langste rand om numerieke instabiliteit te voorkomen.

    Parameters
    ----------
    vertices : np.array(N, 3)
    normal : np.array(3,)

    Returns
    -------
    u, v : np.array(3,), np.array(3,)
    """
    n = len(vertices)
    best_len = 0.0
    best_edge = None
    for i in range(n):
        edge = vertices[(i + 1) % n] - vertices[i]
        elen = np.linalg.norm(edge)
        if elen > best_len:
            best_len = elen
            best_edge = edge

    u = best_edge / best_len
    # Projecteer u in het vlak (verwijder component langs normaal)
    u = u - np.dot(u, normal) * normal
    u_len = np.linalg.norm(u)
    if u_len < 1e-12:
        # Fallback: kies een willekeurige as loodrecht op normaal
        if abs(normal[0]) < 0.9:
            u = np.cross(normal, np.array([1, 0, 0]))
        else:
            u = np.cross(normal, np.array([0, 1, 0]))
        u_len = np.linalg.norm(u)
    u = u / u_len

    v = np.cross(normal, u)
    v = v / np.linalg.norm(v)
    return u, v

def generate_roof_edge_strips(roof_surfaces, d_small, d_large):
    """
    Genereer per dakvlak per rand twee rechthoekige zoekstrips.

    Parameters
    ----------
    roof_surfaces : list[dict]
        Output van extract_roof_surfaces_to_obj (stap 1).
    d_small : float
        Breedte van de kleine strip in meter (standaard 0.5m).
    d_large : float
        Buitenste grens van de grote strip in meter (standaard 1.5m).

    Returns
    -------
    list[dict]
        Per dakvlak:
        {
            'obj_id': str,
            'surface_idx': int,
            'roof_index': int,
            'semantic_info': dict,
            'vlak_normaal': np.array(3,),       # normaalvector van het dakvlak
            'vlak_punt': np.array(3,),           # punt op het dakvlak
            'origineel': np.array(N, 3),         # originele polygoon vertices
            'randen': [
                {
                    'edge_start': np.array(3,),
                    'edge_end': np.array(3,),
                    'outward_dir': np.array(3,), # eenheidsvector naar buiten
                    'klein': np.array(4, 3),      # rechthoek [A, B, B', A']
                    'groot': np.array(4, 3),      # rechthoek [A', B', B'', A'']
                },
                ...
            ]
        }
    """
    results = []

    for roof in roof_surfaces:
        # Neem de eerste (en meestal enige) face
        # Als er meerdere faces zijn, verwerk elk apart
        for face_verts in roof["faces"]:
            verts = np.array(face_verts, dtype=np.float64)
            n = len(verts)

            if n < 3:
                continue

            # --- Bereken vlak-normaal via Newell's methode (robuust) ---
            centroid = verts.mean(axis=0)
            normal = np.zeros(3)
            for i in range(n):
                curr = verts[i]
                nxt = verts[(i + 1) % n]
                normal[0] += (curr[1] - nxt[1]) * (curr[2] + nxt[2])
                normal[1] += (curr[2] - nxt[2]) * (curr[0] + nxt[0])
                normal[2] += (curr[0] - nxt[0]) * (curr[1] + nxt[1])

            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-12:
                # Ontaard vlak (collineaire vertices), skip
                continue

            normal = normal / norm_len

            # Zorg dat normaal "omhoog" wijst (positieve z-component)
            if normal[2] < 0:
                normal = -normal

            # --- Bepaal winding order via signed area in het 2D-vlak ---
            u_ax, v_ax = _compute_plane_axes(verts, normal)

            verts_rel = verts - centroid
            verts_2d_u = verts_rel @ u_ax
            verts_2d_v = verts_rel @ v_ax

            # Signed area (shoelace): positief = CCW, negatief = CW
            signed_area = 0.5 * (
                np.dot(verts_2d_u, np.roll(verts_2d_v, -1))
                - np.dot(verts_2d_v, np.roll(verts_2d_u, -1))
            )
            # cross(normal, edge_vec) wijst INWARD voor CCW winding
            # Dus: CCW (positieve area) -> sign = -1, CW -> sign = +1
            winding_sign = -1.0 if signed_area > 0 else 1.0

            # --- Per rand: bereken outward richting en strips ---
            # Bepaal min/max Z voor dakgoot-detectie
            z_min = verts[:, 2].min()
            z_max = verts[:, 2].max()
            z_range = z_max - z_min
            slope = roof["semantic_info"].get("rf_slope", 0)

            randen = []
            for i in range(n):
                A = verts[i]
                B = verts[(i + 1) % n]

                edge_vec = B - A
                edge_len = np.linalg.norm(edge_vec)
                if edge_len < 1e-9:
                    continue

                # Loodrecht op de rand, in het dakvlak
                # Richting consistent bepaald via winding order
                outward = winding_sign * np.cross(normal, edge_vec)
                outward = outward / np.linalg.norm(outward)

                # Klein: A, B, B + d_small * outward, A + d_small * outward
                A_small = A + d_small * outward
                B_small = B + d_small * outward

                # Groot: A_small, B_small, B + d_large * outward, A + d_large * outward
                A_large = A + d_large * outward
                B_large = B + d_large * outward

                rand_entry = {
                    "edge_start": A,
                    "edge_end": B,
                    "outward_dir": outward,
                    "klein": np.array([A, B, B_small, A_small]),
                    "groot": np.array([A_small, B_small, B_large, A_large]),
                    "is_gutter": False,
                    "is_ridge": False,
                    "klein_h": None,
                    "groot_h": None,
                    "outward_h": None,
                }

                # --- Nok-detectie (ridge) ---
                # Een rand is een nokrand als:
                # 1. Het dak een significante helling heeft (>5°)
                # 2. Beide vertices dicht bij z_max zitten
                # 3. De outward richting naar boven wijst (z > 0)
                if (slope > 5.0 and z_range > 0.5
                        and A[2] > z_max - z_range * 0.25
                        and B[2] > z_max - z_range * 0.25
                        and outward[2] > 0.05):
                    rand_entry["is_ridge"] = True

                # --- Dakgoot-detectie ---
                # Een rand is een dakgootrand als:
                # 1. Het dak een significante helling heeft (>5°)
                # 2. Beide vertices dicht bij z_min zitten
                # 3. De outward richting naar beneden wijst (z < 0)
                if (slope > 5.0 and z_range > 0.5
                        and A[2] < z_min + z_range * 0.25
                        and B[2] < z_min + z_range * 0.25
                        and outward[2] < -0.05):

                    # Horizontale outward: neem XY-component, nul Z
                    outward_h = outward.copy()
                    outward_h[2] = 0.0
                    h_len = np.linalg.norm(outward_h)
                    if h_len > 1e-9:
                        outward_h = outward_h / h_len

                        A_small_h = A + d_small * outward_h
                        B_small_h = B + d_small * outward_h
                        A_large_h = A + d_large * outward_h
                        B_large_h = B + d_large * outward_h

                        rand_entry["is_gutter"] = True
                        rand_entry["outward_h"] = outward_h
                        rand_entry["klein_h"] = np.array(
                            [A, B, B_small_h, A_small_h])
                        rand_entry["groot_h"] = np.array(
                            [A_small_h, B_small_h, B_large_h, A_large_h])

                randen.append(rand_entry)

            results.append({
                "obj_id": roof["obj_id"],
                "surface_idx": roof["surface_idx"],
                "roof_index": roof["roof_index"],
                "semantic_info": roof["semantic_info"],
                "vlak_normaal": normal,
                "vlak_punt": centroid,
                "origineel": verts,
                "randen": randen,
            })

    print(f"[generate_roof_edge_strips] {len(results)} dakvlakken verwerkt, "
          f"totaal {sum(len(r['randen']) for r in results)} rand-strips")
    return results

def classify_points_per_edge(roof_strips, points, max_plane_dist=0.30):
    """
    Classificeer puntenwolkpunten per dakvlak en per rand in de zones
    origineel / klein / groot.

    Geoptimaliseerd met:
    - 3D bounding box pre-filter per dakvlak (reduceert ~95% van de punten)
    - Gevectoriseerde vlak-afstandsberekening
    - Snelle rechthoek-test via as-projectie (i.p.v. ray-casting) voor strips
    - Eenmalige 2D-projectie, hergebruikt voor alle zones

    Parameters
    ----------
    roof_strips : list[dict]
        Output van generate_roof_edge_strips.
    points : np.array(M, 3)
        Puntenwolk in wereldcoördinaten (bijv. uit .las/.laz).
    max_plane_dist : float
        Maximale loodrechte afstand tot het dakvlak in meter (standaard 0.10m).

    Returns
    -------
    list[dict]
        Per dakvlak:
        {
            'obj_id': str,
            'surface_idx': int,
            'roof_index': int,
            'origineel_punten': np.array(K, 3),
            'randen': [
                {
                    'n_klein': int,
                    'n_groot': int,
                    'dichtheid_klein': float,   # punten per m²
                    'dichtheid_groot': float,    # punten per m²
                    'is_oversteek': bool,
                    'punten_klein': np.array,
                    'punten_groot': np.array,
                },
                ...
            ]
        }
    """
    points = np.asarray(points, dtype=np.float64)
    results = []
    _empty_3 = np.empty((0, 3))

    for roof in roof_strips:
        normal = roof["vlak_normaal"]
        plane_pt = roof["vlak_punt"]
        origineel = roof["origineel"]
        randen = roof["randen"]

        # ==============================================================
        # STAP 0: 3D bounding box pre-filter
        # Verzamel alle vertices (origineel + alle strips) en maak een
        # ruime bbox. Punten buiten deze bbox hoeven niet getest.
        # ==============================================================
        all_verts = [origineel]
        for rand in randen:
            all_verts.append(rand["groot"])
            if rand["groot_h"] is not None:
                all_verts.append(rand["groot_h"])
        all_verts = np.vstack(all_verts)
        bbox_min = all_verts.min(axis=0) - max_plane_dist
        bbox_max = all_verts.max(axis=0) + max_plane_dist

        bbox_mask = np.all(
            (points >= bbox_min) & (points <= bbox_max), axis=1
        )
        bbox_pts = points[bbox_mask]

        if len(bbox_pts) == 0:
            results.append(_empty_roof_result(roof, randen))
            continue

        # ==============================================================
        # STAP A: Vlak-afstandsfilter voor SCHUIN vlak (vectorized)
        # ==============================================================
        diffs = bbox_pts - plane_pt
        dists = np.abs(diffs @ normal)
        near_mask = dists < max_plane_dist
        nearby_pts = bbox_pts[near_mask]

        # ==============================================================
        # STAP B+C: 2D-projectie + origineel vlak (als er punten zijn)
        # ==============================================================
        if len(nearby_pts) > 0:
            u, v = _compute_plane_axes(origineel, normal)

            rel = nearby_pts - plane_pt
            pts_u = rel @ u
            pts_v = rel @ v

            orig_rel = origineel - plane_pt
            orig_2d = np.column_stack([orig_rel @ u, orig_rel @ v])
            pts_2d = np.column_stack([pts_u, pts_v])

            in_orig = _points_in_polygon_2d(pts_2d, orig_2d)
            origineel_punten = nearby_pts[in_orig]
        else:
            u, v = _compute_plane_axes(origineel, normal)
            origineel_punten = np.empty((0, 3))
            pts_u = np.empty(0)
            pts_v = np.empty(0)

        # ==============================================================
        # STAP D: Per rand — schuin + horizontaal
        # ==============================================================
        randen_result = []
        for rand in randen:
            A = rand["edge_start"]
            B = rand["edge_end"]
            edge_dir = B - A
            edge_len = np.linalg.norm(edge_dir)
            edge_dir_n = edge_dir / edge_len

            # --- SCHUIN (langs het dakvlak) ---
            n_klein_s, n_groot_s = 0, 0
            d_klein_s, d_groot_s = 0.0, 0.0
            pts_klein_s = np.empty((0, 3))
            pts_groot_s = np.empty((0, 3))

            if len(nearby_pts) > 0:
                outward = rand["outward_dir"]
                ed_u = edge_dir_n @ u
                ed_v = edge_dir_n @ v
                ow_u = outward @ u
                ow_v = outward @ v

                A_rel = A - plane_pt
                A_u, A_v = A_rel @ u, A_rel @ v

                du = pts_u - A_u
                dv = pts_v - A_v
                proj_edge = du * ed_u + dv * ed_v
                proj_out = du * ow_u + dv * ow_v

                d_small_val = np.linalg.norm(rand["klein"][3] - A)
                d_large_val = np.linalg.norm(rand["groot"][3] - A)

                in_klein = (
                    (proj_edge >= -1e-9) & (proj_edge <= edge_len + 1e-9) &
                    (proj_out >= -1e-9) & (proj_out <= d_small_val + 1e-9)
                )
                in_groot = (
                    (proj_edge >= -1e-9) & (proj_edge <= edge_len + 1e-9) &
                    (proj_out >= d_small_val - 1e-9) &
                    (proj_out <= d_large_val + 1e-9)
                )

                n_klein_s = int(in_klein.sum())
                n_groot_s = int(in_groot.sum())

                area_klein = edge_len * d_small_val
                area_groot = edge_len * (d_large_val - d_small_val)
                d_klein_s = n_klein_s / area_klein if area_klein > 1e-9 else 0.0
                d_groot_s = n_groot_s / area_groot if area_groot > 1e-9 else 0.0

                pts_klein_s = nearby_pts[in_klein]
                pts_groot_s = nearby_pts[in_groot]

            # --- HORIZONTAAL (voor dakgootranden) ---
            n_klein_h, n_groot_h = 0, 0
            d_klein_h, d_groot_h = 0.0, 0.0
            pts_klein_h = np.empty((0, 3))
            pts_groot_h = np.empty((0, 3))

            if rand["is_gutter"] and rand["klein_h"] is not None:
                outward_h = rand["outward_h"]
                z_plane = (A[2] + B[2]) / 2.0

                # Filter bbox_pts op horizontaal vlak (z-afstand)
                z_dists = np.abs(bbox_pts[:, 2] - z_plane)
                h_mask = z_dists < max_plane_dist
                h_pts = bbox_pts[h_mask]

                if len(h_pts) > 0:
                    # Projecteer op edge_dir (3D) en outward_h (3D, horizontaal)
                    h_rel = h_pts - A
                    proj_edge_h = h_rel @ edge_dir_n
                    proj_out_h = h_rel @ outward_h

                    d_small_h = np.linalg.norm(rand["klein_h"][3] - A)
                    d_large_h = np.linalg.norm(rand["groot_h"][3] - A)

                    in_klein_h = (
                        (proj_edge_h >= -1e-9) &
                        (proj_edge_h <= edge_len + 1e-9) &
                        (proj_out_h >= -1e-9) &
                        (proj_out_h <= d_small_h + 1e-9)
                    )
                    in_groot_h = (
                        (proj_edge_h >= -1e-9) &
                        (proj_edge_h <= edge_len + 1e-9) &
                        (proj_out_h >= d_small_h - 1e-9) &
                        (proj_out_h <= d_large_h + 1e-9)
                    )

                    n_klein_h = int(in_klein_h.sum())
                    n_groot_h = int(in_groot_h.sum())

                    area_klein_h = edge_len * d_small_h
                    area_groot_h = edge_len * (d_large_h - d_small_h)
                    d_klein_h = n_klein_h / area_klein_h if area_klein_h > 1e-9 else 0.0
                    d_groot_h = n_groot_h / area_groot_h if area_groot_h > 1e-9 else 0.0

                    pts_klein_h = h_pts[in_klein_h]
                    pts_groot_h = h_pts[in_groot_h]

            # --- Beslissingsregel: neem de versie met meer punten ---
            if d_klein_s > 0:
                is_oversteek_s = d_groot_s < d_klein_s * 0.3
            else:
                is_oversteek_s = False

            if d_klein_h > 0:
                is_oversteek_h = d_groot_h < d_klein_h * 0.3
            else:
                is_oversteek_h = False

            randen_result.append({
                # Schuin (langs dakvlak)
                "n_klein": n_klein_s,
                "n_groot": n_groot_s,
                "dichtheid_klein": d_klein_s,
                "dichtheid_groot": d_groot_s,
                "is_oversteek": is_oversteek_s,
                "punten_klein": pts_klein_s,
                "punten_groot": pts_groot_s,
                # Horizontaal (enkel voor dakgootranden)
                "is_gutter": rand["is_gutter"],
                "n_klein_h": n_klein_h,
                "n_groot_h": n_groot_h,
                "dichtheid_klein_h": d_klein_h,
                "dichtheid_groot_h": d_groot_h,
                "is_oversteek_h": is_oversteek_h,
                "punten_klein_h": pts_klein_h,
                "punten_groot_h": pts_groot_h,
            })

        results.append({
            "obj_id": roof["obj_id"],
            "surface_idx": roof["surface_idx"],
            "roof_index": roof["roof_index"],
            "origineel_punten": origineel_punten,
            "randen": randen_result,
        })

    return results

def _empty_roof_result(roof, randen):
    """Helper voor lege resultaten."""
    _empty = np.empty((0, 3))
    return {
        "obj_id": roof["obj_id"],
        "surface_idx": roof["surface_idx"],
        "roof_index": roof["roof_index"],
        "origineel_punten": _empty,
        "randen": [
            {"n_klein": 0, "n_groot": 0,
             "dichtheid_klein": 0.0, "dichtheid_groot": 0.0,
             "is_oversteek": False,
             "punten_klein": _empty, "punten_groot": _empty,
             "is_gutter": rand.get("is_gutter", False),
             "n_klein_h": 0, "n_groot_h": 0,
             "dichtheid_klein_h": 0.0, "dichtheid_groot_h": 0.0,
             "is_oversteek_h": False,
             "punten_klein_h": _empty, "punten_groot_h": _empty}
            for rand in randen
        ],
    }

#Overhangs met RMSE scheiden voor robuustheid

# Hulpfuncties

def _fit_plane_normal(points):
    """
    Fit een vlak door een set punten via SVD en geef de normaal terug.
    De normaal wijst altijd 'omhoog' (positieve z-component).

    Parameters
    ----------
    points : np.array(N, 3)

    Returns
    -------
    np.array(3,) of None
        Normaalvector van het gefitte vlak, of None als fit onmogelijk.
    """
    if len(points) < 3:
        return None
    centroid = points.mean(axis=0)
    centered = points - centroid
    try:
        _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    normal = Vt[-1]
    if normal[2] < 0:
        normal = -normal
    return normal

def _angle_between_normals_deg(n1, n2):
    """
    Bereken de hoek (in graden) tussen twee normaalvectoren.

    Parameters
    ----------
    n1, n2 : np.array(3,)

    Returns
    -------
    float
        Hoek in graden [0, 180].
    """
    cos_angle = np.clip(np.dot(n1, n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(abs(cos_angle))))

def _fit_plane_rmse(points):
    """
    Fit een vlak door de punten via SVD en geef de RMSE van de
    residuen terug (hoe goed de punten een vlak vormen).

    Parameters
    ----------
    points : np.array(N, 3)

    Returns
    -------
    float
        RMSE van de loodrechte afstanden tot het best-fit vlak.
        Hoge waarde = punten vormen geen coherent vlak (ruis).
        Returns inf als fit onmogelijk is.
    """
    if len(points) < 3:
        return float('inf')
    centroid = points.mean(axis=0)
    centered = points - centroid
    try:
        _, S, Vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return float('inf')
    normal = Vt[-1]
    dists = centered @ normal
    return float(np.sqrt(np.mean(dists ** 2)))

def _tight_resegment(points, rand_g, normal, plane_pt,
                     z_gutter, tight_dist):
    """
    Her-segmenteer een gutter-rand met een strengere vlak-afstand.
    Gevectoriseerd: zoekt punten binnen tight_dist van zowel het schuine
    als het horizontale vlak, beperkt tot de klein-strip.

    Returns
    -------
    rmse_s, rmse_h : float
        RMSE voor schuin en horizontaal met tight tolerantie.
    pts_s, pts_h : np.array(N, 3)
        Punten binnen de tight tolerantie per methode.
    """
    A = rand_g["edge_start"]
    B = rand_g["edge_end"]
    edge_dir = B - A
    edge_len = np.linalg.norm(edge_dir)
    edge_dir_n = edge_dir / edge_len

    # Bbox pre-filter (ruim rond de klein strips)
    all_strip_verts = [rand_g["klein"]]
    if rand_g["klein_h"] is not None:
        all_strip_verts.append(rand_g["klein_h"])
    all_strip_verts = np.vstack(all_strip_verts)
    bbox_min = all_strip_verts.min(axis=0) - tight_dist
    bbox_max = all_strip_verts.max(axis=0) + tight_dist

    bbox_mask = np.all((points >= bbox_min) & (points <= bbox_max), axis=1)
    bbox_pts = points[bbox_mask]

    _empty = np.empty((0, 3))

    if len(bbox_pts) == 0:
        return float('inf'), float('inf'), _empty, _empty

    # --- SCHUIN: tight vlak-afstand + binnen klein strip ---
    dists_plane = np.abs((bbox_pts - plane_pt) @ normal)
    near_s = dists_plane < tight_dist
    pts_near_s = bbox_pts[near_s]

    pts_s = _empty
    rmse_s = float('inf')
    if len(pts_near_s) > 0:
        outward = rand_g["outward_dir"]
        rel_s = pts_near_s - A
        proj_edge_s = rel_s @ edge_dir_n
        proj_out_s = rel_s @ outward
        d_small_val = np.linalg.norm(rand_g["klein"][3] - A)

        in_strip_s = (
            (proj_edge_s >= -1e-9) & (proj_edge_s <= edge_len + 1e-9) &
            (proj_out_s >= -1e-9) & (proj_out_s <= d_small_val + 1e-9)
        )
        pts_s = pts_near_s[in_strip_s]
        if len(pts_s) > 0:
            rmse_s = _rmse_to_plane(pts_s, normal, plane_pt)

    # --- HORIZONTAAL: tight Z-afstand + binnen klein_h strip ---
    dists_z = np.abs(bbox_pts[:, 2] - z_gutter)
    # Exclude points on the sloped roof
    dists_roof = np.abs((bbox_pts - plane_pt) @ normal)
    near_h = (dists_z < tight_dist) & (dists_roof >= 0.05)
    pts_near_h = bbox_pts[near_h]

    pts_h = _empty
    rmse_h = float('inf')
    if len(pts_near_h) > 0 and rand_g["klein_h"] is not None:
        outward_h = rand_g["outward_h"]
        rel_h = pts_near_h - A
        proj_edge_h = rel_h @ edge_dir_n
        proj_out_h = rel_h @ outward_h
        d_small_h = np.linalg.norm(rand_g["klein_h"][3] - A)

        in_strip_h = (
            (proj_edge_h >= -1e-9) & (proj_edge_h <= edge_len + 1e-9) &
            (proj_out_h >= -1e-9) & (proj_out_h <= d_small_h + 1e-9)
        )
        pts_h = pts_near_h[in_strip_h]
        if len(pts_h) > 0:
            rmse_h = _rmse_to_horizontal(pts_h, z_gutter)

    return rmse_s, rmse_h, pts_s, pts_h

def _is_oversteek(n_klein, n_groot, min_pts, max_ratio):
    """
    Beslissingsregel: is dit een oversteek?

    True als:
      1. Genoeg punten in de klein-zone (>= min_pts)
      2. De grote zone bevat minder dan max_ratio * n_klein punten

    Voorbeeld: min_pts=5, max_ratio=0.5
      n_klein=100, n_groot=20  → 20 < 100*0.5 = 50  → True (oversteek)
      n_klein=100, n_groot=80  → 80 < 100*0.5 = 50  → False (dak loopt door)
    """
    if n_klein < min_pts:
        return False
    if n_groot < n_klein * max_ratio:
        return True
    return False

def _rmse_to_plane(points, normal, plane_pt):
    """
    RMSE van de loodrechte afstand van punten tot een vlak.
    """
    if len(points) == 0:
        return float('inf')
    dists = (points - plane_pt) @ normal
    return float(np.sqrt(np.mean(dists ** 2)))

def _rmse_to_horizontal(points, z_height):
    """
    RMSE van de Z-afstand van punten tot een horizontaal vlak.
    """
    if len(points) == 0:
        return float('inf')
    dists = points[:, 2] - z_height
    return float(np.sqrt(np.mean(dists ** 2)))

def _merge_unique_points(pts_a, pts_b, tol=0.005):
    """
    Combineer twee puntenverzamelingen, verwijder duplicaten.
    """
    if len(pts_a) == 0:
        return pts_b
    if len(pts_b) == 0:
        return pts_a

    combined = np.vstack([pts_a, pts_b])
    scale = 1.0 / tol
    rounded = np.round(combined * scale).astype(np.int64)
    _, unique_idx = np.unique(rounded, axis=0, return_index=True)
    return combined[np.sort(unique_idx)]

def _estimate_overhang_width(points, edge_start, edge_end, outward_dir):
    """
    Bereken de HORIZONTALE breedte van de oversteek, loodrecht op de
    dakrand geprojecteerd op het XY-vlak.

    Gebruikt het 90e percentiel om uitschieters te filteren.

    Parameters
    ----------
    points : np.array(N, 3)
    edge_start : np.array(3,)
    edge_end : np.array(3,)
    outward_dir : np.array(3,)
        Richting naar buiten (hoeft niet horizontaal te zijn;
        wordt alleen gebruikt om de juiste zijde te kiezen).

    Returns
    -------
    float
        Horizontale oversteekbreedte in meter.
    """
    if len(points) == 0:
        return 0.0

    # Horizontale randrichting
    edge_h = np.array([edge_end[0] - edge_start[0],
                        edge_end[1] - edge_start[1],
                        0.0])
    edge_h_len = np.linalg.norm(edge_h)

    if edge_h_len < 1e-9:
        return 0.0

    edge_h_n = edge_h / edge_h_len

    # Loodrecht op de rand in het XY-vlak (90° rotatie)
    perp_h = np.array([-edge_h_n[1], edge_h_n[0], 0.0])

    # Zorg dat perp_h dezelfde kant op wijst als outward_dir
    if np.dot(perp_h, outward_dir) < 0:
        perp_h = -perp_h

    rel = points - edge_start
    projections = rel @ perp_h
    positive = projections[projections > 0]

    if len(positive) == 0:
        return 0.0

    return max(0.0, float(np.percentile(positive, 90)))

def filter_confirmed_overhangs(classification_result, roof_strips,
                                points=None,
                                min_points=5,
                                max_ratio_groot=0.5,
                                rmse_tiebreak_threshold=0.005,
                                tight_plane_dist=0.05,
                                max_angle_schuin=10.0,
                                max_angle_horiz=10.0,
                                max_rmse_schuin=0.10,
                                max_rmse_horiz=0.30,
                                min_coverage=0.30,
                                max_plane_rmse=0.15):
    """
    Filter classificatieresultaten tot enkel bevestigde dakoversteken.

    Parameters
    ----------
    classification_result : list[dict]
        Output van classify_points_per_edge.
    roof_strips : list[dict]
        Output van generate_roof_edge_strips (nodig voor geometrie).
    points : np.array(M, 3) or None
        Originele puntenwolk. Nodig voor tight-RMSE herberekening.
        Als None wordt de tiebreak overgeslagen.
    min_points : int
        Minimum aantal punten in de klein-zone om een rand te overwegen.
    max_ratio_groot : float
        Maximaal toegelaten verhouding n_groot / n_klein.
    rmse_tiebreak_threshold : float
        Als het RMSE-verschil tussen schuin en horizontaal kleiner is
        dan deze waarde (in meter), wordt een her-segmentatie gedaan
        met tight_plane_dist voor een definitief antwoord.
    tight_plane_dist : float
        Strengere vlak-afstand (in meter) voor de her-segmentatie.
    max_angle_schuin : float
        Maximale hoekafwijking in graden tussen het gefitte vlak van
        de oversteekpunten en het dakvlak (schuine methode).
        Filtert kielgoten en aangrenzende vlakken. Standaard 10.0°.
    max_angle_horiz : float
        Maximale hoekafwijking in graden tussen het gefitte vlak van
        de oversteekpunten en de verticale as [0,0,1] (horizontale
        methode). Standaard 10.0°.
    max_rmse_schuin : float
        Maximale RMSE (in meter) van de oversteekpunten tot het
        dakvlak voor de schuine methode. Standaard 0.10m.
    max_rmse_horiz : float
        Maximale RMSE (in meter) van de oversteekpunten tot het
        horizontale vlak op goothoogte. Standaard 0.30m.
    min_coverage : float
        Minimale dekkingsgraad van de punten langs de rand (0–1).
        Berekend als de span van de punten (5e–95e percentiel van
        de projectie op de randrichting) gedeeld door de randlengte.
        Filtert clusters die slechts een klein stuk van de rand
        bestrijken. Standaard 0.30 (30%).
    max_plane_rmse : float
        Maximale RMSE (in meter) van de punten tot hun eigen
        best-fit vlak (planariteitscheck). Hoge waarde betekent
        dat de punten geen coherent oppervlak vormen (ruis).
        Standaard 0.15m.

    Returns
    -------
    list[dict]
        Per dakvlak met minstens één bevestigde oversteek:
        {
            'obj_id': str,
            'surface_idx': int,
            'roof_index': int,
            'oversteken': [
                {
                    'rand_idx': int,
                    'methode': str,           # 'schuin', 'horizontaal', of 'beide'
                    'punten': np.array(N,3),  # alle punten van de oversteek
                    'breedte': float,          # geschatte breedte in meter
                    'edge_start': np.array(3,),
                    'edge_end': np.array(3,),
                    'edge_len': float,
                    'outward_dir': np.array(3,),
                    'n_punten': int,
                    'dichtheid': float,
                    'schuin': dict,
                    'horizontaal': dict or None,
                },
                ...
            ]
        }
    """
    results = []

    for roof_class, roof_geom in zip(classification_result, roof_strips):
        oversteken = []

        for rand_idx, (rand_c, rand_g) in enumerate(
                zip(roof_class["randen"], roof_geom["randen"])):

            # --- Nok-randen overslaan ---
            if rand_g.get("is_ridge", False):
                continue

            # --- Evalueer schuine methode ---
            schuin_ok = _is_oversteek(
                rand_c["n_klein"], rand_c["n_groot"],
                min_points, max_ratio_groot
            )

            # --- Evalueer horizontale methode (indien gutter) ---
            horiz_ok = False
            if rand_c.get("is_gutter", False):
                horiz_ok = _is_oversteek(
                    rand_c["n_klein_h"], rand_c["n_groot_h"],
                    min_points, max_ratio_groot
                )

            if not schuin_ok and not horiz_ok:
                continue

            # --- Bepaal beste methode via RMSE ---
            tiebreak_used = False
            if schuin_ok and horiz_ok:
                normal = roof_geom["vlak_normaal"]
                plane_pt = roof_geom["vlak_punt"]
                z_gutter = (rand_g["edge_start"][2] + rand_g["edge_end"][2]) / 2.0

                rmse_s = _rmse_to_plane(rand_c["punten_klein"], normal, plane_pt)
                rmse_h = _rmse_to_horizontal(rand_c["punten_klein_h"], z_gutter)

                # Als RMSE's te dicht bij elkaar liggen: her-segmentatie
                if (abs(rmse_s - rmse_h) < rmse_tiebreak_threshold
                        and points is not None):
                    tiebreak_used = True
                    rmse_s_tight, rmse_h_tight, pts_s_tight, pts_h_tight = \
                        _tight_resegment(
                            points, rand_g, normal, plane_pt,
                            z_gutter, tight_plane_dist
                        )
                    if rmse_s_tight <= rmse_h_tight:
                        methode = "schuin"
                        punten = pts_s_tight
                    else:
                        methode = "horizontaal"
                        punten = pts_h_tight
                elif rmse_s <= rmse_h:
                    methode = "schuin"
                    punten = rand_c["punten_klein"]
                else:
                    methode = "horizontaal"
                    punten = rand_c["punten_klein_h"]
            elif schuin_ok:
                methode = "schuin"
                punten = rand_c["punten_klein"]
            else:
                methode = "horizontaal"
                punten = rand_c["punten_klein_h"]

            if len(punten) == 0:
                continue

            # --- Hoekcheck schuine methode ---
            # Controleer of de oversteekpunten op hetzelfde dakvlak
            # liggen. Filtert kielgoten, dakkapellen en aangrenzende
            # vlakken die in de strip vallen.
            if methode == "schuin" and len(punten) >= 3:
                fitted_normal = _fit_plane_normal(punten)
                if fitted_normal is not None:
                    normal = roof_geom["vlak_normaal"]
                    angle_dev = _angle_between_normals_deg(fitted_normal, normal)
                    if angle_dev > max_angle_schuin:
                        continue  # punten liggen op een ander vlak → kielgoot

            # --- Horizontaal-validatie: liggen de punten echt horizontaal? ---
            # Wanneer de horizontale methode wint, fit een vlak door de
            # punten en controleer of dat vlak quasi-horizontaal is
            # (normaal wijkt max max_angle_horiz af van [0,0,1]).
            if methode == "horizontaal" and len(punten) >= 3:
                fitted_normal = _fit_plane_normal(punten)
                if fitted_normal is not None:
                    _vertical = np.array([0.0, 0.0, 1.0])
                    angle_from_horiz = _angle_between_normals_deg(
                        fitted_normal, _vertical
                    )
                    if angle_from_horiz > max_angle_horiz:
                        continue  # punten liggen niet horizontaal

            # --- RMSE-kwaliteitscheck: zijn de punten coherent? ---
            # Verwerp oversteken waarvan de punten te ver van het
            # referentievlak liggen (verspreide / incoherente punten).
            normal = roof_geom["vlak_normaal"]
            plane_pt = roof_geom["vlak_punt"]
            if methode == "schuin":
                rmse_check = _rmse_to_plane(punten, normal, plane_pt)
                if rmse_check > max_rmse_schuin:
                    continue  # punten liggen te ver van dakvlak
            else:  # horizontaal
                z_gut = (rand_g["edge_start"][2] + rand_g["edge_end"][2]) / 2.0
                rmse_check = _rmse_to_horizontal(punten, z_gut)
                if rmse_check > max_rmse_horiz:
                    continue  # punten liggen te ver van horizontaal vlak

            # --- Dekkingsgraad-check: bestrijken punten de hele rand? ---
            edge_dir = rand_g["edge_end"] - rand_g["edge_start"]
            edge_len_check = np.linalg.norm(edge_dir)
            if edge_len_check > 1e-9 and len(punten) >= 3:
                edge_dir_n = edge_dir / edge_len_check
                proj = (punten - rand_g["edge_start"]) @ edge_dir_n
                span = np.percentile(proj, 95) - np.percentile(proj, 5)
                coverage = span / edge_len_check
                if coverage < min_coverage:
                    continue  # punten zitten geconcentreerd in een klein stuk

            # --- Planariteitscheck: vormen de punten een coherent vlak? ---
            if len(punten) >= 3:
                plane_rmse = _fit_plane_rmse(punten)
                if plane_rmse > max_plane_rmse:
                    continue  # punten vormen geen coherent oppervlak

            # --- Bepaal outward richting ---
            if methode == "horizontaal" and rand_g.get("outward_h") is not None:
                outward_measure = rand_g["outward_h"]
            else:
                outward_measure = rand_g["outward_dir"]

            # --- Bereken oversteekbreedte ---
            breedte = _estimate_overhang_width(
                punten, rand_g["edge_start"], rand_g["edge_end"],
                outward_measure
            )

            # --- Bouw resultaat ---
            edge_dir = rand_g["edge_end"] - rand_g["edge_start"]
            edge_len = np.linalg.norm(edge_dir)

            # Bereken RMSE voor beide methoden (ook als maar één actief is)
            normal = roof_geom["vlak_normaal"]
            plane_pt = roof_geom["vlak_punt"]

            rmse_schuin = _rmse_to_plane(
                rand_c["punten_klein"], normal, plane_pt
            ) if len(rand_c["punten_klein"]) > 0 else float('inf')

            rmse_horiz = float('inf')
            if rand_c.get("is_gutter", False) and len(rand_c["punten_klein_h"]) > 0:
                z_gutter = (rand_g["edge_start"][2] + rand_g["edge_end"][2]) / 2.0
                rmse_horiz = _rmse_to_horizontal(
                    rand_c["punten_klein_h"], z_gutter)

            schuin_data = {
                "n_klein": rand_c["n_klein"],
                "n_groot": rand_c["n_groot"],
                "d_klein": rand_c["dichtheid_klein"],
                "d_groot": rand_c["dichtheid_groot"],
                "is_oversteek": schuin_ok,
                "rmse": rmse_schuin,
                "punten_klein": rand_c["punten_klein"],
            }

            horiz_data = None
            if rand_c.get("is_gutter", False):
                horiz_data = {
                    "n_klein": rand_c["n_klein_h"],
                    "n_groot": rand_c["n_groot_h"],
                    "d_klein": rand_c["dichtheid_klein_h"],
                    "d_groot": rand_c["dichtheid_groot_h"],
                    "is_oversteek": horiz_ok,
                    "rmse": rmse_horiz,
                    "punten_klein": rand_c["punten_klein_h"],
                }

            oversteken.append({
                "rand_idx": rand_idx,
                "methode": methode,
                "tiebreak_used": tiebreak_used,
                "punten": punten,
                "breedte": breedte,
                "edge_start": rand_g["edge_start"],
                "edge_end": rand_g["edge_end"],
                "edge_len": edge_len,
                "outward_dir": outward_measure,
                "n_punten": len(punten),
                "dichtheid": len(punten) / (edge_len * breedte)
                    if (edge_len * breedte) > 1e-9 else 0.0,
                "schuin": schuin_data,
                "horizontaal": horiz_data,
            })

        if oversteken:
            results.append({
                "obj_id": roof_class["obj_id"],
                "surface_idx": roof_class["surface_idx"],
                "roof_index": roof_class["roof_index"],
                "oversteken": oversteken,
            })

    n_oversteken = sum(len(r["oversteken"]) for r in results)
    n_daken = len(results)
    print(f"[filter_confirmed_overhangs] {n_oversteken} oversteken "
          f"bevestigd op {n_daken} dakvlakken")
    return results

def print_overhang_summary(confirmed_overhangs):
    """
    Print een overzichtelijke samenvatting van alle bevestigde oversteken.
    """
    print(f"\n{'='*75}")
    print(f"BEVESTIGDE DAKOVERSTEKEN")
    print(f"{'='*75}")

    total = 0
    for roof in confirmed_overhangs:
        print(f"\n  {roof['obj_id']} surface {roof['surface_idx']}:")
        for ov in roof["oversteken"]:
            total += 1
            tb = " [TIEBREAK]" if ov.get("tiebreak_used", False) else ""
            print(f"    Rand {ov['rand_idx']:>2}: "
                  f"breedte={ov['breedte']:.2f}m, "
                  f"randlengte={ov['edge_len']:.2f}m, "
                  f"punten={ov['n_punten']:>5}, "
                  f"methode={ov['methode']}{tb}")

            if ov["horizontaal"] is not None:
                s = ov["schuin"]
                h = ov["horizontaal"]
                print(f"           schuin:  klein={s['n_klein']:>4} "
                      f"({s['d_klein']:>6.1f}/m²)  "
                      f"groot={s['n_groot']:>4} ({s['d_groot']:>6.1f}/m²)  "
                      f"RMSE={s['rmse']:.4f}m  "
                      f"{'<< BEST' if s['is_oversteek'] and s['rmse'] <= h.get('rmse', float('inf')) else ('OK' if s['is_oversteek'] else '--')}")
                print(f"           horiz:   klein={h['n_klein']:>4} "
                      f"({h['d_klein']:>6.1f}/m²)  "
                      f"groot={h['n_groot']:>4} ({h['d_groot']:>6.1f}/m²)  "
                      f"RMSE={h['rmse']:.4f}m  "
                      f"{'<< BEST' if h['is_oversteek'] and h['rmse'] < s.get('rmse', float('inf')) else ('OK' if h['is_oversteek'] else '--')}")

    print(f"\n  Totaal: {total} bevestigde oversteken op "
          f"{len(confirmed_overhangs)} dakvlakken")
    print(f"{'='*75}")

def _estimate_overhang_width(points, edge_start, edge_end, outward_dir):
    """
    Bereken de HORIZONTALE breedte van de oversteek, loodrecht op de
    dakrand geprojecteerd op het XY-vlak.

    Gebruikt het 90e percentiel om uitschieters te filteren.

    Parameters
    ----------
    points : np.array(N, 3)
    edge_start : np.array(3,)
    edge_end : np.array(3,)
    outward_dir : np.array(3,)
        Richting naar buiten (hoeft niet horizontaal te zijn;
        wordt alleen gebruikt om de juiste zijde te kiezen).

    Returns
    -------
    float
        Horizontale oversteekbreedte in meter.
    """
    if len(points) == 0:
        return 0.0

    # Horizontale randrichting
    edge_h = np.array([edge_end[0] - edge_start[0],
                        edge_end[1] - edge_start[1],
                        0.0])
    edge_h_len = np.linalg.norm(edge_h)

    if edge_h_len < 1e-9:
        return 0.0

    edge_h_n = edge_h / edge_h_len

    # Loodrecht op de rand in het XY-vlak (90° rotatie)
    perp_h = np.array([-edge_h_n[1], edge_h_n[0], 0.0])

    # Zorg dat perp_h dezelfde kant op wijst als outward_dir
    if np.dot(perp_h, outward_dir) < 0:
        perp_h = -perp_h

    rel = points - edge_start
    projections = rel @ perp_h
    positive = projections[projections > 0]

    if len(positive) == 0:
        return 0.0

    return max(0.0, float(np.percentile(positive, 90)))

def generate_overhang_strips(confirmed_overhangs):
    """
    Genereer oversteekstrips die IN het dakvlak liggen (zelfde normaal
    als het originele dakvlak, A'B' evenwijdig aan AB).

    De strip wordt geëxtrudeerd langs de outward_dir die al in het
    dakvlak ligt (schuine methode) of horizontaal is (horizontale
    methode), geschaald zodat de horizontale projectie gelijk is
    aan de gemeten breedte.

    Parameters
    ----------
    confirmed_overhangs : list[dict]
        Output van filter_confirmed_overhangs.

    Returns
    -------
    list[dict]
        Per dakvlak met oversteken:
        {
            'obj_id': str,
            'surface_idx': int,
            'roof_index': int,
            'strips': [
                {
                    'rand_idx': int,
                    'methode': str,
                    'breedte_h': float,          # horizontale breedte (input)
                    'breedte_slope': float,       # werkelijke afstand langs vlak
                    'edge_start': np.array(3,),
                    'edge_end': np.array(3,),
                    'edge_len': float,
                    'outward_dir': np.array(3,),  # extrusierichting (in vlak)
                    'normaal': np.array(3,),      # vlak-normaal van de strip
                    'vertices': np.array(4, 3),   # [A, B, B', A'] rechthoek
                },
                ...
            ]
        }
    """
    results = []

    for roof in confirmed_overhangs:
        strips = []

        for ov in roof["oversteken"]:
            A = ov["edge_start"].copy()
            B = ov["edge_end"].copy()
            breedte_h = ov["breedte"]
            outward = ov["outward_dir"]
            methode = ov["methode"]

            if breedte_h < 1e-6:
                continue

            if methode == "horizontaal":
                # --- Horizontale strip: A en B blijven origineel ---
                # Extrudeer puur horizontaal (z=0) zodat de strip
                # aansluit op de dakrand maar horizontaal uitsteekt.
                outward_n = outward[:2]
                outward_2d_len = np.linalg.norm(outward_n)
                if outward_2d_len < 1e-9:
                    continue
                outward_n = outward_n / outward_2d_len

                offset = np.array([breedte_h * outward_n[0],
                                   breedte_h * outward_n[1],
                                   0.0])
                A_out = A + offset
                B_out = B + offset

                # Normaal berekenen uit werkelijke vertices
                edge_vec = B - A
                strip_normal = np.cross(edge_vec, offset)
                sn_len = np.linalg.norm(strip_normal)
                if sn_len < 1e-12:
                    continue
                strip_normal = strip_normal / sn_len
                if strip_normal[2] < 0:
                    strip_normal = -strip_normal

            else:
                # --- Schuine strip: in het dakvlak ---
                outward_len = np.linalg.norm(outward)
                if outward_len < 1e-9:
                    continue
                outward_n_3d = outward / outward_len

                offset = breedte_h * outward_n_3d
                A_out = A + offset
                B_out = B + offset

                edge_vec = B - A
                strip_normal = np.cross(edge_vec, offset)
                n_len = np.linalg.norm(strip_normal)
                if n_len < 1e-12:
                    continue
                strip_normal = strip_normal / n_len
                if strip_normal[2] < 0:
                    strip_normal = -strip_normal

            vertices = np.array([A, B, B_out, A_out])

            breedte_slope = float(np.linalg.norm(offset))

            strips.append({
                "rand_idx": ov["rand_idx"],
                "methode": ov["methode"],
                "breedte_h": breedte_h,
                "breedte_slope": breedte_slope,
                "edge_start": A,
                "edge_end": B,
                "edge_len": ov["edge_len"],
                "outward_dir": outward,
                "normaal": strip_normal,
                "vertices": vertices,
            })

        if strips:
            results.append({
                "obj_id": roof["obj_id"],
                "surface_idx": roof["surface_idx"],
                "roof_index": roof["roof_index"],
                "strips": strips,
            })

    n_strips = sum(len(r["strips"]) for r in results)
    print(f"[generate_overhang_strips] {n_strips} strips "
          f"gegenereerd op {len(results)} dakvlakken")
    for roof in results:
        for s in roof["strips"]:
            print(f"  {roof['obj_id']} s{roof['surface_idx']} "
                  f"rand {s['rand_idx']}: "
                  f"breedte_h={s['breedte_h']:.3f}m, "
                  f"breedte_slope={s['breedte_slope']:.3f}m, "
                  f"methode={s['methode']}")
    return results

def add_overhangs_to_cityjson(city_json, overhang_strips):
    """
    Voeg oversteekstrips toe aan een CityJSON-dict als LOD3-geometrie.
 
    Kopieert de LOD 2.2 geometrie als basis, voegt per oversteekstrip
    een nieuw RoofSurface-vlak toe. Vertices worden gededupliceerd:
    gedeelde punten (A, B) worden hergebruikt.
 
    Werkt in-place op de meegegeven dictionary en geeft deze ook terug,
    zodat de functie naadloos aansluit op de enrichment-pipeline.
 
    Parameters
    ----------
    city_json : dict
        Ingeladen (en eventueel al verrijkte) CityJSON dictionary.
    overhang_strips : list[dict]
        Output van generate_overhang_strips().
 
    Returns
    -------
    dict
        Dezelfde (gewijzigde) CityJSON dictionary.
    """
    scale = city_json["transform"]["scale"]
    translate = city_json["transform"]["translate"]
 
    # --- Bouw vertex-lookup voor deduplicatie ---
    vertex_map = {}
    for idx, v in enumerate(city_json["vertices"]):
        key = tuple(v)
        vertex_map[key] = idx
 
    def _get_or_add_vertex(real_xyz):
        """Converteer real-world coördinaat naar CityJSON integer vertex.
        Hergebruik bestaande vertex als die al bestaat."""
        ix = round((real_xyz[0] - translate[0]) / scale[0])
        iy = round((real_xyz[1] - translate[1]) / scale[1])
        iz = round((real_xyz[2] - translate[2]) / scale[2])
        key = (ix, iy, iz)
 
        if key in vertex_map:
            return vertex_map[key]
 
        new_idx = len(city_json["vertices"])
        city_json["vertices"].append([ix, iy, iz])
        vertex_map[key] = new_idx
        return new_idx
 
    # --- Per CityObject: zoek bijhorende strips en voeg LOD3 toe ---
    n_strips_added = 0
    n_verts_before = len(city_json["vertices"])
 
    for roof_strips in overhang_strips:
        obj_id = roof_strips["obj_id"]
 
        # Zoek het CityObject (direct of als BuildingPart)
        if obj_id in city_json["CityObjects"]:
            part_id = obj_id
        elif f"{obj_id}-0" in city_json["CityObjects"]:
            part_id = f"{obj_id}-0"
        else:
            print(f"  WARN: {obj_id} niet gevonden in CityJSON, skip")
            continue
 
        co = city_json["CityObjects"][part_id]
 
        # Zoek de LOD 2.2 geometrie als basis
        lod22_geom = None
        for geom in co["geometry"]:
            if geom["lod"] == "2.2":
                lod22_geom = geom
                break
 
        if lod22_geom is None:
            print(f"  WARN: Geen LOD 2.2 voor {part_id}, skip")
            continue
 
        # Controleer of LOD3 al bestaat, anders kopieer LOD 2.2
        lod3_geom = None
        for geom in co["geometry"]:
            if geom["lod"] == "3":
                lod3_geom = geom
                break
 
        if lod3_geom is None:
            lod3_geom = copy.deepcopy(lod22_geom)
            lod3_geom["lod"] = "3"
            co["geometry"].append(lod3_geom)
 
        # --- Voeg strips toe als nieuwe faces ---
        boundaries = lod3_geom["boundaries"]
        sem_surfaces = lod3_geom["semantics"]["surfaces"]
        sem_values = lod3_geom["semantics"]["values"]
 
        # Detecteer geometrie-type: MultiSurface heeft een platte
        # values-lijst [0,1,2,...] en boundaries = [[ring], ...],
        # Solid heeft geneste values [[0,1,2,...]] en
        # boundaries = [[[ring], ...]].
        is_multi_surface = (lod3_geom.get("type") == "MultiSurface")
 
        for strip in roof_strips["strips"]:
            verts = strip["vertices"]
            vi = [_get_or_add_vertex(verts[j]) for j in range(4)]
 
            if is_multi_surface:
                boundaries.append([vi])          # MultiSurface: plat
            else:
                boundaries[0].append([vi])       # Solid: genest
 
            new_surf_idx = len(sem_surfaces)
            sem_surfaces.append({
                "type": "RoofSurface",
                "is_overhang": True,
                "overhang_methode": strip["methode"],
                "overhang_breedte": round(strip["breedte_h"], 4),
            })
 
            if is_multi_surface:
                sem_values.append(new_surf_idx)  # MultiSurface: plat
            else:
                sem_values[0].append(new_surf_idx)  # Solid: genest
 
            n_strips_added += 1
 
    n_verts_new = len(city_json["vertices"]) - n_verts_before
 
    print(f"[add_overhangs_to_cityjson] {n_strips_added} overhang-faces "
          f"toegevoegd als LOD3")
    print(f"  Vertices: {n_verts_before} bestaand, "
          f"{n_verts_new} nieuw toegevoegd, "
          f"{n_strips_added * 4 - n_verts_new} hergebruikt (gedeeld met dakvlak)")
    return city_json

def order_quad_corners(corners):
    """
    Sorteert 4 hoekpunten in een consistente volgorde (tegen de klok in)
    zodat ze een geldig quad vormen zonder kruising.
    """
    pts = np.array(corners)
    centroid = pts.mean(axis=0)
    
    # Normaalvector van het vlak bepalen
    v1 = pts[1] - pts[0]
    v2 = pts[2] - pts[0]
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)
    
    # Lokaal 2D assenstelsel op het vlak
    u_axis = pts[0] - centroid
    u_axis = u_axis / np.linalg.norm(u_axis)
    v_axis = np.cross(normal, u_axis)
    
    # Hoek per punt t.o.v. centroid
    angles = []
    for p in pts:
        d = p - centroid
        angle = np.arctan2(np.dot(d, v_axis), np.dot(d, u_axis))
        angles.append(angle)
    
    order = np.argsort(angles)
    return pts[order]

def generate_balkon_overhang_strips(balkon_data, mesh, strip_breedte=1.0,
                                     strip_lengte_marge=0.5):
    """
    Genereer per balkon een horizontale overhangstrip aan de onderkant.
    """
    mesh_center = np.asarray(mesh.vertices).mean(axis=0)
    
    results = []
    
    for entry in balkon_data:
        corners = order_quad_corners(entry['gemiddelde_hoekpunten'])
        
        z_vals = corners[:, 2]
        bottom_indices = np.argsort(z_vals)[:2]
        idx_sorted = sorted(bottom_indices, 
                           key=lambda i: np.argwhere(np.arange(4) == i)[0, 0])
        A = corners[idx_sorted[0]].copy()
        B = corners[idx_sorted[1]].copy()
        
        edge_vec = B - A
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < 1e-9:
            continue
        edge_dir = edge_vec / edge_len
        
        outward_h = np.array([-edge_dir[1], edge_dir[0], 0.0])
        balkon_center = corners.mean(axis=0)
        to_mesh = mesh_center - balkon_center
        if np.dot(outward_h[:2], to_mesh[:2]) > 0:
            outward_h = -outward_h
        
        z_strip = (A[2] + B[2]) / 2.0
        A_strip = A.copy(); A_strip[2] = z_strip
        B_strip = B.copy(); B_strip[2] = z_strip
        
        offset = strip_breedte * outward_h
        A_out = A_strip + offset
        B_out = B_strip + offset
        
        vertices = np.array([A_strip, B_strip, B_out, A_out])
        
        results.append({
            'raamgroep': entry['raamgroep'],
            'edge_start': A_strip,
            'edge_end': B_strip,
            'edge_len': edge_len,
            'edge_dir': edge_dir,
            'outward_dir': outward_h,
            'normaal': np.array([0.0, 0.0, 1.0]),
            'vertices': vertices,
            'strip_breedte': strip_breedte,
            'strip_lengte_marge': strip_lengte_marge,
        })
    
    print(f"[generate_balkon_overhang_strips] {len(results)} strips gegenereerd "
          f"(zoek: {strip_breedte:.1f}m breed, ±{strip_lengte_marge:.1f}m marge langs rand)")
    return results

def _estimate_overhang_width(points, edge_start, edge_end, outward_dir):
    """
    Bereken de HORIZONTALE breedte van de oversteek (90e percentiel).
    """
    if len(points) == 0:
        return 0.0

    edge_h = np.array([edge_end[0] - edge_start[0],
                        edge_end[1] - edge_start[1], 0.0])
    edge_h_len = np.linalg.norm(edge_h)
    if edge_h_len < 1e-9:
        return 0.0

    edge_h_n = edge_h / edge_h_len
    perp_h = np.array([-edge_h_n[1], edge_h_n[0], 0.0])

    if np.dot(perp_h, outward_dir) < 0:
        perp_h = -perp_h

    rel = points - edge_start
    projections = rel @ perp_h
    positive = projections[projections > 0]

    if len(positive) == 0:
        return 0.0

    return max(0.0, float(np.percentile(positive, 90)))

def _estimate_overhang_length(points, edge_start, edge_end):
    """
    Bereken de GEMETEN LENGTE van de oversteek langs de randrichting.
    
    Alle punten worden geprojecteerd op de richting A→B. De lengte is
    het verschil tussen het verste en dichtstbijzijnde punt langs die as.
    
    Gebruikt min/max (niet percentielen): de punten zijn al gefilterd
    door de strakke Z-filter uit pass 2 (±10cm rond mediaan Z), waardoor
    enkel vloerplaat-punten overblijven. Percentielen zouden dan onnodig
    goede randpunten wegknippen.
    
    Returns
    -------
    gemeten_lengte : float
        Totale lengte in meter (max - min projectie).
    start_offset : float
        Positie van het eerste punt t.o.v. A langs de rand.
        Negatief = punten steken voorbij A uit (naar links).
    end_offset : float
        Positie van het laatste punt t.o.v. B langs de rand.
        Positief = punten steken voorbij B uit (naar rechts).
    """
    if len(points) < 3:
        return 0.0, 0.0, 0.0
    
    edge_vec = edge_end - edge_start
    edge_len = np.linalg.norm(edge_vec)
    if edge_len < 1e-9:
        return 0.0, 0.0, 0.0
    
    edge_dir = edge_vec / edge_len
    rel = points - edge_start
    proj = rel[:, 0] * edge_dir[0] + rel[:, 1] * edge_dir[1]
    
    p_min = float(np.min(proj))
    p_max = float(np.max(proj))
    
    return p_max - p_min, p_min, p_max - edge_len

def classify_balkon_overhang_points(overhang_strips, balkon_crops,
                                     max_plane_dist_pass1=0.30,
                                     max_plane_dist_pass2=0.10):
    """
    Twee-pass classificatie van balkon-overhangpunten.

    Parameters
    ----------
    overhang_strips : list[dict]
        Output van generate_balkon_overhang_strips.
    balkon_crops : list[dict]
        Output van crop_balkon_points (in-memory punten).
    """
    crop_lookup = {c['raamgroep']: c['punten'] for c in balkon_crops}

    results = []

    for strip in overhang_strips:
        raamgroep = strip['raamgroep']

        xyz = crop_lookup.get(raamgroep)
        if xyz is None or len(xyz) == 0:
            print(f"  Balkon {raamgroep}: geen punten, skip")
            results.append(_empty_strip_result(raamgroep))
            continue

        A = strip['edge_start']
        B = strip['edge_end']
        edge_len = strip['edge_len']
        edge_dir_n = strip['edge_dir']
        outward = strip['outward_dir']
        breedte = strip['strip_breedte']
        marge = strip['strip_lengte_marge']

        # === PASS 1: Ruim Z-filter → mediaan Z vloerplaat ===
        z_strip = (A[2] + B[2]) / 2.0
        z_dists_1 = xyz[:, 2] - z_strip
        near_mask_1 = np.abs(z_dists_1) < max_plane_dist_pass1
        nearby_1 = xyz[near_mask_1]

        if len(nearby_1) == 0:
            results.append(_empty_strip_result(raamgroep))
            continue

        rel_1 = nearby_1 - A
        proj_edge_1 = rel_1[:, 0] * edge_dir_n[0] + rel_1[:, 1] * edge_dir_n[1]
        proj_out_1  = rel_1[:, 0] * outward[0]     + rel_1[:, 1] * outward[1]

        in_zone_1 = (
            (proj_edge_1 >= -marge) & (proj_edge_1 <= edge_len + marge) &
            (proj_out_1 >= -0.1)    & (proj_out_1 <= breedte + 0.1)
        )
        pass1_pts = nearby_1[in_zone_1]

        if len(pass1_pts) == 0:
            results.append(_empty_strip_result(raamgroep))
            continue

        gemeten_z = float(np.median(pass1_pts[:, 2]))

        # === PASS 2: Strak Z-filter rond gemeten_z ===
        z_dists_2 = xyz[:, 2] - gemeten_z
        near_mask_2 = np.abs(z_dists_2) < max_plane_dist_pass2
        nearby_2 = xyz[near_mask_2]

        if len(nearby_2) == 0:
            results.append(_empty_strip_result(raamgroep))
            continue

        rel_2 = nearby_2 - A
        proj_edge_2 = rel_2[:, 0] * edge_dir_n[0] + rel_2[:, 1] * edge_dir_n[1]
        proj_out_2  = rel_2[:, 0] * outward[0]     + rel_2[:, 1] * outward[1]

        in_zone_2 = (
            (proj_edge_2 >= -marge) & (proj_edge_2 <= edge_len + marge) &
            (proj_out_2 >= -0.1)    & (proj_out_2 <= breedte + 0.1)
        )
        pass2_pts = nearby_2[in_zone_2]
        pass2_z_dists = z_dists_2[near_mask_2][in_zone_2]

        if len(pass2_pts) == 0:
            results.append(_empty_strip_result(raamgroep))
            continue

        rmse = float(np.sqrt(np.mean(pass2_z_dists ** 2)))
        gem_afstand = float(np.mean(pass2_z_dists))

        gemeten_breedte = _estimate_overhang_width(pass2_pts, A, B, outward)
        gemeten_lengte, start_offset, end_offset = _estimate_overhang_length(
            pass2_pts, A, B)

        gemeten_area = gemeten_lengte * gemeten_breedte if gemeten_breedte > 0 else 0.0
        dichtheid = len(pass2_pts) / gemeten_area if gemeten_area > 1e-9 else 0.0
        coverage = min(gemeten_lengte / edge_len, 1.5) if edge_len > 1e-9 else 0.0

        results.append({
            'raamgroep': raamgroep,
            'punten_op_strip': pass2_pts,
            'n_punten': len(pass2_pts),
            'n_punten_pass1': len(pass1_pts),
            'rmse': rmse,
            'gemeten_breedte': gemeten_breedte,
            'gemeten_lengte': gemeten_lengte,
            'gemeten_z': gemeten_z,
            'start_offset': start_offset,
            'end_offset': end_offset,
            'dichtheid': dichtheid,
            'coverage': coverage,
            'gemiddelde_afstand': gem_afstand,
        })

        print(f"  Balkon {raamgroep}: pass1={len(pass1_pts)} → "
              f"pass2={len(pass2_pts)} punten (Z={gemeten_z:.3f}m), "
              f"RMSE={rmse:.4f}m, "
              f"breedte={gemeten_breedte:.3f}m, lengte={gemeten_lengte:.3f}m")

    return results

def _empty_strip_result(raamgroep):
    return {
        'raamgroep': raamgroep,
        'punten_op_strip': np.empty((0, 3)),
        'n_punten': 0,
        'n_punten_pass1': 0,
        'rmse': float('inf'),
        'gemeten_breedte': 0.0,
        'gemeten_lengte': 0.0,
        'gemeten_z': 0.0,
        'start_offset': 0.0,
        'end_offset': 0.0,
        'dichtheid': 0.0,
        'coverage': 0.0,
        'gemiddelde_afstand': 0.0,
    }

def print_balcony_summary(overhang_strips, classification_result):
    """
    Print een overzichtelijke samenvatting van de balkon-overhanganalyse.
    """
    print(f"\n{'='*70}")
    print(f"BALKON OVERHANG ANALYSE")
    print(f"{'='*70}")
    
    for strip, cls in zip(overhang_strips, classification_result):
        rg = strip['raamgroep']
        has_points = cls['n_punten'] > 0
        
        print(f"\n  Balkon {rg}:")
        print(f"    Zoekzone: {strip['edge_len']:.2f}m rand × "
              f"{strip['strip_breedte']:.2f}m zoekbreedte "
              f"(±{strip['strip_lengte_marge']:.1f}m marge)")
        
        if has_points:
            print(f"    Pass 1 → 2:      {cls['n_punten_pass1']} → "
                  f"{cls['n_punten']} punten")
            print(f"    Gemeten Z:       {cls['gemeten_z']:.3f}m")
            print(f"    RMSE:            {cls['rmse']:.4f}m")
            print(f"    Dichtheid:       {cls['dichtheid']:.1f}/m²")
            print(f"    Gemeten breedte: {cls['gemeten_breedte']:.3f}m")
            print(f"    Gemeten lengte:  {cls['gemeten_lengte']:.3f}m "
                  f"(rand={strip['edge_len']:.2f}m, "
                  f"links={cls['start_offset']:+.3f}m, "
                  f"rechts={cls['end_offset']:+.3f}m)")
            print(f"    Coverage:        {cls['coverage']:.0%}")
        else:
            print(f"    Punten:          geen vloerplaat-punten gevonden")
    
    print(f"\n{'='*70}")

def generate_fitted_balkon_obj(overhang_strips, classification_result, 
                                output_path, min_punten=5):
    """
    Genereer een .obj met per balkon een vlak aangepast aan de werkelijke
    afmetingen uit de puntenwolk (lengte, breedte, Z).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fitted_balkons = []
    
    for strip, cls in zip(overhang_strips, classification_result):
        if cls['n_punten'] < min_punten:
            continue
        if cls['gemeten_breedte'] < 0.05 or cls['gemeten_lengte'] < 0.1:
            continue
        
        A = strip['edge_start']
        edge_dir = strip['edge_dir']
        outward = strip['outward_dir']
        z_fitted = cls['gemeten_z']
        
        A_fitted = A + cls['start_offset'] * edge_dir
        A_fitted[2] = z_fitted
        
        B_fitted = A + (strip['edge_len'] + cls['end_offset']) * edge_dir
        B_fitted[2] = z_fitted
        
        A_out = A_fitted + cls['gemeten_breedte'] * outward
        A_out[2] = z_fitted
        
        B_out = B_fitted + cls['gemeten_breedte'] * outward
        B_out[2] = z_fitted
        
        vertices = np.array([A_fitted, B_fitted, B_out, A_out])
        
        fitted_balkons.append({
            'raamgroep': strip['raamgroep'],
            'vertices': vertices,
            'gemeten_lengte': cls['gemeten_lengte'],
            'gemeten_breedte': cls['gemeten_breedte'],
            'gemeten_z': z_fitted,
            'rmse': cls['rmse'],
            'n_punten': cls['n_punten'],
        })
    
    with open(output_path, "w") as f:
        f.write(f"# Gefitte balkonvlakken op basis van puntenwolk\n")
        f.write(f"# Aantal balkons: {len(fitted_balkons)}\n\n")
        
        vert_offset = 0
        for balkon in fitted_balkons:
            rg = balkon['raamgroep']
            f.write(f"# Balkon {rg}: "
                    f"{balkon['gemeten_lengte']:.3f}m × "
                    f"{balkon['gemeten_breedte']:.3f}m, "
                    f"Z={balkon['gemeten_z']:.3f}m, "
                    f"RMSE={balkon['rmse']:.4f}m\n")
            
            for v in balkon['vertices']:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            
            f.write(f"vn 0.000000 0.000000 1.000000\n")
            
            ni = vert_offset // 4 + 1
            idx = [f"{vert_offset + j + 1}//{ni}" for j in range(4)]
            f.write(f"f {' '.join(idx)}\n\n")
            vert_offset += 4
    
    print(f"[generate_fitted_balkon_obj] {len(fitted_balkons)} balkons "
          f"→ {output_path}")
    
    for b in fitted_balkons:
        print(f"  Balkon {b['raamgroep']}: "
              f"{b['gemeten_lengte']:.3f}m × {b['gemeten_breedte']:.3f}m, "
              f"Z={b['gemeten_z']:.3f}m (RMSE={b['rmse']:.4f}m, "
              f"{b['n_punten']} punten)")
    
    return fitted_balkons

def export_overhang_strips_to_obj(overhang_strips, output_path):
    """
    Exporteer de zoekzone-strips als .obj (voor debugging/visualisatie).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write(f"# Balkon overhangstrips (zoekzone)\n")
        f.write(f"# Aantal strips: {len(overhang_strips)}\n\n")
        
        vert_offset = 0
        for strip in overhang_strips:
            f.write(f"# Balkon {strip['raamgroep']} "
                    f"(breedte={strip['strip_breedte']:.3f}m)\n")
            for v in strip['vertices']:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            n = strip['normaal']
            f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
            ni = vert_offset // 4 + 1
            idx = [f"{vert_offset + j + 1}//{ni}" for j in range(4)]
            f.write(f"f {' '.join(idx)}\n\n")
            vert_offset += 4
    
    print(f"[export_overhang_strips_to_obj] {len(overhang_strips)} strips "
          f"→ {output_path}")

def create_balkon_obb(corners, margin_planar=0.5, margin_normal=0.5):
    """
    Maakt een OrientedBoundingBox rond een balkonvlak.
    """
    pts = np.array(corners, dtype=float)
    
    edge_01 = pts[1] - pts[0]
    edge_03 = pts[3] - pts[0]
    edge_12 = pts[2] - pts[1]
    
    u_vec = (edge_01 + (pts[2] - pts[3])) / 2.0
    v_vec = (edge_03 + edge_12) / 2.0
    
    u_len = np.linalg.norm(u_vec)
    v_len = np.linalg.norm(v_vec)
    u_axis = u_vec / u_len
    v_axis = v_vec / v_len
    
    n_axis = np.cross(u_axis, v_axis)
    n_axis = n_axis / np.linalg.norm(n_axis)
    
    R = np.column_stack([u_axis, v_axis, n_axis])
    center = pts.mean(axis=0)
    
    extent = np.array([
        u_len + 2 * margin_planar,
        v_len + 2 * margin_planar,
        2 * margin_normal
    ])
    
    obb = o3d.geometry.OrientedBoundingBox(center, R, extent)
    return obb, R, center, extent

def punten_in_obb(xyz, R, center, extent):
    """
    Geeft een boolean mask terug: True voor punten die binnen de OBB vallen.
    """
    lokaal = (xyz - center) @ R  # projecteer naar lokaal assenstelsel
    half = extent / 2.0
    mask = np.all(np.abs(lokaal) <= half, axis=1)
    return mask

def crop_balkon_points(balkon_data, points, margin_planar=0.5, margin_normal=0.5):
    """
    Snijdt per balkon de relevante punten uit de puntenwolk (in-memory).

    Parameters
    ----------
    balkon_data : list[dict] of list[np.ndarray]
        Accepteert:
        - list[dict] met 'gemiddelde_hoekpunten' (output filter_raamgroepen_by_angle)
        - list[dict] met 'vertices' (output generate_balkon_overhang_strips)
        - list[np.ndarray] van (4,3) hoekpunten (overhang_strips_vertices)
    points : np.array(N, 3)
        Volledige puntenwolk in wereldcoördinaten.
    margin_planar : float
        Uitbreiding langs het vlak in meter.
    margin_normal : float
        Uitbreiding loodrecht op het vlak in meter.

    Returns
    -------
    list[dict]
        Per balkon: {'raamgroep': str, 'punten': np.array(K, 3), 'n_punten': int}
    """
    resultaten = []

    for i, entry in enumerate(balkon_data):
        # Bepaal hoekpunten en raamgroep op basis van input-type
        if isinstance(entry, dict):
            if 'gemiddelde_hoekpunten' in entry:
                raw_corners = entry['gemiddelde_hoekpunten']
            elif 'vertices' in entry:
                raw_corners = entry['vertices']
            else:
                raise KeyError(f"Entry {i} heeft geen 'gemiddelde_hoekpunten' of 'vertices' key")
            raamgroep = entry.get('raamgroep', f'balkon_{i}')
        else:
            # Kale numpy array
            raw_corners = entry
            raamgroep = f'balkon_{i}'

        corners = order_quad_corners(raw_corners)
        _, R, center, extent = create_balkon_obb(corners, margin_planar, margin_normal)

        mask = punten_in_obb(points, R, center, extent)
        crop_pts = points[mask]

        resultaten.append({
            'raamgroep': raamgroep,
            'punten': crop_pts,
            'n_punten': len(crop_pts),
        })

    total = sum(r['n_punten'] for r in resultaten)
    print(f"[crop_balkon_points] {len(resultaten)} balkons, {total} punten totaal")
    return resultaten

def create_virtual_plane_mesh(overhang_strips, overhang_classificatie, 
                               plane_height=8.0, plane_width_extra=2.0):
    """
    Maak een Open3D TriangleMesh van virtuele vlakken, één per balkon.
    
    Elk vlak is parallel aan de gevel maar verschoven met gemeten_breedte
    langs de outward-richting. De vlakken zijn groot genoeg zodat alle 
    camera-rays ze raken.
    
    Parameters
    ----------
    overhang_strips : list[dict]
        Output van generate_balkon_overhang_strips.
    overhang_classificatie : list[dict]
        Output van classify_balkon_overhang_points (met gemeten_breedte).
    plane_height : float
        Hoogte van elk virtueel vlak in meter (boven en onder de onderrand).
    plane_width_extra : float
        Extra breedte links/rechts voorbij de balkonrand in meter.
    
    Returns
    -------
    o3d.geometry.TriangleMesh
        Gecombineerde mesh van alle virtuele vlakken.
    """
    all_vertices = []
    all_triangles = []
    vert_offset = 0
    
    for strip, cls in zip(overhang_strips, overhang_classificatie):
        breedte = cls.get('gemeten_breedte', 0)
        if breedte < 0.05 or cls['n_punten'] < 3:
            continue
        
        A = strip['edge_start']
        B = strip['edge_end']
        edge_dir = strip['edge_dir']
        outward = strip['outward_dir']
        edge_len = strip['edge_len']
        
        # Centerpunt van de onderrand, verschoven naar virtueel vlak
        center = (A + B) / 2.0 + breedte * outward
        
        # Quad hoekpunten: breed langs gevel, hoog langs Z
        half_w = edge_len / 2.0 + plane_width_extra
        
        # 4 hoekpunten: links-onder, rechts-onder, rechts-boven, links-boven
        v0 = center - half_w * edge_dir + np.array([0, 0, -1.0])              # linksonder
        v1 = center + half_w * edge_dir + np.array([0, 0, -1.0])              # rechtsonder
        v2 = center + half_w * edge_dir + np.array([0, 0, plane_height])      # rechtsboven
        v3 = center - half_w * edge_dir + np.array([0, 0, plane_height])      # linksboven
        
        verts = np.array([v0, v1, v2, v3])
        tris = np.array([[0, 1, 2], [0, 2, 3]]) + vert_offset
        
        all_vertices.append(verts)
        all_triangles.append(tris)
        vert_offset += 4
        
        print(f"  Balkon {strip['raamgroep']}: virtueel vlak op {breedte:.3f}m offset, "
              f"{2*half_w:.1f}m breed × {plane_height+1:.1f}m hoog")
    
    if not all_vertices:
        print("WAARSCHUWING: geen virtuele vlakken aangemaakt")
        return o3d.geometry.TriangleMesh()
    
    mesh_virtual = o3d.geometry.TriangleMesh()
    mesh_virtual.vertices = o3d.utility.Vector3dVector(np.vstack(all_vertices))
    mesh_virtual.triangles = o3d.utility.Vector3iVector(np.vstack(all_triangles))
    mesh_virtual.compute_vertex_normals()
    
    print(f"\n[create_virtual_plane_mesh] {len(all_vertices)} vlakken, "
          f"{vert_offset} vertices, {len(np.vstack(all_triangles))} driehoeken")
    return mesh_virtual

def combine_virtual_and_overhang(grouped_virtual_filtered, overhang_strips, 
                                  overhang_classificatie, min_punten=5):
    """
    Combineer de bovenste hoekpunten van de hertracering (virtueel vlak)
    met de onderrand uit de overhang-analyse tot een volledig balkonvlak.
    
    Returns
    -------
    combined_balkons : list[dict]
    obj_string : str
    """
    strip_lookup = {s['raamgroep']: s for s in overhang_strips}
    cls_lookup = {c['raamgroep']: c for c in overhang_classificatie}
    
    combined_balkons = []
    
    for entry in grouped_virtual_filtered:
        rg = entry['raamgroep']
        corners = order_quad_corners(entry['gemiddelde_hoekpunten'])
        
        strip = strip_lookup.get(rg)
        cls = cls_lookup.get(rg)
        if strip is None or cls is None or cls['n_punten'] < min_punten:
            continue
        
        z_vals = corners[:, 2]
        top_idx = np.argsort(z_vals)[-2:]
        top_corners = corners[top_idx]
        
        A_bottom = strip['edge_start'] + cls['start_offset'] * strip['edge_dir']
        A_bottom[2] = cls['gemeten_z']
        
        B_bottom = strip['edge_start'] + (strip['edge_len'] + cls['end_offset']) * strip['edge_dir']
        B_bottom[2] = cls['gemeten_z']
        
        outward = strip['outward_dir']
        breedte = cls['gemeten_breedte']
        A_bottom_offset = A_bottom + breedte * outward
        B_bottom_offset = B_bottom + breedte * outward
        
        edge_dir = strip['edge_dir']
        proj_top = [(np.dot(c - strip['edge_start'], edge_dir), c) for c in top_corners]
        proj_top.sort(key=lambda x: x[0])
        top_left = proj_top[0][1]
        top_right = proj_top[1][1]
        
        proj_A = np.dot(A_bottom_offset - strip['edge_start'], edge_dir)
        proj_B = np.dot(B_bottom_offset - strip['edge_start'], edge_dir)
        if proj_A <= proj_B:
            bottom_left, bottom_right = A_bottom_offset, B_bottom_offset
        else:
            bottom_left, bottom_right = B_bottom_offset, A_bottom_offset
        
        vertices = np.array([bottom_left, bottom_right, top_right, top_left])
        hoogte = top_corners[:, 2].mean() - cls['gemeten_z']
        
        combined_balkons.append({
            'raamgroep': rg,
            'vertices': vertices,
            'hoogte': hoogte,
            'breedte': breedte,
            'gemeten_z': cls['gemeten_z'],
            'top_z': top_corners[:, 2].mean(),
            'rmse': cls['rmse'],
            'n_punten': cls['n_punten'],
        })
        
        print(f"  Balkon {rg}: hoogte={hoogte:.3f}m, breedte={breedte:.3f}m, "
              f"top Z={top_corners[:, 2].mean():.3f}m, bottom Z={cls['gemeten_z']:.3f}m")
    
    # OBJ-string opbouwen
    lines = [
        f"# Gecombineerde balkonvlakken (virtual top + overhang bottom)",
        f"# Aantal balkons: {len(combined_balkons)}",
        "",
    ]
    
    vert_offset = 0
    for balkon in combined_balkons:
        rg = balkon['raamgroep']
        lines.append(f"# Balkon {rg}: "
                     f"hoogte={balkon['hoogte']:.3f}m, "
                     f"breedte={balkon['breedte']:.3f}m, "
                     f"RMSE={balkon['rmse']:.4f}m")
        
        for v in balkon['vertices']:
            lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
        
        e1 = balkon['vertices'][1] - balkon['vertices'][0]
        e2 = balkon['vertices'][3] - balkon['vertices'][0]
        n = np.cross(e1, e2)
        n_len = np.linalg.norm(n)
        if n_len > 1e-12:
            n = n / n_len
        lines.append(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}")
        
        ni = vert_offset // 4 + 1
        idx = [f"{vert_offset + j + 1}//{ni}" for j in range(4)]
        lines.append(f"f {' '.join(idx)}")
        lines.append("")
        vert_offset += 4
    
    obj_string = "\n".join(lines)
    
    print(f"\n[combine_virtual_and_overhang] {len(combined_balkons)} balkons gegenereerd")
    return combined_balkons, obj_string

def create_balkon_compleet(overhang_strips, overhang_classificatie, 
                           grouped_virtual_filtered, min_punten=5):
    """
    Maak een gesloten balkon per detectie: opstand (voorkant) + 
    zijwanden links/rechts + vloerplaat. De achterkant (muur) is open.
    
    Returns
    -------
    balkons : list[dict]
    obj_string : str
    """
    strip_lookup = {s['raamgroep']: s for s in overhang_strips}
    cls_lookup = {c['raamgroep']: c for c in overhang_classificatie}
    
    balkons = []
    
    for entry in grouped_virtual_filtered:
        rg = entry['raamgroep']
        strip = strip_lookup.get(rg)
        cls = cls_lookup.get(rg)
        if strip is None or cls is None or cls['n_punten'] < min_punten:
            continue
        
        breedte = cls['gemeten_breedte']
        if breedte < 0.05:
            continue
        
        corners_virtual = order_quad_corners(entry['gemiddelde_hoekpunten'])
        z_top = np.sort(corners_virtual[:, 2])[-2:].mean()
        z_bottom = cls['gemeten_z']
        hoogte = z_top - z_bottom
        
        if hoogte < 0.1:
            continue
        
        outward = strip['outward_dir']
        edge_dir = strip['edge_dir']
        
        # 4 punten VOOR
        A_front_bottom = strip['edge_start'] + cls['start_offset'] * edge_dir
        A_front_bottom[2] = z_bottom
        A_front_bottom = A_front_bottom + breedte * outward
        
        B_front_bottom = strip['edge_start'] + (strip['edge_len'] + cls['end_offset']) * edge_dir
        B_front_bottom[2] = z_bottom
        B_front_bottom = B_front_bottom + breedte * outward
        
        A_front_top = A_front_bottom.copy()
        A_front_top[2] = z_bottom + hoogte
        
        B_front_top = B_front_bottom.copy()
        B_front_top[2] = z_bottom + hoogte
        
        # 4 punten ACHTER
        A_wall_bottom = A_front_bottom - breedte * outward
        A_wall_top    = A_front_top    - breedte * outward
        B_wall_bottom = B_front_bottom - breedte * outward
        B_wall_top    = B_front_top    - breedte * outward
        
        # 4 vlakken
        front = np.array([A_front_bottom, B_front_bottom, B_front_top, A_front_top])
        left  = np.array([A_wall_bottom, A_front_bottom, A_front_top, A_wall_top])
        right = np.array([B_front_bottom, B_wall_bottom, B_wall_top, B_front_top])
        floor = np.array([A_wall_bottom, B_wall_bottom, B_front_bottom, A_front_bottom])
        
        balkons.append({
            'raamgroep': rg,
            'faces': [front, left, right, floor],
            'hoogte': hoogte,
            'breedte': breedte,
            'gemeten_lengte': cls['gemeten_lengte'],
            'z_bottom': z_bottom,
            'z_top': z_bottom + hoogte,
        })
        
        print(f"  Balkon {rg}: {cls['gemeten_lengte']:.2f}m × {breedte:.2f}m × {hoogte:.2f}m hoog")
    
    # OBJ-string opbouwen
    lines = [
        f"# Complete balkongeometrie (opstand + zijwanden + vloer)",
        f"# Aantal balkons: {len(balkons)}",
        "",
    ]
    
    vert_offset = 0
    for balkon in balkons:
        rg = balkon['raamgroep']
        lines.append(f"# Balkon {rg}: "
                     f"{balkon['gemeten_lengte']:.2f}m × "
                     f"{balkon['breedte']:.2f}m × "
                     f"{balkon['hoogte']:.2f}m")
        
        face_names = ['voorkant', 'links', 'rechts', 'vloer']
        for face, name in zip(balkon['faces'], face_names):
            lines.append(f"# {name}")
            for v in face:
                lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
            
            e1 = face[1] - face[0]
            e2 = face[3] - face[0]
            n = np.cross(e1, e2)
            n_len = np.linalg.norm(n)
            if n_len > 1e-12:
                n = n / n_len
            lines.append(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}")
            
            ni = vert_offset // 4 + 1
            idx = [f"{vert_offset + j + 1}//{ni}" for j in range(4)]
            lines.append(f"f {' '.join(idx)}")
            vert_offset += 4
        
        lines.append("")
    
    obj_string = "\n".join(lines)
    
    print(f"\n[create_balkon_compleet] {len(balkons)} balkons "
          f"({len(balkons)*4} vlakken) gegenereerd")
    return balkons

def add_balkons_to_cityjson(cj, balkons_compleet):
    """
    Voeg balkongeometrie toe aan een CityJSON dict (in-memory) als LOD3.
    
    Per balkon worden 4 vlakken toegevoegd:
    - Vloerplaat → GroundSurface (is_balkon=True)
    - Opstand (voorkant) + zijwanden → WallSurface (is_balkon=True)
    
    Vertices worden gededupliceerd: gedeelde punten worden hergebruikt.
    
    Parameters
    ----------
    cj : dict
        In-memory CityJSON dict (bv. LOD3_window_and_overhang).
        Wordt in-place aangepast.
    balkons_compleet : list[dict]
        Output van create_balkon_compleet. Elke entry heeft:
        - 'faces': [front, left, right, floor], elk np.array(4,3)
        - 'raamgroep', 'hoogte', 'breedte', etc.
    
    Returns
    -------
    cj : dict
        Het aangepaste CityJSON dict (zelfde object als input).
    """
    scale = cj["transform"]["scale"]
    translate = cj["transform"]["translate"]
    
    # --- Vertex-lookup voor deduplicatie ---
    vertex_map = {}
    for idx, v in enumerate(cj["vertices"]):
        key = tuple(v)
        vertex_map[key] = idx
    
    def _get_or_add_vertex(real_xyz):
        ix = round((real_xyz[0] - translate[0]) / scale[0])
        iy = round((real_xyz[1] - translate[1]) / scale[1])
        iz = round((real_xyz[2] - translate[2]) / scale[2])
        key = (ix, iy, iz)
        
        if key in vertex_map:
            return vertex_map[key]
        
        new_idx = len(cj["vertices"])
        cj["vertices"].append([ix, iy, iz])
        vertex_map[key] = new_idx
        return new_idx
    
    # --- Zoek het BuildingPart met LOD3 (of maak aan vanuit LOD 2.2) ---
    n_verts_before = len(cj["vertices"])
    n_faces_added = 0
    balkon_vloer_idx = None
    balkon_wand_idx = None
    
    for obj_id, co in cj["CityObjects"].items():
        lod22_geom = None
        lod3_geom = None
        for geom in co.get("geometry", []):
            if geom.get("lod") == "2.2":
                lod22_geom = geom
            if geom.get("lod") == "3":
                lod3_geom = geom
        
        if lod22_geom is None and lod3_geom is None:
            continue
        
        if lod3_geom is None:
            lod3_geom = copy.deepcopy(lod22_geom)
            lod3_geom["lod"] = "3"
            if lod3_geom.get("type") == "Solid":
                lod3_geom["type"] = "MultiSurface"
                lod3_geom["boundaries"] = lod3_geom["boundaries"][0]
                if "values" in lod3_geom.get("semantics", {}):
                    vals = lod3_geom["semantics"]["values"]
                    if isinstance(vals, list) and len(vals) > 0 and isinstance(vals[0], list):
                        lod3_geom["semantics"]["values"] = vals[0]
            co["geometry"].append(lod3_geom)
            print(f"  LOD3 aangemaakt vanuit LOD 2.2 voor {obj_id}")
        
        boundaries = lod3_geom["boundaries"]
        sem_surfaces = lod3_geom["semantics"]["surfaces"]
        sem_values = lod3_geom["semantics"]["values"]
        
        # --- Semantische surface types toevoegen (eenmalig) ---
        balkon_vloer_idx = None
        balkon_wand_idx = None
        
        for si, surf in enumerate(sem_surfaces):
            if surf.get("is_balkon") and surf.get("type") == "GroundSurface":
                balkon_vloer_idx = si
            if surf.get("is_balkon") and surf.get("type") == "WallSurface":
                balkon_wand_idx = si
        
        if balkon_vloer_idx is None:
            balkon_vloer_idx = len(sem_surfaces)
            sem_surfaces.append({
                "type": "GroundSurface",
                "is_balkon": True,
            })
        
        if balkon_wand_idx is None:
            balkon_wand_idx = len(sem_surfaces)
            sem_surfaces.append({
                "type": "WallSurface",
                "is_balkon": True,
            })
        
        # --- Per balkon: faces toevoegen ---
        for balkon in balkons_compleet:
            front, left, right, floor = balkon['faces']
            
            face_data = [
                (front, balkon_wand_idx, "voorkant"),
                (left,  balkon_wand_idx, "links"),
                (right, balkon_wand_idx, "rechts"),
                (floor, balkon_vloer_idx, "vloer"),
            ]
            
            for face_verts, sem_idx, naam in face_data:
                vi = [_get_or_add_vertex(face_verts[j]) for j in range(4)]
                boundaries.append([vi])
                sem_values.append(sem_idx)
                n_faces_added += 1
        
        break
    
    n_verts_new = len(cj["vertices"]) - n_verts_before
    n_verts_reused = len(balkons_compleet) * 4 * 4 - n_verts_new
    
    print(f"[add_balkons_to_cityjson] {n_faces_added} balkon-faces toegevoegd aan LOD3")
    print(f"  Vertices: {n_verts_before} bestaand, "
          f"{n_verts_new} nieuw, {n_verts_reused} hergebruikt")
    if balkon_vloer_idx is not None:
        print(f"  Semantiek: vloer=surface[{balkon_vloer_idx}], "
              f"wand=surface[{balkon_wand_idx}]")
    
    return cj
