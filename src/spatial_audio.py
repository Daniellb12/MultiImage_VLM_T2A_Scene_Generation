"""Spatial Audio Positioning Module"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import json

try:
    import open3d as o3d
except ImportError:
    raise ImportError("Please install open3d: pip install open3d")

logger = logging.getLogger(__name__)


class SpatialAudioPositioner:
    """Position audio sources in 3D space"""
    
    def __init__(
        self,
        coordinate_system: str = "right_handed",
        intensity_falloff: str = "inverse_square",
        default_intensity: float = 0.8
    ):
        """
        Initialize spatial audio positioner
        
        Args:
            coordinate_system: Coordinate system (right_handed, left_handed)
            intensity_falloff: Falloff type (inverse_square, linear, constant)
            default_intensity: Default intensity for audio sources
        """
        self.coordinate_system = coordinate_system
        self.intensity_falloff = intensity_falloff
        self.default_intensity = default_intensity
        
        logger.info(f"Initialized SpatialAudioPositioner: {coordinate_system}, falloff={intensity_falloff}")
    
    def create_audio_manifest(
        self,
        objects_3d: List[Dict[str, Any]],
        camera_data: Optional[Dict[str, np.ndarray]] = None,
        output_path: str = "data/output/audio_manifest.json"
    ) -> Dict[str, Any]:
        """
        Create spatial audio manifest with 3D positions
        
        Args:
            objects_3d: List of objects with 3D positions and audio files
            camera_data: Optional camera data for listener positions
            output_path: Path to save manifest
        
        Returns:
            Audio manifest dictionary
        """
        logger.info(f"Creating audio manifest for {len(objects_3d)} objects")
        
        # Extract audio sources
        audio_sources = []
        for obj in objects_3d:
            if obj.get('audio_file') and os.path.exists(obj['audio_file']):
                source = {
                    'id': obj['id'],
                    'label': obj['label'],
                    'category': obj.get('category', 'unknown'),
                    'audio_file': obj['audio_file'],
                    'position_3d': obj.get('position_3d', [0.0, 0.0, float(obj.get('depth', 5.0))]),
                    'intensity': self._compute_intensity(obj),
                    'confidence': obj.get('confidence', 0.0),
                    'depth': obj.get('depth', 0.0)
                }
                audio_sources.append(source)
        
        # Extract listener positions from camera data.
        # COLMAP extrinsic: X_cam = R @ X_world + t  →  world centre = -R^T @ t
        # We stored camera_centers in reconstruction.py; fall back to computing
        # it here if that key is missing (e.g. from an older run).
        listener_positions = []
        if camera_data and 'extrinsics' in camera_data:
            centers = camera_data.get('camera_centers', {})
            for camera_id, extrinsic in camera_data['extrinsics'].items():
                R = extrinsic[:3, :3]
                t = extrinsic[:3, 3]
                # World-space camera centre
                if camera_id in centers:
                    position = np.asarray(centers[camera_id]).tolist()
                else:
                    position = (-R.T @ t).tolist()

                # Forward direction in world space (third column of R^T = -R^T[:,2])
                forward = (-R.T[:, 2]).tolist()

                listener_positions.append({
                    'camera_id': int(camera_id) if isinstance(camera_id, (int, np.integer)) else camera_id,
                    'position': position,
                    'forward': forward,
                    'image_name': camera_data.get('image_names', {}).get(camera_id, f"camera_{camera_id}"),
                })
        
        # Create manifest
        manifest = {
            'metadata': {
                'coordinate_system': self.coordinate_system,
                'intensity_falloff': self.intensity_falloff,
                'num_sources': len(audio_sources),
                'num_listeners': len(listener_positions)
            },
            'audio_sources': audio_sources,
            'listener_positions': listener_positions
        }
        
        # Save manifest
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        logger.info(f"Audio manifest saved to: {output_path}")
        logger.info(f"- {len(audio_sources)} audio sources")
        logger.info(f"- {len(listener_positions)} listener positions")
        
        return manifest
    
    def _compute_intensity(self, obj: Dict[str, Any]) -> float:
        """
        Compute audio intensity for an object
        
        Args:
            obj: Object dictionary
        
        Returns:
            Intensity value [0, 1]
        """
        from src.utils import estimate_audio_intensity
        
        # Use distance from origin as proxy
        position = np.array(obj['position_3d'])
        distance = np.linalg.norm(position)
        
        intensity = estimate_audio_intensity(
            obj.get('label', 'unknown'),
            distance,
            self.intensity_falloff
        )
        
        return float(intensity)
    
    def compute_spatial_parameters(
        self,
        source_position: np.ndarray,
        listener_position: np.ndarray,
        listener_forward: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Compute spatial audio parameters for a source relative to listener
        
        Args:
            source_position: 3D position of audio source
            listener_position: 3D position of listener
            listener_forward: Forward direction vector of listener
        
        Returns:
            Dictionary with spatial parameters
        """
        # Direction from listener to source
        direction = source_position - listener_position
        distance = np.linalg.norm(direction)
        
        if distance > 1e-6:
            direction_normalized = direction / distance
        else:
            direction_normalized = np.array([0, 0, 1])
        
        # Compute azimuth and elevation
        azimuth = np.arctan2(direction_normalized[0], direction_normalized[2])
        elevation = np.arcsin(np.clip(direction_normalized[1], -1, 1))
        
        # Compute relative angle if forward direction provided
        if listener_forward is not None:
            listener_forward_normalized = listener_forward / (np.linalg.norm(listener_forward) + 1e-6)
            
            # Angle between forward and source direction (projected to XZ plane)
            forward_xz = np.array([listener_forward_normalized[0], 0, listener_forward_normalized[2]])
            direction_xz = np.array([direction_normalized[0], 0, direction_normalized[2]])
            
            forward_xz_norm = forward_xz / (np.linalg.norm(forward_xz) + 1e-6)
            direction_xz_norm = direction_xz / (np.linalg.norm(direction_xz) + 1e-6)
            
            cos_angle = np.dot(forward_xz_norm, direction_xz_norm)
            relative_azimuth = np.arccos(np.clip(cos_angle, -1, 1))
            
            # Determine left/right
            cross = np.cross(forward_xz_norm, direction_xz_norm)
            if cross[1] < 0:
                relative_azimuth = -relative_azimuth
        else:
            relative_azimuth = azimuth
        
        # Compute attenuation based on distance
        if self.intensity_falloff == "inverse_square":
            attenuation = 1.0 / (1.0 + distance ** 2)
        elif self.intensity_falloff == "linear":
            attenuation = max(0.0, 1.0 - distance / 10.0)
        else:  # constant
            attenuation = 1.0
        
        return {
            'distance': float(distance),
            'azimuth': float(np.degrees(azimuth)),
            'elevation': float(np.degrees(elevation)),
            'relative_azimuth': float(np.degrees(relative_azimuth)),
            'attenuation': float(attenuation),
            'direction': direction_normalized.tolist()
        }
    
    def visualize_audio_scene(
        self,
        manifest: Dict[str, Any],
        point_cloud_path: Optional[str] = None,
        output_path: str = "data/output/audio_visualization.ply"
    ) -> str:
        """
        Create 3D visualization of audio sources and listeners
        
        Args:
            manifest: Audio manifest dictionary
            point_cloud_path: Optional path to existing point cloud
            output_path: Path to save visualization
        
        Returns:
            Path to visualization file
        """
        logger.info("Creating audio scene visualization")
        
        geometries = []
        
        # Load existing point cloud if provided
        if point_cloud_path and os.path.exists(point_cloud_path):
            try:
                pcd = o3d.io.read_point_cloud(point_cloud_path)
                geometries.append(pcd)
                logger.info(f"Loaded point cloud with {len(pcd.points)} points")
            except Exception as e:
                logger.warning(f"Could not load point cloud: {str(e)}")
        
        # Add audio sources as colored spheres
        for source in manifest['audio_sources']:
            position = source['position_3d']
            intensity = source.get('intensity', 0.5)
            
            # Create sphere for audio source
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
            sphere.translate(position)
            
            # Color based on intensity (red = loud, blue = quiet)
            color = np.array([intensity, 0.2, 1.0 - intensity])
            sphere.paint_uniform_color(color)
            
            geometries.append(sphere)
        
        logger.info(f"Added {len(manifest['audio_sources'])} audio sources")
        
        # Add listener positions as cameras/cones
        for listener in manifest['listener_positions']:
            position = listener['position']
            
            # Create cone for listener (camera)
            cone = o3d.geometry.TriangleMesh.create_cone(radius=0.05, height=0.15)
            
            # Orient cone in forward direction
            if 'forward' in listener:
                forward = np.array(listener['forward'])
                # Rotate cone to face forward direction
                # (default cone points up in +Y, we need to rotate to forward)
                default_up = np.array([0, 1, 0])
                forward_normalized = forward / (np.linalg.norm(forward) + 1e-6)
                
                # Compute rotation matrix
                axis = np.cross(default_up, forward_normalized)
                axis_length = np.linalg.norm(axis)
                
                if axis_length > 1e-6:
                    axis = axis / axis_length
                    angle = np.arccos(np.clip(np.dot(default_up, forward_normalized), -1, 1))
                    R = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)
                    cone.rotate(R, center=(0, 0, 0))
            
            cone.translate(position)
            cone.paint_uniform_color([0, 1, 0])  # Green for listeners
            
            geometries.append(cone)
        
        logger.info(f"Added {len(manifest['listener_positions'])} listeners")
        
        # Create coordinate frame at origin
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        geometries.append(coord_frame)
        
        # Combine all TriangleMesh geometries (spheres, cones, frame)
        combined_mesh = o3d.geometry.TriangleMesh()
        for geom in geometries:
            if isinstance(geom, o3d.geometry.TriangleMesh):
                combined_mesh += geom

        # Save visualization
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        if len(geometries) > 0:
            if point_cloud_path and os.path.exists(point_cloud_path):
                # Merge the point cloud with the audio-source markers.
                # Convert the marker mesh to a point cloud so everything fits
                # in a single PLY (open3d can't mix mesh+pcd in one write call).
                pcd = o3d.io.read_point_cloud(point_cloud_path)
                marker_pcd = combined_mesh.sample_points_uniformly(number_of_points=10000)
                merged = pcd + marker_pcd
                o3d.io.write_point_cloud(output_path, merged)
            else:
                o3d.io.write_triangle_mesh(output_path, combined_mesh)

            logger.info(f"Visualization saved to: {output_path}")
        
        # Also create an HTML viewer
        self._create_html_viewer(manifest, output_path.replace('.ply', '.html'))
        
        return output_path
    
    def _create_html_viewer(self, manifest: Dict[str, Any], output_path: str) -> None:
        """
        Create an interactive HTML viewer for the audio scene
        
        Args:
            manifest: Audio manifest
            output_path: Path to save HTML file
        """
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>Spatial Audio Scene Viewer</title>
    <style>
        body {{ margin: 0; font-family: Arial, sans-serif; }}
        #info {{
            position: absolute;
            top: 10px;
            left: 10px;
            background: rgba(0,0,0,0.7);
            color: white;
            padding: 10px;
            border-radius: 5px;
            max-width: 300px;
        }}
        #canvas {{ width: 100%; height: 100vh; }}
    </style>
</head>
<body>
    <div id="info">
        <h3>Spatial Audio Scene</h3>
        <p><strong>Audio Sources:</strong> {len(manifest['audio_sources'])}</p>
        <p><strong>Listeners:</strong> {len(manifest['listener_positions'])}</p>
        <p>Red spheres = Audio sources (brightness = intensity)</p>
        <p>Green cones = Camera/Listener positions</p>
    </div>
    <div id="canvas"></div>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        const manifest = {json.dumps(manifest)};
        
        // Basic Three.js scene setup
        const scene = new THREE.Scene();
        const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        const renderer = new THREE.WebGLRenderer();
        renderer.setSize(window.innerWidth, window.innerHeight);
        document.getElementById('canvas').appendChild(renderer.domElement);
        
        // Add lights
        const ambientLight = new THREE.AmbientLight(0x404040);
        scene.add(ambientLight);
        const directionalLight = new THREE.DirectionalLight(0xffffff, 0.5);
        directionalLight.position.set(1, 1, 1);
        scene.add(directionalLight);
        
        // Add audio sources
        manifest.audio_sources.forEach(source => {{
            const geometry = new THREE.SphereGeometry(0.1, 32, 32);
            const intensity = source.intensity;
            const material = new THREE.MeshPhongMaterial({{
                color: new THREE.Color(intensity, 0.2, 1 - intensity)
            }});
            const sphere = new THREE.Mesh(geometry, material);
            sphere.position.set(source.position_3d[0], source.position_3d[1], source.position_3d[2]);
            scene.add(sphere);
        }});
        
        // Add listeners
        manifest.listener_positions.forEach(listener => {{
            const geometry = new THREE.ConeGeometry(0.05, 0.15, 32);
            const material = new THREE.MeshPhongMaterial({{ color: 0x00ff00 }});
            const cone = new THREE.Mesh(geometry, material);
            cone.position.set(listener.position[0], listener.position[1], listener.position[2]);
            scene.add(cone);
        }});
        
        // Add coordinate axes
        const axesHelper = new THREE.AxesHelper(0.5);
        scene.add(axesHelper);
        
        // Position camera
        camera.position.set(2, 2, 2);
        camera.lookAt(0, 0, 0);
        
        // Animation loop
        function animate() {{
            requestAnimationFrame(animate);
            renderer.render(scene, camera);
        }}
        animate();
        
        // Handle window resize
        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});
        
        // Simple camera controls (drag to rotate)
        let isDragging = false;
        let previousMousePosition = {{ x: 0, y: 0 }};
        
        renderer.domElement.addEventListener('mousedown', () => {{ isDragging = true; }});
        renderer.domElement.addEventListener('mouseup', () => {{ isDragging = false; }});
        renderer.domElement.addEventListener('mousemove', (e) => {{
            if (isDragging) {{
                const deltaX = e.clientX - previousMousePosition.x;
                const deltaY = e.clientY - previousMousePosition.y;
                
                camera.position.x += deltaX * 0.01;
                camera.position.y -= deltaY * 0.01;
                camera.lookAt(0, 0, 0);
            }}
            previousMousePosition = {{ x: e.clientX, y: e.clientY }};
        }});
    </script>
</body>
</html>"""
        
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML viewer saved to: {output_path}")


def export_for_unity(
    manifest: Dict[str, Any],
    output_path: str = "data/output/unity_scene.json"
) -> str:
    """
    Export audio manifest in Unity-compatible format
    
    Args:
        manifest: Audio manifest
        output_path: Output path
    
    Returns:
        Path to Unity export file
    """
    logger.info("Exporting for Unity")
    
    unity_data = {
        'audioSources': [
            {
                'name': source['label'],
                'audioClipPath': source['audio_file'],
                'position': {
                    'x': source['position_3d'][0],
                    'y': source['position_3d'][1],
                    'z': source['position_3d'][2]
                },
                'volume': source['intensity'],
                'spatialBlend': 1.0,  # Fully 3D
                'minDistance': 0.5,
                'maxDistance': 10.0
            }
            for source in manifest['audio_sources']
        ]
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(unity_data, f, indent=2)
    
    logger.info(f"Unity export saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    positioner = SpatialAudioPositioner()
    
    # Test with sample data
    test_objects = [
        {
            'id': 'obj_001',
            'label': 'water fountain',
            'category': 'water',
            'position_3d': [1.0, 0.5, 2.0],
            'audio_file': 'data/audio/obj_001.wav',
            'confidence': 0.9
        },
        {
            'id': 'obj_002',
            'label': 'wind chimes',
            'category': 'decoration',
            'position_3d': [-1.5, 1.2, 1.0],
            'audio_file': 'data/audio/obj_002.wav',
            'confidence': 0.85
        }
    ]
    
    test_camera_data = {
        'extrinsics': {
            0: np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 1.5],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])
        },
        'image_names': {0: 'camera_0.jpg'}
    }
    
    manifest = positioner.create_audio_manifest(test_objects, test_camera_data)
    print(f"Created manifest with {len(manifest['audio_sources'])} sources")
