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


def visualize_audio_scene_gs(
    candidates,
    gs_ply: Optional[str] = None,
    output_path: str = "data/output/gs_scene_viewer.html",
    max_embedded_points: int = 150_000,
) -> str:
    """Create an interactive HTML viewer for the 3DGS scene with audio playback.

    Renders the Gaussian PLY as a particle cloud (sampled points) in Three.js,
    overlays coloured spheres for each enabled audio source, and adds per-source
    HTML5 ``<audio>`` controls below the 3D canvas.

    A trained splat often contains millions of Gaussians; this viewer **subsamples**
    vertices for a self-contained HTML file. SuperSplat and similar tools stream
    the full representation, so they look much denser. Increase
    ``max_embedded_points`` if your machine and browser tolerate a larger file.

    The camera **frames the embedded point cloud** (centroid + bounding radius),
    not the origin, so sources and splats stay in view together.

    Args:
        candidates:   List of ``AudioPlacementCandidate`` from
                      ``src.audio_placement.build_candidates``.
        gs_ply:       Path to the Gaussian PLY (combined composite or raw GS).
                      When present up to ``max_embedded_points`` are embedded as
                      inline JSON to avoid a separate file-serve step.
        output_path:  Destination path for the HTML file.
        max_embedded_points: Cap on vertices embedded (default 150k). Subsampling
                      uses evenly spaced indices along the PLY vertex order for
                      better spatial coverage than uniform random sampling.

    Returns:
        Path to the written HTML file.
    """
    # Collect enabled sources
    enabled_sources = [c for c in candidates if c.enabled]

    # ── Sample point cloud from PLY for inline embedding ──────────────────────
    cloud_points_json = "[]"
    cloud_colors_json = "[]"
    n_total_pts = 0
    # Camera framing from cloud bbox (world / PLY frame, same as audio positions)
    cloud_center_json = json.dumps([0.0, 0.0, 0.0])
    cloud_radius = 5.0
    point_size_js = 0.012
    n_embedded_pts = 0
    if gs_ply and os.path.exists(gs_ply):
        try:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(gs_ply)
            pts = np.asarray(pcd.points)
            cols = np.asarray(pcd.colors) if pcd.has_colors() else np.ones_like(pts) * 0.5
            n_total_pts = len(pts)
            cap = max(1000, int(max_embedded_points))
            if len(pts) > cap:
                # Even stride through vertices — better coverage than i.i.d. random
                # when PLY groups Gaussians spatially or by training order.
                idx = np.linspace(0, len(pts) - 1, num=cap, dtype=np.int64)
                idx = np.unique(idx)
                pts = pts[idx]
                cols = cols[idx]
            cmin = pts.min(axis=0)
            cmax = pts.max(axis=0)
            center = (0.5 * (cmin + cmax)).astype(np.float64)
            extent = float(np.linalg.norm(cmax - cmin)) + 1e-6
            cloud_radius = max(0.5, 0.55 * extent)
            # Point sprite size in world units (Three.js PointsMaterial)
            point_size_js = float(np.clip(0.004 + 0.0012 * extent, 0.006, 0.06))
            cloud_center_json = json.dumps(center.tolist())
            cloud_points_json = json.dumps(pts.tolist())
            cloud_colors_json = json.dumps(cols.tolist())
            n_embedded_pts = len(pts)
            logger.info(
                f"Embedded {len(pts):,} / {n_total_pts:,} cloud points from {gs_ply} "
                f"(cap={cap}); scene radius≈{cloud_radius:.3f}, point_size≈{point_size_js:.4f}"
            )
        except Exception as exc:
            logger.warning(f"Could not sample PLY for viewer: {exc}")

    # ── Build JS array of audio sources ───────────────────────────────────────
    src_js_list = []
    for c in enabled_sources:
        pos = c.final_position
        intensity = c.final_intensity
        # Use relative path so the HTML is portable inside the output folder
        audio_rel = os.path.relpath(c.audio_file, os.path.dirname(output_path)).replace("\\", "/")
        src_js_list.append(
            f"{{id:{json.dumps(c.id)},label:{json.dumps(c.label)},"
            f"audioFile:{json.dumps(audio_rel)},"
            f"position:[{pos[0]:.4f},{pos[1]:.4f},{pos[2]:.4f}],"
            f"intensity:{intensity:.4f}}}"
        )
    sources_js = "[" + ",\n".join(src_js_list) + "]"

    camera_near_js = max(0.001, float(cloud_radius) * 5e-4)
    camera_far_js = max(500.0, float(cloud_radius) * 120.0)
    audio_sphere_r_js = float(np.clip(0.025 + 0.05 * cloud_radius, 0.04, 0.22))

    stat_suffix = (
        f"embedded {n_embedded_pts:,} / {n_total_pts:,} PLY vertices (subsample cap)"
        if n_total_pts
        else "no PLY loaded — spheres only"
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>3DGS Audio Scene Viewer</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #1a1a2e; color: #eee; font-family: 'Segoe UI', sans-serif; display: flex; flex-direction: column; height: 100vh; overflow: hidden; }}
  #header {{ padding: 8px 16px; background: #16213e; display: flex; align-items: center; gap: 16px; flex-shrink: 0; }}
  #header h2 {{ font-size: 1rem; color: #e94560; }}
  #header span {{ font-size: 0.8rem; color: #aaa; }}
  #main {{ display: flex; flex: 1; min-height: 0; }}
  #viewport {{ flex: 1; position: relative; }}
  canvas {{ display: block; width: 100%; height: 100%; }}
  #panel {{ width: 300px; background: #0f3460; overflow-y: auto; padding: 12px; flex-shrink: 0; }}
  #panel h3 {{ font-size: 0.9rem; margin-bottom: 8px; color: #e94560; }}
  .src-card {{ background: #16213e; border-radius: 6px; padding: 10px; margin-bottom: 8px; border-left: 3px solid #e94560; }}
  .src-label {{ font-size: 0.85rem; font-weight: bold; margin-bottom: 4px; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .src-pos {{ font-size: 0.72rem; color: #aaa; margin-bottom: 6px; font-family: monospace; }}
  audio {{ width: 100%; height: 28px; }}
  #tooltip {{ position: absolute; top: 12px; left: 12px; background: rgba(0,0,0,0.7); color: #eee; font-size: 0.75rem; padding: 6px 10px; border-radius: 4px; pointer-events: none; }}
</style>
</head>
<body>
<div id="header">
  <h2>3DGS Audio Scene Viewer</h2>
  <span id="stat">{len(enabled_sources)} audio source(s) &nbsp;|&nbsp; {stat_suffix} &nbsp;|&nbsp; Drag to rotate &nbsp;|&nbsp; Scroll to zoom</span>
</div>
<div id="main">
  <div id="viewport">
    <canvas id="c"></canvas>
    <div id="tooltip">Hover over a sphere to see label</div>
  </div>
  <div id="panel">
    <h3>Audio Sources</h3>
    <div id="src-list"></div>
  </div>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script>
// ── Data ──────────────────────────────────────────────────────────────────
const CLOUD_PTS   = {cloud_points_json};
const CLOUD_COLS  = {cloud_colors_json};
const AUDIO_SRCS  = {sources_js};
const CLOUD_CENTER = {cloud_center_json};
const CLOUD_RADIUS = {float(cloud_radius)};
const POINT_SPRITE = {float(point_size_js)};
const AUDIO_MARKER_R = {float(audio_sphere_r_js)};
const cx = CLOUD_CENTER[0], cy = CLOUD_CENTER[1], cz = CLOUD_CENTER[2];

// ── Three.js Setup ────────────────────────────────────────────────────────
const canvas   = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({{canvas, antialias: true, alpha: true}});
renderer.setClearColor(0x1a1a2e, 1);

const scene  = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, 1, {float(camera_near_js)}, {float(camera_far_js)});

const ambient = new THREE.AmbientLight(0xffffff, 0.6);
scene.add(ambient);
const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
dirLight.position.set(cx + 5, cy + 10, cz + 5);
scene.add(dirLight);

// ── Point cloud ───────────────────────────────────────────────────────────
if (CLOUD_PTS.length > 0) {{
  const geo = new THREE.BufferGeometry();
  const flat = []; const flatC = [];
  for (let i = 0; i < CLOUD_PTS.length; i++) {{
    flat.push(CLOUD_PTS[i][0], CLOUD_PTS[i][1], CLOUD_PTS[i][2]);
    if (CLOUD_COLS.length > i) flatC.push(CLOUD_COLS[i][0], CLOUD_COLS[i][1], CLOUD_COLS[i][2]);
    else flatC.push(0.5, 0.5, 0.5);
  }}
  geo.setAttribute('position', new THREE.Float32BufferAttribute(flat, 3));
  geo.setAttribute('color', new THREE.Float32BufferAttribute(flatC, 3));
  const mat = new THREE.PointsMaterial({{size: POINT_SPRITE, vertexColors: true, sizeAttenuation: true}});
  scene.add(new THREE.Points(geo, mat));
}}

// ── Audio source spheres ──────────────────────────────────────────────────
const spheres = [];
AUDIO_SRCS.forEach((src, idx) => {{
  const i = src.intensity;
  const geo  = new THREE.SphereGeometry(AUDIO_MARKER_R, 24, 24);
  const mat  = new THREE.MeshPhongMaterial({{color: new THREE.Color(i, 0.2, 1-i), emissive: new THREE.Color(i*0.3, 0, (1-i)*0.3)}});
  const mesh = new THREE.Mesh(geo, mat);
  mesh.position.set(src.position[0], src.position[1], src.position[2]);
  mesh.userData = {{label: src.label, idx}};
  scene.add(mesh);
  spheres.push(mesh);
}});

// ── Axes helper at scene centroid ───────────────────────────────────────────
const axLen = Math.max(0.15, CLOUD_RADIUS * 0.15);
const axes = new THREE.AxesHelper(axLen);
axes.position.set(cx, cy, cz);
scene.add(axes);

// ── Resize ────────────────────────────────────────────────────────────────
function resize() {{
  const vp = document.getElementById('viewport');
  const w = vp.clientWidth, h = vp.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}}
window.addEventListener('resize', resize);
resize();

// ── Orbit controls (minimal) — pivot at cloud centroid, not world origin ───
let isDrag = false, lastX = 0, lastY = 0;
let theta = 0, phi = Math.PI / 3;
let radius = CLOUD_RADIUS;
const rMin = Math.max(0.08, CLOUD_RADIUS * 0.12);
const rMax = Math.max(CLOUD_RADIUS * 25.0, 80.0);
function updateCamera() {{
  camera.position.set(
    cx + radius * Math.sin(phi) * Math.sin(theta),
    cy + radius * Math.cos(phi),
    cz + radius * Math.sin(phi) * Math.cos(theta)
  );
  camera.lookAt(cx, cy, cz);
}}
canvas.addEventListener('mousedown', e => {{ isDrag = true; lastX = e.clientX; lastY = e.clientY; }});
canvas.addEventListener('mouseup',   () => isDrag = false);
canvas.addEventListener('mouseleave',() => isDrag = false);
canvas.addEventListener('mousemove', e => {{
  if (!isDrag) {{ rayCast(e); return; }}
  const dx = e.clientX - lastX, dy = e.clientY - lastY;
  theta -= dx * 0.005; phi = Math.max(0.1, Math.min(Math.PI-0.1, phi - dy * 0.005));
  lastX = e.clientX; lastY = e.clientY;
  updateCamera();
}});
canvas.addEventListener('wheel', e => {{
  const step = 0.012 * CLOUD_RADIUS;
  radius = Math.max(rMin, Math.min(rMax, radius + e.deltaY * 0.01 * step));
  updateCamera();
}});
updateCamera();

// ── Raycasting for hover ──────────────────────────────────────────────────
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
const tooltip = document.getElementById('tooltip');
function rayCast(e) {{
  const vp = document.getElementById('viewport');
  const rect = vp.getBoundingClientRect();
  mouse.x =  ((e.clientX - rect.left) / rect.width)  * 2 - 1;
  mouse.y = -((e.clientY - rect.top)  / rect.height) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObjects(spheres);
  if (hits.length > 0) {{
    const {{label}} = hits[0].object.userData;
    tooltip.textContent = label;
    tooltip.style.opacity = 1;
  }} else {{
    tooltip.style.opacity = 0.5;
    tooltip.textContent = 'Drag to rotate | Scroll to zoom';
  }}
}}

// ── Audio source cards ────────────────────────────────────────────────────
const list = document.getElementById('src-list');
AUDIO_SRCS.forEach((src, idx) => {{
  const p = src.position;
  const card = document.createElement('div');
  card.className = 'src-card';
  card.innerHTML = `
    <div class="src-label">${{src.label}}</div>
    <div class="src-pos">[${{p[0].toFixed(2)}}, ${{p[1].toFixed(2)}}, ${{p[2].toFixed(2)}}]  intensity=${{src.intensity.toFixed(2)}}</div>
    <audio controls preload="none" src="${{src.audioFile}}"></audio>
  `;
  card.addEventListener('mouseenter', () => spheres[idx].scale.setScalar(1.5));
  card.addEventListener('mouseleave', () => spheres[idx].scale.setScalar(1.0));
  list.appendChild(card);
}});

// ── Render loop ───────────────────────────────────────────────────────────
(function animate() {{
  requestAnimationFrame(animate);
  renderer.render(scene, camera);
}})();
</script>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    logger.info(f"GS audio viewer saved to: {output_path}")
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
