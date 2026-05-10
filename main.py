"""Main Pipeline Orchestration Script"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, Any
import json

from dotenv import load_dotenv

from src.utils import (
    setup_logging,
    load_config,
    create_output_directories,
    load_images_from_directory,
    save_json
)
from src.image_generation import ImageGenerator
from src.reconstruction import reconstruct_scene
from src.segmentation import SceneSegmenter, project_objects_to_3d
from src.audio_generation import AudioGenerator
from src.spatial_audio import SpatialAudioPositioner, export_for_unity


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Multi-Modal Scene Audio Generation Pipeline"
    )
    
    parser.add_argument(
        '--input',
        type=str,
        default='data/input',
        help='Input directory containing 4 viewpoint images'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/output',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Configuration file path'
    )
    
    parser.add_argument(
        '--skip-generation',
        action='store_true',
        help='Skip image generation step (use existing images only)'
    )
    
    parser.add_argument(
        '--skip-reconstruction',
        action='store_true',
        help='Skip 3D reconstruction step'
    )
    
    parser.add_argument(
        '--skip-segmentation',
        action='store_true',
        help='Skip scene segmentation step'
    )
    
    parser.add_argument(
        '--skip-audio',
        action='store_true',
        help='Skip audio generation step'
    )
    
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu', 'mps'],
        help='Device to run models on'
    )
    
    return parser.parse_args()


class Pipeline:
    """Main pipeline orchestration class"""
    
    def __init__(self, config: Dict[str, Any], args):
        """
        Initialize pipeline
        
        Args:
            config: Configuration dictionary
            args: Command line arguments
        """
        self.config = config
        self.args = args
        self.logger = logging.getLogger(__name__)
        
        # Create output directories
        self.directories = create_output_directories()
        
        # Pipeline state
        self.state = {
            'input_images': [],
            'generated_images': [],
            'all_images': [],
            'reconstruction_results': {},
            'segmentation_results': [],
            'objects_3d': [],
            'objects_with_audio': [],
            'audio_manifest': {}
        }
    
    def run(self) -> Dict[str, Any]:
        """
        Run the complete pipeline
        
        Returns:
            Dictionary with pipeline results
        """
        self.logger.info("=" * 80)
        self.logger.info("Starting Multi-Modal Scene Audio Generation Pipeline")
        self.logger.info("=" * 80)
        
        # Step 1: Load input images
        self.logger.info("\n[1/5] Loading Input Images")
        self.load_input_images()
        
        # Step 2: Generate additional views
        if not self.args.skip_generation:
            self.logger.info("\n[2/5] Generating Additional Viewpoints")
            self.generate_views()
        else:
            self.logger.info("\n[2/5] Skipping image generation")
            self.state['all_images'] = self.state['input_images']
        
        # Step 3: 3D Reconstruction
        if not self.args.skip_reconstruction:
            self.logger.info("\n[3/5] Running 3D Reconstruction")
            self.reconstruct_3d()
        else:
            self.logger.info("\n[3/5] Skipping 3D reconstruction")
        
        # Step 4: Scene Segmentation
        if not self.args.skip_segmentation:
            self.logger.info("\n[4/5] Segmenting Scene Objects")
            self.segment_scene()
        else:
            self.logger.info("\n[4/5] Skipping scene segmentation")
        
        # Step 5: Audio Generation and Positioning
        if not self.args.skip_audio:
            self.logger.info("\n[5/5] Generating and Positioning Audio")
            self.generate_audio()
            if self.state.get('objects_with_audio'):
                self.position_audio()
            else:
                self.logger.warning("No audio objects to position — skipping spatial audio step")
        else:
            self.logger.info("\n[5/5] Skipping audio generation")
        
        # Save final results
        self.logger.info("\n[Final] Saving Results")
        self.save_results()
        
        self.logger.info("\n" + "=" * 80)
        self.logger.info("Pipeline Complete!")
        self.logger.info("=" * 80)
        
        return self.state
    
    def load_input_images(self):
        """Load input images from directory"""
        try:
            images, paths = load_images_from_directory(self.args.input)
            
            if len(images) == 0:
                raise ValueError(f"No images found in {self.args.input}")
            
            self.state['input_images'] = images
            self.state['input_paths'] = paths
            
            self.logger.info(f"✓ Loaded {len(images)} input images")
            
            if len(images) < 4:
                self.logger.warning(f"Expected 4 input images, found {len(images)}")
            
        except Exception as e:
            self.logger.error(f"Failed to load input images: {str(e)}")
            raise
    
    def generate_views(self):
        """Generate additional viewpoints using Gemini"""
        try:
            generator = ImageGenerator(
                model=self.config['image_generation']['model']
            )
            
            # Analyze scene first
            scene_description = generator.analyze_scene(self.state['input_images'])
            self.state['scene_description'] = scene_description
            self.logger.info(f"Scene: {scene_description[:200]}...")
            
            # output_size=None → use model native size (inputs are downscaled to match)
            _raw_size = self.config['image_generation'].get('output_size')
            _output_size = tuple(_raw_size) if _raw_size else None

            # Generate additional views with sequential conditioning
            generated_images = generator.generate_additional_views(
                input_images=self.state['input_images'],
                scene_description=scene_description,
                num_views=self.config['image_generation']['num_additional_views'],
                output_dir=self.directories['generated'],
                output_size=_output_size,
                chain_views=self.config['image_generation'].get('chain_views', True),
            )
            
            self.state['generated_images'] = generated_images
            self.state['all_images'] = self.state['input_images'] + generated_images
            
            self.logger.info(f"✓ Generated {len(generated_images)} additional views")
            self.logger.info(f"✓ Total images: {len(self.state['all_images'])}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate views: {str(e)}")
            self.logger.warning("Continuing with input images only")
            self.state['all_images'] = self.state['input_images']
    
    def reconstruct_3d(self):
        """Run 3D reconstruction"""
        try:
            results = reconstruct_scene(
                images=self.state['all_images'],
                image_paths=self.state.get('input_paths'),
                config=self.config['reconstruction'],
                output_dir=self.directories['reconstruction']
            )
            
            self.state['reconstruction_results'] = results
            
            # Log results
            if results.get('depth_maps'):
                self.logger.info(f"✓ Depth estimation: {len(results['depth_maps'])} depth maps")
            
            if results.get('sparse_reconstruction'):
                sparse = results['sparse_reconstruction']
                self.logger.info(f"✓ COLMAP sparse reconstruction: {len(sparse.images)} images, "
                               f"{len(sparse.points3D)} 3D points")
            
            if results.get('camera_data'):
                cam_data = results['camera_data']
                self.logger.info(f"✓ Camera data: {len(cam_data['intrinsics'])} cameras")
            
            if results.get('dense_point_cloud'):
                self.logger.info(f"✓ Dense point cloud: {results['dense_point_cloud']}")
            
        except Exception as e:
            self.logger.error(f"3D reconstruction failed: {str(e)}")
            self.logger.warning("Some features may be unavailable without reconstruction")
            # Guarantee downstream steps always have a dict to query
            if not self.state.get('reconstruction_results'):
                self.state['reconstruction_results'] = {}
    
    def segment_scene(self):
        """Segment scene objects using Qwen2.5-VL"""
        try:
            segmenter = SceneSegmenter(
                model_name=self.config['segmentation']['model'],
                device=self.args.device,
                quantization=self.config['segmentation']['quantization'],
                max_pixels=self.config['segmentation']['max_pixels'],
                min_pixels=self.config['segmentation']['min_pixels']
            )
            
            # Segment all images
            results = segmenter.segment_batch(
                images=self.state['all_images'],
                output_dir=self.directories['segmentation']
            )
            
            self.state['segmentation_results'] = results
            
            # Count total objects
            total_objects = sum(len(r.get('objects', [])) for r in results)
            self.logger.info(f"✓ Segmentation complete: {total_objects} objects detected")
            
            # Project to 3D if reconstruction available
            if self.state['reconstruction_results'].get('camera_data'):
                self.logger.info("Projecting objects to 3D space...")
                
                objects_3d = project_objects_to_3d(
                    segmentation_results=results,
                    camera_data=self.state['reconstruction_results']['camera_data'],
                    depth_maps=self.state['reconstruction_results'].get('depth_maps')
                )
                
                self.state['objects_3d'] = objects_3d
                self.logger.info(f"✓ Projected {len(objects_3d)} objects to 3D")
            else:
                self.logger.warning("Cannot project to 3D without reconstruction data")
            
        except Exception as e:
            self.logger.error(f"Scene segmentation failed: {str(e)}")
            raise
    
    def generate_audio(self):
        """Generate audio for detected objects"""
        try:
            generator = AudioGenerator(
                use_local=self.config['audio']['use_local_inference'],
                sample_rate=self.config['audio']['sample_rate']
            )
            
            # Use 3D objects if available, otherwise use 2D segmentation
            objects_to_process = self.state.get('objects_3d', [])
            
            if not objects_to_process:
                # Flatten segmentation results
                for result in self.state['segmentation_results']:
                    for i, obj in enumerate(result.get('objects', [])):
                        obj['id'] = f"obj_{result['image_index']:03d}_{i:03d}"
                        objects_to_process.append(obj)
            
            if not objects_to_process:
                self.logger.warning("No objects found for audio generation")
                return
            
            # Limit to reasonable number of objects
            max_objects = 20
            if len(objects_to_process) > max_objects:
                self.logger.warning(f"Too many objects ({len(objects_to_process)}), limiting to {max_objects}")
                # Sort by confidence and take top N
                objects_to_process = sorted(
                    objects_to_process,
                    key=lambda x: x.get('confidence', 0),
                    reverse=True
                )[:max_objects]
            
            # Generate audio for each object
            objects_with_audio = generator.generate_batch(
                objects=objects_to_process,
                output_dir=self.directories['audio'],
                duration=self.config['audio']['clip_duration']
            )
            
            self.state['objects_with_audio'] = objects_with_audio
            
            successful = len([o for o in objects_with_audio if o.get('audio_file')])
            self.logger.info(f"✓ Audio generation complete: {successful}/{len(objects_with_audio)} successful")
            
            # Generate ambient audio if scene description available
            if self.state.get('scene_description'):
                self.logger.info("Generating ambient scene audio...")
                try:
                    ambient_path = generator.generate_ambient_audio(
                        scene_description=self.state['scene_description'],
                        duration=10.0,
                        output_path=os.path.join(self.directories['audio'], 'ambient.wav')
                    )
                    self.state['ambient_audio'] = ambient_path
                    self.logger.info(f"✓ Ambient audio generated: {ambient_path}")
                except Exception as e:
                    self.logger.warning(f"Ambient audio generation failed: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"Audio generation failed: {str(e)}")
            raise
    
    def position_audio(self):
        """Position audio in 3D space"""
        try:
            positioner = SpatialAudioPositioner(
                coordinate_system=self.config['spatial_audio']['coordinate_system'],
                intensity_falloff=self.config['spatial_audio']['intensity_falloff'],
                default_intensity=self.config['spatial_audio']['default_intensity']
            )
            
            # Create audio manifest
            manifest = positioner.create_audio_manifest(
                objects_3d=self.state['objects_with_audio'],
                camera_data=self.state['reconstruction_results'].get('camera_data'),
                output_path=os.path.join(self.args.output, 'audio_manifest.json')
            )
            
            self.state['audio_manifest'] = manifest
            
            self.logger.info(f"✓ Audio manifest created: {len(manifest['audio_sources'])} sources")
            
            # Create visualization
            self.logger.info("Creating visualization...")
            point_cloud = self.state['reconstruction_results'].get('dense_point_cloud')
            
            vis_path = positioner.visualize_audio_scene(
                manifest=manifest,
                point_cloud_path=point_cloud,
                output_path=os.path.join(self.args.output, 'audio_visualization.ply')
            )
            
            self.logger.info(f"✓ Visualization created: {vis_path}")
            self.logger.info(f"✓ HTML viewer: {vis_path.replace('.ply', '.html')}")
            
            # Export for Unity
            unity_path = export_for_unity(
                manifest=manifest,
                output_path=os.path.join(self.args.output, 'unity_scene.json')
            )
            
            self.logger.info(f"✓ Unity export: {unity_path}")
            
        except Exception as e:
            self.logger.error(f"Audio positioning failed: {str(e)}")
            self.logger.warning("Continuing without spatial positioning")
    
    def save_results(self):
        """Save final pipeline results"""
        try:
            # Create summary
            summary = {
                'input_images': len(self.state['input_images']),
                'generated_images': len(self.state.get('generated_images', [])),
                'total_images': len(self.state.get('all_images', [])),
                'reconstruction': {
                    'sparse_points': len(self.state['reconstruction_results'].get('sparse_reconstruction').points3D) if self.state['reconstruction_results'].get('sparse_reconstruction') else 0,
                    'cameras': len(self.state['reconstruction_results'].get('camera_data', {}).get('intrinsics', {})),
                    'has_dense': self.state['reconstruction_results'].get('dense_point_cloud') is not None
                },
                'segmentation': {
                    'objects_detected': sum(len(r.get('objects', [])) for r in self.state.get('segmentation_results', [])),
                    'objects_3d': len(self.state.get('objects_3d', []))
                },
                'audio': {
                    'audio_files_generated': len([o for o in self.state.get('objects_with_audio', []) if o.get('audio_file')]),
                    'audio_sources': len(self.state.get('audio_manifest', {}).get('audio_sources', []))
                },
                'output_files': {
                    'audio_manifest': os.path.join(self.args.output, 'audio_manifest.json'),
                    'visualization_ply': os.path.join(self.args.output, 'audio_visualization.ply'),
                    'visualization_html': os.path.join(self.args.output, 'audio_visualization.html'),
                    'unity_export': os.path.join(self.args.output, 'unity_scene.json')
                }
            }
            
            # Save summary
            summary_path = os.path.join(self.args.output, 'pipeline_summary.json')
            save_json(summary, summary_path)
            
            self.logger.info(f"✓ Pipeline summary saved to: {summary_path}")
            
            # Print summary
            self.logger.info("\n" + "=" * 80)
            self.logger.info("PIPELINE SUMMARY")
            self.logger.info("=" * 80)
            self.logger.info(f"Input Images:           {summary['input_images']}")
            self.logger.info(f"Generated Images:       {summary['generated_images']}")
            self.logger.info(f"Total Images:           {summary['total_images']}")
            self.logger.info(f"3D Points (sparse):     {summary['reconstruction']['sparse_points']}")
            self.logger.info(f"Cameras:                {summary['reconstruction']['cameras']}")
            self.logger.info(f"Objects Detected:       {summary['segmentation']['objects_detected']}")
            self.logger.info(f"Objects in 3D:          {summary['segmentation']['objects_3d']}")
            self.logger.info(f"Audio Files Generated:  {summary['audio']['audio_files_generated']}")
            self.logger.info(f"Audio Sources:          {summary['audio']['audio_sources']}")
            self.logger.info("=" * 80)
            
            self.logger.info("\nOutput Files:")
            for name, path in summary['output_files'].items():
                if os.path.exists(path):
                    self.logger.info(f"  ✓ {name}: {path}")
                else:
                    self.logger.info(f"  ✗ {name}: {path} (not created)")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}")


def main():
    """Main entry point"""
    # Load environment variables
    load_dotenv()
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from: {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        sys.exit(1)
    
    # Check for required API keys
    if not args.skip_generation and not os.getenv("GEMINI_API_KEY"):
        logger.error("GEMINI_API_KEY not found. Set it in .env file or skip image generation with --skip-generation")
        sys.exit(1)
    
    # Create and run pipeline
    try:
        pipeline = Pipeline(config, args)
        results = pipeline.run()
        
        logger.info("\n✓ Pipeline completed successfully!")
        logger.info(f"Results saved to: {args.output}")
        
    except KeyboardInterrupt:
        logger.warning("\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"\nPipeline failed: {str(e)}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
