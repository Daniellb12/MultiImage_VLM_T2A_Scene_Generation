"""Audio Generation Module using MMAudio"""

import os
import shutil
import logging
from typing import List, Dict, Any, Optional
import numpy as np
import time

try:
    from gradio_client import Client
except ImportError:
    raise ImportError("Please install gradio-client: pip install gradio-client")

logger = logging.getLogger(__name__)


class AudioGenerator:
    """Generate foley audio using MMAudio"""
    
    def __init__(
        self,
        use_local: bool = False,
        space_name: str = "hkchengrex/MMAudio",
        sample_rate: int = 44100
    ):
        """
        Initialize audio generator
        
        Args:
            use_local: Whether to use local MMAudio inference (not implemented yet)
            space_name: HuggingFace Space name for MMAudio
            sample_rate: Audio sample rate
        """
        self.use_local = use_local
        self.space_name = space_name
        self.sample_rate = sample_rate
        
        if use_local:
            logger.warning("Local MMAudio inference not implemented yet, falling back to API")
            self.use_local = False
        
        if not self.use_local:
            logger.info(f"Connecting to MMAudio Space: {space_name}")
            try:
                self.client = Client(space_name)
                logger.info("Successfully connected to MMAudio API")
            except Exception as e:
                logger.error(f"Failed to connect to MMAudio Space: {str(e)}")
                raise
    
    def generate_audio(
        self,
        prompt: str,
        duration: float = 4.0,
        negative_prompt: str = "music, speech, singing",
        num_steps: int = 25,
        guidance_scale: float = 4.5,
        seed: Optional[int] = None
    ) -> str:
        """
        Generate audio from text prompt
        
        Args:
            prompt: Text description of the sound
            duration: Duration in seconds
            negative_prompt: What to avoid in generation
            num_steps: Number of diffusion steps
            guidance_scale: Guidance scale for generation
            seed: Random seed for reproducibility
        
        Returns:
            Path to generated audio file
        """
        logger.info(f"Generating audio: '{prompt}' ({duration}s)")
        
        if seed is None:
            seed = int(time.time() * 1000) % (2**31)
        
        try:
            # MMAudio HuggingFace Space API name is "/video_to_audio" for the
            # combined endpoint, but text-only generation uses "/generate_audio".
            # The Space currently exposes the predict endpoint as "/predict";
            # the correct positional-argument order matches the Gradio fn signature:
            #   (video, prompt, negative_prompt, seed, num_steps, cfg_strength, duration)
            # For text-only (no video), pass None for the video argument.
            result = self.client.predict(
                None,           # video input (None = text-only)
                prompt,
                negative_prompt,
                seed,
                num_steps,
                guidance_scale,
                duration,
                api_name="/predict",
            )

            # Result is typically (audio_filepath, video_filepath) or just a path
            if isinstance(result, (list, tuple)) and len(result) > 0:
                audio_path = result[0]
            elif isinstance(result, str):
                audio_path = result
            else:
                raise ValueError(f"Unexpected result type: {type(result)}")

            if audio_path and os.path.exists(str(audio_path)):
                logger.info(f"Audio generated successfully: {audio_path}")
                return str(audio_path)

            logger.error(f"Audio file not found at path: {audio_path}")
            raise ValueError("Failed to get audio file from API")
            
        except Exception as e:
            logger.error(f"Error generating audio: {str(e)}")
            raise
    
    def generate_foley_for_object(
        self,
        object_label: str,
        object_category: str = "unknown",
        duration: float = 4.0
    ) -> str:
        """
        Generate foley audio for a specific object
        
        Args:
            object_label: Label/name of the object
            object_category: Category of the object
            duration: Duration in seconds
        
        Returns:
            Path to generated audio file
        """
        # Create a contextual prompt for better audio generation
        prompt = self._create_foley_prompt(object_label, object_category)
        
        return self.generate_audio(
            prompt=prompt,
            duration=duration,
            negative_prompt="music, speech, singing, voice, talking"
        )
    
    def _create_foley_prompt(self, label: str, category: str) -> str:
        """
        Create a contextual prompt for foley sound generation
        
        Args:
            label: Object label
            category: Object category
        
        Returns:
            Optimized prompt for audio generation
        """
        # Map object categories to appropriate sound descriptions
        sound_mappings = {
            'water': ['flowing water', 'water running', 'gentle stream'],
            'waterfall': ['waterfall cascading', 'water falling', 'rushing waterfall'],
            'fountain': ['water fountain', 'fountain splashing', 'fountain bubbling'],
            'wind': ['wind blowing', 'gentle breeze', 'wind rustling'],
            'leaves': ['leaves rustling', 'wind through leaves', 'tree leaves moving'],
            'door': ['door creaking', 'door opening', 'wooden door'],
            'clock': ['clock ticking', 'ticking sound', 'clock mechanism'],
            'fire': ['fire crackling', 'campfire', 'flames burning'],
            'footsteps': ['footsteps walking', 'footsteps on floor', 'walking sounds'],
            'rain': ['rain falling', 'rainfall', 'raindrops'],
            'ocean': ['ocean waves', 'sea waves', 'beach waves'],
            'birds': ['birds chirping', 'bird sounds', 'birdsong'],
            'traffic': ['traffic sounds', 'cars passing', 'street traffic'],
            'machine': ['machine running', 'mechanical sounds', 'machinery operating'],
            'engine': ['engine running', 'motor sound', 'engine idling'],
            'keyboard': ['keyboard typing', 'typing sounds', 'computer keyboard'],
            'glass': ['glass clinking', 'glass sounds', 'glassware'],
            'metal': ['metal clanging', 'metallic sounds', 'metal objects'],
            'paper': ['paper rustling', 'paper shuffling', 'paper sounds'],
            'wood': ['wood creaking', 'wooden sounds', 'wood knocking'],
        }
        
        # Check if label contains any known sound keywords
        label_lower = label.lower()
        for keyword, sound_options in sound_mappings.items():
            if keyword in label_lower:
                return np.random.choice(sound_options)
        
        # Category-based fallback
        category_lower = category.lower()
        if 'furniture' in category_lower:
            return f"ambient sounds of {label}"
        elif 'nature' in category_lower:
            return f"natural ambient sound of {label}"
        elif 'electronics' in category_lower or 'appliance' in category_lower:
            return f"operating sound of {label}"
        elif 'vehicle' in category_lower:
            return f"{label} running sound"
        
        # Default: use label directly
        return f"sound of {label}"
    
    def generate_batch(
        self,
        objects: List[Dict[str, Any]],
        output_dir: str = "data/audio",
        duration: float = 4.0,
        max_concurrent: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Generate audio for multiple objects
        
        Args:
            objects: List of object dictionaries with 'label' and 'category'
            output_dir: Directory to save audio files
            duration: Duration for each audio clip
            max_concurrent: Maximum concurrent API calls (currently limited to 1)
        
        Returns:
            List of objects with added 'audio_file' field
        """
        logger.info(f"Generating audio for {len(objects)} objects")
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        
        for i, obj in enumerate(objects):
            logger.info(f"Processing object {i+1}/{len(objects)}: {obj.get('label', 'unknown')}")
            
            try:
                # Generate audio
                temp_audio_path = self.generate_foley_for_object(
                    object_label=obj.get('label', 'unknown'),
                    object_category=obj.get('category', 'unknown'),
                    duration=duration
                )
                
                # Copy to output directory with proper naming
                obj_id = obj.get('id', f"obj_{i:03d}")
                output_filename = f"{obj_id}.wav"
                output_path = os.path.join(output_dir, output_filename)

                shutil.copy2(temp_audio_path, output_path)
                
                # Update object with audio path
                obj_with_audio = obj.copy()
                obj_with_audio['audio_file'] = output_path
                obj_with_audio['audio_duration'] = duration
                results.append(obj_with_audio)
                
                logger.info(f"Audio saved to: {output_path}")
                
                # Small delay to avoid rate limiting
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Failed to generate audio for object {i}: {str(e)}")
                # Add object without audio
                obj_with_audio = obj.copy()
                obj_with_audio['audio_file'] = None
                obj_with_audio['error'] = str(e)
                results.append(obj_with_audio)
        
        logger.info(f"Audio generation complete. Generated {len([r for r in results if r.get('audio_file')])} audio files")
        return results
    
    def generate_ambient_audio(
        self,
        scene_description: str,
        duration: float = 10.0,
        output_path: str = "data/audio/ambient.wav"
    ) -> str:
        """
        Generate ambient background audio for the entire scene
        
        Args:
            scene_description: Description of the scene
            duration: Duration in seconds
            output_path: Path to save ambient audio
        
        Returns:
            Path to generated ambient audio
        """
        logger.info("Generating ambient scene audio")
        
        # Create ambient-focused prompt
        prompt = f"ambient environmental sounds for {scene_description}, subtle background atmosphere"
        
        temp_audio = self.generate_audio(
            prompt=prompt,
            duration=duration,
            negative_prompt="music, speech, singing, voice, talking, loud noises"
        )
        
        # Copy to desired location (dirname may be empty if path is bare filename)
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        shutil.copy2(temp_audio, output_path)
        
        logger.info(f"Ambient audio saved to: {output_path}")
        return output_path


def load_audio_info(audio_file: str) -> Dict[str, Any]:
    """
    Load audio file information
    
    Args:
        audio_file: Path to audio file
    
    Returns:
        Dictionary with audio metadata
    """
    try:
        import soundfile as sf
        
        info = sf.info(audio_file)
        
        return {
            'duration': info.duration,
            'sample_rate': info.samplerate,
            'channels': info.channels,
            'format': info.format,
            'subtype': info.subtype
        }
    except ImportError:
        logger.warning("soundfile not installed, cannot load audio info")
        return {}
    except Exception as e:
        logger.error(f"Error loading audio info: {str(e)}")
        return {}


def merge_audio_sources(
    audio_files: List[str],
    output_path: str,
    volumes: Optional[List[float]] = None
) -> str:
    """
    Merge multiple audio sources into a single file
    
    Args:
        audio_files: List of audio file paths
        output_path: Output path for merged audio
        volumes: Optional volume multipliers for each audio source
    
    Returns:
        Path to merged audio file
    """
    try:
        import soundfile as sf
        
        if volumes is None:
            volumes = [1.0] * len(audio_files)
        
        # Load all audio files
        audio_data = []
        max_length = 0
        sample_rate = None
        
        for audio_file, volume in zip(audio_files, volumes):
            data, sr = sf.read(audio_file)
            
            if sample_rate is None:
                sample_rate = sr
            elif sr != sample_rate:
                logger.warning(f"Sample rate mismatch: {sr} vs {sample_rate}")
            
            # Apply volume
            data = data * volume
            
            audio_data.append(data)
            max_length = max(max_length, len(data))
        
        # Pad all audio to same length and sum
        mixed = np.zeros(max_length)
        for data in audio_data:
            if len(data) < max_length:
                data = np.pad(data, (0, max_length - len(data)))
            mixed += data
        
        # Normalize to prevent clipping
        if np.max(np.abs(mixed)) > 1.0:
            mixed = mixed / np.max(np.abs(mixed)) * 0.95
        
        # Save merged audio
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        sf.write(output_path, mixed, sample_rate)
        
        logger.info(f"Merged {len(audio_files)} audio files to: {output_path}")
        return output_path
        
    except ImportError:
        logger.error("soundfile not installed, cannot merge audio")
        raise
    except Exception as e:
        logger.error(f"Error merging audio: {str(e)}")
        raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example usage
    generator = AudioGenerator()
    
    # Test single audio generation
    audio_path = generator.generate_foley_for_object(
        object_label="flowing water",
        object_category="nature",
        duration=4.0
    )
    print(f"Generated audio: {audio_path}")
    
    # Test batch generation
    test_objects = [
        {'id': 'obj_001', 'label': 'running water', 'category': 'nature'},
        {'id': 'obj_002', 'label': 'wind chimes', 'category': 'decoration'},
        {'id': 'obj_003', 'label': 'clock ticking', 'category': 'furniture'}
    ]
    
    results = generator.generate_batch(test_objects, duration=3.0)
    print(f"\nGenerated {len(results)} audio files")
