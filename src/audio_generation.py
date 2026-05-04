"""Audio Generation Module using MMAudio (local inference preferred, HF Space fallback)"""

import logging
import os
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Path to the cloned MMAudio repo — weights are downloaded relative to this dir.
MMAUDIO_REPO = Path(__file__).resolve().parents[1].parent / "MMAudio"


# ── Local inference helpers ────────────────────────────────────────────────────

def _load_local_model(variant: str = "medium_44k", device: str = "cuda"):
#def _load_local_model(variant: str = "large_44k_v2", device: str = "cuda"):

    """Load MMAudio model weights into memory.  Returns (net, feature_utils, fm, seq_cfg)."""
    import torch
    from mmaudio.eval_utils import all_model_cfg
    from mmaudio.model.flow_matching import FlowMatching
    from mmaudio.model.networks import get_my_mmaudio
    from mmaudio.model.utils.features_utils import FeaturesUtils

    # Weights paths inside MMAudio repo are relative — cd into the repo first.
    orig_cwd = Path.cwd()
    os.chdir(MMAUDIO_REPO)
    try:
        model_cfg = all_model_cfg[variant]
        model_cfg.download_if_needed()
        seq_cfg = model_cfg.seq_cfg

        dtype = torch.bfloat16

        # Load entirely outside inference_mode so parameters are normal tensors.
        # A prior model (e.g. Qwen2.5-VL) may have set inference_mode globally
        # in this kernel session; we must neutralise it here to avoid the
        # "Inference tensors cannot be saved for backward" error during the
        # euler ODE loop inside generate().
        with torch.inference_mode(mode=False):
            with torch.no_grad():
                net = get_my_mmaudio(model_cfg.model_name).to(device, dtype).eval()
                net.load_weights(
                    torch.load(model_cfg.model_path, map_location=device, weights_only=True)
                )
                # Clone every parameter so none carry the inference-mode flag.
                for p in net.parameters():
                    p.data = p.data.clone()

                feature_utils = FeaturesUtils(
                    tod_vae_ckpt=model_cfg.vae_path,
                    synchformer_ckpt=model_cfg.synchformer_ckpt,
                    enable_conditions=True,
                    mode=model_cfg.mode,
                    bigvgan_vocoder_ckpt=model_cfg.bigvgan_16k_path,
                    need_vae_encoder=False,
                ).to(device, dtype).eval()
                for p in feature_utils.parameters():
                    p.data = p.data.clone()

        #fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=10)
        fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=25)
    finally:
        os.chdir(orig_cwd)

    logger.info(f"Loaded MMAudio '{variant}' on {device}")
    return net, feature_utils, fm, seq_cfg


@staticmethod
def _run_local(
    prompt: str,
    negative_prompt: str,
    duration: float,
    num_steps: int,
    cfg_strength: float,
    seed: int,
    net,
    feature_utils,
    fm,
    seq_cfg,
    device: str,
) -> np.ndarray:
    """Run one text-to-audio inference pass.  Returns float32 numpy waveform."""
    import torch
    from mmaudio.eval_utils import generate

    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    fm.num_steps = num_steps
    seq_cfg.duration = duration
    net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)

    audios = generate(
        None, None,               # no video → text-only mode
        [prompt],
        negative_text=[negative_prompt],
        feature_utils=feature_utils,
        net=net,
        fm=fm,
        rng=rng,
        cfg_strength=cfg_strength,
    )
    return audios.float().cpu()[0].numpy()   # shape (channels, samples)


# ── AudioGenerator class ───────────────────────────────────────────────────────

class AudioGenerator:
    """Generate foley audio using MMAudio.

    Tries local GPU inference first; falls back to the HuggingFace Gradio Space
    if the local model cannot be loaded (e.g. mmaudio package not found, or
    ``force_api=True`` is passed).
    """

    def __init__(
        self,
        variant: str = "medium_44k",
        force_api: bool = False,
        space_name: str = "hkchengrex/MMAudio",
        sample_rate: int = 44100,
    ):
        self.sample_rate = sample_rate
        self._net = self._feature_utils = self._fm = self._seq_cfg = None
        self._device = None
        self._use_local = False
        self._client = None

        if not force_api:
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._net, self._feature_utils, self._fm, self._seq_cfg = \
                    _load_local_model(variant=variant, device=device)
                self._device = device
                self._use_local = True
                logger.info("AudioGenerator: using LOCAL MMAudio inference")
            except Exception as e:
                logger.warning(f"Local MMAudio unavailable ({e}), falling back to HF Space API")

        if not self._use_local:
            try:
                from gradio_client import Client
                self._client = Client(space_name)
                logger.info(f"AudioGenerator: using HF Space API ({space_name})")
            except Exception as e:
                raise RuntimeError(
                    f"Cannot initialise AudioGenerator — local load failed and "
                    f"Gradio client also failed: {e}"
                ) from e

    # ── core generation ────────────────────────────────────────────────────────

    def generate_audio(
        self,
        prompt: str,
        duration: float = 4.0,
        negative_prompt: str = "music, speech, singing",
        num_steps: int = 25,
        guidance_scale: float = 4.5,
        seed: Optional[int] = None,
    ) -> str:
        """Generate audio from a text prompt.  Returns path to a .wav/.flac file."""
        if seed is None:
            seed = int(time.time() * 1000) % (2 ** 31)

        logger.info(f"Generating audio: '{prompt}' ({duration}s, seed={seed})")

        if self._use_local:
            return self._generate_local(prompt, negative_prompt, duration,
                                        num_steps, guidance_scale, seed)
        return self._generate_api(prompt, negative_prompt, duration,
                                  num_steps, guidance_scale, seed)

    def _generate_local(
        self,
        prompt: str,
        negative_prompt: str,
        duration: float,
        num_steps: int,
        cfg_strength: float,
        seed: int,
    ) -> str:
        import tempfile
        import torch
        import torchaudio

        orig_cwd = Path.cwd()
        os.chdir(MMAUDIO_REPO)
        try:
            # Use no_grad (not inference_mode) so the euler ODE loop inside
            # MMAudio's FlowMatching can still perform in-place tensor ops
            # without hitting the "inference tensor" restriction.
            with torch.inference_mode(mode=False), torch.no_grad():
                waveform = _run_local(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    duration=duration,
                    num_steps=num_steps,
                    cfg_strength=cfg_strength,
                    seed=seed,
                    net=self._net,
                    feature_utils=self._feature_utils,
                    fm=self._fm,
                    seq_cfg=self._seq_cfg,
                    device=self._device,
                )
        finally:
            os.chdir(orig_cwd)

        import torch
        tmp = tempfile.NamedTemporaryFile(suffix=".flac", delete=False)
        tmp.close()
        torchaudio.save(tmp.name, torch.from_numpy(waveform), self._seq_cfg.sampling_rate)
        logger.info(f"Local audio saved to temp: {tmp.name}")
        return tmp.name

    def _generate_api(
        self,
        prompt: str,
        negative_prompt: str,
        duration: float,
        num_steps: int,
        cfg_strength: float,
        seed: int,
    ) -> str:
        result = self._client.predict(
            prompt,
            negative_prompt,
            seed,
            num_steps,
            cfg_strength,
            duration,
            api_name="/text_to_audio",
        )

        if isinstance(result, (list, tuple)) and len(result) > 0:
            audio_path = result[0]
        elif isinstance(result, str):
            audio_path = result
        else:
            raise ValueError(f"Unexpected API result type: {type(result)}")

        if audio_path and os.path.exists(str(audio_path)):
            logger.info(f"API audio at: {audio_path}")
            return str(audio_path)

        raise ValueError(f"Audio file not found at path returned by API: {audio_path}")

    # ── foley helpers ──────────────────────────────────────────────────────────

    def generate_foley_for_object(
        self,
        object_label: str,
        object_category: str = "unknown",
        duration: float = 4.0,
    ) -> str:
        prompt = self._create_foley_prompt(object_label, object_category)
        return self.generate_audio(
            prompt=prompt,
            duration=duration,
            negative_prompt="music, speech, singing, voice, talking",
        )

    def _create_foley_prompt(self, label: str, category: str) -> str:
        sound_mappings = {
            "water":      ["flowing water", "water running", "gentle stream"],
            "waterfall":  ["waterfall cascading", "water falling", "rushing waterfall"],
            "fountain":   ["water fountain", "fountain splashing", "fountain bubbling"],
            "wind":       ["wind blowing", "gentle breeze", "wind rustling"],
            "leaves":     ["leaves rustling", "wind through leaves", "tree leaves moving"],
            "door":       ["door creaking", "door opening", "wooden door"],
            "clock":      ["clock ticking", "ticking sound", "clock mechanism"],
            "fire":       ["fire crackling", "campfire", "flames burning"],
            "footsteps":  ["footsteps walking", "footsteps on floor", "walking sounds"],
            "rain":       ["rain falling", "rainfall", "raindrops"],
            "ocean":      ["ocean waves", "sea waves", "beach waves"],
            "birds":      ["birds chirping", "bird sounds", "birdsong"],
            "traffic":    ["traffic sounds", "cars passing", "street traffic"],
            "machine":    ["machine running", "mechanical sounds", "machinery operating"],
            "engine":     ["engine running", "motor sound", "engine idling"],
            "keyboard":   ["keyboard typing", "typing sounds", "computer keyboard"],
            "glass":      ["glass clinking", "glass sounds", "glassware"],
            "metal":      ["metal clanging", "metallic sounds", "metal objects"],
            "paper":      ["paper rustling", "paper shuffling", "paper sounds"],
            "wood":       ["wood creaking", "wooden sounds", "wood knocking"],
        }
        label_lower = label.lower()
        for keyword, options in sound_mappings.items():
            if keyword in label_lower:
                return np.random.choice(options)

        cat = category.lower()
        if "furniture" in cat:
            return f"ambient sounds of {label}"
        if "nature" in cat:
            return f"natural ambient sound of {label}"
        if "electronics" in cat or "appliance" in cat:
            return f"operating sound of {label}"
        if "vehicle" in cat:
            return f"{label} running sound"
        return f"sound of {label}"

    # ── batch generation ───────────────────────────────────────────────────────

    def generate_batch(
        self,
        objects: List[Dict[str, Any]],
        output_dir: str = "data/audio",
        duration: float = 4.0,
        max_concurrent: int = 1,
    ) -> List[Dict[str, Any]]:
        logger.info(f"Generating audio for {len(objects)} objects")
        os.makedirs(output_dir, exist_ok=True)

        results = []
        for i, obj in enumerate(objects):
            label = obj.get("label", "unknown")
            obj_id = obj.get("id", f"obj_{i:03d}")
            output_path = os.path.join(output_dir, f"{obj_id}.wav")

            # Skip if already generated
            if os.path.exists(output_path):
                logger.info(f"Skipping {label} — already exists at {output_path}")
                obj_out = obj.copy()
                obj_out["audio_file"] = output_path
                obj_out["audio_duration"] = duration
                results.append(obj_out)
                continue

            logger.info(f"Processing {i+1}/{len(objects)}: {label}")
            try:
                temp_path = self.generate_foley_for_object(
                    object_label=label,
                    object_category=obj.get("category", "unknown"),
                    duration=duration,
                )
                shutil.copy2(temp_path, output_path)

                obj_out = obj.copy()
                obj_out["audio_file"] = output_path
                obj_out["audio_duration"] = duration
                results.append(obj_out)
                logger.info(f"Audio saved: {output_path}")

                if not self._use_local:
                    time.sleep(1)   # rate-limit only for API calls

            except Exception as e:
                logger.error(f"Failed for object {i} ({label}): {e}")
                obj_out = obj.copy()
                obj_out["audio_file"] = None
                obj_out["error"] = str(e)
                results.append(obj_out)

        generated = sum(1 for r in results if r.get("audio_file"))
        logger.info(f"Audio generation complete: {generated}/{len(objects)} clips")
        return results

    # ── ambient audio ──────────────────────────────────────────────────────────

    def generate_ambient_audio(
        self,
        scene_description: str,
        duration: float = 10.0,
        output_path: str = "data/audio/ambient.wav",
    ) -> str:
        logger.info("Generating ambient scene audio")
        prompt = (
            f"ambient environmental sounds for {scene_description}, "
            "subtle background atmosphere"
        )
        temp = self.generate_audio(
            prompt=prompt,
            duration=duration,
            negative_prompt="music, speech, singing, voice, talking, loud noises",
        )
        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        shutil.copy2(temp, output_path)
        logger.info(f"Ambient audio saved: {output_path}")
        return output_path


# ── standalone utilities ───────────────────────────────────────────────────────

def load_audio_info(audio_file: str) -> Dict[str, Any]:
    try:
        import soundfile as sf
        info = sf.info(audio_file)
        return {
            "duration": info.duration,
            "sample_rate": info.samplerate,
            "channels": info.channels,
            "format": info.format,
            "subtype": info.subtype,
        }
    except ImportError:
        logger.warning("soundfile not installed, cannot load audio info")
        return {}
    except Exception as e:
        logger.error(f"Error loading audio info: {e}")
        return {}


def merge_audio_sources(
    audio_files: List[str],
    output_path: str,
    volumes: Optional[List[float]] = None,
) -> str:
    try:
        import soundfile as sf

        if volumes is None:
            volumes = [1.0] * len(audio_files)

        audio_data, max_length, sample_rate = [], 0, None
        for af, vol in zip(audio_files, volumes):
            data, sr = sf.read(af)
            if sample_rate is None:
                sample_rate = sr
            elif sr != sample_rate:
                logger.warning(f"Sample rate mismatch: {sr} vs {sample_rate}")
            data = data * vol
            audio_data.append(data)
            max_length = max(max_length, len(data))

        mixed = np.zeros(max_length)
        for data in audio_data:
            if len(data) < max_length:
                data = np.pad(data, (0, max_length - len(data)))
            mixed += data

        if np.max(np.abs(mixed)) > 1.0:
            mixed = mixed / np.max(np.abs(mixed)) * 0.95

        out_dir = os.path.dirname(output_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        sf.write(output_path, mixed, sample_rate)
        logger.info(f"Merged {len(audio_files)} audio files → {output_path}")
        return output_path

    except ImportError:
        logger.error("soundfile not installed, cannot merge audio")
        raise
    except Exception as e:
        logger.error(f"Error merging audio: {e}")
        raise
