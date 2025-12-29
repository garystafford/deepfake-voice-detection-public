"""SageMaker inference handler for deepfake audio detection.

This module provides inference handlers for deploying a Wav2Vec2-based deepfake
audio detection model on Amazon SageMaker. It accepts base64-encoded audio waveforms
via JSON requests and returns classification results.

Author: Gary Stafford
Date: 2025-12-22
"""

import base64
import json
import logging
import os
from os import PathLike
from typing import Any, Dict, Union

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

logger = logging.getLogger(__name__)

# -------------------
# Constants / config
# -------------------
SAMPLE_RATE = 16000
MAX_DURATION = 30  # seconds


# -------------------
# Model + helpers
# -------------------
class DeepfakeDetector:
    def __init__(self, model_path: Union[str, PathLike]):
        """Initialize the DeepfakeDetector with a Wav2Vec2 model.

        Args:
            model_path: Path to the model directory containing config.json,
                       model.safetensors, and preprocessor_config.json.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Model artifacts are expected under model/ inside model_path
        artifact_dir = os.path.join(str(model_path), "model")
        if not os.path.isdir(artifact_dir):
            artifact_dir = str(model_path)

        logger.info(f"Loading model from {artifact_dir}...")

        # Load local feature extractor without network calls
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(
            artifact_dir,
            local_files_only=True,
        )
        self.model = AutoModelForAudioClassification.from_pretrained(
            artifact_dir,
            local_files_only=True,
        ).to(self.device)
        self.model.eval()  # Set to evaluation mode
        logger.info("Model loaded successfully!")

        # Keep model-provided labels as-is.
        # Any downstream renaming/normalization should happen at the streaming server boundary.
        self.id2label = {}
        try:
            raw = getattr(getattr(self.model, "config", None), "id2label", None)
            if isinstance(raw, dict):
                for k, v in raw.items():
                    try:
                        idx = int(k)
                    except Exception:
                        continue
                    if v is None:
                        continue
                    self.id2label[idx] = str(v)
        except Exception:
            self.id2label = {}

    def preprocess_audio(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Prepare raw mono waveform for the model.

        Args:
            audio: Raw mono waveform as a 1D numpy array.
            sample_rate: Audio sampling rate in Hz. Must be 16000.

        Returns:
            Preprocessed float32 audio array truncated to MAX_DURATION seconds.

        Raises:
            ValueError: If sample_rate is not 16000 or audio is not mono (1D).
        """
        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"Invalid sample rate: {sample_rate} Hz. "
                f"Expected {SAMPLE_RATE} Hz for this model."
            )
        if audio.ndim != 1:
            raise ValueError(f"Audio must be mono (1D array), got {audio.ndim}D array.")
        if len(audio) == 0:
            raise ValueError("Audio array is empty.")

        # Ensure float32
        audio = audio.astype(np.float32, copy=False)

        # Truncate to max duration
        max_len = MAX_DURATION * SAMPLE_RATE
        if audio.shape[0] > max_len:
            audio = audio[:max_len]
        return audio

    def detect(self, audio: np.ndarray, sample_rate: int) -> Dict[str, Any]:
        """Detect if audio is real or deepfake from raw waveform.

        Args:
            audio: Raw mono waveform as a 1D numpy array.
            sample_rate: Audio sampling rate in Hz. Must be 16000.

        Returns:
            Dictionary containing:
                - prediction: Model label string (e.g., "real", "fake")
                - confidence: Float confidence score (0.0 to 1.0) for the predicted class
                - probabilities: Dict mapping model label strings to probability scores

        Raises:
            ValueError: If preprocessing or detection fails.
        """
        audio_array = self.preprocess_audio(audio, sample_rate)

        # Extract features
        inputs = self.feature_extractor(
            audio_array,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predictions = torch.softmax(logits, dim=1)

        # Get results
        predicted_class = torch.argmax(predictions, dim=1).item()
        confidence = predictions[0][predicted_class].item()

        # Use model-provided labels directly.
        pred_label = self.id2label.get(int(predicted_class), str(predicted_class))

        # Return per-class probabilities keyed by the model labels.
        probs: Dict[str, float] = {}
        try:
            row = predictions[0].detach().cpu().tolist()
            for idx, p in enumerate(row):
                label = self.id2label.get(int(idx), str(idx))
                probs[str(label)] = float(p)
        except Exception:
            probs = {}

        result = {
            "prediction": str(pred_label),
            "confidence": float(confidence),
            "probabilities": probs,
        }

        return result


# -------------------
# SageMaker handlers
# -------------------
def model_fn(model_dir: str) -> DeepfakeDetector:
    """Load and return the model for SageMaker inference.

    Called once when the SageMaker container is started. Loads the model from
    the unpacked model artifacts in model_dir.

    Args:
        model_dir: Path to the unpacked model artifacts directory containing
                  config.json, model.safetensors, and preprocessor_config.json.

    Returns:
        Initialized DeepfakeDetector instance ready for inference.
    """
    return DeepfakeDetector(model_path=model_dir)


def input_fn(request_body: Any, request_content_type: str) -> Dict[str, Any]:
    """Parse and decode incoming SageMaker inference request.

    Accepts JSON with base64-encoded raw float32 waveform:
    - {"audio_base64": "...", "sample_rate": 16000}

    Args:
        request_body: Raw request body (JSON string or bytes).
        request_content_type: MIME type of the request. Must be "application/json".

    Returns:
        Dictionary with keys:
            - waveform: Decoded numpy float32 array of audio samples
            - sample_rate: Integer sampling rate in Hz

    Raises:
        ValueError: If content type is not JSON, schema is invalid, or required fields are missing.
    """
    if request_content_type != "application/json":
        raise ValueError(
            f"Unsupported content type: '{request_content_type}'. "
            f"Expected 'application/json'."
        )

    payload = json.loads(request_body)

    if "audio_base64" not in payload:
        raise ValueError("Invalid JSON schema. Expected 'audio_base64'.")

    audio_b64 = payload["audio_base64"]
    sample_rate = int(payload["sample_rate"]) if "sample_rate" in payload else 0

    if not audio_b64:
        raise ValueError("Missing required field: 'audio_base64'.")
    if not sample_rate:
        raise ValueError("Missing required field: 'sample_rate'.")

    # Decode base64 raw float32 PCM
    try:
        audio_bytes = base64.b64decode(audio_b64, validate=True)
    except Exception as e:
        raise ValueError(f"Invalid base64 encoding: {e}")

    waveform = np.frombuffer(audio_bytes, dtype=np.float32)

    return {"waveform": waveform, "sample_rate": sample_rate}


def predict_fn(data: Dict[str, Any], model: DeepfakeDetector) -> Dict[str, Any]:
    """Run deepfake detection inference on the provided audio data.

    Args:
        data: Dictionary with "waveform" (np.ndarray) and "sample_rate" (int) keys.
        model: Initialized DeepfakeDetector instance.

    Returns:
        Dictionary containing prediction results (see DeepfakeDetector.detect for schema).
    """
    if "waveform" not in data or "sample_rate" not in data:
        raise ValueError("Data must contain 'waveform' and 'sample_rate' keys.")

    return model.detect(data["waveform"], data["sample_rate"])


def output_fn(prediction: Dict[str, Any], accept: str) -> tuple[str, str]:
    """Serialize prediction results for SageMaker response.

    Args:
        prediction: Dictionary containing detection results.
        accept: Client's Accept header value. Supports "application/json", "text/json", "*/*".

    Returns:
        Tuple of (body, content_type) where:
            - body: JSON-encoded string of prediction results
            - content_type: "application/json"
    """
    body = json.dumps(prediction)
    return body, "application/json"
