# Fine-Tuning Wav2Vec2 for Real-Time Deepfake Audio Detection

![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)
![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.57+-yellow.svg)
![Hugging Face Model](https://img.shields.io/badge/🤗%20Model-wav2vec2-orange.svg)

## Overview

This project demonstrates fine-tuning and deploying a Wav2Vec2 model for real-time deepfake audio detection. The model classifies audio samples as either real or AI-generated (fake) speech with high accuracy. You can run inference locally, in Google Colab, or deploy to AWS SageMaker for production use.

Complete details can be found in the accompanying blog post: [Fine-Tuning Wav2Vec2 for Real-Time Deepfake Audio Detection](https://garystafford.medium.com/fine-tuning-wav2vec2-for-real-time-deepfake-audio-detection-b72d7efebdd7).

- Fine-tuned Model: [garystafford/wav2vec2-deepfake-voice-detector](https://huggingface.co/garystafford/wav2vec2-deepfake-voice-detector)
- Base Model: [Gustking/wav2vec2-large-xlsr-deepfake-audio-classification](https://huggingface.co/Gustking/wav2vec2-large-xlsr-deepfake-audio-classification)
- Dataset: [garystafford/deepfake-audio-detection](https://huggingface.co/datasets/garystafford/deepfake-audio-detection)

## Prerequisites

- **Python**: 3.12 or higher
- **Memory**: Minimum 8GB RAM (16GB+ recommended for faster inference)
- **GPU**: Optional but recommended for faster processing
  - CUDA-compatible GPU with PyTorch CUDA support
  - CPU-only mode works but is slower
- **Audio Formats**: Supports MP3, WAV, and FLAC files
- **Dependencies**: Listed in [requirements.txt](requirements.txt)

## Quick Start

Here's a minimal example to run inference on an audio file:

```python
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import librosa

# Load model
model_name = "garystafford/wav2vec2-deepfake-voice-detector"
model = AutoModelForAudioClassification.from_pretrained(model_name)
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device).eval()

# Run inference
audio, sr = librosa.load("audio_file.mp3", sr=16000, mono=True)
inputs = feature_extractor(audio, sampling_rate=16000, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)

print(f"Real: {probs[0][0]:.2%}, Fake: {probs[0][1]:.2%}")
```

**Expected Output:**

```json
{
  "prediction": "fake",
  "confidence": 0.95,
  "probabilities": {
    "real": 0.05,
    "fake": 0.95
  }
}
```

For detailed examples and batch processing, see the [notebooks](#notebook-overview) below.

## Installation

### Create Python Virtual Environment

Common Notebook dependencies are installed from `requirements.txt`.

### Mac

```bash
brew install python@3.12

python3.12 -m pip install virtualenv --break-system-packages -Uq
python3.12 -m venv .venv
source .venv/bin/activate

python3.12 -m pip install pip -Uq

python3.12 -m pip install -r requirements.txt -Uq
```

### Windows

```ini
py -3.12 -m venv .venv
.\.venv\Scripts\activate
# or posh
.\.venv\Scripts\Activate.ps1

python -m pip install pip -Uq

python -m pip install -r requirements.txt -Uq
```

## Notebook Overview

1. [Local Inference Notebook](local-inference.ipynb): Run inference on your own audio files using the fine-tuned model.
2. [Colab Inference Notebook](colab-inference.ipynb): A Google Colab version of the local inference notebook for easy access without local setup.
3. [SageMaker Deployment Notebook](sagemaker-deployment.ipynb): Deploy the fine-tuned model to AWS SageMaker for scalable inference.

## Audio Samples

Several audio samples are available for testing the model in the [`audio_samples`](audio_samples/) directory. These samples can be used with any of the inference notebooks to verify model performance.

## SageMaker Deployment

Deploy the fine-tuned model to AWS SageMaker for scalable, production-grade inference. The deployment process is detailed in the [`sagemaker-deployment.ipynb`](sagemaker-deployment.ipynb) notebook.

### Deployment Artifacts

- **`code/inference.py`** - Custom inference handler with SageMaker hooks (`model_fn`, `input_fn`, `predict_fn`, `output_fn`) for handling requests
- **`code/requirements.txt`** - Runtime dependencies for the SageMaker container
- **`model/`** - Unpacked model artifacts including:
  - `config.json` - Model configuration
  - `model.safetensors` - Model weights
  - `preprocessor_config.json` - Feature extractor configuration
- **`model.tar.gz`** - Packaged archive containing both `model/` and `code/` directories, ready for SageMaker deployment

## Download Fine-Tuned Model (Optional)

The Notebooks automatically download the fine-tuned model from Hugging Face Model Hub, but you can also manually download it to your local cache using the following code:

```python
from huggingface_hub import snapshot_download

repo_id="garystafford/wav2vec2-deepfake-voice-detector"
snapshot_download(repo_id)

print(f"Model downloaded: {repo_id}")
```

## Download Associated Dataset (Optional)

The dataset used for fine-tuning is available on Hugging Face Datasets Hub. You can download it to your local cache using the following code:

```python
from datasets import load_dataset

repo_id = "garystafford/deepfake-audio-detection"
dataset = load_dataset(repo_id)
print(f"\nLoaded dataset info:")
print(dataset)
```

## Troubleshooting

### Out of Memory Issues

If you encounter memory errors during inference:

- **Reduce audio duration**: Truncate audio files to 30 seconds or less
- **Close other applications**: Free up system memory
- **Use CPU mode**: Set `device = "cpu"` if GPU memory is limited
- **Process in batches**: Instead of loading multiple files at once, process one at a time

### Dependency Installation Problems

If installation fails:

- Ensure Python 3.12+ is installed: `python --version`
- Update pip: `python -m pip install --upgrade pip`
- For Mac M1/M2 users: Use native arm64 Python, not Rosetta
- For Windows users: Install Visual C++ Build Tools if needed for some packages

### Audio Format Issues

If audio files fail to load:

- **Unsupported format**: Convert to MP3, WAV, or FLAC using FFmpeg or online converters
- **Sample rate mismatch**: The model expects 16kHz audio (librosa automatically resamples)
- **Corrupted files**: Verify file integrity and try re-downloading

### Model Download Failures

If model download from Hugging Face fails:

- Check your internet connection
- Verify Hugging Face Hub is accessible: `https://huggingface.co/`
- Try manual download using the code in the [Download Fine-Tuned Model](#download-fine-tuned-model-optional) section
- Check disk space for model cache (models are ~1.2GB)

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
