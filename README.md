# Fine-Tuning Wav2Vec2 for Real-Time Deepfake Audio Detection

Local and SageMaker deployment and inference of a Wav2Vec2 model fine-tuned to detect deepfake audio. Complete details can be found in the accompanying [Fine-Tuning Wav2Vec2 for Real-Time Deepfake Audio Detection](https://garystafford.medium.com/fine-tuning-wav2vec2-for-real-time-deepfake-audio-detection-b72d7efebdd7).

Model: [garystafford/wav2vec2-deepfake-voice-detector](https://huggingface.co/garystafford/wav2vec2-deepfake-voice-detector)

## Create Python Virtual Environment

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

There are several audio samples available for testing the model in the [`audio_samples`](audio_samples/) directory:

## SageMaker Deployment

Code for deploying the fine-tuned model to AWS SageMaker is included in the [`sagemaker-deployment.ipynb`](sagemaker-deployment.ipynb) notebook. The artifacts are stored in the [code](code/) and [model](model/) directories.

## Download Fine-Tuned Model (Optional)

The Notebooks automatically download the fine-tuned model from Hugging Face Model Hub, but you can also manually download it using the following code:

```python
from huggingface_hub import snapshot_download

repo_id="garystafford/wav2vec2-deepfake-voice-detector"
snapshot_download(repo_id, local_dir="wav2vec2-deepfake-voice-detector")

print(f"Model downloaded: {repo_id}")
```

## Download Associated Dataset (Optional)

The dataset used for fine-tuning is available on Hugging Face Datasets Hub. You can download it using the following code:

```python
from datasets import load_dataset

repo_id = "garystafford/deepfake-audio-detection"
dataset = load_dataset(repo_id)
print(f"\nLoaded dataset info:")
print(dataset)
```
