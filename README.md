# Fine-Tuning Wav2Vec2 for Real-Time Deepfake Audio Detection

Local and SageMaker deployment and inference of a Wav2Vec2 model fine-tuned to detect deepfake audio. Complete details can be found in the accompanying [Fine-Tuning Wav2Vec2 for Real-Time Deepfake Audio Detection](https://garystafford.medium.com/fine-tuning-wav2vec2-for-real-time-deepfake-audio-detection-b72d7efebdd7).

## Create Python Environment

### Mac

```bash
brew install python@3.12

python3.12 -m pip install virtualenv --break-system-packages -Uq
python3.12 -m venv .venv
source .venv/bin/activate
python3.12 -m pip install pip -Uq
```

### Windows

```ini
py -3.12 -m venv .venv
.\.venv\Scripts\activate
# or posh
.\.venv\Scripts\Activate.ps1

python -m pip install pip -Uq
```

## Download Fine-Tuned Model

```python
from huggingface_hub import snapshot_download

repo_id="garystafford/wav2vec2-deepfake-voice-detector"
snapshot_download(repo_id, local_dir="wav2vec2-deepfake-voice-detector")

print(f"Model downloaded: {repo_id}")
```

## Download Associated Dataset

```python
from datasets import load_dataset

repo_id = "garystafford/deepfake-audio-detection"
dataset = load_dataset(repo_id)
print(f"\nLoaded dataset info:")
print(dataset)
```
