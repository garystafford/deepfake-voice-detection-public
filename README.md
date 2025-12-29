# Fine-Tuning Wav2Vec2 for Real-Time Deepfake Audio Detection

Local and SageMaker deployment and inference of a Wav2Vec2 model fine-tuned to detect deepfake audio.

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

## Download Model

```bash
git clone https://huggingface.co/garystafford/wav2vec2-deepfake-voice-detector
```
