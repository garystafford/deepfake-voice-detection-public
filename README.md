# Deepfake Video Detection with Amazon SageMaker

## Create Python Environment for Notebook

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

## Download Models

```bash
git clone https://huggingface.co/Gustking/wav2vec2-large-xlsr-deepfake-audio-classification

git clone https://huggingface.co/garystafford/wav2vec2-deepfake-voice-detector
```

```ini
tensorboard --logdir .\logs\deepfake-xlsr --reload_interval 5
# then open http://localhost:6006/ in your browser
```

## Local Inference (CLI)

Run the fine-tuned model locally against any audio file (flac/wav/mp3).

### Usage

```powershell
# Activate the virtual environment first (Windows)
.\.venv\Scripts\Activate.ps1

# Run with a sample file and the checkpoint path
.\.venv\Scripts\python.exe local_test.py --audio youtube_chunks_flac\yt_0000_part_011.flac --model-path deepfake-xlsr-ft\checkpoint-50
```

### Options

- `--audio`: Path to your audio file.
- `--model-path`: Directory containing `model.safetensors`, `config.json`, `preprocessor_config.json`.
  - Defaults to `deepfake-xlsr-ft\checkpoint-50`.
- `--sr`: Target sample rate (default `16000`).

### Example

```powershell
#.venv Python ensures dependencies are available
.\.venv\Scripts\python.exe code\local_test.py --audio audio_samples\example.flac
```

The script prints JSON like:

```json
{
  "prediction": "fake",
  "confidence": 0.67,
  "probabilities": { "real": 0.33, "fake": 0.67 }
}
```

If you move or copy final weights to `model`, you can use:

```powershell
.\.venv\Scripts\python.exe code\local_test.py --audio audio_samples\example.flac --model-path .\model
```

## Fine Tuning Resources

- <https://highspark.co/famous-persuasive-speeches/>
- <https://localai.tools/voice-generator>
- <https://studio.speechify.com/library>
- <https://elevenlabs.io/app/speech-synthesis/text-to-speech>

## Linting

```bash
pip install autoflake black flake8 isort -Uq
pip install "black[jupyter]" -Uq

uvx autoflake --remove-all-unused-imports --in-place --recursive .
black .
isort .
flake8 .
```
