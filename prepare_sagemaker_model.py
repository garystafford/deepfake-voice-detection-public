"""Prepare model artifacts for SageMaker deployment."""

import argparse
import shutil
from pathlib import Path

# Target: SageMaker model structure
MODEL_DIR = Path("model")
CODE_DIR = Path("code")


def prepare_model_artifacts(checkpoint_path: Path):
    """Copy checkpoint files into SageMaker-compatible structure."""

    # Clean and create model directory
    if MODEL_DIR.exists():
        print(f"Removing existing {MODEL_DIR}/")
        shutil.rmtree(MODEL_DIR)
    MODEL_DIR.mkdir(exist_ok=True)

    print(f"Copying model artifacts from {checkpoint_path}/ to {MODEL_DIR}/\n")

    # Required files for SageMaker
    required_files = [
        "config.json",
        "preprocessor_config.json",
        "model.safetensors",
    ]

    for filename in required_files:
        src = checkpoint_path / filename
        dst = MODEL_DIR / filename

        if src.exists():
            shutil.copy2(src, dst)
            print(f"✓ Copied {filename}")
        else:
            print(f"✗ Missing {filename} in checkpoint!")

    # Verify code directory exists
    if not CODE_DIR.exists():
        print(f"\n⚠ {CODE_DIR}/ directory not found!")
        print("Make sure you have:")
        print(f"  - {CODE_DIR}/inference.py")
        print(f"  - {CODE_DIR}/requirements.txt")
    else:
        print(f"\n✓ {CODE_DIR}/ directory exists")
        if (CODE_DIR / "inference.py").exists():
            print(f"  ✓ inference.py found")
        else:
            print(f"  ✗ inference.py missing!")

        if (CODE_DIR / "requirements.txt").exists():
            print(f"  ✓ requirements.txt found")
        else:
            print(f"  ✗ requirements.txt missing!")

    print("\n" + "=" * 60)
    print("SageMaker deployment structure:")
    print("=" * 60)
    print("model/")
    print("├── config.json")
    print("├── preprocessor_config.json")
    print("└── model.safetensors")
    print("code/")
    print("├── inference.py")
    print("└── requirements.txt")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Verify the model/ directory contains all 3 files")
    print("2. Verify code/ contains inference.py and requirements.txt")
    print("3. Package for SageMaker: tar -czf model.tar.gz model/ code/")
    print("4. Upload to S3 and deploy!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare model artifacts for SageMaker deployment"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="wav2vec2-deepfake-voice-detector",
        help="Path to the model checkpoint directory (default: wav2vec2-deepfake-voice-detector)",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.model_path)

    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("\nAvailable checkpoints:")
        base_dir = Path("deepfake-xlsr-ft")
        if base_dir.exists():
            runs = sorted(
                base_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True
            )
            for run in runs[:5]:
                if run.is_dir():
                    print(f"  - {run.name}")
                    checkpoints = [
                        d
                        for d in run.iterdir()
                        if d.is_dir() and d.name.startswith("checkpoint-")
                    ]
                    for cp in checkpoints:
                        print(f"      └── {cp.name}")
        print(
            f"\nUsage: python {Path(__file__).name} --model-path <path-to-checkpoint>"
        )
    else:
        prepare_model_artifacts(checkpoint_path)
