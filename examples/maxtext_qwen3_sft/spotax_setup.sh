#!/bin/bash
# MaxText installation for SpotJAX TPU VMs.
#
# Clones MaxText, installs dependencies, and sets up for Qwen3 SFT.
# Runs before training on each new TPU VM (including after preemption).
set -e

MAXTEXT_DIR="$HOME/maxtext"

echo "=== MaxText Qwen3 SFT Setup ==="

# Clone MaxText (shallow clone for speed)
if [ ! -d "$MAXTEXT_DIR" ]; then
    echo "Cloning MaxText..."
    git clone --depth 1 https://github.com/AI-Hypercomputer/maxtext.git "$MAXTEXT_DIR"
else
    echo "MaxText already cloned"
fi

# Install MaxText with TPU support
echo "Installing MaxText..."
cd "$MAXTEXT_DIR"
uv pip install -e ".[tpu]" --resolution=lowest

# Install MaxText extra dependencies
echo "Installing MaxText extra deps..."
python -m MaxText.install_maxtext_extra_deps

# Install tunix (Google's post-training library, required for SFT)
echo "Installing tunix..."
uv pip install git+https://github.com/google/tunix

# Install PyTorch CPU (for HuggingFace -> MaxText checkpoint conversion)
# Use --extra-index-url so PyPI is still available as fallback
echo "Installing PyTorch CPU..."
uv pip install torch --extra-index-url https://download.pytorch.org/whl/cpu

echo "=== Setup Complete ==="
