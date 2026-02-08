#!/bin/bash
# SpotJAX pre-install script
# This runs before pip install -r requirements.txt
# Uses uv for fast, reliable package installation
# Expects to be run with the venv already activated

echo "=== SpotJAX Setup ==="

# Install gcsfuse if not available
install_gcsfuse() {
    # Check if already installed
    if command -v gcsfuse &> /dev/null; then
        echo "gcsfuse: already installed ($(gcsfuse --version 2>&1 | head -1))"
        return 0
    fi

    # Check common paths where gcsfuse might be installed
    for path in /usr/bin/gcsfuse /usr/local/bin/gcsfuse /snap/bin/gcsfuse; do
        if [ -x "$path" ]; then
            echo "gcsfuse: found at $path"
            export PATH="$(dirname $path):$PATH"
            return 0
        fi
    done

    echo "gcsfuse: not found, installing..."

    # Determine OS release
    RELEASE=$(lsb_release -c -s 2>/dev/null || echo "jammy")

    # Clean up any old broken gcsfuse repo entries
    sudo rm -f /etc/apt/sources.list.d/gcsfuse*.list 2>/dev/null || true

    # Try method 1: Download and install deb directly (most reliable)
    echo "  Trying direct deb install..."
    GCSFUSE_VERSION="2.5.1"  # Latest stable version
    DEB_URL="https://github.com/GoogleCloudPlatform/gcsfuse/releases/download/v${GCSFUSE_VERSION}/gcsfuse_${GCSFUSE_VERSION}_amd64.deb"

    # Use unique temp file to avoid race conditions on multi-node pods
    DEB_FILE="/tmp/gcsfuse_$$.deb"

    if curl -fsSL -o "$DEB_FILE" "$DEB_URL" 2>/dev/null; then
        if sudo dpkg -i "$DEB_FILE" 2>/dev/null; then
            rm -f "$DEB_FILE"
            if command -v gcsfuse &> /dev/null; then
                echo "gcsfuse: installed successfully via deb"
                return 0
            fi
        fi
        rm -f "$DEB_FILE"
    fi

    # Try method 2: apt with proper GPG key
    echo "  Trying apt install..."

    # Download and install GPG key properly
    sudo mkdir -p /usr/share/keyrings
    curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | \
        sudo gpg --yes --dearmor -o /usr/share/keyrings/cloud.google.gpg 2>/dev/null

    # Add repository with signed-by
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt gcsfuse-${RELEASE} main" | \
        sudo tee /etc/apt/sources.list.d/gcsfuse.list > /dev/null

    sudo apt-get update -qq 2>/dev/null
    sudo apt-get install -y -qq gcsfuse 2>/dev/null

    if command -v gcsfuse &> /dev/null; then
        echo "gcsfuse: installed successfully via apt"
        return 0
    fi

    echo "gcsfuse: installation failed"
    return 1
}

# Install bonsai from source
install_bonsai() {
    if python -c "import bonsai" 2>/dev/null; then
        echo "bonsai: already installed"
        return 0
    fi

    echo "bonsai: installing from source..."

    # Clean up any broken previous installs
    uv pip uninstall UNKNOWN 2>/dev/null || true
    uv pip uninstall jax-bonsai 2>/dev/null || true
    uv pip uninstall bonsai 2>/dev/null || true

    BONSAI_DIR="/tmp/bonsai-install-$$"
    rm -rf "$BONSAI_DIR"

    echo "  Cloning jax-ml/bonsai..."
    git clone --depth 1 --quiet https://github.com/jax-ml/bonsai.git "$BONSAI_DIR"

    cd "$BONSAI_DIR"

    # Create __version__ in bonsai/__init__.py (setuptools tries to read this)
    echo '__version__ = "0.1.0"' >> bonsai/__init__.py

    # Install with uv into the active venv
    echo "  Installing..."
    uv pip install . --quiet

    # Cleanup
    rm -rf "$BONSAI_DIR"

    if python -c "import bonsai" 2>/dev/null; then
        echo "bonsai: installed successfully"
        return 0
    else
        echo "bonsai: installation FAILED"
        return 1
    fi
}

# Run installations
gcsfuse_ok=false
if install_gcsfuse; then
    gcsfuse_ok=true
else
    echo "WARNING: gcsfuse installation failed - GCS data loading will not work"
fi

if ! install_bonsai; then
    echo "ERROR: bonsai is required"
    exit 1
fi

if [ "$gcsfuse_ok" = false ]; then
    echo ""
    echo "NOTE: gcsfuse is not installed. You can install it manually with:"
    echo "  curl -fsSL -o /tmp/gcsfuse.deb https://github.com/GoogleCloudPlatform/gcsfuse/releases/download/v2.5.1/gcsfuse_2.5.1_amd64.deb"
    echo "  sudo dpkg -i /tmp/gcsfuse.deb"
fi

echo "=== Setup Complete ==="
