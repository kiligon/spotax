#!/bin/bash
# SpotJAX pre-install script
# This runs before pip install -r requirements.txt
# Uses uv for fast, reliable package installation
# Expects to be run with the venv already activated

set -e

# Install bonsai from source (has broken package metadata - dynamic version fails)
if ! python -c "import bonsai" 2>/dev/null; then
    echo "Installing bonsai from source..."

    # Clean up any broken previous installs
    uv pip uninstall UNKNOWN 2>/dev/null || true
    uv pip uninstall jax-bonsai 2>/dev/null || true
    uv pip uninstall bonsai 2>/dev/null || true

    BONSAI_DIR="/tmp/bonsai-install"
    rm -rf "$BONSAI_DIR"
    git clone --depth 1 https://github.com/jax-ml/bonsai.git "$BONSAI_DIR"

    cd "$BONSAI_DIR"

    # Create __version__ in bonsai/__init__.py (setuptools tries to read this)
    echo '__version__ = "0.1.0"' >> bonsai/__init__.py

    # Fix broken pyproject.toml
    python3 << 'PYFIX'
import re

with open("pyproject.toml", "r") as f:
    content = f.read()

# Replace dynamic = ["version"] with version = "0.1.0"
content = re.sub(r'dynamic\s*=\s*\["version"\]', 'version = "0.1.0"', content)

# Remove [tool.setuptools.dynamic] section entirely
content = re.sub(r'\[tool\.setuptools\.dynamic\]\s*\n[^\[]*', '\n', content)

# Remove upper bound on JAX versions but keep lower bound (bonsai needs 0.8+)
content = re.sub(r'"jax\s*>=\s*([\d.]+),\s*<\s*[\d.]+"', r'"jax >= \1"', content)
content = re.sub(r'"jaxlib\s*>=\s*([\d.]+),\s*<\s*[\d.]+"', r'"jaxlib >= \1"', content)
content = re.sub(r'"flax\s*>=\s*([\d.]+),\s*<\s*[\d.]+"', r'"flax >= \1"', content)

with open("pyproject.toml", "w") as f:
    f.write(content)

print("Fixed pyproject.toml")
PYFIX

    # Fix reshard() incompatibility with Auto axis types
    # reshard() only works with AxisType.Explicit, but default meshes use Auto.
    # Replace reshard() with jax.device_put(x, NamedSharding(mesh, s)) which works with any axis type.
    QWEN3_FILE="bonsai/models/qwen3/modeling.py"
    if [ -f "$QWEN3_FILE" ]; then
        echo "Patching bonsai reshard() -> device_put() for Auto axis type compatibility..."
        sed -i 's/from jax.sharding import PartitionSpec, get_abstract_mesh, reshard/from jax.sharding import PartitionSpec, get_abstract_mesh, NamedSharding/' "$QWEN3_FILE"
        # Replace shard function with debug version
        python3 << 'PATCHSHARD'
import re

with open("bonsai/models/qwen3/modeling.py", "r") as f:
    content = f.read()

# Replace the shard function with a debug version
old_shard = '''def shard(x: jnp.ndarray, s: ShardingSpec):
    mesh = get_abstract_mesh()
    if not mesh.empty and len(mesh.axis_names) > 0:
        return reshard(x, s)
    return x'''

new_shard = '''def shard(x: jnp.ndarray, s: ShardingSpec):
    """Shard array with device_put (patched for Auto axis type compatibility)."""
    mesh = get_abstract_mesh()
    if not mesh.empty and len(mesh.axis_names) > 0:
        # Use jax.lax.with_sharding_constraint instead of device_put inside JIT
        # device_put can cause issues inside traced functions
        return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, s))
    return x'''

content = content.replace(old_shard, new_shard)

with open("bonsai/models/qwen3/modeling.py", "w") as f:
    f.write(content)

print("Patched shard() to use with_sharding_constraint")
PATCHSHARD
    fi

    # Install with uv into the active venv
    uv pip install .
fi
