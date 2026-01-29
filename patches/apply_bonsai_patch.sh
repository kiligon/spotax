#!/bin/bash
# Apply the reshard fix directly to the installed bonsai package
#
# Usage: ./patches/apply_bonsai_patch.sh
#
# This modifies the installed bonsai package in-place. Use the monkey-patch
# approach (import patches.bonsai_monkey_patch) if you prefer not to modify
# installed packages.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BONSAI_FILE=".venv/lib/python3.13/site-packages/bonsai/models/qwen3/modeling.py"

if [ ! -f "$BONSAI_FILE" ]; then
    echo "Error: Cannot find bonsai at $BONSAI_FILE"
    echo "Make sure you're in the project root directory."
    exit 1
fi

# Check if already patched
if grep -q "NamedSharding" "$BONSAI_FILE"; then
    echo "Bonsai appears to already be patched (NamedSharding found)."
    exit 0
fi

echo "Backing up original file..."
cp "$BONSAI_FILE" "$BONSAI_FILE.bak"

echo "Applying patch..."
# Replace the import
sed -i 's/from jax.sharding import PartitionSpec, get_abstract_mesh, reshard/from jax.sharding import PartitionSpec, get_abstract_mesh, NamedSharding/' "$BONSAI_FILE"

# Replace reshard() call with with_sharding_constraint (correct API inside JIT)
sed -i 's/return reshard(x, s)/return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, s))/' "$BONSAI_FILE"

echo "Patch applied successfully!"
echo "Original file backed up to: $BONSAI_FILE.bak"
echo ""
echo "To verify, check the shard() function:"
grep -A5 "def shard" "$BONSAI_FILE"
