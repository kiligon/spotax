"""
Monkey-patch for bonsai to fix reshard() incompatibility with Auto axis types.

The issue: bonsai's shard() function uses jax.sharding.reshard() which only
works with AxisType.Explicit meshes. When using the default AxisType.Auto,
reshard() fails with:
    ValueError: PartitionSpec passed to reshard cannot contain axis names
    that are of type Auto or Manual

The fix: Replace reshard() with jax.device_put() using NamedSharding, which
works with any axis type.

Usage:
    # Import this BEFORE importing bonsai models
    import patches.bonsai_monkey_patch

    # Now import and use bonsai normally
    from bonsai.models.qwen3 import modeling as qwen3
"""

import jax
from jax.sharding import NamedSharding, PartitionSpec, get_abstract_mesh


def _patched_shard(x, s: PartitionSpec):
    """Shard an array according to the given PartitionSpec.

    Uses jax.lax.with_sharding_constraint instead of reshard() to support
    meshes with Auto axis types (the default). reshard() only works with
    Explicit axis types.

    NOTE: with_sharding_constraint is the correct API to use inside JIT-traced
    functions. device_put should only be used outside of JIT context.
    """
    mesh = get_abstract_mesh()
    if not mesh.empty and len(mesh.axis_names) > 0:
        return jax.lax.with_sharding_constraint(x, NamedSharding(mesh, s))
    return x


def patch_bonsai_qwen3():
    """Apply the shard() fix to bonsai.models.qwen3.modeling"""
    try:
        from bonsai.models.qwen3 import modeling
        modeling.shard = _patched_shard
        print("[bonsai_monkey_patch] Patched bonsai.models.qwen3.modeling.shard()")
    except ImportError:
        pass


def patch_bonsai_gemma3():
    """Apply the shard() fix to bonsai.models.gemma3.modeling if it has the same issue"""
    try:
        from bonsai.models.gemma3 import modeling
        if hasattr(modeling, 'shard'):
            modeling.shard = _patched_shard
            print("[bonsai_monkey_patch] Patched bonsai.models.gemma3.modeling.shard()")
    except ImportError:
        pass


def patch_bonsai_llada():
    """Apply the shard() fix to bonsai.models.llada_8b.modeling if it has the same issue"""
    try:
        from bonsai.models.llada_8b import modeling
        if hasattr(modeling, 'shard'):
            modeling.shard = _patched_shard
            print("[bonsai_monkey_patch] Patched bonsai.models.llada_8b.modeling.shard()")
    except ImportError:
        pass


def patch_all():
    """Apply the shard() fix to all known bonsai models that use reshard()"""
    patch_bonsai_qwen3()
    patch_bonsai_gemma3()
    patch_bonsai_llada()


# Auto-apply patches on import
patch_all()
