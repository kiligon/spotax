"""Scaffolding utilities for generating boilerplate code."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, PackageLoader, select_autoescape

from spotjax import __version__


def get_template_env() -> Environment:
    """Get the Jinja2 template environment."""
    return Environment(
        loader=PackageLoader("spotjax", "templates"),
        autoescape=select_autoescape(),
        keep_trailing_newline=True,
    )


def render_spotjax_utils() -> str:
    """Render the spotjax_utils.py template.

    Returns:
        The rendered Python code as a string.
    """
    env = get_template_env()
    template = env.get_template("spotjax_utils.py.j2")
    return template.render(version=__version__)


def generate_spotjax_utils(output_dir: Path, overwrite: bool = False) -> Path:
    """Generate spotjax_utils.py in the specified directory.

    Args:
        output_dir: Directory to write the file to.
        overwrite: If True, overwrite existing file.

    Returns:
        Path to the generated file.

    Raises:
        FileExistsError: If file exists and overwrite is False.
    """
    output_path = output_dir / "spotjax_utils.py"

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"{output_path} already exists. Use --force to overwrite."
        )

    content = render_spotjax_utils()
    output_path.write_text(content)

    return output_path
