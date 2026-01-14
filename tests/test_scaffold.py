"""Tests for SpotJAX scaffolding."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from spotjax import __version__
from spotjax.scaffold import generate_spotjax_utils, render_spotjax_utils


class TestRenderSpotjaxUtils:
    """Tests for render_spotjax_utils."""

    def test_renders_template(self):
        """Test that template renders without errors."""
        content = render_spotjax_utils()
        assert isinstance(content, str)
        assert len(content) > 1000  # Should be substantial

    def test_includes_version(self):
        """Test that rendered content includes version."""
        content = render_spotjax_utils()
        assert f"(v{__version__})" in content

    def test_includes_key_classes(self):
        """Test that rendered content includes key classes."""
        content = render_spotjax_utils()
        assert "class SpotConfig:" in content
        assert "class CheckpointManager:" in content

    def test_includes_key_functions(self):
        """Test that rendered content includes key functions."""
        content = render_spotjax_utils()
        assert "def get_config()" in content
        assert "def setup_distributed(" in content

    def test_includes_signal_handler(self):
        """Test that rendered content includes signal handling."""
        content = render_spotjax_utils()
        assert "SIGTERM" in content
        assert "setup_preemption_handler" in content

    def test_includes_environment_variables(self):
        """Test that rendered content documents environment variables."""
        content = render_spotjax_utils()
        assert "SPOT_CHECKPOINT_DIR" in content
        assert "SPOT_IS_RESTART" in content
        assert "SPOT_WORKER_ID" in content
        assert "JAX_COORDINATOR_ADDRESS" in content

    def test_valid_python_syntax(self):
        """Test that rendered content is valid Python."""
        content = render_spotjax_utils()
        # This will raise SyntaxError if invalid
        compile(content, "<string>", "exec")


class TestGenerateSpotjaxUtils:
    """Tests for generate_spotjax_utils."""

    def test_creates_file(self):
        """Test that file is created."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = generate_spotjax_utils(Path(tmpdir))
            assert output_path.exists()
            assert output_path.name == "spotjax_utils.py"

    def test_file_has_content(self):
        """Test that created file has expected content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = generate_spotjax_utils(Path(tmpdir))
            content = output_path.read_text()
            assert "CheckpointManager" in content
            assert __version__ in content

    def test_raises_if_exists(self):
        """Test that error is raised if file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create the file first
            output_path = Path(tmpdir) / "spotjax_utils.py"
            output_path.write_text("existing content")

            # Should raise
            with pytest.raises(FileExistsError, match="already exists"):
                generate_spotjax_utils(Path(tmpdir))

    def test_overwrites_with_force(self):
        """Test that file is overwritten with overwrite=True."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create the file first
            output_path = Path(tmpdir) / "spotjax_utils.py"
            output_path.write_text("existing content")

            # Should succeed with overwrite
            result = generate_spotjax_utils(Path(tmpdir), overwrite=True)
            assert result == output_path

            # Content should be new
            content = output_path.read_text()
            assert "existing content" not in content
            assert "CheckpointManager" in content

    def test_returns_correct_path(self):
        """Test that correct path is returned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = generate_spotjax_utils(Path(tmpdir))
            assert output_path == Path(tmpdir) / "spotjax_utils.py"
