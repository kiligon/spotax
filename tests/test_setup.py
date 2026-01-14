"""Tests for SpotJAX setup and prerequisites."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from spotjax.setup import (
    PrerequisiteResult,
    SetupChecker,
    check_prerequisites_quick,
)


class TestPrerequisiteResult:
    """Tests for PrerequisiteResult dataclass."""

    def test_create_passed_result(self):
        """Test creating a passed result."""
        result = PrerequisiteResult(
            name="Test Check",
            passed=True,
            message="All good",
        )
        assert result.name == "Test Check"
        assert result.passed is True
        assert result.message == "All good"
        assert result.fix_command is None

    def test_create_failed_result_with_fix(self):
        """Test creating a failed result with fix command."""
        result = PrerequisiteResult(
            name="Test Check",
            passed=False,
            message="Something wrong",
            fix_command="run this command",
        )
        assert result.passed is False
        assert result.fix_command == "run this command"


class TestSetupChecker:
    """Tests for SetupChecker class."""

    def test_ssh_key_paths_defined(self):
        """Test that SSH key paths are defined."""
        assert len(SetupChecker.SSH_KEY_PATHS) > 0
        assert Path.home() / ".ssh" / "google_compute_engine" in SetupChecker.SSH_KEY_PATHS

    def test_check_gcloud_installed_found(self):
        """Test gcloud check when gcloud is installed."""
        checker = SetupChecker()

        with patch("shutil.which") as mock_which:
            mock_which.return_value = "/usr/bin/gcloud"
            result = checker.check_gcloud_installed()

        assert result.passed is True
        assert "/usr/bin/gcloud" in result.message

    def test_check_gcloud_installed_not_found(self):
        """Test gcloud check when gcloud is not installed."""
        checker = SetupChecker()

        with patch("shutil.which") as mock_which:
            mock_which.return_value = None
            result = checker.check_gcloud_installed()

        assert result.passed is False
        assert result.fix_command is not None

    def test_check_ssh_key_found(self):
        """Test SSH key check when key exists."""
        checker = SetupChecker()

        with patch.object(Path, "exists") as mock_exists:
            mock_exists.return_value = True
            result = checker.check_ssh_key()

        assert result.passed is True

    def test_check_ssh_key_not_found(self):
        """Test SSH key check when no key exists."""
        checker = SetupChecker()

        with patch.object(Path, "exists") as mock_exists:
            mock_exists.return_value = False
            result = checker.check_ssh_key()

        assert result.passed is False
        assert "ssh-keygen" in result.fix_command

    def test_all_passed_true(self):
        """Test all_passed returns True when all checks pass."""
        checker = SetupChecker()
        checker.results = [
            PrerequisiteResult("Check 1", True, "OK"),
            PrerequisiteResult("Check 2", True, "OK"),
        ]
        assert checker.all_passed() is True

    def test_all_passed_false(self):
        """Test all_passed returns False when any check fails."""
        checker = SetupChecker()
        checker.results = [
            PrerequisiteResult("Check 1", True, "OK"),
            PrerequisiteResult("Check 2", False, "Failed"),
        ]
        assert checker.all_passed() is False


class TestCheckPrerequisitesQuick:
    """Tests for check_prerequisites_quick function."""

    def test_returns_tuple(self):
        """Test that function returns a tuple."""
        with patch.object(Path, "exists") as mock_exists:
            mock_exists.return_value = False
            result = check_prerequisites_quick()

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_fails_when_ssh_key_missing(self):
        """Test that check fails when SSH key is missing."""
        with patch.object(Path, "exists") as mock_exists:
            mock_exists.return_value = False
            passed, errors = check_prerequisites_quick()

        assert passed is False
        assert any("SSH key" in e for e in errors)
