"""Tests for orchestrator retry logic."""

from __future__ import annotations

from spotjax.orchestrator import AttemptResult, RunResult


class TestRunResult:
    """Tests for RunResult enum."""

    def test_all_result_types_defined(self):
        """Test that all expected result types are defined."""
        assert RunResult.SUCCESS.value == "success"
        assert RunResult.PREEMPTED.value == "preempted"
        assert RunResult.SCRIPT_ERROR.value == "script_error"
        assert RunResult.INFRA_ERROR.value == "infra_error"
        assert RunResult.USER_CANCELLED.value == "user_cancelled"


class TestAttemptResult:
    """Tests for AttemptResult dataclass."""

    def test_create_success_result(self):
        """Test creating a success result."""
        result = AttemptResult(
            result=RunResult.SUCCESS,
            exit_code=0,
            message="Completed successfully",
        )
        assert result.result == RunResult.SUCCESS
        assert result.exit_code == 0
        assert result.message == "Completed successfully"

    def test_create_preempted_result(self):
        """Test creating a preempted result."""
        result = AttemptResult(
            result=RunResult.PREEMPTED,
            exit_code=1,
            message="SSH connection lost",
        )
        assert result.result == RunResult.PREEMPTED
        assert result.exit_code == 1

    def test_create_script_error_result(self):
        """Test creating a script error result."""
        result = AttemptResult(
            result=RunResult.SCRIPT_ERROR,
            exit_code=137,
            message="Script exited with code 137",
        )
        assert result.result == RunResult.SCRIPT_ERROR
        assert result.exit_code == 137

    def test_default_message(self):
        """Test that message defaults to empty string."""
        result = AttemptResult(result=RunResult.SUCCESS, exit_code=0)
        assert result.message == ""


class TestRetryLogic:
    """Tests for retry decision logic."""

    def test_retriable_results(self):
        """Test which results are retriable."""
        # These should trigger retry
        retriable = {RunResult.PREEMPTED, RunResult.INFRA_ERROR, RunResult.SCRIPT_ERROR}

        for result_type in retriable:
            assert result_type in retriable, f"{result_type} should be retriable"

    def test_non_retriable_results(self):
        """Test which results should not trigger retry."""
        non_retriable = {RunResult.SUCCESS, RunResult.USER_CANCELLED}

        for result_type in non_retriable:
            assert result_type in non_retriable, f"{result_type} should not be retriable"
