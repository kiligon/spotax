"""Tests for GCS storage utilities."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from spotjax.infra.storage import StorageProvider


class TestStorageProvider:
    """Tests for StorageProvider class."""

    def test_init(self):
        """Test initialization."""
        provider = StorageProvider(project="test-project")
        assert provider.project == "test-project"
        assert provider._client is None

    @pytest.mark.asyncio
    async def test_bucket_exists_true(self):
        """Test bucket_exists returns True when bucket exists."""
        provider = StorageProvider(project="test-project")

        mock_bucket = MagicMock()
        mock_bucket.reload.return_value = None

        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        provider._client = mock_client

        result = await provider.bucket_exists("test-bucket")

        assert result is True
        mock_client.bucket.assert_called_once_with("test-bucket")
        mock_bucket.reload.assert_called_once()

    @pytest.mark.asyncio
    async def test_bucket_exists_false(self):
        """Test bucket_exists returns False when bucket doesn't exist."""
        from google.api_core import exceptions as gcp_exceptions

        provider = StorageProvider(project="test-project")

        mock_bucket = MagicMock()
        mock_bucket.reload.side_effect = gcp_exceptions.NotFound("Not found")

        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        provider._client = mock_client

        result = await provider.bucket_exists("nonexistent-bucket")

        assert result is False

    @pytest.mark.asyncio
    async def test_create_bucket_success(self):
        """Test create_bucket returns True when bucket is created."""
        provider = StorageProvider(project="test-project")

        mock_bucket = MagicMock()

        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        provider._client = mock_client

        result = await provider.create_bucket("new-bucket", location="US")

        assert result is True
        mock_bucket.create.assert_called_once_with(location="US")

    @pytest.mark.asyncio
    async def test_create_bucket_already_exists(self):
        """Test create_bucket returns False when bucket already exists."""
        from google.api_core import exceptions as gcp_exceptions

        provider = StorageProvider(project="test-project")

        mock_bucket = MagicMock()
        mock_bucket.create.side_effect = gcp_exceptions.Conflict("Already exists")

        mock_client = MagicMock()
        mock_client.bucket.return_value = mock_bucket
        provider._client = mock_client

        result = await provider.create_bucket("existing-bucket")

        assert result is False
