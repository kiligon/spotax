"""Google Cloud Storage utilities."""

from __future__ import annotations

import asyncio

from google.api_core import exceptions as gcp_exceptions
from google.cloud import storage

from spotjax.utils.logging import print_status, print_warning


class StorageProvider:
    """Manages GCS bucket operations.

    All blocking GCS API calls are wrapped in asyncio.to_thread().
    """

    def __init__(self, project: str):
        """Initialize storage provider.

        Args:
            project: GCP project ID
        """
        self.project = project
        self._client: storage.Client | None = None

    @property
    def client(self) -> storage.Client:
        """Get or create GCS client."""
        if self._client is None:
            self._client = storage.Client(project=self.project)
        return self._client

    async def bucket_exists(self, bucket_name: str) -> bool:
        """Check if a bucket exists.

        Args:
            bucket_name: Name of the bucket

        Returns:
            True if bucket exists, False otherwise
        """

        def _check():
            try:
                bucket = self.client.bucket(bucket_name)
                bucket.reload() # why we need to reload?
                return True
            except gcp_exceptions.NotFound:
                return False

        return await asyncio.to_thread(_check)
#
    async def create_bucket(
        self,
        bucket_name: str,
        location: str = "US",
    ) -> bool:
        """Create a bucket if it doesn't exist.

        Args:
            bucket_name: Name of the bucket
            location: GCS location (default: US multi-region)

        Returns:
            True if bucket was created, False if it already existed
        """

        def _create():
            try:
                bucket = self.client.bucket(bucket_name)
                bucket.create(location=location)
                return True
            except gcp_exceptions.Conflict:
                # Bucket already exists
                return False

        return await asyncio.to_thread(_create)

    async def ensure_bucket(
        self,
        bucket_name: str,
        location: str = "US",
    ) -> None:
        """Ensure a bucket exists, creating it if necessary.

        Args:
            bucket_name: Name of the bucket
            location: GCS location for new buckets
        """
        if await self.bucket_exists(bucket_name):
            print_status(f"Using existing bucket: gs://{bucket_name}")
        else:
            print_status(f"Creating bucket: gs://{bucket_name}")
            try:
                await self.create_bucket(bucket_name, location) # don't we duble the function it's already create bucket if not exsis maybe it's ok
                print_status(f"Bucket created: gs://{bucket_name}", style="success")
            except Exception as e:
                print_warning(f"Failed to create bucket: {e}")
                raise


async def ensure_bucket_exists(
    bucket_name: str,
    project: str,
    location: str = "US",
) -> None:
    """Convenience function to ensure a bucket exists.

    Args:
        bucket_name: Name of the bucket
        project: GCP project ID
        location: GCS location for new buckets
    """
    provider = StorageProvider(project)
    await provider.ensure_bucket(bucket_name, location)
