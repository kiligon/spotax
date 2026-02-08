"""Google Cloud TPU infrastructure provider."""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass
from enum import Enum
from google.api_core import exceptions as gcp_exceptions
from google.cloud import tpu_v2

from spotax.utils.logging import console, create_progress, print_status, print_warning

class QueuedResourceState(str, Enum):
    """States for a QueuedResource."""

    STATE_UNSPECIFIED = "STATE_UNSPECIFIED"
    CREATING = "CREATING"
    ACCEPTED = "ACCEPTED"
    PROVISIONING = "PROVISIONING"
    FAILED = "FAILED"
    DELETING = "DELETING"
    ACTIVE = "ACTIVE"
    SUSPENDING = "SUSPENDING"
    SUSPENDED = "SUSPENDED"
    WAITING_FOR_RESOURCES = "WAITING_FOR_RESOURCES"


@dataclass
class TPUNode:
    """Represents a single node in a TPU slice."""

    name: str
    internal_ip: str
    external_ip: str | None = None


class TPUProviderError(Exception):
    """Base exception for TPU provider errors."""

    pass


class TPUCreationError(TPUProviderError):
    """Error during TPU creation."""

    pass


class TPUNotFoundError(TPUProviderError):
    """TPU resource not found."""

    pass


class TPUProvider:
    """Manages TPU VM lifecycle via Google Cloud API.

    All blocking GCP API calls are wrapped in asyncio.to_thread() to avoid
    blocking the event loop (important for SSH keepalives).
    """

    def __init__(self, project: str, zone: str):
        """Initialize TPU provider.

        Args:
            project: GCP project ID
            zone: GCP zone (e.g., us-central2-b)
        """
        self.project = project
        self.zone = zone
        self.region = "-".join(zone.split("-")[:-1])  # us-central2-b -> us-central2

        # Lazy-init clients to avoid blocking on import
        self._tpu_client: tpu_v2.TpuClient | None = None

    @property
    def tpu_client(self) -> tpu_v2.TpuClient:
        """Get or create TPU client."""
        if self._tpu_client is None:
            self._tpu_client = tpu_v2.TpuClient()
        return self._tpu_client

    def _get_parent(self) -> str:
        """Get the parent resource path."""
        return f"projects/{self.project}/locations/{self.zone}"

    def _get_node_name(self, name: str) -> str:
        """Get the full node resource path."""
        return f"{self._get_parent()}/nodes/{name}"

    def _get_queued_resource_name(self, name: str) -> str:
        """Get the full queued resource path."""
        return f"{self._get_parent()}/queuedResources/{name}"

    async def create_tpu(
        self,
        name: str,
        tpu_type: str,
        spot: bool = True,
        software_version: str = "tpu-ubuntu2204-base",
    ) -> str:
        """Create a TPU using the Queued Resources API.

        Args:
            name: Name for the TPU resource
            tpu_type: TPU type (e.g., v4-8, v4-32)
            spot: Whether to use spot/preemptible instances
            software_version: TPU software version

        Returns:
            The queued resource name

        Raises:
            TPUCreationError: If creation fails
        """
        queued_resource_id = name

        # Build the node spec
        node_spec = tpu_v2.QueuedResource.Tpu.NodeSpec(
            parent=self._get_parent(),
            node_id=name,
            node=tpu_v2.Node(
                accelerator_type=tpu_type,
                runtime_version=software_version,
                scheduling_config=tpu_v2.SchedulingConfig(
                    preemptible=spot,
                    reserved=False,
                ),
                # Enable external IPs for SSH access
                network_config=tpu_v2.NetworkConfig(
                    enable_external_ips=True,
                ),
                # Metadata for SSH keys and autocheckpoint
                metadata={
                    "enable-oslogin": "true",
                    "autocheckpoint-enabled": "true",
                },
            ),
        )

        # Build the queued resource request
        queued_resource = tpu_v2.QueuedResource(
            tpu=tpu_v2.QueuedResource.Tpu(node_spec=[node_spec]),
            # Use spot queueing for preemptible instances
            spot=tpu_v2.QueuedResource.Spot() if spot else None,
        )

        request = tpu_v2.CreateQueuedResourceRequest(
            parent=self._get_parent(),
            queued_resource_id=queued_resource_id,
            queued_resource=queued_resource,
        )

        print_status(f"Creating TPU {name} ({tpu_type}) in {self.zone}...")

        try:
            # Run blocking API call in thread
            operation = await asyncio.to_thread(
                self.tpu_client.create_queued_resource, request=request
            )
            # Wait for the operation to complete (just the request, not provisioning)
            result = await asyncio.to_thread(operation.result, timeout=300)
            return result.name
        except gcp_exceptions.AlreadyExists:
            print_warning(f"TPU {name} already exists, attempting to reuse")
            return self._get_queued_resource_name(name)
        except Exception as e:
            raise TPUCreationError(f"Failed to create TPU: {e}") from e

    async def wait_for_ready(
        self,
        name: str,
        timeout: int = 3600,
        poll_interval: int = 10,
    ) -> bool:
        """Wait for TPU to become ready.

        Args:
            name: Queued resource name (full path or just the ID)
            timeout: Maximum time to wait in seconds
            poll_interval: Time between status checks in seconds

        Returns:
            True if TPU is ready, False if timeout

        Raises:
            TPUCreationError: If TPU creation fails
        """
        # Normalize name to full path
        if not name.startswith("projects/"):
            name = self._get_queued_resource_name(name)

        request = tpu_v2.GetQueuedResourceRequest(name=name)

        elapsed = 0
        last_state = None

        with create_progress() as progress:
            task = progress.add_task("Waiting for TPU...", total=None)

            while elapsed < timeout:
                try:
                    resource = await asyncio.to_thread(
                        self.tpu_client.get_queued_resource, request=request
                    )
                except gcp_exceptions.NotFound as e:
                    raise TPUNotFoundError(f"Queued resource {name} not found") from e

                state = resource.state.state
                state_name = QueuedResourceState(state.name).value

                # Update progress message if state changed
                if state_name != last_state:
                    last_state = state_name
                    progress.update(task, description=f"TPU state: {state_name}")
                    console.print(f"  [dim]TPU state: {state_name}[/dim]")

                if state == tpu_v2.QueuedResourceState.State.ACTIVE:
                    progress.update(task, description="TPU ready!")
                    return True

                if state == tpu_v2.QueuedResourceState.State.FAILED:
                    # Get failure reason
                    failure_msg = "Unknown failure"
                    if resource.state.state_initiator:
                        failure_msg = str(resource.state.state_initiator)
                    raise TPUCreationError(f"TPU creation failed: {failure_msg}")

                await asyncio.sleep(poll_interval)
                elapsed += poll_interval

        return False

    async def get_node_internal_ips(self, name: str) -> list[str]:
        """Get internal IPs of all nodes in a TPU slice.

        Args:
            name: Node name (just the ID, not full path)

        Returns:
            List of internal IP addresses, one per node
        """
        full_name = self._get_node_name(name)
        request = tpu_v2.GetNodeRequest(name=full_name)

        try:
            node = await asyncio.to_thread(self.tpu_client.get_node, request=request)
        except gcp_exceptions.NotFound as e:
            raise TPUNotFoundError(f"TPU node {name} not found") from e

        ips = []
        for endpoint in node.network_endpoints:
            if endpoint.ip_address:
                ips.append(endpoint.ip_address)

        if not ips:
            raise TPUProviderError(f"No internal IPs found for TPU {name}")

        return ips

    async def get_node_external_ips(self, name: str) -> list[str]:
        """Get external IPs of all nodes in a TPU slice.

        Args:
            name: Node name (just the ID, not full path)

        Returns:
            List of external IP addresses, one per node
        """
        full_name = self._get_node_name(name)
        request = tpu_v2.GetNodeRequest(name=full_name)

        try:
            node = await asyncio.to_thread(self.tpu_client.get_node, request=request)
        except gcp_exceptions.NotFound as e:
            raise TPUNotFoundError(f"TPU node {name} not found") from e

        ips = []
        for endpoint in node.network_endpoints:
            # access_config may be a single object or a list depending on API version
            access_configs = endpoint.access_config
            if not isinstance(access_configs, (list, tuple)):
                access_configs = [access_configs] if access_configs else []
            for access_config in access_configs:
                if access_config and access_config.external_ip:
                    ips.append(access_config.external_ip)

        if not ips:
            raise TPUProviderError(f"No external IPs found for TPU {name}")

        return ips

    async def delete_tpu(self, name: str) -> None:
        """Delete a TPU and its queued resource.

        Args:
            name: Resource name (just the ID)
        """
        print_status(f"Deleting TPU {name}...")

        # For ACTIVE TPUs, must delete node first, then queued resource
        node_name = self._get_node_name(name)
        queued_name = self._get_queued_resource_name(name)

        # Try to delete the node first
        try:
            node_request = tpu_v2.DeleteNodeRequest(name=node_name)
            operation = await asyncio.to_thread(
                self.tpu_client.delete_node, request=node_request
            )
            await asyncio.to_thread(operation.result, timeout=600)
        except gcp_exceptions.NotFound:
            pass  # Node doesn't exist, continue to queued resource
        except Exception as e:
            print_warning(f"Error deleting node {name}: {e}")

        # Then delete the queued resource
        try:
            queued_request = tpu_v2.DeleteQueuedResourceRequest(name=queued_name)
            operation = await asyncio.to_thread(
                self.tpu_client.delete_queued_resource, request=queued_request
            )
            await asyncio.to_thread(operation.result, timeout=600)
            print_status(f"TPU {name} deleted", style="success")
        except gcp_exceptions.NotFound:
            print_status(f"TPU {name} deleted", style="success")
        except Exception as e:
            print_warning(f"Error deleting queued resource {name}: {e}")

    async def close(self) -> None:
        """Close the TPU client."""
        if self._tpu_client is not None:
            # Client doesn't have async close, but we should clean up
            self._tpu_client = None


def generate_tpu_name(prefix: str = "spotax") -> str:
    """Generate a unique TPU name."""
    short_uuid = uuid.uuid4().hex[:8]
    return f"{prefix}-{short_uuid}"


