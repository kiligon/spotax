"""SSH-based cluster runner using asyncssh."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import asyncssh

from spotjax.utils.logging import print_status, print_warning


@dataclass
class CommandResult:
    """Result of running a command on a node."""

    node_id: int
    stdout: str
    stderr: str
    exit_code: int

    @property
    def success(self) -> bool:
        return self.exit_code == 0


class SSHConnectionError(Exception):
    """Error connecting to a node via SSH."""

    pass


class ClusterRunner:
    """Manages SSH connections to a TPU cluster.

    Uses asyncssh for fully async SSH operations. Connections are pooled
    and kept alive with keepalives to prevent disconnection during long
    operations.
    """

    DEFAULT_SSH_KEY_PATHS = [
        Path.home() / ".ssh" / "google_compute_engine",
        Path.home() / ".ssh" / "id_rsa",
        Path.home() / ".ssh" / "id_ed25519",
    ]

    def __init__(
        self,
        ips: list[str],
        username: str | None = None,
        ssh_key_path: Path | None = None,
        connect_timeout: int = 30,
        keepalive_interval: int = 15,
    ):
        """Initialize cluster runner.

        Args:
            ips: List of IP addresses for each node
            username: SSH username (default: current user or 'root')
            ssh_key_path: Path to SSH private key (default: auto-detect)
            connect_timeout: Timeout for SSH connections in seconds
            keepalive_interval: Interval for SSH keepalive packets
        """
        self.ips = ips
        self.username = username or os.environ.get("USER", "root")
        self.ssh_key_path = ssh_key_path or self._find_ssh_key()
        self.connect_timeout = connect_timeout
        self.keepalive_interval = keepalive_interval

        # Connection pool: node_id -> SSHClientConnection
        self._connections: dict[int, asyncssh.SSHClientConnection] = {}
        self._connect_lock = asyncio.Lock()

    def _find_ssh_key(self) -> Path:
        """Find an SSH key to use."""
        for key_path in self.DEFAULT_SSH_KEY_PATHS:
            if key_path.exists():
                return key_path
        raise SSHConnectionError(
            "No SSH key found. Expected one of: "
            + ", ".join(str(p) for p in self.DEFAULT_SSH_KEY_PATHS)
        )

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the cluster."""
        return len(self.ips)

    async def _connect_node(self, node_id: int, retries: int = 3) -> asyncssh.SSHClientConnection:
        """Connect to a single node with retries.

        Args:
            node_id: The node index
            retries: Number of connection attempts

        Returns:
            SSH connection

        Raises:
            SSHConnectionError: If all retries fail
        """
        ip = self.ips[node_id]
        last_error = None

        for attempt in range(retries):
            try:
                conn = await asyncio.wait_for(
                    asyncssh.connect(
                        ip,
                        username=self.username,
                        client_keys=[str(self.ssh_key_path)],
                        known_hosts=None,  # Skip host key verification for TPU VMs
                        keepalive_interval=self.keepalive_interval,
                        keepalive_count_max=3,
                    ),
                    timeout=self.connect_timeout,
                )
                return conn
            except asyncio.TimeoutError:
                last_error = f"Connection timeout after {self.connect_timeout}s"
            except asyncssh.Error as e:
                last_error = str(e)
            except OSError as e:
                last_error = str(e)

            if attempt < retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds
                print_warning(f"Node {node_id} connection failed, retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)

        raise SSHConnectionError(f"Failed to connect to node {node_id} ({ip}): {last_error}")

    async def connect_all(self) -> None:
        """Establish SSH connections to all nodes in parallel."""
        async with self._connect_lock:
            if self._connections:
                # Already connected
                return

            print_status(f"Connecting to {self.num_nodes} nodes via SSH...")

            # Connect to all nodes in parallel
            tasks = [self._connect_node(i) for i in range(self.num_nodes)]
            try:
                connections = await asyncio.gather(*tasks)
                for i, conn in enumerate(connections):
                    self._connections[i] = conn
                print_status(f"Connected to all {self.num_nodes} nodes", style="success")
            except SSHConnectionError:
                # Clean up any successful connections
                await self.close()
                raise

    async def run_command(
        self,
        cmd: str,
        node: int | None = None,
        env: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> dict[int, CommandResult]:
        """Run a command on all nodes (or a specific node) in parallel.

        Args:
            cmd: Command to run
            node: Specific node to run on (None = all nodes)
            env: Environment variables to set
            timeout: Command timeout in seconds (None = no timeout)

        Returns:
            Dict mapping node_id to CommandResult
        """
        if not self._connections:
            raise SSHConnectionError("Not connected. Call connect_all() first.")

        # Build command with environment variables
        if env:
            env_str = " ".join(f"{k}={v}" for k, v in env.items())
            full_cmd = f"env {env_str} {cmd}"
        else:
            full_cmd = cmd

        # Determine which nodes to run on
        node_ids = [node] if node is not None else list(range(self.num_nodes))

        async def run_on_node(node_id: int) -> CommandResult:
            conn = self._connections[node_id]
            try:
                if timeout:
                    result = await asyncio.wait_for(
                        conn.run(full_cmd, check=False),
                        timeout=timeout,
                    )
                else:
                    result = await conn.run(full_cmd, check=False)
                return CommandResult(
                    node_id=node_id,
                    stdout=result.stdout or "",
                    stderr=result.stderr or "",
                    exit_code=result.exit_status or 0,
                )
            except asyncio.TimeoutError:
                return CommandResult(
                    node_id=node_id,
                    stdout="",
                    stderr=f"Command timed out after {timeout}s",
                    exit_code=-1,
                )
            except asyncssh.Error as e:
                return CommandResult(
                    node_id=node_id,
                    stdout="",
                    stderr=str(e),
                    exit_code=-1,
                )

        # Run on all target nodes in parallel
        tasks = [run_on_node(n) for n in node_ids]
        results = await asyncio.gather(*tasks)

        return {r.node_id: r for r in results}

    async def run_command_streaming(
        self,
        cmd: str,
        env: dict[str, str] | None = None,
        stdout_callback: Callable[[int, str], None] | None = None,
        stderr_callback: Callable[[int, str], None] | None = None,
    ) -> dict[int, int]:
        """Run a command on all nodes with streaming output.

        Args:
            cmd: Command to run
            env: Environment variables to set
            stdout_callback: Called with (node_id, line) for each stdout line
            stderr_callback: Called with (node_id, line) for each stderr line

        Returns:
            Dict mapping node_id to exit_code
        """
        if not self._connections:
            raise SSHConnectionError("Not connected. Call connect_all() first.")

        # Build command with environment variables
        if env:
            env_str = " ".join(f"{k}={v}" for k, v in env.items())
            full_cmd = f"env {env_str} {cmd}"
        else:
            full_cmd = cmd

        async def run_on_node(node_id: int) -> int:
            conn = self._connections[node_id]
            try:
                async with conn.create_process(full_cmd) as process:
                    async def read_stream(stream, callback):
                        try:
                            async for line in stream:
                                if callback:
                                    callback(node_id, line.rstrip("\n"))
                        except asyncssh.Error:
                            pass

                    # Read stdout and stderr concurrently
                    await asyncio.gather(
                        read_stream(process.stdout, stdout_callback),
                        read_stream(process.stderr, stderr_callback),
                    )

                    await process.wait()
                    return process.exit_status or 0
            except asyncssh.Error as e:
                if stderr_callback:
                    stderr_callback(node_id, f"SSH error: {e}")
                return -1

        # Run on all nodes concurrently
        tasks = [run_on_node(i) for i in range(self.num_nodes)]
        results = await asyncio.gather(*tasks)

        return dict(enumerate(results))

    async def upload_directory(
        self,
        local_path: str | Path,
        remote_path: str = "/root/code",
        exclude: list[str] | None = None,
    ) -> None:
        """Sync a local directory to all nodes using rsync.

        Uses asyncio.create_subprocess_exec for true async subprocess handling.

        Args:
            local_path: Local directory to upload
            remote_path: Remote destination path
            exclude: Patterns to exclude from sync
        """
        local_path = Path(local_path).resolve()
        if not local_path.exists():
            raise FileNotFoundError(f"Local path does not exist: {local_path}")

        default_excludes = [
            ".git",
            "__pycache__",
            "*.pyc",
            ".venv",
            "venv",
            "node_modules",
            ".eggs",
            "*.egg-info",
            ".tox",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
        ]
        excludes = exclude or default_excludes

        print_status(f"Uploading {local_path} to {self.num_nodes} nodes...")

        async def rsync_to_node(node_id: int) -> bool:
            ip = self.ips[node_id]
            ssh_key = str(self.ssh_key_path)

            # Build rsync command
            cmd = [
                "rsync",
                "-az",
                "--delete",
                "-e",
                f"ssh -i {ssh_key} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null",
            ]

            # Add excludes
            for pattern in excludes:
                cmd.extend(["--exclude", pattern])

            # Source and destination
            cmd.append(f"{local_path}/")
            cmd.append(f"{self.username}@{ip}:{remote_path}/")

            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
                stdout, stderr = await process.communicate()

                if process.returncode != 0:
                    print_warning(f"rsync to node {node_id} failed: {stderr.decode()}")
                    return False
                return True
            except Exception as e:
                print_warning(f"rsync to node {node_id} error: {e}")
                return False

        # First, create the remote directory on all nodes
        await self.run_command(f"mkdir -p {remote_path}")

        # Rsync to all nodes in parallel
        tasks = [rsync_to_node(i) for i in range(self.num_nodes)]
        results = await asyncio.gather(*tasks)

        success_count = sum(results)
        if success_count == self.num_nodes:
            print_status(f"Upload complete to all {self.num_nodes} nodes", style="success")
        else:
            print_warning(f"Upload succeeded on {success_count}/{self.num_nodes} nodes")

    async def close(self) -> None:
        """Close all SSH connections."""
        async with self._connect_lock:
            for conn in self._connections.values():
                try:
                    conn.close()
                    await conn.wait_closed()
                except Exception:
                    pass
            self._connections.clear()

    async def __aenter__(self) -> ClusterRunner:
        await self.connect_all()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()
