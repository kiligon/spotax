"""Logging utilities for SpotJAX using Rich."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TextIO

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.style import Style
from rich.theme import Theme

# Custom theme for SpotJAX
SPOTJAX_THEME = Theme(
    {
        "info": "cyan",
        "warning": "yellow",
        "error": "red bold",
        "success": "green",
        "node": "magenta",
        "dim": "dim",
    }
)

# Global console instance
console = Console(theme=SPOTJAX_THEME, stderr=True)


def get_log_dir(base_dir: Path | None = None) -> Path:
    """Get the local log directory for this session."""
    if base_dir is None:
        base_dir = Path.home() / ".spotjax" / "logs"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = base_dir / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


class NodeLogger:
    """Logger that prefixes output with node ID and writes to file."""

    # Color palette for different nodes
    NODE_COLORS = [
        "bright_blue",
        "bright_green",
        "bright_yellow",
        "bright_magenta",
        "bright_cyan",
        "bright_red",
        "blue",
        "green",
    ]

    def __init__(
        self,
        node_id: int,
        log_dir: Path,
        stream_to_console: bool = False,
        single_node: bool = False,
    ):
        """Initialize node logger.

        Args:
            node_id: The node index (0, 1, 2, ...)
            log_dir: Directory to write log files
            stream_to_console: Whether to also print to console
            single_node: If True, skip node prefix (for single-VM setups)
        """
        self.node_id = node_id
        self.stream_to_console = stream_to_console
        self.log_dir = log_dir
        self.single_node = single_node

        # Create log file
        self.log_file = log_dir / ("output.log" if single_node else f"node-{node_id}.log")
        self._file_handle: TextIO | None = None

        # Node styling (only used for multi-node)
        color = self.NODE_COLORS[node_id % len(self.NODE_COLORS)]
        self.style = Style(color=color) if not single_node else None
        self.prefix = "" if single_node else f"[node-{node_id}]"

    def open(self) -> None:
        """Open the log file for writing."""
        if self._file_handle is None:
            # Intentionally not using context manager - lifecycle managed by open()/close()
            self._file_handle = open(self.log_file, "w", buffering=1)  # noqa: SIM115

    def close(self) -> None:
        """Close the log file."""
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None

    def write(self, text: str) -> None:
        """Write text to log file and optionally console."""
        # Write to file
        if self._file_handle is not None:
            timestamp = datetime.now().strftime("%H:%M:%S")
            for line in text.splitlines():
                self._file_handle.write(f"[{timestamp}] {line}\n")

        # Write to console if streaming
        if self.stream_to_console:
            for line in text.splitlines():
                if self.single_node:
                    console.print(line, highlight=False)
                else:
                    console.print(f"{self.prefix} {line}", style=self.style, highlight=False)

    def write_stderr(self, text: str) -> None:
        """Write stderr text (highlighted differently)."""
        # Write to file
        if self._file_handle is not None:
            timestamp = datetime.now().strftime("%H:%M:%S")
            for line in text.splitlines():
                self._file_handle.write(f"[{timestamp}] [stderr] {line}\n")

        # Write to console if streaming
        if self.stream_to_console:
            for line in text.splitlines():
                if self.single_node:
                    console.print(line, style=Style(color="red", dim=True), highlight=False)
                else:
                    console.print(
                        f"{self.prefix} {line}",
                        style=Style(color="red", dim=True),
                        highlight=False,
                    )

    def __enter__(self) -> NodeLogger:
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class NodeLoggerManager:
    """Manage loggers for all nodes in a cluster."""

    def __init__(self, num_nodes: int, log_dir: Path, stream_worker: int = 0):
        """Initialize logger manager.

        Args:
            num_nodes: Number of nodes in the cluster
            log_dir: Directory for log files
            stream_worker: Which worker to stream to console (default: 0)
        """
        self.num_nodes = num_nodes
        self.log_dir = log_dir
        self.stream_worker = stream_worker
        self.loggers: list[NodeLogger] = []
        self.single_node = num_nodes == 1

        for i in range(num_nodes):
            logger = NodeLogger(
                node_id=i,
                log_dir=log_dir,
                stream_to_console=(i == stream_worker),
                single_node=self.single_node,
            )
            self.loggers.append(logger)

    def get(self, node_id: int) -> NodeLogger:
        """Get logger for a specific node."""
        if node_id < 0 or node_id >= len(self.loggers):
            raise ValueError(f"Invalid node_id {node_id}, must be 0-{len(self.loggers) - 1}")
        return self.loggers[node_id]

    def open_all(self) -> None:
        """Open all log files."""
        for logger in self.loggers:
            logger.open()

    def close_all(self) -> None:
        """Close all log files."""
        for logger in self.loggers:
            logger.close()

    def __enter__(self) -> NodeLoggerManager:
        self.open_all()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close_all()


def create_progress() -> Progress:
    """Create a progress bar for long-running operations."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )


def print_header(title: str) -> None:
    """Print a styled header."""
    console.print(Panel(title, style="bold cyan", expand=False))


def print_status(message: str, style: str = "info") -> None:
    """Print a status message."""
    console.print(f"[{style}]{message}[/{style}]")


def print_success(message: str) -> None:
    """Print a success message."""
    console.print(f"[success]{message}[/success]")


def print_warning(message: str) -> None:
    """Print a warning message."""
    console.print(f"[warning]Warning:[/warning] {message}")


def print_error(message: str) -> None:
    """Print an error message."""
    # Console is already configured with stderr=True
    console.print(f"[error]Error:[/error] {message}")


def print_config(config: dict) -> None:
    """Print configuration in a formatted way."""
    from rich.table import Table

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="dim")
    table.add_column("Value")

    for key, value in config.items():
        table.add_row(key, str(value))

    console.print(table)
