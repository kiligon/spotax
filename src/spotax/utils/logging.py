"""Logging utilities for Spotax using Rich."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import TextIO

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.style import Style
from rich.theme import Theme

# JAX color palette
JAX_BLUE = "#5e97f6"
JAX_GREEN = "#26a69a"
JAX_PINK = "#ea80fc"

# Custom theme for Spotax using JAX colors
SPOTAX_THEME = Theme(
    {
        "info": JAX_PINK,
        "warning": "bright_yellow",
        "error": "bright_red bold",
        "success": JAX_GREEN,
        "node": JAX_BLUE,
        "dim": "dim",
    }
)

# Global console instance
console = Console(theme=SPOTAX_THEME, stderr=True)


def get_log_dir(base_dir: Path | None = None) -> Path:
    """Get the local log directory for this session."""
    if base_dir is None:
        base_dir = Path.home() / ".spotax" / "logs"
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = base_dir / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


class NodeLogger:
    """Logger that prefixes output with node ID and writes to file."""

    # Color palette for different nodes (JAX-inspired colors)
    NODE_COLORS = [
        JAX_BLUE,       # Node 0 - blue
        JAX_GREEN,      # Node 1 - green
        JAX_PINK,       # Node 2 - pink
        "bright_cyan",  # Node 3
        "bright_yellow",# Node 4
        "#9fa8da",      # Node 5 - light blue
        "#80cbc4",      # Node 6 - light teal
        "#ce93d8",      # Node 7 - light purple
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

        # Write to console if streaming - use yellow instead of red for readability
        if self.stream_to_console:
            stderr_style = Style(color="bright_yellow", italic=True)
            for line in text.splitlines():
                if self.single_node:
                    console.print(line, style=stderr_style, highlight=False)
                else:
                    console.print(
                        f"{self.prefix} {line}",
                        style=stderr_style,
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
    console.print(Panel(title, style=f"bold {JAX_BLUE}", expand=False))


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
