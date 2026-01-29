"""SpotJAX CLI entrypoint."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from spotax import __version__
from spotax.config import SpotJAXConfig
from spotax.orchestrator import run_orchestrator
from spotax.utils.logging import print_error

# Create typer app
app = typer.Typer(
    name="spotax",
    help="CLI tool for orchestrating JAX training on Google Cloud Spot TPUs",
    add_completion=False,
    no_args_is_help=True,
)

console = Console(stderr=True)


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"SpotJAX version {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-v",
            help="Show version and exit",
            callback=version_callback,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """SpotJAX - Orchestrate JAX training on Google Cloud Spot TPUs."""
    pass


@app.command()
def run(
    script: Annotated[
        Path,
        typer.Argument(
            help="Path to the training script to run",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    tpu: Annotated[
        str,
        typer.Option(
            "--tpu",
            "-t",
            help="TPU type (e.g., v4-8, v4-32, v5p-16)",
        ),
    ] = "v4-8", #TODO consider maybe remove as default
    zone: Annotated[
        str,
        typer.Option(
            "--zone",
            "-z",
            help="GCP zone (e.g., us-central2-b)",
        ),
    ] = "us-central2-b", #TODO consider maybe remove as default
    project: Annotated[
        str | None,
        typer.Option(
            "--project",
            "-p",
            help="GCP project ID (auto-detect from environment if not set)", #TODO find how it's finded
        ),
    ] = None,
    bucket: Annotated[
        str | None,
        typer.Option(
            "--bucket",
            "-b",
            help="GCS bucket for checkpoints/logs (auto-create spotax-{project} if not set)",
        ),
    ] = None,
    name: Annotated[
        str | None,
        typer.Option(
            "--name",
            "-n",
            help="Job name/ID (auto-generate timestamp if not set)",
        ),
    ] = None,
    max_retries: Annotated[
        int,
        typer.Option(
            "--max-retries",
            help="Maximum restart attempts on failure",
        ),
    ] = 5, # do we actualy need to do this maybe we need to restact until training is finished?
    worker: Annotated[
        int,
        typer.Option(
            "--stream-worker",
            "-w",
            help="Worker index to stream logs from (0-indexed)",
        ),
    ] = 0,
    code_dir: Annotated[
        Path | None,
        typer.Option(
            "--code-dir",
            "-c",
            help="Directory containing code to sync (default: script's parent directory)",
            exists=True,
            file_okay=False,
            dir_okay=True,
            readable=True,
            resolve_path=True,
        ),
    ] = None,
) -> None:
    """Run a training script on Spot TPUs.

    This command will:
    1. Provision a Spot TPU VM slice
    2. Upload your code to all worker (TPU VM's)
    3. Run the training script
    4. Stream logs from the specified worker
    5. Clean up resources when done

    Example:
        spotax run train.py --tpu v4-8 --zone us-central2-b
    """
    try:
        # Build configuration
        config = SpotJAXConfig.from_cli_args(
            script=str(script),
            tpu_type=tpu,
            zone=zone,
            project=project,
            bucket=bucket,
            job_id=name,
            max_retries=max_retries,
            stream_worker=worker,
            code_dir=str(code_dir) if code_dir else None,
        )
    except ValueError as e:
        print_error(str(e))
        raise typer.Exit(1) from e

    # Check prerequisites before running
    from spotax.setup import check_prerequisites_quick  # i think better to do full check

    prereq_ok, prereq_errors = check_prerequisites_quick()
    if not prereq_ok:
        print_error("Prerequisites not satisfied:")
        for err in prereq_errors:
            console.print(f"  • {err}")
        console.print("\nRun [cyan]spotax setup[/cyan] to check all prerequisites.")
        raise typer.Exit(1)

    # Run the orchestrator
    exit_code = asyncio.run(run_orchestrator(config))
    raise typer.Exit(exit_code)


@app.command()
def setup(
    fix: Annotated[
        bool,
        typer.Option(
            "--fix",
            "-f",
            help="Attempt to automatically fix issues (create SSH keys, register with OS Login)",
        ),
    ] = False,
) -> None:
    """Check and setup SpotJAX prerequisites.

    Verifies that all required tools and credentials are configured:
    - gcloud CLI installed and authenticated
    - Application Default Credentials for API access
    - SSH key for connecting to TPU VMs
    - OS Login registration for SSH access

    Example:
        spotax setup           # Check prerequisites
        spotax setup --fix     # Check and fix issues automatically
    """
    import asyncio

    from spotax.setup import run_setup

    success = asyncio.run(run_setup(fix=fix)) # dont undersund why we use asunc for these i am not sure
    raise typer.Exit(0 if success else 1)


if __name__ == "__main__":
    app()
