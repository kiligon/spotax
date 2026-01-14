"""SpotJAX CLI entrypoint."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from spotjax import __version__
from spotjax.config import SpotJAXConfig
from spotjax.orchestrator import run_orchestrator
from spotjax.utils.logging import print_error

# Create typer app
app = typer.Typer(
    name="spotjax",
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
            help="GCS bucket for checkpoints/logs (auto-create spotjax-{project} if not set)",
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
        spotjax run train.py --tpu v4-8 --zone us-central2-b
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
    from spotjax.setup import check_prerequisites_quick  # i think better to do full check

    prereq_ok, prereq_errors = check_prerequisites_quick()
    if not prereq_ok:
        print_error("Prerequisites not satisfied:")
        for err in prereq_errors:
            console.print(f"  • {err}")
        console.print("\nRun [cyan]spotjax setup[/cyan] to check all prerequisites.")
        raise typer.Exit(1)

    # Run the orchestrator
    exit_code = asyncio.run(run_orchestrator(config))
    raise typer.Exit(exit_code)


@app.command("generate-utils")
def generate_utils(
    output_dir: Annotated[
        Path,
        typer.Argument(
            help="Directory to generate spotjax_utils.py in",
            exists=True,
            file_okay=False,
            dir_okay=True,
            writable=True,
            resolve_path=True,
        ),
    ] = Path("."),
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            "-f",
            help="Overwrite existing spotjax_utils.py if it exists",
        ),
    ] = False,
) -> None:
    """Generate spotjax_utils.py utility template for your project.

    This creates a utility file with best-practice code for:
    - Checkpoint management with Orbax
    - Signal handling for graceful preemption
    - Auto-resume from checkpoints
    - JAX distributed initialization

    The generated file is yours to own and modify. It has no runtime
    dependency on the spotjax package.

    Example:
        spotjax generate-utils ./my_project
        spotjax generate-utils .  # Current directory
    """
    from spotjax.scaffold import generate_spotjax_utils

    try:
        output_path = generate_spotjax_utils(output_dir, overwrite=force)
        console.print(f"[green]Created:[/green] {output_path}")
        console.print()
        #TODO i think it's not really good docs, need to imporve this
        console.print("Next steps:")
        console.print("  1. Install dependencies: [cyan]pip install orbax-checkpoint jax[tpu][/cyan]")
        console.print("  2. Import in your training script:")
        console.print("     [dim]from spotjax_utils import CheckpointManager, get_config, setup_distributed[/dim]")
        console.print()
        console.print("See the generated file for usage examples and documentation.")
    except FileExistsError as e:
        print_error(str(e))
        raise typer.Exit(1) from e


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
        spotjax setup           # Check prerequisites
        spotjax setup --fix     # Check and fix issues automatically
    """
    import asyncio

    from spotjax.setup import run_setup

    success = asyncio.run(run_setup(fix=fix)) # dont undersund why we use asunc for these i am not sure
    raise typer.Exit(0 if success else 1)


@app.command("list-tpus")
def list_tpus(
    project: Annotated[
        str | None,
        typer.Option(
            "--project", #TODO make current project as default value
            "-p",
            help="GCP project ID (auto-detect from environment if not set)",
        ),
    ] = None,
) -> None:
    """List available TPU types by region.

    Shows a table of all TPU accelerator types available across GCP zones.
    Useful for choosing which TPU type and zone to use for your workload.

    Note: Actual availability depends on your project's quota and current demand.

    Example:
        spotjax list-tpus
        spotjax list-tpus --project my-project
    """
    from rich.table import Table

    from spotjax.config import get_default_project
    from spotjax.infra.gcp import list_tpu_types

    # Resolve project
    resolved_project = project or get_default_project()
    if not resolved_project:
        print_error("GCP project not specified. Set --project or GOOGLE_CLOUD_PROJECT env var.")
        raise typer.Exit(1)

    console.print(f"Querying TPU availability for project [cyan]{resolved_project}[/cyan]...")
    console.print()

    try:
        tpu_by_zone = asyncio.run(list_tpu_types(resolved_project))
    except Exception as e:
        print_error(f"Failed to query TPU types: {e}")
        raise typer.Exit(1) from e

    if not tpu_by_zone:
        console.print("[yellow]No TPU types found. Check your project has TPU API enabled.[/yellow]")
        raise typer.Exit(0)

    # Build a table organized by zone
    table = Table(title="Available TPU Types by Zone")
    table.add_column("Zone", style="cyan")
    table.add_column("TPU Types", style="green")

    for zone in sorted(tpu_by_zone.keys()):
        types = tpu_by_zone[zone]
        # Group types for readability
        type_str = ", ".join(types)
        table.add_row(zone, type_str)

    console.print(table)
    console.print()
    console.print("[dim]Note: Actual availability depends on quota and demand.[/dim]")
    console.print("[dim]Request quota at: https://console.cloud.google.com/iam-admin/quotas[/dim]")


@app.command()
def status(
    _name: Annotated[
        str | None,
        typer.Argument(help="Job name to check status of"),
    ] = None,
) -> None:
    """Check the status of a running or recent job.

    Example:
        spotjax status 20260109-143022
    """
    # Phase 2 feature - not implemented yet
    console.print("[yellow]The 'status' command will be available in Phase 2[/yellow]")
    raise typer.Exit(0)


if __name__ == "__main__":
    app()
