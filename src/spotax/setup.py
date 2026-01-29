"""SpotJAX setup and prerequisites checking."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from spotax.utils.logging import print_error, print_status, print_success, print_warning


@dataclass
class PrerequisiteResult:
    """Result of a prerequisite check."""

    name: str
    passed: bool
    message: str
    fix_command: str | None = None


class SetupChecker:
    """Checks and fixes SpotJAX prerequisites."""

    SSH_KEY_PATHS = [
        Path.home() / ".ssh" / "google_compute_engine",
        Path.home() / ".ssh" / "id_rsa",
        Path.home() / ".ssh" / "id_ed25519",
    ]

    def __init__(self):
        self.results: list[PrerequisiteResult] = []

    def check_gcloud_installed(self) -> PrerequisiteResult:
        """Check if gcloud CLI is installed."""
        gcloud_path = shutil.which("gcloud")
        if gcloud_path:
            return PrerequisiteResult(
                name="gcloud CLI",
                passed=True,
                message=f"Found at {gcloud_path}",
            )
        return PrerequisiteResult(
            name="gcloud CLI",
            passed=False,
            message="gcloud CLI not found",
            fix_command="Install from: https://cloud.google.com/sdk/docs/install",
        )

    def check_gcloud_auth(self) -> PrerequisiteResult:
        """Check if gcloud is authenticated."""
        try:
            result = subprocess.run(
                ["gcloud", "auth", "list", "--format=value(account)"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            accounts = result.stdout.strip().split("\n")
            active_accounts = [a for a in accounts if a]

            if active_accounts:
                return PrerequisiteResult(
                    name="GCP Authentication",
                    passed=True,
                    message=f"Logged in as {active_accounts[0]}",
                )
            return PrerequisiteResult(
                name="GCP Authentication",
                passed=False,
                message="No active GCP account",
                fix_command="gcloud auth login",
            )
        except subprocess.TimeoutExpired:
            return PrerequisiteResult(
                name="GCP Authentication",
                passed=False,
                message="gcloud command timed out",
                fix_command="gcloud auth login",
            )
        except Exception as e:
            return PrerequisiteResult(
                name="GCP Authentication",
                passed=False,
                message=f"Error checking auth: {e}",
                fix_command="gcloud auth login",
            )

    def check_application_default_credentials(self) -> PrerequisiteResult:
        """Check if Application Default Credentials are set up."""
        adc_path = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"

        # Also check environment variable
        env_adc = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

        if adc_path.exists():
            return PrerequisiteResult(
                name="Application Default Credentials",
                passed=True,
                message=f"Found at {adc_path}",
            )
        elif env_adc and Path(env_adc).exists():
            return PrerequisiteResult(
                name="Application Default Credentials",
                passed=True,
                message=f"Using GOOGLE_APPLICATION_CREDENTIALS={env_adc}",
            )
        return PrerequisiteResult(
            name="Application Default Credentials",
            passed=False,
            message="ADC not configured (needed for TPU/GCS APIs)",
            fix_command="gcloud auth application-default login",
        )

    def check_ssh_key(self) -> PrerequisiteResult:
        """Check if SSH key exists."""
        for key_path in self.SSH_KEY_PATHS:
            if key_path.exists():
                return PrerequisiteResult(
                    name="SSH Key",
                    passed=True,
                    message=f"Found at {key_path}",
                )

        return PrerequisiteResult(
            name="SSH Key",
            passed=False,
            message="No SSH key found for GCP",
            fix_command="ssh-keygen -t rsa -f ~/.ssh/google_compute_engine -N ''",
        )

    def check_os_login_key(self) -> PrerequisiteResult:
        """Check if SSH key is registered with OS Login."""
        try:
            result = subprocess.run(
                ["gcloud", "compute", "os-login", "ssh-keys", "list", "--format=value(key)"],
                capture_output=True,
                text=True,
                timeout=15,
            )

            if result.returncode == 0 and result.stdout.strip():
                key_count = len(result.stdout.strip().split("\n"))
                return PrerequisiteResult(
                    name="OS Login SSH Key",
                    passed=True,
                    message=f"{key_count} key(s) registered with OS Login",
                )

            return PrerequisiteResult(
                name="OS Login SSH Key",
                passed=False,
                message="No SSH keys registered with OS Login",
                fix_command="gcloud compute os-login ssh-keys add --key-file ~/.ssh/google_compute_engine.pub",
            )
        except subprocess.TimeoutExpired:
            return PrerequisiteResult(
                name="OS Login SSH Key",
                passed=False,
                message="gcloud command timed out",
                fix_command="gcloud compute os-login ssh-keys add --key-file ~/.ssh/google_compute_engine.pub",
            )
        except Exception as e:
            return PrerequisiteResult(
                name="OS Login SSH Key",
                passed=False,
                message=f"Error checking OS Login: {e}",
                fix_command="gcloud compute os-login ssh-keys add --key-file ~/.ssh/google_compute_engine.pub",
            )

    def check_default_project(self) -> PrerequisiteResult:
        """Check if a default GCP project is configured."""
        try:
            result = subprocess.run(
                ["gcloud", "config", "get-value", "project"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            project = result.stdout.strip()
            if project and project != "(unset)":
                return PrerequisiteResult(
                    name="Default GCP Project",
                    passed=True,
                    message=f"Project: {project}",
                )

            return PrerequisiteResult(
                name="Default GCP Project",
                passed=False,
                message="No default project configured",
                fix_command="gcloud config set project YOUR_PROJECT_ID",
            )
        except Exception as e:
            return PrerequisiteResult(
                name="Default GCP Project",
                passed=False,
                message=f"Error checking project: {e}",
                fix_command="gcloud config set project YOUR_PROJECT_ID",
            )

    def run_all_checks(self) -> list[PrerequisiteResult]:
        """Run all prerequisite checks."""
        self.results = [
            self.check_gcloud_installed(),
            self.check_gcloud_auth(),
            self.check_application_default_credentials(),
            self.check_default_project(),
            self.check_ssh_key(),
            self.check_os_login_key(),
        ]
        return self.results

    def all_passed(self) -> bool:
        """Check if all prerequisites passed."""
        return all(r.passed for r in self.results)


def generate_ssh_key(key_path: Path | None = None) -> bool:
    """Generate SSH key for GCP.

    Args:
        key_path: Path for the key (default: ~/.ssh/google_compute_engine)

    Returns:
        True if key was created successfully
    """
    if key_path is None:
        key_path = Path.home() / ".ssh" / "google_compute_engine"

    # Ensure .ssh directory exists
    key_path.parent.mkdir(mode=0o700, exist_ok=True)

    print_status(f"Generating SSH key at {key_path}...")

    try:
        result = subprocess.run(
            [
                "ssh-keygen",
                "-t", "rsa",
                "-b", "4096",
                "-f", str(key_path),
                "-N", "",  # No passphrase
                "-C", "spotax",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print_success(f"SSH key created: {key_path}")
            return True
        else:
            print_error(f"Failed to create SSH key: {result.stderr}")
            return False
    except Exception as e:
        print_error(f"Error creating SSH key: {e}")
        return False


def register_ssh_key_with_os_login(key_path: Path | None = None) -> bool:
    """Register SSH public key with OS Login.

    Args:
        key_path: Path to private key (public key = key_path.pub)

    Returns:
        True if registration succeeded
    """
    if key_path is None:
        key_path = Path.home() / ".ssh" / "google_compute_engine"

    pub_key_path = Path(str(key_path) + ".pub")

    if not pub_key_path.exists():
        print_error(f"Public key not found: {pub_key_path}")
        return False

    print_status("Registering SSH key with OS Login...")

    try:
        result = subprocess.run(
            [
                "gcloud", "compute", "os-login", "ssh-keys", "add",
                f"--key-file={pub_key_path}",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            print_success("SSH key registered with OS Login")
            return True
        else:
            print_error(f"Failed to register SSH key: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print_error("OS Login registration timed out")
        return False
    except Exception as e:
        print_error(f"Error registering SSH key: {e}")
        return False


async def run_setup(fix: bool = False) -> bool:
    """Run SpotJAX setup checks and optionally fix issues.

    Args:
        fix: If True, attempt to fix issues automatically

    Returns:
        True if all checks pass (after fixes)
    """
    from rich.table import Table

    from spotax.utils.logging import console

    print_status("Checking SpotJAX prerequisites...\n")

    checker = SetupChecker()
    results = checker.run_all_checks()

    # Display results as a table
    table = Table(show_header=True, header_style="bold")
    table.add_column("Check", style="cyan")
    table.add_column("Status")
    table.add_column("Details")

    for result in results:
        status = "[green]✓ PASS[/green]" if result.passed else "[red]✗ FAIL[/red]"
        table.add_row(result.name, status, result.message)

    console.print(table)
    console.print()

    if checker.all_passed():
        print_success("All prerequisites satisfied! Ready to use SpotJAX.")
        return True

    # Show failed checks with fix commands
    failed = [r for r in results if not r.passed]

    if not fix:
        print_warning(f"{len(failed)} check(s) failed. Run with --fix to attempt automatic fixes.\n")
        print_status("Manual fix commands:")
        for result in failed:
            if result.fix_command:
                console.print(f"  [dim]{result.name}:[/dim] {result.fix_command}")
        return False

    # Attempt to fix issues
    print_status("Attempting to fix issues...\n")

    for result in failed:
        if result.name == "SSH Key":
            generate_ssh_key()
        elif result.name == "OS Login SSH Key":
            # First ensure we have an SSH key
            ssh_key_path = Path.home() / ".ssh" / "google_compute_engine"
            if not ssh_key_path.exists():
                generate_ssh_key(ssh_key_path)
            register_ssh_key_with_os_login(ssh_key_path)
        elif result.fix_command:
            print_warning(f"Cannot auto-fix '{result.name}'. Please run manually:")
            console.print(f"  {result.fix_command}")

    # Re-run checks
    console.print()
    print_status("Re-checking prerequisites...\n")
    results = checker.run_all_checks()

    # Display updated results
    table = Table(show_header=True, header_style="bold")
    table.add_column("Check", style="cyan")
    table.add_column("Status")
    table.add_column("Details")

    for result in results:
        status = "[green]✓ PASS[/green]" if result.passed else "[red]✗ FAIL[/red]"
        table.add_row(result.name, status, result.message)

    console.print(table)
    console.print()

    if checker.all_passed():
        print_success("All prerequisites satisfied! Ready to use SpotJAX.")
        return True
    else:
        remaining = [r for r in results if not r.passed]
        print_error(f"{len(remaining)} check(s) still failing. Please fix manually.")
        return False


def check_prerequisites_quick() -> tuple[bool, list[str]]:
    """Quick prerequisite check for use before 'spotax run'.

    Returns:
        Tuple of (all_passed, list of error messages)
    """
    errors = []

    # Check SSH key
    ssh_key_found = False
    for key_path in SetupChecker.SSH_KEY_PATHS:
        if key_path.exists():
            ssh_key_found = True
            break

    if not ssh_key_found:
        errors.append("SSH key not found. Run: spotax setup --fix")

    # Check ADC
    adc_path = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
    env_adc = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")

    if not adc_path.exists() and not (env_adc and Path(env_adc).exists()):
        errors.append("GCP credentials not found. Run: gcloud auth application-default login")

    return len(errors) == 0, errors
