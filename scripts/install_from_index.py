"""Install a package from a Python package index with retries."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, help="Package specifier to install.")
    parser.add_argument("--index-url", required=True, help="Primary package index URL.")
    parser.add_argument(
        "--extra-index-url",
        default="https://pypi.org/simple",
        help="Fallback package index URL.",
    )
    parser.add_argument("--retries", type=int, default=10, help="Number of install attempts.")
    parser.add_argument("--delay-seconds", type=int, default=30, help="Delay between attempts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "--index-url",
        args.index_url,
        "--extra-index-url",
        args.extra_index_url,
        args.package,
    ]
    for attempt in range(1, args.retries + 1):
        # The command is constructed from explicit workflow inputs for pip installation.
        result = subprocess.run(command, check=False)  # noqa: S603
        if result.returncode == 0:
            print(f"Installed {args.package} on attempt {attempt}.")
            return
        if attempt == args.retries:
            raise SystemExit(result.returncode)
        print(
            f"Install attempt {attempt} failed for {args.package}. "
            f"Retrying in {args.delay_seconds} seconds..."
        )
        time.sleep(args.delay_seconds)


if __name__ == "__main__":
    main()
