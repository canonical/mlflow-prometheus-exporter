# Copyright 2024 Canonical Ltd.
# See LICENSE file for licensing details.

"""Sanity tests for the mlflow-prometheus-exporter rock."""

import subprocess

import pytest
from charmed_kubeflow_chisme.rock import CheckRock

# rockcraft's python plugin creates the virtual environment at the root of the
# rock, so on the usrmerged ubuntu@24.04 base the venv interpreter is the single
# "python3" on PATH. It is the interpreter that can import mlflow and
# prometheus_client, and it is what the charm invokes when it overrides the
# Pebble command with "python3 mlflow_exporter.py ...".
PYTHON = "python3"


@pytest.mark.abort_on_fail
def test_rock():
    """The rock ships a working Python environment and the exporter script."""
    check_rock = CheckRock("rockcraft.yaml")
    rock_image = check_rock.get_name()
    rock_version = check_rock.get_version()
    local_rock_image = f"{rock_image}:{rock_version}"

    result = subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--entrypoint",
            PYTHON,
            local_rock_image,
            "/mlflow_exporter.py",
            "--help",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    # A successful `--help` run proves the interpreter imported mlflow and
    # prometheus_client and that argparse rendered the exporter's usage.
    assert result.returncode == 0 and "usage" in result.stdout.lower(), (
        "Could not run the exporter with --help inside the rock.\n"
        f"exit={result.returncode}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
