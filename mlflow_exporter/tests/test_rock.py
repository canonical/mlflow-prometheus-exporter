# Copyright 2024 Canonical Ltd.
# See LICENSE file for licensing details.

"""Sanity tests for the mlflow-prometheus-exporter rock."""

import subprocess

import pytest
from charmed_kubeflow_chisme.rock import CheckRock

# The rock uses a `bare` base, so there is no shell available. The exporter is
# therefore exercised by invoking the Python interpreter directly. These are the
# likely locations of the virtual-environment interpreter that has the exporter
# dependencies (mlflow, prometheus_client) installed.
PYTHON_CANDIDATES = ["python3", "/bin/python3", "/usr/bin/python3"]


@pytest.mark.abort_on_fail
def test_rock():
    """The rock ships a working Python environment and the exporter script."""
    check_rock = CheckRock("rockcraft.yaml")
    rock_image = check_rock.get_name()
    rock_version = check_rock.get_version()
    local_rock_image = f"{rock_image}:{rock_version}"

    last_result = None
    for interpreter in PYTHON_CANDIDATES:
        last_result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--entrypoint",
                interpreter,
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
        if last_result.returncode == 0 and "usage" in last_result.stdout.lower():
            return

    pytest.fail(
        "Could not run the exporter with --help inside the rock.\n"
        f"stdout:\n{last_result.stdout}\nstderr:\n{last_result.stderr}"
    )
