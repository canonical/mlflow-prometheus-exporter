#!/usr/bin/env python3
# Copyright 2023 Canonical Ltd.
# See LICENSE file for licensing details.
#

"""Sanity test running the exporter against a local MLflow server, both from the rock.

The rock bundles Python, MLflow and the exporter script, so the MLflow tracking server
and the exporter are both run from the rock image via Docker. This keeps the test
independent of the host Python version: the sanity CI runner is ubuntu-22.04 (Python
3.10), which is older than the rock's ubuntu@24.04 (3.12) base and cannot install the
3.12-pinned exporter dependencies. Only lightweight HTTP checks run on the host, so the
sanity env needs just pytest, requests and tenacity.

Docker host networking lets the host reach the MLflow server and the exporter on
localhost; the sanity workflow runs on a Linux runner, where this is supported.
"""

import subprocess

import pytest
import requests
from charmed_kubeflow_chisme.rock import CheckRock
from tenacity import retry, stop_after_delay, wait_fixed

MLFLOW_PORT = 5000
EXPORTER_PORT = 8999
EXPORTER_TIMEOUT = 5
MLFLOW_SERVER_CONTAINER = "mlflow-server-sanity"
EXPORTER_CONTAINER = "mlflow-exporter-sanity"
MLFLOW_URL = f"http://localhost:{MLFLOW_PORT}/"

# Register a model on the tracking server so that `num_registered_models` becomes 1.
# Creating a version from a bogus URI is expected to fail; only the registered model
# itself needs to exist for the metric.
REGISTER_MODEL_SCRIPT = f"""
import mlflow
from mlflow.exceptions import RestException

mlflow.set_tracking_uri("{MLFLOW_URL}")
try:
    mlflow.register_model("model_name", "model_uri")
except RestException:
    pass
"""


def _remove_container(name):
    """Best-effort removal of a container, ignoring errors if it does not exist."""
    subprocess.run(["docker", "rm", "--force", name], check=False)


@pytest.fixture
def rock_image():
    """Reference (name:version) of the rock image in the local Docker daemon."""
    check_rock = CheckRock("rockcraft.yaml")
    return f"{check_rock.get_name()}:{check_rock.get_version()}"


@pytest.fixture
def exporter_server(rock_image):
    """Run the MLflow server and the exporter from the rock image via Docker."""
    # Clean up any leftovers from a previous aborted run to avoid name conflicts.
    _remove_container(MLFLOW_SERVER_CONTAINER)
    _remove_container(EXPORTER_CONTAINER)

    # The rock bundles the full mlflow distribution, so its `mlflow` CLI can serve.
    subprocess.run(
        [
            "docker",
            "run",
            "--detach",
            "--name",
            MLFLOW_SERVER_CONTAINER,
            "--network",
            "host",
            "--entrypoint",
            "mlflow",
            rock_image,
            "server",
            "--host",
            "0.0.0.0",
            "--port",
            str(MLFLOW_PORT),
        ],
        check=True,
    )
    try:
        _wait_for_mlflow_server()

        # Create a registered model inside the container, using the rock's own mlflow.
        subprocess.run(
            [
                "docker",
                "exec",
                MLFLOW_SERVER_CONTAINER,
                "python3",
                "-c",
                REGISTER_MODEL_SCRIPT,
            ],
            check=True,
        )

        # Run the exporter from the rock, pointing it at the local MLflow server.
        subprocess.run(
            [
                "docker",
                "run",
                "--detach",
                "--name",
                EXPORTER_CONTAINER,
                "--network",
                "host",
                "--entrypoint",
                "python3",
                rock_image,
                "/mlflow_exporter.py",
                "--mlflowurl",
                MLFLOW_URL,
                "-p",
                str(EXPORTER_PORT),
                "-t",
                str(EXPORTER_TIMEOUT),
            ],
            check=True,
        )

        yield
    finally:
        _remove_container(EXPORTER_CONTAINER)
        _remove_container(MLFLOW_SERVER_CONTAINER)


@retry(stop=stop_after_delay(60), wait=wait_fixed(2))
def _wait_for_mlflow_server():
    """Wait until the MLflow tracking server is ready to serve requests."""
    requests.get(f"http://localhost:{MLFLOW_PORT}/health").raise_for_status()


@retry(stop=stop_after_delay(60), wait=wait_fixed(2))
def _verify_metrics():
    """Poll the exporter until it reports the expected MLflow metrics."""
    response = requests.get(f"http://localhost:{EXPORTER_PORT}/metrics")
    response.raise_for_status()
    metrics_text = response.text

    # The default experiment.
    assert 'mlflow_metric{metric_name="num_experiments"} 1.0' in metrics_text
    # The model registered above.
    assert 'mlflow_metric{metric_name="num_registered_models"} 1.0' in metrics_text
    # No runs have been created.
    assert 'mlflow_metric{metric_name="num_runs"} 0' in metrics_text


def test_exporter_reports_mlflow_metrics(exporter_server):
    """The exporter serves metrics collected from the local MLflow server."""
    _verify_metrics()


if __name__ == "__main__":
    pytest.main([__file__])
