#!/usr/bin/env python3
# Copyright 2023 Canonical Ltd.
# See LICENSE file for licensing details.
#

"""Integration tests for testing mlflow exporter with local MLflow server."""

import subprocess
import time

import mlflow
import pytest
import requests
from mlflow.exceptions import RestException
from tenacity import retry, stop_after_delay, wait_fixed

TEST_MLFLOW_URL = "http://localhost:5000/"  # URL for the MLflowserver
TEST_EXPORTER_PORT = 8999
TEST_EXPORTER_TIMEOUT = 20


@pytest.fixture
def exporter_server():
    """Deploy local MLflow server with the exporter on different processes."""
    mlflow_server_process = subprocess.Popen(["mlflow", "server"])
    exporter_process = subprocess.Popen(
        [
            "python",
            "mlflow_exporter/mlflow_exporter.py",
            "--mlflowurl",
            TEST_MLFLOW_URL,
            "-t",
            str(TEST_EXPORTER_TIMEOUT),
            "-p",
            str(TEST_EXPORTER_PORT),
        ]
    )

    yield

    # Clean up after the processes
    exporter_process.terminate()
    mlflow_server_process.terminate()


@retry(stop=stop_after_delay(TEST_EXPORTER_TIMEOUT), wait=wait_fixed(1))
def verify_metrics():
    """Try to get metrices every 1 second."""
    response = requests.get(f"http://localhost:{TEST_EXPORTER_PORT}/metrics")
    response.raise_for_status()  # Raise exception if the request was not successful
    metrics_text = response.text

    assert (
        'mlflow_metric{metric_name="num_experiments"} 1.0' in metrics_text
    )  # This is the default experiment
    assert (
        'mlflow_metric{metric_name="num_registered_models"} 1.0' in metrics_text
    )  # That is the model we have created above
    assert 'mlflow_metric{metric_name="num_runs"} 0' in metrics_text  # No runs so far


def test_exporter_integration(exporter_server):
    """Perform a sample MLflow operation that affects the metrics."""
    mlflow.set_tracking_uri(TEST_MLFLOW_URL)
    try:
        mlflow.register_model("model_name", "model_uri")
    except RestException:
        pass  # Ignore exception as we want to create an empty model for the test

    # Wait for the metrics to be collected and updated using Tenacity
    verify_metrics()


if __name__ == "__main__":
    pytest.main([__file__])
