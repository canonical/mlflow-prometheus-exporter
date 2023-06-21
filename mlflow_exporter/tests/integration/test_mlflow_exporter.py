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

TEST_MLFLOW_URL = "http://localhost:5000/"  # URL for the MLflowserver


@pytest.fixture
def exporter_server():
    """Deploy local MLflow server with the exporter on different processes."""
    mlflow_server_process = subprocess.Popen(["mlflow", "server"])
    time.sleep(10)  # Give some time for the  mlflow server to start up
    exporter_process = subprocess.Popen(
        ["python", "mlflow_exporter/mlflow_exporter.py", "--mlflowurl", TEST_MLFLOW_URL]
    )
    time.sleep(10)  # Give some time for the exporter server to start up

    yield

    # Clean up after the processes
    exporter_process.terminate()
    mlflow_server_process.terminate()


def test_exporter_integration(exporter_server):
    """Perform a sample MLflow operation that affects the metrics."""
    mlflow.set_tracking_uri(TEST_MLFLOW_URL)
    try:
        mlflow.register_model("model_name", "model_uri")
    except RestException:
        pass  # Ignore exception as we want to create empty model for test

    # Wait for the metrics to be collected and updated
    time.sleep(30)

    # Verify the exporter metrics via the /metrics endpoint
    response = requests.get("http://localhost:8000/metrics")

    assert response.status_code == 200
    assert (
        'mlflow_metric{metric_name="num_experiments"} 1.0' in response.text
    )  # This is the default experiment
    assert (
        'mlflow_metric{metric_name="num_registered_models"} 1.0' in response.text
    )  # That is the model we have created above
    assert 'mlflow_metric{metric_name="num_runs"} 0' in response.text  # No runs so far


if __name__ == "__main__":
    pytest.main([__file__])
