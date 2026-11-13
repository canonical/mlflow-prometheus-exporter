#!/usr/bin/env python3
# Copyright 2023 Canonical Ltd.
# See LICENSE file for licensing details.
#

"""Sanity test running the exporter, from the rock, against a local MLflow tracking server.

The MLflow tracking server is run from the MLflow server rock deployed by the mlflow-operator
charm, configured as the charm does: with Basic auth (`--app-name basic-auth`), with its traffic
metrics exposed (`--expose-prometheus`), and either with or without workspaces. The exporter rock
cannot run such a server, as it lacks the server's optional dependencies for these features.
The exporter is run from the rock under test, authenticating to the tracking server via Basic
auth, as the charm's exporter does.

Both are run via Docker, which keeps the test independent of the host Python version: the
sanity CI runner is ubuntu-22.04 (Python 3.10), which is older than the rocks' ubuntu@24.04 (3.12)
base and cannot install the 3.12-pinned exporter dependencies. Only lightweight HTTP checks run on
the host, so the sanity env needs just pytest, requests and tenacity.

Docker host networking lets the host reach the MLflow server and the exporter on
localhost; the sanity workflow runs on a Linux runner, where this is supported.
"""

import subprocess

import pytest
import requests
from charmed_kubeflow_chisme.rock import CheckRock
from tenacity import retry, stop_after_delay, wait_fixed

# NOTE: keep in sync with the MLflow server rock deployed by the mlflow-operator charm, i.e. its
# `oci-image` resource in metadata.yaml at https://github.com/canonical/mlflow-operator:
MLFLOW_SERVER_IMAGE = "docker.io/charmedkubeflow/mlflow:3.15.1-8865b4c-20260831104222"
MLFLOW_PORT = 5000
EXPORTER_PORT = 8999
EXPORTER_TIMEOUT = 5
MLFLOW_SERVER_CONTAINER = "mlflow-server-sanity"
EXPORTER_CONTAINER = "mlflow-exporter-sanity"
MLFLOW_URL = f"http://localhost:{MLFLOW_PORT}/"
# MLflow's default admin user of the basic-auth app, used by the exporter to authenticate:
MLFLOW_USERNAME = "admin"
MLFLOW_PASSWORD = "password1234"
SANITY_TEST_WORKSPACE = "sanity-test-workspace"

# Create one resource of each kind counted by the exporter, in the workspace given as argument if
# any. Run in the MLflow server's container, with its own MLflow, authenticating as the admin.
CREATE_RESOURCES_SCRIPT = f"""
import sys

import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("{MLFLOW_URL}")
if len(sys.argv) > 1:
    mlflow.create_workspace(sys.argv[1])
    mlflow.set_workspace(sys.argv[1])

client = MlflowClient()
experiment_id = client.create_experiment("sanity-test-experiment")
client.create_run(experiment_id)
logged_model = mlflow.initialize_logged_model(
    name="sanity-test-logged-model", experiment_id=experiment_id
)
client.create_registered_model("sanity-test-model")
client.create_model_version(
    "sanity-test-model", source=f"models:/{{logged_model.model_id}}", model_id=logged_model.model_id
)
mlflow.genai.register_prompt(name="sanity-test-prompt", template="Hello, {{{{name}}}}!")
"""

# The resources created by the script above, by metric name:
CREATED_RESOURCES = {
    "num_experiments": 1,
    "num_runs": 1,
    "num_registered_models": 1,
    "num_prompts": 1,
    "num_model_versions": 1,
    "num_logged_models": 1,
}
# The resources that exist on a new tracking server, i.e. the "Default" experiment, by metric name:
INITIAL_RESOURCES = {
    "num_experiments": 1,
    "num_runs": 0,
    "num_registered_models": 0,
    "num_prompts": 0,
    "num_model_versions": 0,
    "num_logged_models": 0,
}


def _remove_container(name):
    """Best-effort removal of a container, ignoring errors if it does not exist."""
    subprocess.run(["docker", "rm", "--force", name], check=False)


def _dump_container_logs(name):
    """Print a container's logs to help debug readiness failures in CI."""
    logs = subprocess.run(
        ["docker", "logs", name],
        capture_output=True,
        text=True,
        check=False,
    )
    print(f"----- docker logs {name} (stdout) -----\n{logs.stdout}")
    print(f"----- docker logs {name} (stderr) -----\n{logs.stderr}")


@pytest.fixture
def rock_image():
    """Reference (name:version) of the rock image in the local Docker daemon."""
    check_rock = CheckRock("rockcraft.yaml")
    return f"{check_rock.get_name()}:{check_rock.get_version()}"


@pytest.fixture(params=[True, False], ids=["workspaces-enabled", "workspaces-disabled"])
def workspaces_enabled(request):
    """Whether the MLflow tracking server runs with workspaces enabled."""
    return request.param


@pytest.fixture
def exporter_server(rock_image, workspaces_enabled):
    """Run the MLflow server and the exporter via Docker, with resources to count."""
    # Clean up any leftovers from a previous aborted run to avoid name conflicts.
    _remove_container(MLFLOW_SERVER_CONTAINER)
    _remove_container(EXPORTER_CONTAINER)

    # Run from a writable working directory, where the basic-auth app's default configuration
    # stores its database, and keep all the server's state there too. A static secret key is
    # required by the basic-auth app.
    subprocess.run(
        [
            "docker",
            "run",
            "--detach",
            "--name",
            MLFLOW_SERVER_CONTAINER,
            "--network",
            "host",
            "--workdir",
            "/tmp",
            "--env",
            "MLFLOW_FLASK_SERVER_SECRET_KEY=sanity-test-secret-key",
            "--entrypoint",
            "mlflow",
            MLFLOW_SERVER_IMAGE,
            "server",
            "--app-name",
            "basic-auth",
            "--host",
            "0.0.0.0",
            "--port",
            str(MLFLOW_PORT),
            "--backend-store-uri",
            "sqlite:////tmp/mlflow.db",
            "--expose-prometheus",
            "/tmp/prometheus",
            "--enable-workspaces" if workspaces_enabled else "--disable-workspaces",
        ],
        check=True,
    )
    try:
        try:
            _wait_for_mlflow_server()
        except Exception:
            _dump_container_logs(MLFLOW_SERVER_CONTAINER)
            raise

        # Create resources inside the server's container, using the server rock's own mlflow.
        subprocess.run(
            [
                "docker",
                "exec",
                "--env",
                f"MLFLOW_TRACKING_USERNAME={MLFLOW_USERNAME}",
                "--env",
                f"MLFLOW_TRACKING_PASSWORD={MLFLOW_PASSWORD}",
                MLFLOW_SERVER_CONTAINER,
                "python3",
                "-c",
                CREATE_RESOURCES_SCRIPT,
                *([SANITY_TEST_WORKSPACE] if workspaces_enabled else []),
            ],
            check=True,
        )

        # Run the exporter from the rock, authenticating to the local MLflow server.
        subprocess.run(
            [
                "docker",
                "run",
                "--detach",
                "--name",
                EXPORTER_CONTAINER,
                "--network",
                "host",
                "--env",
                f"MLFLOW_TRACKING_USERNAME={MLFLOW_USERNAME}",
                "--env",
                f"MLFLOW_TRACKING_PASSWORD={MLFLOW_PASSWORD}",
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


@retry(stop=stop_after_delay(120), wait=wait_fixed(2))
def _wait_for_mlflow_server():
    """Wait until the MLflow tracking server is ready to serve requests."""
    requests.get(f"http://localhost:{MLFLOW_PORT}/health", timeout=5).raise_for_status()


def _expected_samples(workspaces_enabled):
    """Return the samples of resource counts the exporter is expected to report."""
    # the deployment-level metrics count both the initial and the created resources, as totals
    # across workspaces when enabled:
    samples = [
        f'mlflow_metric{{metric_name="{name}"}} {float(INITIAL_RESOURCES[name] + count)}'
        for name, count in CREATED_RESOURCES.items()
    ]
    if workspaces_enabled:
        # the initial resources are in the default workspace, the created ones in a new one:
        counts_per_workspace = {
            "default": INITIAL_RESOURCES,
            SANITY_TEST_WORKSPACE: CREATED_RESOURCES,
        }
        samples += [
            f'mlflow_workspace_metric{{metric_name="{name}",workspace="{workspace}"}} '
            f"{float(count)}"
            for workspace, counts in counts_per_workspace.items()
            for name, count in counts.items()
        ]
    return samples


@retry(stop=stop_after_delay(60), wait=wait_fixed(2))
def _verify_metrics(workspaces_enabled):
    """Poll the exporter until it reports the expected MLflow metrics."""
    response = requests.get(f"http://localhost:{EXPORTER_PORT}/metrics", timeout=5)
    response.raise_for_status()
    metrics_text = response.text

    # The counts of MLflow resources, deployment-level and (with workspaces) per workspace.
    for sample in _expected_samples(workspaces_enabled):
        assert sample in metrics_text
    if not workspaces_enabled:
        assert "mlflow_workspace_metric{" not in metrics_text

    # The tracking server's traffic metrics, re-exposed by the exporter.
    assert "# TYPE mlflow_http_request_total counter" in metrics_text
    assert "# TYPE mlflow_http_request_duration_seconds histogram" in metrics_text
    assert "\nmlflow_http_request_total{" in metrics_text

    # The exporter's own metrics.
    assert "\nprocess_cpu_seconds_total " in metrics_text


def test_exporter_reports_mlflow_metrics(exporter_server, workspaces_enabled):
    """The exporter serves metrics collected from the local MLflow server."""
    # The tracking server requires authentication, also for its traffic metrics, so the exporter
    # can only report the metrics below by authenticating.
    assert requests.get(f"{MLFLOW_URL}metrics", timeout=5).status_code == 401

    try:
        _verify_metrics(workspaces_enabled)
    except Exception:
        _dump_container_logs(EXPORTER_CONTAINER)
        raise


if __name__ == "__main__":
    pytest.main([__file__])
