#!/usr/bin/env python3
# Copyright 2023 Canonical Ltd.
# See LICENSE file for licensing details.
#

"""Run MLflow prometheus exporter server."""

import argparse
import base64
import json
import logging
import os
import ssl
import time
import urllib.error
import urllib.request

import mlflow
from mlflow import MlflowClient
from mlflow.environment_variables import (
    MLFLOW_TRACKING_CLIENT_CERT_PATH,
    MLFLOW_TRACKING_SERVER_CERT_PATH,
)
from prometheus_client import REGISTRY, start_http_server
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.parser import text_string_to_metric_families

logger = logging.getLogger(__name__)

# Path, relative to the MLflow server URL, of the tracking server's own Prometheus endpoint, serving
# its traffic metrics when the server runs with `--expose-prometheus`:
TRACKING_SERVER_METRICS_PATH = "metrics"
# Path, relative to the MLflow server URL, of the tracking server's endpoint reporting its
# capabilities, among which whether workspaces are enabled:
TRACKING_SERVER_INFO_PATH = "api/3.0/mlflow/server-info"
# Upper bound on the exporter's own requests to the tracking server, as the traffic metrics are
# fetched while serving a scrape of the exporter (whose default timeout in Prometheus is 10s):
TRACKING_SERVER_REQUEST_TIMEOUT_SECONDS = 5
# The environment variables the MLflow client reads its Basic auth credentials from:
MLFLOW_TRACKING_USERNAME_ENV_VAR = "MLFLOW_TRACKING_USERNAME"
MLFLOW_TRACKING_PASSWORD_ENV_VAR = "MLFLOW_TRACKING_PASSWORD"

# Metrics counted in each workspace, and across all workspaces at the deployment level:
METRIC_NAMES = (
    "num_experiments",
    "num_runs",
    "num_registered_models",
    "num_prompts",
    "num_model_versions",
    "num_logged_models",
)


def _get_from_tracking_server(url: str) -> str:
    """Return the body of a GET request to the tracking server.

    Basic authentication is used if the MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD
    environment variables have been set, in the same way as the MLflow client does.
    """
    headers = {}
    username = os.getenv(MLFLOW_TRACKING_USERNAME_ENV_VAR)
    password = os.getenv(MLFLOW_TRACKING_PASSWORD_ENV_VAR)
    if username and password:
        credentials = base64.standard_b64encode(
            f"{username}:{password}".encode()
        ).decode()
        headers["Authorization"] = f"Basic {credentials}"

    request = urllib.request.Request(url)
    if "Authorization" in headers:
        request.add_unredirected_header("Authorization", headers["Authorization"])
    ssl_context = ssl.create_default_context(cafile=MLFLOW_TRACKING_SERVER_CERT_PATH.get())
    if client_cert_path := MLFLOW_TRACKING_CLIENT_CERT_PATH.get():
        ssl_context.load_cert_chain(client_cert_path)
    with urllib.request.urlopen(
        request, timeout=TRACKING_SERVER_REQUEST_TIMEOUT_SECONDS, context=ssl_context
    ) as response:
        return response.read().decode("utf-8")


class TrackingServerMetricsCollector:
    """Re-expose the traffic metrics served by the MLflow tracking server's `/metrics` endpoint.

    A failure to fetch or parse them is logged and only leaves these metrics out of the scrape,
    without affecting the exporter's own metrics.
    """

    def __init__(self, mlflow_url: str):
        """Build the URL of the tracking server's metrics endpoint from the MLflow server URL."""
        self._metrics_url = f"{mlflow_url.rstrip('/')}/{TRACKING_SERVER_METRICS_PATH}"

    def describe(self):
        """Describe no metric upfront, so that registering this collector fetches nothing."""
        return []

    def collect(self):
        """Yield the metric families currently served by the tracking server."""
        try:
            # parsing everything before yielding anything, so that a parsing failure never leaves
            # a partial set of metrics (or an error) in the exporter's response:
            families = list(
                text_string_to_metric_families(
                    _get_from_tracking_server(self._metrics_url)
                )
            )
        except Exception as error:
            logger.warning(
                "Failed to collect the tracking server metrics from %s: %s",
                self._metrics_url,
                error,
            )
            return
        yield from families


class MlflowResourcesCollector:
    """Expose the counts of MLflow resources, both deployment-level and per workspace.

    The counts are polled periodically through the MLflow API, as counting them can be expensive,
    and the latest ones are exposed at scrape time.
    """

    def __init__(self):
        """Start with no counts, until the first poll."""
        self._counts = ({}, {})

    def update(self, deployment_counts: dict, workspace_counts: dict):
        """Replace the exposed counts with newly polled ones.

        Args:
            deployment_counts: the deployment-level counts, by metric name.
            workspace_counts: the counts of each workspace, by workspace name and metric name.
        """
        # replacing all the counts at once, as they are concurrently exposed by the HTTP server's
        # thread, and dropping those no longer polled (e.g., of a deleted workspace) along the way:
        self._counts = (deployment_counts, workspace_counts)

    def describe(self):
        """Describe the exposed metric families, without any samples."""
        return list(self._build_families({}, {}))

    def collect(self):
        """Yield the metric families with the latest polled counts."""
        return self._build_families(*self._counts)

    @staticmethod
    def _build_families(deployment_counts: dict, workspace_counts: dict):
        deployment_family = GaugeMetricFamily(
            "mlflow_metric", "MLflow metric", labels=["metric_name"]
        )
        for metric_name, count in deployment_counts.items():
            deployment_family.add_metric([metric_name], count)

        workspace_family = GaugeMetricFamily(
            "mlflow_workspace_metric",
            "MLflow metric per workspace",
            labels=["metric_name", "workspace"],
        )
        for workspace, counts in workspace_counts.items():
            for metric_name, count in counts.items():
                workspace_family.add_metric([metric_name, workspace], count)

        return [deployment_family, workspace_family]


def _workspaces_enabled(mlflow_url: str) -> bool:
    """Return whether the tracking server has workspaces enabled.

    The server's capabilities are queried directly, as the workspace APIs of a server without
    workspaces respond with a 503 status code, which the MLflow client retries for minutes.
    """
    server_info_url = f"{mlflow_url.rstrip('/')}/{TRACKING_SERVER_INFO_PATH}"
    try:
        server_info = json.loads(_get_from_tracking_server(server_info_url))
    except urllib.error.HTTPError as error:
        # same as the MLflow client, for older servers lacking this endpoint (and workspaces):
        if error.code == 404:
            return False
        raise
    return bool(server_info.get("workspaces_enabled", False))


def _count_all(search_page) -> int:
    """Count all the results of a paginated MLflow search.

    The results are counted page by page, without any limit on their total number (unlike, e.g.,
    `mlflow.search_runs`), and without holding them all in memory at once.

    Args:
        search_page: a function returning the page of search results for a given page token.
    """
    count = 0
    page_token = None
    while True:
        page = search_page(page_token)
        count += len(page)
        page_token = page.token
        if not page_token:
            return count


def _count_resources(workspace: str | None) -> dict:
    """Count the MLflow resources of the active workspace, by metric name.

    Each resource is counted independently: one failing to be counted is logged and left out of
    the result, without affecting the others.
    """
    client = MlflowClient()
    # NOTE: both registered models and model versions exclude prompts and their versions:
    page_searches = {
        "num_registered_models": lambda token: client.search_registered_models(
            page_token=token
        ),
        "num_prompts": lambda token: client.search_prompts(page_token=token),
        "num_model_versions": lambda token: client.search_model_versions(
            page_token=token
        ),
    }
    counts = {}
    try:
        experiment_ids = [
            experiment.experiment_id for experiment in mlflow.search_experiments()
        ]
    except Exception as error:
        # Without experiment_ids, we can't search for runs or logged models
        logger.warning(
            "Failed to count num_experiments, num_runs, num_logged_models (workspace: %s): %s",
            workspace,
            error,
        )
    else:
        counts["num_experiments"] = len(experiment_ids)
        # Runs and logged models are only searched in a non-empty list of experiments:
        if experiment_ids:
            page_searches["num_runs"] = lambda token: client.search_runs(
                experiment_ids, page_token=token
            )
            page_searches["num_logged_models"] = (
                lambda token: client.search_logged_models(
                    experiment_ids, page_token=token
                )
            )
        else:
            counts["num_runs"] = counts["num_logged_models"] = 0

    for metric_name, search_page in page_searches.items():
        try:
            counts[metric_name] = _count_all(search_page)
        except Exception as error:
            logger.warning(
                "Failed to count %s (workspace: %s): %s", metric_name, workspace, error
            )
    return counts


def collect_metrics(mlflow_url: str) -> tuple:
    """Collect MLflow data, returning the deployment-level and the per-workspace counts.

    With workspaces enabled, the resources are counted in each workspace, and summed across them
    for the deployment-level counts.
    Without workspaces, only the deployment-level counts are reported, counted server-wide.

    Basic authentication is automatically used if the MLFLOW_TRACKING_USERNAME
    and MLFLOW_TRACKING_PASSWORD environment variables have been set.
    """
    if not _workspaces_enabled(mlflow_url):
        return _count_resources(workspace=None), {}

    workspace_counts = {}
    try:
        for workspace in mlflow.list_workspaces():
            mlflow.set_workspace(workspace.name)
            workspace_counts[workspace.name] = _count_resources(
                workspace=workspace.name
            )
    finally:
        mlflow.set_workspace(None)

    deployment_counts = {
        name: sum(counts[name] for counts in workspace_counts.values())
        for name in METRIC_NAMES
        if all(name in counts for counts in workspace_counts.values())
    }
    return deployment_counts, workspace_counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--port",
        "-p",
        type=int,
        help="Port on which run the exporter server server",
        default=os.getenv("PORT", 8000),
    )
    parser.add_argument(
        "--mlflowurl",
        "-u",
        type=str,
        help="MLflow server url for collecting data",
        default=os.getenv("MLFLOW_URL", "http://localhost:5000/"),
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        help="Timeout for polling new metrics in seconds.",
        default=os.getenv("TIMEOUT", 30),
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.mlflowurl)

    # Expose the counts of MLflow resources, and re-expose the tracking server's traffic metrics,
    # alongside the exporter's own metrics
    resources_collector = MlflowResourcesCollector()
    REGISTRY.register(resources_collector)
    REGISTRY.register(TrackingServerMetricsCollector(args.mlflowurl))

    # Start the Prometheus HTTP server
    start_http_server(args.port)

    # Periodically collect and update the metrics
    while True:
        try:
            resources_collector.update(*collect_metrics(args.mlflowurl))
        except Exception:
            # not to report stale counts, and not to stop serving the other sources of metrics:
            logger.exception("Failed to collect metrics from the MLflow server")
            resources_collector.update({}, {})
        time.sleep(args.timeout)  # Adjust the sleep duration as needed
