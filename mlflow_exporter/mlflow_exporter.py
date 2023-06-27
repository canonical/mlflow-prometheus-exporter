#!/usr/bin/env python3
# Copyright 2023 Canonical Ltd.
# See LICENSE file for licensing details.
#

"""Run MLflow prometheus exporter server."""

import argparse
import os
import time

import mlflow
from prometheus_client import Gauge, start_http_server

mlflow_metric = Gauge("mlflow_metric", "MLflow metric", ["metric_name"])


def collect_metrics():
    """Collect MLflow data and update Prometheus metrics."""
    num_registered_models = len(mlflow.search_registered_models())
    num_experiments = len(mlflow.search_experiments())
    num_runs = len(mlflow.search_runs())

    # Update Prometheus metrics
    mlflow_metric.labels(metric_name="num_registered_models").set(num_registered_models)
    mlflow_metric.labels(metric_name="num_experiments").set(num_experiments)
    mlflow_metric.labels(metric_name="num_runs").set(num_runs)


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

    # Start the Prometheus HTTP server
    start_http_server(args.port)

    # Periodically collect and update the metrics
    while True:
        collect_metrics()
        time.sleep(args.timeout)  # Adjust the sleep duration as needed
