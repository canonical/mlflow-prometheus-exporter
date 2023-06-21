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


class EnvDefault(argparse.Action):
    """
    Argparse Action which sets the value based on environment variable and parameters.

    If the value is missing use environment variable if presented otherwise use default value.
    """

    def __init__(self, envvar, required=True, default=None, **kwargs):
        """Specify the parameters for given argument."""
        if envvar in os.environ:
            default = os.environ[envvar]
        if required and default:
            required = False
        super(EnvDefault, self).__init__(default=default, required=required, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        """Call the method to set the name attribute."""
        setattr(namespace, self.dest, values)


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
        action=EnvDefault,
        envvar="PORT",
        default=8000,
    )
    parser.add_argument(
        "--mlflowurl",
        "-u",
        type=str,
        help="MLflow server url for collecting data",
        action=EnvDefault,
        envvar="MLFLOW_URL",
        default="http://localhost:5000/",
    )
    parser.add_argument(
        "--timeout",
        "-t",
        type=int,
        help="Timeout for pooling new metrics.",
        action=EnvDefault,
        envvar="TIMEOUT",
        default=30,
    )
    args = parser.parse_args()

    mlflow.set_tracking_uri(args.mlflowurl)

    # Start the Prometheus HTTP server
    start_http_server(args.port)

    # Periodically collect and update the metrics
    while True:
        collect_metrics()
        time.sleep(args.timeout)  # Adjust the sleep duration as needed
