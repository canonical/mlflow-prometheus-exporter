# MLflow Prometheus Exporter Docker Image

This Docker image contains the MLflow Prometheus Exporter, which collects additional metrics from MLflow server and exposes them in the Prometheus exposition format.

## Usage

To use this Docker image, you can either pull it from a Docker registry or build it locally.

### Pull from Docker Registry

You can pull the pre-built Docker image from a Docker registry using the following command:

```shell
docker pull charmedkubeflowy/mlflow-prometheus-exporter:tag
```

### Build Locally Docker

If you prefer to build the Docker image locally, you can follow these steps:

1. Clone the repository:

   ```shell
   git clone https://github.com/your-username/mlflow-prometheus-exporter.git
   ```

2. Build the Docker image:

   ```shell
   docker build -t mlflow-prometheus-exporter mlflow_exporter
   ```

3. Run a container using the built image:

   ```shell
   docker run -d -p 8000:8000 --name exporter mlflow-prometheus-exporter
   ```

   Adjust the port mapping (`-p`) and container name (`--name`) as needed.

### Build Locally Rock
This repository contains also `rockcraft.yaml` which can be used to build rock oci image. To build the rock follow these steps:

1. Install essential tools 
   ```
   sudo snap install rockcraft --classic --edge
   sudo snap install skopeo --edge --devmode
   ```
2. Build the rock 
   ```
   cd mlflow_exporter
   # Export the locked runtime dependencies to src/requirements.txt (consumed by the rock build)
   # and pack the rock. Requires poetry and the poetry-plugin-export plugin.
   tox -e pack
   ```
   Alternatively, export the requirements manually and run rockcraft directly:
   ```
   cd mlflow_exporter
   poetry export --directory src --only main --without-hashes --format requirements.txt --output requirements.txt
   rockcraft clean && rockcraft pack --verbosity=trace
   ```
3. Copy the resulted rock to your local Docker registry 
   ```
   sudo skopeo --insecure-policy copy oci-archive:mlflow-prometheus-exporter_3.15.1_amd64.rock docker-daemon:<registry_user>/mlflow-prometheus-exporter:tag
   ```
4. Now you can locally run it using Docker daemon
   ```
   docker run -d -p 8000:8000 --name exporter <registry_user>/mlflow-prometheus-exporter:tag
   ```
5. You can also store it on Dockerhub 
   ```
   docker push <registry_user>/mlflow-prometheus-exporter:tag
   ```

## Configuration

The MLflow Prometheus Exporter can be configured using environment variables:

- `PORT`: The port on which the exporter server will run (default: `8000`).
- `MLFLOW_URL`: The MLflow server URL for collecting data (default: `http://localhost:5000/`).
- `TIMEOUT`: The timeout for polling new metrics in seconds (default: `30`).
- `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD`: The credentials to authenticate to the MLflow server with [Basic auth](https://mlflow.org/docs/latest/self-hosting/security/basic-http-auth/#using-environment-variables), both for collecting data and for fetching its traffic metrics (default: unset, i.e. no authentication).

Example ussage: 
```
python src/mlflow_exporter.py -p 8999 -u http://localhost:31380/ -t 30
PORT=8000 MLFLOW_URL=http://localhost:31380/ TIMEOUT=20 python src/mlflow_exporter.py
MLFLOW_TRACKING_USERNAME=<username> MLFLOW_TRACKING_PASSWORD=<password> python src/mlflow_exporter.py -u http://localhost:31380/
```

## Metrics

The exporter serves the following metrics on its `/metrics` endpoint:

- Counts of MLflow resources, collected through the MLflow API every `TIMEOUT` seconds, for each of the following metric names:
  - `num_experiments`: the number of experiments.
  - `num_runs`: the number of runs.
  - `num_registered_models`: the number of registered models (excluding prompts).
  - `num_prompts`: the number of prompts.
  - `num_model_versions`: the number of model versions (excluding prompt versions).
  - `num_logged_models`: the number of logged models.

  They are reported as:
  - Deployment-level metrics: `mlflow_metric{metric_name="<metric_name>"}`. When the MLflow server runs with workspaces enabled, these are the totals across all the workspaces visible to the exporter's MLflow user, and a total is only reported when it was counted in every workspace.
  - Per-workspace metrics, only when the MLflow server runs with workspaces enabled (`--enable-workspaces`): `mlflow_workspace_metric{metric_name="<metric_name>",workspace="<workspace>"}`, for each workspace visible to the exporter's MLflow user.
- Traffic metrics of the MLflow tracking server (e.g., `mlflow_http_request_total` and `mlflow_http_request_duration_seconds`), fetched from the tracking server's `/metrics` endpoint under `MLFLOW_URL` at each scrape, and re-exposed as they are. They are only available when the tracking server runs with [`--expose-prometheus`](https://mlflow.org/docs/latest/api_reference/cli.html#mlflow-server).
- The exporter's own process and Python runtime metrics (e.g., `process_cpu_seconds_total`).

Each source of metrics is collected independently: when one fails (e.g., the tracking server is unreachable or rejects the credentials), the failure is logged and only the metrics of that source are left out, while the others keep being served. Likewise, each metric of each workspace is counted independently.

## Contributing

If you'd like to contribute to the MLflow Prometheus Exporter, feel free to fork this repository and submit a pull request. Contributions are always welcome!
