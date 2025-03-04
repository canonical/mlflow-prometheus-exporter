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
   docker build -t mlflow-prometheus-exporter .
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
   rockcraft clean && rockcraft pack --verbosity=trace
   ```
3. Copy the resulted rock to your local Docker registry 
   ```
   sudo skopeo --insecure-policy copy oci-archive:mlflow-prometheus-exporter_v1.0.0_22.04_amd64.rock docker-daemon:<registry_user>/mlflow-prometheus-exporter:tag
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

Example ussage: 
```
python mlflow_exporter.py -p 8999 -u http://localhost:31380/ -t 30
PORT=8000 MLFLOW_URL=http://localhost:31380/ TIMEOUT=20 python mlflow_exporter.py
```

## Contributing

If you'd like to contribute to the MLflow Prometheus Exporter, feel free to fork this repository and submit a pull request. Contributions are always welcome!
