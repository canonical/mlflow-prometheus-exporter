FROM ubuntu:24.04
ARG VERSION

ENV TZ=UTC

WORKDIR /app

# to stay consistent with upstream image we separate the dpkg-query to separate layer
RUN mkdir -p /usr/share/rocks; \
    (echo "# os-release" && cat /etc/os-release && echo "# dpkg-query" && dpkg-query -f '${db:Status-Abbrev},${binary:Package},${Version},${source:Package},${Source:Version}\n' -W) > /usr/share/rocks/dpkg.query

RUN set -eux; \
    # install python
    apt-get update; \
    DEBIAN_FRONTEND=noninteractive apt-get full-upgrade -y; \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3-pip \
    tzdata; \
    DEBIAN_FRONTEND=noninteractive apt-get remove --purge --auto-remove -y; \
    rm -rf /var/lib/apt/lists/*

# create an isolated virtual environment to avoid PEP 668 externally-managed errors
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY mlflow_exporter /app

EXPOSE 8000
CMD ["python3", "mlflow_exporter.py"]
