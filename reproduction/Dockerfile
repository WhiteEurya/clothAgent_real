FROM python:3.10-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git graphviz libgl1 libglib2.0-0 libusb-1.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /home/sja/clothAgent_real
COPY reproduction/requirements-cali.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
# Historical scripts use this interpreter path. Keep it as an alias.
RUN mkdir -p /home/sja/miniconda3/envs/cali/bin && \
    ln -s /usr/local/bin/python /home/sja/miniconda3/envs/cali/bin/python
ENV PYTHONPATH=/home/sja/clothAgent_real
CMD ["python", "-m", "cloth_agent.experiments.graph_policy_replay", "--help"]
