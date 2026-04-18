FROM python:3.12-slim

WORKDIR /app

# Install engine dependencies
COPY tensorized_minhash/requirements.txt /app/tensorized_minhash/requirements.txt
RUN pip install --no-cache-dir -r /app/tensorized_minhash/requirements.txt

# Copy engine source
COPY tensorized_minhash/ /app/tensorized_minhash/

COPY log_setup.py /app/log_setup.py
COPY logging.conf /app/logging.conf

# Copy worker
COPY demo/racing/worker_tt/requirements.txt /app/worker_requirements.txt
RUN pip install --no-cache-dir -r /app/worker_requirements.txt

COPY demo/racing/worker_tt/worker.py /app/worker.py

ENV PYTHONUNBUFFERED=1

CMD ["python", "worker.py"]