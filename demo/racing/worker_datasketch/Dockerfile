FROM python:3.12-slim

WORKDIR /app

COPY demo/racing/worker_datasketch/requirements.txt /app/worker_requirements.txt
RUN pip install --no-cache-dir --timeout=400 -r /app/worker_requirements.txt

COPY log_setup.py /app/log_setup.py
COPY logging.conf /app/logging.conf
COPY demo/racing/worker_datasketch/worker.py /app/worker.py

ENV PYTHONUNBUFFERED=1

CMD ["python", "worker.py"]