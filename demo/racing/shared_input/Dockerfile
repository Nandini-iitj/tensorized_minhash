FROM python:3.12-slim

WORKDIR /app

# Engine deps + source (mirrors repo structure so relative path walking works)
COPY tensorized_minhash/requirements.txt /tmp/engine_req.txt
RUN pip install --no-cache-dir -r /tmp/engine_req.txt

COPY tensorized_minhash/ /app/tensorized_minhash/

COPY log_setup.py /app/log_setup.py
COPY logging.conf /app/logging.conf

# Mirror demo/genome/ so parents[2] resolves to /app/demo/ inside the container
COPY demo/genome/ /app/demo/genome/

# Script at the same relative depth as locally: demo/racing/shared_input/
COPY demo/racing/shared_input/generate_input.py /app/demo/racing/shared_input/generate_input.py

ENV PYTHONUNBUFFERED=1

CMD ["sh", "-c", "python /app/demo/racing/shared_input/generate_input.py --load-level ${LOAD_LEVEL:-7} --out-dir ${OUT_DIR:-/shared_input}"]