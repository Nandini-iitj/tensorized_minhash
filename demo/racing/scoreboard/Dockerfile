FROM python:3.12-slim

WORKDIR /app

COPY demo/racing/scoreboard/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --timeout=400 -r /app/requirements.txt


# Also copy shared_input generator so the sidebar button can trigger it
COPY demo/racing/shared_input/ /app/shared_input/
COPY demo/genome/ /app/genome/

COPY demo/racing/scoreboard/app.py /app/app.py

ENV PYTHONUNBUFFERED=1

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]