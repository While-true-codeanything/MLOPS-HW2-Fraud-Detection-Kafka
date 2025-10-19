FROM python:3.12-slim

WORKDIR /app

RUN mkdir -p /app/logs && \
    touch /app/logs/service.log && \
    chmod -R 777 /app/logs

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY fraud_detector/ ./fraud_detector/
COPY weights/ ./weights/


VOLUME /app/input
VOLUME /app/output

CMD ["python", "./fraud_detector/app.py"]