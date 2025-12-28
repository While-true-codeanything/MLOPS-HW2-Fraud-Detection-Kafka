FROM python:3.12-slim

WORKDIR /app

COPY database_logger/ ./database_logger/

COPY database_logger/requirements.txt ./database_logger/requirements.txt
RUN pip install --no-cache-dir -r ./database_logger/requirements.txt

CMD ["python", "./database_logger/app.py"]
