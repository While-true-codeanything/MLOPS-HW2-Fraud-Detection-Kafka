FROM python:3.12-slim

WORKDIR /app

COPY interface/ ./interface/
COPY interface/requirements.txt ./interface/requirements.txt
RUN pip install --no-cache-dir -r ./interface/requirements.txt



EXPOSE 8501

CMD ["streamlit", "run", "./interface/app.py", "--server.port=8501", "--server.address=0.0.0.0"]