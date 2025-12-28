import os
import json
import time
import pandas as pd
from confluent_kafka import Producer

BOOTSTRAP = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
TOPIC = os.getenv("KAFKA_TRANSACTIONS_TOPIC", "transactions")

def main():
    producer = Producer({"bootstrap.servers": BOOTSTRAP})

    df = pd.read_csv("/app/input/test.csv")

    for i, row in df.iterrows():
        payload = {
            "transaction_id": f"{i}",
            "data": row.to_dict()
        }
        producer.produce(TOPIC, value=json.dumps(payload).encode("utf-8"))
        producer.poll(0)
        #time.sleep(0.001)

    producer.flush(10)

if __name__ == "__main__":
    main()
