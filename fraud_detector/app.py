import logging
import os
import time
from pathlib import Path
import argparse
import json
import pandas as pd
from confluent_kafka import Consumer, Producer

from fraud_detector.loader import load_model, load_stats, get_pred_probs
from fraud_detector.scripts import preprocess_df, get_score_from_proba


import sys
THRESHOLD = 0.379

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)



def _parse_bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--transactions_topic', default=os.getenv('KAFKA_TRANSACTIONS_TOPIC', 'transactions'))
    ap.add_argument('--scoring_topic', default=os.getenv('KAFKA_SCORING_TOPIC', 'scoring'))

    ap.add_argument('--weights_dir', default=os.getenv('WEIGHTS_DIR', '/app/weights'))
    ap.add_argument('--bootstrap_servers', default=os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'kafka:9092'))
    return ap.parse_args()


class KafkaFraudService:
    def __init__(self, args):
        self.args = args

        logger.info('Importing pretrained model...')
        weights_path = str(args.weights_dir).rstrip('/').rstrip('\\')
        if not Path(f"{weights_path}/catboost_model.cbm").exists():
            raise FileNotFoundError(f"Model not found: {weights_path}/catboost_model.cbm")
        self.model = load_model(weights_path)
        logger.info('Model successfully loaded')

        user_stats_path = Path(f"{weights_path}/user_stats.csv")
        city_stats_path = Path(f"{weights_path}/city_stats.csv")
        if not user_stats_path.exists() or not city_stats_path.exists():
            raise FileNotFoundError(f"Preprocessor data stats not found: {user_stats_path} or {city_stats_path}")
        self.user_stats, self.city_stats = load_stats(weights_path)
        logger.info('Preprocessor data stats successfully loaded')

        self.consumer = Consumer({
            'bootstrap.servers': args.bootstrap_servers,
            'group.id': 'ml-fraud-detection-scorer',
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': True,
        })
        self.consumer.subscribe([args.transactions_topic])

        self.producer = Producer({'bootstrap.servers': args.bootstrap_servers})
        logger.info(f"Consumer and Producer successfully initialized to: {args.transactions_topic}")

    def parse_message(self, msg_value: bytes):
        data = json.loads(msg_value.decode('utf-8'))
        return data['transaction_id'], pd.DataFrame([data['data']])

    def send_result(self, data):
        self.producer.produce(self.args.scoring_topic, value=json.dumps(data).encode('utf-8'))
        self.producer.flush()

    def run(self):
        logger.info("Kafka ML scoring service started")
        try:
            while True:
                msg = self.consumer.poll(1.0)
                if msg is None:
                    continue
                if msg.error():
                    logger.error(f"Kafka error: {msg.error()}")
                    continue

                try:
                    dfid, dfdata = self.parse_message(msg.value())

                    df_proc = preprocess_df(dfdata, self.user_stats, self.city_stats)
                    proba = float(get_pred_probs(self.model, df_proc)[0])
                    fraud_flag = int(get_score_from_proba(proba))

                    out = {
                        "transaction_id": dfid,
                        "score": proba,
                        "fraud": fraud_flag,
                    }
                    self.send_result(out)

                    logger.info(f"Scored tx={dfid} score={proba:.6f} flag={fraud_flag}")

                except Exception as e:
                    logger.error(f"Error processing message: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Error starting Kafka run service: {e}")


def main():
    args = get_args()
    service = KafkaFraudService(args)
    service.run()


if __name__ == "__main__":
    main()
