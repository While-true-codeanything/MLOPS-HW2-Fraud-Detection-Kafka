import logging
import os
import time
from pathlib import Path
import argparse

from watchdog.observers.polling import PollingObserver as Observer
from watchdog.events import FileSystemEventHandler

from loader import load_model, load_stats, load_test
from scripts import preprocess_df, get_pred_probs, get_topn_feature_importance, \
    get_predict_density_distribution, get_predictions

import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)

logger = logging.getLogger(__name__)
logger.info('Importing pretrained model...')


def _parse_bool(v: str) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir', default=os.getenv('INPUT_DIR', '/app/input'))
    ap.add_argument('--output_dir', default=os.getenv('OUTPUT_DIR', '/app/output'))
    ap.add_argument('--topn', type=int, default=int(os.getenv('TOPN', '5')))
    ap.add_argument('--additional_info', type=_parse_bool, default=_parse_bool(os.getenv('ADDITIONAL_INFO', '1')))
    ap.add_argument('--weights_dir', default=os.getenv('WEIGHTS_DIR', '/app/weights'))
    return ap.parse_args()


class InferenceService(FileSystemEventHandler):
    def __init__(self, input_dir, output_dir, additional_model_info, top_n, weights_path='weights/'):
        super().__init__()
        logger.info('Initializing Monitoring Service')
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

        weights_path = str(weights_path).rstrip('/').rstrip('\\')
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

        self.additional_model_info = additional_model_info
        self.top_n = top_n
        logger.info('Monitoring Service Initialized')

    def process_file(self, path: Path):
        logger.info(f'Processing: {path}')
        try:
            dir_for_loader = path if path.is_dir() else path.parent
            test = load_test(str(dir_for_loader))
            test = preprocess_df(test, self.user_stats, self.city_stats)
            logger.info(f'{path} Successfully preprocessed')
            y_test_proba = get_pred_probs(self.model, test)
            logger.info(f'{path} Successfully predicted probs')
            if self.additional_model_info:
                get_topn_feature_importance(self.model, test, str(self.output_dir), n=self.top_n)
                get_predict_density_distribution(y_test_proba, str(self.output_dir))
                logger.info(f'{path} Additional Predict info successfully saved')
            get_predictions(y_test_proba, path, str(self.output_dir))
            logger.info(f'{path} Predictions saved to predict_{path.name}')
        except Exception as e:
            logger.error(f'Error processing file {path}: {e}', exc_info=True)

    def on_created(self, event):
        if event.is_directory:
            return
        p = Path(event.src_path)
        if p.suffix.lower() != ".csv":
            return
        logger.info(f'File created: {p}')
        self.process_file(p)

    def on_modified(self, event):
        if event.is_directory:
            return
        p = Path(event.src_path)
        if p.suffix.lower() != ".csv":
            return
        logger.info(f'File modified: {p}')
        self.process_file(p)


def main():
    args = get_args()
    inp = Path(args.input_dir)
    out = Path(args.output_dir)
    top_n = int(args.topn)
    additional_info = bool(args.additional_info)
    weights_dir = args.weights_dir

    logger.info('Mounting or Creating input directory...')
    inp.mkdir(parents=True, exist_ok=True)
    logger.info('Mounting or Creating output directory...')
    out.mkdir(parents=True, exist_ok=True)

    handler = InferenceService(inp, out, additional_info, top_n, weights_path=weights_dir)
    observer = Observer()
    logger.info(f'Watching: {inp}')
    observer.schedule(handler, path=str(inp), recursive=True)
    observer.start()
    logger.info('Monitoring Process in effect')

    for p in inp.glob('**/*'):
        if p.is_file() and p.suffix.lower() == '.csv':
            handler.process_file(p)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
    logger.info('File observer ended')


if __name__ == "__main__":
    main()
