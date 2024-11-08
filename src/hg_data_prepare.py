import os
import pandas as pd
import random
import logging
import time
from datetime import datetime

DATA_DIR = '../data/raw_hg'

# TODO: change to argparse
INPUT_TABLES_NAMES = [
    'chillies_gpt_eval_train.csv',
    'chillies_gpt_eval_test.csv',
]

OUTPUT_DIR = '../data/scorer_data'

logger = logging.getLogger(__name__)


def read_tables(
        data_dir: str,
        input_table_names: list[str],
) -> list[tuple[str, pd.DataFrame]]:
    input_data_frames = []
    for table_name in input_table_names:
        table_path = os.path.join(data_dir, table_name)
        logger.info(f'Reading {table_name} from {table_path}')
        df = pd.read_csv(table_path)
        logger.info(f'Table {table_name} contains {len(df)} rows')
        input_data_frames.append((table_name, df))
    return input_data_frames


def prepare_table(
        table_content: pd.DataFrame,
) -> pd.DataFrame:
    result = {
        'id': [],
        'task': [],
        'essay': [],
        'band': [],
        'timestamp': [],
        'iteration': [],
        'source': [],
    }

    for prompt, essay, band in zip(table_content['prompt'], table_content['essay'], table_content['band']):
        processed_task = str(prompt.strip())
        processed_essay = str(essay.strip())

        processed_band = band.strip()
        if processed_band == '<4':
            processed_band = '0'
        processed_band = float(processed_band)

        if len(processed_task) < 10 or len(processed_essay) < 10:
            logger.info(f'Skipped bad row with question and essay: \n{processed_task}\n{processed_essay}')
            continue

        timestamp = str(datetime.fromtimestamp(time.time()))
        result['id'].append(random.randint(1, 10**18))
        result['task'].append(processed_task)
        result['essay'].append(processed_essay)
        result['band'].append(processed_band)
        result['timestamp'].append(timestamp)
        result['iteration'].append(0) # iteration 0 means before any RLFH or additional data mining
        result['source'].append('hf')

    return pd.DataFrame(data=result)


def main():
    logging.basicConfig(level=logging.INFO)
    tables = read_tables(DATA_DIR, INPUT_TABLES_NAMES)
    for table_name, table_content in tables:
        prepared_table = prepare_table(table_content)
        output_table_name = 'iter0_' + table_name
        output_path = os.path.join(OUTPUT_DIR, f'{output_table_name}')
        logger.info(f'Writing processed {table_name} to {output_path}')
        prepared_table.to_csv(output_path)


if __name__ == '__main__':
    main()
