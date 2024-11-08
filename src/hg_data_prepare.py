import os
import pandas as pd
import random
import logging
import time
from datetime import datetime

DATA_DIR = '../data/raw_hg'

INPUT_TABLES = [
    'chillies_gpt_eval_train.csv',
    'chillies_gpt_eval_test.csv',
]

OUTPUT_DIR = '../data/scorer_data'

logger = logging.getLogger(__name__)


def read_tables(
        data_dir: str,
        input_tables: list[str],
) -> list[tuple[str, pd.DataFrame]]:
    input_data_frames = []
    for table in input_tables:
        table_path = os.path.join(data_dir, table)
        logger.info(f'Reading {table} from {table_path}')
        df = pd.read_csv(table_path)
        logger.info(f'Table {table} contains {len(df)} rows')
        input_data_frames.append((table, df))
    return input_data_frames


def prepare_table(
        table: pd.DataFrame,
) -> pd.DataFrame:
    result = {
        'id': [],
        'question': [],
        'essay': [],
        'score': [],
        'timestamp': [],
        'iteration': [],
        'source': [],
    }

    for prompt, essay, band in zip(table['prompt'], table['essay'], table['band']):
        processed_question = str(prompt.strip())
        processed_essay = str(essay.strip())

        processed_score = band.strip()
        if processed_score == '<4':
            processed_score = '0'
        processed_score = float(processed_score)

        if len(processed_question) < 10 or len(processed_essay) < 10:
            logger.info(f'Skipped bad row with question and essay: \n{processed_question}\n{processed_essay}')
            continue

        timestamp = str(datetime.fromtimestamp(time.time()))
        result['id'].append(random.randint(1, 10**18))
        result['question'].append(processed_question)
        result['essay'].append(processed_essay)
        result['score'].append(processed_score)
        result['timestamp'].append(timestamp)
        result['iteration'].append(0) # iteration 0 means before any RLFH or additional data mining
        result['source'].append('hf')

    return pd.DataFrame(data=result)


def main():
    logging.basicConfig(level=logging.INFO)
    tables = read_tables(DATA_DIR, INPUT_TABLES)
    for table_name, table_content in tables:
        prepared_table = prepare_table(table_content)
        output_table_name = 'iter0_' + table_name
        output_path = os.path.join(OUTPUT_DIR, f'{output_table_name}')
        logger.info(f'Writing processed {table_name} to {output_path}')
        prepared_table.to_csv(output_path)


if __name__ == '__main__':
    main()
