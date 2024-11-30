import click
import logging
import os
import time
from datetime import datetime
import pandas as pd
import random

import torch

from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer


logger = logging.getLogger(__name__)

NEW_LINE_STR = '<NL>'


def read_used(
        work_dir: str,
) -> set:
    used = set()
    for file in os.listdir(work_dir):
        if not file.endswith('.csv'):
            logger.info(f'Skipping {file} because not ends with .csv')
            continue
        df_path = os.path.join(work_dir, file)
        logger.info(f'Reading used ids from {df_path}')
        df = pd.read_csv(df_path)
        for row_id in df['from_id']:
            used.add(row_id)
    return used


def collect_from_work_dir(
        work_dir: str,
) -> pd.DataFrame:
    result = []
    for file in os.listdir(work_dir):
        if not file.endswith('.csv'):
            logger.info(f'Skipping {file} because not ends with .csv')
            continue
        df_path = os.path.join(work_dir, file)
        df = pd.read_csv(df_path)
        result.append(df)
    return pd.concat(result)


PROMT = (
    'Your task is to:\n' +

    'Enhance the essay\'s clarity, grammar, vocabulary, and sentence structure while preserving the original meaning and ideas.\n' +
    'Avoid introducing new arguments or changing the essay’s overall structure or tone.\n' +
    'Limit changes to essential improvements—do not rewrite the essay extensively or modify more than necessary.\n' +
    'Please provide only the revised essay\n' +

    'Topic: {task}\n' +
    'Essay: {essay}\n' +

    '-----------------------\n'
    'Write new essay under this line and nothing else: \n'
    '-----------------------\n'
)


@click.command()
@click.option('--work-dir', required=False, type=click.Path(exists=True))
@click.option('--batch-size', default=32, type=int)
@click.option('--output-df-path', type=str)
@click.option('--input-df-path', type=click.Path(exists=True))
@click.option('--cleanup', is_flag=True)
@click.option('--model-name', type=str)
def main(
        work_dir: str,
        batch_size: int,
        output_df_path: str,
        input_df_path: str,
        cleanup: bool,
        model_name: str,
):
    logging.basicConfig(level=logging.INFO)

    logger.info(
        'Starting with:\n' +
        f'work_dir={work_dir}\n' +
        f'batch_size={batch_size}\n' +
        f'output_df_path={output_df_path}\n' +
        f'input_df_path={input_df_path}\n' +
        f'cleanup={cleanup}\n' +
        f'model_name={model_name}\n' +
        '-----------------------'
    )

    timestamp_at_start = str(datetime.fromtimestamp(time.time())).replace(' ', '=')

    if work_dir is None:
        work_dir = os.path.join(os.getcwd(), 'tmp_' + timestamp_at_start)
        logger.info(f'Creating new working dir: {work_dir}')
        os.makedirs(work_dir, exist_ok=False)

    used_ids = read_used(work_dir)
    logger.info(f'Used IDs size: {len(used_ids)}')

    input_df = pd.read_csv(input_df_path)

    batch_id = 0
    while True:
        logger.info('================================')
        logger.info(f'Working with batch_id {batch_id}, current used size is {len(used_ids)}')

        batch = []
        for row_id, task, essay, band in zip(input_df['id'], input_df['task'], input_df['essay'], input_df['band']):
            if len(batch) >= batch_size:
                break

            if row_id in used_ids:
                continue

            used_ids.add(row_id)

            batch.append({
                'id': row_id,
                'task': task,
                'essay': essay,
                'band': band,
            })

        if len(batch) == 0:
            logger.info(f'Empty batch, saving results to {output_df_path} and exiting')
            output_df = collect_from_work_dir(work_dir)
            output_df.to_csv(output_df_path, index=False, header=True)
            return

        logger.info(f'Working for batch size {len(batch)}, first id of batch is {batch[0]["id"]}')

        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

        pipe = pipeline(
            "text-generation",
            model=model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_new_tokens=600,
            pad_token_id=tokenizer.eos_token_id
        )

        built_prompts = [
            PROMT.format(task=sample['task'], essay=sample['essay'].replace(NEW_LINE_STR, '\n'))
            for sample in batch
        ]

        logger.info(f'Running pipeline for {len(built_prompts)} samples')

        pipe_result = pipe(built_prompts, pad_token_id=pipe.tokenizer.eos_token_id)

        logger.info(f'Finished running pipeline for {len(built_prompts)} samples')

        result_rows = {
            'id': [],
            'from_id': [],
            'task': [],
            'essay': [],
            'from_essay': [],
            'band': [],
            'from_band': [],
            'timestamp': [],
            'iteration': [],
            'source': [],
        }

        for sample_orig, sample_res in zip(batch, pipe_result):
            result_rows['id'].append(random.randint(1, 10**18))
            result_rows['from_id'].append(sample_orig['id'])
            result_rows['task'].append(sample_orig['task'])
            new_essay = sample_res[0]['generated_text'].split('-----------------------')[-1]
            result_rows['essay'].append(new_essay.replace('\n', NEW_LINE_STR))
            result_rows['from_essay'].append(sample_orig['essay'])
            result_rows['band'].append(-1)
            result_rows['from_band'].append(sample_orig['band'])
            result_rows['timestamp'].append(timestamp_at_start)
            result_rows['iteration'].append(1)
            result_rows['source'].append(model_name)

        batch_df = pd.DataFrame(result_rows)

        batch_timestamp = str(datetime.fromtimestamp(time.time())).replace(' ', '=')
        batch_df_path = os.path.join(work_dir, f'batch_df_{batch_timestamp}.csv')

        batch_df.to_csv(batch_df_path, index=False, header=True)


if __name__ == '__main__':
    main()