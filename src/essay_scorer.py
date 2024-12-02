import os
import random

import pandas as pd
import click
import logging
from model2vec import StaticModel
import numpy as np
import json
import re

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from tqdm import tqdm

logger = logging.getLogger(__name__)

torch.manual_seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def recursive_generation_from_json_config(
    config: dict,
    used_keys: set,
    current_result: dict,
):
    for key in sorted(config):
        if key in used_keys:
            continue
        new_used_keys = used_keys.copy()
        new_used_keys.add(key)
        for value in config[key]:
            new_current_result = current_result.copy()
            new_current_result[key] = value
            yield from recursive_generation_from_json_config(config, new_used_keys, new_current_result)
        return
    yield current_result

def generate_exp_configs(
    experiments_config_path: str,
):
    with open(experiments_config_path) as experiments_config_file:
        experiments_config = json.load(experiments_config_file)
    hyper_params_configs = experiments_config['hyper_params']
    scorer_model_configs = experiments_config['scorer_model']
    for hyper_params_config in recursive_generation_from_json_config(hyper_params_configs, set(), dict()):
        for scorer_model_config in recursive_generation_from_json_config(scorer_model_configs, set(), dict()):
            yield hyper_params_config, scorer_model_config

def load_embedding_model(
        embedding_mode_name: str,
):
    return StaticModel.from_pretrained(embedding_mode_name)


def load_and_concat_df(
        data_dir: str,
        table_names: list[str],
):
    result = []
    for table_name in table_names:
        table_path = os.path.join(data_dir, f'{table_name}')
        logger.info(f'Loading {table_name} from {table_path}')
        df = pd.read_csv(table_path, engine='python', encoding='utf-8')
        result.append(df)
    return pd.concat(result)


class BandDataset(Dataset):
    def __init__(
            self,
            data_dir: str,
            table_names: list[str],
    ):
        logger.info(f'Building data set from tables: {table_names}')
        df = load_and_concat_df(data_dir, table_names)
        logger.info(f'Dataframe:\n{df}')

        self.tasks = []
        self.essays = []
        self.bands = []
        for task, essay, band in zip(df['task'], df['essay'], df['band']):
            self.tasks.append(task)
            self.essays.append(essay)
            self.bands.append(band)

        logger.info(f'Dataset from tables {table_names} has been built, number of examples {len(self.tasks)}')

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        return (self.tasks[idx], self.essays[idx]), np.array([self.bands[idx]], dtype=np.float32)


class EssayScorer(nn.Module):
    def __init__(
            self,
            scorer_model_config: dict,
    ):
        super().__init__()
        self.float()

        self.config = scorer_model_config

        self.embedding_model = load_embedding_model(scorer_model_config['embeddings_model']['name'])
        self.embedding_cache = dict()
        # embedding_size = scorer_model_config['embeddings_model']['size']

        # "embeds_merging_policy": [
        #     "mean_separate",
        #     "mean_both",
        #     "lstm_separate",
        #     "lstm_both"
        # ],

        # if scorer_model_config['embeds_merging_policy'] == 'lstm_separate':
        #     pass
        # elif scorer_model_config['embeds_merging_policy'] == 'lstm_both':
        #     raise NotImplemented

        last_size = 1024
        layers = []
        for head_layer_desc in scorer_model_config['head_layers']:
            if head_layer_desc == 'relu':
                layers.append(nn.ReLU())
            elif head_layer_desc == 'tanh':
                layers.append(nn.Tanh())
            elif head_layer_desc.startswith('fc_'):
                output_size = int(head_layer_desc.split('_')[-1])
                layers.append(nn.Linear(last_size, output_size))
                last_size = output_size
            elif head_layer_desc.startswith('half_fc_'):
                # apply two independent networks
                # https://github.com/pytorch/vision/issues/720
                output_size = int(head_layer_desc.split('_')[-1])
                pass
            elif head_layer_desc == 'half_product':
                # multiply halfs
                pass
            elif head_layer_desc == 'half_sum':
                # sum halfs
                pass
            else:
                raise Exception(f'Unknown head layer type: {head_layer_desc}')

        self.layers = nn.ModuleList(layers)


    def get_embedding(self, s):
        if s not in self.embedding_cache:
            self.embedding_cache[s] = self.embedding_model.encode([s]).squeeze()
        return self.embedding_cache[s]

    def forward(self, x_raw):
        # x_raw - batch
        # x_raw[0] - list of tasks
        # x_raw[1] - list of essays
        x_embeddings = []
        for task, essay in zip(x_raw[0], x_raw[1]):
            task_embedding = self.get_embedding(task)
            essay_embedding = self.get_embedding(essay)
            x_embeddings.append(np.concatenate([task_embedding, essay_embedding]))
        x = torch.tensor(np.array(x_embeddings))

        for layer in self.layers:
            x = layer(x)

        return x

    def save_to_dir(self, work_dir):
        os.makedirs(work_dir, exist_ok=True)
        with open(os.path.join(work_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)
        torch.save(self.state_dict(), os.path.join(work_dir, 'weights.pt'))

    @staticmethod
    def load(work_dir):
        config = json.load(open(os.path.join(work_dir, 'config.json')))
        essay_scorer = EssayScorer(config)
        essay_scorer.load_state_dict(torch.load(os.path.join(work_dir, 'weights.pt'), weights_only=True))
        return essay_scorer


    def score(
            self,
            task: str,
            essay: str,
    ):
        NEW_LINE_STR = '<NL>'
        task = re.sub(r'[\n\r]', NEW_LINE_STR, task)
        essay = re.sub(r'[\n\r]', NEW_LINE_STR, essay)
        self.eval()
        return self([[task], [essay]])[0]


def train_essay_scorer(
        hyper_params_config,
        dataloader,
        model,
        loss_fn,
        optimizer,
        case_result: dict,
):
    case_result['train_loss'] = []
    model.train()
    last_loss = None
    epoch_loop = tqdm(range(hyper_params_config['max_epochs']))

    early_stop = hyper_params_config['early_stop']
    best_loss = np.inf
    best_loss_epoch = 0

    for epoch in epoch_loop:
        epoch_loop.set_description(f'Epoch {epoch} last_loss {last_loss}')

        for batch, (X, y) in enumerate(dataloader):
            pred = model(X)
            loss = loss_fn(pred, y)
            last_loss = loss.item()

            if last_loss < best_loss:
                best_loss = last_loss
                best_loss_epoch = epoch

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if early_stop != -1 and abs(best_loss_epoch - epoch) > early_stop:
            logger.info(f'Early stopping, last best_loss_epoch: {best_loss_epoch} best_loss: {best_loss}')

        if epoch % 512 == 0:
            case_result['train_loss'].append(last_loss)

    case_result['train_loss'].append(last_loss)


def test_essay_scorer(dataloader, model, loss_fn, case_result):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()

    test_loss /= num_batches
    correct /= size

    case_result['test_loss'] = test_loss


def train_and_test_essay_scorer(
        hyper_params_config: dict,
        scorer_model_config: dict,
        train_data_set: BandDataset,
        test_data_set: BandDataset,
)-> (EssayScorer, dict):
    logger.info('Start training...')

    case_result = {
        'hyper_params_config': hyper_params_config,
        'scorer_model_config': scorer_model_config,
    }

    essay_scorer = EssayScorer(scorer_model_config)
    logger.info(f'essay_scorer: {essay_scorer}')
    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(
        essay_scorer.parameters(),
        lr=hyper_params_config['learning_rate'],
        weight_decay=hyper_params_config['weight_decay'],
    )

    train_essay_scorer(
        hyper_params_config,
        DataLoader(train_data_set, batch_size=hyper_params_config['train_batch_size'], shuffle=True),
        essay_scorer,
        loss_fn,
        optimizer,
        case_result,
    )

    test_essay_scorer(
        DataLoader(test_data_set, batch_size=32, shuffle=True),
        essay_scorer,
        loss_fn,
        case_result,
    )

    return essay_scorer, case_result


@click.command()
@click.option('--data-dir')
@click.option('--train-table-name', 'train_table_names', multiple=True)
@click.option('--test-table-name', 'test_table_names', multiple=True)
@click.option('--experiments-config-path', type=click.Path(exists=True))
@click.option('--work-dir', required=False, type=click.Path(exists=True))
def main(
        data_dir: str,
        train_table_names: list[str],
        test_table_names: list[str],
        experiments_config_path: str,
        work_dir: str,
):
    logging.basicConfig(level=logging.INFO)
    logger.info(
        '=============================================\n' +
        '=============================================\n' +
        '=============================================\n' +
        f'Starting\n' +
        f'Data dir: {data_dir}\n' +
        f'Train: {train_table_names}\n' +
        f'Test: {test_table_names}\n' +
        f'Experiments config path: {experiments_config_path}\n' +
        f'Working dir: {work_dir}\n'
    )

    logger.info(f'device: {device}')

    logger.info('Loading data')
    band_train_dataset = BandDataset(data_dir, train_table_names)
    band_test_dataset = BandDataset(data_dir, test_table_names)

    best_loss = np.inf
    final_result = []

    logger.info('Running experiments')
    exp_configs = list(generate_exp_configs(experiments_config_path))
    random.shuffle(exp_configs)
    for hyper_params_config, scorer_model_config in exp_configs:
        config_iteration = len(final_result)
        logger.info(f'Current config iteration: {config_iteration}')
        logger.info(f'hyper_params_config: {json.dumps(hyper_params_config, indent=2)}')
        logger.info(f'scorer_model_config: {json.dumps(scorer_model_config, indent=2)}')

        scorer_model, case_result = train_and_test_essay_scorer(
            hyper_params_config,
            scorer_model_config,
            band_train_dataset,
            band_test_dataset,
        )

        case_result['train_test_paths'] = {
            'data_dir': data_dir,
            'train_table_names': train_table_names,
            'test_table_names': test_table_names,
        }

        final_result.append(case_result)

        with open(os.path.join(work_dir, 'final_result.json'), 'w', encoding='utf-8') as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)

        if case_result['test_loss'] < best_loss:
            best_loss = case_result['test_loss']
            scorer_model.save_to_dir(os.path.join(work_dir, 'best_model'))


if __name__ == '__main__':
    main()
