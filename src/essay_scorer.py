import os
import pandas as pd
import click
import logging
from model2vec import StaticModel
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn

DATA_DIR = '../data'

logger = logging.getLogger(__name__)

torch.manual_seed(42)


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
            embedding_model,
    ):
        logger.info(f'Building data set from tables: {table_names}')
        df = load_and_concat_df(data_dir, table_names)
        logger.info(f'Dataframe:\n{df}')

        self.X = []
        self.y = []
        for task, essay, band in zip(df['task'], df['essay'], df['band']):
            self.X.append(EssayScorer.get_embedding(embedding_model, task, essay))
            self.y.append(band)
        self.X = np.array(self.X)
        self.y = np.array(self.y)

        logger.info(f'Dataset from tables {table_names} has been built with shape: {self.X.shape} and {self.y.shape}')

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], np.array([self.y[idx]], dtype=np.float32)


class EssayScorerHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.float()

        self.fc1 = nn.Linear(1024, 50)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(50, 50)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(50, 50)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(50, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)

        x = self.fc2(x)
        x = self.relu2(x)

        x = self.fc3(x)
        x = self.relu3(x)

        x = self.fc4(x)
        return x


def train_essay_scorer_head(
        dataloader,
        model,
        loss_fn,
        optimizer
):
    size = len(dataloader.dataset)
    model.train()
    for epoch in range(1000): # return to 1000
        for batch, (X, y) in enumerate(dataloader):
            pred = model(X)
            loss = loss_fn(pred, y)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 10 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")



class EssayScorer:
    def __init__(
            self,
            embedding_model_name: str,
            essay_scorer_head_model,
    ):
        self.embedding_model_name = embedding_model_name
        self.embedding_model = load_embedding_model(embedding_model_name)
        self.essay_scorer_head_model = essay_scorer_head_model

    def save(self, path):
        torch.save(self.essay_scorer_head_model.state_dict(), path)

    @staticmethod
    def load(path):
        head = EssayScorerHead()
        head.load_state_dict(torch.load(path, weights_only=True))
        return EssayScorer(
            'hs-hf/jina-embeddings-v3-distilled',
            head,
        )

    @staticmethod
    def get_embedding(
            embedding_model,
            task: str,
            essay: str,
    ) -> np.ndarray:
        task_embedding = embedding_model.encode([task]).squeeze()
        essay_embedding = embedding_model.encode([essay]).squeeze()
        return np.concatenate([task_embedding, essay_embedding])

    def score(
            self,
            task: str,
            essay: str,
    ):
        embedding = EssayScorer.get_embedding(self.embedding_model, task, essay)
        return self.essay_scorer_head_model(torch.FloatTensor([embedding]))


@click.command()
@click.option('--data-dir', default='../data/scorer_data')
@click.option('--train-table-name', 'train_table_names', multiple=True)
@click.option('--test-table-name', 'test_table_names', multiple=True)
@click.option('--embedding-model-name', default='hs-hf/jina-embeddings-v3-distilled',)
def main(
        data_dir: str,
        train_table_names: list[str],
        test_table_names: list[str],
        embedding_model_name: str,
):
    logging.basicConfig(level=logging.INFO)
    logger.info(
        f'Starting\n' +
        f'Data dir: {data_dir}\n' +
        f'Train: {train_table_names}\n' +
        f'Test: {test_table_names}\n' +
        f'Embedding model name: {embedding_model_name}\n'
    )

    embedding_model = load_embedding_model(embedding_model_name)

    band_train_dataset = BandDataset(data_dir, train_table_names, embedding_model)
    band_test_dataset = BandDataset(data_dir, test_table_names, embedding_model)

    essay_scorer_head = EssayScorerHead()
    loss_fn = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(essay_scorer_head.parameters(), lr=0.001)
    logger.info(f'essay_scorer_head: {essay_scorer_head}')
    train_essay_scorer_head(DataLoader(band_train_dataset, batch_size=1000, shuffle=True), essay_scorer_head, loss_fn, optimizer)

    test_loop(DataLoader(band_test_dataset, batch_size=32, shuffle=True), essay_scorer_head, loss_fn)

    EssayScorer(embedding_model_name, essay_scorer_head).save('./head_model')

    EssayScorer.load('./head_model')


if __name__ == '__main__':
    main()
