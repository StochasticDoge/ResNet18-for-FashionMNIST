import logging
import os
import pickle
import time

import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics as metrics
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from torchvision import datasets
from torchvision.transforms import ToTensor, Normalize, Compose

from seminarski_b.model import Net, MyResNet

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)


device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
out_fpath = os.path.join(os.path.dirname(__file__), 'results')
if not os.path.isdir(out_fpath):
    os.makedirs(out_fpath)


def validate(model, train_loader, val_loader):
    accdict = {}
    for name, loader in [("train", train_loader), ("val", val_loader)]:
        correct = 0
        total = 0

        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(device=device)
                labels = labels.to(device=device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.shape[0]
                correct += int((predicted == labels).sum())

        logger.info("Accuracy {}: {:.2f}".format(name, correct / total))
        accdict[name] = correct / total
    return accdict


def analyze_results(model=None, saved_mfile=None, channels=None):
    if saved_mfile is not None:
        logger.info(f"Analyzing '{saved_mfile}' with {channels} channels")
    if model is None:
        if saved_mfile is not None:
            raise ValueError("If `model` is not provided, "
                             "proivde a path to the saved one instead")
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )
    train_loader = DataLoader(training_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(test_data, batch_size=64, shuffle=False)
    if model is None:
        assert saved_mfile is not None
        model = Net(training_data[0][0], channels)
        model.load_state_dict(torch.load(saved_mfile))
    return validate(model, train_loader, val_loader)


def load_pickle(fname, subdir=None):
    path = out_fpath
    if subdir is not None:
        path = os.path.join(path, subdir)
    path = os.path.join(path, fname)
    with open(path, 'rb') as f:
        return pickle.load(f)


def pickle_obj(fname, obj, subdir=None):
    path = out_fpath
    if subdir is not None:
        path = os.path.join(path, subdir)
    path = os.path.join(path, fname)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


class FashionClassifier:
    def __init__(self, epochs=500, lr=0.1, mini_batch_sz=256, version=1):
        self.epochs = epochs
        self.lr = lr
        self.mini_batch_sz = mini_batch_sz
        self.ver = version
        self._train_loader = None
        self._test_loader = None

    @property
    def train_loader(self):
        if not self._train_loader:
            self._train_loader, self._test_loader = self.load_dataset()
        return self._train_loader

    @property
    def test_loader(self):
        if not self._test_loader:
            self._train_loader, self._test_loader = self.load_dataset()
        return self._test_loader

    def load_dataset(self):
        trn_data = datasets.FashionMNIST(
            root="data",
            train=True,
            download=True,
            transform=ToTensor())
        tst_data = datasets.FashionMNIST(
            root="data",
            train=False,
            download=True,
            transform=ToTensor())
        mean, std = torch.std_mean(trn_data.data.double())
        # TODO: Should we take mean + std over train and test data combined
        transform = Compose([
            ToTensor(),
            Normalize(mean, std)
        ])
        trn_data.transform = transform
        tst_data.transform = transform

        train_loader = torch.utils.data.DataLoader(
            trn_data, batch_size=self.mini_batch_sz, shuffle=True,
            collate_fn=lambda x: tuple(
                x_.to(device) for x_ in default_collate(x))
        )
        test_loader = torch.utils.data.DataLoader(
            tst_data, batch_size=self.mini_batch_sz, shuffle=True,
            collate_fn=lambda x: tuple(
                x_.to(device) for x_ in default_collate(x))
        )
        return train_loader, test_loader

    def evaluate_model(self, model):
        """
        Evaluate various performance metrics on both train and test datasets
        """
        # TODO: Calculate and display other metrics, precision, recall, accuracy
        #   plot the confusion matrix at the end of the pipeline
        results = {}
        for name, loader in [("train", self.train_loader),
                             ("test", self.test_loader)]:

            lbls = []
            pred = []
            with torch.no_grad():
                for imgs, labels in loader:
                    outputs = model(imgs)
                    _, predicted = torch.max(outputs, dim=1)
                    lbls.append(labels)
                    pred.append(predicted)

            labels = torch.cat(lbls).cpu()
            pred = torch.cat(pred).cpu()

            # micro = global stats
            # macro = per label stats then mean
            acc = metrics.accuracy_score(labels, pred)
            prec = metrics.precision_score(labels, pred, average='micro')
            m = metrics.confusion_matrix(labels, pred)
            err = 1 - acc
            results[name] = (m, err, prec)
        return results

    def train(self, n_epochs, optimizer, lr_scheduler, model, loss_fn):
        start = time.time()
        res = dict()
        for n in range(1, n_epochs + 1):
            for imgs, labels in self.train_loader:
                outputs = model(imgs)
                loss = loss_fn(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # evaluation and logging
            errors = self.evaluate_model(model)
            res[n] = errors

            if n % 10 == 0:
                m, err, _ = errors['test']
                plt.imshow(m, cmap='gray')
                plt.show()
            logger.info(
                "Ep {}, time {:d}s, err(%) - {}: {:.1f}, {}: {:.1f}".format(
                    n, round(time.time() - start),
                    'train', 100 * errors['train'][1],
                    'test', 100 * errors['test'][1]))

            if lr_scheduler is not None:
                lr_scheduler.step(errors['test'][1])
        return model, res

    def main(self):
        model = MyResNet()

        opt = optim.SGD(model.parameters(), lr=self.lr,
                        weight_decay=1e-4, momentum=0.9)
        lr_sched = ReduceLROnPlateau(opt, patience=25, min_lr=self.lr / 100)

        # lr_sched = None
        # opt = optim.Adam(model.parameters())
        model, res = self.train(
            n_epochs=self.epochs,
            optimizer=opt,
            model=model,
            loss_fn=nn.CrossEntropyLoss(),
            lr_scheduler=lr_sched,
        )
        self.dump_results(model, res)
        return model, res

    def dump_results(self, model, res):
        """
        Save model's state, evaluation results and hyperparameters locally
        """
        pickle_obj('metrics', res, subdir=f'ver{self.ver}')
        torch.save(model.state_dict(),
                   os.path.join(out_fpath, f'ver{self.ver}', 'resnet'))
        with open(os.path.join(out_fpath, f'ver{self.ver}', 'params.txt'),
                  'wt') as f:
            f.write(f"lr = {self.lr}\nepochs = {self.epochs}"
                    f"\nmini_batch_sz = {self.mini_batch_sz}")


def calculate_patience(df=None):
    if df is not None:
        df = pd.read_pickle(os.path.join(out_fpath, 'ver1', 'df'))
    best = df['test'].iloc[0]
    dist = [(1, best)]
    for i, val in df['test'].items():
        if val < best:
            dist.append((i, val))
            best = val
    diffs = pd.Series([idx - dist[i][0]
                       for i, (idx, val) in enumerate(dist[1:])])
    return diffs, int(diffs.mean() + diffs.std())


# TODO: Regularization
# TODO: Preprocessing:
#   - augmentation (random crops/horizontal, vertical flips)
#   - random translation
#   - random rotation
def main():
    fc = FashionClassifier(epochs=1000, lr=1e-1,
                           mini_batch_sz=1024, version=3)
    fc.main()


if __name__ == '__main__':
    main()
