import pandas as pd
import torch


class PandasDataset(torch.utils.data.Dataset):
    def __init__(self, root, cut):
        self.df = pd.read_csv(root)
        self.df = self.df.iloc[cut[0]:cut[1]]
        self.df = self.df.dropna()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        row = self.df.iloc[item]
        label = row['label'] + 1
        text = row['text']
        return label, text

    def shuffle(self):
        self.df = self.df.sample(frac=1)
        return self
