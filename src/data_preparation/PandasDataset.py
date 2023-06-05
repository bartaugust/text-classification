import pandas as pd
import torch


class PandasDataset(torch.utils.data.Dataset):
    def __init__(self, root, cut,names, add_one):
        self.df = pd.read_csv(root)
        self.df = self.df.iloc[cut[0]:cut[1]]
        self.df = self.df.dropna()
        self.names = names
        self.add_one = add_one

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        row = self.df.iloc[item]
        label = row[self.names.label] + self.add_one
        text = row[self.names.text]
        return label, text

    def shuffle(self):
        self.df = self.df.sample(frac=1)
        return self
