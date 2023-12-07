# -*- coding: utf-8 -*-
"""DataSampling.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1WQVYwxEMQ1FO-XXd9_S5UWV1lkTLlLw0
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import ast
import re

class GraphImageDataset(Dataset):
    def __init__(self, csv_files, transform=None):
        self.data = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
      x, y = self.data.iloc[idx]
      x_tensor = torch.tensor(self.str2list(x), dtype=torch.float32)
      y_out = str(y)
      return x_tensor, y_out

    def str2list(self, string):
      processed_string = re.findall(r'\b\d+\b', string)
      rgb_values = [int(match) for match in processed_string]
      nested_list = [rgb_values[i:i+3] for i in range(0, len(rgb_values), 3)]
      return nested_list
    
# you might want to change the file names based on your address
csv_files1 = ['/content/data_kk0.csv', '/content/data_cr0.csv', '/content/data_gv0.csv', '/content/data_sp0.csv']
csv_files2 = ['/content/data_kk0.csv', '/content/data_cr0.csv', '/content/data_gv0.csv', '/content/data_sp0.csv',
              '/content/data_kk1.csv', '/content/data_cr1.csv', '/content/data_gv1.csv', '/content/data_sp1.csv']
dataset_default = GraphImageDataset(csv_files=csv_files1)
dataset_mixed = GraphImageDataset(csv_files=csv_files2)