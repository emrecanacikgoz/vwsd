import os
import os.path as osp
import json
import re

from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer
import pytorch_lightning as pl

from .util import process_path


ROOT = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)), '..'))


class VWSDDataset(Dataset):
    def __init__(self, root, split, processor=None, tokenizer=None, **kwargs):
        super().__init__()
        self.root = process_path(root)
        self.split = split
        self.transform = processor
        self.tokenizer = tokenizer
        self._load_data()
    
    def _load_data(self):
        assert self.split == 'trial'
        data_file = osp.join(self.root, 'trial.data.txt')
        with open(data_file, 'r') as f:
            items = [line.strip().split('\t') for line in f.readlines()]
            
        label_file = osp.join(self.root, 'trial.gold.txt')
        with open(label_file, 'r') as f:
            labels = [line.strip() for line in f.readlines()]

        assert len(items) == len(labels)
        self.items = list()
        for inputs, label in zip(items, labels):
            self.items.append({
                'word': inputs[0],
                'context': inputs[1],
                'images': inputs[2:],
                'gold': label,
            })

    def _read_image(self, file_name):
        file_path = osp.join(self.root, 'all_images', file_name)
        return Image.open(file_path)

    def _tokenize(self, text):
        return self.tokenizer(text, return_tensors='pt', padding=True)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        images = [self._read_image(file_name) for file_name in item['images']]
        label = item['images'].index(item['gold'])

        if self.transform is not None:
            images = [self.transform(image) for image in self.images]

        word = context = prompt = None
        if self.tokenizer is not None:
            word = self._tokenize(item['word'])
            context = self._tokenize(item['context'])
            prompt = self._tokenize(item['word'] + ' ' + item['context'])

        return {
            'raw_word': item['word'],
            'raw_context': item['context'],
            'image_files': item['images'],
            'gold_image': item['gold'],
            'images': images,
            'tokenized_word': word,
            'tokenized_context': context,
            'tokenized_prompt': prompt,
            'label': label,
        }

