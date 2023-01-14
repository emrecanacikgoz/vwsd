import os
import os.path as osp
import json
import re
from functools import wraps

from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import pytorch_lightning as pl

from .util import process_path


ROOT = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)), '..'))


class VWSDDataset(Dataset):
    def __init__(self, root, split, transform=None, tokenizer=None, **kwargs):
        super().__init__()
        self.root = process_path(root)
        self.split = split
        self.transform = transform
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
            images = [self.transform(x, return_tensors='pt') for x in images]
            images = [x['pixel_values'] for x in images]
            images = torch.cat(images, dim=0).unsqueeze(0)

        word_ids = context_ids = prompt_ids = None
        word_mask = context_mask = prompt_mask = None
        if self.tokenizer is not None:
            word = self._tokenize(item['word'])
            word_ids = word['input_ids']
            word_mask = word['attention_mask']
            context = self._tokenize(item['context'])
            context_ids = context['input_ids']
            context_mask = context['attention_mask']
            prompt = self._tokenize(item['word'] + ' ' + item['context'])
            prompt_ids = prompt['input_ids']
            prompt_mask = prompt['attention_mask']

        return {
            'index': index,
            'raw_word': item['word'],
            'raw_context': item['context'],
            'image_files': item['images'],
            'gold_image': item['gold'],
            'pixel_values': images,
            'word_ids': word_ids,
            'word_mask': word_mask,
            'context_ids': context_ids,
            'context_mask': context_mask,
            'prompt_ids': prompt_ids,
            'prompt_mask': prompt_mask,
            'label': label,
        }


class VWSDDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_dir=None,
        trial_dir=None,
        batch_size=8,
        num_workers=0,
        transform=None,
        tokenizer=None,
    ):
        super().__init__()
        self.train_dir = self.trial_dir = None
        if train_dir is not None:
            self.train_dir = process_path(train_dir)
        if trial_dir is not None:
            self.trial_dir = process_path(trial_dir)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform
        self.tokenizer = tokenizer

    @property
    def pad_token_id(self):
        pad_token_id = 0
        if self.tokenizer is not None:
            pad_token_id = self.tokenizer.pad_token_id
        return pad_token_id

    def setup(self, stage='fit'):
        if stage == 'fit':
            raise NotImplementedError('Training phase is not implemented yet.')
        
        if stage == 'predict' or stage == 'test':
            self.trial_data = self.load_split(split='trial')
    
    def load_split(self, split):
        if split == 'train':
            root = self.train_dir
        elif split == 'trial':
            root = self.trial_dir

        return VWSDDataset(
            root=root,
            split=split,
            transform=self.transform,
            tokenizer=self.tokenizer,
        )

    def test_dataloader(self):
        return DataLoader(
            self.trial_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=clip_collate_fn(self.pad_token_id),
        )

    def predict_dataloader(self):
        return DataLoader(
            self.trial_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=clip_collate_fn(self.pad_token_id),
        )


def clip_collate_fn(pad_token_id):
    def _collate_fn(batch):
        _helper = lambda key: [x[key] for x in batch]
        B = len(batch)
        T = max([x.numel() for x in _helper('prompt_ids')])
        input_ids = torch.empty((B, T), dtype=torch.long)
        input_ids.fill_(pad_token_id)
        attention_mask = torch.empty((B, T), dtype=torch.long)
        attention_mask.fill_(0)

        for i, item in enumerate(batch):
            L = item['prompt_ids'].numel()
            input_ids[i, :L] = item['prompt_ids']
            attention_mask[i, :L] = item['prompt_mask']
        
        pixel_values = torch.cat(_helper('pixel_values'), dim=0)
        labels = torch.tensor(_helper('label')).view(1, -1)

        return {
            'pixel_values': pixel_values,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'indexes': _helper('index'),
            'raw_words': _helper('raw_word'),
            'raw_contexts': _helper('raw_context'),
            'image_files': _helper('image_files'),
        }

    return _collate_fn