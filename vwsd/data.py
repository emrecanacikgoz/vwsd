import os
import os.path as osp
import json
import re
from functools import wraps
import random

from tqdm import tqdm
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer
import pytorch_lightning as pl
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from .util import process_path, clean_text, cnet2str


ROOT = osp.abspath(osp.join(osp.dirname(osp.abspath(__file__)), '..'))


class VWSDDataset(Dataset):
    def __init__(self, root, split, transform=None, tokenizer=None, source="GOOGLE", debug=False, **kwargs):
        super().__init__()
        self.root = process_path(root)
        self.split = split
        self.transform = transform
        self.tokenizer = tokenizer
        #print("Dataset:", source)
        self.source = source
        self.debug = debug
        self._load_data()
    
    def _load_data(self):
        assert self.split in ['trial', 'train', 'validate', 'test']
        if self.split == 'trial':
            data_file = osp.join(self.root, 'trial.data.v1.txt')
            label_file = osp.join(self.root, 'trial.gold.v1.txt')
        elif self.split == 'train':
            data_file = osp.join(self.root, 'train.data.v1.txt')
            label_file = osp.join(self.root, 'train.gold.v1.txt')    

        if self.split in ["train", "trial"]:
            with open(data_file, 'r') as f:
                items = [line.strip().split('\t') for line in f.readlines()]
            
            with open(label_file, 'r') as f:
                labels = [line.strip() for line in f.readlines()]
        else:
            items, labels = [], []

        # TODO randomly add items and labels as seen here
	
        self.items = []
        scraped_items = self._read_scraped_data()

        all_img_paths = []
        for item in scraped_items:
            if "language" in item: # if the example is from WiT
                all_img_paths.append(item["image_path"])

        for item in scraped_items:
            if "language" in item:
                negatives = random.sample(all_img_paths, 9)
                while item["image_path"] in negatives:
                    negatives = random.sample(all_img_paths, 9)
                
                negatives.append(item["image_path"])
                random.shuffle(negatives)
                word, context = item["nl_context"].split()[0], item["nl_context"]
                negatives.insert(0, context)
                negatives.insert(0, word)
                items.append(negatives) # TODO
                labels.append(item["image_path"])

        # TODO Newly added section parts
        assert len(items) == len(labels)
        if self.debug:
            items, labels, scraped_items = items[:40], labels[:40], scraped_items[:40]
 
        for inputs, label, scraped_item in zip(items, labels, scraped_items):
            complete_context = inputs[1]
            prompt = self._clean_scraped_data(scraped_item, complete_context)
            
            self.items.append({
                'word': inputs[0].lower(),
                'context': inputs[1].lower(),
                'images': inputs[2:],
                'gold': label,
                'prompt': prompt.lower(),
            })


        for item in self.items[:20]:
            word, context, prompt = item["word"], item["context"], item["prompt"]
            print(f"\nword: {word}")
            print(f"context: {context}")
            print(f"prompt: {prompt}")


    def _read_scraped_data(self):
        print(os.getcwd(), self.split)
        if self.split == "train":
            file_path = "../vwsd/dummy_data/data_train.json"
        elif self.split == "validate":
            file_path = "../vwsd/dummy_data/data_val.json"
        elif self.split == "trial":
            file_path = "../vwsd/data_trial.json"
        elif self.split == "test":
            file_path = "../vwsd/dummy_data/data_test.json"

        with open(file_path, 'r') as scraped_data:
            scraped_items = list(scraped_data)
        
        return [json.loads(item) for item in scraped_items]


    def _clean_scraped_data(self, scraped_item, context):
        assert self.source in ["GOOGLE", "CNET", "BOTH"], "The used data source does not exist"

        #print("Clean scraped data:", self.source)

        obj, obj_1, obj_2, google = scraped_item["all"], scraped_item["first"], scraped_item["second"], scraped_item["google"]
        CLIP_CONTEXT_LENGTH = 76 - len(context.split()) - 1
        
        
        if self.source == "GOOGLE":
            if len(google) == 0:
                print("Scraped google data is empty")
                info = ""
            elif isinstance(google[0], dict) and "snippets" in google[0]:
                info = clean_text(cnet2str(google[0]["snippets"]))
                if info.count(' ') > CLIP_CONTEXT_LENGTH - 1:
                    truncated_info_tokens = info.split()[:CLIP_CONTEXT_LENGTH]
                    info =  " ".join(truncated_info_tokens)
            else:
                info = " . ".join([s for s in google if isinstance(s, str)])
                info = clean_text(info)
                truncated_info_tokens = info.split()[:CLIP_CONTEXT_LENGTH]
                info = " ".join(truncated_info_tokens)
        
        
        if self.source == "CNET":
            # meaning that we could not find the whole context in ConceptNet
            if "error" in obj and obj["error"]["status"] == 404: 
                first_token, second_token = context.split()[0], context.split()[1]
                info_1, info_2 = cnet2str(obj_1, first_token), cnet2str(obj_2, second_token)
                info = info_1 + " " + info_2
                info = clean_text(info)
                truncated_info_tokens = info.split()[:CLIP_CONTEXT_LENGTH]
                info = " ".join(truncated_info_tokens)
            else:
                info = clean_text(cnet2str(obj, context))
                truncated_info_tokens = info.split()[:CLIP_CONTEXT_LENGTH]
                info = " ".join(truncated_info_tokens)

        prompt = f"{context} . {info}"
        
        return prompt

    # TODO! CHANGE HERE
    def _read_image(self, file_name):
        from_wit = "wit" in file_name
        if from_wit:
            file_path = osp.join("../vwsd/dummy_images", file_name)
        elif self.split == "train":
            file_path = osp.join(self.root, 'train_images_v1', file_name)
        elif self.split == "trial":
            file_path = osp.join(self.root, 'trial_images_v1', file_name)
        return Image.open(file_path)

    def _tokenize(self, text):
        return self.tokenizer(text, return_tensors='pt', max_length=77, padding="max_length", truncation=True)

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
            prompt = self._tokenize(item['prompt'])
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
        debug=False,
        batch_size=8,
        num_workers=0,
        transform=None,
        tokenizer=None,
        source="GOOGLE"
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
        self.debug = debug
        self.source = source

    @property
    def pad_token_id(self):
        pad_token_id = 0
        if self.tokenizer is not None:
            pad_token_id = self.tokenizer.pad_token_id
        return pad_token_id

    def setup(self, stage='fit'):
        if stage == 'fit':
            self.train_data = self.load_split(split='train')
            self.val_data = self.load_split(split='validate')
        elif stage == 'test':
            self.test_data = self.load_split(split='test')
        elif stage == 'predict':
            self.trial_data = self.load_split(split='trial')
    
    def load_split(self, split):
        if split == 'train':
            root = self.train_dir
        elif split == 'trial':
            root = self.trial_dir
        else:
            root = "" #TODO

        return VWSDDataset(
            root=root,
            split=split,
            transform=self.transform,
            tokenizer=self.tokenizer,
            debug=self.debug,
            source=self.source
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=clip_collate_fn(self.pad_token_id),
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=clip_collate_fn(self.pad_token_id),
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data,
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
