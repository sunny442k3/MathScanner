import os
import glob
import json
import torch
from PIL import Image
from functools import partial
from collections import Counter
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader


def read_json(filename):
    with open(filename, 'rb') as f:
        data = json.load(f)
        data = [
            {
                "latex": row["latex"],
                "filename": row["filename"]
            } for row in data
        ]
        return data

        
class CustomDataset(Dataset):

    def __init__(self, paths, labels, width, height):
        self.paths = paths
        self.labels = labels
        self.transform = transforms.Compose([
            transforms.Resize(size=(width, height)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    
    def __len__(self):
        return len(self.paths)


    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        img = self.transform(img)
        label = self.labels[idx]
        return img, label


class Tokenizer:

    def __init__(self):
        self.token2text = {
            0: "<pad>",
            1: "<sos>",
            2: "<eos>",
            3: "<unk>"
        }
        self.text2token = {
            "<pad>": 0,
            "<sos>": 1,
            "<eos>": 2,
            "<unk>":3
        }
        self.vocab = ["<pad>","<sos>","<eos>","<unk>"]
        self.pad_idx = 0
        self.sos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3


    @staticmethod
    def process_text(text):
        text = list(text.lower())
        word = ""
        in_tmp = False
        word_list = []
        for char in text:
            if len(char) == 0:
                continue
            if char == "\\":
                if len(word):
                    word_list.append(word)
                    word = ""
                in_tmp = True
                word += char
                continue
            if in_tmp and (ord(char) >= 97 and ord(char) <= 122):
                word += char 
                continue
            in_tmp = False
            if len(word):
                word_list.append(word)
                word = ""
            word_list.append(char)     
        if len(word):
            word_list.append(word)
        return word_list


    def build_vocab(self, dataset):
        char_list = []
        for label in dataset:
            label = self.process_text(label)
            char_list += label     
        counter_vocab = dict(Counter(char_list))
        counter_vocab = list(counter_vocab.items())
        counter_vocab.sort(key=lambda x: x[1], reverse=True)
        for idx, (word, feq) in enumerate(counter_vocab, 4):
            self.vocab.append(word)
            self.token2text[idx] = word 
            self.text2token[word] = idx 


    def build_vocab_from_dict(self, vocab_dict):
        for idx, (key, val) in enumerate(vocab_dict.items(),4):
            val = val.lower()
            key = int(key)
            if val not in self.vocab and key not in self.token2text:
                self.vocab.append(key)
                self.token2text[key] = val
                self.text2token[val] = key
                


    def __len__(self):
        return len(self.vocab)


    def encode(self, text):
        text = self.process_text(text)
        token = []
        token.append(self.sos_idx)
        for word in text:
            if word in self.vocab:
                token.append(self.text2token[word])
            else:
                token.append(self.unk_idx)
        token.append(self.eos_idx)
        return token

    
    def encode_batch(self, batch_text):
        tokens = []
        for text in batch_text:
            token = self.encode(text)
            tokens.append(token)
        return tokens

    
    def decode(self, token):
        text = []
        for idx in token[1:]:
            text.append(self.token2text[idx])
        return text

    
    def decode_batch(self, tokens):
        texts = []
        for token in tokens:
            text = self.decode(token)
            texts.append(text)
        return texts   


def collate_fn(batch, max_len, pad_idx):
    images = []
    labels = []
    for image, label in batch:
        images.append(image)
        if len(label) != max_len:
            label += [pad_idx]*(max_len - len(label))
        labels.append(label)
    images = torch.stack(images)
    labels = torch.tensor(labels)
    return images, labels


def get_loader(train_dataset, valid_dataset, batch_size=64, shuffle=False, pad_idx=0, max_len=64):
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=partial(collate_fn, max_len=max_len, pad_idx=pad_idx)
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=partial(collate_fn, max_len=max_len, pad_idx=pad_idx)
    )
    return train_loader, valid_loader


def process_data(path, ignore_folder = []):
    if "/" != path[-1]:
        path += "/"
    all_batch = glob.glob(path + "*")
    images_list = []
    labels_list = []
    for batch in all_batch:
        folder_name = batch.split("/")[-1]
        if folder_name in ignore_folder:
            continue
        filename = glob.glob(path + folder_name + "/JSON/*")[0]
        data = read_json(filename)
        images_list += [row["filename"] for row in data]
        labels_list += [row["latex"] for row in data]
    return images_list, labels_list