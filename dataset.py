import re
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


    def _extract_vocab(self, text):
        text = text.lower()
        extract_list = re.findall(r"(\\[a-zA-Z]+)", text)
        extract_list = [i for i in extract_list if i not in ["\\left", "\\right"]]
        extract_list += [
            "\\left(", "\\left|", "\\right)", "\\right|", "\\left\{", "\\right\}", "\\left[", "\\right]"
        ]
        return extract_list


    def build_vocab(self, dataset):
        char_list = []
        for label in dataset:
            label = self._extract_vocab(label)
            char_list += label     
        counter_vocab = dict(Counter(char_list))
        counter_vocab = list(counter_vocab.items())
        counter_vocab.sort(key=lambda x: x[1], reverse=True)
        for idx, (word, feq) in enumerate(counter_vocab, len(self.vocab)):
            if word not in self.vocab:
                self.vocab.append(word)
                self.token2text[idx] = word 
                self.text2token[word] = idx 


    def build_vocab_from_dict(self, vocab_dict):
        for idx, (val, key) in enumerate(vocab_dict.items(),4):
            val = val.lower()
            if val not in self.vocab:
                self.token2text[len(self.vocab)] = val
                self.text2token[val] = len(self.vocab)
                self.vocab.append(val)


    def __len__(self):
        return len(self.vocab)


    def process_text(self, text):
        dict_rep = {}
        for idx, word in enumerate(self.vocab):
            if ("\\" not in word) or (word not in text):
                continue
            text = text.replace(word, f";@;{idx};@;")
            dict_rep[f";@;{idx};@;"] = word
        text = text.split(";@;")
        text = [word for word in text if len(word) > 0]
        text_list = []
        for word in text:
            tmp_word = ";@;" + word + ";@;"
            if tmp_word not in dict_rep:
                word = list(word)
                text_list += word 
                continue
            word = dict_rep[tmp_word]
            text_list.append(word)
        return text_list


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
        if folder_name.split("\\")[-1] in ignore_folder:
            continue
        
        filename = glob.glob(".\\" + folder_name + "\\JSON\\*")[0]
        data = read_json(filename)
        images_list += [".\\"+folder_name+"\\background_images\\"+row["filename"] for row in data]
        labels_list += [row["latex"] for row in data]
    json_data = {
        "images": images_list,
        "latex": labels_list
    }
    with open("./all_data.json", "w") as f:
        json.dump(json_data, f)
    return images_list, labels_list


# if __name__ == "__main__":
#     data = read_json("./dataset/batch_1/JSON/kaggle_data_1.json")
#     test = [i["latex"] for i in data[:5]]
#     tokenizer = Tokenizer()
#     tokenizer.build_vocab(test)
#     tokenizer_tmp = Tokenizer()
#     vocab = json.load(open("./vocab.json", "r"))
#     tokenizer_tmp.build_vocab_from_dict(vocab)
#     print(tokenizer.vocab)
#     print(tokenizer_tmp.vocab)
#     print([i in tokenizer_tmp.vocab for i in tokenizer.vocab])
#     print(tokenizer.process_text('\\lim_{a\\to\\frac{\\pi}{4}}\\frac{\\frac{d}{da}\\left(\\sin{a}+-6\\sec{a}\\right)}{\\frac{d}{da}\\left(a+-4\\frac{\\pi}{4}\\right)}'))