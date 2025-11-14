# data_loader.py
from transformers import AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
import torch
import config

class CodeSearchDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    # 코드 구조에 맞춰 텍스트 추출
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["code"]  # 실제 key는 데이터셋에 따라 조정
        enc = self.tokenizer(
            text, truncation=True, padding="max_length", max_length=self.max_length
        )
        input_ids = torch.tensor(enc["input_ids"])
        attention_mask = torch.tensor(enc["attention_mask"])
        return {"input_ids": input_ids, "attention_mask": attention_mask}

def get_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def get_dataloader(tokenizer, split: str, batch_size: int, shuffle: bool = True):
    dataset = load_dataset(config.DATASET_NAME, config.DATASET_CONFIG)[split]
    ds = CodeSearchDataset(dataset, tokenizer)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

