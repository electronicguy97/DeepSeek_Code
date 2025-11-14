# data_loader.py
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
import config

def load_data():
    # 데이터셋 불러오기
    dataset = load_dataset(config.DATASET_NAME, config.DATASET_CONFIG)

    # 토크나이저 로드
    tokenizer = AutoTokenizer.from_pretrained(
        config.BASE_MODEL,
        padding_side="right"
    )
    tokenizer.pad_token = tokenizer.eos_token

    # 토크나이징 함수
    def tokenize_function(examples):
        # 함수 설명 + 코드 결합
        combined_texts = [
            f"{doc}\n\n{code}" 
            for doc, code in zip(
                examples["func_documentation_string"], 
                examples["func_code_string"]
            )
        ]

        # tokenized = tokenizer(...)
        tokenized = tokenizer(
            combined_texts,
            truncation=True,
            padding="max_length",
            max_length=512
        )

        # labels 추가 (input_ids 복사)
        tokenized["labels"] = tokenized["input_ids"].copy()
        # → torch.tensor() 사용 금지 (map 내부에서는 리스트만 허용)
        # tokenized["labels"] = torch.tensor(tokenized["input_ids"])  # ✅ `torch.tensor()` 사용

        return tokenized

    # 4) 데이터셋 변환
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset["train"].column_names
    )

    return tokenized_dataset, tokenizer
