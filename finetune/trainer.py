from trl import SFTTrainer
from transformers import TrainingArguments
import config


def formatting_func(example):
    text = f"{example['func_documentation_string']}\n\n{example['func_code_string']}"
    return text

def create_trainer(model, tokenizer, dataset):

    tokenized_dataset = dataset["train"].train_test_split(test_size=0.1)
    train_ds = tokenized_dataset["train"]
    eval_ds = tokenized_dataset["test"]

    training_args = TrainingArguments(**config.TRAINING_ARGS)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        #dataset_text_field="combined_text", 
        formatting_func=formatting_func,         
        max_seq_length=512,
        packing=False,
    )

    return trainer
