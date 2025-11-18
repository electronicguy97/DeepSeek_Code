from trl import SFTTrainer
from transformers import TrainingArguments

import config

def create_trainer(model, tokenizer, dataset):

    training_args = TrainingArguments(**config.TRAINING_ARGS)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"]
    )

    return trainer
