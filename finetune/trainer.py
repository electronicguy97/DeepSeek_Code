# trainer.py
import torch
from tqdm import tqdm
import config
from utils import save_checkpoint

def train(model, tokenizer, train_loader, valid_loader=None):
    model.to(config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LR)

    for epoch in range(1, config.NUM_EPOCHS + 1):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{config.NUM_EPOCHS}")
        for batch in loop:
            batch = {k: v.to(config.DEVICE) for k, v in batch.items()}
            outputs = model(**batch, labels=batch["input_ids"])
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            loop.set_postfix(loss=loss.item())

        # 체크포인트 저장
        save_checkpoint(model, tokenizer, config.BASE_MODEL + f"_checkpoint", epoch)

        # 검증이 있다면 여기에 추가 가능
        if valid_loader is not None:
            model.eval()
            # 평가 로직 구현
