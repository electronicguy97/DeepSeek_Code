import config
from data_loader import load_data
from model_builder import build_model
from trainer import create_trainer
from utils import print_line, set_seed, log_config

def main():
    set_seed(config.SEED)
    log_config(config)
    print_line("데이터셋 로딩")
    tokenized, tokenizer = load_data()

    print_line("모델 셋업")
    model = build_model()

    print_line("Initializing trainer...")
    trainer = create_trainer(model, tokenizer, tokenized)

    print_line("학습 시작")
    trainer.train()

    print_line("최종 모델 저장")
    trainer.save_model(config.TRAINING_ARGS["output_dir"])

if __name__ == "__main__":
    main()
