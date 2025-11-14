# model_builder.py
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
import config

def build_model():
    # 양자화 설정
    bnb_config = BitsAndBytesConfig(**config.BNB_PARAMS)

    # 모델 로드
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL,
        device_map="auto",
        quantization_config=bnb_config
    )

    # LoRA 설정 및 적용
    lora_cfg = LoraConfig(**config.LORA_PARAMS) ## **config 사용 - LOAR_PARMS 전체 사용
    model = get_peft_model(model, lora_cfg)

    return model

def build_tokenizer():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.BASE_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
