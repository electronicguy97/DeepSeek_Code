# model_builder.py
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig
from config import BASE_MODEL, LORA_PARAMS, BNB_PARAMS

def load_model():
    bnb_config = BitsAndBytesConfig(**BNB_PARAMS)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        quantization_config=bnb_config
    )

    lora_config = LoraConfig(**LORA_PARAMS)
    # model = get_peft_model(model, lora_config)  # LoRA 적용 예시 (필요시)

    return model
