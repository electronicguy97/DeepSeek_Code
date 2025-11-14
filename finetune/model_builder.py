from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
import config

def build_model():
    model = AutoModelForCausalLM.from_pretrained(
        config.BASE_MODEL,
        device_map="auto",
        quantization_config=config.BNB_CONFIG
    )

    lora = LoraConfig(**config.LORA_PARAMS)
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    return model
