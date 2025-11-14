# config.py
import os
import torch

# 환경 변수 설정
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 디바이스
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 모델/데이터 경로 및 이름
BASE_MODEL = "./DeepSeek-R1-Distill-Llama-8B"
DATASET_NAME = "code_search_net"
DATASET_CONFIG = "python"   # 예: "python", "java" 등

# 학습 하이퍼파라미터
BATCH_SIZE = 8
LR = 5e-5
NUM_EPOCHS = 3
SEED = 42

# LoRA 설정값
LORA_PARAMS = {
    "r": 4,
    "lora_alpha": 16,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "lora_dropout": 0.1,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

# 양자화 설정값 (Bits & Bytes 4-bit)
BNB_PARAMS = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_compute_dtype": torch.float16,
    "llm_int8_enable_fp32_cpu_offload": True,
}
