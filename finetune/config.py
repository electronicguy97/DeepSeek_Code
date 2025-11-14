# config.py
import os
import torch

# ----- 기본 환경 설정 -----
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- 모델 및 데이터 경로 -----
BASE_MODEL = "./DeepSeek-R1-Distill-Llama-8B"
DATASET_NAME = "code_search_net"
DATASET_LANG = "python"

# ----- 학습 관련 설정 -----
BATCH_SIZE = 4
LR = 2e-4
EPOCHS = 3
SEED = 42

# ----- LoRA 설정값 (딕셔너리 형태로 전달용) -----
LORA_PARAMS = {
    "r": 4,
    "lora_alpha": 16,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "lora_dropout": 0.1,
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

# ----- 4bit 양자화 설정값 -----
BNB_PARAMS = {
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_compute_dtype": torch.float16,
    "llm_int8_enable_fp32_cpu_offload": True,
}
