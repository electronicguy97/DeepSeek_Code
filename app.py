import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

st.title("AI 대화 모델")
# 모델 설정
MODEL_PATH = "./deepseek-coder-7b-instruct-v1.5"
device = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_model():
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        #device_map="auto",  
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            low_cpu_mem_usage=False
        ),
        torch_dtype=torch.float16 ).to('cuda')
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    return model, tokenizer

model, tokenizer = load_model()

# 입력 받기
prompt = st.text_area("프롬프트를 입력하세요:")

messages = [
    {
        'role': 'user',
        'content': f"""
        {prompt}
        1. Explain the algorithm in detail.
        2. Write the code and provide comments for each part of the code.
        3. Answer in Korean.
        """
    }
]

if st.button("코드 생성"):
    if prompt:
        with st.spinner("AI가 코드를 생성 중..."):
            # 프롬프트 템플릿 적용
            inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to('cuda')

            # attention_mask가 없는 경우 자동 생성 (기본적으로 1로 채워진 값으로 모든 토큰을 유효한 값으로 처리)
            attention_mask = torch.ones(inputs.size(), device=inputs.device)

            with torch.no_grad():
                outputs = model.generate(inputs, 
                         attention_mask=attention_mask, 
                         max_new_tokens=1024, 
                         do_sample=False, 
                         top_k=50, 
                         top_p=0.95, 
                         num_return_sequences=1, 
                         eos_token_id=tokenizer.eos_token_id,
                         pad_token_id=tokenizer.pad_token_id
                         ).to(device)

            generated_code = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

            st.text_area(generated_code)
    else:
        st.warning("프롬프트를 입력하세요.")