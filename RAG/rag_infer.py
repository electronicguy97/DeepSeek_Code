# rag_infer.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import json
import torch

# 모델 선택
model_name = "../models/finetuned_model"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)
model.eval()

# 벡터 DB 로드
embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
index = faiss.read_index("vector.index")

with open("vector_texts.json", "r", encoding="utf-8") as f:
    texts = json.load(f)

def retrieve(query, top_k=3):
    q_emb = embedder.encode([query])
    D, I = index.search(q_emb, top_k)
    return [texts[idx] for idx in I[0]]

# 프롬프트 생성
def build_prompt(query):
    retrieved = retrieve(query, top_k=5)
    context = "\n\n".join([f"문서 {i+1}: {doc}" for i, doc in enumerate(retrieved)])

    # 파인튜닝하면서 프롬프트는 바꾸지 않았으므로 기존 프롬프트 사용
    messages = [
        {"role": "system", "content": "너는 친절한 AI 어시스턴트입니다. 주어진 문서를 바탕으로 정확하게 답변해 주세요."},
        {"role": "user", "content": f"다음 문서들을 참고해서 질문에 답해 주세요:\n\n{context}\n\n질문: {query}"}
    ]

    # apply_chat_template이 있으면 무조건 이걸 써야 함
    if hasattr(tokenizer, "apply_chat_template"):
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True   # Model Answer 시작 신호
        )
    else:
        # fallback
        prompt = f"<s>{"".join([f"### {m['role']}\n{m['content']}\n" for m in messages])}### Assistant\n"

    return prompt

# 4. 생성 함수
def chat(query):
    prompt = build_prompt(query)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=32000).to(model.device)

    # input_ids 중 프롬프트 부분만 사용해서 생성 시작
    output = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.3,
        top_p=0.9,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        repetition_penalty=1.1
    )

    # 입력 프롬프트 길이 이후만 디코딩 
    response = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    return response.strip()

# 테스트
if __name__ == "__main__":
    q = "create_trainer에서 dataset_text_field는 왜 필요한가?"
    print("질문:", q)
    print("\n답변:")
    print(chat(q))