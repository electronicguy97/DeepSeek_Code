# rag_infer.py
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import json
import torch

# 모델 선택
model_name = "../models/finetuned_model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).cuda()
model.eval()

# 벡터 DB
embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
index = faiss.read_index("vector.index")

with open("vector_texts.json", "r", encoding="utf-8") as f:
    texts = json.load(f)

# Retrieval

def retrieve(query, top_k=3):
    q_emb = embedder.encode([query])
    D, I = index.search(q_emb, top_k)
    return [texts[idx] for idx in I[0]]

# Build Prompt 

def build_prompt(query):
    retrieved_docs = retrieve(query)
    context = "\n".join([f"- {d}" for d in retrieved_docs])

    prompt = f"""
[Context]
{context}

[User Question]
{query}

[Model Answer]
"""
    return prompt

# Model Inference

def chat(query):
    prompt = build_prompt(query)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=300,
        temperature=0.2
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)

# Test

if __name__ == "__main__":
    q = "create_trainer에서 dataset_text_field는 왜 필요한가?"
    print(chat(q))
