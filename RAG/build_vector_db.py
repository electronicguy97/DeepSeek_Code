# build_vector_db.py
from sentence_transformers import SentenceTransformer
import faiss
import json

embedder = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# 파일 불러오기 (PDF나 텍스트 청크 후 저장한 JSON 입력)
with open("docs.json", "r", encoding="utf-8") as f:
    docs = json.load(f)

texts = [d["text"] for d in docs]

# 임베딩
embs = embedder.encode(texts, show_progress_bar=True)
dim = embs.shape[1]

# FAISS index
index = faiss.IndexFlatL2(dim)
print(index)
index.add(embs)

# 저장
faiss.write_index(index, "vector.index")

# 문서도 별도로 저장
with open("vector_texts.json", "w", encoding="utf-8") as f:
    json.dump(texts, f, ensure_ascii=False, indent=2)

print("FAISS vector DB 생성 완료")
