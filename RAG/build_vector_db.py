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
embs = embedder.encode(texts, show_progress_bar=True, normalize_embeddings=True) # 정규화 추가
dim = embs.shape[1]

# FAISS index
index = faiss.IndexFlatIP(dim) # L2 → Inner Product로 변경
index.add(embs.astype('float32'))

# 저장
faiss.write_index(index, "vector.index")

# 문서도 별도로 저장
with open("vector_texts.json", "w", encoding="utf-8") as f:
    json.dump(texts, f, ensure_ascii=False, indent=2)

print("FAISS vector DB 생성 완료")
