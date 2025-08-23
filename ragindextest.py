from src.rag import RAGIndex
idx = RAGIndex(persist_dir="rag_index", model_name="sentence-transformers/all-MiniLM-L6-v2")
idx.build_from_csv("data/train.csv")
print("RAG index built at rag_index/")