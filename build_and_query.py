from rag_utils import RAG


if __name__ == "__main__":
#create RAG and put document to Vector Database
   rag = RAG()
   n = rag.ingest("data/*")
   print(f"Ingested {n} chunks.\n")


# Question and Answer phase
   for q in [
       "What is RAG in one sentence?",
       "What is the late submission policy?",
       "Name some common vector DBs.",
   ]:
       result = rag.answer(q)
       print("Q:", q)
       print("A:", result["answer"], "\n")
