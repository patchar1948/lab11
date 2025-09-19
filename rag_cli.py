from rag_utils import RAG


def main():
   rag = RAG()
   n = rag.ingest("data/*")
   print(f"RAG ready with {n} chunks. Type 'exit' to quit.\n")
   while True:
       q = input("You> ").strip()
       if q.lower() in {"exit", "quit"}: break
       res = rag.answer(q)
       print("\nAssistant>")
       print(res["answer"], "\n")


if __name__ == "__main__":
   main()