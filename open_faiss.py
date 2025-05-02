from langchain_community.vectorstores import FAISS

# Carrega o vectorstore localmente
vectorstore = FAISS.load_local("vectorstore", embeddings=None, allow_dangerous_deserialization=True)

# Acessa os vetores (como matriz numpy)
vetores = vectorstore.index.reconstruct_n(0, vectorstore.index.ntotal)

print(f"Total de vetores: {len(vetores)}")

docs = list(vectorstore.docstore._dict.values())
for i in range(3):
    print(f"\nTexto {i}: {docs[i].page_content[:100]}...")
    print(f"Vetor {i} (10 primeiros valores): {vetores[i]}")