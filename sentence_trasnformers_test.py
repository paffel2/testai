import sentence_transformers

model = sentence_transformers.SentenceTransformer('jinaai/jina-embeddings-v3',trust_remote_code=True)


text_embeddings = model.encode(sentences=['hello world'],task="text-matching")

print(len(text_embeddings.tolist()[0]))