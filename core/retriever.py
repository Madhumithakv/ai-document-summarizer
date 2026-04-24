def retrieve_chunks(query, model, index, chunks, top_k=3):
    query_vec = model.encode([query])
    distances, indices = index.search(query_vec, top_k)

    results = [chunks[i] for i in indices[0]]
    return results