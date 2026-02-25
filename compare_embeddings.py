from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np


def main():
    # Using free local HuggingFace embeddings
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Get embedding for a word.
    vector = embedding_function.embed_query("apple")
    print(f"Vector for 'apple': {vector[:5]}...")  # Show first 5 dims
    print(f"Vector length: {len(vector)}")

    # Compare vectors of two words using cosine similarity
    words = ("apple", "iphone")
    vec_a = np.array(embedding_function.embed_query(words[0]))
    vec_b = np.array(embedding_function.embed_query(words[1]))
    cosine_sim = np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    print(f"Cosine similarity ({words[0]}, {words[1]}): {cosine_sim:.4f}")


if __name__ == "__main__":
    main()
