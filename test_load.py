print("Starting test...")
from langchain_community.document_loaders import TextLoader

print("Loading document...")
loader = TextLoader("data/books/alice_in_wonderland.md", encoding='utf-8')
documents = loader.load()
print(f"Loaded {len(documents)} documents")
print(f"First 200 chars: {documents[0].page_content[:200]}")
print("Done!")
