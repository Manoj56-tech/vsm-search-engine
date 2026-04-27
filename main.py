from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

documents = [
    "I love running and marathons",
    "Running improves health and stamina",
    "Python is great for machine learning",
    "I enjoy long distance running",
    "Artificial intelligence is the future"
]

def search(query):
    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform(documents)

    query_vec = vectorizer.transform([query])
    similarity = cosine_similarity(query_vec, doc_vectors)[0]

    ranked = sorted(enumerate(similarity), key=lambda x: x[1], reverse=True)

    print("\nTop Results:\n")
    for i, score in ranked:
        print(documents[i], " --> ", round(score, 4))

while True:
    q = input("\nEnter query (type 'exit' to stop): ")
    if q.lower() == 'exit':
        break
    search(q)