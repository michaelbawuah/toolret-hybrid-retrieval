import torch
from sentence_transformers import SentenceTransformer


class DenseRetriever:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def encode_documents(self, documents):
        return self.model.encode(
            documents,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    def encode_queries(self, queries):
        return self.model.encode(
            queries,
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def search(self, query_embeddings, document_embeddings, document_ids, k=10):
        scores = torch.matmul(query_embeddings, document_embeddings.T)

        top_scores, top_indices = torch.topk(
            scores,
            k=min(k, document_embeddings.shape[0]),
            dim=1,
        )

        results = []

        for query_scores, query_indices in zip(top_scores, top_indices):
            query_results = []

            for score, index in zip(query_scores, query_indices):
                query_results.append(
                    (document_ids[index.item()], score.item())
                )

            results.append(query_results)

        return results