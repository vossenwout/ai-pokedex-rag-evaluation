import os
from dataclasses import dataclass

from google import genai
from pymilvus import AnnSearchRequest, MilvusClient, WeightedRanker


@dataclass
class PokedexMetadata:
    pokemon: str
    url: str


@dataclass
class PokedexSearchResult:
    text: str
    metadata: PokedexMetadata


class KBEmbeddingError(Exception):
    pass


class KBSearchError(Exception):
    pass


class MilvusKnowledgeBase:
    def __init__(
        self,
        milvus_collection_name: str,
        embedding_model: str = "models/text-embedding-004",
        search_limit: int = 10,
    ):
        self.milvus_collection_name = milvus_collection_name
        if os.getenv("ZILLIZ_CLUSTER_PUBLIC_ENDPOINT") is None:
            raise OSError("ZILLIZ_CLUSTER_PUBLIC_ENDPOINT env var is not set")
        if os.getenv("ZILLIZ_CLUSTER_TOKEN") is None:
            raise OSError("ZILLIZ_CLUSTER_TOKEN env var is not set")
        self.kb = MilvusClient(
            uri=os.getenv("ZILLIZ_CLUSTER_PUBLIC_ENDPOINT"),
            token=os.getenv("ZILLIZ_CLUSTER_TOKEN"),
        )
        self.embedding_model = embedding_model
        if os.getenv("GEMINI_API_KEY") is None:
            raise OSError("GEMINI_API_KEY env var is not set")
        self.gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.search_limit = search_limit

    def _embed_content(self, content: str) -> list[float]:
        try:
            result = self.gemini_client.models.embed_content(
                model=self.embedding_model,
                contents=content,
            )
            return result.embeddings[0].values
        except Exception as e:
            raise KBEmbeddingError(f"Failed to embed content: {e}")

    def hybrid_search(self, query: str) -> list[PokedexSearchResult]:
        ranker = WeightedRanker(0.5, 0.5)
        embedding = self._embed_content(query)
        semantic_search = AnnSearchRequest(
            data=[embedding],
            anns_field="vector",
            param={"metric_type": "COSINE"},
            limit=self.search_limit,
        )
        text_search = AnnSearchRequest(
            data=[query],
            anns_field="sparse",
            param={"level": 10, "metric_type": "BM25"},
            limit=self.search_limit,
        )
        try:
            results = self.kb.hybrid_search(
                collection_name=self.milvus_collection_name,
                reqs=[text_search, semantic_search],
                limit=self.search_limit,
                ranker=ranker,
                output_fields=["metadata", "text"],
            )
        except Exception as e:
            raise KBSearchError(f"Failed to search knowledge base: {e}")

        try:
            return [
                PokedexSearchResult(
                    text=result["entity"]["text"],
                    metadata=PokedexMetadata(
                        pokemon=result["entity"]["metadata"]["pokemon_name"],
                        url=result["entity"]["metadata"]["url"],
                    ),
                )
                for result in results[0]
            ]
        except Exception as e:
            raise KBSearchError(f"Error while parsing search results: {e}")
