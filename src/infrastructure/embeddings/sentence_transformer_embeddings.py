import logging

from src.application.ports.embedding_port import EmbeddingPort
from src.infrastructure.huggingface_auth_fallback import (
    huggingface_invalid_env_token_error,
    huggingface_public_hub_session,
)

logger = logging.getLogger(__name__)


class SentenceTransformerEmbeddings(EmbeddingPort):
    """Sentence-Transformers adapter for dense embeddings."""

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-small",
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._normalize_embeddings = normalize_embeddings
        self._model = self._load_model(model_name=model_name)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        cleaned = [text.strip() for text in texts]
        vectors = self._model.encode(
            cleaned,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        cleaned = query.strip()
        if not cleaned:
            raise ValueError("Query text cannot be empty.")
        vector = self._model.encode(
            [cleaned],
            batch_size=1,
            normalize_embeddings=self._normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
        return vector.tolist()

    @staticmethod
    def _load_model(model_name: str) -> object:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("sentence-transformers is required for embedding generation.") from exc

        logger.info("Loading embedding model: %s", model_name)
        try:
            return SentenceTransformer(model_name)
        except Exception as exc:  # noqa: BLE001
            if huggingface_invalid_env_token_error(exc):
                logger.warning(
                    "HF Hub rejected auth for %s; retrying with token=False (fix or unset HF_TOKEN).",
                    model_name,
                )
                try:
                    with huggingface_public_hub_session():
                        return SentenceTransformer(model_name, token=False)
                except Exception as exc2:  # noqa: BLE001
                    raise ValueError(f"Unable to load embedding model '{model_name}': {exc2}") from exc2
            raise ValueError(f"Unable to load embedding model '{model_name}': {exc}") from exc
