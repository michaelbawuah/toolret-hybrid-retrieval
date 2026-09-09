from __future__ import annotations

from pathlib import Path

from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """
    Lightweight cross-encoder reranker.

    Supports both:
    - Hugging Face model names
    - locally saved fine-tuned models
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
    ):
        self.model_name = model_name

        # -----------------------------------------------------
        # Detect whether model_name refers to a LOCAL model.
        # -----------------------------------------------------

        model_path = Path(model_name).expanduser()

        if model_path.exists():
            resolved_model = str(model_path.resolve())

            print(
                f"Loading local cross-encoder model from: "
                f"{resolved_model}"
            )
        else:
            resolved_model = model_name

            print(
                f"Loading Hugging Face cross-encoder model: "
                f"{resolved_model}"
            )

        kwargs = {}

        if device is not None:
            kwargs["device"] = device

        self.model = CrossEncoder(
            resolved_model,
            **kwargs,
        )

    def rerank(
        self,
        query: str,
        candidates: list[tuple[str, str]],
        top_k: int | None = None,
    ) -> list[tuple[str, float]]:
        """
        Rerank candidate documents for a single query.

        Parameters
        ----------
        query:
            Natural-language ToolRet query.

        candidates:
            List containing:

                (document_id, document_text)

        top_k:
            Optional number of reranked results to return.

        Returns
        -------
        list[tuple[str, float]]

            Ranked pairs:

                (document_id, cross_encoder_score)
        """

        if not candidates:
            return []

        pairs = [
            [
                query,
                document_text,
            ]
            for _document_id, document_text
            in candidates
        ]

        scores = self.model.predict(
            pairs,
            show_progress_bar=False,
        )

        ranked = [
            (
                document_id,
                float(score),
            )
            for (
                document_id,
                _document_text,
            ), score in zip(
                candidates,
                scores,
            )
        ]

        ranked.sort(
            key=lambda item: (
                -item[1],
                item[0],
            )
        )

        if top_k is not None:
            if top_k <= 0:
                return []

            ranked = ranked[:top_k]

        return ranked