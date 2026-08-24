"""
ONNX-exported bge-large-en-v1.5
"""
import argparse
from pathlib import Path

import numpy as np
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

MODEL_NAME = "BAAI/bge-large-en-v1.5"
ONNX_DIR = Path("models/bge-large-onnx")


def export_onnx_model(out_dir: Path = ONNX_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Exporting {MODEL_NAME} to ONNX at {out_dir} ...")
    model = ORTModelForFeatureExtraction.from_pretrained(MODEL_NAME, export=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    print("Done. Quantize separately with `optimum-cli onnxruntime quantize` for int8 if needed.")


class OnnxEmbedder:

    def __init__(self, onnx_dir: Path = ONNX_DIR, num_threads: int = 8,
                 batch_size: int = 32, normalize: bool = True):
        if not Path(onnx_dir).exists():
            raise FileNotFoundError(
                f"{onnx_dir} not found -- run `python -m "
                f"src.auditflow.ingest.index.embedder --export` first."
            )
        from onnxruntime import SessionOptions
        sess_options = SessionOptions()
        sess_options.intra_op_num_threads = num_threads

        self.model = ORTModelForFeatureExtraction.from_pretrained(
            onnx_dir, session_options=sess_options
        )
        self.tokenizer = AutoTokenizer.from_pretrained(onnx_dir)
        self.batch_size = batch_size
        self.normalize = normalize

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        inputs = self.tokenizer(texts, padding=True, truncation=True,
                                 max_length=512, return_tensors="pt")
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0]  # CLS pooling -- BGE's recommended approach
        embeddings = embeddings.detach().numpy()
        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.clip(norms, 1e-9, None)
        return embeddings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            all_embeddings.extend(self._embed_batch(batch).tolist())
        return all_embeddings

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text])[0].tolist()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()
    if args.export:
        export_onnx_model()