import os

# ============================================================
# CPU / Windows Torch compatibility
# ============================================================

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_FORCE_DISABLE_CPP"] = "1"


from pathlib import Path

import torch

from visual_bge.modeling import Visualized_BGE


class VisualBGEEmbedder:
    """
    CPU-safe wrapper around Visualized-BGE-M3.

    Supported inputs:

        text
        image
        multimodal
    """

    def __init__(
        self,
        model_name: str,
        model_weight: str,
        device: str = "cpu",
    ):

        self.device = torch.device(
            device
        )

        self.model_name = model_name
        self.model_weight = model_weight

        if not Path(model_weight).exists():
            raise FileNotFoundError(
                f"Visualized-BGE weights not found: "
                f"{model_weight}"
            )

        print(
            "Loading Visualized-BGE..."
        )

        self.model = Visualized_BGE(
            model_name_bge=model_name,
            model_weight=model_weight,
        )

        self.model.to(
            self.device
        )

        self.model.eval()

        print(
            "Visualized-BGE loaded."
        )

    # ========================================================
    # TEXT
    # ========================================================

    @torch.no_grad()
    def encode_text(
        self,
        text: str,
    ) -> torch.Tensor:

        if not text or not text.strip():
            raise ValueError(
                "Text cannot be empty."
            )

        embedding = self.model.encode(
            text=text
        )

        return self._normalize(
            embedding
        )

    # ========================================================
    # IMAGE
    # ========================================================

    @torch.no_grad()
    def encode_image(
        self,
        image_path: str,
    ) -> torch.Tensor:

        path = Path(
            image_path
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        embedding = self.model.encode(
            image=str(path)
        )

        return self._normalize(
            embedding
        )

    # ========================================================
    # MULTIMODAL
    # ========================================================

    @torch.no_grad()
    def encode_multimodal(
        self,
        text: str,
        image_path: str,
    ) -> torch.Tensor:

        path = Path(
            image_path
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        if not text or not text.strip():
            raise ValueError(
                "Text cannot be empty."
            )

        embedding = self.model.encode(
            text=text,
            image=str(path),
        )

        return self._normalize(
            embedding
        )

    # ========================================================
    # NORMALIZATION
    # ========================================================

    @staticmethod
    def _normalize(
        embedding: torch.Tensor,
    ) -> torch.Tensor:

        return torch.nn.functional.normalize(
            embedding,
            p=2,
            dim=-1,
        )

    # ========================================================
    # NUMPY
    # ========================================================

    @staticmethod
    def to_numpy(
        embedding: torch.Tensor,
    ):

        return (
            embedding
            .detach()
            .cpu()
            .numpy()
        )