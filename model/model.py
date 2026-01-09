import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from base import BaseModel

class FrozenInTime(BaseModel):
    """
    Text-only version of FrozenInTime for:
    - Extension Substep 3
    - EgoVLP / HF text encoder
    - 1792-dim aligned embeddings
    """

    def __init__(
        self,
        text_params,
        target_dim=1792,
        load_checkpoint=None,
    ):
        super().__init__()

        self.text_params = text_params
        self.target_dim = target_dim

        if not text_params.get("pretrained", False):
            raise ValueError("Text model must be pretrained.")

        # -----------------------------
        # Load text encoder
        # -----------------------------
        if text_params["model"].startswith("distilbert"):
            self.text_model = AutoModel.from_pretrained(
                "distilbert-base-uncased",
                cache_dir="pretrained/distilbert-base-uncased",
            )
        else:
            self.text_model = AutoModel.from_pretrained(text_params["model"])

        # Freeze text encoder (IMPORTANT)
        self.text_model.eval()
        for p in self.text_model.parameters():
            p.requires_grad = False

        # -----------------------------
        # Text projection to 1792
        # -----------------------------
        hidden_size = self.text_model.config.hidden_size

        if hidden_size == target_dim:
            # Already aligned (rare but possible)
            self.txt_proj = nn.Identity()
        else:
            # Standard case: 768 → 1792
            self.txt_proj = nn.Linear(hidden_size, target_dim, bias=False)

        # Optional checkpoint loading (rarely needed here)
        if load_checkpoint not in ["", None]:
            checkpoint = torch.load(load_checkpoint, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)
            self.load_state_dict(state_dict, strict=False)

    # -------------------------------------------------
    # Forward (TEXT ONLY)
    # -------------------------------------------------
    def forward(self, data):
        """
        data = {
            "text": {
                "input_ids": Tensor [B, L],
                "attention_mask": Tensor [B, L]
            }
        }
        """
        text_data = data["text"]
        return self.compute_text(text_data)

    # -------------------------------------------------
    # Text encoding (CLS / pooled)
    # -------------------------------------------------
    def compute_text(self, text_data):
        """
        Returns:
            Tensor [B, 1792]
        """
        if self.text_params["model"].startswith("bert"):
            outputs = self.text_model(
                input_ids=text_data["input_ids"],
                attention_mask=text_data["attention_mask"],
            )
            text_embeds = outputs.pooler_output

        elif self.text_params["model"].startswith("distilbert"):
            outputs = self.text_model(**text_data)
            text_embeds = outputs.last_hidden_state[:, 0, :]  # CLS token

        else:
            raise NotImplementedError("Unsupported text model")

        text_embeds = self.txt_proj(text_embeds)
        text_embeds = F.normalize(text_embeds, dim=1)

        return text_embeds


# -------------------------------------------------
# Utility: cosine similarity matrix
# -------------------------------------------------
def sim_matrix(a, b, eps=1e-8):
    """
    a: [N, D]
    b: [M, D]
    returns: [N, M]
    """
    a_norm = a / torch.clamp(a.norm(dim=1, keepdim=True), min=eps)
    b_norm = b / torch.clamp(b.norm(dim=1, keepdim=True), min=eps)
    return torch.mm(a_norm, b_norm.t())


if __name__ == "__main__":
    print("FrozenInTime text-only model (1792-dim) ready.")
