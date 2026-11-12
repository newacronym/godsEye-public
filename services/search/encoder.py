"""Query-time encoders on CPU, from the same SigLIP 2 checkpoint the indexer uses for images, so
query and index vectors share one space: the text tower for questions, the image tower for photos."""
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoTokenizer, SiglipTextModel, SiglipVisionModel

SIGLIP = "google/siglip2-so400m-patch16-384"
TEMPLATE = "a photo of {}."  # SigLIP was trained on alt-text; a light caption template helps short queries


class TextEncoder:
    def __init__(self, name: str = SIGLIP):
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = SiglipTextModel.from_pretrained(name).eval()  # fixed-res SigLIP 2 checkpoints use the SigLIP classes

    @torch.inference_mode()
    def encode(self, texts: list[str]) -> np.ndarray:
        # SigLIP text towers expect fixed-length padding (64 tokens).
        batch = self.tokenizer([TEMPLATE.format(t.strip().lower()) for t in texts], padding="max_length",
                               max_length=64, truncation=True, return_tensors="pt")
        out = self.model(**batch).pooler_output
        return torch.nn.functional.normalize(out, dim=-1).numpy()


def _square(img: Image.Image) -> Image.Image:
    """Pad to a square with grey, exactly as the indexer does before SigLIP's fixed-size resize."""
    w, h = img.size
    side = max(w, h)
    out = Image.new("RGB", (side, side), (128, 128, 128))
    out.paste(img, ((side - w) // 2, (side - h) // 2))
    return out


class ImageEncoder:
    def __init__(self, name: str = SIGLIP):
        self.processor = AutoImageProcessor.from_pretrained(name)
        self.model = SiglipVisionModel.from_pretrained(name).eval()   # pooler output = the indexer's image features

    @torch.inference_mode()
    def encode(self, images: list[Image.Image]) -> np.ndarray:
        pixels = self.processor(images=[_square(i.convert("RGB")) for i in images], return_tensors="pt")["pixel_values"]
        out = self.model(pixel_values=pixels).pooler_output
        return torch.nn.functional.normalize(out, dim=-1).numpy()
