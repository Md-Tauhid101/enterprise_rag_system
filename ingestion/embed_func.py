# embed_func.py
import torch
from PIL import Image
from langchain_community.vectorstores import FAISS

from transformers import CLIPProcessor, CLIPModel
from transformers import AutoTokenizer, AutoModel

# clip embedding model using for only images only
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()

# huggingface embedding model using for text only
MODEL_NAME = "BAAI/bge-base-en-v1.5"
text_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
text_model = AutoModel.from_pretrained(MODEL_NAME)
text_model.eval()


# function for creating text embedding
def embed_text(text: str):
    """
    Create semantic text embeddings using BGE.
    Suitable for RAG, OCR text, tables, long documents.
    """
    inputs = text_tokenizer(
        text,
        return_tensors="pt",
        padding = True,
        truncation = True,
        max_length = 512
    )

    with torch.inference_mode():
        outputs = text_model(**inputs)
        # Mean pooling (standard for BGE)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings[0].cpu().numpy()

# function for creating image embedidng
def embed_image(pil_image: Image.Image):
    """create image embeddings using CLIP"""
    inputs = clip_processor(images=pil_image, return_tensors="pt")
    with torch.inference_mode():
        features = clip_model.get_image_features(**inputs)
        features = features / features.norm(dim=-1, keepdim=True)
    return features[0].cpu().numpy()

# function for creating table embeddings
def embed_table(table_text: str):
    """
    Embed a table by converting it into a structured textual representation
    before embedding. This preserves schema and avoids raw text chaos.
    """

    if not table_text:
        return None
    
    # normalize whitespaces
    cleaned_text = "\n".join(
        line.strip() for line in table_text.splitlines() if line.strip()
    )
    lines = cleaned_text.splitlines()
    #### VALIDATION RULES
    # Must have atleast 2 rows

    if len(lines)<2:
        return None
    
    # Must show column structure
    has_delimiter = any(
        ("|" in line or "\t" in line or " " in line) for line in lines
    )
    if not has_delimiter:
        return None
    
    # Avoid numeric-only garbage
    alpha_ratio = sum(c.isalpha() for c in cleaned_text) / max(len(cleaned_text), 1)
    if alpha_ratio < 0.15:
        return None
    
    #### STRUCTURED REPRESENTATION
    table_representation = (
        "TABLE DATA\n"
        "The following is structured tabular information:\n"
        f"{cleaned_text}"
    )
    return embed_text(table_representation)