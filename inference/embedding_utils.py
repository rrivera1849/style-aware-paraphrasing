
import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer
from tqdm.auto import tqdm

# st = sentence-transformers
valid_models = ["sbert", "cisr", "mud", "crud", "sd"]

def load_st_model(
    HF_id: str
) -> SentenceTransformer:
    model = SentenceTransformer(HF_id)
    model.eval()
    return model

def load_sbert_model(
    HF_id: str = "sentence-transformers/all-mpnet-base-v2"
) -> SentenceTransformer:
    model = load_st_model(HF_id)
    return model

def load_cisr_model(
    HF_id: str = "AnnaWegmann/Style-Embedding"
) -> SentenceTransformer:
    model = load_st_model(HF_id)
    return model

def load_sd_model(
    HF_id: str = "StyleDistance/styledistance"
) -> SentenceTransformer:
    model = load_st_model(HF_id)
    return model

def load_luar_model_and_tokenizer(
    HF_id: str = "rrivera1849/LUAR-MUD"
) -> tuple[AutoModel, AutoTokenizer]:
    luar = AutoModel.from_pretrained(HF_id, trust_remote_code=True)
    luar.eval()
    luar_tok = AutoTokenizer.from_pretrained(HF_id)
    return (luar, luar_tok)

def get_author_embeddings(
    text: list[str],
    function_kwargs: dict,
    model_name: str,
):
    assert model_name in valid_models
    if model_name == "mud" or model_name == "crud":
        out = get_luar_author_embeddings(text, **function_kwargs)
    else:
        out = get_st_author_embeddings(text, **function_kwargs)
        
    return out

def get_instance_embeddings(
    text: list[str],
    function_kwargs: dict,
    model_name: str,
):
    assert model_name in valid_models
    if model_name == "mud" or model_name == "crud":
        out = get_luar_instance_embeddings(text, **function_kwargs)
    else:
        out = get_st_instance_embeddings(text, **function_kwargs)
        
    return out

@torch.no_grad()
def get_st_author_embeddings(
    text: list[str],
    model: SentenceTransformer,
    normalize: bool = True,
) -> torch.Tensor:
    outputs = model.encode(
        text, 
        convert_to_tensor=True, 
        normalize_embeddings=normalize
    )
    outputs = outputs.mean(dim=0, keepdim=True)
    return outputs

@torch.no_grad()
def get_st_instance_embeddings(
    text: list[str],
    model: SentenceTransformer,
    batch_size: int = 32,
    progress_bar: bool = False,
    normalize: bool = True,
) -> torch.Tensor:
    outputs = model.encode(
        text,
        convert_to_tensor=True,
        normalize_embeddings=normalize,
        batch_size=batch_size,
        show_progress_bar=progress_bar,
    )
    return outputs

@torch.no_grad()
def get_luar_author_embeddings(
    text: list[str],
    luar: AutoModel,
    luar_tok: AutoTokenizer,
    normalize: bool = True,
) -> torch.Tensor:
    # output: tensor of shape (1, 512)
    assert isinstance(text, list)
    inputs = luar_tok(
        text,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    ).to(luar.device)
    inputs["input_ids"] = inputs["input_ids"].view(1, len(text), 512)
    inputs["attention_mask"] = inputs["attention_mask"].view(1, len(text), 512)
    outputs = luar(**inputs)
    if normalize:
        outputs = F.normalize(outputs, p=2, dim=-1)
    return outputs

@torch.no_grad()
def get_luar_instance_embeddings(
    text: list[str],
    luar: AutoModel,
    luar_tok: AutoTokenizer,
    batch_size: int = 32,
    progress_bar: bool = False,
    normalize: bool = True,
) -> torch.Tensor:
    # output: tensor of shape (len(text), 512)
    all_outputs = []
    if progress_bar:
        iter = tqdm(range(0, len(text), batch_size))
    else:
        iter = range(0, len(text), batch_size)

    for i in iter:
        batch = text[i:i+batch_size]
        inputs = luar_tok(
            batch,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        ).to(luar.device)
        inputs["input_ids"] = inputs["input_ids"].view(len(batch), 1, 512)
        inputs["attention_mask"] = inputs["attention_mask"].view(len(batch), 1, 512)
        outputs = luar(**inputs)
        all_outputs.append(outputs)
    all_outputs = torch.cat(all_outputs, dim=0)
    if normalize:
        all_outputs = F.normalize(all_outputs, p=2, dim=-1)
    return all_outputs