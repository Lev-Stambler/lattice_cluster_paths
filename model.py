import torch
from typing import Callable, List, Tuple
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from torch.cuda.amp import autocast


TransformerModel = Tuple[AutoModelForCausalLM, AutoTokenizer]

def load_model(model_name: str,
               device,
               quantization=None) -> TransformerModel:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = None
    if quantization is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_name)
    elif quantization == '4bit':
        qc = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=qc, device_map=device)#, torch_dtype=torch.float32)

    elif quantization == '8bit':
        qc = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=qc, device_map=device)#, torch_dtype=torch.float32)
    else:
        raise ValueError("Expected None, 4bit, or 8bit")
    return (model, tokenizer)


def forward_hooked_model(model: TransformerModel, inp: List[str]) -> Tuple[torch.tensor, List[torch.tensor]]:
    model, tokenizer = model
    with torch.no_grad():
        inps = tokenizer(inp, return_tensors='pt').to(model.device)
        r = model.forward(**inps, output_hidden_states=True)
        return r['logits'], r['hidden_states']
