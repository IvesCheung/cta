import inspect

import torch
import torch.nn as nn


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.3):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x + residual


class CTAClassifierHead(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int, dropout: float = 0.3, head_type: str = "mlp"):
        super().__init__()
        bottleneck_dim = max(hidden_dim // 4, 1)

        if head_type == "linear":
            self.head = nn.Linear(hidden_dim, num_classes)
        elif head_type == "cosine":
            self.head = nn.Linear(hidden_dim, num_classes, bias=False)
        elif head_type == "mlp":
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, bottleneck_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(bottleneck_dim, num_classes),
            )
        elif head_type == "ln_mlp":
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, bottleneck_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(bottleneck_dim, num_classes),
            )
        elif head_type == "res_mlp":
            self.head = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                ResidualMLPBlock(hidden_dim, dropout=dropout),
                nn.Linear(hidden_dim, num_classes),
            )
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


def setup_model(model_path: str, num_classes: int, lora_r: int, lora_alpha: int, lora_dropout: float, num_unfrozen_layers: int, head_type: str, device: str = "auto"):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[Model] Loading {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "torch_dtype": torch.float16,
        "trust_remote_code": True,
        "device_map": device,
    }
    model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

    hidden_dim = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    print(f"  hidden_size={hidden_dim}, num_layers={num_layers}")

    for param in model.parameters():
        param.requires_grad = False

    if num_unfrozen_layers > 0:
        layers_to_unfreeze = model.model.layers[-min(num_unfrozen_layers, num_layers):]
        for layer in layers_to_unfreeze:
            for param in layer.parameters():
                param.requires_grad = True
        print(f"  Unfrozen last {len(layers_to_unfreeze)} layers")

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=lora_dropout,
        task_type="FEATURE_EXTRACTION",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    classifier = CTAClassifierHead(hidden_dim, num_classes, dropout=lora_dropout, head_type=head_type).to(model.device)
    trainable = sum(param.numel() for param in classifier.parameters())
    print(f"  Classifier head: {head_type}")
    print(f"  Classifier params: {trainable:,}")
    return model, tokenizer, classifier, hidden_dim


def get_decoder_backbone(model):
    base_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    return base_model.model if hasattr(base_model, "model") else base_model


def create_adamw(params, lr: float, weight_decay: float):
    kwargs = {"lr": lr, "weight_decay": weight_decay}
    if torch.cuda.is_available() and "fused" in inspect.signature(torch.optim.AdamW).parameters:
        kwargs["fused"] = True
    return torch.optim.AdamW(params, **kwargs)
