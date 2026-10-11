import sys
import os
import torch
from unittest.mock import MagicMock
from huggingface_hub import hf_hub_download
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from mamba_ssm.models.config_mamba import MambaConfig
from transformers import LlamaTokenizer


WEIGHTS_DIR = os.environ.get("NVIDIA_MAMBA_WEIGHTS_DIR") or None
REPO_ID = "nvidia/mamba2-8b-3t-4k"


def _hub_download(filename: str) -> str:
    kwargs = {"repo_id": REPO_ID, "filename": filename}
    if WEIGHTS_DIR:
        os.makedirs(WEIGHTS_DIR, exist_ok=True)
        kwargs["local_dir"] = WEIGHTS_DIR
    return hf_hub_download(**kwargs)


def _patch_megatron_imports():
    # The NVIDIA checkpoint was saved with Megatron-LM, so unpickling it tries to
    # import megatron. We don't have Megatron installed, so we inject mocks before
    # torch.load is called. __path__ and __file__ are needed to make Python treat
    # the mock as a proper package rather than a plain module.
    def _make_mock(name):
        mock = MagicMock()
        mock.__name__ = name
        mock.__path__ = []
        mock.__file__ = f"/tmp/{name}.py"
        return mock

    for mod in ["megatron", "megatron.core", "megatron.core.enums", "megatron.core.parallel_state"]:
        sys.modules[mod] = _make_mock(mod)



def _map_nvidia_to_mamba_ssm(checkpoint: dict) -> dict:
    # The checkpoint is nested: model -> language_model -> actual weights
    state_dict = checkpoint.get('model', checkpoint)
    if 'language_model' in state_dict:
        state_dict = state_dict['language_model']

    # NVIDIA's Megatron naming differs from mamba_ssm's expected naming.
    # These replacements translate between the two conventions.
    renames = {
        'decoder.layers.':                  'backbone.layers.',
        'decoder.final_norm.':              'backbone.norm_f.',
        'embedding.word_embeddings.weight': 'backbone.embedding.weight',
        'output_layer.weight':              'lm_head.weight',
    }

    new_dict = {}
    for k, v in state_dict.items():
        for old, new in renames.items():
            k = k.replace(old, new)
        new_dict[k] = v.to(torch.bfloat16)

    return new_dict


def _load_model(mapped_state_dict: dict) -> MambaLMHeadModel:
    # These hyperparameters match the nvidia/mamba2-8b-3t-4k architecture exactly.
    # tie_embeddings=False because NVIDIA trained with separate input/output embeddings.
    config = MambaConfig(
        d_model=4096,
        n_layer=56,
        vocab_size=256000,
        ssm_cfg={
            'layer': 'Mamba2',
            'expand': 2,
            'headdim': 64,
            'd_state': 128,
            'ngroups': 8,
        },
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True,
        pad_vocab_size_multiple=8,
        tie_embeddings=False,
    )
    print("Initializing model on GPU...")
    model = MambaLMHeadModel(config, device="cuda", dtype=torch.bfloat16)

    # strict=False because some optimizer/scheduler state keys in the checkpoint
    # have no counterpart in the inference-only model.
    msg = model.load_state_dict(mapped_state_dict, strict=False)
    print(f"Missing keys: {len(msg.missing_keys)}, Unexpected keys: {len(msg.unexpected_keys)}")

    return model


def get_model():
    _patch_megatron_imports()
    print("Downloading weights...")
    checkpoint_path = _hub_download("release/mp_rank_00/model_optim_rng.pt")
    raw_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    mapped_state_dict = _map_nvidia_to_mamba_ssm(raw_checkpoint)
    model = _load_model(mapped_state_dict)
    return model


def get_tokenizer():
    print("Downloading tokenizer...")
    tokenizer_path = _hub_download(
        "mt_nlg_plus_multilingual_ja_zh_the_stack_frac_015_256k.model"
    )
    tokenizer = LlamaTokenizer(vocab_file=tokenizer_path)
    return tokenizer

