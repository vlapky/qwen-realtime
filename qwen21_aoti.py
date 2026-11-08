import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

from contextvars import ContextVar

from torch._functorch._aot_autograd.subclass_parametrization import unwrap_tensor_subclass_parameters


# --- Minimal copy of spaces.zero.torch.aoti (LazyAOTIModel / aoti_patch) -------------------
# The `spaces` package imports gradio and ZeroGPU machinery, so it is not used on Modal.
# One compiled package is shared by all 32 blocks; each block swaps in its own weights
# (user-managed constants, so the swap is a pointer update, not a copy).


class LazyAOTIModel:
    def __init__(self, archive_file):
        self.archive_file = archive_file
        self.compiled_model = ContextVar("compiled_model", default=None)
        self.loaded_weights = ContextVar("loaded_weights", default=None)

    def __call__(self, weights, check_full_update, *args, **kwargs):
        if (compiled_model := self.compiled_model.get()) is None:
            compiled_model = torch._inductor.aoti_load_package(self.archive_file)
            self.compiled_model.set(compiled_model)
        if (loaded := self.loaded_weights.get()) is None or loaded is not weights:
            fqns = compiled_model.get_constant_fqns()
            constant_map = {name: t for name, t in weights.items() if name in fqns}
            compiled_model.load_constants(constant_map, check_full_update=check_full_update, user_managed=True)
            self.loaded_weights.set(weights)
        return compiled_model(*args, **kwargs)


class _LazyAOTIModelWithWeights:
    def __init__(self, model, weights):
        self.model, self.weights, self.first_call = model, weights, True

    def __call__(self, *args, **kwargs):
        check_full_update, self.first_call = self.first_call, False
        return self.model(self.weights, check_full_update, *args, **kwargs)


def _shallow_clone_module(module):
    clone = object.__new__(module.__class__)
    clone.__dict__ = module.__dict__.copy()
    clone._parameters = module._parameters.copy()
    clone._buffers = module._buffers.copy()
    clone._modules = {k: _shallow_clone_module(v) for k, v in module._modules.items() if v is not None}
    return clone


def aoti_patch(module, aoti_model, call_method="forward"):
    """Route module.<call_method> through the compiled package with this module's weights.
    Call it only after the weights are on their final device: state_dict() is captured here."""
    module_ = _shallow_clone_module(module)
    unwrap_tensor_subclass_parameters(module_)
    setattr(module, call_method, _LazyAOTIModelWithWeights(aoti_model, module_.state_dict()))


BLOCK_NAME = "QwenImage21DecodeBlock"
DECODER_NAME = "QwenImage21Decoder"
PACKAGE_FILENAME = "package.pt2"
INDUCTOR_CONFIGS = {"max_autotune": True, "coordinate_descent_tuning": True, "triton.cudagraphs": False}


def rotate(x, rotary_emb):
    xf = x.float().unflatten(-1, (-1, 2))
    xr, xi = xf[..., 0], xf[..., 1]
    cos = rotary_emb[None, :, None, :, 0]
    sin = rotary_emb[None, :, None, :, 1]
    return torch.stack([xr * cos - xi * sin, xr * sin + xi * cos], dim=-1).flatten(3).type_as(x)


class QwenImage21DecodeBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, hidden_states, modulation, rotary_emb, cached_key, cached_value):
        block = self.block
        attn = block.attn
        mod1, mod2 = modulation[:-1].unsqueeze(1).chunk(2, dim=-1)
        scale1, gate1 = mod1.chunk(2, dim=-1)
        scale2, gate2 = mod2.chunk(2, dim=-1)

        normed = block.img_norm1(hidden_states) * (1 + scale1)
        query = attn.to_q(normed).unflatten(-1, (attn.heads, -1))
        key = attn.to_k(normed).unflatten(-1, (attn.heads, -1))
        value = attn.to_v(normed).unflatten(-1, (attn.heads, -1))
        query = rotate(attn.norm_q(query).to(value.dtype), rotary_emb)
        key = rotate(attn.norm_k(key).to(value.dtype), rotary_emb)
        key = torch.cat([cached_key, key], dim=1)
        value = torch.cat([cached_value, value], dim=1)
        out = F.scaled_dot_product_attention(
            query.permute(0, 2, 1, 3), key.permute(0, 2, 1, 3), value.permute(0, 2, 1, 3)
        )
        out = attn.to_out[0](out.permute(0, 2, 1, 3).flatten(2, 3).type_as(query))
        hidden_states = hidden_states + gate1.tanh() * out

        normed = block.img_norm2(hidden_states) * (1 + scale2)
        return hidden_states + gate2.tanh() * block.img_mlp(normed)


class QwenImage21Decoder(nn.Module):
    def __init__(self, decoder, num_convs):
        super().__init__()
        self.decoder = decoder
        self.num_convs = num_convs

    def forward(self, x):
        return self.decoder(x, feat_cache=[None] * self.num_convs, feat_idx=[0], first_chunk=True)


def block_dynamic_shapes():
    auto = torch.export.Dim.AUTO
    return ({1: auto}, None, {0: auto}, {1: auto}, {1: auto})


def decoder_dynamic_shapes():
    auto = torch.export.Dim.AUTO
    return ({3: auto, 4: auto},)


def _decode_block_forward(compiled, eager):
    def forward(
        hidden_states,
        modulation,
        rotary_emb=None,
        attention_mask=None,
        target_token_mask=None,
        layer_cache=None,
        kv_cache_mode=None,
        cache_write_slice=None,
        segments=None,
        key_valid=None,
    ):
        if (
            kv_cache_mode == "cached"
            and attention_mask is None
            and layer_cache is not None
            and hidden_states.shape[0] == 1
        ):
            cached_key, cached_value = layer_cache.get()
            return compiled(
                hidden_states.contiguous(),
                modulation.contiguous(),
                torch.view_as_real(rotary_emb).contiguous(),
                cached_key,
                cached_value,
            )
        return eager(
            hidden_states,
            modulation,
            rotary_emb=rotary_emb,
            attention_mask=attention_mask,
            target_token_mask=target_token_mask,
            layer_cache=layer_cache,
            kv_cache_mode=kv_cache_mode,
            cache_write_slice=cache_write_slice,
            segments=segments,
            key_valid=key_valid,
        )

    return forward


def _decoder_forward(compiled, eager):
    def forward(x, feat_cache=None, feat_idx=None, first_chunk=False):
        if first_chunk and x.shape[0] == 1 and x.shape[2] == 1:
            return compiled(x.contiguous())
        return eager(x, feat_cache=feat_cache, feat_idx=feat_idx, first_chunk=first_chunk)

    return forward


def aoti_load_pipeline(pipe, repo_id, token=None, blocks=True, decoder=True):
    def fetch(filename, subfolder=None):
        # repo_id may be a local directory with the same layout as the Hub repo.
        if Path(repo_id).is_dir():
            return str(Path(repo_id, subfolder or "", filename))
        return hf_hub_download(repo_id, filename, subfolder=subfolder, token=token)

    config = json.loads(Path(fetch("config.json")).read_text())
    if config.get("torch") != torch.__version__:
        raise RuntimeError(f"artifacts built for torch {config.get('torch')}, runtime has {torch.__version__}")
    loaded = []
    if blocks and BLOCK_NAME in config.get("artifacts", {}):
        block_model = LazyAOTIModel(fetch(PACKAGE_FILENAME, BLOCK_NAME))
        for block in pipe.transformer.transformer_blocks:
            wrapper = QwenImage21DecodeBlock(block)
            aoti_patch(wrapper, block_model)
            block.forward = _decode_block_forward(wrapper, block.forward)
        loaded.append(BLOCK_NAME)
    if decoder and DECODER_NAME in config.get("artifacts", {}):
        decoder_model = LazyAOTIModel(fetch(PACKAGE_FILENAME, DECODER_NAME))
        vae = pipe.vae
        wrapper = QwenImage21Decoder(vae.decoder, vae._cached_conv_counts["decoder"])
        aoti_patch(wrapper, decoder_model)
        vae.decoder.forward = _decoder_forward(wrapper, vae.decoder.forward)
        loaded.append(DECODER_NAME)
    return loaded, config


def aoti_unload_pipeline(pipe):
    """Undo aoti_load_pipeline: drop the instance-level forward overrides so the class
    (eager) forward is used again. Used when the compiled kernels can't run on this GPU."""
    for module in [*pipe.transformer.transformer_blocks, pipe.vae.decoder]:
        if "forward" in module.__dict__:
            del module.forward
