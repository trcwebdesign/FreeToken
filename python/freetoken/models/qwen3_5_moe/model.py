from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.core import get_global_ctx
from freetoken.layers import (
    BaseOP,
    GemmaRMSNorm,
    OPList,
    ParallelLMHead,
    LinearReplicated,
    VocabParallelEmbedding,
)
from freetoken.models.blocks import BaseLLMModel
from freetoken.models.blocks import embed_input_ids
from freetoken.models.qwen3_vl.vision import Qwen3VLVisionModel, QwenVLVisionMixin
from freetoken.utils import nvtx_annotate

from .attention import Qwen3_5Attention
from .gdn import Qwen3_5GatedDeltaNet
from .moe import Qwen3_5DenseMLP, Qwen3_5MoE

if TYPE_CHECKING:
    from freetoken.models.config import ModelConfig


class Qwen3_5DecoderLayer(BaseOP):
    """Pre-norm hybrid block: ``x = x + mixer(input_norm(x)); x = x + moe(post_norm(x))``,
    where the mixer is a GatedDeltaNet (linear layers) or gated attention (full layers).
    All norms are Gemma-style (1+weight)."""

    def __init__(self, config: ModelConfig, layer_id: int, *, prefix: str = ""):
        self._layer_id = layer_id
        self._is_linear = config.is_linear_layer(layer_id)
        if self._is_linear:
            g = config.linear_attention_group()
            assert g is not None
            self.linear_attn = Qwen3_5GatedDeltaNet(
                hidden_size=config.hidden_size,
                num_k_heads=g.num_key_heads,
                num_v_heads=g.num_value_heads,
                head_k_dim=g.key_head_dim,
                head_v_dim=g.value_head_dim,
                conv_kernel_size=g.conv_kernel_dim,
                rms_norm_eps=config.rms_norm_eps,
                layer_id=layer_id,
                quant_config=config.quant,
                prefix=f"{prefix}.linear_attn",
            )
        else:
            self.self_attn = Qwen3_5Attention(config, layer_id, prefix=f"{prefix}.self_attn")
        # Dense variants (num_experts==0, e.g. Qwen3.6-27B) use a plain SwiGLU MLP instead of
        # the routed MoE block; both expose ``forward(hidden)->hidden`` and the same key prefix.
        self.mlp = (
            Qwen3_5MoE(config, layer_id, prefix=f"{prefix}.mlp")
            if config.moe_enabled
            else Qwen3_5DenseMLP(config, prefix=f"{prefix}.mlp")
        )
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, hidden: torch.Tensor, residual: torch.Tensor | None):
        # Residual-stream form: fuse each residual-add into the next RMSNorm
        # (GemmaRMSNorm.forward_add_residual) so add + norm are one kernel per sublayer.
        if residual is None:
            residual = hidden
            hidden = self.input_layernorm.forward(hidden)
        else:
            hidden, residual = self.input_layernorm.forward_add_residual(hidden, residual)
        hidden = self.linear_attn.forward(hidden) if self._is_linear else self.self_attn.forward(hidden)
        hidden, residual = self.post_attention_layernorm.forward_add_residual(hidden, residual)
        hidden = self.mlp.forward(hidden)
        return hidden, residual


class Qwen3_5Model(BaseOP):
    def __init__(self, config: ModelConfig, *, prefix: str = "model"):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [
                Qwen3_5DecoderLayer(config, layer_id, prefix=f"{prefix}.layers.{layer_id}")
                for layer_id in range(config.num_layers)
            ]
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = embed_input_ids(self.embed_tokens, input_ids, get_global_ctx().batch)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        x, _ = self.norm.forward_add_residual(x, residual)
        return x


class Qwen3_5ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Qwen3_5Model(config)
        self.lm_head = ParallelLMHead(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
            tie_word_embeddings=config.tie_word_embeddings,
            tied_embedding=self.model.embed_tokens if config.tie_word_embeddings else None,
            quant_config=config.quant,
            prefix="lm_head",
        )
        super().__init__()
        if getattr(config, "moe_weight_format", None) == "nvfp4":
            _convert_qwen3_5_to_nvfp4(self, config)
        elif getattr(config, "moe_weight_format", None) == "q4_k":
            _convert_qwen3_5_to_gguf(self, config)

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        return self.lm_head.forward(output)


class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLM):
    """The MoE releases share the dense code path: the decoder picks the routed or dense MLP from config.num_experts."""


class Qwen3_5ForConditionalGeneration(QwenVLVisionMixin, Qwen3_5ForCausalLM):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if config.is_multimodal:
            assert not config.vision_config.deepstack_visual_indexes, "Qwen3.5 consumes no DeepStack features"
            self.visual = Qwen3VLVisionModel(config.vision_config, quant_config=config.quant, prefix="visual")


class Qwen3_5MoeForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    """The MoE releases with the vision tower; see Qwen3_5MoeForCausalLM."""


__all__ = [
    "Qwen3_5ForCausalLM",
    "Qwen3_5ForConditionalGeneration",
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5MoeForConditionalGeneration",
]


def _convert_qwen3_5_to_nvfp4(model, config) -> None:
    from freetoken.kernel.triton.nvfp4_linear import Nvfp4DenseColMerged, Nvfp4DenseLinear, Nvfp4LMHead
    from freetoken.models.gemma4.gguf import Nvfp4Embedding

    model.model.embed_tokens = Nvfp4Embedding(config.vocab_size, config.hidden_size)
    model.lm_head = Nvfp4LMHead(config.vocab_size, config.hidden_size)
    for layer in model.model.layers.op_list:
        if layer._is_linear:
            attn = layer.linear_attn
            attn.in_proj = Nvfp4DenseColMerged(attn.in_proj.in_features, attn._in_proj_split)
            attn.out_proj = Nvfp4DenseLinear(attn.value_dim, config.hidden_size)
        else:
            attn = layer.self_attn
            attn.qkv_proj = Nvfp4DenseColMerged(
                attn.qkv_proj.in_features,
                attn._qkv_split,
                has_bias=False,
            )
            attn.o_proj = Nvfp4DenseLinear(attn.qo_attn_dim, config.hidden_size, has_bias=False)
        if config.moe_enabled:
            shared = layer.mlp.shared_expert
        else:
            shared = layer.mlp
        shared.gate_up_proj = Nvfp4DenseColMerged(config.hidden_size, [config.shared_expert_intermediate_size] * 2 if config.moe_enabled else [config.intermediate_size] * 2)
        shared.down_proj = Nvfp4DenseLinear(config.shared_expert_intermediate_size if config.moe_enabled else config.intermediate_size, config.hidden_size)

def _convert_qwen3_5_to_gguf(model, config) -> None:
    from freetoken.layers.gguf import GGUFEmbedding, GGUFMergedLinear, GGUFLinear
    from freetoken.models.gguf.dequant import GGML_Q4_K, GGML_Q6_K

    model.model.embed_tokens = GGUFEmbedding(config.vocab_size, config.hidden_size, GGML_Q4_K)
    model.lm_head = GGUFLinear(config.hidden_size, config.vocab_size, GGML_Q6_K)
    tensor_types = config.gguf_tensor_types or {}
    for layer in model.model.layers.op_list:
        if layer._is_linear:
            attn = layer.linear_attn
            prefix = f"blk.{layer._layer_id}."
            qkv_type = tensor_types.get(prefix + "attn_qkv.weight", GGML_Q6_K)
            gate_type = tensor_types.get(prefix + "attn_gate.weight", GGML_Q4_K)
            alpha_type = tensor_types.get(prefix + "ssm_alpha.weight", GGML_Q4_K)
            beta_type = tensor_types.get(prefix + "ssm_beta.weight", GGML_Q4_K)
            out_type = tensor_types.get(prefix + "ssm_out.weight", GGML_Q4_K)
            attn.in_proj = GGUFMergedLinear(config.hidden_size, [8192, 4096, 32, 32], [qkv_type, gate_type, alpha_type, beta_type])
            attn._split_in_proj = False
            # llama.cpp tiles the GDN V-head columns in ssm_out; the loader untile step
            # materializes this projection densely because the permutation crosses blocks.
            attn.out_proj = LinearReplicated(attn.value_dim, config.hidden_size, has_bias=False)
        else:
            attn = layer.self_attn
            prefix = f"blk.{layer._layer_id}."
            q_type = tensor_types.get(prefix + "attn_q.weight", GGML_Q4_K)
            k_type = tensor_types.get(prefix + "attn_k.weight", GGML_Q4_K)
            v_type = tensor_types.get(prefix + "attn_v.weight", GGML_Q6_K)
            out_type = tensor_types.get(prefix + "attn_output.weight", GGML_Q4_K)
            attn.qkv_proj = GGUFMergedLinear(
                config.hidden_size,
                [attn._qkv_split[0], attn._qkv_split[1], attn._qkv_split[2]],
                [q_type, k_type, v_type],
            )
            attn.o_proj = GGUFLinear(attn.qo_attn_dim, config.hidden_size, out_type)
        shared = layer.mlp.shared_expert
        prefix = f"blk.{layer._layer_id}."
        gate_type = tensor_types.get(prefix + "ffn_gate_shexp.weight", GGML_Q4_K)
        up_type = tensor_types.get(prefix + "ffn_up_shexp.weight", gate_type)
        down_type = tensor_types.get(prefix + "ffn_down_shexp.weight", GGML_Q6_K)
        if gate_type == up_type:
            shared.gate_up_proj = GGUFLinear(config.hidden_size, 2 * config.shared_expert_intermediate_size, gate_type)
        else:
            shared.gate_up_proj = GGUFMergedLinear(
                config.hidden_size,
                [config.shared_expert_intermediate_size] * 2,
                [gate_type, up_type],
            )
        shared.down_proj = GGUFLinear(config.shared_expert_intermediate_size, config.hidden_size, down_type)
