from typing import Optional, Union, Literal

import torch
import transformers

from transformers.models.llama.modeling_llama import (
    LlamaConfig, LlamaDecoderLayer, LlamaRMSNorm, LlamaRotaryEmbedding, Cache, BaseModelOutputWithPast, TransformersKwargs, DynamicCache, create_causal_mask, LlamaPreTrainedModel, GenerationMixin, CausalLMOutputWithPast, Unpack,
)




def _prepare(config, inputs_embeds, attention_mask):
    """
    inputs_embeds is used to obtain batch_size (by .shape[0]), seq_len (by .shape[1]), dtype, and device
    """
    outs = dict()

    use_cache: Optional[bool] = None
    outs["use_cache"] = use_cache

    past_key_values: Optional[Cache] = None
    if use_cache and past_key_values is None:
        past_key_values = DynamicCache()
    outs["past_key_values"] = past_key_values


    cache_position: Optional[torch.LongTensor] = None
    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position: torch.Tensor = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
    outs["cache_position"] = cache_position

    position_ids: Optional[torch.LongTensor] = None
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)
    outs["position_ids"] = position_ids

    logits_to_keep: Union[int, torch.Tensor] = 0
    outs["logits_to_keep"] = logits_to_keep

    causal_mask = create_causal_mask(
        config=config,
        input_embeds=inputs_embeds,
        attention_mask=attention_mask,
        cache_position=cache_position,
        past_key_values=past_key_values,
        position_ids=position_ids,
    )
    outs["causal_mask"] = causal_mask

    return outs

class L1(torch.nn.Module):
    # def __init__(self, config):
    def __init__(self, config, *, embed_tokens=None, rotary_emb=None):
        super().__init__()
        self.config = config
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)

        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        if rotary_emb is not None:
            self.rotary_emb.load_state_dict(rotary_emb.state_dict())

    def forward(
        self,
        input_ids_and_attention_mask:tuple[torch.LongTensor, torch.LongTensor],
    ):
        # print("start l1", input_ids_and_attention_mask[0][:,:4], flush=True)
        input_ids, attention_mask = input_ids_and_attention_mask
        # input_ids, attention_mask = input_ids_and_attention_mask[0], None

        inputs_embeds: Optional[torch.FloatTensor] = None
        if inputs_embeds is None:
            inputs_embeds: torch.Tensor = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        # prepared = _prepare(config=self.config, inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        # use_cache = prepared["use_cache"]
        # past_key_values = prepared["past_key_values"]
        # cache_position = prepared["cache_position"]
        # position_ids = prepared["position_ids"]
        # logits_to_keep = prepared["logits_to_keep"]
        # causal_mask = prepared["causal_mask"]

        # position_embeddings = self.rotary_emb(inputs_embeds, position_ids)
        # outs["position_embeddings"] = position_embeddings

        packed = torch.cat([hidden_states, attention_mask.to(hidden_states)[:,:,None]], 2) # [batch_size, seq_len, hidden_dim + 1]
        return packed
        non_none_tuple_outs = (
            # inputs_embeds,
            # attention_mask,
            hidden_states,
            # position_ids,
            # cache_position,
            # position_embeddings[0], position_embeddings[1],
            # causal_mask,
            # past_key_values,
        )
        return non_none_tuple_outs

    def inherit(self, model:transformers.models.llama.LlamaForCausalLM):
        self.embed_tokens.load_state_dict(model.model.embed_tokens.state_dict())
        self.rotary_emb.load_state_dict(model.model.rotary_emb.state_dict())

class L2(torch.nn.Module):
    # def __init__(self, config, layer_idx):
    def __init__(self, config, layer_idx, *, decoder_layer=None, rotary_emb=None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if decoder_layer is not None:
            self.decoder_layer = decoder_layer
        else:
            self.decoder_layer = LlamaDecoderLayer(config, layer_idx)

        self.rotary_emb = LlamaRotaryEmbedding(config=config)
        if rotary_emb is not None:
            self.rotary_emb.load_state_dict(rotary_emb.state_dict())

    def forward(self, inputs):
        # print("start l2", self.layer_idx, flush=True)
        # hidden_states = inputs["hidden_states"]
        # position_ids = inputs["position_ids"]
        # cache_position = inputs["cache_position"]
        # position_embeddings = inputs["position_embeddings"]

        # hidden_states, position_ids, cache_position, position_embedding_0, position_embedding_1 = inputs

        # causal_mask = inputs["causal_mask"]
        # causal_mask = None

        # past_key_values = inputs["past_key_values"]
        # past_key_values = None

        # hidden_states, = inputs
        # attention_mask = None
        hidden_states = inputs[:,:,:-1]
        attention_mask = inputs[:,:,-1]

        prepared = _prepare(config=self.config, inputs_embeds=hidden_states, attention_mask=attention_mask)
        causal_mask = prepared["causal_mask"]
        position_ids = prepared["position_ids"]
        past_key_values = prepared["past_key_values"]
        cache_position = prepared["cache_position"]

        position_embedding_0, position_embedding_1 = self.rotary_emb(hidden_states, position_ids)

        new_hidden_states = self.decoder_layer(
            hidden_states,
            attention_mask=causal_mask,
            position_ids=position_ids,
            past_key_value=past_key_values,
            cache_position=cache_position,
            position_embeddings=(position_embedding_0, position_embedding_1),
        )

        # outs = dict(inputs)
        # outs["hidden_states"] = new_hidden_states
        # return outs
        # return (new_hidden_states, position_ids, cache_position, position_embeddings_0, position_embeddings_1)

        # return (new_hidden_states,)
        return torch.cat([new_hidden_states, attention_mask[:,:,None]], 2)

    def inherit(self, model:transformers.models.llama.LlamaForCausalLM):
        self.decoder_layer.load_state_dict(model.model.layers[self.layer_idx].state_dict())

class L3(torch.nn.Module):
    def __init__(self, config, *, norm=None):
        super().__init__()
        self.config = config
        if norm is not None:
            self.norm = norm
        else:
            self.norm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, inputs):
        # print("start l3", [i.grad_fn for i in inputs])
        # hidden_states = inputs["hidden_states"]
        # hidden_states, position_ids, cache_position, position_embeddings_0, position_embeddings_1 = inputs

        # hidden_states, = inputs
        hidden_states = inputs[:,:,:-1]

        last_hidden_state = self.norm(hidden_states)
        return last_hidden_state

    def inherit(self, model:transformers.models.llama.LlamaForCausalLM):
        self.norm.load_state_dict(model.model.norm.state_dict())

class L4(torch.nn.Module):
    def __init__(self, config, *, lm_head=None):
        super().__init__()
        self.config = config
        if lm_head is not None:
            self.lm_head = lm_head
        else:
            self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, last_hidden_state):
        # print("start l4", last_hidden_state[:,:3], last_hidden_state.grad_fn)
        logits_to_keep: Union[int, torch.Tensor] = 0
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(last_hidden_state[:, slice_indices, :])
        return logits

    def inherit(self, model:transformers.models.llama.LlamaForCausalLM):
        self.lm_head.load_state_dict(model.lm_head.state_dict())

def convert(model_config, base_model=None, show_progress=True, inherit:Literal["move","copy","init"]="move"):
    if inherit != "init": assert base_model is not None
    if inherit == "move":
        l1 = L1(model_config, embed_tokens=base_model.model.embed_tokens, rotary_emb=base_model.model.rotary_emb)
    elif inherit == "copy":
        l1 = L1(model_config)
        l1.inherit(base_model)
    elif inherit == "init":
        l1 = L1(model_config)
    if show_progress: print("l1")

    if show_progress: print("l2", end="", flush=True)
    l2 = list()
    for layer_idx in range(model_config.num_hidden_layers):
        if inherit == "move":
            l2i = L2(model_config, layer_idx, decoder_layer=base_model.model.layers[layer_idx], rotary_emb=base_model.model.rotary_emb)
        elif inherit == "copy":
            l2i = L2(model_config, layer_idx)
            l2i.inherit(base_model)
        elif inherit == "init":
            l2i = L2(model_config, layer_idx)
        l2.append(l2i)
        if show_progress: print(".", end="", flush=True)
    if show_progress: print("", flush=True)

    if inherit == "move":
        l3 = L3(model_config, norm=base_model.model.norm)
    elif inherit == "copy":
        l3 = L3(model_config)
        l3.inherit(base_model)
    elif inherit == "init":
        l3 = L3(model_config)
    if show_progress: print("l3")

    if inherit == "move":
        l4 = L4(model_config, lm_head=base_model.lm_head)
    elif inherit == "copy":
        l4 = L4(model_config)
        l4.inherit(base_model)
    elif inherit == "init":
        l4 = L4(model_config)
    if show_progress: print("l4")

    seq_model = torch.nn.Sequential(l1, *l2, l3, l4)
    return seq_model
