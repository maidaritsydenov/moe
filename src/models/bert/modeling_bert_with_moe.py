from typing import Optional, Union, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import apply_chunking_to_forward
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_attention_mask_for_sdpa,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.bert.modeling_bert import (
    BertAttention,
    SequenceClassifierOutput,
    BertForSequenceClassification,
    BertEncoder,
    BertEmbeddings,
    BertPooler,
    BertIntermediate,
    BertOutput,
    BertPreTrainedModel,
)


class MoE(nn.Module):
    """
    Represents a Mixture of Experts (MoE) layer.

    This layer combines multiple "expert" networks to handle different aspects of the input data.
    It uses a gating network to determine which experts to use for each input.

    Attributes:
        config: Configuration object containing MoE-related parameters.  Defines the MoE layer's settings.
    """

    def __init__(self, config):
        """
        Initializes the Mixture of Experts (MoE) layer.

            Args:
                config: Configuration object containing MoE-related parameters.

            Returns:
                None
        """
        super().__init__()
        self.config = config
        self.num_experts = getattr(config, "moe_num_experts", 4)
        self.top_k = getattr(config, "moe_top_k", 1)
        self.aux_loss_coef = getattr(config, "moe_aux_loss_coef", 0.1)

        # Создаем экспертов (каждый эксперт - это FFN)
        self.experts = nn.ModuleList(
            [BertIntermediate(config) for _ in range(self.num_experts)]
        )
        self.output = BertOutput(config)  # Общий выходной слой

        # Маршрутизатор (роутер)
        self.router = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.aux_loss = 0.0  # Для накопления вспомогательного лосса

    def forward(self, hidden_states):
        """
        Performs the forward pass of the Mixture of Experts (MoE) layer.

        Args:
            hidden_states: The input hidden states.
            batch_size: The batch size.
            seq_len: The sequence length.

        Returns:
            The output tensor after passing through the MoE layer.
        """
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_flat = hidden_states.view(-1, hidden_dim)  # [B*T, H]
        num_tokens = hidden_flat.size(0)

        # 1) Получаем, какой эксперт на каждый токен:
        router_logits = self.router(hidden_flat)  # [num_tokens, E]
        router_probs = F.softmax(router_logits, dim=-1)
        topk_vals, topk_idx = router_probs.topk(self.top_k, dim=-1)
        topk_idx = topk_idx.squeeze(-1)  # [num_tokens]

        # 2) Подготовим тензор-выход
        intermediate = hidden_flat.new_zeros(num_tokens, self.config.intermediate_size)

        # 3) Для каждого эксперта вычислим его выход исключительно на своих токенах
        for expert_id, expert in enumerate(self.experts):
            mask = topk_idx == expert_id  # булева маска [num_tokens]
            if mask.any():
                inp_i = hidden_flat[mask]  # [num_selected, H]
                out_i = expert(inp_i)  # [num_selected, D]
                intermediate[mask] = out_i

        # 4) Восстанавливаем исходный порядок + остаток
        intermediate = intermediate.view(batch_size, seq_len, -1)
        output = self.output(intermediate, hidden_states)
        return output


class BertLayerWithMoE(nn.Module):
    """
    Implements a BERT layer with Mixture of Experts (MoE) capabilities.

    Attributes:
        config: Configuration object for the layer.
    """

    def __init__(self, config):
        """
        Initializes the MoEBertLayer.

        Args:
            config: Configuration object for the layer.
        """
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention

        if self.add_cross_attention:
            assert self.is_decoder, "Cross-attention requires decoder mode"
            self.crossattention = BertAttention(config)

        # self.intermediate (BertIntermediate) и self.output (BertOutput) FFN на MoE
        self.moe = MoE(config)  # Используем наш MoE-слой

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        """
        Forward pass of the encoder.

        Args:
            hidden_states: The hidden states to be processed.
            attention_mask: Mask used to avoid attention to padding tokens.
            head_mask: Mask used to avoid attention to certain heads.
            encoder_hidden_states: Hidden states from the encoder, used for cross-attention.
            encoder_attention_mask: Attention mask for the encoder hidden states.
            output_attentions: Whether to output attention weights.

        Returns:
            A tuple containing the encoder's output and potentially attention weights.
        """
        # Self-Attention
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[
            1:
        ]  # add self-attentions if we output attention weights

        # Cross-Attention for decoder if applicable
        if self.is_decoder and encoder_hidden_states is not None:
            # ensure cross-attention layers exist if encoder hidden states are passed
            assert hasattr(self, "crossattention"), (
                f"If `encoder_hidden_states` are passed, {self} has to be instantiated "
                "with cross-attention layers by setting `config.add_cross_attention=True`"
            )
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = (
                outputs + cross_attention_outputs[1:]
            )  # add cross-attentions if we output attention weights

        # Feed-forward / Mixture-of-Experts block
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output,
        )
        outputs = (layer_output,) + outputs
        return outputs

    def feed_forward_chunk(self, attention_output):
        """
        Performs a feed-forward pass through the hidden layers.

            Args:
                config: Configuration object containing the number of hidden layers.

            Returns:
                The output of the feed-forward pass.
        """
        return self.moe(attention_output)


class BertMoEEncoder(BertEncoder):
    """
    Encodes input sequences using a Mixture of Experts (MoE) BERT architecture.

    Attributes:
        config: Configuration object for the model.
        add_pooling_layer: Boolean indicating whether to add a pooling layer.
    """

    def __init__(self, config):
        """
        Initializes the model.

        Args:
            config: Configuration object.
            add_pooling_layer: Whether to add a pooling layer.

        Returns:
            None
        """
        super(BertEncoder, self).__init__()
        self.config = config
        self.layer = nn.ModuleList(
            [BertLayerWithMoE(config) for _ in range(config.num_hidden_layers)]
        )
        self.gradient_checkpointing = False


class BertMoEModel(BertPreTrainedModel):
    """
    A BERT model with Mixture of Experts (MoE) capabilities.

    Attributes:
        config: The configuration object for the BERT model.
        add_pooling_layer: Whether to add a pooling layer.
    """

    def __init__(self, config, add_pooling_layer=True):
        """
        Initializes the BertMoEModel.

        Args:
            config: The configuration object for the BERT model.
            add_pooling_layer: Whether to add a pooling layer.

        Returns:
            None
        """
        super().__init__(config)
        self.config = config

        self.attn_implementation = getattr(config, "attn_implementation", "eager")
        self.position_embedding_type = getattr(
            config, "position_embedding_type", "absolute"
        )

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertMoEEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None
        self.init_weights()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        """
        Performs a forward pass through the model to generate logits.

        Args:
            input_ids: Input IDs.
            attention_mask: Attention mask.
            token_type_ids: Token type IDs.
            position_ids: Position IDs.
            head_mask: Head mask.
            inputs_embeds: Input embeddings.
            output_attentions: Whether to output attentions.
            output_hidden_states: Whether to output hidden states.
            return_dict: Whether to return a dict.

        Returns:
            SequenceClassifierOutput: A SequenceClassifierOutput containing the loss, logits,
                hidden states, and attentions.
        """
        r"""
        encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)` or `(batch_size, sequence_length, target_length)`, *optional*):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
            don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
            `decoder_input_ids` of shape `(batch_size, sequence_length)`.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        """
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time"
            )
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = (
            past_key_values[0][0].shape[2] if past_key_values is not None else 0
        )

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(
                    batch_size, seq_length
                )
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=device
                )

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
        )

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length + past_key_values_length), device=device
            )

        use_sdpa_attention_masks = (
            self.attn_implementation == "sdpa"
            and self.position_embedding_type == "absolute"
            and head_mask is None
            and not output_attentions
        )

        # Expand the attention mask
        if use_sdpa_attention_masks and attention_mask.dim() == 2:
            # Expand the attention mask for SDPA.
            # [bsz, seq_len] -> [bsz, 1, seq_len, seq_len]
            if self.config.is_decoder:
                extended_attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask,
                    input_shape,
                    embedding_output,
                    past_key_values_length,
                )
            else:
                extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    attention_mask, embedding_output.dtype, tgt_len=seq_length
                )
        else:
            # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
            # ourselves in which case we just need to make it broadcastable to all heads.
            extended_attention_mask = self.get_extended_attention_mask(
                attention_mask, input_shape
            )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = (
                encoder_hidden_states.size()
            )
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if use_sdpa_attention_masks and encoder_attention_mask.dim() == 2:
                # Expand the attention mask for SDPA.
                # [bsz, seq_len] -> [bsz, 1, seq_len, seq_len]
                encoder_extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    encoder_attention_mask, embedding_output.dtype, tgt_len=seq_length
                )
            else:
                encoder_extended_attention_mask = self.invert_attention_mask(
                    encoder_attention_mask
                )
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = (
            self.pooler(sequence_output) if self.pooler is not None else None
        )

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


class BertMoEForSequenceClassification(BertForSequenceClassification):
    """
    A class for sequence classification using a Mixture of Experts (MoE) architecture
    based on BERT.

    Attributes:
        config: The configuration object for the model.
    """

    def __init__(self, config):
        """
        Initializes the model with the given configuration.

        Args:
            config: The configuration object for the model.
        """
        super(BertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertMoEModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        # Рассчитываем loss
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # Добавляем вспомогательный loss от MoE
        aux_loss = 0.0
        if self.training:
            for layer in self.bert.encoder.layer:
                aux_loss += layer.moe.aux_loss
            if loss is not None:
                loss += aux_loss

        # Формируем вывод
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
