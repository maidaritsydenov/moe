import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers import apply_chunking_to_forward
from transformers.models.bert.modeling_bert import (
    BertConfig,
    BertPreTrainedModel,
    BertAttention,
    SequenceClassifierOutput,
    BertForSequenceClassification,
    BertModel,
    BertEncoder,
    BertEmbeddings,
    BertPooler,
    BertLayer,
    BertIntermediate,
    BertOutput,
)


class MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts = getattr(config, 'moe_num_experts', 4)
        self.top_k = getattr(config, 'moe_top_k', 1)
        self.aux_loss_coef = getattr(config, 'moe_aux_loss_coef', 0.1)

        # Создаем экспертов (каждый эксперт - это FFN)
        self.experts = nn.ModuleList([BertIntermediate(config) for _ in range(self.num_experts)])
        self.output = BertOutput(config)  # Общий выходной слой

        # Маршрутизатор (роутер)
        self.router = nn.Linear(config.hidden_size, self.num_experts, bias=False)
        self.aux_loss = 0.0  # Для накопления вспомогательного лосса

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states_flat = hidden_states.view(-1, hidden_dim)
        num_tokens = hidden_states_flat.size(0)

        # Логиты маршрутизации
        router_logits = self.router(hidden_states_flat)  # [num_tokens, num_experts]
        router_probs = nn.functional.softmax(router_logits, dim=-1)

        # Выбираем top_k экспертов
        top_k_probs, top_k_indices = router_probs.topk(self.top_k, dim=-1)
        top_k_indices = top_k_indices.squeeze(-1)  # [num_tokens, top_k] -> [num_tokens] при top_k=1
        top_k_probs = top_k_probs.squeeze(-1)

        # Сортируем токены по индексу эксперта
        sorted_expert_indices, sorted_index = top_k_indices.sort(dim=0)
        sorted_inputs = hidden_states_flat[sorted_index]

        # Границы групп для каждого эксперта
        diff = sorted_expert_indices[1:] - sorted_expert_indices[:-1]
        boundaries = torch.cat([
            torch.tensor([0], device=hidden_states.device),
            (diff != 0).nonzero(as_tuple=True)[0] + 1,
            torch.tensor([num_tokens], device=hidden_states.device)
        ])

        # Обработка экспертов
        intermediate_output = torch.zeros(num_tokens, self.config.intermediate_size, device=hidden_states.device)
        for i in range(self.num_experts):
            start = boundaries[i]
            end = boundaries[i + 1]
            if start < end:
                expert_input = sorted_inputs[start:end]
                expert_output = self.experts[i](expert_input)
                intermediate_output[start:end] = expert_output

        # Возвращаем исходный порядок
        intermediate_output = intermediate_output[torch.argsort(sorted_index)]

        # Применяем выходной слой (residual + LayerNorm)
        output = self.output(intermediate_output, hidden_states_flat)
        output = output.view(batch_size, seq_len, hidden_dim)

        # Расчет вспомогательного лосса (балансировка нагрузки)
        if self.training:
            density = torch.zeros(self.num_experts, device=router_probs.device)
            density.index_add_(0, top_k_indices, torch.ones(num_tokens, device=router_probs.device))
            density /= num_tokens
            importance = router_probs.sum(dim=0) / num_tokens
            aux_loss = self.aux_loss_coef * self.num_experts * (density * importance).sum()
            self.aux_loss = aux_loss
        else:
            self.aux_loss = 0.0

        return output


class BertLayerWithMoE(nn.Module):
    def __init__(self, config):
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
            output_attentions=False,
    ):
        # Self-Attention
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # Сохраняем attention weights

        # Cross-Attention
        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # Добавляем cross attention weights

        # MoE вместо FFN
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk,
            self.chunk_size_feed_forward,
            self.seq_len_dim,
            attention_output
        )
        outputs = (layer_output,) + outputs
        return outputs

    def feed_forward_chunk(self, attention_output):
        return self.moe(attention_output)


class BertMoEEncoder(BertEncoder):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.config = config
        self.layer = nn.ModuleList([BertLayerWithMoE(config) for _ in range(config.num_hidden_layers)])


class BertMoEModel(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super(BertModel, self).__init__(config)
        self.config = config
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertMoEEncoder(config)
        self.pooler = BertPooler(config) if add_pooling_layer else None
        self.init_weights()


class BertMoEForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config):
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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
