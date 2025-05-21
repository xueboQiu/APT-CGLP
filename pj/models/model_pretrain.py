'''
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''
import gc
from functools import partial
from pj.models.vit import VisionTransformer, interpolate_pos_embed
from pj.models.xbert import BertConfig, BertForMaskedLM

import torch
import torch.nn.functional as F
from torch import nn
from pj.models.gnn import GNN

import random
class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        Initialize the Cross Attention module.
        :param embed_dim: Dimension of the input feature vectors.
        :param num_heads: Number of heads for the multi-head attention mechanism.
        """
        super(CrossAttentionFusion, self).__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.norm = nn.LayerNorm(embed_dim)  # LayerNorm for stabilizing training.
        self.dropout = nn.Dropout(0)  # Dropout for randomly dropping output features.

    def forward(self, feature1, feature2, atte1=None, atte2=None):
        """
        Forward propagation.
        :param feature1: Feature vector 1, shape (batch_size, seq_len, embed_dim).
        :param feature2: Feature vector 2, shape (batch_size, seq_len, embed_dim).
        :param atte1: Attention mask for feature1 (optional).
        :param atte2: Attention mask for feature2 (optional).
        :return: Fused feature vector, shape (batch_size, seq_len, embed_dim).
        """
        # Transform input from (batch_size, seq_len, embed_dim) to (seq_len, batch_size, embed_dim).
        feature1 = feature1.permute(1, 0, 2)  # (seq_len, batch_size, embed_dim)
        feature2 = feature2.permute(1, 0, 2)  # (seq_len, batch_size, embed_dim)

        # key_padding_mask shape is (batch_size, seq_len), where 0 indicates ignore and 1 indicates compute attention.
        key_padding_mask = None
        if atte2 is not None:
            key_padding_mask = (atte2 == 0)  # Convert 0s in atte2 to True (ignore positions).

        # Cross Attention: Use feature1 as Query, feature2 as Key and Value.
        attn_output, _ = self.cross_attention(
            query=feature1,
            key=feature2,
            value=feature2,
            key_padding_mask=key_padding_mask  # Use atte2 to control attention range.
        )

        # Apply Dropout, residual connection, and LayerNorm.
        attn_output = self.norm(self.dropout(attn_output) + feature1)

        # Transform output back to (batch_size, seq_len, embed_dim).
        return attn_output.permute(1, 0, 2)
class CGLP(nn.Module):
    def __init__(self,
                 text_encoder = None,
                 tokenizer = None,
                 config = None,
                 temp = 0.07,
                 init_deit = True
                 ):
        super().__init__()

        self.tokenizer = tokenizer
        self.mlm_probability = config['mlm_probability']
        embed_dim = config['embed_dim']

        graph_width = config['graph_width']
        bert_config = BertConfig.from_json_file(config['bert_config'])
        self.graph_encoder = GNN(node_dim=config['node_dim'],edge_dim=config['edge_dim'],num_layer=config['num_layer'],
                                 emb_dim=config['embed_dim'],graph_width = graph_width, drop_ratio = 0.1)

        self.text_encoder = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config)

        text_width = self.text_encoder.config.hidden_size

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']
        # self.itm_head = nn.Linear(graph_width, 2)
        self.itm_head = nn.Sequential(
            nn.Linear(graph_width, graph_width // 2),
            nn.BatchNorm1d(graph_width // 2),  # Batch Normalization
            nn.ReLU(),
            nn.Linear(graph_width // 2, 2)
        )
        # 初始化 Cross Attention 模块
        self.cross_attention_fusion = CrossAttentionFusion(graph_width, 2)
        self.mgm_head = nn.Linear(text_width, embed_dim)
        # create momentum models
        self.graph_encoder_m = GNN(node_dim=config['node_dim'],edge_dim=config['edge_dim'],num_layer=config['num_layer'],emb_dim=config['embed_dim'],graph_width = graph_width)
        self.text_encoder_m = BertForMaskedLM.from_pretrained(text_encoder, config=bert_config)

        # self.graph_proj = nn.Linear(graph_width, embed_dim)
        # self.text_proj = nn.Linear(text_width, embed_dim)
        # self.graph_proj_m = nn.Linear(graph_width, embed_dim)
        # self.text_proj_m = nn.Linear(text_width, embed_dim)
        self.graph_proj = nn.Sequential(
            nn.Linear(graph_width, embed_dim),
            nn.BatchNorm1d(embed_dim),  # Batch Normalization
            torch.nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.text_proj = nn.Sequential(
            nn.Linear(graph_width, embed_dim),
            nn.BatchNorm1d(embed_dim),  # Batch Normalization
            torch.nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.graph_proj_m = nn.Sequential(
            nn.Linear(graph_width, embed_dim),
            nn.BatchNorm1d(embed_dim),  # Batch Normalization
            torch.nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.text_proj_m = nn.Sequential(
            nn.Linear(graph_width, embed_dim),
            nn.BatchNorm1d(embed_dim),  # Batch Normalization
            torch.nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )

        # all momentum models are not updated by gradient
        self.model_pairs = [[self.graph_encoder,self.graph_encoder_m],
                            [self.graph_proj,self.graph_proj_m],
                            [self.text_encoder,self.text_encoder_m],
                            [self.text_proj,self.text_proj_m],
                           ]
        self.copy_params()

        # create the queue
        self.register_buffer("graph_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.graph_queue = nn.functional.normalize(self.graph_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

    def init_weights(self,m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def embed_text_modality(self, text_encoder, text, momentum, max_len=512, stride=256):

        bs = len(text['input_ids'])  # Should be 8
        batch_sequence_embeddings = []
        batch_sentence_embeddings = []
        # Process each sentence individually
        for bi in range(bs):
            input_ids = text['input_ids'][bi]
            attention_mask = text['attention_mask'][bi]
            seq_len = len(input_ids)

            # If the sentence length is less than or equal to max_len, process directly
            if seq_len <= max_len:
                if momentum:
                    outputs = self.text_encoder_m.bert(
                        input_ids.unsqueeze(0),
                        attention_mask=attention_mask.unsqueeze(0),
                        return_dict=True,
                        mode='text'
                    )
                else:
                    outputs = self.text_encoder.bert(
                        input_ids.unsqueeze(0),
                        attention_mask=attention_mask.unsqueeze(0),
                        return_dict=True,
                        mode='text'
                    )
                sentence_embedding = outputs.last_hidden_state[:, 0, :]  # [1, hidden_size]
                sequence_embedding = outputs.last_hidden_state
                batch_sentence_embeddings.append(sentence_embedding)
                batch_sequence_embeddings.append(sequence_embedding.squeeze(0))
                continue

            # For long sequences, use a sliding window
            # Store token-level embeddings
            sequence_embedding = torch.zeros(seq_len, 768).to(input_ids.device)
            overlap_counts = torch.zeros(seq_len).to(input_ids.device)

            # Store [CLS] embeddings for each window
            cls_embeddings = []

            # Process using a sliding window
            for i in range(0, seq_len - max_len + 1, stride):
                input_ids_chunk = input_ids[i:i + max_len].unsqueeze(0)
                attention_mask_chunk = attention_mask[i:i + max_len].unsqueeze(0)

                if momentum:
                    outputs = self.text_encoder_m.bert(
                        input_ids_chunk,
                        attention_mask=attention_mask_chunk,
                        return_dict=True,
                        mode='text'
                    )
                else:
                    outputs = self.text_encoder.bert(
                        input_ids_chunk,
                        attention_mask=attention_mask_chunk,
                        return_dict=True,
                        mode='text'
                    )
                # Get token-level embeddings
                chunk_embedding = outputs.last_hidden_state.squeeze(0)  # [max_len, hidden_size]
                sequence_embedding[i:i + max_len] = sequence_embedding[i:i + max_len] + chunk_embedding
                overlap_counts[i:i + max_len] = overlap_counts[i:i + max_len] + 1

                # Get [CLS] embedding
                cls_embedding = outputs.last_hidden_state[:, 0, :]  # [1, hidden_size]
                cls_embeddings.append(cls_embedding)

            # Process the last window
            if seq_len > max_len:
                input_ids_chunk = input_ids[-max_len:].unsqueeze(0)
                attention_mask_chunk = attention_mask[-max_len:].unsqueeze(0)

                if momentum:
                    outputs = self.text_encoder_m.bert(
                        input_ids_chunk,
                        attention_mask=attention_mask_chunk,
                        return_dict=True,
                        mode='text'
                    )
                else:
                    outputs = self.text_encoder.bert(
                        input_ids_chunk,
                        attention_mask=attention_mask_chunk,
                        return_dict=True,
                        mode='text'
                    )

                # Get token-level embeddings
                chunk_embedding = outputs.last_hidden_state.squeeze(0)
                sequence_embedding[-max_len:] = sequence_embedding[-max_len:] + chunk_embedding
                overlap_counts[-max_len:] = overlap_counts[-max_len:] + 1

                # Get [CLS] embedding
                cls_embedding = outputs.last_hidden_state[:, 0, :]
                cls_embeddings.append(cls_embedding)

            # Calculate the average of token-level embeddings
            overlap_counts = torch.clamp(overlap_counts, min=1)
            sequence_embedding /= overlap_counts.unsqueeze(-1)

            # Calculate the average of sentence-level embeddings
            cls_embeddings = torch.cat(cls_embeddings, dim=0)  # [num_windows, hidden_size]
            sentence_embedding = torch.mean(cls_embeddings, dim=0, keepdim=True)  # [1, hidden_size]

            batch_sequence_embeddings.append(sequence_embedding)
            batch_sentence_embeddings.append(sentence_embedding)
        return batch_sequence_embeddings, batch_sentence_embeddings
    def overhead_predict_matching(self, graph, text, tokenizer, device=None):

        text_input = tokenizer(text, padding='longest', truncation=True, max_length=3000,
                               return_tensors="pt").to(device)
        graph_embeds_split, text_embeds, graph_feat, text_feat = self.create_graph_text_embeds([graph],text_input)
        # text_embeds = text_embeds.squeeze(0)
        graph_embeds = graph_embeds_split[0]
        graph_attens = torch.ones(graph_embeds.size(0), device=graph_feat.device)
        graph_embed = graph_embeds.unsqueeze(0)  # [1, seq_len, hidden_size]
        graph_att = graph_attens.unsqueeze(0)
        # atten都为1
        output = self.text_encoder.bert(encoder_embeds = text_embeds,
                                        attention_mask = text_input.attention_mask,
                                        encoder_hidden_states = graph_embed,
                                        encoder_attention_mask = graph_att,
                                        return_dict = True,
                                        mode = 'fusion',
                                       )
        gl_embeddings = output.last_hidden_state[:, 0, :]
        gl_output = self.itm_head(gl_embeddings)  # 不要立即使用argmax

        pred = gl_output.argmax(dim=1)
        return pred

    def forward(self, graph, text, alpha=0):
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        text = self.tokenizer(text, padding='longest', truncation=True, max_length=3000, return_tensors="pt").to(
            self.text_encoder.device)

        graph_embeds_split, text_embeds, graph_feat, text_feat = self.create_graph_text_embeds(graph, text,
                                                                                               momentum=False)

        # cosine similarity logits
        sim_i2t = graph_feat @ text_feat.t() / self.temp
        sim_t2i = text_feat @ graph_feat.t() / self.temp
        # hard labels of matched pairs
        # sim_targets = torch.zeros(sim_i2t.size()).to(self.text_encoder.device)
        # sim_targets.fill_diagonal_(1)
        # sim_i2t_targets = alpha * F.softmax(sim_i2t, dim=1) + (1 - alpha) * sim_targets
        # sim_t2i_targets = alpha * F.softmax(sim_t2i, dim=1) + (1 - alpha) * sim_targets

        # # get momentum features
        with torch.no_grad():
            self._momentum_update()
            graph_embeds_split_m, text_embeds_m, graph_feat_m, text_feat_m = self.create_graph_text_embeds(graph, text,
                                                                                                           momentum=True)
            sim_i2t_m = graph_feat_m @ text_feat_m.t() / self.temp
            sim_t2i_m = text_feat_m @ graph_feat_m.t() / self.temp

            sim_targets = torch.zeros(sim_i2t_m.size()).to(self.text_encoder.device)
            sim_targets.fill_diagonal_(1)

            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets

        # loss of i2t and t2i
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()

        loss_ita = (loss_i2t + loss_t2i) / 2
        # Predicted indices for graph-to-text
        pred_i2t = sim_i2t.argmax(dim=1)  # Index of the maximum value in each row
        # Predicted indices for text-to-graph
        pred_t2i = sim_t2i.argmax(dim=1)  # Index of the maximum value in each row
        # Ground truth indices
        target_indices = torch.arange(sim_i2t.size(0)).to(sim_i2t.device)
        # Calculate accuracy
        acc_i2t = (pred_i2t == target_indices).float().mean()  # Accuracy for graph-to-text
        acc_t2i = (pred_t2i == target_indices).float().mean()  # Accuracy for text-to-graph
        # Average accuracy
        ita_acc = (acc_i2t + acc_t2i) / 2

        self._dequeue_and_enqueue(graph_feat_m, text_feat_m)
        ###=================================###
        padded_graph_embeds, graph_atts = self.pad_graph_embeds(graph_embeds_split, device=self.text_encoder.device)
        padded_graph_embeds_m, _ = self.pad_graph_embeds(graph_embeds_split_m, device=self.text_encoder.device)
        # Forward the positive graph-text pair. Special processing for fusing text and graph information, where graph embeddings participate in computation with text embeddings
        # output_pos = self.text_encoder.bert(encoder_embeds = text_embeds,
        #                                 attention_mask = text.attention_mask,
        #                                 encoder_hidden_states = padded_graph_embeds,
        #                                 encoder_attention_mask = graph_atts,
        #                                 return_dict = True,
        #                                 mode = 'fusion',
        #                                )
        output_pos = self.cross_attention_fusion(text_embeds, padded_graph_embeds)
        # select a hard negative graph for each text
        with torch.no_grad():
            bs = graph_atts.size(0)
            weights_i2t = F.softmax(sim_i2t[:, :bs], dim=1)
            weights_t2i = F.softmax(sim_t2i[:, :bs], dim=1)

            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0)

        graph_embeds_neg = []
        graph_atts_neg = []
        for b in range(bs):
            # neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            neg_idx = torch.topk(weights_t2i[b], k=1, largest=True).indices.item()  # absolutely hard negative
            graph_embeds_neg.append(padded_graph_embeds[neg_idx])
            graph_atts_neg.append(graph_atts[neg_idx])
            # graph_embeds_neg.append(graph_feat[neg_idx])
        graph_embeds_neg = torch.stack(graph_embeds_neg, dim=0)
        graph_atts_neg = torch.stack(graph_atts_neg, dim=0)

        # select a hard negative text for each graph
        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            # neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            neg_idx = torch.topk(weights_t2i[b], k=1, largest=True).indices.item()  # absolutely hard negative
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text.attention_mask[neg_idx])
            # text_embeds_neg.append(text_feat[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg], dim=0)

        graph_embeds_all = torch.cat([graph_embeds_neg, padded_graph_embeds], dim=0)
        graph_atts_all = torch.cat([graph_atts_neg, graph_atts], dim=0)

        # output_neg = self.text_encoder.bert(encoder_embeds = text_embeds_all,
        #                                 attention_mask = text_atts_all,
        #                                 encoder_hidden_states = graph_embeds_all,
        #                                 encoder_attention_mask = graph_atts_all,
        #                                 return_dict = True,
        #                                 mode = 'fusion',
        #                                )
        output_neg = self.cross_attention_fusion(text_embeds_all, graph_embeds_all)
        # vl_embeddings = torch.cat([output_pos.last_hidden_state[:,0,:], output_neg.last_hidden_state[:,0,:]],dim=0)
        vl_embeddings = torch.cat([output_pos[:, 0, :], output_neg[:, 0, :]], dim=0)
        # vl_embeddings = torch.cat([output_pos.mean(dim=1), output_neg.mean(dim=1)],dim=0)

        vl_output = self.itm_head(vl_embeddings)

        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)], dim=0).to(
            self.text_encoder.device)

        loss_itm = F.cross_entropy(vl_output, itm_labels)

        itm_acc = (vl_output.argmax(dim=1) == itm_labels).float().mean()
        itm_pos_acc = (vl_output[:bs].argmax(dim=1) == 1).float().mean()
        # print("================================ itm_pos",vl_output[:bs].argmax(dim=1))
        # print(f"Predicted distribution:{vl_output.argmax(dim=1).cpu().numpy()}, acc:{itm_acc}")

        ##================= MGM ========================##
        # Clone tensors to ensure the original tensor is not modified
        padded_graph_embeds = padded_graph_embeds.clone()  # Avoid inplace modification
        updated_graph_atts = graph_atts.clone()  # Avoid inplace modification
        masked_node_features = []  # Store features of masked nodes

        for i in range(padded_graph_embeds.size(0)):
            # Get the number of nodes in the current graph (assuming non-zero `graph_atts` indicate valid nodes)
            num_nodes = (graph_atts[i] != 0).sum().item()

            # Randomly select one node to mask
            mask_idx = torch.randint(0, num_nodes, (1,)).item()

            # Extract and store the masked node's feature (ensure to clone the feature to prevent subsequent modifications)
            masked_node_feature = padded_graph_embeds[i, mask_idx, :].clone()
            masked_node_features.append(masked_node_feature)

            # Set the masked node's feature to zero vector (or other masking marker)
            padded_graph_embeds[i, mask_idx, :] = torch.zeros_like(padded_graph_embeds[i, mask_idx, :])

            # Set the corresponding value in `graph_atts` to 0 to ensure the attention ignores this node
            updated_graph_atts[i, mask_idx] = 0

        # Convert masked node features to tensor
        masked_node_features = torch.stack(masked_node_features)  # [batch_size, embed_dim]
        # Use CrossAttentionFusion module and pass in updated_graph_atts
        mgm_output = self.cross_attention_fusion(
            text_embeds,  # Text embeddings
            padded_graph_embeds,  # Graph embeddings
            atte1=None,  # No mask needed for text embeddings
            atte2=updated_graph_atts  # Attention mask for graph embeddings
        )
        # Forward the positive graph-text pair. Special processing for fusing text and graph information, where graph embeddings participate in computation with text embeddings
        # mgm_output = self.text_encoder.bert(encoder_embeds = text_embeds,
        #                                 attention_mask = text.attention_mask,
        #                                 encoder_hidden_states = padded_graph_embeds,
        #                                 encoder_attention_mask = updated_graph_atts,
        #                                 return_dict = True,
        #                                 mode = 'fusion',
        #                                )

        # Calculate loss
        # Create label tensor (assuming all node pairs are similar here)
        # all_mgm_loss = 1 - nn.functional.cosine_similarity(mgm_output.last_hidden_state[:,0,:], masked_node_features, dim=1)
        all_mgm_loss = 1 - nn.functional.cosine_similarity(mgm_output[:, 0, :], masked_node_features, dim=1)
        loss_mgm = all_mgm_loss.mean()

        ##================= MLM ========================##
        input_ids = text.input_ids.clone()
        labels = input_ids.clone()

        # Example usage:
        max_length = 512
        if input_ids.size(1) > max_length:
            # Randomly select a window
            input_ids, text.attention_mask, labels = self.get_random_window(
                input_ids,
                text.attention_mask,
                labels,
                window_size=max_length
            )
        # Ensure all inputs are contiguous
        input_ids = input_ids.contiguous()
        text.attention_mask = text.attention_mask.contiguous()
        if labels is not None:
            labels = labels.contiguous()
        # Mask on the selected window
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        input_ids, labels = self.mask(input_ids,
                                      self.text_encoder.config.vocab_size,
                                      self.text_encoder.device,
                                      targets=labels,
                                      probability_matrix=probability_matrix)
        with torch.no_grad():
            logits_m = self.text_encoder_m(input_ids,
                                           attention_mask=text.attention_mask,
                                           encoder_hidden_states=padded_graph_embeds_m,
                                           encoder_attention_mask=graph_atts,
                                           return_dict=True,
                                           return_logits=True,
                                           )
        # Feed into the model
        mlm_output = self.text_encoder(input_ids,
                                       attention_mask=text.attention_mask,
                                       encoder_hidden_states=padded_graph_embeds,
                                       encoder_attention_mask=graph_atts,
                                       return_dict=True,
                                       labels=labels,
                                       soft_labels=F.softmax(logits_m, dim=-1),
                                       alpha=alpha
                                       )
        loss_mlm = mlm_output.loss

        return loss_mlm, loss_ita, loss_itm, loss_mgm, itm_acc, ita_acc, itm_pos_acc

    def get_random_window(self, input_ids, attention_mask, labels=None, window_size=512, min_size=128):
        """
        Randomly select a window
        window_size: Maximum window size
        min_size: Minimum window size to avoid too small windows
        """
        seq_length = input_ids.size(1)

        # If the sequence length is less than the window size, return the entire sequence
        if seq_length <= window_size:
            return input_ids, attention_mask, labels

        # Randomly select window size (between min_size and window_size)
        actual_window_size = torch.randint(min_size, min(window_size, seq_length), (1,)).item()

        # Randomly select the start position of the window
        max_start = seq_length - actual_window_size
        start_idx = torch.randint(0, max_start, (1,)).item()
        end_idx = start_idx + actual_window_size

        # Extract the window
        windowed_input_ids = input_ids[:, start_idx:end_idx]
        windowed_attention_mask = attention_mask[:, start_idx:end_idx]
        windowed_labels = labels[:, start_idx:end_idx] if labels is not None else None

        return windowed_input_ids, windowed_attention_mask, windowed_labels

    def predict_matching(self, graph, text, device=None):
        graph_embeds_split, text_embeds, graph_feat, text_feat = self.create_graph_text_embeds(graph, text, momentum=False)

        # Cosine similarity logits
        # sim_i2t = graph_feat[0] @ text_feat.t()
        sim_i2t = graph_feat @ text_feat.t()
        # graph_atts = torch.stack([torch.ones(graph_embed.shape[0]) for graph_embed in graph_embeds_split], dim=0)
        # graph_embeds_split = torch.stack([graph_embed for graph_embed in graph_embeds_split])

        padded_graph_embeds, graph_atts = self.pad_graph_embeds(graph_embeds_split, device=self.text_encoder.device)

        if device is not None:
            graph_embeds, graph_atts = padded_graph_embeds.to(device), graph_atts.to(device)
            text_embeds, text = text_embeds.to(device), text.to(device)

        # output = self.text_encoder.bert(
        #     encoder_embeds=text_embeds,
        #     attention_mask=text.attention_mask,
        #     encoder_hidden_states=graph_embeds,
        #     encoder_attention_mask=graph_atts,
        #     return_dict=True,
        #     mode='fusion'
        # )
        # # Get the representation of the [CLS] token
        # gl_embeddings = output.last_hidden_state[:, 0, :]
        # gl_output = self.itm_head(gl_embeddings).argmax(dim=-1)  # Do not immediately use argmax

        # Fuse graph and text embeddings through cross-attention
        gl_embeddings = self.cross_attention_fusion(text_embeds, graph_embeds)[:, 0, :]  # Get the representation of the [CLS] token

        # Compute the output through the prediction head
        gl_output = self.itm_head(gl_embeddings).argmax(dim=-1)  # [batch_size, num_classes]
        # print("================================ gl_output",gl_output)

        return gl_output, sim_i2t.cpu()
    def pad_graph_embeds(self, graph_embeds_split, device=None):
        # Get the maximum number of nodes in the batch
        max_num_nodes = max([embed.size(0) for embed in graph_embeds_split])
        # Padding each graph's node embeddings to max_num_nodes
        # Pad each graph's number of nodes to max_num_nodes
        padded_graph_embeds = [
            F.pad(embed, (0, 0, 0, max_num_nodes - embed.shape[0]))  # Right-side padding on the node dimension
            for embed in graph_embeds_split
        ]
        # Convert padded_graph_embeds to a tensor with shape (batch_size, max_num_nodes, hidden_size)
        padded_graph_embeds = torch.stack(padded_graph_embeds)
        # Create graph_atts to mark valid nodes (1) and padded nodes (0)
        graph_atts = [
            torch.cat([torch.ones(embed.shape[0]), torch.zeros(max_num_nodes - embed.shape[0])])
            for embed in graph_embeds_split
        ]
        # Convert graph_atts to a tensor with shape (batch_size, max_num_nodes)
        graph_atts = torch.stack(graph_atts)
        # Move to GPU if a device is specified
        if device is not None:
            padded_graph_embeds, graph_atts = padded_graph_embeds.to(device), graph_atts.to(device)

        return padded_graph_embeds, graph_atts
    def create_graph_text_embeds(self, graph, text,momentum=False):

        if graph is None and text is None:
            raise ValueError("Both graph and text cannot be None")

        graph_embeds_split, batch_sequence_embeddings, graph_feat, batch_sentence_embeddings = None, None, None, None

        if graph is not None:
            # graph_sentences = []
            # for g in graph:
            #     # Used to store each source node and its corresponding interaction information
            #     node_interactions = {}
            #     # Get basic information of the graph
            #     edge_index = g.edge_index  # Edge indices [2, num_edges]
            #     edge_attr = g.edge_attr  # Edge attributes (list)
            #     substitutes = g.substitute  # Node attributes (list)
            #     num_edges = g.num_edges  # Number of edges
            #
            #     # Iterate over each edge
            #     for i in range(num_edges):
            #         src_node = edge_index[0, i].item()  # Source node of the edge
            #         tgt_node = edge_index[1, i].item()  # Target node of the edge
            #         edge_description = edge_attr[i]  # Edge attribute description
            #
            #         # Get the attributes of the source node
            #         src_node_name = substitutes[src_node]
            #         tgt_node_name = substitutes[tgt_node]
            #         # Check if src_node_name starts with "process"
            #         if src_node_name.startswith("process"):
            #             # Store the interaction information in the dictionary
            #             if src_node_name not in node_interactions:
            #                 node_interactions[src_node_name] = []
            #             if 'socket NA' not in tgt_node_name:
            #                 node_interactions[src_node_name].append(f"{edge_description} {tgt_node_name}")
            #         elif tgt_node_name.startswith("process"):
            #             # Store the interaction information in the dictionary
            #             if tgt_node_name not in node_interactions:
            #                 node_interactions[tgt_node_name] = []
            #             if 'socket NA' not in src_node_name:
            #                 node_interactions[tgt_node_name].append(f"{edge_description} {src_node_name}")
            #     # Concatenate the interaction information of each source node into sentences
            #     sentences = []
            #     for node, interactions in node_interactions.items():
            #         # Concatenate all interactions of the current node
            #         interactions_str = ", ".join(interactions)
            #         sentences.append(f"{node} {interactions_str}")
            #     # Concatenate all nodes' interaction sentences into a single long sentence
            #     graph_sentences.append("; ".join(sentences))
            # # Perform tokenization on all interaction sentences
            # graph_input = self.tokenizer(graph_sentences, padding='longest', truncation=True, max_length=4000,
            #                                 return_tensors="pt").to(self.text_encoder.device)
            #
            # with torch.no_grad():
            #     graph_embeds_split, graph_feats = self.embed_text_modality(self.text_encoder, graph_input, momentum)
            #
            # graph_embeds_split = torch.stack(graph_embeds_split, dim=0)
            # graph_feats = torch.cat(graph_feats, dim=0)
            # # Normalize and project the graph embeddings
            # if momentum:
            #     graph_feats = F.normalize(self.graph_proj_m(graph_feats), dim=-1)
            # else:
            #     graph_feats = F.normalize(self.graph_proj(graph_feats), dim=-1)
            # =============================================================
            # def create_graph_text_embeds(self, graph, text_sentences):
            # 1. 提取所有 substitute 和 edge_attr 列表，并合并成大批次
            # 1. Extract all substitute and edge_attr lists and merge them into large batches
            all_substitutes = [sub for data in graph for sub in data.substitute]
            all_edge_attrs = [edge for data in graph for edge in data.edge_attr]

            # 2. Perform batch tokenization for all substitute and edge_attr
            x_texts_batch = self.tokenizer(all_substitutes, padding='longest', truncation=True, max_length=400,
                                           return_tensors="pt").to(self.text_encoder.device)
            edge_texts_batch = self.tokenizer(all_edge_attrs, padding='longest', truncation=True, max_length=300,
                                              return_tensors="pt").to(self.text_encoder.device)
            # 3. Use the BERT encoder to obtain features for nodes and edges in batches
            with torch.no_grad():
                if momentum:
                    x_atts_batch = self.text_encoder_m.bert(x_texts_batch.input_ids,
                                                            attention_mask=x_texts_batch.attention_mask, return_dict=True, mode='text')
                    edge_atts_batch = self.text_encoder_m.bert(edge_texts_batch.input_ids,
                                                               attention_mask=edge_texts_batch.attention_mask, return_dict=True, mode='text')
                else:
                    x_atts_batch = self.text_encoder.bert(x_texts_batch.input_ids, attention_mask=x_texts_batch.attention_mask,
                                                          return_dict=True, mode='text')
                    edge_atts_batch = self.text_encoder.bert(edge_texts_batch.input_ids,
                                                             attention_mask=edge_texts_batch.attention_mask, return_dict=True, mode='text')
            x_feat_batch = F.normalize(x_atts_batch.last_hidden_state[:, 0, :], dim=-1).to(self.text_encoder.device)  # Node features
            # x_feat_batch = x_atts_batch.last_hidden_state[:, 0, :]  # Node features
            edge_feat_batch = F.normalize(edge_atts_batch.last_hidden_state[:, 0, :], dim=-1).to(self.text_encoder.device)  # Edge features
            # edge_feat_batch = edge_atts_batch.last_hidden_state[:, 0, :]  # Edge features
            # Assume graph is a list of multiple small graphs
            graph_feats = []  # Used to store the final embeddings of each small graph
            graph_embeds_split = []
            for data in graph:
                # Get the number of nodes and edges in the current small graph
                node_count = data.num_nodes
                edge_count = data.num_edges

                # Get the node and edge features of the current small graph
                x_feats = x_feat_batch[:node_count]  # Node features of the current small graph
                edge_feats = edge_feat_batch[:edge_count]  # Edge features of the current small graph

                # Get the edge index of the current small graph
                edge_index = data.edge_index.to(self.text_encoder.device)

                # Use the graph_encoder to embed the single small graph
                if momentum:
                    graph_embds4nodes = self.graph_encoder_m(x_feats, edge_index, edge_feats)
                else:
                    graph_embds4nodes = self.graph_encoder(x_feats, edge_index, edge_feats)
                graph_embeds_split.append(graph_embds4nodes)
                # Compute the graph-level embedding of the current small graph (using the average of node features as the graph embedding here)
                graph_embed = torch.mean(graph_embds4nodes, dim=0)  # Or use torch.sum, depending on the task requirements
                graph_feats.append(graph_embed)

                # Update x_feat_batch and edge_feat_batch, removing the data of the current small graph
                x_feat_batch = x_feat_batch[node_count:]
                edge_feat_batch = edge_feat_batch[edge_count:]

            # Stack all graph embeddings into a tensor
            graph_feats = torch.stack(graph_feats)
            # graph_embeds_split = torch.stack(graph_embeds_split)
            # Normalize and project the graph embeddings
            if momentum:
                graph_feats = F.normalize(self.graph_proj_m(graph_feats), dim=-1)
            else:
                graph_feats = F.normalize(self.graph_proj(graph_feats), dim=-1)
        # =============================================================
        # text embeddings generated by BERT
        if text is not None:
            with torch.no_grad():
                batch_sequence_embeddings, batch_sentence_embeddings = self.embed_text_modality(self.text_encoder, text,momentum)

            batch_sequence_embeddings = torch.stack(batch_sequence_embeddings, dim=0)
            batch_sentence_embeddings = torch.cat(batch_sentence_embeddings, dim=0)

            if momentum:
                batch_sentence_embeddings = F.normalize(self.text_proj_m(batch_sentence_embeddings), dim=-1)
            else:
                batch_sentence_embeddings = F.normalize(self.text_proj(batch_sentence_embeddings), dim=-1)

            # ============ original without text splitting =================================
            # text_output = self.text_encoder.bert(text.input_ids, attention_mask=text.attention_mask, return_dict=True,
            #                                      mode='text')
            # text_embeds = text_output.last_hidden_state
            # text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)
            # ============ original without text splitting =================================

        return graph_embeds_split, batch_sequence_embeddings, graph_feats, batch_sentence_embeddings

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient


    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)



    @torch.no_grad()
    def _dequeue_and_enqueue(self, graph_feat, text_feat):
        # gather keys before updating queue
        # graph_feats = concat_all_gather(graph_feat)
        # text_feats = concat_all_gather(text_feat)

        # single gpu
        graph_feats = graph_feat
        text_feats = text_feat

        batch_size = graph_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.graph_queue[:, ptr:ptr + batch_size] = graph_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr
    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:
            masked_indices = torch.bernoulli(probability_matrix).bool()

        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False

        if targets is not None:
            targets[~masked_indices] = -100 # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        if targets is not None:
            return input_ids, targets
        else:
            return input_ids


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

