from typing import Dict, List, Optional, Union, Tuple, Callable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from base_bert import BertPreTrainedModel
from utils import *


class BertSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # initialize the linear transformation layers for key, value, query
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # this dropout is applied to normalized attention scores following the original implementation of transformer
    # although it is a bit unusual, we empirically observe that it yields better performance
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

  def transform(self, x, linear_layer):
    # the corresponding linear_layer of k, v, q are used to project the hidden_state (x)
    bs, seq_len = x.shape[:2]
    proj = linear_layer(x)
    # next, we need to produce multiple heads for the proj 
    # this is done by spliting the hidden state to self.num_attention_heads, each of size self.attention_head_size
    proj = proj.view(bs, seq_len, self.num_attention_heads, self.attention_head_size)
    # by proper transpose, we have proj of [bs, num_attention_heads, seq_len, attention_head_size]
    proj = proj.transpose(1, 2)
    return proj

def attention(self, key, query, value, attention_mask=None):
    # Calculate the dot product between query and key
    scores = torch.matmul(query, key.transpose(-2, -1))

    # Scale the scores
    scores = scores / math.sqrt(self.attention_head_size)

    if attention_mask is not None:
        scores += attention_mask

    # Apply softmax to get the attention probabilities
    attention_probs = nn.Softmax(dim=-1)(scores)

    # Apply dropout
    attention_probs = self.dropout(attention_probs)

    # Multiply the attention probabilities with the value to get the final attention vector
    context_layer = torch.matmul(attention_probs, value)

    # Concatenate the heads together
    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(*new_context_layer_shape)

    return context_layer

def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    batch_size, seq_len, hidden_size = hidden_states.size()
    head_size = hidden_size // self.num_attention_heads
        
    # Reshape for multi-head attention
    query = self.query(hidden_states).view(batch_size, seq_len, self.num_attention_heads, head_size).transpose(1, 2)
    key = self.key(hidden_states).view(batch_size, seq_len, self.num_attention_heads, head_size).transpose(1, 2)
    value = self.value(hidden_states).view(batch_size, seq_len, self.num_attention_heads, head_size).transpose(1, 2)
        
    # Calculate attention scores
    attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(head_size)
        
    if attention_mask is not None:
        attention_scores = attention_scores + attention_mask
        
    # Apply softmax to get the attention weights
    attention_probs = F.softmax(attention_scores, dim=-1)
        
    # Apply dropout to the attention weights
    attention_probs = self.dropout(attention_probs)
        
    # Multiply attention probs by value
    context_layer = torch.matmul(attention_probs, value)
        
    # Concatenate heads
    context_layer = context_layer.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        
    return context_layer

class BertLayer(nn.Module):
  def __init__(self, config):
    super().__init__()
    # multi-head attention
    self.self_attention = BertSelfAttention(config)
    # add-norm
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    # feed forward
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_af = F.gelu
    # another add-norm
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

  def add_norm(self, input, output, dense_layer, dropout, ln_layer):
    """
    this function is applied after the multi-head attention layer or the feed forward layer
    input: the input of the previous layer
    output: the output of the previous layer
    dense_layer: used to transform the output
    dropout: the dropout to be applied 
    ln_layer: the layer norm to be applied
    """
    # Hint: Remember that BERT applies to the output of each sub-layer, before it is added to the sub-layer input and normalized 
    ### TODO
    # Apply dense transformation.
    output = dense_layer(output)
    # Apply dropout.
    output = dropout(output)
    # Add the input and output (Residual Connection).
    output = input + output
    # Apply Layer Normalization.
    output = ln_layer(output)
    return output


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: either from the embedding layer (first bert layer) or from the previous bert layer
    as shown in the left of Figure 1 of https://arxiv.org/pdf/1706.03762.pdf 
    each block consists of 
    1. a multi-head attention layer (BertSelfAttention)
    2. a add-norm that takes the input and output of the multi-head attention layer
    3. a feed forward layer
    4. a add-norm that takes the input and output of the feed forward layer
    """
    ### TODO
     # 1. Apply multi-head attention
    attention_output = self.self_attention(hidden_states, attention_mask)
    
    # 2. Add-norm for attention output
    attention_output = self.add_norm(hidden_states, attention_output, self.attention_dense, self.attention_dropout, self.attention_layer_norm)
    
    # 3. Apply feed-forward network on attention output
    interm_output = self.interm_dense(attention_output)
    interm_output = self.interm_af(interm_output)
    
    # 4. Add-norm for feed-forward network output
    layer_output = self.add_norm(attention_output, interm_output, self.out_dense, self.out_dropout, self.out_layer_norm)
    
    return layer_output



class BertModel(BertPreTrainedModel):
  """
  the bert model returns the final embeddings for each token in a sentence
  it consists
  1. embedding (used in self.embed)
  2. a stack of n bert layers (used in self.encode)
  3. a linear transformation layer for [CLS] token (used in self.forward, as given)
  """
  def __init__(self, config):
    super().__init__(config)
    self.config = config

    # embedding
    self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
    self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
    self.tk_type_embedding = nn.Embedding(config.type_vocab_size, config.hidden_size)
    self.embed_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)
    # position_ids (1, len position emb) is a constant, register to buffer
    position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
    self.register_buffer('position_ids', position_ids)

    # bert encoder
    self.bert_layers = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    # for [CLS] token
    self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.pooler_af = nn.Tanh()

    self.init_weights()

  def embed(self, input_ids):
    input_shape = input_ids.size()
    seq_length = input_shape[1]
    
    # Get word embeddings.
    inputs_embeds = self.word_embedding(input_ids)
    
    # Get position embeddings.
    pos_ids = self.position_ids[:, :seq_length]
    pos_embeds = self.pos_embedding(pos_ids)
    
    # Token type embeddings are created as zeros since they're not used in this case.
    tk_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
    tk_type_embeds = self.tk_type_embedding(tk_type_ids)
    
    # Add embeddings together and apply layer normalization and dropout.
    embeddings = inputs_embeds + pos_embeds + tk_type_embeds
    embeddings = self.embed_layer_norm(embeddings)
    embeddings = self.embed_dropout(embeddings)
    
    return embeddings


  def encode(self, hidden_states, attention_mask):
    """
    hidden_states: the output from the embedding layer [batch_size, seq_len, hidden_size]
    attention_mask: [batch_size, seq_len]
    """
    # get the extended attention mask for self attention
    # returns extended_attention_mask of [batch_size, 1, 1, seq_len]
    # non-padding tokens with 0 and padding tokens with a large negative number 
    extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, self.dtype)

    # pass the hidden states through the encoder layers
    for i, layer_module in enumerate(self.bert_layers):
      # feed the encoding from the last bert_layer to the next
      hidden_states = layer_module(hidden_states, extended_attention_mask)

    return hidden_states

  def forward(self, input_ids, attention_mask):
    """
    input_ids: [batch_size, seq_len], seq_len is the max length of the batch
    attention_mask: same size as input_ids, 1 represents non-padding tokens, 0 represents padding tokens
    """
    # get the embedding for each input token
    embedding_output = self.embed(input_ids=input_ids)

    # feed to a transformer (a stack of BertLayers)
    sequence_output = self.encode(embedding_output, attention_mask=attention_mask)

    # get cls token hidden state
    first_tk = sequence_output[:, 0]
    first_tk = self.pooler_dense(first_tk)
    first_tk = self.pooler_af(first_tk)

    return {'last_hidden_state': sequence_output, 'pooler_output': first_tk}
