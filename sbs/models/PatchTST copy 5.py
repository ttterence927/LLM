import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from math import sqrt, log
import matplotlib.pyplot as plt

from transformers import GPT2ForSequenceClassification
from transformers import  GPT2Tokenizer

from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from models.embed2 import DataEmbedding
from embed import DataEmbedding_wo_time

import random
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.device = torch.device('cuda:{}'.format(0))
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features)).to(device=self.device)
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features)).to(device=self.device)

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x
def l2norm(t):
    return F.normalize(t, dim = -1)

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()
        
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, attn_bias):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            attn_bias
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1,
                 output_attention=False, configs=None,
                 attn_scale_init=20):
        super(FullAttention, self).__init__()
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

        self.enc_in = configs.enc_in
       
        self.scale = scale

    def forward(self, queries, keys, values, attn_mask, attn_bias):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        if attn_bias is not None:
            attn_bias = attn_bias.permute(0, 3, 1, 2)
            A = self.dropout(torch.softmax(scores * scale + attn_bias, dim=-1))
        else:
            A = self.dropout(torch.softmax(scores * scale, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.BatchNorm1d(d_model)
        self.norm2 = nn.BatchNorm1d(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, attn_bias=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            attn_bias=attn_bias
        )
        x = x + self.dropout(new_x)
        y = x = self.norm1(x.permute(0, 2, 1)).permute(0, 2, 1)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        y = x + y
        y = self.norm2(y.permute(0, 2, 1)).permute(0, 2, 1)
        return y, attn

class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, attn_bias=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, attn_bias=attn_bias)
                attns.append(attn)

        if self.norm is not None:
            # x = self.norm(x)
            x = self.norm(x.permute(0, 2, 1)).permute(0, 2, 1)

        return x, attns
class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class PatchTST(nn.Module):
    """
    Vanilla Transformer with O(L^2) complexity
    """
    def __init__(self, configs, device):
        super(PatchTST, self).__init__()
        self.is_gpt = configs.is_gpt
        self.enc_in = configs.enc_in
        self.patch_size = configs.patch_size
        self.stride = configs.stride
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1
        #self.patch_num += 1
        self.label_len = configs.label_len
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = False
        self.num_heads = configs.n_heads
        self.factor = 3
        self.activation = 'gelu'
        self.num_classes = configs.num_classes
        self.gpt_layers = configs.gpt_layers
        self.device = torch.device('cuda:{}'.format(0))
        if configs.is_gpt:
            self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

            self.gpt2.h = self.gpt2.h[:self.gpt_layers]
            print("gpt2 = {}".format(self.gpt2))
            for i, (name, param) in enumerate(self.gpt2.named_parameters()):
                if 'ln' in name or 'wpe' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            
            self.gpt2.to(device=self.device)
        # Embedding
        #self.enc_embedding = DataEmbedding_wo_time(self.patch_size, configs.d_model, configs.embed, configs.freq, configs.dropout)

        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, self.factor, attention_dropout=configs.dropout,
                                      output_attention=self.output_attention,
                                      configs=configs),
                                      configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=self.activation
                ) for l in range(configs.e_layers)
            ],
            # norm_layer=torch.nn.LayerNorm(configs.d_model)
            norm_layer=torch.nn.BatchNorm1d(configs.d_model)
        )
        
        self.ln_proj_classification = nn.LayerNorm(configs.d_model * (self.patch_num + 1))
        self.ln_proj_forecasting = nn.LayerNorm(configs.d_model * (self.patch_num + 1))

        # Custom Activation Functions
        self.act_classification = F.relu # Example: ReLU for classification
        self.act_forecasting = F.sigmoid # Example: Sigmoid for forecasting

        # Uncertainty Estimation for Forecasting
        #self.forecast_variance_head = nn.Linear(configs.d_model * (self.patch_num + 1), configs.pred_len)

        self.cnt = 0
        self.classification_head = nn.Linear(configs.d_model * (self.patch_num+1), self.num_classes)
        self.forecast_head = nn.Linear(configs.d_model * (self.patch_num+1), configs.pred_len)
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        self.in_layer = nn.Linear(configs.patch_size *2 , configs.d_model)
        self.out_layer = nn.Linear(configs.d_model * (self.patch_num +1), configs.pred_len * 2)
        ########################DECOMP
        kernel_size = configs.kernel_size
        self.decompsition = series_decomp(kernel_size)
        self.individual = False
        self.channels = configs.enc_in

    def word_to_idx(self, word):
        """ Convert the word to its index in GPT's vocabulary """
        tokens = self.tokenizer.encode(word, add_special_tokens=False)
        return torch.tensor(tokens, dtype=torch.long).to(device=self.device)

    def gpt_embedding_layer(self, word_idx):
        """ Get the embedding from GPT's embedding layer """
        return self.gpt2.wte(word_idx).to(device=self.device)

    def embed_word(self, word, embedding_dim):
        """ Get the GPT embedding for a given word """
        word_idx = self.word_to_idx(word).to(device=self.device)  # Convert the word to its index in GPT's vocabulary
        word_embedding = self.gpt_embedding_layer(word_idx)  # Get the embedding from GPT's embedding layer
        word_embedding = word_embedding.expand(-1, embedding_dim)  # Ensure the embedding matches the required dimensions
        return word_embedding
    def embed_word_to_tensor(self, word, target_shape):
        """ Embed a word and expand it to match the target tensor shape """
        # Get the word embedding
        
        word_idx = self.word_to_idx(word).to(device=self.device)
        word_embedding = self.gpt_embedding_layer(word_idx)

        # Average the embeddings of the tokens
        word_embedding = word_embedding.mean(dim=0)
        return word_embedding

    def forward(self, x, itr):

        B, L, M = x.shape
        seasonal_init, trend_init = self.decompsition(x)
        x = seasonal_init
        revin_layer = RevIN(M)
        x = revin_layer(x, 'norm')
        x = rearrange(x, 'b l m -> b m l')
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x = rearrange(x, 'b m n p -> (b m) n p')
        #print(input_x.shape)


        x_2 = trend_init
        revin_layer2 = RevIN(M)
        x_2 = revin_layer2(x_2, 'norm')
        x_2 = rearrange(x_2, 'b l m -> b m l')
        x_2 = self.padding_patch_layer(x_2)
        x_2 = x_2.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x_2 = rearrange(x_2, 'b m n p -> (b m) n p')

        seasonal_word = self.embed_word_to_tensor("Here are the seasonal decomposition of time series:", x.shape)
        trend_word = self.embed_word_to_tensor("[End of seasonal]Here are the trend decomposition of time series:", x.shape)
        prompt = self.embed_word_to_tensor("[End of trend] Output the [forecasted seasonal decomposition of time series] and [forecasted trend decomposition of time series]", x.shape)

        out = torch.cat([x,  x_2], axis=2)
        out = self.in_layer(out)
        if self.is_gpt:
            outputs = self.gpt2(inputs_embeds=out).last_hidden_state

        classification_activations = self.act_classification(outputs).reshape(B, -1)
        classification_activations = self.ln_proj_classification(classification_activations)
        classification_output = self.classification_head(classification_activations)

        outputs = self.out_layer(outputs.reshape(B*M, -1))

        split_size = outputs.shape[1] // 2
        outputs, outputs2 = torch.split(outputs, split_size, dim=1)
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)
        outputs = revin_layer(outputs, 'denorm')
        outputs2 = rearrange(outputs2, '(b m) l -> b l m', b=B)
        outputs2 = revin_layer2(outputs2, 'denorm')
        #outputs = outputs * stdev
        #outputs = outputs + means
        forecast_output = outputs + outputs2
        return forecast_output, classification_output
    
        input_x = rearrange(tmp, 'b l m -> b m l')
        #print(input_x.shape)
        input_x = self.padding_patch_layer(input_x)
        #print(input_x.shape)
        input_x = input_x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        #print(input_x.shape)
        input_x = rearrange(input_x, 'b m n p -> b n (p m)')
        #print(input_x.shape)
        outputs = self.enc_embedding2(input_x, None)
        #print(outputs.shape)
        outputs, attns = self.encoder(outputs, attn_mask=None)
        #print(outputs.shape)
        if self.is_gpt:
            outputs = self.gpt2(inputs_embeds=outputs).last_hidden_state

        classification_activations = self.act_classification(outputs).reshape(B, -1)
        forecast_activations = self.act_forecasting(outputs).reshape(B, -1)

        classification_activations = self.ln_proj_classification(classification_activations)
        forecast_activations = self.ln_proj_forecasting(forecast_activations)

        # Computing classification and forecast outputs
        classification_output = self.classification_head(classification_activations)
        forecast_output = self.forecast_head(forecast_activations)

        # Uncertainty estimation for forecasting
        #forecast_variance = torch.exp(self.forecast_variance_head(forecast_activations)) # Exponential to ensure variance is positive
        #print(classification_output.shape)
        if self.output_attention:
            return forecast_output, classification_output#, forecast_variance, 0  # attns
        else:
            return forecast_output, classification_output#, forecast_variance

        B, L, M = x_enc.shape
        #32  168 122
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False)+ 1e-5).detach() 
        x_enc /= stdev

        x_enc = rearrange(x_enc, 'b l m -> b m l')
        x_enc = x_enc.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        x_enc = rearrange(x_enc, 'b m n p -> (b m) n p')

        enc_out = self.enc_embedding(x_enc)

        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        #print(B, L, M) 
        forecast_output = self.proj(enc_out.reshape(B*M, -1))
        #print(forecast_output.shape)
        forecast_output = rearrange(forecast_output, '(b m) l -> b l m', m=M)
        #print(forecast_output.shape)
        #forecast_output = forecast_output[:, -self.pred_len:, :1]
        #print(forecast_output.shape)
        forecast_output = forecast_output * stdev#[:, :, :1]
        forecast_output = forecast_output + means#[:, :, :1]
        
        

        


