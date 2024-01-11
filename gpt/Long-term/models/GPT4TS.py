from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch import optim

from transformers import GPT2ForSequenceClassification
# from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from models.hugging_gpt2.GPT2 import GPT2Model
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers import BertTokenizer, BertModel
from einops import rearrange
from transformers import GPT2Tokenizer

# from layers.Embed import DataEmbedding, DataEmbedding_wo_time
# from peft import get_peft_model, LoraConfig, TaskType
import copy
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


class Adapter(nn.Module):
    def __init__(self, in_feat, hid_dim, skip=True):
        super().__init__()
        self.D_fc1 = nn.Linear(in_feat, hid_dim)
        self.D_fc2 = nn.Linear(hid_dim, in_feat)
        self.act = nn.GELU()
        self.skip = skip
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        if self.skip:
            return x + self.drop(self.D_fc2(self.act(self.D_fc1(x))))
        else:
            return self.drop(self.D_fc2(self.act(self.D_fc1(x))))

class SpectModule(nn.Module):
    def __init__(self, freq_len, adapter_len):
        super().__init__()
        self.adapter_len = adapter_len
        # self.weight = nn.Parameter(torch.rand(freq_len, adapter_len//2, dtype=torch.cfloat))
        self.weight_r = nn.Parameter(torch.rand(freq_len, adapter_len//2))
        self.weight_i = nn.Parameter(torch.rand(freq_len, adapter_len//2))
        self.drop = nn.Dropout(0.1)
    
    def forward(self, x):
        # B M N P
        B, M, N, P = x.shape
        x = rearrange(x, 'b m n p -> b m p n')
        x_ft = torch.fft.rfft(x, dim=-1)

        x_real = x_ft.real
        x_imag = x_ft.imag
        x_real = torch.einsum("bmpn, nd->bmpd", x_real, self.weight_r)
        x_imag = torch.einsum("bmpn, nd->bmpd", x_imag, self.weight_i)
        x_ft = torch.view_as_complex(torch.stack([x_real, x_imag], dim=-1))

        res = torch.fft.irfft(x_ft, dim=-1, n=self.adapter_len)
        res = rearrange(res, 'b m p n -> b m n p')

        return self.drop(res)


class SpectBlock(nn.Module):
    def __init__(self, in_feat, freq_len, low_rank=8, adapter_len=8):
        super().__init__()
        self.ln_1 = nn.LayerNorm(in_feat)
        self.ln_2 = nn.LayerNorm(in_feat)
        self.attn = SpectModule(freq_len//2+1, adapter_len)
        # self.mlp = SpectFFN(in_feat, low_rank)
    
    def forward(self, x):
        # B M N P
        x = self.attn(self.ln_1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
        return x


class FFT_adapter(nn.Module):
    def __init__(self, n_layer, in_feat, seq_len):
        super().__init__()
        self.blocks = nn.ModuleList([SpectBlock(in_feat, seq_len) for i in range(n_layer)])

    def forward(self, x):
        # B, M, L
        res_list = []
        # for i, block in enumerate(self.blocks):
        #     x = block(x)
        #     res_list.append(x)
        for i, block in enumerate(self.blocks):
            res_list.append(block(x))
        
        return res_list
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


class Model(nn.Module):
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.patch_size = configs.patch_len
        self.stride = configs.stride
        self.pred_len = configs.pred_len
        self.d_ff = configs.d_ff
        self.gpt_layers = configs.gpt_layers
        self.patch_num = (configs.seq_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride)) 
        if self.stride > 1 or self.patch_size > 1:
            self.patch_num += 1
        
        self.gpt2 = GPT2Model.from_pretrained('gpt2', output_attentions=True, output_hidden_states=True)  # loads a pretrained GPT-2 base model    
        self.gpt2.h = self.gpt2.h[:configs.gpt_layers]
        for i in range(configs.gpt_layers):
            # self.gpt2.h[i].scale = configs.scale
            self.gpt2.h[i].scale = configs.scale
            self.gpt2.h[i].attn.scale = configs.scale
            if configs.T_type == 1:
                self.gpt2.h[i].T_adapter = Adapter(configs.d_model, configs.adapter_dim, skip=False)
                self.gpt2.h[i].T_adapter_gate = torch.nn.Parameter(torch.zeros(1, self.patch_num, 1))
            if configs.C_type == 1:
                self.gpt2.h[i].C_adapter = Adapter(configs.d_model, configs.adapter_dim, skip=False)
                self.gpt2.h[i].C_num = configs.enc_in
                self.gpt2.h[i].C_adapter_gate = torch.nn.Parameter(torch.zeros(1, configs.enc_in, 1))

        self.fft_adapter = FFT_adapter(configs.spect_adapter_layer, configs.enc_in, self.patch_num)
        self.adapter_in_layer = nn.ModuleList([nn.Linear(configs.patch_len, configs.d_model) for i in range(configs.adapter_layer)])
        self.in_layer = nn.Linear(configs.patch_len, configs.d_model)

        self.proj_layer = nn.Linear(configs.d_model, self.d_ff)
        self.out_layer = nn.Linear((self.d_ff * (self.patch_num)* 2) + self.d_ff *3  , configs.pred_len)
        #self.out_layer = nn.Linear((self.d_ff * (self.patch_num)* 2)  , configs.pred_len)
        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if 'adapter' in name:
                param.requires_grad = True
            elif 'ln' in name:
                param.requires_grad = True
            elif 'wpe' in name:
                param.requires_grad = False
            else:
                param.requires_grad = False

        params = sum(p.numel() for p in self.gpt2.parameters() if p.requires_grad)
        print("number of self.gpt2 params: {}".format(params))
        params = sum(p.numel() for p in self.in_layer.parameters() if p.requires_grad)
        print("number of self.in_layer params: {}".format(params))
        params = sum(p.numel() for p in self.out_layer.parameters() if p.requires_grad)
        print("number of self.out_layer params: {}".format(params))
        params = sum(p.numel() for p in self.fft_adapter.parameters() if p.requires_grad)
        print("number of self.fft_adapter params: {}".format(params))
        kernel_size = 25 #configs.kernel_size
        self.decompsition = series_decomp(kernel_size)
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.device = torch.device('cuda:{}'.format(0))

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
    def embed_word_to_tensor(self, word, target, end=False):
        """ Embed a word and expand it to match the target tensor shape """
        # Get the word embedding
        word_idx = self.word_to_idx(word).to(device=self.device)
        word_embedding = self.gpt_embedding_layer(word_idx)

        # Average the embeddings of the tokens
        word_embedding = word_embedding.mean(dim=0)
        #print(word_embedding.shape)
        #print(target.shape)
        # Expand dimensions to match the target tensor
        word_embedding = word_embedding.unsqueeze(0).unsqueeze(0)
        word_embedding = word_embedding.expand(target.size(0), 1, target.size(2))
        
        # Concatenate the word embedding to the target tensor along the first dimension
        if not end:
            result = torch.cat((word_embedding, target), dim=1)
        else:
            result = torch.cat((target, word_embedding), dim=1)
        return result


    def forward(self, x, *args, **kwargs):
        B, L, M = x.shape
        revin_layer = RevIN(M)
        x = revin_layer(x, 'norm')
        seasonal_init, trend_init = self.decompsition(x)
        x = seasonal_init
        
        x = rearrange(x, 'b l m -> b m l')
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        fft_adapter_list = self.fft_adapter(x) # B M N P
        adapters = []
        for i in range(self.gpt_layers - len(fft_adapter_list)):
            adapters.append(None)
        for i in range(len(fft_adapter_list)):
            fft_adapter_list[i] = self.adapter_in_layer[i](fft_adapter_list[i])
            fft_adapter_list[i] = rearrange(fft_adapter_list[i], 'b m n p -> (b m) n p')
            adapters.append(fft_adapter_list[i])
        

        x_2 = trend_init
        
        x_2 = rearrange(x_2, 'b l m -> b m l')
        x_2 = self.padding_patch_layer(x_2)
        x_2 = x_2.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        fft_adapter_list2 = self.fft_adapter(x) # B M N P
        for i in range(self.gpt_layers - len(fft_adapter_list2)):
            adapters.append(None)
        for i in range(len(fft_adapter_list2)):
            fft_adapter_list2[i] = self.adapter_in_layer[i](fft_adapter_list2[i])
            fft_adapter_list2[i] = rearrange(fft_adapter_list2[i], 'b m n p -> (b m) n p')
            adapters.append(fft_adapter_list2[i])


        x = rearrange(x, 'b m n p -> (b m) n p')
        x_2 = rearrange(x_2, 'b m n p -> (b m) n p')
        x = self.in_layer(x)
        x_2 = self.in_layer(x_2)
        x = self.embed_word_to_tensor("As an assistant specializing in time series prediction, you have the vector embeddings from the seasonal decomposition of the ETT time series: [Start of seasonal]", x)
        x_2 = self.embed_word_to_tensor("[End of seasonal];;;; Here are the trend decomposition of ETT time series: [Start of trend]", x_2)
        x_2 = self.embed_word_to_tensor("[End of trend];;;; Your task is to predict the predicted ETT time series(seasonal+trend): [Start of your prediction]", x_2, end=True)
        #print(x_2.shape)
        outputs = torch.cat([x,   x_2], axis=1)
        outputs = self.gpt2(inputs_embeds=outputs, adapters=adapters).last_hidden_state

        outputs = self.proj_layer(outputs)
        outputs = self.out_layer(outputs.reshape(B*M, -1))
        #outputs = rearrange(outputs, '(b m) l -> b l m', b=B)
        split_size = outputs.shape[1] #// 2
        #outputs, outputs2 = torch.split(outputs, split_size, dim=1)
        outputs = rearrange(outputs, '(b m) l -> b l m', b=B)
        #outputs = revin_layer(outputs, 'denorm')
        #outputs2 = rearrange(outputs2, '(b m) l -> b l m', b=B)
        #outputs2 = revin_layer2(outputs2, 'denorm')


        #forecast_output = forecast_output * stdev
        #forecast_output = forecast_output + means
        outputs = revin_layer(outputs, 'denorm')

        return outputs
