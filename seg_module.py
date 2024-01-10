import copy
import torch
import functools

import torch.nn as nn
import torch.nn.functional as F

from torch import nn
from einops import rearrange
from torch import einsum
from torchvision import transforms


class MLP(nn.Module):
    """Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    
    
def resize_fn(img, size):
    return transforms.Resize(size, transforms.InterpolationMode.BICUBIC)(
            transforms.ToPILImage()(img))


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory, pos = None, query_pos = None):
        output = tgt
        
        for layer in self.layers:
            output = layer(output, memory, pos=pos, query_pos=query_pos)

        return output

    
class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, no_norm = False,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, pos = None, query_pos = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt
    
# Projection of x onto y
def proj(x, y):
  return torch.mm(y, x.t()) * y / torch.mm(y, y.t())

# Orthogonalize x wrt list of vectors ys
def gram_schmidt(x, ys):
  for y in ys:
    x = x - proj(x, y)
  return x

def power_iteration(W, u_, update=True, eps=1e-12):
  # Lists holding singular vectors and values
  us, vs, svs = [], [], []
  for i, u in enumerate(u_):
    # Run one step of the power iteration
    with torch.no_grad():
      v = torch.matmul(u, W)
      # Run Gram-Schmidt to subtract components of all other singular vectors
      v = F.normalize(gram_schmidt(v, vs), eps=eps)
      # Add to the list
      vs += [v]
      # Update the other singular vector
      u = torch.matmul(v, W.t())
      # Run Gram-Schmidt to subtract components of all other singular vectors
      u = F.normalize(gram_schmidt(u, us), eps=eps)
      # Add to the list
      us += [u]
      if update:
        u_[i][:] = u
    # Compute this singular value and add it to the list
    svs += [torch.squeeze(torch.matmul(torch.matmul(v, W.t()), u.t()))]
    #svs += [torch.sum(F.linear(u, W.transpose(0, 1)) * v)]
  return svs, us, vs

# Spectral normalization base class 
class SN(object):
  def __init__(self, num_svs, num_itrs, num_outputs, transpose=False, eps=1e-12):
    # Number of power iterations per step
    self.num_itrs = num_itrs
    # Number of singular values
    self.num_svs = num_svs
    # Transposed?
    self.transpose = transpose
    # Epsilon value for avoiding divide-by-0
    self.eps = eps
    # Register a singular vector for each sv
    for i in range(self.num_svs):
      self.register_buffer('u%d' % i, torch.randn(1, num_outputs))
      self.register_buffer('sv%d' % i, torch.ones(1))
  
  # Singular vectors (u side)
  @property
  def u(self):
    return [getattr(self, 'u%d' % i) for i in range(self.num_svs)]

  # Singular values; 
  # note that these buffers are just for logging and are not used in training. 
  @property
  def sv(self):
   return [getattr(self, 'sv%d' % i) for i in range(self.num_svs)]
   
  # Compute the spectrally-normalized weight
  def W_(self):
    W_mat = self.weight.view(self.weight.size(0), -1)
    if self.transpose:
      W_mat = W_mat.t()
    # Apply num_itrs power iterations
    for _ in range(self.num_itrs):
      svs, us, vs = power_iteration(W_mat, self.u, update=self.training, eps=self.eps) 
    # Update the svs
    if self.training:
      with torch.no_grad(): # Make sure to do this in a no_grad() context or you'll get memory leaks!
        for i, sv in enumerate(svs):
          self.sv[i][:] = sv     
    return self.weight / svs[0]

# Linear layer with spectral norm
class SNLinear(nn.Linear, SN):
  def __init__(self, in_features, out_features, bias=True,
               num_svs=1, num_itrs=1, eps=1e-12):
    nn.Linear.__init__(self, in_features, out_features, bias)
    SN.__init__(self, num_svs, num_itrs, out_features, eps=eps)
  def forward(self, x):
    return F.linear(x, self.W_(), self.bias)

# 2D Conv layer with spectral norm
class SNConv2d(nn.Conv2d, SN):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1,
             padding=0, dilation=1, groups=1, bias=True, 
             num_svs=1, num_itrs=1, eps=1e-12):
    nn.Conv2d.__init__(self, in_channels, out_channels, kernel_size, stride, 
                     padding, dilation, groups, bias)
    SN.__init__(self, num_svs, num_itrs, out_channels, eps=eps)    
  def forward(self, x):
    return F.conv2d(x, self.W_(), self.bias, self.stride, 
                    self.padding, self.dilation, self.groups)


class SegBlock(nn.Module):
    def __init__(self, in_channels, out_channels, 
                which_conv=nn.Conv2d, which_linear=None, 
                activation=None, upsample=None):
        super(SegBlock, self).__init__()
        
        self.in_channels, self.out_channels = in_channels, out_channels
        self.which_conv, self.which_linear = which_conv, which_linear
        self.activation = activation
        self.upsample = upsample
        
        self.conv1 = self.which_conv(self.in_channels, self.out_channels)
        self.conv2 = self.which_conv(self.out_channels, self.out_channels)
        self.learnable_sc = in_channels != out_channels or upsample
        if self.learnable_sc:
            self.conv_sc = self.which_conv(in_channels, out_channels, 
                                            kernel_size=1, padding=0)
       
        self.register_buffer('stored_mean1', torch.zeros(in_channels))
        self.register_buffer('stored_var1',  torch.ones(in_channels)) 
        self.register_buffer('stored_mean2', torch.zeros(out_channels))
        self.register_buffer('stored_var2',  torch.ones(out_channels)) 
        
        self.upsample = upsample

    def forward(self, x, y=None):
        x = F.batch_norm(x, self.stored_mean1, self.stored_var1, None, None,
                          self.training, 0.1, 1e-4)
        h = self.activation(x)
        if self.upsample:
            h = self.upsample(h)
            x = self.upsample(x)
        h = self.conv1(h)
        h = F.batch_norm(h, self.stored_mean2, self.stored_var2, None, None,
                          self.training, 0.1, 1e-4)
        
        h = self.activation(h)
        h = self.conv2(h)
        if self.learnable_sc:       
            x = self.conv_sc(x)
        return h + x


class Segmodule(nn.Module):
    
    def __init__(self,
        embedding_dim = 512,
        num_heads = 8,
        num_layers = 3,
        hidden_dim = 2048,
        dropout_rate = 0.3):
        super().__init__()
        
        # output image size will be (embedding_dim, embedding_dim) 
        self.embedding_dim = embedding_dim 
        self.low_feature_size = 32
        self.mid_feature_size = 64
        self.high_feature_size = 128
        self.final_feature_size = 256
        
        self.low_feature_conv = nn.Conv2d(1280*5+640, self.low_feature_size, kernel_size=1, bias=False)
        self.mid_feature_conv = nn.Conv2d(1280+640*4+320, self.mid_feature_size, kernel_size=1, bias=False)
        self.high_feature_conv = nn.Conv2d(640+320*6, self.high_feature_size, kernel_size=1, bias=False)

        
        self.mid_feature_mix_conv = SegBlock(
                                in_channels=self.low_feature_size+self.mid_feature_size,
                                out_channels=self.low_feature_size+self.mid_feature_size,
                                which_conv=functools.partial(SNConv2d, kernel_size=3, padding=1,num_svs=1, num_itrs=1, eps=1e-04),
                                which_linear=functools.partial(SNLinear, num_svs=1, num_itrs=1, eps=1e-04),
                                activation=nn.ReLU(inplace=True),
                                upsample=False,
                            )
        self.high_feature_mix_conv = SegBlock(
                                in_channels=self.low_feature_size+self.mid_feature_size+self.high_feature_size,
                                out_channels=self.low_feature_size+self.mid_feature_size+self.high_feature_size,
                                which_conv=functools.partial(SNConv2d, kernel_size=3, padding=1, num_svs=1, num_itrs=1, eps=1e-04),
                                which_linear=functools.partial(SNLinear, num_svs=1, num_itrs=1, eps=1e-04),
                                activation=nn.ReLU(inplace=True),
                                upsample=False,
                            )
        self.final_feature_conv = SegBlock(
                                in_channels=self.low_feature_size+self.mid_feature_size+self.high_feature_size,
                                out_channels=self.final_feature_size,
                                which_conv=functools.partial(SNConv2d, kernel_size=3, padding=1, num_svs=1, num_itrs=1, eps=1e-04),
                                which_linear=functools.partial(SNLinear, num_svs=1, num_itrs=1, eps=1e-04),
                                activation=nn.ReLU(inplace=True),
                                upsample=False,
                            )
        
        self.input_mlp = MLP(2048, embedding_dim*2, embedding_dim, 3)
        self.output_mlp = MLP(embedding_dim, embedding_dim*2, self.final_feature_size, 3)

        query_dim = [16*32, 16*96, 16*224] # hard-code parameters according to pretrained diffusion models
        self.to_k = nn.ModuleList([nn.Linear(query_dim[i], embedding_dim, bias=False) for i in range(3)])
        self.to_q = nn.ModuleList([nn.Linear(embedding_dim, embedding_dim, bias=False) for _ in range(3)])
        
        decoder_layer = TransformerDecoderLayer(embedding_dim, num_heads, hidden_dim, dropout_rate)
        self.transfromer_decoder = nn.ModuleList([TransformerDecoder(decoder_layer, num_layers) for _ in range(3)])
        
        
    def forward(self, diffusion_feature, text_embedding):
        
        low_features, mid_features, high_features, final_feat = self._prepare_features(diffusion_feature)
        features = [low_features, mid_features, high_features]
        
        batch_size = features[0].size()[0]
        patch_size = 4 # the same as ViT
        text_embedding = rearrange(text_embedding, 'b n d -> (b n) d  ')
        output_query = self.input_mlp(text_embedding)
        
        for i, image_feature in enumerate(features):
            # patch partition
            image_feature = torch.nn.functional.unfold(image_feature, patch_size, stride=patch_size).transpose(1,2).contiguous()
            image_feature = rearrange(image_feature, 'b n d -> (b n) d  ')
            # transformer block
            q = self.to_q[i](output_query)
            k = self.to_k[i](image_feature)
            output_query = self.transfromer_decoder[i](q, k, None)
        
        output_query = rearrange(output_query, '(b n) d -> b n d', b=batch_size)
        mask_embedding = self.output_mlp(output_query)
        seg_result = einsum('b d h w, b n d -> b n h w', final_feat, mask_embedding)
        
        return seg_result
    
    def _prepare_features(self, features, upsample='bilinear'):
        
        low_features = [F.interpolate(i, size=self.low_feature_size, mode=upsample, align_corners=False) for i in features["low"]]
        low_features = torch.cat(low_features, dim=1)
        mid_features = [F.interpolate(i, size=self.mid_feature_size, mode=upsample, align_corners=False) for i in features["mid"]]
        mid_features = torch.cat(mid_features, dim=1)
        high_features = [F.interpolate(i, size=self.high_feature_size, mode=upsample, align_corners=False) for i in features["high"]]
        high_features = torch.cat(high_features, dim=1)
        
        low_features = self.low_feature_conv(low_features) # batch_size, 32, 32, 32
        low_feat = F.interpolate(low_features, size=self.mid_feature_size, mode='bilinear', align_corners=False)
        # low_feat batch_size, 32, 64, 64
        mid_features = self.mid_feature_conv(mid_features) # batch_size, 64, 64, 64
        mid_feat = torch.cat([low_feat, mid_features], dim=1) # batch_size, 96, 64, 64
        mid_feat = self.mid_feature_mix_conv(mid_feat, y=None) # batch_size, 96, 64, 64
        mid_feat = F.interpolate(mid_feat, size=self.high_feature_size, mode='bilinear', align_corners=False)
        # mid_feat batch_size, 96, 128, 128
        high_features = self.high_feature_conv(high_features) # batch_size, 128, 128, 128
        high_feat = torch.cat([mid_feat, high_features], dim=1) # batch_size, 224, 128, 128 
        high_feat = self.high_feature_mix_conv(high_feat, y=None) # batch_size, 224, 128, 128 
        high_feat = F.interpolate(high_feat, size=self.high_feature_size*2, mode='bilinear', align_corners=False)
        # high_feat batch_size, 224, 256, 256
        final_feat = self.final_feature_conv(high_feat, y=None) # batch_size, 256, 256, 256
        final_feat = F.interpolate(final_feat, size=self.embedding_dim, mode='bilinear', align_corners=False)
        # batch_size, 256, 512, 512
        return low_feat, mid_feat, high_feat, final_feat