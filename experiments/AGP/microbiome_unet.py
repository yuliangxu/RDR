# ICFM model: Credit to Ganchao Wei (https://weigcdsb.github.io/)
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from tqdm import tqdm
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
import math

from torchcfm.conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher
)
from torchdyn.core import NeuralODE
from torch.distributions.multivariate_normal import MultivariateNormal
from torchcfm.optimal_transport import OTPlanSampler
from skbio.diversity import beta_diversity
from scipy.stats import describe
import math
import copy
from typing import List

import scipy.spatial.distance as dist
import seaborn as sns
from skbio.diversity import beta_diversity
from skbio.stats.ordination import pcoa

import ot
from numpy.linalg import norm
from scipy.spatial.distance import cdist

# Set device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)

def lt_transform_forward(x, child_node_matrix, out_names, taxa_names, epsilon=1e-6):
    
    # Convert inputs to consistent formats
    if isinstance(child_node_matrix, pd.DataFrame):
        child_node_matrix = child_node_matrix.values
    child_node_matrix = child_node_matrix.astype(str)
    
    n_samples, n_taxa = x.shape
    n_internal = child_node_matrix.shape[0]
    n_nodes = n_taxa + n_internal
    
    # Create mapping from taxon name to index in original data
    name_to_out_index = {name: idx for idx, name in enumerate(out_names)}
    
    # Create indexer to reorder data to phylogenetic tree order
    indexer = [name_to_out_index[name] for name in taxa_names]
    
    # Precompute children indices by parsing node numbers
    children_indices = []
    for i in range(n_internal):
        c1_str, c2_str = child_node_matrix[i]
        try:
            c1_num = int(c1_str.split('_')[1])
            c2_num = int(c2_str.split('_')[1])
            children_indices.append((c1_num - 1, c2_num - 1))
        except Exception as e:
            raise ValueError(f"Error parsing node names: {e}") from e
    
    # Initialize output matrix
    lt_data = np.zeros((n_samples, n_internal))
    
    for i in range(n_samples):
        node_count = np.zeros(n_nodes)
        # Reorder sample to match phylogenetic tree tip order
        node_count[:n_taxa] = x[i, indexer]
        
        # Track computed status
        computed = np.zeros(n_nodes, dtype=bool)
        computed[:n_taxa] = True
        pending = set(range(n_taxa, n_nodes))
        
        # Process nodes in dependency order
        while pending:
            progress = False
            for node_idx in list(pending):
                j = node_idx - n_taxa
                c1_idx, c2_idx = children_indices[j]
                
                if computed[c1_idx] and computed[c2_idx]:
                    # Sum child counts for internal node
                    node_count[node_idx] = node_count[c1_idx] + node_count[c2_idx]
                    computed[node_idx] = True
                    pending.remove(node_idx)
                    progress = True
            
            if not progress and pending:
                raise RuntimeError(f"Stalled at nodes: {pending}")
        
        # Compute log-odds with epsilon smoothing
        for j in range(n_internal):
            c1_idx, c2_idx = children_indices[j]
            count1 = node_count[c1_idx]
            count2 = node_count[c2_idx]
            
            # Apply R-style smoothing: 0->epsilon, 1->1-epsilon
            count1 = epsilon if count1 == 0 else (1 - epsilon if count1 == 1 else count1)
            count2 = epsilon if count2 == 0 else (1 - epsilon if count2 == 1 else count2)
            
            lt_data[i, j] = np.log(count1 / count2)
    
    return lt_data

def smooth_transformation(x, child_node_matrix, out_names, taxa_names, 
                         epsilon=1e-6, noise_scale=1e-7):
    """
    Add noise before transformation to break symmetries
    """
    # Convert to numpy if needed
    if hasattr(x, 'cpu'):
        x = x.cpu().numpy()
    
    # Add symmetric noise (preserves relative abundances)
#     noise = noise_scale * (np.random.rand(*x.shape))
#     x_noisy = x + noise
    
    x_noisy = copy.deepcopy(x)
    x_noisy[x_noisy == 0.] = noise_scale * np.random.rand(len(x[x == 0.]))
    
    
    
    # Renormalize to preserve simplex constraint
    row_sums = x_noisy.sum(axis=1, keepdims=True)
    x_noisy = x_noisy / row_sums
    
    # Apply transformation
    return lt_transform_forward(
        x_noisy,
        child_node_matrix,
        out_names,
        taxa_names,
        epsilon
    )


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    
    :param timesteps: a 1-D Tensor of N indices, one per batch element.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """
        raise NotImplementedError


class ResBlock1D(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    Adapted for 1D vector data.
    """
    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_scale_shift_norm=True,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            nn.LayerNorm(channels),
            nn.SiLU(),
            nn.Linear(channels, self.out_channels),
        )

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        
        self.out_layers = nn.Sequential(
            nn.LayerNorm(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(self.out_channels, self.out_channels),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Linear(channels, self.out_channels)

    def forward(self, x, emb):
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb)
        
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
            
        return self.skip_connection(x) + h


class AttentionBlock1D(nn.Module):
    """
    Self-attention block for 1D data.
    """
    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert channels % num_head_channels == 0
            self.num_heads = channels // num_head_channels
            
        self.norm = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.attention = nn.MultiheadAttention(
            channels, 
            self.num_heads, 
            batch_first=True
        )
        self.proj_out = nn.Linear(channels, channels)

    def forward(self, x):
        b, d = x.shape
        h = self.norm(x)
        
        # Self-attention
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        
        # Reshape for attention (add sequence dimension)
        q = q.unsqueeze(1)  # (b, 1, d)
        k = k.unsqueeze(1)  # (b, 1, d) 
        v = v.unsqueeze(1)  # (b, 1, d)
        
        h, _ = self.attention(q, k, v)
        h = h.squeeze(1)  # (b, d)
        
        h = self.proj_out(h)
        return x + h


class UNet1D(nn.Module):
    """
    1D U-Net adapted for vector field prediction.
    """
    def __init__(
        self,
        dim,
        out_dim=None,
        model_channels=128,
        num_res_blocks=2,
        channel_mult=(1, 2, 4, 8),
        attention_resolutions=(8,),
        dropout=0.1,
        time_varying=True,
        num_heads=4,
        num_head_channels=-1,
        use_scale_shift_norm=True,
    ):
        super().__init__()
        
        self.dim = dim
        self.out_dim = out_dim or dim
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.channel_mult = channel_mult
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.time_varying = time_varying
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels

        # Time embedding
        time_embed_dim = model_channels * 4
        if time_varying:
            self.time_embed = nn.Sequential(
                nn.Linear(model_channels, time_embed_dim),
                nn.SiLU(),
                nn.Linear(time_embed_dim, time_embed_dim),
            )

        # Initial projection
        ch = int(channel_mult[0] * model_channels)
        self.input_blocks = nn.ModuleList([
            nn.Linear(dim, ch)
        ])
        
        input_block_chans = [ch]
        ds = 1
        
        # Downsampling
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock1D(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=int(mult * model_channels),
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(mult * model_channels)
                
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock1D(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                        )
                    )
                    
                self.input_blocks.append(nn.Sequential(*layers))
                input_block_chans.append(ch)
                
            if level != len(channel_mult) - 1:
                # Downsampling via linear projection
                out_ch = ch
                self.input_blocks.append(
                    nn.Sequential(
                        nn.Linear(ch, out_ch),
                        nn.LayerNorm(out_ch)
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2

        # Middle
        self.middle_block = nn.Sequential(
            ResBlock1D(
                ch,
                time_embed_dim,
                dropout,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock1D(
                ch,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
            ),
            ResBlock1D(
                ch,
                time_embed_dim,
                dropout,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        # Upsampling
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock1D(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=int(model_channels * mult),
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = int(model_channels * mult)
                
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock1D(
                            ch,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                        )
                    )
                    
                if level and i == num_res_blocks:
                    # Upsampling via linear projection
                    out_ch = ch
                    layers.append(
                        nn.Sequential(
                            nn.Linear(ch, out_ch),
                            nn.LayerNorm(out_ch)
                        )
                    )
                    ds //= 2
                    
                self.output_blocks.append(nn.Sequential(*layers))

        # Output
        self.out = nn.Sequential(
            nn.LayerNorm(ch),
            nn.SiLU(),
            nn.Linear(ch, self.out_dim),
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        # Zero initialize output
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Apply the model to an input batch.
        
        :param x: an [N x (dim+1)] Tensor if time_varying, else [N x dim]
        :return: an [N x out_dim] Tensor of outputs.
        """
        if self.time_varying:
            x_in = x[:, :-1]
            t = x[:, -1]
            
            # Time embedding
            t_emb = timestep_embedding(t, self.model_channels)
            emb = self.time_embed(t_emb)
        else:
            x_in = x
            emb = None

        # U-Net forward
        h = x_in
        hs = []
        
        # Downsampling
        for module in self.input_blocks:
            if isinstance(module, nn.Sequential):
                for layer in module:
                    if isinstance(layer, TimestepBlock):
                        h = layer(h, emb)
                    else:
                        h = layer(h)
            else:
                h = module(h)
            hs.append(h)
            
        # Middle
        for layer in self.middle_block:
            if isinstance(layer, TimestepBlock):
                h = layer(h, emb)
            else:
                h = layer(h)
                
        # Upsampling with skip connections
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=-1)
            for layer in module:
                if isinstance(layer, TimestepBlock):
                    h = layer(h, emb)
                else:
                    h = layer(h)
                    
        # Output
        return self.out(h)


# Simplified U-Net with fewer parameters
class UNet(nn.Module):
    """
    Simplified U-Net style architecture that matches your MLP interface exactly.
    """
    def __init__(self, dim, out_dim=None, w=128, n_layers=4, 
                 time_varying=False, dropout=0.1):
        super().__init__()
        
        # Match interface
        self.time_varying = time_varying
        if out_dim is None:
            out_dim = dim
            
        # Configure U-Net
        # For dim=122, reasonable channel multipliers
        if dim > 64:
            channel_mult = (1, 2, 4, 4)
        elif dim > 32:
            channel_mult = (1, 2, 4)
        else:
            channel_mult = (1, 2)
            
        # Create U-Net
        self.unet = UNet1D(
            dim=dim,
            out_dim=out_dim,
            model_channels=w,
            num_res_blocks=max(1, n_layers // len(channel_mult)),
            channel_mult=channel_mult,
            attention_resolutions=(4,),
            dropout=dropout,
            time_varying=time_varying,
            num_heads=4,
            use_scale_shift_norm=True,
        )
        
    def forward(self, x):
        return self.unet(x)


    
class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""
    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)
        
    def forward(self, x):
        x_proj = x[:, :, None] * self.W[None, None, :] * 2 * math.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1).squeeze(1)


class Dense(nn.Module):
    """A linear layer with optional bias."""
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
    def forward(self, x):
        return self.linear(x)

class CNNModel(torch.nn.Module):
    """
    Adapted from their CNNModel - uses 1D convolutions with dilations.
    Since we have vector data not sequences, we treat dimensions as 'sequence length'.
    """
    def __init__(self, dim, out_dim=None, w=256, num_cnn_stacks=2, 
                 time_varying=False, dropout=0.1):
        super().__init__()
        self.time_varying = time_varying
        self.dim = dim
        self.num_cnn_stacks = num_cnn_stacks
        if out_dim is None:
            out_dim = dim
            
        # Time embedder
        if time_varying:
            self.time_embedder = nn.Sequential(
                GaussianFourierProjection(embed_dim=w),
                nn.Linear(w, w)
            )
            
        # Input projection - we expand each dimension to w channels
        input_dim = 1 + (1 if time_varying else 0)
        self.linear = nn.Conv1d(input_dim, w, kernel_size=1)
        
        # Stacked dilated convolutions
        self.num_layers = 5 * num_cnn_stacks
        dilations = [1, 1, 4, 16, 64]
        paddings = [0, 0, 0, 0, 0]  # We'll use padding='same' mode
        
        self.convs = nn.ModuleList()
        for stack in range(num_cnn_stacks):
            for dilation, padding in zip(dilations, paddings):
                self.convs.append(
                    nn.Conv1d(w, w, kernel_size=9, dilation=dilation, padding='same')
                )
                
        # Time modulation layers
        self.time_layers = nn.ModuleList([
            Dense(w, w) for _ in range(self.num_layers)
        ]) if time_varying else None
        
        # Normalization
        self.norms = nn.ModuleList([
            nn.LayerNorm(w) for _ in range(self.num_layers)
        ])
        
        # Output projection
        self.final_conv = nn.Sequential(
            nn.Conv1d(w, w, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(w, out_dim, kernel_size=1)
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv1d):
            nn.init.xavier_uniform_(m.weight, gain=0.5)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
                
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Prepare input
        if self.time_varying:
            x_feat = x[:, :-1].unsqueeze(1)  # (batch, 1, dim)
            t = x[:, -1:]
            time_emb = F.relu(self.time_embedder(t))
            
            # Expand time to all dimensions
            t_expanded = t.unsqueeze(1).expand(-1, 1, self.dim)  # (batch, 1, dim)
            feat = torch.cat([x_feat, t_expanded], dim=1)  # (batch, 2, dim)
        else:
            feat = x.unsqueeze(1)  # (batch, 1, dim)
            time_emb = None
            
        # Initial projection
        feat = F.relu(self.linear(feat))  # (batch, w, dim)
        
        # Apply conv layers with residuals
        for i in range(self.num_layers):
            h = self.dropout(feat.clone())
            
            # Time modulation
            if time_emb is not None:
                h = h + self.time_layers[i](time_emb).unsqueeze(2)
                
            # Normalize (reshape for LayerNorm)
            h = h.permute(0, 2, 1)  # (batch, dim, w)
            h = self.norms[i](h)
            h = h.permute(0, 2, 1)  # (batch, w, dim)
            
            # Convolution
            h = F.relu(self.convs[i](h))
            
            # Residual connection
            if h.shape == feat.shape:
                feat = h + feat
            else:
                feat = h
                
        # Output projection
        feat = self.final_conv(feat)  # (batch, out_dim, dim)
        
        # For vector field, we want output per dimension
        if feat.shape[2] == self.dim and feat.shape[1] == 1:
            return feat.squeeze(1)  # (batch, dim)
        else:
            # Average over spatial dimension
            return feat.mean(dim=2)  # (batch, out_dim)    
    
    
    
    
# class MLP(torch.nn.Module):
#     def __init__(self, dim, out_dim=None, w=64, time_varying=False):
#         super().__init__()
#         self.time_varying = time_varying
#         if out_dim is None:
#             out_dim = dim
#         self.net = torch.nn.Sequential(
#             torch.nn.Linear(dim + (1 if time_varying else 0), w),
#             torch.nn.SELU(),
#             torch.nn.Linear(w, w),
#             torch.nn.SELU(),
#             torch.nn.Linear(w, w),
#             torch.nn.SELU(),
#             torch.nn.Linear(w, out_dim),
#         )

#     def forward(self, x):
#         return self.net(x)

class MLP(nn.Module):
    def __init__(self, dim, out_dim=None, w=128, time_varying=False, 
                 dropout_rate=0.2, activation='SELU', num_layers=4):
        """
        Enhanced MLP with regularization features
        Args:
            dim: Input dimension
            out_dim: Output dimension (default = dim)
            w: Hidden layer width
            time_varying: Whether to include time conditioning
            dropout_rate: Dropout probability (0 = no dropout)
            activation: Activation function ('SELU', 'ReLU', 'LeakyReLU')
            num_layers: Total number of layers (input + hidden)
        """
        super().__init__()
        self.time_varying = time_varying
        self.out_dim = out_dim or dim
        self.dropout_rate = dropout_rate
        self.num_layers = max(3, num_layers)  # Minimum 3 layers
        self.activation_type = activation  # Store for reference
        
        # Build network layers
        layers = []
        input_size = dim + (1 if time_varying else 0)
        
        # Input layer
        layers.append(nn.Linear(input_size, w))
        layers.append(self._get_activation(activation))
        
        # Add dropout after input if using SELU
        if dropout_rate > 0:
            if activation == 'SELU':
                layers.append(nn.AlphaDropout(dropout_rate))
            else:
                layers.append(nn.Dropout(dropout_rate))
        
        # Hidden layers
        for _ in range(self.num_layers - 3):  # -3 for input/output layers
            layers.append(nn.Linear(w, w))
            layers.append(self._get_activation(activation))
            if dropout_rate > 0:
                if activation == 'SELU':
                    layers.append(nn.AlphaDropout(dropout_rate))
                else:
                    layers.append(nn.Dropout(dropout_rate))
        
        # Output layer
        layers.append(nn.Linear(w, self.out_dim))
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights using PyTorch's apply mechanism
        self.apply(self._init_weights)
    
    def _get_activation(self, name):
        if name == 'ReLU':
            return nn.ReLU(inplace=True)
        elif name == 'LeakyReLU':
            return nn.LeakyReLU(0.1, inplace=True)
        else:  # Default to SELU
            return nn.SELU(inplace=True)
    
    def _init_weights(self, module):
        """Initialize weights for Linear layers"""
        if isinstance(module, nn.Linear):
            # Determine initialization based on activation type
            if self.activation_type in ['ReLU', 'LeakyReLU']:
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            else:  # SELU - use LeCun initialization
                # Calculate fan-in for the layer
                fan_in = module.weight.size(1)
                # Lecun_normal: std = 1 / sqrt(fan_in)
                std = 1 / math.sqrt(fan_in)
                nn.init.normal_(module.weight, mean=0, std=std)
            
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        return self.net(x)
    

def ema(source, target, decay =  0.9999):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay + source_dict[key].data * (1 - decay)
        )
    

warmup = 2000
def warmup_lr(step):
    return min(step, warmup) / warmup



def cfm(model, optimizer, x1, z1,
        batch_size, nt = 1, sigma = 0, n_epochs = 1000, x0 = None,
        storeCheck = False, epoch_check_step = 100,
        t0 = 0, t1 = 1, ot = False, grad_clip = 1., ema_decay = 0.9999, 
        ema_model = None, useSche = True):
    
    N = x1.shape[0]
    dim = x1.shape[1]
    
    if ema_model is None:
        ema_model = copy.deepcopy(model)
        
    if useSche:
#         print('ische')
        sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr)
    
    nbatch = int(N/batch_size)
    if ot:
        print('iot')
        FM = ExactOptimalTransportConditionalFlowMatcher(sigma=sigma)
    else:
        FM = ConditionalFlowMatcher(sigma=sigma)
    
    n_per_epoch = nbatch*batch_size
    shuffled_idx_all = np.zeros((n_epochs, 2, n_per_epoch), int)
    for ep in range(n_epochs):
        shuffled_idx_all[ep,1,:] = np.random.permutation(x1.shape[0])[0:n_per_epoch]
        if x0 is not None:
            shuffled_idx_all[ep,0,:] = np.random.permutation(x0.shape[0])[0:n_per_epoch]
    
    # batch_idx = np.reshape(np.arange(0,N),[nbatch, batch_size])
    
    losses: List[float] = []
    if storeCheck:
        check_pts = []
        check_steps = []
        
    model.train()
    for k in tqdm(range(n_epochs)):
#     for k in range(n_epochs):
        shuffle_idx_tmp = shuffled_idx_all[k,:,:]
        x0_batch_idx = np.reshape(shuffle_idx_tmp[0,:], [nbatch,-1])
        x1_batch_idx = np.reshape(shuffle_idx_tmp[1,:], [nbatch,-1])
        
        for bb in range(nbatch):
            x1_batch = x1[x1_batch_idx[bb,:],:]
            
            if x0 is None:
                x0_batch = torch.randn_like(x1_batch)
            else:
                x0_batch = x0[x0_batch_idx[bb,:],:]
            
            t_expand = torch.rand(nt*batch_size).type_as(x0_batch)
            t_expand_scale = (t1 - t0) * t_expand + t0
            
            x1_expand = x1_batch.repeat(nt, 1)
            x0_expand = x0_batch.repeat(nt, 1)
            
            _, xt, ut = FM.sample_location_and_conditional_flow(x0_expand, x1_expand, t_expand_scale)
            
            if z1 is None:
                vt = model(torch.cat([xt, t_expand[:, None]], dim=-1))
            else:
                z1_batch = z1[x1_batch_idx[bb,:],:]
                z1_expand = z1_batch.repeat(nt, 1)
                vt = model(torch.cat([xt, z1_expand, t_expand[:, None]], dim=-1))
            
            loss = torch.mean((vt - ut) ** 2)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            
            optimizer.step()
            if useSche:
                sched.step()
            
            ema(model, ema_model, ema_decay)
            optimizer.zero_grad()
            
            # Logging
            losses.append(loss.item())
            
            if storeCheck:
                if k % epoch_check_step == 0:
                    check_pts.append(deepcopy(model.state_dict()))
                    check_steps.append(k)
    if storeCheck:       
        return model, ema_model, losses, check_pts, check_steps
    else:
        return model, ema_model, losses
    

# GP version
def calc_r(ti, tj):
    r = ti[...,None] - tj[...,None,:]
    r[r == 0] = 1e-15
    return r
def k11(r, alpha, l):
    return (alpha**2)*torch.exp(-0.5 * ((r/l)**2))
def k12(r, alpha, l):
    return (alpha**2/l**2)*r*torch.exp(-0.5*((r/l)**2))
def k22(r, alpha, l):
    return (alpha**2/l**4)*(l**2 - r**2)*torch.exp(-0.5*((r/l)**2))

def cov_mat2(ti, tj, alpha, l, sig2_diag = 1e-8):
    
    r = calc_r(ti, tj)
    nB = r.shape[0]
    nt = r.shape[1]
    
    Sig11 = k11(r, alpha, l) + (torch.eye(nt)*sig2_diag).repeat(nB,1,1)
    Sig12 = k12(r, alpha, l)
    Sig21 = Sig12.permute(0, 2, 1)
    Sig22 = k22(r, alpha, l)
    
    block_row1 = torch.cat([Sig11, Sig12], dim=2)
    block_row2 = torch.cat([Sig21, Sig22], dim=2)
    Sig = torch.cat([block_row1, block_row2], dim = 1)
    Sig = (Sig + Sig.permute(0, 2, 1))/2
    
    return Sig

def cov_inv_sing(alpha, l, t_obs, sig2_diag=1e-8):
    
    nt_obs = t_obs.shape[0]
    r_obs_obs = calc_r(t_obs, t_obs)
    
    Sig_22_sing = k11(r_obs_obs, alpha, l) + torch.eye(nt_obs) * sig2_diag
    Sig_22_inv_sing = torch.linalg.inv(Sig_22_sing)
    
    return Sig_22_inv_sing


def samp_x_dx2(t_mat, alpha, l, x_obs, t_obs, sig2_diag=1e-8, antithetic = False):
    
    nB, nt, dim = x_obs.shape[0], t_mat.shape[1], x_obs.shape[2]
    nt_obs = t_obs.shape[0]

    # Compute necessary covariance matrices and kernel functions
    r_obs_x = calc_r(t_obs, t_mat)
    r_obs_obs = calc_r(t_obs, t_obs)
    Sig_11 = cov_mat2(t_mat, t_mat, alpha, l, sig2_diag)
    
    # Precompute parts of the covariance matrices
    k_obs_x, k_obs_dx = k11(r_obs_x, alpha, l), k12(r_obs_x, alpha, l)
    Sig_21 = torch.cat([k_obs_x, k_obs_dx], dim=2)
    Sig_12 = Sig_21.permute(0, 2, 1)

    Sig_22_sing = k11(r_obs_obs, alpha, l) + torch.eye(nt_obs) * sig2_diag
    Sig_22_inv_sing = torch.linalg.inv(Sig_22_sing)
    Sig_22_inv = Sig_22_inv_sing.repeat(nB, 1, 1)

    # Compute conditional covariance matrix with stability adjustment
    Sig_cond = Sig_11 - torch.bmm(torch.bmm(Sig_12, Sig_22_inv), Sig_21)
    Sig_cond = (Sig_cond + Sig_cond.permute(0, 2, 1))/2
    
    svd_add_idx = torch.sum((torch.linalg.eigvals(Sig_cond).real>=0).T, axis = 0) != Sig_cond.shape[1]
    U, S, Vh = torch.linalg.svd(Sig_cond[svd_add_idx,:,:])
    Sig_cond_add = torch.bmm(torch.bmm(Vh.permute(0, 2, 1), torch.diag_embed(S + 1e-8)), Vh)
    Sig_cond[svd_add_idx,:,:] = (Sig_cond_add + Sig_cond_add.permute(0, 2, 1))/2
    
    # Mean adjustment matrix
    mu_A = torch.bmm(Sig_12, Sig_22_inv)
    x_obs_batch = x_obs.reshape(nB, nt_obs, dim)
    mu_new = torch.bmm(mu_A, x_obs_batch).reshape(nB, 2 * nt, dim)

    # Initialize sample matrices
    if antithetic:
        t_mat = t_mat.repeat(2,1)
        x_samps = torch.zeros((nB*2, nt, dim), dtype=x_obs.dtype, device=x_obs.device)
        dx_samps = torch.zeros((nB*2, nt, dim), dtype=x_obs.dtype, device=x_obs.device)
    else:
        x_samps = torch.zeros((nB, nt, dim), dtype=x_obs.dtype, device=x_obs.device)
        dx_samps = torch.zeros((nB, nt, dim), dtype=x_obs.dtype, device=x_obs.device)
    
    # Sampling in batch for all dimensions at once
    try:
        # Reshape mu_new and Sig_cond for compatible shapes
        mu_flat = mu_new.view(nB * dim, 2 * nt)
        Sig_cond_flat = Sig_cond.repeat(dim, 1, 1)
        
        dist = MultivariateNormal(loc= torch.zeros(mu_flat.shape), covariance_matrix=Sig_cond_flat)
        
        if antithetic:
            samp_tmp = dist.rsample()
            samp1 = (samp_tmp + mu_flat).view(nB, 2 * nt, dim)
            samp2 = (-samp_tmp + mu_flat).view(nB, 2 * nt, dim)
            x_dx_samps_flat = torch.cat((samp1, samp2), dim = 0)
        else:
            samp = dist.rsample() + mu_flat
            x_dx_samps_flat = samp.view(nB, 2 * nt, dim)
    except RuntimeError:
        print('Sampling failed; using numpy fallback.')
        if antithetic:
            x_dx_samps_flat = torch.zeros((2*nB, 2 * nt, dim), dtype=x_obs.dtype, device=x_obs.device)
            for bb in range(nB):
                for dd in range(dim):
                    mu_single = mu_new[bb, :, dd].cpu().numpy()
                    cov_single = Sig_cond[bb].cpu().numpy()
                    sample1_tmp = np.random.multivariate_normal(np.zeros(mu_single.shape), cov_single)
                    sample2_tmp = -copy.deepcopy(sample1_tmp)
                    sample1 = sample1_tmp + mu_single
                    sample2 = sample2_tmp + mu_single
                    x_dx_samps_flat[bb, :, dd] = torch.from_numpy(sample1)
                    x_dx_samps_flat[bb + nB, :, dd] = torch.from_numpy(sample2)
            
        else:
            x_dx_samps_flat = torch.zeros((nB, 2 * nt, dim), dtype=x_obs.dtype, device=x_obs.device)
            for bb in range(nB):
                for dd in range(dim):
                    mu_single = mu_new[bb, :, dd].cpu().numpy()
                    cov_single = Sig_cond[bb].cpu().numpy()
                    sample = np.random.multivariate_normal(mu_single, cov_single)
                    x_dx_samps_flat[bb, :, dd] = torch.from_numpy(sample)

    # Separate x and dx samples
    x_samps[:, :, :] = x_dx_samps_flat[:, :nt, :]
    dx_samps[:, :, :] = x_dx_samps_flat[:, nt:, :]

    return x_samps, dx_samps, t_mat

def cond_cov(t_mat, alpha, l, t_obs, sig2_diag=1e-8):
    
    nt = t_mat.shape[1]
    
    nB, nt = t_mat.shape[0], t_mat.shape[1]
    nt_obs = t_obs.shape[0]

    # Compute necessary covariance matrices and kernel functions
    r_obs_x = calc_r(t_obs, t_mat)
    r_obs_obs = calc_r(t_obs, t_obs)
    Sig_11 = cov_mat2(t_mat, t_mat, alpha, l, sig2_diag)
    
    # Precompute parts of the covariance matrices
    k_obs_x, k_obs_dx = k11(r_obs_x, alpha, l), k12(r_obs_x, alpha, l)
    Sig_21 = torch.cat([k_obs_x, k_obs_dx], dim=2)
    Sig_12 = Sig_21.permute(0, 2, 1)

    Sig_22_sing = k11(r_obs_obs, alpha, l) + torch.eye(nt_obs) * sig2_diag
    Sig_22_inv_sing = torch.linalg.inv(Sig_22_sing)
    Sig_22_inv = Sig_22_inv_sing.repeat(nB, 1, 1)

    # Compute conditional covariance matrix with stability adjustment
    Sig_cond = Sig_11 - torch.bmm(torch.bmm(Sig_12, Sig_22_inv), Sig_21)
    Sig_cond = (Sig_cond + Sig_cond.permute(0, 2, 1))/2
    
    svd_add_idx = torch.sum((torch.linalg.eigvals(Sig_cond).real>=0).T, axis = 0) != Sig_cond.shape[1]
    U, S, Vh = torch.linalg.svd(Sig_cond[svd_add_idx,:,:])
    Sig_cond_add = torch.bmm(torch.bmm(Vh.permute(0, 2, 1), torch.diag_embed(S + 1e-8)), Vh)
    Sig_cond[svd_add_idx,:,:] = (Sig_cond_add + Sig_cond_add.permute(0, 2, 1))/2
    
    return Sig_cond

def beta_pdf(t, a, b):
    # Compute the log of the beta PDF for numerical stability.
    log_pdf = (a - 1) * torch.log(t) + (b - 1) * torch.log(1 - t) \
              - (torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b))
    return torch.exp(log_pdf)

def fit_beta_shape(t, x, lr=1e-2, n_iter=1000):
    # Initialize parameters in log space:
    # log_A: amplitude, log_a and log_b for the beta parameters.
    log_A = torch.nn.Parameter(torch.tensor(0.))
    log_a = torch.nn.Parameter(torch.tensor(0.0))
    log_b = torch.nn.Parameter(torch.tensor(0.0))
#     offset = torch.nn.Parameter(torch.tensor(0.0))
    
    t = t.clamp(min=1e-6, max=1-1e-6)
    
#     optimizer = torch.optim.Adam([offset, log_A, log_a, log_b], lr=lr)
    optimizer = torch.optim.Adam([log_A, log_a, log_b], lr=lr)
    
    for i in range(n_iter):
        optimizer.zero_grad()
        
        # Exponentiate parameters to ensure positivity.
        A = torch.exp(log_A)
        a = torch.exp(log_a)
        b = torch.exp(log_b)
        
#         pred = offset + A * beta_pdf(t, a, b)
        pred = A * beta_pdf(t, a, b)
        
        # Use mean squared error loss.
        loss = torch.mean((pred - x)**2)
        loss.backward()
        optimizer.step()
        
        if i % 5000 == 0:
#             print(f"Iter {i}: offset = {offset.item():.4f}, A = {A.item():.4f}, a = {a.item():.4f}, b = {b.item():.4f}, loss = {loss.item():.6f}")
            print(f"Iter {i}: A = {A.item():.4f}, a = {a.item():.4f}, b = {b.item():.4f}, loss = {loss.item():.6f}")
            
    return torch.exp(log_A).item(), torch.exp(log_a).item(), torch.exp(log_b).item()
#     return offset.item(), torch.exp(log_A).item(), torch.exp(log_a).item(), torch.exp(log_b).item()



def GP_CFM(x_data, model, optimizer, alpha, l,
          nt, batch_size, t_obs, n_epochs, sig2_diag = 0., z1 = None, 
          ImpSamp = False, beta_a = 2., beta_b = 2.,
          storeCheck = True, epoch_check_step = 100, ot = False, antithetic = False,
          grad_clip = 1.0, ema_decay = 0.9999, ema_model = None, useSche = True):
    
    N = x_data.shape[0]
    dim = x_data.shape[2]
    
    if ema_model is None:
        ema_model = copy.deepcopy(model)
        
    if useSche:
        print('gpsche')
        sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lr)
    
    if ot:
        print('gpot')
        ot_sampler = OTPlanSampler(method="exact")
    
    if ImpSamp:
        print('gpis')
        m = torch.distributions.beta.Beta(torch.tensor([beta_a]),
                                          torch.tensor([beta_b])) # put more weight on t = 1
    
    nbatch = int(N/batch_size)
    n_per_epoch = nbatch*batch_size
    shuffled_idx_all = np.zeros((n_epochs, 2, n_per_epoch), int)
    for ep in range(n_epochs):
        shuffled_idx_all[ep,1,:] = np.random.permutation(N)[0:n_per_epoch]
    
    Sig_22_inv_sing = cov_inv_sing(alpha, l, t_obs, sig2_diag=sig2_diag)
    
    
    losses: List[float] = []
    if storeCheck:
        check_pts = []
        check_steps = []
        
    model.train()
    for k in tqdm(range(n_epochs)):
        
        shuffle_idx_tmp = shuffled_idx_all[k,:,:]
        x1_batch_idx = np.reshape(shuffle_idx_tmp[1,:], [nbatch,-1])
        
        for bb in range(nbatch):
            x0 = torch.randn((batch_size,dim))
            x1 = x_data[x1_batch_idx[bb,:],1,:]
            
            if ot:
                x0, x1 = ot_sampler.sample_plan(x0, x1)
            
            x_obs = torch.zeros_like(x_data[x1_batch_idx[bb,:],:,:])
            x_obs[:,0,:] = x0
            x_obs[:,1,:] = x1
            
            if ImpSamp:
                t_mat_tmp = m.sample((batch_size,nt))[:,:,0]
            else:
                t_mat_tmp = torch.rand((batch_size,nt))
            
            try:
                xt_batch, ut_batch, t_mat= samp_x_dx2(t_mat_tmp, alpha, l,
                                                      x_obs, t_obs, sig2_diag=sig2_diag,
                                                      antithetic = antithetic)
                
            except:
                print('pass')
                pass
            
            t = torch.reshape(t_mat, (-1, 1))
            xt = torch.reshape(xt_batch, (-1,dim))
            ut = torch.reshape(ut_batch, (-1,dim))
            
            if z1 is None:
                vt = model(torch.cat([xt, t], dim=-1).to(device))
            else:
                z1_batch = z1[x1_batch_idx[bb,:],:]
                vt = model(torch.cat([xt, z1_batch, t], dim=-1).to(device))
            
            if ImpSamp:
                loss = torch.mean((1/torch.exp(m.log_prob(t))[:,None])*((vt - ut) ** 2))
            else:
                loss = torch.mean((vt - ut) ** 2)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
            optimizer.step()
            if useSche:
                sched.step()
            ema(model, ema_model, ema_decay)
            
            optimizer.zero_grad()
            
            # Logging
            losses.append(loss.item())
            
            if storeCheck:
                if k % epoch_check_step == 0:
                    check_pts.append(deepcopy(model.state_dict()))
                    check_steps.append(k)
            
    if storeCheck:       
        return model, ema_model, losses, check_pts, check_steps
    else:
        return model, ema_model, losses


def cunstruct_nn(x_lt, nn_structure = 'mlp', xcov = None, w = 16):
    dim = x_lt.shape[1]
    if xcov is None:

        if nn_structure == 'mlp':
            model = MLP(dim = dim, out_dim = dim, w = w, time_varying=True).to(device)
        elif nn_structure == 'unet':
            model = UNet(dim = dim, out_dim = dim, w = w, time_varying=True).to(device)
        elif nn_structure == 'cnn':
            model = CNNModel(dim=dim, out_dim=dim, w= w, num_cnn_stacks=2, time_varying=True).to(device)
            
    else:

        dim_cov = xcov.shape[1]
        if nn_structure == 'mlp':
            model = MLP(dim = dim + dim_cov, out_dim = dim, w = w, time_varying=True).to(device)
        elif nn_structure == 'unet':
            model = UNet(dim = dim + dim_cov, out_dim = dim, w = w, time_varying=True).to(device)
        elif nn_structure == 'cnn':
            model = CNNModel(dim=dim + dim_cov, out_dim=dim, w= w, num_cnn_stacks=2, time_varying=True).to(device)
            
    return model

def train_model(x_lt, batch_size = 100, lr = 1e-3, nn_structure = 'mlp', 
                cfm_method = 'icfm', n_epochs = 10000,
                sigma = 0., xcov = None,
                alpha = 1, l = 2, sig2_diag = 0.0,
                ImpSamp = True, beta_a = 3., beta_b = 3.,
                grad_clip = 1.0, ema_decay = 0.9999, useSche = True, w = 16, weight_decay=0.):
    
    N = x_lt.shape[0]
    dim = x_lt.shape[1]

    x_data_lt = torch.zeros(N, 2, dim)
    x_data_lt[:,1,:] = x_lt
    t_obs = torch.tensor([0, 1])
    
    model = cunstruct_nn(x_lt, nn_structure, xcov = xcov, w = w)
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
    if cfm_method == 'icfm':
        model,ema_model,_ = cfm(model, optimizer, x_lt.to(device), z1 = xcov,
                                batch_size = batch_size, nt = 1, sigma = sigma,
                                n_epochs = n_epochs, x0 = None, ot = False,
                                grad_clip = grad_clip, ema_decay = ema_decay, useSche = useSche)
    elif cfm_method == 'otcfm':
        model,ema_model,_ = cfm(model, optimizer, x_lt.to(device), z1 = xcov,
                                batch_size = batch_size, nt = 1, sigma = sigma,
                                n_epochs = n_epochs, x0 = None, ot = True,
                                grad_clip = grad_clip, ema_decay = ema_decay, useSche = useSche)
    elif cfm_method == 'gpicfm':
        model,ema_model, _ = GP_CFM(x_data_lt, model, optimizer,
                                    alpha,l,1, batch_size, t_obs,
                                    n_epochs = n_epochs, sig2_diag = sig2_diag, z1 = xcov,
                                    ImpSamp = ImpSamp, beta_a = beta_a, beta_b = beta_b,
                                    storeCheck = False, ot = False, antithetic = True,
                                    grad_clip = grad_clip, ema_decay = ema_decay, useSche = useSche)
        
    elif cfm_method == 'gpotcfm':
        model,ema_model, _ = GP_CFM(x_data_lt, model, optimizer,
                                    alpha,l,1, batch_size, t_obs,
                                    n_epochs = n_epochs, sig2_diag = sig2_diag, z1 = xcov,
                                    ImpSamp = ImpSamp, beta_a = beta_a, beta_b = beta_b,
                                    storeCheck = False, ot = True, antithetic = True,
                                    grad_clip = grad_clip, ema_decay = ema_decay, useSche = useSche)
    
    return model,ema_model

def train_model2(x_lt, model, ema_model, batch_size = 100, lr = 1e-3, nn_structure = 'mlp', 
                cfm_method = 'icfm', n_epochs = 10000,
                sigma = 0., xcov = None,
                alpha = 1, l = 2, sig2_diag = 0.0,
                ImpSamp = True, beta_a = 3., beta_b = 3.,
                grad_clip = 1.0, ema_decay = 0.9999, useSche = False, weight_decay=0.):
    
    N = x_lt.shape[0]
    dim = x_lt.shape[1]

    x_data_lt = torch.zeros(N, 2, dim)
    x_data_lt[:,1,:] = x_lt
    t_obs = torch.tensor([0, 1])
    
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
    if cfm_method == 'icfm':
        model,ema_model,_ = cfm(model, optimizer, x_lt.to(device), z1 = xcov,
                                batch_size = batch_size, nt = 1, sigma = sigma,
                                n_epochs = n_epochs, x0 = None, ot = False, ema_model = ema_model,
                                useSche = False, grad_clip = grad_clip, ema_decay = ema_decay)
    elif cfm_method == 'otcfm':
        model,ema_model,_ = cfm(model, optimizer, x_lt.to(device), z1 = xcov,
                                batch_size = batch_size, nt = 1, sigma = sigma,
                                n_epochs = n_epochs, x0 = None, ot = True, ema_model = ema_model,
                                useSche = False, grad_clip = grad_clip, ema_decay = ema_decay)
    elif cfm_method == 'gpicfm':
        model,ema_model, _ = GP_CFM(x_data_lt, model, optimizer,
                                    alpha,l,1, batch_size, t_obs,
                                    n_epochs = n_epochs, sig2_diag = sig2_diag, z1 = xcov,
                                    ImpSamp = ImpSamp, beta_a = beta_a, beta_b = beta_b,
                                    storeCheck = False, ot = False, antithetic = True,
                                    grad_clip = grad_clip, ema_decay = ema_decay,
                                    useSche = False,
                                    ema_model = ema_model)
        
    elif cfm_method == 'gpotcfm':
        model,ema_model, _ = GP_CFM(x_data_lt, model, optimizer,
                                    alpha,l,1, batch_size, t_obs,
                                    n_epochs = n_epochs, sig2_diag = sig2_diag, z1 = xcov,
                                    ImpSamp = ImpSamp, beta_a = beta_a, beta_b = beta_b,
                                    storeCheck = False, ot = True, antithetic = True,
                                    grad_clip = grad_clip, ema_decay = ema_decay,
                                    useSche = False,
                                    ema_model = ema_model)
    
    return model,ema_model

class torch_wrapper(torch.nn.Module):
    """Wraps model to torchdyn compatible format."""

    def __init__(self, model, Xcov = None):
        super().__init__()
        self.model = model
        self.Xcov = Xcov

    def forward(self, t, x, *args, **kwargs):
        if self.Xcov is None:
            x_expand = x
        else:
            x_expand = torch.cat([x, self.Xcov], -1)
        return self.model(torch.cat([x_expand, t.repeat(x_expand.shape[0])[:, None]], 1))


def gen_traj(model, dim, n_samp, nt_gen, seed, batch_size, x_start=None, z=None):
    
    traj_list = []
    
    # Determine the model wrapper based on whether z is provided
    if z is None:
        node = NeuralODE(torch_wrapper(model), solver="dopri5",
                         sensitivity="adjoint", atol=1e-4, rtol=1e-4)
    else:
        node = NeuralODE(torch_wrapper(model, z), solver="dopri5",
                         sensitivity="adjoint", atol=1e-4, rtol=1e-4)
    
    # Process samples in batches
    for i in tqdm(range(0, n_samp, batch_size)):
        current_batch_size = min(batch_size, n_samp - i)
        
        # Generate or slice x_start for the current batch
        if x_start is None:
            torch.manual_seed(seed + i)  # Adjust seed for each batch
            x_batch = torch.randn(current_batch_size, dim)
        else:
            x_batch = x_start[i:i + current_batch_size]
        
        # Compute trajectory for the batch
        with torch.no_grad():
            traj_batch = node.trajectory(x_batch, t_span=torch.linspace(0, 1, nt_gen))
        
        traj_list.append(traj_batch)
    
    # Combine trajectories from all batches
    traj = torch.cat(traj_list, dim=1)  # Shape: [nt_gen, n_samp, dim]
    return traj

def gen_samp(model, dim, n_samp, nt_gen, seed, batch_size=None, x_start=None, z=None):
    if batch_size is None:
        batch_size = n_samp
    traj = gen_traj(model, dim, n_samp, nt_gen, seed, batch_size, x_start, z)
    samp = traj[-1, :]  # Extract the final sample
    return samp

def lt_transform_back(log_odds, child_node_matrix, taxa_names, out_names, epsilon=1e-10):
    """
    Transform log-odds back to relative abundances using phylogenetic tree structure.
    Improved version with cleaner logic.
    """
    
    # Initialize the reconstructed matrix
    reconstructed = np.zeros((len(child_node_matrix), 2), dtype=float)
    
    # Numerically stable sigmoid
    def stable_sigmoid(x):
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
    
    reconstructed[:, 0] = stable_sigmoid(log_odds)  # Probability for child 1
    reconstructed[:, 1] = 1 - reconstructed[:, 0]    # Probability for child 2
    
    # Tree processing logic
    if hasattr(child_node_matrix, 'iloc'):
        # pandas DataFrame - original logic works well
        for ii in range(len(child_node_matrix)):
            c1 = child_node_matrix.iloc[ii, 0]
            c2 = child_node_matrix.iloc[ii, 1]
            
            # Find and update children in the matrix
            if c1 in child_node_matrix.index:
                c1_idx = child_node_matrix.index.get_loc(c1)
                reconstructed[c1_idx, :] *= reconstructed[ii, 0]
            
            if c2 in child_node_matrix.index:
                c2_idx = child_node_matrix.index.get_loc(c2)
                reconstructed[c2_idx, :] *= reconstructed[ii, 1]
    else:
        # For numpy arrays - implement proper logic
        n_internal = len(child_node_matrix)
        n_taxa = n_internal + 1
        
        # Build mapping of node names to row indices
        node_to_row = {}
        for i in range(n_internal):
            node_name = f"Node_{n_taxa + i + 1}"
            node_to_row[node_name] = i
        
        # Process the tree
        for ii in range(n_internal):
            c1_str = str(child_node_matrix[ii, 0])
            c2_str = str(child_node_matrix[ii, 1])
            
            # Update children if they're internal nodes
            if c1_str in node_to_row:
                c1_idx = node_to_row[c1_str]
                reconstructed[c1_idx, :] *= reconstructed[ii, 0]
            
            if c2_str in node_to_row:
                c2_idx = node_to_row[c2_str]
                reconstructed[c2_idx, :] *= reconstructed[ii, 1]
    
    # Tip extraction - extract values for leaf nodes
    n_tips = len(log_odds) + 1
    reconstructed_tip = np.zeros(n_tips, dtype=float)
    
    for ii in range(len(child_node_matrix)):
        # Get child node names
        if hasattr(child_node_matrix, 'iloc'):
            c1_str = str(child_node_matrix.iloc[ii, 0])
            c2_str = str(child_node_matrix.iloc[ii, 1])
        else:
            c1_str = str(child_node_matrix[ii, 0])
            c2_str = str(child_node_matrix[ii, 1])
        
        # Extract node numbers and assign to tips
        try:
            c1_num = int(c1_str.replace("Node_", ""))
            c2_num = int(c2_str.replace("Node_", ""))
            
            if c1_num <= n_tips:
                reconstructed_tip[c1_num - 1] = reconstructed[ii, 0]
            if c2_num <= n_tips:
                reconstructed_tip[c2_num - 1] = reconstructed[ii, 1]
        except:
            pass
    
    # Reorder to match output taxa order
    taxa_names_flat = np.array(taxa_names).flatten()
    out_names_flat = np.array(out_names).flatten()
    
    # Match R's order() and match() functions - this is what the original does
    sorted_indices = np.argsort(taxa_names_flat)
    reorder_indices = np.searchsorted(taxa_names_flat[sorted_indices], out_names_flat)
    final_indices = sorted_indices[reorder_indices]
    
    # Extract final results
    out_ra = reconstructed_tip[final_indices]
    
    return out_ra


def reconstruct_lt_all(lt_samp, child_node_matrix, tree_taxa_pd, outTaxa_pd):
    """
    Reconstruct relative abundances for all samples from log-odds transformation.
    Maintains the same interface as the original function.
    """
    
    # Convert torch tensor to numpy if needed
    if hasattr(lt_samp, 'cpu'):
        lt_data = lt_samp.T.cpu().detach().numpy()  # Note the transpose
    else:
        lt_data = lt_samp.T  # Note the transpose
    
    # Get values from pandas objects if needed
    if hasattr(tree_taxa_pd, 'values'):
        taxa_names = tree_taxa_pd.values
    else:
        taxa_names = tree_taxa_pd
        
    if hasattr(outTaxa_pd, 'values'):
        out_names = outTaxa_pd.values
    else:
        out_names = outTaxa_pd
    
    # Process each column (sample)
    results = []
    for col in range(lt_data.shape[1]):
        result = lt_transform_back(
            lt_data[:, col],
            child_node_matrix=child_node_matrix,
            taxa_names=taxa_names,
            out_names=out_names,
        )
        results.append(result)
    
    # Stack and transpose back
    reconstructed_ra_all = np.stack(results, axis=1)
    return reconstructed_ra_all.T

def bray_curtis(generated_data, test_data, train_data = None):
    if train_data is None:
        aggregated_data = np.concatenate([generated_data, test_data], axis=0)
        group_labels = ((['generated'] * generated_data.shape[0])+ 
                        (['test'] * test_data.shape[0]))
    else:
        aggregated_data = np.concatenate([generated_data, test_data, train_data], axis=0)
        group_labels = ((['generated'] * generated_data.shape[0])+ 
                        (['test'] * test_data.shape[0]) +
                       (['train'] * train_data.shape[0]))
    
    sample_ids = [f"S{i+1}" for i in range(aggregated_data.shape[0])]
    metadata = pd.DataFrame({'Sample': sample_ids, 'Group': group_labels}).set_index('Sample')
    bray_curtis_dm = beta_diversity("braycurtis",
                                aggregated_data,
                                sample_ids)
    
    bray_df = pd.DataFrame(bray_curtis_dm.data, index=sample_ids, columns=sample_ids)
    
    return aggregated_data, metadata, bray_curtis_dm, bray_df

def mean_bc(bray_df, metadata):
    group1_samples = metadata[metadata["Group"] == "generated"].index
    group2_samples = metadata[metadata["Group"] == "test"].index
    between_group_distances = bray_df.loc[group1_samples, group2_samples].values.flatten()
    
    return np.mean(between_group_distances)


def mean_sd_bc(bray_df, metadata):
    group1_samples = metadata[metadata["Group"] == "generated"].index
    group2_samples = metadata[metadata["Group"] == "test"].index
    between_group_distances = bray_df.loc[group1_samples, group2_samples].values.flatten()
    
    # Calculate mean and standard error
    mean_distance = np.mean(between_group_distances)
    std_distance = np.std(between_group_distances, ddof=1)  # Sample standard deviation
    n = len(between_group_distances)
    standard_error = std_distance / np.sqrt(n)
    
    return {
        'mean': mean_distance,
        'std': std_distance,
        'se': standard_error,
        'n': n,
        'distances': between_group_distances  # Optional: return raw distances for further analysis
    }



def stacked_bar_one(reconstructed_ra_all, plot_name, taxa_name, n_sub=None, width=12, height=6,
                   idx_taxa = None):
    # ——— Build global taxa → color map ———
    all_taxa = sorted(taxa_name['V3'].unique())
    n_taxa = len(all_taxa)
    base_cmap = plt.get_cmap('tab20') if n_taxa <= 20 else plt.get_cmap('tab20b')
    taxa2color = {
        t: base_cmap(i / (n_taxa - 1))
        for i, t in enumerate(all_taxa)
    }

    # ——— Prepare data ———
    if n_sub is None:
        n_sub = reconstructed_ra_all.shape[1]
    df = pd.DataFrame(reconstructed_ra_all.T, columns=taxa_name['V3'])
    grouped = df.groupby(axis=1, level=0).sum()
    c_names = grouped.columns
    X = grouped.values[:n_sub, :]

    # sort samples by largest component
    idx_samp = np.argsort(-X.max(axis=1))
    X = X[idx_samp]

    # sort taxa by mean abundance
    if idx_taxa is None:
        means = X.mean(axis=0)
        idx_taxa = np.argsort(-means)
    X = X[:, idx_taxa]
    c_names = c_names[idx_taxa]

    # pull colors for the bars in abundance‐order
    bar_colors = [taxa2color[t] for t in c_names]

    # ——— Plot ———
    fig, ax = plt.subplots(figsize=(width, height))
    bottom = np.zeros(n_sub)
    for i, t in enumerate(c_names):
        ax.bar(np.arange(n_sub), X[:, i], bottom=bottom, width=1.0, color=bar_colors[i])
        bottom += X[:, i]

    # ——— Static alphabetical colorbar ———
    full_colors = [taxa2color[t] for t in all_taxa]
    cmap_full = mcolors.ListedColormap(full_colors)
    bounds = np.arange(n_taxa+1) - 0.5
    norm = mcolors.BoundaryNorm(bounds, cmap_full.N)
    sm = cm.ScalarMappable(cmap=cmap_full, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ticks=np.arange(n_taxa), ax=ax)
    cbar.ax.set_yticklabels(all_taxa)
    cbar.set_label("Taxa (alphabetical)")

    ax.set_xlabel("Sample Index (sorted)")
    ax.set_ylabel("Proportion")
    ax.set_title(plot_name)
    plt.tight_layout()
    
    return idx_taxa

def stacked_bar_one2(ax, reconstructed_ra_all, taxa_name, plot_name, n_sub=None,
                    idx_taxa = None):
    # ——— Build global taxa → color map ———
    all_taxa = sorted(taxa_name['V3'].unique())
    n_taxa = len(all_taxa)
    base_cmap = plt.get_cmap('tab20') if n_taxa <= 20 else plt.get_cmap('tab20b')
    taxa2color = {
        t: base_cmap(i / (n_taxa - 1))
        for i, t in enumerate(all_taxa)
    }

    # ——— Prepare data ———
    if n_sub is None:
        n_sub = reconstructed_ra_all.shape[1]
    df = pd.DataFrame(reconstructed_ra_all.T, columns=taxa_name['V3'])
    grouped = df.groupby(axis=1, level=0).sum()
    c_names = grouped.columns
    X = grouped.values[:n_sub, :]

    # sort samples
    idx_samp = np.argsort(-X.max(axis=1))
    X = X[idx_samp]

    # sort taxa by mean abundance
    if idx_taxa is None:
        means = X.mean(axis=0)
        idx_taxa = np.argsort(-means)
    X = X[:, idx_taxa]
    c_names = c_names[idx_taxa]
    

    # pull colors for bars
    bar_colors = [taxa2color[t] for t in c_names]

    # ——— Plot ———
    bottom = np.zeros(n_sub)
    for i, t in enumerate(c_names):
        ax.bar(np.arange(n_sub), X[:, i], bottom=bottom, width=1.0, color=bar_colors[i])
        bottom += X[:, i]

    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Proportion")
    ax.set_title(plot_name)
    
    return idx_taxa


def shannon_entropy(x, tol=0.):
    mask = x > tol
    result = np.zeros_like(x)
    result[mask] = x[mask] * np.log(x[mask])
    return -np.sum(result, axis=-1)

def get_sparsity(x, tol=0.):
    return np.sum(x <= tol, axis=-1) / x.shape[1]


def sparsity_shannon(x_raw_np, gen_samp, SEED = 256, TOL = 1e-4):

    print("Sparsity")
    print(pd.DataFrame(
        [describe(get_sparsity(x_raw_np, TOL)),
         describe(get_sparsity(gen_samp, TOL)),], 
        index=['Original', 'generated']))

    print("Shannon Entropy")
    print(pd.DataFrame(
        [describe(shannon_entropy(x_raw_np)),
         describe(shannon_entropy(gen_samp)),], 
        index=['Original', 'generated']))
    
    
def umap_ori(ax, trans_ref, generated, title):
    
    gen_embedding = trans_ref.transform(generated)
    ref_embedding = trans_ref.embedding_
    
    umap_df_gen = pd.DataFrame(gen_embedding, columns=["UMAP1", "UMAP2"])
    umap_df_gen["Group"] = "generated"  # Add Group Labels
    
    umap_df_ref = pd.DataFrame(ref_embedding, columns=["UMAP1", "UMAP2"])
    umap_df_ref["Group"] = "test"  # Add Group Labels

    # Combine for proper legend handling
    umap_df = pd.concat([umap_df_gen, umap_df_ref])

    # Plot with hue for legend
    sns.scatterplot(ax=ax, data=umap_df, x="UMAP1", y="UMAP2", hue="Group",
                    s=60, edgecolor="black", alpha=0.6)

    # Labels and Title
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(title)
    ax.legend(title="Group")  # Ensure legend is shown
    
def umap_ori2(ax, trans_ref, generated, traindata, title):
    
    train_embedding = trans_ref.transform(traindata)
    gen_embedding = trans_ref.transform(generated)
    ref_embedding = trans_ref.embedding_
    
    umap_df_train = pd.DataFrame(train_embedding, columns=["UMAP1", "UMAP2"])
    umap_df_train["Group"] = "train"  # Add Group Labels
    
    umap_df_gen = pd.DataFrame(gen_embedding, columns=["UMAP1", "UMAP2"])
    umap_df_gen["Group"] = "generated"  # Add Group Labels
    
    umap_df_ref = pd.DataFrame(ref_embedding, columns=["UMAP1", "UMAP2"])
    umap_df_ref["Group"] = "test"  # Add Group Labels

    # Combine for proper legend handling
    umap_df = pd.concat([umap_df_gen, umap_df_ref, umap_df_train])

    # Plot with hue for legend
    sns.scatterplot(ax=ax, data=umap_df, x="UMAP1", y="UMAP2", hue="Group",
                    s=60, edgecolor="black", alpha=0.4)

    # Labels and Title
    ax.set_xlabel("UMAP1")
    ax.set_ylabel("UMAP2")
    ax.set_title(title)
    ax.legend(title="Group")  # Ensure legend is shown
    
def create_reference_pcoa(reference_datasets, dataset_names):
    # Combine all datasets to create reference space
    all_data = []
    all_metadata = []
    dataset_indices = {}
    current_idx = 0
    
    for i, (generated_data, test_data, train_data) in enumerate(reference_datasets):
        # Get data for this dataset
        agg_data, metadata, _, _ = bray_curtis(generated_data, test_data, train_data)
        
        # Add dataset identifier to metadata
        metadata['Dataset'] = dataset_names[i]
        metadata['Original_Index'] = range(len(metadata))
        
        # Store indices for this dataset
        dataset_indices[dataset_names[i]] = (current_idx, current_idx + len(agg_data))
        current_idx += len(agg_data)
        
        all_data.append(agg_data)
        all_metadata.append(metadata)
    
    # Combine all data
    combined_data = np.concatenate(all_data, axis=0)
    combined_metadata = pd.concat(all_metadata, ignore_index=True)
    
    # Create sample IDs for combined dataset
    combined_sample_ids = [f"Combined_S{i+1}" for i in range(combined_data.shape[0])]
    combined_metadata.index = combined_sample_ids
    
    # Calculate Bray-Curtis distance for combined dataset
    combined_bc_dm = beta_diversity("braycurtis", combined_data, combined_sample_ids)
    
    # Perform PCoA on combined dataset
    reference_pcoa_results = pcoa(combined_bc_dm)
    
    # Extract coordinates for each original dataset
    reference_coords = {}
    for dataset_name, (start_idx, end_idx) in dataset_indices.items():
        coords_df = pd.DataFrame(
            reference_pcoa_results.samples[['PC1', 'PC2']].iloc[start_idx:end_idx],
            index=combined_sample_ids[start_idx:end_idx]
        )
        # Add back the original metadata
        original_metadata = combined_metadata.iloc[start_idx:end_idx][['Group']].copy()
        original_metadata.index = coords_df.index
        coords_df = coords_df.merge(original_metadata, left_index=True, right_index=True)
        reference_coords[dataset_name] = coords_df
    
    return reference_pcoa_results, reference_coords, dataset_indices
    
def pcoa_bc_consistent(ax, pcoa_results, coords_df, title, global_coords_df=None, dataset_idx=None, reverse=False):
    """
    Plot PCoA using consistent reference space coordinates.
    
    Parameters:
    -----------
    ax : matplotlib axis
    pcoa_results : PCoA results from reference space
    coords_df : DataFrame with PC1, PC2, and Group columns for this specific dataset
    title : str
    global_coords_df : DataFrame with PC1, PC2 columns for ALL datasets combined (for consistent limits)
    dataset_idx : tuple (start, end) indices if highlighting specific dataset
    reverse : bool, if True reverses plotting order (test plotted on top)
    """
    
    # Define consistent color mapping: generated=1st color, test=2nd color
    colors = sns.color_palette("tab10")  # Default seaborn palette
    custom_palette = {'generated': colors[0], 'test': colors[1]}
    
    if reverse:
        # Plot in reverse order: test first (bottom layer), then generated (top layer)
        # This makes generated points appear on top when overlapping
        plot_order = ['test', 'generated']
    else:
        # Default order: generated first (bottom layer), then test (top layer)  
        plot_order = ['generated', 'test']
    
    # Plot each group separately to control layering order
    for group in plot_order:
        group_data = coords_df[coords_df['Group'] == group]
        if not group_data.empty:
            sns.scatterplot(ax=ax, data=group_data, x="PC1", y="PC2", 
                           color=custom_palette[group], label=group,
                           s=30, edgecolor="black", alpha=0.7)
    
    ax.set_xlabel(f"PC1 ({pcoa_results.proportion_explained[0]:.2%} variance explained)")
    ax.set_ylabel(f"PC2 ({pcoa_results.proportion_explained[1]:.2%} variance explained)")
    ax.set_title(title)
    ax.legend(title="Group")
    
    # Set consistent axis limits based on GLOBAL coordinate range (all datasets combined)
    if global_coords_df is not None:
        # Use global coordinates for consistent limits across all plots
        all_pc1 = global_coords_df['PC1']
        all_pc2 = global_coords_df['PC2']
    else:
        # Fallback to current dataset if global coordinates not provided
        all_pc1 = coords_df['PC1']
        all_pc2 = coords_df['PC2']
    
    pc1_range = all_pc1.max() - all_pc1.min()
    pc2_range = all_pc2.max() - all_pc2.min()
    
    ax.set_xlim(all_pc1.min() - 0.1 * pc1_range, all_pc1.max() + 0.1 * pc1_range)
    ax.set_ylim(all_pc2.min() - 0.1 * pc2_range, all_pc2.max() + 0.1 * pc2_range)
    
def clr_transform_matrix(X, eps=1e-15):
    # Avoid log(0) by adding a small eps
    X_safe = X + eps
    # Geometric mean per row (shape: (n,))
    gm = np.exp(np.mean(np.log(X_safe), axis=1))
    # Broadcast divide, then take log
    # gm[:, None] makes gm shape (n, 1), so dividing X_safe by gm is row-wise
    return np.log(X_safe / gm[:, None])

def aitchison_distance_between_sets_pairwise(X, Y, eps=1e-15):
    # 1) CLR-transform each row
    U = clr_transform_matrix(X, eps=eps)  # shape (nX, D)
    V = clr_transform_matrix(Y, eps=eps)  # shape (nY, D)
    
    # 2) Compute pairwise Euclidean distances between CLR-transformed rows
    dist_matrix = cdist(U, V, metric="euclidean")  # shape (nX, nY)
    
    # 3) Return the mean of all pairwise distances
    return dist_matrix.mean()


def pairwise_ad_mean_sd(X, Y, eps=1e-15):
    # 1) CLR-transform each row
    U = clr_transform_matrix(X, eps=eps)  # shape (nX, D)
    V = clr_transform_matrix(Y, eps=eps)  # shape (nY, D)
    
    # 2) Compute pairwise Euclidean distances between CLR-transformed rows
    dist_matrix = cdist(U, V, metric="euclidean")  # shape (nX, nY)
    
    # 3) Flatten to get all pairwise distances
    all_distances = dist_matrix.flatten()
    
    # 4) Calculate mean and standard error
    mean_distance = np.mean(all_distances)
    std_distance = np.std(all_distances, ddof=1)  # Sample standard deviation
    n = len(all_distances)
    standard_error = std_distance / np.sqrt(n)
    
    return {
        'mean': mean_distance,
        'std': std_distance,
        'se': standard_error,
        'n': n,
        'distances': all_distances  # Optional: return raw distances for further analysis
    }



def w2_distance_between_sets(X, Y):
    N, D = X.shape
    M, _ = Y.shape
    a = np.ones(N) / N
    b = np.ones(M) / M
    # cost matrix = pairwise squared Euclidean distances
    C = ot.dist(X, Y, metric='sqeuclidean')
    w2_sq = ot.emd2(a, b, C)
    return np.sqrt(w2_sq)