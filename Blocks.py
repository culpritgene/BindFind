import torch
from torch import nn
from torch.nn import functional as F
from ModernHopfield import StatePattern, Hopfield
import math

class qkv_transform(nn.Conv1d):
    """Conv1d for qkv_transform"""

class sConv1d(nn.Conv1d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(sConv1d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv1d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class AxialAttention1d(nn.Module):
    def __init__(self, in_planes, out_planes, length=None, kernel_size=56, groups=8, bias=False, proximal=True,
                 reshaped=False):
        assert (in_planes % groups == 0) and (out_planes % groups == 0)
        if not proximal:
            assert length % kernel_size == 0
        super(AxialAttention1d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.Lenght = length
        self.groups = groups
        self.group_planes = out_planes // groups
        self.kernel_size = kernel_size
        self.bias = bias
        self.proximal = proximal
        self.reshaped = reshaped

        # padding of the sequence (equivalent to roll with zeros at the end of the sequence)

        # reshaping into 2D map / slicing of the original sequence into segments of length = self.kernel_size
        # ... functional realization in forward

        ### Multi-head self attention
        # projection from d_input into d_hidden = out_planes*2
        self.qkv_transform = qkv_transform(in_planes, out_planes * 2, kernel_size=1, stride=1,
                                           padding=0, bias=False)
        self.bn_qkv = nn.BatchNorm1d(out_planes * 2)
        self.bn_similarity = nn.BatchNorm2d(groups * 3)
        self.bn_output = nn.BatchNorm1d(out_planes * 2)

        # Position embedding
        if self.proximal:
            self.positional_dim = self.kernel_size
        else:
            self.positional_dim = (self.Lenght // self.kernel_size)
        self.relative = nn.Parameter(torch.randn(self.group_planes * 2, 2 * self.positional_dim - 1),
                                     requires_grad=True)
        query_index = torch.arange(self.positional_dim).unsqueeze(0)
        key_index = torch.arange(self.positional_dim).unsqueeze(1)
        relative_index = key_index - query_index + self.positional_dim - 1
        self.register_buffer('flatten_index', relative_index.view(-1))

        self.reset_parameters()

    def forward(self, x):
        if not self.reshaped:
            L, N, C = x.shape
            x = x.permute(1, 2, 0)
            x = x.view(N, C, L // self.kernel_size, self.kernel_size)
        if self.proximal:
            x = x.permute(0, 2, 1, 3)
        else:
            x = x.permute(0, 3, 1, 2)  # N, W, C, H
        N, W, C, H = x.shape
        x = x.contiguous().view(N * W, C, H)

        # Transformations
        qkv = self.bn_qkv(self.qkv_transform(x))
        q, k, v = torch.split(qkv.reshape(N * W, self.groups, self.group_planes * 2, H),
                              [self.group_planes // 2, self.group_planes // 2, self.group_planes], dim=2)

        # Calculate position embedding
        all_embeddings = torch.index_select(self.relative, 1, self.flatten_index).view(self.group_planes * 2,
                                                                                       self.positional_dim,
                                                                                       self.positional_dim)
        q_embedding, k_embedding, v_embedding = torch.split(all_embeddings,
                                                            [self.group_planes // 2, self.group_planes // 2,
                                                             self.group_planes], dim=0)

        qr = torch.einsum('bgci,cij->bgij', q, q_embedding)
        kr = torch.einsum('bgci,cij->bgij', k, k_embedding).transpose(2, 3)
        qk = torch.einsum('bgci,bgcj->bgij', q, k)
        stacked_similarity = torch.cat([qk, qr, kr], dim=1)
        stacked_similarity = self.bn_similarity(stacked_similarity).view(N * W, 3, self.groups, H, H).sum(dim=1)
        # stacked_similarity = self.bn_qr(qr) + self.bn_kr(kr) + self.bn_qk(qk)
        # (N, groups, H, H, W)
        similarity = F.softmax(stacked_similarity, dim=3)

        sv = torch.einsum('bgij,bgcj->bgci', similarity, v)
        sve = torch.einsum('bgij,cij->bgci', similarity, v_embedding)
        stacked_output = torch.cat([sv, sve], dim=-1).view(N * W, self.out_planes * 2, H)
        output = self.bn_output(stacked_output).view(N, W, self.out_planes, 2, H).sum(dim=-2)

        if self.proximal:
            output = output.permute(0, 2, 1, 3)
        else:
            output = output.permute(0, 2, 3, 1)

        if self.reshaped:
            return output
        return output.contiguous().view(N, self.out_planes, L).permute(2, 0, 1)

    def reset_parameters(self):
        self.qkv_transform.weight.data.normal_(0, math.sqrt(1. / self.in_planes))
        # nn.init.uniform_(self.relative, -0.1, 0.1)
        nn.init.normal_(self.relative, 0., math.sqrt(1. / self.group_planes))

class AxialBlock1d(nn.Module):

    def __init__(self, C_in, C_hid, C_out, length, kernel_size=56, groups=1,
                 activation='relu', norm_layer=None, marginal_att=False, dropout=0.0, weight_standard=False):
        assert length % kernel_size == 0
        super(AxialBlock1d, self).__init__()
        Conv1d = sConv1d if weight_standard else nn.Conv1d
        if norm_layer is None:
            norm_layer = nn.BatchNorm1d
        self.kernel_size = kernel_size
        self.C_hid = C_hid
        self.marginal_att = marginal_att
        self.conv_down = Conv1d(C_in, C_hid, kernel_size=1, groups=groups, bias=False)
        self.bn1 = norm_layer(C_hid)
        C_hid0 = C_hid * 2 if marginal_att else C_hid
        self.proxy_block1 = AxialAttention1d(C_hid0, C_hid, groups=groups, kernel_size=kernel_size,
                                            length=length, proximal=True, reshaped=True)
        self.tele_block = AxialAttention1d(C_hid, C_hid, groups=groups, kernel_size=kernel_size,
                                           length=length, proximal=False, reshaped=True)
        self.proxy_block2 = AxialAttention1d(C_hid, C_hid, groups=groups, kernel_size=kernel_size,
                                            length=length, proximal=True, reshaped=True)

        self.conv_up = Conv1d(C_hid, C_out, kernel_size=1, groups=groups, bias=False)
        self.bn2 = norm_layer(C_out)
        self.act = choose_activation(activation)
        self.resweight = nn.Parameter(torch.Tensor([0]))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        identity = x

        out = self.conv_down(x)
        out = self.bn1(out)
        out = self.dropout(self.act(out))

        N, C, L = out.shape
        out = out.view(N, C, L // self.kernel_size, self.kernel_size)

        ### calculate marginals
        if self.marginal_att:
            proxy_marginal = out.mean(dim=3)
            tele_marginal = out.mean(dim=2)
            marg_att = torch.einsum('bci,bcj->bcij', proxy_marginal, tele_marginal)
            out = torch.cat([out, marg_att], dim=1)

        out = self.proxy_block1(out)
        out = self.dropout(self.tele_block(out))
        out = self.proxy_block2(out)
        out = self.act(out)

        out = out.contiguous().view(N, self.C_hid, L)  # .permute(2,0,1)

        out = self.conv_up(out)
        out = self.bn2(out)

        out = identity + out*self.resweight
        out = self.dropout(self.act(out))

        return out

class ResNetBlock1d(nn.Module):
    def __init__(self, Cin: int, Cout: int, kernel_size: int, padding=1, stride=1,
                 dropout_rate=0, batchnorm1d=(True, True), activation='relu',
                 shortcut_type='A', **kwargs):
        super(ResNetBlock1d, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=Cin, out_channels=Cout, kernel_size=kernel_size,
                               stride=stride, padding=padding, **kwargs)
        self.conv2 = nn.Conv1d(in_channels=Cout, out_channels=Cout, kernel_size=kernel_size,
                               stride=1, padding=padding, **kwargs)
        self.act1 = choose_activation(activation) if isinstance(activation, str) else activation
        self.act2 = choose_activation(activation) if isinstance(activation, str) else activation

        self.bn1 = nn.BatchNorm1d(num_features=Cout, momentum=0.1, affine=True) if batchnorm1d[0] else lambda x: x
        self.bn2 = nn.BatchNorm1d(num_features=Cout, momentum=0.1, affine=True) if batchnorm1d[1] else lambda x: x

        if shortcut_type == 'A':
            self.shortcut = lambda x: F.pad(x[..., ::stride, ::stride],
                                            pad=[0, 0, 0, 0, (Cout - Cin) // 2, (Cout - Cin) // 2], mode="constant",
                                            value=0)
        elif shortcut_type == 'B':
            self.shortcut = nn.Conv1d(in_channels=Cin, out_channels=Cout, kernel_size=(1, 1), padding=0)
        self.dropout1 = None
        self.dropout2 = None
        if dropout_rate != 0:
            self.dropout1 = nn.Dropout(p=dropout_rate)
            self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, x0):
        x = self.bn1(self.conv1(x0))
        if self.dropout1:
            x = self.dropout1(x)
        x = self.act1(x)
        x = self.bn2(self.conv2(x))
        if self.dropout2:
            x = self.dropout2(x)
        x += self.shortcut(x0)
        x = self.act2(x)
        return x

class RZTXEncoderLayer(nn.Module):
    r"""RZTXEncoderLayer is made up of self-attn and feedforward network with
    residual weights for faster convergece.
    This encoder layer is based on the paper "ReZero is All You Need:
    Fast Convergence at Large Depth".
    Thomas Bachlechner∗, Bodhisattwa Prasad Majumder∗, Huanru Henry Mao∗,
    Garrison W. Cottrell, Julian McAuley. 2020.
    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of intermediate layer, relu or gelu (default=relu).
        use_res_init: Use residual initialization
    Examples::
        >>> encoder_layer = RZTXEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu'):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.resweight = nn.Parameter(torch.Tensor([0]))

        if activation == "relu":
            self.activation = F.relu
        elif activation == "gelu":
            self.activation = F.gelu

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # type: (Tensor, Optional[Tensor], Optional[Tensor]) -> Tensor
        r"""Pass the input through the encoder layer.
        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
        Shape:
            see the docs in PyTroch Transformer class.
        """
        # Self attention layer
        src2 = src
        src2 = self.self_attn(src2, src2, src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)

        src2 = src2[0] # no attention weights
        src2 = src2 * self.resweight
        src = src + self.dropout1(src2)

        # Pointiwse FF Layer
        src2 = src
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src2 = src2 * self.resweight
        src = src + self.dropout2(src2)
        return src

class HopfieldConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, hidden_length, num_memories=100, maxpool_kernel=2,
                 stride=2, groups=2, beta=1.0, activation=lambda x: x, norm=False, dropout=0.0,
                 weight_standard=False):
        super(HopfieldConv, self).__init__()
        Conv1d = sConv1d if weight_standard else nn.Conv1d

        Norm = nn.BatchNorm1d if norm else lambda x: x

        self.ChannelPool = Conv1d(in_channels, hidden_channels, kernel_size=1, groups=groups)
        self.Maxpool = nn.MaxPool1d(maxpool_kernel, stride)
        self.patterns = StatePattern(hidden_length, quantity=num_memories, batch_first=True)
        self.Unpool = nn.ConvTranspose1d(hidden_channels, in_channels, kernel_size=maxpool_kernel,
                                         stride=stride, groups=groups)
        self.act = choose_activation(activation)
        self.norm1 = Norm(hidden_channels)
        self.norm2 = Norm(in_channels)
        self.RZ = nn.Parameter(torch.Tensor([0.1]))
        self.beta = beta
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        identity = x
        x = F.dropout(self.norm1(self.act(self.ChannelPool(x))))
        x = self.Maxpool(x)
        x = x
        _, xz, _ = self.patterns(x)
        attn = F.softmax(torch.bmm(self.beta * x, xz.transpose(2, 1)), dim=-1)
        x = torch.bmm(attn, xz)
        x = F.dropout(self.norm2(self.act(self.Unpool(x))))
        x = identity + self.RZ * x
        return x

class HopfieldConv2(nn.Module):
    def __init__(self, in_channels, hidden_channels, hidden_length, num_memories=100, maxpool_kernel=2,
                 stride=2, groups=2, beta=1.0, activation=lambda x: x, norm=None, dropout=0.0,
                 weight_standard=False):
        super(HopfieldConv2, self).__init__()
        Conv1d = sConv1d if weight_standard else nn.Conv1d

        if norm == 'BN':
            Norm = nn.BatchNorm1d
        elif norm == 'GN':
            Norm = lambda x: nn.GroupNorm(num_groups=groups, num_channels=x)
        elif not norm:
            print('Initializing HopfieldConv without intermediate Norm layers!')
            Norm = lambda x: lambda x: x
        else:
            raise TypeError(f'{norm} is not supported as a valid Norm layer option')

        self.norm0 = Norm(in_channels)
        self.ChannelPool = Conv1d(in_channels, hidden_channels, kernel_size=1, groups=groups)
        self.Maxpool = nn.MaxPool1d(maxpool_kernel, stride)
        self.patterns = StatePattern(hidden_length, quantity=num_memories, batch_first=True)
        self.Unpool = nn.ConvTranspose1d(hidden_channels, in_channels, kernel_size=maxpool_kernel,
                                         stride=stride, groups=groups)
        self.act = choose_activation(activation)
        self.norm1 = Norm(hidden_channels)
        self.norm2 = Norm(in_channels)
        self.RZ = nn.Parameter(torch.Tensor([0.1]))
        self.beta = beta
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        identity = x
        x = self.norm0(x)
        x = F.dropout(self.norm1(self.act(self.ChannelPool(x))))
        x = self.Maxpool(x)
        x = x
        _, xz, _ = self.patterns(x)
        attn = F.softmax(torch.bmm(self.beta * x, xz.transpose(2, 1)), dim=-1)
        x = torch.bmm(attn, xz)
        x = F.dropout(self.norm2(self.act(self.Unpool(x))))
        x = identity + self.RZ * x
        return x

class HopfieldConvDecoder(nn.Module):
    def __init__(self, in_channels1, in_channels2, hidden_channels, hidden_length, maxpool_kernel=2,
                 stride=2, groups=2, beta=1.0, activation=lambda x: x, norm=None, dropout=0.0,
                 weight_standard=False):
        super(HopfieldConvDecoder, self).__init__()
        Conv1d = sConv1d if weight_standard else nn.Conv1d

        if norm == 'BN':
            Norm = nn.BatchNorm1d
        elif norm == 'GN':
            Norm = lambda x: nn.GroupNorm(num_groups=groups, num_channels=x)
        elif not norm:
            print('Initializing HopfieldConv without intermediate Norm layers!')
            Norm = lambda x: lambda x: x
        else:
            raise TypeError(f'{norm} is not supported as a valid Norm layer option')

        self.norm1 = Norm(in_channels1)
        self.norm2 = Norm(in_channels2)
        self.ChannelPool1 = Conv1d(in_channels1, hidden_channels, kernel_size=1, groups=groups)
        self.ChannelPool2 = Conv1d(in_channels2, hidden_channels, kernel_size=1, groups=groups)
        self.Maxpool = nn.MaxPool1d(maxpool_kernel, stride)

        self.act = choose_activation(activation)
        self.norm3 = Norm(hidden_channels)
        self.norm4 = Norm(hidden_channels)
        self.RZ = nn.Parameter(torch.Tensor([0.1]))
        self.beta = beta
        self.dropout = nn.Dropout(dropout)

    def forward(self, x1, x2):
        x1 = self.norm1(x1)
        x2 = self.norm2(x2)

        x1 = F.dropout(self.norm3(self.act(self.ChannelPool1(x1))))
        x2 = F.dropout(self.norm4(self.act(self.ChannelPool2(x2))))
        x1 = self.Maxpool(x1)
        x2 = self.Maxpool(x2)

        dotp = torch.bmm(self.beta * x1, x2.transpose(2, 1))
        attn1 = F.softmax(dotp, dim=-1)
        x2 = torch.bmm(attn1, x2)

        attn2 = F.softmax(dotp, dim=0)
        x1 = torch.bmm(attn2, x1)

        return x1, x2


class ConvTrunk(nn.Module):
    def __init__(self, in_channels, hid_channels, groups=4, dropout=0.1, activation='relu', weight_standard=False):
        super(ConvTrunk, self).__init__()
        Conv1d = sConv1d if weight_standard else nn.Conv1d
        self.conv1 = Conv1d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = Conv1d(16, hid_channels,  kernel_size=3, dilation=1, padding=1, groups=2)
        self.BNorm1 = nn.BatchNorm1d(hid_channels)
        self.conv3 = Conv1d(hid_channels, hid_channels, kernel_size=1, dilation=1, padding=0, groups=groups)
        self.GroupNorm1 = nn.GroupNorm(num_groups=groups, num_channels=hid_channels)
        self.conv4 = Conv1d(hid_channels, hid_channels, kernel_size=3, dilation=1, padding=1, groups=groups)
        self.GroupNorm2 = nn.GroupNorm(num_groups=groups, num_channels=hid_channels)
        self.Pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # 200 -> 100

        self.act = choose_activation(activation)
        self.dropout = nn.Dropout(dropout)
        self.RZ = nn.ParameterList([nn.Parameter(torch.Tensor([0.1])) for _ in range(3)])

    def forward(self, x):
        x = x.transpose(2, 1)
        N, C, L = x.shape
        x = F.relu(self.conv1(x))
        x1 = F.dropout(self.BNorm1(self.act(self.conv2(x))), p=0.2)
        x2 = F.dropout(self.GroupNorm1(self.act(self.conv3(x1))), p=0.3)
        x = x2 + self.RZ[0] * x1
        x = F.dropout(self.GroupNorm2(self.act(self.conv4(x))), p=0.3)
        x = x + self.RZ[1] * x2
        xf = self.Pool1(x)
        return xf, x

class SeqEncoder(nn.Module):
    def __init__(self, input_dim, main_hid, gr, second_hid=256):
        super(SeqEncoder, self).__init__()
        self.ConvTrunk = ConvTrunk(in_channels=input_dim, hid_channels=main_hid, groups=gr, activation='relu')
        self.SelfAttn1 = RZTXEncoderLayer(main_hid, 4, dim_feedforward=second_hid, dropout=0.1)
        self.SelfAttn2 = RZTXEncoderLayer(main_hid, 4, dim_feedforward=second_hid, dropout=0.1)

    def forward(self, x):
        x, x01 = self.ConvTrunk(x)
        x = x.permute(2,0,1)
        x = self.SelfAttn1(x)
        x = self.SelfAttn2(x)
        return x

class AxialNet(nn.Module):
    def __init__(self, in_channels, hid_channels, length, kernel_sizes=(20, 10), out_channels=None,
                 groups=4, dropout=0.1, activation='relu', weight_standard=False):
        super(AxialNet, self).__init__()
        if not out_channels:
            out_channels = in_channels
        self.axialblocks = torch.nn.ModuleList()

        for ks in kernel_sizes:
            block = AxialBlock1d(C_in=in_channels, C_hid=hid_channels, C_out=out_channels, length=length, groups=groups,
                                 kernel_size=ks, activation=activation, marginal_att=True, dropout=dropout,
                                 weight_standard=weight_standard)
            self.axialblocks.append(block)

    def forward(self, x):
        for b in self.axialblocks:
            x = b(x)
        return x

class HopfieldConvNet(nn.Module):
    def __init__(self, in_channels, hid_channels, hidden_lengths, kernel_sizes=(2,5), strides=(2,5),
                 num_memories=(100,100,100), groups=4, beta=1.0, norm='BN', dropout=0.1, activation='relu',
                 weight_standard=False):
        super(HopfieldConvNet, self).__init__()

        self.HopfieldConvBlocks = torch.nn.ModuleList()

        print(norm)
        for hl, ker, stride, mems in zip(hidden_lengths, kernel_sizes, strides, num_memories):
            block = HopfieldConv2(in_channels, hid_channels, hidden_length=hl,
                                          maxpool_kernel=ker, stride=stride, num_memories=mems, groups=groups, beta=beta,
                                          activation=activation, norm=norm, weight_standard=weight_standard,
                                          dropout=dropout)

            self.HopfieldConvBlocks.append(block)

    def forward(self, x):
        for b in self.HopfieldConvBlocks:
            x = b(x)
        return x

class HighColumnNet(nn.Module):
    def __init__(self, d_model, hid_channels, num_memories, dim_feedforward=256,
                 groups=4, dropout=0.1, RZTX=True, activation='relu', norm=True):
        super(HighColumnNet, self).__init__()
        self.LetterPatterns = StatePattern(d_model, quantity=num_memories, batch_first=False)
        self.HopfieldLocal = Hopfield(input_size=d_model, hidden_size=hid_channels, output_size=d_model, num_heads=groups,
                                      add_zero_association=True, pattern_projection_as_connected=True, dropout=dropout, batch_first=False)
        if RZTX:
            self.SelfAttn = RZTXEncoderLayer(d_model, nhead=groups, dim_feedforward=dim_feedforward, dropout=dropout,
                                             activation=activation)
        else:
            self.SelfAttn = nn.TransformerEncoderLayer(d_model, nhead=groups, dim_feedforward=dim_feedforward,
                                                       dropout=dropout, activation=activation)
        self.RZ = self.RZ = nn.Parameter(torch.Tensor([0.1]))

    def forward(self, x, bpp=None):
        x = x.permute(2,0,1)
        identity = x
        _, xz, _ = self.LetterPatterns(x)
        x = self.HopfieldLocal((xz, x, xz))
        x = identity + self.RZ*x
        # print(x.shape)
        # if bpp!=None:
        #     print(bpp.shape)
        x = self.SelfAttn(x, src_mask=bpp)
        x = x.permute(1,2,0)
        return x



def choose_activation(name):
    if name == 'relu':
        return F.relu
    elif name == 'selu':
        return F.selu
    elif name == 'leaky_relu':
        return F.leaky_relu
    elif name == 'sigmoid':
        return F.sigmoid
    elif name == 'tanh':
        return F.tanh
    elif name == 'None':
        return lambda x: x
    else:
        raise ValueError(f'Function {name} is not available as an activation function for this model.')