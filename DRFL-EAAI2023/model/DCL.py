import torch
import torch.nn as nn
from functools import reduce
import copy
import logging
import math
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)
class Fushion_Module(nn.Module):
    def __init__(self, in_ch):
        super(Fushion_Module, self).__init__()
    def forward(self,decode,encode):
        out = torch.cat([decode, encode], 1)
        return out
class EdgeAttention(nn.Module):
    def __init__(self, planes,kernel_size=3):
        super(EdgeAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(planes, 1, kernel_size, padding=padding, bias=False)
        self.conv2 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        #print(x.shape)
        #torch.Size([1, 128, 64, 64])
        avg_out = torch.mean(x, dim=1, keepdim=True)

        x_edge = x - avg_out

        x_edge = self.conv1(x_edge)

        x = torch.cat([x_edge, avg_out], dim=1)
        x = self.conv2(x)
        return self.sigmoid(x)
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(planes)
        self.ea = EdgeAttention(planes)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.ca(out)*out
        out = self.ea(out)*out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
class Encode_layer(nn.Module):
    def __init__(self, in_ch, out_ch,relu_layer=nn.LeakyReLU,norm_layer=nn.GroupNorm):
        super(Encode_layer, self).__init__()
        self.Basic=BasicBlock(in_ch,in_ch)
        relu= relu_layer(0.2,True)
        conv=nn.Conv2d(in_ch, out_ch, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        bn=norm_layer(out_ch,out_ch)
        down=[ conv,bn,relu]
        self.down = nn.Sequential(*down)
    def forward(self, x):
        x=self.Basic(x)
        x = self.down(x)
        return x
class Decode_layer(nn.Module):
    def __init__(self, in_ch, out_ch,relu_layer=nn.PReLU,norm_layer=nn.GroupNorm,dropout=False):
        super(Decode_layer, self).__init__()
        #relu = relu_layer(True)
        self.Basic = BasicBlock(in_ch, in_ch)
        relu=relu_layer(num_parameters=1, init=0.25)
        conv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        bn = norm_layer(out_ch,out_ch)
        Dropout=nn.Dropout(0.5)
        up = [conv,bn,relu]
        if dropout==True:
            up.append(Dropout)
        self.up = nn.Sequential(*up)
    def forward(self, x):
        x=self.Basic(x)
        x = self.up(x)
        return x
class End_layer(nn.Module):
    def __init__(self, in_ch=128):
        super(End_layer, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_ch, 1, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x
class End_layer2(nn.Module):
    def __init__(self, in_ch=128):
        super(End_layer2, self).__init__()
        self.model = nn.Sequential(

            nn.ConvTranspose2d(in_ch, 1, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),

            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x
class End_layer3(nn.Module):
    def __init__(self, in_ch=128):
        super(End_layer3, self).__init__()
        self.model = nn.Sequential(

            nn.ConvTranspose2d(in_ch, 1, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),

            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x

class Softnethead(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self,input_nc):
        super(Softnethead, self).__init__()
        self.firstConv=nn.Conv2d(1, 63, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.firstrelu= nn.LeakyReLU(0.2,True)
        self.encode1 = Encode_layer(64, 64)
        self.encode2 = Encode_layer(64, 64)
        self.encode3 = Encode_layer(64, 64)
        self.encode4 = Encode_layer(64, 64)

        self.decode1 = Decode_layer(64, 64, dropout=True)
        self.decode2 = Decode_layer(128, 64)
        self.decode3 = Decode_layer(128, 64)
        self.decode4 = Decode_layer(128, 64)
        self.end = End_layer3(128)

        self.endSigmoid = nn.Sigmoid()
        self.fushion1=  Fushion_Module(64)
        self.fushion2 = Fushion_Module(64)
        self.fushion3 = Fushion_Module(64)
        self.fushion4 = Fushion_Module(64)
        self.fushion5 = Fushion_Module(64)



    def forward(self, x,sr):
        #print(sr.shape)512
        #print(x.shape)256
        e0 = self.firstConv(sr)
        e0=self.firstrelu(e0)

        e0 = torch.cat([e0,x], 1)
        e1 = self.encode1(e0)

        e2 = self.encode2(e1)
        #print(e2.shape)
        e3 = self.encode3(e2)
        #print(e3.shape)
        e4 = self.encode4(e3)

        d2 = self.decode1(e4)

        f2 = self.fushion2(d2, e3)
        d3 = self.decode2(f2)
        f3 = self.fushion3(d3, e2)
        d4 = self.decode3(f3)
        f4 = self.fushion4(d4, e1)
        d5 = self.decode4(f4)
        f5 = self.fushion5(d5, e0)
        #print(f5.shape)
        out = self.end(f5)

        return out

class ConvBNPReLU(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        args:
            nIn: number of input channels
            nOut: number of output channels
            kSize: kernel size
            stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output

class Softnet(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self,config,input_nc):
        super(Softnet, self).__init__()
        self.firstConv=nn.Conv2d(input_nc, 64, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False)
        self.firstrelu= nn.LeakyReLU(0.2,True)
        self.encode1 = Encode_layer(64, 128)
        self.encode2 = Encode_layer(128, 256)
        self.encode3 = Encode_layer(256, 512)
        self.encode4 = Encode_layer(512, 512)

        self.decode1 = Decode_layer(512, 512, dropout=True)
        self.decode2 = Decode_layer(1024, 256)
        self.decode3 = Decode_layer(512, 128)
        self.decode4 = Decode_layer(256, 64)

        self.decode5 = Decode_layer(128, 128)
        self.end = End_layer(192)
        self.end2 = End_layer2(128)

        self.fushion1=  Fushion_Module(512)
        self.fushion2 = Fushion_Module(512)
        self.fushion3 = Fushion_Module(256)
        self.fushion4 = Fushion_Module(128)
        self.fushion5 = Fushion_Module(64)
        self.Softnethead = Softnethead(5)
        #ConvBNPReLU
        self.transformer2 = Transformer2(config)
        self.transformer = Transformer(config)
        self.heatmap = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
    def forward(self, x):

        e0 = self.firstConv(x)
        e0=self.firstrelu(e0)
        e1 = self.encode1(e0)

        e2 = self.encode2(e1)
        #print(e2.shape)
        e3 = self.encode3(e2)
        #print(e3.shape)
        e4 = self.encode4(e3)

        # seg
        d2 = self.decode1(e4)
        f2 = self.fushion2(d2, e3)
        d3 = self.decode2(f2)
        f3 = self.fushion3(d3, e2)
        d4 = self.decode3(f3)
        f4 = self.fushion4(d4, e1)
        d5 = self.decode4(f4)
        d5_a = self.transformer(d5)
        #print(d5_a.shape)# [1, 64, 128, 128])



        # sr
        d2 = self.decode1(e4)
        f2 = self.fushion2(d2, e3)
        d3 = self.decode2(f2)
        f3 = self.fushion3(d3, e2)
        d4 = self.decode3(f3)
        f4 = self.fushion4(d4, e1)
        d5sr = self.decode4(f4)
        #print(d5sr.shape)#1, 64, 128, 128]
        d5sr_a = self.transformer(d5sr)
        #print(d5sr_a.shape)[1, 64, 128, 128])
        f5sr = self.fushion5(d5sr, e0)
        #print(d5sr.shape)  # [1, 64, 128, 128])
        outsr = self.decode5(f5sr)
        # print(outsr.shape) [1, 128, 256, 256])
        # outsrt = self.transformer2(outsr)
        # print(outsrt.shape)[1, 64, 256, 256]
        out2 = self.end2(outsr)

        #d5_a = self.heatmap(d5_a)
        #d5sr_a = self.heatmap(d5sr_a)
        sr = self.transformer2(d5_a,d5sr_a)
        #print(sr.shape)#[1, 1, 128, 128]
        sr = d5sr * sr
        #print(sr.shape)[1, 64, 128, 128]

        # print(d5.shape)#1, 64, 128, 128]
        d5 = torch.cat([d5, sr], 1)
        #print(d5.shape)[1, 128, 128, 128]
        #d5 = d5 + sr
        f5 = self.fushion5(d5, e0)
        # print(f5.shape) #torch.Size([1, 128, 128, 128])
        out = self.end(f5)
        # print(out.shape) #torch.Size([1, 1, 256, 256])



        #print(out2.shape) #torch.Size([1, 1, 512, 512])
        #print(d5.shape)torch.Size([1, 1, 256, 256])
        #print(d5sr.shape)
        #fu_feat = torch.cat([out,x], 1)

        bin = self.Softnethead(out,out2)
        return out,out2,bin,d5_a,d5sr_a



logger = logging.getLogger(__name__)
def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": torch.nn.functional.relu, "relu": torch.nn.functional.relu, "swish": swish}

class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights
class AttentionW(nn.Module):
    def __init__(self, config, vis):
        super(AttentionW, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.g_q = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        self.g_k = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        self.g_v = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        query_layer = torch.mul(query_layer, torch.sigmoid(self.g_q))
        key_layer = torch.mul(key_layer, torch.sigmoid(self.g_k))
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        value_layer = torch.mul(value_layer, torch.sigmoid(self.g_v))
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights
class AttentionW2(nn.Module):
    def __init__(self, config, vis):
        super(AttentionW2, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.g_q = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        self.g_k = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        self.g_v = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states,hidden_states2):
        mixed_query_layer = self.query(hidden_states2)
        mixed_key_layer = self.key(hidden_states2)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)
        query_layer = torch.mul(query_layer, torch.sigmoid(self.g_q))
        key_layer = torch.mul(key_layer, torch.sigmoid(self.g_k))
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        value_layer = torch.mul(value_layer, torch.sigmoid(self.g_v))
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:   # ResNet
            grid_size = config.patches["grid"]
            #print(img_size)
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            #print(patch_size)
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            print(patch_size)
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False


        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])


    def forward(self, x):
        #print(x.shape)
        #torch.Size([6, 3, 256, 256])


        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        #print(x.shape)
        #torch.Size([6, 1024, 16, 16])
        x4 = x
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        #torch.Size([6, 768, 16, 16])

        #print(x.shape)
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        #print(x.shape)


        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features ,x4

class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = AttentionW(config, vis)

    def forward(self, x):
        h = x

        x = self.attention_norm(x)
        print()
        x, weights = self.attn(x)
        x = x + h
        print()
        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

class Block2(nn.Module):
    def __init__(self, config, vis):
        super(Block2, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = AttentionW2(config, vis)

    def forward(self, x,x2):
        h = x
        x = self.attention_norm(x)
        x2 = self.attention_norm(x2)
        x, weights = self.attn(x,x2)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights

class Encoder2(nn.Module):
    def __init__(self, config, vis):
        super(Encoder2, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block2(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states,hidden_states2):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states,hidden_states2)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights

class Transformer(nn.Module):
    def __init__(self, config, vis=False):
        super(Transformer, self).__init__()

        self.encoder = Encoder(config, vis)
        #grid_size = config.patches["grid"]
        # print(img_size)

        img_size = 8
        img_size = _pair(img_size)
        patch_size = (8,8)
        #print(patch_size)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        #print(n_patches)
        self.position_embeddings = nn.Parameter(torch.zeros(1, 64, 768))
        self.start2 = nn.Conv2d(64, 3, kernel_size=1,dilation=6)
        self.patch_embeddings = Conv2d(in_channels=64,
                                       out_channels=768,
                                       kernel_size=16,
                                       stride=16)

        self.end = nn.Conv2d(3, 64, kernel_size=1)


        self.start1 = nn.Conv2d(64, 3, kernel_size=1, dilation=12)
        self.dropout = Dropout(config.transformer["dropout_rate"])
    def forward(self, input_ids):

        x = self.patch_embeddings(input_ids)
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        embeddings = x + self.position_embeddings
        encoded, attn_weights = self.encoder(embeddings)  # (B, n_patch, hidden)
        B, n_patch, hidden = encoded.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int((n_patch)), int((n_patch))
        x = encoded.permute(0, 2, 1)
        x = x.contiguous().view(B, 3, h*2, w*2)
        x = self.start1(input_ids) + x + self.start2(input_ids)

        x = self.end(x)
        return x

class Transformer2(nn.Module):
    def __init__(self, config, vis=False):
        super(Transformer2, self).__init__()
        self.encoder2 = Encoder2(config, vis)

        img_size = 8
        img_size = _pair(img_size)
        patch_size = (8,8)
        #print(patch_size)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        #print(n_patches)
        self.position_embeddings = nn.Parameter(torch.zeros(1, 64, 768))
        self.position_embeddings2 = nn.Parameter(torch.zeros(1, 64, 768))
        self.start2 = nn.Conv2d(64, 3, kernel_size=1,dilation=6)
        self.patch_embeddings = Conv2d(in_channels=64,
                                       out_channels=768,
                                       kernel_size=16,
                                       stride=16)
        #16384
        self.end = nn.Conv2d(3, 1, kernel_size=1)


        self.start1 = nn.Conv2d(64, 3, kernel_size=1, dilation=12)
        self.dropout = Dropout(config.transformer["dropout_rate"])
    def forward(self, input_ids,input_ids2):

        x = self.patch_embeddings(input_ids)
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        embeddings = x + self.position_embeddings

        x2 = self.patch_embeddings(input_ids2)
        x2 = x2.flatten(2)
        x2 = x2.transpose(-1, -2)  # (B, n_patches, hidden)
        embeddings2 = x2 + self.position_embeddings2

        encoded, attn_weights = self.encoder2(embeddings,embeddings2)  # (B, n_patch, hidden)
        B, n_patch, hidden = encoded.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int((n_patch)), int((n_patch))
        x = encoded.permute(0, 2, 1)
        x = x.contiguous().view(B, 3, h*2, w*2)
        x = self.start1(input_ids) + x + self.start2(input_ids)

        x = self.end(x)
        return x


