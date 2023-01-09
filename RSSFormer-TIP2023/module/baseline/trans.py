from module.baseline.base_hrnet.hrnet_encoder import HRNetEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from ever.core import registry
import ever as er
from module.CGFL import SegmentationLoss

BatchNorm2d = nn.BatchNorm2d

BN_MOMENTUM = 0.1


class SimpleFusion(nn.Module):
    def __init__(self, in_channels):
        super(SimpleFusion, self).__init__()
        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(True)
        )

    def forward(self, feat_list):
        # print(feat_list[0].shape)
        # print(feat_list[1].shape)
        # print(feat_list[2].shape)
        # print(feat_list[3].shape)
        # torch.Size([16, 32, 128, 128])
        # torch.Size([16, 64, 64, 64])
        # torch.Size([16, 128, 32, 32])
        # torch.Size([16, 256, 16, 16])

        x0 = feat_list[0]
        x0_h, x0_w = x0.size(2), x0.size(3)
        x1 = F.interpolate(feat_list[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x2 = F.interpolate(feat_list[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x3 = F.interpolate(feat_list[3], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x = torch.cat([x0, x1, x2, x3], dim=1)
        x = self.fuse_conv(x)
        return x


@registry.MODEL.register('trans')
class trans(er.ERModule):
    def __init__(self, config):
        super(trans, self).__init__(config)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.backbone = HRNetEncoder(self.config.backbone)
        self.neck = SimpleFusion(self.config.neck.in_channels)
        self.head = nn.Sequential(
            nn.Conv2d(self.config.head.in_channels, self.config.classes, 1),
            nn.UpsamplingBilinear2d(scale_factor=self.config.head.upsample_scale),
        )
        self.loss = SegmentationLoss(self.config.loss)

    def forward(self, x, y=None):
        pred_list = self.backbone(x)

        logit = self.neck(pred_list)
        # print('logit',logit.shape)    torch.Size([16, 480, 128, 128])
        logit = self.head(logit)
        # print('logit', logit.shape) torch.Size([16, 7, 512, 512])

        if self.training:
            y_true = y['cls']
            return self.loss(logit, y_true.long())
        else:
            return logit.softmax(dim=1)

    def set_default_config(self):
        self.config.update(dict(
            backbone=dict(
                hrnet_type='hrnetv2_w48',
                pretrained=False,
                norm_eval=False,
                frozen_stages=-1,
                with_cp=False,
                with_gc=False,
            ),
            neck=dict(
                in_channels=720,
            ),
            classes=7,
            head=dict(
                in_channels=720,
                upsample_scale=4.0,
            ),
            loss=dict(

                ce=dict(),
            )
        ))



import copy
import math
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
def swish(x):
    return x * torch.sigmoid(x)
ACT2FN = {"relu": torch.nn.functional.relu}
import ml_collections

def get_config():
    """Returns the ViT-B/16 configuration."""
    config = ml_collections.ConfigDict()
    config.patches = ml_collections.ConfigDict({'size': (16, 16)})
    config.hidden_size = 170 # 768
    config.transformer = ml_collections.ConfigDict()
    config.transformer.mlp_dim = 3072
    config.transformer.num_heads = 12 #12 7
    config.transformer.num_layers = 1
    config.transformer.attention_dropout_rate = 0.0
    config.transformer.dropout_rate = 0.1
    config.classifier = 'seg'
    config.representation_size = None
    config.patch_size = 16
    config.decoder_channels = (256, 128, 64, 16)
    config.n_classes = 2
    config.activation = 'softmax'
    return config
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
        #print(hidden_states.shape)
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
        #print(context_layer.shape)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        #print(context_layer.shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights

class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["relu"]
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
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)

        x, weights = self.attn(x)
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
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states2)

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
    def __init__(self, vis=False):
        super(Transformer, self).__init__()
        config = get_config()
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
        self.start = nn.Conv2d(64, 128, kernel_size=1,dilation=3)
        self.patch_embeddings = Conv2d(in_channels=64,
                                       out_channels=768,
                                       kernel_size=16,
                                       stride=16)
        #16384
        self.end = nn.Conv2d(3, 64, kernel_size=1)


        self.start1 = nn.Conv2d(64, 3, kernel_size=1, dilation=3)
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, input_ids, input_ids2):
        x = input_ids #Q K
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        embeddings = x #+ self.position_embeddings

        x2 = input_ids2 #V
        x2 = x2.flatten(2)
        x2 = x2.transpose(-1, -2)  # (B, n_patches, hidden)
        embeddings2 = x2 #+ self.position_embeddings2

        print(embeddings.shape)
        print(embeddings2.shape)
        print("-------------------------------------")
        # torch.Size([2, 16, 224, 224])
        # torch.Size([2, 50176, 16])
        # torch.Size([2, 50176, 16])

        encoded, attn_weights = self.encoder2(embeddings, embeddings2)  # (B, n_patch, hidden)
        B, n_patch, hidden = encoded.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int((n_patch)), int((n_patch))
        x = encoded.permute(0, 2, 1)
        x = x.contiguous().view(B, 3, h * 2, w * 2)

        print(x.shape)

        x = self.end(x)
        return x
