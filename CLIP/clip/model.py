# import sys
# sys.path.append("..")
from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import math

from .reducer import TokenReduction

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.flatten(start_dim=2).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)



class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, reduce: str = 'ours'):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask
        if reduce == 'ours':
            self.token_reducer = TokenReduction(batch_first=False)
        self.reduce = reduce

    def attention(self, x: torch.Tensor, token_size: torch.Tensor, token_id: torch.Tensor, text: torch.Tensor = None):
        if self.attn_mask is not None and (self.attn_mask.shape[0] != x.shape[0] or self.attn_mask.device != x.device):
            self.attn_mask = torch.empty(x.shape[0], x.shape[0]).fill_(float("-inf")).triu_(1).to(dtype=x.dtype, device=x.device)
        if text is None: 
            return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask, token_size=token_size)

    def forward(self, x: torch.Tensor, token_id: torch.Tensor = None, text: torch.Tensor = None):
        if self.reduce == 'trips':
            x_attn, metirc = self.attention(self.ln_1(x), None, None, text=text)
            x = x + x_attn
            x = self.token_reducer(x, self.ln_2(x), metirc)
            x = x + self.mlp(self.ln_2(x))
            return x

        x_attn, metirc = self.attention(self.ln_1(x), self.token_reducer.token_status["token_size"] if hasattr(self, 'token_reducer') else None, token_id)
        x = x + x_attn
        if hasattr(self, 'token_reducer'):
            x, token_id = self.token_reducer(x, metirc, token_id=token_id)
        x = x + self.mlp(self.ln_2(x))
        if token_id is not None:
            return x, token_id
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, reduce: str = 'ours', r: int = 0):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask, reduce if r > 0 else 'none') for _ in range(layers)])

        ### 
        self.token_status = {"r": r}
        for module in self.modules():
            if isinstance(module, TokenReduction):
                module.token_status = self.token_status
        self.reduce=reduce
        self.cross_token = [None for i in range(self.layers)]

    def forward(self, x: torch.Tensor, token_id: torch.Tensor = None, text: torch.Tensor = None):
        self.token_status['token_size'] = torch.ones_like(x[..., 0, None]).permute(1, 0, 2)
        if token_id is not None:
            idx = torch.arange(x.shape[1])
            if self.reduce == 'ours' and self.token_status["r"] > 0:
                token_id = torch.cat([token_id, token_id[:, 0, None]], dim=-1)
            for i, module in enumerate(self.resblocks):
                if self.reduce == 'ours':
                    cross_token = x[token_id.argmax(dim=-1), idx].detach() + self.cross_text[:, i, :]
                    self.cross_token[i] = cross_token
                    x = torch.cat([x[:-1], cross_token[None, ...]], dim=0)
                x, token_id = module(x, token_id, text=text)
            return x, token_id
        else:
            for i, module in enumerate(self.resblocks):
                if self.reduce == 'ours':
                    cross_token = x[-1, ...].detach() + self.cross_image[:, i, :]
                    self.cross_token[i] = cross_token
                    x = torch.cat([x[:-1], cross_token[None, ...]], dim=0)
                x = module(x, text=text)
            return x


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int, reduce: str = 'ours', r: int = 0):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, reduce=reduce, r=r)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        ### 
        self.reduce=reduce

    def forward(self, x: torch.Tensor, text: torch.Tensor = None):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        if self.reduce == 'ours':
            cross_token = x[:, 0, None, :]
            x = torch.cat([x, cross_token.expand(x.shape[0], -1, -1)], dim=1) ###

        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND ###
        x = self.transformer(x, text=text)
        x = x.permute(1, 0, 2)  # LND -> NLD ###

        x_cls = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x_cls = x_cls @ self.proj

        if self.reduce == 'ours' and self.training:
            x_cross = F.layer_norm(torch.cat(self.transformer.cross_token), \
                self.ln_pre.normalized_shape, self.ln_pre.weight.detach(), self.ln_pre.bias.detach(), self.ln_pre.eps)
            x_cross = x_cross @ self.proj.detach()
            return (x_cls, x_cross)
        return x_cls


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 reduce: str,
                 r: dict
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim,
                reduce=reduce,
                r=r['rv']
            )

        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
            reduce=reduce,
            r=r['rl']
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

        self.reduce=reduce
        if self.reduce=='ours':
            self.visual.transformer.cross_image = nn.Parameter(torch.zeros(1, vision_layers, vision_width)) ##
            self.transformer.cross_text = nn.Parameter(torch.zeros(1, transformer_layers, transformer_width)) ##

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)


    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    @property
    def device(self):
        return self.visual.conv1.weight.device

    def encode_image(self, image, text=None):
        return self.visual(image.type(self.dtype), text=text)

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        if self.reduce == 'ours' :
            cross_token = x[torch.arange(x.shape[0]), text.argmax(dim=-1), None, :]	
            x = torch.cat([x, cross_token], dim=1)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x, text = self.transformer(x, text)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_final(x).type(self.dtype)
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x_eos = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        if self.reduce == 'ours' and self.training:
            x_cross = F.layer_norm(torch.cat(self.transformer.cross_token), \
                self.ln_final.normalized_shape, self.ln_final.weight.detach(), self.ln_final.bias.detach(), self.ln_final.eps)
            x_cross = x_cross @ self.text_projection.detach()
            return (x_eos, x_cross)
        return x_eos

    def forward(self, image, caption, alpha, idx): 
        
        with torch.no_grad():
            self.logit_scale.clamp_(0, 4.6052)
        
        idx = idx.view(-1,1)
        idx_all = idx.t()
        pos_idx = torch.eq(idx, idx_all).float()       
        sim_targets = pos_idx / pos_idx.sum(1,keepdim=True)   
    
        image_features = self.encode_image(image)
        text = self.tokenize(caption).to(image.device)
        text_features = self.encode_text(text)


        if self.reduce=='ours' and self.training:
            image_features, image_cross = image_features
            text_features, text_cross = text_features
            cross_mean = (F.softmax(image_cross, dim=-1) + F.softmax(text_cross, dim=-1)) / 2
            loss_kd = (F.kl_div(F.log_softmax(image_cross, dim=-1), cross_mean, reduction='batchmean') + \
                F.kl_div(F.log_softmax(text_cross, dim=-1), cross_mean, reduction='batchmean')) / 2

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        loss_i2t = -torch.sum(F.log_softmax(logits_per_image, dim=1)*sim_targets,dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(logits_per_text, dim=1)*sim_targets,dim=1).mean() 

        loss_ita = (loss_i2t+loss_t2i)/2

        return (loss_ita, loss_kd) if (self.reduce=='ours' and self.training) else loss_ita


    @torch.no_grad()    
    def init_cross(self):
        self.cross_image.data.copy_(self.visual.class_embedding.data[None, None, ...])
        self.cross_text.data.copy_(self.token_embedding.weight.data[None, -1, None, :])
    

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat, idxs):
        # gather keys before updating queue
        image_feats = concat_all_gather(image_feat)
        text_feats = concat_all_gather(text_feat)
        

        batch_size = image_feats.shape[0]
        ptr = int(self.ptr_queue)
        assert self.queue_size % batch_size == 0  # for simplicity

        if ptr % batch_size != 0:
            ptr = (ptr // batch_size) * batch_size

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        self.idx_queue[:, ptr:ptr + batch_size] = idxs.T
        ptr = (ptr + batch_size) % self.queue_size # move pointer

        self.ptr_queue[0] = ptr  


    @torch.no_grad()       
    def reset_queue(self):
        self.image_queue = torch.randn(self.embed_dim, self.queue_size).to(device=self.image_queue.device)
        self.text_queue = torch.randn(self.embed_dim, self.queue_size).to(device=self.text_queue.device)
        self.idx_queue = torch.full((1,self.queue_size),-100).to(device=self.idx_queue.device)
        self.ptr_queue = torch.zeros(1, dtype=torch.long).to(device=self.ptr_queue.device)
        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)




def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)


def build_model(state_dict: dict, evaluate: str, reduce: str, r: dict):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [len(set(k.split(".")[2] for k in state_dict if k.startswith(f"visual.layer{b}"))) for b in [1, 2, 3, 4]]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round((state_dict["visual.attnpool.positional_embedding"].shape[0] - 1) ** 0.5)
        vision_patch_size = None
        assert output_width ** 2 + 1 == state_dict["visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith("transformer.resblocks")))

    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers,
        reduce=reduce, r=r
    )

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    if not evaluate:
        convert_weights(model) 
        model.load_state_dict(state_dict, strict=False)
        # model.proj_inverse.data.copy_(torch.linalg.pinv(model.visual.proj.float()).data) # rebuttal: trips
        return model
    else:
        model.load_state_dict(state_dict)
        return model.eval()

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