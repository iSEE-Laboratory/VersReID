import math
from functools import partial
from itertools import repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._six import container_abcs


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
to_2tuple = _ntuple(2)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, stride_size=20, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        stride_size_tuple = to_2tuple(stride_size)
        self.num_x = (img_size[1] - patch_size[1]) // stride_size_tuple[1] + 1
        self.num_y = (img_size[0] - patch_size[0]) // stride_size_tuple[0] + 1
        print('using stride: {}, and patch number is num_y{} * num_x{}'.format(stride_size, self.num_y, self.num_x))
        num_patches = self.num_x * self.num_y
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        B, C, H, W = x.shape

        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)

        x = x.flatten(2).transpose(1, 2)  # [64, 8, 768]
        return x


class TransReID(nn.Module):
    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, num_classes=1000, embed_dim=768,
                 depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., camera=0,
                 view=0,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, local_feature=False, sie_xishu=1.0,
                 scene_nums=0, scene_prompt_type=0, scene_prompt_nums=0, aux_loss=False, aux_prompt_nums=0):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.local_feature = local_feature
        self.patch_embed = PatchEmbed(img_size=img_size,
                                      patch_size=patch_size,
                                      stride_size=stride_size,
                                      in_chans=in_chans,
                                      embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches

        self.scene_nums = scene_nums
        self.scene_prompt_type = scene_prompt_type
        self.scene_prompt_nums = scene_prompt_nums
        self.aux_prompt_nums = aux_prompt_nums

        if self.scene_prompt_type == 1:
            print("Using scene prompt with {} scenes, each scene with {} tokens...".format(self.scene_nums,
                                                                                           self.scene_prompt_nums))
            # k, n, c
            self.scene_prompt_token = nn.Parameter(torch.zeros(self.scene_nums, self.scene_prompt_nums, self.embed_dim))
            trunc_normal_(self.scene_prompt_token, std=.02)
            if aux_loss and self.aux_prompt_nums > 0:
                print("Using aux prompt with {} tokens...".format(self.aux_prompt_nums))
                self.aux_prompt_token = nn.Parameter(torch.zeros(1, self.aux_prompt_nums, self.embed_dim))
                trunc_normal_(self.aux_prompt_token, std=.02)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.cam_num = camera
        self.view_num = view
        self.sie_xishu = sie_xishu
        # Initialize SIE Embedding
        if camera > 1 and view > 1:
            self.sie_embed = nn.Parameter(torch.zeros(camera * view, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=.02)
            print('camera number is : {} and viewpoint number is : {}'.format(camera, view))
            print('using SIE_Lambda is : {}'.format(sie_xishu))
        elif camera > 1:
            self.sie_embed = nn.Parameter(torch.zeros(camera, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=.02)
            print('camera number is : {}'.format(camera))
            print('using SIE_Lambda is : {}'.format(sie_xishu))
        elif view > 1:
            self.sie_embed = nn.Parameter(torch.zeros(view, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=.02)
            print('viewpoint number is : {}'.format(view))
            print('using SIE_Lambda is : {}'.format(sie_xishu))

        print('using drop_out rate is : {}'.format(drop_rate))
        print('using attn_drop_out rate is : {}'.format(attn_drop_rate))
        print('using drop_path rate is : {}'.format(drop_path_rate))

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.fc = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'scene_prompt_token', 'aux_prompt_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.fc = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, camera_id, view_id, dataset_label):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)

        if self.cam_num > 0 and self.view_num > 0:
            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[camera_id * self.view_num + view_id]
        elif self.cam_num > 0:
            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[camera_id]
        elif self.view_num > 0:
            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[view_id]
        else:
            x = x + self.pos_embed

        if self.scene_prompt_type == 1 and (self.aux_prompt_nums > 0 or self.scene_prompt_nums > 0):
            prompt_tokens = []
            for i in range(x.size()[0]):
                if isinstance(dataset_label, int):  # for testing
                    if dataset_label != -1:
                        prompt_tokens.append(self.scene_prompt_token[dataset_label].unsqueeze(0))
                    else:
                        if self.aux_prompt_nums <= 0:
                            break
                        prompt_tokens.append(self.aux_prompt_token)
                elif len(dataset_label.size()) == 1:  # for training GT feat
                    prompt_tokens.append(self.scene_prompt_token[dataset_label[i]].unsqueeze(0))

                # The below condition is discarded in the new version of the code (always False)
                elif len(dataset_label.size()) == 2:  # for training feat
                    dataset_label = torch.softmax(dataset_label, dim=1)
                    scene_prompt_token = torch.einsum('i,ijk->jk', dataset_label[i], self.scene_prompt_token)
                    prompt_tokens.append(scene_prompt_token.unsqueeze(0))

            if len(prompt_tokens) == x.size(0):
                prompt_tokens = torch.cat(prompt_tokens, dim=0)
                x = torch.cat((x, prompt_tokens), dim=1)

        x = self.pos_drop(x)

        if self.local_feature:
            for blk in self.blocks[:-1]:
                x = blk(x)
            return x
        else:
            for blk in self.blocks:
                x = blk(x)
            x = self.norm(x)
            return x

    def forward(self, x, cam_label=None, view_label=None, dataset_label=None, mid_feat=False):
        x = self.forward_features(x, cam_label, view_label, dataset_label)
        if mid_feat:
            return x
        else:
            return x[:, 0]

    def load_param(self, model_path, param_dict=None):
        if param_dict is None:
            param_dict = torch.load(model_path, map_location='cpu')
        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for k, v in param_dict.items():
            if 'head' in k or 'dist' in k:
                continue
            if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = self.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
                # To resize pos embedding when using model at different size from pretrained weights
                if model_path is not None and 'distilled' in model_path:
                    print('distill need to choose right cls token in the pth')
                    v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
                v = resize_pos_embed(v, self.pos_embed, self.patch_embed.num_y, self.patch_embed.num_x, model_path)
            try:
                self.state_dict()[k].copy_(v)
            except:
                print('===========================ERROR=========================')
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape,
                                                                                                self.state_dict()[
                                                                                                    k].shape))


def resize_pos_embed(posemb, posemb_new, hight, width, model_path):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = posemb_new.shape[1]

    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]
    ntok_new -= 1

    print('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape,
                                                                                                posemb_new.shape, hight,
                                                                                                width))
    if posemb_grid.size()[0] == 196:
        posemb_grid = posemb_grid.reshape(1, 14, 14, -1).permute(0, 3, 1, 2)
    if posemb_grid.size()[0] == 128:
        posemb_grid = posemb_grid.reshape(1, 16, 8, -1).permute(0, 3, 1, 2)
    if posemb_grid.size()[0] == 192:
        posemb_grid = posemb_grid.reshape(1, 24, 8, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb


def vit_base_patch16_224_TransReID(img_size=(256, 128),
                                   stride_size=16,
                                   drop_rate=0.0,
                                   attn_drop_rate=0.0,
                                   drop_path_rate=0.1,
                                   camera=0,
                                   view=0,
                                   local_feature=False,
                                   sie_xishu=1.5,
                                   **kwargs):
    model = TransReID(img_size=img_size,
                      patch_size=16,
                      stride_size=stride_size,
                      embed_dim=768,
                      depth=12,
                      num_heads=12,
                      mlp_ratio=4,
                      qkv_bias=True,
                      camera=camera,
                      view=view,
                      drop_path_rate=drop_path_rate,
                      drop_rate=drop_rate,
                      attn_drop_rate=attn_drop_rate,
                      norm_layer=partial(nn.LayerNorm, eps=1e-6),
                      sie_xishu=sie_xishu,
                      local_feature=local_feature,
                      **kwargs)
    return model


def vit_small_patch16_224_TransReID(img_size=(256, 128),
                                    stride_size=16,
                                    drop_rate=0.,
                                    attn_drop_rate=0.,
                                    drop_path_rate=0.1,
                                    camera=0, view=0,
                                    local_feature=False,
                                    sie_xishu=1.5,
                                    **kwargs):
    kwargs.setdefault('qk_scale', 384 ** -0.5)
    model = TransReID(img_size=img_size,
                      patch_size=16,
                      stride_size=stride_size,
                      embed_dim=384,
                      depth=12,
                      num_heads=6,
                      mlp_ratio=4.,
                      qkv_bias=True,
                      drop_path_rate=drop_path_rate,
                      camera=camera,
                      view=view,
                      drop_rate=drop_rate,
                      attn_drop_rate=attn_drop_rate,
                      norm_layer=partial(nn.LayerNorm, eps=1e-6),
                      sie_xishu=sie_xishu,
                      local_feature=local_feature,
                      **kwargs)
    return model


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        print("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
              "The distribution of values may be incorrect.", )

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        tensor.uniform_(2 * l - 1, 2 * u - 1)

        tensor.erfinv_()

        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
