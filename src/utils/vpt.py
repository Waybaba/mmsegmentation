import torch
import torch.nn as nn
import numpy as np

class PadPrompter(nn.Module):
    def __init__(self, args):
        super(PadPrompter, self).__init__()
        pad_size = args.prompt_size
        image_size = args.image_size

        h, w = image_size

        self.base_size = (h - pad_size*2, w - pad_size*2)
        self.pad_up = nn.Parameter(torch.zeros([1, 3, pad_size, w]))
        self.pad_down = nn.Parameter(torch.zeros([1, 3, pad_size, w]))
        self.pad_left = nn.Parameter(torch.zeros([1, 3, h - pad_size*2, pad_size]))
        self.pad_right = nn.Parameter(torch.zeros([1, 3, h - pad_size*2, pad_size]))

    def forward(self, x):
        base = torch.zeros(1, 3, *self.base_size).to(x.device)
        prompt = torch.cat([self.pad_left, base, self.pad_right], dim=3)
        prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=2)
        prompt = torch.cat(x.size(0) * [prompt])

        # rotate the prompt (b, c, h, w) -> (b, c, w, h)
        prompt = prompt.permute(0, 1, 3, 2)

        prompt *= 255

        return x + prompt


class FixedPatchPrompter(nn.Module):
    def __init__(self, args):
        super(FixedPatchPrompter, self).__init__()
        self.isize = args.image_size
        self.psize = args.prompt_size
        self.patch = nn.Parameter(torch.zeros([1, 3, self.psize, self.psize]))

    def forward(self, x):
        prompt = torch.zeros([1, 3, *self.isize]).to(x.device)
        prompt[:, :, :self.psize, :self.psize] = self.patch

        prompt = prompt.permute(0, 1, 3, 2)

        prompt *= 255

        return x + prompt


class RandomPatchPrompter(nn.Module):
    def __init__(self, args):
        super(RandomPatchPrompter, self).__init__()
        self.isize = args.image_size
        self.psize = args.prompt_size
        self.patch = nn.Parameter(torch.zeros([1, 3, self.psize, self.psize]))

    def forward(self, x):
        x_ = np.random.choice(self.isize[0] - self.psize)
        y_ = np.random.choice(self.isize[1] - self.psize)

        prompt = torch.zeros([1, 3, *self.isize]).to(x.device)
        prompt[:, :, x_:x_ + self.psize, y_:y_ + self.psize] = self.patch
        
        prompt = prompt.permute(0, 1, 3, 2)

        prompt *= 255

        return x + prompt

class CNNPrompter(nn.Module):
    """
    A prompter that uses a CNN to generate the new image.
    Convolutions are applied to the original image to generate the prompt.
    the stride is 1 and the padding is the same as the kernel size.
    Different from previous prompters, this prompter is not a fixed patch.
    Also, the args.prompt_size should be a list which indicates the kernel size of each layer.
    args.psize:
        a list of kernel size for each layer.
    args.res_add:
        True means that the output is add to the original image.
        Also, the kernel would be inited with 0 instead of 1 at
        the center of the kernel.
    """
    def __init__(self, args):
        super(CNNPrompter, self).__init__()
        self.isize = args.image_size
        self.psize = args.prompt_size
        self.res_add = args.res_add
        self.weight_init_type = args.weight_init_type # origin or random
        assert hasattr(self.psize, "__iter__"), "args.prompt_size should be a list"
        self.layers = nn.ModuleList()
        for l_idx, k_size in enumerate(self.psize):
            # create weight
            if self.weight_init_type == "zero":
                kernel = torch.zeros([3, 3, k_size, k_size])
                bias = torch.zeros([3])
            elif self.weight_init_type == "normal":
                kernel = torch.normal(0, 1.0, [3, 3, k_size, k_size])
                bias = torch.normal(0, 1.0, [3])
            elif self.weight_init_type == "origin":
                if self.res_add:
                    kernel = torch.zeros([3, 3, k_size, k_size])
                    bias = torch.zeros([3])
                else:
                    # diagonal kernel which output the same color as the input
                    kernel = torch.zeros([3, 3, k_size, k_size])
                    for kn_idx in range(3):
                        kernel[kn_idx, kn_idx, k_size//2, k_size//2] = 1
                    bias = torch.zeros([3])
            else:
                raise ValueError("weight_init_type should be one of [zero, normal, origin]")
            # init params
            self.layers.append(nn.Conv2d(3, 3, k_size, 1, k_size//2, bias=True))
            kernel = nn.Parameter(kernel)
            bias = nn.Parameter(bias)
            self.layers[-1].weight = kernel
            self.layers[-1].bias = bias
            # add activation
            if not (l_idx == (len(self.psize) - 1)):
                self.layers.append(nn.ReLU())
    
    def forward(self, x):
        prompt = x
        for layer in self.layers:
            prompt = layer(prompt)
        if self.res_add:
            return x + prompt
        else:
            return prompt
        

def padding(args):
    return PadPrompter(args)


def fixed_patch(args):
    return FixedPatchPrompter(args)


def random_patch(args):
    return RandomPatchPrompter(args)



### The following is additonal code outside the original file
# the original file is from https://github.com/hjbahng/visual_prompting
# API:
#   VisutalPrompter(type, dynamic_cfg, prompt_size=32)
#   type: one of "padding", "fixed_patch", "random_patch"
#   prompt_size: the size of the prompt, default to 32
#   dynamic_cfg: a dict with key "image_size" and value a tuple of (h, w)

def wrapper(base_class):
    class Wrapper(base_class):
        def __init__(self, dynamic_cfg, prompt_size=32):
            # if dynamic_cfg["image_size"][0] != dynamic_cfg["image_size"][1]:
            #     raise ValueError("image_size should be square for Visual Prompter e.g. (224, 224)")
            # make a empty config for the base class
            args = type('args', (object,), {})()
            args.prompt_size = prompt_size
            for k, v in dynamic_cfg.items():
                setattr(args, k, v)
            super().__init__(args=args)
    return Wrapper

PadPrompter_ = wrapper(PadPrompter)
FixedPatchPrompter_ = wrapper(FixedPatchPrompter)
RandomPatchPrompter_ = wrapper(RandomPatchPrompter)
CNNPrompter_ = wrapper(CNNPrompter)

def VisutalPrompter(type, dynamic_cfg={}, prompt_size=32, **kwargs):
    dynamic_cfg.update(kwargs)
    if type == "padding":
        return PadPrompter_(dynamic_cfg, prompt_size)
    elif type == "fixed_patch":
        return FixedPatchPrompter_(dynamic_cfg, prompt_size)
    elif type == "random_patch":
        return RandomPatchPrompter_(dynamic_cfg, prompt_size)
    elif type == "cnn":
        return CNNPrompter_(dynamic_cfg, prompt_size)
    else:
        raise ValueError("type should be one of padding, fixed_patch, random_patch")
