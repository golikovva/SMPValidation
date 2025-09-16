###############################################################################
# BSD 3-Clause License
#
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#
# Author & Contact: Guilin Liu (guilinl@nvidia.com)
###############################################################################

import torch
import torch.nn.functional as F
from torch import nn, cuda, Tensor
from torch.autograd import Variable


class PartialConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False  

        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False

        super(PartialConv2d, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])
            
        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2], input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in
                        
                self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)

                # for mixed precision training, change 1e-8 to 1e-6
                self.mask_ratio = self.slide_winsize/(self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)


        raw_out = super(PartialConv2d, self).forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)


        if self.return_mask:
            return output, self.update_mask
        else:
            return output


class PartialConv2dFixed(nn.Conv2d):
    def __init__(self, *args, **kwargs):

        # whether the mask is multi-channel or not
        if 'multi_channel' in kwargs:
            self.multi_channel = kwargs['multi_channel']
            kwargs.pop('multi_channel')
        else:
            self.multi_channel = False  

        if 'return_mask' in kwargs:
            self.return_mask = kwargs['return_mask']
            kwargs.pop('return_mask')
        else:
            self.return_mask = False

        super(PartialConv2dFixed, self).__init__(*args, **kwargs)

        if self.multi_channel:
            self.weight_maskUpdater = torch.ones(self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
        else:
            self.weight_maskUpdater = torch.ones(1, 1, self.kernel_size[0], self.kernel_size[1])
            
        self.slide_winsize = self.weight_maskUpdater.shape[1] * self.weight_maskUpdater.shape[2] * self.weight_maskUpdater.shape[3]

        self.last_size = (None, None, None, None)
        self.update_mask = None
        self.mask_ratio = None

    def forward(self, input, mask_in=None):
        assert len(input.shape) == 4
        if mask_in is not None or self.last_size != tuple(input.shape):
            self.last_size = tuple(input.shape)

            with torch.no_grad():
                if self.weight_maskUpdater.type() != input.type():
                    self.weight_maskUpdater = self.weight_maskUpdater.to(input)

                if mask_in is None:
                    # if mask is not provided, create a mask
                    if self.multi_channel:
                        mask = torch.ones(input.data.shape[0], input.data.shape[1], input.data.shape[2], input.data.shape[3]).to(input)
                    else:
                        mask = torch.ones(1, 1, input.data.shape[2], input.data.shape[3]).to(input)
                else:
                    mask = mask_in
                        
                # self.update_mask = F.conv2d(mask, self.weight_maskUpdater, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)
                self.update_mask = F.conv2d(mask, self.weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=1)

                self.mask_ratio = self.weight.sum()/(self.update_mask + 1e-8)
                # self.mask_ratio = torch.max(self.update_mask)/(self.update_mask + 1e-8)
                self.update_mask = torch.clamp(self.update_mask, 0, 1)
                # self.mask_ratio = torch.mul(self.mask_ratio, self.update_mask)


        raw_out = super(PartialConv2dFixed, self).forward(torch.mul(input, mask) if mask_in is not None else input)

        if self.bias is not None:
            bias_view = self.bias.view(1, self.out_channels, 1, 1)
            output = torch.mul(raw_out - bias_view, self.mask_ratio) + bias_view
            output = torch.mul(output, self.update_mask)
        else:
            output = torch.mul(raw_out, self.mask_ratio)


        if self.return_mask:
            return output, self.update_mask
        else:
            return output
        

def _fspecial_gauss_1d(size: int, sigma: float) -> Tensor:
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.unsqueeze(0).unsqueeze(0)

class GaussianPartialConv2d(nn.Module):
    def __init__(self, channels, win_size, n_sigma=3.0, multi_channel=False):
        super().__init__()
        assert win_size % 2 == 1, "Window size must be odd"
        
        # Calculate sigma to maintain consistent Gaussian coverage
        sigma = (win_size - 1) / (2 * n_sigma)
        self.padding = win_size // 2
        
        # Create 2D Gaussian kernel
        kernel_1d = _fspecial_gauss_1d(win_size, sigma)
        kernel_2d = torch.outer(kernel_1d[0, 0], kernel_1d[0, 0])
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # (1, 1, win_size, win_size)
        kernel_2d = kernel_2d.repeat(channels, 1, 1, 1)  # (channels, 1, win_size, win_size)
        
        # Configure partial convolution
        self.conv = PartialConv2dFixed(
            channels, channels, win_size, 
            padding=self.padding, bias=False, 
            groups=channels, multi_channel=multi_channel,
            return_mask=True
        )
        self.conv.weight = nn.Parameter(kernel_2d, requires_grad=False)
        
    def forward(self, input, mask_in=None):
        return self.conv(input, mask_in)
    

def local_std(
    X: Tensor, 
    mask: Tensor, 
    win_size: int = 11, 
    n_sigma: float = 3.0,  # Controls coverage (3.0 = 99.7% of distribution)
    eps: float = 1e-8,
    device: str | torch.device = 'cpu',
) -> Tensor:
    """
    Compute local standard deviation with consistent Gaussian coverage.
    
    Args:
        X: Input tensor (N, C, H, W)
        mask: Binary mask (1=valid, 0=invalid)
        win_size: Size of Gaussian kernel (must be odd)
        n_sigma: Number of std deviations from center to edge (default=3.0)
        eps: Small value for numerical stability
        
    Returns:
        Local standard deviation tensor (same shape as X)
    """
    X = torch.as_tensor(X).float().to(device)
    mask = torch.as_tensor(mask).float().to(device)
    assert X.shape == mask.shape, "X and mask must have same shape"
    assert win_size % 2 == 1, "Window size must be odd"
    
    # Determine mask type (single or multi-channel)
    multi_channel = (mask.size(1) > 1)
    
    # Create Gaussian partial convolution
    gaussian_conv = GaussianPartialConv2d(
        X.size(1), win_size, n_sigma, multi_channel
    ).to(X.device)
    
    # Compute local mean
    mean, updated_mask = gaussian_conv(X, mask)
    
    # Compute local mean of squares
    mean_sq, _ = gaussian_conv(X * X, mask)
    
    # Compute variance and standard deviation
    variance = mean_sq - mean.pow(2)
    variance = torch.clamp(variance, min=0)  # Ensure non-negativity
    std = torch.sqrt(variance + eps).cpu().numpy()
    
    return std