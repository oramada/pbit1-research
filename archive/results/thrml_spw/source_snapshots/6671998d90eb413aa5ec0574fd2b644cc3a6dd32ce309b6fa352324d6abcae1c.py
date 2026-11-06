"""Masked affine integer convolution using one free p-bit per kernel weight."""
import numpy as np
from results.thrml_spw.integer_activation_native.gated_layer import IntegerGateLayer


def patches(x, kernel, stride, padding, dilation, lower, span):
    """NCHW inputs; padding has a separate mask, never a rounded zero code."""
    n, channels, height, width = x.shape
    kh, kw = kernel
    oh = (height + 2*padding - dilation*(kh-1)-1)//stride + 1
    ow = (width + 2*padding - dilation*(kw-1)-1)//stride + 1
    if min(oh, ow) < 1 or span <= 0:
        raise ValueError('Invalid convolution geometry or activation range')
    codes = np.zeros((n*oh*ow, channels*kh*kw), np.uint8)
    mask = np.zeros_like(codes)
    for image in range(n):
        for row in range(oh):
            for col in range(ow):
                dst = (image*oh+row)*ow+col
                for channel in range(channels):
                    for kr in range(kh):
                        for kc in range(kw):
                            i = (channel*kh+kr)*kw+kc
                            rr = row*stride-padding+kr*dilation
                            cc = col*stride-padding+kc*dilation
                            if 0 <= rr < height and 0 <= cc < width:
                                v = np.clip((float(x[image,channel,rr,cc])-lower)/span, 0, 1)
                                codes[dst,i] = np.rint(255*v)
                                mask[dst,i] = 1
    # Observe both code and validity with identical weight states, not two runs.
    return np.concatenate([codes,mask]), codes, mask, (n,oh,ow)


def decode(gated, shape, codes, mask, samples, lower, span, center, radius, bias):
    positions = codes.shape[0]
    g, m = gated[:,:positions], gated[:,positions:]
    sum_x = np.sum(mask*(lower+span*codes.astype(float)/255),axis=1)
    sampled = lower*m.astype(float)/samples + (span/255)*g.astype(float)/samples
    y = sum_x[None,:,None]*(center-radius) + 2*radius*sampled + bias
    n,oh,ow = shape
    return y.reshape(y.shape[0],n,oh,ow,-1).transpose(0,1,4,2,3)


def layer(kernel_shape, output_channels, observed_rows):
    return IntegerGateLayer(int(np.prod(kernel_shape)),output_channels,observed_rows)
