from torch import nn
from torch.autograd import Function
import torch.nn.functional as F
import torch
from torch.nn.modules.utils import _triple, _pair, _single

import adapool_cuda


class CUDA_ADAPOOL1d(Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, beta, kernel=2, stride=None, return_mask=False):

        assert input.dtype==beta.dtype, '`input` and `beta` are not of the same dtype.'
        beta = torch.clamp(beta , 0., 1.)
        no_batch = False
        if len(input.size()) == 2:
            no_batch = True
            input.unsqueeze_(0)
        B, C, D = input.shape
        kernel = _single(kernel)
        if stride is None:
            stride = kernel
        else:
            stride = _single(stride)

        oD = (D-kernel[0]) // stride[0] + 1

        output = input.new_zeros((B, C, oD))
        if return_mask:
            mask = input.new_zeros((B, kernel[0] * oD))
        else:
            mask = input.new_zeros((1))

        adapool_cuda.forward_1d(input.contiguous(), beta, kernel, stride, output, return_mask, mask)
        ctx.save_for_backward(input, beta)
        ctx.kernel = kernel
        ctx.stride = stride
        if return_mask:
            mask_ = mask.detach().clone()
            mask_.requires_grad = False
            CUDA_ADAPOOL1d.mask = mask_
        output = torch.nan_to_num(output)
        if no_batch:
            return output.squeeze_(0)
        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):

        grad_input = torch.zeros_like(ctx.saved_tensors[0])
        grad_beta = torch.zeros_like(ctx.saved_tensors[1])

        saved = [grad_output] + list(ctx.saved_tensors) + [ctx.kernel, ctx.stride, grad_input, grad_beta]
        adapool_cuda.backward_1d(*saved)

        return torch.nan_to_num(saved[-2]), torch.nan_to_num(saved[-1]), None, None, None





class CUDA_ADAPOOL1d_EDSCW(Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, kernel=2, stride=None, return_mask=False):

        no_batch = False
        if len(input.size()) == 2:
            no_batch = True
            input.unsqueeze_(0)
        B, C, D = input.shape
        kernel = _single(kernel)
        if stride is None:
            stride = kernel
        else:
            stride = _single(stride)
        oD = (D-kernel[0]) // stride[0] + 1
        output = input.new_zeros((B, C, oD))
        if return_mask:
            mask = input.new_zeros((B, kernel[0] * oD))
        else:
            mask = input.new_zeros((1))

        adapool_cuda.forward_1d_edscw(input.contiguous(), kernel, stride, output, return_mask, mask)
        ctx.save_for_backward(input)
        ctx.kernel = kernel
        ctx.stride = stride
        if return_mask:
            mask_ = mask.detach().clone()
            mask_.requires_grad = False
            CUDA_ADAPOOL1d_EDSCW.mask = mask_
        output = torch.nan_to_num(output)
        if no_batch:
            return output.squeeze_(0)
        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):

        grad_input = torch.zeros_like(ctx.saved_tensors[0])

        saved = [grad_output] + list(ctx.saved_tensors) + [ctx.kernel, ctx.stride, grad_input]
        adapool_cuda.backward_1d_edscw(*saved)

        return torch.nan_to_num(saved[-1]), None, None, None



class CUDA_IDWPOOL1d(Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, kernel=2, stride=None, return_mask=False):

        no_batch = False
        if len(input.size()) == 2:
            no_batch = True
            input.unsqueeze_(0)
        B, C, D = input.shape
        kernel = _single(kernel)
        if stride is None:
            stride = kernel
        else:
            stride = _single(stride)
        oD = (D-kernel[0]) // stride[0] + 1
        output = input.new_zeros((B, C, oD))
        if return_mask:
            mask = input.new_zeros((B, kernel[0] * oD))
        else:
            mask = input.new_zeros((1))

        adapool_cuda.forward_1d_idw(input.contiguous(), kernel, stride, output, return_mask, mask)
        ctx.save_for_backward(input)
        ctx.kernel = kernel
        ctx.stride = stride
        if return_mask:
            mask_ = mask.detach().clone()
            mask_.requires_grad = False
            CUDA_ADAPOOL1d_EDSCW.mask = mask_
        output = torch.nan_to_num(output)
        if no_batch:
            return output.squeeze_(0)
        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):

        grad_input = torch.zeros_like(ctx.saved_tensors[0])

        saved = [grad_output] + list(ctx.saved_tensors) + [ctx.kernel, ctx.stride, grad_input]
        adapool_cuda.backward_1d_idw(*saved)

        return torch.nan_to_num(saved[-1]), None, None, None


class CUDA_ADAPOOL1d_EM(Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, kernel=2, stride=None, return_mask=False):

        no_batch = False
        if len(input.size()) == 2:
            no_batch = True
            input.unsqueeze_(0)
        B, C, D = input.shape
        kernel = _single(kernel)
        if stride is None:
            stride = kernel
        else:
            stride = _single(stride)
        oD = (D-kernel[0]) // stride[0] + 1
        output = input.new_zeros((B, C, oD))
        if return_mask:
            mask = input.new_zeros((B, kernel[0] * oD))
        else:
            mask = input.new_zeros((1))

        adapool_cuda.forward_1d_em(input.contiguous(), kernel, stride, output, return_mask, mask)
        ctx.save_for_backward(input)
        ctx.kernel = kernel
        ctx.stride = stride
        if return_mask:
            mask_ = mask.detach().clone()
            mask_.requires_grad = False
            CUDA_ADAPOOL1d_EM.mask = mask_
        output = torch.nan_to_num(output)
        if no_batch:
            return output.squeeze_(0)
        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):

        grad_input = torch.zeros_like(ctx.saved_tensors[0])

        saved = [grad_output] + list(ctx.saved_tensors) + [ctx.kernel, ctx.stride, grad_input]
        adapool_cuda.backward_1d_em(*saved)

        return torch.nan_to_num(saved[-1]), None, None, None



class CUDA_ADAPOOL2d(Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, beta, kernel=2, stride=None, return_mask=False):

        assert input.dtype==beta.dtype, '`input` and `beta` are not of the same dtype.'
        beta = torch.clamp(beta , 0., 1.)
        no_batch = False
        if len(input.size()) == 3:
            no_batch = True
            input.unsqueeze_(0)
        B, C, H, W = input.shape
        kernel = _pair(kernel)
        if stride is None:
            stride = kernel
        else:
            stride = _pair(stride)

        oH = (H - kernel[0]) // stride[0] + 1
        oW = (W - kernel[1]) // stride[1] + 1

        output = input.new_zeros((B, C, oH, oW))
        if return_mask:
            mask = input.new_zeros((B, kernel[0] * oH, kernel[1] * oW))
        else:
            mask = input.new_zeros((1))

        adapool_cuda.forward_2d(input.contiguous(), beta, kernel, stride, output, return_mask, mask)
        ctx.save_for_backward(input, beta)
        ctx.kernel = kernel
        ctx.stride = stride
        if return_mask:
            mask_ = mask.detach().clone()
            mask_.requires_grad = False
            CUDA_ADAPOOL2d.mask = mask_
        output = torch.nan_to_num(output)
        if no_batch:
            return output.squeeze_(0)
        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):

        grad_input = torch.zeros_like(ctx.saved_tensors[0])
        grad_beta = torch.zeros_like(ctx.saved_tensors[1])

        saved = [grad_output] + list(ctx.saved_tensors) + [ctx.kernel, ctx.stride, grad_input, grad_beta]
        adapool_cuda.backward_2d(*saved)

        return torch.nan_to_num(saved[-2]), torch.nan_to_num(saved[-1]), None, None, None





class CUDA_ADAPOOL2d_EDSCW(Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, kernel=2, stride=None, return_mask=False):
        no_batch = False
        if len(input.size()) == 3:
            no_batch = True
            input.unsqueeze_(0)
        B, C, H, W = input.shape
        kernel = _pair(kernel)
        if stride is None:
            stride = kernel
        else:
            stride = _pair(stride)

        oH = (H - kernel[0]) // stride[0] + 1
        oW = (W - kernel[1]) // stride[1] + 1

        output = input.new_zeros((B, C, oH, oW))
        if return_mask:
            mask = input.new_zeros((B, kernel[0] * oH, kernel[1] * oW))
        else:
            mask = input.new_zeros((1))

        adapool_cuda.forward_2d_edscw(input.contiguous(), kernel, stride, output, return_mask, mask)
        ctx.save_for_backward(input)
        ctx.kernel = kernel
        ctx.stride = stride
        if return_mask:
            mask_ = mask.detach().clone()
            mask_.requires_grad = False
            CUDA_ADAPOOL2d_EDSCW.mask = mask_
        output = torch.nan_to_num(output)
        if no_batch:
            return output.squeeze_(0)
        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):

        grad_input = torch.zeros_like(ctx.saved_tensors[0])

        saved = [grad_output] + list(ctx.saved_tensors) + [ctx.kernel, ctx.stride, grad_input]
        adapool_cuda.backward_2d_edscw(*saved)

        return torch.nan_to_num(saved[-1]), None, None, None


class CUDA_IDWPOOL2d(Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, kernel=2, stride=None, return_mask=False):
        no_batch = False
        if len(input.size()) == 3:
            no_batch = True
            input.unsqueeze_(0)
        B, C, H, W = input.shape
        kernel = _pair(kernel)
        if stride is None:
            stride = kernel
        else:
            stride = _pair(stride)

        oH = (H - kernel[0]) // stride[0] + 1
        oW = (W - kernel[1]) // stride[1] + 1

        output = input.new_zeros((B, C, oH, oW))
        if return_mask:
            mask = input.new_zeros((B, kernel[0] * oH, kernel[1] * oW))
        else:
            mask = input.new_zeros((1))

        adapool_cuda.forward_2d_idw(input.contiguous(), kernel, stride, output, return_mask, mask)
        ctx.save_for_backward(input)
        ctx.kernel = kernel
        ctx.stride = stride
        if return_mask:
            mask_ = mask.detach().clone()
            mask_.requires_grad = False
            CUDA_ADAPOOL2d_EDSCW.mask = mask_
        output = torch.nan_to_num(output)
        if no_batch:
            return output.squeeze_(0)
        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):

        grad_input = torch.zeros_like(ctx.saved_tensors[0])

        saved = [grad_output] + list(ctx.saved_tensors) + [ctx.kernel, ctx.stride, grad_input]
        adapool_cuda.backward_2d_idw(*saved)

        return torch.nan_to_num(saved[-1]), None, None, None


class CUDA_ADAPOOL2d_EM(Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, kernel=2, stride=None, return_mask=False):

        no_batch = False
        if len(input.size()) == 3:
            no_batch = True
            input.unsqueeze_(0)
        B, C, H, W = input.shape
        kernel = _pair(kernel)
        if stride is None:
            stride = kernel
        else:
            stride = _pair(stride)

        oH = (H - kernel[0]) // stride[0] + 1
        oW = (W - kernel[1]) // stride[1] + 1

        output = input.new_zeros((B, C, oH, oW))
        if return_mask:
            mask = input.new_zeros((B, kernel[0] * oH, kernel[1] * oW))
        else:
            mask = input.new_zeros((1))

        adapool_cuda.forward_2d_em(input.contiguous(), kernel, stride, output, return_mask, mask)
        ctx.save_for_backward(input)
        ctx.kernel = kernel
        ctx.stride = stride
        if return_mask:
            mask_ = mask.detach().clone()
            mask_.requires_grad = False
            CUDA_ADAPOOL2d_EM.mask = mask_
        output = torch.nan_to_num(output)
        if no_batch:
            return output.squeeze_(0)
        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):

        grad_input = torch.zeros_like(ctx.saved_tensors[0])

        saved = [grad_output] + list(ctx.saved_tensors) + [ctx.kernel, ctx.stride, grad_input]
        adapool_cuda.backward_2d_em(*saved)

        return torch.nan_to_num(saved[-1]), None, None, None





class CUDA_ADAPOOL3d(Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, beta, kernel=2, stride=None, return_mask=False):
        assert input.dtype==beta.dtype, '`input` and `beta` are not of the same dtype.'
        beta = torch.clamp(beta , 0., 1.)
        no_batch = False
        if len(input.size()) == 3:
            no_batch = True
            input.unsqueeze_(0)
        B, C, D, H, W = input.shape
        kernel = _triple(kernel)
        if stride is None:
            stride = kernel
        else:
            stride = _triple(stride)
        oD = (D - kernel[0]) // stride[0] + 1
        oH = (H - kernel[1]) // stride[1] + 1
        oW = (W - kernel[2]) // stride[2] + 1

        output = input.new_zeros((B, C, oD, oH, oW))
        if return_mask:
            mask = input.new_zeros((B, kernel[0] * oD, kernel[1] * oH, kernel[2] * oW))
        else:
            mask = input.new_zeros((1))

        adapool_cuda.forward_3d(input.contiguous(), beta, kernel, stride, output, return_mask, mask)
        ctx.save_for_backward(input, beta)
        ctx.kernel = kernel
        ctx.stride = stride
        if return_mask:
            mask_ = mask.detach().clone()
            mask_.requires_grad = False
            CUDA_ADAPOOL3d.mask = mask_
        output = torch.nan_to_num(output)
        if no_batch:
            return output.squeeze_(0)
        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):

        grad_input = torch.zeros_like(ctx.saved_tensors[0])
        grad_beta = torch.zeros_like(ctx.saved_tensors[1])

        saved = [grad_output] + list(ctx.saved_tensors) + [ctx.kernel, ctx.stride, grad_input, grad_beta]
        adapool_cuda.backward_3d(*saved)

        return torch.nan_to_num(saved[-2]), torch.nan_to_num(saved[-1]), None, None, None





class CUDA_ADAPOOL3d_EDSCW(Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, kernel=2, stride=None, return_mask=False):

        no_batch = False
        if len(input.size()) == 3:
            no_batch = True
            input.unsqueeze_(0)
        B, C, D, H, W = input.shape
        kernel = _triple(kernel)
        if stride is None:
            stride = kernel
        else:
            stride = _triple(stride)
        oD = (D - kernel[0]) // stride[0] + 1
        oH = (H - kernel[1]) // stride[1] + 1
        oW = (W - kernel[2]) // stride[2] + 1

        output = input.new_zeros((B, C, oD, oH, oW))
        if return_mask:
            mask = input.new_zeros((B, kernel[0] * oD, kernel[1] * oH, kernel[2] * oW))
        else:
            mask = input.new_zeros((1))

        adapool_cuda.forward_3d_edscw(input.contiguous(), kernel, stride, output, return_mask, mask)
        ctx.save_for_backward(input)
        ctx.kernel = kernel
        ctx.stride = stride
        if return_mask:
            mask_ = mask.detach().clone()
            mask_.requires_grad = False
            CUDA_ADAPOOL3d_EDSCW.mask = mask_
        output = torch.nan_to_num(output)
        if no_batch:
            return output.squeeze_(0)
        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):

        grad_input = torch.zeros_like(ctx.saved_tensors[0])

        saved = [grad_output] + list(ctx.saved_tensors) + [ctx.kernel, ctx.stride, grad_input]
        adapool_cuda.backward_3d_edscw(*saved)

        return torch.nan_to_num(saved[-1]), None, None, None


class CUDA_IDWPOOL3d(Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, kernel=2, stride=None, return_mask=False):

        no_batch = False
        if len(input.size()) == 3:
            no_batch = True
            input.unsqueeze_(0)
        B, C, D, H, W = input.shape
        kernel = _triple(kernel)
        if stride is None:
            stride = kernel
        else:
            stride = _triple(stride)
        oD = (D - kernel[0]) // stride[0] + 1
        oH = (H - kernel[1]) // stride[1] + 1
        oW = (W - kernel[2]) // stride[2] + 1

        output = input.new_zeros((B, C, oD, oH, oW))
        if return_mask:
            mask = input.new_zeros((B, kernel[0] * oD, kernel[1] * oH, kernel[2] * oW))
        else:
            mask = input.new_zeros((1))

        adapool_cuda.forward_3d_idw(input.contiguous(), kernel, stride, output, return_mask, mask)
        ctx.save_for_backward(input)
        ctx.kernel = kernel
        ctx.stride = stride
        if return_mask:
            mask_ = mask.detach().clone()
            mask_.requires_grad = False
            CUDA_ADAPOOL3d_EDSCW.mask = mask_
        output = torch.nan_to_num(output)
        if no_batch:
            return output.squeeze_(0)
        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):

        grad_input = torch.zeros_like(ctx.saved_tensors[0])

        saved = [grad_output] + list(ctx.saved_tensors) + [ctx.kernel, ctx.stride, grad_input]
        adapool_cuda.backward_3d_idw(*saved)

        return torch.nan_to_num(saved[-1]), None, None, None


class CUDA_ADAPOOL3d_EM(Function):

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, input, kernel=2, stride=None, return_mask=False):

        no_batch = False
        if len(input.size()) == 3:
            no_batch = True
            input.unsqueeze_(0)
        B, C, D, H, W = input.shape
        kernel = _triple(kernel)
        if stride is None:
            stride = kernel
        else:
            stride = _triple(stride)
        oD = (D - kernel[0]) // stride[0] + 1
        oH = (H - kernel[1]) // stride[1] + 1
        oW = (W - kernel[2]) // stride[2] + 1

        output = input.new_zeros((B, C, oD, oH, oW))
        if return_mask:
            mask = input.new_zeros((B, kernel[0] * oD, kernel[1] * oH, kernel[2] * oW))
        else:
            mask = input.new_zeros((1))

        adapool_cuda.forward_3d_em(input.contiguous(), kernel, stride, output, return_mask, mask)
        ctx.save_for_backward(input)
        ctx.kernel = kernel
        ctx.stride = stride
        if return_mask:
            mask_ = mask.detach().clone()
            mask_.requires_grad = False
            CUDA_ADAPOOL3d_EM.mask = mask_
        output = torch.nan_to_num(output)
        if no_batch:
            return output.squeeze_(0)
        return output

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_output):

        grad_input = torch.zeros_like(ctx.saved_tensors[0])

        saved = [grad_output] + list(ctx.saved_tensors) + [ctx.kernel, ctx.stride, grad_input]
        adapool_cuda.backward_3d_em(*saved)

        return torch.nan_to_num(saved[-1]), None, None, None



'''
---  S T A R T  O F  F U N C T I O N  A D A P O O L 1 D ---
    [About]
        Class used for downsampling based on adaptive exponential pooling (AdaPool)
    [Args]
        - beta: Tensor, for the beta parameter to be used during pooling. Should be the same as the
                size of the output tensor.
        - kernel_size: Integer or Tuple, for the kernel size to be used for downsampling. If an `Integer`
                       is used, a `Tuple` is created for the rest of the dimensions. Defaults to 2.
        - stride: Integer or Tuple, for the steps taken between kernels (i.e. strides). If `None` the
                  strides become equal to the `kernel_size` tuple. Defaults to `None`.
        - return_mask: Bool, for returning the computed regional mask used for pooling.
        - native: Bool, for using the singl-native adapool operation or using a combination of `EM` and
                  `EDSCW` pooling. Currently only non-native implementation is supported by CUDA-compatible
                  GPUs. This is due to hardware constraints based on the threads per block. Defaults to `False`.
    [Returns]
         - PyTorch Tensor, subsampled based on the specified `kernel_size` and `stride`
'''
def adapool1d(x, beta=None, kernel_size=2, stride=None, return_mask=False, native=False):
    if stride is None:
        stride = kernel_size
    kernel_size = _single(kernel_size)
    stride = _single(stride)

    assert beta is not None, 'Function called with `None`/undefined `beta` parameter.'
    shape = [(x.shape[-1] - kernel_size[-1]) // stride[-1] + 1]
    beta_shape = list(beta.shape)
    shape_d = [s*kernel_size[i] for i,s in enumerate(shape)]

    assert shape == beta_shape or beta_shape==[1], 'Required `beta` shape {0} does not match given shape {1}'.format(shape, beta_shape)
    assert x.is_cuda, 'Only CUDA implementation supported!'

    if not native:
        x = beta*CUDA_ADAPOOL1d_EDSCW.apply(x, kernel_size, stride, return_mask) + (1.-beta)*CUDA_ADAPOOL1d_EM.apply(x, kernel_size, stride, return_mask)
    else:
        x = CUDA_ADAPOOL1d.apply(x, beta, kernel_size, stride, return_mask)

    # Replace `NaN's
    if not return_mask:
        return torch.nan_to_num(x)
    else:
        if not native:
            return torch.nan_to_num(x), (CUDA_ADAPOOL1d_EDSCW.mask,CUDA_ADAPOOL1d_EM.mask,beta)
        else:
            return torch.nan_to_num(x), CUDA_ADAPOOL1d.mask
'''
---  E N D  O F  F U N C T I O N  A D A P O O L 1 D ---
'''



'''
---  S T A R T  O F  F U N C T I O N  A D A P O O L 2 D ---
    [About]
        Class used for downsampling based on adaptive exponential pooling (AdaPool)
    [Args]
        - beta: Tensor, for the beta parameter to be used during pooling. Should be the same as the
                size of the output tensor.
        - kernel_size: Integer or Tuple, for the kernel size to be used for downsampling. If an `Integer`
                       is used, a `Tuple` is created for the rest of the dimensions. Defaults to 2.
        - stride: Integer or Tuple, for the steps taken between kernels (i.e. strides). If `None` the
                  strides become equal to the `kernel_size` tuple. Defaults to `None`.
        - return_mask: Bool, for returning the computed regional mask used for pooling.
        - native: Bool, for using the singl-native adapool operation or using a combination of `EM` and
                  `EDSCW` pooling. Currently only non-native implementation is supported by CUDA-compatible
                  GPUs. This is due to hardware constraints based on the threads per block. Defaults to `False`.
    [Returns]
         - PyTorch Tensor, subsampled based on the specified `kernel_size` and `stride`
'''
def adapool2d(x, beta=None, kernel_size=2, stride=None, return_mask=False, native=False):
    if stride is None:
        stride = kernel_size
    kernel_size = _pair(kernel_size)
    stride = _pair(stride)

    assert beta is not None, 'Function called with `None`/undefined `beta` parameter.'
    shape = [(x.shape[-2] - kernel_size[-2]) // stride[-2] + 1 ,
             (x.shape[-1] - kernel_size[-1]) // stride[-1] + 1]
    beta_shape = list(beta.shape)
    shape_d = [s*kernel_size[i] for i,s in enumerate(shape)]

    assert shape == beta_shape or beta_shape==[1,1], 'Required `beta` shape {0} does not match given shape {1}'.format(shape, beta_shape)
    assert x.is_cuda, 'Only CUDA implementation supported!'

    if not native:
        x = beta*CUDA_ADAPOOL2d_EDSCW.apply(x, kernel_size, stride, return_mask) + (1.-beta)*CUDA_ADAPOOL2d_EM.apply(x, kernel_size, stride, return_mask)
    else:
        x = CUDA_ADAPOOL2d.apply(x, beta, kernel_size, stride, return_mask)

    # Replace `NaN's
    if not return_mask:
        return torch.nan_to_num(x)
    else:
        if not native:
            return torch.nan_to_num(x), (CUDA_ADAPOOL2d_EDSCW.mask,CUDA_ADAPOOL2d_EM.mask,beta)
        else:
            return torch.nan_to_num(x), CUDA_ADAPOOL2d.mask
'''
---  E N D  O F  F U N C T I O N  A D A P O O L 2 D ---
'''



'''
---  S T A R T  O F  F U N C T I O N  A D A P O O L 3 D ---
    [About]
        Class used for downsampling based on adaptive exponential pooling (AdaPool)
    [Args]
        - beta: Tensor, for the beta parameter to be used during pooling. Should be the same as the
                size of the output tensor.
        - kernel_size: Integer or Tuple, for the kernel size to be used for downsampling. If an `Integer`
                       is used, a `Tuple` is created for the rest of the dimensions. Defaults to 2.
        - stride: Integer or Tuple, for the steps taken between kernels (i.e. strides). If `None` the
                  strides become equal to the `kernel_size` tuple. Defaults to `None`.
        - return_mask: Bool, for returning the computed regional mask used for pooling.
        - native: Bool, for using the singl-native adapool operation or using a combination of `EM` and
                  `EDSCW` pooling. Currently only non-native implementation is supported by CUDA-compatible
                  GPUs. This is due to hardware constraints based on the threads per block. Defaults to `False`.
    [Returns]
         - PyTorch Tensor, subsampled based on the specified `kernel_size` and `stride`
'''
def adapool3d(x, beta=None, kernel_size=2, stride=None, return_mask=False, native=False):
    if stride is None:
        stride = kernel_size
    kernel_size = _triple(kernel_size)
    stride = _triple(stride)

    assert beta is not None, 'Function called with `None`/undefined `beta` parameter.'
    shape = [(x.shape[-3] - kernel_size[-3]) // stride[-3] + 1 ,
             (x.shape[-2] - kernel_size[-2]) // stride[-2] + 1 ,
             (x.shape[-1] - kernel_size[-1]) // stride[-1] + 1]
    beta_shape = list(beta.shape)
    shape_d = [s*kernel_size[i] for i,s in enumerate(shape)]

    assert shape==beta_shape or beta_shape==[1,1,1], 'Required `beta` shape {0} does not match given shape {1}'.format(shape, beta_shape)
    assert x.is_cuda, 'Only CUDA implementation supported!'

    if not native:
        x = beta*CUDA_ADAPOOL3d_EDSCW.apply(x, kernel_size, stride, return_mask) + (1. - beta)*CUDA_ADAPOOL3d_EM.apply(x, kernel_size, stride, return_mask)
    else:
        x = CUDA_ADAPOOL3d.apply(x, beta, kernel_size, stride, return_mask)

    # Replace `NaN's
    if not return_mask:
        return torch.nan_to_num(x)
    else:
        if not native:
            return torch.nan_to_num(x), (CUDA_ADAPOOL3d_EDSCW.mask,CUDA_ADAPOOL3d_EM.mask,beta)
        else:
            return torch.nan_to_num(x), CUDA_ADAPOOL3d.mask
'''
---  E N D  O F  F U N C T I O N  A D A P O O L 3 D ---
'''



'''
---  E N D  O F  F U N C T I O N  E J V S W P O O L 1 D ---
    [About]
        Class used for downsampling based on Exponential Jaccard Vector Similarity Weighting (SoftAvgPool)
    [Args]
        - kernel_size: Integer or Tuple, for the kernel size to be used for downsampling. If an `Integer`
                       is used, a `Tuple` is created for the rest of the dimensions. Defaults to 2.
        - stride: Integer or Tuple, for the steps taken between kernels (i.e. strides). If `None` the
                  strides become equal to the `kernel_size` tuple. Defaults to `None`.
        - return_mask: Bool, for returning the computed regional mask used for pooling.
    [Returns]
         - PyTorch Tensor, subsampled based on the specified `kernel_size` and `stride`
'''
def edscwpool1d(x, kernel_size=2, stride=None, return_mask=False):
    if stride is None:
        stride = kernel_size
    kernel_size = _single(kernel_size)
    stride = _single(stride)

    assert x.is_cuda, 'Only CUDA implementation supported!'

    x = CUDA_ADAPOOL1d_EDSCW.apply(x, kernel_size, stride, return_mask)

    # Replace `NaN's
    if not return_mask:
        return torch.nan_to_num(x)
    else:
        return torch.nan_to_num(x), CUDA_ADAPOOL1d_EDSCW.mask
'''
---  E N D  O F  F U N C T I O N  E J V S W P O O L 1 D ---
'''



'''
--- S T A R T  O F  F U N C T I O N  E M P O O L 1 D ---
    [About]
        Class used for downsampling based on Exponential Maximum weighting (SoftPool)
    [Args]
        - kernel_size: Integer or Tuple, for the kernel size to be used for downsampling. If an `Integer`
                       is used, a `Tuple` is created for the rest of the dimensions. Defaults to 2.
        - stride: Integer or Tuple, for the steps taken between kernels (i.e. strides). If `None` the
                  strides become equal to the `kernel_size` tuple. Defaults to `None`.
        - return_mask: Bool, for returning the computed regional mask used for pooling.
    [Returns]
         - PyTorch Tensor, subsampled based on the specified `kernel_size` and `stride`
'''
def empool1d(x, kernel_size=2, stride=None, return_mask=False):
    if stride is None:
        stride = kernel_size
    kernel_size = _single(kernel_size)
    stride = _single(stride)

    assert x.is_cuda, 'Only CUDA implementation supported!'

    x = CUDA_ADAPOOL1d_EM.apply(x, kernel_size, stride, return_mask)

    # Replace `NaN's
    if not return_mask:
        return torch.nan_to_num(x)
    else:
        return torch.nan_to_num(x), CUDA_ADAPOOL1d_EM.mask
'''
--- E N D  O F  F U N C T I O N  E M P O O L 1 D ---
'''



'''
---  S T A R T  O F  F U N C T I O N  I D W P O O L 1 D ---
    [About]
        Class used for downsampling based on Inverse Distance Weighting (IDW)
    [Args]
        - kernel_size: Integer or Tuple, for the kernel size to be used for downsampling. If an `Integer`
                       is used, a `Tuple` is created for the rest of the dimensions. Defaults to 2.
        - stride: Integer or Tuple, for the steps taken between kernels (i.e. strides). If `None` the
                  strides become equal to the `kernel_size` tuple. Defaults to `None`.
        - return_mask: Bool, for returning the computed regional mask used for pooling.
    [Returns]
         - PyTorch Tensor, subsampled based on the specified `kernel_size` and `stride`
'''
def idwpool1d(x, kernel_size=2, stride=None, return_mask=False):
    if stride is None:
        stride = kernel_size
    kernel_size = _single(kernel_size)
    stride = _single(stride)

    assert x.is_cuda, 'Only CUDA implementation supported!'

    x = CUDA_IDWPOOL1d.apply(x, kernel_size, stride, return_mask)

    # Replace `NaN's
    if not return_mask:
        return torch.nan_to_num(x)
    else:
        return torch.nan_to_num(x), CUDA_IDWPOOL1d.mask
'''
---  E N D  O F  F U N C T I O N  I D W P O O L 1 D ---
'''



'''
---  E N D  O F  F U N C T I O N  E J V S W P O O L 2 D ---
    [About]
        Class used for downsampling based on Exponential Jaccard Vector Similarity Weighting (SoftAvgPool)
    [Args]
        - kernel_size: Integer or Tuple, for the kernel size to be used for downsampling. If an `Integer`
                       is used, a `Tuple` is created for the rest of the dimensions. Defaults to 2.
        - stride: Integer or Tuple, for the steps taken between kernels (i.e. strides). If `None` the
                  strides become equal to the `kernel_size` tuple. Defaults to `None`.
        - return_mask: Bool, for returning the computed regional mask used for pooling.
    [Returns]
         - PyTorch Tensor, subsampled based on the specified `kernel_size` and `stride`
'''
def edscwpool2d(x, kernel_size=2, stride=None, return_mask=False):
    if stride is None:
        stride = kernel_size
    kernel_size = _pair(kernel_size)
    stride = _pair(stride)

    assert x.is_cuda, 'Only CUDA implementation supported!'

    x = CUDA_ADAPOOL2d_EDSCW.apply(x, kernel_size, stride, return_mask)

    # Replace `NaN's
    if not return_mask:
        return torch.nan_to_num(x)
    else:
        return torch.nan_to_num(x), CUDA_ADAPOOL2d_EDSCW.mask
'''
---  E N D  O F  F U N C T I O N  E J V S W P O O L 2 D ---
'''



'''
--- S T A R T  O F  F U N C T I O N  E M P O O L 2 D ---
    [About]
        Class used for downsampling based on Exponential Maximum weighting (SoftPool)
    [Args]
        - kernel_size: Integer or Tuple, for the kernel size to be used for downsampling. If an `Integer`
                       is used, a `Tuple` is created for the rest of the dimensions. Defaults to 2.
        - stride: Integer or Tuple, for the steps taken between kernels (i.e. strides). If `None` the
                  strides become equal to the `kernel_size` tuple. Defaults to `None`.
        - return_mask: Bool, for returning the computed regional mask used for pooling.
    [Returns]
         - PyTorch Tensor, subsampled based on the specified `kernel_size` and `stride`
'''
def empool2d(x, kernel_size=2, stride=None, return_mask=False):
    if stride is None:
        stride = kernel_size
    kernel_size = _pair(kernel_size)
    stride = _pair(stride)

    assert x.is_cuda, 'Only CUDA implementation supported!'

    x = CUDA_ADAPOOL2d_EM.apply(x, kernel_size, stride, return_mask)

    # Replace `NaN's
    if not return_mask:
        return torch.nan_to_num(x)
    else:
        return torch.nan_to_num(x), CUDA_ADAPOOL2d_EM.mask
'''
--- S T A R T  O F  F U N C T I O N  E M P O O L 2 D ---
'''



'''
---  S T A R T  O F  F U N C T I O N  I D W P O O L 2 D ---
    [About]
        Class used for downsampling based on Inverse Distance Weighting (IDW)
    [Args]
        - kernel_size: Integer or Tuple, for the kernel size to be used for downsampling. If an `Integer`
                       is used, a `Tuple` is created for the rest of the dimensions. Defaults to 2.
        - stride: Integer or Tuple, for the steps taken between kernels (i.e. strides). If `None` the
                  strides become equal to the `kernel_size` tuple. Defaults to `None`.
        - return_mask: Bool, for returning the computed regional mask used for pooling.
    [Returns]
         - PyTorch Tensor, subsampled based on the specified `kernel_size` and `stride`
'''
def idwpool2d(x, kernel_size=2, stride=None, return_mask=False):
    if stride is None:
        stride = kernel_size
    kernel_size = _pair(kernel_size)
    stride = _pair(stride)

    assert x.is_cuda, 'Only CUDA implementation supported!'

    x = CUDA_IDWPOOL2d.apply(x, kernel_size, stride, return_mask)

    # Replace `NaN's
    if not return_mask:
        return torch.nan_to_num(x)
    else:
        return torch.nan_to_num(x), CUDA_IDWPOOL2d.mask
'''
---  E N D  O F  F U N C T I O N  I D W P O O L 2 D ---
'''



'''
---  E N D  O F  F U N C T I O N  E J V S W P O O L 3 D ---
    [About]
        Class used for downsampling based on Exponential Jaccard Vector Similarity Weighting (SoftAvgPool)
    [Args]
        - kernel_size: Integer or Tuple, for the kernel size to be used for downsampling. If an `Integer`
                       is used, a `Tuple` is created for the rest of the dimensions. Defaults to 2.
        - stride: Integer or Tuple, for the steps taken between kernels (i.e. strides). If `None` the
                  strides become equal to the `kernel_size` tuple. Defaults to `None`.
        - return_mask: Bool, for returning the computed regional mask used for pooling.
    [Returns]
         - PyTorch Tensor, subsampled based on the specified `kernel_size` and `stride`
'''
def edscwpool3d(x, kernel_size=2, stride=None, return_mask=False):
    if stride is None:
        stride = kernel_size
    kernel_size = _triple(kernel_size)
    stride = _triple(stride)

    assert x.is_cuda, 'Only CUDA implementation supported!'

    x = CUDA_ADAPOOL3d_EDSCW.apply(x, kernel_size, stride, return_mask)

    # Replace `NaN's
    if not return_mask:
        return torch.nan_to_num(x)
    else:
        return torch.nan_to_num(x), CUDA_ADAPOOL3d_EDSCW.mask
'''
---  E N D  O F  F U N C T I O N  E J V S W P O O L 3 D ---
'''



'''
--- S T A R T  O F  F U N C T I O N  E M P O O L 3 D ---
    [About]
        Class used for downsampling based on Exponential Maximum weighting (SoftPool)
    [Args]
        - kernel_size: Integer or Tuple, for the kernel size to be used for downsampling. If an `Integer`
                       is used, a `Tuple` is created for the rest of the dimensions. Defaults to 2.
        - stride: Integer or Tuple, for the steps taken between kernels (i.e. strides). If `None` the
                  strides become equal to the `kernel_size` tuple. Defaults to `None`.
        - return_mask: Bool, for returning the computed regional mask used for pooling.
    [Returns]
         - PyTorch Tensor, subsampled based on the specified `kernel_size` and `stride`
'''
def empool3d(x, kernel_size=2, stride=None, return_mask=False):
    if stride is None:
        stride = kernel_size
    kernel_size = _triple(kernel_size)
    stride = _triple(stride)

    assert x.is_cuda, 'Only CUDA implementation supported!'

    x = CUDA_ADAPOOL3d_EM.apply(x, kernel_size, stride, return_mask)

    # Replace `NaN's
    if not return_mask:
        return torch.nan_to_num(x)
    else:
        return torch.nan_to_num(x), CUDA_ADAPOOL3d_EM.mask
'''
--- E N D  O F  F U N C T I O N  E M P O O L 3 D ---
'''



'''
---  S T A R T  O F  F U N C T I O N  I D W P O O L 3 D ---
    [About]
        Class used for downsampling based on Inverse Distance Weighting (IDW)
    [Args]
        - kernel_size: Integer or Tuple, for the kernel size to be used for downsampling. If an `Integer`
                       is used, a `Tuple` is created for the rest of the dimensions. Defaults to 2.
        - stride: Integer or Tuple, for the steps taken between kernels (i.e. strides). If `None` the
                  strides become equal to the `kernel_size` tuple. Defaults to `None`.
        - return_mask: Bool, for returning the computed regional mask used for pooling.
    [Returns]
         - PyTorch Tensor, subsampled based on the specified `kernel_size` and `stride`
'''
def idwpool3d(x, kernel_size=2, stride=None, return_mask=False):
    if stride is None:
        stride = kernel_size
    kernel_size = _triple(kernel_size)
    stride = _triple(stride)

    assert x.is_cuda, 'Only CUDA implementation supported!'

    x = CUDA_IDWPOOL3d.apply(x, kernel_size, stride, return_mask)

    # Replace `NaN's
    if not return_mask:
        return torch.nan_to_num(x)
    else:
        return torch.nan_to_num(x), CUDA_IDWPOOL3d.mask
'''
---  E N D  O F  F U N C T I O N  I D W P O O L 3 D ---
'''



'''
--- S T A R T  O F  F U N C T I O N  A D A U N P O O L ---
    [About]
        Function for upsampling provided the mask(s) used for pooling
    [Args]
        - mask: PyTorch Tensor containing the mask or tuple object containing the two masks used
        alongside the `beta` value.
        - interpolate: Bool, use interpolation for upsampling. If the provided input is already
        upsampled and interpolate is `False`. Only the masks will be applied. Defaults to `True`.
    [Returns]
        - PyTorch Tensor, upsampled to the same size as the provided mask.
'''
def adaunpool(x, mask=None, interpolate=True):
    assert mask is not None, 'Function called with `None`/undefined `mask` parameter.'
    if interpolate:
        if isinstance(mask, tuple):
            mask_edscw = torch.clamp(mask[0], 0., 1.)
            mask_em = torch.clamp(mask[1], 0., 1.)
            x1 = F.interpolate(x*mask[2], size=mask[0].shape[1:], mode='area').transpose(0,1)
            x2 = F.interpolate(x*(1.-mask[2]), size=mask[1].shape[1:], mode='area').transpose(0,1)
            return (x1*mask_edscw.unsqueeze(0) + x2*mask_em.unsqueeze(0)).transpose(0,1)
        else:
            mask = torch.clamp(mask, 0., 1.)
            x = F.interpolate(x, size=mask.shape[1:], mode='area').transpose(0,1)
            return (x*mask.unsqueeze(0) + x*(1.- mask.unsqueeze(0))).transpose(0,1)
    else:
        if isinstance(mask, tuple):
            mask_edscw = torch.clamp(mask[0], 0., 1.)
            mask_em = torch.clamp(mask[1], 0., 1.)
            return (x.transpose(0,1)*mask_edscw.unsqueeze(0) + x.transpose(0,1)*mask_em.unsqueeze(0)).transpose(0,1)
        else:
             mask = torch.clamp(mask, 0., 1.)
             return (x.transpose(0,1)*mask.unsqueeze(0) + x.transpose(0,1)*(1.- mask.unsqueeze(0))).transpose(0,1)
'''
--- E N D  O F  F U N C T I O N  A D A U N P O O L ---
'''



'''
===  S T A R T  O F  C L A S S  A D A P O O L 1 D ===
    [About]
        Class used for downsampling based on adaptive exponential pooling (AdaPool)
    [Args]
        - kernel_size: Integer or Tuple, for the kernel size to be used for downsampling. If an `Integer`
                       is used, a `Tuple` is created for the rest of the dimensions. Defaults to 2.
        - beta: Tensor or Tuple, for the beta parameter to be used during pooling. Should include the
                size of the output tensor.
        - stride: Integer or Tuple, for the steps taken between kernels (i.e. strides). If `None` the
                  strides become equal to the `kernel_size` tuple. Defaults to `None`.
        - beta_trainable: Bool, for parameterizing beta. If `False` beta should be initialized with a
                          tensor through the `beta` parameter.
        - return_mask: Bool, for returning the computed regional mask used for pooling.
        - device: String, for the CUDA device(s) to hold the data. Defaults to `None`
        - dtype: Data type object to be used when initializing `beta`. Defaults to `None`.
        - native: Bool, for using the singl-native adapool operation or using a combination of `EM` and
                  `EDSCW` pooling. Currently only non-native implementation is supported by CUDA-compatible
                  GPUs. This is due to hardware constraints based on the threads per block. Defaults to `False`.
    [Returns]
         - PyTorch Tensor, subsampled based on the specified `kernel_size` and `stride`
'''
class AdaPool1d(torch.nn.Module):
    def __init__(self, kernel_size=2, beta=None, stride=None,
                 beta_trainable=True, return_mask=False, device=None,
                 dtype=None, native=False):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(AdaPool1d, self).__init__()
        if stride is None:
            stride = kernel_size
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride)

        assert isinstance(native, bool), 'Argument `native` should be boolean'
        self.native = native

        assert isinstance(beta, tuple) or torch.is_tensor(beta) or isinstance(beta, int), 'Agument `beta` can only be initialized with Tuple, Tensor or Int type objects and should correspond to size (oD)'
        if isinstance(beta, tuple):
            beta = torch.randn(beta, **factory_kwargs)
        elif isinstance(beta, int):
            beta = torch.randn((beta), **factory_kwargs)
        else:
            beta = beta.to(**factory_kwargs)
        beta = torch.clamp(beta, 0., 1.)

        self.return_mask = return_mask

        if beta_trainable:
            self.register_parameter(name='beta', param=torch.nn.Parameter(beta))
        else:
            self.register_buffer(name='beta', param=torch.nn.Parameter(beta))


    def forward(self, x):
        self.beta.data.clamp(0., 1.)
        return adapool1d(x, beta=self.beta, kernel_size=self.kernel_size, stride=self.stride, return_mask=self.return_mask, native=self.native)
'''
===  S T A R T  O F  C L A S S  A D A P O O L 1 D ===
'''



'''
===  S T A R T  O F  C L A S S  E J V S W P O O L 1 D ===
    [About]
        Class used for downsampling based on Exponential Jaccard Vector Similarity Weighting (SoftAvgPool)
    [Args]
        - kernel_size: Integer or Tuple, for the kernel size to be used for downsampling. If an `Integer`
                       is used, a `Tuple` is created for the rest of the dimensions. Defaults to 2.
        - stride: Integer or Tuple, for the steps taken between kernels (i.e. strides). If `None` the
                  strides become equal to the `kernel_size` tuple. Defaults to `None`.
        - return_mask: Bool, for returning the computed regional mask used for pooling.
    [Returns]
         - PyTorch Tensor, subsampled based on the specified `kernel_size` and `stride`
'''
class EDSCWPool1d(torch.nn.Module):
    def __init__(self, kernel_size=2, stride=None, return_mask=False):
        super(EDSCWPool1d, self).__init__()
        if stride is None:
            stride = kernel_size
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride)
        self.return_mask = return_mask


    def forward(self, x):
        return edscwpool1d(x, kernel_size=self.kernel_size, stride=self.stride, return_mask=self.return_mask)
'''
===  E N D  O F  C L A S S  E J V S W P O O L 1 D ===
'''



'''
===  S T A R T  O F  C L A S S  E M P O O L 1 D ===
    [About]
        Class used for downsampling based on Exponential Maximum weighting (SoftPool)
    [Args]
        - kernel_size: Integer or Tuple, for the kernel size to be used for downsampling. If an `Integer`
                       is used, a `Tuple` is created for the rest of the dimensions. Defaults to 2.
        - stride: Integer or Tuple, for the steps taken between kernels (i.e. strides). If `None` the
                  strides become equal to the `kernel_size` tuple. Defaults to `None`.
        - return_mask: Bool, for returning the computed regional mask used for pooling.
    [Returns]
         - PyTorch Tensor, subsampled based on the specified `kernel_size` and `stride`
'''
class EMPool1d(torch.nn.Module):
    def __init__(self, kernel_size=2, stride=None, return_mask=False):
        super(EMPool1d, self).__init__()
        if stride is None:
            stride = kernel_size
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride)
        self.return_mask = return_mask


    def forward(self, x):
        return empool1d(x, kernel_size=self.kernel_size, stride=self.stride, return_mask=self.return_mask)
'''
===  E N D  O F  C L A S S  E M P O O L 1 D ===
'''



'''
===  S T A R T  O F  C L A S S  I D W P O O L 1 D ===
    [About]
        Class used for downsampling based on Inverse Distance Weighting (IDW)
    [Args]
        - kernel_size: Integer or Tuple, for the kernel size to be used for downsampling. If an `Integer`
                       is used, a `Tuple` is created for the rest of the dimensions. Defaults to 2.
        - stride: Integer or Tuple, for the steps taken between kernels (i.e. strides). If `None` the
                  strides become equal to the `kernel_size` tuple. Defaults to `None`.
        - return_mask: Bool, for returning the computed regional mask used for pooling.
    [Returns]
         - PyTorch Tensor, subsampled based on the specified `kernel_size` and `stride`
'''
class IDWPool1d(torch.nn.Module):
    def __init__(self, kernel_size=2, stride=None, return_mask=False):
        super(IDWPool1d, self).__init__()
        if stride is None:
            stride = kernel_size
        self.kernel_size = _single(kernel_size)
        self.stride = _single(stride)
        self.return_mask = return_mask


    def forward(self, x):
        return idwpool1d(x, kernel_size=self.kernel_size, stride=self.stride, return_mask=self.return_mask)
'''
===  E N D  O F  C L A S S  I D W P O O L 1 D ===
'''



'''
===  S T A R T  O F  C L A S S  A D A P O O L 2 D ===
    [About]
        Class used for downsampling based on adaptive exponential pooling (AdaPool)
    [Args]
        - kernel_size: Integer or Tuple, for the kernel size to be used for downsampling. If an `Integer`
                       is used, a `Tuple` is created for the rest of the dimensions. Defaults to 2.
        - beta: Tensor or Tuple, for the beta parameter to be used during pooling. Should include the
                size of the output tensor.
        - stride: Integer or Tuple, for the steps taken between kernels (i.e. strides). If `None` the
                  strides become equal to the `kernel_size` tuple. Defaults to `None`.
        - beta_trainable: Bool, for parameterizing beta. If `False` beta should be initialized with a
                          tensor through the `beta` parameter.
        - return_mask: Bool, for returning the computed regional mask used for pooling.
        - device: String, for the CUDA device(s) to hold the data. Defaults to `None`
        - dtype: Data type object to be used when initializing `beta`. Defaults to `None`.
        - native: Bool, for using the singl-native adapool operation or using a combination of `EM` and
                  `EDSCW` pooling. Currently only non-native implementation is supported by CUDA-compatible
                  GPUs. This is due to hardware constraints based on the threads per block. Defaults to `False`.
    [Returns]
         - PyTorch Tensor, subsampled based on the specified `kernel_size` and `stride`
'''
class AdaPool2d(torch.nn.Module):
    def __init__(self, kernel_size=2, beta=None, stride=None,
                 beta_trainable=True,return_mask=False, device=None,
                 dtype=None, native=False):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(AdaPool2d, self).__init__()
        if stride is None:
            stride = kernel_size
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)

        assert isinstance(native, bool), 'Argument `native` should be boolean'
        self.native = native

        assert isinstance(beta, tuple) or torch.is_tensor(beta), 'Agument `beta` can only be initialized with Tuple or Tensor type objects and should correspond to size (oH, oW)'
        if isinstance(beta, tuple):
            beta = torch.randn(beta, **factory_kwargs)
        else:
            beta = beta.to(**factory_kwargs)
        beta = torch.clamp(beta, 0., 1.)

        self.return_mask = return_mask

        if beta_trainable:
            self.register_parameter(name='beta', param=torch.nn.Parameter(beta))
        else:
            self.register_buffer(name='beta', param=torch.nn.Parameter(beta))


    def forward(self, x):
        self.beta.data.clamp(0., 1.)
        return adapool2d(x, beta=self.beta, kernel_size=self.kernel_size, stride=self.stride, return_mask=self.return_mask, native=self.native)
'''
===  S T A R T  O F  C L A S S  A D A P O O L 2 D ===
'''



'''
===  S T A R T  O F  C L A S S  E J V S W P O O L 2 D ===
    [About]
        Class used for downsampling based on Exponential Jaccard Vector Similarity Weighting (SoftAvgPool)
    [Args]
        - kernel_size: Integer or Tuple, for the kernel size to be used for downsampling. If an `Integer`
                       is used, a `Tuple` is created for the rest of the dimensions. Defaults to 2.
        - stride: Integer or Tuple, for the steps taken between kernels (i.e. strides). If `None` the
                  strides become equal to the `kernel_size` tuple. Defaults to `None`.
        - return_mask: Bool, for returning the computed regional mask used for pooling.
    [Returns]
         - PyTorch Tensor, subsampled based on the specified `kernel_size` and `stride`
'''
class EDSCWPool2d(torch.nn.Module):
    def __init__(self, kernel_size=2, stride=None,return_mask=False):
        super(EDSCWPool2d, self).__init__()
        if stride is None:
            stride = kernel_size
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.return_mask = return_mask


    def forward(self, x):
        return edscwpool2d(x, kernel_size=self.kernel_size, stride=self.stride, return_mask=self.return_mask)
'''
===  E N D  O F  C L A S S  E J V S W P O O L 2 D ===
'''



'''
===  S T A R T  O F  C L A S S  E M P O O L 2 D ===
    [About]
        Class used for downsampling based on Exponential Maximum weighting (SoftPool)
    [Args]
        - kernel_size: Integer or Tuple, for the kernel size to be used for downsampling. If an `Integer`
                       is used, a `Tuple` is created for the rest of the dimensions. Defaults to 2.
        - stride: Integer or Tuple, for the steps taken between kernels (i.e. strides). If `None` the
                  strides become equal to the `kernel_size` tuple. Defaults to `None`.
        - return_mask: Bool, for returning the computed regional mask used for pooling.
    [Returns]
         - PyTorch Tensor, subsampled based on the specified `kernel_size` and `stride`
'''
class EMPool2d(torch.nn.Module):
    def __init__(self, kernel_size=2, stride=None,return_mask=False):
        super(EMPool2d, self).__init__()
        if stride is None:
            stride = kernel_size
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.return_mask = return_mask


    def forward(self, x):
        return empool2d(x, kernel_size=self.kernel_size, stride=self.stride, return_mask=self.return_mask)
'''
===  E N D  O F  C L A S S  E M P O O L 2 D ===
'''



'''
===  S T A R T  O F  C L A S S  I D W P O O L 2 D ===
    [About]
        Class used for downsampling based on Inverse Distance Weighting (IDW)
    [Args]
        - kernel_size: Integer or Tuple, for the kernel size to be used for downsampling. If an `Integer`
                       is used, a `Tuple` is created for the rest of the dimensions. Defaults to 2.
        - stride: Integer or Tuple, for the steps taken between kernels (i.e. strides). If `None` the
                  strides become equal to the `kernel_size` tuple. Defaults to `None`.
        - return_mask: Bool, for returning the computed regional mask used for pooling.
    [Returns]
         - PyTorch Tensor, subsampled based on the specified `kernel_size` and `stride`
'''
class IDWPool2d(torch.nn.Module):
    def __init__(self, kernel_size=2, stride=None,return_mask=False):
        super(IDWPool2d, self).__init__()
        if stride is None:
            stride = kernel_size
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.return_mask = return_mask


    def forward(self, x):
        return idwpool2d(x, kernel_size=self.kernel_size, stride=self.stride, return_mask=self.return_mask)
'''
===  E N D  O F  C L A S S  I D W P O O L 2 D ===
'''



'''
===  S T A R T  O F  C L A S S  A D A P O O L 3 D ===
    [About]
        Class used for downsampling based on adaptive exponential pooling (AdaPool)
    [Args]
        - kernel_size: Integer or Tuple, for the kernel size to be used for downsampling. If an `Integer`
                       is used, a `Tuple` is created for the rest of the dimensions. Defaults to 2.
        - beta: Tensor or Tuple, for the beta parameter to be used during pooling. Should include the
                size of the output tensor.
        - stride: Integer or Tuple, for the steps taken between kernels (i.e. strides). If `None` the
                  strides become equal to the `kernel_size` tuple. Defaults to `None`.
        - beta_trainable: Bool, for parameterizing beta. If `False` beta should be initialized with a
                          tensor through the `beta` parameter.
        - return_mask: Bool, for returning the computed regional mask used for pooling.
        - device: String, for the CUDA device(s) to hold the data. Defaults to `None`
        - dtype: Data type object to be used when initializing `beta`. Defaults to `None`.
        - native: Bool, for using the singl-native adapool operation or using a combination of `EM` and
                  `EDSCW` pooling. Currently only non-native implementation is supported by CUDA-compatible
                  GPUs. This is due to hardware constraints based on the threads per block. Defaults to `False`.
    [Returns]
         - PyTorch Tensor, subsampled based on the specified `kernel_size` and `stride`
'''
class AdaPool3d(torch.nn.Module):
    def __init__(self, kernel_size=2, beta=None, stride=None,
                 beta_trainable=True, return_mask=False,
                 device=None, dtype=None, native=False):
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(AdaPool3d, self).__init__()
        if stride is None:
            stride = kernel_size
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)

        assert isinstance(native, bool), 'Argument `native` should be boolean'
        self.native = native

        assert isinstance(beta, tuple) or torch.is_tensor(beta), 'Agument `beta` can only be initialized with Tuple or Tensor type objects and should correspond to size (oD, oH, oW)'
        if isinstance(beta, tuple):
            beta = torch.randn(beta, **factory_kwargs)
        else:
            beta = beta.to(**factory_kwargs)
        beta = torch.clamp(beta, 0., 1.)

        self.return_mask = return_mask

        if beta_trainable:
            self.register_parameter(name='beta', param=torch.nn.Parameter(beta))
        else:
            self.register_buffer(name='beta', param=torch.nn.Parameter(beta))


    def forward(self, x):
        self.beta.data.clamp(0., 1.)
        return adapool3d(x, beta=self.beta, kernel_size=self.kernel_size, stride=self.stride, return_mask=self.return_mask, native=self.native)

'''
===  S T A R T  O F  C L A S S  E J V S W P O O L 3 D ===
    [About]
        Class used for downsampling based on Exponential Jaccard Vector Similarity Weighting (SoftAvgPool)
    [Args]
        - kernel_size: Integer or Tuple, for the kernel size to be used for downsampling. If an `Integer`
                       is used, a `Tuple` is created for the rest of the dimensions. Defaults to 2.
        - stride: Integer or Tuple, for the steps taken between kernels (i.e. strides). If `None` the
                  strides become equal to the `kernel_size` tuple. Defaults to `None`.
        - return_mask: Bool, for returning the computed regional mask used for pooling.
    [Returns]
         - PyTorch Tensor, subsampled based on the specified `kernel_size` and `stride`
'''
class EDSCWPool3d(torch.nn.Module):
    def __init__(self, kernel_size=2, stride=None, return_mask=False):
        super(EDSCWPool3d, self).__init__()
        if stride is None:
            stride = kernel_size
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.return_mask = return_mask


    def forward(self, x):
        return edscwpool3d(x, kernel_size=self.kernel_size, stride=self.stride, return_mask=self.return_mask)
'''
===  E N D  O F  C L A S S  E J V S W P O O L 3 D ===
'''



'''
===  S T A R T  O F  C L A S S  E M P O O L 3 D ===
    [About]
        Class used for downsampling based on Exponential Maximum weighting (SoftPool)
    [Args]
        - kernel_size: Integer or Tuple, for the kernel size to be used for downsampling. If an `Integer`
                       is used, a `Tuple` is created for the rest of the dimensions. Defaults to 2.
        - stride: Integer or Tuple, for the steps taken between kernels (i.e. strides). If `None` the
                  strides become equal to the `kernel_size` tuple. Defaults to `None`.
        - return_mask: Bool, for returning the computed regional mask used for pooling.
    [Returns]
         - PyTorch Tensor, subsampled based on the specified `kernel_size` and `stride`
'''
class EMPool3d(torch.nn.Module):
    def __init__(self, kernel_size=2, stride=None, return_mask=False):
        super(EMPool3d, self).__init__()
        if stride is None:
            stride = kernel_size
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.return_mask = return_mask


    def forward(self, x):
        return empool3d(x, kernel_size=self.kernel_size, stride=self.stride, return_mask=self.return_mask)
'''
===  E N D  O F  C L A S S  I D W P O O L 3 D ===
'''




'''
===  S T A R T  O F  C L A S S  I D W P O O L 3 D ===
    [About]
        Class used for downsampling based on Inverse Distance Weighting (IDW)
    [Args]
        - kernel_size: Integer or Tuple, for the kernel size to be used for downsampling. If an `Integer`
                       is used, a `Tuple` is created for the rest of the dimensions. Defaults to 2.
        - stride: Integer or Tuple, for the steps taken between kernels (i.e. strides). If `None` the
                  strides become equal to the `kernel_size` tuple. Defaults to `None`.
        - return_mask: Bool, for returning the computed regional mask used for pooling.
    [Returns]
         - PyTorch Tensor, subsampled based on the specified `kernel_size` and `stride`
'''
class IDWPool3d(torch.nn.Module):
    def __init__(self, kernel_size=2, stride=None, return_mask=False):
        super(IDWPool3d, self).__init__()
        if stride is None:
            stride = kernel_size
        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.return_mask = return_mask


    def forward(self, x):
        return idwpool3d(x, kernel_size=self.kernel_size, stride=self.stride, return_mask=self.return_mask)
'''
===  E N D  O F  C L A S S  I D W P O O L 3 D ===
'''



'''
===  S T A R T  O F  C L A S S  A D A U N P O O L 1 D ===
    [About]
        Class used for upsampling provided the mask(s) used for pooling
    [Args]
        - mask: PyTorch Tensor containing the mask or tuple object containing the two masks used
        alongside the `beta` value.
    [Returns]
        - PyTorch Tensor, upsampled to the same size as the provided mask.
'''
class AdaUnpool1d(torch.nn.Module):
    def __init__(self, mask=None):
        super(AdaUnpool1d, self).__init__()
        if stride is None:
            stride = kernel_size
        assert mask != None, '`mask` cannot be `None`!'
        self.mask = mask

    def forward(self, x):
        return adaunpool(x, mask=self.mask)
'''
===  E N D  O F  C L A S S  A D A U N P O O L 1 D ===
'''



'''
===  S T A R T  O F  C L A S S  A D A U N P O O L 2 D ===
    [About]
        Class used for upsampling provided the mask(s) used for pooling
    [Args]
        - mask: PyTorch Tensor containing the mask or tuple object containing the two masks used
        alongside the `beta` value.
    [Returns]
        - PyTorch Tensor, upsampled to the same size as the provided mask.
'''
class AdaUnpool2d(torch.nn.Module):
    def __init__(self, mask=None):
        super(AdaUnpool2d, self).__init__()
        if stride is None:
            stride = kernel_size
        assert mask != None, '`mask` cannot be `None`!'
        self.mask = mask

    def forward(self, x):
        return adaunpool(x, mask=self.mask)
'''
===  E N D  O F  C L A S S  A D A U N P O O L 2 D ===
'''



'''
===  S T A R T  O F  C L A S S  A D A U N P O O L 3 D ===
    [About]
        Class used for upsampling provided the mask(s) used for pooling
    [Args]
        - mask: PyTorch Tensor containing the mask or tuple object containing the two masks used
        alongside the `beta` value.
    [Returns]
        - PyTorch Tensor, upsampled to the same size as the provided mask.
'''
class AdaUnpool3d(torch.nn.Module):
    def __init__(self, mask=None):
        super(AdaUnpool3d, self).__init__()
        if stride is None:
            stride = kernel_size
        assert mask != None, '`mask` cannot be `None`!'
        self.mask = mask

    def forward(self, x):
        return adaunpool(x, mask=self.mask)
'''
===  E N D  O F  C L A S S  A D A U N P O O L 3 D ===
'''
