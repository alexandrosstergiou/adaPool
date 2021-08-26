#include <torch/extension.h>
#include <vector>

// CUDA forward declarations

int AdaPool1dForwardLauncher(const at::Tensor input, const at::Tensor beta,
                             const int batches, const int channels,
                             const int dim, const int kernel_d,
                             const int stride_d, at::Tensor output,
                             const bool return_mask, at::Tensor mask);

int Ada_EDSCW_Pool1dForwardLauncher(const at::Tensor input, const int batches,
                                   const int channels, const int dim,
                                   const int kernel_d, const int stride_d,
                                   at::Tensor output, const bool return_mask,
                                   at::Tensor mask);

int IDW_Pool1dForwardLauncher(const at::Tensor input, const int batches,
                              const int channels, const int dim,
                              const int kernel_d, const int stride_d,
                              at::Tensor output, const bool return_mask,
                              at::Tensor mask);

int Ada_EM_Pool1dForwardLauncher(const at::Tensor input, const int batches,
                                 const int channels, const int dim,
                                 const int kernel_d, const int stride_d,
                                 at::Tensor output, const bool return_mask,
                                 at::Tensor mask);

int AdaPool1dBackwardLauncher(const at::Tensor output_grad, const at::Tensor input,
                              const at::Tensor beta, const int batches,
                              const int channels, const int dim,
                              const int kernel_d, const int stride_d,
                              at::Tensor input_grad, at::Tensor beta_grad);

int Ada_EDSCW_Pool1dBackwardLauncher(const at::Tensor output_grad, const at::Tensor input,
                                    const int batches, const int channels,
                                    const int dim, const int kernel_d,
                                    const int stride_d, at::Tensor input_grad);

int IDW_Pool1dBackwardLauncher(const at::Tensor output_grad, const at::Tensor input,
                               const int batches, const int channels,
                               const int dim, const int kernel_d,
                               const int stride_d, at::Tensor input_grad);

int Ada_EM_Pool1dBackwardLauncher(const at::Tensor output_grad, const at::Tensor input,
                                  const int batches, const int channels,
                                  const int dim, const int kernel_d,
                                  const int stride_d, at::Tensor input_grad);

int AdaPool2dForwardLauncher(const at::Tensor input, const at::Tensor beta,
                             const int batches, const int channels,
                             const int height, const int width,
                             const int kernel_h, const int kernel_w,
                             const int stride_h, const int stride_w,
                             at::Tensor output, const bool return_mask,
                             at::Tensor mask);

int Ada_EDSCW_Pool2dForwardLauncher(const at::Tensor input, const int batches,
                                   const int channels, const int height,
                                   const int width, const int kernel_h,
                                   const int kernel_w, const int stride_h,
                                   const int stride_w, at::Tensor output,
                                   const bool return_mask, at::Tensor mask);

int IDW_Pool2dForwardLauncher(const at::Tensor input, const int batches,
                              const int channels, const int height,
                              const int width, const int kernel_h,
                              const int kernel_w, const int stride_h,
                              const int stride_w, at::Tensor output,
                              const bool return_mask, at::Tensor mask);

int Ada_EM_Pool2dForwardLauncher(const at::Tensor input, const int batches,
                                 const int channels, const int height,
                                 const int width, const int kernel_h,
                                 const int kernel_w, const int stride_h,
                                 const int stride_w, at::Tensor output,
                                 const bool return_mask, at::Tensor mask);

int AdaPool2dBackwardLauncher(const at::Tensor output_grad, const at::Tensor input,
                              const at::Tensor beta, const int batches,
                              const int channels, const int height,
                              const int width, const int kernel_h,
                              const int kernel_w, const int stride_h,
                              const int stride_w, at::Tensor input_grad,
                              at::Tensor beta_grad);

int Ada_EDSCW_Pool2dBackwardLauncher(const at::Tensor output_grad, const at::Tensor input,
                                    const int batches, const int channels,
                                    const int height, const int width,
                                    const int kernel_h, const int kernel_w,
                                    const int stride_h, const int stride_w,
                                    at::Tensor input_grad);

int IDW_Pool2dBackwardLauncher(const at::Tensor output_grad, const at::Tensor input,
                               const int batches, const int channels,
                               const int height, const int width,
                               const int kernel_h, const int kernel_w,
                               const int stride_h, const int stride_w,
                               at::Tensor input_grad);

int Ada_EM_Pool2dBackwardLauncher(const at::Tensor output_grad, const at::Tensor input,
                                  const int batches, const int channels,
                                  const int height, const int width,
                                  const int kernel_h, const int kernel_w,
                                  const int stride_h, const int stride_w,
                                  at::Tensor input_grad);

int AdaPool3dForwardLauncher(const at::Tensor input, const at::Tensor beta,
                             const int batches, const int channels,
                             const int depth, const int height,
                             const int width, const int kernel_d,
                             const int kernel_h, const int kernel_w,
                             const int stride_d, const int stride_h,
                             const int stride_w, at::Tensor output,
                             const bool return_mask, at::Tensor mask);

int Ada_EDSCW_Pool3dForwardLauncher(const at::Tensor input, const int batches,
                                   const int channels, const int depth,
                                   const int height, const int width,
                                   const int kernel_d, const int kernel_h,
                                   const int kernel_w, const int stride_d,
                                   const int stride_h, const int stride_w,
                                   at::Tensor output, const bool return_mask,
                                   at::Tensor mask);

int IDW_Pool3dForwardLauncher(const at::Tensor input, const int batches,
                              const int channels, const int depth,
                              const int height, const int width,
                              const int kernel_d, const int kernel_h,
                              const int kernel_w, const int stride_d,
                              const int stride_h, const int stride_w,
                              at::Tensor output, const bool return_mask,
                              at::Tensor mask);

int Ada_EM_Pool3dForwardLauncher(const at::Tensor input, const int batches,
                                 const int channels, const int depth,
                                 const int height, const int width,
                                 const int kernel_d, const int kernel_h,
                                 const int kernel_w, const int stride_d,
                                 const int stride_h, const int stride_w,
                                 at::Tensor output, const bool return_mask,
                                 at::Tensor mask);

int AdaPool3dBackwardLauncher(const at::Tensor output_grad, const at::Tensor input,
                              const at::Tensor beta, const int batches,
                              const int channels, const int depth,
                              const int height, const int width,
                              const int kernel_d, const int kernel_h,
                              const int kernel_w, const int stride_d,
                              const int stride_h, const int stride_w,
                              at::Tensor input_grad, at::Tensor beta_grad);

int Ada_EDSCW_Pool3dBackwardLauncher(const at::Tensor output_grad, const at::Tensor input,
                                    const int batches, const int channels,
                                    const int depth, const int height,
                                    const int width, const int kernel_d,
                                    const int kernel_h, const int kernel_w,
                                    const int stride_d, const int stride_h,
                                    const int stride_w, at::Tensor input_grad);

int IDW_Pool3dBackwardLauncher(const at::Tensor output_grad, const at::Tensor input,
                               const int batches, const int channels,
                               const int depth, const int height,
                               const int width, const int kernel_d,
                               const int kernel_h, const int kernel_w,
                               const int stride_d, const int stride_h,
                               const int stride_w, at::Tensor input_grad);

int Ada_EM_Pool3dBackwardLauncher(const at::Tensor output_grad, const at::Tensor input,
                                  const int batches, const int channels,
                                  const int depth, const int height,
                                  const int width, const int kernel_d,
                                  const int kernel_h, const int kernel_w,
                                  const int stride_d, const int stride_h,
                                  const int stride_w, at::Tensor input_grad);



// C++ interface

#define CHECK_CUDA(x) AT_ASSERT(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERT(x.is_contiguous(), #x " must be a contiguous tensor");
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x);

int adapool1d_forward(at::Tensor input, at::Tensor beta,
                      const std::tuple<int> kernel, const std::tuple<int> stride,
                      at::Tensor output, bool return_mask,
                      at::Tensor mask) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_INPUT(mask);
    CHECK_INPUT(beta);

    int batches = input.size(0);
    int channels = input.size(1);
    int dim = input.size(2);

    int kernel_d = std::get<0>(kernel);
    int stride_d = std::get<0>(stride);

    AdaPool1dForwardLauncher(input, beta,
                             batches, channels,
                             dim, kernel_d,
                             stride_d, output,
                             return_mask, mask);
    return 1;
}

int ada_edscw_pool1d_forward(at::Tensor input, const std::tuple<int> kernel,
                            const std::tuple<int> stride, at::Tensor output,
                            bool return_mask, at::Tensor mask) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_INPUT(mask);

    int batches = input.size(0);
    int channels = input.size(1);
    int dim = input.size(2);

    int kernel_d = std::get<0>(kernel);
    int stride_d = std::get<0>(stride);

    Ada_EDSCW_Pool1dForwardLauncher(input, batches,
                                   channels, dim,
                                   kernel_d, stride_d,
                                   output, return_mask,
                                   mask);
    return 1;
}

int idw_pool1d_forward(at::Tensor input, const std::tuple<int> kernel,
                       const std::tuple<int> stride, at::Tensor output,
                       bool return_mask, at::Tensor mask) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_INPUT(mask);

    int batches = input.size(0);
    int channels = input.size(1);
    int dim = input.size(2);

    int kernel_d = std::get<0>(kernel);
    int stride_d = std::get<0>(stride);

    IDW_Pool1dForwardLauncher(input, batches,
                              channels, dim,
                              kernel_d, stride_d,
                              output, return_mask,
                              mask);
    return 1;
}

int ada_em_pool1d_forward(at::Tensor input, const std::tuple<int> kernel,
                          const std::tuple<int> stride, at::Tensor output,
                          bool return_mask, at::Tensor mask) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_INPUT(mask);

    int batches = input.size(0);
    int channels = input.size(1);
    int dim = input.size(2);

    int kernel_d = std::get<0>(kernel);
    int stride_d = std::get<0>(stride);

    Ada_EM_Pool1dForwardLauncher(input, batches,
                                 channels, dim,
                                 kernel_d, stride_d,
                                 output, return_mask,
                                 mask);
    return 1;
}


int adapool1d_backward(const at::Tensor output_grad, const at::Tensor input,
                       const at::Tensor beta, const std::tuple<int> kernel,
                       const std::tuple<int> stride, at::Tensor input_grad,
                       at::Tensor beta_grad) {
    CHECK_INPUT(output_grad);
    CHECK_INPUT(input);
    CHECK_INPUT(beta);
    CHECK_INPUT(input_grad);
    CHECK_INPUT(beta_grad);

    int batches = input_grad.size(0);
    int channels = input_grad.size(1);
    int dim = input_grad.size(2);

    int kernel_d = std::get<0>(kernel);
    int stride_d = std::get<0>(stride);

    AdaPool1dBackwardLauncher(output_grad, input,
                              beta, batches,
                              channels, dim,
                              kernel_d, stride_d,
                              input_grad, beta_grad);
    return 1;
}

int ada_edscw_pool1d_backward(const at::Tensor output_grad, const at::Tensor input,
                             const std::tuple<int> kernel, const std::tuple<int> stride,
                             at::Tensor input_grad) {
    CHECK_INPUT(output_grad);
    CHECK_INPUT(input);
    CHECK_INPUT(input_grad);

    int batches = input_grad.size(0);
    int channels = input_grad.size(1);
    int dim = input_grad.size(2);

    int kernel_d = std::get<0>(kernel);
    int stride_d = std::get<0>(stride);

    Ada_EDSCW_Pool1dBackwardLauncher(output_grad, input,
                                    batches, channels,
                                    dim, kernel_d,
                                    stride_d, input_grad);
    return 1;
}

int idw_pool1d_backward(const at::Tensor output_grad, const at::Tensor input,
                        const std::tuple<int> kernel, const std::tuple<int> stride,
                        at::Tensor input_grad) {
    CHECK_INPUT(output_grad);
    CHECK_INPUT(input);
    CHECK_INPUT(input_grad);

    int batches = input_grad.size(0);
    int channels = input_grad.size(1);
    int dim = input_grad.size(2);

    int kernel_d = std::get<0>(kernel);
    int stride_d = std::get<0>(stride);

    IDW_Pool1dBackwardLauncher(output_grad, input,
                               batches, channels,
                               dim, kernel_d,
                               stride_d, input_grad);
    return 1;
}

int ada_em_pool1d_backward(const at::Tensor output_grad, const at::Tensor input,
                           const std::tuple<int> kernel, const std::tuple<int> stride,
                           at::Tensor input_grad) {
    CHECK_INPUT(output_grad);
    CHECK_INPUT(input);
    CHECK_INPUT(input_grad);

    int batches = input_grad.size(0);
    int channels = input_grad.size(1);
    int dim = input_grad.size(2);

    int kernel_d = std::get<0>(kernel);
    int stride_d = std::get<0>(stride);

    Ada_EM_Pool1dBackwardLauncher(output_grad, input,
                                  batches, channels,
                                  dim, kernel_d,
                                  stride_d, input_grad);
    return 1;
}


int adapool2d_forward(at::Tensor input, at::Tensor beta, const std::tuple<int, int> kernel,
                      const std::tuple<int, int> stride, at::Tensor output,
                      bool return_mask, at::Tensor mask) {
    CHECK_INPUT(input);
    CHECK_INPUT(beta);
    CHECK_INPUT(output);
    CHECK_INPUT(mask);

    int batches = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);

    int kernel_h = std::get<0>(kernel);
    int kernel_w = std::get<1>(kernel);
    int stride_h = std::get<0>(stride);
    int stride_w = std::get<1>(stride);

    AdaPool2dForwardLauncher(input, beta,
                             batches, channels,
                             height, width,
                             kernel_h, kernel_w,
                             stride_h, stride_w,
                             output, return_mask,
                             mask);
    return 1;
}

int ada_edscw_pool2d_forward(at::Tensor input, const std::tuple<int, int> kernel,
                            const std::tuple<int, int> stride, at::Tensor output,
                            bool return_mask, at::Tensor mask) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_INPUT(mask);

    int batches = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);

    int kernel_h = std::get<0>(kernel);
    int kernel_w = std::get<1>(kernel);
    int stride_h = std::get<0>(stride);
    int stride_w = std::get<1>(stride);

    Ada_EDSCW_Pool2dForwardLauncher(input, batches,
                                   channels, height,
                                   width, kernel_h,
                                   kernel_w, stride_h,
                                   stride_w, output,
                                   return_mask, mask);
    return 1;
}

int idw_pool2d_forward(at::Tensor input, const std::tuple<int, int> kernel,
                       const std::tuple<int, int> stride, at::Tensor output,
                       bool return_mask, at::Tensor mask) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_INPUT(mask);

    int batches = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);

    int kernel_h = std::get<0>(kernel);
    int kernel_w = std::get<1>(kernel);
    int stride_h = std::get<0>(stride);
    int stride_w = std::get<1>(stride);

    IDW_Pool2dForwardLauncher(input, batches,
                              channels, height,
                              width, kernel_h,
                              kernel_w, stride_h,
                              stride_w, output,
                              return_mask, mask);
    return 1;
}


int ada_em_pool2d_forward(at::Tensor input, const std::tuple<int, int> kernel,
                          const std::tuple<int, int> stride, at::Tensor output,
                          bool return_mask, at::Tensor mask) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_INPUT(mask);

    int batches = input.size(0);
    int channels = input.size(1);
    int height = input.size(2);
    int width = input.size(3);

    int kernel_h = std::get<0>(kernel);
    int kernel_w = std::get<1>(kernel);
    int stride_h = std::get<0>(stride);
    int stride_w = std::get<1>(stride);

    Ada_EM_Pool2dForwardLauncher(input, batches,
                                 channels, height,
                                 width, kernel_h,
                                 kernel_w, stride_h,
                                 stride_w, output,
                                 return_mask, mask);
    return 1;
}

int adapool2d_backward(const at::Tensor output_grad, const at::Tensor input,
                       const at::Tensor beta, const std::tuple<int, int> kernel,
                       const std::tuple<int, int> stride, at::Tensor input_grad,
                       at::Tensor beta_grad) {
    CHECK_INPUT(output_grad);
    CHECK_INPUT(input);
    CHECK_INPUT(beta);
    CHECK_INPUT(input_grad);
    CHECK_INPUT(beta_grad);

    int batches = input_grad.size(0);
    int channels = input_grad.size(1);
    int height = input_grad.size(2);
    int width = input_grad.size(3);

    int kernel_h = std::get<0>(kernel);
    int kernel_w = std::get<1>(kernel);
    int stride_h = std::get<0>(stride);
    int stride_w = std::get<1>(stride);

    AdaPool2dBackwardLauncher(output_grad, input,
                              beta, batches,
                              channels, height,
                              width, kernel_h,
                              kernel_w, stride_h,
                              stride_w, input_grad,
                              beta_grad);
    return 1;
}

int ada_edscw_pool2d_backward(const at::Tensor output_grad, const at::Tensor input,
                             const std::tuple<int, int> kernel, const std::tuple<int, int> stride,
                             at::Tensor input_grad) {
    CHECK_INPUT(output_grad);
    CHECK_INPUT(input);
    CHECK_INPUT(input_grad);

    int batches = input_grad.size(0);
    int channels = input_grad.size(1);
    int height = input_grad.size(2);
    int width = input_grad.size(3);

    int kernel_h = std::get<0>(kernel);
    int kernel_w = std::get<1>(kernel);
    int stride_h = std::get<0>(stride);
    int stride_w = std::get<1>(stride);

    Ada_EDSCW_Pool2dBackwardLauncher(output_grad, input,
                                    batches, channels,
                                    height, width,
                                    kernel_h, kernel_w,
                                    stride_h, stride_w,
                                    input_grad);
    return 1;
}

int idw_pool2d_backward(const at::Tensor output_grad, const at::Tensor input,
                        const std::tuple<int, int> kernel, const std::tuple<int, int> stride,
                        at::Tensor input_grad) {
    CHECK_INPUT(output_grad);
    CHECK_INPUT(input);
    CHECK_INPUT(input_grad);

    int batches = input_grad.size(0);
    int channels = input_grad.size(1);
    int height = input_grad.size(2);
    int width = input_grad.size(3);

    int kernel_h = std::get<0>(kernel);
    int kernel_w = std::get<1>(kernel);
    int stride_h = std::get<0>(stride);
    int stride_w = std::get<1>(stride);

    IDW_Pool2dBackwardLauncher(output_grad, input,
                               batches, channels,
                               height, width,
                               kernel_h, kernel_w,
                               stride_h, stride_w,
                               input_grad);
    return 1;
}

int ada_em_pool2d_backward(const at::Tensor output_grad, const at::Tensor input,
                           const std::tuple<int, int> kernel, const std::tuple<int, int> stride,
                           at::Tensor input_grad) {
    CHECK_INPUT(output_grad);
    CHECK_INPUT(input);
    CHECK_INPUT(input_grad);

    int batches = input_grad.size(0);
    int channels = input_grad.size(1);
    int height = input_grad.size(2);
    int width = input_grad.size(3);

    int kernel_h = std::get<0>(kernel);
    int kernel_w = std::get<1>(kernel);
    int stride_h = std::get<0>(stride);
    int stride_w = std::get<1>(stride);

    Ada_EM_Pool2dBackwardLauncher(output_grad, input,
                                  batches, channels,
                                  height, width,
                                  kernel_h, kernel_w,
                                  stride_h, stride_w,
                                  input_grad);
    return 1;
}

int adapool3d_forward(at::Tensor input, at::Tensor beta,
                      const std::tuple<int, int, int> kernel, const std::tuple<int, int, int> stride,
                      at::Tensor output, bool return_mask, at::Tensor mask) {
    CHECK_INPUT(input);
    CHECK_INPUT(beta);
    CHECK_INPUT(output);
    CHECK_INPUT(mask);

    int batches = input.size(0);
    int channels = input.size(1);
    int depth = input.size(2);
    int height = input.size(3);
    int width = input.size(4);

    int kernel_d = std::get<0>(kernel);
    int kernel_h = std::get<1>(kernel);
    int kernel_w = std::get<2>(kernel);
    int stride_d = std::get<0>(stride);
    int stride_h = std::get<1>(stride);
    int stride_w = std::get<2>(stride);

    AdaPool3dForwardLauncher(input, beta,
                             batches, channels,
                             depth, height,
                             width, kernel_d,
                             kernel_h, kernel_w,
                             stride_d, stride_h,
                             stride_w, output,
                             return_mask, mask);
    return 1;
}

int ada_edscw_pool3d_forward(at::Tensor input, const std::tuple<int, int, int> kernel,
                            const std::tuple<int, int, int> stride, at::Tensor output,
                            bool return_mask, at::Tensor mask) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_INPUT(mask);

    int batches = input.size(0);
    int channels = input.size(1);
    int depth = input.size(2);
    int height = input.size(3);
    int width = input.size(4);

    int kernel_d = std::get<0>(kernel);
    int kernel_h = std::get<1>(kernel);
    int kernel_w = std::get<2>(kernel);
    int stride_d = std::get<0>(stride);
    int stride_h = std::get<1>(stride);
    int stride_w = std::get<2>(stride);

    Ada_EDSCW_Pool3dForwardLauncher(input, batches,
                                   channels, depth,
                                   height, width,
                                   kernel_d, kernel_h,
                                   kernel_w, stride_d,
                                   stride_h, stride_w,
                                   output, return_mask,
                                   mask);
    return 1;
}

int idw_pool3d_forward(at::Tensor input, const std::tuple<int, int, int> kernel,
                       const std::tuple<int, int, int> stride, at::Tensor output,
                       bool return_mask, at::Tensor mask) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_INPUT(mask);

    int batches = input.size(0);
    int channels = input.size(1);
    int depth = input.size(2);
    int height = input.size(3);
    int width = input.size(4);

    int kernel_d = std::get<0>(kernel);
    int kernel_h = std::get<1>(kernel);
    int kernel_w = std::get<2>(kernel);
    int stride_d = std::get<0>(stride);
    int stride_h = std::get<1>(stride);
    int stride_w = std::get<2>(stride);

    IDW_Pool3dForwardLauncher(input, batches,
                              channels, depth,
                              height, width,
                              kernel_d, kernel_h,
                              kernel_w, stride_d,
                              stride_h, stride_w,
                              output, return_mask,
                              mask);
    return 1;
}


int ada_em_pool3d_forward(at::Tensor input, const std::tuple<int, int, int> kernel,
                          const std::tuple<int, int, int> stride, at::Tensor output,
                          bool return_mask, at::Tensor mask) {
    CHECK_INPUT(input);
    CHECK_INPUT(output);
    CHECK_INPUT(mask);

    int batches = input.size(0);
    int channels = input.size(1);
    int depth = input.size(2);
    int height = input.size(3);
    int width = input.size(4);

    int kernel_d = std::get<0>(kernel);
    int kernel_h = std::get<1>(kernel);
    int kernel_w = std::get<2>(kernel);
    int stride_d = std::get<0>(stride);
    int stride_h = std::get<1>(stride);
    int stride_w = std::get<2>(stride);

    Ada_EM_Pool3dForwardLauncher(input, batches,
                                 channels, depth,
                                 height, width,
                                 kernel_d, kernel_h,
                                 kernel_w, stride_d,
                                 stride_h, stride_w,
                                 output, return_mask,
                                 mask);
    return 1;
}


int adapool3d_backward(const at::Tensor output_grad, const at::Tensor input,
                       const at::Tensor beta, const std::tuple<int, int, int> kernel,
                       const std::tuple<int, int, int> stride, at::Tensor input_grad,
                       at::Tensor beta_grad) {
    CHECK_INPUT(output_grad);
    CHECK_INPUT(input);
    CHECK_INPUT(beta);
    CHECK_INPUT(input_grad);
    CHECK_INPUT(beta_grad);

    int batches = input_grad.size(0);
    int channels = input_grad.size(1);
    int depth = input_grad.size(2);
    int height = input_grad.size(3);
    int width = input_grad.size(4);

    int kernel_d = std::get<0>(kernel);
    int kernel_h = std::get<1>(kernel);
    int kernel_w = std::get<2>(kernel);
    int stride_d = std::get<0>(stride);
    int stride_h = std::get<1>(stride);
    int stride_w = std::get<2>(stride);

    AdaPool3dBackwardLauncher(output_grad, input,
                              beta, batches,
                              channels, depth,
                              height, width,
                              kernel_d, kernel_h,
                              kernel_w, stride_d,
                              stride_h, stride_w,
                              input_grad, beta_grad);
    return 1;
}

int ada_edscw_pool3d_backward(const at::Tensor output_grad, const at::Tensor input,
                             const std::tuple<int, int, int> kernel, const std::tuple<int, int, int> stride,
                             at::Tensor input_grad) {
    CHECK_INPUT(output_grad);
    CHECK_INPUT(input);
    CHECK_INPUT(input_grad);

    int batches = input_grad.size(0);
    int channels = input_grad.size(1);
    int depth = input_grad.size(2);
    int height = input_grad.size(3);
    int width = input_grad.size(4);

    int kernel_d = std::get<0>(kernel);
    int kernel_h = std::get<1>(kernel);
    int kernel_w = std::get<2>(kernel);
    int stride_d = std::get<0>(stride);
    int stride_h = std::get<1>(stride);
    int stride_w = std::get<2>(stride);

    Ada_EDSCW_Pool3dBackwardLauncher(output_grad, input,
                                    batches, channels,
                                    depth, height,
                                    width, kernel_d,
                                    kernel_h, kernel_w,
                                    stride_d, stride_h,
                                    stride_w, input_grad);
    return 1;
}

int idw_pool3d_backward(const at::Tensor output_grad, const at::Tensor input,
                        const std::tuple<int, int, int> kernel, const std::tuple<int, int, int> stride,
                        at::Tensor input_grad){
    CHECK_INPUT(output_grad);
    CHECK_INPUT(input);
    CHECK_INPUT(input_grad);

    int batches = input_grad.size(0);
    int channels = input_grad.size(1);
    int depth = input_grad.size(2);
    int height = input_grad.size(3);
    int width = input_grad.size(4);

    int kernel_d = std::get<0>(kernel);
    int kernel_h = std::get<1>(kernel);
    int kernel_w = std::get<2>(kernel);
    int stride_d = std::get<0>(stride);
    int stride_h = std::get<1>(stride);
    int stride_w = std::get<2>(stride);

    IDW_Pool3dBackwardLauncher(output_grad, input,
                               batches, channels,
                               depth, height,
                               width, kernel_d,
                               kernel_h, kernel_w,
                               stride_d, stride_h,
                               stride_w, input_grad);
    return 1;
}

int ada_em_pool3d_backward(const at::Tensor output_grad, const at::Tensor input,
                           const std::tuple<int, int, int> kernel, const std::tuple<int, int, int> stride,
                           at::Tensor input_grad) {
    CHECK_INPUT(output_grad);
    CHECK_INPUT(input);
    CHECK_INPUT(input_grad);

    int batches = input_grad.size(0);
    int channels = input_grad.size(1);
    int depth = input_grad.size(2);
    int height = input_grad.size(3);
    int width = input_grad.size(4);

    int kernel_d = std::get<0>(kernel);
    int kernel_h = std::get<1>(kernel);
    int kernel_w = std::get<2>(kernel);
    int stride_d = std::get<0>(stride);
    int stride_h = std::get<1>(stride);
    int stride_w = std::get<2>(stride);

    Ada_EM_Pool3dBackwardLauncher(output_grad, input,
                                  batches, channels,
                                  depth, height,
                                  width, kernel_d,
                                  kernel_h, kernel_w,
                                  stride_d, stride_h,
                                  stride_w, input_grad);
    return 1;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_1d", &adapool1d_forward, "AdaPool1d forward");
  m.def("forward_1d_edscw", &ada_edscw_pool1d_forward, "AdaPool1d forward (EDSCW)");
  m.def("forward_1d_em", &ada_em_pool1d_forward, "AdaPool1d forward (EM)");
  m.def("forward_1d_idw", &idw_pool1d_forward, "IDWPool1d forward");

  m.def("backward_1d", &adapool1d_backward, "AdaPool1d backward");
  m.def("backward_1d_edscw", &ada_edscw_pool1d_backward, "AdaPool1d backward (EDSCW)");
  m.def("backward_1d_em", &ada_em_pool1d_backward, "AdaPool1d backward (EM)");
  m.def("backward_1d_idw", &idw_pool1d_backward, "IDWPool1d backward");


  m.def("forward_2d", &adapool2d_forward, "AdaPool2d forward");
  m.def("forward_2d_edscw", &ada_edscw_pool2d_forward, "AdaPool2d forward (EDSCW)");
  m.def("forward_2d_em", &ada_em_pool2d_forward, "AdaPool2d forward (EM)");
  m.def("forward_2d_idw", &idw_pool2d_forward, "IDWPool2d forward");


  m.def("backward_2d", &adapool2d_backward, "AdaPool2d backward");
  m.def("backward_2d_edscw", &ada_edscw_pool2d_backward, "AdaPool2d backward (EDSCW)");
  m.def("backward_2d_em", &ada_em_pool2d_backward, "AdaPool2d backward (EM)");
  m.def("backward_2d_idw", &idw_pool2d_backward, "IDWPool2d backward");


  m.def("forward_3d", &adapool3d_forward, "AdaPool3d forward");
  m.def("forward_3d_edscw", &ada_edscw_pool3d_forward, "AdaPool3d forward (EDSCW)");
  m.def("forward_3d_em", &ada_em_pool3d_forward, "AdaPool3d forward (EM)");
  m.def("forward_3d_idw", &idw_pool3d_forward, "IDWPool3d forward");


  m.def("backward_3d", &adapool3d_backward, "AdaPool3d backward");
  m.def("backward_3d_edscw", &ada_edscw_pool3d_backward, "AdaPool3d backward (EDSCW)");
  m.def("backward_3d_em", &ada_em_pool3d_backward, "AdaPool3d backward (EM)");
  m.def("backward_3d_idw", &idw_pool3d_backward, "IDWPool3d backward");
}
