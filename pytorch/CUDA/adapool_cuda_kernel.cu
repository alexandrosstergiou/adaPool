#include <float.h>
#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>

#include "limits.cuh"

using namespace at;  // fix for pytorch<=0.4.1

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 1024

inline int GET_BLOCKS(const int N) {
  int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  int max_block_num = 65535;
  return min(optimal_block_num, max_block_num);
}

//type-safe sign
template <typename scalar_t>
__device__ scalar_t sgn(scalar_t val) {
    return (scalar_t(0) < val) - (val < scalar_t(0));
}

// Overflow and Underflow clamp
template <typename scalar_t>
__device__  scalar_t clamp(const scalar_t n, const scalar_t lower, const scalar_t upper) {
  const scalar_t tmp = abs(n);
  const scalar_t result = max(lower, min(tmp, upper));
  return result * sgn(n);
}

// Psudo-Huber distance
template <typename scalar_t>
__device__  scalar_t huber(const scalar_t x1, const scalar_t x2, const scalar_t delta) {
  const scalar_t result = pow(delta,2) * (sqrt(1.+pow((x1-x2)/delta,2)) - 1.);
  //const scalar_t result = abs(x1-x2);//sqrt(pow(x1-x2,2));
  return result;
}

// Euclidean distance
template <typename scalar_t>
__device__  scalar_t l2(const scalar_t x1, const scalar_t x2) {
  //const scalar_t result = abs(x1-x2)/sqrt(pow(x1-x2,2));
  const scalar_t result = (x1*x2)/(sqrt(pow(x1,2))+sqrt(pow(x2,2)));
  //const scalar_t result = (2*x1*x2)/((x1*x1)+(x2*x2));
  return result;
}

// DSC
template <typename scalar_t>
__device__  scalar_t dsc(const scalar_t x1, const scalar_t x2) {
  //const scalar_t result = (x1*x2)/(sqrt(pow(x1,2))+sqrt(pow(x2,2)));
  const scalar_t result = (2*abs(x1*x2))/(x1*x1+x2*x2);
  return result;
}


template <typename scalar_t>
__global__ void AdaPool1dForward(const int nthreads,
                                 const scalar_t *bottom_input, const scalar_t *bottom_beta,
                                 const int batches, const int channels,
                                 const int dim, const int kernel_d,
                                 const int stride_d, scalar_t *output_data,
                                 const bool return_mask, scalar_t *mask){
    int pooled_dim = dim/stride_d;
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      int pd = index % pooled_dim;
      int c = (index / pooled_dim) % channels;
      int n = index / pooled_dim / channels;

      const int offset = (n * channels + c) * dim;
      const scalar_t *offset_bottom_input = bottom_input + offset;

      const int base_d = pd*stride_d - kernel_d/2;

      scalar_t act_sum = 0.;
      scalar_t mask_sum_avg = 0.;
      scalar_t mask_sum_max = 0.;

      output_data[index] = 0.;
      const scalar_t upper = n_limits<scalar_t>::max();
      const scalar_t lower = n_limits<scalar_t>::min();
      const scalar_t zero = 0.;
      const scalar_t one = 1.;

      const scalar_t b = clamp(bottom_beta[index], zero, one);

      int count = 0.;

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;
        if(d_offset >= dim || d_offset < 0)continue;
        const int offset = d_offset;

        act_sum += offset_bottom_input[offset];
        count += 1;
      }
      scalar_t act_avg = act_sum/count;

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;
        if(d_offset >= dim || d_offset < 0)continue;
        const int offset = d_offset;

        scalar_t dist = dsc(offset_bottom_input[offset], act_avg);

        mask_sum_avg += exp(dist);
        mask_sum_max += exp(offset_bottom_input[offset]);
      }
      // Overflow check
      mask_sum_avg = clamp(mask_sum_avg, lower, upper);
      mask_sum_max = clamp(mask_sum_max, lower, upper);


      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;
        if(d_offset >= dim || d_offset < 0)continue;
        const int offset = d_offset;

        scalar_t dist = dsc(offset_bottom_input[offset], act_avg);

        scalar_t mask_ = b * exp(dist)/mask_sum_avg + (1. - b) * exp(offset_bottom_input[offset])/ mask_sum_max;
        mask_ = clamp(mask_, zero, upper);

        if (return_mask)mask[offset]= mask_;

        output_data[index] += offset_bottom_input[offset] * mask_;
        output_data[index] = clamp(output_data[index], zero, upper);
      }
    }
}


template <typename scalar_t>
__global__ void AdaPool2dForward(const int nthreads,
                                  const scalar_t *bottom_input, const scalar_t *bottom_beta,
                                  const int batches, const int channels,
                                  const int height, const int width,
                                  const int kernel_h, const int kernel_w,
                                  const int stride_h, const int stride_w,
                                  scalar_t *output_data, const bool return_mask,
                                  scalar_t *mask){
    int pooled_height = height/stride_h;
    int pooled_width = width/stride_w;
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int c = (index / pooled_width / pooled_height) % channels;
      int n = index / pooled_width / pooled_height / channels;

      const int offset = (n * channels + c) * height * width;
      const scalar_t *offset_bottom_input = bottom_input + offset;

      const int base_y = ph*stride_h - kernel_h/2;
      const int base_x = pw*stride_w - kernel_w/2;

      scalar_t act_sum = 0.;
      scalar_t mask_sum_avg = 0.;
      scalar_t mask_sum_max = 0.;

      output_data[index] = 0.;
      const scalar_t upper = n_limits<scalar_t>::max();
      const scalar_t lower = n_limits<scalar_t>::min();
      const scalar_t zero = 0.;
      const scalar_t one = 1.;

      const scalar_t b = clamp(bottom_beta[index], zero, one);

      int count = 0.;

      for(int iy=0; iy<kernel_h; iy++){
        const int y_offset = base_y + iy;
        if(y_offset >= height || y_offset < 0)continue;
        for(int ix=0; ix<kernel_w; ix++){
          const int x_offset = base_x + ix;
          if(x_offset >= width || x_offset < 0)continue;
          const int offset = y_offset*width + x_offset;

          act_sum += offset_bottom_input[offset];
          count += 1;
        }
      }
      scalar_t act_avg = act_sum/count;

      for(int iy=0; iy<kernel_h; iy++){
        const int y_offset = base_y + iy;

        if(y_offset >= height || y_offset < 0)continue;
        for(int ix=0; ix<kernel_w; ix++){
          const int x_offset = base_x + ix;

          if(x_offset >= width || x_offset < 0)continue;
          const int offset = y_offset*width + x_offset;

          scalar_t dist = dsc(offset_bottom_input[offset], act_avg);

          mask_sum_avg += exp(dist);
          mask_sum_max += exp(offset_bottom_input[offset]);

        }
      }
      // Overflow check
      mask_sum_avg = clamp(mask_sum_avg, lower, upper);
      mask_sum_max = clamp(mask_sum_max, lower, upper);


      for(int iy=0; iy<kernel_h; iy++){
        const int y_offset = base_y + iy;

        if(y_offset >= height || y_offset < 0)continue;
        for(int ix=0; ix<kernel_w; ix++){
          const int x_offset = base_x + ix;

          if(x_offset >= width || x_offset < 0)continue;
          const int offset = y_offset*width + x_offset;

          scalar_t dist = dsc(offset_bottom_input[offset], act_avg);

          scalar_t mask_ = b * exp(dist)/  mask_sum_avg + (1. - b) * exp(offset_bottom_input[offset])/  mask_sum_max;
          mask_ = clamp(mask_, zero, upper);

          if (return_mask)mask[offset]= mask_;

          output_data[index] += offset_bottom_input[offset] * mask_;
          output_data[index] = clamp(output_data[index], zero, upper);
        }
      }
    }
}


template <typename scalar_t>
__global__ void AdaPool3dForward(const int nthreads,
                                  const scalar_t *bottom_input, const scalar_t *bottom_beta,
                                  const int batches, const int channels,
                                  const int depth, const int height,
                                  const int width, const int kernel_d,
                                  const int kernel_h, const int kernel_w,
                                  const int stride_d, const int stride_h,
                                  const int stride_w, scalar_t *output_data,
                                  const bool return_mask, scalar_t *mask){
    int pooled_depth = depth/stride_d;
    int pooled_height = height/stride_h;
    int pooled_width = width/stride_w;
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int pd = (index / pooled_width / pooled_height) % pooled_depth;
      int c = (index / pooled_width / pooled_height / pooled_depth) % channels;
      int n = index / pooled_width / pooled_height / pooled_depth / channels;

      const int offset = (n * channels + c) * depth * height * width;
      const scalar_t *offset_bottom_input = bottom_input + offset;

      const int base_d = pd*stride_d - kernel_d/2;
      const int base_y = ph*stride_h - kernel_h/2;
      const int base_x = pw*stride_w - kernel_w/2;

      scalar_t act_sum = 0.;
      scalar_t mask_sum_avg = 0.;
      scalar_t mask_sum_max = 0.;

      output_data[index] = 0.;
      const scalar_t upper = n_limits<scalar_t>::max();
      const scalar_t lower = n_limits<scalar_t>::min();
      const scalar_t zero = 0.;
      const scalar_t one = 1.;

      const scalar_t b = clamp(bottom_beta[index], zero, one);

      int count = 0.;

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;
        if(d_offset >= depth || d_offset < 0)continue;
        for(int iy=0; iy<kernel_h; iy++){
          const int y_offset = base_y + iy;
          if(y_offset >= height || y_offset < 0)continue;
          for(int ix=0; ix<kernel_w; ix++){
            const int x_offset = base_x + ix;
            if(x_offset >= width || x_offset < 0)continue;
            const int offset = d_offset*height + y_offset*width + x_offset;

            act_sum += offset_bottom_input[offset];
            count += 1;
          }
        }
      }
      scalar_t act_avg = act_sum/count;


      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;

        if(d_offset >= depth || d_offset < 0)continue;
        for(int iy=0; iy<kernel_h; iy++){
          const int y_offset = base_y + iy;

          if(y_offset >= height || y_offset < 0)continue;
          for(int ix=0; ix<kernel_w; ix++){
            const int x_offset = base_x + ix;

            if(x_offset >= width || x_offset < 0)continue;
            const int offset = d_offset*height + y_offset*width + x_offset;

            scalar_t dist = dsc(offset_bottom_input[offset], act_avg);

            mask_sum_avg += exp(dist);
            mask_sum_max += exp(offset_bottom_input[offset]);

          }
        }
      }
      // Overflow check
      mask_sum_avg = clamp(mask_sum_avg, lower, upper);
      mask_sum_max = clamp(mask_sum_max, lower, upper);

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;

        if(d_offset >= depth || d_offset < 0)continue;
        for(int iy=0; iy<kernel_h; iy++){
          const int y_offset = base_y + iy;

          if(y_offset >= height || y_offset < 0)continue;
          for(int ix=0; ix<kernel_w; ix++){
            const int x_offset = base_x + ix;

            if(x_offset >= width || x_offset < 0)continue;
            const int offset = d_offset*height + y_offset*width + x_offset;

            scalar_t dist = dsc(offset_bottom_input[offset], act_avg);


            scalar_t mask_ = b * exp(dist)/mask_sum_avg + (1. - b) * exp(offset_bottom_input[offset])/mask_sum_max;
            mask_ = clamp(mask_, zero, upper);

            if (return_mask)mask[offset]= mask_;

            output_data[index] += offset_bottom_input[offset] * mask_;
            output_data[index] = clamp(output_data[index], zero, upper);

          }
        }
      }
    }
}


int AdaPool1dForwardLauncher(const at::Tensor input, const at::Tensor beta,
                             const int batches, const int channels,
                             const int dim, const int kernel_d,
                             const int stride_d, at::Tensor output,
                             const bool return_mask, at::Tensor mask){
    const int output_size = batches * dim/stride_d * channels;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "AdaPool1dLauncherForward", ([&] {
        const scalar_t *bottom_input = input.data_ptr<scalar_t>();
        const scalar_t *bottom_beta = beta.data_ptr<scalar_t>();
        scalar_t *output_data = output.data_ptr<scalar_t>();
        scalar_t *mask_data = mask.data_ptr<scalar_t>();

        AdaPool1dForward<scalar_t>
        <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
          output_size, bottom_input,
          bottom_beta, batches,
          channels, dim,
          kernel_d, stride_d,
          output_data, return_mask,
          mask_data);
        })
      );

    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
      fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
      exit(-1);
    }
  return 1;
}

int AdaPool2dForwardLauncher(const at::Tensor input, const at::Tensor beta,
                             const int batches, const int channels,
                             const int height, const int width,
                             const int kernel_h, const int kernel_w,
                             const int stride_h, const int stride_w,
                             at::Tensor output, const bool return_mask,
                             at::Tensor mask){
    const int output_size = batches * height/stride_h * width/stride_w * channels;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "AdaPool2dLauncherForward", ([&] {
        const scalar_t *bottom_input = input.data_ptr<scalar_t>();
        const scalar_t *bottom_beta = beta.data_ptr<scalar_t>();
        scalar_t *output_data = output.data_ptr<scalar_t>();
        scalar_t *mask_data = mask.data_ptr<scalar_t>();

        AdaPool2dForward<scalar_t>
        <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
          output_size, bottom_input,
          bottom_beta, batches,
          channels, height,
          width, kernel_h,
          kernel_w, stride_h,
          stride_w, output_data,
          return_mask, mask_data);
        })
      );

    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
      fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
      exit(-1);
    }
  return 1;
}

int AdaPool3dForwardLauncher(const at::Tensor input, const at::Tensor beta,
                             const int batches, const int channels,
                             const int depth, const int height,
                             const int width, const int kernel_d,
                             const int kernel_h, const int kernel_w,
                             const int stride_d, const int stride_h,
                             const int stride_w, at::Tensor output,
                             const bool return_mask, at::Tensor mask){
    const int output_size = batches * depth/stride_d * height/stride_h * width/stride_w * channels;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "AdaPool3dLauncherForward", ([&] {
        const scalar_t *bottom_input = input.data_ptr<scalar_t>();
        const scalar_t *bottom_beta = beta.data_ptr<scalar_t>();
        scalar_t *output_data = output.data_ptr<scalar_t>();
        scalar_t *mask_data = mask.data_ptr<scalar_t>();


        AdaPool3dForward<scalar_t>
        <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
          output_size, bottom_input,
          bottom_beta, batches,
          channels, depth,
          height, width,
          kernel_d, kernel_h,
          kernel_w, stride_d,
          stride_h, stride_w,
          output_data, return_mask,
          mask_data);
        })
      );

    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
      fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
      exit(-1);
    }
  return 1;
}


template <typename scalar_t>
__global__ void AdaPool1dBackward(const int nthreads,
                                   const scalar_t *diff_output, const scalar_t *data_input,
                                   const scalar_t *data_beta, const int batches,
                                   const int channels, const int dim,
                                   const int kernel_d, const int stride_d,
                                   scalar_t *diff_input, scalar_t *diff_beta){
    int pooled_dim = dim/stride_d;
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      int pd = index % pooled_dim;
      int c = (index / pooled_dim) % channels;
      int n = index / pooled_dim / channels;

      const int offset0 = (n * channels + c) * dim;
      const scalar_t *offset_data_input = data_input + offset0;

      const scalar_t diff_output_index = diff_output[index];
      scalar_t *offset_diff_input = diff_input + offset0;
      const int base_d = pd*stride_d - kernel_d/2;

      scalar_t act_sum = 0.;
      scalar_t mask_sum_avg = 0.;
      scalar_t mask_sum_max = 0.;

      const scalar_t zero = 0.;
      const scalar_t one = 1.;
      const scalar_t upper = n_limits<scalar_t>::max();
      const scalar_t lower = n_limits<scalar_t>::min();

      const scalar_t b = clamp(data_beta[index], zero, one);

      int count = 0.;

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;
        if(d_offset >= dim || d_offset < 0)continue;
        const int offset = d_offset;

        act_sum += offset_data_input[offset];
        count += 1;
      }
      scalar_t act_avg = act_sum/count;

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;

        if(d_offset >= dim || d_offset < 0)continue;
        const int offset = d_offset;

        scalar_t dist = dsc(offset_data_input[offset], act_avg);

        mask_sum_avg += exp(dist);
        mask_sum_max += exp(offset_data_input[offset]);

      }
      // Overflow check
      mask_sum_avg = clamp(mask_sum_avg, lower, upper);
      mask_sum_max = clamp(mask_sum_max, lower, upper);

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;

        if(d_offset >= dim || d_offset < 0)continue;
          const int offset = d_offset;

          scalar_t dist = dsc(offset_data_input[offset], act_avg);

          scalar_t mask = b * exp(dist)/mask_sum_avg + (1. - b) * exp(offset_data_input[offset])/mask_sum_max;
          mask = clamp(mask, zero, upper);

          scalar_t weighted_grad = diff_output_index * mask;

          // Underflow check
          weighted_grad = clamp(weighted_grad, zero, upper);

          atomicAdd(offset_diff_input+offset, weighted_grad);
      }
    }
}

template <typename scalar_t>
__global__ void AdaPool2dBackward(const int nthreads,
                              const scalar_t *diff_output, const scalar_t *data_input,
                              const scalar_t *data_beta, const int batches,
                              const int channels, const int height,
                              const int width, const int kernel_h,
                              const int kernel_w, const int stride_h,
                              const int stride_w, scalar_t *diff_input,
                              scalar_t *diff_beta){
    int pooled_height = height/stride_h;
    int pooled_width = width/stride_w;
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int c = (index / pooled_width / pooled_height) % channels;
      int n = index / pooled_width / pooled_height / channels;

      const int offset0 = (n * channels + c) * height * width;
      const scalar_t *offset_data_input = data_input + offset0;

      const scalar_t diff_output_index = diff_output[index];

      scalar_t *offset_diff_input = diff_input + offset0;

      const int base_y = ph*stride_h - kernel_h/2;
      const int base_x = pw*stride_w - kernel_w/2;

      scalar_t act_sum = 0.;
      scalar_t mask_sum_avg = 0.;
      scalar_t mask_sum_max = 0.;

      scalar_t zero = 0.;
      const scalar_t one = 1.;
      const scalar_t upper = n_limits<scalar_t>::max();
      const scalar_t lower = n_limits<scalar_t>::min();

      const scalar_t b = clamp(data_beta[index], zero, one);

      int count = 0.;

      for(int iy=0; iy<kernel_h; iy++){
        const int y_offset = base_y + iy;
        if(y_offset >= height || y_offset < 0)continue;
        for(int ix=0; ix<kernel_w; ix++){
          const int x_offset = base_x + ix;
          if(x_offset >= width || x_offset < 0)continue;
          const int offset = y_offset*width + x_offset;

          act_sum += offset_data_input[offset];
          count += 1;
        }
      }
      scalar_t act_avg = act_sum/count;

      for(int iy=0; iy<kernel_h; iy++){
        const int y_offset = base_y + iy;

        if(y_offset >= height || y_offset < 0)continue;
        for(int ix=0; ix<kernel_w; ix++){
          const int x_offset = base_x + ix;

          if(x_offset >= width || x_offset < 0)continue;
          const int offset = y_offset*width + x_offset;

          scalar_t dist = dsc(offset_data_input[offset], act_avg);

          mask_sum_avg += exp(dist);
          mask_sum_max += exp(offset_data_input[offset]);

        }
      }
      // Overflow check
      mask_sum_avg = clamp(mask_sum_avg, lower, upper);
      mask_sum_max = clamp(mask_sum_max, lower, upper);

      for(int iy=0; iy<kernel_h; iy++){
        const int y_offset = base_y + iy;

        if(y_offset >= height || y_offset < 0)continue;
        for(int ix=0; ix<kernel_w; ix++){
          const int x_offset = base_x + ix;

          if(x_offset >= width || x_offset < 0)continue;
            const int offset = y_offset*width + x_offset;

            scalar_t dist = dsc(offset_data_input[offset], act_avg);

            scalar_t mask = b * exp(dist)/mask_sum_avg + (1. - b) * exp(offset_data_input[offset])/mask_sum_max;
            mask = clamp(mask, zero, upper);

            scalar_t weighted_grad = diff_output_index * mask;

            // Underflow check
            weighted_grad = clamp(weighted_grad, zero, upper);

            atomicAdd(offset_diff_input+offset, weighted_grad);
        }
      }
    }
}

template <typename scalar_t>
__global__ void AdaPool3dBackward(const int nthreads,
                                  const scalar_t *diff_output, const scalar_t *data_input,
                                  const scalar_t *data_beta, const int batches,
                                  const int channels, const int depth,
                                  const int height, const int width,
                                  const int kernel_d, const int kernel_h,
                                  const int kernel_w , const int stride_d,
                                  const int stride_h, const int stride_w,
                                  scalar_t *diff_input, scalar_t *diff_beta){
    int pooled_depth = depth/stride_d;
    int pooled_height = width/stride_h;
    int pooled_width = width/stride_w;
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int pd = (index / pooled_width / pooled_height) % pooled_depth;
      int c = (index / pooled_width / pooled_height / pooled_depth) % channels;
      int n = index / pooled_width / pooled_height / pooled_depth / channels;

      const int offset0 = (n * channels + c) * depth * height * width;
      const scalar_t *offset_data_input = data_input + offset0;

      const scalar_t diff_output_index = diff_output[index];

      scalar_t *offset_diff_input = diff_input + offset0;

      const int base_d = pd*stride_d - kernel_d/2;
      const int base_y = ph*stride_h - kernel_h/2;
      const int base_x = pw*stride_w - kernel_w/2;


      scalar_t act_sum = 0.;
      scalar_t mask_sum_avg = 0.;
      scalar_t mask_sum_max = 0.;

      scalar_t zero = 0.;
      const scalar_t one = 1.;
      const scalar_t upper = n_limits<scalar_t>::max();
      const scalar_t lower = n_limits<scalar_t>::min();

      const scalar_t b = clamp(data_beta[index], zero, one);

      int count = 0.;

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;
        if(d_offset >= depth || d_offset < 0)continue;
        for(int iy=0; iy<kernel_h; iy++){
          const int y_offset = base_y + iy;
          if(y_offset >= height || y_offset < 0)continue;
          for(int ix=0; ix<kernel_w; ix++){
            const int x_offset = base_x + ix;
            if(x_offset >= width || x_offset < 0)continue;
            const int offset = d_offset*height + y_offset*width + x_offset;

            act_sum += offset_data_input[offset];
            count += 1;
          }
        }
      }
      scalar_t act_avg = act_sum/count;

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;

        if(d_offset >= depth || d_offset < 0)continue;
        for(int iy=0; iy<kernel_h; iy++){
          const int y_offset = base_y + iy;

          if(y_offset >= height || y_offset < 0)continue;
          for(int ix=0; ix<kernel_w; ix++){
            const int x_offset = base_x + ix;

            if(x_offset >= width || x_offset < 0)continue;
              const int offset = d_offset*height + y_offset*width + x_offset;

              scalar_t dist = dsc(offset_data_input[offset], act_avg);

              mask_sum_avg += exp(dist);
              mask_sum_max += exp(offset_data_input[offset]);

          }
        }
      }
      // Overflow check
      mask_sum_avg = clamp(mask_sum_avg, lower, upper);
      mask_sum_max = clamp(mask_sum_max, lower, upper);

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;

        if(d_offset >= depth || d_offset < 0)continue;
        for(int iy=0; iy<kernel_h; iy++){
          const int y_offset = base_y + iy;

          if(y_offset >= height || y_offset < 0)continue;
          for(int ix=0; ix<kernel_w; ix++){
            const int x_offset = base_x + ix;

            if(x_offset >= width || x_offset < 0)continue;
              const int offset = d_offset*height + y_offset*width + x_offset;

              scalar_t dist = dsc(offset_data_input[offset], act_avg);

              scalar_t mask = b * exp(dist)/mask_sum_avg + (1. - b) * exp(offset_data_input[offset])/mask_sum_max;
              mask = clamp(mask, zero, upper);

              scalar_t weighted_grad = diff_output_index/mask;

              // Underflow check
              weighted_grad = clamp(weighted_grad, zero, upper);

              atomicAdd(offset_diff_input+offset, weighted_grad);
          }
        }
      }
    }
}

int AdaPool1dBackwardLauncher(const at::Tensor output_grad, const at::Tensor input,
                              const at::Tensor beta, const int batches,
                              const int channels, const int dim,
                              const int kernel_d, const int stride_d,
                              at::Tensor input_grad, at::Tensor beta_grad){

    const int output_size = batches * dim/stride_d * channels;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "AdaPool1dLauncherBackward", ([&] {
        scalar_t *diff_input = input_grad.data_ptr<scalar_t>();
        scalar_t *diff_beta = beta_grad.data_ptr<scalar_t>();
        const scalar_t *diff_output = output_grad.data_ptr<scalar_t>();
        const scalar_t *data_input = input.data_ptr<scalar_t>();
        const scalar_t *data_beta = beta.data_ptr<scalar_t>();

        AdaPool1dBackward<scalar_t>
        <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
          output_size, diff_output,
          data_input, data_beta,
          batches, channels,
          dim, kernel_d,
          stride_d, diff_input,
          diff_beta);
        }
        )
        );

    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
      fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
      exit(-1);
    }
  return 1;
}

int AdaPool2dBackwardLauncher(const at::Tensor output_grad, const at::Tensor input,
                               const at::Tensor beta, const int batches,
                               const int channels, const int height,
                               const int width, const int kernel_h,
                               const int kernel_w, const int stride_h,
                               const int stride_w, at::Tensor input_grad,
                               at::Tensor beta_grad){

    const int output_size = batches * height/stride_h * width/stride_w * channels;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "AdaPool2dLauncherBackward", ([&] {
        scalar_t *diff_input = input_grad.data_ptr<scalar_t>();
        scalar_t *diff_beta = beta_grad.data_ptr<scalar_t>();
        const scalar_t *diff_output = output_grad.data_ptr<scalar_t>();
        const scalar_t *data_input = input.data_ptr<scalar_t>();
        const scalar_t *data_beta = beta.data_ptr<scalar_t>();

        AdaPool2dBackward<scalar_t>
        <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
          output_size, diff_output,
          data_input, data_beta,
          batches, channels,
          height, width,
          kernel_h, kernel_w,
          stride_h, stride_w,
          diff_input, diff_beta);
        }
        )
        );

    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
      fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
      exit(-1);
    }
  return 1;
}

int AdaPool3dBackwardLauncher(const at::Tensor output_grad, const at::Tensor input,
                              const at::Tensor beta, const int batches,
                              const int channels, const int depth,
                              const int height, const int width,
                              const int kernel_d, const int kernel_h,
                              const int kernel_w, const int stride_d,
                              const int stride_h, const int stride_w,
                              at::Tensor input_grad, at::Tensor beta_grad){

    const int output_size = batches * depth/stride_d * height/stride_h * width/stride_w * channels;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "AdaPool3dLauncherBackward", ([&] {
        scalar_t *diff_input = input_grad.data_ptr<scalar_t>();
        scalar_t *diff_beta = beta_grad.data_ptr<scalar_t>();
        const scalar_t *diff_output = output_grad.data_ptr<scalar_t>();
        const scalar_t *data_input = input.data_ptr<scalar_t>();
        const scalar_t *data_beta = beta.data_ptr<scalar_t>();

        AdaPool3dBackward<scalar_t>
        <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
          output_size, diff_output,
          data_input, data_beta,
          batches, channels,
          depth, height,
          width, kernel_d,
          kernel_h, kernel_w,
          stride_d, stride_h,
          stride_w, diff_input,
          diff_beta);
        }
        )
        );

    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
      fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
      exit(-1);
    }
  return 1;
}





template <typename scalar_t>
__global__ void Ada_EDSCW_Pool1dForward(const int nthreads,
                                       const scalar_t *bottom_input, const int batches,
                                       const int channels, const int dim,
                                       const int kernel_d, const int stride_d,
                                       scalar_t *output_data, const bool return_mask,
                                       scalar_t *mask){
    int pooled_dim = dim/stride_d;
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      int pd = index % pooled_dim;
      int c = (index / pooled_dim) % channels;
      int n = index / pooled_dim / channels;

      const int offset = (n * channels + c) * dim;
      const scalar_t *offset_bottom_input = bottom_input + offset;

      const int base_d = pd*stride_d - kernel_d/2;

      scalar_t act_sum = 0.;
      scalar_t mask_sum_avg = 0.;

      output_data[index] = 0.;
      const scalar_t upper = n_limits<scalar_t>::max();
      const scalar_t lower = n_limits<scalar_t>::min();
      const scalar_t zero = 0.;

      int count = 0.;

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;
        if(d_offset >= dim || d_offset < 0)continue;
        const int offset = d_offset;

        act_sum += offset_bottom_input[offset];
        count += 1;
      }
      scalar_t act_avg = act_sum/count;

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;
        if(d_offset >= dim || d_offset < 0)continue;
        const int offset = d_offset;

        scalar_t dist = dsc(offset_bottom_input[offset], act_avg);

        mask_sum_avg += exp(dist);

      }
      // Overflow check
      mask_sum_avg = clamp(mask_sum_avg, lower, upper);


      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;

        if(d_offset >= dim || d_offset < 0)continue;
        const int offset = d_offset;

        scalar_t dist = dsc(offset_bottom_input[offset], act_avg);

        scalar_t mask_ = exp(dist)/mask_sum_avg;

        if (return_mask)mask[offset]= mask_;

        output_data[index] += offset_bottom_input[offset] * mask_;
        output_data[index] = clamp(output_data[index], zero, upper);
      }
    }
}


template <typename scalar_t>
__global__ void Ada_EDSCW_Pool2dForward(const int nthreads,
                                       const scalar_t *bottom_input, const int batches,
                                       const int channels, const int height,
                                       const int width, const int kernel_h,
                                       const int kernel_w, const int stride_h,
                                       const int stride_w, scalar_t *output_data,
                                       const bool return_mask, scalar_t *mask){
    int pooled_height = height/stride_h;
    int pooled_width = width/stride_w;
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int c = (index / pooled_width / pooled_height) % channels;
      int n = index / pooled_width / pooled_height / channels;

      const int offset = (n * channels + c) * height * width;
      const scalar_t *offset_bottom_input = bottom_input + offset;

      const int base_y = ph*stride_h - kernel_h/2;
      const int base_x = pw*stride_w - kernel_w/2;

      scalar_t act_sum = 0.;
      scalar_t mask_sum_avg = 0.;

      output_data[index] = 0.;
      const scalar_t upper = n_limits<scalar_t>::max();
      const scalar_t lower = n_limits<scalar_t>::min();
      const scalar_t zero = 0.;

      int count = 0.;

      for(int iy=0; iy<kernel_h; iy++){
        const int y_offset = base_y + iy;
        if(y_offset >= height || y_offset < 0)continue;
        for(int ix=0; ix<kernel_w; ix++){
          const int x_offset = base_x + ix;
          if(x_offset >= width || x_offset < 0)continue;
          const int offset = y_offset*width + x_offset;

          act_sum += offset_bottom_input[offset];
          count += 1;
        }
      }
      scalar_t act_avg = act_sum/count;

      for(int iy=0; iy<kernel_h; iy++){
        const int y_offset = base_y + iy;

        if(y_offset >= height || y_offset < 0)continue;
        for(int ix=0; ix<kernel_w; ix++){
          const int x_offset = base_x + ix;

          if(x_offset >= width || x_offset < 0)continue;
          const int offset = y_offset*width + x_offset;

          scalar_t dist = dsc(offset_bottom_input[offset], act_avg);

          mask_sum_avg += exp(dist);

        }
      }
      // Overflow check
      mask_sum_avg = clamp(mask_sum_avg, lower, upper);


      for(int iy=0; iy<kernel_h; iy++){
        const int y_offset = base_y + iy;

        if(y_offset >= height || y_offset < 0)continue;
        for(int ix=0; ix<kernel_w; ix++){
          const int x_offset = base_x + ix;

          if(x_offset >= width || x_offset < 0)continue;
          const int offset = y_offset*width + x_offset;

          scalar_t dist = dsc(offset_bottom_input[offset], act_avg);

          scalar_t mask_ = exp(dist)/mask_sum_avg;

          if (return_mask)mask[offset]= mask_;

          output_data[index] += offset_bottom_input[offset] * mask_;
          output_data[index] = clamp(output_data[index], zero, upper);
        }
      }
    }
}


template <typename scalar_t>
__global__ void Ada_EDSCW_Pool3dForward(const int nthreads,
                                       const scalar_t *bottom_input, const int batches,
                                       const int channels, const int depth,
                                       const int height, const int width,
                                       const int kernel_d, const int kernel_h,
                                       const int kernel_w, const int stride_d,
                                       const int stride_h, const int stride_w,
                                       scalar_t *output_data, const bool return_mask,
                                       scalar_t *mask){
    int pooled_depth = depth/stride_d;
    int pooled_height = height/stride_h;
    int pooled_width = width/stride_w;
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int pd = (index / pooled_width / pooled_height) % pooled_depth;
      int c = (index / pooled_width / pooled_height / pooled_depth) % channels;
      int n = index / pooled_width / pooled_height / pooled_depth / channels;

      const int offset = (n * channels + c) * depth * height * width;
      const scalar_t *offset_bottom_input = bottom_input + offset;

      const int base_d = pd*stride_d - kernel_d/2;
      const int base_y = ph*stride_h - kernel_h/2;
      const int base_x = pw*stride_w - kernel_w/2;

      scalar_t act_sum = 0.;
      scalar_t mask_sum_avg = 0.;

      output_data[index] = 0.;
      const scalar_t upper = n_limits<scalar_t>::max();
      const scalar_t lower = n_limits<scalar_t>::min();
      const scalar_t zero = 0.;

      int count = 0.;

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;
        if(d_offset >= depth || d_offset < 0)continue;
        for(int iy=0; iy<kernel_h; iy++){
          const int y_offset = base_y + iy;
          if(y_offset >= height || y_offset < 0)continue;
          for(int ix=0; ix<kernel_w; ix++){
            const int x_offset = base_x + ix;
            if(x_offset >= width || x_offset < 0)continue;
            const int offset = d_offset*height + y_offset*width + x_offset;

            act_sum += offset_bottom_input[offset];
            count += 1;
          }
        }
      }
      scalar_t act_avg = act_sum/count;


      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;

        if(d_offset >= depth || d_offset < 0)continue;
        for(int iy=0; iy<kernel_h; iy++){
          const int y_offset = base_y + iy;

          if(y_offset >= height || y_offset < 0)continue;
          for(int ix=0; ix<kernel_w; ix++){
            const int x_offset = base_x + ix;

            if(x_offset >= width || x_offset < 0)continue;
            const int offset = d_offset*height + y_offset*width + x_offset;

            scalar_t dist = dsc(offset_bottom_input[offset], act_avg);

            mask_sum_avg += exp(dist);

          }
        }
      }
      // Overflow check
      mask_sum_avg = clamp(mask_sum_avg, lower, upper);

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;

        if(d_offset >= depth || d_offset < 0)continue;
        for(int iy=0; iy<kernel_h; iy++){
          const int y_offset = base_y + iy;

          if(y_offset >= height || y_offset < 0)continue;
          for(int ix=0; ix<kernel_w; ix++){
            const int x_offset = base_x + ix;

            if(x_offset >= width || x_offset < 0)continue;
            const int offset = d_offset*height + y_offset*width + x_offset;

            scalar_t dist = dsc(offset_bottom_input[offset], act_avg);

            scalar_t mask_ = exp(dist)/mask_sum_avg;

            if (return_mask)mask[offset]= mask_;

            output_data[index] += offset_bottom_input[offset] * mask_;
            output_data[index] = clamp(output_data[index], zero, upper);

          }
        }
      }
    }
}


int Ada_EDSCW_Pool1dForwardLauncher(const at::Tensor input, const int batches,
                                   const int channels, const int dim,
                                   const int kernel_d, const int stride_d,
                                   at::Tensor output, const bool return_mask,
                                   at::Tensor mask){
    const int output_size = batches * dim/stride_d * channels;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "Ada_EDSCW_Pool1dLauncherForward", ([&] {
        const scalar_t *bottom_input = input.data_ptr<scalar_t>();
        scalar_t *output_data = output.data_ptr<scalar_t>();
        scalar_t *mask_data = mask.data_ptr<scalar_t>();

        Ada_EDSCW_Pool1dForward<scalar_t>
        <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
          output_size, bottom_input,
          batches, channels,
          dim, kernel_d,
          stride_d, output_data,
          return_mask, mask_data);
        })
      );

    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
      fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
      exit(-1);
    }
  return 1;
}

int Ada_EDSCW_Pool2dForwardLauncher(const at::Tensor input, const int batches,
                                   const int channels, const int height,
                                   const int width, const int kernel_h,
                                   const int kernel_w, const int stride_h,
                                   const int stride_w, at::Tensor output,
                                   const bool return_mask, at::Tensor mask){
    const int output_size = batches * height/stride_h * width/stride_w * channels;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "Ada_EDSCW_Pool2dLauncherForward", ([&] {
        const scalar_t *bottom_input = input.data_ptr<scalar_t>();
        scalar_t *output_data = output.data_ptr<scalar_t>();
        scalar_t *mask_data = mask.data_ptr<scalar_t>();

        Ada_EDSCW_Pool2dForward<scalar_t>
        <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
          output_size, bottom_input,
          batches, channels,
          height, width,
          kernel_h, kernel_w,
          stride_h, stride_w,
          output_data, return_mask,
          mask_data);
        })
      );

    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
      fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
      exit(-1);
    }
  return 1;
}

int Ada_EDSCW_Pool3dForwardLauncher(const at::Tensor input,  const int batches,
                                   const int channels, const int depth,
                                   const int height, const int width,
                                   const int kernel_d, const int kernel_h,
                                   const int kernel_w, const int stride_d,
                                   const int stride_h, const int stride_w,
                                   at::Tensor output, const bool return_mask,
                                   at::Tensor mask){
    const int output_size = batches * depth/stride_d * height/stride_h * width/stride_w * channels;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "Ada_EDSCW_Pool3dLauncherForward", ([&] {
        const scalar_t *bottom_input = input.data_ptr<scalar_t>();
        scalar_t *output_data = output.data_ptr<scalar_t>();
        scalar_t *mask_data = mask.data_ptr<scalar_t>();


        Ada_EDSCW_Pool3dForward<scalar_t>
        <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
          output_size, bottom_input,
          batches, channels,
          depth, height,
          width, kernel_d,
          kernel_h, kernel_w,
          stride_d, stride_h,
          stride_w, output_data,
          return_mask, mask_data);
        })
      );

    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
      fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
      exit(-1);
    }
  return 1;
}


template <typename scalar_t>
__global__ void Ada_EDSCW_Pool1dBackward(const int nthreads,
                                        const scalar_t *diff_output, const scalar_t *data_input,
                                        const int batches, const int channels,
                                        const int dim, const int kernel_d,
                                        const int stride_d, scalar_t *diff_input){
    int pooled_dim = dim/stride_d;
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      int pd = index % pooled_dim;
      int c = (index / pooled_dim) % channels;
      int n = index / pooled_dim / channels;

      const int offset0 = (n * channels + c) * dim;
      const scalar_t *offset_data_input = data_input + offset0;

      const scalar_t diff_output_index = diff_output[index];

      scalar_t *offset_diff_input = diff_input + offset0;

      const int base_d = pd*stride_d - kernel_d/2;

      scalar_t act_sum = 0.;
      scalar_t mask_sum_avg = 0.;

      const scalar_t upper = n_limits<scalar_t>::max();
      const scalar_t lower = n_limits<scalar_t>::min();

      int count = 0.;

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;
        if(d_offset >= dim || d_offset < 0)continue;
        const int offset = d_offset;

        act_sum += offset_data_input[offset];
        count += 1;
      }
      scalar_t act_avg = act_sum/count;

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;

        if(d_offset >= dim || d_offset < 0)continue;
        const int offset = d_offset;

        scalar_t dist = dsc(offset_data_input[offset], act_avg);

        mask_sum_avg += exp(dist);

      }
      // Overflow check
      mask_sum_avg = clamp(mask_sum_avg, lower, upper);

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;

        if(d_offset >= dim || d_offset < 0)continue;
          const int offset = d_offset;

          scalar_t dist = dsc(offset_data_input[offset], act_avg);

          scalar_t mask = exp(dist)/mask_sum_avg;

          scalar_t weighted_grad = diff_output_index * mask;

          // Underflow check
          weighted_grad = clamp(weighted_grad, lower, upper);

          atomicAdd(offset_diff_input+offset, weighted_grad);
      }
    }
}

template <typename scalar_t>
__global__ void Ada_EDSCW_Pool2dBackward(const int nthreads,
                                        const scalar_t *diff_output, const scalar_t *data_input,
                                        const int batches, const int channels,
                                        const int height, const int width,
                                        const int kernel_h, const int kernel_w,
                                        const int stride_h, const int stride_w,
                                        scalar_t *diff_input){
    int pooled_height = height/stride_h;
    int pooled_width = width/stride_w;
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int c = (index / pooled_width / pooled_height) % channels;
      int n = index / pooled_width / pooled_height / channels;

      const int offset0 = (n * channels + c) * height * width;
      const scalar_t *offset_data_input = data_input + offset0;

      const scalar_t diff_output_index = diff_output[index];

      scalar_t *offset_diff_input = diff_input + offset0;

      const int base_y = ph*stride_h - kernel_h/2;
      const int base_x = pw*stride_w - kernel_w/2;

      scalar_t act_sum = 0.;
      scalar_t mask_sum_avg = 0.;

      const scalar_t upper = n_limits<scalar_t>::max();
      const scalar_t lower = n_limits<scalar_t>::min();

      int count = 0.;

      for(int iy=0; iy<kernel_h; iy++){
        const int y_offset = base_y + iy;
        if(y_offset >= height || y_offset < 0)continue;
        for(int ix=0; ix<kernel_w; ix++){
          const int x_offset = base_x + ix;
          if(x_offset >= width || x_offset < 0)continue;
          const int offset = y_offset*width + x_offset;

          act_sum += offset_data_input[offset];
          count += 1;
        }
      }
      scalar_t act_avg = act_sum/count;

      for(int iy=0; iy<kernel_h; iy++){
        const int y_offset = base_y + iy;

        if(y_offset >= height || y_offset < 0)continue;
        for(int ix=0; ix<kernel_w; ix++){
          const int x_offset = base_x + ix;

          if(x_offset >= width || x_offset < 0)continue;
          const int offset = y_offset*width + x_offset;

          scalar_t dist = dsc(offset_data_input[offset], act_avg);

          mask_sum_avg += exp(dist);

        }
      }
      // Overflow check
      mask_sum_avg = clamp(mask_sum_avg, lower, upper);

      for(int iy=0; iy<kernel_h; iy++){
        const int y_offset = base_y + iy;

        if(y_offset >= height || y_offset < 0)continue;
        for(int ix=0; ix<kernel_w; ix++){
          const int x_offset = base_x + ix;

          if(x_offset >= width || x_offset < 0)continue;
            const int offset = y_offset*width + x_offset;

            scalar_t dist = dsc(offset_data_input[offset], act_avg);

            scalar_t mask = exp(dist)/mask_sum_avg;

            scalar_t weighted_grad = diff_output_index * mask;

            // Underflow check
            weighted_grad = clamp(weighted_grad, lower, upper);

            atomicAdd(offset_diff_input+offset, weighted_grad);
        }
      }
    }
}

template <typename scalar_t>
__global__ void Ada_EDSCW_Pool3dBackward(const int nthreads,
                                        const scalar_t *diff_output, const scalar_t *data_input,
                                        const int batches, const int channels,
                                        const int depth, const int height,
                                        const int width, const int kernel_d,
                                        const int kernel_h, const int kernel_w ,
                                        const int stride_d, const int stride_h,
                                        const int stride_w, scalar_t *diff_input){
    int pooled_depth = depth/stride_d;
    int pooled_height = width/stride_h;
    int pooled_width = width/stride_w;
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int pd = (index / pooled_width / pooled_height) % pooled_depth;
      int c = (index / pooled_width / pooled_height / pooled_depth) % channels;
      int n = index / pooled_width / pooled_height / pooled_depth / channels;

      const int offset0 = (n * channels + c) * depth * height * width;
      const scalar_t *offset_data_input = data_input + offset0;

      const scalar_t diff_output_index = diff_output[index];

      scalar_t *offset_diff_input = diff_input + offset0;

      const int base_d = pd*stride_d - kernel_d/2;
      const int base_y = ph*stride_h - kernel_h/2;
      const int base_x = pw*stride_w - kernel_w/2;


      scalar_t act_sum = 0.;
      scalar_t mask_sum_avg = 0.;

      const scalar_t upper = n_limits<scalar_t>::max();
      const scalar_t lower = n_limits<scalar_t>::min();

      int count = 0.;

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;
        if(d_offset >= depth || d_offset < 0)continue;
        for(int iy=0; iy<kernel_h; iy++){
          const int y_offset = base_y + iy;
          if(y_offset >= height || y_offset < 0)continue;
          for(int ix=0; ix<kernel_w; ix++){
            const int x_offset = base_x + ix;
            if(x_offset >= width || x_offset < 0)continue;
            const int offset = d_offset*height + y_offset*width + x_offset;

            act_sum += offset_data_input[offset];
            count += 1;
          }
        }
      }
      scalar_t act_avg = act_sum/count;

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;

        if(d_offset >= depth || d_offset < 0)continue;
        for(int iy=0; iy<kernel_h; iy++){
          const int y_offset = base_y + iy;

          if(y_offset >= height || y_offset < 0)continue;
          for(int ix=0; ix<kernel_w; ix++){
            const int x_offset = base_x + ix;

            if(x_offset >= width || x_offset < 0)continue;
              const int offset = d_offset*height + y_offset*width + x_offset;

              scalar_t dist = dsc(offset_data_input[offset], act_avg);

              mask_sum_avg += exp(dist);

          }
        }
      }
      // Overflow check
      mask_sum_avg = clamp(mask_sum_avg, lower, upper);

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;

        if(d_offset >= depth || d_offset < 0)continue;
        for(int iy=0; iy<kernel_h; iy++){
          const int y_offset = base_y + iy;

          if(y_offset >= height || y_offset < 0)continue;
          for(int ix=0; ix<kernel_w; ix++){
            const int x_offset = base_x + ix;

            if(x_offset >= width || x_offset < 0)continue;
              const int offset = d_offset*height + y_offset*width + x_offset;

              scalar_t dist = dsc(offset_data_input[offset], act_avg);

              scalar_t mask = exp(dist)/mask_sum_avg;

              scalar_t weighted_grad = diff_output_index/mask;

              // Underflow check
              weighted_grad = clamp(weighted_grad, lower, upper);

              atomicAdd(offset_diff_input+offset, weighted_grad);
          }
        }
      }
    }
}

int Ada_EDSCW_Pool1dBackwardLauncher(const at::Tensor output_grad, const at::Tensor input,
                                    const int batches, const int channels,
                                    const int dim, const int kernel_d,
                                    const int stride_d, at::Tensor input_grad){

    const int output_size = batches * dim/stride_d * channels;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "Ada_EDSCW_Pool1dLauncherBackward", ([&] {
        scalar_t *diff_input = input_grad.data_ptr<scalar_t>();
        const scalar_t *diff_output = output_grad.data_ptr<scalar_t>();
        const scalar_t *data_input = input.data_ptr<scalar_t>();

        Ada_EDSCW_Pool1dBackward<scalar_t>
        <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
          output_size, diff_output,
          data_input, batches,
          channels, dim,
          kernel_d, stride_d,
          diff_input);
        }
        )
        );

    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
      fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
      exit(-1);
    }
  return 1;
}

int Ada_EDSCW_Pool2dBackwardLauncher(const at::Tensor output_grad, const at::Tensor input,
                                    const int batches, const int channels,
                                    const int height, const int width,
                                    const int kernel_h, const int kernel_w,
                                    const int stride_h, const int stride_w,
                                    at::Tensor input_grad){

    const int output_size = batches * height/stride_h * width/stride_w * channels;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "Ada_EDSCW_Pool2dLauncherBackward", ([&] {
        scalar_t *diff_input = input_grad.data_ptr<scalar_t>();
        const scalar_t *diff_output = output_grad.data_ptr<scalar_t>();
        const scalar_t *data_input = input.data_ptr<scalar_t>();

        Ada_EDSCW_Pool2dBackward<scalar_t>
        <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
          output_size, diff_output,
          data_input, batches,
          channels, height,
          width, kernel_h,
          kernel_w, stride_h,
          stride_w, diff_input);
        }
        )
        );

    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
      fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
      exit(-1);
    }
  return 1;
}

int Ada_EDSCW_Pool3dBackwardLauncher(const at::Tensor output_grad, const at::Tensor input,
                                    const int batches, const int channels,
                                    const int depth, const int height,
                                    const int width, const int kernel_d,
                                    const int kernel_h, const int kernel_w,
                                    const int stride_d, const int stride_h,
                                    const int stride_w, at::Tensor input_grad){

    const int output_size = batches * depth/stride_d * height/stride_h * width/stride_w * channels;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "Ada_EDSCW_Pool3dLauncherBackward", ([&] {
        scalar_t *diff_input = input_grad.data_ptr<scalar_t>();
        const scalar_t *diff_output = output_grad.data_ptr<scalar_t>();
        const scalar_t *data_input = input.data_ptr<scalar_t>();

        Ada_EDSCW_Pool3dBackward<scalar_t>
        <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
          output_size, diff_output,
          data_input, batches,
          channels, depth,
          height, width,
          kernel_d, kernel_h,
          kernel_w, stride_d,
          stride_h, stride_w,
          diff_input);
        }
        )
        );

    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
      fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
      exit(-1);
    }
  return 1;
}





template <typename scalar_t>
__global__ void Ada_EM_Pool1dForward(const int nthreads,
                                     const scalar_t *bottom_input, const int batches,
                                     const int channels, const int dim,
                                     const int kernel_d, const int stride_d,
                                     scalar_t *output_data, const bool return_mask,
                                     scalar_t *mask){
    int pooled_dim = dim/stride_d;
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      int pd = index % pooled_dim;
      int c = (index / pooled_dim) % channels;
      int n = index / pooled_dim / channels;

      const int offset = (n * channels + c) * dim;
      const scalar_t *offset_bottom_input = bottom_input + offset;

      const int base_d = pd*stride_d - kernel_d/2;

      scalar_t mask_sum_max = 0.;

      output_data[index] = 0.;
      const scalar_t upper = n_limits<scalar_t>::max();
      const scalar_t lower = n_limits<scalar_t>::min();
      const scalar_t zero = 0.;

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;
        if(d_offset >= dim || d_offset < 0)continue;
        const int offset = d_offset;

        mask_sum_max += exp(offset_bottom_input[offset]);

      }
      // Overflow check
      mask_sum_max = clamp(mask_sum_max, lower, upper);

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;

        if(d_offset >= dim || d_offset < 0)continue;
        const int offset = d_offset;

        scalar_t mask_ = exp(offset_bottom_input[offset])/ mask_sum_max;

        if (return_mask)mask[offset]= mask_;

        output_data[index] += offset_bottom_input[offset] * mask_;
        output_data[index] = clamp(output_data[index], zero, upper);
      }
    }
}


template <typename scalar_t>
__global__ void Ada_EM_Pool2dForward(const int nthreads,
                                  const scalar_t *bottom_input, const int batches,
                                  const int channels, const int height,
                                  const int width, const int kernel_h,
                                  const int kernel_w, const int stride_h,
                                  const int stride_w, scalar_t *output_data,
                                  const bool return_mask, scalar_t *mask){
    int pooled_height = height/stride_h;
    int pooled_width = width/stride_w;
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int c = (index / pooled_width / pooled_height) % channels;
      int n = index / pooled_width / pooled_height / channels;

      const int offset = (n * channels + c) * height * width;
      const scalar_t *offset_bottom_input = bottom_input + offset;

      const int base_y = ph*stride_h - kernel_h/2;
      const int base_x = pw*stride_w - kernel_w/2;

      scalar_t mask_sum_max = 0.;

      output_data[index] = 0.;
      const scalar_t upper = n_limits<scalar_t>::max();
      const scalar_t lower = n_limits<scalar_t>::min();
      const scalar_t zero = 0.;

      for(int iy=0; iy<kernel_h; iy++){
        const int y_offset = base_y + iy;

        if(y_offset >= height || y_offset < 0)continue;
        for(int ix=0; ix<kernel_w; ix++){
          const int x_offset = base_x + ix;

          if(x_offset >= width || x_offset < 0)continue;
          const int offset = y_offset*width + x_offset;

          mask_sum_max += exp(offset_bottom_input[offset]);

        }
      }
      // Overflow check
      mask_sum_max = clamp(mask_sum_max, lower, upper);


      for(int iy=0; iy<kernel_h; iy++){
        const int y_offset = base_y + iy;

        if(y_offset >= height || y_offset < 0)continue;
        for(int ix=0; ix<kernel_w; ix++){
          const int x_offset = base_x + ix;

          if(x_offset >= width || x_offset < 0)continue;
          const int offset = y_offset*width + x_offset;

          scalar_t mask_ = exp(offset_bottom_input[offset])/  mask_sum_max;

          if (return_mask)mask[offset]= mask_;

          output_data[index] += offset_bottom_input[offset] * mask_;
          output_data[index] = clamp(output_data[index], zero, upper);
        }
      }
    }
}


template <typename scalar_t>
__global__ void Ada_EM_Pool3dForward(const int nthreads,
                                    const scalar_t *bottom_input, const int batches,
                                    const int channels, const int depth,
                                    const int height, const int width,
                                    const int kernel_d, const int kernel_h,
                                    const int kernel_w, const int stride_d,
                                    const int stride_h, const int stride_w,
                                    scalar_t *output_data, const bool return_mask,
                                    scalar_t *mask){
    int pooled_depth = depth/stride_d;
    int pooled_height = height/stride_h;
    int pooled_width = width/stride_w;
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int pd = (index / pooled_width / pooled_height) % pooled_depth;
      int c = (index / pooled_width / pooled_height / pooled_depth) % channels;
      int n = index / pooled_width / pooled_height / pooled_depth / channels;

      const int offset = (n * channels + c) * depth * height * width;
      const scalar_t *offset_bottom_input = bottom_input + offset;

      const int base_d = pd*stride_d - kernel_d/2;
      const int base_y = ph*stride_h - kernel_h/2;
      const int base_x = pw*stride_w - kernel_w/2;

      scalar_t mask_sum_max = 0.;

      output_data[index] = 0.;
      const scalar_t upper = n_limits<scalar_t>::max();
      const scalar_t lower = n_limits<scalar_t>::min();
      const scalar_t zero = 0.;

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;

        if(d_offset >= depth || d_offset < 0)continue;
        for(int iy=0; iy<kernel_h; iy++){
          const int y_offset = base_y + iy;

          if(y_offset >= height || y_offset < 0)continue;
          for(int ix=0; ix<kernel_w; ix++){
            const int x_offset = base_x + ix;

            if(x_offset >= width || x_offset < 0)continue;
            const int offset = d_offset*height + y_offset*width + x_offset;

            mask_sum_max += exp(offset_bottom_input[offset]);

          }
        }
      }
      // Overflow check
      mask_sum_max = clamp(mask_sum_max, lower, upper);

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;

        if(d_offset >= depth || d_offset < 0)continue;
        for(int iy=0; iy<kernel_h; iy++){
          const int y_offset = base_y + iy;

          if(y_offset >= height || y_offset < 0)continue;
          for(int ix=0; ix<kernel_w; ix++){
            const int x_offset = base_x + ix;

            if(x_offset >= width || x_offset < 0)continue;
            const int offset = d_offset*height + y_offset*width + x_offset;

            scalar_t mask_ = exp(offset_bottom_input[offset])/mask_sum_max;

            if (return_mask)mask[offset]= mask_;

            output_data[index] += offset_bottom_input[offset] * mask_;
            output_data[index] = clamp(output_data[index], zero, upper);

          }
        }
      }
    }
}


int Ada_EM_Pool1dForwardLauncher(const at::Tensor input, const int batches,
                                 const int channels, const int dim,
                                 const int kernel_d, const int stride_d,
                                 at::Tensor output, const bool return_mask,
                                 at::Tensor mask){
    const int output_size = batches * dim/stride_d * channels;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "Ada_EM_Pool1dLauncherForward", ([&] {
        const scalar_t *bottom_input = input.data_ptr<scalar_t>();
        scalar_t *output_data = output.data_ptr<scalar_t>();
        scalar_t *mask_data = mask.data_ptr<scalar_t>();

        Ada_EM_Pool1dForward<scalar_t>
        <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
          output_size, bottom_input,
          batches, channels,
          dim, kernel_d,
          stride_d, output_data,
          return_mask, mask_data);
        })
      );

    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
      fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
      exit(-1);
    }
  return 1;
}

int Ada_EM_Pool2dForwardLauncher(const at::Tensor input, const int batches,
                                 const int channels, const int height,
                                 const int width, const int kernel_h,
                                 const int kernel_w, const int stride_h,
                                 const int stride_w, at::Tensor output,
                                 const bool return_mask, at::Tensor mask){
    const int output_size = batches * height/stride_h * width/stride_w * channels;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "Ada_EM_Pool2dLauncherForward", ([&] {
        const scalar_t *bottom_input = input.data_ptr<scalar_t>();
        scalar_t *output_data = output.data_ptr<scalar_t>();
        scalar_t *mask_data = mask.data_ptr<scalar_t>();

        Ada_EM_Pool2dForward<scalar_t>
        <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
          output_size, bottom_input,
          batches, channels,
          height, width,
          kernel_h, kernel_w,
          stride_h, stride_w,
          output_data, return_mask,
          mask_data);
        })
      );

    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
      fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
      exit(-1);
    }
  return 1;
}

int Ada_EM_Pool3dForwardLauncher(const at::Tensor input, const int batches,
                                 const int channels, const int depth,
                                 const int height, const int width,
                                 const int kernel_d, const int kernel_h,
                                 const int kernel_w, const int stride_d,
                                 const int stride_h, const int stride_w,
                                 at::Tensor output, const bool return_mask,
                                 at::Tensor mask){
    const int output_size = batches * depth/stride_d * height/stride_h * width/stride_w * channels;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "Ada_EM_Pool3dLauncherForward", ([&] {
        const scalar_t *bottom_input = input.data_ptr<scalar_t>();
        scalar_t *output_data = output.data_ptr<scalar_t>();
        scalar_t *mask_data = mask.data_ptr<scalar_t>();


        Ada_EM_Pool3dForward<scalar_t>
        <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
          output_size, bottom_input,
          batches, channels,
          depth, height,
          width, kernel_d,
          kernel_h, kernel_w,
          stride_d, stride_h,
          stride_w, output_data,
          return_mask, mask_data);
        })
      );

    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
      fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
      exit(-1);
    }
  return 1;
}


template <typename scalar_t>
__global__ void Ada_EM_Pool1dBackward(const int nthreads,
                                      const scalar_t *diff_output, const scalar_t *data_input,
                                      const int batches, const int channels,
                                      const int dim, const int kernel_d,
                                      const int stride_d, scalar_t *diff_input){
    int pooled_dim = dim/stride_d;
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      int pd = index % pooled_dim;
      int c = (index / pooled_dim) % channels;
      int n = index / pooled_dim / channels;

      const int offset0 = (n * channels + c) * dim;
      const scalar_t *offset_data_input = data_input + offset0;

      const scalar_t diff_output_index = diff_output[index];

      scalar_t *offset_diff_input = diff_input + offset0;

      const int base_d = pd*stride_d - kernel_d/2;

      scalar_t mask_sum_max = 0.;
      const scalar_t upper = n_limits<scalar_t>::max();
      const scalar_t lower = n_limits<scalar_t>::min();

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;

        if(d_offset >= dim || d_offset < 0)continue;
        const int offset = d_offset;

        mask_sum_max += exp(offset_data_input[offset]);

      }
      // Overflow check
      mask_sum_max = clamp(mask_sum_max, lower, upper);

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;

        if(d_offset >= dim || d_offset < 0)continue;
          const int offset = d_offset;

          scalar_t mask = exp(offset_data_input[offset])/mask_sum_max;

          scalar_t weighted_grad = diff_output_index * mask;

          // Underflow check
          weighted_grad = clamp(weighted_grad, lower, upper);

          atomicAdd(offset_diff_input+offset, weighted_grad);
      }
    }
}

template <typename scalar_t>
__global__ void Ada_EM_Pool2dBackward(const int nthreads,
                              const scalar_t *diff_output, const scalar_t *data_input,
                              const int batches, const int channels,
                              const int height, const int width,
                              const int kernel_h, const int kernel_w,
                              const int stride_h, const int stride_w,
                              scalar_t *diff_input){
    int pooled_height = height/stride_h;
    int pooled_width = width/stride_w;
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int c = (index / pooled_width / pooled_height) % channels;
      int n = index / pooled_width / pooled_height / channels;

      const int offset0 = (n * channels + c) * height * width;
      const scalar_t *offset_data_input = data_input + offset0;

      const scalar_t diff_output_index = diff_output[index];

      scalar_t *offset_diff_input = diff_input + offset0;

      const int base_y = ph*stride_h - kernel_h/2;
      const int base_x = pw*stride_w - kernel_w/2;

      scalar_t mask_sum_max = 0.;

      const scalar_t upper = n_limits<scalar_t>::max();
      const scalar_t lower = n_limits<scalar_t>::min();

      for(int iy=0; iy<kernel_h; iy++){
        const int y_offset = base_y + iy;

        if(y_offset >= height || y_offset < 0)continue;
        for(int ix=0; ix<kernel_w; ix++){
          const int x_offset = base_x + ix;

          if(x_offset >= width || x_offset < 0)continue;
          const int offset = y_offset*width + x_offset;

          mask_sum_max += exp(offset_data_input[offset]);

        }
      }
      // Overflow check
      mask_sum_max = clamp(mask_sum_max, lower, upper);

      for(int iy=0; iy<kernel_h; iy++){
        const int y_offset = base_y + iy;

        if(y_offset >= height || y_offset < 0)continue;
        for(int ix=0; ix<kernel_w; ix++){
          const int x_offset = base_x + ix;

          if(x_offset >= width || x_offset < 0)continue;
            const int offset = y_offset*width + x_offset;

            scalar_t mask = exp(offset_data_input[offset])/mask_sum_max;

            scalar_t weighted_grad = diff_output_index * mask;

            // Underflow check
            weighted_grad = clamp(weighted_grad, lower, upper);

            atomicAdd(offset_diff_input+offset, weighted_grad);
        }
      }
    }
}

template <typename scalar_t>
__global__ void Ada_EM_Pool3dBackward(const int nthreads,
                              const scalar_t *diff_output, const scalar_t *data_input,
                              const int batches, const int channels,
                              const int depth, const int height,
                              const int width, const int kernel_d,
                              const int kernel_h, const int kernel_w ,
                              const int stride_d, const int stride_h,
                              const int stride_w, scalar_t *diff_input){
    int pooled_depth = depth/stride_d;
    int pooled_height = width/stride_h;
    int pooled_width = width/stride_w;
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int pd = (index / pooled_width / pooled_height) % pooled_depth;
      int c = (index / pooled_width / pooled_height / pooled_depth) % channels;
      int n = index / pooled_width / pooled_height / pooled_depth / channels;

      const int offset0 = (n * channels + c) * depth * height * width;
      const scalar_t *offset_data_input = data_input + offset0;

      const scalar_t diff_output_index = diff_output[index];

      scalar_t *offset_diff_input = diff_input + offset0;

      const int base_d = pd*stride_d - kernel_d/2;

      const int base_y = ph*stride_h - kernel_h/2;

      const int base_x = pw*stride_w - kernel_w/2;

      scalar_t mask_sum_max = 0.;

      const scalar_t upper = n_limits<scalar_t>::max();
      const scalar_t lower = n_limits<scalar_t>::min();


      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;

        if(d_offset >= depth || d_offset < 0)continue;
        for(int iy=0; iy<kernel_h; iy++){
          const int y_offset = base_y + iy;

          if(y_offset >= height || y_offset < 0)continue;
          for(int ix=0; ix<kernel_w; ix++){
            const int x_offset = base_x + ix;

            if(x_offset >= width || x_offset < 0)continue;
              const int offset = d_offset*height + y_offset*width + x_offset;

              mask_sum_max += exp(offset_data_input[offset]);

          }
        }
      }
      // Overflow check
      mask_sum_max = clamp(mask_sum_max, lower, upper);

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;

        if(d_offset >= depth || d_offset < 0)continue;
        for(int iy=0; iy<kernel_h; iy++){
          const int y_offset = base_y + iy;

          if(y_offset >= height || y_offset < 0)continue;
          for(int ix=0; ix<kernel_w; ix++){
            const int x_offset = base_x + ix;

            if(x_offset >= width || x_offset < 0)continue;
              const int offset = d_offset*height + y_offset*width + x_offset;

              scalar_t mask = exp(offset_data_input[offset])/mask_sum_max;

              scalar_t weighted_grad = diff_output_index/mask;

              // Underflow check
              weighted_grad = clamp(weighted_grad, lower, upper);

              atomicAdd(offset_diff_input+offset, weighted_grad);
          }
        }
      }
    }
}

int Ada_EM_Pool1dBackwardLauncher(const at::Tensor output_grad, const at::Tensor input,
                               const int batches, const int channels,
                               const int dim, const int kernel_d,
                               const int stride_d, at::Tensor input_grad){

    const int output_size = batches * dim/stride_d * channels;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "Ada_EM_Pool1dLauncherBackward", ([&] {
        scalar_t *diff_input = input_grad.data_ptr<scalar_t>();
        const scalar_t *diff_output = output_grad.data_ptr<scalar_t>();
        const scalar_t *data_input = input.data_ptr<scalar_t>();

        Ada_EM_Pool1dBackward<scalar_t>
        <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
          output_size, diff_output,
          data_input, batches,
          channels, dim,
          kernel_d, stride_d,
          diff_input);
        }
        )
        );

    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
      fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
      exit(-1);
    }
  return 1;
}

int Ada_EM_Pool2dBackwardLauncher(const at::Tensor output_grad, const at::Tensor input,
                               const int batches, const int channels,
                               const int height, const int width,
                               const int kernel_h, const int kernel_w,
                               const int stride_h, const int stride_w,
                               at::Tensor input_grad){

    const int output_size = batches * height/stride_h * width/stride_w * channels;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "Ada_EM_Pool2dLauncherBackward", ([&] {
        scalar_t *diff_input = input_grad.data_ptr<scalar_t>();
        const scalar_t *diff_output = output_grad.data_ptr<scalar_t>();
        const scalar_t *data_input = input.data_ptr<scalar_t>();
        Ada_EM_Pool2dBackward<scalar_t>
        <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
          output_size, diff_output,
          data_input, batches,
          channels, height,
          width, kernel_h,
          kernel_w, stride_h,
          stride_w, diff_input);
        }
        )
        );

    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
      fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
      exit(-1);
    }
  return 1;
}

int Ada_EM_Pool3dBackwardLauncher(const at::Tensor output_grad, const at::Tensor input,
                               const int batches, const int channels,
                               const int depth, const int height,
                               const int width, const int kernel_d,
                               const int kernel_h, const int kernel_w,
                               const int stride_d, const int stride_h,
                               const int stride_w, at::Tensor input_grad){

    const int output_size = batches * depth/stride_d * height/stride_h * width/stride_w * channels;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "Ada_EM_Pool3dLauncherBackward", ([&] {
        scalar_t *diff_input = input_grad.data_ptr<scalar_t>();
        const scalar_t *diff_output = output_grad.data_ptr<scalar_t>();
        const scalar_t *data_input = input.data_ptr<scalar_t>();

        Ada_EM_Pool3dBackward<scalar_t>
        <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
          output_size, diff_output,
          data_input, batches,
          channels, depth, height,
          width, kernel_d,
          kernel_h, kernel_w,
          stride_d, stride_h,
          stride_w, diff_input);
        }
        )
        );

    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
      fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
      exit(-1);
    }
  return 1;
}





template <typename scalar_t>
__global__ void IDW_Pool1dForward(const int nthreads,
                                  const scalar_t *bottom_input, const int batches,
                                  const int channels, const int dim,
                                  const int kernel_d, const int stride_d,
                                  scalar_t *output_data, const bool return_mask,
                                  scalar_t *mask){
    int pooled_dim = dim/stride_d;
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      int pd = index % pooled_dim;
      int c = (index / pooled_dim) % channels;
      int n = index / pooled_dim / channels;

      const int offset = (n * channels + c) * dim;
      const scalar_t *offset_bottom_input = bottom_input + offset;

      const int base_d = pd*stride_d - kernel_d/2;

      scalar_t act_sum = 0.;
      scalar_t mask_sum_avg = 0.;

      output_data[index] = 0.;
      const scalar_t upper = n_limits<scalar_t>::max();
      const scalar_t lower = n_limits<scalar_t>::min();
      const scalar_t zero = 0.;

      int count = 0.;

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;
        if(d_offset >= dim || d_offset < 0)continue;
        const int offset = d_offset;

        act_sum += offset_bottom_input[offset];
        count += 1;
      }
      scalar_t act_avg = act_sum/count;

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;
        if(d_offset >= dim || d_offset < 0)continue;
        const int offset = d_offset;

        scalar_t dist = l2(offset_bottom_input[offset], act_avg);

        mask_sum_avg += pow(dist,-1);

      }
      // Overflow check
      mask_sum_avg = clamp(mask_sum_avg, lower, upper);


      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;

        if(d_offset >= dim || d_offset < 0)continue;
        const int offset = d_offset;

        scalar_t dist = l2(offset_bottom_input[offset], act_avg);

        scalar_t mask_ = pow(dist,-1)/mask_sum_avg;

        if (return_mask)mask[offset]= mask_;

        output_data[index] += offset_bottom_input[offset] * mask_;
        output_data[index] = clamp(output_data[index], zero, upper);
      }
    }
}


template <typename scalar_t>
__global__ void IDW_Pool2dForward(const int nthreads,
                                  const scalar_t *bottom_input, const int batches,
                                  const int channels, const int height,
                                  const int width, const int kernel_h,
                                  const int kernel_w, const int stride_h,
                                  const int stride_w, scalar_t *output_data,
                                  const bool return_mask, scalar_t *mask){
    int pooled_height = height/stride_h;
    int pooled_width = width/stride_w;
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int c = (index / pooled_width / pooled_height) % channels;
      int n = index / pooled_width / pooled_height / channels;

      const int offset = (n * channels + c) * height * width;
      const scalar_t *offset_bottom_input = bottom_input + offset;

      const int base_y = ph*stride_h - kernel_h/2;
      const int base_x = pw*stride_w - kernel_w/2;

      scalar_t act_sum = 0.;
      scalar_t mask_sum_avg = 0.;

      output_data[index] = 0.;
      const scalar_t upper = n_limits<scalar_t>::max();
      const scalar_t lower = n_limits<scalar_t>::min();
      const scalar_t zero = 0.;

      int count = 0.;

      for(int iy=0; iy<kernel_h; iy++){
        const int y_offset = base_y + iy;
        if(y_offset >= height || y_offset < 0)continue;
        for(int ix=0; ix<kernel_w; ix++){
          const int x_offset = base_x + ix;
          if(x_offset >= width || x_offset < 0)continue;
          const int offset = y_offset*width + x_offset;

          act_sum += offset_bottom_input[offset];
          count += 1;
        }
      }
      scalar_t act_avg = act_sum/count;

      for(int iy=0; iy<kernel_h; iy++){
        const int y_offset = base_y + iy;

        if(y_offset >= height || y_offset < 0)continue;
        for(int ix=0; ix<kernel_w; ix++){
          const int x_offset = base_x + ix;

          if(x_offset >= width || x_offset < 0)continue;
          const int offset = y_offset*width + x_offset;

          scalar_t dist = l2(offset_bottom_input[offset], act_avg);

          mask_sum_avg += pow(dist,-1);

        }
      }
      // Overflow check
      mask_sum_avg = clamp(mask_sum_avg, lower, upper);


      for(int iy=0; iy<kernel_h; iy++){
        const int y_offset = base_y + iy;

        if(y_offset >= height || y_offset < 0)continue;
        for(int ix=0; ix<kernel_w; ix++){
          const int x_offset = base_x + ix;

          if(x_offset >= width || x_offset < 0)continue;
          const int offset = y_offset*width + x_offset;

          scalar_t dist = l2(offset_bottom_input[offset], act_avg);

          scalar_t mask_ = pow(dist,-1)/mask_sum_avg;

          if (return_mask)mask[offset]= mask_;

          output_data[index] += offset_bottom_input[offset] * mask_;
          output_data[index] = clamp(output_data[index], zero, upper);
        }
      }
    }
}


template <typename scalar_t>
__global__ void IDW_Pool3dForward(const int nthreads,
                                  const scalar_t *bottom_input, const int batches,
                                  const int channels, const int depth,
                                  const int height, const int width,
                                  const int kernel_d, const int kernel_h,
                                  const int kernel_w, const int stride_d,
                                  const int stride_h, const int stride_w,
                                  scalar_t *output_data, const bool return_mask,
                                  scalar_t *mask){
    int pooled_depth = depth/stride_d;
    int pooled_height = height/stride_h;
    int pooled_width = width/stride_w;
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int pd = (index / pooled_width / pooled_height) % pooled_depth;
      int c = (index / pooled_width / pooled_height / pooled_depth) % channels;
      int n = index / pooled_width / pooled_height / pooled_depth / channels;

      const int offset = (n * channels + c) * depth * height * width;
      const scalar_t *offset_bottom_input = bottom_input + offset;

      const int base_d = pd*stride_d - kernel_d/2;
      const int base_y = ph*stride_h - kernel_h/2;
      const int base_x = pw*stride_w - kernel_w/2;

      scalar_t act_sum = 0.;
      scalar_t mask_sum_avg = 0.;

      output_data[index] = 0.;
      const scalar_t upper = n_limits<scalar_t>::max();
      const scalar_t lower = n_limits<scalar_t>::min();
      const scalar_t zero = 0.;

      int count = 0.;

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;
        if(d_offset >= depth || d_offset < 0)continue;
        for(int iy=0; iy<kernel_h; iy++){
          const int y_offset = base_y + iy;
          if(y_offset >= height || y_offset < 0)continue;
          for(int ix=0; ix<kernel_w; ix++){
            const int x_offset = base_x + ix;
            if(x_offset >= width || x_offset < 0)continue;
            const int offset = d_offset*height + y_offset*width + x_offset;

            act_sum += offset_bottom_input[offset];
            count += 1;
          }
        }
      }
      scalar_t act_avg = act_sum/count;


      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;

        if(d_offset >= depth || d_offset < 0)continue;
        for(int iy=0; iy<kernel_h; iy++){
          const int y_offset = base_y + iy;

          if(y_offset >= height || y_offset < 0)continue;
          for(int ix=0; ix<kernel_w; ix++){
            const int x_offset = base_x + ix;

            if(x_offset >= width || x_offset < 0)continue;
            const int offset = d_offset*height + y_offset*width + x_offset;

            scalar_t dist = l2(offset_bottom_input[offset], act_avg);

            mask_sum_avg += pow(dist,-1);

          }
        }
      }
      // Overflow check
      mask_sum_avg = clamp(mask_sum_avg, lower, upper);

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;

        if(d_offset >= depth || d_offset < 0)continue;
        for(int iy=0; iy<kernel_h; iy++){
          const int y_offset = base_y + iy;

          if(y_offset >= height || y_offset < 0)continue;
          for(int ix=0; ix<kernel_w; ix++){
            const int x_offset = base_x + ix;

            if(x_offset >= width || x_offset < 0)continue;
            const int offset = d_offset*height + y_offset*width + x_offset;

            scalar_t dist = l2(offset_bottom_input[offset], act_avg);

            scalar_t mask_ = pow(dist,-1)/mask_sum_avg;

            if (return_mask)mask[offset]= mask_;

            output_data[index] += offset_bottom_input[offset] * mask_;
            output_data[index] = clamp(output_data[index], zero, upper);

          }
        }
      }
    }
}


int IDW_Pool1dForwardLauncher(const at::Tensor input, const int batches,
                              const int channels, const int dim,
                              const int kernel_d, const int stride_d,
                              at::Tensor output, const bool return_mask,
                              at::Tensor mask){
    const int output_size = batches * dim/stride_d * channels;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "IDW_Pool1dLauncherForward", ([&] {
        const scalar_t *bottom_input = input.data_ptr<scalar_t>();
        scalar_t *output_data = output.data_ptr<scalar_t>();
        scalar_t *mask_data = mask.data_ptr<scalar_t>();

        IDW_Pool1dForward<scalar_t>
        <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
          output_size, bottom_input,
          batches, channels,
          dim, kernel_d,
          stride_d, output_data,
          return_mask, mask_data);
        })
      );

    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
      fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
      exit(-1);
    }
  return 1;
}

int IDW_Pool2dForwardLauncher(const at::Tensor input, const int batches,
                              const int channels, const int height,
                              const int width, const int kernel_h,
                              const int kernel_w, const int stride_h,
                              const int stride_w, at::Tensor output,
                              const bool return_mask, at::Tensor mask){
    const int output_size = batches * height/stride_h * width/stride_w * channels;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "IDW_Pool2dLauncherForward", ([&] {
        const scalar_t *bottom_input = input.data_ptr<scalar_t>();
        scalar_t *output_data = output.data_ptr<scalar_t>();
        scalar_t *mask_data = mask.data_ptr<scalar_t>();

        IDW_Pool2dForward<scalar_t>
        <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
          output_size, bottom_input,
          batches, channels,
          height, width,
          kernel_h, kernel_w,
          stride_h, stride_w,
          output_data, return_mask,
          mask_data);
        })
      );

    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
      fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
      exit(-1);
    }
  return 1;
}

int IDW_Pool3dForwardLauncher(const at::Tensor input, const int batches,
                              const int channels, const int depth,
                              const int height, const int width,
                              const int kernel_d, const int kernel_h,
                              const int kernel_w, const int stride_d,
                              const int stride_h, const int stride_w,
                              at::Tensor output, const bool return_mask,
                              at::Tensor mask){
    const int output_size = batches * depth/stride_d * height/stride_h * width/stride_w * channels;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "IDW_Pool3dLauncherForward", ([&] {
        const scalar_t *bottom_input = input.data_ptr<scalar_t>();
        scalar_t *output_data = output.data_ptr<scalar_t>();
        scalar_t *mask_data = mask.data_ptr<scalar_t>();


        IDW_Pool3dForward<scalar_t>
        <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
          output_size, bottom_input,
          batches, channels,
          depth, height,
          width, kernel_d,
          kernel_h, kernel_w,
          stride_d, stride_h,
          stride_w, output_data,
          return_mask, mask_data);
        })
      );

    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
      fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
      exit(-1);
    }
  return 1;
}


template <typename scalar_t>
__global__ void IDW_Pool1dBackward(const int nthreads,
                                   const scalar_t *diff_output, const scalar_t *data_input,
                                   const int batches, const int channels,
                                   const int dim, const int kernel_d,
                                   const int stride_d, scalar_t *diff_input){
    int pooled_dim = dim/stride_d;
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      int pd = index % pooled_dim;
      int c = (index / pooled_dim) % channels;
      int n = index / pooled_dim / channels;

      const int offset0 = (n * channels + c) * dim;
      const scalar_t *offset_data_input = data_input + offset0;

      const scalar_t diff_output_index = diff_output[index];

      scalar_t *offset_diff_input = diff_input + offset0;

      const int base_d = pd*stride_d - kernel_d/2;

      scalar_t act_sum = 0.;
      scalar_t mask_sum_avg = 0.;

      const scalar_t upper = n_limits<scalar_t>::max();
      const scalar_t lower = n_limits<scalar_t>::min();

      int count = 0.;

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;
        if(d_offset >= dim || d_offset < 0)continue;
        const int offset = d_offset;

        act_sum += offset_data_input[offset];
        count += 1;
      }
      scalar_t act_avg = act_sum/count;

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;

        if(d_offset >= dim || d_offset < 0)continue;
        const int offset = d_offset;

        scalar_t dist = l2(offset_data_input[offset], act_avg);

        mask_sum_avg += pow(dist,-1);

      }
      // Overflow check
      mask_sum_avg = clamp(mask_sum_avg, lower, upper);

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;

        if(d_offset >= dim || d_offset < 0)continue;
          const int offset = d_offset;

          scalar_t dist = l2(offset_data_input[offset], act_avg);

          scalar_t mask = pow(dist,-1)/mask_sum_avg;

          scalar_t weighted_grad = diff_output_index * mask;

          // Underflow check
          weighted_grad = clamp(weighted_grad, lower, upper);

          atomicAdd(offset_diff_input+offset, weighted_grad);
      }
    }
}

template <typename scalar_t>
__global__ void IDW_Pool2dBackward(const int nthreads,
                                   const scalar_t *diff_output, const scalar_t *data_input,
                                   const int batches, const int channels,
                                   const int height, const int width,
                                   const int kernel_h, const int kernel_w,
                                   const int stride_h,const int stride_w,
                                   scalar_t *diff_input){
    int pooled_height = height/stride_h;
    int pooled_width = width/stride_w;
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int c = (index / pooled_width / pooled_height) % channels;
      int n = index / pooled_width / pooled_height / channels;

      const int offset0 = (n * channels + c) * height * width;
      const scalar_t *offset_data_input = data_input + offset0;

      const scalar_t diff_output_index = diff_output[index];

      scalar_t *offset_diff_input = diff_input + offset0;

      const int base_y = ph*stride_h - kernel_h/2;
      const int base_x = pw*stride_w - kernel_w/2;

      scalar_t act_sum = 0.;
      scalar_t mask_sum_avg = 0.;

      const scalar_t upper = n_limits<scalar_t>::max();
      const scalar_t lower = n_limits<scalar_t>::min();

      int count = 0.;

      for(int iy=0; iy<kernel_h; iy++){
        const int y_offset = base_y + iy;
        if(y_offset >= height || y_offset < 0)continue;
        for(int ix=0; ix<kernel_w; ix++){
          const int x_offset = base_x + ix;
          if(x_offset >= width || x_offset < 0)continue;
          const int offset = y_offset*width + x_offset;

          act_sum += offset_data_input[offset];
          count += 1;
        }
      }
      scalar_t act_avg = act_sum/count;

      for(int iy=0; iy<kernel_h; iy++){
        const int y_offset = base_y + iy;

        if(y_offset >= height || y_offset < 0)continue;
        for(int ix=0; ix<kernel_w; ix++){
          const int x_offset = base_x + ix;

          if(x_offset >= width || x_offset < 0)continue;
          const int offset = y_offset*width + x_offset;

          scalar_t dist = l2(offset_data_input[offset], act_avg);

          mask_sum_avg += pow(dist,-1);

        }
      }
      // Overflow check
      mask_sum_avg = clamp(mask_sum_avg, lower, upper);

      for(int iy=0; iy<kernel_h; iy++){
        const int y_offset = base_y + iy;

        if(y_offset >= height || y_offset < 0)continue;
        for(int ix=0; ix<kernel_w; ix++){
          const int x_offset = base_x + ix;

          if(x_offset >= width || x_offset < 0)continue;
            const int offset = y_offset*width + x_offset;

            scalar_t dist = l2(offset_data_input[offset], act_avg);

            scalar_t mask = pow(dist,-1)/mask_sum_avg;

            scalar_t weighted_grad = diff_output_index * mask;

            // Underflow check
            weighted_grad = clamp(weighted_grad, lower, upper);

            atomicAdd(offset_diff_input+offset, weighted_grad);
        }
      }
    }
}

template <typename scalar_t>
__global__ void IDW_Pool3dBackward(const int nthreads,
                                   const scalar_t *diff_output, const scalar_t *data_input,
                                   const int batches, const int channels,
                                   const int depth, const int height,
                                   const int width, const int kernel_d,
                                   const int kernel_h, const int kernel_w ,
                                   const int stride_d, const int stride_h,
                                   const int stride_w, scalar_t *diff_input){
    int pooled_depth = depth/stride_d;
    int pooled_height = width/stride_h;
    int pooled_width = width/stride_w;
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int pd = (index / pooled_width / pooled_height) % pooled_depth;
      int c = (index / pooled_width / pooled_height / pooled_depth) % channels;
      int n = index / pooled_width / pooled_height / pooled_depth / channels;

      const int offset0 = (n * channels + c) * depth * height * width;
      const scalar_t *offset_data_input = data_input + offset0;

      const scalar_t diff_output_index = diff_output[index];

      scalar_t *offset_diff_input = diff_input + offset0;

      const int base_d = pd*stride_d - kernel_d/2;
      const int base_y = ph*stride_h - kernel_h/2;
      const int base_x = pw*stride_w - kernel_w/2;


      scalar_t act_sum = 0.;
      scalar_t mask_sum_avg = 0.;

      const scalar_t upper = n_limits<scalar_t>::max();
      const scalar_t lower = n_limits<scalar_t>::min();

      int count = 0.;

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;
        if(d_offset >= depth || d_offset < 0)continue;
        for(int iy=0; iy<kernel_h; iy++){
          const int y_offset = base_y + iy;
          if(y_offset >= height || y_offset < 0)continue;
          for(int ix=0; ix<kernel_w; ix++){
            const int x_offset = base_x + ix;
            if(x_offset >= width || x_offset < 0)continue;
            const int offset = d_offset*height + y_offset*width + x_offset;

            act_sum += offset_data_input[offset];
            count += 1;
          }
        }
      }
      scalar_t act_avg = act_sum/count;

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;

        if(d_offset >= depth || d_offset < 0)continue;
        for(int iy=0; iy<kernel_h; iy++){
          const int y_offset = base_y + iy;

          if(y_offset >= height || y_offset < 0)continue;
          for(int ix=0; ix<kernel_w; ix++){
            const int x_offset = base_x + ix;

            if(x_offset >= width || x_offset < 0)continue;
              const int offset = d_offset*height + y_offset*width + x_offset;

              scalar_t dist = l2(offset_data_input[offset], act_avg);

              mask_sum_avg += pow(dist,-1);

          }
        }
      }
      // Overflow check
      mask_sum_avg = clamp(mask_sum_avg, lower, upper);

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;

        if(d_offset >= depth || d_offset < 0)continue;
        for(int iy=0; iy<kernel_h; iy++){
          const int y_offset = base_y + iy;

          if(y_offset >= height || y_offset < 0)continue;
          for(int ix=0; ix<kernel_w; ix++){
            const int x_offset = base_x + ix;

            if(x_offset >= width || x_offset < 0)continue;
              const int offset = d_offset*height + y_offset*width + x_offset;

              scalar_t dist = l2(offset_data_input[offset], act_avg);

              scalar_t mask = pow(dist,-1)/mask_sum_avg;

              scalar_t weighted_grad = diff_output_index/mask;

              // Underflow check
              weighted_grad = clamp(weighted_grad, lower, upper);

              atomicAdd(offset_diff_input+offset, weighted_grad);
          }
        }
      }
    }
}

int IDW_Pool1dBackwardLauncher(const at::Tensor output_grad, const at::Tensor input,
                               const int batches, const int channels,
                               const int dim, const int kernel_d,
                               const int stride_d, at::Tensor input_grad){

    const int output_size = batches * dim/stride_d * channels;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "IDW_Pool1dLauncherBackward", ([&] {
        scalar_t *diff_input = input_grad.data_ptr<scalar_t>();
        const scalar_t *diff_output = output_grad.data_ptr<scalar_t>();
        const scalar_t *data_input = input.data_ptr<scalar_t>();

        IDW_Pool1dBackward<scalar_t>
        <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
          output_size, diff_output,
          data_input, batches,
          channels, dim,
          kernel_d, stride_d,
          diff_input);
        }
        )
        );

    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
      fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
      exit(-1);
    }
  return 1;
}

int IDW_Pool2dBackwardLauncher(const at::Tensor output_grad, const at::Tensor input,
                               const int batches, const int channels,
                               const int height, const int width,
                               const int kernel_h, const int kernel_w,
                               const int stride_h, const int stride_w,
                               at::Tensor input_grad){

    const int output_size = batches * height/stride_h * width/stride_w * channels;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "IDW_Pool2dLauncherBackward", ([&] {
        scalar_t *diff_input = input_grad.data_ptr<scalar_t>();
        const scalar_t *diff_output = output_grad.data_ptr<scalar_t>();
        const scalar_t *data_input = input.data_ptr<scalar_t>();

        IDW_Pool2dBackward<scalar_t>
        <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
          output_size, diff_output,
          data_input, batches,
          channels, height,
          width, kernel_h,
          kernel_w, stride_h,
          stride_w, diff_input);
        }
        )
        );

    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
      fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
      exit(-1);
    }
  return 1;
}

int IDW_Pool3dBackwardLauncher(const at::Tensor output_grad, const at::Tensor input,
                               const int batches, const int channels,
                               const int depth, const int height,
                               const int width, const int kernel_d,
                               const int kernel_h, const int kernel_w,
                               const int stride_d, const int stride_h,
                               const int stride_w, at::Tensor input_grad){

    const int output_size = batches * depth/stride_d * height/stride_h * width/stride_w * channels;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        input.scalar_type(), "IDW_Pool3dLauncherBackward", ([&] {
        scalar_t *diff_input = input_grad.data_ptr<scalar_t>();
        const scalar_t *diff_output = output_grad.data_ptr<scalar_t>();
        const scalar_t *data_input = input.data_ptr<scalar_t>();

        IDW_Pool3dBackward<scalar_t>
        <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
          output_size, diff_output,
          data_input, batches,
          channels, depth,
          height, width,
          kernel_d, kernel_h,
          kernel_w, stride_d,
          stride_h, stride_w,
          diff_input);
        }
        )
        );

    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err) {
      fprintf(stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString(err));
      exit(-1);
    }
  return 1;
}
