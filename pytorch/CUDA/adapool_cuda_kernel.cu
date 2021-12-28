#include <float.h>
#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>
# include <stdio.h>
#include "limits.cuh"

using namespace at;  // fix for pytorch<=0.4.1

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 1024


// `max_block_num` may need to change based on the CUDA version
// Use the following guide:
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications
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

// Euclidean distance (L2)
template <typename scalar_t>
__device__  scalar_t l2(const scalar_t x1, const scalar_t x2) {
  const scalar_t result = sqrt(pow(x1-x2,2));
  return result;
}

// City block distance (L1)
template <typename scalar_t>
__device__  scalar_t l1(const scalar_t x1, const scalar_t x2) {
  const scalar_t result = abs(x1-x2);
  return result;
}

// DSC
template <typename scalar_t>
__device__  scalar_t dsc(const scalar_t x1, const scalar_t x2) {
  const scalar_t result = (2*abs(x1*x2))/(pow(x1,2)+pow(x2,2));
  return result;
}

/*
---  S T A R T  O F  T E M P L A T E  A D A P O O L 1 D F O R W A R D ---
    [About]
        CUDA template utilising `CUDA_1D_KERNEL_LOOP` macro for iterating over the tensor.
        ! Note: This is the native implementation of adaPool1d and because of the large requirement
        in resources it may be unstable. Further refinement may be required!
    [Params]
        - No default for template parameter
    [Args]
        - nthreads: (constant) Integer, for the number of threads. From NVidia's website: "CUDA architecture limits the numbers of threads per block (1024 threads per block limit)".
        - bottom_input: (constant) Scalar_t, tensor for the input data.
        - bottom_beta: (constant) Scalar_t, tensor with beta parameter to be used during pooling.
        - batches: (constant) Integer, for the number of batches.
        - channels: (constant) Integer, for the number of channels.
        - dim: (constant) Integer, specifies the size of the iterable dimension of the tensor to pool over.
        - kernel_d: (constant) Integer, for the size of the kernel.
        - stride_d: (constant) Integer, for the steps taken between kernels.
        - output_data: constant) Scalar_t, tensor to assign the calculated output data.
        - return_mask: (constant) Boolean, if the calculated mask should be returned.
        - mask: (constant) Scalar_t, tensor of size # kernel ops (output size) x `kernel_d`.

*/
template <typename scalar_t>
__global__ void AdaPool1dForward(const int nthreads,
                                 const scalar_t *bottom_input, const scalar_t *bottom_beta,
                                 const int batches, const int channels,
                                 const int dim, const int kernel_d,
                                 const int stride_d, scalar_t *output_data,
                                 const bool return_mask, scalar_t *mask){
    int pooled_dim = dim/stride_d;
    // Run in parallel for each cell within each kernel region
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      int pd = index % pooled_dim; // index of each kernel operation in relation to the position in the input
      int c = (index / pooled_dim) % channels;
      int n = index / pooled_dim / channels;

      const int offset = (n * channels + c) * dim; // initial offset
      const scalar_t *offset_bottom_input = bottom_input + offset;

      const int base_d = pd*stride_d; // start cell index for each kernel
      if (base_d > dim - kernel_d)break; // limit iterations for the index of the final kernel location in the input

      // --- Initialisations happen here ----
      scalar_t act_sum = 0.;
      scalar_t mask_sum_avg = 0.;
      scalar_t mask_sum_max = 0.;

      output_data[index] = 0.;
      const scalar_t upper = n_limits<scalar_t>::max();
      const scalar_t lower = n_limits<scalar_t>::min();
      const scalar_t zero = 0.;
      const scalar_t one = 1.;

      const scalar_t b = clamp(bottom_beta[index], zero, one);

      int count = 0.; // used for calculating the average

      // Iterate over inputs cells within each kernel region in the input
      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;

        if(d_offset >= dim || d_offset < 0)continue; // check if the offset index is valid (not larger than or equal to the size of the dimension) OR smaller than 0 (for fool proofing)
        const int offset = d_offset;

        // Use this for verbose when debugging
        //printf("(pd: %d), base_d: %d, id: %d, d_offset: %d \n", pd, base_d, id, d_offset);

        act_sum += offset_bottom_input[offset];
        count += 1;
      }
      scalar_t act_avg = act_sum/count; // average calculation

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;

        if(d_offset >= dim || d_offset < 0)continue;
        const int offset = d_offset;

        scalar_t dist = dsc(offset_bottom_input[offset], act_avg); // Dice Sørensen Coefficient calculation

        mask_sum_avg += exp(dist); // SoftAvg (sum)
        mask_sum_max += exp(offset_bottom_input[offset]); // SoftMax (sum)
      }
      // Over/Under-flow checks
      mask_sum_avg = clamp(mask_sum_avg, lower, upper);
      mask_sum_max = clamp(mask_sum_max, lower, upper);


      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id; // offset adjustment

        if(d_offset >= dim || d_offset < 0)continue;
        const int offset = d_offset;

        scalar_t dist = dsc(offset_bottom_input[offset], act_avg); // kernel region cell DSC

        scalar_t mask_ = b * exp(dist)/mask_sum_avg; // soft Inverse Coefficient Weighting
        mask_ = mask_ +  (1. - b) * exp(offset_bottom_input[offset])/ mask_sum_max; // SoftMax
        mask_ = clamp(mask_, zero, upper); // Over/Under-flow

        if (return_mask) {
          int offset_m = (pd * kernel_d) + id; // calculate mask offset
          //printf("offset mask: %d \n", offset_m);
          mask[offset_m]= mask_;
        }

        output_data[index] += offset_bottom_input[offset] * mask_;
        output_data[index] = clamp(output_data[index], zero, upper);
      }
    }
}
/*
---  E N D  O F  T E M P L A T E  A D A P O O L 1 D F O R W A R D ---
*/


/*
---  S T A R T  O F  T E M P L A T E  A D A P O O L 2 D F O R W A R D ---
    [About]
        CUDA template utilising `CUDA_1D_KERNEL_LOOP` macro for iterating over the tensor.
        ! Note: This is the native implementation of adaPool2d and because of the large requirement
        in resources it may be unstable. Further refinement may be required!
    [Params]
        - No default for template parameter
    [Args]
        - nthreads: (constant) Integer, for the number of threads. From NVidia's website: "CUDA architecture limits the numbers of threads per block (1024 threads per block limit)".
        - bottom_input: (constant) Scalar_t, tensor for the input data.
        - bottom_beta: (constant) Scalar_t, tensor with beta parameter to be used during pooling.
        - batches: (constant) Integer, for the number of batches.
        - channels: (constant) Integer, for the number of channels.
        - height: (constant) Integer, specifies the size of the (iterable) height/y dimension of the tensor to pool over.
        - width: (constant) Integer, specifies the size of the (iterable) width/x dimension of the tensor to pool over.
        - kernel_h: (constant) Integer, for the height of the kernel.
        - kernel_w: (constant) Integer, for the width of the kernel.
        - stride_h: (constant) Integer, for the steps taken over the height/y dimension between kernels.
        - stride_w: (constant) Integer, for the steps taken over the width/x dimension between kernels.
        - output_data: constant) Scalar_t, tensor to assign the calculated output data.
        - return_mask: (constant) Boolean, if the calculated mask should be returned.
        - mask: (constant) Scalar_t, tensor of size # kernel ops (output size) x `kernel_h` x `kernel_w`.

*/
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
    // Run in parallel for each cell within each kernel region
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      int pw = index % pooled_width; // index over width of each kernel operation in relation to the position in the input
      int ph = (index / pooled_width) % pooled_height; // index  over height of each kernel operation in relation to the position in the input
      int c = (index / pooled_width / pooled_height) % channels;
      int n = index / pooled_width / pooled_height / channels;

      const int offset = (n * channels + c) * height * width; // initial offset
      const scalar_t *offset_bottom_input = bottom_input + offset;

      const int base_y = ph * stride_h; // start cell index over height/y for each kernel
      if (base_y > height - kernel_h)break; // limit height/y iterations for the index of the final kernel location in the input

      const int base_x = pw * stride_w; // start cell index over width/x for each kernel
      if (base_x > width - kernel_w)break; // limit width/x iterations for the index of the final kernel location in the input

      // --- Initialisations happen here ----
      scalar_t act_sum = 0.;
      scalar_t mask_sum_avg = 0.;
      scalar_t mask_sum_max = 0.;

      output_data[index] = 0.;
      const scalar_t upper = n_limits<scalar_t>::max();
      const scalar_t lower = n_limits<scalar_t>::min();
      const scalar_t zero = 0.;
      const scalar_t one = 1.;

      const scalar_t b = clamp(bottom_beta[index], zero, one);

      int count = 0.; // used for calculating the average

      // Iterate over inputs cells within each kernel region in the input
      for(int iy=0; iy<kernel_h; iy++){
        const int y_offset = base_y + iy;

        if(y_offset >= height || y_offset < 0)continue; // check if the offset index over y is valid (not larger than or equal to the size of the dimension) OR smaller than 0 (for fool proofing)

        for(int ix=0; ix<kernel_w; ix++){
          const int x_offset = base_x + ix;

          if(x_offset >= width || x_offset < 0)continue; // check if the offset index over x is valid (not larger than or equal to the size of the dimension) OR smaller than 0 (for fool proofing)

          const int offset = y_offset*width + x_offset;

          // Use this for verbose when debugging
          // printf("(ph: %d, pw: %d), base_y: %d, base_x: %d, iy: %d, ix: %d offset: %d \n", ph, pw, base_y, base_x, iy, ix, offset)

          act_sum += offset_bottom_input[offset];
          count += 1;
        }
      }
      scalar_t act_avg = act_sum/count; // average calculation

      for(int iy=0; iy<kernel_h; iy++){
        const int y_offset = base_y + iy;

        if(y_offset >= height || y_offset < 0)continue;

        for(int ix=0; ix<kernel_w; ix++){
          const int x_offset = base_x + ix;

          if(x_offset >= width || x_offset < 0)continue;
          const int offset = y_offset*width + x_offset;

          scalar_t dist = dsc(offset_bottom_input[offset], act_avg); // Dice Sørensen Coefficient calculation

          mask_sum_avg += exp(dist); // SoftAvg (sum)
          mask_sum_max += exp(offset_bottom_input[offset]); // SoftMax (sum)

        }
      }
      // Overflow check
      mask_sum_avg = clamp(mask_sum_avg, lower, upper);
      mask_sum_max = clamp(mask_sum_max, lower, upper);


      for(int iy=0; iy<kernel_h; iy++){
        const int y_offset = base_y + iy; // offset adjustment (y-based)

        if(y_offset >= height || y_offset < 0)continue;

        for(int ix=0; ix<kernel_w; ix++){
          const int x_offset = base_x + ix;// offset adjustment (x-based)

          if(x_offset >= width || x_offset < 0)continue;
          const int offset = y_offset*width + x_offset;

          scalar_t dist = dsc(offset_bottom_input[offset], act_avg); // kernel region cell DSC

          scalar_t mask_ = b * exp(dist)/  mask_sum_avg; // soft Inverse Coefficient Weighting
          mask_ = mask_ + (1. - b) * exp(offset_bottom_input[offset])/  mask_sum_max; // SoftMax
          mask_ = clamp(mask_, zero, upper); // Over/Under-flow

          if (return_mask) {
            // mask size = num_kernel_ops x kernel_size
            int oW = (width - kernel_w) / stride_w + 1 ;// calculate number of operations
            int w_mask = oW * kernel_w; // mask width
            int offset_m = ((ph * kernel_h) + iy) * w_mask; // offset over H dimension
            offset_m = offset_m + (pw * kernel_h) + ix; // offset over both H and W dimension
            // Use this for verbose when debugging
            // printf("offset mask: %d \n", offset_m);
            mask[offset_m]= mask_;
          }

          output_data[index] += offset_bottom_input[offset] * mask_;
          output_data[index] = clamp(output_data[index], zero, upper);
        }
      }
    }
}
/*
---  E N D  O F  T E M P L A T E  A D A P O O L 2 D F O R W A R D ---
*/


/*
---  S T A R T  O F  T E M P L A T E  A D A P O O L 3 D F O R W A R D ---
    [About]
        CUDA template utilising `CUDA_1D_KERNEL_LOOP` macro for iterating over the tensor.
        ! Note: This is the native implementation of adaPool3d and because of the large requirement
        in resources it may be unstable. Further refinement may be required!
    [Params]
        - No default for template parameter
    [Args]
        - nthreads: (constant) Integer, for the number of threads. From NVidia's website: "CUDA architecture limits the numbers of threads per block (1024 threads per block limit)".
        - bottom_input: (constant) Scalar_t, tensor for the input data.
        - bottom_beta: (constant) Scalar_t, tensor with beta parameter to be used during pooling.
        - batches: (constant) Integer, for the number of batches.
        - channels: (constant) Integer, for the number of channels.
        - depth: (constant) Integer, specifies the size of the (iterable) depth/d dimension of the tensor to pool over.
        - height: (constant) Integer, specifies the size of the (iterable) height/y dimension of the tensor to pool over.
        - width: (constant) Integer, specifies the size of the (iterable) width/x dimension of the tensor to pool over.
        - kernel_d: (constant) Integer, for the depth of the kernel.
        - kernel_h: (constant) Integer, for the height of the kernel.
        - kernel_w: (constant) Integer, for the width of the kernel.
        - stride_d: (constant) Integer, for the steps taken over the depth/d dimension between kernels.
        - stride_h: (constant) Integer, for the steps taken over the height/y dimension between kernels.
        - stride_w: (constant) Integer, for the steps taken over the width/x dimension between kernels.
        - output_data: constant) Scalar_t, tensor to assign the calculated output data.
        - return_mask: (constant) Boolean, if the calculated mask should be returned.
        - mask: (constant) Scalar_t, tensor of size # kernel ops (output size) x `kernel_d` x `kernel_h` x `kernel_w`.

*/
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
    // Run in parallel for each cell within each kernel region
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      int pw = index % pooled_width; // index over width of each kernel operation in relation to the position in the input
      int ph = (index / pooled_width) % pooled_height; // index over height of each kernel operation in relation to the position in the input
      int pd = (index / pooled_width / pooled_height) % pooled_depth; // index over depth of each kernel operation in relation to the position in the input
      int c = (index / pooled_width / pooled_height / pooled_depth) % channels;
      int n = index / pooled_width / pooled_height / pooled_depth / channels;

      const int offset = (n * channels + c) * depth * height * width; // initial offset
      const scalar_t *offset_bottom_input = bottom_input + offset;

      const int base_d = pd*stride_d; // start cell index over depth/d for each kernel
      if (base_d > depth - kernel_d)break; // limit depth/d iterations for the index of the final kernel location in the input

      const int base_y = ph*stride_h; // start cell index over height/y for each kernel
      if (base_y > height - kernel_h)break; // limit height/y iterations for the index of the final kernel location in the input

      const int base_x = pw*stride_w; // start cell index over width/x for each kernel
      if (base_x > width - kernel_w)break; // limit width/x iterations for the index of the final kernel location in the input

      // --- Initialisations happen here ----
      scalar_t act_sum = 0.;
      scalar_t mask_sum_avg = 0.;
      scalar_t mask_sum_max = 0.;

      output_data[index] = 0.;
      const scalar_t upper = n_limits<scalar_t>::max();
      const scalar_t lower = n_limits<scalar_t>::min();
      const scalar_t zero = 0.;
      const scalar_t one = 1.;

      const scalar_t b = clamp(bottom_beta[index], zero, one);

      int count = 0.; // used for calculating the average

      // Iterate over inputs cells within each kernel region in the input
      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;

        if(d_offset >= depth || d_offset < 0)continue; // check if the offset index over d is valid (not larger than or equal to the size of the dimension) OR smaller than 0 (for fool proofing)

        for(int iy=0; iy<kernel_h; iy++){
          const int y_offset = base_y + iy;

          if(y_offset >= height || y_offset < 0)continue; // check if the offset index over y is valid (not larger than or equal to the size of the dimension) OR smaller than 0 (for fool proofing)

          for(int ix=0; ix<kernel_w; ix++){
            const int x_offset = base_x + ix;

            if(x_offset >= width || x_offset < 0)continue; // check if the offset index over x is valid (not larger than or equal to the size of the dimension) OR smaller than 0 (for fool proofing)

            const int offset = d_offset*height + y_offset*width + x_offset;

            // Use this for verbose when debugging
            // printf("(pd: %d, ph: %d, pw: %d), base_d: %d, base_y: %d, base_x: %d, id: %d, iy: %d, ix: %d, offset: %d \n", pd, ph, pw, base_d, base_y, base_x, id, iy, ix, offset);

            act_sum += offset_bottom_input[offset];
            count += 1;
          }
        }
      }
      scalar_t act_avg = act_sum/count; // average calculation


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

            scalar_t dist = dsc(offset_bottom_input[offset], act_avg); // Dice Sørensen Coefficient calculation

            mask_sum_avg += exp(dist); // SoftAvg (sum)
            mask_sum_max += exp(offset_bottom_input[offset]); // SoftMax (sum)

          }
        }
      }
      // Overflow check
      mask_sum_avg = clamp(mask_sum_avg, lower, upper);
      mask_sum_max = clamp(mask_sum_max, lower, upper);

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id; // offset adjustment (d-based)

        if(d_offset >= depth || d_offset < 0)continue;
        for(int iy=0; iy<kernel_h; iy++){
          const int y_offset = base_y + iy; // offset adjustment (y-based)

          if(y_offset >= height || y_offset < 0)continue;
          for(int ix=0; ix<kernel_w; ix++){
            const int x_offset = base_x + ix; // offset adjustment (x-based)

            if(x_offset >= width || x_offset < 0)continue;
            const int offset = d_offset*height + y_offset*width + x_offset;

            scalar_t dist = dsc(offset_bottom_input[offset], act_avg); // kernel region cell DSC


            scalar_t mask_ = b * exp(dist)/mask_sum_avg; // soft Inverse Coefficient Weighting
            mask_ = mask_ + (1. - b) * exp(offset_bottom_input[offset])/mask_sum_max; // SoftMax
            mask_ = clamp(mask_, zero, upper); // Over/Under-flow

            if (return_mask) {
              // mask size = num_kernel_ops x kernel_size
              int oH = (height - kernel_h) / stride_h + 1 ;// calculate number of H-dim operations
              int h_mask = oH * kernel_h; // mask height

              int oW = (width - kernel_w) / stride_w + 1 ;// calculate number of W-dim operations
              int w_mask = oW * kernel_w; // mask width

              int offset_m = ((pd * kernel_d) + id) * h_mask * w_mask; // offset over D dimension
              offset_m = offset_m + ((ph * kernel_h) + iy) * w_mask; // offset over H dimension
              offset_m = offset_m + (pw * kernel_w) + ix; // offset over D, H and W dimension
              // Use this for verbose when debugging
              // printf("offset mask: %d \n", offset_m);
              mask[offset_m]= mask_;
            }

            output_data[index] += offset_bottom_input[offset] * mask_;
            output_data[index] = clamp(output_data[index], zero, upper);

          }
        }
      }
    }
}
/*
---  E N D  O F  T E M P L A T E  A D A P O O L 3 D F O R W A R D ---
*/


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

/*
---  S T A R T  O F  T E M P L A T E  A D A P O O L 1 D B A C K W A R D ---
    [About]
        CUDA template utilising `CUDA_1D_KERNEL_LOOP` macro for iterating over the tensor.
        ! Note: This is used by the native implementation of adaPool1d. Because of the large requirement
        in resources it may be unstable. Further refinement may be required!
    [Params]
        - No default for template parameter
    [Args]
        - nthreads: (constant) Integer, for the number of threads. From NVidia's website: "CUDA architecture limits the numbers of threads per block (1024 threads per block limit)".
        - diff_output: (constant) Scalar_t, tensor for the output's gradients.
        - data_input: (constant) Scalar_t, tensor for the input data.
        - data_beta: (constant) Scalar_t, tensor with beta parameter to be used during pooling.
        - batches: (constant) Integer, for the number of batches.
        - channels: (constant) Integer, for the number of channels.
        - dim: (constant) Integer, specifies the size of the iterable dimension of the tensor to pool over.
        - kernel_d: (constant) Integer, for the size of the kernel.
        - stride_d: (constant) Integer, for the steps taken between kernels.
        - diff_input: (constant) Scalar_t, tensor for the gradients (to be calculated) of the input data.
        - diff_beta: (constant) Scalar_t, tensor for the gradients (to be calculated) of the beta param.

*/
template <typename scalar_t>
__global__ void AdaPool1dBackward(const int nthreads,
                                   const scalar_t *diff_output, const scalar_t *data_input,
                                   const scalar_t *data_beta, const int batches,
                                   const int channels, const int dim,
                                   const int kernel_d, const int stride_d,
                                   scalar_t *diff_input, scalar_t *diff_beta){
    int pooled_dim = dim/stride_d;
    // Run in parallel for each cell within each kernel region
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      int pd = index % pooled_dim; // index of each kernel operation in relation to the position in the input
      int c = (index / pooled_dim) % channels;
      int n = index / pooled_dim / channels;

      const int offset0 = (n * channels + c) * dim; // initial offset
      const scalar_t *offset_data_input = data_input + offset0; // offset based on the input data

      const scalar_t diff_output_index = diff_output[index]; // offset based on the output gradients
      scalar_t *offset_diff_input = diff_input + offset0; // offset based on the input gradients

      const int base_d = pd*stride_d; // start cell index for each kernel

      // --- Initialisations happen here ----
      scalar_t act_sum = 0.;
      scalar_t mask_sum_avg = 0.;
      scalar_t mask_sum_max = 0.;

      const scalar_t zero = 0.;
      const scalar_t one = 1.;
      const scalar_t upper = n_limits<scalar_t>::max();
      const scalar_t lower = n_limits<scalar_t>::min();

      const scalar_t b = clamp(data_beta[index], zero, one);

      int count = 0.; // used for calculating the average

      // Iterate over inputs cells within each kernel region in the input
      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;

        if(d_offset >= dim || d_offset < 0)continue; // check if the offset index is valid (not larger than or equal to the size of the dimension) OR smaller than 0 (for fool proofing)
        const int offset = d_offset;

        // Use this for verbose when debugging
        //printf("(pd: %d), base_d: %d, id: %d, d_offset: %d \n", pd, base_d, id, d_offset);

        act_sum += offset_data_input[offset];
        count += 1;
      }
      scalar_t act_avg = act_sum/count; // average calculation

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;

        if(d_offset >= dim || d_offset < 0)continue;
        const int offset = d_offset;

        scalar_t dist = dsc(offset_data_input[offset], act_avg); // Dice Sørensen Coefficient calculation

        mask_sum_avg += exp(dist); // SoftAvg (sum)
        mask_sum_max += exp(offset_data_input[offset]); // SoftMax (sum)

      }
      // Over/Under-flow checks
      mask_sum_avg = clamp(mask_sum_avg, lower, upper);
      mask_sum_max = clamp(mask_sum_max, lower, upper);

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id; // offset adjustment

        if(d_offset >= dim || d_offset < 0)continue;
          const int offset = d_offset;

          scalar_t dist = dsc(offset_data_input[offset], act_avg); // kernel region cell DSC

          scalar_t mask_ = b * exp(dist)/mask_sum_avg; // soft Inverse Coefficient Weighting
          mask_ = mask_ +  (1. - b) * exp(offset_data_input[offset])/ mask_sum_max; // SoftMax
          mask_ = clamp(mask_, zero, upper); // Over/Under-flow

          scalar_t weighted_grad = diff_output_index * mask_; // use mask over the output gradients

          // Underflow check
          weighted_grad = clamp(weighted_grad, zero, upper);

          atomicAdd(offset_diff_input+offset, weighted_grad);
      }
    }
}
/*
---  E N D  O F  T E M P L A T E  A D A P O O L 1 D B A C K W A R D ---
*/


/*
---  S T A R T  O F  T E M P L A T E  A D A P O O L 2 D B A C K W A R D ---
    [About]
        CUDA template utilising `CUDA_1D_KERNEL_LOOP` macro for iterating over the tensor.
        ! Note: This is used by the native implementation of adaPool2d. Because of the large requirement
        in resources it may be unstable. Further refinement may be required!
    [Params]
        - No default for template parameter
    [Args]
        - nthreads: (constant) Integer, for the number of threads. From NVidia's website: "CUDA architecture limits the numbers of threads per block (1024 threads per block limit)".
        - diff_output: (constant) Scalar_t, tensor for the output's gradients.
        - data_input: (constant) Scalar_t, tensor for the input data.
        - data_beta: (constant) Scalar_t, tensor with beta parameter to be used during pooling.
        - batches: (constant) Integer, for the number of batches.
        - channels: (constant) Integer, for the number of channels.
        - height: (constant) Integer, specifies the size of the (iterable) height/y dimension of the tensor to pool over.
        - width: (constant) Integer, specifies the size of the (iterable) width/x dimension of the tensor to pool over.
        - kernel_h: (constant) Integer, for the height of the kernel.
        - kernel_w: (constant) Integer, for the width of the kernel.
        - stride_h: (constant) Integer, for the steps taken over the height/y dimension between kernels.
        - stride_w: (constant) Integer, for the steps taken over the width/x dimension between kernels.
        - diff_input: (constant) Scalar_t, tensor for the gradients (to be calculated) of the input data.
        - diff_beta: (constant) Scalar_t, tensor for the gradients (to be calculated) of the beta param.

*/
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
    // Run in parallel for each cell within each kernel region
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      int pw = index % pooled_width; // index over width of each kernel operation in relation to the position in the input
      int ph = (index / pooled_width) % pooled_height; // index  over height of each kernel operation in relation to the position in the input
      int c = (index / pooled_width / pooled_height) % channels;
      int n = index / pooled_width / pooled_height / channels;

      const int offset0 = (n * channels + c) * height * width; // initial offset
      const scalar_t *offset_data_input = data_input + offset0; // offset based on the input data

      const scalar_t diff_output_index = diff_output[index]; // offset based on the output gradients
      scalar_t *offset_diff_input = diff_input + offset0; // offset based on the input gradients

      const int base_y = ph * stride_h; // start cell index over height/y for each kernel
      if (base_y > height - kernel_h)break; // limit height/y iterations for the index of the final kernel location in the input

      const int base_x = pw * stride_w; // start cell index over width/x for each kernel
      if (base_x > width - kernel_w)break; // limit width/x iterations for the index of the final kernel location in the input

      // --- Initialisations happen here ----
      scalar_t act_sum = 0.;
      scalar_t mask_sum_avg = 0.;
      scalar_t mask_sum_max = 0.;

      scalar_t zero = 0.;
      const scalar_t one = 1.;
      const scalar_t upper = n_limits<scalar_t>::max();
      const scalar_t lower = n_limits<scalar_t>::min();

      const scalar_t b = clamp(data_beta[index], zero, one);

      int count = 0.; // used for calculating the average

      // Iterate over inputs cells within each kernel region in the input
      for(int iy=0; iy<kernel_h; iy++){
        const int y_offset = base_y + iy;

        if(y_offset >= height || y_offset < 0)continue; // check if the offset index over y is valid (not larger than or equal to the size of the dimension) OR smaller than 0 (for fool proofing)

        for(int ix=0; ix<kernel_w; ix++){
          const int x_offset = base_x + ix;

          if(x_offset >= width || x_offset < 0)continue; // check if the offset index over x is valid (not larger than or equal to the size of the dimension) OR smaller than 0 (for fool proofing)

          const int offset = y_offset*width + x_offset;

          // Use this for verbose when debugging
          // printf("(ph: %d, pw: %d), base_y: %d, base_x: %d, iy: %d, ix: %d offset: %d \n", ph, pw, base_y, base_x, iy, ix, offset)

          act_sum += offset_data_input[offset];
          count += 1;
        }
      }
      scalar_t act_avg = act_sum/count; // average calculation

      for(int iy=0; iy<kernel_h; iy++){
        const int y_offset = base_y + iy;

        if(y_offset >= height || y_offset < 0)continue;

        for(int ix=0; ix<kernel_w; ix++){
          const int x_offset = base_x + ix;

          if(x_offset >= width || x_offset < 0)continue;
          const int offset = y_offset*width + x_offset;

          scalar_t dist = dsc(offset_data_input[offset], act_avg); // Dice Sørensen Coefficient calculation

          mask_sum_avg += exp(dist); // SoftAvg (sum)
          mask_sum_max += exp(offset_data_input[offset]); // SoftMax (sum)

        }
      }
      // Overflow check
      mask_sum_avg = clamp(mask_sum_avg, lower, upper);
      mask_sum_max = clamp(mask_sum_max, lower, upper);

      for(int iy=0; iy<kernel_h; iy++){
        const int y_offset = base_y + iy; // offset adjustment (y-based)

        if(y_offset >= height || y_offset < 0)continue;

        for(int ix=0; ix<kernel_w; ix++){
          const int x_offset = base_x + ix; // offset adjustment (x-based)

          if(x_offset >= width || x_offset < 0)continue;
            const int offset = y_offset*width + x_offset;

            scalar_t dist = dsc(offset_data_input[offset], act_avg); // kernel region cell DSC

            scalar_t mask_ = b * exp(dist)/  mask_sum_avg; // soft Inverse Coefficient Weighting
            mask_ = mask_ + (1. - b) * exp(offset_data_input[offset])/  mask_sum_max; // SoftMax
            mask_ = clamp(mask_, zero, upper); // Over/Under-flow

            scalar_t weighted_grad = diff_output_index * mask_; // use mask over the output gradients

            // Underflow check
            weighted_grad = clamp(weighted_grad, zero, upper);

            atomicAdd(offset_diff_input+offset, weighted_grad);
        }
      }
    }
}
/*
---  E N D  O F  T E M P L A T E  A D A P O O L 2 D B A C K W A R D ---
*/


/*
---  S T A R T  O F  T E M P L A T E  A D A P O O L 3 D B A C K W A R D ---
    [About]
        CUDA template utilising `CUDA_1D_KERNEL_LOOP` macro for iterating over the tensor.
        ! Note: This is used by the native implementation of adaPool3d. Because of the large requirement
        in resources it may be unstable. Further refinement may be required!
    [Params]
        - No default for template parameter
    [Args]
        - nthreads: (constant) Integer, for the number of threads. From NVidia's website: "CUDA architecture limits the numbers of threads per block (1024 threads per block limit)".
        - diff_output: (constant) Scalar_t, tensor for the output's gradients.
        - data_input: (constant) Scalar_t, tensor for the input data.
        - data_beta: (constant) Scalar_t, tensor with beta parameter to be used during pooling.
        - batches: (constant) Integer, for the number of batches.
        - channels: (constant) Integer, for the number of channels.
        - depth: (constant) Integer, specifies the size of the (iterable) depth/d dimension of the tensor to pool over.
        - height: (constant) Integer, specifies the size of the (iterable) height/y dimension of the tensor to pool over.
        - width: (constant) Integer, specifies the size of the (iterable) width/x dimension of the tensor to pool over.
        - kernel_d: (constant) Integer, for the depth of the kernel.
        - kernel_h: (constant) Integer, for the height of the kernel.
        - kernel_w: (constant) Integer, for the width of the kernel.
        - stride_d: (constant) Integer, for the steps taken over the depth/d dimension between kernels.
        - stride_h: (constant) Integer, for the steps taken over the height/y dimension between kernels.
        - stride_w: (constant) Integer, for the steps taken over the width/x dimension between kernels.
        - diff_input: (constant) Scalar_t, tensor for the gradients (to be calculated) of the input data.
        - diff_beta: (constant) Scalar_t, tensor for the gradients (to be calculated) of the beta param.

*/
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
    // Run in parallel for each cell within each kernel region
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      int pw = index % pooled_width; // index over width of each kernel operation in relation to the position in the input
      int ph = (index / pooled_width) % pooled_height; // index over height of each kernel operation in relation to the position in the input
      int pd = (index / pooled_width / pooled_height) % pooled_depth; // index over depth of each kernel operation in relation to the position in the input
      int c = (index / pooled_width / pooled_height / pooled_depth) % channels;
      int n = index / pooled_width / pooled_height / pooled_depth / channels;

      const int offset0 = (n * channels + c) * depth * height * width; // initial offset
      const scalar_t *offset_data_input = data_input + offset0; // offset based on the input data

      const scalar_t diff_output_index = diff_output[index]; // offset based on the output gradients
      scalar_t *offset_diff_input = diff_input + offset0; // offset based on the input gradients

      const int base_d = pd*stride_d; // start cell index over depth/d for each kernel
      if (base_d > depth - kernel_d)break; // limit depth/d iterations for the index of the final kernel location in the input

      const int base_y = ph*stride_h; // start cell index over height/y for each kernel
      if (base_y > height - kernel_h)break; // limit height/y iterations for the index of the final kernel location in the input

      const int base_x = pw*stride_w; // start cell index over width/x for each kernel
      if (base_x > width - kernel_w)break; // limit width/x iterations for the index of the final kernel location in the input

      // --- Initialisations happen here ----
      scalar_t act_sum = 0.;
      scalar_t mask_sum_avg = 0.;
      scalar_t mask_sum_max = 0.;

      scalar_t zero = 0.;
      const scalar_t one = 1.;
      const scalar_t upper = n_limits<scalar_t>::max();
      const scalar_t lower = n_limits<scalar_t>::min();

      const scalar_t b = clamp(data_beta[index], zero, one);

      int count = 0.; // used for calculating the average

      // Iterate over inputs cells within each kernel region in the input
      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;

        if(d_offset >= depth || d_offset < 0)continue; // check if the offset index over d is valid (not larger than or equal to the size of the dimension) OR smaller than 0 (for fool proofing)

        for(int iy=0; iy<kernel_h; iy++){
          const int y_offset = base_y + iy;

          if(y_offset >= height || y_offset < 0)continue; // check if the offset index over y is valid (not larger than or equal to the size of the dimension) OR smaller than 0 (for fool proofing)

          for(int ix=0; ix<kernel_w; ix++){
            const int x_offset = base_x + ix;

            if(x_offset >= width || x_offset < 0)continue; // check if the offset index over x is valid (not larger than or equal to the size of the dimension) OR smaller than 0 (for fool proofing)

            const int offset = d_offset*height + y_offset*width + x_offset;

            // Use this for verbose when debugging
            // printf("(pd: %d, ph: %d, pw: %d), base_d: %d, base_y: %d, base_x: %d, id: %d, iy: %d, ix: %d, offset: %d \n", pd, ph, pw, base_d, base_y, base_x, id, iy, ix, offset);

            act_sum += offset_data_input[offset];
            count += 1;
          }
        }
      }
      scalar_t act_avg = act_sum/count; // average calculation

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

              scalar_t dist = dsc(offset_data_input[offset], act_avg); // Dice Sørensen Coefficient calculation

              mask_sum_avg += exp(dist); // SoftAvg (sum)
              mask_sum_max += exp(offset_data_input[offset]); // SoftMax (sum)

          }
        }
      }
      // Overflow check
      mask_sum_avg = clamp(mask_sum_avg, lower, upper);
      mask_sum_max = clamp(mask_sum_max, lower, upper);

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id; // offset adjustment (d-based)

        if(d_offset >= depth || d_offset < 0)continue;
        for(int iy=0; iy<kernel_h; iy++){
          const int y_offset = base_y + iy; // offset adjustment (y-based)

          if(y_offset >= height || y_offset < 0)continue;
          for(int ix=0; ix<kernel_w; ix++){
            const int x_offset = base_x + ix; // offset adjustment (x-based)

            if(x_offset >= width || x_offset < 0)continue;
              const int offset = d_offset*height + y_offset*width + x_offset;

              scalar_t dist = dsc(offset_data_input[offset], act_avg); // kernel region cell DSC

              scalar_t mask_ = b * exp(dist)/  mask_sum_avg; // soft Inverse Coefficient Weighting
              mask_ = mask_ + (1. - b) * exp(offset_data_input[offset])/  mask_sum_max; // SoftMax
              mask_ = clamp(mask_, zero, upper); // Over/Under-flow

              scalar_t weighted_grad = diff_output_index * mask_; // use mask over the output gradients

              // Underflow check
              weighted_grad = clamp(weighted_grad, zero, upper);

              atomicAdd(offset_diff_input+offset, weighted_grad);
          }
        }
      }
    }
}
/*
---  E N D  O F  T E M P L A T E  A D A P O O L 3 D B A C K W A R D ---
*/

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




/*
---  S T A R T  O F  T E M P L A T E  A D A _ E D S C W _ P O O L 1 D F O R W A R D ---
    [About]
        CUDA template utilising `CUDA_1D_KERNEL_LOOP` macro for iterating over the tensor.
    [Params]
        - No default for template parameter
    [Args]
        - nthreads: (constant) Integer, for the number of threads. From NVidia's website: "CUDA architecture limits the numbers of threads per block (1024 threads per block limit)".
        - bottom_input: (constant) Scalar_t, tensor for the input data.
        - batches: (constant) Integer, for the number of batches.
        - channels: (constant) Integer, for the number of channels.
        - dim: (constant) Integer, specifies the size of the iterable dimension of the tensor to pool over.
        - kernel_d: (constant) Integer, for the size of the kernel.
        - stride_d: (constant) Integer, for the steps taken between kernels.
        - output_data: constant) Scalar_t, tensor to assign the calculated output data.
        - return_mask: (constant) Boolean, if the calculated mask should be returned.
        - mask: (constant) Scalar_t, tensor of size # kernel ops (output size) x `kernel_d`.

*/
template <typename scalar_t>
__global__ void Ada_EDSCW_Pool1dForward(const int nthreads,
                                       const scalar_t *bottom_input, const int batches,
                                       const int channels, const int dim,
                                       const int kernel_d, const int stride_d,
                                       scalar_t *output_data, const bool return_mask,
                                       scalar_t *mask){
    int pooled_dim = dim/stride_d;
    // Run in parallel for each cell within each kernel region
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      int pd = index % pooled_dim; // index of each kernel operation in relation to the position in the input
      int c = (index / pooled_dim) % channels;
      int n = index / pooled_dim / channels;

      const int offset = (n * channels + c) * dim; // initial offset
      const scalar_t *offset_bottom_input = bottom_input + offset;

      const int base_d = pd*stride_d; // start cell index for each kernel
      if (base_d > dim - kernel_d)break; // limit iterations based on the position of the final kernel application over the input

      // --- Initialisations happen here ----
      output_data[index] = 0.;
      const scalar_t upper = n_limits<scalar_t>::max();
      const scalar_t lower = n_limits<scalar_t>::min();
      const scalar_t zero = 0.;

      scalar_t act_sum = 0.;
      scalar_t mask_sum_avg = 0.;

      int count = 0.; // used for calculating the average

      // Iterate over inputs cells within each kernel region in the input
      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;

        if(d_offset >= dim || d_offset < 0)continue; // check if the offset index is valid (not larger than or equal to the size of the dimension) OR smaller than 0 (for fool proofing)
        const int offset = d_offset;

        // Use this for verbose when debugging
        //printf("(pd: %d), base_d: %d, id: %d, d_offset: %d \n", pd, base_d, id, d_offset);

        act_sum += offset_bottom_input[offset];
        count += 1;
      }
      scalar_t act_avg = act_sum/count; // average

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id; // offset adjustment

        if(d_offset >= dim || d_offset < 0)continue;
        const int offset = d_offset;

        scalar_t dist = dsc(offset_bottom_input[offset], act_avg); // Dice Sørensen Coefficient calculation

        mask_sum_avg += exp(dist);

      }
      // Overflow check
      mask_sum_avg = clamp(mask_sum_avg, lower, upper);


      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;// offset adjustment

        if(d_offset >= dim || d_offset < 0)continue;
        const int offset = d_offset;

        scalar_t dist = dsc(offset_bottom_input[offset], act_avg);// kernel region cell DSC

        scalar_t mask_ = exp(dist)/mask_sum_avg;// soft Inverse Coefficient Weighting

        if (return_mask) {
          const int offset_m = (pd * kernel_d) + id; // calculate mask offset
          // Use this for verbose when debugging
          //printf("offset mask: %d \n", offset_m);
          mask[offset_m]= mask_;
        }

        output_data[index] += offset_bottom_input[offset] * mask_;
        output_data[index] = clamp(output_data[index], zero, upper);
      }
    }
}
/*
---  E N D  O F  T E M P L A T E  A D A _ E D S C W _ P O O L 1 D F O R W A R D ---
*/


/*
---  S T A R T  O F  T E M P L A T E  A D A _ E D S C W _ P O O L 2 D F O R W A R D ---
    [About]
        CUDA template utilising `CUDA_1D_KERNEL_LOOP` macro for iterating over the tensor.
    [Params]
        - No default for template parameter
    [Args]
        - nthreads: (constant) Integer, for the number of threads. From NVidia's website: "CUDA architecture limits the numbers of threads per block (1024 threads per block limit)".
        - bottom_input: (constant) Scalar_t, tensor for the input data.
        - batches: (constant) Integer, for the number of batches.
        - channels: (constant) Integer, for the number of channels.
        - height: (constant) Integer, specifies the size of the (iterable) height/y dimension of the tensor to pool over.
        - width: (constant) Integer, specifies the size of the (iterable) width/x dimension of the tensor to pool over.
        - kernel_h: (constant) Integer, for the height of the kernel.
        - kernel_w: (constant) Integer, for the width of the kernel.
        - stride_h: (constant) Integer, for the steps taken over the height/y dimension between kernels.
        - stride_w: (constant) Integer, for the steps taken over the width/x dimension between kernels.
        - output_data: constant) Scalar_t, tensor to assign the calculated output data.
        - return_mask: (constant) Boolean, if the calculated mask should be returned.
        - mask: (constant) Scalar_t, tensor of size # kernel ops (output size) x `kernel_h` x `kernel_w`.

*/
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
    // Run in parallel for each cell within each kernel region
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      int pw = index % pooled_width; // index over width of each kernel operation in relation to the position in the input
      int ph = (index / pooled_width) % pooled_height; // index  over height of each kernel operation in relation to the position in the input
      int c = (index / pooled_width / pooled_height) % channels;
      int n = index / pooled_width / pooled_height / channels;

      const int offset = (n * channels + c) * height * width; // initial offset
      const scalar_t *offset_bottom_input = bottom_input + offset;

      const int base_y = ph*stride_h; // start cell index over height/y for each kernel
      if (base_y > height - kernel_h)break; // limit height/y iterations for the index of the final kernel location in the input

      const int base_x = pw*stride_w; // start cell index over width/x for each kernel
      if (base_x > width - kernel_w)break; // limit width/x iterations for the index of the final kernel location in the input

      // --- Initialisations happen here ----
      scalar_t act_sum = 0.;
      scalar_t mask_sum_avg = 0.;

      output_data[index] = 0.;
      const scalar_t upper = n_limits<scalar_t>::max();
      const scalar_t lower = n_limits<scalar_t>::min();
      const scalar_t zero = 0.;

      int count = 0.; // used for calculating the average

      // Iterate over inputs cells within each kernel region in the input
      for(int iy=0; iy<kernel_h; iy++){
        const int y_offset = base_y + iy;

        if(y_offset >= height || y_offset < 0)continue; // check if the offset index over y is valid (not larger than or equal to the size of the dimension) OR smaller than 0 (for fool proofing)

        for(int ix=0; ix<kernel_w; ix++){
          const int x_offset = base_x + ix;

          if(x_offset >= width || x_offset < 0)continue; // check if the offset index over x is valid (not larger than or equal to the size of the dimension) OR smaller than 0 (for fool proofing)

          const int offset = y_offset*width + x_offset;

          // Use this for verbose when debugging
          // printf("(ph: %d, pw: %d), base_y: %d, base_x: %d, iy: %d, ix: %d offset: %d \n", ph, pw, base_y, base_x, iy, ix, offset)

          act_sum += offset_bottom_input[offset];
          count += 1;
        }
      }
      scalar_t act_avg = act_sum/count; // average calculation

      for(int iy=0; iy<kernel_h; iy++){
        const int y_offset = base_y + iy;

        if(y_offset >= height || y_offset < 0)continue;
        for(int ix=0; ix<kernel_w; ix++){
          const int x_offset = base_x + ix;

          if(x_offset >= width || x_offset < 0)continue;
          const int offset = y_offset*width + x_offset;

          scalar_t dist = dsc(offset_bottom_input[offset], act_avg); // Dice Sørensen Coefficient calculation

          mask_sum_avg += exp(dist);

        }
      }
      // Overflow check
      mask_sum_avg = clamp(mask_sum_avg, lower, upper);


      for(int iy=0; iy<kernel_h; iy++){
        const int y_offset = base_y + iy; // offset adjustment (y-based)

        if(y_offset >= height || y_offset < 0)continue;
        for(int ix=0; ix<kernel_w; ix++){
          const int x_offset = base_x + ix; // offset adjustment (x-based)

          if(x_offset >= width || x_offset < 0)continue;
          const int offset = y_offset*width + x_offset;

          scalar_t dist = dsc(offset_bottom_input[offset], act_avg); // kernel region cell DSC

          scalar_t mask_ = exp(dist)/mask_sum_avg; // soft Inverse Coefficient Weighting

          if (return_mask) {
            // mask size = num_kernel_ops x kernel_size
            int oW = (width - kernel_w) / stride_w + 1 ;// calculate number of operations
            int w_mask = oW * kernel_w; // mask width
            int offset_m = ((ph * kernel_h) + iy) * w_mask; // offset over H dimension
            offset_m = offset_m + (pw * kernel_h) + ix; // offset over both H and W dimension
            // Use this for verbose when debugging
            // printf("offset mask: %d \n", offset_m);
            mask[offset_m]= mask_;
          }

          output_data[index] += offset_bottom_input[offset] * mask_;
          output_data[index] = clamp(output_data[index], zero, upper);
        }
      }
    }
}
/*
---  E N D  O F  T E M P L A T E  A D A _ E D S C W _ P O O L 2 D F O R W A R D ---
*/


/*
---  S T A R T  O F  T E M P L A T E  A D A _ E D S C W _ P O O L 3 D F O R W A R D ---
    [About]
        CUDA template utilising `CUDA_1D_KERNEL_LOOP` macro for iterating over the tensor.
    [Params]
        - No default for template parameter
    [Args]
        - nthreads: (constant) Integer, for the number of threads. From NVidia's website: "CUDA architecture limits the numbers of threads per block (1024 threads per block limit)".
        - bottom_input: (constant) Scalar_t, tensor for the input data.
        - batches: (constant) Integer, for the number of batches.
        - channels: (constant) Integer, for the number of channels.
        - depth: (constant) Integer, specifies the size of the (iterable) depth/d dimension of the tensor to pool over.
        - height: (constant) Integer, specifies the size of the (iterable) height/y dimension of the tensor to pool over.
        - width: (constant) Integer, specifies the size of the (iterable) width/x dimension of the tensor to pool over.
        - kernel_d: (constant) Integer, for the depth of the kernel.
        - kernel_h: (constant) Integer, for the height of the kernel.
        - kernel_w: (constant) Integer, for the width of the kernel.
        - stride_d: (constant) Integer, for the steps taken over the depth/d dimension between kernels.
        - stride_h: (constant) Integer, for the steps taken over the height/y dimension between kernels.
        - stride_w: (constant) Integer, for the steps taken over the width/x dimension between kernels.
        - output_data: constant) Scalar_t, tensor to assign the calculated output data.
        - return_mask: (constant) Boolean, if the calculated mask should be returned.
        - mask: (constant) Scalar_t, tensor of size # kernel ops (output size) x `kernel_d` x `kernel_h` x `kernel_w`.

*/
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
    // Run in parallel for each cell within each kernel region
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      int pw = index % pooled_width; // index over width of each kernel operation in relation to the position in the input
      int ph = (index / pooled_width) % pooled_height; // index over height of each kernel operation in relation to the position in the input
      int pd = (index / pooled_width / pooled_height) % pooled_depth; // index over depth of each kernel operation in relation to the position in the input
      int c = (index / pooled_width / pooled_height / pooled_depth) % channels;
      int n = index / pooled_width / pooled_height / pooled_depth / channels;

      const int offset = (n * channels + c) * depth * height * width; // initial offset
      const scalar_t *offset_bottom_input = bottom_input + offset;

      const int base_d = pd*stride_d; // start cell index over depth/d for each kernel
      if (base_d > depth - kernel_d)break; // limit depth/d iterations for the index of the final kernel location in the input

      const int base_y = ph*stride_h; // start cell index over height/y for each kernel
      if (base_y > height - kernel_h)break; // limit height/y iterations for the index of the final kernel location in the input

      const int base_x = pw*stride_w; // start cell index over width/x for each kernel
      if (base_x > width - kernel_w)break; // limit width/x iterations for the index of the final kernel location in the input

      // --- Initialisations happen here ----
      scalar_t act_sum = 0.;
      scalar_t mask_sum_avg = 0.;

      output_data[index] = 0.;
      const scalar_t upper = n_limits<scalar_t>::max();
      const scalar_t lower = n_limits<scalar_t>::min();
      const scalar_t zero = 0.;

      int count = 0.; // used for calculating the average

      // Iterate over inputs cells within each kernel region in the input
      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;

        if(d_offset >= depth || d_offset < 0)continue; // check if the offset index over d is valid (not larger than or equal to the size of the dimension) OR smaller than 0 (for fool proofing)

        for(int iy=0; iy<kernel_h; iy++){
          const int y_offset = base_y + iy;

          if(y_offset >= height || y_offset < 0)continue; // check if the offset index over y is valid (not larger than or equal to the size of the dimension) OR smaller than 0 (for fool proofing)

          for(int ix=0; ix<kernel_w; ix++){
            const int x_offset = base_x + ix;

            if(x_offset >= width || x_offset < 0)continue; // check if the offset index over x is valid (not larger than or equal to the size of the dimension) OR smaller than 0 (for fool proofing)

            const int offset = d_offset*height + y_offset*width + x_offset;

            // Use this for verbose when debugging
            // printf("(pd: %d, ph: %d, pw: %d), base_d: %d, base_y: %d, base_x: %d, id: %d, iy: %d, ix: %d, offset: %d \n", pd, ph, pw, base_d, base_y, base_x, id, iy, ix, offset);

            act_sum += offset_bottom_input[offset];
            count += 1;
          }
        }
      }
      scalar_t act_avg = act_sum/count; // average calculation


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

            scalar_t dist = dsc(offset_bottom_input[offset], act_avg); // Dice Sørensen Coefficient calculation

            mask_sum_avg += exp(dist);

          }
        }
      }
      // Overflow check
      mask_sum_avg = clamp(mask_sum_avg, lower, upper);

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id; // offset adjustment (d-based)

        if(d_offset >= depth || d_offset < 0)continue;
        for(int iy=0; iy<kernel_h; iy++){
          const int y_offset = base_y + iy; // offset adjustment (y-based)

          if(y_offset >= height || y_offset < 0)continue;
          for(int ix=0; ix<kernel_w; ix++){
            const int x_offset = base_x + ix; // offset adjustment (x-based)

            if(x_offset >= width || x_offset < 0)continue;
            const int offset = d_offset*height + y_offset*width + x_offset;

            scalar_t dist = dsc(offset_bottom_input[offset], act_avg); // kernel region cell DSC

            scalar_t mask_ = exp(dist)/mask_sum_avg;

            if (return_mask) {
              // mask size = num_kernel_ops x kernel_size
              int oH = (height - kernel_h) / stride_h + 1; // calculate number of H-dim operations
              int h_mask = oH * kernel_h; // mask height

              int oW = (width - kernel_w) / stride_w + 1; // calculate number of W-dim operations
              int w_mask = oW * kernel_w; // mask width

              int offset_m = ((pd * kernel_d) + id) * h_mask * w_mask; // offset over D dimension
              offset_m = offset_m + ((ph * kernel_h) + iy) * w_mask; // offset over H dimension
              offset_m = offset_m + (pw * kernel_w) + ix; // offset over D, H and W dimension
              // Use this for verbose when debugging
              // printf("offset mask: %d \n", offset_m);
              mask[offset_m]= mask_;
            }

            output_data[index] += offset_bottom_input[offset] * mask_;
            output_data[index] = clamp(output_data[index], zero, upper);
          }
        }
      }
    }
}
/*
---  E N D  O F  T E M P L A T E  A D A _ E S D C W _ P O O L 3 D F O R W A R D ---
*/


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

/*
---  S T A R T  O F  T E M P L A T E  A D A _ E D S C W _ P O O L 1 D B A C K W A R D ---
    [About]
        CUDA template utilising `CUDA_1D_KERNEL_LOOP` macro for iterating over the tensor.
        ! Note: This is used by the native implementation of adaPool1d. Because of the large requirement
        in resources it may be unstable. Further refinement may be required!
    [Params]
        - No default for template parameter
    [Args]
        - nthreads: (constant) Integer, for the number of threads. From NVidia's website: "CUDA architecture limits the numbers of threads per block (1024 threads per block limit)".
        - diff_output: (constant) Scalar_t, tensor for the output's gradients.
        - data_beta: (constant) Scalar_t, tensor with beta parameter to be used during pooling.
        - batches: (constant) Integer, for the number of batches.
        - channels: (constant) Integer, for the number of channels.
        - dim: (constant) Integer, specifies the size of the iterable dimension of the tensor to pool over.
        - kernel_d: (constant) Integer, for the size of the kernel.
        - stride_d: (constant) Integer, for the steps taken between kernels.
        - diff_input: (constant) Scalar_t, tensor for the gradients (to be calculated) of the input data.

*/
template <typename scalar_t>
__global__ void Ada_EDSCW_Pool1dBackward(const int nthreads,
                                        const scalar_t *diff_output, const scalar_t *data_input,
                                        const int batches, const int channels,
                                        const int dim, const int kernel_d,
                                        const int stride_d, scalar_t *diff_input){
    int pooled_dim = dim/stride_d;
    // Run in parallel for each cell within each kernel region
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      int pd = index % pooled_dim; // index of each kernel operation in relation to the position in the input
      int c = (index / pooled_dim) % channels;
      int n = index / pooled_dim / channels;

      const int offset0 = (n * channels + c) * dim; // initial offset
      const scalar_t *offset_data_input = data_input + offset0; // offset based on the input data

      const scalar_t diff_output_index = diff_output[index]; // offset based on the output gradients
      scalar_t *offset_diff_input = diff_input + offset0; // offset based on the input gradients

      const int base_d = pd*stride_d; // start cell index for each kernel

      // --- Initialisations happen here ----
      scalar_t act_sum = 0.;
      scalar_t mask_sum_avg = 0.;

      const scalar_t upper = n_limits<scalar_t>::max();
      const scalar_t lower = n_limits<scalar_t>::min();

      int count = 0.; // used for calculating the average

      // Iterate over inputs cells within each kernel region in the input
      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;

        if(d_offset >= dim || d_offset < 0)continue; // check if the offset index is valid (not larger than or equal to the size of the dimension) OR smaller than 0 (for fool proofing)
        const int offset = d_offset;

        // Use this for verbose when debugging
        //printf("(pd: %d), base_d: %d, id: %d, d_offset: %d \n", pd, base_d, id, d_offset);

        act_sum += offset_data_input[offset];
        count += 1;
      }
      scalar_t act_avg = act_sum/count; // average calculation

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;

        if(d_offset >= dim || d_offset < 0)continue;
        const int offset = d_offset;

        scalar_t dist = dsc(offset_data_input[offset], act_avg); // Dice Sørensen Coefficient calculation

        mask_sum_avg += exp(dist); // SoftAvg (sum)

      }
      // Over/Under-flow checks
      mask_sum_avg = clamp(mask_sum_avg, lower, upper);

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id; // offset adjustment

        if(d_offset >= dim || d_offset < 0)continue;
          const int offset = d_offset;

          scalar_t dist = dsc(offset_data_input[offset], act_avg); // kernel region cell DSC

          scalar_t mask_ = exp(dist)/mask_sum_avg; // soft Inverse Coefficient Weighting

          scalar_t weighted_grad = diff_output_index * mask_; // use mask over the output gradients

          // Underflow check
          weighted_grad = clamp(weighted_grad, lower, upper);

          atomicAdd(offset_diff_input+offset, weighted_grad);
      }
    }
}
/*
---  E N D  O F  T E M P L A T E  A D A _ E D S C W _ P O O L 1 D B A C K W A R D ---
*/



/*
---  S T A R T  O F  T E M P L A T E  A D A _ E S D C W _ P O O L 2 D B A C K W A R D ---
    [About]
        CUDA template utilising `CUDA_1D_KERNEL_LOOP` macro for iterating over the tensor.
    [Params]
        - No default for template parameter
    [Args]
        - nthreads: (constant) Integer, for the number of threads. From NVidia's website: "CUDA architecture limits the numbers of threads per block (1024 threads per block limit)".
        - diff_output: (constant) Scalar_t, tensor for the output's gradients.
        - data_input: (constant) Scalar_t, tensor for the input data.
        - batches: (constant) Integer, for the number of batches.
        - channels: (constant) Integer, for the number of channels.
        - height: (constant) Integer, specifies the size of the (iterable) height/y dimension of the tensor to pool over.
        - width: (constant) Integer, specifies the size of the (iterable) width/x dimension of the tensor to pool over.
        - kernel_h: (constant) Integer, for the height of the kernel.
        - kernel_w: (constant) Integer, for the width of the kernel.
        - stride_h: (constant) Integer, for the steps taken over the height/y dimension between kernels.
        - stride_w: (constant) Integer, for the steps taken over the width/x dimension between kernels.
        - diff_input: (constant) Scalar_t, tensor for the gradients (to be calculated) of the input data.

*/
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
    // Run in parallel for each cell within each kernel region
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      int pw = index % pooled_width; // index over width of each kernel operation in relation to the position in the input
      int ph = (index / pooled_width) % pooled_height; // index  over height of each kernel operation in relation to the position in the input
      int c = (index / pooled_width / pooled_height) % channels;
      int n = index / pooled_width / pooled_height / channels;

      const int offset0 = (n * channels + c) * height * width; // initial offset
      const scalar_t *offset_data_input = data_input + offset0; // offset based on the input data

      const scalar_t diff_output_index = diff_output[index]; // offset based on the output gradients
      scalar_t *offset_diff_input = diff_input + offset0; // offset based on the input gradients

      const int base_y = ph * stride_h; // start cell index over height/y for each kernel
      if (base_y > height - kernel_h)break; // limit height/y iterations for the index of the final kernel location in the input

      const int base_x = pw * stride_w; // start cell index over width/x for each kernel
      if (base_x > width - kernel_w)break; // limit width/x iterations for the index of the final kernel location in the input

      // --- Initialisations happen here ----
      scalar_t act_sum = 0.;
      scalar_t mask_sum_avg = 0.;

      const scalar_t upper = n_limits<scalar_t>::max();
      const scalar_t lower = n_limits<scalar_t>::min();

      int count = 0.; // used for calculating the average

      // Iterate over inputs cells within each kernel region in the input
      for(int iy=0; iy<kernel_h; iy++){
        const int y_offset = base_y + iy;

        if(y_offset >= height || y_offset < 0)continue; // check if the offset index over y is valid (not larger than or equal to the size of the dimension) OR smaller than 0 (for fool proofing)

        for(int ix=0; ix<kernel_w; ix++){
          const int x_offset = base_x + ix;

          if(x_offset >= width || x_offset < 0)continue; // check if the offset index over x is valid (not larger than or equal to the size of the dimension) OR smaller than 0 (for fool proofing)

          const int offset = y_offset*width + x_offset;

          // Use this for verbose when debugging
          // printf("(ph: %d, pw: %d), base_y: %d, base_x: %d, iy: %d, ix: %d offset: %d \n", ph, pw, base_y, base_x, iy, ix, offset)

          act_sum += offset_data_input[offset];
          count += 1;
        }
      }
      scalar_t act_avg = act_sum/count; // average calculation

      for(int iy=0; iy<kernel_h; iy++){
        const int y_offset = base_y + iy;

        if(y_offset >= height || y_offset < 0)continue;

        for(int ix=0; ix<kernel_w; ix++){
          const int x_offset = base_x + ix;

          if(x_offset >= width || x_offset < 0)continue;
          const int offset = y_offset*width + x_offset;

          scalar_t dist = dsc(offset_data_input[offset], act_avg); // Dice Sørensen Coefficient calculation

          mask_sum_avg += exp(dist); // SoftMax (sum)

        }
      }
      // Overflow check
      mask_sum_avg = clamp(mask_sum_avg, lower, upper);

      for(int iy=0; iy<kernel_h; iy++){
        const int y_offset = base_y + iy; // offset adjustment (y-based)

        if(y_offset >= height || y_offset < 0)continue;

        for(int ix=0; ix<kernel_w; ix++){
          const int x_offset = base_x + ix; // offset adjustment (x-based)

          if(x_offset >= width || x_offset < 0)continue;
            const int offset = y_offset*width + x_offset;

            scalar_t dist = dsc(offset_data_input[offset], act_avg); // kernel region cell DSC

            scalar_t mask_ = exp(dist)/mask_sum_avg; // soft Inverse Coefficient Weighting

            scalar_t weighted_grad = diff_output_index * mask_; // use mask over the output gradients

            // Underflow check
            weighted_grad = clamp(weighted_grad, lower, upper);

            atomicAdd(offset_diff_input+offset, weighted_grad);
        }
      }
    }
}
/*
---  E N D  O F  T E M P L A T E  A D A _ E D S C W _ P O O L 2 D B A C K W A R D ---
*/


/*
---  S T A R T  O F  T E M P L A T E  A D A _ E D S C W _ P O O L 3 D B A C K W A R D ---
    [About]
        CUDA template utilising `CUDA_1D_KERNEL_LOOP` macro for iterating over the tensor.
    [Params]
        - No default for template parameter
    [Args]
        - nthreads: (constant) Integer, for the number of threads. From NVidia's website: "CUDA architecture limits the numbers of threads per block (1024 threads per block limit)".
        - diff_output: (constant) Scalar_t, tensor for the output's gradients.
        - data_input: (constant) Scalar_t, tensor for the input data.
        - batches: (constant) Integer, for the number of batches.
        - channels: (constant) Integer, for the number of channels.
        - depth: (constant) Integer, specifies the size of the (iterable) depth/d dimension of the tensor to pool over.
        - height: (constant) Integer, specifies the size of the (iterable) height/y dimension of the tensor to pool over.
        - width: (constant) Integer, specifies the size of the (iterable) width/x dimension of the tensor to pool over.
        - kernel_d: (constant) Integer, for the depth of the kernel.
        - kernel_h: (constant) Integer, for the height of the kernel.
        - kernel_w: (constant) Integer, for the width of the kernel.
        - stride_d: (constant) Integer, for the steps taken over the depth/d dimension between kernels.
        - stride_h: (constant) Integer, for the steps taken over the height/y dimension between kernels.
        - stride_w: (constant) Integer, for the steps taken over the width/x dimension between kernels.
        - diff_input: (constant) Scalar_t, tensor for the gradients (to be calculated) of the input data.

*/
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
    // Run in parallel for each cell within each kernel region
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      int pw = index % pooled_width; // index over width of each kernel operation in relation to the position in the input
      int ph = (index / pooled_width) % pooled_height; // index over height of each kernel operation in relation to the position in the input
      int pd = (index / pooled_width / pooled_height) % pooled_depth; // index over depth of each kernel operation in relation to the position in the input
      int c = (index / pooled_width / pooled_height / pooled_depth) % channels;
      int n = index / pooled_width / pooled_height / pooled_depth / channels;

      const int offset0 = (n * channels + c) * depth * height * width; // initial offset
      const scalar_t *offset_data_input = data_input + offset0; // offset based on the input data

      const scalar_t diff_output_index = diff_output[index]; // offset based on the output gradients
      scalar_t *offset_diff_input = diff_input + offset0; // offset based on the input gradients

      const int base_d = pd*stride_d; // start cell index over depth/d for each kernel
      if (base_d > depth - kernel_d)break; // limit depth/d iterations for the index of the final kernel location in the input

      const int base_y = ph*stride_h; // start cell index over height/y for each kernel
      if (base_y > height - kernel_h)break; // limit height/y iterations for the index of the final kernel location in the input

      const int base_x = pw*stride_w; // start cell index over width/x for each kernel
      if (base_x > width - kernel_w)break; // limit width/x iterations for the index of the final kernel location in the input

      // --- Initialisations happen here ----
      scalar_t act_sum = 0.;
      scalar_t mask_sum_avg = 0.;

      const scalar_t upper = n_limits<scalar_t>::max();
      const scalar_t lower = n_limits<scalar_t>::min();

      int count = 0.; // used for calculating the average

      // Iterate over inputs cells within each kernel region in the input
      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;

        if(d_offset >= depth || d_offset < 0)continue; // check if the offset index over d is valid (not larger than or equal to the size of the dimension) OR smaller than 0 (for fool proofing)

        for(int iy=0; iy<kernel_h; iy++){
          const int y_offset = base_y + iy;

          if(y_offset >= height || y_offset < 0)continue; // check if the offset index over y is valid (not larger than or equal to the size of the dimension) OR smaller than 0 (for fool proofing)

          for(int ix=0; ix<kernel_w; ix++){
            const int x_offset = base_x + ix;

            if(x_offset >= width || x_offset < 0)continue; // check if the offset index over x is valid (not larger than or equal to the size of the dimension) OR smaller than 0 (for fool proofing)

            const int offset = d_offset*height + y_offset*width + x_offset;

            // Use this for verbose when debugging
            // printf("(pd: %d, ph: %d, pw: %d), base_d: %d, base_y: %d, base_x: %d, id: %d, iy: %d, ix: %d, offset: %d \n", pd, ph, pw, base_d, base_y, base_x, id, iy, ix, offset);

            act_sum += offset_data_input[offset];
            count += 1;
          }
        }
      }
      scalar_t act_avg = act_sum/count; // average calculation

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

              scalar_t dist = dsc(offset_data_input[offset], act_avg); // Dice Sørensen Coefficient calculation

              mask_sum_avg += exp(dist); // SoftAvg (sum)

          }
        }
      }
      // Overflow check
      mask_sum_avg = clamp(mask_sum_avg, lower, upper);

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id; // offset adjustment (d-based)

        if(d_offset >= depth || d_offset < 0)continue;
        for(int iy=0; iy<kernel_h; iy++){
          const int y_offset = base_y + iy; // offset adjustment (y-based)

          if(y_offset >= height || y_offset < 0)continue;
          for(int ix=0; ix<kernel_w; ix++){
            const int x_offset = base_x + ix; // offset adjustment (x-based)

            if(x_offset >= width || x_offset < 0)continue;
              const int offset = d_offset*height + y_offset*width + x_offset;

              scalar_t dist = dsc(offset_data_input[offset], act_avg); // kernel region cell DSC

              scalar_t mask_ = exp(dist)/mask_sum_avg;

              scalar_t weighted_grad = diff_output_index * mask_; // use mask over the output gradients

              // Underflow check
              weighted_grad = clamp(weighted_grad, lower, upper);

              atomicAdd(offset_diff_input+offset, weighted_grad);
          }
        }
      }
    }
}
/*
---  E N D  O F  T E M P L A T E  A D A _ E D S C W _ P O O L 3 D B A C K W A R D ---
*/

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




/*
---  S T A R T  O F  T E M P L A T E  A D A _ E M _ P O O L 1 D F O R W A R D ---
    [About]
        CUDA template utilising `CUDA_1D_KERNEL_LOOP` macro for iterating over the tensor.
    [Params]
        - No default for template parameter
    [Args]
        - nthreads: (constant) Integer, for the number of threads. From NVidia's website: "CUDA architecture limits the numbers of threads per block (1024 threads per block limit)".
        - bottom_input: (constant) Scalar_t, tensor for the input data.
        - batches: (constant) Integer, for the number of batches.
        - channels: (constant) Integer, for the number of channels.
        - dim: (constant) Integer, specifies the size of the iterable dimension of the tensor to pool over.
        - kernel_d: (constant) Integer, for the size of the kernel.
        - stride_d: (constant) Integer, for the steps taken between kernels.
        - output_data: constant) Scalar_t, tensor to assign the calculated output data.
        - return_mask: (constant) Boolean, if the calculated mask should be returned.
        - mask: (constant) Scalar_t, tensor of size # kernel ops (output size) x `kernel_d`.

*/
template <typename scalar_t>
__global__ void Ada_EM_Pool1dForward(const int nthreads,
                                     const scalar_t *bottom_input, const int batches,
                                     const int channels, const int dim,
                                     const int kernel_d, const int stride_d,
                                     scalar_t *output_data, const bool return_mask,
                                     scalar_t *mask){
    int pooled_dim = dim/stride_d;
    // Run in parallel for each cell within each kernel region
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      int pd = index % pooled_dim;// index of each kernel operation in relation to the position in the input
      int c = (index / pooled_dim) % channels;
      int n = index / pooled_dim / channels;

      const int offset = (n * channels + c) * dim; // initial offset
      const scalar_t *offset_bottom_input = bottom_input + offset;

      const int base_d = pd*stride_d; // start cell index for each kernel
      if (base_d > dim - kernel_d)break; // limit iterations based on the position of the final kernel application over the input

      // --- Initialisations happen here ----
      scalar_t mask_sum_max = 0.;

      output_data[index] = 0.;
      const scalar_t upper = n_limits<scalar_t>::max();
      const scalar_t lower = n_limits<scalar_t>::min();
      const scalar_t zero = 0.;

      // Iterate over inputs cells within each kernel region in the input
      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;

        if(d_offset >= dim || d_offset < 0)continue;// check if the offset index is valid (not larger than or equal to the size of the dimension) OR smaller than 0 (for fool proofing)
        const int offset = d_offset;

        // Use this for verbose when debugging
        //printf("(pd: %d), base_d: %d, id: %d, d_offset: %d \n", pd, base_d, id, d_offset);

        mask_sum_max += exp(offset_bottom_input[offset]);

      }
      // Overflow check
      mask_sum_max = clamp(mask_sum_max, lower, upper);

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;

        if(d_offset >= dim || d_offset < 0)continue;
        const int offset = d_offset;

        scalar_t mask_ = exp(offset_bottom_input[offset])/ mask_sum_max;// SoftMax

        if (return_mask) {
          int offset_m = (pd * kernel_d) + id; // calculate mask offset
          //printf("offset mask: %d \n", offset_m);
          mask[offset_m]= mask_;
        }

        output_data[index] += offset_bottom_input[offset] * mask_;
        output_data[index] = clamp(output_data[index], zero, upper);
      }
    }
}
/*
---  E N D  O F  T E M P L A T E  A D A _ E M _ P O O L 1 D F O R W A R D ---
*/


/*
---  S T A R T  O F  T E M P L A T E  A D A _ E M _ P O O L 2 D F O R W A R D ---
    [About]
        CUDA template utilising `CUDA_1D_KERNEL_LOOP` macro for iterating over the tensor.
    [Params]
        - No default for template parameter
    [Args]
        - nthreads: (constant) Integer, for the number of threads. From NVidia's website: "CUDA architecture limits the numbers of threads per block (1024 threads per block limit)".
        - bottom_input: (constant) Scalar_t, tensor for the input data.
        - batches: (constant) Integer, for the number of batches.
        - channels: (constant) Integer, for the number of channels.
        - height: (constant) Integer, specifies the size of the (iterable) height/y dimension of the tensor to pool over.
        - width: (constant) Integer, specifies the size of the (iterable) width/x dimension of the tensor to pool over.
        - kernel_h: (constant) Integer, for the height of the kernel.
        - kernel_w: (constant) Integer, for the width of the kernel.
        - stride_h: (constant) Integer, for the steps taken over the height/y dimension between kernels.
        - stride_w: (constant) Integer, for the steps taken over the width/x dimension between kernels.
        - output_data: constant) Scalar_t, tensor to assign the calculated output data.
        - return_mask: (constant) Boolean, if the calculated mask should be returned.
        - mask: (constant) Scalar_t, tensor of size # kernel ops (output size) x `kernel_h` x `kernel_w`.

*/
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
    // Run in parallel for each cell within each kernel region
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      int pw = index % pooled_width; // index over width of each kernel operation in relation to the position in the input
      int ph = (index / pooled_width) % pooled_height; // index  over height of each kernel operation in relation to the position in the input
      int c = (index / pooled_width / pooled_height) % channels;
      int n = index / pooled_width / pooled_height / channels;

      const int offset = (n * channels + c) * height * width; // initial offset
      const scalar_t *offset_bottom_input = bottom_input + offset;

      const int base_y = ph*stride_h;// start cell index over height/y for each kernel
      if (base_y > height - kernel_h)break; // limit height/y iterations for the index of the final kernel location in the input

      const int base_x = pw*stride_w; // start cell index over width/x for each kernel
      if (base_x > width - kernel_w)break; // limit width/x iterations for the index of the final kernel location in the input

      // --- Initialisations happen here ----
      scalar_t mask_sum_max = 0.;

      output_data[index] = 0.;
      const scalar_t upper = n_limits<scalar_t>::max();
      const scalar_t lower = n_limits<scalar_t>::min();
      const scalar_t zero = 0.;

      // Iterate over inputs cells within each kernel region in the input
      for(int iy=0; iy<kernel_h; iy++){
        const int y_offset = base_y + iy;

        if(y_offset >= height || y_offset < 0)continue; // check if the offset index over y is valid (not larger than or equal to the size of the dimension) OR smaller than 0 (for fool proofing)

        for(int ix=0; ix<kernel_w; ix++){
          const int x_offset = base_x + ix;

          if(x_offset >= width || x_offset < 0)continue; // check if the offset index over x is valid (not larger than or equal to the size of the dimension) OR smaller than 0 (for fool proofing)

          const int offset = y_offset*width + x_offset;

          // Use this for verbose when debugging
          // printf("(ph: %d, pw: %d), base_y: %d, base_x: %d, iy: %d, ix: %d offset: %d \n", ph, pw, base_y, base_x, iy, ix, offset)

          mask_sum_max += exp(offset_bottom_input[offset]);

        }
      }
      // Overflow check
      mask_sum_max = clamp(mask_sum_max, lower, upper);


      for(int iy=0; iy<kernel_h; iy++){
        const int y_offset = base_y + iy; // offset adjustment (y-based)

        if(y_offset >= height || y_offset < 0)continue;

        for(int ix=0; ix<kernel_w; ix++){
          const int x_offset = base_x + ix; // offset adjustment (x-based)

          if(x_offset >= width || x_offset < 0)continue;
          const int offset = y_offset*width + x_offset; // x+y adjusted offset

          scalar_t mask_ = exp(offset_bottom_input[offset])/  mask_sum_max; // SoftMax

          if (return_mask) {
            // mask size = num_kernel_ops x kernel_size
            int oW = (width - kernel_w) / stride_w + 1 ;// calculate number of operations
            int w_mask = oW * kernel_w; // mask width
            int offset_m = ((ph * kernel_h) + iy) * w_mask; // offset over H dimension
            offset_m = offset_m + (pw * kernel_h) + ix; // offset over both H and W dimension
            // Use this for verbose when debugging
            // printf("offset mask: %d \n", offset_m);
            mask[offset_m]= mask_;
          }

          output_data[index] += offset_bottom_input[offset] * mask_;
          output_data[index] = clamp(output_data[index], zero, upper);
        }
      }
    }
}
/*
---  E N D  O F  T E M P L A T E  A D A _ E M _ P O O L 2 D F O R W A R D ---
*/



/*
---  S T A R T  O F  T E M P L A T E  A D A _ E M _ P O O L 3 D F O R W A R D ---
    [About]
        CUDA template utilising `CUDA_1D_KERNEL_LOOP` macro for iterating over the tensor.
    [Params]
        - No default for template parameter
    [Args]
        - nthreads: (constant) Integer, for the number of threads. From NVidia's website: "CUDA architecture limits the numbers of threads per block (1024 threads per block limit)".
        - bottom_input: (constant) Scalar_t, tensor for the input data.
        - batches: (constant) Integer, for the number of batches.
        - channels: (constant) Integer, for the number of channels.
        - depth: (constant) Integer, specifies the size of the (iterable) depth/d dimension of the tensor to pool over.
        - height: (constant) Integer, specifies the size of the (iterable) height/y dimension of the tensor to pool over.
        - width: (constant) Integer, specifies the size of the (iterable) width/x dimension of the tensor to pool over.
        - kernel_d: (constant) Integer, for the depth of the kernel.
        - kernel_h: (constant) Integer, for the height of the kernel.
        - kernel_w: (constant) Integer, for the width of the kernel.
        - stride_d: (constant) Integer, for the steps taken over the depth/d dimension between kernels.
        - stride_h: (constant) Integer, for the steps taken over the height/y dimension between kernels.
        - stride_w: (constant) Integer, for the steps taken over the width/x dimension between kernels.
        - output_data: constant) Scalar_t, tensor to assign the calculated output data.
        - return_mask: (constant) Boolean, if the calculated mask should be returned.
        - mask: (constant) Scalar_t, tensor of size # kernel ops (output size) x `kernel_d` x `kernel_h` x `kernel_w`.

*/
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
    // Run in parallel for each cell within each kernel region
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      int pw = index % pooled_width; // index over width of each kernel operation in relation to the position in the input
      int ph = (index / pooled_width) % pooled_height; // index over height of each kernel operation in relation to the position in the input
      int pd = (index / pooled_width / pooled_height) % pooled_depth; // index over depth of each kernel operation in relation to the position in the input
      int c = (index / pooled_width / pooled_height / pooled_depth) % channels;
      int n = index / pooled_width / pooled_height / pooled_depth / channels;

      const int offset = (n * channels + c) * depth * height * width; // initial offset
      const scalar_t *offset_bottom_input = bottom_input + offset;

      const int base_d = pd*stride_d; // start cell index over depth/d for each kernel
      if (base_d > depth - kernel_d)break; // limit depth/d iterations for the index of the final kernel location in the input

      const int base_y = ph*stride_h; // start cell index over height/y for each kernel
      if (base_y > height - kernel_h)break; // limit height/y iterations for the index of the final kernel location in the input

      const int base_x = pw*stride_w; // start cell index over width/x for each kernel
      if (base_x > width - kernel_w)break; // limit width/x iterations for the index of the final kernel location in the input

      // --- Initialisations happen here ----
      scalar_t mask_sum_max = 0.;

      output_data[index] = 0.;
      const scalar_t upper = n_limits<scalar_t>::max();
      const scalar_t lower = n_limits<scalar_t>::min();
      const scalar_t zero = 0.;

      // Iterate over inputs cells within each kernel region in the input
      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;

        if(d_offset >= depth || d_offset < 0)continue; // check if the offset index over d is valid (not larger than or equal to the size of the dimension) OR smaller than 0 (for fool proofing)

        for(int iy=0; iy<kernel_h; iy++){
          const int y_offset = base_y + iy;

          if(y_offset >= height || y_offset < 0)continue; // check if the offset index over y is valid (not larger than or equal to the size of the dimension) OR smaller than 0 (for fool proofing)

          for(int ix=0; ix<kernel_w; ix++){
            const int x_offset = base_x + ix;

            if(x_offset >= width || x_offset < 0)continue; // check if the offset index over x is valid (not larger than or equal to the size of the dimension) OR smaller than 0 (for fool proofing)

            const int offset = d_offset*height + y_offset*width + x_offset;

            // Use this for verbose when debugging
            // printf("(pd: %d, ph: %d, pw: %d), base_d: %d, base_y: %d, base_x: %d, id: %d, iy: %d, ix: %d, offset: %d \n", pd, ph, pw, base_d, base_y, base_x, id, iy, ix, offset);

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

            if (return_mask) {
              // mask size = num_kernel_ops x kernel_size
              int oH = (height - kernel_h) / stride_h + 1; // calculate number of H-dim operations
              int h_mask = oH * kernel_h; // mask height

              int oW = (width - kernel_w) / stride_w + 1; // calculate number of W-dim operations
              int w_mask = oW * kernel_w; // mask width

              int offset_m = ((pd * kernel_d) + id) * h_mask * w_mask; // offset over D dimension
              offset_m = offset_m + ((ph * kernel_h) + iy) * w_mask; // offset over H dimension
              offset_m = offset_m + (pw * kernel_w) + ix; // offset over D, H and W dimension
              // Use this for verbose when debugging
              // printf("offset mask: %d \n", offset_m);
              mask[offset_m]= mask_;
            }


            output_data[index] += offset_bottom_input[offset] * mask_;
            output_data[index] = clamp(output_data[index], zero, upper);

          }
        }
      }
    }
}
/*
---  E N D  O F  T E M P L A T E  A D A _ E M _ P O O L 3 D F O R W A R D ---
*/

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


/*
---  S T A R T  O F  T E M P L A T E  A D A _ E M _ P O O L 1 D B A C K W A R D ---
    [About]
        CUDA template utilising `CUDA_1D_KERNEL_LOOP` macro for iterating over the tensor.
    [Params]
        - No default for template parameter
    [Args]
        - nthreads: (constant) Integer, for the number of threads. From NVidia's website: "CUDA architecture limits the numbers of threads per block (1024 threads per block limit)".
        - diff_output: (constant) Scalar_t, tensor for the output's gradients.
        - data_input: (constant) Scalar_t, tensor for the input data.
        - batches: (constant) Integer, for the number of batches.
        - channels: (constant) Integer, for the number of channels.
        - dim: (constant) Integer, specifies the size of the iterable dimension of the tensor to pool over.
        - kernel_d: (constant) Integer, for the size of the kernel.
        - stride_d: (constant) Integer, for the steps taken between kernels.
        - diff_input: (constant) Scalar_t, tensor for the gradients (to be calculated) of the input data.

*/
template <typename scalar_t>
__global__ void Ada_EM_Pool1dBackward(const int nthreads,
                                      const scalar_t *diff_output, const scalar_t *data_input,
                                      const int batches, const int channels,
                                      const int dim, const int kernel_d,
                                      const int stride_d, scalar_t *diff_input){
    int pooled_dim = dim/stride_d;
    // Run in parallel for each cell within each kernel region
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      int pd = index % pooled_dim; // index of each kernel operation in relation to the position in the input
      int c = (index / pooled_dim) % channels;
      int n = index / pooled_dim / channels;

      const int offset0 = (n * channels + c) * dim; // initial offset
      const scalar_t *offset_data_input = data_input + offset0; // offset based on the input data

      const scalar_t diff_output_index = diff_output[index]; // offset based on the output gradients
      scalar_t *offset_diff_input = diff_input + offset0; // offset based on the input gradients

      const int base_d = pd*stride_d; // start cell index for each kernel

      // --- Initialisations happen here ----
      scalar_t mask_sum_max = 0.;
      const scalar_t upper = n_limits<scalar_t>::max();
      const scalar_t lower = n_limits<scalar_t>::min();

      // Iterate over inputs cells within each kernel region in the input
      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;

        if(d_offset >= dim || d_offset < 0)continue; // check if the offset index is valid (not larger than or equal to the size of the dimension) OR smaller than 0 (for fool proofing)
        const int offset = d_offset;

        // Use this for verbose when debugging
        //printf("(pd: %d), base_d: %d, id: %d, d_offset: %d \n", pd, base_d, id, d_offset);

        mask_sum_max += exp(offset_data_input[offset]);

      }
      // Overflow check
      mask_sum_max = clamp(mask_sum_max, lower, upper);

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;

        if(d_offset >= dim || d_offset < 0)continue;
          const int offset = d_offset;

          scalar_t mask_ = exp(offset_data_input[offset])/mask_sum_max; // SoftMax

          scalar_t weighted_grad = diff_output_index * mask_; // use mask over the output gradients

          // Underflow check
          weighted_grad = clamp(weighted_grad, lower, upper);

          atomicAdd(offset_diff_input+offset, weighted_grad);
      }
    }
}
/*
---  E N D  O F  T E M P L A T E  A D A _ E M _ P O O L 1 D B A C K W A R D ---
*/


/*
---  S T A R T  O F  T E M P L A T E  A D A _ E M _ P O O L 2 D B A C K W A R D ---
    [About]
        CUDA template utilising `CUDA_1D_KERNEL_LOOP` macro for iterating over the tensor.
    [Params]
        - No default for template parameter
    [Args]
        - nthreads: (constant) Integer, for the number of threads. From NVidia's website: "CUDA architecture limits the numbers of threads per block (1024 threads per block limit)".
        - diff_output: (constant) Scalar_t, tensor for the output's gradients.
        - data_input: (constant) Scalar_t, tensor for the input data.
        - batches: (constant) Integer, for the number of batches.
        - channels: (constant) Integer, for the number of channels.
        - height: (constant) Integer, specifies the size of the (iterable) height/y dimension of the tensor to pool over.
        - width: (constant) Integer, specifies the size of the (iterable) width/x dimension of the tensor to pool over.
        - kernel_h: (constant) Integer, for the height of the kernel.
        - kernel_w: (constant) Integer, for the width of the kernel.
        - stride_h: (constant) Integer, for the steps taken over the height/y dimension between kernels.
        - stride_w: (constant) Integer, for the steps taken over the width/x dimension between kernels.
        - diff_input: (constant) Scalar_t, tensor for the gradients (to be calculated) of the input data.

*/
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
    // Run in parallel for each cell within each kernel region
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      int pw = index % pooled_width; // index over width of each kernel operation in relation to the position in the input
      int ph = (index / pooled_width) % pooled_height; // index  over height of each kernel operation in relation to the position in the input
      int c = (index / pooled_width / pooled_height) % channels;
      int n = index / pooled_width / pooled_height / channels;

      const int offset0 = (n * channels + c) * height * width; // initial offset
      const scalar_t *offset_data_input = data_input + offset0; // offset based on the input data

      const scalar_t diff_output_index = diff_output[index]; // offset based on the output gradients
      scalar_t *offset_diff_input = diff_input + offset0; // offset based on the input gradients

      const int base_y = ph * stride_h; // start cell index over height/y for each kernel
      if (base_y > height - kernel_h)break; // limit height/y iterations for the index of the final kernel location in the input

      const int base_x = pw * stride_w; // start cell index over width/x for each kernel
      if (base_x > width - kernel_w)break; // limit width/x iterations for the index of the final kernel location in the input

      // --- Initialisations happen here ----
      scalar_t mask_sum_max = 0.;

      const scalar_t upper = n_limits<scalar_t>::max();
      const scalar_t lower = n_limits<scalar_t>::min();

      // Iterate over inputs cells within each kernel region in the input
      for(int iy=0; iy<kernel_h; iy++){
        const int y_offset = base_y + iy;

        if(y_offset >= height || y_offset < 0)continue; // check if the offset index over y is valid (not larger than or equal to the size of the dimension) OR smaller than 0 (for fool proofing)

        for(int ix=0; ix<kernel_w; ix++){
          const int x_offset = base_x + ix;

          if(x_offset >= width || x_offset < 0)continue; // check if the offset index over x is valid (not larger than or equal to the size of the dimension) OR smaller than 0 (for fool proofing)

          const int offset = y_offset*width + x_offset;

          // Use this for verbose when debugging
          // printf("(ph: %d, pw: %d), base_y: %d, base_x: %d, iy: %d, ix: %d offset: %d \n", ph, pw, base_y, base_x, iy, ix, offset)

          mask_sum_max += exp(offset_data_input[offset]);

        }
      }
      // Overflow check
      mask_sum_max = clamp(mask_sum_max, lower, upper);

      for(int iy=0; iy<kernel_h; iy++){
        const int y_offset = base_y + iy; // offset adjustment (y-based)

        if(y_offset >= height || y_offset < 0)continue;
        for(int ix=0; ix<kernel_w; ix++){
          const int x_offset = base_x + ix;

          if(x_offset >= width || x_offset < 0)continue;
            const int offset = y_offset*width + x_offset; // offset adjustment (x-based)

            scalar_t mask_ = exp(offset_data_input[offset])/mask_sum_max; // SoftMax (sum)

            scalar_t weighted_grad = diff_output_index * mask_; // use mask over the output gradients

            // Underflow check
            weighted_grad = clamp(weighted_grad, lower, upper);

            atomicAdd(offset_diff_input+offset, weighted_grad);
        }
      }
    }
}
/*
---  E N D  O F  T E M P L A T E  A D A _ E M _ P O O L 2 D B A C K W A R D ---
*/


/*
---  S T A R T  O F  T E M P L A T E  A D A _ E M _ P O O L 3 D B A C K W A R D ---
    [About]
        CUDA template utilising `CUDA_1D_KERNEL_LOOP` macro for iterating over the tensor.
    [Params]
        - No default for template parameter
    [Args]
        - nthreads: (constant) Integer, for the number of threads. From NVidia's website: "CUDA architecture limits the numbers of threads per block (1024 threads per block limit)".
        - diff_output: (constant) Scalar_t, tensor for the output's gradients.
        - data_input: (constant) Scalar_t, tensor for the input data.
        - batches: (constant) Integer, for the number of batches.
        - channels: (constant) Integer, for the number of channels.
        - depth: (constant) Integer, specifies the size of the (iterable) depth/d dimension of the tensor to pool over.
        - height: (constant) Integer, specifies the size of the (iterable) height/y dimension of the tensor to pool over.
        - width: (constant) Integer, specifies the size of the (iterable) width/x dimension of the tensor to pool over.
        - kernel_d: (constant) Integer, for the depth of the kernel.
        - kernel_h: (constant) Integer, for the height of the kernel.
        - kernel_w: (constant) Integer, for the width of the kernel.
        - stride_d: (constant) Integer, for the steps taken over the depth/d dimension between kernels.
        - stride_h: (constant) Integer, for the steps taken over the height/y dimension between kernels.
        - stride_w: (constant) Integer, for the steps taken over the width/x dimension between kernels.
        - diff_input: (constant) Scalar_t, tensor for the gradients (to be calculated) of the input data.

*/
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
      int pw = index % pooled_width; // index over width of each kernel operation in relation to the position in the input
      int ph = (index / pooled_width) % pooled_height; // index over height of each kernel operation in relation to the position in the input
      int pd = (index / pooled_width / pooled_height) % pooled_depth; // index over depth of each kernel operation in relation to the position in the input
      int c = (index / pooled_width / pooled_height / pooled_depth) % channels;
      int n = index / pooled_width / pooled_height / pooled_depth / channels;

      const int offset0 = (n * channels + c) * depth * height * width; // initial offset
      const scalar_t *offset_data_input = data_input + offset0; // offset based on the input data

      const scalar_t diff_output_index = diff_output[index]; // offset based on the output gradients
      scalar_t *offset_diff_input = diff_input + offset0; // offset based on the input gradients

      const int base_d = pd*stride_d; // start cell index over depth/d for each kernel
      if (base_d > depth - kernel_d)break; // limit depth/d iterations for the index of the final kernel location in the input

      const int base_y = ph*stride_h; // start cell index over height/y for each kernel
      if (base_y > height - kernel_h)break; // limit height/y iterations for the index of the final kernel location in the input

      const int base_x = pw*stride_w; // start cell index over width/x for each kernel
      if (base_x > width - kernel_w)break; // limit width/x iterations for the index of the final kernel location in the input

      // --- Initialisations happen here ----
      scalar_t mask_sum_max = 0.;

      const scalar_t upper = n_limits<scalar_t>::max();
      const scalar_t lower = n_limits<scalar_t>::min();

      // Iterate over inputs cells within each kernel region in the input
      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;

        if(d_offset >= depth || d_offset < 0)continue; // check if the offset index over d is valid (not larger than or equal to the size of the dimension) OR smaller than 0 (for fool proofing)

        for(int iy=0; iy<kernel_h; iy++){
          const int y_offset = base_y + iy;

          if(y_offset >= height || y_offset < 0)continue; // check if the offset index over y is valid (not larger than or equal to the size of the dimension) OR smaller than 0 (for fool proofing)

          for(int ix=0; ix<kernel_w; ix++){
            const int x_offset = base_x + ix;

            if(x_offset >= width || x_offset < 0)continue; // check if the offset index over x is valid (not larger than or equal to the size of the dimension) OR smaller than 0 (for fool proofing)

            const int offset = d_offset*height + y_offset*width + x_offset;

            // Use this for verbose when debugging
            // printf("(pd: %d, ph: %d, pw: %d), base_d: %d, base_y: %d, base_x: %d, id: %d, iy: %d, ix: %d, offset: %d \n", pd, ph, pw, base_d, base_y, base_x, id, iy, ix, offset);

            mask_sum_max += exp(offset_data_input[offset]);

          }
        }
      }
      // Overflow check
      mask_sum_max = clamp(mask_sum_max, lower, upper);

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id; // offset adjustment (d-based)

        if(d_offset >= depth || d_offset < 0)continue;
        for(int iy=0; iy<kernel_h; iy++){
          const int y_offset = base_y + iy; // offset adjustment (y-based)

          if(y_offset >= height || y_offset < 0)continue;
          for(int ix=0; ix<kernel_w; ix++){
            const int x_offset = base_x + ix; // offset adjustment (x-based)

            if(x_offset >= width || x_offset < 0)continue;
              const int offset = d_offset*height + y_offset*width + x_offset;

              scalar_t mask_ = exp(offset_data_input[offset])/mask_sum_max; // SoftMax

              scalar_t weighted_grad = diff_output_index * mask_; // use mask over the output gradients

              // Underflow check
              weighted_grad = clamp(weighted_grad, lower, upper);

              atomicAdd(offset_diff_input+offset, weighted_grad);
          }
        }
      }
    }
}
/*
---  E N D  O F  T E M P L A T E  A D A _ E M _ P O O L 3 D B A C K W A R D ---
*/

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



/*
---  S T A R T  O F  T E M P L A T E  I D W _ P O O L 1 D F O R W A R D ---
    [About]
        CUDA template utilising `CUDA_1D_KERNEL_LOOP` macro for iterating over the tensor.
    [Params]
        - No default for template parameter
    [Args]
        - nthreads: (constant) Integer, for the number of threads. From NVidia's website: "CUDA architecture limits the numbers of threads per block (1024 threads per block limit)".
        - bottom_input: (constant) Scalar_t, tensor for the input data.
        - batches: (constant) Integer, for the number of batches.
        - channels: (constant) Integer, for the number of channels.
        - dim: (constant) Integer, specifies the size of the iterable dimension of the tensor to pool over.
        - kernel_d: (constant) Integer, for the size of the kernel.
        - stride_d: (constant) Integer, for the steps taken between kernels.
        - output_data: constant) Scalar_t, tensor to assign the calculated output data.
        - return_mask: (constant) Boolean, if the calculated mask should be returned.
        - mask: (constant) Scalar_t, tensor of size # kernel ops (output size) x `kernel_d`.

*/
template <typename scalar_t>
__global__ void IDW_Pool1dForward(const int nthreads,
                                  const scalar_t *bottom_input, const int batches,
                                  const int channels, const int dim,
                                  const int kernel_d, const int stride_d,
                                  scalar_t *output_data, const bool return_mask,
                                  scalar_t *mask){
    int pooled_dim = dim/stride_d;
    // Run in parallel for each cell within each kernel region
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      int pd = index % pooled_dim; // index of each kernel operation in relation to the position in the input
      int c = (index / pooled_dim) % channels;
      int n = index / pooled_dim / channels;

      const int offset = (n * channels + c) * dim; // initial offset
      const scalar_t *offset_bottom_input = bottom_input + offset;

      const int base_d = pd*stride_d; // start cell index for each kernel
      if (base_d > dim - kernel_d)break; // limit iterations based on the position of the final kernel application over the input

      // --- Initialisations happen here ----
      scalar_t act_sum = 0.;
      scalar_t mask_sum_avg = 0.;

      output_data[index] = 0.;
      const scalar_t upper = n_limits<scalar_t>::max();
      const scalar_t lower = n_limits<scalar_t>::min();
      const scalar_t zero = 0.;

      int count = 0.; // used for calculating the average

      // Iterate over inputs cells within each kernel region in the input
      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;

        if(d_offset >= dim || d_offset < 0)continue; // check if the offset index is valid (not larger than or equal to the size of the dimension) OR smaller than 0 (for fool proofing)
        const int offset = d_offset;

        // Use this for verbose when debugging
        //printf("(pd: %d), base_d: %d, id: %d, d_offset: %d \n", pd, base_d, id, d_offset);

        act_sum += offset_bottom_input[offset];
        count += 1;
      }
      scalar_t act_avg = act_sum/count; // average

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id; // offset adjustment

        if(d_offset >= dim || d_offset < 0)continue;
        const int offset = d_offset;

        // ! Note: Change the template call if you woulbe prefer to use a different distance method (L1/Huber etc.)
        scalar_t dist = l2(offset_bottom_input[offset], act_avg); // L2 distance

        mask_sum_avg += pow(dist,-1);

      }
      // Overflow check
      mask_sum_avg = clamp(mask_sum_avg, lower, upper);


      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;

        if(d_offset >= dim || d_offset < 0)continue;
        const int offset = d_offset;

        scalar_t dist = l2(offset_bottom_input[offset], act_avg);

        scalar_t mask_ = pow(dist,-1)/mask_sum_avg;// IDW

        if (return_mask) {
          int offset_m = (pd * kernel_d) + id; // calculate mask offset
          //printf("offset mask: %d \n", offset_m);
          mask[offset_m]= mask_;
        }

        output_data[index] += offset_bottom_input[offset] * mask_;
        output_data[index] = clamp(output_data[index], zero, upper);
      }
    }
}
/*
---  E N D  O F  T E M P L A T E  I D W _ P O O L 1 D F O R W A R D ---
*/


/*
---  S T A R T  O F  T E M P L A T E  I D W _ P O O L 2 D F O R W A R D ---
    [About]
        CUDA template utilising `CUDA_1D_KERNEL_LOOP` macro for iterating over the tensor.
    [Params]
        - No default for template parameter
    [Args]
        - nthreads: (constant) Integer, for the number of threads. From NVidia's website: "CUDA architecture limits the numbers of threads per block (1024 threads per block limit)".
        - bottom_input: (constant) Scalar_t, tensor for the input data.
        - batches: (constant) Integer, for the number of batches.
        - channels: (constant) Integer, for the number of channels.
        - height: (constant) Integer, specifies the size of the (iterable) height/y dimension of the tensor to pool over.
        - width: (constant) Integer, specifies the size of the (iterable) width/x dimension of the tensor to pool over.
        - kernel_h: (constant) Integer, for the height of the kernel.
        - kernel_w: (constant) Integer, for the width of the kernel.
        - stride_h: (constant) Integer, for the steps taken over the height/y dimension between kernels.
        - stride_w: (constant) Integer, for the steps taken over the width/x dimension between kernels.
        - output_data: constant) Scalar_t, tensor to assign the calculated output data.
        - return_mask: (constant) Boolean, if the calculated mask should be returned.
        - mask: (constant) Scalar_t, tensor of size # kernel ops (output size) x `kernel_h` x `kernel_w`.

*/
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
    // Run in parallel for each cell within each kernel region
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      int pw = index % pooled_width; // index over width of each kernel operation in relation to the position in the input
      int ph = (index / pooled_width) % pooled_height; // index  over height of each kernel operation in relation to the position in the input
      int c = (index / pooled_width / pooled_height) % channels;
      int n = index / pooled_width / pooled_height / channels;

      const int offset = (n * channels + c) * height * width; // initial offset
      const scalar_t *offset_bottom_input = bottom_input + offset;

      const int base_y = ph * stride_h; // start cell index over height/y for each kernel
      if (base_y > height - kernel_h)break; // limit height/y iterations for the index of the final kernel location in the input

      const int base_x = pw * stride_w; // start cell index over width/x for each kernel
      if (base_x > width - kernel_w)break; // limit width/x iterations for the index of the final kernel location in the input

      // --- Initialisations happen here ----
      scalar_t act_sum = 0.;
      scalar_t mask_sum_avg = 0.;

      output_data[index] = 0.;
      const scalar_t upper = n_limits<scalar_t>::max();
      const scalar_t lower = n_limits<scalar_t>::min();
      const scalar_t zero = 0.;

      int count = 0.; // used for calculating the average

      // Iterate over inputs cells within each kernel region in the input
      for(int iy=0; iy<kernel_h; iy++){
        const int y_offset = base_y + iy;

        if(y_offset >= height || y_offset < 0)continue; // check if the offset index over y is valid (not larger than or equal to the size of the dimension) OR smaller than 0 (for fool proofing)

        for(int ix=0; ix<kernel_w; ix++){
          const int x_offset = base_x + ix;
          if(x_offset >= width || x_offset < 0)continue;
          // check if the offset index over x is valid (not larger than or equal to the size of the dimension) OR smaller than 0 (for fool proofing)

          const int offset = y_offset*width + x_offset;

          // Use this for verbose when debugging
          // printf("(ph: %d, pw: %d), base_y: %d, base_x: %d, iy: %d, ix: %d offset: %d \n", ph, pw, base_y, base_x, iy, ix, offset)

          act_sum += offset_bottom_input[offset];
          count += 1;
        }
      }
      scalar_t act_avg = act_sum/count; // average calculation

      for(int iy=0; iy<kernel_h; iy++){
        const int y_offset = base_y + iy;

        if(y_offset >= height || y_offset < 0)continue;
        for(int ix=0; ix<kernel_w; ix++){
          const int x_offset = base_x + ix;

          if(x_offset >= width || x_offset < 0)continue;
          const int offset = y_offset*width + x_offset;

          // ! Note: Change the template call if you woulbe prefer to use a different distance method (L1/Huber etc.)
          scalar_t dist = l2(offset_bottom_input[offset], act_avg); // L2 distance

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

          scalar_t mask_ = pow(dist,-1)/mask_sum_avg;// IDW

          if (return_mask) {
            // mask size = num_kernel_ops x kernel_size
            int oW = (width - kernel_w) / stride_w + 1 ;// calculate number of operations
            int w_mask = oW * kernel_w; // mask width
            int offset_m = ((ph * kernel_h) + iy) * w_mask; // offset over H dimension
            offset_m = offset_m + (pw * kernel_h) + ix; // offset over both H and W dimension
            // Use this for verbose when debugging
            // printf("offset mask: %d \n", offset_m);
            mask[offset_m]= mask_;
          }

          output_data[index] += offset_bottom_input[offset] * mask_;
          output_data[index] = clamp(output_data[index], zero, upper);
        }
      }
    }
}
/*
---  E N D  O F  T E M P L A T E  I D W _ P O O L 2 D F O R W A R D ---
*/


/*
---  S T A R T  O F  T E M P L A T E  I D W _ P O O L 3 D F O R W A R D ---
    [About]
        CUDA template utilising `CUDA_1D_KERNEL_LOOP` macro for iterating over the tensor.
    [Params]
        - No default for template parameter
    [Args]
        - nthreads: (constant) Integer, for the number of threads. From NVidia's website: "CUDA architecture limits the numbers of threads per block (1024 threads per block limit)".
        - bottom_input: (constant) Scalar_t, tensor for the input data.
        - batches: (constant) Integer, for the number of batches.
        - channels: (constant) Integer, for the number of channels.
        - depth: (constant) Integer, specifies the size of the (iterable) depth/d dimension of the tensor to pool over.
        - height: (constant) Integer, specifies the size of the (iterable) height/y dimension of the tensor to pool over.
        - width: (constant) Integer, specifies the size of the (iterable) width/x dimension of the tensor to pool over.
        - kernel_d: (constant) Integer, for the depth of the kernel.
        - kernel_h: (constant) Integer, for the height of the kernel.
        - kernel_w: (constant) Integer, for the width of the kernel.
        - stride_d: (constant) Integer, for the steps taken over the depth/d dimension between kernels.
        - stride_h: (constant) Integer, for the steps taken over the height/y dimension between kernels.
        - stride_w: (constant) Integer, for the steps taken over the width/x dimension between kernels.
        - output_data: constant) Scalar_t, tensor to assign the calculated output data.
        - return_mask: (constant) Boolean, if the calculated mask should be returned.
        - mask: (constant) Scalar_t, tensor of size # kernel ops (output size) x `kernel_d` x `kernel_h` x `kernel_w`.

*/
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
    // Run in parallel for each cell within each kernel region
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      int pw = index % pooled_width; // index over width of each kernel operation in relation to the position in the input
      int ph = (index / pooled_width) % pooled_height; // index over height of each kernel operation in relation to the position in the input
      int pd = (index / pooled_width / pooled_height) % pooled_depth; // index over depth of each kernel operation in relation to the position in the input
      int c = (index / pooled_width / pooled_height / pooled_depth) % channels;
      int n = index / pooled_width / pooled_height / pooled_depth / channels;

      const int offset = (n * channels + c) * depth * height * width; // initial offset
      const scalar_t *offset_bottom_input = bottom_input + offset;

      const int base_d = pd*stride_d; // start cell index over depth/d for each kernel
      if (base_d > depth - kernel_d)break; // limit depth/d iterations for the index of the final kernel location in the input

      const int base_y = ph*stride_h; // start cell index over height/y for each kernel
      if (base_y > height - kernel_h)break; // limit height/y iterations for the index of the final kernel location in the input

      const int base_x = pw*stride_w; // start cell index over width/x for each kernel
      if (base_x > width - kernel_w)break; // limit width/x iterations for the index of the final kernel location in the input

      // --- Initialisations happen here ----
      scalar_t act_sum = 0.;
      scalar_t mask_sum_avg = 0.;

      output_data[index] = 0.;
      const scalar_t upper = n_limits<scalar_t>::max();
      const scalar_t lower = n_limits<scalar_t>::min();
      const scalar_t zero = 0.;

      int count = 0.;

      // Iterate over inputs cells within each kernel region in the input
      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;

        if(d_offset >= depth || d_offset < 0)continue; // check if the offset index over d is valid (not larger than or equal to the size of the dimension) OR smaller than 0 (for fool proofing)

        for(int iy=0; iy<kernel_h; iy++){
          const int y_offset = base_y + iy;

          if(y_offset >= height || y_offset < 0)continue; // check if the offset index over y is valid (not larger than or equal to the size of the dimension) OR smaller than 0 (for fool proofing)

          for(int ix=0; ix<kernel_w; ix++){
            const int x_offset = base_x + ix;

            if(x_offset >= width || x_offset < 0)continue; // check if the offset index over x is valid (not larger than or equal to the size of the dimension) OR smaller than 0 (for fool proofing)

            const int offset = d_offset*height + y_offset*width + x_offset;

            // Use this for verbose when debugging
            // printf("(pd: %d, ph: %d, pw: %d), base_d: %d, base_y: %d, base_x: %d, id: %d, iy: %d, ix: %d, offset: %d \n", pd, ph, pw, base_d, base_y, base_x, id, iy, ix, offset);

            act_sum += offset_bottom_input[offset];
            count += 1;
          }
        }
      }
      scalar_t act_avg = act_sum/count; // average calculation


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

            // ! Note: Change the template call if you woulbe prefer to use a different distance method (L1/Huber etc.)
            scalar_t dist = l2(offset_bottom_input[offset], act_avg); // L2 distance

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

            scalar_t mask_ = pow(dist,-1)/mask_sum_avg; // IDW

            if (return_mask) {
              // mask size = num_kernel_ops x kernel_size
              int oH = (height - kernel_h) / stride_h + 1; // calculate number of H-dim operations
              int h_mask = oH * kernel_h; // mask height

              int oW = (width - kernel_w) / stride_w + 1; // calculate number of W-dim operations
              int w_mask = oW * kernel_w; // mask width

              int offset_m = ((pd * kernel_d) + id) * h_mask * w_mask; // offset over D dimension
              offset_m = offset_m + ((ph * kernel_h) + iy) * w_mask; // offset over H dimension
              offset_m = offset_m + (pw * kernel_w) + ix; // offset over D, H and W dimension
              // Use this for verbose when debugging
              // printf("offset mask: %d \n", offset_m);
              mask[offset_m]= mask_;
            }


            output_data[index] += offset_bottom_input[offset] * mask_;
            output_data[index] = clamp(output_data[index], zero, upper);

          }
        }
      }
    }
}
/*
---  E N D  O F  T E M P L A T E  I D W _ P O O L 3 D F O R W A R D ---
*/

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


/*
---  S T A R T  O F  T E M P L A T E  I D W P O O L 1 D B A C K W A R D ---
    [About]
        CUDA template utilising `CUDA_1D_KERNEL_LOOP` macro for iterating over the tensor.
    [Params]
        - No default for template parameter
    [Args]
        - nthreads: (constant) Integer, for the number of threads. From NVidia's website: "CUDA architecture limits the numbers of threads per block (1024 threads per block limit)".
        - diff_output: (constant) Scalar_t, tensor for the output's gradients.
        - data_input: (constant) Scalar_t, tensor for the input data.
        - batches: (constant) Integer, for the number of batches.
        - channels: (constant) Integer, for the number of channels.
        - dim: (constant) Integer, specifies the size of the iterable dimension of the tensor to pool over.
        - kernel_d: (constant) Integer, for the size of the kernel.
        - stride_d: (constant) Integer, for the steps taken between kernels.
        - diff_input: (constant) Scalar_t, tensor for the gradients (to be calculated) of the input data.

*/
template <typename scalar_t>
__global__ void IDW_Pool1dBackward(const int nthreads,
                                   const scalar_t *diff_output, const scalar_t *data_input,
                                   const int batches, const int channels,
                                   const int dim, const int kernel_d,
                                   const int stride_d, scalar_t *diff_input){
    int pooled_dim = dim/stride_d;
    // Run in parallel for each cell within each kernel region
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      int pd = index % pooled_dim; // index of each kernel operation in relation to the position in the input
      int c = (index / pooled_dim) % channels;
      int n = index / pooled_dim / channels;

      const int offset0 = (n * channels + c) * dim; // initial offset
      const scalar_t *offset_data_input = data_input + offset0; // offset based on the input data

      const scalar_t diff_output_index = diff_output[index]; // offset based on the output gradients
      scalar_t *offset_diff_input = diff_input + offset0; // offset based on the input gradients

      const int base_d = pd*stride_d; // start cell index for each kernel

      // --- Initialisations happen here ----
      scalar_t act_sum = 0.;
      scalar_t mask_sum_avg = 0.;

      const scalar_t upper = n_limits<scalar_t>::max();
      const scalar_t lower = n_limits<scalar_t>::min();

      int count = 0.; // used for calculating the average

      // Iterate over inputs cells within each kernel region in the input
      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;

        if(d_offset >= dim || d_offset < 0)continue;// check if the offset index is valid (not larger than or equal to the size of the dimension) OR smaller than 0 (for fool proofing)
        const int offset = d_offset;

        // Use this for verbose when debugging
        //printf("(pd: %d), base_d: %d, id: %d, d_offset: %d \n", pd, base_d, id, d_offset);

        act_sum += offset_data_input[offset]; // average calculation
        count += 1;
      }
      scalar_t act_avg = act_sum/count;

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;

        if(d_offset >= dim || d_offset < 0)continue;
        const int offset = d_offset;

        // ! Note: Change the template call if you woulbe prefer to use a different distance method (L1/Huber etc.)
        scalar_t dist = l2(offset_data_input[offset], act_avg); // L2 distance

        mask_sum_avg += pow(dist,-1);

      }
      // Overflow check
      mask_sum_avg = clamp(mask_sum_avg, lower, upper);

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;

        if(d_offset >= dim || d_offset < 0)continue;
          const int offset = d_offset;

          scalar_t dist = l2(offset_data_input[offset], act_avg);

          scalar_t mask = pow(dist,-1)/mask_sum_avg; // IDW

          scalar_t weighted_grad = diff_output_index * mask; // use mask over the output gradients

          // Underflow check
          weighted_grad = clamp(weighted_grad, lower, upper);

          atomicAdd(offset_diff_input+offset, weighted_grad);
      }
    }
}
/*
---  E N D  O F  T E M P L A T E  I D W P O O L 1 D B A C K W A R D ---
*/


/*
---  S T A R T  O F  T E M P L A T E  A D A P O O L 2 D B A C K W A R D ---
    [About]
        CUDA template utilising `CUDA_1D_KERNEL_LOOP` macro for iterating over the tensor.
        ! Note: This is used by the native implementation of adaPool2d. Because of the large requirement
        in resources it may be unstable. Further refinement may be required!
    [Params]
        - No default for template parameter
    [Args]
        - nthreads: (constant) Integer, for the number of threads. From NVidia's website: "CUDA architecture limits the numbers of threads per block (1024 threads per block limit)".
        - diff_output: (constant) Scalar_t, tensor for the output's gradients.
        - data_input: (constant) Scalar_t, tensor for the input data.
        - data_beta: (constant) Scalar_t, tensor with beta parameter to be used during pooling.
        - batches: (constant) Integer, for the number of batches.
        - channels: (constant) Integer, for the number of channels.
        - height: (constant) Integer, specifies the size of the (iterable) height/y dimension of the tensor to pool over.
        - width: (constant) Integer, specifies the size of the (iterable) width/x dimension of the tensor to pool over.
        - kernel_h: (constant) Integer, for the height of the kernel.
        - kernel_w: (constant) Integer, for the width of the kernel.
        - stride_h: (constant) Integer, for the steps taken over the height/y dimension between kernels.
        - stride_w: (constant) Integer, for the steps taken over the width/x dimension between kernels.
        - diff_input: (constant) Scalar_t, tensor for the gradients (to be calculated) of the input data.
        - diff_beta: (constant) Scalar_t, tensor for the gradients (to be calculated) of the beta param.

*/
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
    // Run in parallel for each cell within each kernel region
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      int pw = index % pooled_width; // index over width of each kernel operation in relation to the position in the input
      int ph = (index / pooled_width) % pooled_height; // index  over height of each kernel operation in relation to the position in the input
      int c = (index / pooled_width / pooled_height) % channels;
      int n = index / pooled_width / pooled_height / channels;

      const int offset0 = (n * channels + c) * height * width; // initial offset
      const scalar_t *offset_data_input = data_input + offset0; // offset based on the input data

      const scalar_t diff_output_index = diff_output[index]; // offset based on the output gradients
      scalar_t *offset_diff_input = diff_input + offset0; // offset based on the input gradients

      const int base_y = ph * stride_h; // start cell index over height/y for each kernel
      if (base_y > height - kernel_h)break; // limit height/y iterations for the index of the final kernel location in the input

      const int base_x = pw * stride_w; // start cell index over width/x for each kernel
      if (base_x > width - kernel_w)break; // limit width/x iterations for the index of the final kernel location in the input

      // --- Initialisations happen here ----
      scalar_t act_sum = 0.;
      scalar_t mask_sum_avg = 0.;

      const scalar_t upper = n_limits<scalar_t>::max();
      const scalar_t lower = n_limits<scalar_t>::min();

      int count = 0.; // used for calculating the average

      // Iterate over inputs cells within each kernel region in the input
      for(int iy=0; iy<kernel_h; iy++){
        const int y_offset = base_y + iy;

        if(y_offset >= height || y_offset < 0)continue; // check if the offset index over y is valid (not larger than or equal to the size of the dimension) OR smaller than 0 (for fool proofing)

        for(int ix=0; ix<kernel_w; ix++){
          const int x_offset = base_x + ix;

          if(x_offset >= width || x_offset < 0)continue; // check if the offset index over x is valid (not larger than or equal to the size of the dimension) OR smaller than 0 (for fool proofing)

          const int offset = y_offset*width + x_offset;

          // Use this for verbose when debugging
          // printf("(ph: %d, pw: %d), base_y: %d, base_x: %d, iy: %d, ix: %d offset: %d \n", ph, pw, base_y, base_x, iy, ix, offset)

          act_sum += offset_data_input[offset];
          count += 1;
        }
      }
      scalar_t act_avg = act_sum/count; // average calculation

      for(int iy=0; iy<kernel_h; iy++){
        const int y_offset = base_y + iy;

        if(y_offset >= height || y_offset < 0)continue;
        for(int ix=0; ix<kernel_w; ix++){
          const int x_offset = base_x + ix;

          if(x_offset >= width || x_offset < 0)continue;
          const int offset = y_offset*width + x_offset;

          // ! Note: Change the template call if you woulbe prefer to use a different distance method (L1/Huber etc.)
          scalar_t dist = l2(offset_data_input[offset], act_avg); // L2 distance

          mask_sum_avg += pow(dist,-1);

        }
      }
      // Overflow check
      mask_sum_avg = clamp(mask_sum_avg, lower, upper);

      for(int iy=0; iy<kernel_h; iy++){
        const int y_offset = base_y + iy; // offset adjustment (y-based)

        if(y_offset >= height || y_offset < 0)continue;
        for(int ix=0; ix<kernel_w; ix++){
          const int x_offset = base_x + ix; // offset adjustment (x-based)

          if(x_offset >= width || x_offset < 0)continue;
            const int offset = y_offset*width + x_offset;

            scalar_t dist = l2(offset_data_input[offset], act_avg);

            scalar_t mask = pow(dist,-1)/mask_sum_avg; // IDW

            scalar_t weighted_grad = diff_output_index * mask; // use mask over the output gradients

            // Underflow check
            weighted_grad = clamp(weighted_grad, lower, upper);

            atomicAdd(offset_diff_input+offset, weighted_grad);
        }
      }
    }
}
/*
---  E N D  O F  T E M P L A T E  I D W P O O L 2 D B A C K W A R D ---
*/

/*
---  S T A R T  O F  T E M P L A T E  I D W _ P O O L 3 D B A C K W A R D ---
    [About]
        CUDA template utilising `CUDA_1D_KERNEL_LOOP` macro for iterating over the tensor.
    [Params]
        - No default for template parameter
    [Args]
        - nthreads: (constant) Integer, for the number of threads. From NVidia's website: "CUDA architecture limits the numbers of threads per block (1024 threads per block limit)".
        - diff_output: (constant) Scalar_t, tensor for the output's gradients.
        - data_input: (constant) Scalar_t, tensor for the input data.
        - batches: (constant) Integer, for the number of batches.
        - channels: (constant) Integer, for the number of channels.
        - depth: (constant) Integer, specifies the size of the (iterable) depth/d dimension of the tensor to pool over.
        - height: (constant) Integer, specifies the size of the (iterable) height/y dimension of the tensor to pool over.
        - width: (constant) Integer, specifies the size of the (iterable) width/x dimension of the tensor to pool over.
        - kernel_d: (constant) Integer, for the depth of the kernel.
        - kernel_h: (constant) Integer, for the height of the kernel.
        - kernel_w: (constant) Integer, for the width of the kernel.
        - stride_d: (constant) Integer, for the steps taken over the depth/d dimension between kernels.
        - stride_h: (constant) Integer, for the steps taken over the height/y dimension between kernels.
        - stride_w: (constant) Integer, for the steps taken over the width/x dimension between kernels.
        - diff_input: (constant) Scalar_t, tensor for the gradients (to be calculated) of the input data.

*/
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
    // Run in parallel for each cell within each kernel region
    CUDA_1D_KERNEL_LOOP(index, nthreads) {
      int pw = index % pooled_width; // index over width of each kernel operation in relation to the position in the input
      int ph = (index / pooled_width) % pooled_height; // index over height of each kernel operation in relation to the position in the input
      int pd = (index / pooled_width / pooled_height) % pooled_depth; // index over depth of each kernel operation in relation to the position in the input
      int c = (index / pooled_width / pooled_height / pooled_depth) % channels;
      int n = index / pooled_width / pooled_height / pooled_depth / channels;

      const int offset0 = (n * channels + c) * depth * height * width; // initial offset
      const scalar_t *offset_data_input = data_input + offset0; // offset based on the input data

      const scalar_t diff_output_index = diff_output[index]; // offset based on the output gradients
      scalar_t *offset_diff_input = diff_input + offset0; // offset based on the input gradients

      const int base_d = pd*stride_d; // start cell index over depth/d for each kernel
      if (base_d > depth - kernel_d)break; // limit depth/d iterations for the index of the final kernel location in the input

      const int base_y = ph*stride_h; // start cell index over height/y for each kernel
      if (base_y > height - kernel_h)break; // limit height/y iterations for the index of the final kernel location in the input

      const int base_x = pw*stride_w; // start cell index over width/x for each kernel
      if (base_x > width - kernel_w)break; // limit width/x iterations for the index of the final kernel location in the input

      // --- Initialisations happen here ----
      scalar_t act_sum = 0.;
      scalar_t mask_sum_avg = 0.;

      const scalar_t upper = n_limits<scalar_t>::max();
      const scalar_t lower = n_limits<scalar_t>::min();

      int count = 0.; // used for calculating the average

      // Iterate over inputs cells within each kernel region in the input
      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id;

        if(d_offset >= depth || d_offset < 0)continue; // check if the offset index over d is valid (not larger than or equal to the size of the dimension) OR smaller than 0 (for fool proofing)

        for(int iy=0; iy<kernel_h; iy++){
          const int y_offset = base_y + iy;

          if(y_offset >= height || y_offset < 0)continue; // check if the offset index over y is valid (not larger than or equal to the size of the dimension) OR smaller than 0 (for fool proofing)

          for(int ix=0; ix<kernel_w; ix++){
            const int x_offset = base_x + ix;

            if(x_offset >= width || x_offset < 0)continue; // check if the offset index over x is valid (not larger than or equal to the size of the dimension) OR smaller than 0 (for fool proofing)

            const int offset = d_offset*height + y_offset*width + x_offset;

            // Use this for verbose when debugging
            // printf("(pd: %d, ph: %d, pw: %d), base_d: %d, base_y: %d, base_x: %d, id: %d, iy: %d, ix: %d, offset: %d \n", pd, ph, pw, base_d, base_y, base_x, id, iy, ix, offset);

            act_sum += offset_data_input[offset];
            count += 1;
          }
        }
      }
      scalar_t act_avg = act_sum/count; // average calculation

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

              // ! Note: Change the template call if you woulbe prefer to use a different distance method (L1/Huber etc.)
              scalar_t dist = l2(offset_data_input[offset], act_avg); // L2 distance

              mask_sum_avg += pow(dist,-1);

          }
        }
      }
      // Overflow check
      mask_sum_avg = clamp(mask_sum_avg, lower, upper);

      for(int id=0; id<kernel_d; id++){
        const int d_offset = base_d + id; // offset adjustment (d-based)

        if(d_offset >= depth || d_offset < 0)continue;
        for(int iy=0; iy<kernel_h; iy++){
          const int y_offset = base_y + iy; // offset adjustment (y-based)

          if(y_offset >= height || y_offset < 0)continue;
          for(int ix=0; ix<kernel_w; ix++){
            const int x_offset = base_x + ix; // offset adjustment (x-based)

            if(x_offset >= width || x_offset < 0)continue;
              const int offset = d_offset*height + y_offset*width + x_offset;

              scalar_t dist = l2(offset_data_input[offset], act_avg);

              scalar_t mask = pow(dist,-1)/mask_sum_avg; // IDW

              scalar_t weighted_grad = diff_output_index/mask; // use mask over the output gradients

              // Underflow check
              weighted_grad = clamp(weighted_grad, lower, upper);

              atomicAdd(offset_diff_input+offset, weighted_grad);
          }
        }
      }
    }
}
/*
---  E N D  O F  T E M P L A T E  I D W _ P O O L 3 D B A C K W A R D ---
*/


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
