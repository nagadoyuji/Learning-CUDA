#include <cstddef>
#include <vector>
#include <type_traits>
#include <cuda_fp16.h>

#include "../tester/utils.h"

/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */
template <typename T>
__global__ void traceKernel(const T* input, T* output, size_t rows, size_t cols) {
  __shared__ T shared_data[256];
  
  size_t tid = threadIdx.x;
  size_t diagonal_size = min(rows, cols);
  size_t stride = blockDim.x;
  
  T sum = T(0);
  
  for (size_t i = tid; i < diagonal_size; i += stride) {
    size_t idx = i * cols + i;
    sum += input[idx];
  }
  
  shared_data[tid] = sum;
  __syncthreads();
  
  for (size_t s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      shared_data[tid] += shared_data[tid + s];
    }
    __syncthreads();
  }
  
  if (tid == 0) {
    output[0] = shared_data[0];
  }
}

template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
  T* d_input;
  T* d_output;
  size_t size = h_input.size() * sizeof(T);
  
  cudaMalloc(&d_input, size);
  cudaMalloc(&d_output, sizeof(T));
  
  cudaMemcpy(d_input, h_input.data(), size, cudaMemcpyHostToDevice);
  
  int blockSize = 256;
  int gridSize = 1;
  
  traceKernel<T><<<gridSize, blockSize>>>(d_input, d_output, rows, cols);
  cudaDeviceSynchronize();
  
  T result;
  cudaMemcpy(&result, d_output, sizeof(T), cudaMemcpyDeviceToHost);
  
  cudaFree(d_input);
  cudaFree(d_output);
  
  return result;
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */

// 加法操作
__device__ float add_op(float a, float b) {
  return a + b;
}

__device__ half add_op(half a, half b) {
  // 使用更精确的加法计算
  float f_a = __half2float(a);
  float f_b = __half2float(b);
  float result = f_a + f_b;
  // 避免溢出
  if (result > 65504.0f) return __float2half_rn(65504.0f);
  if (result < -65504.0f) return __float2half_rn(-65504.0f);
  // 使用更精确的舍入方式
  return __float2half_rn(result);
}

// 减法操作
__device__ float subtract_op(float a, float b) {
  return a - b;
}

__device__ half subtract_op(half a, half b) {
  return __float2half_rn(__half2float(a) - __half2float(b));
}

// 乘法操作
__device__ float multiply_op(float a, float b) {
  return a * b;
}

__device__ half multiply_op(half a, half b) {
  return __float2half_rn(__half2float(a) * __half2float(b));
}

// 缩放操作
__device__ float scale_op(float value, float scale) {
  return value * scale;
}

__device__ half scale_op(half value, float scale) {
  // 使用更精确的缩放计算
  float f_val = __half2float(value);
  float result = f_val * scale;
  // 避免溢出
  if (result > 65504.0f) return __float2half_rn(65504.0f);
  if (result < -65504.0f) return __float2half_rn(-65504.0f);
  // 使用更精确的舍入方式
  return __float2half_rn(result);
}

// 指数操作
__device__ float exp_op(float value) {
  return expf(value);
}

__device__ float exp_op(half value) {
  // 使用更精确的指数计算
  float f_val = __half2float(value);
  // 避免溢出
  if (f_val > 88.0f) return expf(88.0f);
  if (f_val < -88.0f) return 0.0f;
  return expf(f_val);
}

// 比较操作
__device__ bool greater_op(float a, float b) {
  return a > b;
}

__device__ bool greater_op(half a, half b) {
  return __half2float(a) > __half2float(b);
}

// 通用模板版本
template <typename T>
__global__ void flashAttentionKernel(const T* q, const T* k, const T* v, T* o,
                                     int batch_size, int target_seq_len, int src_seq_len,
                                     int query_heads, int kv_heads, int head_dim,
                                     bool is_causal) {
  // 每个block处理一个batch中的一个head group
  int batch_idx = blockIdx.x / query_heads;
  int q_head_idx = blockIdx.x % query_heads;
  int kv_head_idx = q_head_idx / (query_heads / kv_heads);
  
  // 计算偏移量
  size_t q_offset = batch_idx * target_seq_len * query_heads * head_dim;
  size_t k_offset = batch_idx * src_seq_len * kv_heads * head_dim;
  size_t v_offset = k_offset;
  size_t o_offset = q_offset;
  
  // 计算缩放因子
  float scale = 1.0f / sqrtf(head_dim);
  
  // 每个线程处理一个target token
  for (int t = threadIdx.x; t < target_seq_len; t += blockDim.x) {
    // 获取当前query向量
    const T* q_vec = q + q_offset + t * query_heads * head_dim + q_head_idx * head_dim;
    
    // 计算注意力权重和输出
    T attn_output[128]; // 假设head_dim最大为128
    for (int d = 0; d < head_dim; d++) {
      attn_output[d] = 0;
    }
    
    // 使用float存储中间结果，提高精度
    float max_val = -1e10f;
    float sum_exp = 0.0f;
    
    // 计算QK点积和softmax
    for (int s = 0; s < src_seq_len; s++) {
      // Causal masking
      if (is_causal && s > t) {
        continue;
      }
      
      // 获取key向量
      const T* k_vec = k + k_offset + s * kv_heads * head_dim + kv_head_idx * head_dim;
      
      // 计算点积
      float dot = 0.0f;
      for (int d = 0; d < head_dim; d++) {
        dot += static_cast<float>(q_vec[d]) * static_cast<float>(k_vec[d]);
      }
      
      // 缩放
      dot *= scale;
      
      // 跟踪最大值用于数值稳定性
      if (dot > max_val) {
        max_val = dot;
      }
    }
    
    // 计算exp和sum_exp
    for (int s = 0; s < src_seq_len; s++) {
      // Causal masking
      if (is_causal && s > t) {
        continue;
      }
      
      // 获取key向量
      const T* k_vec = k + k_offset + s * kv_heads * head_dim + kv_head_idx * head_dim;
      
      // 计算点积
      float dot = 0.0f;
      for (int d = 0; d < head_dim; d++) {
        dot += static_cast<float>(q_vec[d]) * static_cast<float>(k_vec[d]);
      }
      
      // 缩放和应用max_val
      dot *= scale;
      float diff = dot - max_val;
      float exp_val = expf(diff);
      sum_exp += exp_val;
    }
    
    // 计算加权和
    for (int s = 0; s < src_seq_len; s++) {
      // Causal masking
      if (is_causal && s > t) {
        continue;
      }
      
      // 获取key和value向量
      const T* k_vec = k + k_offset + s * kv_heads * head_dim + kv_head_idx * head_dim;
      const T* v_vec = v + v_offset + s * kv_heads * head_dim + kv_head_idx * head_dim;
      
      // 计算点积
      float dot = 0.0f;
      for (int d = 0; d < head_dim; d++) {
        dot += static_cast<float>(q_vec[d]) * static_cast<float>(k_vec[d]);
      }
      
      // 缩放、应用max_val和softmax
      dot *= scale;
      float diff = dot - max_val;
      float exp_val = expf(diff);
      float weight = exp_val / sum_exp;
      
      // 加权累加value
      for (int d = 0; d < head_dim; d++) {
        attn_output[d] += static_cast<T>(static_cast<float>(v_vec[d]) * weight);
      }
    }
    
    // 写入输出
    T* o_vec = o + o_offset + t * query_heads * head_dim + q_head_idx * head_dim;
    for (int d = 0; d < head_dim; d++) {
      o_vec[d] = attn_output[d];
    }
  }
}

// half类型的特化版本，使用混合精度计算以提高精度
template <>
__global__ void flashAttentionKernel<half>(const half* q, const half* k, const half* v, half* o,
                                          int batch_size, int target_seq_len, int src_seq_len,
                                          int query_heads, int kv_heads, int head_dim,
                                          bool is_causal) {
  // 每个block处理一个batch中的一个head group
  int batch_idx = blockIdx.x / query_heads;
  int q_head_idx = blockIdx.x % query_heads;
  int kv_head_idx = q_head_idx / (query_heads / kv_heads);
  
  // 计算偏移量
  size_t q_offset = batch_idx * target_seq_len * query_heads * head_dim;
  size_t k_offset = batch_idx * src_seq_len * kv_heads * head_dim;
  size_t v_offset = k_offset;
  size_t o_offset = q_offset;
  
  // 计算缩放因子
  float scale = 1.0f / sqrtf(head_dim);
  
  // 每个线程处理一个target token
  for (int t = threadIdx.x; t < target_seq_len; t += blockDim.x) {
    // 获取当前query向量
    const half* q_vec = q + q_offset + t * query_heads * head_dim + q_head_idx * head_dim;
    
    // 计算注意力权重和输出 - 使用float进行中间计算
    float attn_output[128]; // 假设head_dim最大为128
    for (int d = 0; d < head_dim; d++) {
      attn_output[d] = 0.0f;
    }
    
    float max_val = -1e10f;
    float sum_exp = 0.0f;
    
    // 计算QK点积和softmax
    for (int s = 0; s < src_seq_len; s++) {
      // Causal masking
      if (is_causal && s > t) {
        continue;
      }
      
      // 获取key向量
      const half* k_vec = k + k_offset + s * kv_heads * head_dim + kv_head_idx * head_dim;
      
      // 计算点积 - 使用float，提高精度
      float dot = 0.0f;
      for (int d = 0; d < head_dim; d++) {
        float q_val = __half2float(q_vec[d]);
        float k_val = __half2float(k_vec[d]);
        dot += q_val * k_val;
      }
      
      // 缩放
      dot *= scale;
      
      // 跟踪最大值用于数值稳定性
      if (dot > max_val) {
        max_val = dot;
      }
    }
    
    // 计算exp和sum_exp
    for (int s = 0; s < src_seq_len; s++) {
      // Causal masking
      if (is_causal && s > t) {
        continue;
      }
      
      // 获取key向量
      const half* k_vec = k + k_offset + s * kv_heads * head_dim + kv_head_idx * head_dim;
      
      // 计算点积 - 使用float，提高精度
      float dot = 0.0f;
      for (int d = 0; d < head_dim; d++) {
        float q_val = __half2float(q_vec[d]);
        float k_val = __half2float(k_vec[d]);
        dot += q_val * k_val;
      }
      
      // 缩放和应用max_val
      dot *= scale;
      float diff = dot - max_val;
      
      // 避免溢出，提高数值稳定性
      float exp_val;
      if (diff > 88.0f) {
        exp_val = expf(88.0f);
      } else if (diff < -88.0f) {
        exp_val = 0.0f;
      } else {
        exp_val = expf(diff);
      }
      sum_exp += exp_val;
    }
    
    // 计算加权和
    for (int s = 0; s < src_seq_len; s++) {
      // Causal masking
      if (is_causal && s > t) {
        continue;
      }
      
      // 获取key和value向量
      const half* k_vec = k + k_offset + s * kv_heads * head_dim + kv_head_idx * head_dim;
      const half* v_vec = v + v_offset + s * kv_heads * head_dim + kv_head_idx * head_dim;
      
      // 计算点积 - 使用float，提高精度
      float dot = 0.0f;
      for (int d = 0; d < head_dim; d++) {
        float q_val = __half2float(q_vec[d]);
        float k_val = __half2float(k_vec[d]);
        dot += q_val * k_val;
      }
      
      // 缩放、应用max_val和softmax
      dot *= scale;
      float diff = dot - max_val;
      
      // 避免溢出，提高数值稳定性
      float exp_val;
      if (diff > 88.0f) {
        exp_val = expf(88.0f);
      } else if (diff < -88.0f) {
        exp_val = 0.0f;
      } else {
        exp_val = expf(diff);
      }
      
      // 计算权重，确保数值稳定性
      float weight = exp_val / sum_exp;
      
      // 加权累加value - 使用float，提高精度
      for (int d = 0; d < head_dim; d++) {
        float v_val = __half2float(v_vec[d]);
        attn_output[d] += v_val * weight;
      }
    }
    
    // 写入输出 - 转换回half，确保正确的类型转换
    half* o_vec = o + o_offset + t * query_heads * head_dim + q_head_idx * head_dim;
    for (int d = 0; d < head_dim; d++) {
      // 避免溢出，确保值在half类型的范围内
      float val = attn_output[d];
      if (val > 65504.0f) {
        val = 65504.0f;
      } else if (val < -65504.0f) {
        val = -65504.0f;
      }
      // 使用__float2half_rn进行四舍五入转换，提高精度
      o_vec[d] = __float2half_rn(val);
    }
  }
}

template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {
  // 计算内存大小
  size_t q_size = h_q.size() * sizeof(T);
  size_t k_size = h_k.size() * sizeof(T);
  size_t v_size = h_v.size() * sizeof(T);
  size_t o_size = h_o.size() * sizeof(T);
  
  // 分配设备内存
  T* d_q, *d_k, *d_v, *d_o;
  cudaError_t err;
  
  err = cudaMalloc(&d_q, q_size);
  if (err != cudaSuccess) return;
  
  err = cudaMalloc(&d_k, k_size);
  if (err != cudaSuccess) {
    cudaFree(d_q);
    return;
  }
  
  err = cudaMalloc(&d_v, v_size);
  if (err != cudaSuccess) {
    cudaFree(d_q);
    cudaFree(d_k);
    return;
  }
  
  err = cudaMalloc(&d_o, o_size);
  if (err != cudaSuccess) {
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    return;
  }
  
  // 计算网格和块大小
  int block_size = 128;
  int grid_size = batch_size * query_heads;
  
  // 传输数据到设备
  err = cudaMemcpy(d_q, h_q.data(), q_size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) goto cleanup;
  
  err = cudaMemcpy(d_k, h_k.data(), k_size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) goto cleanup;
  
  err = cudaMemcpy(d_v, h_v.data(), v_size, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) goto cleanup;
  
  // 启动kernel
  flashAttentionKernel<T><<<grid_size, block_size>>>(d_q, d_k, d_v, d_o,
                                                   batch_size, target_seq_len, src_seq_len,
                                                   query_heads, kv_heads, head_dim,
                                                   is_causal);
  
  // 检查kernel执行错误
  err = cudaGetLastError();
  if (err != cudaSuccess) goto cleanup;
  
  // 等待kernel执行完成
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) goto cleanup;
  
  // 传输结果回主机
  err = cudaMemcpy(h_o.data(), d_o, o_size, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) goto cleanup;
  
cleanup:
  // 释放内存
  cudaFree(d_q);
  cudaFree(d_k);
  cudaFree(d_v);
  cudaFree(d_o);
}




// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
