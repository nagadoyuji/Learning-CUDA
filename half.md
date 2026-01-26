 CUDA 中对half（半精度浮点数，FP16）数据类型的处理方式，包括定义、运算、内存操作和硬件支持等核心内容，这是 CUDA 优化显存占用和计算效率的重要知识点。
1. half数据类型的基础认知
half是 16 位浮点数（FP16），相比 32 位float（FP32）：
显存占用减少 50%，可提升内存带宽利用率；
计算吞吐量更高（部分 GPU 架构），但精度较低（有效数字约 3-4 位）；
CUDA 中half的定义在<cuda_fp16.h>头文件中（CUDA 7.0 + 支持）。
2. CUDA 处理half的核心方式
（1）数据类型定义与转换
首先需包含头文件，然后可直接定义half变量，或在float/half间转换：
cpp
运行
#include <cuda_runtime.h>
#include <cuda_fp16.h>  // 必须包含，否则无法识别half类型

// 主机端/设备端通用的half操作示例
__global__ void half_operation_kernel(half* __restrict__ d_out, const half* __restrict__ d_in, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        // 1. 直接定义half常量
        const half h_const = __float2half(2.0f);  // float转half（编译期/运行期）
        
        // 2. half与half运算
        half h_val = d_in[idx] * h_const;
        
        // 3. half转float（如需高精度计算）
        float f_val = __half2float(h_val);
        f_val += 1.0f;
        
        // 4. float转回half，写入输出
        d_out[idx] = __float2half(f_val);
    }
}

// 主机端调用示例
int main() {
    const int n = 1024;
    size_t size = n * sizeof(half);
    
    // 分配显存（half类型的内存分配与float一致）
    half *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    
    // 主机端初始化half数据（先转float再转half）
    half* h_in = (half*)malloc(size);
    for (int i = 0; i < n; i++) {
        h_in[i] = __float2half((float)i);
    }
    
    // 主机→设备拷贝（half的内存拷贝与普通类型一致）
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    
    // 启动核函数
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    half_operation_kernel<<<grid, block>>>(d_out, d_in, n);
    
    // 释放内存
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    return 0;
}
（2）half的运算函数（核心）
CUDA 不支持half的直接算术运算符（如+/*），需使用内置函数：
功能	内置函数示例	说明
float 转 half	__float2half(float x)	运行期转换
half 转 float	__half2float(half x)	运行期转换
编译期 float 转 half	__float2half_rn(float x)	四舍五入，编译期优化
half 加法	hadd(half a, half b)	a + b
half 乘法	hmul(half a, half b)	a * b
half 乘加（核心）	hfma(half a, half b, half c)	a*b + c（单指令，高效）
（3）硬件支持与架构差异
不同 GPU 架构对half的支持程度不同，这是处理half的关键：
Kepler (SM 3.x)：仅支持half的内存存储 / 加载，运算需先转float；
Maxwell (SM 5.x)：部分支持half运算，但效率一般；
Pascal (SM 6.x)：全面支持half运算（FP16 指令）；
Volta/Turing/Ampere (SM 7.x/8.x/9.x)：
支持half的高效运算（Tensor Core 加速）；
支持__half2（两个half打包），进一步提升吞吐量；
Ampere 及以上还支持 BF16（脑浮点数，兼容 FP16）。
（4）half的批量操作（__half2）
为提升效率，CUDA 提供__half2类型（两个half打包成 32 位），可一次操作两个half：
cpp
运行
__global__ void half2_operation_kernel(__half2* __restrict__ d_out, const __half2* __restrict__ d_in, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n/2) {  // 一次处理2个元素，n需为偶数
        // __half2的乘加操作，一次处理两个half
        __half2 h2_val = hfma2(d_in[idx], __float2half2_rn(2.0f), __float2half2_rn(1.0f));
        d_out[idx] = h2_val;
    }
}
（5）Tensor Core 加速（高性能场景）
在 Volta 及以上架构中，half是 Tensor Core 的核心输入类型，可通过 CUDA 的cublasLt或warp-level函数调用 Tensor Core，实现矩阵乘法的超高吞吐量：
cpp
运行
// 示例：使用cublas进行FP16矩阵乘法（依赖cublas库）
#include <cublas_v2.h>

void half_matmul(half* d_C, const half* d_A, const half* d_B, int m, int n, int k) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    const half alpha = __float2half(1.0f);
    const half beta = __float2half(0.0f);
    
    // cublasHgemm：FP16矩阵乘法（H代表half）
    cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                m, n, k,
                &alpha,
                d_A, m,
                d_B, k,
                &beta,
                d_C, m);
    
    cublasDestroy(handle);
}
3. 注意事项
精度问题：half的精度远低于float，适合对精度要求不高的场景（如图像处理、深度学习），避免用于高精度科学计算；
核函数编译：需确保 CUDA 编译器版本≥7.0，且编译时指定正确的架构（如-arch=sm_75）；
主机端限制：CPU 通常不原生支持half运算，主机端操作half需先转float。
总结
CUDA 中处理half需包含<cuda_fp16.h>，通过内置函数实现half与float的转换及算术运算，不支持直接使用+/*等运算符；
不同 GPU 架构对half的支持差异大，Volta 及以上架构可通过 Tensor Core 实现half的超高吞吐量运算；
常用优化手段：使用__half2批量操作、结合__restrict__优化内存访问、利用 cublas 等库调用硬件加速。