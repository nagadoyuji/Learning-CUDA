CUDA 提供的内置接口（内置函数）有哪些类别，以及各类接口的核心用途和使用场景，这是高效编写 CUDA 核函数、充分利用 GPU 硬件能力的关键。
CUDA 内置接口是 NVIDIA 为 GPU 架构定制的底层函数，无需额外链接库，直接内联到核函数中执行，覆盖数学运算、内存操作、线程同步、硬件特性调用等核心场景。以下按功能分类详细说明：
1. 基础数学运算函数
这是最常用的内置接口，覆盖标量 / 向量的基础算术、三角函数、指数对数等，适配float/double/half等类型，部分函数有精度 / 速度优化版本。
核心分类与示例
功能类型	常用函数（float 版）	说明	扩展（half/double 版）
基础算术	fadd/fmul/fdiv/fmod	加减乘除 / 取模（可替代运算符，更可控）	hadd/hmul（half）、dadd（double）
乘加（核心）	fmaf(a,b,c)	ab + c（单指令，比 ab+c 快）	hfma（half）、dfma（double）
三角函数	sin/cos/tan/asin/acos/atan	标准三角函数	hsin/dsin
指数 / 对数	exp/log/log2/log10	指数、自然对数、2/10 底对数	hexp/dlog
幂运算	pow/sqrt/rsqrt	幂、平方根、倒数平方根（rsqrt 更快）	hrsqrt/dsqrt
绝对值 / 符号	fabs/fsign	绝对值、符号提取	hfabs/dabs
比较运算	fmax/fmin/fclamp(x,min,max)	最大值 / 最小值 / 范围限制	hmax/dmin
代码示例：基础数学函数使用
cpp
运行
__global__ void math_kernel(float* __restrict__ d_out, const float* __restrict__ d_in, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        float x = d_in[idx];
        // 乘加（单指令，高效）
        float res1 = fmaf(x, 2.0f, 1.0f);  // x*2 + 1
        // 倒数平方根（GPU高频优化操作）
        float res2 = rsqrtf(x);            // 1/sqrt(x)，比1/sqrt(x)快
        // 范围限制（clamp到[0, 1]）
        float res3 = fclamp(res1, 0.0f, 1.0f);
        d_out[idx] = res2 * res3;
    }
}
2. 向量 / 打包类型运算函数
针对 CUDA 向量化的float2/float4、half2等打包类型，提供批量运算接口，单指令处理多个元素，提升吞吐量。
核心函数示例
打包类型	核心函数	说明
float2	v2add/v2mul/v2fma	2 个 float 同时加 / 乘 / 乘加
float4	v4add/v4mul/v4fma	4 个 float 同时加 / 乘 / 乘加
half2	hfma2/hadd2/hmul2	2 个 half 同时乘加 / 加 / 乘（Tensor Core 优化）
代码示例：float4 向量运算
cpp
运行
__global__ void vector_math_kernel(float* __restrict__ d_out, const float* __restrict__ d_in, int n) {
    int vec_idx = threadIdx.x + blockIdx.x * blockDim.x;
    int base_idx = vec_idx * 4;
    if (base_idx + 3 < n) {
        float4 in = reinterpret_cast<const float4*>(d_in)[vec_idx];
        // 4个float同时乘加：in.x*2+1, in.y*2+1, in.z*2+1, in.w*2+1
        float4 out = v4fma(in, make_float4(2.0f), make_float4(1.0f));
        reinterpret_cast<float4*>(d_out)[vec_idx] = out;
    }
}
3. 内存操作函数
用于精细控制 GPU 内存访问（全局 / 共享 / 常量内存），优化内存带宽和延迟。
核心分类与示例
内存类型	常用函数	说明
全局内存	__ldg(const T* ptr)	只读缓存加载（提升全局内存读取速度）
共享内存	__syncthreads()	块内线程同步（共享内存操作必用）
内存预取	__prefetch(const T* ptr)	提前将内存数据加载到缓存，减少延迟
内存原子	atomicAdd/atomicMax/atomicExch	全局 / 共享内存原子操作（避免竞争）
对齐加载	__ldg_gather4(const float* ptr)	非连续 4 元素加载（适配稀疏访问）
代码示例：只读缓存 + 原子操作
cpp
运行
__global__ void memory_kernel(float* __restrict__ d_out, const float* __restrict__ d_in, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        // __ldg：全局内存只读缓存加载（比直接读取快）
        float val = __ldg(&d_in[idx]);
        // 原子加：多线程安全更新d_out[0]
        atomicAdd(&d_out[0], val);
    }
}
4. 线程 / 块 / 网格控制函数
用于获取线程、块、网格的硬件信息，实现灵活的并行调度。
核心函数（必记）
函数	说明
threadIdx.x/y/z	当前线程在块内的 x/y/z 维度索引
blockIdx.x/y/z	当前块在网格内的 x/y/z 维度索引
blockDim.x/y/z	每个块的 x/y/z 维度线程数
gridDim.x/y/z	网格的 x/y/z 维度块数
warpSize	每个 warp 的线程数（固定 32）
__lane_id()	当前线程在 warp 内的索引（0-31）
代码示例：线程索引计算
cpp
运行
__global__ void thread_ctrl_kernel(float* __restrict__ d_out, int n) {
    // 2D网格+2D块的全局索引计算（图像处理常用）
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int global_idx = y * gridDim.x * blockDim.x + x;
    
    if (global_idx < n) {
        d_out[global_idx] = (float)global_idx;
    }
}
5. Warp 级协作函数（SM7.0+）
针对 GPU 的 warp（32 线程组）特性，提供高效的 warp 内数据交换、同步接口，是高性能优化的核心。
核心函数
函数	说明
__shfl_sync(mask, val, lane)	warp 内线程间数据交换（无内存开销）
__reduce_sync(mask, val)	warp 内数据归约（求和 / 最大值等）
__ballot_sync(mask, cond)	warp 内条件投票（返回 32 位掩码）
代码示例：warp 内归约求和
cpp
运行
__global__ void warp_reduce_kernel(float* __restrict__ d_out, const float* __restrict__ d_in, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    float val = 0.0f;
    if (idx < n) val = d_in[idx];
    
    // warp内求和（mask=0xffffffff表示所有32个线程参与）
    val = __reduce_sync(0xffffffff, val);
    
    // 仅warp内第0个线程写入结果
    if (__lane_id() == 0) {
        atomicAdd(d_out, val);
    }
}
6. 硬件特性 / 性能函数
用于获取 GPU 硬件信息、计时或控制执行模式。
核心函数
函数	说明
clock64()	获取 GPU 时钟周期数（核函数计时）
__nvvm_read_ptx_sreg_tid_x()	底层 PTX 线程索引（高级优化）
__noinline__	禁止函数内联（控制代码布局）
__launch_bounds__(max_threads)	限制块内最大线程数（编译器优化）
代码示例：核函数计时
cpp
运行
__global__ void timer_kernel(unsigned long long* __restrict__ d_time) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_time[0] = clock64();  // 记录起始时钟
    }
    __syncthreads();
    
    // 模拟计算
    float val = 0.0f;
    for (int i = 0; i < 1000; i++) val += sinf((float)i);
    
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_time[1] = clock64();  // 记录结束时钟
    }
}
7. 半精度（half）专用函数
针对half类型的专属运算接口（需包含<cuda_fp16.h>），前文提到的hfma/hadd等均属于此类。
核心转换 / 运算函数
函数	说明
__float2half(float)	float 转 half（运行期）
__half2float(half)	half 转 float（运行期）
__float2half2_rn	float 转 half2（编译期，四舍五入）
hfma2(half2, half2)	half2 类型乘加（Tensor Core 加速）
关键使用原则
优先用内置函数而非手动实现：如fmaf比a*b+c快（单指令），rsqrt比1/sqrt(x)快（硬件加速）；
类型匹配：half类型必须用h前缀函数，double用d前缀，避免隐式转换损耗；
架构适配：warp 级函数（__shfl_sync）仅支持 SM7.0+，编译时需指定-arch=sm_75等；
精度选择：部分函数有_rn（四舍五入）、_rz（向零取整）版本，如__float2half_rn，按需选择。
总结
CUDA 内置接口按功能可分为数学运算、向量运算、内存操作、线程控制、warp 级协作、硬件特性、half 专用 7 大类，覆盖核函数开发全场景；
高频核心接口：fmaf（乘加）、__ldg（只读缓存）、__syncthreads（块同步）、__shfl_sync（warp 数据交换）、atomicAdd（原子操作）；
使用关键：匹配数据类型（float/half/double）、适配 GPU 架构、优先选择硬件加速的内置函数（如 rsqrt、hfma2）以最大化性能。