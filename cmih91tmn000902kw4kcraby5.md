---
title: "Optimizing GEMM: GPU Architecture Essentials"
seoTitle: "Mastering GEMM: Key GPU Architecture Insights"
datePublished: Thu Nov 27 2025 09:46:58 GMT+0000 (Coordinated Universal Time)
cuid: cmih91tmn000902kw4kcraby5
slug: optimizing-gemm-gpu-architecture-essentials
ogImage: https://cdn.hashnode.com/res/hashnode/image/upload/v1764236657192/ce9f6164-1df0-4c99-9f0d-ff413a6440d2.jpeg
tags: ai, ml, cuda, gpu-nvidia-amd, cudathread, gemm

---

Every time you ask ChatGPT a question, get a movie recommendation on Netflix, or watch your phone recognize faces in photos, billions of matrix multiplications are happening behind the scenes. This fundamental mathematical operation has become the computational backbone of modern artificial intelligence.

**GEMM (General Matrix Multiplication)** is the technical term that we will be using often in this blog for matrix multiplication operations. GEMM is omnipresent in machine learning, when a neural network predicts house prices using linear regression, it's performing matrix multiplications to combine features and weights. When a recurrent neural network analyzes product review sentiment, GEMM operations process sequential data through hidden states. When a convolutional neural network suggest captions for Instagram photos, countless matrix operations extract and combine visual features.

The AI revolution from classic machine learning (pre-2010) through deep learning (2010s) to today's generative AI era has been enabled by our ability to perform these matrix operations at scale. Modern large language models like the ones powering Gemini, ChatGPT, and Claude require a large number of matrix operations for both **training** and **inference**. For instance A single forward pass through a transformer architecture involves matrix multiplications with dimensions in the thousands, repeated across multiple layers number of layers depends on the size and architecture of the model, ex:- Llama model frameworks like Pytorch & Tensorflow abstract these details from us being in IT(Software & Hardware) we come across abstractions every day, while abstractions help us to focus on doing what we need to they hide lot of details which can lead to issues ex:- ORM(Object Relational Mapping) frameworks like Hibernate abstracts how the queries are being constructed and sent to underlying database, often this becomes bottleneck for performance of the applications as developers are abstracted from internals of the framework. I was curious to understand what goes under the hood when we are using **torch.matmul() , model(inputs) - Forward Pass & loss.bacward() - Backward Pass** during modelling & inference process of model , my curiosity led me to this topic of understanding the nuts and bolts framework that powers the ML models , when I began peeling back the layers of abstraction in ML frameworks like PyTorch and TensorFlow, I discovered a fascinating world of optimization and admired the people & efforts behind these optimizations. The journey from a simple matrix multiplication to GPU-accelerated GEMM operations involves critical decisions at different layers of the whole infrastructure memory hierarchies, thread management, communication protocols & network bandwidth etc…

This curiosity about how GEMM operations work led me to explore CUDA architecture and the specialized libraries that PyTorch calls behind the scenes. The topic is inherently complex, and I approached it without any CUDA background while trying to understand model training architectures. I wanted to create the resource I wish I had—a simple, approachable guide for anyone interested in GPU computing but hesitant due to the steep learning curve. As I am visual learne and like to understand concepts through analogies, hence I've added visualizations and analogies wherever possible to make these concepts more intuitive.

In this blog series, I'll take you through that same journey—from naive matrix multiplication to highly optimized GPU implementations. We'll explore why simple approaches fail at scale, how GEMM leverages GPU memory hierarchies\*\*,\*\* and how hardware and software innovations are continuously incorporated into the libraries and frameworks we use for building ML workloads.To follow along, we need basic understanding of GPU architecture and terminology. While this topic is vast,we'll keep the scope focused on what's essential for understanding GEMM operations.

This is just the tip of the iceberg, and as they say, the devil is in the details. Understanding these fundamentals helps us appreciate the sophistication of GPU computing and recognize the engineering effort that goes on behind the scenes—in both hardware and software. This is work that most of us take for granted every time we call a simple function like **torch.matmul** in PyTorch.

In PyTorch, matrix multiplication is just one line of code. Behind that single line lies an entire world of optimizations: C++ dispatch, cuBLAS libraries, hand-tuned CUDA kernels, and billions of coordinated hardware operations.

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1763768893346/79e8f1d5-1b99-4911-bc1a-fbe8b5f9eb21.png align="center")

![Layers of GPU Computing Abstraction ](https://cdn.hashnode.com/res/hashnode/image/upload/v1761378556588/424d510d-a8f6-44dd-aeb0-70b0c4751db5.png align="center")

**Image-1: Layers of GPU Computing Abstraction**  
**From CUDA cores at the hardware level to PyTorch/TensorFlow at the application level**

# GPU Architecture

Like the universe exist in two dimensions (**Time** & **Space**), computing systems—whether CPU or GPU—has two fundamental dimensions: **Memory and Execution**. All applications being built scales across two these two dimensions.

If an application requires more execution capabilities through computing units, it's called a **compute-bound application**. One example would be computing hash values for data encryption using algorithms like RSA256. On the other dimension, applications that need less computing power but require more memory and I/O operations are termed **memory-bound applications**. Some applications lie at the intersection of both, needing significant computing and memory resources. This distinction is crucial to understand because we'll encounter it multiple times throughout our discussions. Think of this like fractals - recurring pattern that appears at different levels of system design.

There is often confusion between working of CPUs and GPUs, but they address fundamentally different problems—one size does not fit all. For instance, we can classify databases as OLTP (transaction processing) or OLAP (analytical processing). While both solve different problems, one can't replace the other. The same applies to CPUs and GPUs. CPU computing units are designed to handle complex problems with limited parallelism, while GPU computing units are designed to scale massively with the capability to solve problems across different domains. This space is currently evolving rapidly—NVIDIA introduces new architectures every year during their GTC sessions, reflecting the dynamic nature of GPU computing.

To optimize matrix multiplication at the scale required for ML/AI workloads, we need a better understanding of GPU architecture, processing units, and memory hierarchies—the two cornerstones of any GPU workload. **Think of it like a manufacturing plant producing cars:** it has critical resources like technicians, tools, raw materials, and storage space for parts. GPUs also have critical resources, some of which are highlighted below. While each NVIDIA GPU architecture has numerous configurations, we're focusing on a few important ones relevant to our context. You can find full architecture details for [Ampere](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)

## Processing Units (The Skilled Technicians):

* **Threads:** Individual workers
    
* **Warps:** Small coordinated group of 32 threads
    
* **Thread Blocks:** Teams of workers assigned to complete a task
    
* **CUDA Cores and Tensor Cores:** Individual workstations that serve different purpose in Workshop
    
* **Streaming Multiprocessors (SMs):** The main production lines where work gets done, think of each SM as workshop
    

## Memory Hierarchy (storage for parts and materials):

* **Registers:** immediate tools required for Technician (fastest, most limited)
    
* **Shared Memory/L1 Cache:** Workshop floor storage (fast but limited)
    
* **L2 Cache:** Regional storage - Shared across all SMs
    
* **Global RAM:** The main warehouse (large but distant)
    

<iframe src="https://ghpages.vvnasantosh.net/visualizations/gpu_memory_process_hierarchy.html" width="130%" height="1700px" style="border:none;zoom:0.6;max-width:100%;display:block;margin:0">
</iframe>

**Image 2: A100 GPU Architecture - Memory and Processing Units Overview**

<iframe src="https://ghpages.vvnasantosh.net/visualizations/memory_processing_mapping_notebook.html" width="130%" height="2400px" style="border:none;zoom:0.6;max-width:100%;display:block;margin:0">
</iframe>

**Image 3: Memory ↔ Processing Unit Mapping**

## Resource Constraints:

Just as a manufacturing plant has limited skilled technicians and shop floor space, GPUs have finite resources. For NVIDIA's A100 (Ampere architecture), we must be aware of these constraints before writing CUDA functions or kernels to achieve our objectives:

* **108 Streaming Multiprocessors**
    
* **2,048 maximum threads per SM**
    
* **64 maximum Warps per SM**
    
* **1024 Threads per Block**
    
* **64 FP32 cores + 32 FP64 cores + 4 Tensor cores per SM**
    
* **Total Threads =** `2048 Threads per SM * 108 SM = 221,184 Threads per GPU`
    
* **Global (HBM) memory:** `40GB`
    
* **L2 Cache:** `40MB`
    
* **Shared-memory+L1(Shared across Thread Block):** `192KB`
    
* **Maximum Shared-Memory Configurable:** `164KB`
    
* **Registers:** `64K 32-bit Registers per SM, 255 Registers per Thread`
    

# CUDA Programming Terminology:

## Kernel:

A kernel is a function that executes on the GPU to achieve our computational objective. In our manufacturing analogy, it's the instruction manual that guides each technician on the steps to follow for their specific task.

For matrix multiplication, our kernel would contain the instructions for computing `C[i][j] = A[i][k] * B[k][j]` for assigned matrix elements, The time taken for model training and inference is directly dependent on kernel performance. One particularly important kernel that has optimized the attention process in recent times is FlashAttention.

Though the GPU performs the actual computation, the CPU orchestrates the process. Before the kernel is launched on the GPU, the CPU copies the data to GPU HBM Memory (Global Memory).

<div data-node-type="callout">
<div data-node-type="callout-emoji">💡</div>
<div data-node-type="callout-text"><strong>Analogy</strong>: <strong>The way I like to think about it is like maritime logistics where cargo is transferred to smaller vessels before reaching the main ship</strong></div>
</div>

* Declare Host Variables (Host = CPU)
    
* Declare Device Variables (Device = GPU)
    
* Allocate Device Memory (Memory to Store the Data Required in Global Memory HBM)
    
* Copy Host → Device (Transfer input data to GPU)
    
* Launch Kernel (Execute Main Function)
    
* Copy Device → Host (Retrieve results from GPU to CPU)
    
* Free Memory(Clean up alloacted memory)
    

## Thread Hierarchy

To perform computations on the GPU, we write **kernels**. Whether developing custom kernels or using existing ones, each kernel requires **processing units (CUDA/Tensor cores)** and **memory resources** for execution. These resources are organized according to configurations specified during kernel launch.

The resource allocation involves a two-tier responsibility model:

* **Developer-controlled:** Grid dimensions, Block dimensions, shared memory usage, Register usage
    
* **Hardware/scheduler-controlled:** Warp scheduling, Streaming Multiprocessor assignment, Instruction scheduling
    

Developers don't specify individual threads directly. Instead, the number of required threads is expressed through Grid and Thread Blocks configuration, which uses a hierarchical approach: **Grid → Thread Blocks → Warps → Threads**. Both Grids and Blocks configurations can be expressed using 1D, 2D, and 3D dimensions to address problem statements belonging to different domains.

### **Warp**

A critical aspect of CUDA's execution model is the Warp. This is not controlled by developers. While we specify the grid and block dimensions, we don't specify the warp size, which is controlled by schedulers and is constant at 32 threads per warp executed simultaneously.

An important note about warps is that even if we specify just 1 thread per block, the warp scheduler still allocates a full warp of 32 threads, with 31 threads remaining unused. This significantly impacts SM utilization and should be considered when designing kernel launch configurations. The table below shows the impact of thread block size on warp utilization:

| **Threads per Block** | **Warps Used** | **Warp Utilization** |
| --- | --- | --- |
| 16 Threads | 1 Warp | 50% (16/32) |
| 32 Threads | 1 Warp | 100% (32/32) |
| 48 Threads | 2 Warps | 75% (48/64) |
| 64 Threads | 2 Warps | 100% (64/64) |
| 96 Threads | 3 Warps | 100% (96/96) |

### Grid Configuration

**Grid Dimension (gridDim):** The grid dimension specifies the number of blocks in the grid. It determines how many blocks are launched to execute the kernel.

### Thread Block Configuration

**Block Dimension (blockDim):** The block dimension specifies the number of threads within a single block.

### Kernel Launch

`kernel<<<gridDim, blockDim>>>(parameters)`

* Thread blocks per grid: 16 × 16 = 256 thread blocks
    
* Threads per block: 16 × 16 = 256 threads
    
* Warps per Block : 256 / 32 = 8 Warps per Block
    
* Total Warps : 8 \* 256 = 2048 Warps
    
* Total threads: 256 × 256 = 65,536 threads
    

This is one possible configuration for processing a 256×256 matrix where each thread handles one matrix element(This is not Optimal configuration)

![](https://cdn.hashnode.com/res/hashnode/image/upload/v1761400350581/b22d9645-aa7a-4f97-9946-2f8054727187.png align="center")

**Image 4: Demonstrating 1D Grid and Block Dimensions**

**Note: For more detailed information on CUDA Thread Hierarchy. Please refer to the official CUDA programming guide** **CUDA** [**documentation**](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-hierarchy)

## SIMT (Single Instruction Multiple Thread)

SIMT is the execution model used by CUDA kernels. Consider multiplying matrix A of size (256, 256) by a constant C. The instruction `A[i][j] = C * A[i][j]` remains the same for all iterations except for the index values. This nested loop can be converted to execute in parallel, where each thread calculates one element on the GPU.

This is achieved by issuing the same instruction to a group of 32 threads (a warp), as discussed earlier. Though threads are the fundamental building blocks for GPU processing, instructions are issued at the warp level. Hence, when an instruction is issued, it's executed by all active threads in the warp.

**Note: In modern architectures, there are modifications where collections of threads work together on computations (like Thread Block Clusters), but this basic SIMT model covers the fundamental concept.**

```python
for i in range(256):
    for j in range(256):
        A[i][j] = C * A[i][j]
```

### Logical vs Physical Representation

In the above example of multiplying the matrix by constant C, the indexing representation is a logical representation. Internally, the GPU operates on linear indexes, so the data must be converted to a linear index. This conversion can be done explicitly by the developer or handled automatically by the compiler. When the code is compiled, it converts to the respective linear index. Below is a visual representation of how the logical representation can be converted to physical representation:

* Linear thread IDs within each warp (32 consecutive threads)
    
* Linear memory addresses in global memory
    
* Sequential instruction execution within warps
    

Thread indexes are linear. When accessing matrix elements, we can use either 2D or linear indexing. The image below shows how 2D indexes can be converted to linear indexing.

<iframe src="https://ghpages.vvnasantosh.net/visualizations/matrix_2d_to_1d_notebook" width="160%" height="1280px" style="border:none;zoom:0.6;max-width:100%;display:block;margin:0">
</iframe>

**Image 5: CUDA 2D to 1D Matrix Indexing**  

<iframe src="https://ghpages.vvnasantosh.net/visualizations/matrix_3d_to_1d_notebook" width="140%" height="1900px" style="border:none;zoom:0.6;max-width:100%;display:block;margin:0">
</iframe>

**Image 6: CUDA 3D to 1D Matrix Indexing**

## Memory Coalescing : Efficient Data Access Patterns

The way matrix elements are stored in memory significantly impacts kernel performance. Understanding how threads access memory is crucial for achieving optimal GPU performance in matrix operations to avoid Kernels starving for data due to slow memory access.

GPU memory operations can be categorized into three phases:

1. **Load - Transfer data from global memory to registers directly or via shared memory**
    
2. **Compute - Execute the actual mathematical operations on data in registers**
    
3. **Store operations - Write results back to global memory**
    

The important point to consider is that memory bandwidth is bottleneck in few GPU kernels not computational power , for such kernels optimizing memory access patterns improves the performance of the kernel.

**Memory coalescing** occurs when threads within the same warp access consecutive memory addresses. When this happens, the GPU can combine multiple memory requests into fewer, wider memory transactions, dramatically improving bandwidth utilization.

**Note**:- **Coalescing is determined by the access pattern of the 32 threads within a single Warp**

### Memory Storage Layouts

By default, C++ stores matrices in **row-major layout** while Fortran uses **column-major layout**. This is analogous to how databases can be organized as row-oriented (traditional RDBMS) or column-oriented (analytical databases) based on storage patterns.

Whether we are using CPU or GPU for computation, data first needs to be brought to execution units (ALU in CPU or CUDA Cores in GPU), which is accomplished using load instructions. Upon load instruction, data is moved from Global Memory to Registers either directly or through Shared Memory. Registers are high-speed memory.

To keep it simple for our discussion, each load instruction can only access a certain number of bytes in one cycle. When all of our threads in a warp are accessing adjacent locations, one load instruction can fetch all the inputs required for the threads in the warp to be processed. If the threads of the same warp are accessing elements that are distant, this leads to multiple load instructions from Global Memory, which is a time-consuming process. Transfer of data from Global Memory is constrained by bandwidth. CUDA cores complete their instructions and wait for the data while the load instruction is still fetching data from Global Memory. Hence, it is paramount to ensure that data should be arranged such that threads in a warp access consecutive elements for their respective operations.

### Analogy:

In our manufacturing analogy, if all the technicians are working on the same part of a car, they all require parts from the same bin or nearby bins. One supply truck can make a single trip to load all parts from consecutive bins. However, if each technician is working on a different part of the car that requires parts from different bins, this makes it difficult to load all parts from different bins efficiently. This might require multiple iterations to bring parts required for all technicians, which impacts the overall execution of the manufacturing process.

Let us go through some examples of coalesced and non-coalesced patterns to understand this better.

### Coalesced Memory Access Pattern

Consider matrix multiplication for matrices **A=(1024×1024)**, **B=(1024×1024)**, **C=(1024×1024)** using a configuration that results in **optimal coalesced memory access**.

With `blockDim(32,1)`, each warp contains 32 threads that access **32 consecutive memory locations** within the same matrix row.

##### **Visualizations**

<iframe src="https://ghpages.vvnasantosh.net/visualizations/coalesced_grid_layout.html" width="160%" height="1400px" style="border:none;zoom:0.8;max-width:100%;display:block;margin:0">
</iframe>

**Image 7: Coalesced Grid Layout**

<iframe src="https://ghpages.vvnasantosh.net/visualizations/coalesced_block_layout.html" width="160%" height="1100px" style="border:none;zoom:0.7;max-width:100%;display:block;margin:0">
</iframe>

**Image 8: Thread Organization Within a Single Block**

<iframe src="https://ghpages.vvnasantosh.net/visualizations/coalesced_warp_patterns.html" width="160%" height="2300px" style="border:none;zoom:0.9;max-width:100%;display:block;margin:0">
</iframe>

##### **Thread Block (0,0) - Warp Access Pattern:**

**Thread 0:**

```plaintext
Parameters: 
----------
blockIdx.x=0, blockIdx.y=0 , threadIdx.x=0, threadIdx.y=0

Calculation:
-------------
row = 0 * 1 + 0 = 0
col = 0 * 32 + 0 = 0
idx = 0 * 1024 + 0 = 0
```

**Thread 1:**

```plaintext
Parameters: 
------------
blockIdx.x=0, blockIdx.y=0, threadIdx.x=1, threadIdx.y=0

Calculation:
------------
row = 0 * 1 + 0 = 0
col = 0 * 32 + 1 = 1
idx = 0 * 1024 + 1 = 1
```

**Thread 2:**

```plaintext
Parameters: 
-------------
blockIdx.x=0, blockIdx.y=0, threadIdx.x=2, threadIdx.y=0

Calculation:
-----------
row = 0 * 1 + 0 = 0
col = 0 * 32 + 2 = 2
idx = 0 * 1024 + 2 = 2
```

**…continuing pattern**

**Thread 31:**

```plaintext
Parameters: 
----------
blockIdx.x=0, blockIdx.y=0, threadIdx.x=31, threadIdx.y=0

Calculation:
----------
row = 0 * 1 + 0 = 0
col = 0 * 32 + 31 = 31
idx = 0 * 1024 + 31 = 31
```

<div data-node-type="callout">
<div data-node-type="callout-emoji">💡</div>
<div data-node-type="callout-text"><strong>Perfect Coalescing:</strong> All 32 threads access consecutive indices <strong>0, 1, 2, …, 31</strong></div>
</div>

##### **Thread Block (31,0) - Final Block in Row 0**

**Thread 0:**

```plaintext
Parameters: 
---------
blockIdx.x=31, blockIdx.y=0, threadIdx.x=0, threadIdx.y=0

Calculation:
----------
row = 0 * 1 + 0 = 0
col = 31 * 32 + 0 = 992
idx = 0 * 1024 + 992 = 992
```

**Thread 1:**

```plaintext
Parameters: 
------------
blockIdx.x=31, blockIdx.y=0, threadIdx.x=1, threadIdx.y=0

Calculation:
------------
row = 0 * 1 + 0 = 0
col = 31 * 32 + 1 = 993
idx = 0 * 1024 + 993 = 993
```

**…continuing pattern**

**Thread 31:**

```plaintext
Parameters: 
----------
blockIdx.x=31, blockIdx.y=0, threadIdx.x=31, threadIdx.y=0

Calculation:
------------
row = 0 * 1 + 0 = 0
col = 31 * 32 + 31 = 1023
idx = 0 * 1024 + 1023 = 1023
```

<div data-node-type="callout">
<div data-node-type="callout-emoji">💡</div>
<div data-node-type="callout-text"><strong>Perfect Coalescing:</strong> All 32 threads access consecutive indices <strong>992, 993, 994, …, 1023</strong></div>
</div>

#### Access Pattern(Block Wise)

| **Thread Block** | **Indices Accessed** | **Pattern** |
| --- | --- | --- |
| Block (0,0) | 0, 1, 2, …, 31 | Consecutive |
| Block (1,0) | 32, 33, 34, …, 63 | Consecutive |
| Block (2,0) | 64, 65, 66, …, 95 | Consecutive |
| ….. |  |  |
| Block(31,0) | 992, 993, 994, …, 1023 | Consecutive |

#### Key Advantages

1. **No Memory Gaps:** Every thread in a warp accesses consecutive memory addresses
    
2. **Single Memory Transaction:** Each warp requires only 1 memory transaction
    
3. **Optimal Bandwidth Utilization:** 100% of memory bandwidth is effectively used
    
4. **Maximum Performance:** Optimal memory access pattern for GPU architecture
    

#### Warp-Level Analysis

Each thread block contains exactly **32 threads which is equivalent to 1 warp**, each block has at most 1 Warp for the configuration

| Thread Block | Warp | Index Access Pattern |
| --- | --- | --- |
| Block(0,0) | Warp-0 | 0-31 |
| Block(1,0) | Warp-0 | 32-63 |
| Block(2,0) | Warp-0 | 64-95 |
| ….. |  |  |
| Block(31,0) | Warp-0 | 992-1023 |

<div data-node-type="callout">
<div data-node-type="callout-emoji">💡</div>
<div data-node-type="callout-text">The reason for perfect Coealscing in our case is <code>blockDim(32,1)</code> aligns perfectly with GPU Warp architecture, ensuring each warp accesses a single contiguous block of memory within the same matrix row.</div>
</div>

### Non-Coalesced Memory Access Pattern

Consider matrix multiplication for matrices **A=(1024×1024)**, **B=(1024×1024)**, **C=(1024×1024)** using a configuration that results in **non-coalesced memory access**.

#### Configuration Parameters

```plaintext
gridDim = (64,64)   => 64 Thread Blocks in X dimension, 64 Thread Blocks in Y dimension
blockDim = (16,16)  => 16 Threads in X dimension, 16 Threads in Y dimension

gridDim.x = 64      gridDim.y = 64  
blockDim.x = 16     blockDim.y = 16

threadIdx.x  0-15    blockIdx.x  0-63
threadIdx.y  0-15    blockIdx.y  0-63
```

* **Thread Blocks:** 64 × 64 = 4,096 blocks
    
* **Threads per Block:** 16 × 16 = 256 threads
    
* **Warps per Block:** 256 ÷ 32 = 8 warps
    
* **Threads:** 1,048,576 threads (perfect for 1024×1024 matrix)
    

#### Thread Block Organization (blockIdx.x, blockIdx.y) :

→ → → → → → → → **blockIdx.x** ← ← ← ← ← ← ← ← ←

|  | 0 | 1 | 63 |
| --- | --- | --- | --- |
| 0 | Thread Block(0,0) | Thread Block(1,0) | Thread Block(63,0) |
| 1 | Thread Block(0,1) | Thread Block(1,1) | Thread Block(63,1) |
| 2 | Thread Block(0,2) | Thread Block(1,2) | Thread Block(63,2) |
| …….. |  | …….. | …….. |
| 63 | Thread Block(0,63) | Thread Block(1,63) | Thread Block(63,63) |

#### Thread Organization within Each Block

|  | 0 | 1 | 2 | …….. | 15 |
| --- | --- | --- | --- | --- | --- |
| 0 | threadIdx(0,0) | threadIdx(1,0) | threadIdx(2,0) | …….. | threadIdx(15,0) |
| 1 | threadIdx(0,1) | threadIdx(1,1) | threadIdx(2,1) | …….. | threadIdx(15,1) |
| 2 | threadIdx(0,2) | threadIdx(1,2) | threadIdx(2,2) | …….. | threadIdx(15,2) |
| …….. | …….. | …….. | …….. | …….. | …….. |
| 15 | threadIdx(0,15) | threadIdx(1,15) | threadIdx(2,15) | …….. | threadIdx(15,15) |

```plaintext
row = blockIdx.y * blockDim.y + threadIdx.y
col = blockIdx.x * blockDim.x + threadIdx.x
idx = row * matrix_width + col    (where matrix_width = 1024)
```

#### Visualizations

<iframe src="https://ghpages.vvnasantosh.net/visualizations/non_coalesced_grid_layout.html" width="100%" height="1300px" style="border:none;zoom:0.9;max-width:100%;display:block;margin:0">
</iframe>

**Image 9: Non-Coalesced Grid Layout**

<iframe src="https://ghpages.vvnasantosh.net/visualizations/non_coalesced_block_layout.html" width="160%" height="2300px" style="border:none;zoom:0.9;max-width:100%;display:block;margin:0">
</iframe>

**Image 10: Warp Distribution in 2D Thread Block Layout**

<iframe src="https://ghpages.vvnasantosh.net/visualizations/non_coalesced_warp_patterns.html" width="160%" height="2500px" style="border:none;zoom:0.9;max-width:100%;display:block;margin:0">
</iframe>

**Image 11: Non-Coalesced Analysis - Warps**

#### Problem:

With `blockDim(16,16)`, each warp contains 32 consecutive threads that span **2 matrix rows**, creating large memory gaps.

##### **Warp 0 (Threads 0-31) in Thread Block (0,0):**

**First 16 threads (threadIdx.y = 0):**

```plaintext
Parameters:
-----------
blockIdx.x=0, blockIdx.y=0, threadIdx.x=0-15, threadIdx.y=0

Calculation:
----------
row = 0 * 16 + 0 = 0
col = 0 * 16 + (0-15) = 0-15
idx = 0 * 1024 + (0-15) = 0-15

Memory indices accessed: 0, 1, 2, 3, ..., 14, 15
```

**Next 16 threads (threadIdx.y = 1):**

```plaintext
Parameters: 
-----------
blockIdx.x=0, blockIdx.y=0, threadIdx.x=0-15, threadIdx.y=1

Calculation:
-----------
row = 0 * 16 + 1 = 1
col = 0 * 16 + (0-15) = 0-15
idx = 1 * 1024 + (0-15) = 1024-1039

Memory indices accessed: 1024, 1025, 1026, ..., 1038, 1039
```

<div data-node-type="callout">
<div data-node-type="callout-emoji">💡</div>
<div data-node-type="callout-text"><strong>Gap Analysis:</strong> <strong>1009 elements</strong> between indices 15 and 1024!</div>
</div>

##### **Final Warp Example - Warp 7 (Threads 224-255):**

**Threads 224-239 (threadIdx.y = 14):**

```plaintext
Parameters: 
----------
blockIdx.x=0, blockIdx.y=0, threadIdx.x=0-15, threadIdx.y=14

Calculation:
-----------
row = 0 * 16 + 14 = 14
col = 0 * 16 + (0-15) = 0-15
idx = 14 * 1024 + (0-15) = 14336-14351
```

**Threads 240-255 (threadIdx.y = 15):**

```plaintext
Parameters: 
----------
blockIdx.x=0, blockIdx.y=0, threadIdx.x=0-15, threadIdx.y=15

Calculation:
-----------
row = 0 * 16 + 15 = 15
col = 0 * 16 + (0-15) = 0-15
idx = 15 * 1024 + (0-15) = 15360-15375
```

#### Summary: Non-Coalesced Access Pattern

| **Warp** | Indices (First 16 Threads) | Indices (Next 16 Threads) | Coalesced |
| --- | --- | --- | --- |
| Warp-0 | 0-15 (row 0) | 1024-1039 (row 1) | No |
| Warp-1 | 2048-2063 (row 2) | 3072-3087 (row 3) | No |
| Warp 2 | 4096-4111 (row 4) | 5120-5135 (row 5) | No |
|  |  |  |  |
| Warp 7 | 14336-14351 (row 14) | 15360-15375 (row 15) | No |

<div data-node-type="callout">
<div data-node-type="callout-emoji">💡</div>
<div data-node-type="callout-text">Issue with <code>blockDim(16,16)</code> : It forces warps to span multiple matrix rows, breaking the consecutive memory access pattern required for optimal coalescing.</div>
</div>

## Compute vs Memory Bound Kernels

GPU kernels can be classified as either **memory bound** or **compute bound** based on their primary performance bottleneck. Understanding this classification is crucial for choosing the right optimization approach.

Like every computing system, GPUs have two fundamental dimensions: **execution capability** and **memory bandwidth**. Every system has scaling limits, and to improve performance, we can optimize along either dimension until we reach the maximum limits of available resources.

### Key Classification Metric: Arithmetic Intensity

<div data-node-type="callout">
<div data-node-type="callout-emoji">💡</div>
<div data-node-type="callout-text"><strong>Arithmetic Intensity = FLOPs (Floating Point Operations) / Bytes Accessed Peak Arithmetic Intensity for A100 = 9.75 FLOP/byte</strong> <strong>Above 9.75 FLOP/byte: Compute-bound(can potentially use full 19.5 TFLOPS) Below 9.75 FLOP/byte: Memory-bound (limited by 2.0 TB/s bandwidth)</strong></div>
</div>

### Kernel Classification Definitions

### Compute-Bound Kernels

**Definition**: A kernel is compute-bound when its performance is limited by the speed of its mathematical operations. The GPU's compute units (CUDA Cores, Tensor Cores) are the primary bottleneck, and their utilization is the limiting factor.

**Optimization Strategies:**

To optimize compute-bound kernels, the focus is on maximizing the efficiency of mathematical operations. This involves:

* **Ensuring High Occupancy**: Launching enough threads to fully saturate the GPU’s Streaming Multiprocessors (SMs) and keep the compute units busy.
    
* **Leveraging Specialized Hardware:** Using specialized cores like [Tensor Cores](https://www.digitalocean.com/community/tutorials/understanding-tensor-cores) for matrix multiplication.
    
* **Multi-GPU Scaling:** For workloads that exceed a single GPU’s capacity, distributing the work across multiple GPUs can increase overall throughput.
    

### Memory Bound Kernels

**Definition**: A kernel is memory-bound when its performance is limited by the speed at which data can be transferred to and from the GPU's memory. The primary bottleneck is memory bandwidth.

**Key Indicators:**

* Low Compute Utilization (cores are idle, waiting for data).
    
* Poor cache hit rates
    
* High memory access latency
    
* Memory access inefficiencies (poor coalescing, cache thrashing)
    

**Optimization Strategies:**

To optimize memory-bound kernels, the focus is on reducing the amount of data transferred from high-latency memory and improving access patterns. This is achieved by:

* **Maximize data reuse**: Keep frequently accessed data in faster memory levels
    
* **Optimize access patterns**: Ensuring memory accesses are coalesced to reduce the number of memory transactions
    
* **Use memory hierarchy efficiently**: Thread Registers → L1 Cache/Shared Memory → L2 Cache → Global Memory (HBM2)
    
* **Use lower precision**: Mixed precision or reduced precision where high accuracy isn’t critical
    

# Summary

In this blog explored the foundational concepts of GPU computing and GEMM operations that power modern AI workloads. We looked at the layers of abstraction from ML frameworks down to GPU hardware.

* GPU Architecture (A100/Ampere) - Understanding the two fundamental dimensions: Memory & Execution, along with resource constraints that define optimization boundaries
    
* CUDA Execution Hierarchy - How Grid → Thread Blocks → Warps → Threads organize parallel execution, and why warps (32 threads) are the critical execution unit
    
* Memory Coalescing - Importance of access patterns when 32 threads in a warp access consecutive memory addresses, the GPU combines requests into a single efficient transaction, this is more critical for memory bound kernels.
    
* Kernel Classification - Understanding compute-bound vs. memory-bound kernels through Arithmetic Intensity, and choosing the right optimization strategy
    

# What’s Next

In upcoming blogs, we'll delve into GEMM Optimizations by building on the concepts discussed

* **Naive GEMM Implementation** - Without any optimizations except using multiple threads to process
    
* **Tiled GEMM** - Leveraging shared memory for data reuse
    
* **Advanced Optimizations** - Warp & Register Tiling
    
* **Introduction to CUDA Libraries** - Importance of **CUTLASS & cuBLAS** in GEMM context