---
title: "A study on CUDA async memcpy"
description: >-
  A study on CUDA async exeuctions, including PTX and C++ barrier/pipeline abstractions
date: 2026-02-27
categories: [CUDA]
tags: [CUDA, Performance]
---

## Prologue

As a CUDA programming beginner, I started my journey from implementing a vanilla GEMM and try to optimize it. As I dug deeper, I found all the performant implementations need some ways to exploit async memory access to hide the long latency between global memory and shared memory. 

However, I found the related learning resources are scattered around the Internet, and they’re approaching this problem from different perspectives, utilizing different levels of API, which gave me a hard time to understand the whole picture. I was puzzled by many questions, like “why there are so many overloads of the `cuda::memcpy_async`?”, “when using `cuda::pipeline`, why sometimes we need a shared state with number of stages provided, while some other times we just need `make_pipeline()` without any params?”, “how does the high level interface translates to lower level primitives?”, and so on.

I spent quite some time trying to find the answers, and finally I feel I’m confident enough to write down my learnings. If you have similar questions as mine, I hope this blog can explain some of them. I will try my best to make this blog correct, but of course if you find any errors, please do let me know.

Please note that in this blog I will cover mostly pre-Hopper era, for example I will not cover TMA bulk copy, nor thread block cluster shared memory etc. The new features of Hopper+ arch produce a bit more complexity, and I hope I will cover them in future posts.

## Why async memcpy

A very typical pattern in our kernel is like that 

1. copy data from global memory to shared memory and/or register files
2. process the data in shared memory and/or register files
3. write the data back to global memory

The data movement latency between global memory and shared memory is significantly slower than the compute. To avoid the CUDA Core and Tensor Core stall waiting for data, the traditional mitigation is to create multiple active threads (maybe more accurately, thread warps), so that when some threads are waiting for data, another group of threads can be scheduled. This is called increasing the “occupancy”. However, increasing occupancy is not always possible or not always enough, so another approach, asynchronous memory data copy, is utilized.  

Conceptually, the asynchronous execution can be seen as if running by another thread (referred as async proxy in Nvidia doc) in parallel, other than the thread that initiates the async execution. And we just need to make sure there are mechanisms to let the initiating thread know the execution is complete. Please note that the async execution is not necessarily to be only data copying, but also computation, though our current focus is only data copying.

## PTX: mbarrier based and async-group based

PTX (Parallel Thread Execution) is an intermediate representation used in CUDA. It provides a stable instruction set architecture that allows CUDA code to be portable across different GPU generations while still enabling hardware-specific optimizations. The CUDA C++ code will be compiled to PTX then eventually to SASS. We don’t usually write code at PTX level except for the most performance critical part.

Surprisingly, I found it’s easier to start from this low-level primitives to understand the CUDA async execution. There are two mechanisms at PTX level: mbarrier-based mechanism (`cp.async.mbarrier.*` ) and async-group mechanism (`cp.async.{commit/wait}_group` ). All the higher level constructions that we will talk later will be eventually lowering to one of these two. Both mechanisms use `cp.async` instruction to initiate the actual copying asynchronously, the difference is how to handle the completion. 

### async-group based approach

For this approach, after issuing one or more `cp.async` instructions, we add a `cp.async.commit_group` to commit all the previous uncommitted async copy as a group. Afterwards, we call `cp.async.wait_group N` to wait for the groups to complete. Note that there is an important parameter `N` for the `wait_group`, which means we wait until only the most recent `N` committed groups are still in-flight. So N=0 means waiting for all the in-flight groups are done. (There is an equivalent `cp.async.wait_all`, btw)

Here is an example:

```
cp.async.ca.shared.global [shrd1], [gbl1], 4;
cp.async.cg.shared.global [shrd2], [gbl2], 16;
cp.async.ca.shared.global [shrd3], [gbl3], 8;
cp.async.commit_group;  // End of group 1

cp.async.cg.shared.global [shrd4], [gbl4], 16;
cp.async.commit_group;  // End of group 2

cp.async.cg.shared.global [shrd5], [gbl5], 16;
cp.async.commit_group;  // End of group 3

cp.async.wait_group 1;  // waits for group 1 and group 2 to complete

cp.async.cg.shared.global [shrd6], [gbl6], 16;
cp.async.commit_group;  // End of group 4

cp.async.wait_group 0;  // wait for group 3 and group 4 to complete
```

Note that in the above snippet, only the `cp.async.wait_group` is blocking operation. For example, we can freely insert other instructions between `commit_group` and `wait_group`, and these instructions will run in parallel with the background memcpy.

Also note that these operations are per-thread. There is no cross-thread synchronization, unlike the barrier approach we will discuss below. 

As we can see, it simulates a FIFO queue of async copy groups, which is exactly the semantics of higher-level C++ `cuda::pipeline` that we will talk later.

### mbarrier based approach

[mbarrier approach](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=cp%2520async%2520commit_group#parallel-synchronization-and-communication-instructions-mbarrier) is centered around a barrier object. Barrier object is an opaque object created in *shared memory*. It contains these important informations:

- Current phase of this mbarrier object
- Count of pending arrivals for the current phase of this mbarrier object
- Count of expected arrivals for the next phase of this mbarrier object

(There is also a tx-count info which is available in newer arch but we skip it for now.)

If you know the `std::barrier` in C++ std library, the semantics is pretty much same. At the init, we use `mbarrier.init [addr], N` to init a barrier object at `[addr]` and specifying the pending arrival count `N`.  Then each parallel thread may call `mbarrier.arrive` to decrease the pending count by 1. Once the pending count reach 0, the barrier will atomically change to next phase and reset the pending arrivals. 

We can use non-blocking`mbarrier.try_wait` to detect the phase completion. It will immediately return true or false. Or we can use `mbarrier.test_wait`, which will be potentially blocking for a given time limit before return.

Please note that the above workflow is already a legit one, but we haven’t yet mention how to bind the `cp.async` with the barrier object. To achieve that we need an additional special instruction `cp.async.mbarrier.arrive [addr]` , which will “bind” all prior [`cp.async`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=cp%2520async%2520commit_group#data-movement-and-conversion-instructions-cp-async) operations initiated by the executing thread to the barrier object at `[addr]`. Based on the PTX doc, once the `cp.async.mbarrier.arrive` is executed, the barrier’s pending count will increase by 1 immediately. And when the memory copy is completed by async thread, the pending count will decrease by 1. So the net effect is still zero, but it makes sure the barrier phase won’t complete before the async memcpy is done.

Here is an example that I copied from Nvidia [PTX doc](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=cp%2520async%2520commit_group#parallel-synchronization-and-communication-instructions-mbarrier-test-wait-try-wait):

```
.reg .b64 state;
.shared .b64 shMem2;
.shared .b64 shard1, shard2;
.global .b64 gbl1, gbl2;

mbarrier.init.shared.b64 [shMem2], threadCount; // init barrier arrival count
...
cp.async.ca.shared.global [shard1], [gbl1], 4;
cp.async.cg.shared.global [shard2], [gbl2], 16;

// accounts for arrive-on from prior cp.async operation
cp.async.mbarrier.arrive.shared.b64 [shMem2]; // hook async copy with barrier
...
mbarrier.arrive.shared.b64 state, [shMem2]; // reduce pending count

waitLoop:
mbarrier.test_wait.shared::cta.b64 p, [shMem2], state;
@!p bra waitLoop;  // busy-loop waiting
```

(No need to bother syntax details for now. We won’t really write PTX code.)

## CUDA C++: barrier, pipeline and memcpy_async

On top of PTX, there are multiple higher-level interfaces we can use, as mentioned in the [CUDA programming guide](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html). Per my observation, the `C++ cuda::ptx` and `C Primitives` are relatively thin wrapper of the PTX instructions, which should be easy to understand after our previous discussions. So I will be focusing on the more general and higher level interfaces (which I think it’s part of so-called [`CCCL/libcu++` library](https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/synchronization_primitives.html). Particularly we will discuss `cuda::barrier`, `cuda::pipeline` and how they work together with `cuda::memcpy_async()` .

Before diving into details, I’d like to mention that in CUDA C++ there is a concept of [thread scopes](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#thread-scopes), and the most relevant ones to this context is `cuda::thread_scope_thread` (memory operations are visible only to the local thread) and `cuda::thread_scope_block` (memory operations are visible to other threads in the same thread block). When we create `cuda::barrier` and `cuda::pipeline`, we need specify the thread scope, and as we will see, different thread scope will cause very different semantics and ptx instructions.

### cuda::barrier

The interface of `cuda::barrier` is not too different from the `mbarrier.*` PTX. Besides binding with async memcpy, the benefit that barrier can provide is as [quoted](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#asynchronous-barriers):

> An asynchronous barrier differs from a typical single-stage barrier (`__syncthreads()`) in that the notification by a thread that it has reached the barrier (the “arrival”) is separated from the operation of waiting for other threads to arrive at the barrier (the “wait”). This separation increases execution efficiency by allowing a thread to perform additional operations unrelated to the barrier, making more efficient use of the wait time.

IMHO thread-scoped barrier isn’t quite useful, or say, no advantage comparing with the thread-scoped pipeline that we’ll talk later soon. (Correct me if I’m wrong.) So let’s just take a look the block-scoped example:

```cpp
#include <cuda/barrier>
#include <cooperative_groups.h>

__device__ void compute(float *data, int iteration);

__global__ void split_arrive_wait(int iteration_count, float *data)
{
  using barrier_t = cuda::barrier<cuda::thread_scope_block>;
  __shared__ barrier_t bar;
  auto block = cooperative_groups::this_thread_block();

  if (block.thread_rank() == 0)
  {
    // Initialize barrier with expected arrival count.
    init(&bar, block.size());
  }
  block.sync();

  for (int i = 0; i < iteration_count; ++i)
  {
    /* code before arrive */

    // This thread arrives. Arrival does not block a thread.
    barrier_t::arrival_token token = bar.arrive();

    compute(data, i);

    // Wait for all threads participating in the barrier to complete bar.arrive().
    bar.wait(std::move(token));

    /* code after wait */
  }
}
```

As we can see, we use `thread_scope_block`, and the barrier is located in shared memory. In thread 0, we init the barrier with thread block size as pending arrival count. In the for loop, we have `compute()` between the `bar.arrive()` and `bar.wait()`. The benefit is that if the load is not very balanced, the threads reach bar.arrive() earlier can do the unrelated `compute()` without waiting for other slower threads. 

Apparently, for the `/* code before arrive */` part, the most typical operation is async copy from gmem to smem. We can see there are many overloads for the [`cuda::memcpy_async`](https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/asynchronous_operations/memcpy_async.html#libcudacxx-extended-api-asynchronous-operations-memcpy-async) The first two are about barriers. As you can see we need pass in a barrier object as the last parameter. Inside the function, it will be translated to the `cp.async.mbarrier.arrive` ptx (see above section), to hook this memcpy with the barrier.

### cuda::pipeline

The pipeline can be seen as a FIFO queue with a specified capacity (”stages”). At creation time, we specify the capacity of the queue (usually a small number like 2 or 3). There are four related methods as listed below:

| `producer_acquire` | Acquires an available stage from pipeline’s internal queue. If there is no available stage, blocks until some thread calls `consumer_release`. |
| `producer_commit` | Commits the asynchronous operations issued after the `producer_acquire` call on the currently acquired stage of the pipeline. |
| `consumer_wait` | Waits for completion of asynchronous operations in the oldest stage of the pipeline. |
| `consumer_release` | Releases the oldest stage of the pipeline to the pipeline object for reuse. The released stage can be then acquired by a producer. |

Though the semantics are same, the implementation could be very different for thread scoped pipeline vs block scoped pipeline. 

#### thread-scoped pipeline

For thread-scoped pipeline, we don’t really need anything in the shared memory. As a result, it can easily translate to the async-group based PTX. We already hinted that the semantics of commit_group/wait_group can be seen as a FIFO queue, and it’s also per-thread. That’s also the reason the we don’t need provide a shared state in this version of pipeline creation, like this:

```cpp
// Create a pipeline at thread scope
constexpr auto scope = cuda::thread_scope_thread;
cuda::pipeline<scope> pipeline = cuda::make_pipeline();
```

See this example for [thread-scoped pipeline example](https://github.com/roba269/gpu-learning/blob/main/async_study/test_pipeline_thread_scope.cu), and the corresponding [PTX](https://github.com/roba269/gpu-learning/blob/main/async_study/test_pipeline_thread_scope.ptx). As we can see the ptx is based on commit_group / wait_group approach.

If your kernel doesn’t need cross-thread data sharing, thread-scoped pipeline could be a good choice. Unfortunately I feel like most of the realistic work in this GenAI era (like GEMM, Attention) do need cross-thread data sharing.

#### block-scoped pipeline

On the other hand, block-scope pipeline requires some state in shared memory so that multiple threads can agree on the status of the pipeline (like if there is an available stage etc). 

So the block-scoped pipeline creation is like this:

```cpp
// Create a pipeline at block scope
constexpr auto scope = cuda::thread_scope_block;
constexpr auto stages_count = 2;
__shared__ cuda::pipeline_shared_state<scope, stages_count> shared_state;
auto pipeline = cuda::make_pipeline(group, &shared_state);
```

If you still remember the two PTX mechanisms described before, you will find we have to use the mbarrier approach for this case. We will need multiple mbarrier objects for the stages, this is a proof-of-concept implementation that use barriers to implement block-scoped pipeline:

```cpp
template <int stage_count>
class Pipeline {
private:
    int _head, _tail;
    __shared__ cuda::barrier _produced[stage_count], _consumed[stage_count];
public:
    Pipeline(int producer_count, int consumer_count) : _head(0), _tail(0) {
        if (threadIdx.x == 0) {
            for (int i = 0 ; i < stage_count ; ++i) {
                _produced[i].init(producer_count);
                _consumed[i].init(consumer_count);
            }
        }
        // set consumed to be ready so that the producer can start immediately.
        if (is_this_thread_consumer()) {
            for (int i = 0 ; i < stage_count ; ++i) {
                _consumed[i].arrive();
            }
        }
    }
    void producer_aquire() {
        _consumed[_head].wait();
    }
    void producer_commit() {
        _produced[_head].arrive();
        _head = (_head + 1) % stage_count;
    }
    void consumer_wait() {
        _produced[_tail].wait();
    }
    void consumer_release() {
        _consumed[_tail].arrive();
        _tail = (_tail + 1) % stage_count;
    }
}
```

(Note: Just for illustrating the idea. I did’t really test it, not even sure if it compile!) 

And see this example for [thread-scoped pipeline](https://github.com/roba269/gpu-learning/blob/main/async_study/test_pipeline_block_scope.cu) and the corresponding [PTX](https://github.com/roba269/gpu-learning/blob/main/async_study/test_pipeline_block_scope.ptx) As we can see the PTX is based on mbarrier approach.

## “Real world” example

Finally let’s make a relative more realistic example. This is a strange shaped matrix multiplication with M = N = 32 and relatively large K. We only launch 1 thread block which contains 32 thread (i.e. 1 warp), to amplify the result of async memcpy. We iterate along the K, each of the (K/32) iteration compute a 32 * 32 subtile from A with a 32 * 32 subtile from B, and accumulate the results.

We compare the vanilla implementation with the double-buffer implementation. For the double-buffer implementation, the idea is that we create two buffers (some one called pingpong?) for the subtile, while the current subtile is being computed, we asynchronously copy the next subtile from global memory. We use 2-stage thread-scoped cuda::pipeline to implement it.

```cpp
// baseline implemenation
__global__ void ref_kernel(float *gA, float *gB, float *gC, int M, int N, int K) {
    __shared__ float sA[WARP_SIZE], sB[WARP_SIZE];
    float acc[WARP_SIZE] = {0};
    for (int phase = 0 ; phase < K ; ++phase) {
        // collectively load one column of A and one row of B from gmem to smem
        sA[threadIdx.x] = gA[threadIdx.x * K + phase];
        sB[threadIdx.x] = gB[phase * N + threadIdx.x];
        __syncthreads();
        // each thread is responsible for one row of final result
        for (int i = 0 ; i < WARP_SIZE ; ++i)
            acc[i] += sA[threadIdx.x] * sB[i];
        __syncthreads();
    }
    for (int i = 0 ; i < WARP_SIZE ; ++i)
        gC[threadIdx.x * N + i] = acc[i];
}

// pingpong pipeline implementation
__global__ void pipeline_kernel(float *gA, float *gB, float *gC, int M, int N, int K) {
    constexpr auto scope = cuda::thread_scope_block;
    __shared__ cuda::pipeline_shared_state<scope, 2> shared_state;
    auto group = cooperative_groups::this_thread_block();
    auto pipe = cuda::make_pipeline(group, &shared_state);
    
    __shared__ float sA[2][WARP_SIZE], sB[2][WARP_SIZE];
    float acc[WARP_SIZE] = {0};
    pipe.producer_acquire();
    cuda::memcpy_async(&sA[0][threadIdx.x], &gA[threadIdx.x * K], sizeof(float), pipe);
    cuda::memcpy_async(&sB[0][threadIdx.x], &gB[threadIdx.x], sizeof(float), pipe);
    pipe.producer_commit();

    for (int produce_phase = 1 ; produce_phase < K ; ++produce_phase) {
        pipe.producer_acquire();
        cuda::memcpy_async(&sA[produce_phase % 2][threadIdx.x], &gA[threadIdx.x * K + produce_phase], sizeof(float), pipe);
        cuda::memcpy_async(&sB[produce_phase % 2][threadIdx.x], &gB[produce_phase * N + threadIdx.x], sizeof(float), pipe);
        pipe.producer_commit();
        pipe.consumer_wait();
        for (int i = 0 ; i < WARP_SIZE ; ++i)
            acc[i] += sA[(produce_phase-1)%2][threadIdx.x] * sB[(produce_phase-1)%2][i];
        pipe.consumer_release();
    }

    pipe.consumer_wait();
    for (int i = 0 ; i < WARP_SIZE ; ++i)
        acc[i] += sA[(K-1)%2][threadIdx.x] * sB[(K-1)%2][i];    
    pipe.consumer_release();

    for (int i = 0 ; i < WARP_SIZE ; ++i)
        gC[threadIdx.x * N + i] = acc[i];
}

ref_kernel<<<1,32>>>(d_A, d_B, d_C, 32, 32, 65536);
pipeline_kernel<<<1,32>>>(d_A, d_B, d_C, 32, 32, 65536);

```

See the full code [here](https://github.com/roba269/gpu-learning/blob/main/async_study/gemm_async.cu)

When running on my RTX 4070, the running time is **12040 ms** (baseline) vs **9875 ms** (pingpong pipeline). Note that it’s definitely not a good speed because we only use 1 warp to do everything. The purpose is to show the effective ness of async memcpy, instead of optimize the GEMM. 

In the previous example, all the threads have equal responsibility, i.e. they all serve as both producer and consumer. Another common approach is so called warp-specialization, whose idea is that some warps are dedicated for producer while some others are dedicates for consumer. In terms of implementation, it uses another overload of make_pipeline function, providing an additional parameter to specify the number of producer count. See [this example in the official doc](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#producer-consumer-pattern-through-warp-specialization)

In the CUTLASS library both approaches are provided. So far I’m not clear about which approach should be used in which scenarios. Hopefully I can figure that out later and post my findings.

## Summary

In this blog post, I've explored the landscape of CUDA async memory copy mechanisms, from the low-level PTX instructions (`cp.async` with `commit_group` / `wait_group` and `mbarrier`) to the high-level `cuda::barrier` and `cuda::pipeline` abstraction. Understanding these different layers and when to use thread-scoped versus block-scoped pipelines is crucial for writing efficient CUDA code that can hide memory latency. Through practical examples, we've seen how async memcpy with double-buffering can provide meaningful performance improvements.

## Reference

[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) Chapter 3.2, 4.9, 4.10, 4.11

CUDA Core Compute Libraries Doc [Synchronization Primitives](https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/synchronization_primitives.html), [Asynchronous Operations](https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/asynchronous_operations.html)

PTX Doc [mbarrier](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=cp%2520async%2520commit_group#parallel-synchronization-and-communication-instructions-mbarrier), [async copy](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=cp%2520async%2520commit_group#data-movement-and-conversion-instructions-asynchronous-copy)

[CUDA C++ Programming Guide (Legacy)](https://docs.nvidia.com/cuda/cuda-c-programming-guide) Chapter 10.26, 10.27, 10.28
