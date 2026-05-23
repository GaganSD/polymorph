# Comparative Technical Evaluation of Gated DeltaNet-2 and Mamba-3 MIMO for Sub-Millisecond Context Compression and Long-Sequence Binary Classification

Processing sequences in linear time without the quadratic computational and memory bottlenecks of standard self-attention has emerged as a critical requirement for serving ultra-long context windows. For enterprise sequence classifiers tasked with sub-millisecond context compression over sequences exceeding $100\text{k}$ tokens, architectures must be evaluated on both mathematical representational capacity and physical hardware constraints.

This technical report evaluates Gated DeltaNet-2—a linear attention model featuring decoupled key-value memory editing —against Mamba-3 MIMO, an "inference-first" State Space Model (SSM) using second-order discretization, complex-valued state tracking, and Multi-Input Multi-Output (MIMO) updates. The objective of this analysis is to determine whether Gated DeltaNet-2's delta-rule mechanics provide superior token-level precision for binary classification over long contexts compared to the Mamba-3 paradigm, and whether VRAM and latency profiles justify a pivot in the core machine learning engine.

## Architectural Frameworks and Mathematical Paradigms

To establish a baseline for evaluation, the mathematical formulations of Gated DeltaNet-2 and Mamba-3 MIMO must be analyzed. These architectures represent different approaches to sequence mixing: Gated DeltaNet-2 refines the fast-weight update mechanisms of linear attention, while Mamba-3 MIMO extends continuous-time linear dynamical systems.

## The Gated DeltaNet-2 Fast-Weight Engine

Linear attention models maintain a persistent recurrent state matrix $S_t \in \mathbb{R}^{d_k \times d_v}$. Standard linear attention updates this state additively, which can cause historical associations to be overwritten or degraded over long sequences. The delta rule mitigates this by reading the existing association at the current key coordinate and subtracting it before writing a new value.

```
Linear Attention State Updates:

1. Additive Update (Standard):
   S_t = S_{t-1} + k_t * v_t^T
   (Information accumulates continuously, leading to representation scramble)

2. Gated Delta Rule-2 (Decoupled):
   S_t = (I - k_t * (b_t ⊙ k_t)^T) * D_t * S_{t-1} + k_t * (w_t ⊙ v_t)^T
   (Surgical erasure on key axis via b_t, selective writing on value axis via w_t)
```

Prior formulations, including Gated DeltaNet and Kimi Delta Attention (KDA), restrict active memory editing by tying the erase and write decisions to a single scalar gate, $\beta_t \in $. Gated DeltaNet-2 decouples these axes. Erasing is parameterized as a key-side operation determining which coordinates of the existing state should be removed, while writing is parameterized as a value-side operation determining which coordinates of the incoming value should be committed. For input token embedding $x_t \in \mathbb{R}^d$, the model projects query $q_t \in \mathbb{R}^{d_k}$, key $k_t \in \mathbb{R}^{d_k}$, and value $v_t \in \mathbb{R}^{d_v}$ matrices. The channel-wise erase gate $b_t \in ^{d_k}$ and write gate $w_t \in ^{d_v}$ are generated via independent projections :

$$b_t = \sigma(W_b x_t)$$

$$w_t = \sigma(W_w x_t)$$

The channel-wise log-decay $g_t$ and decay matrix $D_t = \text{Diag}(\alpha_t)$ are parameterized as :

$$g_t = -\exp(a) \odot \text{softplus}(W_f x_t + \delta)$$

$$\alpha_t = \exp(g_t)$$

To prevent precision loss when calculating cumulative log-decay across long contexts, the decay activation is computed in FP32 before being consumed by the recurrent kernel. Gated DeltaNet-2 also supports a negative-eigenvalue variant to stabilize state transitions, scaling the erase gate to $^{d_k}$ while keeping the write gate in $^{d_v}$ to preserve value magnitude. The modulated erase vector $e_t$ and write vector $z_t$ are defined as :

$$e_t = b_t \odot k_t$$

$$z_t = w_t \odot v_t$$

Applying decay before the active memory edit yields the state evolution under Gated Delta Rule-2 :

$$\bar{S}_t = D_t S_{t-1}$$

$$r_t = \bar{S}_t^T e_t$$

$$S_t = \bar{S}_t + k_t (z_t - r_t)^T$$

This can be written as a single recurrence equation :

$$S_t = \left(I - k_t (b_t \odot k_t)^T\right) D_t S_{t-1} + k_t (w_t \odot v_t)^T$$

The output representation is retrieved via :

$$o_t = S_t^T q_t$$

By using $b_t \odot k_t$ on the key axis, Gated DeltaNet-2 makes the read/erase direction channel-selective. Similarly, using $w_t \odot v_t$ on the value axis makes the write update channel-selective. When both gates collapse to a scalar ($b_t = \beta_t \mathbf{1}_{d_k}$, $w_t = \beta_t \mathbf{1}_{d_v}$), the model recovers KDA; if the channel-wise decay collapses to a scalar, it recovers the original Gated DeltaNet.

## The Mamba-3 MIMO State Space Engine

Mamba-3 models continuous-time linear dynamics using State Space Model (SSM) principles :

$$\dot{h}(t) = A(t)h(t) + B(t)x(t)$$

$$y(t) = C(t)^T h(t)$$

where $h(t) \in \mathbb{R}^N$ represents the hidden state, $x(t)$ the input sequence, and $A(t)$, $B(t)$, and $C(t)$ are time-varying, data-dependent transition parameters. Mamba-3 introduces three primary modifications to this continuous formulation:

1. Exponential-

Trapezoidal Discretization — Mamba-1 and Mamba-2 discretized the continuous system using first-order "exponential-Euler" approximations. Mamba-3 uses a second-order "exponential-trapezoidal" discretization scheme, raising the local truncation error to $\mathcal{O}(\Delta_t^3)$:

$$h_t = e^{\Delta_t A_t} h_{t-1} + (1-\lambda_t)\Delta_t e^{\Delta_t A_t} B_{t-1} x_{t-1} + \lambda_t \Delta_t B_t x_t$$

which is parameterized as:

$$h_t = \alpha_t h_{t-1} + \beta_t B_{t-1} x_{t-1} + \gamma_t B_t x_t$$

where $\lambda_t \in $ is a data-dependent scalar. Expanding this recurrence reveals an implicit convolution over the SSM input, allowing the model to operate without the external short causal convolutions used in Mamba-1 and Mamba-2.

| Discretization Method | $\alpha_t$ Factor | $\beta_t$ Factor | $\gamma_t$ Factor | Source Context |

| --- | --- | --- | --- | --- |

| Forward Euler | $I + \Delta_t A$ | — | $\Delta_t$ | Standard Baseline |

| Backward Euler | $(I - \Delta_t A)^{-1}$ | — | $(I - \Delta_t A)^{-1} \Delta_t$ | Standard Baseline |

| Trapezoidal | $(I - \frac{\Delta_t}{2} A)^{-1} (I + \frac{\Delta_t}{2} A)$ | — | $(I - \frac{\Delta_t}{2} A)^{-1} \Delta_t$ | Classical S4 |

| Exponential-Euler | $\exp(\Delta_t A_t)$ | — | $\Delta_t$ | Mamba-1, Mamba-2 |

| Exponential-Trapezoidal | $\exp(\Delta_t A_t) @@MATH15@@ | $\lambda_t \Delta_t$ | Mamba-3 |

2. Complex-

Valued State Transitions (The RoPE Trick) — While real-valued state updates can scale representations, they struggle with tracking structured cycles or relative patterns over long contexts. Mamba-3 addresses this by using complex-valued states ($h_t \in \mathbb{C}^{N/2}$), which represent rotation-like dynamics. This complex transition matrix is implemented via a data-dependent Rotary Positional Embedding (RoPE) applied directly to the input and output projections. This avoids the need to compile custom complex CUDA kernels.

3. Multi-

Input Multi-Output (MIMO) Projections — Standard SSM updates are Single-Input Single-Output (SISO) operations where the state update relies on an outer product of scalar inputs. Mamba-3 MIMO projects inputs into vector-sized representations $x_t \in \mathbb{R}^R$, transitioning the update step to a matrix-multiplication-based operation. This architectural shift increases decoding FLOPs by up to $4\times$ at a fixed state size without increasing wall-clock decode latency, maximizing GPU utilization during the memory-bound decoding phase.

## Hardware-Efficient Processing and State Update Dynamics

For sequence classification over $100\text{k}+$ token horizons, parallel training throughput and fast sequential decoding are critical operational metrics. Both Gated DeltaNet-2 and Mamba-3 are designed to utilize modern hardware, but they rely on different mathematical approaches to achieve efficiency. Parallel Training Mechanics: WY Chunkwise vs. Selective ScanTraining long sequences sequentially is computationally impractical on modern GPUs. To resolve this, both architectures use parallelized training paths : Gated DeltaNet-2 Chunkwise WY DerivationGated DeltaNet-2 splits the sequence into chunks of size $C$. Cumulative channel-wise decay factors are absorbed into asymmetric erase and write terms within each chunk. For a given chunk, let the decay-normalized matrices be :

$$\bar{E} = \gamma \odot (B \odot K)$$

$$Z = W \odot V$$

The intra-chunk interaction matrix $T \in \mathbb{R}^{C \times C}$ is defined as a strictly lower triangular causal pattern :

$$T = \text{tril}(\bar{E}\bar{K}^T, -1)$$

Using a forward substitution step inside each GPU shared memory block, the correction matrix $A$ is computed as :

$$A = (I + T)^{-1}$$

The WY auxiliary matrices are then computed as :

$$Y = A \bar{E}$$

$$U = A Z$$

Here, $Y$ acts as the erase-side auxiliary and $U$ acts as the write-side auxiliary. These terms allow the end-of-chunk state matrix $S[n+1]$ to be computed via highly parallelized matrix multiplications (GEMMs) :

$$S[n+1] = \text{Diag}(\gamma_C) S[n] + K^T Y S[n] + U^T \dots$$

This formulation parallelizes intra-chunk operations, allowing Gated DeltaNet-2 to maintain flat training throughput scaling over long sequence lengths. Mamba-3 Selective ScanMamba-3 parallelizes training by treating the linear recurrence as a parallel associative scan, mapping the selective state equations directly to Tensor Cores. Because its transition equations are associative, the model can resolve the entire sequence update in parallel across GPU threads. Decode Phase and Arithmetic Intensity During sequence decoding, models process inputs token-by-token, shifting the bottleneck from raw compute to memory bandwidth (moving the hidden state between high-bandwidth memory and the execution units). In standard SISO SSMs, the update requires loading the state vector and projecting the scalar input, yielding low arithmetic intensity. Mamba-3's MIMO formulation restructures the decode update into a dense matrix multiplication. By processing vector-sized inputs $x_t \in \mathbb{R}^R$, the model increases the ratio of FLOPs to memory transactions. This allows Mamba-3 MIMO to utilize idle Tensor Cores, performing more computation per memory transfer and matching Mamba-2's decode latency while serving a more expressive state. Gated DeltaNet-2 maintains near-flat decoding latency over long context windows, but it requires two additional projection steps per token to compute the channel-wise erase gate $b_t$ and write gate $w_t$. This adds a small, constant computational overhead compared to models using scalar gates.

## Quantitative Evaluation: Long-Context Precision and Retrieval

To evaluate context compression and classification accuracy over $100\text{k}+$ token sequences, we analyze empirical benchmarks measuring retrieval precision and state-tracking capabilities.

### Long-Context Precision on RULER Needle-in-a-Haystack Benchmarks

The RULER benchmark evaluates a model's ability to locate and retrieve specific information (needles) embedded within long sequences of distracting text (haystacks). Single-key (S-NIAH) and multi-key (MK-NIAH) configurations measure the model's robustness against representation degradation and memory interference.

| Evaluation Benchmark & Context Setting | Gated DeltaNet-2 | Mamba-3 (MIMO) | Kimi Delta Attention (KDA) | Gated DeltaNet (GDN-1) |
| --- | --- | --- | --- | --- |
| S-NIAH-2 @4K (Recurrent) | 93.0% | 64.2% | 89.0% | 87.2% |
| S-NIAH-3 @2K (Recurrent) | 89.8% | 72.4% | 63.2% | 54.2% |
| MK-NIAH-1 @4K (Recurrent) | 37.8% | 18.0% | 28.0% | 27.8% |
| S-NIAH-2 @4K (Hybrid) | 57.9% | 53.0% | 56.0% | 57.3% |
| S-NIAH-3 @2K (Hybrid) | 99.0% | 98.4% | 93.4% | 91.2% |
| MK-NIAH-1 @4K (Hybrid) | 48.0% | 46.6% | 40.4% | 44.8% |
| Real-world Retrieval Avg (Recurrent) | 29.88% | 28.35% | 28.67% | 28.09% |
| Real-world Retrieval Avg (Hybrid) | 42.28% | 40.11% | 40.14% | 39.11% |

These results demonstrate Gated DeltaNet-2's precision advantage in recurrent configurations, particularly in high-interference settings. On MK-NIAH-1 @4K, Gated DeltaNet-2 achieves an accuracy of $37.8\%$, more than doubling the $18.0\%$ achieved by Mamba-3 MIMO. This difference in retrieval precision is explained by their state-updating mechanisms. Additive updates in SSMs compress all historical information into a shared state space. Over very long sequences ($100\text{k}+$ tokens), writing many consecutive associations causes representations to overlap, leading to memory degradation. In contrast, Gated DeltaNet-2's decoupled channel-wise erase gate ($b_t$) can selectively target and clear specific key coordinates in the state before writing new value coordinates via the write gate ($w_t$). This channel-wise memory editing acts as a targeted eraser, allowing the model to isolate and protect competing associations. This is supported by ablation studies indicating that the erase gate ($b_t$) is the primary driver of Gated DeltaNet-2's retrieval gains.

### Language Modeling and Commonsense Reasoning

At the 1.3B/1.5B scale, pretraining performance on standard language modeling and reasoning tasks indicates overall representational capacity.

| Model Specification (1.3B–1.5B Scale) | WikiText Perplexity ↓ | Lambada Perplexity ↓ | Lambada Accuracy ↑ | Standard Commonsense Avg ↑ |
| --- | --- | --- | --- | --- |
| Gated DeltaNet-2 (Recurrent) | 15.90 | 11.41 | 48.09% | 53.11% |
| Mamba-3 MIMO (Recurrent) | 16.45 | 11.66 | 47.82% | 52.39% |
| Kimi Delta Attention (KDA) | 16.81 | 11.68 | 48.13% | 52.28% |
| Gated DeltaNet (GDN-1) | 16.40 | 11.89 | 49.62% | 52.07% |
| Mamba-2 | 16.79 | 12.38 | 45.24% | 51.82% |
| Gated DeltaNet-2 (Hybrid) | 15.62 | 10.43 | 50.90% | 53.97% |
| Mamba-3 MIMO (Hybrid) | 15.81 | 10.92 | 49.82% | 52.72% |

Across both pure recurrent and hybrid configurations, Gated DeltaNet-2 out-performs matching-scale Mamba baselines. In pure recurrent settings, Gated DeltaNet-2 achieves an average accuracy of $53.11\%$, compared to $52.39\%$ for Mamba-3 MIMO and $51.82\%$ for Mamba-2. In hybrid settings (integrating sliding window self-attention), Gated DeltaNet-2 reaches $53.97\%$ average accuracy.

## Physical Systems Engineering: VRAM and Latency Footprints

For sub-millisecond sequence classification over $100\text{k}+$ token inputs, the modeling precision of an architecture must be evaluated alongside its physical footprint and execution latency on hosting hardware.

### VRAM Footprint Scaling Characteristics

In standard Transformers, the Key-Value (KV) cache grows linearly with sequence length, resulting in quadratic memory growth. At $100\text{k}+$ token horizons, the KV cache requires dozens of gigabytes of VRAM, limiting sequence processing on standard hosting hardware. Both Gated DeltaNet-2 and Mamba-3 avoid this memory explosion by compressing historical context into a fixed-size recurrent state. This state remains constant in size regardless of the sequence length, ensuring $O(1)$ memory consumption during decoding.

```
VRAM Consumption over Ultra-Long Contexts (100k+ Tokens):

Transformer (Linear KV-Cache Growth)     /
                                        /
                                       /
                                      /____
Gated DeltaNet-2 / Mamba-3 (O(1) Constant Recurrent State) _______________

+---------------------- Sequence Length
```

The memory footprint characteristics of these architectures are analyzed below: Mamba-3 MIMO: The VRAM consumption is determined by:

$$\text{VRAM}_{\text{SSM}} \approx (\text{Parameters} \times \text{Bytes Per Dtype} \times 1.07) + \text{State Overhead}$$

The $1.07$ factor accounts for active activation and runtime buffers, which are smaller than those of Transformers because there is no KV cache headroom to reserve. A key hardware benefit of Mamba-3 is state compaction: it matches the perplexity and performance of Mamba-2 while using half its predecessor's state size ($N=64$ vs. $128$). This allows a 1.5B Mamba-3 model to operate with a minimal state overhead. Gated DeltaNet-2: The state is a matrix $S_t \in \mathbb{R}^{d_k \times d_v}$ per head. For $16$ heads with $d_k = d_v = 128$, the recurrent state size is $16 \times 128 \times 128 = 262,144$ parameters per layer. For a $1.3\text{B}$ parameter architecture, this state remains constant across the $100\text{k}+$ sequence. This fixed state overhead is stable and avoids the memory allocation spikes associated with Transformers.

### Computational Latency and Real-Time Decoding Performance

To achieve sub-millisecond context compression and classification, the time spent processing each incoming token must remain low and stable.

```
Token Decode Latency (Wall-Clock Time on GPU)

Gated DeltaNet-2 (Small constant projection overhead)     /
                                                         /
                                                        /_______________
Mamba-3 MIMO (MIMO matrix-multiplies maintain flat latency) _______________

+------------------------ Sequence Length (100k+ Tokens)
```

**Mamba-3 MIMO Latency:** Mamba-3 is designed to optimize execution times during the decoding phase. By replacing standard causal convolutions with implicit convolutions in its discretization-based recurrence, it reduces external convolution dependencies. Furthermore, the MIMO matrix-multiplication-based state update allows the model to perform up to four times more mathematical operations in parallel per step. This improves hardware utilization on modern GPUs, matching the decoding latency of simpler models while processing richer state transitions.

**Gated DeltaNet-2 Latency:** Gated DeltaNet-2 scales near-flat with sequence length, maintaining high throughput via custom Triton kernels. However, because it projects two additional channel-wise gates ($W_b$ and $W_w$) per token, it introduces a small, constant computational overhead compared to models using scalar gates. This overhead remains small but can affect execution times under tight latency constraints.

## Strategic Decision Analysis and Technical Recommendation

To guide the decision on whether to pivot from Mamba-3 MIMO to Gated DeltaNet-2 for a linear-time sequence classifier operating over $100\text{k}+$ token sequences, we evaluate their trade-offs across key operational requirements:

### Task Demands: Key Retrieval vs. Continuous Tracking

**Precision-Oriented Keyword and Key-Value Retrieval:** If the classification task relies on identifying specific, sparse triggers or keywords across a long context (such as detecting document boundary anomalies, contract compliance terms, or specific log file triggers), Gated DeltaNet-2 is highly effective. Its decoupled erase/write gates act as a targeted eraser, preventing representation overlap and maintaining high retrieval precision over long sequences. This is supported by its superior performance on multi-key retrieval benchmarks.

**State Transition and Continuous Logic Tracking:** If the classification depends on tracking continuous mathematical state transitions, cyclic patterns, or continuous arithmetic variables (such as log-arithmetic verification or sequence logic checking), Mamba-3 MIMO is better suited. Its complex-valued states capture rotational dynamics, allowing it to solve modular arithmetic and logic tracking tasks where real-valued models struggle.

### Systems Engineering: VRAM and Latency Feasibility

**Hardware Footprint and State Compaction:** Mamba-3 MIMO matches the performance of its predecessor using half the state size ($N=64$ vs. $128$). This state compaction reduces VRAM overhead and minimizes memory access latency during decoding.

**Execution Latency and Hardware Utilization:** Gated DeltaNet-2 is efficient during parallel training via its WY chunkwise Triton formulation. However, during sequential token-by-token decoding, it requires additional channel-wise gate projections, adding a small constant computational overhead. Mamba-3 MIMO's matrix-multiplication updates improve hardware utilization on modern GPUs, performing more compute operations per memory transfer to maintain flat decode latency.

### Summary of Technical Trade-offs

| Comparative Metric | Gated DeltaNet-2 | Mamba-3 MIMO | Operational Implications |
| --- | --- | --- | --- |
| Long-Context Retrieval ($100\text{k}+$ Tokens) | Excellent ($37.8\%\text{–}48.0\%$ MK-NIAH) | Moderate ($18.0\%\text{–}46.6\%$ MK-NIAH) | Gated DeltaNet-2's decoupled erase gate targets specific key coordinates, minimizing interference. |
| Decode Latency Profile | Sub-millisecond, with minor gate projection overhead | Sub-millisecond, with $4\times$ FLOP efficiency gains | Mamba-3 MIMO leverages matrix multiplications to maintain flat wall-clock decoding latency. |
| VRAM Consumption | $O(1)$ constant, with head-dimension state overhead | $O(1)$ constant, matching perplexity with half the state size | Mamba-3's smaller state size ($N=64$) reduces footprint and memory traffic. |
| State Dynamics | Real-valued fast-weight matrix edits | Complex-valued rotations via the RoPE trick | Mamba-3's complex-valued transitions are optimized for tracking structured cycles and logic. |
| Training Throughput | Parallel chunkwise WY algorithm | Parallel selective scan | Both architectures support parallel training, but Mamba-3 has a more mature hardware kernel ecosystem. |

### Technical Recommendations

Based on this comparative evaluation, we recommend the following deployment paths:

**Pivot to Gated DeltaNet-2 for Sparse Anomaly Detection and Keyword-Based Classification:** If the classification task requires identifying specific, sparse triggers or key-value pairings within $100\text{k}+$ sequences, Gated DeltaNet-2's decoupled erase and write gates prevent representation scrambling and maintain high retrieval precision. This is supported by its superior multi-key retrieval performance compared to Mamba architectures.

**Deploy Gated DeltaNet-2 in a Hybrid Configuration with Sliding Window Attention (SWA):** To maximize classification performance, Gated DeltaNet-2 should be implemented in a hybrid architecture integrating local sliding window self-attention. This hybrid combination achieves the highest overall accuracy on long-context benchmarks ($53.97\%$ average accuracy and $48.0\%$ multi-key retrieval success at $4\text{k}$ context lengths). This configuration balances linear-time global processing with high-fidelity local attention.

**Retain Mamba-3 MIMO for Continuous Logic Tracking and Maximum Decode Throughput:** If the classification task relies on tracking continuous variables, state machine updates, or algorithmic logic, Mamba-3's complex-valued state tracking is better suited. Additionally, if decode latency and GPU cost-to-throughput ratios are primary operational constraints, its MIMO dense matrix updates maximize Tensor Core utilization to maintain flat decoding latency.
