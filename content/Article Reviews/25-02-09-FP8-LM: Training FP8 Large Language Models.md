---
title: FP8-LM - Training FP8 Large Language Models
date: '2025-02-09T21:08:00Z'
draft: false
tags:
    - NLP
    - Transformers
    - Deep Learning
    - LLM's

categories:
    - Research Reviews
    - Machine Learning
    - Artificial Intelligence

authors:
    - Houwen Peng
    - Kan Wu
    - Yixuan Wei
    - Guoshuai Zhao
    - Yuxiang Yang
    - Ze Liu
    - Yifan Xiong
    - Ziyue Yang
    - Bolin Ni
    - Jingcheng Hu
    - Ruihang Li
    - Miaosen Zhang
    - Chen Li
    - Jia Ning
    - Ruizhe Wang
    - Zheng Zhang
    - Shuguang Liu
    - Joe Chau
    - Han Hu
    - Peng Cheng

github:
    - https://github.com/Azure/MS-AMP

summary: "A review of the Transformers ability to reason and their relation to seperated knowledge paper"
doi: "https://arxiv.org/abs/2310.18313"
pdf: "https://arxiv.org/pdf/2310.18313"
---
## Introduction

Training Large Language Models (LLMs) is slow and costly; they are computationally expensive, requiring a lot of compute power, memory allocation, and time to effectively train. The current paradigm of training LLMs uses 16-bit number representations (BF16), 32-bit number representations (FP32), or a mixture of both.

The reason BF16 has been used as opposed to FP16 is because BF16 offers a wider range of potential values at the expense of precision, due to its exponent occupying more of the bits rather than the mantissa, which prevents underflow and overflow in comparison to FP16. This leads to better numerical stability while matching the performance of the full-precision FP32 using half the number of bits leading to a reduction in its memory footprints, improving computational efficiency and reducing communication overhead.

So although higher-precision number representations have the capability of representing numbers more precisely at the expense of extra space in memory and higher operational costs for mathematical operations (this may be worth the tradeoff in particular domains of scientific research, engineering simulations, and financial modeling where small uncertainties may compound leading to inaccurate calculations and therefore inaccurate conclusions), in the domain of deep learning the computational overhead is high. This means models must find a balance between the accuracy of high-precision floating point representations of numbers, and lower-precision representations that speed up and increase the efficiency of computations.

The current paradigm of deep learning emphasises increasing compute power to increase computational speeds of higher-precision numbers at the expense of increased compute costs. Additionally, it almost mandates that to improve the predictive capacity of neural networks, the only step forward is more data and compute power. This has set a precedent of maintaining higher-precision representations within hardware without offering true lower-precision representational capacities i.e. even if the GPUs/TPUs/CPUs offer FP8 support, their fundamental architectures are optimised for BF16, offering only minor perceived improvements in computational efficiency and speed when using FP8.

This paper theorizes that using a lower-precision number representation may lead to reductions in cost because they have lower communication overheads, faster numerical operations, and smaller memory footprints without significant deteriorations in overall predictive accuracy, which challenges the current approach of needing increased compute power for progression in the field of deep learning. It also argues that this change in number representation does not require changes to hyperparameters during training to maintain accuracy, signifying a significant (yet simple) improvement in the efficiency of training deep neural networks. And finally, it argues that these improvements are not linear, but rather scale multiplicatively as the size of the neural architecture (which has an exponentially increasing number of weights).


---

## Key Contributions & Insights

This paper defines a novel approach FP8 mixed-precision training framework that incorporates 8-bit weights and gradients, 8-bit optimizers, and 8-bit distributed parallel training to allow for faster and more memory-efficient training while maintaining predictive accuracy of the models. This implementation does not require changes to hyper-parameters and training receipts. It addresses issues such as data underflow or overflow and quantization errors, which cause numerical instability and irreversible divergences throughout training, by using two techniques ***precision decoupling*** and ***automatic scaling***. By using FP8 instead of BF16, they have noted a 39% reduction in real memory usage and 75% faster training.


### **Automatic Scaling**
Training LLMs with reduced-precision FP8 has its challenges; data overflow and underflow are much more problematic as the dynamic range and representation precision of FP8 is much lower than that of FP16 and BF16, which leads to training collapses caused by loss spikes (or even NaNs) and vanishing gradients. To address this, *tensor scaling techniques* are proposed - this is the idea where we multiply higher precision values with a scaling factor before converting them to FP8 in order to preserve the number in a range that corresponds with the representable range of FP8. This scaling factor is determined dynamically by checking the running statistics of the gradients and weights within the layers. This allows for higher-precision numbers to preserve their gradient values within the representation range when converted to FP8, alleviating underflow and overflow occurrences.

***FP8 Gradient and All-Reduce Communication***

The typical method of storing gradients for computation is done in 16/32-bit datatypes, which results in high bandwidth requirements for collective communication during training. This is when parameters (such as gradients, model weights, and activations) need to be exchanged *between* GPUs, usually through some collective communication operation such as AllReduce (summing gradients across multiple GPUs), Broadcasting (sending updated weights to all GPUs), and AllGather (Sharing activations across GPUs for specific architectures). The higher the size of the datatype - i.e. the number of bits it takes in memory - the more bytes are required to be communicated for each number representation. For example, if a model had 100 parameters, storing and communicating in FP32 would require 4*100 bytes, while FP16/BF16 cuts this cost in half.

As a model size increases, the number of parameters may grow exponentially, leading to exponentially increasing communication overhead. Additionally, communication overhead also depends on the number of GPUs, which means that although GPUs help distribute computation, it also increases synchronization costs between the GPUs. These combined may lead to a non-trivial growing bottleneck during training as the complexity of the model increases, and the number of parallel batches are processed.

Directly applying FP8 to gradients leads to a decrease in accuracy because of the underflow and overflow problems arising from the low-bit AllReduce operation. The AllReduce operation typically aggregates gradients across GPUs using *pre-scaling* (dividing gradients before summation) and *post-scaling* (dividing after summation) within a batch of inputs, where the goal is to average the gradient tensor at a given $ i $th layer, denoted as $ g_i $, across different GPUs. But using the *pre-scaling* and *post-scaling* leads to an issue of underflowing and overflowing respectively, so this paper proposes *automatic scaling* as a method to address this issue.

Additionally, FP8 needs per-tensor scaling factors, but current GPU communication frameworks do not efficiently support reducing them across GPUs. Therefore, the scalar that is defined is calculated as a single global scaler that is shared across all GPUs, which ensures that all gradient tensors for a given $i$th layer, $g_i$, use the same shared scaling factor when quantized into FP8 format. This approach significantly reduces communication overhead by limiting the number of scalers transmitted, making synchronization steps highly efficient and allowing low-bit gradient communication without extra complexity

### **Precision Decoupling**
It has been shown that reducing the precision of an optimisers variables leads to accuracy degradation, which raises the question: which variables in the optimizer must be kept at high precision? This is where precision decoupling comes, where the goal is to decouple the influence of data precision on the variables in the optimizer and investigate which one can be assigned lower precision.

***FP8 Optimizers***

Traditional optimizers, such as Adam, maintain copies of their model weights, gradients, first-order and second-order gradient moments in 32-bit float format for numerical stability. This leads to large 16byte overhead per parameter during training. To try and reduce this, a mixed-precision approach is taken: Gradient statistics can use lower precision, while the master weights necessitate high precision; direction of the gradient holds greater significance than its magnitude.
- Master weights need higher precision because during optimization weight updates can become extremely small or large making it harder to decipher the correct direction at low precision; the higher the precision helps prevent loss of information when updating weights, ensuring stable and accurate training in the correct direction.
- The first- and second-order gradient moments are used more to scale magnitude than direction so do not require as high precision
    - The first-order gradient moment can tolerate a high quantization error and can be assigned with low-precision, albeit with some degradation.
    - The second-order moment requires a higher precision than the first-order gradient moment because calculating the square of gradients for the second-order gradient moment might lead to data underflow due to the typically small gradient values.


### Adapting to FP8 to Different Distributed Parallel Training

Distributed strategies are necessary for parallelizing training across multiple GPUs and are often used in a complementary way to increase parallelism and efficient training. These methods include
- *Data parallelism*: Distributing the same model across multiple devices, each processing a different subset of the data simultaneously.
    - Since this approach doesn't involve splitting individual tensors or layers, integrating FP8 precision doesn't require specific modifications.
- *Pipeline parallelism*: Dividing the model into sequential stages, each assigned to a different device, allowing different batches of data to be processed concurrently through the pipeline.
    - Similar to data parallelism, this method doesn't necessitate additional FP8-specific adjustments because it doesn't involve partitioning tensors within layers and independently allocates distinct layers to different portions of the GPU.
    - Although it does not need to directly find a method to implement FP8, it is essential to ensure data integrity and congruency between the layers
- *Tensor parallelism*: Splitting individual tensors (such as weight matrices) across multiple devices to perform computations like matrix multiplications in parallel.
    - This type of distribution requires the sharded weights and activation tensors to be converted into FP8 format for computations in linear layers.
    - This enables both forward computations and backward gradient communications to utilize FP8 precision, enhancing efficiency.
- *Sequence parallelism*: Distribute subsequences of the initial input into a transformer models across multiple devices, enabling parallel processing of different parts of the input sequence.
    - This type of distribution requires a converter, $g$, to manage the transition between sequence and tensor parallel regions.
    - During the forward pass, an all-gather operation collects sequence partitions, while in the backward pass, a reduce-scatter operation handles tensor segments.
    - Incorporating FP8 datatype conversion before these operations reduces communication costs across GPUs by transmitting low-bit FP8 activations.

- *Zero Redundancy Optimizer (ZeRO)*: Memory optimization technique that builds on data parallelism
    - Partitions the model states (parameters, gradients, and optimizer states) across multiple devices reducing memory redundancies
    - Applying FP8 to this method is difficult because of managing scaling factors associated with FP8
    - Solution: Allocate each tensor as a whole across devices rather than partitioning into sub-tensors. This ensures that tensor scaling factors are distributed along with the tensor and reduces communication and computational complexity.


### Results

Using FP8 mixed-precision did not lead to significant differences in the loss curves and demonstrate that the proposed FP8 mixed-precision scheme can achieve equivalent performance to those of its higher-precision counterparts across a diverse array of model scales, additionally there is comparable zero-short performance in comparison to downstream tasks in comparison to BF16 model counterparts. Furthermore, the communication costs and memory footprints dramatically decreased in comparison to its higher-precision counter parts, which further validates that models pre-trained with FP8 low-precision maintain both accuracy and intrinsic in-context learning capabilities comparable to those of higher-precision while lowering computational cost, communication overhead and reducing memory footprint.

These results also extrapolate to different approaches to LLMs, such as fine-tuning and reinforcement learning from human feedback, leading to dramatic decreases in memory utilization in the RL scenario (32% concerning model weights and 62% concerning optimizer states). This further reinforces the versatility and innovative solution FP8 has in advancing and improving the training process of LLMs while maintaining predictive quality.

---

## Evaluation of the Paper

This paper introduced a groudbreaking, paradigm shifting FP8 mixed-precision training framework that significantly improves memory efficieny and training speeds without significant sacrifices in performance. It provides a comprehensive open-source implementation of FP8 accross a variety of training components - gradients, optimizers, distributed communication, and parallel training strategies - as well as provides performance benchmarks in a variety of different training paradigms - pretraining, fine-tuning and even reinforcement learning with human feedback tasks - differentiating itself from prior approaches and providing a general solution to the field of LLM's as a whole.

With the results from this paper, it offers a way to make highly scalable solution for multi-GPU distributed training and opens the door to training even larger models at a lower computational cost and a faster pace. Its numerical stability methods of using automatic scaling rather than pre- and post-scaling ensures that FP8 weights and gradients are effectively trained through their corresponding optimizers at scale, and the decoupling of precision to ensure as much of the parameters are represented in FP8 increasing efficiency while preventing significant degradations in the model's performance.

*These results alone are paradigm-shifting.*

Additionally, the paper has done all this with the current limitations of GPU communication and optimization for FP8 operations, demonstrating that these results are yet to see their peak advancements in the domain of training and optimizing a transformer based architectures.

The paper has also led to the more recent explosion of the DeepSeek model that utilizes FP8 mixed-precision in its training (along with some other innovative approaches, [see DeepSeek-R1](https://senankassem.com/posts/25-02-11-deepSeek-r1-incentivizing-reasoning-capability-in-llms-via-reinforcement-learning/ 'DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning')) in building a competent model with Chain-of-Thought (CoT) reasoning at a fraction of the cost of current large-scale LLMs. It provides the groundwork for advancements in the field and opens up the door for future research related to the field of effective computation and communication within and between GPUs, the effects of scaling up FP8 models [(example)](https://openreview.net/forum?id=E1EHO0imOb&utm_source=chatgpt.com 'Scaling to a Trillion Tokens'), as well as in regards to our understanding of knowledge representations and updating the vector embeddings.

Now comes some more critical analysis of this ground-breaking paper:

*FP8 Training may not be as robust for all architectures*. The paper only focuses on GPT-style models (7B to 175B parameters); Different architectures, such as diffusion models, may exhibit different numerical stability issues with FP8 and the methods may not generalize well to non-transformer architectures. However, Mixture-of-Experts (MoE) - implemented by DeepSeek's R1 model - has been shown to work with this FP8 mixed-precision training paradigm, so there may be more generalizability than less.

*There is a lack of evaluation on long-context or novel tasks*. While FP8 achieves comparable pretraining loss and zero-shot performance, there is no detailed analysis of fine-grained accuracy trade-offs in tasks that require long-term dependencies (e.g., long-context reasoning tasks). Additionally, FP8’s impact on out-of-distribution generalization is also unexplored.

*The proposed global minimum scaling factor synchronization is efficient but introduces additional computation*. Synchronizing scaling factors across GPUs could become a bottleneck at extreme scales (e.g., trillion-parameter models), but then again, there may be other bottle necks before this is ever an issue since the scaling factor grows logarithmically to the number of parameters as it scales via the tensors that carry the weights.

*The FP8 benefits are highly hardware-dependent*, and their benefits may not transfer well to older GPUs (e.g., A100), or may require specific fine-tuning of the processing unit to benefit from this implementation.

There is a *lack of comparative analysis with other FP8 approaches*. There exist alternative FP8 training methods beyond Nvidia TE; research such as Graphcore's FP8 work or other mixed-precision techniques (e.g., hybrid 8-bit/16-bit schemes) could provide better baselines for comparison in their results as Nvidia TE only operates at the matrix multiplication level (linear forward-pass layer) rather than at the optimizer and syncronization layers.

*FP8 impact on inference remains unexplored*. This paper primarly focuses on training efficiency without really touching on any computation or memory advantages at inference. Since FP8 training reduces memory footprints and computational overhead, it is worth investigating whether it also enables better model compression or quantization at inference time as well. This may provide insights into its viability for real-world deployment, particularly in resource-constrained environments.*

Finally, there is *no detailed analysis of quantization artifacts*. While automatic scaling mitigates quantization errors, the paper does not provide a deep analysis of gradient statistics under FP8. This raises questions like "Do certain layers (e.g., attention vs. feedforward) suffer more from FP8 quantization?" and "What is the effect of FP8 on weight sparsity or activation distributions?"

---

## Final Thoughts

This paper was very thorough in its approach, addressing many different aspects of the underlying processes that allow for the training of neural networks. The approach affords a more efficient approach to training models without dramatically increasing complexity. The solution also seems to scale as the LLMs grow larger in their parameter count, which leads me to believe this is a "low-hanging fruit" implementation to improve training of future models, as well as generalizes to a variety of training paradigms. This paper opens the door to more open-source research-focused LLMs which may push out understanding of knowledge representation within vectors as well as accessibility to larger models. Additionally it opens the door for more optimized FP8 GPUs and iterations that allow for true observations in the compute and cost optimization FP8 mixed-precision implementations provide.

Overall, FP8-LM is a significant step toward making LLM training more efficient. The proposed full-stack FP8 training framework unlocks substantial memory, communication, and compute savings, making scaling up next-gen foundation models more feasible.

However, questions remain about robustness across different architectures, long-context tasks, and alternative quantization strategies. Future research should explore broader applications of FP8 and hybrid precision methods to further optimize LLM training.

Areas for Further Exploration
1. Extending to Other Architectures
- Future work should evaluate FP8 on different architectures beyond transformers (e.g., diffusion models, CNNs, MoE models).
- Applying FP8 to vision or multimodal models could reveal new insights.
2. Long-Context and Novel Training Analysis
- Evaluating FP8 models on long-context benchmarks (e.g., LongBench, Needle-in-a-Haystack) would help determine if FP8 impacts context retention.
- Evaluating FP8 models on out-of-distribution data may also determine the effectiveness of the generalizability and capacities of these models in comparison to their higher-precision counterparts.
3. Inference Considerations
- Investigating if FP8 models allows for better model compression or quantization at inference as well.

