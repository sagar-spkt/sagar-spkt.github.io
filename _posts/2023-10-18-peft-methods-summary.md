---
title: 'PEFT Cheat Sheet: Succinct Explanations to the Numerous PEFT Methods for LLM'
date: 2023-10-18
permalink: /posts/2023/10/peft-methods-summary/
excerpt: "Have a quick look into the Parameter-Efficient Fine-Tuning (PEFT) methods for Large Language Models (LLMs) and discover how these techniques are revolutionizing machine learning by optimizing computational resources without compromising performance."
tags:
  - Large Language Models
  - Machine Learning
  - Natural Language Processing
  - LLMs
  - PEFT
  - Finetuning
  - LoRA
  - QLoRA
  - NLP
  - Parameter-Efficient Fine-Tuning
  - PEFT Cheatsheet
---

Parameter-efficient fine-tuning (PEFT) of large language models (LLMs) is a critical area of focus in today's machine learning research, driven by the need to optimize computational resources without compromising performance. Fine-tuning a pre-trained model for a specific task can often require substantial computational power, making it a bottleneck for many real-world applications. PEFT methods aim to mitigate this challenge by efficiently leveraging the existing parameters of a pre-trained model while adding, modifying, or reconfiguring a minimal number of parameters for the target task. This blog post will delve into various PEFT methods for LLMs, categorizing them into additive, selective, reparametrization-based, and hybrid methods. This classification of PEFT methods is taken from [Lialin et al., 2023](https://arxiv.org/pdf/2303.15647.pdf) Each method will be succinctly described, highlighting its unique approach and potential benefits.

## Traditional Fine-Tuning
Before going into the list of PEFT methods, let's explain how we traditionally adapted our pretrained models for a downstream task. The process involves taking a pre-trained model, which has already learned useful features from a large-scale dataset, and further training it on a specific task using a smaller, task-specific dataset.

The process of traditional fine-tuning is as follows:
* Pre-training: A model is first trained on a large-scale dataset. This is often an unsupervised task, such as language modelling, where the model learns to predict the next word in a sentence. During this pre-training phase, the model learns a general understanding of the language, including its syntax, semantics, and some level of world knowledge.
* Fine-tuning: After pre-training, the model is then fine-tuned on a specific task using a smaller, task-specific dataset. This could be any supervised NLP task like sentiment analysis, question answering, or named entity recognition. During this phase, all the model parameters are updated to optimize for the specific task.

## Parameter-Efficient Fine-Tuning
In recent years, the use of large language models has revolutionized the field of natural language processing (NLP). These models, such as GPT-3 and BLOOM, are capable of generating human-like text and understanding the context of a language. However, fine-tuning these models traditionally can be computationally expensive and require a lot of memory due to the large number of parameters involved. To tackle this problem, there has been a upsurge in research community to find the efficient way to fine-tune pretrained models for downstream task. These founds and to be found methods are collectively called "Parameter-Efficient Fine-Tuning" methods, PEFT in short.

If one tries to list all the PEFT methods available online, it might take forever. Here, I try to list and provide quick explanation to some popular one. Most of them are extracted from survey paper by [Lialin et al., 2023](https://arxiv.org/pdf/2303.15647.pdf). However, I've included some that the paper not listed or some that are published after the survey paper. Basically, the paper categorized PEFT methods based on the conceptual framework underlying the approach. 

<style>
  details {
    margin-left: 2em;
  }

  summary {
    margin-left: -2em;
  }
    summary h1,
    summary h2,
    summary h3,
    summary h4,
    summary h5,
    summary h6 {
        display: inline;
    }
</style>

[Click Methods to Expand their Explanations ]

<details>
<summary><h3>1. Additive Methods</h3></summary>

Additive methods for fine-tuning language models involve expanding the pre-existing pre-trained model with supplementary parameters or layers, and then training only those newly added parameters. Despite the potential increase in complexity, adding parameters can enhance training time and memory efficiency by shrinking the size of gradients and the optimizer states. Consequently, this approach can enhance the fine-tuning of larger networks or the use of larger micro-batch sizes, thus enhancing GPU training throughput and reducing communication volume in distributed setups. Based on the way parameters are added, it is divided into Adapters, Soft Prompting, and Others.
<details>
<summary><h4>1.1 Adapters</h4></summary>
Adapters are a method that introduces small, fully-connected networks after Transformer sub-layers.

<details>
<summary><h5>1.1.1 Adapters</h5></summary>

Adapters add fully-connected networks with a small hidden dimension after attention and feed-forward network (FFN) layers in a Transformer. Although this approach reduces the parameters updated during training, it creates inference overhead due to the added layers.

</details>

<details>
<summary><h5>1.1.2 AdaMix</h5></summary>

AdaMix uses multiple adapters in a mixture-of-experts (MoE) fashion. Unlike a regular MoE, which selects and weights multiple experts using a routing network, AdaMix randomly selects a single expert for each forward pass. This strategy minimizes computational costs and barely degrades the performance.

</details>

</details>

<details>
<summary><h4>1.2 Soft Prompts</h4></summary>

Soft prompts involve fine-tuning a portion of the model’s input embeddings via gradient descent. This approach transforms the problem of finding prompts in a discrete space(textual prompts) into a continuous optimization problem.

<details>
<summary><h5>1.2.1 Prompt Tuning</h5></summary>

Prompt tuning introduces a trainable tensor, commonly referred to as a "soft prompt", which is prepended to the model's input embeddings. This tensor is directly optimized through gradient descent. This method requires storing a small task-specific soft prompt and enables mixed-task inference using the original pre-trained model.

</details>

<details>
<summary><h5>1.2.2 Prefix Tuning</h5></summary>

Prefix tuning is a method used to address the instability of prompt tuning. Instead of only adding a soft prompt to the model input, trainable parameters are prepended to the hidden states of all layers. The same prefix is prepended to all of the hidden states.

</details>

<details>
<summary><h5>1.2.3 P-Tuning</h5></summary>

P-Tuning is another form of soft prompting, which employs a prompt encoder (a bidirectional long-short term memory network or LSTM) to optimize the prompt parameters. The prompt tokens can be inserted anywhere in the input sequence, and are not restricted to only the beginning.

</details>

<details>
<summary><h5>1.2.4 Intrinsic Prompt Tuning (IPT)</h5></summary>

Intrinsic Prompt Tuning (IPT) hypothesizes that the space used to define soft prompt parameters contains an "intrinsic task subspace" that can differentiate between various tasks. It introduces an autoencoder to (de)compress the soft prompt. Despite reducing the number of parameters for the soft prompt, the requirement to train the autoencoder makes it practically infeasible.

</details>

</details>

<details>
<summary><h4>1.3 Other Additive Approaches</h4></summary>

Beyond adapters and soft prompts, there are other methods of adding parameters that do not strictly follow the concepts of adapters or soft prompts.

<details>
<summary><h5>1.3.1 Knowledge Distillation</h5></summary>

Knowledge distillation is a technique that transfers knowledge from a larger, high-performing model (the teacher model) to a smaller model (the student model). The teacher model's output probabilities serve as soft targets for training the student model, enabling the student model to benefit from the teacher model's knowledge and generalize better.

</details>

<details>
<summary><h5>1.3.2 Ladder-Side Tuning (LST)</h5></summary>

Ladder-Side Tuning (LST) trains a small transformer network on the side of the pre-trained network. This side network combines the hidden states of the pre-trained backbone network with its own hidden states, using the pre-trained model as a feature extractor. Backpropagation is only computed through the side network, saving on both memory and compute during training.

</details>

<details>
<summary><h5>1.3.3 IA3</h5></summary>

(IA)3 is a method that learns new parameters (vectors) lv, lk, lff which rescale key, value, and hidden FFN activations in each transformer layer. This method produces very low overhead during parameter updates in fine-tuning.

</details>

</details>
</details>

<details>
<summary><h3>2. Selective Methods</h3></summary>

Selective methods for parameter-efficient fine-tuning involve optimizing a subset of a model's existing parameters. The selection can be based on layer depth, layer type, or even individual parameters. Here are some popular selective methods:

<details>
<summary><h4>2.1 Quantization</h4></summary>

Quantization is a method that reduces the precision of model parameters to lower memory and computational requirements. In traditional deep learning models, parameters are usually stored as 32-bit floating-point numbers. Quantization, however, allows these parameters to be represented with lower bit precision, such as 8-bit integers. This reduction in precision significantly lowers the memory footprint and speeds up computations.

</details>
<details>
<summary><h4>2.2 BitFit</h4></summary>

BitFit is a method that fine-tunes only the biases of the network. For every linear or convolutional layer, the weight matrix is kept constant, and only the bias vector is optimized. This approach is particularly efficient as it reduces the number of parameters that need to be updated during training.

</details>
<details>
<summary><h4>2.3 Pruning</h4></summary>

Pruning is a technique that involves removing unnecessary weights or connections from a pre-trained model. By identifying and eliminating redundant or less important parameters, the model’s size and computational requirements can be significantly reduced. Pruning can be performed based on different criteria, such as magnitude-based pruning or structured pruning. Magnitude-based pruning removes weights with small magnitudes, while structured pruning removes entire neurons or filters based on their importance.

</details>
<details>
<summary><h4>2.4 DiffPruning</h4></summary>

DiffPruning aims to achieve parameter efficiency by learning a sparse update of a neural network’s weights. The method introduces a learnable binary mask on the weights, denoted by δ = z ◦ ∆W, where ◦ represents the Hadamard product. This parameter mask is learned during model fine-tuning as part of the regularization objective, which is a differentiable approximation to the L0 norm of the update vector δ. This method requires more memory than traditional fine-tuning, as it involves optimizing all parameters during training in addition to the learnable binary mask.

</details>
<details>
<summary><h4>2.5 Freeze and Reconfigure (FAR)</h4></summary>

The Freeze and Reconfigure (FAR) method selects columns of parameter matrices to prune and reconfigures linear layers into trainable and frozen. In the first stage, the most important rows of parameter matrices are identified for updating. This process is similar to structured pruning and can use any pruning method. In the second stage, the network is reconfigured by splitting each parameter tensor into trainable and frozen components. After training, the parameters can be reconfigured back, removing any inference overhead.

</details>
<details>
<summary><h4>2.6 FishMask</h4></summary>

FishMask is a sparse fine-tuning method that selects the top-p parameters of the model based on their Fisher information. Fisher information measures the amount of information that an observable random variable carries about an unknown parameter of a distribution that models the variable.

</details>
<details>
<summary><h4>2.7 ULMFit</h4></summary>

ULMFit achieves fine-tuning using gradual unfreezing. Instead of fine-tuning all layers at once, which risks catastrophic forgetting, ULMFit gradually unfreezes the model starting from the last layer. The last layer is unfrozen first and all unfrozen layers are fine-tuned for one epoch. Then the next group of frozen layers is unfrozen and fine-tuned and the process is repeated until all layers are fine-tuned until convergence at the last iteration.

</details>
</details>
<details>
<summary><h3>3. Reparametrization-based Methods</h3></summary>

Reparametrization-based methods aim to find the low-rank representation (essentially smaller dimensions) of the updates that will be incorporated into the parameters of a pretrained model for a downstream task. The principle behind this is that neural networks possess equivalent low-dimensional representations.

<details>
<summary><h4>3.1 Intrinsic SAID</h4></summary>

Intrinsic SAID uses the Fastfood transform to reparametrize the update to the model weights. The model weights, which will be added to the pretrained model weight, are learned through the matrices H (Hadamard matrix), G (random diagonal matrix with independent standard normal entries), B (random diagonal matrix with equal probability ±1 entries), and Π (random permutation matrix). After training, the matrix M=HGΠHB is added to the pretrained model weights. This method essentially transforms the model update operation into a more manageable format.

</details>
<details>
<summary><h4>3.2 LoRA</h4></summary>

LoRA, or Low Rank Adaptation of LLM, takes inspiration from IntrinsicSAID and proposes a simpler way to perform low-rank fine-tuning. The update for a weight matrix in LoRA is decomposed into a product of just two low-rank matrices, unlike the Fastfood Transform used in IntrinsicSAID. This simplification reduces the complexity of the update operation and makes it more efficient.

</details>
<details>
<summary><h4>3.3 KronA</h4></summary>

KronA replaces the matrix factorization in LoRA with a matrix factorization through a Kronecker product δW = WA ⊗ WB. This yields a better rank per parameters tradeoff because the Kronecker product maintains the rank of the original matrices being multiplied. In other words, rank(A ⊗ B) = rank A·rank B. This method allows for more efficient use of parameters and maintains the rank properties of the original matrices.

#### AdaLoRA
</details>
<details>
<summary><h4>3.4 AdaLoRA</h4></summary>

AdaLoRA proposes an SVD (Singular Value Decomposition) inspired decomposition of the adapter matrices and develops various importance scores to assess which triplets in the SVD decomposition can be removed. This allows adaptively tuning the ranks of the adapter matrices across layers. This method provides a dynamic way to adjust the rank of the adapter matrices, allowing for more flexibility and efficiency in the fine-tuning process.

</details>
</details>
<details>
<summary><h3>4. Hybrid Methods</h3></summary>

Hybrid methods in parameter-efficient fine-tuning (PEFT) for large language models (LLMs) amalgamate ideas from different PEFT categories to optimize performance while minimizing computational expenses associated with fine-tuning extensive neural networks. They are essentially a harmonious blend of multiple strategies, each contributing its strengths and counteracting the weaknesses of others, thereby leading to enhanced performance and efficiency.

<details>
<summary><h4>4.1 Quantized LoRA (QLoRA)</h4></summary>

Quantized LoRA (QLoRA) is a hybrid method that begins with quantizing the pretrained LLM, followed by standard LoRA training. QLoRA introduces a series of innovative features to conserve memory without compromising performance. These include 4-bit NormalFloat (NF4), a novel data type that is ideally suited for normally distributed weights, Double quantization, a technique to reduce the average memory footprint by quantizing the quantization constants, Paged optimizers, a tool to manage memory spikes.

</details>
<details>
<summary><h4>4.2 SparseAdapter</h4></summary>

The SparseAdapter method employs a large hidden dimension for the added module and prunes about 40% of the values at initialization. While it consistently outperforms its non-sparse counterpart with the same trainable parameter count, it's important to note that the training and inference costs can be higher due to hardware support requirements for sparse tensors and operations. Additionally, calculating the pruning mask for this method may necessitate obtaining gradients for all newly added parameters.

</details>
<details>
<summary><h4>4.3 MAM Adapters</h4></summary>

MAM Adapters is a hybrid approach that combines the concepts of adapters and soft prompting. It capitalizes on the fact that scaled parallel adapters perform better than sequentially-placed adapters, and an adapter placed in parallel to the Feed Forward Network (FFN) outperforms multi-head attention-parallel adapters. Moreover, it utilizes the efficiency of soft prompts in modifying attentions by altering just 0.1% of the parameters.

</details>
<details>
<summary><h4>4.4 UniPELT</h4></summary>

UniPELT is a hybrid method that incorporates LoRA, Prefix-tuning, and Adapters. Specifically, it uses LoRA reparametrization for WQ and WV attention matrices, applies prefix-tuning to keys and values of each layer, and adds adapters after the feed-forward layer of the transformer block. For each of these modules, gating is implemented as a linear layer that projects the module input into a dimension of size one, applies a sigmoid activation, and averages the resulting vector over the sequence length.

</details>
<details>
<summary><h4>4.5 Compacter</h4></summary>

The Compacter method, as proposed by Karimi Mahabadi et al., 2021, leverages the Kronecker product, low-rank matrices, and parameter sharing across layers to generate adapter weights.

</details>
<details>
<summary><h4>4.6 S4</h4></summary>

The S4 method carries out a thorough exploration of diverse combinations of parameter-efficient fine-tuning techniques. Its search space includes dividing consecutive layers into four uneven groups, allocating variable amounts of trainable parameters to each layer, deciding which groups to fine-tune, and determining the PEFT methods to apply to each group.

</details>
</details>

## Conclusion

In conclusion, the pursuit of parameter efficiency in fine-tuning LLMs is a critical aspect of contemporary machine learning research. By successfully leveraging the existing parameters of a pre-trained model and minimizing the addition or modification of new parameters, PEFT methods offer a promising solution to the computational and memory challenges associated with fine-tuning large models. As we continue to push the boundaries of what's possible with machine learning and artificial intelligence, these methods will undoubtedly play a pivotal role in shaping the future of the field.