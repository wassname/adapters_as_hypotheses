Title: CLOVER: Cross-Layer Orthogonal Vectors Pruning and Fine-Tuning

URL Source: https://arxiv.org/pdf/2411.17426

Published Time: Mon, 03 Feb 2025 01:40:12 GMT

Number of Pages: 16

Markdown Content:
# CLOVER: Cross-Layer Orthogonal Vectors Pruning and Fine-Tuning 

Fanxu Meng 1 2 Pingzhi Tang 1 Fan Jiang 1 Muhan Zhang 1 2 

## Abstract 

Decoder-only models generate tokens autoregres-sively by caching key/value vectors, but as the cache grows, inference becomes memory-bound. To address this issue, we introduce CLOVER (Cross-Layer Orthogonal Vectors), a novel ap-proach that treats pairs of attention layers as a set of low-rank decompositions. CLOVER applies Singular Value Decomposition (SVD) to the Q-K

and V -O pairs within each attention head. The resulting singular values can either guide pruning or serve as trainable parameters for efficient fine-tuning of all orthogonal vectors. After pruning or fine-tuning, these values are reintegrated into the model without increasing its parameter count. We apply CLOVER to various models, including GPT-2 XL, DeepSeek-V2-Lite, Whisper-Large-v3, Stable Diffusion XL, and LLaMA-3.2-11B-Vision. Our results demonstrate that CLOVER significantly improves pruning efficiency. For in-stance, the perplexity of pruning 70% of the Q-K

pairs in GPT-2 XL is similar to that of pruning just 8% with vanilla methods. Fine-tuning the sin-gular values further results in a full-rank update, outperforming state-of-the-art methods (LoRA, DoRA, HiRA, and PiSSA) by 7.6%, 5.5%, 3.8%, and 0.7%, respectively, on eight commonsense tasks for LLaMA-2 7B. 

## 1. Introduction 

In recent years, Large Language Models (LLMs) have rapidly evolved into essential tools for productivity (OpenAI, 2024; Anthropic, 2024; Team et al., 2024a). Open-source models (AI@Meta, 2024; Mistral, 2024; Qwen, 2024; Liu et al., 2024b; Team et al., 2024b; Abdin et al., 2024) have also narrowed the performance gap with closed-source mod-els. The success of LLMs is largely attributed to Next Token Prediction (Radford, 2018; Brown et al., 2020), where to-kens are predicted sequentially, with attention computed between each token and all preceding ones. To avoid redun-dant computations, key-value features are cached. However, as model size grows, the overhead of caching becomes sub-stantial, leading to memory and communication bottlenecks. 𝑊 !

## 𝑊 " 𝑊 #

## 𝑊 $

# 𝑥 

# 𝑦 

(a) Multi-Head Attention 𝑈 !" 

## 𝑉 !" 𝑈 #$ 

## 𝑉 #$ 

# 𝑥 

# 𝑦  

> 𝑆 !"
> 𝑆 #$

(b) CLOVER 0.08 0.7 1

> Pruning Ratio
> 3
> 4
> 5
> 6
> 7
> Log of Perplexity
> Vanilla
> CLOVER

(c) Pruning without Training 0.125 0.25 0.375 0.5 0.625  

> 10
> 12
> 14
> 16
> 18
> 20
> 22
> 24
> Perplexity
> Vanilla
> CLOVer
> CLOVer

(d) Fine-Tuning Pruned Model 

Figure 1. (a) We treat the Query-Key and Value-Output layers within a single attention head as a unified structure. (b) Apply SVD to obtain two sets of singular vectors for initializing the Q-K and V-O layers, along with singular values that guide pruning or enable efficient full-rank fine-tuning. (c) This cross-layer orthogo-nalization strategy allows for higher pruning rates. (d) The pruned model maintains strong performance after fine-tuning. 

For instance, a 65B parameter model (Touvron et al., 2023) with 8-bit key-value quantization requires over 86GB of GPU memory to store 512K tokens, exceeding the capacity of a single H100-80GB GPU (Sun et al., 2024). To enable efficient training and inference, we introduce CLOVER (Cross-Layer Orthogonal Vectors), a novel method that orthogonalizes the Query, Key, Value, and Out-put vectors without generating additional transformation matrices. As shown in Figure 1a, we treat the Q-K and V -

O pairs in each attention head as a low-rank decomposition of WQK and WV O . By crossing these layers and perform-1

> arXiv:2411.17426v3 [cs.LG] 31 Jan 2025 CLOVER: Cross-Layer Orthogonal Vectors

ing SVD on WQK and WV O , the Query, Key, Value, and Output vectors become orthogonal within each attention head. Figure 1b illustrates how the resulting singular val-ues can guide pruning or serve as trainable parameters for efficient fine-tuning. After pruning or fine-tuning, these values can be reintegrated into the model without increasing its parameter count. Notably, previous methods, such as SVFT (Lingam et al., 2024), obtain orthogonal vectors by directly performing orthogonal decomposition on the matrix at each layer, which results in an accompanying transfor-mation matrix, doubling the parameter count. In contrast, CLOVER treats the Q-K pairs as transformation matrices for each other, and similarly for the V -O pairs. CLOVER only generates a small set of singular values to guide prun-ing and fine-tuning, which can be merged back into the model without increasing inference costs. 

By orthogonalizing the vectors, we eliminate linear re-dundancy. Attention heads contain numerous non-zero norm vectors. Directly pruning these vectors would degrade performance, but orthogonalizing them allows us to repre-sent the entire attention head’s space using a small set of orthogonal bases. The remaining vectors are nearly zero, making them safe to prune. As shown in Figure 1c, prun-ing an average of 45 vectors in the query-key pair using CLOVER results in a perplexity similar to that of vanilla pruning, which prunes only 5 vectors. Moreover, CLOVER generates a singular value matrix between the Q-K and 

V -O pairs. By updating this matrix during fine-tuning, 

CLOVER learns linear combinations of all orthogonal bases within each attention head. In contrast, PiSSA can only learn from a subset of orthogonal vectors, potentially causing some data projections to approach zero in those di-rections, leading to non-functional adapters during training. As shown in Figure 1d, fine-tuning a very small number of singular values can achieve performance close to that of fine-tuning all attention heads. We summarize the contribution of our paper as follows: • We treat the Q-K and V-O pairs in each attention head as low-rank approximations of WQK and WV O . By performing SVD, we orthogonalize the attention head without adding extra transformation matrices. • This orthogonalization reduces linear redundancy, is compatible with any pruning method, and allows for higher pruning ratios. Pruning 46.42% of the vectors in Whisper’s attention head preserves performance with-out requiring additional training. • CLOVER enables efficient full-rank updates, surpass-ing SOTA methods such as LoRA, DoRA, HiRA, and PiSSA on eight commonsense reasoning tasks across LLaMA 7B/13B, LLaMA-2-7B, and LLaMA-3-8B, with additional analyses highlighting its advantages. 

## 2. Related Work 

LLM Compression To mitigate the high memory de-mands of KV Caches in long-context models, several tech-niques have been proposed. These include reducing se-quence length with linear attention (Katharopoulos et al., 2020; Wang et al., 2020; Peng et al., 2023; Gu & Dao, 2023; De et al., 2024), dynamic token pruning (Fu et al., 2024; Jo & Shin, 2024; Li et al., 2024b), compressing the key-value rank (Shazeer, 2019; Ainslie et al., 2023; Liu et al., 2024a; Yu et al., 2024), and pruning head dimensions (Ashkboos et al., 2024; Xia et al., 2023; Sun et al., 2023). Additional approaches include sharing key-value representations across layers (Sun et al., 2024; Brandon et al., 2024; Liu et al., 2024c; Zuhri et al., 2024) and quantizing KV cache weights and activations (Frantar et al., 2022; Dettmers et al., 2022; Xiao et al., 2023; Liu et al., 2024e; Hooper et al., 2024). Among them, structure pruning is hardware-friendly but can reduce performance when non-zero dimensions are removed (Ma et al., 2023). Fine-tuning can recover some of the lost performance, but it’s computationally expensive. To address this, Parameter Efficient Fine-Tuning (PEFT) methods are used (Guo et al., 2023). 

Parameter Efficient Fine-Tuning. Several strategies have been introduced to minimize fine-tuning parameters while maintaining performance. These include low-rank adaptation (Hu et al., 2021), partial-parameter fine-tuning (Zaken et al., 2021; Lawton et al., 2023; Zhao et al., 2020; Sung et al., 2021; Ansell et al., 2021; Xu et al., 2021; Guo et al., 2020; Fu et al., 2023), soft prompt fine-tuning (Ham-bardzumyan et al., 2021; Lester et al., 2021; Li & Liang, 2021; Liu et al., 2023b; Vu et al., 2021; Asai et al., 2022; Wang et al., 2023), and sparse matrix fine-tuning (Qiu et al., 2023; Liu et al., 2023a; Yuan et al., 2024). Among these, LoRA is widely used due to its simplicity and effectiveness, with recent works enhancing it further (Zhang et al., 2023; Zi et al., 2023; Liu et al., 2024d; Zhao et al., 2024; Jiang et al., 2024). PiSSA (Meng et al., 2024) improves conver-gence speed by initializing adapters with principal singular values and vectors, also reducing quantization error (Wang et al., 2024a;b; Li et al., 2024a). However, PiSSA is limited by its use of a fixed set of orthogonal bases. SVFT (Lingam et al., 2024) directly applies Singular Value Decomposition (SVD) to the original matrix, but this increases the number of parameters, raising computational overhead and reducing efficiency. The CLOVER method addresses these issues by treating the Query-Key pairs in each attention head as low-rank matrices. Using orthogonal decomposition, CLOVER eliminates the need for additional transformation matrices. Instead, it leverages a small set of singular values to linearly combine orthogonal vectors, making the approach more parameter-efficient. After fine-tuning, the adapter can be smoothly reintegrated into the original matrix structure. 2CLOVER: Cross-Layer Orthogonal Vectors 

## 3. CLOVER: Cross-Layer Orthogonal Vectors 

Below is a step-by-step explanation of CLOVER method and explain why it can update orthogonal decompose the Query, Key, Value, Output layers in Multi-Head Attention without need introduce any transfer matrix. We mainly use the computation of the Q-K pair in as an example. Then extended to the V -O pair. 

Multi-Head Self-Attention Setup. In a multi-head self-attention mechanism with H heads, each head h ∈{1, . . . , H } computes an attention score as: attn (Qh, K h) = softmax 

 QhK⊤

> h√d



,

where H is the number of attention heads, d is the dimen-sionality of each head, X ∈ Rn×D is the input matrix ( n

is the sequence length, D is the total hidden dimension), 

Qh, K h ∈ Rn×d are the query and key representations for head h, WQ, W K ∈ RD×H×d are weights for projecting the input X into queries and keys. Specifically, the queries and keys for head h are obtained by multiplying X with the corresponding “slice” of WQ and 

WK , respectively: 

Qh = X W [: ,h, :]  

> Q

, Kh = X W [: ,h, :]  

> K

.

Cross Layers Merging. Substituting Qh and Kh into 

QhK⊤ 

> h

, we have: 

QhK⊤ 

> h

= X W [: ,h, :] 

> Q

 W [: ,h, :] 

> K

⊤X⊤.

Notice that the original weights W [: ,h, :]  

> Q

and W [: ,h, :]  

> K

are each in RD×d, once multiplied together, the resulting ma-trix W hQK = W [: ,h, :] 

> Q

 W [: ,h, :] 

> K

⊤ has dimension D × D.Since d ≪ D, using W hQK directly in computations—or storing it as trainable parameters—would be highly ineffi-cient, limiting the use cases of such parameter merging. 

Cross Layers Orthogonal Decomposition To address the large size of W hQK , we factorize W hQK via SVD: 

W hQK = U hQK ShQK V hQK ,

where U hQK is a D × D orthogonal matrix, ShQK is a D × D

diagonal matrix of singular values, V hQK is another D × D

orthogonal matrix. Since W [: ,h, :]  

> Q

and W [: ,h, :]  

> K

each have shape RD×d, the rank of W hQK is at most d. Thus the actual non-zero singular values in ShQK are at most d. We can truncate the SVD to keep only the top-r singular values without loss: 

W hQK = U hQK [: , : r] ShQK [: r, : r]  V hQK [: , : r]⊤,

where r ≤ d.The process can be easily applied to WV and WO , as intro-duced in Appendix A.1. 

CLOVER for Pruning After performing SVD, we can rewrite the weight matrix W hQK as follows: 

W hQK = U hQK [: , : r] ShQK [: r, : r]

| {z } 

> ˜Uh∈RD×r

 V hQK [: , : r]⊤

| {z } 

> ˜Vh∈Rr×D

.

Instead of storing the full matrices W hQ and W hK ∈ RD×d,we store the smaller factors ˜U h ∈ RD×r and ˜V h ∈ Rr×D ,which can be significantly smaller than the original matrix since r ≤ d ≪ D. This leads to a reduction in memory usage and computational cost. Additionally, we can prune 

singular values (and their corresponding singular vectors) below a chosen threshold. This further reduces the parame-ter count and computational overhead. 

CLOVER for Fine-Tuning CLOVER can be used not only for pruning, but also for parameter-efficient fine-tuning. We freeze the matrices U hQK [: , : r] and V hQK [: , : r], and only fine-tune the singular values ShQK [: r, : r].In contrast to SVFT, which factorizes the entire weight ma-trices WQ, W K , W V , W O ∈ RD×D individually, CLOVER factorizes the merged weights W hQK and W hOV within each attention head, significantly reducing the parameters. By applying SVD factorization within each attention head, CLOVER constrains the effective rank of the cross-layer matrix to d. As a result, the tunable matrix SQK has a size bounded by RH×d×d (considering all heads). In compari-son, SVFT requires factorizing large matrices each into three components ( U, S, V ∈ RD×D ), leading to a significant in-crease in parameter count and computational overhead, even with sparse updates for the singular values S.For example, consider the LLaMA 2-7B model with H =32 attention heads and a head dimension of d = 128 . By factorizing each head separately, the largest size for SQK 

is O(32 × 128 × 128) , which is significantly smaller than factorizing a R4096 ×4096 matrix. This makes CLOVER’s parameter efficiency comparable to that of a LoRA config-uration with rank 32, as shown in Appendix A.2, but with additional potential for pruning. 3CLOVER: Cross-Layer Orthogonal Vectors  

> Table 1. Pruning GPT-2-XL’s attention layers with CLOVER and vanilla pruning at various ratios, evaluating perplexity on Wikitext2 (lower is better), and fine-tuning on OpenWebText with different token budgets. The base model’s perplexity is 14.78.

Pruning Ratio w/o Training Perplexity( ↓) 66M Tokens Perplexity ( ↓) 131M Tokens Perplexity ( ↓)

Vanilla CLOVER Vanilla CLOVER CLOVER † Vanilla CLOVER CLOVER †

12.5% 33.76 15.89 16.04 15.45 15.67 16.38 15.77 15.42 

25.0% 78.36 17.45 16.93 15.70 15.89 17.07 16.05 15.75 

37.5% 159.4 20.95 18.17 16.17 16.60 18.14 16.48 16.41 

50.0% 338.9 35.12 20.45 17.22 17.63 19.02 17.13 17.71 62.5% 538.5 85.25 24.65 19.32 20.64 21.44 18.40 20.39 75.0% 708.8 187.4 36.04 24.65 29.28 27.22 20.99 28.44 

## 4. Experiments 

As detailed in Section 3, CLOVER is highly effective for both pruning and fine-tuning. We presents a series of experi-ments to validate these capabilities. In Section 4.1, we com-pare CLOVER with Vanilla pruning on a GPT-2-XL model (Radford et al., 2019). CLOVER results in less performance degradation, while Vanilla pruning significantly harms the model’s performance, making recovery difficult even with fine-tuning. In Section 4.2, we conduct fine-tuning experi-ments on eight commonsense tasks, comparing CLOVER with state-of-the-art methods. The results show the effec-tiveness of CLOVER’s linear combinations of all orthogonal vectors. In Section 4.3, CLOVER is applied to various mod-els. We visualize how it removes linear redundancy between vectors, enabling more efficient pruning. In Section 4.4, we demonstrate CLOVER’s ability to perform significant prun-ing on the Whisper model, which exhibits substantial linear redundancy, without requiring fine-tuning. In Section 4.5, we explain the importance of learning from all the orthogo-nal vectors by analyzing the projection of data features onto different directions in the model. In Section 4.6, we confirm CLOVER’s full-rank update capability by visualizing the singular value distribution of ∆W from various methods. Finally, in Section 4.7, we show how CLOVER fine-tunes the model using its inherent properties, without introducing “intrusive dimension” like LoRA, which may risk model degradation (Shuttleworth et al., 2024). 

4.1. CLOVER for Large Ratio Pruning 

Due to the need to compute attention between each token and all preceding tokens, compressing atten-tion—particularly the key-value layers—is crucial, despite the larger number of parameters in the MLP. CLOVER rep-resents each attention head with a small number of vectors. Since it only modifies the initialization, it can be combined with any other pruning technique. This paper validates the proposed method using basic structured pruning on GPT-2-XL, rather than targeting state-of-the-art performance. We initialize GPT-2-XL with CLOVER, then prune small singu-lar values based on their magnitude. To maintain inference efficiency, we apply the same pruning rate across all layers, removing a fixed percentage of the smallest singular vectors. The singular values, S, are then merged into the U and V

matrices. For comparison, we also prune without CLOVER orthogonalization, using L2-norms for pruning. After prun-ing, we evaluate perplexity on the WikiText-2 (Merity et al., 2016) dataset. We then fine-tune the pruned models on the OpenWebText (Gokaslan & Cohen, 2019) dataset following nanoGPT 1. To minimize disruption to the original model, we fine-tune only the pruned attention layers, leaving the MLP, embedding layers, and LM head unchanged. In the CLOVER † case, after pruning, S is not immediately merged into the U and V matrices but is used for parameter-efficient fine-tuning, with the merging occurring afterward. We ad-just the learning rate from 6e-4 to 6e-3 and remove weight decay, while keeping other hyperparameters consistent with the other two methods. Based on Table 1, CLOVER causes less damage to the model than Vanilla pruning, as it transfers functionality into fewer orthogonal bases. For example, pruning 50% of the parameters without further fine-tuning, CLOVER’s perplex-ity only increases by 1.38 ×, while Vanilla pruning increases by 21.9 ×. After fine-tuning, CLOVER’s performance far exceeds that of Vanilla pruning. Due to its lower model disruption, CLOVER requires fewer tokens for fine-tuning to restore performance (e.g., perplexity with 66M tokens is close to that with 131M tokens), whereas Vanilla pruning needs more tokens, resulting in higher costs and potential degradation in out-of-domain tasks. Furthermore, by fine-tuning only the singular values from the SVD decomposition and the attention layer biases, CLOVER achieves recovery with fewer training resources and parameter changes. At lower pruning rates, CLOVER even outperforms full atten-tion layer training. However, when pruning rates are too high, accuracy loss becomes significant, and the available parameters for fine-tuning become insufficient (e.g., at 75% pruning, only 0.15% of the original attention layer parame-ters are updated). 

> 1https://github.com/karpathy/nanoGPT

4CLOVER: Cross-Layer Orthogonal Vectors  

> Table 2. Accuracy comparison of LLaMA 7B/13B, LLaMA2 7B, and LLaMA3 8B with various PEFT methods on eight commonsense reasoning datasets. Results of LoRA and DoRA are taken from (Liu et al., 2024d). Results of HiRA are taken from (Anonymous, 2025).

Model Method Params BoolQ PIQA SIQA Hella Swag Wino Grande ARC-e ARC-c OBQA Avg. 

ChatGPT - - 73.1 85.4 68.5 78.5 66.1 89.8 79.9 74.8 77.0 LLaMA-7B Series 0.99% 63.0 79.2 76.3 67.9 75.7 74.5 57.1 72.4 70.8 Parallel 3.54% 67.9 76.4 78.8 69.8 78.9 73.7 57.3 75.2 72.2 LoRA 0.83% 68.9 80.7 77.4 78.1 78.8 77.8 61.3 74.8 74.7 DoRA 0.84% 69.7 83.4 78.6 87.2 81.0 81.9 66.2 79.2 78.4 PiSSA 0.83% 74.1 85.4 81.5 94.0 85.0 85.6 72.1 84.2 82.7 CLOVER 0.83% 72.9 86.34 82.1 94.9 85.4 87.5 74.4 86.4 83.7 

LLaMA-13B Series 0.80% 71.8 83 79.2 88.1 82.4 82.5 67.3 81.8 79.5 Parallel 2.89% 72.5 84.9 79.8 92.1 84.7 84.2 71.2 82.4 81.4 LoRA 0.67% 72.1 83.5 80.5 90.5 83.7 82.8 68.3 82.4 80.5 DoRA 0.68% 72.4 84.9 81.5 92.4 84.2 84.2 69.6 82.8 81.5 PiSSA 0.67% 74.6 88.0 82.9 95.5 87.0 90.3 77.2 88.2 85.4 CLOVER 0.67% 75.2 88.4 83.1 96.0 87.8 89.7 79.3 89.8 86.2 

LLaMA2-7B LoRA 0.83% 69.8 79.9 79.5 83.6 82.6 79.8 64.7 81.0 77.6 DoRA 0.84% 71.8 83.7 76.0 89.1 82.6 83.7 68.2 82.4 79.7 HiRA 0.83% 71.2 83.4 79.5 88.1 84.0 86.7 73.8 84.6 81.4 PiSSA 0.83% 75.0 87.0 81.6 95.0 86.5 88.5 75.9 86.4 84.5 CLOVER 0.83% 75.0 86.4 82.0 95.1 87.5 89.6 76.6 89.4 85.2 

LLaMA3-8B LoRA 0.70% 70.8 85.2 79.9 91.7 84.3 84.2 71.2 79.0 80.8 DoRA 0.71% 74.6 89.3 79.9 95.5 85.6 90.5 80.4 85.8 85.2 HiRA 0.70% 75.4 89.7 81.2 95.4 87.7 93.3 82.9 88.3 86.7 PiSSA 0.70% 77.2 90.0 82.9 96.6 88.4 93.6 82.4 87.4 87.3 CLOVER 0.47% 76.4 89.3 82.1 96.9 89.9 93.6 84.5 90.6 87.9 

4.2. CLOVER for Full-Rank Fine-Tuning 

In this section, we evaluate CLOVER against LoRA (Hu et al., 2021), DoRA (Liu et al., 2024d), HiRA (Anonymous, 2025), and PiSSA (Meng et al., 2024) on commonsense reasoning tasks, excluding SVFT (Lingam et al., 2024) due to its significant overhead. The tasks are divided into eight sub-tasks, as outlined in Table 4. Following the DoRA setup, we fine-tune the Commonsense-170k dataset and evaluate each sub-task’s test set. We apply orthogonal de-composition to the Value-Output and fine-tune the resulting singular value matrix. Due to the non-linear RoPE(Su et al., 2024) operation between the query and key, we perform orthogonal decomposition in the Key layer and fine-tune the transition matrix. Similarly, we treat the 64 consecu-tive dimensions in the MLP.Up layer as a head, applying orthogonal decomposition and updating the transition ma-trix. The learnable parameters of LLaMA 7B/13B (Touvron et al., 2023) and LLaMA-2-7B (AI@Meta, 2023) match LoRA/DoRA/HiRA/PiSSA with rank 32 updates. LLaMA-3-8B (AI@Meta, 2024) has 2/3 of the trainable parameters compared to the other models. For a fair comparison, we use the hyperparameters from DoRA (3 epochs, batch size 16, linear scheduler learning rate). We adjusted the learning rate based on DoRA’s approach and found that CLOVER per-forms best with lr=1e-4, which we applied across all models. PiSSA was trained using the same hyperparameters, but with a learning rate of 2e-5, as specified in its original pa-per. Due to the stable performance of PiSSA and CLOVER during training, we did not perform validation every 80 it-erations, as done in DoRA, to select the best-performing model on the validation set for testing. Instead, we trained for the full 3 epochs and used the final model for testing. HiRA’s results are taken directly from its original paper, while the other results are sourced from DoRA’s paper. Ta-ble 2 demonstrates that CLOVER consistently outperforms all other methods across all models and tasks. Specifically, on LLaMA 7B, CLOVER outperforms LoRA, DoRA, and PiSSA by 9%, 5.3%, and 1%, respectively. On LLaMA 13B, CLOVER outperforms these methods by 5.7%, 4.7%, and 0.8%. On LLaMA-2-7B, CLOVER surpasses LoRA, DoRA, HiRA, and PiSSA by 7.6%, 5.5%, 3.8%, and 0.7%. Even on LLaMA-3-8B, with fewer trainable parameters, CLOVER outperforms by 7.1%, 2.7%, 1.2%, and 0.6%. CLOVER leads in most sub-tasks and ranks second in a few. 5CLOVER: Cross-Layer Orthogonal Vectors 0 50 100   

> 0.0
> 0.5
> 1.0
> 1.5
> 2.0
> WQ WK
> (10, 0.93)
> Vanilla
> CLOVER
> 050 100
> Sorted Dimensions
> 0.0
> 0.2
> 0.4
> 0.6
> WV WTO
> (72, 0.13)
> Vanilla
> CLOVER

(a) DeepSeek-V2-Lite 0 25 50 75     

> 0
> 1
> 2
> 3
> 4
> 5
> WQ WK
> (1, 2.66)
> Vanilla
> CLOVER
> 025 50 75
> Sorted Dimensions
> 0.0
> 0.2
> 0.4
> 0.6
> WV WTO
> (27, 0.29)
> Vanilla
> CLOVER

(b) Llama-3.2-Vision 0 20 40 60     

> 0.00
> 0.25
> 0.50
> 0.75
> 1.00
> 1.25
> 1.50
> WQ WK
> (7, 0.63)
> Vanilla
> CLOVER
> 020 40 60
> Sorted Dimensions
> 0.0
> 0.2
> 0.4
> 0.6
> 0.8
> WV WTO
> (35, 0.16)
> Vanilla
> CLOVER

(c) Whisper-Large-v3 0 20 40 60     

> 1
> 2
> 3
> WQ WK
> (29, 1.25)
> Vanilla
> CLOVER
> 020 40 60
> Sorted Dimensions
> 0.2
> 0.4
> 0.6
> 0.8
> WV WTO
> (26, 0.46)
> Vanilla
> CLOVER

(d) SDXL 0 50 100    

> 0.00
> 0.02
> 0.04
> 0.06
> 0.08
> WQ WK
> (9, 0.04)
> Vanilla
> CLOVER
> 050 100
> Sorted Dimensions
> 0.00
> 0.02
> 0.04
> 0.06
> 0.08
> WV WTO
> (44, 0.02)
> Vanilla
> CLOVER

(e) CLIP-ViT-BigG 

Figure 2. CLOVER (orange) uses fewer orthogonal basis vectors than Vanilla Pruning (blue) to span the attention head space. The first row shows the importance of Q-K dimensions, and the second row shows V-O dimensions. After the red dot, CLOVER’s importance is lower, and pruning these vectors results in less performance loss. 

4.3. CLOVER Removal Redundant Vectors 

CLOVER achieves a higher pruning ratio due to the sig-nificant linear redundancy present in the model. By repre-senting the entire attention head with only a small number of orthogonal vectors, CLOVER effectively removes this redundancy. To illustrate the advantages of CLOVER in eliminating linear redundancy, we apply it to a variety range of models, including the large language model DeepSeek-V2-Lite (DeepSeek-AI, 2024), the multimodal automatic speech recognition and speech translation model Whisper-Large-v3 (Radford et al., 2023), the multimodal instruction-tuned image reasoning generative models LLaMA-3.2-11B-Vision (AI@Meta, 2024), the image encoder CLIP-ViT-bigG (Cherti et al., 2022), and the image generation model Stable Diffusion XL (Podell et al., 2023). We compute the 

L2 norm for each dimension (equal to singular values) in both the Q-K pair and the V-O pair, sorting the values in descending order within each attention head for better visu-alization. For comparison, we also perform Vanilla Pruning, which does not utilize CLOVER initialization but instead sorts directly based on the L2 norm. Figure 2 showcases the first attention head from the first layer of each model. In the first column of the figure, depict-ing the Q-K norm, we observe that in the original model, the importance of each dimension is relatively balanced (e.g. Figure 2c). This balanced distribution is a result of the linear redundancy, where different directions are inter-twined, making it challenging to prune individual directions without negatively affecting the model’s performance. How-ever, after applying CLOVER’s orthogonal decomposition, only a small number of orthogonal bases on the left side exhibit significantly large norms. These vectors span al-most the entire attention head’s space, and the remaining vectors have norms that approach zero, indicating that they are already represented by the dominant singular vectors and can be pruned without loss of performance. Beyond the red intersection point, CLOVER’s remaining vectors exhibit consistently lower importance than those in Vanilla Pruning, meaning pruning these vectors results in less per-formance degradation. This demonstrates why CLOVER enables a higher pruning ratio. A similar trend is observed for the V-O pair, although the model’s inherent sparsity is less pronounced than in the Q-K pair, making the effect less noticeable. Still, in most models, pruning half of the vectors has a smaller impact on performance compared to Vanilla Pruning. Notably, in CLIP-ViT-bigG (Figure 2e), a proportion of the vectors already have a norm of zero, allowing for safe pruning. 

4.4. CLOVER for Training-Free Pruning 

As demonstrated by the prominent low-rank properties in Figure 2c, we applied pruning to the Whisper-large-v3 model (Radford et al., 2023). To intuitively highlight the effectiveness of CLOVER pruning, we present an example using an audio input from the LibriSpeech Long dataset (Gandhi et al., 2023). For reference, the waveform of this input is shown in Figure 3, and the corresponding target translation script is provided in Appendix A.4. After applying CLOVER to orthogonalize the vec-tors, we pruned vectors with magnitudes close to zero (∥WQ∥∥ WK ∥ ≤ 5 × 10 −3 and ∥WV ∥∥ W ⊤ 

> O

∥ ≤ 6 × 10 −3). This pruning achieved ratios of 56.01% and 36.82% for the parameters in Q-K Pair and V -O Pair, respectively. Re-markably, the model’s output remains nearly unchanged, with only one error, which has been highlighted in the text using strikethrough and red for clarity: 6CLOVER: Cross-Layer Orthogonal Vectors 0 2e5 4e5 6e5 8e5 10e5 

> Samples
> 0.5
> 0.0
> 0.5
> Amplitude

Figure 3. An audio waveform from the librispeech dataset. 

Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel. Nor is Mr. Quilter’s manner less interesting than his matter. He tells us that at this festive season of the year, with Christmas and roast beef looming before us, similes drawn from eating and its results occur most readily to the mind. He has grave doubts whether Sir Frederick Layton’s work is really Greek after all, and can discover in it but little of rocky Ithaca. Linnell’s pictures are a sort of Up Guards and Adam paintings, and Mason’s exquisite idles are as national as a jingo poem. Mr. Birkett Foster’s landscapes smile at one much in the same way that Mr. Carker used to flash his teeth. And , and Mr. John Collier gives his sitter a cheerful slap on the back before he says, like a shampooer in a Turkish bath, next man. 

In contrast, using a vanilla pruning method with the same pruning ratio, the model completely fails to produce valid outputs: 

... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... ... 

This example validates our earlier claim that straightfor-ward pruning of non-zero dimensions can lead to accumu-lated loss. In contrast, CLOVER effectively eliminates lin-ear redundancy, enabling a significantly higher pruning ra-tio. When the linear redundancy is sufficiently pronounced, CLOVER can even achieve a high pruning ratio without the need for fine-tuning to recover performance. 

4.5. Necessity of Full-Direction Fine-Tuning 

Besides pruning with a large ratio, CLOVER is capable of learning linear combinations of all orthogonal vectors within each attention head. This capability allows CLOVER to resemble full-parameter fine-tuning more closely. To highlight the advantages of updating all orthogonal bases, we randomly sampled 16 instances from the Commonsense dataset, fed them into the model, and performed SVD to the model. We then recorded the projection magnitudes of input features along all orthogonal directions. Figure 4 visualizes the results for the middle layer, revealing the following insights: 1) Without accounting for the scaling effect of singular val-ues, the projection magnitude along the principal singular vector consistently exceeds that in other directions. This ob-servation supports PiSSA’s approach, which updates based Top 256 

> 10%
> Next 256
> 7%
> Remaining
> 76%
> Bottom 256
> 7%

(a) PiSSA Top 256  

> 6%
> Next 256
> 7%
> Remaining
> 81%
> Bottom 256
> 6%

(b) LoRA Top 256  

> 18%
> Next 256 10%
> Remaining
> 65%
> Bottom 256
> 7%

(c) PiSSA with Singular Value 100% (d) CLOVER 

Figure 4. Proportion of data projections across different compo-nents in random directions (LoRA) versus orthogonal directions (PiSSA), as well as all orthogonal directions (CLOVER). 

on the principal singular values and vectors, leading to im-proved training performance. In contrast, LoRA projects in random directions, resulting in uniform projection magni-tudes across all directions. 2) The singular values in the original model reflect the im-portance of each direction in the pretraining task. The model amplifies the components along directions with larger sin-gular values and suppresses those along smaller singular values. Therefore, it is crucial to consider the scaling effect of singular values. As shown in Figure 4c, the projection magnitude along the principal singular vector direction in-creases to 18%. 3) While more data projections align with the principal singular vector at higher ranks, 82% of the feature compo-nents are still projected onto other directions. In extreme cases, if a task is entirely orthogonal to the vectors used by PiSSA, training on such a task may result in zero gradients, thereby limiting its learning capacity. Under the same rank constraint, 94% of the feature components in LoRA are pro-jected outside the LoRA adapter, making it more susceptible to the zero-gradient problem. Since CLOVER updates across all orthogonal directions, as shown in Figure 4d it effectively mitigates this issue. Con-sequently, CLOVER outperforms both LoRA and PiSSA in multi-task learning, even when using the same or fewer learnable parameters (Section 4.2). 7CLOVER: Cross-Layer Orthogonal Vectors 

4.6. Visualizing Rank Updates 

To demonstrate CLOVER achieves full-rank updates, we multiply the updated singular values with their correspond-ing singular vectors and perform SVD on the base model (SQK applied to the Key layer, SV O to the Value layer, and 

SU D to the Up layer). We take LoRA, and Full Fine-tuning for comparing. Figure 5 shows the singular value of the middle layer in LLaMA-2-7B, revealing that CLOVER and Full Fine-tuning achieve full-rank updates, while LoRA is constrained by its low-rank design. 0 1000 2000 3000 4000 

> 0.0
> 0.5
> 1.0
> 1.5
> 2.0

(a) Full Fine-Tuning 0 1000 2000 3000 4000  

> 0
> 2
> 4
> 6

(b) LoRA 0 1000 2000 3000 4000  

> 0.0
> 0.2
> 0.4
> 0.6
> 0.8

(c) CLOVER 

Figure 5. ∆W is low rank in LoRA, while full rank for Full-Fine-Tuning and CLOVER. 

4.7. CLOVER Avoids Intrusive Dimensions 

Recent research (Shuttleworth et al., 2024) has highlighted an issue with LoRA, referred to as the “intrusive dimensions” phenomenon. As illustrated in Figure 6b, LoRA introduces new random directions into the model, which possess large magnitudes and thus precede all the original singular vectors. The study suggests that these “intrusive dimensions” can degrade the model’s performance, exacerbating catastrophic forgetting during continual learning with LoRA. In contrast, CLOVER addresses this issue by fixing all orthogonal bases and updating only the vector combinations. As a result, the changes introduced by CLOVER fine-tuning closely resemble those generated by full parameter fine-tuning, as shown in Figure 6a and Figure 6c. 0 100 200 300 400 

> 0
> 100
> 200
> 300
> 400

(a) Full Fine-Tuning 0 100 200 300 400  

> 0
> 100
> 200
> 300
> 400

(b) LoRA 0 100 200 300 400  

> 0
> 100
> 200
> 300
> 400

(c) CLOVER 

Figure 6. Intruder dimensions phenomenal in LoRA, which does not exist in Full Fine-Tuning and CLOVER. 

## 5. Conclusion and Limitations 

In this paper, we introduce Cross-Layer Orthogonal Vectors (CLVOER), a method that orthogonalizes vectors within attention heads without requiring additional transformation matrices. This orthogonalization process condenses effec-tive parameters into fewer vectors, improving the pruning ratio. By fine-tuning the singular values obtained through orthogonalization, CLVOER learns linear combinations of orthogonal bases, enabling full-rank updates. When applied to prune 50% of the attention head parameters in GPT-2XL, CLVOER results in a perplexity that is just one-tenth of that achieved by standard pruning methods. For Whisper-Large-v3, CLVOER removes 46.42% of the parameters without fine-tuning, while preserving model performance. Furthermore, when used for fine-tuning, CLVOER outper-forms state-of-the-art methods such as LoRA, DoRA, HiRA, and PiSSA, achieving superior results with equal or fewer trainable parameters. We also demonstrate how CLVOER removes linear redundancy to facilitate pruning and discuss the necessity of fine-tuning across all orthogonal bases. Vi-sual comparisons of models fine-tuned with different meth-ods further illustrate its effectiveness. Despite its advantages, CLVOER has some limitations. When nonlinear operations are present between Q-K or V-O pairs (such as with the widely-used RoPE (Su et al., 2024)), cross-layer orthogonalization is not feasible. In these cases, we instead perform head-wise orthogonalization within the Key layer during fine-tuning. Fortunately, CLVOER Fine-Tuning can apply intra-layer attention head orthogo-nalization, while CLOVER Pruning remains applicable to many popular models, including DeepSeek (DeepSeek-AI, 2024; Liu et al., 2024b)(which uses Decoupled RoPE), ViT and SDXL (which use absolute positional encoding), and BLOOM (Le Scao et al., 2023) (which employs Alibi rela-tive positional encoding (Press et al., 2021)). Additionally, as a newly proposed method, our current evaluation fo-cuses primarily on basic pruning tasks and does not include comparisons with other state-of-the-art pruning techniques. However, because CLVOER does not alter the model struc-ture and only updates the initialization method, it can be combined with existing pruning methods to further enhance their effectiveness. As a novel technique, CLVOER holds considerable promise for future applications. For instance, it could be combined with quantization methods to eliminate outliers, guide prun-ing and fine-tuning based on data feature directions, or even inspire new model architectures. 8CLOVER: Cross-Layer Orthogonal Vectors 

## Impact Statement 

This paper proposes a cross-layer orthogonal initialization method to guide model pruning and efficient fine-tuning, of-fering valuable insights for the application and development of large models. Both application directions aim to reduce training and inference costs, lower computational overhead, decrease power consumption, and minimize carbon emis-sions. 

## References 

Abdin, M., Aneja, J., Behl, H., Bubeck, S., Eldan, R., Gunasekar, S., Harrison, M., Hewett, R. J., Javaheripi, M., Kauffmann, P., et al. Phi-4 technical report. arXiv preprint arXiv:2412.08905, 2024. AI@Meta. Llama 2: Open foundation and fine-tuned chat models. CoRR , abs/2307.09288, 2023. doi: 10. 48550/arXiv.2307.09288. URL https://doi.org/ 10.48550/arXiv.2307.09288 .AI@Meta. Llama 3 model card, 2024. URL 

https://github.com/meta-llama/llama3/ blob/main/MODEL_CARD.md .Ainslie, J., Lee-Thorp, J., de Jong, M., Zemlyanskiy, Y., Lebr ´on, F., and Sanghai, S. Gqa: Training generalized multi-query transformer models from multi-head check-points. arXiv preprint arXiv:2305.13245, 2023. Anonymous. HiRA: Parameter-efficient hadamard high-rank adaptation for large language models. In The Thirteenth International Conference on Learning Representations , 2025. URL https://openreview. net/forum?id=TwJrTz9cRS .Ansell, A., Ponti, E. M., Korhonen, A., and Vuli ´c, I. Composable sparse fine-tuning for cross-lingual trans-fer. arXiv preprint arXiv:2110.07560, 2021. Anthropic. Claude 3.5 sonnet, 2024. URL https://www. anthropic.com/news/claude-3-5-sonnet .Asai, A., Salehi, M., Peters, M. E., and Hajishirzi, H. Attempt: Parameter-efficient multi-task tuning via at-tentional mixtures of soft prompts. arXiv preprint arXiv:2205.11961, 2022. Ashkboos, S., Croci, M. L., Nascimento, M. G. d., Hoefler, T., and Hensman, J. Slicegpt: Compress large language models by deleting rows and columns. arXiv preprint arXiv:2401.15024, 2024. Bisk, Y., Zellers, R., Gao, J., Choi, Y., et al. Piqa: Reason-ing about physical commonsense in natural language. In Proceedings of the AAAI conference on artificial intelligence, volume 34, pp. 7432–7439, 2020. Brandon, W., Mishra, M., Nrusimha, A., Panda, R., and Kelly, J. R. Reducing transformer key-value cache size with cross-layer attention. arXiv preprint arXiv:2405.12981, 2024. Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. Advances in neural information processing systems , 33: 1877–1901, 2020. Cherti, M., Beaumont, R., Wightman, R., Wortsman, M., Ilharco, G., Gordon, C., Schuhmann, C., Schmidt, L., and Jitsev, J. Reproducible scaling laws for contrastive language-image learning. arXiv preprint arXiv:2212.07143, 2022. Clark, C., Lee, K., Chang, M.-W., Kwiatkowski, T., Collins, M., and Toutanova, K. Boolq: Exploring the surpris-ing difficulty of natural yes/no questions. arXiv preprint arXiv:1905.10044, 2019. Clark, P., Cowhey, I., Etzioni, O., Khot, T., Sabharwal, A., Schoenick, C., and Tafjord, O. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv preprint arXiv:1803.05457, 2018. De, S., Smith, S. L., Fernando, A., Botev, A., Cristian-Muraru, G., Gu, A., Haroun, R., Berrada, L., Chen, Y., Srinivasan, S., et al. Griffin: Mixing gated linear recur-rences with local attention for efficient language models. arXiv preprint arXiv:2402.19427, 2024. DeepSeek-AI. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model. CoRR ,abs/2405.04434, 2024. URL https://doi.org/10. 48550/arXiv.2405.04434 .Dettmers, T., Lewis, M., Belkada, Y., and Zettlemoyer, L. Gpt3. int8 (): 8-bit matrix multiplication for transform-ers at scale. Advances in Neural Information Processing Systems, 35:30318–30332, 2022. Frantar, E., Ashkboos, S., Hoefler, T., and Alistarh, D. Gptq: Accurate post-training quantization for generative pre-trained transformers. arXiv preprint arXiv:2210.17323, 2022. Fu, Q., Cho, M., Merth, T., Mehta, S., Rastegari, M., and Najibi, M. Lazyllm: Dynamic token pruning for efficient long context llm inference. arXiv preprint arXiv:2407.14057, 2024. Fu, Z., Yang, H., So, A. M.-C., Lam, W., Bing, L., and Collier, N. On the effectiveness of parameter-efficient fine-tuning. In Proceedings of the AAAI Conference on Artificial Intelligence , volume 37, pp. 12799–12807, 2023. 9CLOVER: Cross-Layer Orthogonal Vectors 

Gandhi, S., von Platen, P., and Rush, A. M. Distil-whisper: Robust knowledge distillation via large-scale pseudo la-belling. arXiv preprint arXiv:2311.00430, 2023. Gokaslan, A. and Cohen, V. Openwebtext cor-pus. http://Skylion007.github.io/ OpenWebTextCorpus , 2019. Gu, A. and Dao, T. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023. Guo, D., Rush, A. M., and Kim, Y. Parameter-efficient transfer learning with diff pruning. arXiv preprint arXiv:2012.07463, 2020. Guo, S., Xu, J., Zhang, L. L., and Yang, M. Com-presso: Structured pruning with collaborative prompting learns compact large language models. arXiv preprint arXiv:2310.05015, 2023. Hambardzumyan, K., Khachatrian, H., and May, J. Warp: Word-level adversarial reprogramming. arXiv preprint arXiv:2101.00121, 2021. Hooper, C., Kim, S., Mohammadzadeh, H., Mahoney, M. W., Shao, Y. S., Keutzer, K., and Gholami, A. Kvquant: Towards 10 million context length llm in-ference with kv cache quantization. arXiv preprint arXiv:2401.18079, 2024. Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. Lora: Low-rank adaptation of large language models. arXiv preprint arXiv:2106.09685, 2021. Jiang, T., Huang, S., Luo, S., Zhang, Z., Huang, H., Wei, F., Deng, W., Sun, F., Zhang, Q., Wang, D., et al. Mora: High-rank updating for parameter-efficient fine-tuning. arXiv preprint arXiv:2405.12130, 2024. Jo, H.-r. and Shin, D. A2sf: Accumulative attention scoring with forgetting factor for token pruning in transformer decoder. arXiv preprint arXiv:2407.20485, 2024. Katharopoulos, A., Vyas, A., Pappas, N., and Fleuret, F. Transformers are rnns: Fast autoregressive transform-ers with linear attention. In International conference on machine learning, pp. 5156–5165. PMLR, 2020. Lawton, N., Kumar, A., Thattai, G., Galstyan, A., and Steeg, G. V. Neural architecture search for parameter-efficient fine-tuning of large pre-trained language models. arXiv preprint arXiv:2305.16597, 2023. Le Scao, T., Fan, A., Akiki, C., Pavlick, E., Ili ´c, S., Hesslow, D., Castagn ´e, R., Luccioni, A. S., Yvon, F., Gall ´e, M., et al. Bloom: A 176b-parameter open-access multilingual language model. 2023. Lester, B., Al-Rfou, R., and Constant, N. The power of scale for parameter-efficient prompt tuning. arXiv preprint arXiv:2104.08691, 2021. Li, M., Lin, Y., Zhang, Z., Cai, T., Li, X., Guo, J., Xie, E., Meng, C., Zhu, J.-Y., and Han, S. Svdqunat: Absorb-ing outliers by low-rank components for 4-bit diffusion models. arXiv preprint arXiv:2411.05007, 2024a. Li, X. L. and Liang, P. Prefix-tuning: Optimizing continuous prompts for generation. arXiv preprint arXiv:2101.00190, 2021. Li, Y., Huang, Y., Yang, B., Venkitesh, B., Locatelli, A., Ye, H., Cai, T., Lewis, P., and Chen, D. Snapkv: Llm knows what you are looking for before generation. arXiv preprint arXiv:2404.14469, 2024b. Lingam, V., Tejaswi, A., Vavre, A., Shetty, A., Gudur, G. K., Ghosh, J., Dimakis, A., Choi, E., Bojchevski, A., and Sanghavi, S. Svft: Parameter-efficient fine-tuning with singular vectors. arXiv preprint arXiv:2405.19597, 2024. Liu, A., Feng, B., Wang, B., Wang, B., Liu, B., Zhao, C., Dengr, C., Ruan, C., Dai, D., Guo, D., et al. Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model. arXiv preprint arXiv:2405.04434 ,2024a. Liu, A., Feng, B., Xue, B., Wang, B., Wu, B., Lu, C., Zhao, C., Deng, C., Zhang, C., Ruan, C., et al. Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437 ,2024b. Liu, A., Liu, J., Pan, Z., He, Y., Haffari, G., and Zhuang, B. Minicache: Kv cache compression in depth dimension for large language models. arXiv preprint arXiv:2405.14366, 2024c. Liu, S.-Y., Wang, C.-Y., Yin, H., Molchanov, P., Wang, Y.-C. F., Cheng, K.-T., and Chen, M.-H. Dora: Weight-decomposed low-rank adaptation. arXiv preprint arXiv:2402.09353, 2024d. Liu, W., Qiu, Z., Feng, Y., Xiu, Y., Xue, Y., Yu, L., Feng, H., Liu, Z., Heo, J., Peng, S., et al. Parameter-efficient orthogonal finetuning via butterfly factorization. arXiv preprint arXiv:2311.06243, 2023a. Liu, X., Zheng, Y., Du, Z., Ding, M., Qian, Y., Yang, Z., and Tang, J. Gpt understands, too. AI Open, 2023b. Liu, Z., Yuan, J., Jin, H., Zhong, S., Xu, Z., Braverman, V., Chen, B., and Hu, X. Kivi: A tuning-free asym-metric 2bit quantization for kv cache. arXiv preprint arXiv:2402.02750, 2024e. 10 CLOVER: Cross-Layer Orthogonal Vectors 

Ma, X., Fang, G., and Wang, X. Llm-pruner: On the struc-tural pruning of large language models. Advances in neural information processing systems , 36:21702–21720, 2023. Meng, F., Wang, Z., and Zhang, M. Pissa: Principal singular values and singular vectors adaptation of large language models. arXiv preprint arXiv:2404.02948, 2024. Merity, S., Xiong, C., Bradbury, J., and Socher, R. Pointer sentinel mixture models, 2016. Mihaylov, T., Clark, P., Khot, T., and Sabharwal, A. Can a suit of armor conduct electricity? a new dataset for open book question answering. arXiv preprint arXiv:1809.02789, 2018. Mistral. Cheaper, better, faster, stronger: Continuing to push the frontier of ai and making it accessible to all, 2024. URL https://mistral.ai/news/ mixtral-8x22b .OpenAI. Hello GPT-4o, 2024. URL https://openai. com/index/hello-gpt-4o/ .Peng, B., Alcaide, E., Anthony, Q., Albalak, A., Arcadinho, S., Biderman, S., Cao, H., Cheng, X., Chung, M., Grella, M., et al. Rwkv: Reinventing rnns for the transformer era. arXiv preprint arXiv:2305.13048, 2023. Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., M ¨uller, J., Penna, J., and Rombach, R. Sdxl: Im-proving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023. Press, O., Smith, N. A., and Lewis, M. Train short, test long: Attention with linear biases enables input length extrapolation. arXiv preprint arXiv:2108.12409, 2021. Qiu, Z., Liu, W., Feng, H., Xue, Y., Feng, Y., Liu, Z., Zhang, D., Weller, A., and Sch ¨olkopf, B. Controlling text-to-image diffusion by orthogonal finetuning. Advances in Neural Information Processing Systems , 36:79320– 79362, 2023. Qwen. Qwen2.5: A party of foundation models, 2024. URL 

https://qwenlm.github.io/blog/qwen2.5 .Radford, A. Improving language understanding by genera-tive pre-training. 2018. Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I., et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019. Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., and Sutskever, I. Robust speech recognition via large-scale weak supervision. In International conference on machine learning, pp. 28492–28518. PMLR, 2023. Sakaguchi, K., Bras, R. L., Bhagavatula, C., and Choi, Y. Winogrande: An adversarial winograd schema challenge at scale. Communications of the ACM , 64(9):99–106, 2021. Sap, M., Rashkin, H., Chen, D., LeBras, R., and Choi, Y. Socialiqa: Commonsense reasoning about social interac-tions. arXiv preprint arXiv:1904.09728, 2019. Shazeer, N. Fast transformer decoding: One write-head is all you need. arXiv preprint arXiv:1911.02150, 2019. Shuttleworth, R., Andreas, J., Torralba, A., and Sharma, P. Lora vs full fine-tuning: An illusion of equivalence. arXiv preprint arXiv:2410.21228, 2024. Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., and Liu, Y. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024. Sun, M., Liu, Z., Bair, A., and Kolter, J. Z. A simple and effective pruning approach for large language models. arXiv preprint arXiv:2306.11695, 2023. Sun, Y., Dong, L., Zhu, Y., Huang, S., Wang, W., Ma, S., Zhang, Q., Wang, J., and Wei, F. You only cache once: Decoder-decoder architectures for language models. arXiv preprint arXiv:2405.05254, 2024. Sung, Y.-L., Nair, V., and Raffel, C. A. Training neural networks with fixed sparse masks. Advances in Neural Information Processing Systems , 34:24193– 24205, 2021. Team, G., Georgiev, P., Lei, V. I., Burnell, R., Bai, L., Gulati, A., Tanzer, G., Vincent, D., Pan, Z., Wang, S., et al. Gemini 1.5: Unlocking multimodal understand-ing across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024a. Team, G., Riviere, M., Pathak, S., Sessa, P. G., Hardin, C., Bhupatiraju, S., Hussenot, L., Mesnard, T., Shahri-ari, B., Ram ´e, A., et al. Gemma 2: Improving open language models at a practical size. arXiv preprint arXiv:2408.00118, 2024b. Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Rozi `ere, B., Goyal, N., Hambro, E., Azhar, F., et al. LLaMA: Open and efficient founda-tion language models. arXiv preprint arXiv:2302.13971, 2023. Vu, T., Lester, B., Constant, N., Al-Rfou, R., and Cer, D. Spot: Better frozen model adaptation through soft prompt transfer. arXiv preprint arXiv:2110.07904, 2021. Wang, S., Li, B. Z., Khabsa, M., Fang, H., and Ma, H. Linformer: Self-attention with linear complexity. arXiv preprint arXiv:2006.04768, 2020. 11 CLOVER: Cross-Layer Orthogonal Vectors 

Wang, S., Yu, L., and Li, J. Lora-ga: Low-rank adap-tation with gradient approximation. arXiv preprint arXiv:2407.05000, 2024a. Wang, Z., Panda, R., Karlinsky, L., Feris, R., Sun, H., and Kim, Y. Multitask prompt tuning enables parameter-efficient transfer learning. arXiv preprint arXiv:2303.02861, 2023. Wang, Z., Liang, J., He, R., Wang, Z., and Tan, T. Lora-pro: Are low-rank adapters properly optimized? arXiv preprint arXiv:2407.18242, 2024b. Xia, M., Gao, T., Zeng, Z., and Chen, D. Sheared llama: Accelerating language model pre-training via structured pruning. arXiv preprint arXiv:2310.06694, 2023. Xiao, G., Lin, J., Seznec, M., Wu, H., Demouth, J., and Han, S. Smoothquant: Accurate and efficient post-training quantization for large language models. In International Conference on Machine Learning , pp. 38087–38099. PMLR, 2023. Xu, R., Luo, F., Zhang, Z., Tan, C., Chang, B., Huang, S., and Huang, F. Raise a child in large language model: Towards effective and generalizable fine-tuning. arXiv preprint arXiv:2109.05687, 2021. Yu, H., Yang, Z., Li, S., Li, Y., and Wu, J. Effectively com-press kv heads for llm. arXiv preprint arXiv:2406.07056, 2024. Yuan, S., Liu, H., and Xu, H. Bridging the gap between low-rank and orthogonal adaptation via householder reflection adaptation. arXiv preprint arXiv:2405.17484, 2024. Zaken, E. B., Ravfogel, S., and Goldberg, Y. Bitfit: Simple parameter-efficient fine-tuning for transformer-based masked language-models. arXiv preprint arXiv:2106.10199, 2021. Zellers, R., Holtzman, A., Bisk, Y., Farhadi, A., and Choi, Y. Hellaswag: Can a machine really finish your sentence? arXiv preprint arXiv:1905.07830, 2019. Zhang, Q., Chen, M., Bukharin, A., Karampatziakis, N., He, P., Cheng, Y., Chen, W., and Zhao, T. Adalora: Adaptive budget allocation for parameter-efficient fine-tuning. arXiv preprint arXiv:2303.10512, 2023. Zhao, J., Zhang, Z., Chen, B., Wang, Z., Anandkumar, A., and Tian, Y. Galore: Memory-efficient llm train-ing by gradient low-rank projection. arXiv preprint arXiv:2403.03507, 2024. Zhao, M., Lin, T., Mi, F., Jaggi, M., and Sch ¨utze, H. Mask-ing as an efficient alternative to finetuning for pretrained language models. arXiv preprint arXiv:2004.12406 ,2020. Zi, B., Qi, X., Wang, L., Wang, J., Wong, K.-F., and Zhang, L. Delta-lora: Fine-tuning high-rank parame-ters with the delta of low-rank matrices. arXiv preprint arXiv:2309.02411, 2023. Zuhri, Z. M. K., Adilazuarda, M. F., Purwarianti, A., and Aji, A. F. Mlkv: Multi-layer key-value heads for memory efficient transformer decoding. arXiv preprint arXiv:2406.09297, 2024. 12 CLOVER: Cross-Layer Orthogonal Vectors 

## A. Appendix 

A.1. Cross Layer Orthogonal Vectors in Value and Output layers 

In the main text, we only presented the orthogonalization process for the Q-K pair. Here, we provide the method for orthogonalizing the V-O pair. Additionally, for up-down layers, the output dimension of the Up layer can be reshaped into block number × block size, followed by performing orthogonal decomposition within each block. 

Y = attn(Q, K) V W O , V = XW V ∈ Rb×h×n×d (1) 

= attn(Q, K) XW V WO , WV WO = WV O = U SV ∈ Rh×D×D (2) 

= attn(Q, K) XU SV, S[: ,r vo :,r vo :] = SV O ∈ Rh×rvo ×rvo = 0 , r vo ≤ d. (3) 

= attn(Q, K) XU V O SV O VV O , UV O ∈ RD×h×rvo , VV O ∈ Rh×rvo ×D . (4) Through this series of transformations, WV and WO can be equivalently replaced by orthogonal vectors UV O and VV O ,along with the diagonal matrix SV O . Since rvo ≤ d, the singular zero values and their corresponding singular vectors can be safely pruned. After guided pruning, SV O can be merged into UV O and VV O , resulting in no additional computational overhead. 

A.2. Hyperparameters 

Table 3 presents a comparison of hyperparameters for different fine-tuning methods on commonsense tasks. The target model remains the same for LoRA, DoRA, HiRA, and PiSSA. However, DoRA introduces an additional magnitude module, leading to a slightly higher parameter count. In a single layer of LoRA, the trainable parameters are as follows: In LoRA, the trainable parameters are: 

Q = 4096 × 32 + 4096 × 32 

K = 4096 × 32 + 4096 × 32 

V = 4096 × 32 + 4096 × 32 

Up = 4096 × 32 + 11008 × 32 

Down = 4096 × 32 + 11008 × 32 

The total sum is 1,753,088. In CLOVER, the trainable parameters are: 

QK = 32 × 128 × 128 

V O = 32 × 128 × 128 

U D = 172 × 64 × 64 

The total sum is also 1,753,088. Since CLOVER inserts trainable parameters across layers, we use the Q-K pair notation to represent its target model. When CLVOER updates parameters within an attention head, the number of trainable parameters matches exactly that of LoRA at rank 32. To adjust the number of learnable parameters, CLOVER can either span multiple heads or split a single head into multiple blocks. Both PiSSA and CLOVER exhibit stable training performance. Therefore, instead of validating every 80 steps, we omit frequent validation, improving training efficiency. 

A.3. Detail Information of Dataset 

The commonsense reasoning tasks consist of 8 subtasks, each with predefined training and testing sets, as described by LLM-Adapters (Hu et al., 2023). The following table lists the details of each sub-dataset. 13 CLOVER: Cross-Layer Orthogonal Vectors  

> Table 3. Detailed Training Hyperparameters. Q-K,V-O, U-D means CLVOER update pair of orthogonal vectors.

Method Target Evaluation steps LR Scheduler Batch size Warmup Steps Epochs 

LoRA Q,K,V,U,D 80 3e-4 Linear 16 100 3DoRA Q,K,V,U,D 80 2e-4 Linear 16 100 3HiRA Q,K,V,U,D 80 2e-4/2e-4 Linear 32 100 3PiSSA Q,K,V,U,D – 2e-5 Linear 16 100 3CLOVER Q-K,V-O, U-D – 1e-4 Linear 16 100 3                            

> Table 4. Details of datasets for commonsense reasoning tasks.
> Dataset Train Test About BoolQ (Clark et al., 2019) 9,427 3,270 Naturally occurring yes/no questions from unconstrained settings. PIQA (Bisk et al., 2020) 16,113 1,838 Questions with two solutions requiring physical commonsense. SIQA (Sap et al., 2019) 33,410 1,954 Reasoning about actions and social implications. HellaSwag (Zellers et al., 2019) 39,905 10,042 Commonsense NLI questions with context and endings. WinoGrande (Sakaguchi et al., 2021) 40,398 1,267 Fill-in-the-blank task with binary options. ARC-e (Clark et al., 2018) 2,251 2,376 Grade-school multiple-choice science questions in Easy sets. ARC-c (Clark et al., 2018) 1,119 1,172 Grade-school multiple-choice science questions in Challenge sets. OBQA (Mihaylov et al., 2018) 4,957 500 Questions requiring multi-step reasoning and commonsense knowledge.

For WinoGrande, the original dataset includes multiple partitions: [xs, s, m, l, xl, debiased]. While LLM-Adapters simply concatenated all these partitions, note that the “xl” partition actually includes all others, leading to extensive data duplication. After removing duplicates, the training data is reduced from 63.2K to 40.4K instances. Additionally, in the LLM-Adapters paper, the training set sizes of ARC Challenge and ARC Easy were reversed by mistake; here, we correct that error. 

A.4. LibriSpeech Long dataset target transcript 

Below is the reference text of the LibriSpeech Long dataset for comparison. 

Mr. Quilter is the apostle of the middle classes, and we are glad to welcome his gospel. Nor is Mr. Quilter’s manner less interesting than his matter. He tells us that at this festive season of the year, with Christmas and roast beef looming before us, similes drawn from eating and its results occur most readily to the mind. He has grave doubts whether Sir Frederick Layton’s work is really Greek after all, and can discover in it but little of rocky Ithaca. Linnell’s pictures are a sort of Up Guards and Adam paintings, and Mason’s exquisite idles are as national as a jingo poem. Mr. Birkett Foster’s landscapes smile at one much in the same way that Mr. Carker used to flash his teeth, and Mr. John Collier gives his sitter a cheerful slap on the back before he says, like a shampooer in a Turkish bath, next man. 

In fact, with Vanilla Pruning ratios of just 22.31% and 6.69% for WQ-WK and WV -WO , respectively, the model’s output is already significantly degraded. 

Mr. Colter is the personal of the classes, and we are glad to welcome his gospel. Nor is Mr. Colter’s manner less interesting than his manner. He tells us that at this festive season of the year, with Christmas and roast beef looming before us, similarly he is drawn from eating and its results occur most readily to the mind. He is very dull, so very frequently, and is very Greek after all, and can discover in it but little of Rocky Ithaca. The Nell’s pictures are sort of up-guard to Adam’s paintings, and Mason’s exquisite idylls are as national as a jingle poem. Mr. Burke and Foster’s landscapes smile at one much in the same way as Mr. Parker, Mr. Flash is tits. And Mr. John Collier gives his sitter a cheerful slap on the back before he says like a shampoo and a Turkish bath, Next man. 

A.5. Visualizing more attention heads 

In Section 4.3, we only presented the first attention head in the first layer. Here, we provide a broader view by showcasing more attention heads. Figure 7 illustrates the L2 norm of all Q-K heads in the first, middle, and last layers of Whisper-Large-14 CLOVER: Cross-Layer Orthogonal Vectors 

v3. Figure 8 shows the L2 norm of all Q-K heads in the first, middle, and last layers of ViT-bigG. From these figures, we can observe that CLOVER consistently represents the entire attention head with fewer orthogonal bases across all layers and all attention heads. This property forms the foundation of CLVOER’s effectiveness in enhancing pruning. 0 200 400 600 800 1000 1200             

> 0
> 2
> 4
> 6
> 8
> 10
> Layer.0.qk
> Absorb and Decompose
> Vanilla
> 0200 400 600 800 1000 1200
> 0
> 2
> 4
> 6
> 8
> Layer.15.qk
> 0200 400 600 800 1000 1200
> 0.0
> 2.5
> 5.0
> 7.5
> 10.0
> 12.5
> 15.0
> Layer.31.qk

Figure 7. The L2-norm for the 0-th, 15-th, and 31-st attention layers in the Whisper-large-v3 encoder. The blue line represents the results after redundancy removal using the CLOVER method, while the orange line depicts the L2-norm directly computed for each dimension. 

15 CLOVER: Cross-Layer Orthogonal Vectors 0 250 500 750 1000 1250 1500  

> 0.0
> 0.1
> 0.2
> 0.3
> 0.4
> 0.5

Layer.0.qk        

> Absorb and Decompose
> Vanilla
> 0250 500 750 1000 1250 1500
> 0.0
> 0.5
> 1.0
> 1.5
> 2.0
> 2.5

Layer.23.qk        

> 0250 500 750 1000 1250 1500
> 0.0
> 0.2
> 0.4
> 0.6
> 0.8
> 1.0

Layer.47.qk 

Figure 8. The L2-norm for the 0-th, 15-th, and 31-st attention layers in the ViT-bigG. The blue line represents the results after redundancy removal using the CLOVER method, while the orange line depicts the L2-norm directly computed for each dimension. 

16
