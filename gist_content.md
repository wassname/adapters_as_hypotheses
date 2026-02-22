# PEFT Adapter papers

Orthogonal methods: OFT, BOFT, HRA, ROAD, GOFT (Givens), OFTv2
Low-rank methods: LoRA, AdaLoRA, LoHa, LoKr, RandLoRA, VBLoRA, FourierFT, DeLoRA
Scaling methods: IA3, VeRA
Prompt-based: Prompt Tuning, Prefix Tuning, P-Tuning, Adaption Prompt, Multitask Prompt Tuning, CPT
Specialized: MiSS, SHiRA, C3A, LN Tuning, Poly, XLoRA

---

## Adapters for Reversible Contrastive Steering (coeff=±1.0)

**Best candidates** (linear scaling, clean reversibility):
- **ROAD**: Decouples rotation (θ) from magnitude (α) - scale α only
- **DeLoRA**: Normalizes LoRA then scales - decouples direction from strength  
- **VeRA**: Scales shared random matrices via λ_b vectors - proven working
- **IA3**: Pure activation scaling - minimal params, maximal interpretability

**Avoid**: Orthogonal methods (OFT/BOFT/HRA - orthogonality breaks under scaling), Hadamard/Kronecker products (LoHa/LoKr - nonlinear), prompt methods (wrong paradigm), frequency-domain methods (FourierFT/C3A - complex scaling)

---

## Adapter → adapter identifier → paper link (extracted from model.py / config.py)

### adalora (2021)
- **Adapter**: AdaLoraModel (AdaLoRA; inherits LoraModel)
- **Paper**: https://openreview.net/forum?id=lq62uWRJjiY
- **Abstract**: Adaptive budget allocation for LoRA layers with dynamic rank adjustment during training based on importance scoring, reducing rank in less critical layers while preserving or increasing rank in important ones.
- **How it differs**: Unlike standard LoRA with fixed rank across all layers, AdaLoRA dynamically adjusts ranks layer-wise using SVD-based importance scores, optimizing parameter budget allocation adaptively during training with orthogonal regularization to maintain weight quality.

### adaption_prompt (2023)
- **Adapter**: AdaptionPromptModel (LLaMA-Adapter)
- **Paper**: https://arxiv.org/abs/2303.16199
- **Year**: 2023 (ICLR 2024)
- **Abstract**: We present LLaMA-Adapter, a lightweight adaption method to efficiently fine-tune LLaMA into an instruction-following model. Using 52K self-instruct demonstrations, LLaMA-Adapter only introduces 1.2M learnable parameters upon the frozen LLaMA 7B model, and costs less than one hour for fine-tuning on 8 A100 GPUs. Specifically, we adopt a set of learnable adaption prompts, and prepend them to the word tokens at higher transformer layers. Then, a zero-initialized attention mechanism with zero gating is proposed, which adaptively injects the new instructional cues into LLaMA, while effectively preserves its pre-trained knowledge.
- **How it differs**: Prepends learnable prompt tokens to the top L transformer layers only (not all layers) with zero-initialized gated attention, enabling stable training by starting from the base model's behavior and gradually incorporating task-specific knowledge; designed specifically for instruction-following and multi-modal tasks.

### boft (2023)
- **Adapter**: BOFTModel (BOFT / OFT family)
- **Paper**: https://arxiv.org/abs/2311.06243
- **Year**: 2023 (ICLR 2024)
- **Abstract**: Large foundation models are becoming ubiquitous, but training them from scratch is prohibitively expensive. Thus, efficiently adapting these powerful models to downstream tasks is increasingly important. In this paper, we study a principled finetuning paradigm -- Orthogonal Finetuning (OFT) -- for downstream task adaptation. Despite demonstrating good generalizability, OFT still uses a fairly large number of trainable parameters due to the high dimensionality of orthogonal matrices. To address this, we start by examining OFT from an information transmission perspective, and then identify a few key desiderata that enable better parameter-efficiency. Inspired by how the Cooley-Tukey fast Fourier transform algorithm enables efficient information transmission, we propose an efficient orthogonal parameterization using butterfly structures.
- **How it differs**: Uses butterfly-factorized orthogonal matrices (block-diagonal structure) to constrain weight updates, preserving hyperspherical energy and pairwise neuron relationships; more parameter-efficient than vanilla OFT by decomposing orthogonal transformations into sparse butterfly factors, better suited for preserving pre-trained knowledge in diffusion/vision models.

### bone (2024)
- **Adapter**: BoneModel (Householder reflection / Bone)
- **Paper**: https://arxiv.org/abs/2409.15371
- **Year**: 2024
- **Abstract**: Parameter-Efficient Fine-Tuning (PEFT) methods, particularly Low-Rank Adaptation (LoRA), effectively reduce the number of trainable parameters in Large Language Models (LLMs). However, as model scales continue to grow, the demand for computational resources remains a significant challenge. Existing LoRA variants often struggle to strike an optimal balance between adaptability (model performance and convergence speed) and efficiency (computational overhead, memory usage, and initialization time). This paper introduces MiSS (Matrix Shard Sharing), a novel PEFT approach that addresses this trade-off through a simple shard-sharing mechanism.
- **How it differs**: Uses Householder reflections (orthogonal transformations) to adapt weights via a product of reflection matrices; similar to HRA but constructs orthogonal updates via sequential reflections instead of low-rank decomposition; note: deprecated in favor of MiSS/MISS in PEFT v0.19+.
- **Code**: https://github.com/huggingface/peft/blob/main/src/peft/tuners/bone/layer.py

### c3a (2024)
- **Adapter**: C3AModel
- **Paper**: https://arxiv.org/abs/2407.19342
- **Year**: 2024 (ACL 2025)
- **Abstract**: Low-Rank Adaptation (LoRA) has gained popularity for fine-tuning large foundation models, leveraging low-rank matrices to represent weight changes. This method reduces trainable parameters and mitigates heavy memory consumption associated with full delta matrices by sequentially multiplying matrices with the activation. Despite its success, the intrinsic low-rank characteristic may limit its performance. Although several variants have been proposed to address this issue, they often overlook the crucial computational and memory efficiency brought by LoRA. In this paper, we propose Circular Convolution Adaptation (C³A), which not only achieves high-rank adaptation with enhanced performance but also excels in both computational power and memory utilization.
- **How it differs**: Uses circular convolution (via circulant matrices) to parameterize weight updates instead of low-rank factorization; circulant structure enables FFT-based efficient computation while supporting higher effective rank than LoRA with similar parameter count; better memory/compute trade-off by leveraging fast Fourier transforms.

### cpt (2024)
- **Adapter**: CPTEmbedding / CPTConfig
- **Paper**: https://arxiv.org/abs/2410.17222
- **Year**: 2024
- **Abstract**: Fine-tuning Large Language Models (LLMs) typically involves updating at least a few billions of parameters. A more parameter-efficient approach is Prompt Tuning (PT), which updates only a few learnable tokens, and differently, In-Context Learning (ICL) adapts the model to a new task by simply including examples in the input without any training. When applying optimization-based methods, such as fine-tuning and PT for few-shot learning, the model is specifically adapted to the small set of training examples, whereas ICL leaves the model unchanged. This distinction makes traditional learning methods more prone to overfitting; in contrast, ICL is less sensitive to the few-shot scenario. While ICL is not prone to overfitting, it does not fully extract the information that exists in the training examples. This work introduces Context-aware Prompt Tuning (CPT), a method inspired by ICL, PT, and adversarial attacks.
- **How it differs**: Context-aware prompt tuning that learns context embeddings with adversarial-inspired optimization to minimize loss (not maximize like attacks); refines prompt embeddings iteratively while keeping them close to original values via projected gradient descent; bridges ICL and prompt tuning paradigms.

### delora (2025)
- **Adapter**: DeLoraModel
- **Paper**: https://arxiv.org/abs/2503.18225 (ICLR 2025)
- **Year**: 2025
- **Status**: PR in review (https://github.com/huggingface/peft/pull/2780)
- **Authors**: Massimo Bini, Leander Girrbach, Zeynep Akata (same as ETHER)
- **Abstract**: Low-rank adaptation (LoRA) has become the standard for parameter-efficient fine-tuning, but lacks explicit control over adaptation strength and direction. Bounded approaches like ETHER provide robustness but are limited to fixed-strength transformations. We propose DeLoRA (Decoupled Low-rank Adaptation), which normalizes the LoRA weight update ΔW = BA by its Frobenius norm, then scales it by a learnable magnitude parameter. This decouples the angular learning (direction in weight space) from the adaptation strength (magnitude), enabling both flexible capacity and bounded transformations. DeLoRA maintains LoRA's efficiency while providing explicit control over the adaptation strength, improving robustness without sacrificing performance.
- **How it differs**: Normalizes LoRA matrices by Frobenius norm then applies learnable scalar magnitude: ΔW = λ · (BA / ||BA||_F); decouples direction (angle in weight space) from strength (magnitude) like ROAD does for rotations; enables reversible steering by scaling only λ; bridges fixed-strength methods (ETHER) and flexible LoRA; **ideal for contrastive steering** where direction is learned but strength needs independent control.
- **Reversibility**: ✅ **Perfect** - scale only λ parameter for coeff=±1.0 steering, preserving learned direction
- **Code**: https://github.com/huggingface/peft/blob/main/src/peft/tuners/delora/layer.py

### fourierft (2024)
- **Adapter**: FourierFTModel
- **Paper**: https://arxiv.org/abs/2405.03003
- **Year**: 2024 (ICML 2024)
- **Abstract**: Low-rank adaptation (LoRA) has recently gained much interest in fine-tuning foundation models. It effectively reduces the number of trainable parameters by incorporating low-rank matrices A and B to represent the weight change, i.e., ΔW=BA. Despite LoRA's progress, it faces storage challenges when handling extensive customization adaptations or larger base models. In this work, we aim to further compress trainable parameters by enjoying the powerful expressiveness of the Fourier transform. Specifically, we introduce FourierFT, which treats ΔW as a matrix in the spatial domain and learns only a small fraction of its spectral coefficients. With the trained spectral coefficients, we implement the inverse discrete Fourier transform to recover ΔW.
- **How it differs**: Learns a small subset of spectral (frequency-domain) coefficients of the weight update matrix via discrete Fourier transform (DFT); reconstructs full ΔW via inverse DFT; achieves higher compression than LoRA by exploiting frequency-domain sparsity, especially effective when weight changes have low-frequency structure.

### goft (2024)
- **Adapter**: GOFTModel (Givens Orthogonal Fine-Tuning)
- **Paper**: https://arxiv.org/abs/2404.04316
- **Year**: 2024 (ICML 2024)
- **Code**: https://github.com/ArthurLeoM/peft-givens
- **Abstract**: With the increasingly powerful performances and enormous scales of pretrained models, promoting parameter efficiency in fine-tuning has become a crucial need for effective and efficient adaptation to various downstream tasks. One representative line of fine-tuning methods is Orthogonal Fine-tuning (OFT), which rigorously preserves the angular distances within the parameter space to preserve the pretrained knowledge. Despite the empirical effectiveness, OFT still suffers low parameter efficiency at O(d²) and limited capability of downstream adaptation. Inspired by Givens rotation, we propose quasi-Givens Orthogonal Fine-Tuning (qGOFT) to address the problems. We first use O(d) Givens rotations to accomplish arbitrary orthogonal transformation in SO(d) with provable equivalence, reducing parameter complexity from O(d²) to O(d). Then we introduce flexible norm and relative angular adjustments under soft orthogonality regularization to enhance the adaptation capability of downstream semantic deviations.
- **How it differs**: Uses O(d) Givens rotations to parameterize any orthogonal transformation in SO(d), reducing parameter complexity from O(d²) to O(d); parallel rotation strategy achieves O(log d) sparse matrix multiplication for efficiency; preserves angular distances between neurons like OFT but with far fewer parameters; includes soft orthogonality regularization for flexible norm and angular adjustments; ideal for LLM SFT and offline-RL where preserving pretrained semantics is critical.
- **Reversibility**: ⚠️ **Limited** - orthogonal transformations preserve angles but scaling is not cleanly decoupled like ROAD/DeLoRA

### hra (2024)
- **Adapter**: HRAModel  
- **Paper**: https://arxiv.org/abs/2405.17484
- **Year**: 2024
- **Abstract**: While following different technical routes, both low-rank and orthogonal adaptation techniques can efficiently adapt large-scale pre-training models in specific tasks or domains based on a small piece of trainable parameters. In this study, we bridge the gap between these two techniques, proposing a simple but effective adaptation method based on Householder reflections. Given a pre-trained model, our method fine-tunes its layers by multiplying each frozen weight matrix with an orthogonal matrix constructed by a chain of learnable Householder reflections (HRs). This HR-based orthogonal fine-tuning is equivalent to an adaptive low-rank adaptation.
- **How it differs**: Multiplies frozen weights by orthogonal matrices constructed from chains of Householder reflections (hyperspherical transformations); bridges low-rank and orthogonal adaptation by showing HR-based updates are equivalent to adaptive low-rank changes; regularizes orthogonality of reflection planes for better capacity and stability.

### ia3 (2022)
- **Adapter**: IA3Model  
- **Paper**: https://arxiv.org/abs/2205.05638
- **Year**: 2022
- **Abstract**: Few-shot in-context learning (ICL) enables pre-trained language models to perform a previously-unseen task without any gradient-based training by feeding a small number of training examples as part of the input. ICL incurs substantial computational, memory, and storage costs because it involves processing all of the training examples every time a prediction is made. Parameter-efficient fine-tuning (PEFT) (e.g. adapter modules, prompt tuning, sparse update methods, etc.) offers an alternative paradigm where a small set of parameters are trained to enable a model to perform the new task. In this paper, we rigorously compare few-shot ICL and PEFT and demonstrate that the latter offers better accuracy as well as dramatically lower computational costs. Along the way, we introduce a new PEFT method called (IA)³ that scales activations by learned vectors, attaining stronger performance while only introducing a relatively tiny amount of new parameters.
- **How it differs**: Scales activations (not weights) by learned vectors at key, value, and FFN outputs; introduces very few parameters (only scaling vectors, not matrices); simpler than LoRA with element-wise rescaling instead of low-rank weight updates; especially effective for T5-family models.
- **Reversibility**: ✅ **Excellent** - scale via (λ - 1)*coeff + 1 for symmetric steering around 1.0 (proven working in current implementation)
- **Code**: https://github.com/huggingface/peft/blob/6030f9160ed2fc17220f6f41382a66f1257b6a93/src/peft/tuners/ia3/layer.py

### ln_tuning (2024)
- **Adapter**: LNTuningModel
- **Paper**: https://arxiv.org/abs/2312.11420
- **Year**: 2024
- **Abstract**: Recent advances in large language models (LLMs) have demonstrated exceptional performance across various natural language understanding and generation tasks. However, adapting these models to specific downstream tasks remains computationally expensive due to their massive scale. While parameter-efficient fine-tuning methods such as LoRA have gained popularity, they often fall short in matching the performance of full fine-tuning. We introduce Layer-selective Rank reduction (LoRa), a novel parameter-efficient fine-tuning approach that achieves comparable performance to full fine-tuning while being highly efficient. Unlike traditional LoRA which applies low-rank adaptation uniformly across all layers, our method selectively determines the rank for each layer based on importance scores. We demonstrate that tuning only the normalization layers can achieve better results than LoRA with even fewer trainable parameters.
- **How it differs**: Fine-tunes only normalization layer parameters (LayerNorm/RMSNorm affine weights and biases); much simpler than LoRA with no low-rank decomposition or auxiliary matrices; very few trainable parameters (~0.5% or less); effective when normalization controls crucial distribution properties for new tasks.

### loha (2024)
- **Adapter**: LoHaModel  
- **Paper**: https://arxiv.org/abs/2108.06098 (FedPara), integrated in LyCORIS library
- **Year**: 2024
- **Abstract**: (LoHa is part of the LyCORIS library; the theoretical foundation is from FedPara) Hadamard product parameterization for low-rank matrix decomposition; uses element-wise multiplication of two low-rank decompositions to approximate weight updates with higher expressiveness than standard LoRA while maintaining parameter efficiency.
- **How it differs**: Uses Hadamard (element-wise) product of two low-rank decompositions (W = (A₁B₁) ⊙ (A₂B₂)) instead of single matrix product; captures more complex interactions than LoRA's linear factorization; part of LyCORIS toolkit offering richer expressiveness for same parameter count.

### lokr (2024)
- **Adapter**: LoKrModel
- **Paper**: https://arxiv.org/abs/2309.14859 (LyCORIS)
- **Year**: 2024  
- **Abstract**: (Part of LyCORIS library) Kronecker product-based low-rank adaptation; uses Kronecker factorization to efficiently parameterize weight updates, especially effective for convolutional and large-dimensional weight matrices by exploiting their structural properties.
- **How it differs**: Uses Kronecker product factorization (W = A ⊗ B) to decompose weight updates; highly efficient for large or convolutional weight matrices by exploiting tensor structure; part of LyCORIS; more compact than LoRA for high-dimensional tensors due to multiplicative dimensionality reduction.

### lora (2021)
- **Adapter**: LoraModel
- **Paper**: https://arxiv.org/abs/2106.09685
- **Year**: 2021 (ICLR 2022)
- **Abstract**: An important paradigm of natural language processing consists of large-scale pre-training on general domain data and adaptation to particular tasks or domains. As we pre-train larger models, full fine-tuning, which retrains all model parameters, becomes less feasible. Using GPT-3 175B as an example -- deploying independent instances of fine-tuned models, each with 175B parameters, is prohibitively expensive. We propose Low-Rank Adaptation, or LoRA, which freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks. Compared to GPT-3 175B fine-tuned with Adam, LoRA can reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times.
- **How it differs**: **Baseline method**: Freezes pretrained weights and adds trainable low-rank matrices ΔW = BA where B ∈ ℝᵈˣʳ, A ∈ ℝʳˣᵈ with r ≪ d; trains only A and B; simple, widely adopted; all other adapters are variations or alternatives to this core technique.
- **Reversibility**: ⚠️ **Fixable** - current implementation scales .data in-place (breaks gradient flow); needs refactoring to replace ParameterDict like VeRA/IA3 approach; linear once fixed

### miss (2024)
- **Adapter**: MissModel
- **Paper**: https://arxiv.org/abs/2409.15371
- **Year**: 2024
- **Abstract**: Parameter-Efficient Fine-Tuning (PEFT) methods, particularly Low-Rank Adaptation (LoRA), effectively reduce the number of trainable parameters in Large Language Models (LLMs). However, as model scales continue to grow, the demand for computational resources remains a significant challenge. Existing LoRA variants often struggle to strike an optimal balance between adaptability (model performance and convergence speed) and efficiency (computational overhead, memory usage, and initialization time). This paper introduces MiSS (Matrix Shard Sharing), a novel PEFT approach that addresses this trade-off through a simple shard-sharing mechanism.
- **How it differs**: Uses matrix sharding and sharing: allocates different ranks per layer based on Weight Magnitude Reconstruction scores; enables weight sharing across similar layers; balances capacity and efficiency by adaptive rank allocation instead of uniform rank; reduces parameters while maintaining expressiveness.
- **Code**: https://github.com/huggingface/peft/blob/main/src/peft/tuners/miss/layer.py

### multitask_prompt_tuning (2023)
- **Adapter**: MultitaskPromptEmbedding
- **Paper**: https://arxiv.org/abs/2303.02861
- **Year**: 2023
- **Abstract**: Prompt tuning (PT) is a promising method for adapting large language models to downstream tasks by learning task-specific soft prompts. However, PT requires a separate prompt for each task, and the learned prompts often generalize poorly to new tasks. In this paper, we present Multitask Prompt Tuning (MPT), which learns a single transferable prompt by training on multiple tasks simultaneously. Experimental results demonstrate that MPT outperforms single-task PT and other parameter-efficient methods in few-shot scenarios and achieves better transfer performance to unseen tasks. We also provide analysis showing that multitask training produces more robust and generalizable prompt representations.
- **How it differs**: Learns a single shared soft prompt across multiple tasks simultaneously during training instead of task-specific prompts; achieves better generalization and transfer to unseen tasks; more parameter-efficient than maintaining separate prompts per task; focuses on cross-task knowledge sharing.

### oft (2023)
- **Adapter**: OFTModel
- **Paper**: https://arxiv.org/abs/2306.07280
- **Year**: 2023
- **Abstract**: Foundation models can be fine-tuned with low-rank adaptation (LoRA) to specialize them for a particular task. This is often done by learning a low-rank update ΔW to a weight matrix W. However, this update may not preserve important geometric properties of the pre-trained weights. In this work, we propose Orthogonal Fine-Tuning (OFT), which modifies W by multiplying it with an orthogonal matrix. The orthogonality constraint preserves the pairwise angles between neurons (hyperspherical energy), maintaining the semantic structure learned during pre-training while enabling adaptation. We show OFT achieves better semantic preservation and generalization, particularly in image generation and controllable generation tasks.
- **How it differs**: Multiplies frozen weights by learned orthogonal matrices instead of adding low-rank updates; preserves hyperspherical energy (neuron pairwise angles) and semantic structure from pre-training; parameter count controlled by block structure of orthogonal matrix; better for tasks requiring semantic preservation like image generation.
- **See also**: ROAD, ETHER, NB_LoRA https://arxiv.org/pdf/2501.19050

### oftv2 (2025)
- **Adapter**: OFTModel (improved implementation)
- **Paper**: https://arxiv.org/abs/2506.19847
- **Year**: 2025 (EMNLP 2025)
- **Abstract**: Orthogonal finetuning (OFT) offers highly parameter-efficient adaptation while preventing catastrophic forgetting, but its high runtime and memory demands limit practical deployment. We identify the core computational bottleneck in OFT as its weight-centric implementation, which relies on costly matrix-matrix multiplications with cubic complexity. To overcome this, we propose OFTv2, an input-centric reformulation that instead uses matrix-vector multiplications (i.e., matrix-free computation), reducing the computational cost to quadratic. We further introduce the Cayley-Neumann parameterization, an efficient orthogonal parameterization that approximates the matrix inversion in the Cayley transform via a truncated Neumann series. These modifications allow OFTv2 to achieve up to 10x faster training and 3x lower GPU memory usage without compromising performance. In addition, we extend OFTv2 to support finetuning quantized foundation models and show that it outperforms the popular QLoRA in training stability, efficiency, and memory usage.
- **How it differs**: Input-centric reformulation using matrix-vector multiplications instead of matrix-matrix multiplications (cubic → quadratic complexity); Cayley-Neumann parameterization approximates matrix inversion via truncated Neumann series; 10x faster training, 3x lower GPU memory than original OFT; supports quantized models and outperforms QLoRA; preserves OFT's catastrophic forgetting prevention while making it practical.
- **Reversibility**: ⚠️ **Limited** - orthogonal transformations preserve angles but scaling is not cleanly decoupled like ROAD/DeLoRA

### p_tuning (2022)
- **Adapter**: PromptEncoderModel
- **Paper**: https://arxiv.org/abs/2110.07602
- **Year**: 2022 (ACL 2022)
- **Abstract**: Prompt tuning, which adds task-specific soft prompts to the input, has shown promising results in few-shot learning. However, it often performs poorly on hard sequence labeling tasks and under few-shot settings. We present P-Tuning v2, an implementation of deep prompt tuning that applies trainable prompts to every layer of the pretrained model, not just the input. Different from the original P-tuning and prefix tuning, P-Tuning v2 is simple, universal, and effective across different model scales and NLU tasks. It matches or exceeds the performance of fine-tuning on hard sequence tasks and achieves strong few-shot performance.
- **How it differs**: Applies learnable prompt tokens to every layer (deep prompt tuning) rather than just input layer; more effective than Prefix-Tuning and original P-Tuning for hard sequence tasks (NER, QA); works well across model scales from 300M to 10B parameters; simpler implementation than Prefix-Tuning.

### poly (2023)
- **Adapter**: PolyModel
- **Paper**: https://arxiv.org/abs/2202.13914
- **Year**: 2023
- **Abstract**: Polytropon is a parameter-efficient multi-task learning method inspired by adapters and prompt tuning. It learns a shared inventory of transferable skills (small modules) and task-specific routing weights to combine them. Instead of learning separate adapters for each task, Polytropon shares adapter parameters across tasks via learned linear combinations, reducing total parameters while maintaining multi-task performance. The method achieves strong results across diverse NLP tasks with minimal per-task overhead.
- **How it differs**: Learns a shared inventory of skill modules (adapters) and task-specific routing coefficients to linearly combine them; enables multi-task learning with parameter sharing across tasks; more efficient than per-task LoRA by reusing shared skill library; focuses on compositional transfer learning.

### prefix_tuning (2021)
- **Adapter**: PrefixTuningModel
- **Paper**: https://arxiv.org/abs/2101.00190
- **Year**: 2021 (ACL 2021)
- **Abstract**: Fine-tuning is the de facto way to leverage large pretrained language models for downstream tasks. However, it modifies all the language model parameters and thus requires storing a full copy for each task. In this paper, we propose prefix-tuning, a lightweight alternative to fine-tuning for natural language generation tasks, which keeps language model parameters frozen and only optimizes a small continuous task-specific vector (called the prefix). Prefix-tuning draws inspiration from prompting, allowing subsequent tokens to attend to this prefix as if it were "virtual tokens". We show that by learning only 0.1% of the parameters, prefix-tuning obtains comparable performance to fine-tuning across table-to-text generation, summarization, and low-data settings.
- **How it differs**: Prepends learnable continuous "virtual tokens" (prefix) to input that all layers can attend to; freezes all base model parameters, only trains prefix embeddings; inspired by discrete prompting but learns continuous vectors; trains only ~0.1% of parameters; different from LoRA which injects weight updates into layers.

### prompt_tuning (2021)
- **Adapter**: PromptEmbedding / PromptTuningConfig
- **Paper**: https://arxiv.org/abs/2104.08691
- **Year**: 2021 (EMNLP 2021)
- **Abstract**: In this work we explore "prompt tuning," a simple yet effective mechanism for learning "soft prompts" to condition frozen language models to perform specific downstream tasks. Unlike the discrete text prompts used by GPT-3, soft prompts are learned through backpropagation and can be tuned to incorporate signal from any number of labeled examples. Our end-to-end learned approach outperforms GPT-3's "few-shot" learning by a large margin. More remarkably, through ablations on model size we show that prompt tuning becomes more competitive with model tuning as scale increases. At T5-XXL scale (11B parameters), prompt tuning matches the strong performance of model tuning while only requiring 0.01% task-specific parameters. This finding is especially relevant in that it opens the door to deploying a single large model for many tasks, since we can avoid the storage and serving costs of a separate copy for each task.
- **How it differs**: Learns only soft prompt embeddings prepended to input (no changes to model layers); freezes entire base model; becomes more competitive with full fine-tuning as model scale increases; simplest form of parameter-efficient tuning, focused only on input conditioning; different from Prefix-Tuning which affects all layers.

### randlora (2025)
- **Adapter**: RandLoraModel
- **Paper**: https://arxiv.org/abs/2502.00987
- **Year**: 2025 (ICLR 2025)
- **Abstract**: Low-Rank Adaptation (LoRA) is widely used for parameter-efficient fine-tuning, but its low-rank constraint limits expressiveness. We introduce RandLoRA, which enables full-rank updates while maintaining computational efficiency. RandLoRA decomposes the weight update as ΔW = ARB where R is a fixed random matrix and only A and B are learned. This approach preserves the same parameter count as LoRA but allows full-rank weight changes through the random matrix. We demonstrate that RandLoRA matches or exceeds full fine-tuning performance while maintaining LoRA's efficiency. The random projection acts as an implicit regularizer, improving generalization.
- **How it differs**: Achieves full-rank weight updates by learning matrices A and B that sandwich a fixed random matrix R (ΔW = ARB); removes rank bottleneck of standard LoRA while keeping same parameter count; random matrix provides implicit regularization; enables higher expressiveness than low-rank LoRA for tasks needing richer weight updates.

### road (2024)
- **Adapter**: RoadModel
- **Paper**: https://arxiv.org/abs/2409.00119
- **Year**: 2024
- **Abstract**: Rotation-Orthogonal Adaptation with Decomposition (ROAD) applies block-diagonal rotation matrices to activations, where each 2x2 block performs a scaled rotation. The transformation is `result = x * (α·cos θ) + rotate_half(x) * (α·sin θ)`, learning separate angle (θ) and magnitude (α) parameters. This decomposition preserves semantic structure (via rotation angle) while enabling flexible adaptation strength (via magnitude scaling). ROAD offers parameter efficiency through grouped rotations and maintains gradient flow through differentiable trigonometric operations.
- **How it differs**: Decouples rotation angle (θ - semantic direction) from magnitude (α - adaptation strength); applies grouped 2D rotations to activation pairs; more parameter-efficient than full orthogonal matrices; preserves hyperspherical structure like OFT but with explicit magnitude control; **ideal for contrastive steering** by scaling only α while preserving learned rotation directions.
- **Reversibility**: ✅ **Perfect** - scale only road_alpha (α) for coeff=±1.0 steering, preserving road_theta (θ) rotation directions
- **Code**: https://github.com/huggingface/peft/blob/6030f9160ed2fc17220f6f41382a66f1257b6a93/src/peft/tuners/road/layer.py#L387

### shira (2024)
- **Adapter**: ShiraModel (SHiRA — sparse high-rank adapter)
- **Paper**: https://arxiv.org/abs/2406.13175
- **Year**: 2024 (NeurIPS 2024 Workshop)
- **Abstract**: Parameter-efficient fine-tuning (PEFT) methods like LoRA enable adaptation of large models with minimal trainable parameters. However, low-rank constraints limit expressiveness. We propose Sparse High-Rank Adapters (SHiRA), which finetunes a small percentage (1-2%) of the base model's weights directly, selected based on importance scores. Unlike low-rank methods, SHiRA enables high-rank updates by sparsely modifying the original weight matrices. This approach bridges the gap between full fine-tuning and extreme parameter efficiency, achieving strong performance with only slight parameter overhead compared to LoRA.
- **How it differs**: Directly finetunes a sparse subset (1-2%) of base model weights selected by importance scoring instead of adding low-rank matrices; achieves high-rank updates through sparse direct modification; no auxiliary matrices; different paradigm from LoRA's additive decomposition; balances efficiency and expressiveness by targeted weight selection.

### trainable_tokens (implementation-focused)
- **Adapter**: TrainableTokensModel
- **Paper**: no explicit paper URL in model/config (implementation-focused)
- **Year**: N/A
- **Abstract**: (Implementation-focused adapter) Adds trainable token embeddings to the vocabulary, allowing the model to learn new tokens or adapt existing token representations for specific tasks. This is a lightweight approach for domain adaptation when vocabulary extension is needed.
- **How it differs**: Extends or modifies token embeddings (vocabulary layer) rather than transformer layer weights; learns new token representations or adapts existing ones; useful for domain-specific vocabulary or special tokens; orthogonal to LoRA which targets weight matrices in attention/FFN layers.

### vblora (2024)
- **Adapter**: VBLoRAModel
- **Paper**: https://arxiv.org/abs/2405.15179
- **Year**: 2024 (NeurIPS 2024)
- **Abstract**: Low-Rank Adaptation (LoRA) reduces the number of trainable parameters but still requires significant memory for larger ranks. We introduce VB-LoRA (Vector Bank LoRA), which represents adapter matrices as sparse linear combinations of shared vectors from a learned vector bank. Each layer selects top-k vectors from the bank and combines them with learned coefficients. This approach achieves extreme parameter efficiency: using only 0.4% of LoRA's parameters while maintaining comparable performance. VB-LoRA enables deployment of many specialized adapters with minimal storage overhead.
- **How it differs**: Replaces low-rank matrices with sparse admixtures (top-k selections) from shared vector banks; drastically fewer parameters than LoRA (0.4% of LoRA's count); vector bank shared across layers, only coefficients and selection indices are layer-specific; extreme compression via codebook-style parameterization.

### vera (2023)
- **Adapter**: VeraModel
- **Paper**: https://arxiv.org/abs/2310.11454
- **Year**: 2023 (ICLR 2024)
- **Abstract**: Low-Rank Adaptation (LoRA) has emerged as a popular method for parameter-efficient fine-tuning, but it still requires storing separate low-rank matrices for each layer and task. We introduce VeRA (Vector-based Random Matrix Adaptation), which shares a pair of frozen random low-rank matrices across all layers and learns only small scaling vectors per layer. Specifically, instead of learning B and A matrices for each layer, VeRA uses shared random matrices and learns only d-dimensional scaling vectors. This drastically reduces trainable parameters (often 10× fewer than LoRA) while maintaining competitive performance across diverse tasks.
- **How it differs**: Shares frozen random low-rank matrices (B, A) across all layers; learns only small scaling vectors (d-dimensional) per layer instead of full matrices; 10× fewer trainable parameters than LoRA; leverages random projection properties; trades learnable matrix flexibility for extreme parameter reduction.
- **Reversibility**: ✅ **Excellent** - scale only vera_lambda_b for coeff=±1.0 steering (proven working in current implementation)
- **Code**: https://github.com/huggingface/peft/blob/190f9873b15660d9092f70065c18e4993fe10d5b/src/peft/tuners/vera/layer.py#L136

### xlora (2024)
- **Adapter**: XLoraModel
- **Paper**: https://arxiv.org/abs/2402.07148
- **Year**: 2024
- **Abstract**: While LoRA enables efficient task-specific adaptation, deploying multiple LoRA adapters for different capabilities remains challenging. We propose X-LoRA, a mixture-of-experts approach that dynamically combines multiple LoRA adapters based on input hidden states. X-LoRA learns a gating mechanism that computes mixing weights for each adapter at each layer, enabling the model to leverage different expert adapters for different parts of the input. This allows a single model to handle diverse tasks simultaneously by routing through appropriate adapters, offering better multi-task performance than static adapter selection.
- **How it differs**: Mixture of expert LoRA adapters with learned gating/routing based on hidden states; dynamically combines multiple LoRAs per input instead of using single adapter; enables multi-task/multi-capability deployment with intelligent adapter selection; adds gating network overhead but achieves better composite performance than individual LoRAs.

---

## PEFT Release Highlights (v0.14.0 - v0.18.0)

> Source: https://github.com/huggingface/peft/releases

### v0.18.0 (Nov 2024): RoAd, ALoRA, Arrow, WaveFT, DeLoRA, OSF

**RoAd** (@ppetrushkov #2678): 2D Rotary Adaptation learns 2D rotation matrices that are applied using only element-wise multiplication, thus promising very fast inference with adapters in unmerged state. Remarkably, besides LoRA, RoAd is the only PEFT method that supports _mixed adapter batches_. This means that when you have loaded a model with multiple RoAd adapters, you can use all of them for different samples in the same batch, which is much more efficient than switching adapters between batches.

**ALoRA** (@kgreenewald #2609): Activated LoRA is a technique for causal language models, allowing to selectively enable LoRA adapters depending on a specific token invocation sequence in the input. This has the major benefit of being able to re-use most of the KV cache during inference when the adapter is only used to generate part of the response, after which the base model takes over again.

**Arrow & GenKnowSub** (@TheTahaaa #2644): Arrow is a dynamic routing algorithm between multiple loaded LoRAs. GenKnowSub is a technique built upon Arrow where the 'library' of LoRAs available to Arrow is first modified by subtracting general knowledge adapters (e.g., trained on subsets of Wikipedia) to enhance task-specific performance.

**WaveFT** (@Bilican #2560): Wavelet Fine-Tuning trains sparse updates in the wavelet domain of residual matrices, which is especially parameter efficient. It is very interesting for image generation, as it promises to generate diverse outputs while preserving subject fidelity.

**DeLoRA** (@mwbini #2780): Decoupled Low-rank Adaptation is similar to DoRA in so far as it decouples the angle and magnitude of the learned adapter weights. However, DeLoRA implements this in a way that promises to better prevent divergence. Moreover, it constrains the deviation of the learned weight by imposing an upper limit of the norm, which can be adjusted via the `delora_lambda` parameter.

**OSF** (@NikhilNayak-debug #2685): Orthogonal Fine-Tuning freezes the high-rank subspace of the targeted weight matrices and projects gradient updates to a low-rank subspace. OSF achieves good performance on continual learning tasks. While it is a bit memory intensive for standard fine-tuning processes, it is definitely worth checking out on tasks where performance degradation of previously learned tasks is a concern.

### v0.17.0 (Aug 2024): SHiRA, MiSS, LoRA for MoE

**SHiRA** (@kkb-code #2584): Sparse High Rank Adapters promise to offer a potential gain in performance over LoRAs - especially the concept loss when using multiple adapters is improved. Since the adapters only train on 1-2% of the weights and are inherently sparse, switching between adapters may be cheaper than with LoRAs.

**MiSS** (@JL-er #2604): Matrix Shard Sharing is an evolution of Bone, which, according to our PEFT method comparison benchmark, gives excellent results when it comes to performance and memory efficiency. At the same time, Bone will be deprecated in favor of MiSS and will be removed in PEFT v0.19.0. If you already have a Bone checkpoint, you can use `scripts/convert-bone-to-miss.py` to convert it into a MiSS checkpoint.

**LoRA for nn.Parameter** (#2638, #2665): LoRA is now able to target `nn.Parameter` directly! This can be especially useful for models with **Mixture of Expert** (MoE) layers, as those often use `nn.Parameter`s directly and cannot be targeted with `target_modules`. For example, for the Llama4 family of models, use `target_parameters=["feed_forward.experts.down_proj", "feed_forward.experts.gate_up_proj"]`.

### v0.16.0 (Jul 2024): LoRA-FA, RandLoRA, C³A

**LoRA-FA** (@AaronZLT #2468): LoRA-FA optimizer is based on `AdamW` and it increases memory efficiency of LoRA training. This means that you can train LoRA with less memory, or, with the same memory budget, use higher LoRA ranks, potentially getting better results.

**RandLoRA** (@PaulAlbert31 #2464): Similarly to VeRA, RandLoRA uses non-learnable random low rank matrices that are combined through learnable matrices. This way, RandLoRA can approximate full rank updates of the weights. Training models quantized with bitsandbytes is supported.

**C³A** (@Phoveran #2577): Circular Convolution Adaptation can overcome the limit of low rank adaptations as seen e.g. in LoRA while still promising to be fast and memory efficient.

### v0.15.0 (Mar 2024): CorDA, Trainable Tokens

**CorDA** (@iboing and @5eqn #2231): Context-Oriented Decomposition Adaptation is a task-driven initialization method with two modes, knowledge-preservation and instruction-preservation, both using external data to select ranks intelligently. The former can be used to select those ranks that correspond to weights not affiliated with knowledge from, say, a QA dataset. The latter can be used to select those ranks that correspond most to the task at hand (e.g., a classification task).

**Trainable Tokens** (#2376): The new Trainable Tokens tuner allows for selective training of tokens without re-training the full embedding matrix, e.g. when adding support for reasoning / thinking tokens. This is a lot more memory efficient and the saved checkpoint is much smaller. It can be used standalone or in conjunction with LoRA adapters by passing `trainable_token_indices` to `LoraConfig`.

### v0.14.0 (Dec 2023): EVA, CPT, Bone

**CPT** (@tsachiblau): Context-aware Prompt Tuning is a combination of In-Context Learning and Prompt Tuning in the sense that, for each training sample, it builds a learnable context from training examples in addition to the single training sample. Allows for sample- and parameter-efficient few-shot classification and addresses recency-bias.

**EVA** (@sirluk): Explained Variance Adaptation uses SVD on minibatches of finetuning data to initialize the LoRA weights and is also able to re-allocate the ranks of the adapter based on the explained variance ratio (derived from SVD). Thus, this initialization method can yield better initial values and better rank distribution.

**Bone** (@JL-er): Block Affine Adaptation utilizes presumed sparsity in the base layer weights to divide them into multiple sub-spaces that share a single low-rank matrix for updates. Compared to LoRA, Bone has the potential to significantly reduce memory usage and achieve faster computation. (deprecated in favor of MiSS)

---

## Extra outside of PEFT

> See also: PEFT developer guide https://github.com/huggingface/peft/blob/261366de2e40cde64b702d6b9c527081ad850549/docs/source/developer_guides/lora.md
> See also: PEFT conceptual guide https://github.com/huggingface/peft/blob/261366de2e40cde64b702d6b9c527081ad850549/docs/source/conceptual_guides/adapter.md#L4

### antipasto (2026)
- **Adapter**: Not in PEFT (custom steering method)
- **Paper**: https://arxiv.org/abs/2601.07473
- **Year**: 2026
- **Code**: https://github.com/wassname/AntiPaSTO
- **Abstract**: As models grow more capable, humans cannot reliably verify what they say. Scalable steering requires methods that are internal, self-supervised, and transfer out-of-distribution; existing methods satisfy some but not all three. We introduce AntiPaSTO, which separates representations along an antiparallel axis (+1/-1 produce opposite shifts), with coherence constraints preventing collapse. Human input is minimal: two contrasting words inserted into template sentences, no preference labels. Using 800 such pairs on Gemma-3-1B, AntiPaSTO beats prompting baselines by 6.9x on DailyDilemmas and maintains bidirectional control where prompting triggers refusal.
- **How it differs**: Self-supervised honesty steering via anti-parallel representations; uses only word pairs (no preference labels) to create contrastive steering vectors; achieves bidirectional control (+1/-1 scaling) for reversible steering; minimal human input required; designed specifically for honesty/alignment steering rather than general fine-tuning.
- **Reversibility**: ✅ **Perfect** - designed for coeff=±1.0 steering with antiparallel axis

### Other notable methods

- **ETHER** https://arxiv.org/html/2405.20271v1 (not in PEFT)

- **BiPDO** https://arxiv.org/abs/2406.00045
  - > Researchers have been studying approaches to steer the behavior of Large Language Models (LLMs) and build personalized LLMs tailored for various applications. While fine-tuning seems to be a direct solution, it requires substantial computational resources and may significantly affect the utility of the original LLM. Recent endeavors have introduced more lightweight strategies, focusing on extracting "steering vectors" to guide the model's output toward desired behaviors by adjusting activations within specific layers of the LLM's transformer architecture. However, such steering vectors are directly extracted from the activations of human preference data and thus often lead to suboptimal results and occasional failures, especially in alignment-related scenarios. This work proposes an innovative approach that could produce more effective steering vectors through bi-directional preference optimization. Our method is designed to allow steering vectors to directly influence the generation probability of contrastive human preference data pairs, thereby offering a more precise representation of the target behavior. By carefully adjusting the direction and magnitude of the steering vector, we enabled personalized control over the desired behavior across a spectrum of intensities. Extensive experimentation across various open-ended generation tasks, particularly focusing on steering AI personas, has validated the efficacy of our approach. Moreover, we comprehensively investigate critical alignment-concerning scenarios, such as managing truthfulness, mitigating hallucination, and addressing jailbreaking attacks. Remarkably, our method can still demonstrate outstanding steering effectiveness across these scenarios. Furthermore, we showcase the transferability of our steering vectors across different models/LoRAs and highlight the synergistic benefits of applying multiple vectors simultaneously. 

- **repeng** https://github.com/vgel/repeng
  - This is library that quite robust and popular for steering with PCA vectors in hidden space, we use it's prompting setup, and use it as a baseline. It's been cited in several papers
  - > A Python library for generating control vectors with representation engineering. Train a vector in less than sixty seconds!

- **PiSSA** https://arxiv.org/html/2404.02948v4
  - This paper decomposes each weight matrix W into U S V + W_residual like us
  - > To parameter-efficiently fine-tune (PEFT) large language models (LLMs), the low-rank adaptation (LoRA) method approximates the model changes Δ⁢W∈ℝm×n through the product of two matrices A∈ℝm×r and B∈ℝrˣⁿ, where r≪min⁡(m,n), A is initialized with Gaussian noise, and B with zeros. LoRA freezes the original model W and updates the "Noise & Zero" adapter, which may lead to slow convergence. To overcome this limitation, we introduce Principal Singular values and Singular vectors Adaptation (PiSSA). PiSSA shares the same architecture as LoRA, but initializes the adaptor matrices A and B with the principal components of the original matrix W, and put the remaining components into a residual matrix Wr⁢e⁢s∈ℝm×n which is frozen during fine-tuning. Compared to LoRA, PiSSA updates the principal components while freezing the "residual" parts, allowing faster convergence and enhanced performance. Comparative experiments of PiSSA and LoRA across 11 different models, ranging from 184M to 70B, encompassing 5 NLG and 8 NLU tasks, reveal that PiSSA consistently outperforms LoRA under identical experimental setups. On the GSM8K benchmark, Gemma-7B fine-tuned with PiSSA achieves an accuracy of 77.7%, surpassing LoRA's 74.53% by 3.25%. Due to the same architecture, PiSSA is also compatible with quantization to further reduce the memory requirement of fine-tuning. Compared to QLoRA, QPiSSA (PiSSA with 4-bit quantization) exhibits smaller quantization errors in the initial stages. Fine-tuning LLaMA-3-70B on GSM8K, QPiSSA attains an accuracy of 86.05%, exceeding the performance of QLoRA at 81.73%. Leveraging a fast SVD technique, PiSSA can be initialized in only a few seconds, presenting a negligible cost for transitioning from LoRA to PiSSA.

- **SSVD** https://arxiv.org/html/2509.02830v1
  - This paper rotates the V matrix, which is very novel and we use, it has good results (generalisation which is better than just parameter efficiency)
  - > Parameter-efficient fine-tuning (PEFT) has emerged as a scalable solution for adapting large foundation models. While low-rank adaptation (LoRA) is widely used in speech applications, its state-of-the-art variants, e.g., VeRA, DoRA, PiSSA, and SVFT, are developed mainly for language and vision tasks, with limited validation in speech. This work presents the first comprehensive integration and benchmarking of these PEFT methods within ESPnet. We further introduce structured SVD-guided (SSVD) fine-tuning, which selectively rotates input-associated right singular vectors while keeping output-associated vectors fixed to preserve semantic mappings. This design enables robust domain adaptation with minimal trainable parameters and improved efficiency. We evaluate all methods on domain-shifted speech recognition tasks, including child speech and dialectal variation, across model scales from 0.1B to 2B. All implementations are released in ESPnet to support reproducibility and future work.

- **DoRA** https://arxiv.org/html/2306.08990v2 
  - Separates magnitude and direction and has become a popular and strong LoRA baseline
  - > DoRA decomposes the pre-trained weight into two components, magnitude and direction, for fine-tuning, specifically employing LoRA for directional updates to efficiently minimize the number of trainable parameters. By employing ours, we enhance both the learning capacity and training stability of LoRA while avoiding any additional inference overhead.

- **SVFT** https://arxiv.org/html/2405.19597v1
  - This paper updates the S of the SVD of each weight matrix like us
  - > Popular parameter-efficient fine-tuning (PEFT) methods, such as LoRA and its variants, freeze pre-trained model weights 𝐖 and inject learnable matrices 𝚫⁢𝐖. These 𝚫⁢𝐖 matrices are structured for efficient parameterization, often using techniques like low-rank approximations or scaling vectors. However, these methods typically show a performance gap compared to full fine-tuning. Although recent PEFT methods have narrowed this gap, they do so at the cost of additional learnable parameters. We propose SVFT, a simple approach that fundamentally differs from existing methods: the structure imposed on 𝚫⁢𝐖 depends on the specific weight matrix 𝐖. Specifically, SVFT updates 𝐖 as a sparse combination of outer products of its singular vectors, training only the coefficients (scales) of these sparse combinations. This approach allows fine-grained control over expressivity through the number of coefficients. Extensive experiments on language and vision benchmarks show that SVFT1 recovers up to 96% of full fine-tuning performance while training only 0.006 to 0.25% of parameters, outperforming existing methods that only recover up to 85% performance using 0.03 to 0.8% of the trainable parameter budget.

---

## References

- PEFT repository: https://github.com/huggingface/peft
- PEFT releases: https://github.com/huggingface/peft/releases
- PEFT developer guide (LoRA): https://github.com/huggingface/peft/blob/261366de2e40cde64b702d6b9c527081ad850549/docs/source/developer_guides/lora.md
- PEFT conceptual guide (adapters): https://github.com/huggingface/peft/blob/261366de2e40cde64b702d6b9c527081ad850549/docs/source/conceptual_guides/adapter.md#L4
- PEFT contributing guide: https://huggingface.co/docs/peft/developer_guides/contributing