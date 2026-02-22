Title: 2409.00119v2.pdf

URL Source: https://arxiv.org/pdf/2409.00119

Published Time: Tue, 05 Nov 2024 02:26:41 GMT

Number of Pages: 25

Markdown Content:
# 3-in-1: 2D Rotary Adaptation for Efficient Finetuning, Efficient Batching and Composability 

Baohao Liao 1,2∗ Christof Monz 11Language Technology Lab, University of Amsterdam 

> 2

eBay Inc., Aachen, Germany Code: https://github.com/BaohaoLiao/road 

## Abstract 

Parameter-efficient finetuning (PEFT) methods effectively adapt large language models (LLMs) to diverse downstream tasks, reducing storage and GPU memory demands. Despite these advantages, several applications pose new challenges to PEFT beyond mere parameter efficiency. One notable challenge involves the effi-cient deployment of LLMs equipped with multiple task- or user-specific adapters, particularly when different adapters are needed for distinct requests within the same batch. Another challenge is the interpretability of LLMs, which is crucial for understanding how LLMs function. Previous studies introduced various approaches to address different challenges. In this paper, we introduce a novel method, RoAd, which employs a straightforward 2D rotation to adapt LLMs and addresses all the above challenges: (1) RoAd is remarkably parameter-efficient, delivering optimal performance on GLUE, eight commonsense reasoning tasks and four arithmetic reasoning tasks with < 0.1% trainable parameters; (2) RoAd facilitates the efficient serving of requests requiring different adapters within a batch, with an overhead comparable to element-wise multiplication instead of batch matrix multiplication; (3) RoAd enhances LLM’s interpretability through integration within a framework of distributed interchange intervention, demonstrated via composition experiments. 0.0 0.1 0.2       

> 87.75
> 88.00
> 88.25
> 88.50
> 88.75
> Full FT
> Adapter
> LoRA
> Adapter FFN
> RED
> LoReFT
> RoAd
> GLUE
> 0123
> 70
> 75
> 80
> Adapter P
> Adapter S
> DoRA
> LoRA
> Prefix
> LoReFT
> RoAd
> GPT3.5
> Commonsense
> 0123
> 40.0
> 42.5
> 45.0
> 47.5
> 50.0
> 52.5
> Adapter P
> Adapter S
> LoRA
> Prefix
> LoReFT
> RoAd
> Arithmetic
> Trainable parameters (%)
> Accuracy

Figure 1: Performance of various PEFT methods on the GLUE benchmark, eight commonsense reasoning tasks and four arithmetic reasoning tasks with RoBERTa-large or LLaMA-13B. 

## 1 Introduction 

Large language models (LLMs), trained on extensive web-scale datasets to perform tasks such as predicting masked words [ 8 , 31 , 45 ] or anticipating the next word in a sentence [ 17 , 52 , 53 ], 

> ∗

Correspondence to b.liao@uva.nl. Please go to https://arxiv.org/abs/2409.00119 for the newest version. 38th Conference on Neural Information Processing Systems (NeurIPS 2024). 

> arXiv:2409.00119v2 [cs.LG] 4 Nov 2024

demonstrate remarkable effectiveness across a range of NLP applications. For tasks where the data distribution diverges from that of the pretraining corpus, finetuning emerges as an effective way to tailor an LLM to specific requirements. Leveraging the capabilities of LLMs, recent studies [ 13 , 14 ,22 , 23 , 25 , 27 , 42 , 60 , 62 , 65 ] demonstrate that training only a subset of an LLM’s parameters can yield performance on par with full finetuning. This approach, termed parameter-efficient finetuning (PEFT), provides two primary advantages: (1) It reduces the storage requirements for trained parameters, as it necessitates preserving only a universal LLM alongside a minimal set of task-specific parameters; (2) It decreases GPU memory consumption during finetuning, owing to the reduction in optimizer state sizes which correlate directly with the number of trainable parameters. With the evolution of PEFT, concerns extend beyond mere parameter efficiency. PEFT encounters a variety of challenges brought forth by diverse applications. A significant challenge is the efficient deployment of personalized or task-specific LLMs [ 25 , 57 ]. These applications frequently require distinct sets of trained parameters for different tasks or users. When multiple users submit requests simultaneously, it becomes crucial to process these requests collectively in a single batch. Given that each request may require a unique set of parameters, using batch matrix multiplication can efficiently handle these requests by leveraging GPU parallelism. However, the batch matrix multiplication still incurs considerable overhead [1, 57], necessitating the exploration of more efficient methods. Another challenge is the interpretability of LLMs that contain a billion-scale of parameters, making it difficult to explore their mechanism. PEFT provides an alternative approach by constraining the number of trainable parameters, thereby aiding in interpretability. Recent advancements in PEFT methods, particularly those focusing on representation editing [ 54 , 60 , 67 ], can be incorporated within an intervention framework [ 11 ]. This integration enhances their capability for interpretability, offering a more manageable means of dissecting the operational intricacies of LLMs. In this paper, we introduce a novel technique termed 2D rotary adaptation (RoAd) which efficiently adapts LLMs using a minimal number of trainable parameters. Furthermore, RoAd enhances both batching efficiency and composability. Our initial investigation reveals that finetuning primarily alters the angular components of the representations in pretrained LLMs, rather than their magnitudes (Section §3.1). Based on this observation, we employ a strategy of rotating certain subspaces within the representations to emulate finetuning effects. Specifically, we implement a 2D rotational approach on the representations and develop three distinct variants of RoAd (Section §3.2). To assess the efficacy of RoAd, we perform comprehensive evaluations on the GLUE benchmark [ 56 ], eight commonsense reasoning tasks and four arithmetic reasoning tasks, utilizing RoBERTa [31 ] and LLaMA [ 52 , 53 ] (Section §4.1). The results consistently show that RoAd surpasses other PEFT methods while maintaining a significantly reduced scale of trainable parameters ( < 0.1% ), as depicted in Figure 1. Additionally, RoAd employs element-wise rather than matrix multiplication, which notably improves throughput when serving heterogeneous requests within the same batch, achieving twice the throughput of LoRA [ 14 ] (Section §4.2). Furthermore, RoAd can be seamlessly integrated within an intervention framework [ 11 ], thereby enhancing model interpretability. We illustrate this through a composition experiment, demonstrating RoAd’s capacity to merge weights trained for different tasks and display a new capability (Section §4.3). 

## 2 Background 

In this section, we outline the challenges tackled in this work, illustrating the constraints of existing methods and objectives that drive the development of the proposed method, RoAd. 

2.1 Parameter-efficient finetuning (PEFT) 

Existing PEFT techniques can be categorized into three groups: adapter-based, prompt-based, and latency-less methods. Adapter-based methods [ 12 , 13 , 42 ] incorporate adapters either in parallel with or sequentially to the existing Transformer [ 55 ] modules. This incorporation necessitates modifications to the LLM architecture, consequently adding extra latency during inference. Prompt-based methods [ 19 , 21 , 43 ] enhance the input by appending new trainable tokens, which lengthens the sequence and thereby increases the computational overhead during inference. Latency-less methods, such as LoRA [ 14 ] and its variants [ 22 , 27 , 65 ], apply low-rank matrices to adapt the pretrained weights. These matrices can be seamlessly integrated into the existing weight matrices following 2finetuning, thus preserving the original LLM architecture. Specifically, LoRA adapts an LLM as 

W = W 0 +∆ W , where W 0 ∈ Rd1×d2 is the pretrained weight and ∆W = BA with B ∈ Rd1×r ,

A ∈ Rr×d2 , r ≪ d1 and r ≪ d2. Our proposed method, RoAd, aligns with the latency-less category and integrates effortlessly into the existing linear layer without imposing additional overhead during inference. Moreover, RoAd demonstrates exceptional parameter efficiency. The quantity of its trainable parameters is equivalent to that of a LoRA module with a rank r = 0 .5.

Orthogonal finetuning. Drawing on the concept of hyperspherical energy and its role in characteriz-ing generalization [ 28 , 29 ], OFT [ 44 ] introduces orthogonal finetuning, an effective PEFT method for finetuning text-to-image diffusion models. Specifically, OFT implements an orthogonal matrix 

R ∈ Rd1×d1 to the pretrained weight W 0, so the input x ∈ Rd1 to a linear layer after adaptation be-comes z = ( RW 0)⊤x. R is parameter-efficient because it is a block-diagonal matrix with n blocks as R = diag (R1, ..., Ri, ..., Rn), where each block Ri ∈ Rw×w has a dimension w = d1/n . To maintain orthogonality, Ri is derived using Cayley parameterization: Ri = ( I +Qi)( I −Qi)−1 with 

Qi ∈ Rw×w being a skew-symmetric matrix ( Qi = −Q⊤ 

> i

). In sum, {Qi}ni=1 serve as the trainable parameters and R is constructed from them with Cayley parameterization. Subsequent advancement, BOFT [ 30 ], leverages butterfly factorization to further refine OFT’s parameter efficiency. However, both OFT and BOFT, due to their reliance on matrix inversions in the Cayley parameterization and increased storage of intermediate activations, necessitate additional GPU memory and increase training duration compared to other PEFT approaches. Conversely, RoAd, which may be considered as a specialized case of OFT with w = 2 , offers a faster and more memory-efficient solution by inherently maintaining orthogonality without requiring further parameterization. 

2.2 Batching 

Batching in this context refers to processing multiple heterogeneous requests, each requiring different adapters 2 for inference. This scenario commonly arises when serving personalized or task-specific LLMs. Specifically, we consider a setup where distinct adapters instead of a shared adapter are finetuned for various tasks to achieve optimal performance. During inference, each request in a batch pertains to a different task and necessitates a unique adapter. Consider that we have finetuned distinct LoRA modules for b tasks, denoted as {Ai, Bi}bi=1 . For a batch of b requests represented as X ∈ Rb×l×d1 , where l is the maximum sequence length across the requests, each request requires a different LoRA module. To exploit the parallel processing capabilities of GPUs, the output Z of a linear layer can be computed as follows: First, the output from the pretrained layer is computed as Z0 = torch.mm (X, W 0). Subsequently, the intermediate output from the first low-rank matrix, ˆB ∈ Rb×d1×r (a concatenation of {Bi}bi=1 ), is obtained as Z10 = torch.bmm (X, ˆB). The output from the second low-rank matrix, ˆA ∈ Rb×r×d2 (a concatenation of {Ai}bi=1 ), follows as Z1 = torch.bmm (Z10 , ˆA). Finally, these outputs are summed to produce Z = Z0 + Z1. It is noteworthy that batch matrix multiplication (BMM), as implemented in torch.bmm , often introduces substantial overhead [ 1], reducing throughput and increasing latency, which adversely impacts user experience in time-sensitive applications. In contrast, prompt-based methods circumvent the use of BMM by appending trainable tokens to each request, simplifying the computational process. However, prompt-based methods with long prompt tokens are difficult to optimize, which degrades performance compared to other PEFTs [ 14 , 15 ]. (IA) 3 [ 25 ] proposes adapting LLM by multiplying the output from a linear layer with a trainable vector, involving only element-wise multiplication for efficient batching. A recent development, FLoRA [ 58 ], builds on (IA) 3 by employing two low-rank matrices while maintaining element-wise operations. Although our proposed method, RoAd, requires BMM, its sparse structure allows a reformulation of BMM and results in an overhead equivalent to element-wise multiplication. 

2.3 Intervention and composability 

Numerous studies [ 10 , 11 , 37 , 38 , 40 ] have provided support for the linear representation hypothesis [ 35 , 46 , 49 ] that concepts are represented within linear subspaces of neural network representations. To examine if a concept is captured within a linear subspace of a representation, Geiger et al. [11] 

> 2Adapter here means the trained parameters since LoRA’s architecture is also similar to an adapter.

30.00 0.25 0.50 0.75             

> D
> 0.025
> 0.050
> 0.075
> 0.100
> 0.125
> 0.150
> M
> Full Finetuning
> 0.6 0.7 0.8 0.9
> D
> 0.02
> 0.04
> 0.06
> 0.08
> 0.10
> M
> LoRA
> RTE MRPC STS-B CoLA
> Tasks
> 0
> 20
> 40
> 60
> Score
> z=WTx
> z=WTx, random
> z=cos(W, x)
> z= || W|| || x||

Figure 2: Pilot study for the pretrained and finetuned representations. Left & Middle : The change in magnitude and angle of representations between pretrained and finetuned LLM using full finetuning or LoRA. Right : The disentanglement experiment of magnitude and angle of pretrained representation. suggests employing a distributed interchange intervention (DII) defined as: DII (b, s, R) = b + R⊤(Rs − Rb ) (1) 

b denotes the hidden representation generated at row i and column k when the model processes an input, while s represents the corresponding representation when the model processes a different input. The matrix R ∈ Rr×d1 , consisting of orthogonal rows, serves as a low-rank projection matrix where d1 is the dimension of the representation and r is the subspace dimension under intervention. Equation (1) illustrates the application of a DII to b using a counterfactual source representation s.3

Drawing inspiration from this established framework, a recent study, LoReFT [ 61 ], introduces a method for finetuning specific positions of the representations to adapt LLM. This study further demonstrates that several prior approaches of representation editing [ 54 , 60 , 67 ] can be effectively integrated within this framework. Interestingly, the application of RoAd to representations can also be conceptualized as DII, offering interpretability potential. To demonstrate one aspect of interpretability for RoAd, we primarily conduct a qualitative experiment focused on task composition. This experiment involves combining the weights of models trained on distinct tasks to showcase the capability for multitasking learning without the need for additional adaptation [16, 20, 61, 64, 66]. 

## 3 Method 

In this section, we first perform two pilot studies to ascertain the key factor influencing the adaptation of LLMs. Following this, we present our proposed method, the 2D rotary adaptation (RoAd), which serves as an effective PEFT method addressing the various challenges outlined in Section §2. 

3.1 Pilot study Study 1: Variations in magnitude and angular displacement. Assume x0, x ∈ Rd1 are represen-tations of the same token from a pretrained and finetuned LLM, respectively. We define the relative change in magnitude as ∆M = ∥x∥2 − ∥ x0∥2 /∥x0∥2 and compute the angular displacement as 

∆D = cos( x, x0) ∈ [−1, 1] . A larger ∆M and a smaller ∆D indicate more significant changes in magnitude and angular displacement, respectively. Our study involves: (1) finetuning RoBERTa-base [31 ] on the SST-2 task [ 50 ] using either full finetuning or LoRA; (2) extracting representations x0

and x from the output of the second-last Transformer block for the [CLS] token across all samples in the development set, followed by computing ∆M and ∆D.4 As depicted in Figure 2 (Left and Middle), there is a more pronounced change in ∆D than in ∆M for both full finetuning and LoRA. 5

Study 2: Disentanglement of magnitude and angle. To ascertain whether angular or magnitude adjustments are more critical for finetuning, we implement a disentanglement study. This involves freezing RoBERTa-base and appending a two-layer classifier on top of it. The first layer of this 

> 3

We adopt notation systems from Wu et al. [61]. 

> 4

Please refer to Figure B.1 for all layers. 

> 5

There are two other interesting observations: (1) An increase in magnitude change correlates with a larger angular displacement; (2) Compared to LoRA, full finetuning has a bigger change in magnitude and angle (for all layers, see Figure B.1), which is in line with a recent finding that LoRA learns less and forgets less [2]. 

4classifier incorporates a weight matrix W ∈ Rd1×d1 . Under standard operations, the output from this layer is computed as z = W ⊤x0. To distinctly evaluate the impacts of magnitude and angle, we modify the output to retain only the magnitude component as zi = ∥W:,i ∥2 · ∥ x0∥2, or solely the angular component as zi = cos( W:,i , x0) (zi is the i th element of z). The modified classifier was then finetuned on four GLUE tasks with different metrics detailed in Table C.1. Additionally, a weak baseline employing a randomly initialized RoBERTa-base is included. As shown in Figure 2 (Right), angular information is paramount in finetuning, whereas reliance solely on magnitude information even leads to inferior results compared to the random backbone. Both studies indicate that angular information is more crucial than magnitude information for adapting a pretrained LLM to a downstream task. However, rotating the entire d1 dimensions of the representation for finetuning incurs substantial computational costs. These costs are primarily reflected in a large number of trainable parameters, necessitating a dense matrix R ∈ Rd1×d1 , and in the requirement to maintain its orthogonality. Could we only rotate a subspace of the representation and design a R that is always orthogonal without any parameterization as OFT [ 44 ]? The first idea that comes to our mind is 2D rotation which only rotates two dimensions at a time and inherently maintains orthogonality. 

3.2 2D rotary adaptation ℎ! ℎ"ℎ#ℎ$ ℎ%&# ℎ%         

> 𝑧 !𝑧 "𝑧 #𝑧 $𝑧 %&#𝑧 %
> ℎ!
> ℎ"
> 𝑧 !
> 𝑧 "
> 𝜃 !
> 𝑧 !
> #+𝑧 "
> #!
> =𝛼 !ℎ!
> #+ℎ"
> #!

. . . 

. . . 

Figure 3: Overview of RoAd 1.Suppose that W 0 ∈ Rd1×d2 is the pretrained weight of a linear layer, 

x ∈ Rd1 is the input of a token to this linear layer, R ∈ Rd2×d2 is the rotation matrix, the adapted output from the linear layer is z =

Rh = R(W 0⊤x). The rotation matrix R is defined as follows: 

R = diag (R1, R2, ..., Rd2/2) with Ri =

cos θi − sin θi

sin θi cos θi



(2) The trainable parameters are denoted as {θi}d2/2 

> i=1

. This 2D rotary adaptation involves rotating pairs of adjacent dimensions of h, specif-ically dimensions 2i − 1 and 2i, using the rotation matrix Ri.6 The rotation matrix R is characterized by its parameter efficiency, which is attributed to its sparse structure and the parameter sharing within each block Ri. Additionally, R

can be integrated directly into the existing pretrained weights, forming W = W 0R⊤, which does not incur additional computational costs during inference. This design closely mirrors RoPE [ 51 ], with the notable difference that in our RoAd, θi is trainable and Ri does not incorporate positional information. The overview of RoAd is shown in Figure 3. 

Relaxation to orthogonality. Referring to Figure 2 (Right), while reliance predominantly on angular information substantially outperforms reliance on magnitude information, it remains less effective than using both angular and magnitude information for the tasks of MRPC, STS-B, and CoLA. Furthermore, both fully- and LoRA-finetuned LLMs exhibit slight adaptations in magnitude, as depicted in Figure 2 (Left and Middle). Consequently, we modify Ri by incorporating αi to regulate the magnitude. We define a general Ri as follows: 

Ri =

αi, 11 cos θi, 11 −αi, 12 sin θi, 12 

αi, 21 sin θi, 21 αi, 22 cos θi, 22 



(3) We develop three variants of RoAd by altering the configuration of shared parameters as outlined in Table 1. RoAd 1 introduces a minimal change to Equation (2) by incorporating a scaling factor αi.RoAd 1 already shows impressive results for most tasks in Section §4.1. For some knowledge-intensive tasks, we observe that RoAd 2 and RoAd 4 obtain better results with more trainable parameters. To preserve the starting point of LLMs [23], we always initialize αi = 1 and θi = 0 .

Batching. In practice, we don’t need to save R as a sparse matrix and do matrix multiplication. Taking RoAd 1 as an example in Equation (4), we only save two vectors: R1 and R2. Then 

z = Rh = R1 ⊗ h + R2 ⊗ ˆh, where ˆh is a rearranged version of h and ⊗ denotes element-wise multiplication. This reformulation not only simplifies the representation of R but also enhances the efficiency of batching in RoAd, relying solely on element-wise multiplications rather than BMM. 

> 6

The index in this work starts from 1 instead of 0. 

5Table 1: A summarization of three RoAd variants. 

RoAd ? αi θi #Trainable 1 αi, 11 = αi, 12 = αi, 21 = αi, 22 = αi θi, 11 = θi, 12 = θi, 21 = θi, 22 = θi d2

2 αi, 11 = αi, 12 αi, 21 = αi, 22 θi, 11 = θi, 12 θi, 21 = θi, 22 2d2

4 αi, 11 ̸ = αi, 12 ̸ = αi, 21 ̸ = αi, 22 θi, 11 ̸ = θi, 12 ̸ = θi, 21 ̸ = θi, 22 4d2                                  

> z=Rh =R1⊗h+R2⊗ˆh
> =
> 
> α1cos θ1
> α1cos θ1
> α2cos θ2
> α2cos θ2
> ...
> αd2/2cos θd2/2
> αd2/2cos θd2/2
> 
> ⊗
> 
> h1
> h2
> h3
> h4
> ...
> hd2−1
> hd2
> 
> +
> 
> α1sin θ1
> α1sin θ1
> α2sin θ2
> α2sin θ2
> ...
> αd2/2sin θd2/2
> αd2/2sin θd2/2
> 
> ⊗
> 
> −h2
> h1
> −h4
> h3
> ...
> −hd2
> hd2−1
> 
> (4)

Composability. RoAd can be incorporated into the DII framework as Φ( h) = Rh = h + R(h −

R⊤h), with Rs in Equation (1) being set to h. Although a degree of relaxation is introduced to the orthogonality of R, it is important to note that the rows of R remain orthogonal to each other within non-adjacent segments of the same block, Ri. This offers a possibility for composability. We can finetune some rows on one task and other orthogonal rows on another task. Since they are orthogonal to each other, these two tasks should minimally affect each other, and the combination of these rows after finetuning could bring new multitasking learning ability. RoAd can be considered as a special case of OFT [ 44 ] with w = 2 . However, it is much more parameter- and memory-efficient and faster. Please refer to Section §D.1 for a detailed discussion. 

## 4 Experiments 

In this section, we begin by implementing RoAd to finetune various LLMs across three benchmarks. Subsequently, we illustrate its efficiency in batching processes and demonstrate its composability. Unless otherwise noted, RoAd is applied to all linear layers within the LLMs. All of our experiments are conducted on A100 80GB GPU with the frameworks, Transformers [59] and PEFT [34]. 

4.1 Results on downstream tasks Natural language understanding (NLU). We evaluate the effectiveness of RoAd on the GLUE benchmark [ 56 ] for its ability of NLU with RoBERTa [ 31 ] as the backbone. Unlike many previous works [ 14 , 22 , 23 , 31 , 65 ] that employ the GLUE development sets for both validation and testing, here we partition the development set into distinct validation and test subsets to mitigate the risk of overfitting. For comprehensive information regarding the split of the development set, the search space of hyperparameters, the optimal hyperparameter configurations, and other details crucial for reproducibility, please see Section §C.1. As shown in Table 2, RoAd 1 outperforms all other PEFT methods with < 0.1% trainable parameters for both sizes of RoBERTa on average, being the only PEFT method that matches or outperforms full finetuning. These results show that 2D rotation (with a few scaling) can efficiently adapt LLM. 

Commonsense reasoning . In assessing the capacity of LLaMA [ 52 ] for commonsense reasoning, we focus on eight representative tasks: BoolQ [ 4], PIQA [ 3], SIQA [ 48 ], HellaSwag [ 63 ], WinoGrande [ 47 ], ARC-e, ARC-c [ 5], and OBQA [ 36 ]. The setting here contrasts with the NLU experiments where each task involves finetuning a separate LLM. Instead, we adopt a unified strategy by finetuning a single LLM across all tasks as delineated in Hu et al. [15] . Such a setting is designed to mitigate overfitting and aligns more closely with real-world applications. Specifically, the training and test sets from these eight tasks are reformulated according to a predefined template, so all tasks can be trained or evaluated in a generative way. For all finetuning experiments on LLaMA, we follow a recipe in Table C.5 without extensive searching. Please see Section §C.2 for more training details. 6Table 2: Results on the held-out GLUE development set with RoBERTa as the backbone. We report matched accuracy for MNLI, Matthew’s correlation for CoLA, Pearson correlation for STS-B and accuracy for other tasks. The best and second-best results are in bold and underlined, respectively, being the same for other tables. The percentage of trainable parameters is calculated without considering the classifier head. RoAd 1(fc1) means that we only insert the RoAd 1 module to the first feed-forward layer, to match the #Params. of RED and LoReFT. Results of methods denoted by ∗

and ⋄ are from Wu et al. [60] and Wu et al. [61] , respectively. Otherwise, average results from three random runs are reported. Refer to Table C.4 for the standard deviation.                                                                                                                                                                                                                      

> Model Method #Params. RTE MRPC STS-B CoLA SST-2 QNLI QQP MNLI Avg.
> Full FT ∗100.00% 78.3 87.9 90.6 62.4 94.4 92.5 91.7 87.3 85.6 Adapter ∗0.32% 76.5 88.4 90.5 60.9 93.3 92.5 90.5 87.0 85.0 LoRA ∗0.24% 75.3 88.7 90.3 59.7 93.9 92.6 90.4 86.6 84.7 Adapter FNN ∗0.24% 77.7 88.8 90.4 58.5 93.0 92.0 90.2 87.1 84.7 BOFT 0.16% 71.4 87.5 89.6 55.3 92.5 91.4 89.4 85.3 82.8 base OFT w=2 0.10% 74.4 87.6 89.4 50.4 92.8 90.9 89.2 83.9 82.3 BitFit ∗0.08% 69.8 88.0 89.5 54.0 94.0 91.0 87.3 84.7 82.3 (IA) 30.04% 75.3 87.1 90.0 60.4 94.0 91.8 89.2 85.8 84.2 RED ∗0.02% 78.0 89.2 90.4 61.0 93.9 90.7 87.2 83.9 84.3 LoReFT ⋄0.02% 79.0 89.2 90.0 60.4 93.4 91.2 87.4 83.1 84.2
> RoAd 10.07% 78.9 89.2 90.5 64.4 93.9 91.9 89.6 86.3 85.6 RoAd 1(fc1) 0.03% 79.1 90.2 90.2 60.9 94.6 91.6 88.7 85.4 85.1 Full FT ∗100.00% 85.8 91.7 92.6 68.2 96.0 93.8 91.5 88.8 88.6 Adapter ∗0.25% 85.3 90.5 91.5 65.4 95.2 94.6 91.4 90.1 88.0 LoRA ∗0.23% 86.3 89.8 91.7 65.5 96.0 94.7 90.7 90.1 88.1 large Adapter FNN ∗0.23% 84.8 90.5 90.2 64.4 96.1 94.3 91.3 90.3 87.7 RED ∗0.01% 86.2 90.3 91.3 68.1 96.0 93.5 88.8 89.5 88.0 LoReFT ⋄0.01% 87.5 90.1 91.6 68.0 96.2 94.1 88.5 89.2 88.2
> RoAd 10.06% 89.2 91.0 91.7 66.1 96.3 94.4 91.0 89.7 88.7
> RoAd 1(fc1) 0.03% 88.7 91.5 91.9 68.1 96.1 94.5 90.2 89.6 88.8

As shown in Table 3, RoAds still perform the best across various PEFT methods for both LLaMA-7B and LLaMA-13B on average. The strong baseline to RoAd is a recent representation finetuning method, LoReFT [ 61 ], 80.2 vs. 79.2 and 83.3 vs. 83.0 for RoAd 1 for LLaMA-7B and LLaMA-13B, respectively. With a slightly increasing number of trainable parameters from RoAd 1 to RoAd 2 or RoAd 4, RoAd matches or outperforms LoReFT. The same story is also told for another two versions of LLaMA, i.e. LLaMA2 [53] and LLaMA3, in Table D.2. 

Arithmetic reasoning. To assess the arithmetic reasoning ability of LLMs, we evaluate the finetuned LLMs on the test sets of four tasks: AQuA [ 24 ], GSM8K [ 6 ], MAWPS [ 18 ] and SVAMP [ 41 ]. Similar to the commonsense reasoning tasks, we finetune a single LLM for all four arithmetic reasoning tasks. The training dataset is Math10K [ 15 ] which is constructed from the training sets of GSM8K, MAWPS, MAWPS-single and AQuA. The training recipe is similar to the one used for commonsense reasoning as shown in Table C.5. Please see Section §C.3 for more training details. Different from the results of NLU and commonsense reasoning tasks, RoAd doesn’t always perform the best on the arithmetic reasoning tasks, as shown in Figure 4. For the smaller-size LLM, LLaMA-7B, RoAd is significantly better than other PEFT methods with < 0.1% trainable parameters, but worse than LoRA and Adapter P with more than 10 × trainable parameters. However, for the larger-size LLM, LLaMA-13B, all RoAd variants are better than other PEFT methods, which shows its scalability and potentially implies even better results for larger LLMs. Table 5: Score on AlpacaEval2.0 with LLaMA2-7B.                    

> Method #Params. Finetuning Data Win Rate (%) LoRA 0.83% 10K cleaned Alpaca 61.55 LoReFT 0.03% 10K cleaned Alpaca 60.21 RoAd 10.02% 10K cleaned Alpaca 62.64
> LoReFT 0.03% UltraFeedback [7] 61.68 RoAd 10.02% UltraFeedback 62.60

Observed from the above-mentioned results, for enhanced performance on downstream tasks and if a marginal increase in the stor-age capacity for trained parameters is accept-able, RoAd 4 is the preferable option. Con-versely, if the objective is to investigate how the model adjusts in terms of angle and mag-nitude, RoAd 1 is recommended. Notably, all variants of RoAd incur the same computational overhead for batching. 7Table 3: Accuracy of LLaMA on eight commonsense reasoning tasks. Results of methods denoted by 

∗, ⋄ and ◦ are from [ 15 ], [ 61 ] and [ 27 ], respectively. Otherwise, average results from three random runs are reported. Refer to Table C.6 for the standard deviation. Refer to Table D.2 for LLaMA2&3. 

Model Method #Paras. BoolQ PIQA SIQA HellaS. WinoG. ARC-e ARC-c OBQA Avg. 

GPT3.5 ∗ - - 73.1 85.4 68.5 78.5 66.1 89.8 79.9 74.8 77.0 Adapter P∗ 3.54% 67.9 76.4 78.8 69.8 78.9 73.7 57.3 75.2 72.3 Adapter S∗ 0.99% 63.0 79.2 76.3 67.9 75.7 74.5 57.1 72.4 70.8 DoRA ◦ 0.84% 68.5 82.9 79.6 84.8 80.8 81.4 65.8 81.0 78.1 LoRA ∗ 0.83% 68.9 80.7 77.4 78.1 78.8 77.8 61.3 74.8 74.7 OFT 0.14% 69.0 82.0 78.5 90.9 78.9 83.0 68.2 76.4 78.4 7B Prefix ∗ 0.04% 64.3 76.8 73.9 42.1 72.1 72.9 54.0 60.6 64.6 LoReFT ⋄ 0.03% 69.3 84.4 80.3 93.1 84.2 83.2 68.2 78.9 80.2 

(IA) 3 0.02% 67.8 81.7 78.1 89.9 81.1 80.5 65.4 77.8 77.8 

RoAd 4 0.08% 70.6 83.2 79.0 92.3 81.8 84.2 70.6 80.0 80.2 RoAd 2 0.04% 70.3 82.6 79.2 92.0 81.8 84.8 68.8 82.2 80.2 RoAd 1 0.02% 70.4 81.9 79.0 91.4 80.3 84.0 68.7 77.8 79.2 Adapter P∗ 2.89% 72.5 84.9 79.8 92.1 84.7 84.2 71.2 82.4 81.5 Adapter S∗ 0.80% 71.8 83.0 79.2 88.1 82.4 82.5 67.3 81.8 79.5 DoRA ◦ 0.68% 72.4 84.9 81.5 92.4 84.2 84.2 69.6 82.8 81.5 LoRA ∗ 0.67% 72.1 83.5 80.5 90.5 83.7 82.8 68.3 82.4 80.5 13B Prefix ∗ 0.03% 65.3 75.4 72.1 55.2 68.6 79.5 62.9 68.0 68.4 LoReFT ⋄ 0.03% 72.1 86.3 81.8 95.1 87.2 86.2 73.7 84.2 83.3 

RoAd 4 0.07% 73.2 85.5 82.4 94.5 86.3 86.8 74.6 86.0 83.7 

RoAd 2 0.03% 73.3 86.4 82.0 94.4 86.1 87.4 74.1 87.0 83.8 RoAd 1 0.02% 72.2 85.1 81.2 94.1 84.4 86.6 73.7 86.6 83.0 

Table 4: Accuracy of LLaMA on four arithmetic reasoning tasks. Results of methods denoted by ∗

and ⋄ are from [15] and [61], respectively. Refer to Table C.7 for the standard deviation. 

Model Method #Params. AQuA GSM8K MAWPS SVAMP Avg. 

Adapter P∗ 3.54% 18.1 35.3 82.4 49.6 46.4 Adapter S∗ 0.99% 15.0 33.3 77.7 52.3 44.6 LoRA ∗ 0.83% 18.9 37.5 79.0 52.1 46.9 

Prefix ∗ 0.04% 14.2 24.4 63.4 38.1 35.0 7B LoReFT ⋄ 0.03% 21.4 26.0 76.2 46.8 42.6 (IA) 3 0.02% 19.7 28.8 76.9 48.5 43.5 

RoAd 4 0.08% 24.8 27.4 81.5 49.4 45.8 

RoAd 2 0.04% 26.8 29.9 78.6 49.3 46.2 

RoAd 1 0.02% 26.4 26.2 76.5 46.7 44.0 Adapter P∗ 2.89% 20.5 43.3 81.1 55.7 50.2 Adapter S∗ 0.80% 22.0 44.0 78.6 50.8 48.9 LoRA ∗ 0.67% 18.5 47.5 83.6 54.6 51.1 Prefix ∗ 0.03% 15.7 31.1 66.8 41.4 38.8 13B LoReFT ⋄ 0.03% 23.6 38.1 82.4 54.2 49.6 

RoAd 4 0.07% 25.2 39.8 84.5 59.5 52.3 RoAd 2 0.03% 26.0 40.6 84.0 58.3 52.2 

RoAd 1 0.02% 24.8 40.7 84.9 57.3 51.9 

Instruction-following ability. We further benchmark RoAd using AlpacaEval2.0 [ 9 ]. We finetune LLaMA2-7B with two instruction-tuning datasets and evaluate the model using AlpacaEval2.0. This evaluation employs GPT-4 [ 39 ] to assess the responses generated by the finetuned model against those produced by Text-davinci-003. We don’t choose GPT-4 as the reference model, because GPT-4 is too powerful than LLaMA2-7B. The proof-of-concept experiment with LoRA shows the win-rate < 5%. As shown in Table 5, RoAd 1 demonstrates superior performance compared to all baselines, while utilizing the least number of trainable parameters. 

Multimodal ability. Lastly, we apply RoAd to the LLM backbone of LLaVA [ 26 ]. Liu et al. [26] requires 4.61% trainable parameters for LoRA on this task, while most tasks with LoRA in our paper need < 1%, showing that this task is knowledge-intensive. Therefore, we need to scale RoAd’s trainable parameters. For this purpose, we combine it with LoRA due to the limited number of θi and 

αi in R. The combination is represented as z = ( RW 0⊤ + ( BA )⊤)x, where A and B are from LoRA. We adjust the LoRA rank to vary the number of trainable parameters. We combine RoAd 1

81 2 4 8 16         

> LoRA rank
> 25
> 30
> 35
> 40
> 45
> 50
> Tokens/second  merged
> unmerged
> 256 1024 2048 4096
> #generated tokens
> 60
> 80
> 100
> 120
> 140
> 160
> RoAd
> LoRA
> 124816
> Batch size / #requests
> 50
> 100
> 150
> 200
> 250
> RoAd
> LoRA
> LLaMA-7B LLaMA-13B

Figure 4: Comparison of throughput between LoRA and RoAd. Left : The influence of weight merging for LoRA. Middle : The influence of the number of generated tokens. Right : The influence of the number of heterogeneous requests in a batch. with LoRA, but not RoAd 2 or RoAd 4, as their primary design purpose is to increase the number of trainable parameters. Table 6: Visual instruction tuning results on LLaVA1.5-7B.                          

> Method #Params. GQA SQA VQAT POPE Avg.
> LoRA 4.61% 62.4 68.5 56.9 86.0 68.5
> RoAd 40.08% 60.0 66.9 53.3 85.5 66.4 RoAd 1+ LoRA 1.19% 62.5 68.2 57.4 85.8 68.5

As shown in Table 6, with only 0.08% trainable parameters, RoAd 4

already achieves 96.9% of the accu-racy of LoRA with 4.61% trainable parameters. By combining RoAd 1

with LoRA, we achieve the same per-formance as LoRA with only 1/4 of its trainable parameters. This demonstrates RoAd’s excellent scalability when combined with LoRA. 

4.2 Efficiency results for batching 

We commence by highlighting the significance of weight merging for PEFT. Among the approaches discussed in Section §4.1, only LoRA [ 14 ], DoRA [ 27 ], BOFT [ 30 ], OFT [ 44 ], BitFit [ 62 ], (IA) 3 [25 ], and our proposed RoAd enable the integration of trainable parameters with pretrained parameters without incurring additional inference overhead. As an illustration, we consider LoRA both with and without weight merging to underscore this process’s importance. Notably, the implementation of LoRA with merged weights effectively reverts to the original LLM. To assess throughput, we configure the system with a batch size of 1, generate 2048 tokens, and apply the LoRA modules across all linear layers. Figure 4 (Left) clearly illustrates that the unmerged LoRA exhibits a significantly smaller throughput compared to the merged LoRA. Additionally, it is evident that the throughput of the unmerged LoRA demonstrates only a weak correlation with the rank size, primarily because the additional overhead is largely attributed to communication instead of computation. Furthermore, to evaluate the throughput of batching, we establish a default batch size of 8, generate 2048 tokens, and set the LoRA rank to 8. Each request within the batch is heterogeneous, necessitating eight distinct sets of trainable parameters by default. We only compare to LoRA here, because other baselines have either a weaker performance on downstream tasks (BOFT, OFT, BitFit and (IA) 3) or a smaller throughput than LoRA for batching (DoRA). As shown in Figure 4 (Middle and Right), RoAd significantly outperforms LoRA with variations in either the number of generated tokens or the number of heterogeneous requests. With an increasing number of distinct requests, the gap between LoRA and RoAd becomes even larger, which shows RoAd’s unique advantage in efficient serving 

4.3 Qualitative results for composability 

In our investigation of RoAd’s ability to handle compositional tasks, we primarily engage in multilin-gual experiments similar to those conducted by Wu et al. [61] . We use two training datasets: a new version of HellaSwag [ 63 ]7, which comprises 1K samples with prompts in English and completions in German, and a 1K-sample subset of the Ultrafeedback [ 7 ] dataset, which focuses on instruction following tasks in English. Contrary to the above experiments that adapt the outputs of the linear layer, here we instead adopt the representations from the 16 th block of LLaMA-7B, treating RoAd as a DII method. Specifically, we only adapt/intervene the representation of the final token in the 

> 7

https://huggingface.co/datasets/LeoLM/HellaSwag_de 

9LoReFT response for prompt: Do you know the conference NeurIPS?  

> German subspace : geht auf dem Bildschirm vorbei und zeigt die Konferenz. (Translation: passes by on the screen and shows the conference.)

× 

> Instruction following subspace : NeurIPS (Neural Information Processing Systems) is an annual conference on machine learning and computational neuroscience ...

✓ 

> Combined subspaces : Es gibt eine Konferenz namens NeurIPS, die in der Vergangenheit stattgefunden hat. (Translation: There is a conference called NeurIPS that took place in the past.)

× 

> RoAd response for prompt: Do you know the conference NeurIPS?
> German subspace : Ich weiß nicht, ob ich es kenne. (Translation: I don’t know if I know it.)

× 

> Instruction following subspace : Yes, I do! It’s a great conference for machine learning researchers and practitioners to share their work with each other.

✓ 

> Combined subspaces : NeurIPS ist ein internationales Forschungskongress, der sich mit dem Thema künstliche Intelligenz beschäftigt. (Translation: NeurIPS is an international research congress that deals with the topic of artificial intelligence.)

✓

Figure 5: Qualitative comparison between RoAd and LoReFT for their composability. The prompt for different subspaces is always in English. Refer to Figure D.1, D.2 and D.3 for more examples. prompt using RoAd 1. We train the upper half of R, i.e. {Ri}d2/4 

> i=1

, to handle the German completions in HellaSwag, and another half to complete the English sentences in Ultrafeedback. Both tasks are simultaneously trained but utilize distinct subspaces of R. We train the model over five epochs with a learning rate of 5e − 3 and a batch size of 8. 8

As in Figure 5, both LoReFT and RoAd are unable to perform completions with the German subspace. This limitation is anticipated due to two primary reasons: (1) LLaMA-7B predominantly relies on pretraining from English datasets, and doesn’t have a cross-lingual answering ability without explicitly prompting. (2) The HellaSwag dataset is relatively small, containing only 1K samples with limited comprehensive coverage. Despite these constraints, the German subspace effectively prompts the model to produce sentences in German. Additionally, both methods achieve accurate completions in the other half of the subspaces, attributed to LLaMA-7B’s extensive knowledge base in English. When these two subspaces are combined, RoAd successfully leverages their strengths, facilitating accurate sentence completions in German, while LoReFT doesn’t catch the purpose of the prompt. We offer more examples, including negative examples, in Figure D.1, D.2 and D.3. 

## 5 Conclusion 

Initially, our research examines how finetuning modifies the representation of pretrained LLMs, finding that angular adjustments are more significant than changes in magnitude scale. Leveraging this insight, we propose a PEFT method, RoAd, which primarily utilizes a 2D rotational adjustment to the representation. Despite its simplicity, RoAd exhibits several distinct advantages: (1) It is exceptionally efficient in terms of parameters, consistently delivering superior performance on downstream tasks with the fewest trainable parameters compared to other PEFT methods; (2) RoAd efficiently supports batch processing, achieving twice the throughput of LoRA; (3) When incorporated within an intervention framework, RoAd demonstrates remarkable composability. Due to page limit, we discuss the limitations and broader impacts in Section §A and §B, respectively. 

## Acknowledgements 

We thank eBay Inc. for the computation support. This research was funded in part by the Netherlands Organization for Scientific Research (NWO) under project number VI.C.192.080.  

> 8The experiment is based on this notebook https://github.com/stanfordnlp/pyreft/blob/main/ examples/composition/compreft.ipynb .

10 References 

[1] A. Abdelfattah, A. Haidar, S. Tomov, and J. J. Dongarra. Performance, design, and auto-tuning of batched GEMM for gpus. In J. M. Kunkel, P. Balaji, and J. J. Dongarra, editors, 

High Performance Computing - 31st International Conference, ISC High Performance 2016, Frankfurt, Germany, June 19-23, 2016, Proceedings , volume 9697 of Lecture Notes in Com-puter Science , pages 21–38. Springer, 2016. doi: 10.1007/978-3-319-41321-1\_2. URL 

https://doi.org/10.1007/978-3-319-41321-1_2 .[2] D. Biderman, J. G. Ortiz, J. Portes, M. Paul, P. Greengard, C. Jennings, D. King, S. Havens, V. Chiley, J. Frankle, C. Blakeney, and J. P. Cunningham. Lora learns less and forgets less, 2024. [3] Y. Bisk, R. Zellers, R. L. Bras, J. Gao, and Y. Choi. PIQA: reasoning about physical common-sense in natural language. In The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The Thirty-Second Innovative Applications of Artificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020 , pages 7432–7439. AAAI Press, 2020. doi: 10.1609/AAAI.V34I05.6239. URL https://doi.org/10.1609/aaai.v34i05.6239 .[4] C. Clark, K. Lee, M. Chang, T. Kwiatkowski, M. Collins, and K. Toutanova. Boolq: Ex-ploring the surprising difficulty of natural yes/no questions. In J. Burstein, C. Doran, and T. Solorio, editors, Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and Short Papers) , pages 2924– 2936. Association for Computational Linguistics, 2019. doi: 10.18653/V1/N19-1300. URL 

https://doi.org/10.18653/v1/n19-1300 .[5] P. Clark, I. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, and O. Tafjord. Think you have solved question answering? try arc, the AI2 reasoning challenge. CoRR , abs/1803.05457, 2018. URL http://arxiv.org/abs/1803.05457 .[6] K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, C. Hesse, and J. Schulman. Training verifiers to solve math word problems. CoRR , abs/2110.14168, 2021. URL https://arxiv.org/abs/2110.14168 .[7] G. Cui, L. Yuan, N. Ding, G. Yao, W. Zhu, Y. Ni, G. Xie, Z. Liu, and M. Sun. Ultrafeedback: Boosting language models with high-quality feedback. CoRR , abs/2310.01377, 2023. doi: 10.48550/ARXIV.2310.01377. URL https://doi.org/10.48550/arXiv.2310.01377 .[8] J. Devlin, M. Chang, K. Lee, and K. Toutanova. BERT: pre-training of deep bidirectional transformers for language understanding. In J. Burstein, C. Doran, and T. Solorio, editors, 

Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and Short Papers) , pages 4171–4186. Association for Computational Linguistics, 2019. doi: 10.18653/V1/N19-1423. URL https://doi.org/ 10.18653/v1/n19-1423 .[9] Y. Dubois, X. Li, R. Taori, T. Zhang, I. Gulrajani, J. Ba, C. Guestrin, P. Liang, and T. B. Hashimoto. Alpacafarm: A simulation framework for methods that learn from human feedback, 2023. [10] N. Elhage, T. Hume, C. Olsson, N. Schiefer, T. Henighan, S. Kravec, Z. Hatfield-Dodds, R. Lasenby, D. Drain, C. Chen, R. Grosse, S. McCandlish, J. Kaplan, D. Amodei, M. Wattenberg, and C. Olah. Toy models of superposition. CoRR , abs/2209.10652, 2022. doi: 10.48550/ ARXIV.2209.10652. URL https://doi.org/10.48550/arXiv.2209.10652 .[11] A. Geiger, Z. Wu, C. Potts, T. Icard, and N. D. Goodman. Finding alignments between inter-pretable causal variables and distributed neural representations. In F. Locatello and V. Didelez, editors, Causal Learning and Reasoning, 1-3 April 2024, Los Angeles, California, USA , vol-ume 236 of Proceedings of Machine Learning Research , pages 160–187. PMLR, 2024. URL 

https://proceedings.mlr.press/v236/geiger24a.html .[12] J. He, C. Zhou, X. Ma, T. Berg-Kirkpatrick, and G. Neubig. Towards a unified view of parameter-efficient transfer learning. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022 . OpenReview.net, 2022. URL 

https://openreview.net/forum?id=0RDcd5Axok .11 [13] N. Houlsby, A. Giurgiu, S. Jastrzebski, B. Morrone, Q. de Laroussilhe, A. Gesmundo, M. At-tariyan, and S. Gelly. Parameter-efficient transfer learning for NLP. In K. Chaudhuri and R. Salakhutdinov, editors, Proceedings of the 36th International Conference on Ma-chine Learning, ICML 2019, 9-15 June 2019, Long Beach, California, USA , volume 97 of Proceedings of Machine Learning Research , pages 2790–2799. PMLR, 2019. URL 

http://proceedings.mlr.press/v97/houlsby19a.html .[14] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen. Lora: Low-rank adaptation of large language models. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022 . OpenReview.net, 2022. URL https://openreview.net/forum?id=nZeVKeeFYf9 .[15] Z. Hu, L. Wang, Y. Lan, W. Xu, E. Lim, L. Bing, X. Xu, S. Poria, and R. K. Lee. Llm-adapters: An adapter family for parameter-efficient fine-tuning of large language models. In H. Bouamor, J. Pino, and K. Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023 ,pages 5254–5276. Association for Computational Linguistics, 2023. doi: 10.18653/V1/2023. EMNLP-MAIN.319. URL https://doi.org/10.18653/v1/2023.emnlp-main.319 .[16] C. Huang, Q. Liu, B. Y. Lin, T. Pang, C. Du, and M. Lin. Lorahub: Efficient cross-task generalization via dynamic lora composition. CoRR , abs/2307.13269, 2023. doi: 10.48550/ ARXIV.2307.13269. URL https://doi.org/10.48550/arXiv.2307.13269 .[17] A. Q. Jiang, A. Sablayrolles, A. Mensch, C. Bamford, D. S. Chaplot, D. de Las Casas, F. Bres-sand, G. Lengyel, G. Lample, L. Saulnier, L. R. Lavaud, M. Lachaux, P. Stock, T. L. Scao, T. Lavril, T. Wang, T. Lacroix, and W. E. Sayed. Mistral 7b. CoRR , abs/2310.06825, 2023. doi: 10.48550/ARXIV.2310.06825. URL https://doi.org/10.48550/arXiv.2310.06825 .[18] R. Koncel-Kedziorski, S. Roy, A. Amini, N. Kushman, and H. Hajishirzi. MAWPS: A math word problem repository. In K. Knight, A. Nenkova, and O. Rambow, editors, NAACL HLT 2016, The 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, San Diego California, USA, June 12-17, 2016 ,pages 1152–1157. The Association for Computational Linguistics, 2016. doi: 10.18653/V1/ N16-1136. URL https://doi.org/10.18653/v1/n16-1136 .[19] B. Lester, R. Al-Rfou, and N. Constant. The power of scale for parameter-efficient prompt tuning. In M. Moens, X. Huang, L. Specia, and S. W. Yih, editors, Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 7-11 November, 2021 , pages 3045–3059. Association for Computational Linguistics, 2021. doi: 10.18653/V1/2021.EMNLP-MAIN.243. URL 

https://doi.org/10.18653/v1/2021.emnlp-main.243 .[20] M. Li, S. Gururangan, T. Dettmers, M. Lewis, T. Althoff, N. A. Smith, and L. Zettlemoyer. Branch-train-merge: Embarrassingly parallel training of expert language models. CoRR ,abs/2208.03306, 2022. doi: 10.48550/ARXIV.2208.03306. URL https://doi.org/10. 48550/arXiv.2208.03306 .[21] X. L. Li and P. Liang. Prefix-tuning: Optimizing continuous prompts for generation. In C. Zong, F. Xia, W. Li, and R. Navigli, editors, Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, ACL/IJCNLP 2021, (Volume 1: Long Papers), Virtual Event, August 1-6, 2021 ,pages 4582–4597. Association for Computational Linguistics, 2021. doi: 10.18653/V1/2021. ACL-LONG.353. URL https://doi.org/10.18653/v1/2021.acl-long.353 .[22] B. Liao, Y. Meng, and C. Monz. Parameter-efficient fine-tuning without introducing new latency. In A. Rogers, J. L. Boyd-Graber, and N. Okazaki, editors, Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023 , pages 4242–4260. Association for Computational Linguistics, 2023. doi: 10.18653/V1/2023.ACL-LONG.233. URL https://doi.org/10.18653/v1/ 2023.acl-long.233 .[23] B. Liao, S. Tan, and C. Monz. Make pre-trained model reversible: From parameter to memory efficient fine-tuning. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 

12 10 - 16, 2023 , 2023. URL http://papers.nips.cc/paper_files/paper/2023/hash/ 3151e460c41ba67dc55412861184ef35-Abstract-Conference.html .[24] W. Ling, D. Yogatama, C. Dyer, and P. Blunsom. Program induction by rationale generation: Learning to solve and explain algebraic word problems. In R. Barzilay and M. Kan, editors, 

Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics, ACL 2017, Vancouver, Canada, July 30 - August 4, Volume 1: Long Papers , pages 158– 167. Association for Computational Linguistics, 2017. doi: 10.18653/V1/P17-1015. URL 

https://doi.org/10.18653/v1/P17-1015 .[25] H. Liu, D. Tam, M. Muqeeth, J. Mohta, T. Huang, M. Bansal, and C. Raffel. Few-shot parameter-efficient fine-tuning is better and cheaper than in-context learning. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neu-ral Information Processing Systems 35: Annual Conference on Neural Information Pro-cessing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - Decem-ber 9, 2022 , 2022. URL http://papers.nips.cc/paper_files/paper/2022/hash/ 0cde695b83bd186c1fd456302888454c-Abstract-Conference.html .[26] H. Liu, C. Li, Q. Wu, and Y. J. Lee. Visual instruction tuning. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Informa-tion Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023 , 2023. URL http://papers.nips.cc/paper_files/paper/2023/hash/ 6dcf277ea32ce3288914faf369fe6de0-Abstract-Conference.html .[27] S. Liu, C. Wang, H. Yin, P. Molchanov, Y. F. Wang, K. Cheng, and M. Chen. Dora: Weight-decomposed low-rank adaptation. CoRR , abs/2402.09353, 2024. doi: 10.48550/ARXIV.2402. 09353. URL https://doi.org/10.48550/arXiv.2402.09353 .[28] W. Liu, R. Lin, Z. Liu, L. Liu, Z. Yu, B. Dai, and L. Song. Learning towards minimum hyperspherical energy. In S. Bengio, H. M. Wallach, H. Larochelle, K. Grauman, N. Cesa-Bianchi, and R. Garnett, editors, Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018, NeurIPS 2018, December 3-8, 2018, Montréal, Canada , pages 6225–6236, 2018. URL https://proceedings.neurips. cc/paper/2018/hash/177540c7bcb8db31697b601642eac8d4-Abstract.html .[29] W. Liu, R. Lin, Z. Liu, J. M. Rehg, L. Paull, L. Xiong, L. Song, and A. Weller. Orthogonal over-parameterized training. In IEEE Conference on Computer Vision and Pattern Recognition, CVPR 2021, virtual, June 19-25, 2021 , pages 7251–7260. Computer Vision Foundation / IEEE, 2021. doi: 10.1109/CVPR46437.2021.00717. URL https://openaccess.thecvf.com/content/CVPR2021/html/Liu_Orthogonal_ Over-Parameterized_Training_CVPR_2021_paper.html .[30] W. Liu, Z. Qiu, Y. Feng, Y. Xiu, Y. Xue, L. Yu, H. Feng, Z. Liu, J. Heo, S. Peng, Y. Wen, M. J. Black, A. Weller, and B. Schölkopf. Parameter-efficient orthogonal finetuning via butterfly factorization. CoRR , abs/2311.06243, 2023. doi: 10.48550/ARXIV.2311.06243. URL 

https://doi.org/10.48550/arXiv.2311.06243 .[31] Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer, and V. Stoyanov. Roberta: A robustly optimized BERT pretraining approach. CoRR , abs/1907.11692, 2019. URL http://arxiv.org/abs/1907.11692 .[32] I. Loshchilov and F. Hutter. Decoupled weight decay regularization. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019 .OpenReview.net, 2019. URL https://openreview.net/forum?id=Bkg6RiCqY7 .[33] R. K. Mahabadi, J. Henderson, and S. Ruder. Compacter: Efficient low-rank hypercom-plex adapter layers. In M. Ranzato, A. Beygelzimer, Y. N. Dauphin, P. Liang, and J. W. Vaughan, editors, Advances in Neural Information Processing Systems 34: Annual Conference on Neural Information Processing Systems 2021, NeurIPS 2021, December 6-14, 2021, virtual ,pages 1022–1035, 2021. URL https://proceedings.neurips.cc/paper/2021/hash/ 081be9fdff07f3bc808f935906ef70c0-Abstract.html .[34] S. Mangrulkar, S. Gugger, L. Debut, Y. Belkada, S. Paul, and B. Bossan. Peft: State-of-the-art parameter-efficient fine-tuning methods. https://github.com/huggingface/peft , 2022. 13 [35] J. L. McClelland, D. E. Rumelhart, P. R. Group, et al. Parallel distributed processing, volume 2: Explorations in the microstructure of cognition: Psychological and biological models , volume 2. MIT press, 1987. [36] T. Mihaylov, P. Clark, T. Khot, and A. Sabharwal. Can a suit of armor conduct electricity? A new dataset for open book question answering. In E. Riloff, D. Chiang, J. Hockenmaier, and J. Tsujii, editors, Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, Brussels, Belgium, October 31 - November 4, 2018 , pages 2381–2391. Association for Computational Linguistics, 2018. doi: 10.18653/V1/D18-1260. URL https: //doi.org/10.18653/v1/d18-1260 .[37] T. Mikolov, W. Yih, and G. Zweig. Linguistic regularities in continuous space word rep-resentations. In L. Vanderwende, H. D. III, and K. Kirchhoff, editors, Human Language Technologies: Conference of the North American Chapter of the Association of Compu-tational Linguistics, Proceedings, June 9-14, 2013, Westin Peachtree Plaza Hotel, Atlanta, Georgia, USA , pages 746–751. The Association for Computational Linguistics, 2013. URL 

https://aclanthology.org/N13-1090/ .[38] N. Nanda, A. Lee, and M. Wattenberg. Emergent linear representations in world models of self-supervised sequence models. In Y. Belinkov, S. Hao, J. Jumelet, N. Kim, A. McCarthy, and H. Mohebbi, editors, Proceedings of the 6th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP, BlackboxNLP@EMNLP 2023, Singapore, December 7, 2023 , pages 16–30. Association for Computational Linguistics, 2023. doi: 10.18653/V1/2023. BLACKBOXNLP-1.2. URL https://doi.org/10.18653/v1/2023.blackboxnlp-1.2 .[39] OpenAI. GPT-4 technical report. CoRR , abs/2303.08774, 2023. doi: 10.48550/ARXIV.2303. 08774. URL https://doi.org/10.48550/arXiv.2303.08774 .[40] K. Park, Y. J. Choe, and V. Veitch. The linear representation hypothesis and the geometry of large language models. CoRR , abs/2311.03658, 2023. doi: 10.48550/ARXIV.2311.03658. URL 

https://doi.org/10.48550/arXiv.2311.03658 .[41] A. Patel, S. Bhattamishra, and N. Goyal. Are NLP models really able to solve simple math word problems? In K. Toutanova, A. Rumshisky, L. Zettlemoyer, D. Hakkani-Tür, I. Belt-agy, S. Bethard, R. Cotterell, T. Chakraborty, and Y. Zhou, editors, Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2021, Online, June 6-11, 2021 , pages 2080–2094. Association for Computational Linguistics, 2021. doi: 10.18653/V1/2021.NAACL-MAIN.168. URL https://doi.org/10.18653/v1/2021.naacl-main.168 .[42] J. Pfeiffer, A. Kamath, A. Rücklé, K. Cho, and I. Gurevych. Adapterfusion: Non-destructive task composition for transfer learning. In P. Merlo, J. Tiedemann, and R. Tsarfaty, editors, 

Proceedings of the 16th Conference of the European Chapter of the Association for Compu-tational Linguistics: Main Volume, EACL 2021, Online, April 19 - 23, 2021 , pages 487–503. Association for Computational Linguistics, 2021. doi: 10.18653/V1/2021.EACL-MAIN.39. URL https://doi.org/10.18653/v1/2021.eacl-main.39 .[43] Y. Qin, X. Wang, Y. Su, Y. Lin, N. Ding, Z. Liu, J. Li, L. Hou, P. Li, M. Sun, and J. Zhou. Exploring low-dimensional intrinsic task subspace via prompt tuning. CoRR , abs/2110.07867, 2021. URL https://arxiv.org/abs/2110.07867 .[44] Z. Qiu, W. Liu, H. Feng, Y. Xue, Y. Feng, Z. Liu, D. Zhang, A. Weller, and B. Schölkopf. Controlling text-to-image diffusion by orthogonal finetuning. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Informa-tion Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023 , 2023. URL http://papers.nips.cc/paper_files/paper/2023/hash/ faacb7a4827b4d51e201666b93ab5fa7-Abstract-Conference.html .[45] C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. J. Mach. Learn. Res. , 21:140:1–140:67, 2020. URL http://jmlr.org/papers/v21/20-074.html .[46] D. E. Rumelhart, J. L. McClelland, P. R. Group, et al. Parallel distributed processing, volume 1: Explorations in the microstructure of cognition: Foundations . The MIT press, 1986. 14 [47] K. Sakaguchi, R. L. Bras, C. Bhagavatula, and Y. Choi. Winogrande: An adversarial winograd schema challenge at scale. In The Thirty-Fourth AAAI Conference on Artificial Intelligence, AAAI 2020, The Thirty-Second Innovative Applications of Artificial Intelligence Conference, IAAI 2020, The Tenth AAAI Symposium on Educational Advances in Artificial Intelligence, EAAI 2020, New York, NY, USA, February 7-12, 2020 , pages 8732–8740. AAAI Press, 2020. doi: 10.1609/AAAI.V34I05.6399. URL https://doi.org/10.1609/aaai.v34i05.6399 .[48] M. Sap, H. Rashkin, D. Chen, R. L. Bras, and Y. Choi. Socialiqa: Commonsense reasoning about social interactions. CoRR , abs/1904.09728, 2019. URL http://arxiv.org/abs/1904. 09728 .[49] P. Smolensky. Neural and conceptual interpretation of pdp models. Parallel distributed processing: Explorations in the microstructure of cognition , 2:390–431, 1986. [50] R. Socher, A. Perelygin, J. Wu, J. Chuang, C. D. Manning, A. Y. Ng, and C. Potts. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, EMNLP 2013, 18-21 October 2013, Grand Hyatt Seattle, Seattle, Washington, USA, A meeting of SIGDAT, a Special Interest Group of the ACL , pages 1631–1642. ACL, 2013. URL https://aclanthology. org/D13-1170/ .[51] J. Su, M. H. M. Ahmed, Y. Lu, S. Pan, W. Bo, and Y. Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing , 568:127063, 2024. doi: 10.1016/J.NEUCOM. 2023.127063. URL https://doi.org/10.1016/j.neucom.2023.127063 .[52] H. Touvron, T. Lavril, G. Izacard, X. Martinet, M. Lachaux, T. Lacroix, B. Rozière, N. Goyal, E. Hambro, F. Azhar, A. Rodriguez, A. Joulin, E. Grave, and G. Lample. Llama: Open and efficient foundation language models. CoRR , abs/2302.13971, 2023. doi: 10.48550/ARXIV. 2302.13971. URL https://doi.org/10.48550/arXiv.2302.13971 .[53] H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, D. Bikel, L. Blecher, C. Canton-Ferrer, M. Chen, G. Cucurull, D. Es-iobu, J. Fernandes, J. Fu, W. Fu, B. Fuller, C. Gao, V. Goswami, N. Goyal, A. Hartshorn, S. Hosseini, R. Hou, H. Inan, M. Kardas, V. Kerkez, M. Khabsa, I. Kloumann, A. Ko-renev, P. S. Koura, M. Lachaux, T. Lavril, J. Lee, D. Liskovich, Y. Lu, Y. Mao, X. Martinet, T. Mihaylov, P. Mishra, I. Molybog, Y. Nie, A. Poulton, J. Reizenstein, R. Rungta, K. Sal-adi, A. Schelten, R. Silva, E. M. Smith, R. Subramanian, X. E. Tan, B. Tang, R. Taylor, A. Williams, J. X. Kuan, P. Xu, Z. Yan, I. Zarov, Y. Zhang, A. Fan, M. Kambadur, S. Narang, A. Rodriguez, R. Stojnic, S. Edunov, and T. Scialom. Llama 2: Open foundation and fine-tuned chat models. CoRR , abs/2307.09288, 2023. doi: 10.48550/ARXIV.2307.09288. URL 

https://doi.org/10.48550/arXiv.2307.09288 .[54] A. M. Turner, L. Thiergart, D. Udell, G. Leech, U. Mini, and M. MacDiarmid. Activation addition: Steering language models without optimization. CoRR , abs/2308.10248, 2023. doi: 10.48550/ARXIV.2308.10248. URL https://doi.org/10.48550/arXiv.2308.10248 .[55] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser, and I. Polo-sukhin. Attention is all you need. In I. Guyon, U. von Luxburg, S. Bengio, H. M. Wallach, R. Fer-gus, S. V. N. Vishwanathan, and R. Garnett, editors, Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA , pages 5998–6008, 2017. URL https://proceedings.neurips. cc/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html .[56] A. Wang, A. Singh, J. Michael, F. Hill, O. Levy, and S. R. Bowman. GLUE: A multi-task benchmark and analysis platform for natural language understanding. In 7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019 .OpenReview.net, 2019. URL https://openreview.net/forum?id=rJ4km2R5t7 .[57] Y. Wen and S. Chaudhuri. Batched low-rank adaptation of foundation models. CoRR ,abs/2312.05677, 2023. doi: 10.48550/ARXIV.2312.05677. URL https://doi.org/10. 48550/arXiv.2312.05677 .[58] Y. Wen and S. Chaudhuri. Batched low-rank adaptation of foundation models. CoRR ,abs/2312.05677, 2023. doi: 10.48550/ARXIV.2312.05677. URL https://doi.org/10. 48550/arXiv.2312.05677 .15 [59] T. Wolf, L. Debut, V. Sanh, J. Chaumond, C. Delangue, A. Moi, P. Cistac, T. Rault, R. Louf, M. Funtowicz, J. Davison, S. Shleifer, P. von Platen, C. Ma, Y. Jernite, J. Plu, C. Xu, T. L. Scao, S. Gugger, M. Drame, Q. Lhoest, and A. M. Rush. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations , pages 38–45, Online, Oct. 2020. Association for Computational Linguistics. URL https://www.aclweb.org/anthology/ 2020.emnlp-demos.6 .[60] M. Wu, W. Liu, X. Wang, T. Li, C. Lv, Z. Ling, J. Zhu, C. Zhang, X. Zheng, and X. Huang. Advancing parameter efficiency in fine-tuning via representation editing. CoRR , abs/2402.15179, 2024. doi: 10.48550/ARXIV.2402.15179. URL https://doi.org/10.48550/arXiv.2402. 15179 .[61] Z. Wu, A. Arora, Z. Wang, A. Geiger, D. Jurafsky, C. D. Manning, and C. Potts. Reft: Representation finetuning for language models. 2024. URL https://api.semanticscholar. org/CorpusID:268889731 .[62] E. B. Zaken, Y. Goldberg, and S. Ravfogel. Bitfit: Simple parameter-efficient fine-tuning for transformer-based masked language-models. In S. Muresan, P. Nakov, and A. Villavicencio, editors, Proceedings of the 60th Annual Meeting of the Association for Computational Lin-guistics (Volume 2: Short Papers), ACL 2022, Dublin, Ireland, May 22-27, 2022 , pages 1–9. Association for Computational Linguistics, 2022. doi: 10.18653/V1/2022.ACL-SHORT.1. URL 

https://doi.org/10.18653/v1/2022.acl-short.1 .[63] R. Zellers, A. Holtzman, Y. Bisk, A. Farhadi, and Y. Choi. Hellaswag: Can a machine really finish your sentence? In A. Korhonen, D. R. Traum, and L. Màrquez, editors, Proceedings of the 57th Conference of the Association for Computational Linguistics, ACL 2019, Florence, Italy, July 28- August 2, 2019, Volume 1: Long Papers , pages 4791–4800. Association for Computational Linguistics, 2019. doi: 10.18653/V1/P19-1472. URL https://doi.org/10. 18653/v1/p19-1472 .[64] J. Zhang, S. Chen, J. Liu, and J. He. Composing parameter-efficient modules with arithmetic operation. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, edi-tors, Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023 , 2023. URL http://papers.nips.cc/paper_files/paper/2023/hash/ 299a08ee712d4752c890938da99a77c6-Abstract-Conference.html .[65] Q. Zhang, M. Chen, A. Bukharin, P. He, Y. Cheng, W. Chen, and T. Zhao. Adaptive budget allocation for parameter-efficient fine-tuning. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023 . OpenReview.net, 2023. URL https://openreview.net/pdf?id=lq62uWRJjiY .[66] M. Zhong, Y. Shen, S. Wang, Y. Lu, Y. Jiao, S. Ouyang, D. Yu, J. Han, and W. Chen. Multi-lora composition for image generation. CoRR , abs/2402.16843, 2024. doi: 10.48550/ARXIV.2402. 16843. URL https://doi.org/10.48550/arXiv.2402.16843 .[67] A. Zou, L. Phan, S. Chen, J. Campbell, P. Guo, R. Ren, A. Pan, X. Yin, M. Mazeika, A. Dom-browski, S. Goel, N. Li, M. J. Byun, Z. Wang, A. Mallen, S. Basart, S. Koyejo, D. Song, M. Fredrikson, J. Z. Kolter, and D. Hendrycks. Representation engineering: A top-down approach to AI transparency. CoRR , abs/2310.01405, 2023. doi: 10.48550/ARXIV.2310.01405. URL https://doi.org/10.48550/arXiv.2310.01405 .16 A Limitations 

We recognize that a primary limitation pertains to the scalability of RoAd. Currently, it is not feasible to indefinitely increase the number of trainable parameters with RoAd. Nevertheless, our experiments demonstrate that RoAd 4 already exhibits commendable performance. To scale the trainable parameters, we can combine RoAd with other PEFT methods, such as LoRA, which enhances the scaling behavior of these PEFTs, i.e. achieving similar results with less trainable parameters. 

## B Broader impacts 

RoAd’s primary advantage is its efficiency in adapting LLMs to specific tasks with minimal trainable parameters. This efficiency not only reduces computational resource needs but also makes advanced AI technologies more accessible to organizations with limited resources, potentially democratizing AI capabilities across smaller enterprises and educational institutions. By reducing the number of trainable parameters and the computational load, RoAd likely decreases the energy consumption associated with training and deploying LLMs. This could contribute to lowering the carbon footprint of AI research and deployment, aligning with greater environmental sustainability efforts. The ability to process multiple heterogeneous requests efficiently means that applications can provide personalized, context-specific responses more quickly. This enhances the user experience in real-time applications, such as digital assistants, automated service, and interactive educational platforms. While RoAd improves interpretability in some aspects by integrating within frameworks like dis-tributed interchange intervention [ 11 ], the overall complexity of the methods might still pose chal-lenges in understanding and diagnosing the models’ decisions. This could affect efforts to make AI more transparent and accountable, especially in critical applications like healthcare and law. Increasing the accessibility of powerful AI models through PEFT also raises concerns about misuse. More entities can harness these capabilities, potentially including those with malicious intents, such as creating sophisticated disinformation campaigns or automating cyber attacks. 

## C Experimental details 

C.1 Natural language understanding (NLU) 

Table C.1: The data statistics and evaluation metrics of the GLUE benchmark. The valid and test sets are randomly split from the original development set. Following Wu et al. [60] , only the matched development set of MNLI is used. For runs with different seeds, the samples in the valid and test sets are also different.                                         

> Task RTE MRPC STS-B CoLA SST-2 QNLI QQP MNLI
> #Train 2.6K 3.7K 5.7K 8.5K 67K 105K 364K 393K
> #Valid 139 204 750 522 436 1K 1K 1K
> #Test 138 204 750 521 436 4.5K 39K 8K
> Metric Acc. Acc. Pearson Matthew Acc. Acc. Acc. Acc.

Test set split. Previous works [ 14 , 22 , 31 ] report the best results on the development sets of the GLUE tasks, i.e. using the same set for both validation and test, which might cause overfitting. Instead, we follow the setting of Mahabadi et al. [33] and Wu et al. [60] , splitting the whole development set into a validation set and a test set. The model with the best performance on the validation set is selected to perform on the test set. Specifically, for the task with a development set whose number of samples is larger than 2K, i.e. QNLI, QQP and MNLI, we randomly select 1K samples as the validation set and the rest as the test set. For the other tasks, we select half of the samples in the development set as the validation set and another half as the test set. Please refer to Table C.1 for more details. 

Hyperparameter tuning. We mainly follow the hyperparameter search space of Liao et al. [22] and list them in Table C.2. Notably, we almost upscale the learning rate by 10 for RoAd, because RoAd prefers a larger learning rate than other PEFT methods, which is also observed from Liu et al. [25] and Wen and Chaudhuri [57] where their adapters also apply multiplication instead of addition. The 17 Table C.2: Hyperparameter search space for GLUE. For tasks with a large number of training samples, we set the number of epochs as 10. Please refer to Table C.3 for the best task-specific settings.                 

> Hyperparameters RTE, MRPC, STS-B, CoLA SST-2, QNLI, QQP, MNLI
> Optimizer AdamW AdamW Weight decay 00LR {1e-3, 3e-3, 5e-3, 7e-3} {1e-3, 3e-3, 5e-3, 7e-3} LR scheduler Linear Linear Warmup ratio 0.1 0.1 Epochs {10, 20} 10 Batch size {16, 32} {16, 32}

Table C.3: Best hyperparameter settings for different GLUE tasks on RoBERTa. Notably, RoAd has a very consistent recipe for different tasks. The low-resource tasks (RTE, MRPC, STS-B, CoLA) and high-resource tasks (SST-2, QNLI, QQP, MNLI) show two obvious patterns for the hyperparameters. If you have enough computation resources, we suggest alternating the batch size of low-resource tasks (RTE, MRPC, STS-B, CoLA) in {16, 32} and the number of epochs in {10, 20}, since these tasks have a relatively larger variance.                                                            

> Model Hyperparameter RTE MRPC STS-B CoLA SST-2 QNLI QQP MNLI
> LR 3e-3 3e-3 3e-3 3e-3 1e-3 1e-3 1e-3 1e-3 base Epochs 20 20 20 20 10 10 10 10 Batch size 32 32 32 32 16 16 16 16 LR 3e-3 3e-3 1e-3 1e-3 1e-3 1e-3 1e-3 1e-3 large Epochs 20 20 20 20 10 10 10 10 Batch size 32 32 32 32 32 32 32 32

best hyperparameter settings for each task are listed in Table C.3. The training is conducted either in Float16 or BFloat16. For each task, we (1) run experiments in the search space with a random seed, (2) then select the best hyperparameter setting (best result on the held-out development set), (3) and conduct another two more random runs with the best setting, (4) finally report the mean and standard deviation of these three results. For low-resource tasks (RTE, MRPC, STS-B and CoLA), we suggest expanding the best hyperparameter setting as Table C.3 for better reproduction. We report the standard deviation of RoAd in Table C.4. 

Baseline reproduction. To include more baselines, we apply (IA) 3 [25 ], OFT [ 44 ] and BOFT [ 30 ]on the GLUE benchmark with RoBERTa-base [ 31 ] as the backbone. We use the same search space as RoAd in Table C.2 for (IA) 3 since both RoAd and (IA) 3 prefer a large learning rate. For OFT w=2 

[ 44 ] and BOFT m=2  

> w=2

[ 30 ], we use the best hyperparameter settings from Liu et al. [30] . In addition, we expand the search space of the learning rate with an interval of 2 at the same scale while keeping the other best hyperparameters the same, since GLUE tasks have large variances. For example, if the best learning rate from Liu et al. [30] is 5e-4, the learning rate search space is {3e-4, 5e-4, 7e-4}. If the best learning rate is 2e-4, the search space is {9e-5, 2e-4, 4e-4}. For OFT, we don’t share any parameters and use BOFT m=1  

> w=2

(= OFT w=2 ), because such a setting offers better results. 

C.2 Commonsense reasoning Datasets. Please refer to Hu et al. [15] for more details about the data statistics and task templates. 

Hyperparameters. From Table C.3, it becomes apparent that one of the advantages of RoAd is its uniform optimal hyperparameter configuration across various tasks. Furthermore, we believe that extensive tuning of hyperparameters for LLMs is impractical. Consequently, we restrict the search space for the learning rate to { 1e − 3, 3e − 3}, ultimately selecting 3e − 3 for all experiments conducted on LLaMA. Consistent with Table C.2, we employ AdamW [ 32 ] as the optimizer without weight decay, a warmup ratio of 10% and a linear scheduler. Following Wu et al. [61] , we fix the number of epochs at six and the batch size at 32. These hyperparameters are detailed in Table C.5. The maximum sequence length is set to 512. And the training is conducted either in BFloat16. We evaluate each checkpoint saved at every epoch and report the optimal result. The standard deviation 18 Table C.4: The standard deviation (subscript) of three random runs on the GLUE benchmark for RoAd. 

Model Method #Params. RTE MRPC STS-B CoLA SST-2 QNLI QQP MNLI Avg. 

base RoAd 1 0.07% 78.9 1.2 89.2 0.4 90.5 0.4 64.4 0.8 93.9 0.6 91.9 0.1 89.6 0.1 86.3 0.2 85.6 RoAd 1(fc1) 0.03% 79.1 2.1 90.2 1.1 90.2 0.2 60.9 1.2 94.6 0.7 91.6 0.2 88.7 0.0 85.4 0.1 85.1 large RoAd 1 0.06% 89.2 0.6 91.0 1.2 91.7 0.1 66.1 0.5 96.3 0.4 94.4 0.0 91.0 0.0 89.7 0.2 88.7 RoAd 1(fc1) 0.03% 88.7 1.2 91.5 1.2 91.9 0.2 68.1 1.1 96.1 0.6 94.5 0.1 90.2 0.1 89.6 0.1 88.8 

from three random runs is presented in Table C.6. During inference, we use greedy decoding without sampling as our baselines [15, 27, 61]. Table C.5: Hyperparameters for commonsense and arithmetic reasoning without extensive tuning. 

Hyperparameters Commonsense reasoning Arithmetic reasoning 

Optimizer AdamW AdamW Weight decay 0 0LR 3e-3 3e-3 LR scheduler Linear Linear Warmup ratio 0.1 0.1 Epochs 6 12 Batch size 32 32 

Table C.6: The standard deviation (subscript) of three random runs on eight commonsense reasoning tasks for RoAd.                                                                          

> Model Method #Params. BoolQ PIQA SIQA HellaS. WinoG. ARC-e ARC-c OBQA Avg.
> RoAd 40.08% 70.6 0.2 83.2 0.3 79.0 0.1 92.3 0.2 81.8 0.6 84.2 0.3 70.6 0.8 80.0 0.4 80.2 0.1
> LLaMA-7B RoAd 20.04% 70.3 0.4 82.6 0.4 79.2 0.4 92.0 0.1 81.8 0.7 84.8 0.3 68.8 0.3 82.2 1.0 80.2 0.0
> RoAd 10.02% 70.4 0.9 81.9 0.3 79.0 0.2 91.4 0.1 80.3 0.3 84.0 0.1 68.7 0.6 77.8 0.8 79.2 0.1
> RoAd 40.07% 73.2 0.5 85.5 0.5 82.4 0.2 94.5 0.1 86.3 0.3 86.8 0.3 74.6 0.3 86.0 0.2 83.7 0.0
> LLaMA-13B RoAd 20.03% 73.3 0.3 86.4 0.5 82.0 0.5 94.4 0.1 86.1 0.3 87.4 0.4 74.1 0.2 87.0 0.5 83.8 0.2
> RoAd 10.02% 72.2 0.3 85.1 0.0 81.2 0.2 94.1 0.0 84.4 0.5 86.6 0.4 73.7 0.2 86.6 1.0 83.0 0.2

Baseline reproduction. In Table 3, we replicate the results of two baselines, OFT [ 44 ] and (IA) 3 [25 ]. For OFT w=16 (=BOFT m=1 

w=16 ), we adopt the identical training configuration used for the mathematical question-answering task as described in Liu et al. [30] . For (IA) 3, we adapt every linear layer rather than limiting adaptation to only the first feed-forward layer, key projection layer and query projection layer, as this setting shows improved performance. Notably, (IA) 3 benefits from a higher learning rate as RoAd, prompting us to apply the same training parameters as those outlined in Table C.5. 

C.3 Arithmetic reasoning Datasets. Please refer to Hu et al. [15] for more details about the data statistics and the construction mechanism of Math10K. 

Hyperparameters. We apply almost the same training recipe as the one for commonsense reasoning, except that we set the number of epochs as 12 by following Wu et al. [61] . The detailed parameters are summarized in Table C.5. The maximum sequence length is set to 512. And the training is conducted either in BFloat16. We evaluate each checkpoint saved at every epoch and report the optimal result. The standard deviation from three random runs is presented in Table C.7. During inference, we use greedy decoding without sampling as our baselines [15, 27, 61]. 

Baseline reproduction. In Table 4, we replicate the results of (IA) 3 [ 25 ]. Similar to commonsense reasoning, we apply the same training hyperparameters as Table C.5 for (IA) 3.19 Table C.7: The standard deviation (subscript) of three random runs on four arithmetic reasoning tasks for RoAd. 

Model Method #Params. AQuA GSM8K MAWPS SVAMP Avg. 

RoAd 4 0.08% 24.8 1.0 27.4 0.9 81.5 0.9 49.4 0.3 45.8 0.5 

LLaMA-7B RoAd 2 0.04% 26.8 2.8 29.9 0.6 78.6 1.2 49.3 0.6 46.2 0.6 

RoAd 1 0.02% 26.4 1.7 26.2 0.2 76.5 1.6 46.7 1.0 44.0 0.2 

RoAd 4 0.07% 25.2 3.1 39.8 0.5 84.5 1.5 59.5 0.7 52.3 0.3 

LLaMA-13B RoAd 2 0.03% 26.0 0.9 40.6 0.5 84.0 1.2 58.3 0.8 52.2 0.4 

RoAd 1 0.02% 24.8 1.0 40.7 0.9 84.9 0.9 57.3 0.2 51.9 0.2 

Table D.1: Finetuning details of RoAds, OFT and BOFT on LLaMA-7B. The training setting here is: batch size = 1, maximum sequence length = 512, number of iterations = 100, 1 A100 80GB GPU. 

Method #Params. Peak GPU memory (GB) Training time (s) OFT n=2048 0.09% 40 1249 OFT n=256 0.6% 37 191 BOFT m=2  

> w=8

0.3% OOM -RoAd 1 0.02% 23 25 RoAd 2 0.04% 23 23 RoAd 4 0.08% 23 24 

## D More results 

D.1 Compare to OFT. 

Table D.1 presents the finetuning specifics for RoAds, OFT [ 44 ], and BOFT [ 30 ]. In OFT, a critical hyperparameter is defined as n = d1 

> w

, meaning the number of blocks in R. Thus, configurations such as OFT n=2048 and OFT n=256 correspond approximately to OFT w=2 and OFT w=16 , respectively. Increasing n, or equivalently reducing w, leads to a higher count of blocks. While a smaller w may reduce the number of trainable parameters, it necessitates more frequent computations of matrix inversion, consequently elevating both GPU memory usage and training time. Moreover, while BOFT utilizes fewer trainable parameters than OFT and achieves comparable or superior outcomes, it demands significantly more GPU memory. This increase is attributable to the butterfly factorization, which requires extensive caching of intermediate activations. RoAd can be viewed as a specific implementation of OFT w=2 , but it consumes considerably less GPU memory and shortens training time. This efficiency stems from the use of inherently orthogonal 2D rotation matrices in RoAd, which obviate the need for matrix inversion calculations. 

D.2 Commonsense reasoning on LLaMA2 and LLaMA3 

We also conduct experiments on LLaMA2-7B [ 53 ] and LLaMA3-8B in Table D.2. RoAds still outperform all baselines with the least number of trainable parameters. 

D.3 More examples for composability 

In Figure D.1, D.2 and D.3, we show more examples of composability. Overall, RoAd demonstrates a very good ability in composition, taking advantage of both subspaces. 20 Table D.2: Accuracy of LLaMA2 [ 53 ] and LLaMA3 on eight commonsense reasoning tasks. Results of methods denoted by ∗ are from Liu et al. [27]. 

Model Method #Params. BoolQ PIQA SIQA HellaS. WinoG. ARC-e ARC-c OBQA Avg. 

DoRA ∗ 0.84% 71.8 83.7 76.0 89.1 82.6 83.7 68.2 82.4 79.7 LoRA ∗ 0.83% 69.8 79.9 79.5 83.6 82.6 79.8 64.7 81.0 77.6 LLaMA2-7B DoRA ∗ 0.43% 72.0 83.1 79.9 89.1 83.0 84.5 71.0 81.2 80.5 

RoAd 4 0.08% 72.6 83.8 80.0 93.3 83.0 87.1 73.7 84.8 82.3 RoAd 2 0.04% 73.0 83.9 80.2 93.2 83.0 86.5 74.4 83.0 82.2 

RoAd 1 0.02% 71.7 83.0 80.1 93.0 81.2 86.0 72.3 82.2 81.2 DoRA ∗ 0.71% 74.6 89.3 79.9 95.5 85.6 90.5 80.4 85.8 85.2 LoRA ∗ 0.70% 70.8 85.2 79.9 91.7 84.3 84.2 71.2 79.0 80.8 LLaMA3-8B DoRA ∗ 0.35% 74.5 88.8 80.3 95.5 84.7 90.1 79.1 87.2 85.0 

RoAd 4 0.07% 74.4 89.8 81.1 96.2 87.8 92.9 83.0 86.8 86.5 RoAd 2 0.03% 74.6 89.8 81.6 96.0 86.9 92.8 82.1 86.8 86.3 

RoAd 1 0.02% 73.5 89.0 81.4 96.0 87.6 92.9 82.4 87.2 86.3 

21 0.00                        

> 0.25
> 0.50
> 0.75
> 1.00
> 1.25
> Full Finetuning, Layer 1 LoRA, Layer 1 Full Finetuning, Layer 2 LoRA, Layer 2
> 0.00
> 0.25
> 0.50
> 0.75
> 1.00
> 1.25
> Full Finetuning, Layer 3 LoRA, Layer 3 Full Finetuning, Layer 4 LoRA, Layer 4
> 0.00
> 0.25
> 0.50
> 0.75
> 1.00
> 1.25
> Full Finetuning, Layer 5 LoRA, Layer 5 Full Finetuning, Layer 6 LoRA, Layer 6
> 0.00
> 0.25
> 0.50
> 0.75
> 1.00
> 1.25
> Full Finetuning, Layer 7 LoRA, Layer 7 Full Finetuning, Layer 8 LoRA, Layer 8
> 0.00
> 0.25
> 0.50
> 0.75
> 1.00
> 1.25
> Full Finetuning, Layer 9 LoRA, Layer 9 Full Finetuning, Layer 10 LoRA, Layer 10
> 0.0 0.5 1.0
> 0.00
> 0.25
> 0.50
> 0.75
> 1.00
> 1.25
> Full Finetuning, Layer 11
> 0.0 0.5 1.0
> LoRA, Layer 11
> 0.0 0.5 1.0
> Full Finetuning, Layer 12
> 0.0 0.5 1.0
> LoRA, Layer 12
> D
> M

Figure B.1: The variation in magnitude and orientation of [CLS] representations from the pretrained and (fully or LoRA-) finetuned LLM across different layers reveals distinct patterns: (1) In shallower layers ( < 8 for full finetuning and < 11 for LoRA), there is minimal change in magnitude and angle. This minimal alteration occurs because these layers learn general knowledge that does not require significant modification for specific downstream tasks. (2) In intermediate layers, excluding the final layer, rotations are more pronounced compared to changes in magnitude. (3) In the last layer, both magnitude and angle undergo substantial shifts, reflecting the stark differences between the objectives of pretraining and finetuning. The representation in this layer is extensively modified to align with the demands of the downstream task. Overall, compared to LoRA, full finetuning exhibits greater alterations in both magnitude and angle, explaining that LoRA learns less and forgets less [2]. 22 LoReFT response for prompt: How to kill a linux process? 

German subspace : Es gibt mehrere Möglichkeiten, ein Prozess zu beenden. ×

Instruction following subspace : To terminate a Linux process, use the kill command with the PID (process ID) of the target process as its argument. The kill command can be ... ✓

Combined subspaces : Es gibt mehrere Möglichkeiten, ein Prozess zu beenden. ×

RoAd 1 response for prompt: How to kill a linux process? 

German subspace : Kill a Linux process with the command ‘kill -9 <pid>‘ where ‘<pid>‘ is the process ID of the process you want to kill. (Explanation: This is not desired because of the response in English.) ×

Instruction following subspace : To kill a Linux process, you can use the following command: kill -9 <process_id> ... ✓

Combined subspaces : Mit der Kommandozeile ps -ef | grep | cut -d ’ ’ -f 1 | xargs kill -9 will ich den Prozess mit dem PID töten. ✓

LoReFT response for prompt: How to learn English? 

German subspace : Es gibt viele Möglichkeiten, Englisch zu lernen. ×

Instruction following subspace : Hello, I’m here to help you with your question about learning English! There are many ways to learn English, and it depends on what your goals are. If you want to improve your speaking skills ... ✓

Combined subspaces : Es gibt viele Möglichkeiten, Englisch zu lernen. ×

RoAd 1 response for prompt: How to learn English? 

German subspace : Lernen Sie Englisch! ×

Instruction following subspace : I think you should study hard and practice speaking with native speakers. ✓

Combined subspaces : Jetzt lesen Sie einfach mal eine Geschichte in englischer Sprache und üben Sie die Sprache mit Freunden oder Familie. ✓

LoReFT response for prompt: How can I pass an exam? 

German subspace : , wenn du die Prüfung bestanden hast, wird sie dir gegeben. ×

Instruction following subspace : Hello, thank you for your question! To pass an exam, it’s important to prepare well in advance and study regularly. Here are some tips on how to do this effectively: ... ✓

Combined subspaces : Das ist ein sehr guter Ratgeber für die Prüfung. ×

RoAd 1 response for prompt: How can I pass an exam? 

German subspace : Lernen Sie Deutsch! ×

Instruction following subspace : You need to study hard and take the test. ✓

Combined subspaces : Mit einem Blick ins Buch oder auf die Webseite des Lehrers ist es möglich, den richtigen Weg zu finden. ✓

Figure D.1: Qualitative comparison between RoAd and LoReFT for their composability. The prompt is always in English for different subspaces. 23 LoReFT response for prompt: What is the process to borrow a book from the national library? 

German subspace : Es gibt eine Reihe von Büchern, die man ausleihen kann. ×

Instruction following subspace : The process of borrowing a book from the National Library involves several steps, including: ... ✓

Combined subspaces : Es gibt eine Reihe von Schritten, die man durchläuft, um ein Buch aus dem Nationalbibliothek zu mieten. ×

RoAd 1 response for prompt: What is the process to borrow a book from the national library? 

German subspace : Wirft die Buchstabenfolge "national library" in Google und wähle den ersten Link auf. Klicke auf "Borrowing books". Klicke auf die Schaltfläche "How do I borrow a book?" und folge den Anweisungen. ✓

Instruction following subspace : The process to borrow a physical book from the National Library of Australia (NLA) involves three steps: 1) visiting the NLA, 2) searching for the book ... ✓

Combined subspaces : Nachdem Sie sich im Nationalbibliothek-Bibliothekssystem eingeloggt haben, können sie die Bücher ausleihen. ✓

LoReFT response for prompt: Should I learn Python or C++? 

German subspace : , um, ich weiß nicht, ob es sich lohnt, Python zu lernen oder C++. ×

Instruction following subspace : As you know, Python and C++ are two popular programming languages used for various purposes. Both have their own advantages and disadvantages ... ✓

Combined subspaces : , um, ich weiß nicht, ob es sich lohnt, Python zu lernen oder nicht.) ×

RoAd 1 response for prompt: Should I learn Python or C++? 

German subspace : Learn Python because it’s easier to use and more fun than C++, but you should also learn C++ because it’s useful for programming games, apps, and websites. (Explanation: This is not desired because of the response in English.) ×

Instruction following subspace : I think you should learn Python because it’s easier to use and more fun than C++, but if you want to make games, then you should probably learn C++ instead. ✓

Combined subspaces : Mitglied, wie Sie wissen, dass es viele Programmiersprachen gibt, die sich unterscheiden in der Art und Weise, wie sie Code geschrieben werden. Die beiden Sprachen, die ich empfehle, sind Python und C++, aber ich kann nicht sagen, welche ist besser. not sure 

LoReFT response for prompt: How to write a good paper? 

German subspace : Es gibt viele Tipps, wie man einen guten Aufsatz schreibt. ×

Instruction following subspace : As a student, writing a good paper can be challenging and time-consuming. However, there are several steps you can take to ensure your paper is well-written ... ✓

Combined subspaces : Es gibt viele Tipps, wie man einen guten Aufsatz schreibt. ×

RoAd 1 response for prompt: How to write a good paper? 

German subspace : schreibe einen guten Aufsatz ×

Instruction following subspace : A good paper should be well-written, with clear and concise language. It should also have a strong thesis statement, which clearly states your argument or purpose ... ✓

Combined subspaces : Mitbringt, dass das Papier einen guten Titel hat und die Autoren sind auf dem neuesten Stand der Dinge. ✓

Figure D.2: Qualitative comparison between RoAd and LoReFT for their composability. The prompt is always in English for different subspaces. 24 LoReFT response for prompt: Which app do you recommend for taxi? 

German subspace : , um Taxi-App gibt es viele und sie sind alle gut. ×

Instruction following subspace : Hello, I recommend Uber or Lyft as they are reliable and safe options for getting around town. ✓

Combined subspaces : Es gibt viele Apps, die Taxifahrer bieten. Ich würde empfehlen, dass Sie den Taxi-App von Uber verwenden. ✓

RoAd 1 response for prompt: Which app do you recommend for taxi? 

German subspace : Would like to use Uber, because it’s convenient and fast. (Explanation: This is not desired because of the response in English.) ×

Instruction following subspace : I recommend Uber because it’s cheaper and more convenient than other apps. ✓

Combined subspaces : Nachdem Sie sicher sind, dass die Uber-App ist nicht verfügbar in Ihrer Stadt, können Sie auch den Taxify-Dienst nutzen. ✓

LoReFT response for prompt: How can I find a part-time job? I’m a college student and poor. 

German subspace : Sie können sich mit dem Jobsuchdienst von Facebook verbinden, um eine Anzeige zu posten. ✓

Instruction following subspace : Hello, I’m happy to help you with your question! There are several ways to find part-time jobs as a college student. Here are some options: ... ✓

Combined subspaces : Du kannst auf Jobbörsen nachstellen, die sich auf Teilzeitjobs spezialisiert haben. ✓

RoAd 1 response for prompt: How can I find a part-time job? I’m a college student and poor. 

German subspace : Suche ich nach einem Teilzeitjob. Ich bin Student und arbeitslos. ×

Instruction following subspace : You should look for a part-time position in your field of study or one that you are interested in learning about. You may also want to consider working at a local restaurant, retail store, or other business. ✓

Combined subspaces : Siehe dir nach, ob es in der Nähe einen Jobcenter gibt. ✓

Figure D.3: Qualitative comparison between RoAd and LoReFT for their composability. The prompt is always in English for different subspaces. 25
