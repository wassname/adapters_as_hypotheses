Title: Low-Curvature Projections for Scalable Non-Destructive LLM Editing

URL Source: https://arxiv.org/html/2602.15823

Markdown Content:
arXiv is now an independent nonprofit!
Learn more
×
Back to arXiv
Why HTML?
Report Issue
Back to Abstract
Download PDF
Abstract
1Introduction
2The Model editing problem
3CrispEdit: Curvature-Restricted In-Situ Parameter Editing
4Experiments
5Related work
6Conclusion and future work
References
ANotation
BProof of Bregman divergence quadratic form
CProof of Proposition 1
DProof of matrix-free projection
EAdditional details on LLM experiments
FQualitative case study
GAdditional tables
License: CC BY 4.0
arXiv:2602.15823v2 [cs.LG] 01 May 2026
CrispEdit: Low-Curvature Projections for Scalable Non-Destructive LLM Editing
Zarif Ikram
Arad Firouzkouhi
Stephen Tu
Mahdi Soltanolkotabi
Paria Rashidinejad
University of Southern California
{zikram,firouzko,stephen.tu,soltanol,paria.rashidinejad}@usc.edu
(February 17, 2026)
Abstract

A central challenge in large language model (LLM) editing is capability preservation: methods that successfully change targeted behavior can quietly game the editing proxy and corrupt general capabilities, producing degenerate behaviors reminiscent of proxy/reward hacking. We present CrispEdit, a scalable and principled second-order editing algorithm that treats capability preservation as an explicit constraint, unifying and generalizing several existing editing approaches. CrispEdit formulates editing as constrained optimization and enforces the constraint by projecting edit updates onto the low-curvature subspace of the capability-loss landscape. At the crux of CrispEdit is expressing capability constraint via Bregman divergence, whose quadratic form yields the Gauss–Newton Hessian exactly and even when the base model is not trained to convergence. We make this second-order procedure efficient at the LLM scale using Kronecker-factored approximate curvature (K-FAC) and a novel matrix-free projector that exploits Kronecker structure to avoid constructing massive projection matrices. Across standard model-editing benchmarks, CrispEdit achieves high edit success while keeping capability degradation below 1% on average across datasets, significantly improving over prior editors.

\website

https://crispedit.github.io \codehttps://github.com/zarifikram/CrispEdit \emails

1Introduction
Figure 1:Comparison overview of CrispEdit. CrispEdit achieves strong edit reliability and generality, with and without QA context, while preserving broad base capabilities (MMLU, GSM8K, IFEval, ARC-C, TruthfulQA) on LLaMA-3-8B-Instruct.

Large language models (LLMs) are rapidly becoming a shared backbone for knowledge work, spanning search and question answering (gao2023retrieval; lewis2020retrieval), science (jumper2021highly), software development (chen2021evaluating), decision support (lopez2023can), and education (kasneci2023chatgpt). Yet every day, facts shift, new discoveries land, products ship, hallucinations or unsafe behaviors are uncovered, quickly making the models stale. Retraining from scratch is the cleanest way to absorb this drift, but it is also the most expensive and slowest. Model editing (sinitsineditable; de2021editing; mitchell2022memory; wang2024knowledge) offers a practical alternative: apply targeted updates to correct a fact, insert new knowledge, remove unsafe behavior, personalize the style while leaving everything else intact.

Figure 2:Geometric interpretation of CrispEdit compared to baseline editing strategies. Top left: Standard gradient descent effectively minimizes edit loss but moves perpendicular to the capability contours, resulting in high capability loss (degradation). Top right: Projecting onto the nullspace of activation covariance is overly conservative; it preserves representations but restricts the update too heavily to successfully optimize the edit loss. Bottom: CrispEdit projects the update onto the low-curvature subspace of the capability loss. This allows changes in representations to satisfy the edit while moving along the “valley” of the landscape to maintain general model capabilities.

In many cases, edits may appear to succeed while quietly degrading broader capabilities reminiscent of reward/proxy hacking gao2023scaling. This degradation can manifest as brittle reasoning, weaker instruction-following, or even broken fluency. In response, prior work has introduced heuristic guardrails: restrict updates to a small set of parameters (hu2022lora; yu2024melo), localize “where the knowledge lives,” (meng2022locating; yang2025finetuning; gu2025ultraedit) or constrain representation changes (e.g., via subject-centric “knowledge vectors”) (meng2023massediting; fang2025alphaedit). Despite improvements, these methods tend to bake in strong assumptions about edit structure (e.g., explicit subjects/entities) and impose constraints in parameter or representation space that are only indirectly tied to capability preservation, resulting in a poor edit–preservation trade-off. Indeed, editors built on such constraints still perform poorly when tested in the wild under natural autoregressive generation, despite looking strong under unrealistic teacher-forced evaluation that scaffolds the ground-truth prefix and target length (yang-etal-2025-mirage).

In this paper, we adopt a first-principles formulation of model editing: an edit should reduce an edit loss while leaving broader capabilities effectively unchanged. Accordingly, we pose editing as a constrained optimization problem that seeks to minimize the edit loss subject to negligible changes in capability loss, measured on a designated capability set via a distance metric (Section˜2). Standard approaches that replace the constraint with a soft penalty typically require nontrivial tuning and can be prohibitively expensive when the capability set is far larger than the edit set. This motivates us to ask: How to enforce capability preservation directly, without turning editing into full retraining?

To address this, we introduce CrispEdit (Curvature-Restricted In-Situ Parameter Editing), a scalable non-destructive editor, built around the following core ideas.

1. Preserving capabilities with low-curvature projections. A core idea behind CrispEdit is that not all parameter directions are equally important for preserving a model’s capabilities. Recent work shows that the curvature of the pretrained loss landscape can be characterized by the Hessian, which is observed to be highly anisotropic: sharp in a small number of directions and flat in others (sagun2017empirical; oymak2019generalization; pmlr-v97-ghorbani19b; kalra2026scalable). CrispEdit exploits this structure by projecting updates into low-curvature subspaces of Hessian, effectively “hiding” parameter movement where capabilities are minimally affected (see Figure˜2 and Section˜3.1).

2. Avoiding base model convergence requirement with Bregman constraint. A quadratic approximation based on the standard Hessian—which instantiates our formulation with a Euclidean distance—requires assuming that the base model is trained to (near)-convergence, which is rarely satisfied in practice for modern large networks. We resolve this by measuring capability preservation using Bregman divergence. This choice yields a quadratic form expressed exactly in terms of the Gauss-Newton Hessian (GNH), even when the base model is not trained to convergence, avoiding stationarity assumptions.

3. Representation constraints as a restrictive special case. Our Bregman-GNH formulation also sheds light on several successful prior heuristics. We prove (see Proposition˜1) popular editors such as AlphaEdit (fang2025alphaedit) and Adam-NSCL (wang2021training) solve an approximate special case of our framework, but do so within far more restrictive and lower-dimensional subspaces, leading to a worse capability preservation-edit tradeoff (Figure˜1).

4. Scalable matrix-free low-curvature projectors. The remaining challenge is scale: how can we efficiently compute and store curvature information for billion-parameter transformers? CrispEdit addresses this with two key ideas:

(a) 

The resulting GNH is amenable to accurate approximations via Kronecker-factored approximate curvature (martens2015optimizing, K-FAC), which we leverage to enable efficient computation of the low-curvature projection matrix.

(b) 

Instead of explicitly constructing a low-curvature projection matrix, we introduce (Section˜3.3) a matrix-free projector that exploits the Kronecker eigen-structure: rotate gradients into a factored eigenbasis, mask high-curvature components, and rotate back. This makes constraint-aware second-order editing feasible and enables precomputing capability curvature statistics once and reusing them across many future edits, amortizing cost and enabling batch and sequential editing.

Experimental results. We evaluate CrispEdit in both small- and large-scale regimes. In controlled small-scale experiments on image classification (MNIST 
↦
 FashionMNIST), where calculating exact curvature is feasible, we show that Hessian low-curvature projections yield the strongest capability preservation, and that K-FAC closely tracks this behavior cheaply. We then scale CrispEdit to edit LLMs (e.g., LLaMA-3-8B-Instruct and Qwen-2.5-1.5B-Instruct) and evaluate them as used in practice: edits should be reliable in standalone autoregressive generations, generalize across semantically equivalent in-scope queries, and remain local, preserving out-of-scope knowledge and broad skills such as reasoning, instruction-following, and truthfulness. We further test our method in both batch editing, where many edits are applied at once, and sequential editing, where batches of edits are applied to the model sequentially. Across settings, CrispEdit consistently improves the edit–capability trade-off, achieving strong edit success while substantially reducing capability degradation, with modest compute and storage requirements.1

2The Model editing problem

Let 
𝑓
𝜽
:
𝒳
↦
𝒴
 denote a model with parameters 
𝜽
∈
Θ
⊆
ℝ
𝑝
, mapping inputs 
𝑥
∈
𝒳
 to outputs 
𝑦
∈
𝒴
. Model editing seeks to update a pretrained (base) model 
𝑓
𝜽
0
 with initial parameters 
𝜽
0
, using a provided edit target pair 
(
𝑥
,
𝑦
)
∈
𝒳
×
𝒴
, while preserving the existing capabilities of the base model. We formalize this as follows.

Let 
𝒟
cap
=
{
(
𝑥
𝑖
,
𝑦
𝑖
)
}
𝑖
=
1
𝑛
 be a reference dataset that serves as a proxy for capabilities we wish to preserve, an exemplar of the domains on which the model should continue to perform well. We formulate capability preservation through the empirical loss

	
ℒ
cap
​
(
𝜽
;
𝒟
cap
)
:=
1
𝑛
​
∑
𝑖
=
1
𝑛
ℓ
​
(
𝑓
𝜽
​
(
𝑥
𝑖
)
,
𝑦
𝑖
)
,
	

where 
ℓ
​
(
𝑦
^
,
𝑦
)
 is a task-appropriate loss (e.g., cross entropy). Preserving capabilities then means keeping 
ℒ
​
(
𝜽
;
𝒟
cap
)
 close to its pre-edit value, i.e., 
ℒ
​
(
𝜽
;
𝒟
cap
)
≈
ℒ
​
(
𝜽
0
;
𝒟
cap
)
. Let 
𝒟
edit
=
{
(
𝑥
𝑖
,
𝑦
𝑖
)
}
𝑖
=
1
𝑇
 be the edit dataset containing the desired edit pairs. We write 
ℒ
edit
​
(
𝜽
;
𝒟
edit
)
 to denote an edit loss, such as the negative log-likelihood of edit outputs. Using the language of constrained optimization, a natural optimization problem that expresses our desire to minimize edit loss subject to preserving capabilities is the following:2

	
min
𝜽
∈
Θ
⁡
ℒ
edit
​
(
𝜽
)
s.t.
𝖽
​
(
ℒ
cap
​
(
𝜽
)
,
ℒ
cap
​
(
𝜽
0
)
)
≤
𝜀
,
		
(1)

where 
𝖽
​
(
⋅
,
⋅
)
 is a measure of distance, such as the difference between the two loss values or the Bregman divergence, and 
𝜀
 is a small tolerance value. The above formulation is general, unifying and extending many existing model editing frameworks as we discuss in Section˜5.

While Problem (1) rigorously expresses our desired intent for model editing, actually solving (1), especially at LLM scale, is challenging due to the hard constraint. We note that we focus on the constrained formulation above in lieu of the standard Lagrangian relaxation to (1), namely 
min
𝜽
∈
Θ
⁡
ℒ
edit
​
(
𝜽
)
+
𝜆
​
𝖽
​
(
ℒ
cap
​
(
𝜽
)
,
ℒ
cap
​
(
𝜽
0
)
)
. This is due to the fact that in typical operating regimes 
𝑛
 (the number of reference pairs) far exceeds 
𝑇
 (the number of edits), and the computational overhead of gradient-based optimization on the unconstrained problem can be non-trivial. We avoid this complexity by considering an alternative approach to approximating (1) based on low-curvature projections.

3CrispEdit: Curvature-Restricted In-Situ Parameter Editing

We now present CrispEdit for solving Problem (1). The key idea is to edit only along directions that are locally “safe” for maintaining capabilities as informed by the constraint. We start in Section˜3.1 with a simple instantiation of CrispEdit under the standard capability loss difference and derive a principled curvature-restricted model-editing algorithm. Then, in Section˜3.2, we leverage Bregman divergences to derive a practical editing approach that scales to billion-parameter LLMs.

For what follows, we assume that both the maps 
𝑦
^
↦
ℓ
​
(
𝑦
^
,
𝑦
)
 and 
𝜽
↦
𝑓
𝜽
​
(
𝑥
)
 are twice continuously differentiable over their respective domains. This immediately holds for architectures with smooth activation functions such as GeLU/SwiGLU. Furthermore, this assumption can readily be relaxed to functions that are twice differentiable except on a measure zero set, such as architectures with ReLU activations; for simplicity of exposition, we omit these details.

3.1Preserving capabilities with low-curvature updates

We first consider the distance measure to be the standard distance 
𝖽
​
(
𝑎
,
𝑏
)
=
|
𝑎
−
𝑏
|
. Furthermore, in this section, we assume that the base parameters 
𝜽
0
 are a local minima of the capabilities loss 
ℒ
cap
​
(
𝜽
)
; we remove this assumption in Section˜3.2 by using a different distance measure. Applying a second-order Taylor expansion to the constraint in (1) yields 
ℒ
cap
​
(
𝜽
)
−
ℒ
cap
​
(
𝜽
0
)
≈
1
2
​
(
𝜽
−
𝜽
0
)
⊤
​
𝑯
cap
​
(
𝜽
−
𝜽
0
)
, where 
𝑯
cap
≔
∇
𝜃
2
ℒ
cap
​
(
𝜽
0
)
 is the Hessian of the capability loss function evaluated at 
𝜽
0
, and the first term in Taylor expansion is zero because 
∇
𝜃
ℒ
cap
​
(
𝜽
0
)
=
0
. Under this setting, (1) can be approximated by optimizing the following quadratically constrained optimization problem:

	
min
𝜽
∈
Θ
⁡
ℒ
edit
​
(
𝜽
)
s.t.
(
𝜽
−
𝜽
0
)
⊤
​
𝑯
cap
​
(
𝜽
−
𝜽
0
)
≤
𝜀
.
		
(2)

In the deep learning literature, it is well-understood that in typical overparameterized settings, the Hessian 
𝑯
cap
 at the end of training is usually low-rank sagun2017empirical; oymak2019generalization; pmlr-v97-ghorbani19b. Thus, the ellipsoidal constraint in (2) offers many parameter directions around 
𝜽
0
 of low-curvature, where the capability loss 
ℒ
cap
 remains (approximately) invariant. These low-curvature directions enable the optimization (2) to decrease the edit loss 
ℒ
edit
, while limiting loss of capabilities. Furthermore, compared to the Lagrange relaxation objective, the quadratic constraint offers several key advantages:

(a) 

Strict control of capability loss: The ellipsoidal constraint can be enforced via projected gradient or trust-region methods, enabling strict control of tolerated capability degradation; we discuss this shortly.

(b) 

Scalability to billion-parameter models: The second-order relaxation of the constraint forms the foundation for efficiently scaling our approach to billion-parameter LLMs leveraging Bregman divergences (cf. Section˜3.2).

(c) 

Pre-computation: The curvature model 
𝑯
cap
 can be precomputed once and reused across many subsequent edits, amortizing cost and enabling sequential and online interventions (cf. Section˜3.4).

Projected low-curvature gradient descent. We can enforce the constraint in (2) by ensuring that the weight changes 
Δ
​
𝜽
=
𝜽
−
𝜽
0
 are in the (approximate) null-space of the Hessian 
𝑯
cap
, i.e., 
𝑯
cap
​
Δ
​
𝜽
≈
0
 which is equivalent to 
Δ
​
𝜽
∈
𝖭𝗎𝗅𝗅
​
(
𝑯
cap
)
. A sufficient condition to enforce the constraint during gradient descent is projecting the gradients to the approximate null-space of 
𝑯
cap
 at every gradient step.

Let 
𝑯
cap
=
𝑼
​
𝚺
​
𝑼
⊤
 be the eigen-decomposition of 
𝑯
cap
, where 
𝚺
=
𝖽𝗂𝖺𝗀
​
(
𝜎
1
,
…
,
𝜎
𝑝
)
 and 
𝜎
1
≥
⋯
≥
𝜎
𝑝
≥
0
 (recall that 
𝜽
0
 is locally optimal). We construct a low-curvature projector by discarding the top eigenspace. Concretely, given an energy threshold 
𝛾
∈
(
0
,
1
)
, let 
𝑘
:=
min
⁡
{
𝑟
∈
[
𝑝
]
∣
∑
𝑖
=
1
𝑟
𝜎
𝑖
/
∑
𝑖
=
1
𝑝
𝜎
𝑖
≥
𝛾
}
 denote the minimum index capturing 
𝛾
-fraction of the eigenspectrum. Then, the orthogonal projection to the remaining directions 
𝑼
>
𝑘
≔
[
𝑢
𝑘
+
1
​
∣
…
∣
​
𝑢
𝑝
]
 can be computed as:

	
𝒈
𝑡
proj
=
𝑷
𝛾
​
∇
𝜽
ℒ
edit
​
(
𝜽
𝑡
)
,
where
𝑷
𝛾
:=
𝑼
>
𝑘
​
𝑼
>
𝑘
⊤
.
		
(3)

Intuitively, the projection in (3) removes the components of the edit gradient that point in the directions where capability loss is sensitive. We will refer to the subspace spanned by 
𝑼
>
𝑘
 as the 
𝛾
-approximate nullspace.

3.2Gauss-Newton constraint via Bregman divergence

In Section˜3.1 and deriving (2), we assumed that 
∇
𝜽
ℒ
cap
​
(
𝜽
0
)
=
0
. However, in training neural networks, especially LLMs, one typically does not train the network to convergence, to avoid overfitting. Moreover, the capability loss can only be viewed as a mere proxy to the pretraining loss. To avoid relying on the linear term vanishing, we instantiate CrispEdit using a Bregman divergence that is always first-order flat at 
𝜽
0
.

Definition 1 (Bregman divergence). 

For a pair 
(
𝑥
,
𝑦
)
 and loss 
ℓ
​
(
⋅
,
⋅
)
, define the Bregman divergence:

	
𝖽
ℓ
,
𝑦
Breg
(
𝑓
𝜽
(
𝑥
)
,
	
𝑓
𝜽
0
(
𝑥
)
)
:=
ℓ
(
𝑓
𝜽
(
𝑥
)
,
𝑦
)
−
ℓ
(
𝑓
𝜽
0
(
𝑥
)
,
𝑦
)
−
⟨
∇
ℓ
(
𝑓
𝜽
0
(
𝑥
)
,
𝑦
)
,
𝑓
𝜽
(
𝑥
)
−
𝑓
𝜽
0
(
𝑥
)
⟩
.
	

With this definition, we now consider a distance defined as 
𝖽
​
(
ℒ
cap
​
(
𝜽
)
,
ℒ
cap
​
(
𝜽
0
)
)
:=
1
𝑛
​
∑
𝑖
=
1
𝑛
𝖽
ℓ
,
𝑦
𝑖
Breg
​
(
𝑓
𝜽
​
(
𝑥
𝑖
)
,
𝑓
𝜽
0
​
(
𝑥
𝑖
)
)
. A key property of Bregman divergence is that in the second-order Taylor approximation, the gradient is zero for any fixed 
𝜽
, resulting in the following (cf. Appendix B):

	
𝖽
ℓ
Breg
​
(
ℒ
cap
​
(
𝜽
)
,
ℒ
cap
​
(
𝜽
0
)
)
	
≈
1
2
​
(
𝜽
−
𝜽
0
)
⊤
​
𝑮
cap
​
(
𝜽
−
𝜽
0
)
,
	

where 
𝑮
cap
 is the Gauss-Newton Hessian (GNH, also referred to as the Generalized Gauss-Newton), defined as 
𝑮
cap
:=
𝔼
𝒟
cap
​
[
𝑱
⊤
​
𝑯
𝑦
^
​
𝑱
]
. Here, 
𝑱
=
∇
𝜽
𝑓
𝜽
​
(
𝑥
)
 is the network’s parameter-output Jacobian, and 
𝑯
𝑦
^
=
∇
𝑦
^
2
ℓ
 is the Hessian of the loss with respect to the network’s outputs, with the expectation taken empirically over the dataset 
𝒟
cap
. Importantly, 
𝑮
cap
 is well-behaved for overparameterized and partially trained networks, and lends itself to reliable and scalable approximations which we explore below.

Connections to existing model editing methods. It turns out many existing heuristic model editing methods can be viewed as solving the problem (2) via conservative approximations of the quadratic constraint, and with more restrictive assumptions. For example, the popular AlphaEdit technique (fang2025alphaedit) (and related methods like Adam-NSCL (wang2021training)) can be viewed as solving the following approximate optimization problem:

	
	
min
𝜽
⁡
ℒ
edit
​
(
𝜽
)
s.t.
𝜽
−
𝜽
0
∈
𝖭𝗎𝗅𝗅
​
(
𝑲
cap
)
.
		
(4)

Here, matrix 
𝑲
cap
 is constructed from the so-called knowledge vectors for a particular MLP layer, used for preserving capabilities in certain domains of interest. We show that AlphaEdit solves a special, more restrictive problem compared to our approach; the proof can be found in Appendix C.

Proposition 1 (AlphaEdit is more conservative). 

Fix an MLP layer 
𝑙
 and consider updating only the weights of layer 
𝑙
. Let 
𝐊
cap
𝑙
≔
𝐈
⊗
[
𝐚
𝑙
−
1
1
,
…
,
𝐚
𝑙
−
1
𝑛
]
 be the layer-input activations on the capability dataset, and 
𝐆
cap
𝑙
 be the GNH. Then, 
𝖭𝗎𝗅𝗅
​
(
𝐊
cap
𝑙
)
⊆
𝖭𝗎𝗅𝗅
​
(
𝐆
cap
𝑙
)
.

Unlike AlphaEdit’s representation-level restriction via 
𝑲
cap
, our method preserves capabilities through loss curvatures via 
𝑮
cap
. Furthermore, our approach can update multiple layers simultaneously, whereas AlphaEdit edits one layer at a time; consequently, a direct comparison requires matching the edited parameter subset. Proposition˜1 shows that even if we artificially restrict our method to a single layer 
𝑙
, the feasible update subspace defined by the corresponding layerwise GNH is a superset of AlphaEdit’s layerwise subspace. We emphasize that this constraint of the form can be significantly more restrictive than our approach. In particular, 
𝖭𝗎𝗅𝗅
​
(
𝑲
cap
)
 can be a subspace of much smaller dimension than the nullspace of the GNH. Furthermore, in contrast to the knowledge matrix, in practice, the GNH is known to be flat in many directions, e.g., due to network overparameterization sagun2017empirical; oymak2019generalization. Therefore, the constraint in AlphaEdit can be significantly more restrictive, leading to a worse tradeoff between preserving prior capabilities and applying the new edits, as evidenced by our comparative analysis in Section˜4.2.

Result: Representational constraint is a restrictive special case of our formulation
We prove that heuristic methods like AlphaEdit enforce updates within the nullspace of layer inputs (
𝑲
cap
), which is a strict subset of the nullspace of the loss curvature (
𝑮
cap
) utilized by our method. Consequently, AlphaEdit solves a significantly more constrained optimization problem, limiting its accessible parameter space and resulting in a worse tradeoff between editing efficacy and capability preservation.
3.3K-FAC for scalable, matrix-free projections

The remaining obstacle is scale: 
𝑮
cap
 is expensive to compute and represent as a matrix. To address this, we approximate 
𝑮
cap
 with Kronecker-Factored Approximate Curvature (K-FAC)3 (martens2015optimizing; george2018fast). At a high level, K-FAC approximates 
𝑮
cap
 as a block-diagonal matrix, i.e., 
𝑮
cap
≈
blkdiag
​
(
𝑮
cap
1
,
…
,
𝑮
cap
𝐿
)
 for a network with 
𝐿
 layers. To describe each block-diagonal approximation, suppose that layer 
𝑙
 of an MLP computes its outputs as follows: 
𝒔
𝑙
=
𝑾
𝑙
​
𝒂
𝑙
−
1
 and 
𝒂
𝑙
=
𝜙
𝑙
​
(
𝒔
𝑙
)
, where 
𝒂
𝑙
−
1
∈
ℝ
𝑑
in
 are input activations, 
𝑾
𝑙
∈
ℝ
𝑑
out
×
𝑑
in
 are layer weights (including any bias terms), and 
𝒔
𝑙
∈
ℝ
𝑑
out
 are layer pre-activations. Let 
𝒈
𝑙
=
∇
𝒔
𝑙
log
⁡
𝑝
​
(
𝑦
^
∣
𝑥
)
 denote the pseudo-gradients of preactivations. Then, the K-FAC approximation of GNH for layer 
𝑙
 is given by:

	
𝑮
cap
𝑙
≈
𝔼
​
[
𝒂
𝑙
−
1
​
𝒂
𝑙
−
1
⊤
]
⊗
𝔼
​
[
𝒈
𝑙
​
𝒈
𝑙
⊤
]
≔
𝑨
𝑙
−
1
⊗
𝑺
𝑙
.
		
(5)

Here, 
𝑨
𝑙
−
1
 and 
𝑺
𝑙
 are uncentered covariance matrices of the activations and preactivation pseudo-gradients, respectively, with the expectation taken with respect to the capabilities dataset 
𝒟
cap
. This reduces the per-layer storage requirements from 
𝑂
​
(
𝑑
in
2
​
𝑑
out
2
)
 to 
𝑂
​
(
𝑑
in
2
+
𝑑
out
2
)
 memory.

Matrix-free projections without forming 
𝑃
𝛾
(
𝑙
)
. Even with K-FAC approximations in place, explicitly materializing a projector matrix 
𝑷
𝛾
(
𝑙
)
 for the 
𝛾
-approximate nullspace of 
𝑮
cap
(
𝑙
)
 is memory-prohibitive. Thus, we now describe an efficient method to project onto the 
𝛾
-approximate nullspace that does not require explicitly forming 
𝑷
𝛾
(
𝑙
)
. The key idea behind our approach is the fact that the eigenvalues/eigenvectors of a Kronecker product 
𝑴
⊗
𝑵
 are simply the product of the eigenvalues/eigenvectors of 
𝑴
 and 
𝑵
. Specifically, let 
𝑨
𝑙
−
1
=
𝑼
in
​
𝚲
in
​
𝑼
in
⊤
 and 
𝑺
𝑙
−
1
=
𝑼
out
​
𝚲
out
​
𝑼
out
⊤
 denote the respective eigendecompositions of 
𝑨
𝑙
−
1
 and 
𝑺
𝑙
−
1
. We show in Appendix D, for a weight-gradient 
𝑸
𝑙
=
∇
𝑾
𝑙
𝐿
edit
​
(
𝜽
)
, the projected gradient 
𝑸
𝑙
proj
=
mat
(
𝑷
𝛾
(
𝑙
)
​
vec
(
𝑸
𝑙
)
)
 can be written as:

	
𝑸
𝑙
proj
	
=
𝑼
out
​
(
(
𝑼
out
⊤
​
𝑸
𝑙
​
𝑼
in
)
⊙
𝑴
)
​
𝑼
in
⊤
,
		
(6)

where 
⊙
 denotes the Hadamard (entry-wise) matrix product and 
𝑴
𝑖
​
𝑗
=
𝟏
​
[
𝜆
𝑖
out
​
𝜆
𝑗
in
≤
𝜆
𝛾
]
 is a binary mask that selects low-curvature components of the Kronecker matrix; 
𝜆
𝛾
 denotes the largest eigenvalue associated with the 
𝛾
-approximate nullspace of 
𝑷
𝛾
ℓ
. Using this formula, one never needs to form the 
𝑑
in
​
𝑑
out
×
𝑑
in
​
𝑑
out
 projector, further reducing the storage requirement from 
𝑂
​
(
𝑑
in
2
​
𝑑
out
2
)
 to 
𝑂
​
(
𝑑
in
2
+
𝑑
out
2
+
𝑑
in
​
𝑑
out
)
. With this projection in hand, we are ready to define CrispEdit, presented in Algorithm˜1.

Algorithm 1 CrispEdit
0: 
𝜽
0
, 
𝒟
cap
, 
𝒟
edit
, number of epochs 
𝐸
.
0: Edited model parameters 
𝜽
.
1: Compute K-FAC factors 
(
𝑨
𝑙
−
1
,
𝑺
𝑙
)
 for all finetuned layers 
𝑙
 on 
ℒ
​
(
𝜽
;
𝒟
cap
)
; cache 
𝑼
out
(
𝑙
)
,
𝑼
in
(
𝑙
)
, and projection mask 
𝑴
(
𝑙
)
 for each layer (computed via SVD).
2: Initialize parameters 
𝜽
←
𝜽
0
.
3: for 
𝑒
=
1
 to 
𝐸
 do
4:  for each minibatch 
ℬ
⊂
𝒟
edit
 do
5:   Compute gradient 
𝑸
𝑙
 for each fine-tuned layer 
𝑙
.
6:   Project gradient to 
𝑸
𝑙
proj
 (cf. Equation˜6)
7:   Update parameters 
𝜽
 using 
𝑸
𝑙
proj
.
8:  end for
9: end for
Result: K-FAC enables scalable matrix-free curvature projection.
Directly storing the GNH or its projector is memory-prohibitive. We overcome this by approximating the GNH with K-FAC and deriving a matrix-free projection update. By exploiting the Kronecker structure, we project gradients using only the eigendecompositions of the smaller factor matrices 
𝑨
 and 
𝑺
, avoiding the materialization of the full high-dimensional projector. This reduces memory complexity from 
𝑂
​
(
𝑑
in
2
​
𝑑
out
2
)
 to 
𝑂
​
(
𝑑
in
2
+
𝑑
out
2
)
, enabling CrispEdit to scale efficiently.
Algorithm 2 CrispEdit-Seq
0: 
𝜽
0
, 
𝒟
cap
, edits 
𝒟
edit
(
1
)
,
…
,
𝒟
edit
(
𝐾
)
.
0: Edited models 
𝜽
1
,
…
,
𝜽
𝐾
 (updated sequentially).
1: Compute K-FAC factors 
{
𝑨
(
𝑙
−
1
)
,
𝑺
(
𝑙
)
}
 on 
ℒ
​
(
𝜽
;
𝒟
cap
)
.
2: Initialize 
{
𝑨
acc
(
𝑙
−
1
)
,
𝑺
acc
(
𝑙
)
}
←
{
𝑨
cap
(
𝑙
−
1
)
,
𝑺
cap
(
𝑙
)
}
.
3: for 
𝑘
=
1
 to 
𝐾
 do
4:  Solve (1) for 
𝜽
𝑘
 with edit loss 
ℒ
​
(
𝜽
;
𝒟
edit
(
𝑘
)
)
, using layer-wise 
𝛾
-approximate nullspace projections induced by 
{
𝑨
acc
(
𝑙
−
1
)
,
𝑺
acc
(
𝑙
)
}
 (cf. Algorithm˜1).
5:  Compute K-FAC factors 
{
𝑨
edit
,
𝑘
(
𝑙
−
1
)
,
𝑺
edit
,
𝑘
(
𝑙
)
}
 for 
𝒟
edit
(
𝑘
)
.
6:  Aggregate K-FAC factors 
{
𝑨
acc
(
𝑙
−
1
)
,
𝑺
acc
(
𝑙
)
}
 with 
{
𝑨
edit
,
𝑘
(
𝑙
−
1
)
,
𝑺
edit
,
𝑘
(
𝑙
)
}
 via streaming averages.
7: end for
3.4Sequential editing via online projection updates

Up to this point, we have described CrispEdit in a batch editing setting, where we assume all the edits 
𝒟
edit
 are gathered at once, and the base model is updated to incorporate all the edits. A complementary setting is one of sequential editing, where edits (single instances or batches) arrive over time and the model is updated from 
𝑓
𝜽
0
 to 
𝑓
𝜽
1
,
…
,
𝑓
𝜽
𝐾
 in 
𝐾
 successive rounds. Here, at every round 
𝑘
, the goal is to preserve both the base capabilities and the earlier edits in rounds 
1
 to 
𝑘
−
1
 applied to the model. This setting is closely connected to continual (or lifelong) learning de2021continual; shi2025continual and inherits its core failure mode catastrophic forgetting. Batch editing can be viewed as “breadth-first”, integrating many edits at once, whereas sequential editing is “depth-first”, repeatedly revising the model as the new edit data arrive yang2025finetuning.

Concretely, consider a sequence of edit data that arrives over time in chunks: 
𝒟
edit
(
1
)
,
…
,
𝒟
edit
(
𝐾
)
. A naïve algorithm at every round 
𝑘
 sets 
𝒟
edit
=
∪
𝑖
=
1
𝑘
𝒟
edit
(
𝑖
)
, and approximately solves problem (1) using Algorithm˜1. However, this naïve approach must keep all edits around, which can be infeasible and/or impractical for large 
𝐾
 or privacy-sensitive settings yao2023editing. To address these issues, we develop an algorithm (Algorithm˜2) which sequentially maintains the requisite statistics to implement 
𝛾
-approximate nullspace projection. The key idea behind Algorithm˜2 is that the 
𝑨
𝑙
−
1
 and 
𝑺
𝑙
 factors from K-FAC (cf. (5)) are memory-efficient sufficient statistics to summarize the approximate nullspace of the capability loss and the previous edit losses. By updating these statistics online after each round 
𝑘
, we can simultaneously minimize 
ℒ
​
(
𝜽
;
𝒟
edit
(
𝑘
)
)
 while treating both capabilities and the existing edit losses as hard constraints.

4Experiments
4.1Comparison of various second-order constraints

To understand the effect of various second-order constraints on capability preservation in model editing, we consider a simple setting where calculating the Hessian of the model is tractable. Since this is prohibitive for large LLMs, we use LeNet-5 (lecun2002gradient) as a representative model. We pre-train the model to 99% test accuracy on the MNIST dataset (lecun1998mnist) and fine-tune it on the Fashion-MNIST dataset (xiao2017fashion). In this setting, we treat the MNIST loss as the capabilities objective, and the Fashion-MNIST loss as the edit objective.

Figure 3:Tradeoff between pre-training accuracy (capability preservation) and post-training performance (edit efficacy) for different nullspace projection methods. We fine-tune a LeNet-5 model pre-trained on MNIST on Fashion-MNIST in the 
𝛾
-approximate nullspace of the embeddings (Adam-NSCL) Hessian along with Hessian approximations Gauss-Newton Hessian, K-FAC, and EK-FAC (CrispEdit), over a range of energy thresholds 
𝛾
.

For the fine-tuning phase, we first compute the 
𝛾
-approximate nullspace projector of the Hessian of the pre-train loss, applying projected gradient descent (PGD) to fine-tune a one hidden-layer MLP, as described in Section˜3.1. To address the inaccuracy of the projector caused by parameter drift, we recalculate the 
𝛾
-approximate nullspace projector every time parameter changes more than 25%. To understand the trade-off curve between pre-train and fine-tune test accuracy, we sweep over a range of energy threshold 
𝛾
=
1
−
10
−
𝑘
 with 
𝑘
∈
[
1
10
,
7
]
. We then compare this algorithm against running PGD onto four alternative approximate nullspaces: (a) activation covariance (cf. Adam-NSCL wang2021training), (b) Gauss-Newton Hessian, (c) K-FAC martens2015optimizing, and (d) eigenvalue-corrected K-FAC (EK-FAC) george2018fast.

Our results, which illustrate the trade-off between pre-train and fine-tune performance for both the Hessian-based algorithm and the four alternatives (a)-(d), are shown in Figure˜3. We highlight three findings: (i) Projecting gradient updates onto the 
𝛾
-approximate nullspace of the Hessian provides an effective strategy for improving fine-tune accuracy on Fashion-MNIST while maintaining base MNIST performance. (ii) The GNH approach yields a trade-off curve that is quite competitive with the Hessian approach, illustrating the efficacy of the Bregman constraint. This, however, is not the case with the activation covariance used by Adam-NSCL. (iii) Both K-FAC and EK-FAC approximate the performance of the GNH approach reasonably well. The last point (iii) is promising, as it suggests that using K-FAC when we are unable to compute the full Hessian (e.g., LLMs) is a viable approach as we demonstrate next.

Table 1:Comparison of CrispEdit with existing methods on editing LLaMA-3-8B-Instruct. Rel and Gen denote reliability and generalization. We edit 3,000 samples from three datasets, evaluate edits with WILD, and measure base capability on five benchmarks. Values that are best or within 5% of best are in bold.
Data	Method	Edited Capabilities	Base Capabilities	Time
QA Context	No Context					
Rel	Gen	Rel	Gen	MMLU	IFEval	TruthfulQA	ARC-C	GSM8K

ZsRE
	LLaMA-3-8B-Instruct	2.1	1.7	2.9	2.1	69.5	69.3	50.7	58.0	73.5	
MEMIT	0.1	0.0	0.1	0.1	22.9	0.0	51.3	23.5	0.0	9h 27m
AlphaEdit	70.1	60.6	48.1	39.4	52.7	47.7	46.3	40.5	45.5	7h 19m
Adam-NSCL	16.6	15.5	1.9	2.0	69.2	29.6	50.8	42.0	39.5	29m 19s
LocBF-FT	69.5	59.7	25.2	22.1	69.5	70.1	51.6	54.0	75.5	22m 15s
UltraEdit	20.0	16.3	22.7	17.4	69.3	72.5	51.8	54.5	73.0	3m 23s
MEND	0.0	0.0	0.0	0.0	22.9	18.2	0.0	26.0	0.0	58m 20s
FT	46.8	43.1	9.9	8.3	69.3	45.0	48.7	43.0	50.0	4m 32s
FT Sequential	3.6	3.5	0.9	1.2	68.8	19.4	52.8	40.5	6.5	9m 17s
LoRA	9.1	7.4	18.7	7.2	67.8	70.8	52.0	56.0	71.0	47m 24s
LoRA Sequential	4.4	4.0	1.3	0.9	67.3	64.6	56.0	47.0	67.0	3h 12m
CrispEdit	80.5	69.0	57.4	50.9	69.5	67.9	50.5	55.0	76.0	4m 6s
CrispEdit-Seq	71.1	62.9	72.8	60.6	67.8	70.2	53.6	52.0	74.0	43m 36s

CounterFact
	LLaMA-3-8B-Instruct	1.2	1.0	0.3	0.6	69.5	69.3	50.7	58.0	73.5	
MEMIT	0.0	0.0	0.0	0.0	24.6	18.6	49.6	21.0	0.0	7h 30m
AlphaEdit	74.9	57.0	50.5	44.1	47.4	32.9	41.5	40.5	37.5	5h 56m
Adam-NSCL	19.1	8.5	1.7	1.8	68.6	22.8	57.1	39.5	16.5	24m 9s
LocBF-FT	61.1	41.6	10.9	13.3	69.4	65.0	51.3	52.5	74.0	14m 40s
UltraEdit	18.1	12.4	10.2	9.3	69.2	68.6	49.2	52.0	74.0	3m 9s
MEND	0.0	0.0	0.0	0.0	22.9	18.2	0.0	26.0	0.0	17m 42s
FT	12.3	6.0	1.6	2.2	67.4	22.7	50.4	40.0	18.0	4m 12s
FT Sequential	19.1	10.6	1.3	2.2	33.4	20.4	51.3	31.5	0.0	6m 45s
LoRA	13.2	8.3	9.5	2.7	68.2	68.8	53.4	53.0	71.0	51m 34s
LoRA Sequential	6.5	4.8	1.6	2.0	67.3	62.4	53.9	40.0	71.0	2h 16m
CrispEdit	79.4	55.9	38.4	32.4	69.3	67.5	49.5	54.0	76.5	3m 17s
CrispEdit-Seq	66.5	43.8	39.1	29.2	67.9	68.5	56.6	54.0	73.0	34m 39s

WikiBigEdit
	LLaMA-3-8B-Instruct	9.3	9.1	16.4	16.1	69.5	69.3	50.7	58.0	73.5	
MEMIT	0.0	0.0	0.0	0.0	24.6	13.6	52.3	23.5	0.0	10h 42m
AlphaEdit	72.9	66.8	73.9	68.3	58.5	61.6	50.2	50.5	58.0	7h 37m
Adam-NSCL	13.6	13.6	3.4	3.4	69.2	45.3	50.2	42.5	39.0	30m 45s
LocBF-FT	50.4	46.7	16.7	15.7	69.2	73.2	52.0	55.5	73.5	15m 47s
UltraEdit	59.2	54.8	55.4	52.0	69.3	67.7	52.4	53.5	74.5	3m 15s
MEND	0.0	0.0	0.0	0.0	22.9	18.2	0.0	26.0	0.0	38m 36s
FT	23.3	23.2	4.2	4.3	69.5	49.4	49.2	42.5	59.0	5m 12s
FT Sequential	13.4	12.6	1.8	1.5	68.1	34.5	51.8	43.0	29.5	10m 13s
LoRA	30.0	25.8	27.0	15.7	67.8	70.7	55.4	48.0	75.0	58m 42s
LoRA Sequential	20.9	18.7	7.9	7.3	67.8	73.8	54.4	48.0	71.0	4h 54m
CrispEdit	77.0	70.2	28.4	30.5	69.3	70.5	51.8	55.0	74.0	6m 29s
CrispEdit-Seq	66.7	59.8	40.8	38.6	69.2	68.8	50.4	53.0	73.0	38m 47s
4.2Large-scale LLM evaluations

We now study scaling CrispEdit to billion-parameter LLMs, predominately focusing on LLaMA-3-8B-Instruct. We investigate the following: (i) How well can we edit the model? (ii) Do the edits generalize for different contexts? (iii) To what extent can we preserve the model capabilities?

Datasets, metrics, and evaluation. We edit the base model on 3000 samples of three standard model editing datasets: ZsRE levy-etal-2017-zero, CounterFact meng2022locating, and WikiBigEdit thede25awikibigedit. We report two standard edit metrics (de2021editing; yang-etal-2025-mirage): reliability (or efficacy) asks whether the edited model produces an acceptable answer to a given edit query, and generalization asks whether the effects of an edit extend to semantically related contexts. All three datasets contain rewrite prompts for efficacy evaluation, and paraphrased prompts for generalization evaluation. To measure capability degradation, we benchmark edited and base models on diverse tasks: MMLU (hendrycks2020measuring), IFEval (zhou2023instruction), TruthfulQA (lin2022truthfulqa), ARC-Challenge (clark2018think), and GSM8k (cobbe2021training).

An edited LM should apply the edits in a conversational manner and across different contexts. Yet, due to the computational costs, prior works (fang2025alphaedit; gu2025ultraedit) typically use likelihood-based, teacher-forced evaluation that leak both content and length of the ground truth, leading to overestimated performance (yang-etal-2025-mirage). To better capture realistic editing behavior, we follow the WILD evaluation protocol (yang-etal-2025-mirage) that combines context-guided autoregressive decoding of LLM responses with LLM-as-a-judge evaluation. We adopt WILD with EasyEdit (wang2024easyedit), evaluating prompts both with and without QA context. While we do not anticipate any real-world carry-over, we include teacher-forced evaluations in Table˜3 (Appendix) for completeness.

Method and baselines. We edit the base model with CrispEdit by first computing K-FAC caches on Wikipedia samples for five MLP down-projection layers, and then fine-tuning them with PGD in the 
𝛾
-approximate nullspace of caches (cf. Algorithm˜1). We compare against a range of baselines. MEMIT (meng2023massediting) and AlphaEdit (fang2025alphaedit) follow the locate-then-edit paradigm; Adam-NSCL (wang2021training) performs PGD in the feature covariance nullspace; UltraEdit (gu2025ultraedit) leverages sensitivity analysis with online statistics; MEND (mitchell2022fast) uses a hypernetwork to predict parameter changes, FT and LoRA (hu2022lora; zhang2023adaptive) performs standard and low-rank fine-tuning, respectively; and LocBF-FT (yang2025finetuning) constrains fine-tuning to a single, hyperparameter-tuned layer. For more details about the evaluation and baselines, see Appendix E.

Figure 4:Runtime comparison of CrispEdit with other methods. We apply a number of model editing methods to edit LLaMA-3-8B-Instruct on 3,000 ZsRE samples and measure the wall-clock time for execution.

Key results. We report our key results in Table˜1. Across all datasets, we find two consistent patterns. First, aggressive editing approaches—including MEMIT, MEND, FT, and Adam-NSCL—exhibit substantial degradation. While these methods perform well under teacher-forced evaluation (cf. Table˜3, Appendix), the degraded base capabilities adversely affect their editing performance under autoregressive decoding (cf. Appendix F). Second, conservative editing strategies, which restrict updates to limited parameter subspaces, better preserve base capabilities but lead to a suboptimal edited capabilities. AlphaEdit remains a strong baseline of this class, yet it degrades the model’s base capabilities due to its limited nullspace estimate, in addition to needing additional subject-centric representations. In comparison, CrispEdit consistently tops editing performance while preserving the base capabilities nearly intact. Furthermore, it remains computationally efficient (cf. Figure˜4), as it only augments standard fine-tuning with PGD.

Result: CrispEdit achieves superior editing performance while preserving base model capabilities.
Prior editing methods often trade capability preservation for edit quality. Approaches like FT and Adam-NSCL can lead to substantial degradation under autoregressive decoding, while conservative methods such as AlphaEdit require pushing the energy threshold so low that the resulting nullspace becomes a loose approximation, thus improving edits at the cost of base capabilities. CrispEdit consistently yields a better trade-off and achieves high edit performance with nearly intact base capabilities, all the while maintaining computational efficiency via projected gradient descent fine-tuning.

Ablations. We now discuss key findings from ablation experiments; results are provided in Appendix G.

(i) Robustness to energy threshold 
𝛾
. We vary the threshold 
𝛾
 from 0.5 to 0.99. Table˜8 shows that CrispEdit’s base capability preservation is reasonably robust to the threshold, even with 
𝛾
 as small as 
0.5
.

(ii) Sensitivity to the size of capability dataset 
𝑛
. We vary 
𝑛
=
|
𝒟
cap
|
 from 10 to 100,000. Surprisingly, as Table˜7 shows, CrispEdit stays robust across a range of dataset sizes, maintaining strong base capability even with as few as 100 samples. This suggests CrispEdit requires only a small cache to be effective. This raises a question: are capability dataset needed at all? To validate the importance of capability dataset, we run standard finetuning with no projections (i.e., 
𝑛
=
0
). As Figure˜5 shows, while CrispEdit is robust to 
𝑛
, lack of projection yields a detrimental effect on capability preservation. Furthermore, capability preservation can improve edit performance in autoregressive evaluation through maintaining fluency and reliable instruction-following during generation.

Figure 5: Effect of capability dataset size 
𝑛
 on editing performance and base capability preservation. We edit LLaMA-3-8B-Instruct on 3,000 ZsRE samples using CrispEdit for a range of 
𝑛
 and measure the editing performance and base capability preservation.

(iii) Scaling to larger datasets. We increase the size of the the edit dataset, using up to 10,000 ZsRE samples. As Table˜4 shows, CrispEdit scales robustly from 3,000 to 10,000 edits. In contrast, the baselines degrade or plateau at larger scales due to sequential editing, restrictive layer choices, or limited adaptation capacity. Notably, while LocBF-FT performs competitively at 3k edits, its performance drops significantly at 10k edits. This degradation stems from its restriction to a single layer, which lacks the representational capacity required to manage larger-scale knowledge updates.

(iv) Sensitivity to model families. We use CrispEdit to apply 3,000 ZsRE edits to Qwen-2.5-1.5B-Instruct, and compare it against strong baselines. As Table˜5 shows, our method retains its advantages, achieving strong editing performances while retaining base capabilities.

Takeaway: CrispEdit is robust to hyperparameter choices and scales to large-scale editing.
Our ablations demonstrate that CrispEdit is highly resilient to variations in the energy threshold 
𝛾
 and remains effective with a minimal capability cache (as few as 100 samples), though the projection mechanism itself remains essential. Unlike baselines that face capacity bottlenecks at scale (e.g., LocBF-FT), CrispEdit maintains performance up to 10,000 edits and generalizes effectively across different model architectures like Qwen-2.5-1.5B-Instruct.
Figure 6:Consequence of scaling the number of edits up to 10,000. We edit LLaMA-3-8B-Instruct on 3,000 and 10,000 ZsRE samples using several model editing methods and measure their reliability and generality with QA context. Here, darker hue corresponds to larger editing samples.
Figure 7:Evolution of CrispEdit-Seq performance. CrispEdit-Seq shows stronger editing performance whilst retaining previous edits.

Sequential editing with CrispEdit-Seq. Table˜1 shows that CrispEdit-Seq matches the strength of CrispEdit in sequential editing. CrispEdit-Seq also reasonably matches the sequential editing performance of AlphaEdit (the strongest competitor), while retaining base capabilities nearly intact and operating 
8
×
 faster. Figure˜7 shows that CrispEdit-Seq retains previously edited knowledge despite being a depth-first fine-tuning method, challenging previous assumptions that depth-first methods are ill-suited for sequential model editing (yang2025finetuning).

5Related work

Memory-based approaches employ additional memory components to store edits outside its parameters. These components can be in the form of axillary models (dong2022calibrating; mitchell2022memory; hartvigsen2023aging; wang2024wise), in-context learning (wang2024wise, WISE), low-rank adapters (yu2024melo, MELO), or retrieval-based alignment (jiang2024learning, LTE). Compared to these methods, CrispEdit does not assume any data, memory, or architectural augmentations for inference.

Locate-then-edit based approaches aim to locate a set of parameters responsible for a undesired behavior and edit them. They rely on the assumption that feed-forward networks contain the knowledge in models (geva2021transformer; geva2022transformer; dai2022knowledge) and precisely edit the neurons responsible for particular information. They often assume structures in the dataset such as subject or entity (meng2022locating; meng2023massediting; gupta2024unified; fang2025alphaedit; pan2025precise) and relations (dai2022knowledge). An exception to these is gu2025ultraedit, which uses representations of the last token for its localization calculation. In contrast, CrispEdit does not assume any edit structure and does not require locating specific parameters.

Hypernet-based approaches treat predicting parameters shifts as a meta-learning problem and learns a separate network to solve the problem. These methods take the underlying optimization problem of locate-then-edit methods and uses an hypernetwork to predict the parameter shifts, such as mitchell2022fast solving the optimization speed of meng2022locating and tan2024massive solving the least squares problem of meng2023massediting. Recently, li2025reinforced treats the dual optimization problem of model stability and edit quality by treating the hypernetwork as a reinforcement learning (RL) agent. Compared to these methods, CrispEdit has no additional network for predicting parameters shifts.

Constrained fine-tuning approaches perform GD-based finetuning with additional constraints such as weight decay (rawat2021modifying, FT-L), null-space projection (wang2021training, Adam-NSCL), prompt-masking (zhang2024comprehensive, FT-M), low-rank update (yu2024melo, MELO) or strict layer choice (yang2025finetuning, LocBF-FT). CrispEdit builds on this line by combining FT-M with PGD, deriving the projection from a constrained-optimization view of capability preservation leveraging the loss curvature. In this way, CrispEdit aims to reduce the amount of manual strictness (e.g., highly restrictive layer choices or aggressive update limitations) sometimes required for constrained fine-tuning baselines, while retaining the simplicity and scalability of standard fine-tuning. Closest to our method is Adam-NSCL, which applies PGD in the null space of activation covariances. We show that Adam-NSCL is a special, more conservative case (Proposition˜1) and CrispEdit empirically outperforms it.

Continual learning (CL) is closely related to model editing that studies sequential updates while mitigating catastrophic forgetting. Existing methods broadly fall into three categories: regularization-based methods aim to preserve relevant parameters (zenke2017continual), replay-based methods aim to efficiently replay past memories during training (shin2017continual; rebuffi2017icarl), and architecture-based methods adjust model architecture on the fly (rusu2016progressive). Relevant to our work are curvature aware methods, most notably elastic weight consolidation (kirkpatrick2017overcoming, EWC), which estimates old task curvature with the Fisher and adds it as a penalty alongside standard loss to minimize curvature change. Relatedly, li2024hessian performs automatic rank selection with Hessian information of the loss w.r.t base weights and low rank perturbation on the weights to obtain task weights. Recently, gupta2024unified unify different CL methods under a single Bregman divergence-based objective. In comparison, CrispEdit avoids per-step auxiliary loss calculation and scales to LLM editing.

6Conclusion and future work

We formulate model editing as a quadratically constrained optimization problem, introducing CrispEdit and its sequential variant as scalable approaches for editing billion-parameter LLMs while preserving capabilities. Our method leverages Gauss–Newton Hessian eigenspaces, induced by a Bregman divergence constraint, to identify low-curvature directions where the capabilities loss is nearly invariant. We use K-FAC to design efficient projection onto these nullspaces, making CrispEdit practical at LLM scale.

Our work opens up several exciting avenues for future work. The first direction is exploring the use of CrispEdit in other applications, such as safety (e.g., editing out harmful generation and/or hallucinations) and personalization (e.g., changing response style to suit user preferences). Another interesting direction is to utilize CrispEdit for learning interpretable models, e.g., training models to minimize some notion of model complexity such as weight sparsity, feature disentanglement, etc., subject to maintaining model capabilities. Finally, on the algorithmic side, alternative techniques for non-linear constrained optimization, such as trust-region and sequential quadratic programming methods, could enable CrispEdit to take larger, more aggressive fine-tuning steps leading to further improvements on edit capabilities while preserving base capabilities.

References
Appendix ANotation

General notations. We use bold lowercase letters (e.g., 
𝜽
) for vectors and bold uppercase letters (e.g., 
𝑯
) for matrices. For a matrix 
𝑴
, 
𝖭𝗎𝗅𝗅
​
(
𝑴
)
 denotes its null space. The identity matrix is denoted by 
𝑰
. For vectors 
𝒖
,
𝒗
, 
⟨
𝒖
,
𝒗
⟩
 denotes the standard inner product. The operator 
⊙
 denotes the Hadamard (element-wise) product. We denote sets by calligraphy letters e.g., 
𝒳
. We write 
𝔼
𝒟
​
[
𝜙
​
(
𝑧
)
]
=
1
𝑛
​
∑
𝑖
𝜙
​
(
𝑧
𝑖
)
 to denote the empirical expectation of function 
𝜙
​
(
𝑧
)
 using the dataset 
𝒟
=
{
𝑧
𝑖
}
𝑖
=
1
𝑛
. 
⊗
 denotes the Kronecker product. For a subspace 
𝑆
⊆
ℝ
𝑑
, 
𝑃
𝑆
∈
ℝ
𝑑
×
𝑑
 denotes the orthogonal projection onto 
𝑆
.

Models and parameters. Let 
𝑓
𝜽
:
𝒳
→
𝒴
 denote a parametric model with parameters 
𝜽
∈
Θ
⊆
ℝ
𝑝
. The pretrained (base) model parameters are denoted by 
𝜽
0
. We write 
Δ
​
𝜽
:=
𝜽
−
𝜽
0
 for parameter updates.

Datasets. We distinguish between: (i) a capability dataset 
𝒟
cap
=
{
(
𝑥
𝑖
,
𝑦
𝑖
)
}
𝑖
=
1
𝑛
, used as a proxy for behaviors to be preserved, and (ii) an edit dataset 
𝒟
edit
=
{
(
𝑥
𝑖
,
𝑦
𝑖
)
}
𝑖
=
1
𝑇
, specifying desired edits. Typically 
𝑛
≫
𝑇
.

Losses and objectives. Let 
ℓ
​
(
𝑦
^
,
𝑦
)
 denote a task-appropriate loss (e.g., cross-entropy). The empirical capability loss is

	
ℒ
cap
​
(
𝜽
)
=
1
𝑛
​
∑
𝑖
=
1
𝑛
ℓ
​
(
𝑓
𝜽
​
(
𝑥
𝑖
)
,
𝑦
𝑖
)
,
	

and 
ℒ
edit
​
(
𝜽
)
 denotes the edit loss evaluated on 
𝒟
edit
. We measure deviations in capability loss using a distance function 
𝖽
​
(
⋅
,
⋅
)
, including absolute loss differences and Bregman divergences.

Second-order quantities. We denote by

	
𝑯
cap
:=
∇
𝜽
2
ℒ
cap
​
(
𝜽
0
)
	

the Hessian of the capability loss at the base model parameters. When using Bregman divergences, the quadratic approximation is governed by the Gauss–Newton Hessian (GNH),

	
𝑮
cap
:=
𝔼
(
𝑥
,
𝑦
)
∼
𝒟
cap
​
[
𝑱
⊤
​
𝑯
𝑦
^
​
𝑱
]
,
	

where 
𝑱
=
∇
𝜽
𝑓
𝜽
​
(
𝑥
)
 is the parameter–output Jacobian and 
𝑯
𝑦
^
=
∇
𝑦
^
2
ℓ
 is the Hessian of the loss with respect to model outputs.

Low-curvature subspaces. Let 
𝑴
∈
{
𝑯
cap
,
𝑮
cap
}
 admit an eigendecomposition 
𝑴
=
𝑼
​
𝚺
​
𝑼
⊤
, with eigenvalues 
𝜎
1
≥
⋯
≥
𝜎
𝑝
≥
0
. For a threshold 
𝛾
∈
(
0
,
1
)
, we define 
𝑘
 as the smallest index such that

	
∑
𝑖
=
1
𝑘
𝜎
𝑖
/
∑
𝑖
=
1
𝑝
𝜎
𝑖
≥
𝛾
.
	

The 
𝛾
-approximate nullspace is spanned by 
𝑼
>
𝑘
=
[
𝒖
𝑘
+
1
,
…
,
𝒖
𝑝
]
, and the corresponding projector is

	
𝑷
𝛾
:=
𝑼
>
𝑘
​
𝑼
>
𝑘
⊤
.
	

Layerwise notation and K-FAC. For an MLP layer 
ℓ
, we denote input activations by 
𝒂
ℓ
−
1
, weights by 
𝑾
ℓ
∈
ℝ
𝑑
out
×
𝑑
in
, and pre-activation pseudo-gradients by 
𝒈
ℓ
. Under the K-FAC approximation, the layerwise GNH block is approximated as

	
𝑮
cap
(
ℓ
)
≈
𝑨
ℓ
−
1
⊗
𝑺
ℓ
,
	

where 
𝑨
ℓ
−
1
=
𝔼
​
[
𝒂
ℓ
−
1
​
𝒂
ℓ
−
1
⊤
]
 and 
𝑺
ℓ
=
𝔼
​
[
𝒈
ℓ
​
𝒈
ℓ
⊤
]
.

Operators. We use 
vec
(
⋅
)
 and 
mat
(
⋅
)
 to denote vectorization and reshaping operators between matrix and vector forms.

Appendix BProof of Bregman divergence quadratic form

The following lemma computes a second order approximation to Bregman divergence associated with a loss function 
ℓ
.

Proposition 2 (Quadratic Approximation of Bregman Divergence). 

Fix an input 
𝐱
 and parameters 
𝛉
0
∈
ℝ
𝑝
. Assume 
𝐟
𝛉
​
(
𝐱
)
:
𝒳
→
ℝ
𝑚
 is 
𝐶
2
 in 
𝛉
 and 
ℓ
:
ℝ
𝑚
→
ℝ
 is convex and 
𝐶
2
. Define the Bregman divergence

	
𝐷
ℓ
​
(
𝒂
,
𝒃
)
=
ℓ
​
(
𝒂
)
−
ℓ
​
(
𝒃
)
−
⟨
∇
ℓ
​
(
𝒃
)
,
𝒂
−
𝒃
⟩
.
		
(7)

Denote the Jacobian by 
𝐉
​
(
𝛉
)
:=
∇
𝛉
𝐟
𝛉
​
(
𝐱
)
∈
ℝ
𝑚
×
𝑝
 and the output Hessian by 
𝐇
ℓ
​
(
𝐮
)
:=
∇
𝐮
2
ℓ
​
(
𝐮
)
∈
ℝ
𝑚
×
𝑚
. Then, there exists 
𝜌
>
0
 such that for all 
Δ
​
𝛉
 with 
|
Δ
​
𝛉
|
≤
𝜌

	
𝐷
ℓ
​
(
𝒇
𝜽
0
+
Δ
​
𝜽
​
(
𝒙
)
,
𝒇
𝜽
0
​
(
𝒙
)
)
	
=
1
2
​
Δ
​
𝜽
⊤
​
[
𝑱
​
(
𝜽
0
)
⊤
​
𝑯
ℓ
​
(
𝒇
𝜽
0
​
(
𝒙
)
)
​
𝑱
​
(
𝜽
0
)
]
​
Δ
​
𝜽
+
𝑜
​
(
‖
Δ
​
𝜽
‖
2
)
.
	
Proof.

By the chain rule,

	
∇
𝜽
𝐷
ℓ
​
(
𝒇
𝜽
​
(
𝒙
)
,
𝒇
𝜽
0
​
(
𝒙
)
)
	
=
𝑱
​
(
𝜽
)
⊤
​
(
∇
ℓ
​
(
𝒇
𝜽
​
(
𝒙
)
)
−
∇
ℓ
​
(
𝒇
𝜽
0
​
(
𝒙
)
)
)
,
	

which evaluates to zero at 
𝜽
=
𝜽
0
. Differentiating again gives the following decomposition:

	
∇
𝜽
2
𝐷
ℓ
​
(
𝒇
𝜽
​
(
𝒙
)
,
𝒇
𝜽
0
​
(
𝒙
)
)
	
=
𝑱
(
𝜽
)
⊤
𝑯
ℓ
(
𝒇
𝜽
(
𝒙
)
)
𝑱
(
𝜽
)
+
∑
𝑗
=
1
𝑚
(
[
∇
𝒂
𝐷
ℓ
(
𝒂
,
𝒇
𝜽
0
(
𝒙
)
)
]
𝑗
|
𝒂
=
𝒇
𝜽
​
(
𝒙
)
)
∇
𝜽
2
[
𝒇
𝜽
(
𝒙
)
]
𝑗
.
	

At 
𝜽
=
𝜽
0
, 
∇
𝒂
𝐷
ℓ
​
(
𝒇
𝜽
0
​
(
𝒙
)
,
𝒇
𝜽
0
​
(
𝒙
)
)
=
0
 and thus the second term in the above equation evaluates to zero. Therefore,

	
∇
𝜽
2
𝐷
ℓ
​
(
𝒇
𝜽
​
(
𝒙
)
,
𝒇
𝜽
0
​
(
𝒙
)
)
|
𝜽
=
𝜽
0
	
=
𝑱
​
(
𝜽
0
)
⊤
​
𝑯
ℓ
​
(
𝒇
𝜽
0
​
(
𝒙
)
)
​
𝑱
​
(
𝜽
0
)
.
	

Thus, by the second order Taylor approximation of 
𝐷
ℓ
​
(
𝒇
𝜽
​
(
𝒙
)
,
𝒇
𝜽
0
​
(
𝒙
)
)
 around 
𝜽
0
, we conclude

	
𝐷
ℓ
​
(
𝒇
𝜽
0
+
Δ
​
𝜽
​
(
𝒙
)
,
𝒇
𝜽
0
​
(
𝒙
)
)
	
=
1
2
​
Δ
​
𝜽
⊤
​
[
𝑱
​
(
𝜽
0
)
⊤
​
𝑯
ℓ
​
(
𝒇
𝜽
0
​
(
𝒙
)
)
​
𝑱
​
(
𝜽
0
)
]
​
Δ
​
𝜽
+
𝑜
​
(
‖
Δ
​
𝜽
‖
2
)
.
	

∎

Appendix CProof of Proposition 1

Throughout the proof, we drop the dependency on layer 
ℓ
 for notation simplicity. We show that any vectors that belong to the null space of 
𝑲
cap
 also belongs to the null space of 
𝑮
cap
. We interpret 
Δ
​
𝑾
ℓ
∈
Null
​
(
𝑲
cap
ℓ
)
 as the constraint 
Δ
​
𝑾
ℓ
​
𝑲
cap
ℓ
=
0
 (equivalently, 
(
(
𝑲
cap
ℓ
)
⊤
⊗
𝑰
𝑑
out
)
​
vec
​
(
Δ
​
𝑾
ℓ
)
=
0
 under column-wise vectorization). We keep all network parameters fixed except the layer-
ℓ
 weight matrix 
𝑾
∈
ℝ
𝑑
out
×
𝑑
in
. Define the parameter-space representation of layer-
ℓ
 weights and updates by 
𝒘
≔
vec
​
(
𝑾
)
∈
ℝ
𝑑
out
​
𝑑
in
,
 and 
Δ
​
𝒘
≔
vec
​
(
Δ
​
𝑾
)
∈
ℝ
𝑑
out
​
𝑑
in
.
 Define the downstream map 
𝑓
:
ℝ
𝑑
out
→
ℝ
𝑚
 to be the function that takes the layer pre-activation 
𝒔
ℓ
 at layer 
ℓ
 (with all other parameters held fixed) to the network output. Thus, for each capability example 
𝑖
∈
[
𝑛
]
,

	
𝒚
𝑖
​
(
𝑾
)
=
𝑓
​
(
𝑾
​
𝒂
ℓ
−
1
𝑖
)
∈
ℝ
𝑚
.
	

Let 
𝑱
𝑓
​
(
𝒔
ℓ
)
≔
∇
𝒔
ℓ
𝑓
​
(
𝒔
ℓ
)
∈
ℝ
𝑚
×
𝑑
out
 denote the Jacobian of 
𝑓
 at 
𝒔
ℓ
. By the chain rule,

	
∇
𝒘
𝒚
𝑖
​
(
𝑾
0
)
	
=
𝑱
𝑓
​
(
𝑾
0
​
𝒂
ℓ
−
1
𝑖
)
​
∇
𝒘
(
𝑾
​
𝒂
ℓ
−
1
𝑖
)
|
𝑾
=
𝑾
0
.
	

The map 
𝑾
↦
𝑾
​
𝒂
ℓ
−
1
𝑖
 is linear, and its Jacobian under 
𝒘
=
vec
​
(
𝑾
)
 is

	
∇
𝒘
(
𝑾
​
𝒂
ℓ
−
1
𝑖
)
=
𝑰
𝑑
out
⊗
(
𝒂
ℓ
−
1
𝑖
)
⊤
,
	

so the per-example Jacobian with respect to 
𝒘
 can be written as

	
𝑱
𝑖
≔
∇
𝒘
𝒚
𝑖
​
(
𝑾
0
)
=
𝑱
𝑓
​
(
𝑾
0
​
𝒂
ℓ
−
1
𝑖
)
​
(
𝑰
𝑑
out
⊗
(
𝒂
ℓ
−
1
𝑖
)
⊤
)
.
	

Now let 
Δ
​
𝑾
∈
𝖭𝗎𝗅𝗅
​
(
𝑲
cap
)
, i.e. 
Δ
​
𝑾
​
𝒂
ℓ
−
1
𝑖
=
𝟎
 for all 
𝑖
∈
[
𝑛
]
. Using the identity

	
(
𝑰
𝑑
out
⊗
𝒙
⊤
)
​
Δ
​
𝒘
=
Δ
​
𝑾
​
𝒙
for any 
​
𝒙
∈
ℝ
𝑑
in
,
	

we obtain

	
(
𝑰
𝑑
out
⊗
(
𝒂
ℓ
−
1
𝑖
)
⊤
)
​
Δ
​
𝒘
=
Δ
​
𝑾
​
𝒂
ℓ
−
1
𝑖
=
𝟎
∀
𝑖
∈
[
𝑛
]
,
	

and hence 
𝑱
𝑖
​
Δ
​
𝒘
=
𝟎
 for all 
𝑖
. By definition, the layer Gauss–Newton Hessian for the capability objective has the form

	
𝑮
cap
=
∑
𝑖
=
1
𝑛
𝑱
𝑖
⊤
​
𝑯
𝑖
​
𝑱
𝑖
,
	

where each 
𝑯
𝑖
⪰
𝟎
. Therefore, for any vector 
𝒗
,

	
𝒗
⊤
​
𝑮
cap
​
𝒗
=
∑
𝑖
=
1
𝑛
(
𝑱
𝑖
​
𝒗
)
⊤
​
𝑯
𝑖
​
(
𝑱
𝑖
​
𝒗
)
,
	

so if 
𝑱
𝑖
​
𝒗
=
𝟎
 for all 
𝑖
 then 
𝒗
⊤
​
𝑮
cap
​
𝒗
=
0
, which implies 
𝑮
cap
​
𝒗
=
𝟎
 since 
𝑮
cap
⪰
𝟎
. Applying this with 
𝒗
=
Δ
​
𝒘
 and using 
𝑱
𝑖
​
Δ
​
𝒘
=
𝟎
 for all 
𝑖
, we conclude 
𝑮
cap
​
Δ
​
𝒘
=
𝟎
, i.e. 
Δ
​
𝑾
∈
𝖭𝗎𝗅𝗅
​
(
𝑮
cap
)
.

Appendix DProof of matrix-free projection
Proposition 3. 

Let 
𝐀
∈
ℝ
𝑛
𝐴
×
𝑛
𝐴
, 
𝐁
∈
ℝ
𝑛
𝐵
×
𝑛
𝐵
 be two positive semi-definite matrices, 
𝐂
:=
𝐁
⊗
𝐀
 denote the Kronecker product, and let 
𝐗
∈
ℝ
𝑛
𝐴
×
𝑛
𝐵
. Let 
𝜏
:
ℝ
≥
0
↦
{
0
,
1
}
 denote any predicate function, and define the following subspace:

	
𝑆
:=
span
​
{
𝒖
∈
ℝ
𝑛
𝐴
​
𝑛
𝐵
∣
the pair 
(
𝜆
,
𝒖
)
 is an eigenvalue/vector pair of 
𝑪
 with 
𝜏
​
(
𝜆
)
=
1
}
.
	

We have that:

	
mat
(
𝑷
𝑆
​
vec
(
𝑿
)
)
=
𝑼
𝐴
​
(
(
𝑼
𝐴
⊤
​
𝑿
​
𝑼
𝐵
)
⊙
𝑴
)
​
𝑼
𝐵
⊤
,
		
(8)

where 
𝐀
=
𝐔
𝐴
​
diag
​
(
𝜆
𝐴
,
1
,
…
,
𝜆
𝐴
,
𝑛
𝐴
)
​
𝐔
𝐴
⊤
 and 
𝐁
=
𝐔
𝐵
​
diag
​
(
𝜆
𝐵
,
1
,
…
,
𝜆
𝐵
,
𝑛
𝐵
)
​
𝐔
𝐵
⊤
 are the eigen-decompositions of 
𝐀
,
𝐁
 respectively, and 
𝐌
∈
ℝ
𝑛
𝐴
×
𝑛
𝐵
 with 
𝑀
𝑖
​
𝑗
=
𝜏
​
(
𝜆
𝐴
,
𝑖
⋅
𝜆
𝐵
,
𝑗
)
 is the mask matrix corresponding to the predicate function 
𝜏
.

Before we give the proof, we remark that 
mat
:
ℝ
𝑛
𝐴
​
𝑛
𝐵
↦
ℝ
𝑛
𝐴
×
𝑛
𝐵
 above is understood to be the functional inverse of 
vec
:
ℝ
𝑛
𝐴
×
𝑛
𝐵
↦
ℝ
𝑛
𝐴
​
𝑛
𝐵
, i.e., 
mat
(
vec
(
𝑿
)
)
=
𝑿
 for any 
𝑿
∈
ℝ
𝑛
𝐴
×
𝑛
𝐵
.

Proof.

Let us order the columns of 
𝑼
𝐴
 (resp. 
𝑼
𝐵
) as 
𝒖
𝐴
,
𝑖
 (resp. 
𝒖
𝐵
,
𝑗
). From basic properties of Kronecker products, the eigenvalues and eigenvectors of 
𝑪
 are given by 
𝜆
𝐴
,
𝑖
​
𝜆
𝐵
,
𝑗
 and 
𝒖
𝐵
,
𝑗
⊗
𝒖
𝐴
,
𝑖
, with 
𝑖
∈
[
𝑛
𝐴
]
 and 
𝑗
∈
[
𝑛
𝐵
]
. Therefore, 
𝑷
𝑆
 can be written as:

	
𝑷
𝑆
=
∑
𝑖
,
𝑗
=
1
𝑛
𝐴
,
𝑛
𝐵
𝜏
​
(
𝜆
𝐴
,
𝑖
​
𝜆
𝐵
,
𝑗
)
​
(
𝒖
𝐵
,
𝑗
​
𝒖
𝐵
,
𝑗
⊤
⊗
𝒖
𝐴
,
𝑖
​
𝒖
𝐴
,
𝑖
⊤
)
.
	

Hence, using the identity 
vec
(
𝑭
​
𝑿
​
𝑮
)
=
(
𝑮
⊤
⊗
𝑭
)
​
vec
(
𝑿
)
 for any size-conforming 
𝑭
,
𝑿
,
𝑮
,

	
𝑷
𝑆
​
vec
(
𝑿
)
	
=
∑
𝑖
,
𝑗
=
1
𝑛
𝐴
,
𝑛
𝐵
𝜏
​
(
𝜆
𝐴
,
𝑖
​
𝜆
𝐵
,
𝑗
)
​
(
𝒖
𝐵
,
𝑗
​
𝒖
𝐵
,
𝑗
⊤
⊗
𝒖
𝐴
,
𝑖
​
𝒖
𝐴
,
𝑖
⊤
)
​
vec
(
𝑿
)
	
		
=
∑
𝑖
,
𝑗
=
1
𝑛
𝐴
,
𝑛
𝐵
𝜏
​
(
𝜆
𝐴
,
𝑖
​
𝜆
𝐵
,
𝑗
)
​
vec
(
𝒖
𝐴
,
𝑖
​
𝒖
𝐴
,
𝑖
⊤
​
𝑿
​
𝒖
𝐵
,
𝑗
​
𝒖
𝐵
,
𝑗
⊤
)
	
		
=
vec
(
∑
𝑖
,
𝑗
=
1
𝑛
𝐴
,
𝑛
𝐵
𝜏
​
(
𝜆
𝐴
,
𝑖
​
𝜆
𝐵
,
𝑗
)
​
𝒖
𝐴
,
𝑖
⊤
​
𝑿
​
𝒖
𝐵
,
𝑗
⋅
𝒖
𝐴
,
𝑖
​
𝒖
𝐵
,
𝑗
⊤
)
	
		
=
vec
(
𝑼
𝐴
​
(
(
𝑼
𝐴
⊤
​
𝑿
​
𝑼
𝐵
)
⊙
𝑴
)
​
𝑼
𝐵
⊤
)
.
	

Hence the claim follows by taking 
mat
(
⋅
)
 on each side. ∎

Appendix EAdditional details on LLM experiments
Base capability evaluation.

We evaluate the base capabilities of edited models using the lm-evaluation-harness (eval-harness). We benchmark performance on a diverse set of standard reasoning and knowledge tasks, including IFEval, TruthfulQA (MC2), MMLU (5-shot), GSM8K with chain-of-thought prompting (8-shot), and ARC-Challenge (25-shot). For each task, we evaluate 200 examples, applying the chat template and multi-turn few-shot formatting.

Editing performance evaluation.

We use EasyEdit (wang2024easyedit) for evaluation. Except for Table˜3 where we perform teacher-forcing, we follow WILD (yang-etal-2025-mirage) protocol for evaluation. For “No Context”, we use the dataset questions as is. For “QA Context”, That is, we contextualize prompt by appending the template “Please answer the question: \n\nQ: {question}\nA:”, and autoregressively generate up to 40 tokens using predefined stop tokens 
[
.
,
\n
,
eos
]
. We evaluate the generated outputs with gpt-4o-mini (see Figure˜8 for the exact prompt).

Prompt for LLM-as-a-Judge
Your job is to look at a question, a gold target, and a predicted answer, and then assign a grade 
→
 of either ["CORRECT", "INCORRECT"].
The following are examples of CORRECT predicted answers.
Question: What are the names of Barack Obama’s children?
Gold target: Malia Obama and Sasha Obama
Predicted answer 1: sasha and malia obama
Predicted answer 2: Malia and Sasha Obama are the names of Barack Obama’s children.
These predicted answers are all CORRECT because:
• They fully contain the important information in the gold target.
• They do not contain any information that contradicts the gold target.
The following are examples of INCORRECT predicted answers.
Question: What are the names of Barack Obama’s children?
Gold target: Malia and Sasha
Predicted answer 1: Malia.
Predicted answer 2: Malia, Sasha, and Susan.
Predicted answer 3: Malia and Sasha, Malia and Sasha,
Malia and Sasha, Malia and Sasha (repeated answer)
These predicted answers are all INCORRECT because:
• A factual statement in the answer contradicts the gold target or contains repeated content.
Here is a sample. Simply reply with either CORRECT or INCORRECT.
Question: {question}
Gold target: {target}
Predicted answer: {predicted_answer}
According to the gold target, please grade the predicted answer of this question as one of:
• A: CORRECT
• B: INCORRECT
Just return the letters “A” or “B”, with no text around it.
Figure 8:The complete prompt used to employ an LLM as a judge. The judge provides binary assessments (correct or incorrect) based on a given question, gold target answer, and predicted answer.

CrispEdit implementation. For experiments reported in Table˜1, CrispEdit uses 
(
𝑛
,
𝛾
)
=
(
10
,
000
,
0.9
)
 for CounterFact and WikiBigEdit and 
(
𝑛
,
𝛾
)
=
(
10
,
000
,
0.7
)
 for ZsRE, while CrispEdit-Seq uses 
(
𝑛
,
𝛾
)
=
(
30
,
0.999
)
 for ZsRE and CounterFact and 
(
𝑛
,
𝛾
)
=
(
200
,
0.995
)
 for WikiBigEdit. For ZsRE10k experiment reported in Table˜4, CrispEdit uses 
(
𝑛
,
𝛾
)
=
(
1
,
000
,
0.7
)
. For our Qwen-2.5-1.5B-Instruct implementation Table˜5, CrispEdit uses 
(
𝑛
,
𝛾
)
=
(
1000
,
0.7
)
 for ZsRE and 
(
𝑛
,
𝛾
)
=
(
1000
,
0.9
)
 for Counterfact and WikiBigEdit, while CrispEdit-Seq uses 
(
𝑛
,
𝛾
)
=
(
30
,
0.995
)
.

All other hyperparameters are kept fixed across experiments and follow Table˜2.

Non-trivial K-FAC implementation for CrispEdit-Seq. We now discuss one non-trivial design choice made in our implementation. We found that masking prompt tokens for K-FAC calculation (mirroring the fine-tuning setup) yielded suboptimal performance, even with a larger number of tokens (Table˜6). Instead, in our K-FAC calculation for edit samples, we calculate the next token prediction loss over the entire prompt–target sequence. While we are not sure about the underlying cause of this behavior, we suspect that it arises from our relaxed assumption of token independence during K-FAC calculation.

Table 2:Default hyperparameters used for CrispEdit and CrispEdit-Seq.
Hyperparameter	Value
Editing layers (LLaMA-3-8B-Instruct)	{19, 20, 21, 22, 23}
Editing layers (Qwen-2.5-1.5B-Instruct)	{4, 5, 6}
Number of steps	25
Early stopping	0.01
Batch size	32
Chunk size (CrispEdit-Seq)	100
Learning rate (Adam)	
5
×
10
−
4

Baseline implementation. All our baselines follow the code and hyperparameters provided by the EasyEdit framwork. Such hyperparameters come from the original authors of respective baselines that tuned their method for LLaMA-3-8B-Instruct.

Appendix FQualitative case study
Model Editing Case Study 1
Editing Prompt	
What voice type does Marina Rebeka have?

Edit Target	
mezzo-srano

Generation Output
Adam-NSCL	
mezzo-srano-srano-srano-srano-srano-srano-srano-srano-srano-srano-srano-srano-srano-srano-srano-srano

LocBFFT	
mezzo-oprano

AlphaEdit	
mezzo-soprano

UltraEdit	
mezzo soprano

FT	
mezzo-srano-srano-srano-srano-srano-srano-srano-srano-srano-srano-srano-srano-srano-srano-srano-srano

CrispEdit	
mezzo-srano
Model Editing Case Study 2
Editing Prompt	
What is the status of Cebu flowerpecker?

Edit Target	
endangered species

Generation Output
Adam-NSCL	
endangered species Data Deficient species endangered species endangered species Data Deficient species endangered species endangered species endangered species endangered species endangered species endangered species endangered species endangered species endangered species endangered species endangered species endangered species endangered species endangered species endangered species endangered species endangered species endangered species

LocBFFT	
endangered species

AlphaEdit	
endangered

UltraEdit	
critically endangered species

FT	
endangered species species endangered species endangered species endangered species endangered species endangered species endangered species endangered species endangered species endangered species endangered species endangered species endangered species endangered species endangered species endangered species endangered species endangered species endangered species endangered species endangered species endangered species endangered species endangered species endangered

CrispEdit	
endangered species
Appendix GAdditional tables
Table 3:Comparison of CrispEdit with existing methods on editing LLaMA-3-8B-Instruct in the teacher-forcing evaluation pipeline. Rel, gen, Spec denote reliability, generality, and specificity, respectively. We perform model editing on 3,000 samples of three representative datasets and evaluate editing performance and base performance following teacher-forcing setup of meng2023massediting; meng2022locating; fang2025alphaedit. Results that are the highest or within 5% of the highest results are highlighted in bold.
Method	ZsRE	CounterFact	WikiBigEdit
Rel	Gen	Spec	Rel	Gen	Spec	Rel	Gen	Spec
Llama 3 8B Instruct	25.7	25.1	37.8	0.9	1.2	89.4	34.0	34.8	32.8
MEMIT	0.0	0.0	0.0	0.0	0.0	49.4	0.5	0.5	0.0
AlphaEdit	86.7	77.8	32.4	94.3	72.0	69.1	95.0	89.0	42.0
Adam-NSCL	98.8	92.4	22.1	99.5	81.5	47.7	99.7	97.5	36.4
LocBF-FT	99.1	91.1	34.5	99.7	72.7	44.6	99.9	96.8	42.8
UltraEdit	61.9	57.3	31.5	28.0	18.7	51.4	87.4	84.7	47.8
MEND	0.0	0.0	0.1	0.0	0.0	0.0	0.0	0.0	0.0
FT	99.1	93.1	22.9	99.7	82.0	47.9	99.8	97.6	36.3
FT Sequential	79.7	76.6	16.8	78.6	59.8	51.6	93.5	90.5	29.0
LoRA	93.4	60.6	30.9	93.8	17.8	42.9	99.3	82.4	44.4
LoRA Sequential	36.8	32.7	21.7	20.9	10.4	57.0	70.2	65.4	36.0
CrispEdit	99.1	92.1	32.3	99.8	73.0	55.2	99.9	97.1	44.7
CrispEdit-Seq	98.3	91.4	30.2	99.5	62.9	52.3	99.9	96.7	39.5
Table 4:Influence of scaling to larger editing dataset. Rel and Gen denote reliability and generality, respectively. We perform model editing on 10,000 samples of ZsRE and evaluate editing performance with WILD framework and base performance with five representative benchmarks. Results that are the highest or within 5% of the highest results are highlighted in bold.
Data	Method	Edited Capabilities	Base Capabilities
QA Context	No Context					
Rel	Gen	Rel	Gen	MMLU	IFEval	TruthfulQA	ARC-C	GSM8K

ZsRE 10K
	LLaMA-3-8B-Instruct	2.0	1.5	2.9	2.1	69.5	69.3	50.7	58.0	73.5
LocBF-FT	53.5	47.7	11.5	11.6	68.0	67.6	50.7	50.0	73.0
UltraEdit	20.1	16.7	12.6	10.4	67.9	68.9	49.8	46.0	73.0
Adam-NSCL	1.2	1.1	0.4	0.7	68.2	14.8	54.0	35.0	2.0
AlphaEdit	0.3	0.2	0.1	0.0	22.8	20.9	53.9	22.0	0.0
CrispEdit	77.4	68.7	31.1	28.9	68.5	69.9	50.2	52.0	71.0
Table 5:Comparison of CrispEdit with existing methods on editing Qwen-2.5-1.5B-Instruct. Rel and gen denote reliability and generality, respectively. We perform model editing on 3,000 samples of ZsRE and evaluate editing performance with WILD framework and base performance with five representative benchmarks. Results that are the highest or within 5% of the highest results are highlighted in bold.
Data	Model	Edited Capabilities	Base Capabilities
QA Context	No Context					
Rel	Gen	Rel	Gen	MMLU	IFEval	TruthfulQA	ARC-C	GSM8K

ZsRE
	Qwen 2.5 1.5B	3.5	4.0	2.3	2.0	61.9	48.3	50.9	52.0	58.0
FT	35.4	29.6	32.2	25.5	50.0	24.8	49.8	34.5	35.5
LocBF-FT	71.4	52.9	38.0	30.6	59.6	42.0	54.6	44.0	54.0
AlphaEdit	7.2	4.3	6.2	4.2	24.9	12.4	44.7	21.5	2.0
UltraEdit	11.3	9.8	18.2	11.8	62.3	47.7	52.1	50.0	54.0
Adam-NSCL	62.6	50.5	21.4	15.3	59.3	38.0	46.0	44.0	32.0
CrispEdit (Batch) 	77.8	61.0	52.6	44.0	57.8	32.8	46.4	42.0	58.5
CrispEdit (Seq) 	55.5	40.7	77.7	51.6	59.3	39.5	46.0	42.0	59.0

CounterFact
	Qwen 2.5 1.5B	2.0	1.8	0.9	0.7	61.9	48.3	50.9	52.0	58.0
FT	22.3	28.4	8.9	14.2	34.8	15.1	45.7	23.5	6.5
LocBF-FT	58.2	32.6	46.8	21.5	59.3	39.0	46.6	40.5	56.0
AlphaEdit	22.6	14.1	31.2	16.8	24.4	12.9	46.8	19.0	1.5
UltraEdit	10.8	8.5	14.4	5.9	62.4	41.9	44.8	41.5	62.0
Adam-NSCL	5.9	4.9	3.4	1.5	60.5	18.5	48.3	36.0	4.5
CrispEdit (Batch) 	63.3	34.4	67.0	29.9	61.5	40.5	47.3	44.0	58.5
CrispEdit (Seq) 	64.6	41.8	60.3	27.9	58.4	39.9	47.7	43.0	58.0

WikiBigEdit
	Qwen 2.5 1.5B	8.4	8.6	7.0	6.4	61.9	48.3	50.9	52.0	58.0
FT	59.5	50.4	42.2	37.0	54.3	30.7	46.2	39.5	52.0
LocBF-FT	76.8	61.9	66.0	55.2	60.4	34.8	46.1	43.5	58.0
AlphaEdit	0.7	0.7	1.5	1.3	24.4	13.2	48.9	23.5	1.0
UltraEdit	27.5	25.8	53.2	45.5	62.5	41.4	44.5	44.5	60.0
Adam-NSCL	31.9	28.4	11.6	10.4	62.3	36.2	46.4	41.5	33.0
CrispEdit (Batch) 	62.9	52.0	57.3	46.3	61.2	38.7	47.0	45.5	58.5
CrispEdit (Seq) 	53.4	43.3	83.4	60.0	59.9	34.0	47.9	44.5	55.0
Table 6:Effect of prompt masking during K-FAC calculation. Even with larger number of tokens for computing K-FAC, prompt masking leads to suboptimal performance with CrispEdit-Seq.
Method	Rel
CrispEdit (chunk size = 100)	71.1
CrispEdit (chunk size = 500, prompt masking)	12
Table 7:Influence of the size of capability dataset 
𝑛
 on editing performances and base capability preservation. Across a range of 
𝑛
, we set 
𝛾
=
0.9
 for CrispEdit, perform model editing on 3,000 samples of ZsRE, and evaluate editing performance with WILD framework and base performance with five representative benchmarks. Results that are the highest or within 5% of the highest results are highlighted in bold. CrispEdit remains robust across a wide range of 
𝑛
. Highlighted model represents data used in Table˜1.
Data	Sample Size	Edited Capabilities	Base Capabilities
QA Context	No Context					
Rel	Gen	Rel	Gen	MMLU	IFEval	TruthfulQA	ARC-C	GSM8K

ZsRE
	LLaMA-3-8B-Instruct	2.1	1.7	2.9	2.1	69.5	69.3	50.7	58.0	73.5
No Projection (FT)	46.8	43.1	9.9	8.3	69.3	45.0	48.7	43.0	50.0

𝑛
=
10
	53.6	48.5	10.6	9.3	69.1	48.8	50.8	42.5	57.5

𝑛
=
50
	69.8	62.9	24.9	24.5	69.3	68.3	51.8	53.0	74.0

𝑛
=
100
	74.2	66.0	35.8	31.4	69.4	68.1	50.4	52.0	75.0

𝑛
=
500
	78.4	65.9	54.4	47.2	69.5	72.3	51.5	54.5	75.0

𝑛
=
1000
	75.9	63.9	48.8	41.3	69.4	72.3	50.4	54.0	74.5

𝑛
=
10000
	71.2	57.9	48.0	40.3	69.4	68.4	50.3	59.5	73.0

𝑛
=
50000
	71.0	57.3	47.3	39.9	69.2	68.9	50.2	57.0	75.5

𝑛
=
100000
	69.9	55.5	54.2	43.8	69.3	68.3	50.1	56.5	72.0
Table 8:Influence of energy threshold 
𝛾
 on editing performances and base capability preservation. Across a range of 
𝛾
, we set 
𝑛
=
10
,
000
 for CrispEdit, perform model editing on 3,000 samples of three representative datasets, and evaluate editing performance with WILD framework and base performance with five representative benchmarks. Results that are the highest or within 5% of the highest results are highlighted in bold. CrispEdit remains robust across a wide range of 
𝛾
.
Data	Energy threshold	Edited Capabilities	Base Capabilities
QA Context	No Context					
Rel	Gen	Rel	Gen	MMLU	IFEval	TruthfulQA	ARC-C	GSM8K

ZsRE
	LLaMA-3-8B-Instruct	2.1	1.7	2.9	2.1	69.5	69.3	50.7	58.0	73.5
CrispEdit (
𝛾
=
0.5
)	77.4	68.3	43.4	39.1	69.5	67.8	50.5	52.0	77.5
CrispEdit (
𝛾
=
0.6
)	77.8	67.8	56.0	48.0	69.5	70.2	51.0	53.5	75.5
CrispEdit (
𝛾
=
0.7
)	80.5	69.0	57.4	50.9	69.5	67.9	50.5	55.0	76.0
CrispEdit (
𝛾
=
0.8
)	80.3	68.3	52.3	46.0	69.2	66.7	50.1	56.0	77.0
CrispEdit (
𝛾
=
0.9
)	71.2	57.9	48.0	40.3	69.4	68.4	50.3	59.5	73.0
CrispEdit (
𝛾
=
0.95
)	62.7	48.5	38.5	31.7	69.4	68.4	50.4	56.0	76.0
CrispEdit (
𝛾
=
0.99
)	37.8	28.8	35.3	27.6	69.4	68.9	51.1	57.5	73.0

CounterFact
	LLaMA-3-8B-Instruct	1.2	1.0	0.3	0.6	69.5	69.3	50.7	58.0	73.5
CrispEdit (
𝛾
=
0.5
)	45.5	31.5	5.7	7.2	68.5	50.4	51.4	50.0	43.5
CrispEdit (
𝛾
=
0.6
)	65.5	48.7	9.8	14.3	69.7	63.2	52.4	55.5	75.0
CrispEdit (
𝛾
=
0.7
)	75.7	57.3	15.5	20.4	69.4	66.8	51.7	55.0	72.5
CrispEdit (
𝛾
=
0.8
)	79.2	57.4	21.4	25.8	69.5	69.4	49.8	54.5	73.5
CrispEdit (
𝛾
=
0.9
)	79.4	55.9	38.4	32.4	69.3	67.5	49.5	54.0	76.5
CrispEdit (
𝛾
=
0.95
)	72.0	47.5	46.3	33.0	69.4	67.8	50.3	57.0	74.0
CrispEdit (
𝛾
=
0.99
)	51.6	27.7	45.3	26.8	69.4	68.2	51.7	56.0	76.5

WikiBigEdit
	LLaMA-3-8B-Instruct	9.3	9.1	16.4	16.1	69.5	69.3	50.7	58.0	73.5
CrispEdit (
𝛾
=
0.5
)	62.6	58.7	14.3	14.9	69.0	68.8	50.6	55.0	72.5
CrispEdit (
𝛾
=
0.6
)	66.5	60.8	17.4	19.1	69.3	68.2	51.4	53.0	75.0
CrispEdit (
𝛾
=
0.7
)	76.2	69.2	26.3	27.8	69.2	68.8	51.1	54.0	76.5
CrispEdit (
𝛾
=
0.8
)	77.2	72.1	21.2	24.4	69.4	69.1	50.4	55.0	76.5
CrispEdit (
𝛾
=
0.9
)	77.0	70.2	28.4	30.5	69.3	70.5	51.8	55.0	74.0
CrispEdit (
𝛾
=
0.95
)	76.9	68.9	23.4	27.3	69.2	62.6	51.2	57.5	74.5
CrispEdit (
𝛾
=
0.99
)	67.6	57.2	34.4	32.3	69.3	62.5	52.6	58.0	70.5
Experimental support, please view the build logs for errors. Generated by L A T E xml  .
Instructions for reporting errors

We are continuing to improve HTML versions of papers, and your feedback helps enhance accessibility and mobile support. To report errors in the HTML that will help us improve conversion and rendering, choose any of the methods listed below:

Click the "Report Issue" button, located in the page header.

Tip: You can select the relevant text first, to include it in your report.

Our team has already identified the following issues. We appreciate your time reviewing and reporting rendering errors we may not have found yet. Your efforts will help us improve the HTML versions for all readers, because disability should not be a barrier to accessing research. Thank you for your continued support in championing open access for all.

Have a free development cycle? Help support accessibility at arXiv! Our collaborators at LaTeXML maintain a list of packages that need conversion, and welcome developer contributions.

We gratefully acknowledge support from our major funders, member institutions, and all contributors.
About
·
Help
·
Contact
·
Subscribe
·
Copyright
·
Privacy
·
Accessibility
·
Operational Status
(opens in new tab)
Major funding support from
