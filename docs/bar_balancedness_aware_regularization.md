Title: Implicit Regularization of Sharpness-Aware Minimization for Scale-Invariant Problems

URL Source: https://arxiv.org/html/2410.14802

Markdown Content:
 Abstract
1Introduction
2Preliminaries
3SAM for Non-Overparametrized Problems
4SAM for Overparametrized Problems
5Implicit Regularization Made Explicit
6Numerical Experiments
7Discussions
 References
Implicit Regularization of Sharpness-Aware Minimization for Scale-Invariant Problems
Bingcong Li &Liang Zhang &Niao He
Department of Computer Science ETH Zurich, Switzerland {bingcong.li, liang.zhang, niao.he}@inf.ethz.ch
Abstract

Sharpness-aware minimization (SAM) improves generalization of various deep learning tasks. Motivated by popular architectures such as LoRA, we explore the implicit regularization of SAM for scale-invariant problems involving two groups of variables. Instead of focusing on commonly used sharpness, this work introduces a concept termed balancedness, defined as the difference between the squared norm of two variables. This allows us to depict richer global behaviors of SAM. In particular, our theoretical and empirical findings reveal that i) SAM promotes balancedness; and ii) the regularization on balancedness is data-responsive – outliers have stronger impact. The latter coincides with empirical observations that SAM outperforms SGD in the presence of outliers. Leveraging the implicit regularization, we develop a resource-efficient SAM variant, balancedness-aware regularization (BAR), tailored for scale-invariant problems such as finetuning language models with LoRA. BAR saves
95
%
 computational overhead of SAM, with enhanced test performance across various tasks on RoBERTa, GPT2, and OPT-1.3B.

1Introduction

Sharpness-aware minimization (SAM) is emerging as an appealing optimizer, because it enhances generalization performance on various downstream tasks across vision and language applications (Foret et al., 2021; Chen et al., 2022; Bahri et al., 2022). The success of SAM is typically explained using its implicit regularization (IR) toward a flat solution (Wen et al., 2023a).

However, existing results only characterize sharpness/flatness near local minima (Wen et al., 2023a). Little is known about early convergence, despite its crucial role in SAM’s implicit regularization (Agarwala and Dauphin, 2023). In addition, theoretical understanding of SAM highly hinges upon the existence of positive eigenvalues of Hessians (Wen et al., 2023a), leaving gaps in nonconvex scenarios where the Hessian can be negative definite. The limitations above lead to our first question (Q1): can we broaden the scope of implicit regularization to depict global behaviors in SAM?

Moreover, scenarios where SAM popularizes often involve certain form of data anomalies, such as outliers and large data variance. SAM has provable generalization benefits on sparse coding problems in the small signal-to-noise ratio (SNR) regime (Chen et al., 2023). Remarkable performance of SAM is also observed under distributional shifts, e.g., domain adaptation (Wang et al., 2023), meta-learning (Abbas et al., 2022), and transfer learning in language models (Bahri et al., 2022; Sherborne et al., 2023). Evidences above motivate our second question (Q2): can implicit regularization of SAM reflect its enhanced performance under data anomalies?

This work answers both Q1 and Q2 within a class of scale-invariant problems. The focus on scale-invariance is motivated by its prominence in deep learning architectures. Consider variables
𝐱
∈
ℝ
𝑑
1
 and
𝐲
∈
ℝ
𝑑
2
, both in high-dimensional space. The problems of interest can be categorized into non-overparametrization (NOP) and overparametrization (OP), based on whether the dimension of variables (
𝑑
1
+
𝑑
2
) is greater than dimension of
dom
⁢
𝑓
,



NOP:
⁢
min
𝐱
,
𝐲
⁡
𝑓
𝑛
⁢
(
𝐱𝐲
⊤
)
=
𝔼
𝜉
∼
𝒟
⁢
[
𝑓
𝑛
𝜉
⁢
(
𝐱𝐲
⊤
)
]
,

(1a)



OP:
⁢
min
𝐱
,
𝐲
⁡
𝑓
𝑜
⁢
(
𝐱
⊤
⁢
𝐲
)
=
𝔼
𝜉
∼
𝒟
⁢
[
𝑓
𝑜
𝜉
⁢
(
𝐱
⊤
⁢
𝐲
)
]
.

(1b)

Here,
𝑑
1
=
𝑑
2
 is assumed for OP, and
𝒟
 denotes the training data. For both cases, the losses are nonconvex in
(
𝐱
,
𝐲
)
. Scale-invariance refers to that
(
𝛼
⁢
𝐱
,
𝐲
/
𝛼
)
 share the same objective value
∀
𝛼
≠
0
. It naturally calls for implicit regularization from optimization algorithms to determine the value of
𝛼
. We focus on two-variable problems in the main text for simplicity and generalize the results to multi-layer cases in the appendix. Problems (1a) and (1b) are inspired by widely-adopted modules in deep learning, where low rank adapters (LoRA) for finetuning language models is NOP, and softmax in attention falls in OP framework (Hu et al., 2022; Vaswani et al., 2017).



(a) non-overparametrized (NOP) 	(b) overparametrized (OP)
Figure 1:Implicit regularization of SAM on balancedness. The losses for NOP and OP are
𝔼
⁢
[
‖
𝐱𝐲
⊤
−
(
𝐀
+
𝛼
⁢
𝐍
)
‖
2
]
 and
𝔼
⁢
[
‖
𝐱
⊤
⁢
𝐲
−
(
𝑎
+
𝛼
⁢
𝑛
)
‖
2
]
, respectively. Here,
𝐀
 is the ground truth matrix,
𝐍
 is the Gaussian noise, and
𝛼
 controls the SNR. Left of (a) and (b):
|
‖
𝐱
𝑡
‖
2
−
‖
𝐲
𝑡
‖
2
|
 vs. iteration. Right of (a) and (b):
|
‖
𝐠
𝐱
𝑡
‖
2
−
‖
𝐠
𝐲
𝑡
‖
2
|
 vs. iteration, where
(
𝐠
𝐱
𝑡
,
𝐠
𝐲
𝑡
)
 denotes stochastic gradients.

This work studies SAM’s implicit regularization on balancedness, defined as
ℬ
𝑡
=
1
2
⁢
(
‖
𝐱
𝑡
‖
2
−
‖
𝐲
𝑡
‖
2
)
. Balancedness is a useful alternative to sharpness for (1) because: i) it enables us to go beyond local minima and describe the behavior over SAM’s entire trajectory; ii) analyses and assumptions can be significantly simplified when working with
ℬ
𝑡
; and, iii) it enables a data-driven perspective for understanding SAM. Building on balancedness, we answer our major questions.

For Q1, we prove that even with imbalanced initialization, SAM drives
|
ℬ
𝑡
|
→
0
 for OP, while ensuring a small
|
ℬ
𝑡
|
 in NOP. In contrast, we also prove that balancedness of SGD is unchanged over iterations. This clear distinction between SAM and SGD is illustrated in Fig. 1. Thanks to the adoption of balancedness, our results on implicit regularization have no requirement on the batchsize compared to (Wen et al., 2023a) and can be extended to explain
𝑚
-sharpness in (Foret et al., 2021).

Regarding Q2, we present analytical and empirical evidences that data anomalies (e.g., samples with large noise) have stronger impact on balancedness for both NOP and OP. Fig. 1 showcases an example where SAM is applied on the same problem with different SNRs. Smaller SNR (i.e., larger
𝛼
) promotes balancedness faster. Being more balanced with noisy data also aligns well with previous studies (Chen et al., 2023; Wang et al., 2023), which show that SAM performs better than SGD under data anomalies. This data-driven behavior of SAM is well depicted through balancedness.

Our theoretical understanding on balancedness also cultivates practical tools. In particular, we explicify the implicit regularization of SAM as a data-driven regularizer. When applied on top of, e.g., SGD, it enables a computationally efficient variant of SAM, balancedness-aware regularization (BAR), suited for scale-invariant problems such as finetuning language models with LoRA (Hu et al., 2022). BAR eliminates the need to compute the second gradient in SAM, thereby significantly reducing overhead in large-scale settings. BAR improves the test performance of LoRA on three representative downstream tasks on RoBERTa, GPT2, and OPT, while saving
95
%
 computational overhead of SAM. Moreover, this is the first efficient SAM approach derived from SAM’s implicit regularization. In a nutshell, our contribution can be summarized as:

❖

Theories. Balancedness is introduced as a new metric for implicit regularization in SAM. Compared to sharpness, balancedness enables us to depict richer behaviors – SAM favors balanced solutions for both NOP and OP, and data anomalies have stronger regularization on balancedness.

❖

Practice. Implicit regularization of SAM is made explicit for practical merits. The resulting approach, balancedness-aware regularization (BAR), improves accuracy for finetuning language models with LoRA, while significantly saving computational overhead of SAM.

Notation. Bold lowercase (capital) letters denote column vectors (matrices);
∥
⋅
∥
 stands for
ℓ
2
 (Frobenius) norm of a vector (matrix), and
(
⋅
)
⊤
 refers to transpose.

1.1Related Work

Related topics are streamlined here, with comprehensive discussions deferred to Apdx. A.2.

Scale-invariance in deep learning. Scale-invariant modules are prevalent in modern neural networks, such as LoRA, ReLU networks, and softmax in attention. However, scale-invariant problems are not yet fully understood, especially from a theoretical perspective. Neyshabur et al. (2018) develop scale-invariant PAC-Bayesian bounds for ReLU networks. A scale-invariant SGD is developed in (Neyshabur et al., 2015), and this approach becomes more practical recently in (Gonon et al., 2024). Linear neural networks entail scale-invariance and overparametrization simultaneously, and IR of (S)GD on quadratic loss is established in (Arora et al., 2018; Du et al., 2018; Gidel et al., 2019). IR of GD for softmax attention in transformers is studied in (Sheen et al., 2024) assuming linearly separable data. It is pointed out in (Dinh et al., 2017) that sharpness is sensitive to scaling, while our results indicate that when taking the training trajectory into account, SAM excludes extreme scaling.

Mechanism behind SAM. To theoretically explain the success of SAM, Bartlett et al. (2023) analyze sharpness on quadratic losses. Wen et al. (2023a) focus on sharpness of SAM near the solution manifold on smooth loss functions, requiring batchsize to be 1 in the stochastic case. Andriushchenko and Flammarion (2022) consider sparsity of SAM on (overparametrized) diagonal linear networks on a regression problem. Chen et al. (2023) study the benign overfitting of SAM on a two-layer ReLU network. In general, existing studies on SAM’s implicit regularization focus more on sharpness and do not fully capture scale-invariance. In comparison, our results i) are Hessian-free and hence sharpness-free; ii) have no constraint on batchsize; and iii) hold for both NOP and OP.

SAM variants. Approaches in (Kim et al., 2022; Kwon et al., 2021) modify SAM for efficiency under coordinate-wise ill-scaling, while our results suggest that SAM favors balancedness between layers. Computationally efficient SAM variants are developed through reusing or sparsifying gradients (Liu et al., 2022; Mi et al., 2022); stochastic perturbation (Du et al., 2022a); switching to SGD (Jiang et al., 2023); and connecting with distillation (Du et al., 2022b). Our BAR can be viewed as resource-efficient SAM applied specifically for scale-invariant problems such as LoRA. Different from existing works, BAR is the first to take inspiration from the implicit regularization of SAM.

2Preliminaries

This section briefly reviews SAM and then compares sharpness with balancedness. For a smoother presentation, our main numerical benchmark, LoRA (Hu et al., 2022), is revisited in Sec. 5.

2.1Recap of SAM
Algorithm 1 SAM (Foret et al., 2021)
1:Initialize:
𝐰
0
,
𝜌
,
𝑇
,
𝜂
2:for 
𝑡
=
0
,
…
,
𝑇
−
1
 do
3:     Sample
𝜉
 to get a minibatch
ℳ
𝑡
4:     Define stochastic gradient on
ℳ
𝑡
 as
∇
ℎ
𝑡
⁢
(
⋅
)
5:     Find
𝜖
𝑡
=
𝜌
⁢
∇
ℎ
𝑡
⁢
(
𝐰
𝑡
)
/
‖
∇
ℎ
𝑡
⁢
(
𝐰
𝑡
)
‖
6:     Update via
𝐰
𝑡
+
1
=
𝐰
𝑡
−
𝜂
⁢
∇
ℎ
𝑡
⁢
(
𝐰
𝑡
+
𝜖
𝑡
)
7:end for

Sharpness-aware minimization (SAM) is designed originally to seek for solutions in flat basins. The idea is formalized by enforcing small loss around the entire neighborhood in parameter space, i.e.,
min
𝐰
⁡
max
‖
𝜖
‖
≤
𝜌
⁡
ℎ
⁢
(
𝐰
+
𝜖
)
, where
𝜌
 is the radius of considered neighborhood, and
ℎ
⁢
(
𝐰
)
:=
𝔼
𝜉
⁢
[
ℎ
𝜉
⁢
(
𝐰
)
]
. Practical implementation of SAM is summarized under Alg. 1. It is proved in (Wen et al., 2023a) that
‖
∇
ℎ
𝑡
⁢
(
𝐰
)
‖
≠
0
 (in line 5) holds for any
𝜌
 under most initialization. Based on this result and similar to (Dai et al., 2023), we assume that SAM iterates are well-defined.

Limitation of sharpness. Coming naturally with SAM is the so-termed sharpness, given by
𝒮
⁢
(
𝐰
)
:=
max
‖
𝜖
‖
≤
𝜌
⁡
ℎ
⁢
(
𝐰
+
𝜖
)
−
ℎ
⁢
(
𝐰
)
. When
‖
∇
ℎ
⁢
(
𝐰
)
‖
→
0
,
𝒮
⁢
(
𝐰
)
 can be approximated using (scaled) largest eigenvalue of Hessian (Zhuang et al., 2022). This approximation is widely exploited in literature to study the implicit regularization of SAM. Consequently, most results only hold locally – behaviors near
‖
∇
ℎ
⁢
(
𝐰
)
‖
→
0
 are studied. In addition, sharpness (the largest eigenvalue) is not always informative for scale-invariant problems (1). Consider
ℎ
⁢
(
𝑥
,
𝑦
)
=
𝑥
⁢
𝑦
 for example. The sharpness is
1
 for any
(
𝑥
,
𝑦
)
 – these points are not distinguishable in terms of sharpness.

2.2Prelude on Balancedness

Balancedness
ℬ
𝑡
:=
1
2
⁢
(
‖
𝐱
𝑡
‖
2
−
‖
𝐲
𝑡
‖
2
)
 turns out to be an intriguing alternative to sharpness on the scale-invariant problem (1). Being a global metric, balancedness is capable of describing the entire trajectory of an algorithm, regardless of proximity to critical points or definiteness of Hessian.

How does
ℬ
𝑡
 evolve in different algorithms? To set a comparing benchmark of SAM, we first borrow results from previous works on SGD. Following implicit regularization literature such as (Arora et al., 2018, 2019b; Wen et al., 2023a), we consider SGD with infinitesimally small learning rate
𝜂
→
0
 for the NOP problem (1a)


𝐱
𝑡
+
1
=
𝐱
𝑡
−
𝜂
⁢
𝐠
𝐱
𝑡
,
𝐲
𝑡
+
1
=
𝐲
𝑡
−
𝜂
⁢
𝐠
𝐲
𝑡
.

(2)
Theorem 1 ((Arora et al., 2018, 2019a; Ji and Telgarsky, 2019; Ahn et al., 2023)).

When applying SGD on the NOP (1a), the limiting flow with
𝜂
→
0
 satisfies
‖
𝐱
𝑡
‖
2
−
‖
𝐲
𝑡
‖
2
=
‖
𝐱
0
‖
2
−
‖
𝐲
0
‖
2
 for all
𝑡
>
0
. In other words,
d
⁢
ℬ
𝑡
d
⁢
𝑡
=
0
 holds.

Theorem 1 shows that
ℬ
𝑡
≡
ℬ
0
 given
𝜂
→
0
. A graphical illustration can be found in Fig. 1 (a). Another interesting observation is that given the same initialization,
ℬ
𝑡
 is fixed for SGD regardless of training datasets. This suggests that SGD is less adaptive to data. A similar result of Theorem 1 can be established for SGD on OP. The full statement is deferred to Apdx. C.1; see also Fig. 1 (b).

Merits of being balance. Because
ℬ
0
 is preserved, SGD is sensitive to initialization. For example,
(
𝐱
0
,
𝐲
0
)
 and
(
2
⁢
𝐱
0
,
0.5
⁢
𝐲
0
)
 can result in extremely different trajectories, although the same objective value is shared at initialization. Most of existing works initialize
ℬ
0
≈
0
 to promote optimization benefits, because the variance of stochastic gradient is small and the local curvature is harmonized around a balanced solution. Take the stochastic gradient of NOP on minibatch
ℳ
 for example


𝐠
𝐱
=
1
|
ℳ
|
⁢
[
∑
𝜉
∈
ℳ
∇
𝑓
𝑛
𝜉
⁢
(
𝐱𝐲
⊤
)
]
⁢
𝐲
,
𝐠
𝐲
=
1
|
ℳ
|
⁢
[
∑
𝜉
∈
ℳ
∇
𝑓
𝑛
𝜉
⁢
(
𝐱𝐲
⊤
)
]
⊤
⁢
𝐱
.

(3)

Assuming bounded variance
𝔼
⁢
[
‖
1
|
ℳ
|
⁢
∑
𝜉
∈
ℳ
∇
𝑓
𝑛
𝜉
⁢
(
𝐱𝐲
⊤
)
−
∇
𝑓
𝑛
⁢
(
𝐱𝐲
⊤
)
‖
2
]
≤
𝜎
2
, it can be seen that the variance of
[
𝐠
𝐱
,
𝐠
𝐲
]
 is bounded by
𝜎
2
⁢
(
‖
𝐱
‖
2
+
‖
𝐲
‖
2
)
. In other words, among
{
(
𝐱
,
𝐲
)
|
𝐱𝐲
⊤
=
𝐖
}
, gradient variance is minimized if
‖
𝐱
‖
=
‖
𝐲
‖
, i.e., being balance. Moreover, block smoothness parameters
𝐿
𝑛
𝐱
 and
𝐿
𝑛
𝐲
1 also hint upon the difficulties for optimization, where large values typically correspond to slow convergence (Bottou et al., 2018; Nesterov, 2004). With the help of Assumption 1 (in the next subsection), it can be seen that
𝐿
𝑛
𝐱
=
𝐿
𝑛
⁢
‖
𝐲
‖
2
 and
𝐿
𝑛
𝐲
=
𝐿
𝑛
⁢
‖
𝐱
‖
2
. In other words, a large
|
ℬ
𝑡
|
 implies difficulty for optimizing one variable than the other. For these reasons, balancedness is well-appreciated in domains such as matrix factorization/sensing – a special case of (1a) (Tu et al., 2016; Bartlett et al., 2018; Du et al., 2018; Ge et al., 2017). It is also observed that balanced neural networks are easier to optimize relative to unbalanced ones (Neyshabur et al., 2015).

2.3Assumptions and Prerequisites

To gain theoretical insights of scale-invariant problems in (1), we assume that the loss has Lipschitz continuous gradient on
dom
⁢
𝑓
 following common nonconvex optimization and SAM analyses (Bottou et al., 2018; Andriushchenko and Flammarion, 2022; Wen et al., 2023a).

Assumption 1.

Let
𝐖
∈
ℝ
𝑑
1
×
𝑑
2
, and
𝑤
∈
ℝ
. For each
𝜉
,
𝑓
𝑛
𝜉
⁢
(
𝐖
)
 and
𝑓
𝑜
𝜉
⁢
(
𝑤
)
 in (1) have
𝐿
𝑛
, and
𝐿
𝑜
 Lipschitz continuous gradient, respectively.

Scale-invariant problems are challenging to solve even on simple problems in Fig. 1. Even GD can diverge on some manually crafted initialization (De Sa et al., 2015; Arora et al., 2019a). With proper hyperparameters this rarely happens in practice; hence, we focus on scenarios where SGD and SAM do not diverge. This assumption is weaker than the global convergence needed in (Andriushchenko and Flammarion, 2022), and is similar to the assumption on existence (Wen et al., 2023a).

3SAM for Non-Overparametrized Problems

This section tackles the implicit regularization of SAM on NOP (1a). Motivated by practical scenarios such as LoRA, we focus on cases initialized with large
|
ℬ
0
|
.

When ambiguity is absent, the subscript in
𝑓
𝑛
 and
𝐿
𝑛
 is ignored in this section for convenience. Applying Alg. 1 on NOP, the update of SAM can be written as



𝐱
~
𝑡
=
𝐱
𝑡
+
𝜌
⁢
𝑢
𝑡
⁢
𝐠
𝐱
𝑡
,

𝐲
~
𝑡
=
𝐲
𝑡
+
𝜌
⁢
𝑢
𝑡
⁢
𝐠
𝐲
𝑡

(4a)


𝐠
𝐱
~
𝑡
=
∇
𝑓
𝑡
⁢
(
𝐱
~
𝑡
⁢
𝐲
~
𝑡
⊤
)
⁢
𝐲
~
𝑡
,

𝐠
𝐲
~
𝑡
=
[
∇
𝑓
𝑡
⁢
(
𝐱
~
𝑡
⁢
𝐲
~
𝑡
⊤
)
]
⊤
⁢
𝐱
~
𝑡

(4b)


𝐱
𝑡
+
1
=
𝐱
𝑡
−
𝜂
⁢
𝐠
𝐱
~
𝑡
,

𝐲
𝑡
+
1
=
𝐲
𝑡
−
𝜂
⁢
𝐠
𝐲
~
𝑡

(4c)

where
𝜌
>
0
 is the radius of SAM perturbation;
𝑢
𝑡
:=
1
/
‖
𝐠
𝐱
𝑡
‖
2
+
‖
𝐠
𝐲
𝑡
‖
2
; and
𝑓
𝑡
,
∇
𝑓
𝑡
 denote the loss, stochastic gradient on minibatch
ℳ
𝑡
, respectively.

Theorem 2.

(Dynamics of SAM.) Suppose that Assumption 1 holds. Consider SAM for NOP in (4) with a sufficiently small
𝜌
. Let
ℬ
𝑡
:=
1
2
⁢
(
‖
𝐱
𝑡
‖
2
−
‖
𝐲
𝑡
‖
2
)
. For some
|
𝒜
𝑡
|
=
𝒪
⁢
(
𝜌
2
⁢
𝐿
)
 and
𝜂
→
0
, the limiting flow of SAM guarantees that


d
⁢
ℬ
𝑡
d
⁢
𝑡
=
𝜌
⁢
‖
𝐠
𝐱
𝑡
‖
2
−
‖
𝐠
𝐲
𝑡
‖
2
‖
𝐠
𝐱
𝑡
‖
2
+
‖
𝐠
𝐲
𝑡
‖
2
+
𝒜
𝑡
.

(5)

Moreover, the change on
ℬ
𝑡
 depends on the difference of stochastic gradients on
𝐱
𝑡
 and
𝐲
𝑡
, i.e.,


𝜌
⁢
|
‖
𝐠
𝐱
𝑡
‖
−
‖
𝐠
𝐲
𝑡
‖
|
−
𝒪
⁢
(
𝜌
2
⁢
𝐿
)
≤
|
𝑑
⁢
ℬ
𝑡
𝑑
⁢
𝑡
|
≤
𝜌
⁢
|
‖
𝐠
𝐱
𝑡
‖
2
−
‖
𝐠
𝐲
𝑡
‖
2
|
+
𝒪
⁢
(
𝜌
2
⁢
𝐿
)
.

(6)

Unlike SGD for which
d
⁢
ℬ
𝑡
d
⁢
𝑡
=
0
, Theorem 2 states that the balancedness for SAM is driven by gradient difference
‖
𝐠
𝐱
𝑡
‖
2
−
‖
𝐠
𝐲
𝑡
‖
2
. To gain some intuition, if we estimate
‖
𝐠
𝐱
𝑡
‖
2
−
‖
𝐠
𝐲
𝑡
‖
2
∝
‖
𝐲
𝑡
‖
2
−
‖
𝐱
𝑡
‖
2
 based on (3) and ignore
𝒜
𝑡
, it can be seen that
d
⁢
ℬ
𝑡
d
⁢
𝑡
∝
−
𝜌
⁢
ℬ
𝑡
. This indicates the contraction on
|
ℬ
𝑡
|
. A graphical illustration on decreasing
|
ℬ
𝑡
|
, and its relation with gradient difference can be found in Figs. 1 (a) and 2 (a). Moreover, this implicit regularization on balancedness is global as it holds for all
𝑡
 regardless of whether
(
𝐱
𝑡
,
𝐲
𝑡
)
 is close to local optima. Thanks to adopting balancedness as the metric, Theorem 2 also poses no requirement on the batchsize.

SAM promotes balancedness. As discussed in Section 2.2, unbalancedness is burdensome for optimization. SAM overcomes this by implicitly favoring relatively balanced solutions.

Corollary 1.

(Informal.) Under some regularity conditions, there exists
ℬ
¯
𝑡
𝜌
≥
0
 such that whenever
|
ℬ
𝑡
|
>
ℬ
¯
𝑡
𝜌
, the magnitude of
ℬ
𝑡
 shrinks, where
ℬ
¯
𝑡
𝜌
 can be found in (21) at appendix.

Corollary 1 shows that SAM promotes balancedness until
|
ℬ
𝑡
|
 reaches lower bounds
ℬ
¯
𝑡
𝜌
. Because
ℬ
¯
𝑡
𝜌
 depends on SAM’s trajectory, we plot
1
𝑇
⁢
∫
0
𝑇
ℬ
¯
𝑡
𝜌
⁢
𝑑
𝑡
 using dotted lines for better visualization in Fig. 2 (a). It can be seen that our calculation on
ℬ
¯
𝑡
𝜌
 almost matches the balancedness of SAM after sufficient convergence. Being balance also reveals that the benefit of SAM can come from optimization, which is a perspective typically ignored in literature.



(a) threshold of balancedness
ℬ
¯
𝑡
𝜌
	(b) relation with regularization
Figure 2:Implicit regularization of SAM on NOP
𝔼
⁢
[
‖
𝐱𝐲
⊤
−
(
𝐀
+
𝛼
⁢
𝐍
)
‖
2
]
, where
𝛼
 controls SNR. (a) the threshold of balancedness
ℬ
¯
𝑡
𝜌
 in Corollary 1; (b) implicit vs. explicit regularization.

Noisy data have stronger impact on balancedness. Although our discussions extend to more general problems, for simplicity we consider the example in Fig. 2 (a), i.e.,
𝔼
⁢
[
‖
𝐱𝐲
⊤
−
(
𝐀
+
𝛼
⁢
𝐍
)
‖
2
]
, where
𝐀
 is ground truth;
𝐍
 is data noise; and
𝛼
 determines SNR. For this problem, noisy data directly lead to noisy gradients. It can be seen in Fig. 2 (a) that smaller SNR coincides with faster decreasing of
|
ℬ
𝑡
|
. To explain such a data-responsive behavior in implicit regularization, Theorem 2 states that balancedness changes largely when the difference of
‖
𝐠
𝐲
𝑡
‖
 and
‖
𝐠
𝐱
𝑡
‖
 is large. Since
𝔼
⁢
[
‖
𝐠
𝐲
𝑡
‖
2
−
‖
𝐠
𝐱
𝑡
‖
2
]
∝
𝛼
2
 if assuming elements of
𝐍
 to be iid unit Gaussian variables, it thus implies that a small SNR (large
𝛼
) offers large regularization on balancedness.

Extension to LoRA (multi-layer two-variable NOP). For LoRA, the objective is to minimize
𝐷
 blocks of variables simultaneously, i.e.,
min
⁡
𝔼
𝜉
⁢
[
𝑓
𝜉
⁢
(
{
𝐱
𝑙
⁢
𝐲
𝑙
⊤
}
𝑙
=
1
𝐷
)
]
. It is established in Theorem 5 in appendix that SAM cultivates balancedness in a layer-wise fashion, i.e., the magnitude of
ℬ
𝑡
,
𝑙
:=
1
2
⁢
(
‖
𝐱
𝑡
,
𝑙
‖
2
−
‖
𝐲
𝑡
,
𝑙
‖
2
)
 cannot be large for each
𝑙
. However, the
|
d
⁢
ℬ
𝑡
,
𝑙
/
d
⁢
𝑡
|
 can be
𝒪
⁢
(
𝐷
)
 times smaller than Theorem 2 in the worst case because of the additional variables.

Validation of IR on modern architectures. Going beyond the infinitesimally small step size, we adopt
𝜂
=
0.1
 on modern language models to validate our theoretical findings. We consider finetuning a RoBERTa-large with LoRA for few-shot learning tasks. More details can be found later in Section 6.1. Balancedness of SAM and SGD on different layers in various datasets are plotted in Fig. 3. SAM has a clear trend of promoting balancedness, aligning well with our theoretical predictions.

4SAM for Overparametrized Problems

Next, we focus on SAM’s implicit regularization on OP (1b). Overparametrization enables SAM to have stronger regularization on balancedness. Subscripts in
𝑓
𝑜
 and
𝐿
𝑜
 are omitted for convenience. SAM’s per iteration update for OP can be summarized as



𝐱
~
𝑡
=
𝐱
𝑡
+
𝜌
⁢
𝑢
𝑡
⁢
𝐲
𝑡
,

𝐲
~
𝑡
=
𝐲
𝑡
+
𝜌
⁢
𝑢
𝑡
⁢
𝐱
𝑡

(7a)


𝐠
𝐱
~
𝑡
=
𝑓
𝑡
′
⁢
(
𝐱
~
𝑡
⊤
⁢
𝐲
~
𝑡
)
⁢
𝐲
~
𝑡
,

𝐠
𝐲
~
𝑡
=
𝑓
𝑡
′
⁢
(
𝐱
~
𝑡
⊤
⁢
𝐲
~
𝑡
)
⁢
𝐱
~
𝑡

(7b)


𝐱
𝑡
+
1
=
𝐱
𝑡
−
𝜂
⁢
𝐠
𝐱
~
𝑡
,

𝐲
𝑡
+
1
=
𝐲
𝑡
−
𝜂
⁢
𝐠
𝐲
~
𝑡

(7c)

where
𝑢
𝑡
:=
sgn
⁢
(
𝑓
𝑡
′
⁢
(
𝐱
𝑡
⊤
⁢
𝐲
𝑡
)
)
/
‖
𝐱
𝑡
‖
2
+
‖
𝐲
𝑡
‖
2
;
𝑓
𝑡
 and
𝑓
𝑡
′
 denote the loss, stochastic gradient on minibatch
ℳ
𝑡
, respectively. Different from NOP, SAM has stronger regularization on balancedness, where
|
ℬ
𝑡
|
 decreases whenever the norm of stochastic gradient is large. To see this, it is convenient to define
𝒞
𝑡
:=
|
‖
𝐱
𝑡
‖
−
‖
𝐲
𝑡
‖
|
. Note that
𝒞
𝑡
≤
2
⁢
|
ℬ
𝑡
|
.

Theorem 3.

Consider
𝜂
→
0
 for (7). The limiting flow of SAM on OP ensures a decreasing magnitude of
ℬ
𝑡
 whenever
|
𝑓
𝑡
′
⁢
(
𝐱
𝑡
⊤
⁢
𝐲
𝑡
)
|
⋅
𝒞
𝑡
>
𝒪
⁢
(
𝜌
⁢
𝐿
⁢
|
ℬ
𝑡
|
)
. Moreover, the speed of decrease can be lower- and upper- bounded as


𝜌
⁢
|
𝑓
𝑡
′
⁢
(
𝐱
𝑡
⊤
⁢
𝐲
𝑡
)
|
⋅
𝒞
𝑡
−
𝒪
⁢
(
𝜌
2
⁢
𝐿
⁢
|
ℬ
𝑡
|
)
≤
|
𝑑
⁢
ℬ
𝑡
𝑑
⁢
𝑡
|
≤
𝜌
⁢
|
𝑓
𝑡
′
⁢
(
𝐱
𝑡
⊤
⁢
𝐲
𝑡
)
|
⁢
2
⁢
|
ℬ
𝑡
|
+
𝒪
⁢
(
𝜌
2
⁢
𝐿
⁢
|
ℬ
𝑡
|
)
.


Given
𝜌
→
0
 and sufficiently noisy data, Theorem 3 implies that
|
ℬ
𝑡
|
→
0
. Moreover, Theorem 3 also states that the regularization power on balancedness is related to both gradient norm and balancedness itself. The elbow-shaped curve of
|
ℬ
𝑡
|
 in Fig. 1 (b) demonstrates that the regularization power is reducing, as both gradient norm and balancedness shrink over time.

Figure 3:Implicit regularization of SAM on LoRA. We consider few shot learning with LoRA on a RoBERTa-large. For datasets RTE, SST-5, and MNLI, 1st, 12th and 24th query layers’
2
⁢
|
ℬ
𝑡
,
𝑙
|
 are plotted, respectively. The layers are chosen to represent early, middle, and final stages of RoBERTa. The averaged
ℬ
¯
𝑡
,
𝑙
𝜌
 in Corollary 1 is
0.37
,
0.21
, and
0.29
, respectively.

Noisy data have stronger impact on balancedness. As shown in Fig. 1 (b), balancedness is promoted faster on problems with lower SNR. This data-responsive behavior can be already seen from Theorem 3, because
|
d
⁢
ℬ
𝑡
/
d
⁢
𝑡
|
 is directly related with
|
𝑓
𝑡
′
⁢
(
𝐱
𝑡
⊤
⁢
𝐲
𝑡
)
|
, and
𝔼
⁢
[
|
𝑓
𝑡
′
⁢
(
𝐱
𝑡
⊤
⁢
𝐲
𝑡
)
|
]
 is clearly larger when data are more noisy. In other words, SAM exploits noisy data for possible optimization merits from balancedness (see discussions in Sec. 2.2). Overall, the implicit regularization on balancedness aligns well with the empirical observations in presence of data anomalies (Wang et al., 2023; Sherborne et al., 2023), where SAM outperforms SGD by a large margin.

Extension to
𝑚
-sharpness.
𝑚
-sharpness is a variant of SAM suitable for distributed training. It is observed to empirically improve SAM’s performance (Foret et al., 2021).
𝑚
-sharpness evenly divides minibatch
ℳ
𝑡
 into
𝑚
 disjoint subsets, i.e.,
{
𝑓
𝑡
,
𝑗
}
𝑗
=
1
𝑚
, and perform SAM update independently on each subset; see (38) in appendix. It turns out that
𝑚
-sharpness can also be explained using balancedness. With formal proofs in Apdx. C.3, the IR of
𝑚
-sharpness amounts to substitute
|
𝑓
𝑡
′
⁢
(
𝐱
𝑡
⊤
⁢
𝐲
𝑡
)
|
 in Theorem 3 with
1
𝑚
⁢
∑
𝑗
=
1
𝑚
|
𝑓
𝑡
,
𝑗
′
⁢
(
𝐱
𝑡
⊤
⁢
𝐲
𝑡
)
|
. This means that the regularization on balancedness from
𝑚
-sharpness is more profound than vanilla SAM, because
1
𝑚
⁢
∑
𝑗
=
1
𝑚
|
𝑓
𝑡
,
𝑗
′
⁢
(
𝐱
𝑡
⊤
⁢
𝐲
𝑡
)
|
≥
|
𝑓
𝑡
′
⁢
(
𝐱
𝑡
⊤
⁢
𝐲
𝑡
)
|
.

Finally, we connect balancedness with sharpness on local minima of OP.

Lemma 1.

Let
𝒲
∗
=
{
(
𝐱
,
𝐲
)
|
𝐱
⊤
⁢
𝐲
=
𝑤
,
𝑓
′
⁢
(
𝑤
)
=
0
,
𝑓
′′
⁢
(
𝑤
)
>
0
}
 be non-empty. For the OP problem (1b), minimizing sharpness within
𝒲
∗
 is equivalent to finding
ℬ
=
0
 in
𝒲
∗
.

This link showcases that by studying balancedness we can also obtain the implicit regularization on sharpness for free. A concurrent work also links balancedness with sharpness (the largest eigenvalue) for some one-hidden layer neural networks (Singh and Hofmann, 2024). Compared with (Wen et al., 2023a), this is achieved with less assumptions and simplified analyses. More importantly, balancedness enables us to cope with arbitrary batchsize, to explain SAM’s stronger regularization with noisy data, and to extend results to
𝑚
-sharpness.

5Implicit Regularization Made Explicit

Next, insights from our theoretical understanding of SAM are leveraged to build practical tools. We adopt LoRA (Hu et al., 2022) as our major numerical benchmark for scale-invariant problems given its prevalence in practice. More diverse examples on both OP and NOP can be found in Apdx. A.3. Compared to full parameter-tuning, LoRA is more economical in terms of memory not only for finetuning, but also for serving multiple downstream tasks. LoRA and its variants are actively developed and well welcomed by the community; see e.g., HuggingFace’s PEFT codebase.2

5.1Overview of LoRA

Given a pretrained model with frozen weight
𝐖
𝑙
∈
ℝ
𝑑
1
×
𝑑
2
 on a particular layer
𝑙
, the objective of LoRA is to find low rank matrices
𝐗
𝑙
∈
ℝ
𝑑
1
×
𝑟
, and
𝐘
𝑙
∈
ℝ
𝑑
2
×
𝑟
 with
𝑟
≪
min
⁡
{
𝑑
1
,
𝑑
2
}
 such that the loss is minimized for a downstream task, i.e.,


min
{
𝐗
𝑙
,
𝐘
𝑙
}
𝑙
⁡
ℒ
⁢
(
{
𝐖
𝑙
+
𝐗
𝑙
⁢
𝐘
𝑙
⊤
}
𝑙
)
.

(8)

LoRA enjoys parameter efficiency for finetuning thanks to the low-rank matrices
𝐗
𝑙
 and
𝐘
𝑙
. For instance, it only requires 0.8M trainable parameters to finetune a 355M-parameter RoBERTa-large (Hu et al., 2022). The outer product of
𝐗
𝑙
 and
𝐘
𝑙
 induces scale-invariance, and the number of variables renders it NOP. The downside of LoRA, on the other hand, is the drop on test performance due to the parsimony on trainable parameters. Unbalancedness is also unavoidable for LoRA, due to the need of initializing at
𝐗
𝑙
∼
𝒩
⁢
(
0
,
𝜎
2
)
,
𝐘
𝑙
=
𝟎
; see an example of RoBERTa-large in Fig. 3. The unbalancedness leads to instability of LoRA when finetuning RoBERTa on datasets SST-2 and MNLI; see more details in Apdx. D.4.

Integrating SAM with LoRA is a case with mutual benefits – LoRA reduces the additional memory requirement of SAM, while SAM not only overcomes the distributional shift in finetuning (Zhou et al., 2022), but also mitigates the possible inefficiency associated with LoRA’s unbalancedness.

5.2Balancedness-Aware Regularization (BAR)

However, directly applying SAM variants on LoRA exhibits two concerns: i) SAM doubles computational cost due to the need of two gradients; and ii) additional efforts are required to integrate SAM with gradient accumulation and low-precision training (HuggingFace,), which are common techniques for memory and runtime efficiency in large-scale finetuning. Note that concern i) is annoying given the size of language models, especially in setups involving model parallelism.

Our balancedness-aware regularization (BAR) is a highly efficient approach to address both concerns, and it fixes the accuracy drop of LoRA relative to full-parameter finetuning. BAR is also the first efficient SAM variant derived from implicit regularization. The key observation for our algorithm design is that SAM’s implicit regularization on balancedness can be achieved with an explicit regularizer
𝛼
𝑡
⁢
|
𝐱
⊤
⁢
𝐱
−
𝐲
⊤
⁢
𝐲
|
. This regularizer originates from matrix sensing; see e.g., (Tu et al., 2016; Ge et al., 2017). For OP, choosing
𝛼
𝑡
:=
𝒪
⁢
(
|
𝑓
′
⁢
(
𝐱
𝑡
⊤
⁢
𝐲
𝑡
)
|
/
‖
𝐱
𝑡
‖
2
+
‖
𝐲
𝑡
‖
2
)
 recovers SAM’s dynamic on
ℬ
𝑡
 up to an error of
𝒪
⁢
(
𝜌
2
)
; cf. Lemma 2 in appendix. By ignoring this error, it can be seen that
ℬ
𝑡
 decreases when
‖
𝐱
𝑡
‖
≥
‖
𝐲
𝑡
‖
. Following this dynamic, we regulate balancedness based on whether
‖
𝐱
𝑡
‖
≥
‖
𝐲
𝑡
‖
. The resultant approach is termed as overparamterized BAR (oBAR) to reflect its source in OP.

On the other hand, because LoRA is NOP inherently, we take inspiration from Theorem 2 – dropping the term
𝒜
𝑡
 and mimicking dynamics of SAM. In particular, we regulate the objective with
𝛼
𝑡
⁢
(
𝐱
⊤
⁢
𝐱
−
𝐲
⊤
⁢
𝐲
)
 if
‖
𝐠
𝐱
𝑡
‖
2
<
‖
𝐠
𝐲
𝑡
‖
2
; otherwise
𝛼
𝑡
⁢
(
𝐲
⊤
⁢
𝐲
−
𝐱
⊤
⁢
𝐱
)
. The resultant approach is termed as nBAR. A graphical illustration can be found in Fig. 2 (b). It can be observed that nBAR shares similar performance as SAM on NOP. Both nBAR and oBAR can be implemented in the same manner as weight decay, and their detailed steps are summarized in Algs. 2 and 3, respectively.

Another benefit of BAR, in additional to the lightweight computation, is that it can be applied individually on each LoRA layer. As previously discussed (cf. Theorem 5), the number of layers has a negative impact on balancedness. By overcoming this “curse of multi-layer”, BAR can induce better test performance over SAM.

Schedule of
𝛼
𝑡
. In both nBAR and oBAR, one can employ a decreasing scheduler for
𝛼
𝑡
 for algorithmic flexibility. This is motivated by the fact that for both NOP and OP problems, the implicit regularization of SAM is less powerful after sufficient balancedness or near optimal. Commonly adopted cosine and linear schedules work smoothly.

Algorithm 2 nBAR
1:Initialize: learning rate
{
𝜂
𝑡
}
, regularization coefficient
{
𝛼
𝑡
}
2:for 
𝑡
=
0
,
…
,
𝑇
−
1
 do
3:     Get stochastic gradient
𝐠
𝐱
𝑡
 and
𝐠
𝐲
𝑡
4:     if 
‖
𝐠
𝐱
𝑡
‖
≥
‖
𝐠
𝐲
𝑡
‖
 then
5:         
𝐱
𝑡
←
(
1
+
𝛼
𝑡
⁢
𝜂
𝑡
)
⁢
𝐱
𝑡
6:         
𝐲
𝑡
←
(
1
−
𝛼
𝑡
⁢
𝜂
𝑡
)
⁢
𝐲
𝑡
7:     else
8:         
𝐱
𝑡
←
(
1
−
𝛼
𝑡
⁢
𝜂
𝑡
)
⁢
𝐱
𝑡
9:         
𝐲
𝑡
←
(
1
+
𝛼
𝑡
⁢
𝜂
𝑡
)
⁢
𝐲
𝑡
10:     end if
11:     Optimizer update (via Adam or SGD)
12:end for
	Algorithm 3 oBAR
1:Initialize: learning rate
{
𝜂
𝑡
}
, regularization coefficient
{
𝛼
𝑡
}
2:for 
𝑡
=
0
,
…
,
𝑇
−
1
 do
3:     Get stochastic gradient
𝐠
𝐱
𝑡
 and
𝐠
𝐲
𝑡
4:     if 
‖
𝐱
𝑡
‖
≥
‖
𝐲
𝑡
‖
 then
5:         
𝐱
𝑡
←
(
1
−
𝛼
𝑡
⁢
𝜂
𝑡
)
⁢
𝐱
𝑡
6:         
𝐲
𝑡
←
(
1
+
𝛼
𝑡
⁢
𝜂
𝑡
)
⁢
𝐲
𝑡
7:     else
8:         
𝐱
𝑡
←
(
1
+
𝛼
𝑡
⁢
𝜂
𝑡
)
⁢
𝐱
𝑡
9:         
𝐲
𝑡
←
(
1
−
𝛼
𝑡
⁢
𝜂
𝑡
)
⁢
𝐲
𝑡
10:     end if
11:     Optimizer update (via Adam or SGD)
12:end for
6Numerical Experiments

To demonstrate the effectiveness of BAR, numerical experiments are conducted on various deep learning tasks using language models (LMs). Bold and underlined numbers are used to highlight the best and second best performance, respectively. More experimental details can be found in Apdx. D. Code is available at https://github.com/BingcongLi/BAR.

6.1Few-shot Learning with RoBERTa-large and OPT-1.3B
Table 1:Few shot learning on RoBERTa (355M).
†
 denotes results reported by (Malladi et al., 2023)
RoBERTa	SST-2	SST-5	SNLI	MNLI	RTE	TREC	avg (
↑
)
LoRA	91.1±0.8	52.3±2.9	84.3
±
0.3
	78.1±1.3	77.5±2.3	96.6±1.0	80.0
LoRA-SAM	92.2±0.4	54.2±2.0	85.5±0.7	78.7±1.0	80.6±4.3	96.7±0.2	81.3
LoRA-oBAR	91.5±0.9	54.5±2.7	84.9±0.5	78.3±2.2	79.7±2.0	96.7±0.5	80.9
LoRA-nBAR	91.4±0.5	55.0±2.0	84.9±1.4	78.1±0.2	81.0±1.0	96.7±1.0	81.2
Zero-Shot† 	79.0	35.5	50.2	48.8	51.4	32.0	49.5
Table 2:Runtime of BAR (normalized to LoRA, 1x) on OPT-1.3B. SAM relies on FP32 for stability. LoRA and BAR adopt FP16 training since this is the default choice for large models. nBAR and oBAR share similar runtime, hence reported together.
runtime (
↓
)	SST-2	CB	RTE	COPA	ReCoRD	SQuAD
LoRA-SAM	4.43x	3.34x	4.10x	3.28x	4.35x	3.54x
LoRA-BAR	1.05x	1.03x	1.04x	1.05x	1.04x	1.03x
Table 3:Performance of BAR for few shot learning using OPT-1.3B.
OPT-1.3B	SST-2	CB	RTE	COPA	ReCoRD	SQuAD	avg (
↑
)
Prefix	92.9±1.0	71.6±3.0	65.2±2.6	73.0±1.0	69.7±1.0	82.1±1.4	75.8
LoRA	93.1±0.2	72.6±3.7	69.1±4.8	78.0±0.0	70.8±1.0	81.9±1.8	77.6
LoRA-SAM	93.5±0.5	74.3±1.0	70.6±2.7	78.0±0.0	70.9±1.2	83.0±0.7	78.4
LoRA-oBAR	93.6±0.6	75.6±4.5	70.4±4.8	78.0±0.0	70.9±0.8	82.5±0.5	78.5
LoRA-nBAR	93.7±0.7	79.8±4.4	70.5±2.4	78.0±0.0	71.0±1.0	82.3±1.8	79.2
Zero-Shot	53.6	39.3	53.1	75.0	70.2	27.2	53.1
Table 4:Finetuning RoBERTa (355M) with BAR. Results marked with
†
 are taken from (Hu et al., 2022), and those with
∗
 refer to Adapter
P
 in (Hu et al., 2022).
RoBERTa	# para	STS-B	RTE	MRPC	CoLA	QQP	avg (
↑
)
FT† 	355M	92.4	86.6	90.9	68.0	90.2	85.6
Adapter∗ 	0.8M	91.9
±
0.4	80.1
±
2.9	89.7
±
1.2	67.8
±
2.5	91.7
±
0.2	84.2
LoRA	0.8M	92.4
±
0.1	88.2
±
0.6	89.6
±
0.5	64.8
±
1.4	91.4
±
0.1	85.3
LoRA-oBAR	0.8M	92.6
±
0.1	88.7
±
0.2	90.3
±
0.9	65.1
±
1.0	91.6
±
0.1	85.7
LoRA-nBAR	0.8M	92.6
±
0.2	89.2
±
1.3	90.3
±
0.4	65.6
±
1.2	91.6
±
0.1	85.9

The first task to consider is few-shot learning with LoRA (Malladi et al., 2023), where the goal is to finetune a language model with a small training set. We follow the settings in (Malladi et al., 2023), and choose the backbones as RoBERTa-large, a masked LM with 355M parameters, and OPT-1.3B, an autoregressive LM (Liu et al., 2019; Zhang et al., 2022).

Results of the proposed oBAR and nBAR on RoBERTa-large are summarized in Table 1. As indicated by the zero-shot performance, the distributional shift between finetuning and pretraining datasets is obvious. This is a natural setting suitable for SAM and BAR. The averaged test accuracy is improved by
0.9
 and
1.2
 via oBAR and nBAR, respectively. The performance of nBAR is close to SAM. Moreover, BAR saves 74% additional runtime of SAM; see more details in Table 7 in the appendix.

The proposed nBAR and oBAR perform even better when scaling up to OPT-1.3B. BAR reduces the overhead of SAM by more than
95
%
 because of its compatibility with FP16 training; see Table 2. Note that applying FP16 directly with SAM leads to underflow; see more in Apdx. D. This signifies the flexibility of BAR over SAM when scaling to large problems, as FP16 is the default choice for LMs. Prefix tuning (Li and Liang, 2021) is also included as a benchmark for comparisons on test performance. We report F1 score for SQuAD and accuracy for other datasets in Table 3. The averaged improvement over LoRA is
0.9
 and
1.6
 from oBAR and nBAR, respectively, both outperforming SAM. We conjecture that the performance gap between SAM and BAR comes from their different effectiveness in regularizing balancedness. Balancedness of a particular layer is decreasing slower in SAM due to multiple layers, as shown in Theorem 5, while BAR promotes balancedness faster as it can be applied individually on each LoRA layer. Comparing the absolute improvement for RoBERTa-large (355M) and OPT-1.3B, it is conjectured that BAR has more potential for larger models, and the verification is left for future due to hardware constraints.

6.2Finetuning with RoBERTa-large

Having demonstrated the power of BAR in few-shot learning, we then apply it to finetune RoBERTa-large with LoRA. The results can be found in Table 4. It can be observed that nBAR and oBAR improve the performance of LoRA and prefix tuning (Li and Liang, 2021) on most of tested datasets. On average, oBAR leads to a gain of
0.4
, and nBAR raises the test performance by
0.6
. BAR thereby fills the gap of test performance between LoRA (0.8M) and full-parameter (355M) finetuning.

6.3Text Generation on GPT2-medium

Lastly, we consider BAR on a text-generation problem using GPT2-medium, a model with 345M parameters. Results on WebNLG (Gardent et al., 2017) are reported in Table 5. It can be seen that oBAR matches the performance of prefix tuning, while nBAR achieves the best BLEU score.

Table 5:Finetuning GPT2 (345M) with BAR on WebNLG. Results of prefix tuning and full-parameter finetuning are obtained from (Hu et al., 2022).
GPT2	FT∗	Prefix∗	LoRA	LoRA-oBAR	LoRA-nBAR
# param	354M	0.35M	0.35M	0.35M	0.35M
BLEU (
↑
)	46.5	55.1	54.99
±
0.24	55.15
±
0.19	55.20
±
0.16
7Discussions

This work provides theoretical and empirical evidence on the implicit regularization of SAM for both scale-invariant NOP and OP problems. Balancedness, as an alternative to commonly adopted sharpness, is employed as the metric to capture global and data-responsive behaviors of SAM. We find that i) SAM promotes variables to have (relatively) balanced norms; and ii) noisy data have stronger impact on balancedness. Lastly, we explicify the implicit regularization as a data-driven regularizer to foster the design of a computationally efficient SAM variant, termed BAR. The effectiveness of BAR is demonstrated using various tasks on RoBERTa-large, GPT2 and OPT. BAR saves
95
%
 overhead of SAM and enhances the accuracy of LoRA to the level of full-parameter finetuning.

Limitation and Future directions.

Our approach, BAR, is best applied on scale-invariant modules in neural networks. Finetuning language models with LoRA, as a popular option in practice, is a setting naturally suitable for our approach. However, our approach does not apply for linear models, e.g., logistic regression. Regarding future directions, an interesting one is whether SAM has other forms of implicit regularization beyond balancedness and sharpness. The exploration of other scale-invariant architectures beyond LoRA, e.g., the softmax function in attention, is also deferred to future work.

Acknowledgements

We thank anonymous reviewers for their suggestions. BL is supported by Swiss National Science Foundation (SNSF) Project Funding No. 200021-207343. LZ gratefully acknowledges funding by the Max Planck ETH Center for Learning Systems (CLS). NH is supported by ETH research grant funded through ETH Zurich Foundations and SNSF Project Funding No. 200021-207343.

References
Abbas et al. (2022)	Momin Abbas, Quan Xiao, Lisha Chen, Pin-Yu Chen, and Tianyi Chen.Sharp-MAML: Sharpness-aware model-agnostic meta learning.In Proc. Int. Conf. Machine Learning, pages 10–32. PMLR, 2022.
Agarwala and Dauphin (2023)	Atish Agarwala and Yann Dauphin.SAM operates far from home: eigenvalue regularization as a dynamical phenomenon.In Proc. Int. Conf. Machine Learning, pages 152–168. PMLR, 2023.
Ahn et al. (2023)	Kwangjun Ahn, Sébastien Bubeck, Sinho Chewi, Yin Tat Lee, Felipe Suarez, and Yi Zhang.Learning threshold neurons via edge of stability.In Proc. Adv. Neural Info. Processing Systems, volume 36, 2023.
Andriushchenko and Flammarion (2022)	Maksym Andriushchenko and Nicolas Flammarion.Towards understanding sharpness-aware minimization.In Proc. Int. Conf. Machine Learning, pages 639–668. PMLR, 2022.
Arora et al. (2018)	Sanjeev Arora, Nadav Cohen, and Elad Hazan.On the optimization of deep networks: Implicit acceleration by overparameterization.In Proc. Int. Conf. Machine Learning, pages 244–253. PMLR, 2018.
Arora et al. (2019a)	Sanjeev Arora, Nadav Cohen, Noah Golowich, and Wei Hu.A convergence analysis of gradient descent for deep linear neural networks.In Proc. Int. Conf. Learning Represention, 2019a.
Arora et al. (2019b)	Sanjeev Arora, Nadav Cohen, Wei Hu, and Yuping Luo.Implicit regularization in deep matrix factorization.In Proc. Adv. Neural Info. Processing Systems, volume 32, 2019b.
Arora et al. (2019c)	Sanjeev Arora, Simon Du, Wei Hu, Zhiyuan Li, and Ruosong Wang.Fine-grained analysis of optimization and generalization for overparameterized two-layer neural networks.In Proc. Int. Conf. Machine Learning, pages 322–332. PMLR, 2019c.
Arora et al. (2022)	Sanjeev Arora, Zhiyuan Li, and Abhishek Panigrahi.Understanding gradient descent on the edge of stability in deep learning.In Proc. Int. Conf. Machine Learning, pages 948–1024. PMLR, 2022.
Bahri et al. (2022)	Dara Bahri, Hossein Mobahi, and Yi Tay.Sharpness-aware minimization improves language model generalization.In Proc. Conf. Assoc. Comput. Linguist. Meet., pages 7360–7371, 2022.
Barrett and Dherin (2021)	David Barrett and Benoit Dherin.Implicit gradient regularization.In Proc. Int. Conf. Learning Represention, 2021.
Bartlett et al. (2018)	Peter Bartlett, Dave Helmbold, and Philip Long.Gradient descent with identity initialization efficiently learns positive definite linear transformations by deep residual networks.In Proc. Int. Conf. Machine Learning, pages 521–530. PMLR, 2018.
Bartlett et al. (2023)	Peter Bartlett, Philip Long, and Olivier Bousquet.The dynamics of sharpness-aware minimization: Bouncing across ravines and drifting towards wide minima.J. Mach. Learn. Res., 24(316):1–36, 2023.
Bottou et al. (2018)	Léon Bottou, Frank E Curtis, and Jorge Nocedal.Optimization methods for large-scale machine learning.SIAM Review, 60(2):223–311, 2018.
Bowman et al. (2015)	Samuel Bowman, Gabor Angeli, Christopher Potts, and Christopher D Manning.A large annotated corpus for learning natural language inference.In Proc. Conf. Empir. Methods Nat. Lang. Process., pages 632–642, 2015.
Cer et al. (2017)	Daniel Cer, Mona Diab, Eneko Agirre, Iñigo Lopez-Gazpio, and Lucia Specia.SemEval-2017 task 1: Semantic textual similarity-multilingual and cross-lingual focused evaluation.In Proc. Int. Workshop Semant. Eval., pages 1–14. ACL, 2017.
Chaudhari et al. (2017)	Pratik Chaudhari, Anna Choromanska, Stefano Soatto, Yann LeCun, Carlo Baldassi, Christian Borgs, Jennifer Chayes, Levent Sagun, and Riccardo Zecchina.Entropy-SGD: Biasing gradient descent into wide valleys.In Proc. Int. Conf. Learning Represention, 2017.
Chen et al. (2022)	Xiangning Chen, Cho-Jui Hsieh, and Boqing Gong.When vision transformers outperform ResNets without pre-training or strong data augmentations.In Proc. Int. Conf. Learning Represention, 2022.
Chen et al. (2024)	Yukang Chen, Shengju Qian, Haotian Tang, Xin Lai, Zhijian Liu, Song Han, and Jiaya Jia.Long-LoRA: Efficient fine-tuning of long-context large language models.In Proc. Int. Conf. Learning Represention, 2024.
Chen et al. (2023)	Zixiang Chen, Junkai Zhang, Yiwen Kou, Xiangning Chen, Cho-Jui Hsieh, and Quanquan Gu.Why does sharpness-aware minimization generalize better than SGD?In Proc. Adv. Neural Info. Processing Systems, volume 36, 2023.
Dai et al. (2023)	Yan Dai, Kwangjun Ahn, and Suvrit Sra.The crucial role of normalization in sharpness-aware minimization.In Proc. Adv. Neural Info. Processing Systems, volume 36, 2023.
De Marneffe et al. (2019)	Marie-Catherine De Marneffe, Mandy Simons, and Judith Tonhauser.The CommitmentBank: Investigating projection in naturally occurring discourse.Proc. Sinn und Bedeutung, 23(2):107–124, 2019.
De Sa et al. (2015)	Christopher De Sa, Christopher Re, and Kunle Olukotun.Global convergence of stochastic gradient descent for some non-convex matrix problems.In Proc. Int. Conf. Machine Learning, pages 2332–2341. PMLR, 2015.
Dettmers et al. (2023)	Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer.QLoRA: Efficient finetuning of quantized LLMs.In Proc. Adv. Neural Info. Processing Systems, volume 36, 2023.
Dinh et al. (2017)	Laurent Dinh, Razvan Pascanu, Samy Bengio, and Yoshua Bengio.Sharp minima can generalize for deep nets.In Proc. Int. Conf. Machine Learning, pages 1019–1028. PMLR, 2017.
Dolan and Brockett (2005)	Bill Dolan and Chris Brockett.Automatically constructing a corpus of sentential paraphrases.In Proc. Int. Workshop Paraphrasing, 2005.
Du et al. (2022a)	Jiawei Du, Hanshu Yan, Jiashi Feng, Joey Tianyi Zhou, Liangli Zhen, Rick Siow Mong Goh, and Vincent Y. F. Tan.Efficient sharpness-aware minimization for improved training of neural networks.In Proc. Int. Conf. Learning Represention, 2022a.
Du et al. (2022b)	Jiawei Du, Daquan Zhou, Jiashi Feng, Vincent Y. F. Tan, and Joey Tianyi Zhou.Sharpness-aware training for free.In Proc. Adv. Neural Info. Processing Systems, 2022b.
Du et al. (2018)	Simon S Du, Wei Hu, and Jason D Lee.Algorithmic regularization in learning deep homogeneous models: Layers are automatically balanced.In Proc. Adv. Neural Info. Processing Systems, volume 31, 2018.
Dziugaite and Roy (2017)	Gintare Karolina Dziugaite and Daniel M. Roy.Computing nonvacuous generalization bounds for deep (stochastic) neural networks with many more parameters than training data.In Proc. Conf. Uncerntainty in Artif. Intel., 2017.
Foret et al. (2021)	Pierre Foret, Ariel Kleiner, Hossein Mobahi, and Behnam Neyshabur.Sharpness-aware minimization for efficiently improving generalization.In Proc. Int. Conf. Learning Represention, 2021.
Gardent et al. (2017)	Claire Gardent, Anastasia Shimorina, Shashi Narayan, and Laura Perez-Beltrachini.The WebNLG challenge: Generating text from RDF data.In Proc. Int. Conf. Nat. Lang. Gener., pages 124–133. ACL, 2017.
Ge et al. (2017)	Rong Ge, Chi Jin, and Yi Zheng.No spurious local minima in nonconvex low rank problems: A unified geometric analysis.In Proc. Int. Conf. Machine Learning, pages 1233–1242. PMLR, 2017.
Gidel et al. (2019)	Gauthier Gidel, Francis Bach, and Simon Lacoste-Julien.Implicit regularization of discrete gradient dynamics in linear neural networks.In Proc. Adv. Neural Info. Processing Systems, volume 32, 2019.
Gonon et al. (2024)	Antoine Gonon, Nicolas Brisebarre, Elisa Riccietti, and Rémi Gribonval.A path-norm toolkit for modern networks: consequences, promises and challenges.In Proc. Int. Conf. Learning Represention, 2024.
Houlsby et al. (2019)	Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin De Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly.Parameter-efficient transfer learning for NLP.In Proc. Int. Conf. Machine Learning, pages 2790–2799. PMLR, 2019.
Hu et al. (2022)	Edward Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen.LoRA: Low-rank adaptation of large language models.In Proc. Int. Conf. Learning Represention, 2022.
(38)	HuggingFace.Gradient accumulation.URL https://huggingface.co/docs/accelerate/en/usage_guides/gradient_accumulation.
Izmailov et al. (2018)	Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry P. Vetrov, and Andrew Gordon Wilson.Averaging weights leads to wider optima and better generalization.In Proc. Conf. Uncerntainty in Artif. Intel., pages 876–885, 2018.
Jastrzębski et al. (2017)	Stanisław Jastrzębski, Zachary Kenton, Devansh Arpit, Nicolas Ballas, Asja Fischer, Yoshua Bengio, and Amos Storkey.Three factors influencing minima in SGD.arXiv:1711.04623, 2017.
Ji and Telgarsky (2019)	Ziwei Ji and Matus Telgarsky.Gradient descent aligns the layers of deep linear networks.In Proc. Int. Conf. Learning Represention, 2019.
Jiang et al. (2023)	Weisen Jiang, Hansi Yang, Yu Zhang, and James Kwok.An adaptive policy to employ sharpness-aware minimization.In Proc. Int. Conf. Learning Represention, 2023.
Jiang et al. (2020)	Yiding Jiang, Behnam Neyshabur, Hossein Mobahi, Dilip Krishnan, and Samy Bengio.Fantastic generalization measures and where to find them.In Proc. Int. Conf. Learning Represention, 2020.
Keskar et al. (2016)	Nitish Shirish Keskar, Dheevatsa Mudigere, Jorge Nocedal, Mikhail Smelyanskiy, and Ping Tak Peter Tang.On large-batch training for deep learning: Generalization gap and sharp minima.In Proc. Int. Conf. Learning Represention, 2016.
Kim et al. (2022)	Minyoung Kim, Da Li, Shell Xu Hu, and Timothy M. Hospedales.Fisher SAM: Information geometry and sharpness aware minimisation.In Proc. Int. Conf. Machine Learning, pages 11148–11161, 2022.
Kopiczko et al. (2024)	Dawid Jan Kopiczko, Tijmen Blankevoort, and Yuki M Asano.VeRA: Vector-based random matrix adaptation.In Proc. Int. Conf. Learning Represention, 2024.
Kwon et al. (2021)	Jungmin Kwon, Jeongseop Kim, Hyunseo Park, and In Kwon Choi.ASAM: Adaptive sharpness-aware minimization for scale-invariant learning of deep neural networks.In Proc. Int. Conf. Machine Learning, pages 5905–5914. PMLR, 2021.
Li and Giannakis (2023)	Bingcong Li and Georgios B Giannakis.Enhancing sharpness-aware optimization through variance suppression.In Proc. Adv. Neural Info. Processing Systems, volume 36, 2023.
Li and Liang (2021)	Xiang Lisa Li and Percy Liang.Prefix-tuning: Optimizing continuous prompts for generation.In Proc. Conf. Assoc. Comput. Linguist. Meet., pages 4582–4597, 2021.
Li et al. (2022)	Zhiyuan Li, Tianhao Wang, and Sanjeev Arora.What happens after SGD reaches zero loss? – A mathematical framework.In Proc. Int. Conf. Learning Represention, 2022.
Liu et al. (2019)	Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov.RoBERTa: A robustly optimized BERT pretraining approach.arXiv preprint arXiv:1907.11692, 2019.
Liu et al. (2022)	Yong Liu, Siqi Mai, Xiangning Chen, Cho-Jui Hsieh, and Yang You.Towards efficient and scalable sharpness-aware minimization.In Proc. Conf. Computer Vision and Pattern Recognition, pages 12350–12360, 2022.
Loshchilov and Hutter (2019)	Ilya Loshchilov and Frank Hutter.Decoupled weight decay regularization.In Proc. Int. Conf. Learning Represention, 2019.
Lyu and Li (2020)	Kaifeng Lyu and Jian Li.Gradient descent maximizes the margin of homogeneous neural networks.In Proc. Int. Conf. Learning Represention, 2020.
Malladi et al. (2023)	Sadhika Malladi, Tianyu Gao, Eshaan Nichani, Alex Damian, Jason D. Lee, Danqi Chen, and Sanjeev Arora.Fine-tuning language models with just forward passes.In Proc. Adv. Neural Info. Processing Systems, volume 36, 2023.
Mi et al. (2022)	Peng Mi, Li Shen, Tianhe Ren, Yiyi Zhou, Xiaoshuai Sun, Rongrong Ji, and Dacheng Tao.Make sharpness-aware minimization stronger: A sparsified perturbation approach.In Proc. Adv. Neural Info. Processing Systems, volume 35, 2022.
Nesterov (2004)	Yurii Nesterov.Introductory lectures on convex optimization: A basic course, volume 87.Springer Science & Business Media, 2004.
Neyshabur et al. (2015)	Behnam Neyshabur, Russ R Salakhutdinov, and Nati Srebro.Path-SGD: Path-normalized optimization in deep neural networks.In Proc. Adv. Neural Info. Processing Systems, volume 28, 2015.
Neyshabur et al. (2017)	Behnam Neyshabur, Srinadh Bhojanapalli, David Mcallester, Nathan Srebro, and Nati Srebro.Exploring generalization in deep learning.In Proc. Adv. Neural Info. Processing Systems, volume 30, pages 5947–5956, 2017.
Neyshabur et al. (2018)	Behnam Neyshabur, Srinadh Bhojanapalli, and Nathan Srebro.A PAC-bayesian approach to spectrally-normalized margin bounds for neural networks.In Proc. Int. Conf. Learning Represention, 2018.
Rajpurkar et al. (2016)	Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang.SQuAD: 100,000+ questions for machine comprehension of text.In Proc. Conf. Empir. Methods Nat. Lang. Process., pages 2383–2392, 2016.
Rajpurkar et al. (2018)	Pranav Rajpurkar, Robin Jia, and Percy Liang.Know what you don’t know: Unanswerable questions for SQuAD.In Proc. Conf. Assoc. Comput. Linguist. Meet., pages 784–789, 2018.
Roemmele et al. (2011)	Melissa Roemmele, Cosmin Adrian Bejan, and Andrew S Gordon.Choice of plausible alternatives: An evaluation of commonsense causal reasoning.In AAAI Spring Symposium Series, 2011.
Sheen et al. (2024)	Heejune Sheen, Siyu Chen, Tianhao Wang, and Harrison H Zhou.Implicit regularization of gradient flow on one-layer softmax attention.arXiv preprint arXiv:2403.08699, 2024.
Sherborne et al. (2023)	Tom Sherborne, Naomi Saphra, Pradeep Dasigi, and Hao Peng.TRAM: Bridging trust regions and sharpness aware minimization.In Proc. Int. Conf. Learning Represention, 2023.
Si and Yun (2023)	Dongkuk Si and Chulhee Yun.Practical sharpness-aware minimization cannot converge all the way to optima.In Proc. Adv. Neural Info. Processing Systems, volume 36, 2023.
Singh and Hofmann (2024)	Sidak Pal Singh and Thomas Hofmann.Closed form of the hessian spectrum for some neural networks.In High-dimensional Learning Dynamics 2024: The Emergence of Structure and Reasoning, 2024.
Socher et al. (2013)	Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D Manning, Andrew Y Ng, and Christopher Potts.Recursive deep models for semantic compositionality over a sentiment treebank.In Proc. Conf. Empir. Methods Nat. Lang. Process., pages 1631–1642, 2013.
Tahmasebi et al. (2024)	Behrooz Tahmasebi, Ashkan Soleymani, Dara Bahri, Stefanie Jegelka, and Patrick Jaillet.A universal class of sharpness-aware minimization algorithms.arXiv preprint arXiv:2406.03682, 2024.
Tu et al. (2016)	Stephen Tu, Ross Boczar, Max Simchowitz, Mahdi Soltanolkotabi, and Ben Recht.Low-rank solutions of linear matrix equations via procrustes flow.In Proc. Int. Conf. Machine Learning, pages 964–973. PMLR, 2016.
Vaswani et al. (2017)	Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin.Attention is all you need.In Proc. Adv. Neural Info. Processing Systems, volume 30, 2017.
Voorhees and Tice (2000)	Ellen M Voorhees and Dawn M Tice.Building a question answering test collection.In Proc. Annu. Int. ACM SIGIR Conf. Res. Dev. Inf. Retr., pages 200–207, 2000.
Wang et al. (2019a)	Alex Wang, Yada Pruksachatkun, Nikita Nangia, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel Bowman.SuperGLUE: A stickier benchmark for general-purpose language understanding systems.In Proc. Adv. Neural Info. Processing Systems, volume 32, 2019a.
Wang et al. (2019b)	Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R Bowman.GLUE: A multi-task benchmark and analysis platform for natural language understanding.In Proc. Int. Conf. Learning Represention, 2019b.
Wang et al. (2023)	Pengfei Wang, Zhaoxiang Zhang, Zhen Lei, and Lei Zhang.Sharpness-aware gradient matching for domain generalization.In Proc. Conf. Computer Vision and Pattern Recognition, pages 3769–3778, 2023.
Wang and Mao (2022)	Ziqiao Wang and Yongyi Mao.On the generalization of models trained with SGD: Information-theoretic bounds and implications.In Proc. Int. Conf. Learning Represention, 2022.
Warstadt et al. (2019)	Alex Warstadt, Amanpreet Singh, and Samuel R Bowman.Neural network acceptability judgments.Trans. Assoc. Comput. Linguist., 7:625–641, 2019.
Wen et al. (2023a)	Kaiyue Wen, Tengyu Ma, and Z hiyuan Li.How does sharpness-aware minimization minimizes sharpness.In Proc. Int. Conf. Learning Represention, 2023a.
Wen et al. (2023b)	Kaiyue Wen, Tengyu Ma, and Zhiyuan Li.Sharpness minimization algorithms do not only minimize sharpness to achieve better generalization.In Proc. Adv. Neural Info. Processing Systems, volume 36, 2023b.
Williams et al. (2018)	Adina Williams, Nikita Nangia, and Samuel R Bowman.A broad-coverage challenge corpus for sentence understanding through inference.In Proc. Conf. North Am. Chapter Assoc. Comput. Linguist., pages 1112–1122, 2018.
Woodworth et al. (2020)	Blake Woodworth, Suriya Gunasekar, Jason D Lee, Edward Moroshko, Pedro Savarese, Itay Golan, Daniel Soudry, and Nathan Srebro.Kernel and rich regimes in overparametrized models.In Proc. Annual Conf. Learning Theory, pages 3635–3673. PMLR, 2020.
Wu et al. (2020)	Dongxian Wu, Shu-Tao Xia, and Yisen Wang.Adversarial weight perturbation helps robust generalization.In Proc. Adv. Neural Info. Processing Systems, volume 33, pages 2958–2969, 2020.
Xia et al. (2024)	Wenhan Xia, Chengwei Qin, and Elad Hazan.Chain of LoRA: Efficient fine-tuning of language models via residual learning.arXiv preprint arXiv:2401.04151, 2024.
Zhang et al. (2023a)	Qingru Zhang, Minshuo Chen, Alexander Bukharin, Pengcheng He, Yu Cheng, Weizhu Chen, and Tuo Zhao.Adaptive budget allocation for parameter-efficient fine-tuning.In Proc. Int. Conf. Learning Represention, 2023a.
Zhang et al. (2023b)	Ruipeng Zhang, Ziqing Fan, Jiangchao Yao, Ya Zhang, and Yanfeng Wang.Domain-inspired sharpness aware minimization under domain shifts.In Proc. Int. Conf. Learning Represention, 2023b.
Zhang et al. (2018)	Sheng Zhang, Xiaodong Liu, Jingjing Liu, Jianfeng Gao, Kevin Duh, and Benjamin Van Durme.ReCoRD: Bridging the gap between human and machine commonsense reading comprehension.arXiv preprint arXiv:1810.12885, 2018.
Zhang et al. (2022)	Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al.OPT: Open pre-trained transformer language models.arXiv preprint arXiv:2205.01068, 2022.
Zhao et al. (2022)	Yang Zhao, Hao Zhang, and Xiuyuan Hu.Penalizing gradient norm for efficiently improving generalization in deep learning.In Proc. Int. Conf. Machine Learning, pages 26982–26992, 2022.
Zhou et al. (2022)	Wenxuan Zhou, Fangyu Liu, Huan Zhang, and Muhao Chen.Sharpness-aware minimization with dynamic reweighting.In Proc. Conf. Empir. Methods Nat. Lang. Process., pages 5686–5699, 2022.
Zhuang et al. (2022)	Juntang Zhuang, Boqing Gong, Liangzhe Yuan, Yin Cui, Hartwig Adam, Nicha Dvornek, Sekhar Tatikonda, James Duncan, and Ting Liu.Surrogate gap minimization improves sharpness-aware training.In Proc. Int. Conf. Learning Represention, 2022.

Supplementary Document for
“Implicit Regularization of Sharpness-Aware Minimization
for Scale-Invariant Problems”

Appendix AMissing Details
A.1Broad Impact

The theories and approaches are applicable across various scenarios. The proposed algorithmic tool simplifies finetuning language models, improves performance of downstream tasks, and consumes less resource compared to SAM. For tasks such as sentiment classification, our approach facilitates real world systems such as recommendation by improving accuracy. However, caution is advised when the downstream tasks of language models involve generation. For these tasks, users should thoroughly review generated content and consider to implement gating methods to ensure safety and trustworthiness.

A.2More on Related Work

Sharpness and generalization. Sharpness is observed to relate with generalization of SGD in deep learning (Keskar et al., 2016). It is found that sharpness varies with the ratio between learning rate and batchsize in SGD (Jastrzębski et al., 2017). Large scale experiments also indicate sharpness-based measures align with generalization in practical scenarios (Jiang et al., 2020; Chen et al., 2022). Theoretical understandings on generalization error using sharpness-related metrics can be found in e.g., (Dziugaite and Roy, 2017; Neyshabur et al., 2017; Wang and Mao, 2022). There is a large body of literature exploring sharpness for improved generalization. Entropy SGD leverages local entropy in search of a flat valley (Chaudhari et al., 2017). A similar approach as SAM is also developed in (Wu et al., 2020) while putting more emphases on adversarial robustness. Stochastic weight averaging is proposed for finding flatter minima in (Izmailov et al., 2018). It is shown later in (Wen et al., 2023b) that the interplay between sharpness and generalization subtly depends on data distributions and model architectures, and there are unveiled reasons beyond sharpness for the benefit of SAM.

SAM variants. Although SAM is successful in various deep learning tasks, it can be improved further by leveraging local geometry in a fine-grained manner. For example, results in (Zhao et al., 2022; Barrett and Dherin, 2021) link SAM with gradient norm penalization. Zhuang et al. (2022) optimize sharpness gap and training loss jointly. A more accurate manner to solve inner maximization in SAM is developed in (Li and Giannakis, 2023). SAM and its variants are also widely applied to domain generalization problems; see e.g., (Zhang et al., 2023b; Wang et al., 2023).

Other perspectives for SAM. The convergence of SAM is comprehensively studied in (Si and Yun, 2023). Agarwala and Dauphin (2023) focus on the edge-of-stability-like behavior of unnormalized SAM on quadratic problems. Dai et al. (2023) argue that the normalization in SAM, i.e., line 5 of Alg. 1, is critical. Sharpness measure is generalized to any functions of Hessian in (Tahmasebi et al., 2024). However, even the generalized sharpness cannot provide implicit regularization for simple functions such as
ℎ
⁢
(
𝑥
,
𝑦
)
=
𝑥
⁢
𝑦
, because the Hessian is the same for all
(
𝑥
,
𝑦
)
. In addition, when Hessian is negative definite, some of the generalized sharpness measures (e.g., determinate of Hessian) may not be necessarily meaningful.

Implicit regularization. The regularization effect can come from optimization algorithms rather than directly from the regularizer in objective functions. This type of the behavior is termed as implicit regularization or implicit bias of the optimizer. The implicit regularization of (S)GD is studied from multiple perspectives, such as margin (Ji and Telgarsky, 2019; Lyu and Li, 2020), kernel (Arora et al., 2019c), and Hessian (Li et al., 2022; Arora et al., 2022). Initialization can also determine the implicit regularization (Woodworth et al., 2020). Most of these works explore the overparametrization regime.

LoRA and parameter-efficient finetuning. LoRA (Hu et al., 2022), our major numerical benchmark, is an instance of parameter-efficient finetuning (PEFT) approaches. PEFT reduces the resource requirement for large language models on various downstream tasks, at the cost of possible accuracy drops on test performance. The latter, together with the transfer learning setup jointly motivate the adoption of SAM. Other commonly adopted PEFT methods include, e.g., adapters (Houlsby et al., 2019) and prefix tuning (Li and Liang, 2021). There are also various efforts to further improve LoRA via adaptivity (Zhang et al., 2023a), chaining (Xia et al., 2024), aggressive parameter saving (Kopiczko et al., 2024), low-bit training (Dettmers et al., 2023), and modifications for long-sequences (Chen et al., 2024). Most of these efforts are orthogonal to BAR proposed in this work.

A.3Additional Applications of Scale-Invariant Problems in Deep Learning

Attention in transformers. Attention is one of the backbones of modern neural networks (Vaswani et al., 2017). Given the input
𝐃
, attention can be written as


min
𝐐
,
𝐊
,
𝐕
⁡
softmax
⁢
(
1
𝛼
⁢
𝐃𝐐𝐊
⊤
⁢
𝐃
⊤
)
⁢
𝐃𝐕

(9)

where
{
𝐐
,
𝐊
,
𝐕
}
 are query, key, and value matrices to be optimized. This is a scale-invariant problem because scaling
{
𝐐
,
𝐊
}
 does not modify the objective function. Considering the number of variables, the optimization of
{
𝐐
,
𝐊
}
 is considered as OP.

Two-layer linear neural networks. This problem is a simplified version of two-layer ReLU neural nets, and its objective can be defined as


𝑓
⁢
(
𝐖
1
,
𝐖
2
)
=
1
2
⁢
𝔼
(
𝐚
,
𝐛
)
⁢
[
‖
𝐖
1
⁢
𝐖
2
⁢
𝐚
−
𝐛
‖
2
]
.

(10)

This is usually adopted as an example for overparametrization, and can be extended to deeper linear neural networks; see e.g., (Arora et al., 2019a). Moreover, it is known that the optimization for such problem is quite challenging, and GD can fail to converge if
𝐖
1
 and
𝐖
2
 are not initialized with balancedness (Arora et al., 2019a). An extension of (10) is two-layer ReLU networks, which are widely adopted in theoretical frameworks to understand the behavior of neural networks. ReLU networks are scale-invariant, but only when the scaling factor is positive.

Other examples. For ResNets, two-variable scale-invariant submodules also include affine BatchNorm and the subsequent convolutional layer. For transformers, scale-invariant submodules besides attention include LayerNorm and its subsequent linear layer.

A.4SAM Pays More Attention to Difficult Examples

Testing example for NOP. The problem presented below is adopted in Fig. 1 (a) and Fig. 2 for visualization of SAM’s behavior on NOP. We consider a special case of problem (1a), where the goal is to fit (rank-1) matrices by minimizing


𝑓
𝑛
⁢
(
𝐱
,
𝐲
)
=
𝔼
𝜉
⁢
[
‖
𝐱𝐲
⊤
−
(
𝐀
+
𝛼
⁢
𝐍
𝜉
)
‖
2
]

(11)

where
𝐀
∈
ℝ
3
×
3
:=
diag
⁢
[
0.5
,
0
,
0
]
 and
𝐍
𝜉
∈
ℝ
3
×
3
 denote the ground truth and Gaussian noise, respectively; and
𝛼
 controls the SNR. Here we choose
𝐍
𝜉
:=
diag
⁢
[
1.0
,
0.8
,
0.5
]
⁢
𝐔
𝜉
, where entries of
𝐔
𝜉
 are unit Gaussian random variables.

In our simulation of Fig. 1 (a), we set the step size to be
𝜂
=
10
−
4
 and the total number of iterations as
𝑇
=
10
5
 for both SGD and SAM. Parameter
𝜌
 is chosen as
0.1
 for SAM. For both algorithms, initialization is
𝐱
0
=
[
0.2
,
−
0.1
,
0.3
]
⊤
 and
𝐲
0
=
−
3
⁢
𝐱
0
. Note that we choose a small step size to mimic the settings of our theorems.

Testing example for OP. The problem presented below is adopted in Fig. 1 (b) for visualization of SAM on OP. A special case of problem (1b) is considered with objective function


𝑓
𝑜
⁢
(
𝐱
,
𝐲
)
=
𝔼
𝜉
⁢
[
‖
𝐱
⊤
⁢
𝐲
−
(
𝑎
+
𝛼
⁢
𝑛
𝜉
)
‖
2
]

(12)

where
𝑎
∈
ℝ
 and
𝑛
𝜉
∈
ℝ
 denote the ground truth and Gaussian noise, respectively. We choose
𝑎
=
0.5
 and
𝑛
𝜉
 as a unit Gaussian random variable. Here,
𝛼
 controls the SNR of this problem.

In our simulation of Fig. 1 (b), we set
𝜂
=
10
−
4
 and
𝑇
=
10
5
 for both SGD and SAM. Parameter
𝜌
 is set as
0.2
 for SAM. For both algorithms, initialization is
𝐱
0
=
[
0.2
,
−
0.1
,
0.3
]
⊤
 and
𝐲
0
=
−
3
⁢
𝐱
0
.

A.5Scale-Invariance in OP

Scale-invariance also bothers OP in the same fashion as it burdens NOP. For completeness, the scale-invariance of OP can be verified by


𝑓
𝑜
⁢
(
𝐱
⊤
⁢
𝐲
)
=
𝑓
𝑜
⁢
(
(
𝛼
⁢
𝐱
)
⊤
⁢
(
1
𝛼
⁢
𝐲
)
)
,
∀
𝛼
≠
0
.

(13)

An optimizer has to determine
𝛼
 for OP despite it does not influence objective value. Hence, scaling is redundant for OP.

Similar to NOP, the (stochastic) gradient of OP is not scale-invariant. In particular, given a minibatch of data
ℳ
, the stochastic gradient for OP (1b) can be written as


𝐠
𝐱
=
1
|
ℳ
|
⁢
[
∑
𝜉
∈
ℳ
(
𝑓
𝑜
𝜉
)
′
⁢
(
𝐱
⊤
⁢
𝐲
)
]
⁢
𝐲
,
𝐠
𝐲
=
1
|
ℳ
|
⁢
[
∑
𝜉
∈
ℳ
(
𝑓
𝑜
𝜉
)
′
⁢
(
𝐱
⊤
⁢
𝐲
)
]
⁢
𝐱
.

(14)

Consequently, being balance also brings optimization benefits for OP as discussed previously in Section 2.2 .

A.6BAR in Detail
Figure 4:The value of
𝑓
⁢
(
𝑥
,
𝑦
)
. Once SGD reaches the dotted line, i.e., the hard constraint
|
𝑥
|
=
|
𝑦
|
, it can only converge to a saddle point
(
0
,
0
)
.

BAR is inspired jointly from the balancedness-promoting regularizer
|
‖
𝐱
𝑡
‖
2
−
‖
𝐲
𝑡
‖
2
|
 and the dynamics of SAM on both NOP and OP. The implementation of BAR is similar as weight decay in AdamW (Loshchilov and Hutter, 2019).

Here we use nBAR as an example. If ignoring
𝒜
𝑡
 in Theorem 2, it can be seen that
ℬ
𝑡
 for NOP decreases whenever
‖
𝐠
𝐱
𝑡
‖
<
‖
𝐠
𝐲
𝑡
‖
. In other words, the balancedness of SAM is driven by the difference between the gradient norms at
𝐱
𝑡
 and
𝐲
𝑡
. nBAR mimics this and triggers balancedness when stochastic gradients
𝐠
𝐱
𝑡
 and
𝐠
𝐲
𝑡
 are not balanced; see Alg. 2.

Finally, we illustrate more on the reasons for employing regularization in OP rather than posing
‖
𝐱
𝑡
‖
=
‖
𝐲
𝑡
‖
 as a hard constraint or initializing in a balanced manner, i.e.,
‖
𝐱
0
‖
=
‖
𝐲
0
‖
. First, it is quite clear that
‖
𝐱
‖
=
‖
𝐲
‖
 is a nonconvex set and how to project on such a set is still debatable. Second, the ‘symmetry’ associated with the scale-invariant problems does not always favor this constraint. For the purpose of graphical illustration, we consider a
2
-dimensional example
𝑓
⁢
(
𝑥
,
𝑦
)
=
30000
⁢
(
𝑥
⁢
𝑦
−
0.005
)
2
. It is quite clear that the objective is symmetric regarding the line
𝑥
=
−
𝑦
, which satisfies
|
𝑥
|
=
|
𝑦
|
; see Fig. 4. However, it is not hard to see that SGD can never leave
𝑥
=
−
𝑦
 once it reaches this line via a hard constraint or initialized on this line. In other words, directly adding
‖
𝐱
‖
=
‖
𝐲
‖
 as a constraint can trap the algorithm at saddle points. This symmetric pattern is even more complicated in high dimension, i.e., symmetry over multiple lines or hyperplanes. Hence, one should be extremely careful about this hard constraint, and regularization is a safer and more practical choice.

Appendix BMissing Proofs for NOP
B.1Proof of Theorem 1
Proof.

For notational convenience, we let
𝐆
𝑡
:=
∇
𝑓
𝑡
⁢
(
𝐱
𝑡
⁢
𝐲
𝑡
⊤
)
. Then, we have that


d
⁢
‖
𝐱
𝑡
‖
2
d
⁢
𝑡
=
2
⁢
𝐱
𝑡
⊤
⁢
d
⁢
𝐱
𝑡
d
⁢
𝑡
=
−
2
⁢
𝐱
𝑡
⊤
⁢
𝐠
𝐱
𝑡
=
−
2
⁢
𝐱
𝑡
⊤
⁢
𝐆
𝑡
⁢
𝐲
𝑡
.


Similarly, we have that


d
⁢
‖
𝐲
𝑡
‖
2
d
⁢
𝑡
=
2
⁢
𝐲
𝑡
⊤
⁢
d
⁢
𝐲
𝑡
d
⁢
𝑡
=
−
2
⁢
𝐲
𝑡
⊤
⁢
𝐠
𝐲
𝑡
=
−
2
⁢
𝐲
𝑡
⊤
⁢
𝐆
𝑡
⊤
⁢
𝐱
𝑡
.


Combining these two inequalities, we arrive at


d
⁢
‖
𝐱
𝑡
‖
2
d
⁢
𝑡
−
d
⁢
‖
𝐲
𝑡
‖
2
d
⁢
𝑡
=
0
.


The proof is thus completed. ∎

B.2Extension to Stochastic Normalized Gradient Descent (SNGD)

Next, we extend Theorem 1 to SNGD, whose updates can be written as


𝐱
𝑡
+
1
=
𝐱
𝑡
−
𝜂
⁢
𝐠
𝐱
𝑡
‖
𝐠
𝐱
𝑡
‖
2
+
‖
𝐠
𝐲
𝑡
‖
2
,

𝐲
𝑡
+
1
=
𝐲
𝑡
−
𝜂
⁢
𝐠
𝐲
𝑡
‖
𝐠
𝐱
𝑡
‖
2
+
‖
𝐠
𝐲
𝑡
‖
2
.

(15)
Theorem 4.

When applying SNGD (15) on NOP problem (1a), the limiting flow with
𝜂
→
0
 guarantees that
‖
𝐱
𝑡
‖
2
−
‖
𝐲
𝑡
‖
2
=
‖
𝐱
0
‖
2
−
‖
𝐲
0
‖
2
 for all
𝑡
>
0
. In other words,
d
⁢
ℬ
𝑡
d
⁢
𝑡
=
0
 holds.

Proof.

For notational convenience, we let
𝐆
𝑡
:=
∇
𝑓
𝑡
⁢
(
𝐱
𝑡
⁢
𝐲
𝑡
⊤
)
. Then, we have that


d
⁢
‖
𝐱
𝑡
‖
2
d
⁢
𝑡
=
2
⁢
𝐱
𝑡
⊤
⁢
d
⁢
𝐱
𝑡
d
⁢
𝑡
=
−
2
⁢
𝐱
𝑡
⊤
⁢
𝐠
𝐱
𝑡
‖
𝐠
𝐱
𝑡
‖
2
+
‖
𝐠
𝐲
𝑡
‖
2
=
−
2
⁢
𝐱
𝑡
⊤
⁢
𝐆
𝑡
⁢
𝐲
𝑡
‖
𝐠
𝐱
𝑡
‖
2
+
‖
𝐠
𝐲
𝑡
‖
2
.


Similarly, we have that


d
⁢
‖
𝐲
𝑡
‖
2
d
⁢
𝑡
=
2
⁢
𝐲
𝑡
⊤
⁢
d
⁢
𝐲
𝑡
d
⁢
𝑡
=
−
2
⁢
𝐲
𝑡
⊤
⁢
𝐠
𝐲
𝑡
‖
𝐠
𝐱
𝑡
‖
2
+
‖
𝐠
𝐲
𝑡
‖
2
=
−
2
⁢
𝐲
𝑡
⊤
⁢
𝐆
𝑡
⊤
⁢
𝐱
𝑡
‖
𝐠
𝐱
𝑡
‖
2
+
‖
𝐠
𝐲
𝑡
‖
2
.


Combining these two inequalities, we arrive at


d
⁢
‖
𝐱
𝑡
‖
2
d
⁢
𝑡
−
d
⁢
‖
𝐲
𝑡
‖
2
d
⁢
𝑡
=
0
.


The proof is thus completed. ∎

B.3Proof of Theorem 2
Proof.

Denote
𝐆
𝑡
=
∇
𝑓
𝑡
⁢
(
𝐱
𝑡
⁢
𝐲
𝑡
⊤
)
 and
𝐆
~
𝑡
=
∇
𝑓
𝑡
⁢
(
𝐱
~
𝑡
⁢
𝐲
~
𝑡
⊤
)
 for notational convenience. Following SAM updates in (4) and setting
𝜂
→
0
, we have that


d
⁢
𝐱
𝑡
d
⁢
𝑡
=
−
𝐆
~
𝑡
⁢
(
𝐲
𝑡
+
𝜌
⁢
𝑢
𝑡
⁢
𝐆
𝑡
⊤
⁢
𝐱
𝑡
)
,
d
⁢
𝐲
𝑡
d
⁢
𝑡
=
−
𝐆
~
𝑡
⊤
⁢
(
𝐱
𝑡
+
𝜌
⁢
𝑢
𝑡
⁢
𝐆
𝑡
⁢
𝐲
𝑡
)
.


This gives that



1
2
⁢
d
⁢
(
‖
𝐱
𝑡
‖
2
−
‖
𝐲
𝑡
‖
2
)
d
⁢
𝑡

=
𝜌
⁢
𝑢
𝑡
⁢
[
𝐲
𝑡
⊤
⁢
𝐆
~
𝑡
⊤
⁢
𝐆
𝑡
⁢
𝐲
𝑡
−
𝐱
𝑡
⊤
⁢
𝐆
~
𝑡
⁢
𝐆
𝑡
⊤
⁢
𝐱
𝑡
]

(16a)


=
𝜌
⁢
𝑢
𝑡
⁢
[
‖
𝐠
𝐱
𝑡
‖
2
−
‖
𝐠
𝐲
𝑡
‖
2
]
+
𝜌
⁢
𝑢
𝑡
⁢
[
𝐲
𝑡
⊤
⁢
(
𝐆
~
𝑡
−
𝐆
𝑡
)
⊤
⁢
𝐠
𝐱
𝑡
−
𝐱
𝑡
⊤
⁢
(
𝐆
~
𝑡
−
𝐆
𝑡
)
⁢
𝐠
𝐲
𝑡
]
⏟
:=
𝒜
𝑡
.

(16b)

The second term in (16b) is
𝒜
𝑡
 in Theorem 2. Next, we give upper bound on
|
𝒜
𝑡
|
. Using Assumption 1, we have that


‖
𝐆
~
𝑡
−
𝐆
𝑡
‖

≤
𝐿
⁢
‖
𝐱
~
𝑡
⁢
𝐲
~
𝑡
⊤
−
𝐱
𝑡
⁢
𝐲
𝑡
⊤
‖


=
𝐿
⁢
‖
𝜌
⁢
𝑢
𝑡
⁢
(
𝐱
𝑡
⁢
𝐠
𝐲
𝑡
⊤
+
𝐠
𝐱
𝑡
⁢
𝐲
𝑡
⊤
)
+
𝜌
2
⁢
𝑢
𝑡
2
⁢
𝐠
𝐱
𝑡
⁢
𝐠
𝐲
𝑡
⊤
‖


≤
(
𝑎
)
𝐿
⁢
𝜌
⁢
‖
𝐱
𝑡
⁢
𝐠
𝐲
𝑡
⊤
+
𝐠
𝐱
𝑡
⁢
𝐲
𝑡
⊤
‖
‖
𝐠
𝐱
𝑡
‖
2
+
‖
𝐠
𝐲
𝑡
‖
2
+
𝐿
⁢
𝜌
2
⁢
‖
𝐠
𝐱
𝑡
⁢
𝐠
𝐲
𝑡
⊤
‖
‖
𝐠
𝐱
𝑡
‖
2
+
‖
𝐠
𝐲
𝑡
‖
2


≤
(
𝑏
)
𝐿
⁢
𝜌
⁢
(
‖
𝐱
𝑡
‖
+
‖
𝐲
𝑡
‖
)
+
𝐿
⁢
𝜌
2
2
=
𝒪
⁢
(
𝐿
⁢
𝜌
)


where (a) uses the definition of
𝑢
𝑡
; (b) follows from
‖
𝐚𝐛
⊤
‖
=
‖
𝐚
‖
⁢
‖
𝐛
‖
 and the finite convergence assumption. To bound
𝒜
𝑡
, we also have


𝜌
⁢
𝑢
𝑡
⁢
|
𝐲
𝑡
⊤
⁢
(
𝐆
~
𝑡
−
𝐆
𝑡
)
⊤
⁢
𝐠
𝐱
𝑡
|

=
𝜌
⁢
|
𝐲
𝑡
⊤
⁢
(
𝐆
~
𝑡
−
𝐆
𝑡
)
⊤
⁢
𝐠
𝐱
𝑡
|
‖
𝐠
𝐱
𝑡
‖
2
+
‖
𝐠
𝐲
𝑡
‖
2
≤
𝜌
⁢
|
𝐲
𝑡
⊤
⁢
(
𝐆
~
𝑡
−
𝐆
𝑡
)
⊤
⁢
𝐠
𝐱
𝑡
|
‖
𝐠
𝐱
𝑡
‖


≤
𝜌
⁢
‖
𝐆
~
𝑡
−
𝐆
𝑡
‖
⁢
‖
𝐲
𝑡
‖
=
𝒪
⁢
(
𝐿
⁢
𝜌
2
)

(17)

where the last line also uses the finite convergence. We can bound
𝜌
⁢
𝑢
𝑡
⁢
|
𝐱
𝑡
⊤
⁢
(
𝐆
~
𝑡
−
𝐆
𝑡
)
⁢
𝐠
𝐲
𝑡
|
=
𝒪
⁢
(
𝜌
2
⁢
𝐿
)
 in a similar manner. Combining (B.3) with (16b) gives the bound on
|
𝒜
𝑡
|
=
𝒪
⁢
(
𝜌
2
⁢
𝐿
)
 . ∎

B.4Proof of Corollary 1

Here, we prove the formal version of Corollary 1.

Corollary 2.

Suppose that
‖
𝐠
𝐱
𝑡
‖
>
0
 and
‖
𝐠
𝐲
𝑡
‖
>
0
 and
𝜌
→
0
, then there exists
ℬ
¯
𝑡
 such that the magnitude of
ℬ
𝑡
 shrinks whenever
|
ℬ
𝑡
|
>
ℬ
¯
𝑡
.

Proof.

Without loss of generality, we suppose that
ℬ
𝑡
>
0
, i.e.,
‖
𝐱
𝑡
‖
>
‖
𝐲
𝑡
‖
>
0
. Let
𝐱
¯
𝑡
 and
𝐲
¯
𝑡
 be the scaled version of
𝐱
𝑡
 and
𝐲
𝑡
 such that
‖
𝐱
¯
𝑡
‖
=
‖
𝐲
¯
𝑡
‖
 and
𝐱
¯
𝑡
⁢
𝐲
¯
𝑡
⊤
=
𝐱
𝑡
⁢
𝐲
𝑡
⊤
 are satisfied. This suggests that
𝐱
𝑡
=
𝛼
𝑡
⁢
𝐱
¯
𝑡
 and
𝐲
𝑡
=
𝐲
¯
𝑡
/
𝛼
𝑡
, where
𝛼
𝑡
=
‖
𝐱
𝑡
‖
/
‖
𝐲
𝑡
‖
. Next, we show that whenever
ℬ
𝑡
 is large enough, we have that


d
⁢
ℬ
𝑡
d
⁢
𝑡
=
𝜌
⁢
‖
𝐠
𝐱
𝑡
‖
2
−
‖
𝐠
𝐲
𝑡
‖
2
‖
𝐠
𝐱
𝑡
‖
2
+
‖
𝐠
𝐲
𝑡
‖
2
+
𝒪
⁢
(
𝜌
2
⁢
𝐿
)
<
0
.

(18)

Since
𝜌
→
0
, we only need to show that for some small
𝜖
=
𝒪
⁢
(
𝜌
⁢
𝐿
)
≥
0
,


‖
𝐠
𝐱
𝑡
‖
2
−
‖
𝐠
𝐲
𝑡
‖
2
‖
𝐠
𝐱
𝑡
‖
2
+
‖
𝐠
𝐲
𝑡
‖
2
<
−
𝜖
.

(19)

By the definition of
𝐠
𝐱
𝑡
,
𝐠
𝐲
𝑡
 and
𝐱
¯
𝑡
,
𝐲
¯
𝑡
, we have that (19) can be rewritten as


𝛼
𝑡
2
⁢
‖
𝐆
𝑡
⊤
⁢
𝐱
¯
𝑡
‖
2
−
‖
𝐆
𝑡
⁢
𝐲
¯
𝑡
‖
2
/
𝛼
𝑡
2
𝛼
𝑡
2
⁢
‖
𝐆
𝑡
⊤
⁢
𝐱
¯
𝑡
‖
2
+
‖
𝐆
𝑡
⁢
𝐲
¯
𝑡
‖
2
/
𝛼
𝑡
2
>
𝜖
.

(20)

Note that the function
ℎ
⁢
(
𝑧
)
:=
(
𝑎
⁢
𝑧
−
𝑏
/
𝑧
)
/
𝑎
⁢
𝑧
+
𝑏
/
𝑧
 is monotonically increasing in
𝑧
 when
𝑎
,
𝑏
>
0
 and
𝑧
>
0
 as
ℎ
′
⁢
(
𝑧
)
=
(
𝑎
2
⁢
𝑧
+
6
⁢
𝑎
⁢
𝑏
/
𝑧
+
𝑏
2
/
𝑧
3
)
/
(
2
⁢
(
𝑎
⁢
𝑧
+
𝑏
/
𝑧
)
3
/
2
)
>
0
. This implies that
ℎ
⁢
(
𝑧
)
>
0
 when
𝑧
>
𝑏
/
𝑎
, and thus the condition in (20) can be satisfied for
𝜖
=
𝒪
⁢
(
𝜌
⁢
𝐿
)
→
0
 when
𝛼
𝑡
2
>
𝛼
¯
2
, where
𝛼
¯
2
:=
‖
𝐆
𝑡
⁢
𝐲
¯
𝑡
‖
/
‖
𝐆
𝑡
⊤
⁢
𝐱
¯
𝑡
‖
. This condition on
𝛼
𝑡
 is equivalent to


ℬ
𝑡

=
1
2
⁢
(
‖
𝐱
𝑡
‖
2
−
‖
𝐲
𝑡
‖
2
)


=
1
2
⁢
(
‖
𝛼
𝑡
⁢
𝐱
¯
𝑡
‖
2
−
‖
𝐲
¯
𝑡
/
𝛼
𝑡
‖
2
)


>
1
2
⁢
(
‖
𝛼
¯
⁢
𝐱
¯
𝑡
‖
2
−
‖
𝐲
¯
𝑡
/
𝛼
¯
‖
2
)
.


Combining everything together, we have that
d
⁢
ℬ
𝑡
d
⁢
𝑡
<
0
 if


ℬ
𝑡
>
ℬ
¯
𝑡
:=
1
2
⁢
(
‖
𝛼
¯
⁢
𝐱
¯
𝑡
‖
2
−
‖
𝐲
¯
𝑡
/
𝛼
¯
‖
2
)
.

(21)

The proof is thus completed. We also note that in the case of
𝜌
>
0
, the same condition as (21) can be derived by obtaining the inverse function of
ℎ
⁢
(
𝑧
)
 evaluated at
𝜖
=
𝒪
⁢
(
𝜌
⁢
𝐿
)
, and the corresponding
𝛼
¯
𝜌
 and
ℬ
¯
𝑡
𝜌
 can be defined similarly. ∎

B.5Extension to LoRA (layer-wise NOP problem)

Let
𝑙
∈
{
1
,
2
,
…
,
𝐷
}
 be the layer index. Denote
𝑓
𝑡
 as the loss function on minibatch
ℳ
𝑡
. To simplify the notation, we also let
𝐆
𝑡
,
𝑙
:=
∇
𝐱
𝑡
,
𝑙
⁢
𝐲
𝑡
,
𝑙
⊤
𝑓
𝑡
⁢
(
{
𝐱
𝑡
,
𝑙
,
𝐲
𝑡
,
𝑙
}
𝑙
)
,
𝐆
~
𝑡
,
𝑙
:=
∇
𝐱
~
𝑡
,
𝑙
⁢
𝐲
~
𝑡
,
𝑙
⊤
𝑓
𝑡
⁢
(
{
𝐱
~
𝑡
,
𝑙
,
𝐲
~
𝑡
,
𝑙
}
𝑙
)
, and
𝑢
𝑡
:=
1
/
∑
𝑙
=
1
𝐷
(
‖
𝐠
𝐱
𝑡
,
𝑙
‖
2
+
‖
𝐠
𝐲
𝑡
,
𝑙
‖
2
)
. The update of SAM for layer
𝑙
 can be written as



𝐱
~
𝑡
,
𝑙
=
𝐱
𝑡
,
𝑙
+
𝜌
⁢
𝑢
𝑡
⁢
𝐆
𝑡
,
𝑙
⁢
𝐲
𝑡
,
𝑙
,

𝐲
~
𝑡
,
𝑙
=
𝐲
𝑡
,
𝑙
+
𝜌
⁢
𝑢
𝑡
⁢
𝐆
𝑡
,
𝑙
⊤
⁢
𝐱
𝑡
,
𝑙

(22a)


𝐠
𝐱
~
𝑡
,
𝑙
=
𝐆
~
𝑡
,
𝑙
⁢
𝐲
~
𝑡
,
𝑙
,

𝐠
𝐲
~
𝑡
,
𝑙
=
𝐆
~
𝑡
,
𝑙
⊤
⁢
𝐱
~
𝑡
,
𝑙

(22b)


𝐱
𝑡
+
1
,
𝑙
=
𝐱
𝑡
,
𝑙
−
𝜂
⁢
𝐠
𝐱
~
𝑡
,
𝑙
,

𝐲
𝑡
+
1
,
𝑙
=
𝐲
𝑡
,
𝑙
−
𝜂
⁢
𝐠
𝐲
~
𝑡
,
𝑙
.

(22c)

Refined assumption for LoRA. Direct translating Assumption 1 to our multi-layer setting gives


‖
∇
𝑓
𝑡
⁢
(
{
𝐱
𝑙
⁢
𝐲
𝑙
⊤
}
𝑙
)
−
∇
𝑓
𝑡
⁢
(
{
𝐚
𝑙
⁢
𝐛
𝑙
⊤
}
𝑙
)
‖
2
≤
𝐿
2
⁢
∑
𝑙
=
1
𝐷
‖
𝐱
𝑙
⁢
𝐲
𝑙
⊤
−
𝐚
𝑙
⁢
𝐛
𝑙
⊤
‖
2
.

(23)

However, the above assumption is loose, and our proof only needs block-wise smoothness, i.e.,


‖
∇
𝑙
𝑓
𝑡
⁢
(
𝐱
𝑙
⁢
𝐲
𝑙
⊤
)
−
∇
𝑙
𝑓
𝑡
⁢
(
𝐚
𝑙
⁢
𝐛
𝑙
⊤
)
‖
2
≤
𝐿
^
2
⁢
‖
𝐱
𝑙
⁢
𝐲
𝑙
⊤
−
𝐚
𝑙
⁢
𝐛
𝑙
⊤
‖
2
,
∀
𝑙

(24)

where
∇
𝑙
 refers to the gradient on
𝐱
𝑙
⁢
𝐲
𝑙
⊤
. It can be seen that
𝐷
⁢
𝐿
^
≥
𝐿
, but one can assume that
𝐷
⁢
𝐿
^
≈
𝐿
 for intuitive understandings.

Theorem 5.

Suppose that block smoothness assumption in (24) holds. Consider the limiting flow of SAM in (22) with
𝜂
→
0
 and a sufficiently small
𝜌
. Let
ℬ
𝑡
,
𝑙
:=
1
2
⁢
(
‖
𝐱
𝑡
,
𝑙
‖
2
−
‖
𝐲
𝑡
,
𝑙
‖
2
)
 and
ℬ
𝑡
=
∑
𝑙
=
1
𝐷
ℬ
𝑡
,
𝑙
. For some
|
𝒜
𝑡
|
=
𝒪
⁢
(
𝜌
2
⁢
𝐿
^
)
, SAM guarantees that


d
⁢
ℬ
𝑡
d
⁢
𝑡
=
𝜌
⁢
∑
𝑙
=
1
𝐷
‖
𝐠
𝐱
𝑡
,
𝑙
‖
2
−
∑
𝑙
=
1
𝐷
‖
𝐠
𝐲
𝑡
,
𝑙
‖
2
∑
𝑙
=
1
𝐷
‖
𝐠
𝐱
𝑡
,
𝑙
‖
2
+
∑
𝑙
=
1
𝐷
‖
𝐠
𝐲
𝑡
,
𝑙
‖
2
+
𝒜
𝑡
.

(25)

Furthermore, for per layer balancedness it satisfies that for some
|
𝒜
𝑡
,
𝑙
|
=
𝒪
⁢
(
𝜌
2
⁢
𝐿
^
)
.


d
⁢
ℬ
𝑡
,
𝑙
d
⁢
𝑡
=
𝜌
⁢
‖
𝐠
𝐱
𝑡
,
𝑙
‖
2
−
‖
𝐠
𝐲
𝑡
,
𝑙
‖
2
∑
𝑙
=
1
𝐷
‖
𝐠
𝐱
𝑡
,
𝑙
‖
2
+
∑
𝑙
=
1
𝐷
‖
𝐠
𝐲
𝑡
,
𝑙
‖
2
+
𝒜
𝑡
,
𝑖
.

(26)

Understanding Theorem 5.
𝒜
𝑡
,
𝑖
 and
𝒜
𝑡
 are at the same order because of the possible unbalancedness among gradient norms for different layers. Comparing per layer balancedness
ℬ
𝑡
,
𝑙
 with Theorem 2, it can be roughly estimate that the regularization power is
𝒪
⁢
(
𝐷
)
 times smaller in
ℬ
𝑡
,
𝑙
. This estimation comes from
𝐿
^
≈
𝐿
/
𝐷
, and the first term is also
𝒪
⁢
(
𝐷
)
 smaller than the same term in Theorem 2. In other words, the regularization on balancedness can be reduced by
𝒪
⁢
(
𝐷
)
 times in LoRA in the worst case, and the worst case comes from gradient unbalancedness among layers.

Proof.

Following (22) and setting
𝜂
→
0
, we have that


d
⁢
𝐱
𝑡
,
𝑙
d
⁢
𝑡
=
−
𝐆
~
𝑡
,
𝑙
⁢
(
𝐲
𝑡
,
𝑙
+
𝜌
⁢
𝑢
𝑡
⁢
𝐆
𝑡
,
𝑙
⊤
⁢
𝐱
𝑡
,
𝑙
)
,
d
⁢
𝐲
𝑡
,
𝑙
d
⁢
𝑡
=
−
𝐆
~
𝑡
,
𝑙
⊤
⁢
(
𝐱
𝑡
,
𝑙
+
𝜌
⁢
𝑢
𝑡
⁢
𝐆
𝑡
,
𝑙
⁢
𝐲
𝑡
,
𝑙
)
.


This gives that



d
⁢
ℬ
𝑡
,
𝑙
d
⁢
𝑡

=
𝜌
⁢
𝑢
𝑡
⁢
[
𝐲
𝑡
,
𝑙
⊤
⁢
𝐆
~
𝑡
,
𝑙
⊤
⁢
𝐆
𝑡
,
𝑙
⁢
𝐲
𝑡
,
𝑙
−
𝐱
𝑡
,
𝑙
⊤
⁢
𝐆
~
𝑡
,
𝑙
⁢
𝐆
𝑡
,
𝑙
⊤
⁢
𝐱
𝑡
,
𝑙
]

(27a)


=
𝜌
⁢
𝑢
𝑡
⁢
[
‖
𝐠
𝐱
𝑡
,
𝑙
‖
2
−
‖
𝐠
𝐲
𝑡
,
𝑙
‖
2
]
+
𝜌
⁢
𝑢
𝑡
⁢
[
𝐲
𝑡
,
𝑙
⊤
⁢
(
𝐆
~
𝑡
,
𝑙
−
𝐆
𝑡
,
𝑙
)
⊤
⁢
𝐠
𝐱
𝑡
,
𝑙
−
𝐱
𝑡
,
𝑙
⊤
⁢
(
𝐆
~
𝑡
,
𝑙
−
𝐆
𝑡
,
𝑙
)
⁢
𝐠
𝐲
𝑡
,
𝑙
]
⏟
:=
𝒜
𝑡
,
𝑙
.

(27b)

Proof for (25). Let
𝒜
𝑡
:=
∑
𝑙
𝒜
𝑡
,
𝑙
. To start with, we have that


‖
𝐆
~
𝑡
,
𝑙
−
𝐆
𝑡
,
𝑙
‖

≤
𝐿
^
⁢
‖
𝐱
~
𝑡
,
𝑙
⁢
𝐲
~
𝑡
,
𝑙
⊤
−
𝐱
𝑡
,
𝑙
⁢
𝐲
𝑡
,
𝑙
⊤
‖


=
𝐿
^
⁢
‖
𝜌
⁢
𝑢
𝑡
⁢
(
𝐱
𝑡
,
𝑙
⁢
𝐠
𝐲
𝑡
,
𝑙
⊤
+
𝐠
𝐱
𝑡
,
𝑙
⁢
𝐲
𝑡
,
𝑙
⊤
)
+
𝜌
2
⁢
𝑢
𝑡
2
⁢
𝐠
𝐱
𝑡
,
𝑙
⁢
𝐠
𝐲
𝑡
,
𝑙
⊤
‖


Next, based on finite convergence assumption, we have that


𝜌
⁢
𝑢
𝑡
⁢
∑
𝑙
=
1
𝐷
|
𝐲
𝑡
,
𝑙
⊤
⁢
(
𝐆
~
𝑡
,
𝑙
−
𝐆
𝑡
,
𝑙
)
⊤
⁢
𝐠
𝐱
𝑡
,
𝑙
|

(28)


≤
∑
𝑙
=
1
𝐷
𝒪
⁢
(
𝜌
⁢
𝑢
𝑡
⁢
‖
𝐆
~
𝑡
,
𝑙
−
𝐆
𝑡
,
𝑙
‖
⋅
‖
𝐠
𝐱
𝑡
,
𝑙
‖
)


≤
(
𝑎
)
∑
𝑙
=
1
𝐷
𝒪
⁢
(
𝜌
2
⁢
𝑢
𝑡
2
⁢
𝐿
^
⁢
‖
𝐱
𝑡
,
𝑙
⁢
𝐠
𝐲
𝑡
,
𝑙
⊤
+
𝐠
𝐱
𝑡
,
𝑙
⁢
𝐲
𝑡
,
𝑙
⊤
‖
⋅
‖
𝐠
𝐱
𝑡
,
𝑙
‖
)


≤
(
𝑏
)
∑
𝑙
=
1
𝐷
𝒪
⁢
(
𝜌
2
⁢
𝑢
𝑡
2
⁢
𝐿
^
⁢
(
‖
𝐠
𝐲
𝑡
,
𝑙
‖
+
‖
𝐠
𝐱
𝑡
,
𝑙
‖
)
⋅
‖
𝐠
𝐱
𝑡
,
𝑙
‖
)


=
𝜌
2
⁢
𝐿
^
⋅
𝒪
⁢
(
∑
𝑙
=
1
𝐷
‖
𝐠
𝐱
𝑡
,
𝑙
‖
2
∑
𝑙
=
1
𝐷
(
‖
𝐠
𝐱
𝑡
,
𝑙
‖
2
+
‖
𝐠
𝐲
𝑡
,
𝑙
‖
2
)
+
∑
𝑙
=
1
𝐷
‖
𝐠
𝐱
𝑡
,
𝑙
‖
⁢
‖
𝐠
𝐲
𝑡
,
𝑙
‖
∑
𝑙
=
1
𝐷
(
‖
𝐠
𝐱
𝑡
,
𝑙
‖
2
+
‖
𝐠
𝐲
𝑡
,
𝑙
‖
2
)
)


=
𝒪
⁢
(
𝜌
2
⁢
𝐿
^
)


where in (a) we use the fact that
𝜌
 is chosen small; (b) uses finite convergence assumption and
‖
𝐚𝐛
⊤
‖
=
‖
𝐚
‖
⁢
‖
𝐛
‖
. Using similar arguments, we can bound
𝒜
𝑡
=
𝒪
⁢
(
𝜌
2
⁢
𝐿
^
)
.

Proof for (26). Next, we give upper bound on
|
𝒜
𝑡
,
𝑙
|
. Using similar argument as (28), we have that


𝜌
⁢
𝑢
𝑡
⁢
|
𝐲
𝑡
,
𝑙
⊤
⁢
(
𝐆
~
𝑡
,
𝑙
−
𝐆
𝑡
,
𝑙
)
⊤
⁢
𝐠
𝐱
𝑡
,
𝑙
|

(29)


≤
𝒪
⁢
(
𝜌
2
⁢
𝑢
𝑡
2
⁢
𝐿
^
⁢
(
‖
𝐠
𝐲
𝑡
,
𝑙
‖
+
‖
𝐠
𝐱
𝑡
,
𝑙
‖
)
⋅
‖
𝐠
𝐱
𝑡
,
𝑙
‖
)


=
𝜌
2
⁢
𝐿
^
⋅
𝒪
⁢
(
‖
𝐠
𝐱
𝑡
,
𝑙
‖
2
∑
𝑙
=
1
𝐷
(
‖
𝐠
𝐱
𝑡
,
𝑙
‖
2
+
‖
𝐠
𝐲
𝑡
,
𝑙
‖
2
)
+
‖
𝐠
𝐱
𝑡
,
𝑙
‖
⁢
‖
𝐠
𝐲
𝑡
,
𝑙
‖
∑
𝑙
=
1
𝐷
(
‖
𝐠
𝐱
𝑡
,
𝑙
‖
2
+
‖
𝐠
𝐲
𝑡
,
𝑙
‖
2
)
)
.

(30)

Using (29), we have that


|
𝒜
𝑡
,
𝑙
|

≤
𝜌
2
⁢
𝐿
^
⋅
𝒪
⁢
(
‖
𝐠
𝐱
𝑡
,
𝑙
‖
2
+
‖
𝐠
𝐲
𝑡
,
𝑙
‖
2
∑
𝑙
=
1
𝐷
(
‖
𝐠
𝐱
𝑡
,
𝑙
‖
2
+
‖
𝐠
𝐲
𝑡
,
𝑙
‖
2
)
+
‖
𝐠
𝐱
𝑡
,
𝑙
‖
⁢
‖
𝐠
𝐲
𝑡
,
𝑙
‖
∑
𝑙
=
1
𝐷
(
‖
𝐠
𝐱
𝑡
,
𝑙
‖
2
+
‖
𝐠
𝐲
𝑡
,
𝑙
‖
2
)
)


=
𝒪
⁢
(
𝜌
2
⁢
𝐿
^
)
.


The proof is is thus completed. ∎

Appendix CMissing Proofs for OP
C.1Unbalancedness of SGD in OP
Theorem 6.

Applied SGD or SNGD on problem (1b), both of them ensure that
‖
𝐱
𝑡
‖
2
−
‖
𝐲
𝑡
‖
2
=
‖
𝐱
0
‖
2
−
‖
𝐲
0
‖
2
 for all
𝑡
>
0
. In other words,
ℬ
𝑡
 keeps unchanged.

Proof.

We consider SGD and NSGD separately.

SGD. It is straightforward to see that


d
⁢
‖
𝐱
𝑡
‖
2
d
⁢
𝑡
=
−
2
⁢
𝑓
𝑡
′
⁢
(
𝐱
𝑡
⊤
⁢
𝐲
𝑡
)
⁢
𝐱
𝑡
⊤
⁢
𝐲
𝑡
=
d
⁢
‖
𝐲
𝑡
‖
2
d
⁢
𝑡
.


This completes the proof of SGD.

NSGD. The gradient update of NSGD is


d
⁢
𝐱
𝑡
d
⁢
𝑡
=
−
𝐠
𝐱
𝑡
‖
𝐠
𝐱
𝑡
‖
2
+
‖
𝐠
𝐲
𝑡
‖
2
,
d
⁢
𝐲
𝑡
d
⁢
𝑡
=
−
𝐠
𝐲
𝑡
‖
𝐠
𝐱
𝑡
‖
2
+
‖
𝐠
𝐲
𝑡
‖
2
.

(31)

Then we have that for NSGD,


d
⁢
‖
𝐱
𝑡
‖
2
d
⁢
𝑡
=
−
2
⁢
𝑓
𝑡
′
⁢
(
𝐱
𝑡
⊤
⁢
𝐲
𝑡
)
⁢
𝐱
𝑡
⊤
⁢
𝐲
𝑡
‖
𝐠
𝐱
𝑡
‖
2
+
‖
𝐠
𝐲
𝑡
‖
2
=
d
⁢
‖
𝐲
𝑡
‖
2
d
⁢
𝑡
.


This gives the result for SNGD. ∎

C.2Proof of Theorem 3

To prove this theorem, we first focus on the dynamic of SAM.

Lemma 2.

Suppose that Assumption 1 holds. Consider the limiting flow of SAM in (7) with
𝜂
→
0
. Let
ℬ
𝑡
:=
1
2
⁢
(
‖
𝐱
𝑡
‖
2
−
‖
𝐲
𝑡
‖
2
)
 and
𝜌
 be small. Then, for some
|
𝒜
𝑡
|
=
𝒪
⁢
(
𝜌
2
⁢
𝐿
⁢
|
ℬ
𝑡
|
)
, SAM guarantees


d
⁢
ℬ
𝑡
d
⁢
𝑡
=
−
2
⁢
𝜌
⁢
|
𝑓
𝑡
′
⁢
(
𝐱
𝑡
⊤
⁢
𝐲
𝑡
)
|
‖
𝐱
𝑡
‖
2
+
‖
𝐲
𝑡
‖
2
⁢
ℬ
𝑡
+
𝒜
𝑡
.

(32)
Proof.

For notational convenience, we write
𝑓
𝑡
′
:=
𝑓
𝑡
′
⁢
(
𝐱
𝑡
⊤
⁢
𝐲
𝑡
)
 and
𝑓
~
𝑡
′
:=
𝑓
𝑡
′
⁢
(
𝐱
~
𝑡
⊤
⁢
𝐲
~
𝑡
)
. Using similar arguments as Theorem 2, we have that


1
2
⁢
d
d
⁢
𝑡
⁢
(
‖
𝐱
𝑡
‖
2
−
‖
𝐲
𝑡
‖
2
)

=
−
𝜌
⁢
𝑢
𝑡
⁢
𝑓
~
𝑡
′
⋅
(
‖
𝐱
𝑡
‖
2
−
‖
𝐲
𝑡
‖
2
)

(33)


=
−
𝜌
⁢
sgn
⁢
(
𝑓
𝑡
′
)
⁢
𝑓
~
𝑡
′
‖
𝐱
𝑡
‖
2
+
‖
𝐲
𝑡
‖
2
⋅
(
‖
𝐱
𝑡
‖
2
−
‖
𝐲
𝑡
‖
2
)


=
−
𝜌
⁢
|
𝑓
𝑡
′
|
‖
𝐱
𝑡
‖
2
+
‖
𝐲
𝑡
‖
2
⋅
(
‖
𝐱
𝑡
‖
2
−
‖
𝐲
𝑡
‖
2
)


+
𝜌
⁢
sgn
⁢
(
𝑓
𝑡
′
)
⁢
(
𝑓
𝑡
′
−
𝑓
~
𝑡
′
)
‖
𝐱
𝑡
‖
2
+
‖
𝐲
𝑡
‖
2
⋅
(
‖
𝐱
𝑡
‖
2
−
‖
𝐲
𝑡
‖
2
)
⏟
:=
𝒜
𝑡
.


Next we bound
|
𝒜
𝑡
|
. To start with, we have that


|
𝐱
~
𝑡
⊤
⁢
𝐲
~
𝑡
−
𝐱
𝑡
⊤
⁢
𝐲
𝑡
|

=
|
𝜌
2
⁢
𝑢
𝑡
2
⁢
𝐱
𝑡
⊤
⁢
𝐲
𝑡
+
𝜌
⁢
𝑢
𝑡
⁢
‖
𝐱
𝑡
‖
2
+
𝜌
⁢
𝑢
𝑡
⁢
‖
𝐲
𝑡
‖
2
|

(34)


≤
𝜌
2
⁢
|
𝐱
𝑡
⊤
⁢
𝐲
𝑡
|
‖
𝐱
𝑡
‖
2
+
‖
𝐲
𝑡
‖
2
+
𝜌
⁢
‖
𝐱
𝑡
‖
2
+
‖
𝐲
𝑡
‖
2


≤
𝜌
2
2
+
𝜌
⁢
‖
𝐱
𝑡
‖
2
+
‖
𝐲
𝑡
‖
2
.


Using Assumption 1 and (34), we arrive at


|
𝑓
𝑡
′
−
𝑓
𝑡
′
~
|
≤
𝐿
⁢
|
𝐱
~
𝑡
⊤
⁢
𝐲
~
𝑡
−
𝐱
𝑡
⊤
⁢
𝐲
𝑡
|
=
𝒪
⁢
(
𝜌
⁢
𝐿
⁢
‖
𝐱
𝑡
‖
2
+
‖
𝐲
𝑡
‖
2
)
.

(35)

Hence, we arrive at


|
𝒜
𝑡
|
≤
𝜌
⁢
|
𝑓
𝑡
′
−
𝑓
𝑡
′
~
|
⁢
|
‖
𝐱
𝑡
‖
2
−
‖
𝐲
𝑡
‖
2
‖
𝐱
𝑡
‖
2
+
‖
𝐲
𝑡
‖
2
|
=
𝒪
⁢
(
𝜌
2
⁢
𝐿
⁢
|
ℬ
𝑡
|
)
.


The proof is thus completed. ∎

Next, the proof of Theorem 3 is provided.

Proof.

Lemma 2 has already indicated the concentration of
ℬ
𝑡
 towards
0
, if the magnitude of the first term is larger than
|
𝒜
𝑡
|
. To see this, notice that we can lower bound
2
⁢
|
ℬ
𝑡
|
/
‖
𝐱
𝑡
‖
2
+
‖
𝐲
𝑡
‖
2
 by


|
‖
𝐱
𝑡
‖
2
−
‖
𝐲
𝑡
‖
2
‖
𝐱
𝑡
‖
2
+
‖
𝐲
𝑡
‖
2
|
=
|
(
‖
𝐱
𝑡
‖
+
‖
𝐲
𝑡
‖
)
⁢
(
‖
𝐱
𝑡
‖
−
‖
𝐲
𝑡
‖
)
|
‖
𝐱
𝑡
‖
2
+
‖
𝐲
𝑡
‖
2
|
|
≥
|
‖
𝐱
𝑡
‖
−
‖
𝐲
𝑡
‖
|
=
𝒞
𝑡
.

(36)

Hence, long as
𝜌
⁢
|
𝑓
𝑡
′
⁢
(
𝐱
𝑡
⊤
⁢
𝐲
𝑡
)
|
⋅
𝒞
𝑡
>
𝒪
⁢
(
𝜌
2
⁢
𝐿
⁢
|
ℬ
𝑡
|
)
, we have the first term dominating the dynamic of SAM, leading to contraction of
ℬ
𝑡
. This completes the proof to the first part.

Next we prove the second part, which is the lower- and upper- bound on
ℬ
𝑡
. The lower bound can be seen from (36). For the upper bound, we have


|
‖
𝐱
𝑡
‖
2
−
‖
𝐲
𝑡
‖
2
‖
𝐱
𝑡
‖
2
+
‖
𝐲
𝑡
‖
2
|
≤
|
‖
𝐱
𝑡
‖
2
−
‖
𝐲
𝑡
‖
2
|
‖
𝐱
𝑡
‖
2
−
‖
𝐲
𝑡
‖
2
|
|
=
2
⁢
|
ℬ
𝑡
|
.

(37)

Plugging (37) into (33) finishes the proof. ∎

C.3
𝑚
-sharpness for OP

𝑚
-sharpness is a variant of SAM that is empirically observed to improve generalization, and it is especially useful for distributed training on multiple GPUs (Foret et al., 2021). However, the reason behind the improved performance is not fully understood. (Andriushchenko and Flammarion, 2022) show that
𝑚
-sharpness is more sparse-promoting for diagonal linear neural networks minimized via a quadratic loss. However, diagonal linear networks are not scale-invariant.

For consistent notation with (7), we use
𝑓
𝑡
⁢
(
⋅
)
 to denote the loss function on minibatch
ℳ
𝑡
. In
𝑚
-sharpness, the minibatch
ℳ
𝑡
 is divided into
𝑚
 disjoint subsets. Without loss of generality, we also assume that the minibatch is evenly divided. We denote the loss function on each subset as
𝑓
𝑡
,
𝑖
,
𝑖
∈
{
1
,
2
,
…
,
𝑚
}
. Note that we have
1
𝑚
⁢
∑
𝑖
=
1
𝑚
𝑓
𝑡
,
𝑖
=
𝑓
𝑡
. With these definitions, the update of
𝑚
-sharpness can be written as



𝐱
~
𝑡
,
𝑖
=
𝐱
𝑡
+
𝜌
⁢
𝑢
𝑡
,
𝑖
⁢
𝐲
𝑡
,

𝐲
~
𝑡
,
𝑖
=
𝐲
𝑡
+
𝜌
⁢
𝑢
𝑡
,
𝑖
⁢
𝐱
𝑡

(38a)


𝐠
𝐱
~
𝑡
,
𝑖
𝑖
=
𝑓
𝑡
,
𝑖
′
⁢
(
𝐱
~
𝑡
,
𝑖
⊤
⁢
𝐲
~
𝑡
,
𝑖
)
⁢
𝐲
~
𝑡
,
𝑖
,

𝐠
𝐲
~
𝑡
,
𝑖
𝑖
=
𝑓
𝑡
,
𝑖
′
⁢
(
𝐱
~
𝑡
,
𝑖
⊤
⁢
𝐲
~
𝑡
,
𝑖
)
⁢
𝐱
~
𝑡
,
𝑖

(38b)


𝐱
𝑡
+
1
=
𝐱
𝑡
−
𝜂
⁢
1
𝑚
⁢
∑
𝑖
=
1
𝑚
𝐠
𝐱
~
𝑡
,
𝑖
𝑖
,

𝐲
𝑡
+
1
=
𝐲
𝑡
−
𝜂
⁢
1
𝑚
⁢
∑
𝑖
=
1
𝑚
𝐠
𝐲
~
𝑡
,
𝑖
𝑖
.

(38c)

where
𝑢
𝑡
,
𝑖
:=
sgn
⁢
(
𝑓
𝑡
,
𝑖
′
⁢
(
𝐱
𝑡
⊤
⁢
𝐲
𝑡
)
)
/
‖
𝐱
𝑡
‖
2
+
‖
𝐲
𝑡
‖
2
. Comparing with the SAM update for OP in (7), the difference is that perturbed gradient is calculated on each
𝑓
𝑡
,
𝑖
. Next, we analyze the dynamic of SAM with
𝑚
-sharpness.

Lemma 3.

Suppose that Assumption 1 holds. Consider the limiting flow of SAM in (38) with
𝜂
→
0
. Let
ℬ
𝑡
:=
1
2
⁢
(
‖
𝐱
𝑡
‖
2
−
‖
𝐲
𝑡
‖
2
)
 and
𝜌
 be small. Then, for some
|
𝒜
𝑡
|
=
𝒪
⁢
(
𝜌
2
⁢
𝐿
)
, SAM guarantees that


d
⁢
ℬ
𝑡
d
⁢
𝑡
=
−
2
⁢
𝜌
𝑚
⁢
∑
𝑖
=
1
𝑚
|
𝑓
𝑡
,
𝑖
′
⁢
(
𝐱
𝑡
⊤
⁢
𝐲
𝑡
)
|
‖
𝐱
𝑡
‖
2
+
‖
𝐲
𝑡
‖
2
⁢
ℬ
𝑡
+
𝒜
𝑡
.

(39)
Proof.

For notational convenience, we write
𝑓
𝑡
,
𝑖
′
:=
𝑓
𝑡
,
𝑖
′
⁢
(
𝐱
𝑡
⊤
⁢
𝐲
𝑡
)
 and
𝑓
~
𝑡
,
𝑖
′
:=
𝑓
𝑡
,
𝑖
′
⁢
(
𝐱
~
𝑡
,
𝑖
⊤
⁢
𝐲
~
𝑡
,
𝑖
)
. Then, we have that


1
2
⁢
d
d
⁢
𝑡
⁢
(
‖
𝐱
𝑡
‖
2
−
‖
𝐲
𝑡
‖
2
)

=
−
𝜌
𝑚
⁢
∑
𝑖
=
1
𝑚
𝑢
𝑡
,
𝑖
⁢
𝑓
~
𝑡
,
𝑖
′
⋅
(
‖
𝐱
𝑡
‖
2
−
‖
𝐲
𝑡
‖
2
)

(40)


=
−
𝜌
𝑚
⁢
∑
𝑖
=
1
𝑚
sgn
⁢
(
𝑓
𝑡
,
𝑖
′
)
⁢
𝑓
~
𝑡
,
𝑖
′
‖
𝐱
𝑡
‖
2
+
‖
𝐲
𝑡
‖
2
⋅
(
‖
𝐱
𝑡
‖
2
−
‖
𝐲
𝑡
‖
2
)


=
−
𝜌
𝑚
⁢
∑
𝑖
=
1
𝑚
|
𝑓
𝑡
,
𝑖
′
|
‖
𝐱
𝑡
‖
2
+
‖
𝐲
𝑡
‖
2
⋅
(
‖
𝐱
𝑡
‖
2
−
‖
𝐲
𝑡
‖
2
)


+
𝜌
𝑚
⁢
∑
𝑖
=
1
𝑚
sgn
⁢
(
𝑓
𝑡
,
𝑖
′
)
⁢
(
𝑓
𝑡
,
𝑖
′
−
𝑓
~
𝑡
,
𝑖
′
)
‖
𝐱
𝑡
‖
2
+
‖
𝐲
𝑡
‖
2
⋅
(
‖
𝐱
𝑡
‖
2
−
‖
𝐲
𝑡
‖
2
)
⏟
:=
𝒜
𝑡
,
𝑖
.


Next, using (34) and Assumption 1, we have


|
𝑓
𝑡
,
𝑖
′
−
𝑓
~
𝑡
,
𝑖
′
|
≤
𝐿
⁢
|
𝐱
~
𝑡
,
𝑖
⊤
⁢
𝐲
~
𝑡
,
𝑖
−
𝐱
𝑡
⊤
⁢
𝐲
𝑡
|
=
𝒪
⁢
(
𝜌
⁢
𝐿
⁢
‖
𝐱
𝑡
‖
2
+
‖
𝐲
𝑡
‖
2
)
.


Hence, we can bound
|
𝒜
𝑡
,
𝑖
|
 as


|
𝒜
𝑡
,
𝑖
|
≤
|
𝑓
𝑡
,
𝑖
′
−
𝑓
~
𝑡
,
𝑖
′
|
⁢
|
‖
𝐱
𝑡
‖
2
−
‖
𝐲
𝑡
‖
2
‖
𝐱
𝑡
‖
2
+
‖
𝐲
𝑡
‖
2
|
=
𝒪
⁢
(
𝜌
⁢
𝐿
⁢
|
ℬ
𝑡
|
)
.


The proof is thus completed by plugging
|
𝒜
𝑡
,
𝑖
|
 into (40). ∎

C.4Extension to Layer-wise OP

We start with the notation. Let
𝑙
∈
{
1
,
2
,
…
,
𝐷
}
 be the layer index. Denote
𝑓
𝑡
 as the loss on minibatch
ℳ
𝑡
. Let
𝑓
𝑡
,
𝑙
′
:=
∇
𝑙
𝑓
𝑡
⁢
(
{
𝐱
𝑡
,
𝑙
⊤
⁢
𝐲
𝑡
,
𝑙
}
𝑙
)
, i.e., the
𝑙
-th entry of gradient (w.r.t. the variable
𝐱
𝑡
,
𝑙
⊤
⁢
𝐲
𝑡
,
𝑙
),
𝑓
~
𝑡
,
𝑙
′
:=
∇
𝑙
𝑓
𝑡
⁢
(
{
𝐱
~
𝑡
,
𝑙
⊤
⁢
𝐲
~
𝑡
,
𝑙
}
𝑙
)
, and
𝑢
𝑡
:=
1
/
∑
𝑙
=
1
𝐷
|
𝑓
𝑡
,
𝑙
′
|
2
⁢
[
‖
𝐱
𝑡
,
𝑙
‖
2
+
‖
𝐲
𝑡
,
𝑙
‖
2
]
. The update of SAM for layer
𝑙
 can be written as



𝐱
~
𝑡
,
𝑙
=
𝐱
𝑡
,
𝑙
+
𝜌
⁢
𝑢
𝑡
⁢
𝑓
𝑡
,
𝑙
′
⁢
𝐲
𝑡
,
𝑙
,

𝐲
~
𝑡
,
𝑙
=
𝐲
𝑡
,
𝑙
+
𝜌
⁢
𝑢
𝑡
⁢
𝑓
𝑡
,
𝑙
′
⁢
𝐱
𝑡
,
𝑙
,

(41a)


𝐠
𝐱
~
𝑡
,
𝑙
=
𝑓
~
𝑡
,
𝑙
′
⁢
𝐲
~
𝑡
,
𝑙
,

𝐠
𝐲
~
𝑡
,
𝑙
=
𝑓
~
𝑡
,
𝑙
′
⁢
𝐱
~
𝑡
,
𝑙

(41b)


𝐱
𝑡
+
1
,
𝑙
=
𝐱
𝑡
,
𝑙
−
𝜂
⁢
𝐠
𝐱
~
𝑡
,
𝑙
,

𝐲
𝑡
+
1
,
𝑙
=
𝐲
𝑡
,
𝑙
−
𝜂
⁢
𝐠
𝐲
~
𝑡
,
𝑙
.

(41c)

Refined assumption for LoRA. Our proof only needs block-wise smoothness, i.e.,


|
∇
𝑙
𝑓
𝑡
⁢
(
𝐱
𝑙
⊤
⁢
𝐲
𝑙
)
−
∇
𝑙
𝑓
𝑡
⁢
(
𝐚
𝑙
⊤
⁢
𝐛
𝑙
)
|
2
≤
𝐿
^
2
⁢
|
𝐱
𝑙
⊤
⁢
𝐲
𝑙
−
𝐚
𝑙
⊤
⁢
𝐛
𝑙
|
2
,
∀
𝑙
,

(42)

where
∇
𝑙
 refers to the gradient on
𝐱
𝑙
⊤
⁢
𝐲
𝑙
. It can be seen that
𝐷
⁢
𝐿
^
≥
𝐿
, but one can assume that
𝐷
⁢
𝐿
^
≈
𝐿
 for more clear intuition.

Theorem 7.

Suppose that block smoothness assumption in (42) holds. Consider the limiting flow of SAM in (41) with
𝜂
→
0
 and a sufficiently small
𝜌
. Let
ℬ
𝑡
,
𝑙
:=
1
2
⁢
(
‖
𝐱
𝑡
,
𝑙
‖
2
−
‖
𝐲
𝑡
,
𝑙
‖
2
)
 and
ℬ
𝑡
max
=
max
𝑙
⁡
|
ℬ
𝑡
,
𝑙
|
. For some
|
𝒜
𝑡
|
=
𝒪
⁢
(
𝜌
2
⁢
𝐿
^
⁢
ℬ
𝑡
max
)
, SAM guarantees that


d
⁢
ℬ
𝑡
d
⁢
𝑡
=
−
𝜌
⁢
∑
𝑙
=
1
𝐷
|
𝑓
𝑡
,
𝑙
′
|
2
⁢
(
‖
𝐱
𝑡
,
𝑙
‖
2
−
‖
𝐲
𝑡
,
𝑙
‖
2
)
∑
𝑙
=
1
𝐷
|
𝑓
𝑡
,
𝑙
′
|
2
⁢
[
‖
𝐱
𝑡
,
𝑙
‖
2
+
‖
𝐲
𝑡
,
𝑙
‖
2
]
+
𝒜
𝑡
.

(43)

Furthermore, for some
|
𝒜
𝑡
,
𝑙
|
=
𝒪
⁢
(
𝜌
2
⁢
𝐿
^
⁢
|
ℬ
𝑡
,
𝑙
|
)
, per layer balancedness satisfies that


d
⁢
ℬ
𝑡
,
𝑙
d
⁢
𝑡
=
−
𝜌
⁢
|
𝑓
𝑡
,
𝑙
′
|
2
⁢
(
‖
𝐱
𝑡
,
𝑙
‖
2
−
‖
𝐲
𝑡
,
𝑙
‖
2
)
∑
𝑙
=
1
𝐷
|
𝑓
𝑡
,
𝑙
′
|
2
⁢
[
‖
𝐱
𝑡
,
𝑙
‖
2
+
‖
𝐲
𝑡
,
𝑙
‖
2
]
+
𝒜
𝑡
,
𝑖
.

(44)
Proof.

Using a similar derivation as before, we have that


1
2
⁢
d
d
⁢
𝑡
⁢
(
‖
𝐱
𝑡
,
𝑙
‖
2
−
‖
𝐲
𝑡
,
𝑙
‖
2
)

=
−
𝜌
⁢
𝑢
𝑡
⁢
|
𝑓
𝑡
,
𝑙
′
|
2
⋅
(
‖
𝐱
𝑡
,
𝑙
‖
2
−
‖
𝐲
𝑡
,
𝑙
‖
2
)


+
𝜌
⁢
𝑢
𝑡
⁢
𝑓
𝑡
,
𝑙
′
⁢
(
𝑓
𝑡
,
𝑙
′
−
𝑓
~
𝑡
,
𝑙
′
)
⋅
(
‖
𝐱
𝑡
,
𝑙
‖
2
−
‖
𝐲
𝑡
,
𝑙
‖
2
)
⏟
:=
𝒜
𝑡
,
𝑙


Next, based on (42), we have that


|
𝑓
𝑡
,
𝑙
′
−
𝑓
~
𝑡
,
𝑙
′
|
≤
𝐿
^
⁢
|
𝐱
~
𝑡
,
𝑙
⊤
⁢
𝐲
~
𝑡
,
𝑙
−
𝐱
𝑡
,
𝑙
⊤
⁢
𝐲
𝑡
,
𝑙
|
≤
𝜌
⁢
𝐿
^
⁢
𝑢
𝑡
⁢
|
𝑓
𝑡
,
𝑙
′
|
⁢
(
‖
𝐱
𝑡
,
𝑙
‖
2
+
‖
𝐲
𝑡
,
𝑙
‖
2
)
+
𝜌
2
⁢
𝐿
^
⁢
𝑢
𝑡
2
⁢
|
𝑓
𝑡
,
𝑙
′
|
2
⁢
|
𝐱
𝑡
,
𝑙
⊤
⁢
𝐲
𝑡
,
𝑙
|
.


Combining these two equations, and applying similar argument as Theorem 5, it is not difficult to arrive at
|
𝒜
𝑡
,
𝑖
|
=
𝒪
⁢
(
𝜌
2
⁢
𝐿
^
⁢
|
ℬ
𝑡
,
𝑙
|
)
 and
|
𝒜
𝑡
|
=
𝒪
⁢
(
𝜌
2
⁢
𝐿
^
⁢
ℬ
𝑡
max
)
. ∎

C.5Proof of Lemma 1
Proof.

Within
𝒲
∗
, the Hessian on
(
𝐱
,
𝐲
)
 can be calculated as
𝑓
′′
⁢
(
𝐱
⊤
⁢
𝐲
)
⁢
[
𝐲
⊤
,
𝐱
⊤
]
⊤
⁢
[
𝐲
⊤
,
𝐱
⊤
]
. The largest eigenvalue is
𝑓
′′
⁢
(
𝑤
)
⁢
(
‖
𝐱
‖
2
+
‖
𝐲
‖
2
)
. By the AM-GM inequality, it can be seen that the largest eigenvalue is minimized when
‖
𝐱
‖
=
‖
𝐲
‖
, whose balancedness is
0
. ∎

Appendix DMissing Experimental Details

We mainly focus on finetuning LMs with LoRA. This setting naturally includes distributional shift – the finetuning dataset does not usually have the same distribution as the pretraining dataset as validated through zero-shot performance. All experiments are performed on a server with AMD EPYC 7742 CPUs and NVIDIA GeForce RTX 3090 GPUs each with 24GiB memory. All numerical results from Section 6 report test performance (e.g., accuracy, F1 scores, or BLEU scores) and the standard deviation across multiple runs.

D.1Details on Datasets

Our evaluations are carried out on commonly-used datasets in the literature.

GLUE benchmark. GLUE is designed to provide a general-purpose evaluation of language understanding (Wang et al., 2019b). Those adopted in our work include MNLI (inference, (Williams et al., 2018)), SST-2 (sentiment analysis, (Socher et al., 2013)), MRPC (paraphrase detection, (Dolan and Brockett, 2005)), CoLA (linguistic acceptability (Warstadt et al., 2019)), QNLI (inference (Rajpurkar et al., 2018)), QQP3 (question-answering), RTE4 (inference), and STS-B (textual similarity (Cer et al., 2017)). These datasets are released under different permissive licenses.

SuperGLUE benchmark. SuperGLUE (Wang et al., 2019a) is another commonly adopted benchmark for language understanding and is more challenging compared with GLUE. The considered datasets include CB (inference, (De Marneffe et al., 2019)), ReCoRD (multiple-choice question answering (Zhang et al., 2018)), COPA (question answering (Roemmele et al., 2011)). These datasets are released under different permissive licenses.

WebNLG Challenge. This dataset is commonly used for data-to-text evaluation (Gardent et al., 2017). It has 22K examples in total with 14 distinct categories. Among them, 9 are seen during training, and the unseen training data are used to test the generalization performance. The dataset is released under license CC BY-NC-SA 4.0.

Additional datasets. We also use SQuAD (question answering (Rajpurkar et al., 2016)) in our experiments, which is released under license CC BY-SA 4.0. Other datasets include TREC (topic classification (Voorhees and Tice, 2000)) and SNLI (inference (Bowman et al., 2015)). Both of them are licensed under CC BY-SA 4.0.

D.2Details on Language Models

We summarize the adopted language models in our evaluation. All model checkpoints are obtained from HuggingFace.

RoBERTa-large. This is a
355
M parameter model. The model checkpoint5 is released under the MIT license.

OPT-1.3B. The model checkpoint6 is released under a non-commercial license. 7

GPT2-medium. This is a
345
M parameter model. Its checkpoint8 is under MIT License.

D.3Few-shot Learning with RoBERTa and OPT

Experiments on RoBERTa-large. We follow the
𝑘
-shot learning setup in (Malladi et al., 2023) and focus on classification tasks. The training set contains
𝑘
=
512
 samples per class while the test set has
1000
 samples. We also employ prompts for finetuning; where the adopted prompts are the same as those in (Malladi et al., 2023, Table 13). AdamW is adopted as the base optimizer, and hyperparameters are tuned from those in Table 6. Our experiments are averaged over
3
 random trials. The estimated runtime is about 5 minutes per dataset.

Table 6:Hyperparameters used for few-shot learning with RoBERTa-large.
Hyper-parameters	Values
LoRA
𝑟
 (rank)	8
LoRA
𝛼
	16
# iterations	1000
batchsize	16
learning rate	1
×
10
−
4
, 3
×
10
−
4
, 5
×
10
−
4


𝜌
 for SAM	0.05, 0.1, 0.2

𝜇
0
 for BAR	0.5, 1.0, 2.0
scheduler for BAR	linear, cosine

The per-iteration runtime on the SST-5 dataset of BAR, SAM, and the baseline optimizer are compared in Table 7. It can be seen that SAM is much more slower than the baseline approach, and BAR reduces 74% additional runtime of SAM, while achieving comparable accuracy. We believe that this runtime saving can be even larger with additional engineering efforts such as kernel fusion, which we leave for future work. This validates the computational efficiency of BAR.

Table 7:Per-iteration runtime for finetuning RoBERTa-large on SST5.
SST5	baseline	SAM	BAR
time (s)	0.105	0.265	0.146

Experiments on OPT. For OPT-1.3B, we consider tasks from the SuperGLUE benchmark covering classification and multiple-choice. We also consider generation tasks on SQuAD. Following (Malladi et al., 2023), we randomly sample
1000
 data for training and the other
1000
 for testing. AdamW is adopted as base optimizer. The hyperparameters adopted are searched over values in Table 8. Estimated runtime is less than or around 10 minutes, depending on the dataset.

If we directly apply FP16 training with SAM, underflow can happen if one does not take care of the gradient scaling on the two gradients calculated per iteration. This means that SAM is not flexible enough to be integrated with the codebase for large scale training, as FP16 is the default choice for finetuning LMs. We employ FP32 to bypass the issue with SAM. Consequently, the training speed is significantly slowed down; see a summary in Table 9. It further demonstrates the effectiveness of BAR for large scale-training.

Overall, the results for few-shot learning indicate that given limited data, BAR can effectively improve generalization using significantly reduced computational resources relative to SAM.

Table 8:Hyperparameters used for few-shot learning with OPT-1.3B.
Hyper-parameters	Values
LoRA
𝑟
 (rank)	8
LoRA
𝛼
	16
# iterations	1000
batchsize	2, 4, 8
learning rate	1
×
10
−
5
, 1
×
10
−
4
, 5
×
10
−
4


𝜌
 for SAM	0.05, 0.1, 0.2

𝜇
0
 for BAR	0.2, 0.5, 1.0, 2.0
scheduler for BAR	linear, cosine
Table 9:Per-iteration runtime for finetuning OPT-1.3B on RTE.
RTE	baseline	SAM	BAR
precision	FP16	FP32	FP16
time (s)	0.1671	0.708	0.1731
D.4Finetuning with RoBERTa-large
Table 10:Experiments on finetuning RoBERTa (355M). Results marked with
†
 are taken from (Hu et al., 2022), and those with
∗
 refer to Adapter
P
 in (Hu et al., 2022).
RoBERTa	# para	SST2	STS-B	RTE	QQP	QNLI	MRPC	MNLI	CoLA	avg
FT† 	355M	96.4	92.4	86.6	92.2	94.7	90.9	90.2	68.0	88.9
Adapter∗ 	0.8M	96.6	91.9	80.1	91.7	94.8	89.7	-	67.8	-
LoRA	0.8M	95.8	92.4	88.2	91.4	94.7	89.6	90.6	64.8	88.4
LoRA-oBAR	0.8M	96.0	92.6	88.7	91.6	94.8	90.3	90.6	65.1	88.7
LoRA-nBAR	0.8M	96.0	92.6	89.2	91.6	94.7	90.3	90.8	65.6	88.9

Our implementation is inspired from (Hu et al., 2022)9, which is under MIT License. The hyperparameters are chosen the same as provided in its GitHub Repo. AdamW is adopted as the base optimizer. However, we employ single GPU rather than multiple ones and use gradient accumulation rather than parallelism due to memory constraint. We also note that there could be failure cases for LoRA using certain seed, e.g., SST-2 with seed 1 and MNLI with seed 2. These cases are ignored when comparing. We consider the GLUE benchmark and report the mismatched accuracy for MNLI, Matthew’s correlation for CoLA, Pearson correlation for STS-B, and accuracy for other datasets. Larger values indicate better results for all datasets. For LoRA, we employ
𝑟
=
8
 and
𝛼
=
16
. Experiments are conducted over three random trials for all datasets, with the exception of QQP, for which only two trials are performed due to its large size. The results of final test performance can be found in Table 10. Estimated runtime varies for different datasets from 2 to 15 hours, except for QQP which takes 3 days on our device.

For the hyperparameters of oBAR and nBAR,
𝜇
0
 is typically chosen from
{
0.2
,
0.5
,
1.0
}
; however, for QQP, a value of
0.05
 is used. The scheduler is chosen from linear and constant. We also observe that for datasets such as COLA and RTE, setting weight decay as
0
 works best for BAR.

D.5GPT2 medium on WebNLG Challenge

AdamW is adopted as base optimizer. The hyperparameters can be found in Table 11. Our results are obtained from three random trials. Each trial takes roughly 8 hours on our hardware.

Table 11:Hyperparameters used for GPT2.
Hyper-parameters	Values
LoRA
𝑟
 (rank)	4
LoRA
𝛼
	32
# epochs	5
batchsize	8
learning rate	2
×
10
−
4

label Smooth	0.1

𝜇
0
 for BAR	0.1, 0.15, 0.2, 0.25, 0.3
scheduler for BAR	linear, constant
beam size	10
length penalty	0.8
Generated on Fri Oct 18 18:16:53 2024 by LaTeXML
Report Issue
Report Issue for Selection
