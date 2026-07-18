Title: TRAM: Bridging Trust Regions and Sharpness Aware Minimization

URL Source: https://arxiv.org/html/2310.03646

Markdown Content:
Back to arXiv

This is experimental HTML to improve accessibility. We invite you to report rendering errors. 
Use Alt+Y to toggle on accessible reporting links and Alt+Shift+Y to toggle off.
Learn more about this project and help improve conversions.

Why HTML?
Report Issue
Back to Abstract
Download PDF
1Introduction
2Background
3TRAM: Trust Region Aware Minimization
4Results
5Conclusion
6Acknowledgments

HTML conversions sometimes display errors due to content that did not convert correctly from the source. This paper uses the following packages that are not yet supported by the HTML conversion tool. Feedback on these issues are not necessary; they are known and are being worked on.

failed: tikz-dependency
failed: boldline
failed: extdash
failed: pythonhighlight

Authors: achieve the best HTML results from your LaTeX submissions by following these best practices.

License: CC BY 4.0
arXiv:2310.03646v2 [cs.LG] 12 Mar 2024
TRAM: Bridging Trust Regions and Sharpness Aware Minimization
Tom Sherborne
1
 Naomi Saphra
2
  Pradeep Dasigi
3
  Hao Peng
4
⁣
*


1
University of Edinburgh  
2
Kempner Institute, Harvard University  
3
Allen Institute for AI

4
University of Illinois Urbana-Champaign
tom.sherborne@ed.ac.uk, nsaphra@fas.harvard.edu
pradeepd@allenai.org, haopeng@illinois.edu
 This work was done while Tom Sherborne and Hao Peng were at the Allen Institute for AI.
Abstract

Sharpness-aware minimization (SAM) reports improving domain generalization by reducing the loss surface curvature in the parameter space. However, generalization during fine-tuning is often more dependent on the transferability of representations in the function space. Trust-region methods (TR) target this goal by regularizing representation curvature to reduce catastrophic forgetting of pre-trained task-agnostic information while adopting task-specific skills. We consider unifying these strategies for low curvature in both parameter space and function space to improve out-of-domain (OOD) generalization. We propose Trust Region Aware Minimization (TRAM), a SAM algorithm fine-tuning for low parameter sharpness and smooth, informative representations preserving pre-trained structure. TRAM uses a trust region bound to inform the SAM adversarial neighborhood, introducing an awareness of function curvature within optimization for flatter minima. We empirically validate TRAM in vision (cross-dataset adaptation) and text (OOD language modeling, zero-shot cross-lingual transfer) tasks where robust domain transfer and representation generality are critical. TRAM outperforms SAM- and TR-based optimization across all tasks, notably surpassing competing methods for hard transfer between anticorrelated domains. TRAM establishes a novel standard in fine-tuning for domain-generalizable models with minimal additional computation over previous sharpness-aware methods.

1Introduction

Neural model training requires navigating over a complex, non-convex loss surface (Frankle, 2020) towards a good local minimum. Studying loss surfaces and training dynamics has led to many algorithmic advances (Izmailov et al., 2018; Foret et al., 2021; Chen et al., 2023) and regularization schemes (Srivastava et al., 2014; Ioffe & Szegedy, 2015) to improve optimization. One such strategy is to exploit an association between generalization and flat minima, defined by Hochreiter & Schmidhuber (1994) as “region[s] in weight space with the property that each weight vector from that region has [a] similar small error”. Intuitively, a flatter, or less sharp (Keskar et al., 2017), minimum will generalize better, as the loss function will be non-increasing under distribution shift. Recent work has developed a family of sharpness-aware minimization (SAM) algorithms targeting flat minima by jointly minimizing a worst-case generalization bound and local parameter sharpness (Foret et al., 2021; Kwon et al., 2021; Kim et al., 2022; Möllenhoff & Khan, 2023).

While flat minima methods report widespread improvement over conventional optimizers (Kaddour et al., 2022), we argue that they are not fully connected to the modern fine-tuning paradigm, wherein a task-specific model inherits parameters from a pre-trained model instead of being trained from scratch (Wang et al., 2019; Liang et al., 2020). In these settings, focusing on local properties of the loss landscape (e.g., sharpness) may fail by suboptimally exploiting useful generic task-agnostic structures within pre-trained representations. In this work, we propose to combine sharpness-aware minimization with the robust transfer of pre-trained information (in representation space) for fine-tuning scenarios requiring out-of-distribution knowledge for successful adaptation.

Figure 1:TRAM introduces an awareness of function curvature (i.e., the trust region) into sharpness-aware minimization. (left) TRAM estimates the size of the trust region, 
𝑑
, around 
𝑓
⁢
(
𝑥
)
 in green. (right) the loss contour in parameter space following Kwon et al. (2021) where blue is the typical loss; red is the maximized worst-case loss for ASAM; and green is the maximized loss within the subdomain constrained for function smoothness.

Existing methods to improve leveraging pre-trained structure during fine-tuning include trust region regularization (Schulman et al., 2015; Jiang et al., 2020; Aghajanyan et al., 2021) or adversarial perturbation (Zhu et al., 2020; He et al., 2021). These methods focus on the curvature of the function itself e.g., by encouraging smooth local changes in representations. The intuition is that lower representation curvature during fine-tuning limits a function from catastrophically forgetting (French, 1999, inter alia) useful information from pre-training. This representation smoothing approach contrasts with SAM-style optimization for parameter smoothness. Both perspectives show empirical improvement in downstream tasks (Aghajanyan et al., 2021; Bahri et al., 2022), but a fusion of these strategies is presently under-explored.

To this end, we propose TRAM: Trust Region Aware Minimization, a fine-tuning algorithm for out-of-distribution generalization combining the success of both sharpness-aware and trust region optimization. TRAM uses a trust region bound to inform the SAM adversarial neighborhood, introducing an awareness of function curvature within optimization for flatter minima. The resulting algorithm yields low-sharpness parameters and improved adaptation of pre-trained models to downstream tasks. To illustrate TRAM’s advantage over strong baselines in retaining generic representations, we focus on distribution transfer challenges within Transformer-based models. Our contributions are:

• 

We propose a new optimization algorithm: Trust Region Aware Minimization integrates representation smoothing regularization into sharpness-aware minimization. We propose and contrast multiple variants of TRAM based on differing perspectives on trust region estimation and efficiency trade-offs (Section 3).1

• 

We highlight that TRAM empirically improves generalization for multiple out-of-distribution adaptation tasks across vision and natural language: cross-dataset adaptation for image classification, cross-domain language modeling and zero-shot cross-lingual transfer (Section 4).

• 

We analyze how TRAM limits catastrophic forgetting and optimizes flatter minima to improve fine-tuning. By characterizing major and minor distribution shifts, we identify how TRAM outperforms the trend in anticorrelated generalization scenarios. Our analysis verifies that TRAM optimizes a smoother loss surface for both in-domain and out-of-domain distributions. TRAM also improves representation similarity between seen and unseen distributions to improve cross-domain classification (Section 4).

2Background

We describe SAM and trust region optimization, highlighting how these approaches have similar goals. Our motivation for TRAM is the unifying features of each approach outlined in Table 1.

Notation: We consider function 
𝑓
:
𝑋
→
𝑌
 parameterized by weights 
𝜃
 and evaluated by loss function 
ℓ
:
𝑌
×
𝑌
→
ℝ
+
. The expected loss on true distribution 
𝒟
 is 
𝐿
𝒟
⁢
(
𝜃
)
=
𝔼
(
𝑥
,
𝑦
)
∼
𝒟
⁢
[
ℓ
⁢
(
𝑦
,
𝑓
⁢
(
𝑥
;
𝜃
)
)
]
 and the empirical estimate is 
𝐿
𝑆
=
1
𝑛
⁢
∑
𝑆
ℓ
⁢
(
𝑦
𝑖
,
𝑓
⁢
(
𝑥
𝑖
;
𝜃
)
)
 sampling 
𝑛
 training samples, 
𝑆
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
, from 
𝒟
. Functional distance on model outputs is measured by the Kullback-Leibler divergence 
𝐷
KL
(
𝑝
|
|
𝑞
)
 between target 
𝑝
 and estimate 
𝑞
. We describe successful domain transfer to distribution 
𝒟
′
 as a non-increasing loss for sample 
𝑆
′
∼
𝒟
′
.

Sharpness-Aware Minimization: Foret et al. (2021) define local sharpness as 
max
∥
𝜖
∥
2
≤
𝜌
⁡
𝐿
𝑆
⁢
(
𝜃
+
𝜖
)
−
𝐿
𝑆
⁢
(
𝜃
)
. The SAM objective (Equation 1) regularizes parameter magnitude to minimize this sharpness metric jointly with loss within local parameter neighborhood 
𝜌
.

	
𝐿
𝑆
SAM
=
min
𝜃
⁡
max
∥
𝜖
∥
2
≤
𝜌
⁡
𝐿
𝑆
⁢
(
𝜃
+
𝜖
)
+
𝜆
2
⁢
∥
𝜃
∥
2
2
		
(1)
	
𝜖
ASAM
∗
=
𝜌
⁢
𝜃
2
⁢
∇
𝐿
𝑆
∥
𝜃
⁢
∇
𝐿
𝑆
∥
2
		
(2)

This min-max optimization problem is solved in alternating stages. Initial ascent perturbs parameters 
𝜃
 to 
𝜃
+
𝜖
, where 
𝜖
 is a perturbation maximizing loss (to minimize local sharpness). The feasible region for perturbation 
𝜖
 is a Euclidean spherical neighborhood with radius 
𝜌
>
0
. Successive descent evaluates gradients at 
𝜃
+
𝜖
 for gradient descent at 
𝜃
 using the local worst-case loss.

The optimal 
𝜖
, the perturbation for worst-case loss within the 
𝜌
-ball, is the source of ongoing debate. Foret et al. (2021) express a closed-form solution setting 
𝜖
 as the radius 
𝜌
 scaled by the normalized gradient. Kwon et al. (2021) propose Adaptive SAM (ASAM) to improve SAM with invariance to the loss scaling. For ASAM, each parameter within 
𝜃
 is perturbed by 
𝜌
 scaled by parameter gradient and the parameter norm (Equation 2). TRAM follows SAM in setting 
𝜖
 with scale invariance and also augments 
𝜖
 such that the update in 
𝜃
 respects a maximum divergence in the function space.

Trust Region Regularization: Trust region regularization encourages low curvature during optimization by regularizing the function output distribution with respect to a previous step’s distribution. A fine-tuned model with high curvature (i.e., distance) to pre-trained representations may struggle to connect task-specific knowledge with novel domains. This approach proves successful in penalizing large policy updates in reinforcement learning (Schulman et al., 2015), encouraging local smoothness to adversarial perturbation (Jiang et al., 2020) and minimizing catastrophic forgetting for domain transfer (Aghajanyan et al., 2021).

Equation 3 defines the objective under Trust Region Policy Optimization (TRPO; Schulman et al., 2015) constraining loss, 
𝐿
𝑆
, with a regularization term 
𝑑
𝜃
. TRPO idealizes smoothness in 
𝑓
⁢
(
𝑥
)
 by regularizing local function similarity to the previous iterate. The update at 
𝑡
 is constrained such that changes in probability density, 
𝑝
𝑓
(
⋅
|
𝑥
,
𝜃
)
 are no larger than some 
𝜀
 measured by divergence 
𝑑
:
𝒴
×
𝒴
→
ℝ
+
. There are several ways of defining 
𝑑
—we consider options in Equations 4 to 5.

	
min
𝜃
⁡
𝐿
𝑆
⁢
(
𝜃
)
⁢
subject
⁢
to
⁢
𝑑
𝜃
≤
𝜀
		
(3)

Equation 4 estimates the trust region as the KL divergence between predictive distributions at the previous and current step. Intuitively, penalizing divergence from prior steps encourages the function to stay “close” to the previous distribution i.e., within the trust region of equivalent output. Across training, 
𝑑
𝜃
 encourages small updates with low curvature between fine-tuned and pre-trained models.

	
𝑑
𝜃
⁢
(
𝜃
𝑡
−
1
,
𝜃
𝑡
)
	
=
𝔼
𝑥
∼
𝐷
[
𝐷
KL
(
𝑝
𝑓
(
⋅
|
𝑥
,
𝜃
𝑡
−
1
)
|
|
𝑝
𝑓
(
⋅
|
𝑥
,
𝜃
𝑡
)
)
]
		
(4)

Equation 5 provides the penalty from R3F (Aghajanyan et al., 2021) where 
𝑑
𝑥
 estimates the trust region by sampling from inputs under parametric noise. This penalizes the divergence between 
𝑝
𝑓
(
⋅
|
𝑥
,
𝜃
𝑡
)
 and 
𝑝
𝑓
(
⋅
|
𝑥
+
𝑧
,
𝜃
𝑡
)
 for some zero-mean noise 
𝑧
∼
𝒩
⁢
(
0
,
𝜎
2
)
. R3F proposes that sampling 
𝑧
 estimates the trust region by simulating a distribution shift in 
𝑝
𝑓
 corresponding to perturbed 
𝑥
+
𝑧
. This encourages similarity to a neighborhood around 
𝑓
⁢
(
𝑥
,
𝜃
)
 with equivalent output. Either approach estimates the permissible distance for an update in 
𝜃
 without increasing local representation curvature. We focus on trust region methods to improve generalization across distributions via improved leveraging of pre-trained structure (Jiang et al., 2020, inter alia).

	
𝑑
𝑥
⁢
(
𝑥
+
𝑧
,
𝑥
)
	
=
𝔼
𝑧
∼
𝒩
[
𝐷
KL
(
𝑝
𝑓
(
⋅
|
𝑥
+
𝑧
,
𝜃
)
|
|
𝑝
𝑓
(
⋅
|
𝑥
,
𝜃
)
)
]
		
(5)

Comparison: SAM, TRPO, and R3F have similar goals in searching for generalizable solutions while appearing superficially distinct. We compare the broad motivations and qualities of methods in Table 1, highlighting both perspectives optimize for smoothness in different spaces.

SAM minimizes sharpness within a neighborhood in 
𝜃
 set by scalar parameter 
𝜌
. Trust region regularization penalizes loss by scalar distance 
𝑑
𝜃
⁢
or
⁢
𝑑
𝑥
. We hypothesize that this regularization can inform the size of the SAM neighborhood. Can we jointly minimize sharpness and penalize high curvature in representations? Considering the sharpness objective  
(
max
∥
𝜖
∥
2
≤
𝜌
⁡
𝐿
𝑆
⁢
(
𝜃
+
𝜖
)
−
𝐿
𝑆
⁢
(
𝜃
)
)
 we consider if this 
𝜖
 can also satisfy the Equation 3 constraint of 
𝑑
𝜃
⁢
or
⁢
𝑑
𝑥
<
𝜀
. Our intuition here is to minimize parameter sharpness (i.e., SAM) only within an update promoting low representation curvature. Combining the features of these solutions could improve generalization to unseen distributions during fine-tuning.

Table 1:Comparison between SAM-style, trust region and TRAM learning. SAM optimizes parameters for low sharpness, trust region methods optimize for low-curvature representations. TRAM combines these strategies to bound SAM-style learning within a trust region neighborhood.
	Goal	
𝜖
	Distance	Domain	Gradient	Forward/Backward
SAM-style	Low-sharpness 
𝜃
	Equation 2	—	
𝜌
-ball	
∇
𝐿
𝑆
 at 
𝜃
+
𝜖
	2 
→
, 2 
←

Trust region	Low-curvature 
𝑓
⁢
(
𝑦
|
𝑥
,
𝜃
)
	—	
𝑑
𝜃
⁢
or
⁢
𝑑
𝑥
	
𝐷
KL
 over Distance	
∇
𝐿
𝑆
+
𝑑
𝜃
⁢
or
⁢
𝑑
𝑥
	2 
→
, 1 
←

TRAM	Both	Equation 7	
𝑑
𝜃
⁢
or
⁢
𝑑
𝑥
	
𝑑
𝜃
-
or
⁢
𝑑
𝑥
-ball	
∇
𝐿
𝑆
 at 
𝜃
+
𝜖
	3 
→
, 2 
←
3TRAM: Trust Region Aware Minimization

We consider methods improving generalization by encouraging low-sharpness parameters and task transfer by encouraging low curvature in representation space. We introduce TRAM: Trust Region Aware Minimization unifying sharpness-aware and trust region optimization. Kim et al. (2022) raise that the 
𝜌
 hyperparameter defining the ascent neighborhood in SAM is an “ad hoc” scaling with little relationship to the loss landscape or parameter geometry. We propose to instead define the ascent region by a trust region in representation space.

TRAM substitutes 
𝜌
 in Equation 2 with the trust region metric, 
𝑑
:
𝒴
×
𝒴
→
ℝ
+
∗
, as defined in Section 2. We estimate trust regions using the divergence from a prior model distribution (
𝑑
𝜃
, Equation 4) or divergence from the current distribution under parametric noise (
𝑑
𝑥
, Equation 5). TRAM constrains the maximization domain for ascent (i.e., 
𝜃
→
𝜃
+
𝜖
) to the parameter corollary for the trust region i.e., 
max
∥
𝜖
∥
2
≤
𝑑
 substituted within Equation 1. TRAM perturbs 
𝜃
 with a loss perturbation only within the parameter neighborhood constrained for low representation curvature. This introduces function curvature awareness within TRAM in addition to the sharpness-awareness objective for flatter minima. In contrast, the maximization region, 
𝜌
 in SAM/ASAM has no sensitivity to function curvature. We build TRAM on ASAM, and not SAM, after observing strictly better performance in our preliminary experiments.

	
∇
𝐿
TRAM
(
𝜃
)
=
∂
𝐿
𝑆
∂
𝜃
|
𝜃
=
𝜃
+
𝜖
TRAM
∗
		
(6)
	
𝜖
TRAM
∗
=
𝑑
⁢
𝜃
2
⁢
∇
𝐿
𝑆
⁢
(
𝜃
𝑡
)
∥
𝜃
⁢
∇
𝐿
𝑆
⁢
(
𝜃
𝑡
)
∥
2
		
(7)

The gradient descent update in TRAM is Equation 6, where 
𝜖
TRAM
∗
 is solved as Equation 7 by direct substitution of 
𝜌
 in ASAM. Algorithm 1 in Section B.6 details the full training algorithm for TRAM based on the SAM-style min-max optimization routine. TRAM does not require tuning a 
𝜌
 hyperparameter for stable training. TRAM using 
𝑑
𝜃
 introduces no new hyperparameters, and using 
𝑑
𝑥
 requires only tuning 
𝜎
 for additive noise 
𝑧
. We hypothesize that TRAM jointly minimizes parameter sharpness and representation curvature to minimize catastrophic forgetting of pre-trained structure. Our results in Section 4 empirically validate this hypothesis.

Connection to ASAM: The geometric interpretation of TRAM frames the maximization domain defined by 
𝑑
 as a subdomain of the 
𝜌
-radius Euclidean ball defined in ASAM. Whereas ASAM defines a fixed radius by 
𝜌
 at each step, TRAM instead uses the nonzero 
𝑑
 radius constraining the maximization domain to additionally satisfy the trust region constraint outlined in Equations 3 to 5. Foret et al. (2021, Theorem 2) defines a PAC-Bayesian generalization bound for SAM on 
𝐿
𝒟
 assuming 
𝜌
>
0
. Kwon et al. (2021, Theorem 3) identify a similarly valid bound when considering the norm-adaptive scaling on 
𝜖
ASAM
∗
 as in Equation 2. We assume 
𝑑
≤
𝜌
 for similar asymptotic behavior for 
𝜖
TRAM
∗
 to 
𝜖
ASAM
∗
. We infer that TRAM inherits the existing generalization bound of ASAM for any 
𝜌
>
0
 directly substituted for 
𝑑
 i.e., TRAM is a subsolution of ASAM. We can constrain 
𝑑
 such that 
max
𝜃
¬
⁢
𝑡
⁡
𝑑
𝜃
⁢
(
𝜃
¬
⁢
𝑡
,
𝜃
𝑡
)
≤
𝜌
 or 
max
𝑧
⁡
𝑑
𝑥
⁢
(
𝑥
+
𝑧
,
𝑥
)
≤
𝜌
 to enforce this bound 
𝑑
∈
(
0
,
𝜌
]
. However, we empirically observe this constraint is satisfied for the optimal setting of 
𝜌
 in ASAM.

Improving Efficiency with TRAM-Fisher: Kim et al. (2022) propose an alternative to SAM removing the Euclidean assumption for parameter geometry. Fisher SAM (FSAM) instead exploits the statistical manifold induced by the Fisher Information metric of predictive distribution of the function, 
𝑝
𝑓
⁢
(
𝑦
|
𝑥
,
𝜃
)
 (Amari, 1998) to set 
𝜖
. This measures statistical divergence between 
𝜃
 and 
𝜃
+
𝜖
 resulting in 
𝜖
FSAM
∗
 in Equation 8 defining an ellipsoid around 
𝜃
 scaled by the Fisher Information matrix, 
𝐹
⁢
(
𝜃
)
. 
𝐹
⁢
(
𝜃
)
 is prohibitively expensive at scale and is approximated with Equation 9, the diagonal of the squared gradient sum for each batch 
𝐵
.

	
𝜖
FSAM
∗
=
𝐹
⁢
(
𝜃
)
−
1
⁢
∇
𝐿
𝑆
∇
𝐿
𝑆
⁢
𝐹
⁢
(
𝜃
)
−
1
⁢
∇
𝐿
𝑆
		
(8)
	
𝐹
^
⁢
(
𝜃
)
=
Diag
⁢
(
1
|
𝐵
|
⁢
∑
𝑖
∈
𝐵
(
log
⁡
𝑝
𝑓
⁢
(
𝑦
𝑖
|
𝑥
𝑖
,
𝜃
)
)
)
2
		
(9)

We propose TRAM-Fisher as an efficient variant of TRAM inspired by Fisher SAM. Where FSAM measures the Fisher Information geometry of 
𝜃
 under input 
𝑥
, we instead sample the geometry of 
𝜃
 under the trust region estimation from 
𝑥
+
𝑧
. Our proposal is minimal: replace 
𝑝
⁢
(
𝑦
𝑖
|
𝑥
𝑖
,
𝜃
)
 with 
𝑝
⁢
(
𝑦
𝑖
|
𝑥
𝑖
+
𝑧
𝑖
,
𝜃
)
 to estimate the Fisher Information Matrix of the trust region neighborhood as 
𝔼
𝑧
∼
𝒩
⁢
[
𝐹
^
⁢
(
𝑥
+
𝑧
;
𝜃
)
]
. We sample parametric noise 
{
𝑧
𝑖
}
𝑖
=
0
|
𝐵
|
 identically to TRAM and now scale learning with the information geometry of the low curvature neighborhood, 
𝑓
⁢
(
𝑥
+
𝑧
)
. TRAM-Fisher uses the same number of forward/backward passes as FSAM and only requires additional processing to sample 
𝑧
 and compute 
𝑥
+
𝑧
. TRAM-Fisher matches FSAM in runtime efficiency (with marginal additional operations) and performs competitively across our experiments. The full TRAM-Fisher algorithm is shown in Section B.6.

Summary: We propose three variants of TRAM, and TRAM-Fisher, summarized in Table 2. TRAM-
𝜃
𝑡
−
1
 follows TRPO (Schulman et al., 2015) in using previous step parameters, 
𝜃
𝑡
−
1
, to measure the trust region. We also propose a simplification of TRAM-
𝜃
𝑡
−
1
 estimating the trust region using 
𝑑
𝜃
 between current 
𝜃
𝑡
 and pre-trained model 
𝜃
0
. TRAM-
𝜃
0
 improves training efficiency by removing an updating 
𝜃
𝑡
−
1
 state. TRAM-
𝑥
 follows R3F (Aghajanyan et al., 2021) using noise-based trust region measurement with additional hyperparameter 
𝑧
 for sampling parametric noise. Practically, TRAM requires one additional forward pass adding marginal overhead to the extant complexity of SAM-style training. Despite this additional cost, Section 4 identifies empirical benefits to TRAM and targeted improvement to out-of-domain loss surface sharpness and cross-domain representation similarity.

Table 2:We propose four variants of TRAM based on different trust region estimations. TRAM-
𝜃
𝑡
−
1
 uses divergence against the previous step; TRAM-
𝜃
0
 is a simplifying heuristic of this divergence against the pre-trained model only; and TRAM-
𝑥
 uses noised input divergence, 
𝑑
𝑥
. TRAM-Fisher extends FSAM by measuring the Fisher Information metric around the trust region.
Variant	Trust region measurement	
𝜖
	Domain	Forward/Backward
TRAM-
𝜃
𝑡
−
1
	
𝑑
𝜃
⁢
(
𝜃
𝑡
−
1
,
𝜃
𝑡
)
	Equation 7	
𝑑
𝜃
-ball	3 
→
, 2 
←

TRAM-
𝜃
0
	
𝑑
𝜃
⁢
(
𝜃
0
,
𝜃
𝑡
)
	Equation 7	
𝑑
𝜃
-ball	3 
→
, 2 
←

TRAM-
𝑥
	
𝑑
𝑥
⁢
(
𝑥
+
𝑧
,
𝑥
)
,
𝑧
∼
𝒩
⁢
(
0
,
𝜎
2
)
	Equation 7	
𝑑
𝑥
-ball	3 
→
, 2 
←

TRAM-Fisher	
𝐹
^
⁢
(
𝑥
+
𝑧
;
𝜃
)
,
𝑧
∼
𝒩
⁢
(
0
,
𝜎
2
)
	Equation 8	
𝐹
^
-ellipse	2 
→
, 2 
←

We outline our datasets in Appendix A, and experiment design in Appendix B for both vision and language modalities. We compare to gradient descent methods (SGD, Adam), sharpness aware methods (SAM, ASAM, FSAM), and trust region methods (TRPO, R3F, MESA) further detailed in Section B.2. Broadly, we investigate the hypothesis that out-of-distribution generalization improves by jointly minimizing parameter sharpness and representation curvature in the function.

4Results
4.1Cross-dataset image classification
Table 3:Cross-dataset adaptation from ImageNet to CIFAR-100, Stanford Cars and Oxford Flowers. We report Top-1 classification accuracy averaged over five runs, 
±
 the 95% confidence interval, for direct comparison to Kim et al. (2022).
	CIFAR-100 
(
↑
)
	Cars 
(
↑
)
	Flowers 
(
↑
)

SGD	
87.97
±
0.12	
92.85
±
0.31	
94.53
±
0.20
SAM	87.99
±
0.09	93.29
±
0.01	95.05
±
0.06
ASAM	
87.97
±
0.08	93.28
±
0.02	95.08
±
0.10
FSAM	88.39
±
0.13	93.42
±
0.01	95.26
±
0.03
TRAM-
𝜃
𝑡
−
1
	88.47
±
0.16	
93.49
±
0.04	
97.07
±
0.10
TRAM-
𝜃
0
	88.31
±
0.09	93.16
±
0.07	95.53
±
0.10
TRAM-
𝑥
	
88.78
±
0.01	93.32
±
0.11	96.34
±
0.03
TRAM-Fisher	88.02
±
0.18	93.12
±
0.13	94.90
±
0.11

First, we validate the performance of TRAM in a standardized setting for comparison to other SAM-style optimizers. We evaluate adapting ViT-base (Dosovitskiy et al., 2021) from ImageNet pre-training to image classification fine-tuning. We follow the setup of Kim et al. (2022, Section 5.1) evaluating adaptation to CIFAR-100 (Krizhevsky, 2009), Stanford Cars (Krause et al., 2013), and Oxford Flowers (Nilsback & Zisserman, 2008). Section B.3 details our experiment design.

Table 3 details the Top-1 accuracy results for this experiment with direct comparison to Kim et al. (2022, Table 3). The best-performing variant of TRAM (TRAM-
𝜃
𝑡
−
1
 or TRAM-
𝑥
) is significantly superior to the closest FSAM competitor (
𝑝
<
0.01
). Other variants of TRAM, TRAM-
𝜃
0
 or TRAM-Fisher, are largely competitive with prior methods. Our observations validate the hypothesis that TRAM improves adaptation across datasets during fine-tuning for image classification. This comparison acts as a sanity check and demonstrates the utility of our method compared to other SAM-style optimizers. TRAM yields improved fine-tuned image classification models by encouraging smoothness in parameter and function space.

4.2Cross-domain language modeling
Table 4:M2D2 perplexity (lower is better) on Wikipedia (upper) & S2ORC (lower) splits. TRAM-
𝜃
𝑡
−
1
 significantly improves over prior work (
𝑝
<
0.01
 Kolmogorov-Smirnov test). Results are grouped as: (i) optimizers; (ii) trust region methods; and (iii) TRAM variants. The leftmost column is the training domain and we evaluate zero-shot perplexity on ten domains unseen during fine-tuning (full details in Appendix A). ZS Avg. is the macro-average of all zero-shot domains.
Wiki	Soc.	Cult.	Gen.	Health.	Hist.	Human.	Math.	Nat.	Phil.	Rel	Tech.	ZS Avg. 
↓

GPT-2	27.2	27.7	27.8	24.5	29.2	28.8	28.6	29.4	27.8	27.7	28.7	28.0
Adam	
24.8
	
26.3
	
26.4
	
23.6
	
27.2
	
27.0
	
27.4
	
27.6
	
26.3
	
25.8
	27.4	
26.5

SAM	24.5	25.9	26.0	23.1	26.9	26.6	26.6	27.2	25.8	25.5	27.0	26.1
ASAM	
24.8
	25.4	25.6	22.5	27.1	26.4	26.3	26.7	25.5	25.5	
28.1
	25.9
FSAM	21.7	23.0	23.3	20.6	23.9	23.7	23.8	24.0	23.1	22.8	24.0	23.2
TRPO	21.8	23.0	23.3	20.7	24.0	23.7	23.8	24.0	23.1	22.8	24.1	23.3
R3F	21.8	23.0	23.3	20.7	24.0	23.7	23.8	24.0	23.1	22.8	24.1	23.3
MESA	23.1	24.0	24.3	21.5	25.4	24.9	24.8	25.2	24.1	24.0	25.1	24.3
TRAM-
𝑥
	21.9	23.1	23.4	20.7	24.0	23.3	23.9	23.9	23.2	22.7	23.9	23.2
TRAM-
𝜃
𝑡
−
1
	
20.9
	
22.4
	
22.7
	
20.1
	
23.1
	
22.9
	
23.2
	
23.3
	
22.4
	
22.0
	
23.4
	
22.5

TRAM-
𝜃
0
	21.9	23.1	23.4	20.7	23.9	23.3	23.9	23.8	23.1	22.7	23.9	23.2
TRAM-Fisher	22.5	23.7	24.0	21.3	24.6	24.0	24.7	24.6	23.8	23.3	24.6	23.9
S2ORC	Math	Art	Astro	CondM.	CS	Econ.	NLin.	Phil.	Phys.	QBio	Stat	ZS Avg. 
↓

GPT-2	27.6	35.8	32.4	30.9	27.9	29.5	27.6	33.7	33.5	30.9	23.4	30.6
Adam	11.4	44.2	33.9	20.1	21.2	21.0	14.7	41.9	29.5	30.8	16.9	27.4
SAM	10.5	45.3	33.2	18.7	20.3	20.0	13.7	42.4	28.3	30.2	16.1	26.8
ASAM	10.3	45.6	33.2	18.5	20.1	19.8	13.5	42.6	28.2	30.2	15.9	26.8
FSAM	10.4	45.6	33.3	18.5	20.2	19.9	13.5	42.7	28.3	30.2	15.9	26.8
TRPO	10.4	46.0	33.4	18.6	20.3	20.0	13.6	42.9	28.4	30.4	16.0	26.9
R3F	10.4	46.0	33.4	18.6	20.2	20.0	13.6	42.9	28.4	30.4	16.0	26.9
MESA	
11.9
	
44.1
	
34.1
	
20.8
	
21.7
	
21.6
	
15.3
	
41.7
	
30.0
	
31.0
	
17.4
	
27.8

TRAM-
𝑥
	10.4	44.9	33.0	18.6	20.1	19.9	13.6	42.0	28.1	30.0	15.9	26.6
TRAM-
𝜃
𝑡
−
1
	
9.6
	
46.8
	32.5	
17.2
	
19.2
	
18.9
	
12.6
	
43.3
	
27.0
	
29.6
	
15.0
	
26.2

TRAM-
𝜃
0
	10.4	44.8	33.0	18.6	20.1	19.9	13.6	42.0	28.2	30.0	15.9	26.6
TRAM-Fisher	10.5	46.1	
32.4
	18.7	20.3	20.0	13.6	43.0	28.2	30.3	16.0	26.9

We now consider zero-shot cross-domain language modeling using the M2D2 Corpus (Reid et al., 2022) outlined in Appendix A. We hypothesize that TRAM can improve domain transfer in language modeling by retaining domain-agnostic information from pre-training when fine-tuning to a specific domain. We train a GPT-2 Base model (Radford et al., 2019) on the largest domain in each split of M2D2 (Soc. domain 379M tokens for Wikipedia and Math 1.4B tokens for S2ORC) and evaluate perplexity across ten domains unseen during fine-tuning. Section B.4 details our complete experiment design.

Our results in Table 4 validate our hypothesis for TRAM in the cross-domain setting to improve out-of-domain language modeling fine-tuning on a single domain. All TRAM variants (excluding TRAM-Fisher) perform comparably or above competitors in zero-shot transfer across both splits of M2D2. TRAM improves domain transfer in fine-tuned models by better leveraging pre-trained information from unseen domains within the smoother minima idealized by SAM-style training. Generally, the naive Adam baseline or the MESA trust region comparison perform poorest at cross-domain language modeling for Wikipedia or S2ORC splits respectively. As with image classification, FSAM is the strongest competitor to TRAM. The best variant in both splits is TRAM-
𝜃
𝑡
−
1
 improving in-domain and average zero-shot perplexity. TRAM-
𝜃
𝑡
−
1
 uses the TRPO method of estimating the trust region using the parameters of the previous step. This variant always yields the lowest perplexity in the training domain and the majority of similar and distant zero-shot domains. We additionally verify that TRAM performs competitively at a larger model scale using GPT2-XL (1.5B parameters) in Table 10 in Section C.2. We also compare against a naive combination of methods (e.g., ASAM+TRPO) in Section C.3.

TRAM improves perplexity for all domains in the Wikipedia split, where all zero-shot domains are positively correlated with the training domain perplexity. However, we observe that perplexity degrades for domains distant from the fine-tuning domain in S2ORC (Math) which benefit less from shared features. Given that neither SAM-style nor trust region methods inverted this anticorrelation trend, it is unsurprising that TRAM follows suit. This confounder results in the overall best model, TRAM-
𝜃
𝑡
−
1
, reporting the worst performance for the distant domains where the overall poorest model, MESA, reports the best performance. We suggest that optimization alone may be insufficient to improve zero-shot domain adaptation for larger distribution shifts. We discuss further the correlation between domain-specific perplexity in Section C.1.

4.2.1Easy and hard generalization

When evaluating performance variation between different distributional shifts—we find that TRAM improves on all prior work for minor shifts (e.g., Math to Physics/Phys.) and generally matches or improves on a negative trend for major shifts (e.g., Math to Art). Discussion of out-of-domain generalization often overlooks differences between major and minor shifts. In practice, in-domain performance has a very different relationship to performance when generalizing to a major domain shift rather than a minor shift. Considering minor distribution shifts, accuracy is strongly correlated on in-domain and out-of-domain datasets (Miller et al., 2021). However, major distribution shifts may lead to scenarios where performance is instead anticorrelated with in-domain accuracy (Teney et al., 2022). Considering these scenarios in the S2ORC task, we observe that models trained using TRAM often perform better on new domains than their in-domain performance would predict. Furthermore, TRAM improves perplexity across both minor and major distribution shifts.

\phantomsubcaption
\phantomsubcaption
\phantomsubcaption
Figure 2:Perplexity on S2ORC training domain (Math) and zero-shot domains. We report perplexity across: (2) domains correlated with Math as Stem domains (see Section C.1), (2) Art domain, and (2) the Philosophy (Phil.) domain. Each figure includes linear regression trends: the blue dotted trend is for prior work and green dashed line includes all TRAM variants. Positive slope (
𝜌
>
0
) represents correlated domains, negative slope (
𝜌
<
0
) represents anticorrelated domains. We report Pearson 
𝜌
 correlation for the blue trend noting 
𝑝
<
0.01
 significance.

Figure 2 shows the close positive correlation between performance on the training domain (Math) and the average across all other STEM disciplines, considering all optimization approaches. As detailed in Section C.1, performance correlates with 
𝜌
>
0.8
 between Math and each individual STEM category. Considering the blue dotted trend for previous optimization methods (excluding TRAM), we see that all TRAM optimizers fall on or marginally below the line. This result suggests that TRAM not only supports in-domain performance but specifically improves generalization to similar domains.

By contrast, we find there is generally a trade-off between performance on Math and the hardest anticorrelated domains: Art (Figure 2) and Philosophy (Phil, Figure 2). Both TRAM-
𝑥
 and TRAM-
𝜃
0
 fall far below the trend for previous algorithms where in-domain improvement worsens out-of-domain perplexity. TRAM not only matches or outperforms existing methods on easier generalization cases, but exhibits a lesser trade-off between easy and hard generalization compared to all previous approaches.

4.3Zero-shot cross-lingual transfer

Finally, we now consider if TRAM improves cross-lingual adaptation during monolingual fine-tuning. We adapt a multilingual pre-trained model to an English entailment classification task (NLI) and then evaluate the zero-shot cross-lingual capability for the model to classify entailment from inputs in 14 unseen languages. We hypothesize that TRAM benefits cross-lingual transfer via improved application of multilingual pre-trained information to a task with only English training data. In general, languages closer to English (e.g., French, German) are “easier” for transfer than distant or low-resource languages (e.g., Urdu, Swahili) (Ahmad et al., 2019). An ideal system will produce equivalent cross-lingual transfer for all zero-shot languages. Our complete experiment design is outlined in Section B.5. We train an XLM-Roberta-based model (Conneau et al., 2020a) on English MultiNLI (Williams et al., 2018) and report accuracy results for the XNLI cross-lingual entailment benchmark (discussed in Appendix A).

Table 5 highlights that TRAM improves over all competing methods for the cross-lingual transfer objective, similar to our findings for cross-dataset image classification and cross-domain language modeling. Similar to the above tasks, TRAM-
𝑥
 and TRAM-
𝜃
𝑡
−
1
 are the best-performing algorithms reporting both the strongest in-domain and average out-of-domain accuracy. Either TRAM variant is the best method across all individual languages. We identify that all methods worsen for languages distant from English in a similar trend to language modeling for anticorrelated domains. However, here TRAM is strictly superior to any other method for both near and distant languages to English. Notably, TRAM-Fisher significantly improves upon FSAM (
𝑝
<
0.01
) despite the close similarity in methods. Given the additional forward pass required for TRAM-
𝑥
, TRAM-Fisher represents a better performance-complexity trade-off which is competitive in some tasks. We analyze the loss surface and representation transfer in Table 6 to verify that TRAM extends a low-curvature loss surface and representation smoothness to all zero-shot languages. In Section C.5, we train a model using TRAM with alternative distances for trust region measurement to analyze the criticality of using KL divergence. We observe that TRAM is robust to multiple distances with marginal degradation. These results empirically verify our hypothesis that training with complementary SAM-style and trust region methods improves the language transferability of a fine-tuned model.

Table 5:XNLI accuracy (higher is better) for training language (En) and 14 zero-shot target languages summarised by ZS Avg. (key in Appendix A). All TRAM variants significantly outperform other methods (
𝑝
<
0.01
 Wilcoxon test). Results are grouped as: (i) optimizers; (ii) trust region methods; and (ii) TRAM variants. We report the mean across 20 seeds with standard deviation in Table 13.
	En	Ar	Bg	De	El	Es	Fr	Hi	Ru	Sw	Th	Tr	Ur	Vi	Zh	ZS Avg. 
↑

Adam	
83.9
	
71.2
	
77.1
	
75.7
	
75.2
	
78.3
	77.6	69.6	
74.9
	64.6	
71.2
	
72.2
	65.8	74.1	
73.1
	
72.9

SAM	84.8	72.1	78.1	76.7	75.7	79.0	77.9	69.8	75.7	65.2	71.8	73.1	66.8	75.1	74.2	73.7
ASAM	85.0	72.0	78.4	76.9	76.1	79.5	78.5	70.4	76.1	65.2	72.5	73.4	66.9	75.5	74.2	74.0
FSAM	84.7	72.2	78.1	76.9	76.0	79.3	78.4	70.0	76.1	65.1	72.2	73.0	66.8	75.3	74.2	73.8
TRPO	84.9	71.3	77.7	76.2	75.3	78.6	
77.3
	
69.2
	75.2	64.4	71.6	72.4	
65.3
	
73.8
	73.3	73.0
R3F	85.5	72.7	78.9	77.5	76.8	79.9	79.2	70.7	76.8	66.2	72.9	73.9	66.6	75.8	74.6	74.5
MESA	84.9	71.9	77.9	76.7	75.7	78.8	77.8	69.6	75.8	
64.1
	72.1	72.4	65.7	74.4	73.9	73.3
TRAM-
𝑥
	
86.2
	
73.5
	
79.8
	
78.3
	
77.5
	
80.9
	79.6	71.4	
77.5
	66.0	
73.8
	
74.3
	
67.6
	
76.7
	
75.9
	
75.2

TRAM-
𝜃
𝑡
−
1
	
86.2
	73.1	79.5	78.2	77.0	80.2	
79.7
	
71.5
	
77.5
	
66.4
	73.3	74.2	67.5	
76.7
	75.8	75.0
TRAM-
𝜃
0
	85.6	72.9	79.3	77.8	77.4	80.2	79.6	71.2	77.1	65.9	73.3	74.2	67.5	
76.7
	75.8	74.9
TRAM-Fisher	84.3	73.1	78.7	77.1	76.2	79.5	78.4	71.4	76.6	65.7	73.2	73.6	67.5	75.5	75.5	74.4

Loss surface dynamics: Investigating the loss surface, we test the hypothesis that TRAM leads to flatter minima on both in-domain and out-of-domain data. We evaluate validation set 
𝜖
-sharpness (Keskar et al., 2017), defined in Section B.7, across 20 trained models. We report in-domain (for English) and out-of-domain (zero-shot languages) 
𝜖
-sharpness in Table 6 across TRAM and baselines (omitting models which under-performed). Most methods unsurprisingly demonstrate a lower in-domain sharpness but poorer out-of-domain sharpness. TRAM yields a smoother solution for both the in-domain and out-of-domain regions of the loss surface. We also observe an improved average Pearson correlation (and lower variance) between in-distribution and out-of-distribution sharpness using TRAM. This infers that the relationship between loss surfaces of different distributions is more desirably predictable with TRAM. Notably, other SAM-style methods are worse than Adam for out-of-domain sharpness—suggesting that current SAM algorithms (excluding TRAM) are possibly “sharpness-aware” only within the training distribution.

Table 6:Analysis of (a) 
𝜖
-sharpness and (b) CKA representation similarity for TRAM. We measure each metric using the XNLI validation set and report for the training language (En) and the zero-shot languages (ZS). We report mean of 20 runs 
±
 standard deviation across languages and the Pearson correlation between En and ZS Avg. 
𝜖
-sharpness across runs.
(a) 
𝜖
-sharpness 
↓
	En	ZS Avg.	Pearson 
𝜌

Adam	2.16	1.98
±
 0.79	0.29
±
0.20
SAM	1.43	3.32
±
 0.96	0.26
±
0.34
ASAM	2.57	2.22
±
 0.79	0.38
±
0.12
FSAM	2.34	2.62
±
 0.29	0.27
±
0.71
TRPO	6.17	2.36
±
 1.02	0.52
±
0.25
R3F	6.22	2.56
±
 1.21	0.50
±
0.12
MESA	2.76	5.48
±
 0.75	0.21
±
0.25
TRAM-
𝜃
𝑡
−
1
	
0.50
	
1.19
±
0.38	0.60
±
0.15
TRAM-
𝜃
0
	0.75	1.92
±
0.24	0.58
±
0.27
TRAM-
𝑥
	0.61	1.49
±
 0.49	
0.75
±
0.18
TRAM-Fisher	1.67	2.02
±
 0.40	0.42
±
0.37
(b) CKA 
↑
	En	ZS Avg.
Adam	0.69	0.44
±
 0.10
SAM	0.69	0.42
±
 0.10
ASAM	0.69	0.42
±
 0.10
FSAM	0.73	0.48
±
 0.10
TRPO	0.70	0.45
±
 0.10
R3F	0.66	0.40
±
 0.10
MESA	0.67	0.42
±
 0.10
TRAM-
𝜃
𝑡
−
1
	
0.77
	
0.57
±
 0.10
TRAM-
𝜃
0
	0.69	0.45
±
 0.11
TRAM-
𝑥
	0.75	0.54
±
 0.11
TRAM-Fisher	0.72	0.49
±
 0.10

Representation transfer: We analyze the similarity of pre-trained and fine-tuned representations for the same setup of XNLI. We hypothesize that if TRAM optimizes within the trust region, pre- and post-fine-tuned representations will be more similar to allow better usage of pre-trained structure. We measure this relationship using CKA similarity (Kornblith et al., 2019) defined in Section B.8. Similar to the previous analysis, we observe that TRAM produces representations that are more similar to pre-trained XLM-Roberta representations than any competitor. This applies to both the En case and the ZS Avg. case, with all other models performing similarly to the Adam baseline. Counterintuitively, trust region methods perform no better than SAM-style methods which do not explicitly target representational similarity. This observation could be related to recent insight into the smoothness side effects of training with SAM (Wen et al., 2023). We additionally raise that neither metric in Table 6 shows a similar trend to our empirical findings—comparisons here do not strictly reflect similar performance variation on specific tasks. Despite empirical improvement, recent work questions if sharpness meaningfully correlates with generalization (Juneja et al., 2023; Andriushchenko et al., 2023). Extending TRAM should further evaluate this relationship and investigate how trust region measurement could inform better predictors of generalization capability.

5Conclusion

We present TRAM: Trust Region Aware Minimization. TRAM optimizes for smoothness in both parameter and function spaces to improve domain generalization during fine-tuning. TRAM inherits the capability of SAM to optimize towards flatter minima and integrates trust region awareness to ensure low local curvature between output representations. We evaluate TRAM on out-of-distribution scenarios, where the model must generalize to new distributions unseen during training. In this setup, TRAM proves more effective than SAM-style optimization or trust region methods. Our analysis identifies how TRAM bucks the anticorrelated trend for major distribution shifts, learns a flatter out-of-domain loss surface, and improves representation similarity for data unseen during fine-tuning.

6Acknowledgments

TS gratefully acknowledges the support of the UK Engineering and Physical Sciences Research Council (grant EP/W002876/1). This work has been made possible in part by a gift from the Chan Zuckerberg Initiative Foundation to establish the Kempner Institute for the Study of Natural and Artificial Intelligence.

References
Aghajanyan et al. (2021)
↑
	Armen Aghajanyan, Akshat Shrivastava, Anchit Gupta, Naman Goyal, Luke Zettlemoyer, and Sonal Gupta.Better fine-tuning by reducing representational collapse.In International Conference on Learning Representations, 2021.URL https://openreview.net/forum?id=OQ08SN70M1V.
Ahmad et al. (2019)
↑
	Wasi Ahmad, Zhisong Zhang, Xuezhe Ma, Eduard Hovy, Kai-Wei Chang, and Nanyun Peng.On difficulties of cross-lingual transfer with order differences: A case study on dependency parsing.In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pp.  2440–2452, Minneapolis, Minnesota, June 2019. Association for Computational Linguistics.doi: 10.18653/v1/N19-1253.URL https://aclanthology.org/N19-1253.
Amari (1998)
↑
	Shun-ichi Amari.Natural Gradient Works Efficiently in Learning.Neural Computation, 10(2):251–276, 02 1998.ISSN 0899-7667.doi: 10.1162/089976698300017746.URL https://doi.org/10.1162/089976698300017746.
Andriushchenko et al. (2023)
↑
	Maksym Andriushchenko, Francesco Croce, Maximilian Müller, Matthias Hein, and Nicolas Flammarion.A modern look at the relationship between sharpness and generalization.In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pp.  840–902. PMLR, 23–29 Jul 2023.URL https://proceedings.mlr.press/v202/andriushchenko23a.html.
Bahri et al. (2022)
↑
	Dara Bahri, Hossein Mobahi, and Yi Tay.Sharpness-aware minimization improves language model generalization.In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp.  7360–7371, Dublin, Ireland, May 2022. Association for Computational Linguistics.doi: 10.18653/v1/2022.acl-long.508.URL https://aclanthology.org/2022.acl-long.508.
Chen et al. (2023)
↑
	Xiangning Chen, Chen Liang, Da Huang, Esteban Real, Kaiyuan Wang, Yao Liu, Hieu Pham, Xuanyi Dong, Thang Luong, Cho-Jui Hsieh, Yifeng Lu, and Quoc V. Le.Symbolic discovery of optimization algorithms, 2023.URL https://arxiv.org/abs/2302.06675.
Chronopoulou et al. (2022)
↑
	Alexandra Chronopoulou, Matthew Peters, and Jesse Dodge.Efficient hierarchical domain adaptation for pretrained language models.In Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp.  1336–1351, Seattle, United States, July 2022. Association for Computational Linguistics.doi: 10.18653/v1/2022.naacl-main.96.URL https://aclanthology.org/2022.naacl-main.96.
Chronopoulou et al. (2023)
↑
	Alexandra Chronopoulou, Matthew Peters, Alexander Fraser, and Jesse Dodge.AdapterSoup: Weight averaging to improve generalization of pretrained language models.In Findings of the Association for Computational Linguistics: EACL 2023, pp.  2054–2063, Dubrovnik, Croatia, May 2023. Association for Computational Linguistics.URL https://aclanthology.org/2023.findings-eacl.153.
Conneau et al. (2018)
↑
	Alexis Conneau, Ruty Rinott, Guillaume Lample, Adina Williams, Samuel Bowman, Holger Schwenk, and Veselin Stoyanov.XNLI: Evaluating cross-lingual sentence representations.In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pp.  2475–2485, Brussels, Belgium, October-November 2018. Association for Computational Linguistics.doi: 10.18653/v1/D18-1269.URL https://aclanthology.org/D18-1269.
Conneau et al. (2020a)
↑
	Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov.Unsupervised cross-lingual representation learning at scale.In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pp.  8440–8451, Online, July 2020a. Association for Computational Linguistics.doi: 10.18653/v1/2020.acl-main.747.URL https://aclanthology.org/2020.acl-main.747.
Conneau et al. (2020b)
↑
	Alexis Conneau, Shijie Wu, Haoran Li, Luke Zettlemoyer, and Veselin Stoyanov.Emerging cross-lingual structure in pretrained language models.In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pp.  6022–6034, Online, July 2020b. Association for Computational Linguistics.doi: 10.18653/v1/2020.acl-main.536.URL https://aclanthology.org/2020.acl-main.536.
Deng et al. (2009)
↑
	Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei.Imagenet: A large-scale hierarchical image database.In 2009 IEEE Conference on Computer Vision and Pattern Recognition, pp.  248–255, 2009.doi: 10.1109/CVPR.2009.5206848.
Dosovitskiy et al. (2021)
↑
	Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby.An image is worth 16x16 words: Transformers for image recognition at scale.In International Conference on Learning Representations, 2021.URL https://openreview.net/forum?id=YicbFdNTTy.
Du et al. (2022)
↑
	Jiawei Du, Zhou Daquan, Jiashi Feng, Vincent Tan, and Joey Tianyi Zhou.Sharpness-aware training for free.In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), Advances in Neural Information Processing Systems, 2022.URL https://openreview.net/forum?id=xK6wRfL2mv7.
Foret et al. (2021)
↑
	Pierre Foret, Ariel Kleiner, Hossein Mobahi, and Behnam Neyshabur.Sharpness-aware minimization for efficiently improving generalization.In International Conference on Learning Representations, 2021.URL https://openreview.net/forum?id=6Tm1mposlrM.
Frankle (2020)
↑
	Jonathan Frankle.Revisiting "qualitatively characterizing neural network optimization problems".ArXiv, abs/2012.06898, 2020.URL https://api.semanticscholar.org/CorpusID:229152287.
French (1999)
↑
	Robert M. French.Catastrophic forgetting in connectionist networks.Trends in Cognitive Sciences, 3(4):128–135, 1999.ISSN 1364-6613.doi: https://doi.org/10.1016/S1364-6613(99)01294-2.URL https://www.sciencedirect.com/science/article/pii/S1364661399012942.
Gretton et al. (2012)
↑
	Arthur Gretton, Karsten M. Borgwardt, Malte J. Rasch, Bernhard Schölkopf, and Alexander Smola.A kernel two-sample test.Journal of Machine Learning Research, 13(25):723–773, 2012.URL http://jmlr.org/papers/v13/gretton12a.html.
He et al. (2021)
↑
	Pengcheng He, Xiaodong Liu, Jianfeng Gao, and Weizhu Chen.{DEBERTA}: {DECODING}-{enhanced} {bert} {with} {disentangled} {attention}.In International Conference on Learning Representations, 2021.URL https://openreview.net/forum?id=XPZIaotutsD.
Hochreiter & Schmidhuber (1994)
↑
	Sepp Hochreiter and Jürgen Schmidhuber.Simplifying neural nets by discovering flat minima.In G. Tesauro, D. Touretzky, and T. Leen (eds.), Advances in Neural Information Processing Systems, volume 7. MIT Press, 1994.URL https://proceedings.neurips.cc/paper_files/paper/1994/file/01882513d5fa7c329e940dda99b12147-Paper.pdf.
Ioffe & Szegedy (2015)
↑
	Sergey Ioffe and Christian Szegedy.Batch normalization: Accelerating deep network training by reducing internal covariate shift.In Francis Bach and David Blei (eds.), Proceedings of the 32nd International Conference on Machine Learning, volume 37 of Proceedings of Machine Learning Research, pp.  448–456, Lille, France, 07–09 Jul 2015. PMLR.URL https://proceedings.mlr.press/v37/ioffe15.html.
Izmailov et al. (2018)
↑
	Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, and Andrew Gordon Wilson.Averaging weights leads to wider optima and better generalization.In Conference on Uncertainty in Artificial Intelligence, 2018.URL http://arxiv.org/abs/1803.05407.
Jiang et al. (2020)
↑
	Haoming Jiang, Pengcheng He, Weizhu Chen, Xiaodong Liu, Jianfeng Gao, and Tuo Zhao.SMART: Robust and efficient fine-tuning for pre-trained natural language models through principled regularized optimization.In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pp.  2177–2190, Online, July 2020. Association for Computational Linguistics.doi: 10.18653/v1/2020.acl-main.197.URL https://aclanthology.org/2020.acl-main.197.
Juneja et al. (2023)
↑
	Jeevesh Juneja, Rachit Bansal, Kyunghyun Cho, João Sedoc, and Naomi Saphra.Linear connectivity reveals generalization strategies.In The Eleventh International Conference on Learning Representations, 2023.URL https://openreview.net/forum?id=hY6M0JHl3uL.
Kaddour et al. (2022)
↑
	Jean Kaddour, Linqing Liu, Ricardo Silva, and Matt J Kusner.When do flat minima optimizers work?In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh (eds.), Advances in Neural Information Processing Systems, volume 35, pp.  16577–16595. Curran Associates, Inc., 2022.URL https://proceedings.neurips.cc/paper_files/paper/2022/file/69b5534586d6c035a96b49c86dbeece8-Paper-Conference.pdf.
Keskar et al. (2017)
↑
	Nitish Shirish Keskar, Dheevatsa Mudigere, Jorge Nocedal, Mikhail Smelyanskiy, and Ping Tak Peter Tang.On large-batch training for deep learning: Generalization gap and sharp minima.In International Conference on Learning Representations, 2017.URL https://openreview.net/forum?id=H1oyRlYgg.
Kim et al. (2022)
↑
	Minyoung Kim, Da Li, Shell X Hu, and Timothy Hospedales.Fisher SAM: Information geometry and sharpness aware minimisation.In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato (eds.), Proceedings of the 39th International Conference on Machine Learning, volume 162 of Proceedings of Machine Learning Research, pp.  11148–11161. PMLR, 17–23 Jul 2022.URL https://proceedings.mlr.press/v162/kim22f.html.
Kingma & Ba (2017)
↑
	Diederik P. Kingma and Jimmy Ba.Adam: A method for stochastic optimization, 2017.
Kornblith et al. (2019)
↑
	Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton.Similarity of neural network representations revisited.In Kamalika Chaudhuri and Ruslan Salakhutdinov (eds.), Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pp. 3519–3529. PMLR, 09–15 Jun 2019.URL https://proceedings.mlr.press/v97/kornblith19a.html.
Krause et al. (2013)
↑
	Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei.3d object representations for fine-grained categorization.In 2013 IEEE International Conference on Computer Vision Workshops, pp.  554–561, 2013.doi: 10.1109/ICCVW.2013.77.
Krizhevsky (2009)
↑
	Alex Krizhevsky.Learning multiple layers of features from tiny images.pp.  32–33, 2009.URL https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf.
Kwon et al. (2021)
↑
	Jungmin Kwon, Jeongseop Kim, Hyunseo Park, and In Kwon Choi.Asam: Adaptive sharpness-aware minimization for scale-invariant learning of deep neural networks.In Marina Meila and Tong Zhang (eds.), Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, pp.  5905–5914. PMLR, 18–24 Jul 2021.URL https://proceedings.mlr.press/v139/kwon21b.html.
Liang et al. (2020)
↑
	Yaobo Liang, Nan Duan, Yeyun Gong, Ning Wu, Fenfei Guo, Weizhen Qi, Ming Gong, Linjun Shou, Daxin Jiang, Guihong Cao, Xiaodong Fan, Ruofei Zhang, Rahul Agrawal, Edward Cui, Sining Wei, Taroon Bharti, Ying Qiao, Jiun-Hung Chen, Winnie Wu, Shuguang Liu, Fan Yang, Daniel Campos, Rangan Majumder, and Ming Zhou.XGLUE: A new benchmark dataset for cross-lingual pre-training, understanding and generation.In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp.  6008–6018, Online, November 2020. Association for Computational Linguistics.doi: 10.18653/v1/2020.emnlp-main.484.URL https://aclanthology.org/2020.emnlp-main.484.
Lo et al. (2020)
↑
	Kyle Lo, Lucy Lu Wang, Mark Neumann, Rodney Kinney, and Daniel Weld.S2ORC: The semantic scholar open research corpus.In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pp.  4969–4983, Online, July 2020. Association for Computational Linguistics.doi: 10.18653/v1/2020.acl-main.447.URL https://aclanthology.org/2020.acl-main.447.
Miller et al. (2021)
↑
	John Miller, Rohan Taori, Aditi Raghunathan, Shiori Sagawa, Pang Wei Koh, Vaishaal Shankar, Percy Liang, Yair Carmon, and Ludwig Schmidt.Accuracy on the line: On the strong correlation between out-of-distribution and in-distribution generalization, 2021.
Möllenhoff & Khan (2023)
↑
	Thomas Möllenhoff and Mohammad Emtiyaz Khan.SAM as an optimal relaxation of bayes.In The Eleventh International Conference on Learning Representations, 2023.URL https://openreview.net/forum?id=k4fevFqSQcX.
Nilsback & Zisserman (2008)
↑
	Maria-Elena Nilsback and Andrew Zisserman.Automated flower classification over a large number of classes.In 2008 Sixth Indian Conference on Computer Vision, Graphics & Image Processing, pp.  722–729, 2008.doi: 10.1109/ICVGIP.2008.47.
Radford et al. (2019)
↑
	Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al.Language models are unsupervised multitask learners.OpenAI blog, 1(8):9, 2019.
Raffel et al. (2020)
↑
	Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu.Exploring the limits of transfer learning with a unified text-to-text transformer.Journal of Machine Learning Research, 21(140):1–67, 2020.URL http://jmlr.org/papers/v21/20-074.html.
Reid et al. (2022)
↑
	Machel Reid, Victor Zhong, Suchin Gururangan, and Luke Zettlemoyer.M2D2: A massively multi-domain language modeling dataset.In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp.  964–975, Abu Dhabi, United Arab Emirates, December 2022. Association for Computational Linguistics.URL https://aclanthology.org/2022.emnlp-main.63.
Schulman et al. (2015)
↑
	John Schulman, Sergey Levine, Pieter Abbeel, Michael Jordan, and Philipp Moritz.Trust region policy optimization.In Francis Bach and David Blei (eds.), Proceedings of the 32nd International Conference on Machine Learning, volume 37 of Proceedings of Machine Learning Research, pp.  1889–1897, Lille, France, 07–09 Jul 2015. PMLR.URL https://proceedings.mlr.press/v37/schulman15.html.
Srivastava et al. (2014)
↑
	Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov.Dropout: A simple way to prevent neural networks from overfitting.Journal of Machine Learning Research, 15(56):1929–1958, 2014.URL http://jmlr.org/papers/v15/srivastava14a.html.
Teney et al. (2022)
↑
	Damien Teney, Seong Joon Oh, and Ehsan Abbasnejad.Id and ood performance are sometimes inversely correlated on real-world datasets.ArXiv, abs/2209.00613, 2022.URL https://api.semanticscholar.org/CorpusID:251979643.
Wang et al. (2019)
↑
	Alex Wang, Amanpreet Singh, Julian Michael, Felix Hill, Omer Levy, and Samuel R. Bowman.GLUE: A multi-task benchmark and analysis platform for natural language understanding.In International Conference on Learning Representations, 2019.URL https://openreview.net/forum?id=rJ4km2R5t7.
Wen et al. (2023)
↑
	Kaiyue Wen, Zhiyuan Li, and Tengyu Ma.Sharpness minimization algorithms do not only minimize sharpness to achieve better generalization, 2023.
Williams et al. (2018)
↑
	Adina Williams, Nikita Nangia, and Samuel Bowman.A broad-coverage challenge corpus for sentence understanding through inference.In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long Papers), pp.  1112–1122, New Orleans, Louisiana, June 2018. Association for Computational Linguistics.doi: 10.18653/v1/N18-1101.URL https://aclanthology.org/N18-1101.
Zhu et al. (2020)
↑
	Chen Zhu, Yu Cheng, Zhe Gan, Siqi Sun, Tom Goldstein, and Jingjing Liu.Freelb: Enhanced adversarial training for natural language understanding.In International Conference on Learning Representations, 2020.URL https://openreview.net/forum?id=BygzbyHFvB.
Appendix AData Splits
Vision Datasets

For vision modality experiments, we evaluate cross-dataset transfer from ImageNet (Deng et al., 2009) to CIFAR-100 (Krizhevsky, 2009), Stanford Cars (Krause et al., 2013), and Oxford Flowers (Nilsback & Zisserman, 2008). We source all datasets from HuggingFace2 using the default training/testing partitions.

Language Datasets

We evaluate the M2D2 dataset (Reid et al., 2022) for cross-domain language modeling. M2D2 contains two groups: 11 domains from the S2ORC corpus of ArXiv listings (Lo et al., 2020) and an archive of Wikipedia articles. We train a language model on each split’s largest domain and evaluate zero-shot generalization to ten domains unseen during fine-tuning. Evaluation uses token-level perplexity across each domain. Table 7 details the partition sizes (in tokens) for each domain in M2D2.

Zero-shot cross-lingual transfer is evaluated using MultiNLI and XNLI for entailment classification. In this task, a model predicts an entailment label (neutral, entailment, contradiction) between sentence pairs. We use only English language MultiNLI (Williams et al., 2018) for training data and evaluate the trained model on the 14 unseen natural languages in XNLI (Conneau et al., 2018) during test time. These datasets are balanced in label classes and we report accuracy per language in our results. A complete breakdown of partition sizes is shown in Table 8.

Table 7:Data splits for M2D2 (Reid et al., 2022) across Wikipedia and S2ORC (Lo et al., 2020). For simplicity, we do not consider the fine-grained subdomains in each domain. All data sourced from Huggingface (huggingface.co/datasets/machelreid/m2d2)
Split	Domain	Abbrev.	Size (Tokens)	Training Domain	Train Tokens	Validation Tokens	Test Tokens
Wiki	Culture and the arts	Cult.	289M		—	—	34.33M
General reference	Gen.	60M		—	—	2.38M
Health and fitness	Health.	116M		—	—	6.83M
History and events	Hist.	226M		—	—	11.65M
Human activities	Human.	343M		—	—	12.41M
Mathematics and logic	Math.	52M		—	—	1.65M
Natural and physical sciences	Nat.	189M		—	—	13.45M
Philosophy and thinking	Phil.	165M		—	—	2.32M
Religion and belief systems	Rel	64M		—	—	5.44M
Society and social sciences	Soc.	397M	✓	380M	11.8M	11.74M
Technology and applied sciences	Tech.	297M		—	—	11.78M
S2ORC	Art	Art	98M		—	—	1.06M
Astrophysics	Astro	728M		—	—	1.14M
Condensed matter	CondM.	688M		—	—	1.17M
Computer science	CS	1.1B		—	—	1.17M
Economics	Econ.	11M		—	—	1.16M
Mathematics	Math	1.4B	✓	1.1B	1.46M	1.40M
Nonlinear sciences	NLin.	134M		—	—	1.28M
Philosophy	Phil.	156M		—	—	1.06M
Physics	Phys.	737M		—	—	1.12M
Quantitative biology	QBio	336M		—	—	1.08M
Statistics	Stat	450M		—	—	1.19M
Table 8:Data splits for XNLI (Conneau et al., 2018). The Training data in English is sourced from the MultiNLI dataset (Williams et al., 2018) with translations provided for XNLI. Model selection during training uses only the English validation data. Validation data for other languages is used to measure 
𝜖
-sharness in our analysis. We omit data splits not used in this work. All data sourced from HuggingFace (huggingface.co/datasets/xnli).
XNLI	Abbrev.	Train Sentences	Validation Sentences	Test Sentences
English	En	393K	2.5K	5K
Arabic	Ar	—	2.5K	5K
Bulgarian	Bg	—	2.5K	5K
German	De	—	2.5K	5K
Greek	El	—	2.5K	5K
Spanish	Es	—	2.5K	5K
French	Fr	—	2.5K	5K
Hindi	Hi	—	2.5K	5K
Russian	Ru	—	2.5K	5K
Swahili	Sw	—	2.5K	5K
Thai	Th	—	2.5K	5K
Turkish	Tr	—	2.5K	5K
Urdu	Ur	—	2.5K	5K
Vietnamese	Vi	—	2.5K	5K
Chinese (Simplified)	Zh	—	—	5K
Appendix BAdditional Experimental Details
B.1Model training

We fine-tune each pre-trained model without any freezing or additional task-specific parameters where possible. We also do not explore fine-tuning with low-rank approximations or adapters i.e., ‘full fine-tuning’. This setup isolates the contribution of the optimization algorithm over additional capacity in the model. For image classification and cross-lingual entailment classification, we follow fine-tuning norms and only introduce a new dataset-specific ‘head’ to predict dataset-specific logits. For language tasks, we fine-tune each pre-trained model for 50,000 steps using an initial learning rate of 
2
×
10
−
5
, a polynomial decay schedule, and 10,000 step learning rate warmup. We use Adam (Kingma & Ba, 2017), with a decay factor setting 
(
𝛽
1
,
𝛽
2
)
=
(
0.9
,
0.99
)
, as the base optimizer for each SAM-style and TR method unless mentioned otherwise. When using validation loss for model selection, we use only the validation partition of the training domain to reflect a stricter evaluation setup without access to additional domains during training. All models are trained 
1
×
A100 80GB GPU for under 72 hours except for GPT2-XL experiments in Section C.2.

B.2Baselines

We compare to a naive SGD baseline for vision experiments following Kim et al. (2022). Our naive baseline for language experiments is Adam (Kingma & Ba, 2017) without any augmentation setting decay factors as 
(
𝛽
1
,
𝛽
2
)
=
(
0.9
,
0.99
)
. All algorithms listed below use Adam as the inner optimizer for the final update (e.g., Algorithm 1 Step 6).

For sharpness-aware methods: we compare to SAM (
𝜌
=
0.05
, Foret et al., 2021), Adaptive SAM (ASAM, 
𝜌
=
0.5
, Kwon et al., 2021) and Fisher SAM (FSAM, 
𝛾
=
0.1
,
𝜂
=
0.1
, Kim et al., 2022).

For trust region methods: we compare to Trust Region Policy Optimization (TRPO, Schulman et al., 2015), R3F (
𝜎
=
0.1
, Aghajanyan et al., 2021), and MESA (Du et al., 2022). MESA is a variant of TRPO regularizing output representation divergence between current 
𝜃
𝑡
 and the exponential moving average of previous 
𝜃
<
𝑡
 with decay factor 0.999. For trust-region methods, we add the regularizer directly to the task-specific loss function with a weighting coefficient of 
𝜆
=
0.1
 (in Equation 3).

B.3Cross-dataset transfer for image classification

We implement the same cross-dataset adaptation setup as Kim et al. (2022) as a ‘sanity check’ directly comparing TRAM to prior methods in the same setting. This setup is not strictly similar to the ‘out-of-distribution’ scenario we report for language tasks—this experiment verifies that TRAM is performant on standard benchmarks and valuably evaluates TRAM in the vision modality. The objective is to adapt ViT-base (Dosovitskiy et al., 2021) from ImageNet pre-training (Deng et al., 2009) to additional image classification tasks. We evaluate dataset adaptation to CIFAR-100 (Krizhevsky, 2009), Oxford Flowers (Nilsback & Zisserman, 2008) and Stanford Cars (Krause et al., 2013) datasets. Our hypothesis is that TRAM can improve applying information from ImageNet to additional datasets with different labels and input data.

We match the experimental setting of Kim et al. (2022): fine-tuning ViT-base-16 for 200 epochs with a base optimizer of SGD, an initial learning rate of 
5
×
10
−
4
, and a cosine learning rate decay with no warmup or restarts. We do not use early stopping to match prior work and use the final model regardless of validation loss. We report the average Top-1 accuracy over 5 runs, 
±
 the 95% confidence interval, in Table 3 for direct comparison to Kim et al. (2022, Table 3).

B.4Cross-domain language modeling

We consider zero-shot cross-domain language modeling using the M2D2 Corpus (Reid et al., 2022). Our hypothesis is that TRAM can better apply language modeling information from large text corpora to improve out-of-domain perplexity when fine-tuning to a specific domain. For S2ORC, we train on the “Math” domain (Math, 1.4B tokens) and for Wikipedia, we train on the “Society and social sciences” domain (Soc., 379M tokens). We use the 112M parameter GPT-2 base model (Radford et al., 2019) with a batch size of 16 blocks of 1024 tokens following the setup of prior work (Reid et al., 2022; Chronopoulou et al., 2022; 2023). We evaluate generalization via perplexity for each test domain. We also evaluate a zero-shot baseline (i.e., GPT-2 before fine-tuning) to contrast with the same model before domain-specific adaptation. To reduce computation, we train one model with one random seed per algorithm.

B.5Zero-shot cross-lingual transfer

We test zero-shot cross-lingual transfer by fine-tuning a multilingual model on an English task and then evaluating the model in other languages. We hypothesize that TRAM can improve task transfer across languages by improving the usage of information from multilingual pre-training during monolingual fine-tuning. A poorer model may ‘forget’ other languages during the adaptation process. We evaluate transfer from English to additional languages by predicting labels for the XNLI test set after training the model for NLI only in English. We use the 250M XLM-Roberta Base multilingual pre-trained model (Conneau et al., 2020a) with a classification head trained from scratch. This model uses a batch size of 32 examples using only English validation loss for model selection. Each reported result is averaged across 20 runs of varying random seeds to control for variation in loss surface.

B.6Training algorithms
Algorithm 1 Trust Region Aware Minimization
  Input: Training set 
𝑆
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
, loss function 
ℓ
, learning rate 
𝛼
, model parameters 
𝜃
, noise standard deviation 
𝜎
 {if noise-estimated trust region}.
  for 
𝑡
=
1
,
2
,
…
 do
     (1) Sample batch of 
𝐵
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
0
|
𝐵
|
 data from 
𝑆
.
     (2) Compute the predictive distribution, 
𝑝
𝑓
(
⋅
|
𝑥
𝐵
,
𝜃
𝑡
)
, and gradient of the batch loss 
∇
𝐿
𝐵
⁢
(
𝜃
)
.
     (3) Compute trust region distance 
𝑑
 as:      
𝑑
𝜃
 using 
𝑝
𝑓
(
⋅
|
𝑥
𝐵
,
𝜃
𝑡
−
1
)
 (Equation 4) or      
𝑑
𝑥
 using 
𝑝
𝑓
(
⋅
|
𝑥
𝐵
+
𝑧
,
𝜃
𝑡
)
,
𝑧
∼
𝑁
(
0
,
𝜎
2
)
 (Equation 5).
     (4) Compute 
𝜖
𝑇
⁢
𝑅
⁢
𝐴
⁢
𝑀
∗
:       
𝜖
TRAM
∗
=
𝑑
⁢
𝜃
2
⁢
∇
𝐿
𝑆
⁢
(
𝜃
𝑡
)
/
‖
𝜃
⁢
∇
𝐿
𝑆
⁢
(
𝜃
𝑡
)
‖
2
     (5) Ascent step perturbing 
𝜃
 to 
𝜃
+
𝜖
𝑇
⁢
𝑅
⁢
𝐴
⁢
𝑀
∗
.
     (6) Compute gradient at 
𝜃
+
𝜖
𝑇
⁢
𝑅
⁢
𝐴
⁢
𝑀
∗
 as Equation 6:       
∇
𝐿
TRAM
(
𝜃
)
=
∂
𝐿
𝑆
∂
𝜃
|
𝜃
=
𝜃
+
𝜖
TRAM
∗
     (7) Gradient descent update: 
𝜃
←
𝜃
−
𝛼
⁢
∇
𝐿
TRAM
⁢
(
𝜃
)
.
  end for
 
Algorithm 2 Trust Region Aware Minimization with Fisher Information Matrix (TRAM-Fisher)
  Input: Training set 
𝑆
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
, loss function 
ℓ
, learning rate 
𝛼
, model parameters 
𝜃
, noise standard deviation 
𝜎
  for 
𝑡
=
1
,
2
,
…
 do
     1) Sample batch of 
𝐵
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
0
|
𝐵
|
 data from 
𝑆
.
     2) Compute the predictive distribution, 
𝑝
𝑓
(
⋅
|
𝑥
𝐵
,
𝜃
𝑡
)
, and gradient of the batch loss 
∇
𝐿
𝐵
⁢
(
𝜃
)
.
     3) Sample input noise 
𝑧
∼
𝑁
⁢
(
0
,
𝜎
2
⁢
𝐼
|
𝜃
|
)
.
     4) Approximate the Fisher Information Matrix at 
𝑥
+
𝑧
:        
𝐹
^
⁢
(
𝑥
+
𝑧
;
𝜃
)
=
Diag
⁢
(
1
|
𝐵
|
⁢
∑
𝑖
∈
𝐵
(
log
⁡
𝑝
𝑓
⁢
(
𝑦
𝑖
|
𝑥
𝑖
+
𝑧
𝑖
,
𝜃
)
)
)
2
     5) Compute 
𝜖
TRAM
−
F
∗
:       
𝜖
TRAM
−
F
∗
=
𝐹
^
⁢
(
𝑥
+
𝑧
;
𝜃
)
−
1
⁢
∇
𝐿
𝑆
∇
𝐿
𝑆
⁢
𝐹
^
⁢
(
𝑥
+
𝑧
;
𝜃
)
−
1
⁢
∇
𝐿
𝑆
.
     6) Ascent step perturbing 
𝜃
 to 
𝜃
+
𝜖
𝑇
⁢
𝑅
⁢
𝐴
⁢
𝑀
−
𝐹
∗
.
     7) Compute gradient at 
𝜃
+
𝜖
𝑇
⁢
𝑅
⁢
𝐴
⁢
𝑀
−
𝐹
∗
 as Equation 6:       
∇
𝐿
TRAM
−
F
(
𝜃
)
=
∂
𝐿
𝑆
∂
𝜃
|
𝜃
=
𝜃
+
𝜖
TRAM
∗
     8) Gradient descent update: 
𝜃
←
𝜃
−
𝛼
⁢
∇
𝐿
TRAM
−
F
⁢
(
𝜃
)
.
  end for

The training algorithm for TRAM is outlined in Algorithm 1 using different metrics for trust region estimation, 
𝑑
, outlined in Section 3. Algorithm 2 details the TRAM-Fisher algorithm. Practically, this modifies Algorithm 1 in removing one forward pass to estimate the trust region distance and instead approximate the Fisher Information Matrix of the trust region neighborhood in representation space.

B.7Measuring sharpness

We follow Keskar et al. (2017) in evaluating model 
𝜖
-sharpness as Equation 10 where 
ℓ
 is the loss function, 
𝑥
∈
ℝ
𝑛
 are 
𝑛
 model parameters, 
𝐴
∈
ℝ
𝑛
×
𝑝
 is a matrix restricting the 
𝜖
-sharpness to a subspace of 
𝑝
 parameters (
𝐴
+
 is the pseudo-inverse of 
𝐴
) and 
𝒞
𝜀
 is defined as Equation 11 denoting a “box” region around the solution over which loss is maximized.

	
𝜙
𝑥
,
𝑓
⁢
(
𝜖
,
𝐴
)
	
:=
max
𝑦
∈
𝒞
𝜖
⁡
ℓ
⁢
(
𝑥
+
𝐴
⁢
𝑦
)
−
ℓ
⁢
(
𝑥
)
1
+
ℓ
⁢
(
𝑥
)
×
100
		
(10)

	
𝒞
𝜖
	
=
{
𝑧
∈
ℝ
𝑝
:
−
𝜖
⁢
(
|
(
𝐴
+
⁢
𝑥
)
𝑖
|
+
1
)
≤
𝑧
𝑖
≤
𝜖
⁢
(
|
(
𝐴
+
⁢
𝑥
)
𝑖
|
+
1
)
⁢
∀
𝑖
∈
[
𝑝
]
}
		
(11)

For our measurement of 
𝜖
-sharpness, we set 
𝐴
 to the identity matrix 
𝐼
𝑛
×
𝑛
 to measure over the complete model. We measure 
𝜖
-sharpness over the validation set of XNLI in all languages comparing between original loss 
ℓ
⁢
(
𝑥
)
 and maximized loss 
max
𝑦
∈
𝒞
𝜖
⁡
ℓ
⁢
(
𝑥
+
𝐴
⁢
𝑦
)
. We follow the 
𝜖
-sharpness setup of Juneja et al. (2023) using an SGD optimizer, learning rate of 
8
×
10
−
5
, a 32 example batch size, accumulation over 4 steps and 
𝜖
 of 
1
×
10
−
5
.

B.8Measuring representation similarity

We follow Kornblith et al. (2019) and Conneau et al. (2020b) in evaluating cross-lingual similarity using Centered Kernel Alignment (CKA). At a language level, CKA computes a similarity score between matrix 
𝑋
 and 
𝑌
 where 
𝑋
,
𝑌
∈
ℝ
𝑛
×
𝑑
 are dense matrices of 
𝑛
 outputs of 
𝑑
-dimensional representations from each model. We compute linear CKA similarity as Equation 12 using the Frobenius norm. For our cross-lingual transfer experiments, we use the base model output for each example (i.e., the representation before the classification head) to evaluate similarity.

	
CKA
⁢
(
𝑋
,
𝑌
)
=
∥
𝑌
𝑇
⁢
𝑋
∥
𝐹
2
∥
𝑋
𝑇
⁢
𝑋
∥
𝐹
⁢
∥
𝑌
𝑇
⁢
𝑌
∥
𝐹
		
(12)
Appendix CAdditional results
C.1Domain correlations for S2ORC
Table 9:Pearson correlation between training domains and zero-shot domains for M2D2. We report how the change in training domain correlates with changes in zero-shot perplexity to analyze how different domains improve or worsen during fine-tuning. All domains are correlated with Soc. for the Wikipedia split. Art and Phil. domains are anti-correlated with Math training domain for S2ORC indicating a major distribution shift.
Wiki Domain	
𝜌
 to Soc.	
𝑝
<
0.01
⁢
?
	S2ORC Domain	
𝜌
 to Math	
𝑝
<
0.01
⁢
?
	STEM?
Cult.	0.982	✓	Art	-0.861	✓	
Gen.	0.983	✓	Astro	0.812	✓	✓
Health.	0.970	✓	CondM.	0.999	✓	✓
Hist.	0.998	✓	CS	0.996	✓	✓
Human.	0.980	✓	Econ.	0.997	✓	✓
Math.	0.976	✓	NLin.	1.000	✓	✓
Nat.	0.982	✓	Phil.	-0.825	✓	
Phil.	0.985	✓	Phys.	0.991	✓	✓
Rel	0.994	✓	QBio	0.932	✓	✓
Tech.	0.983	✓	Stat	0.998	✓	✓
ZS Avg.	0.990	✓	ZS Avg.	0.968	✓	
			STEM Avg	0.998	✓	

Table 9 details the correlation between zero-shot and training domain perplexity across methods. We omit the combination approaches (e.g., ASAM+R3F) due to poor performance. For Wikipedia, all domains are correlated with the training domain indicating that the domain-specific fine tuning on Soc. domain has a net positive improvement on all zero-shot domains. This trend is not consistent for S2ORC where we observe that Art and Phil. domains are anti-correlated with the Math training domain. Improvement to Math perplexity worsens the performance on these domains across all methods. As discussed in Section 4.2.1, TRAM reports perplexity below this trend to perform better than expected for a negatively correlated trend. For comparison, we contrast the correlations between positively correlated domains (grouped as an average entitled STEM) and anticorrelated domains in Figure 2.

C.2Training GPT2-XL with TRAM
Table 10:M2D2 perplexity across training algorithms for GPT2-XL. We fine-tune on the Math domain M2D2 S2ORC split and evaluate in-domain and out-of-domain perplexity. We evaluate TRAM, competitive comparisons and a GPT2-XL zero-shot baseline. We omit algorithms demonstrating poorer results in smaller scale experiments to limit computation demands. As in Table 4, TRAM performs strongly compared to all comparisons. We report the average zero-shot perplexity (ZS Avg.) as the summary metric to judge domain transfer capability (lower is better). Worst perplexity (excluding zero-shot) is red, best is green.
S2ORC	Math	Art	Phil.	Astro	CondM.	CS	Econ.	NLin.	Phys.	QBio	Stat	ZS Avg. 
↓

GPT2-XL	16.9	22.8	21.2	19.8	19.0	17.3	18.5	17.8	20.7	19.8	14.9	19.2
Adam	8.7	
30.4
	
28.2
	
24.0
	
14.9
	
15.4
	15.4	11.4	
21.4
	22.1	12.6	19.6
SAM	8.7	29.3	28.0	22.6	14.6	15.1	15.1	11.2	20.5	21.4	12.3	19.0
ASAM	7.9	28.0	26.1	21.8	13.4	14.1	14.1	10.4	19.4	20.3	11.4	17.9
FSAM	
7.8
	26.7	25.0	21.1	
13.1
	
13.7
	
13.7
	
10.2
	
18.8
	19.6	
11.2
	
17.3

TRPO	8.9	27.9	26.4	23.0	
14.9
	15.3	15.3	11.5	20.8	21.3	12.5	18.9
R3F	8.9	27.9	26.4	23.0	
14.9
	15.3	15.3	11.5	20.8	21.3	12.5	18.9
MESA	
9.1
	28.7	26.7	23.7	14.8	15.0	
16.3
	
13.1
	20.7	
23.2
	
12.8
	19.5
TRAM-
𝜃
𝑡
−
1
	8.3	
25.2
	
23.8
	
20.1
	13.7	14.0	14.2	10.7	18.9	
19.4
	11.5	
17.2

TRAM-
𝑥
	8.3	25.3	
23.8
	20.2	13.8	14.1	14.2	10.8	19.0	19.5	11.6	
17.2

Bahri et al. (2022) report that training with SAM is effective over all sizes of T5 (Raffel et al., 2020). We verify if this improvement trend extends to TRAM by training a GPT2-XL model (1.5B parameters) on the same language modeling task for 100,000 steps. The setup is the same as described in Appendix B but we use 4 A100 GPUs for training each with a batch size per device of 4 blocks 
×
 1024 tokens. Perplexity for S2ORC domains is shown in Table 10 where we observe similar trends to the 112M parameter GPT2 model. We choose not to run these larger experiments on methods with poor performance in Table 4 (e.g., combined approaches, TRAM-Fisher) to limit computation demands. Zero-shot GPT2-XL is a stronger baseline here which some methods struggle to improve upon despite improvement in the training domain. TRAM-
𝜃
𝑡
−
1
 and TRAM-
𝑥
 perform similarly reporting the lowest perplexity in four domains. The most competitive adjacent algorithm is FSAM reporting the lowest perplexity in seven domains. The difference between FSAM and either TRAM algorithm is not significant here, as we observed for smaller models in Table 4.

C.3Results from combining optimization algorithms
Table 11:M2D2 perplexity (lower is better) on Wikipedia (upper) & S2ORC (lower) splits. TRAM-
𝜃
𝑡
−
1
 significantly improves over prior work (
𝑝
<
0.01
 Kolmogorov-Smirnov test). Results are grouped as: (i) optimizers; (ii) trust region methods; (iii) combined SAM optimizers and trust region methods; and (iv) TRAM variants. The leftmost column is the training domain and we evaluate zero-shot perplexity on ten domains unseen during fine-tuning (full details in Appendix A). ZS Avg. is the macro-average of all zero-shot domains.
Wiki	Soc.	Cult.	Gen.	Health.	Hist.	Human.	Math.	Nat.	Phil.	Rel	Tech.	ZS Avg. 
↓

GPT-2	27.2	27.7	27.8	24.5	29.2	28.8	28.6	29.4	27.8	27.7	28.7	28.0
Adam	24.8	26.3	26.4	23.6	27.2	27.0	27.4	27.6	26.3	25.8	27.4	26.5
SAM	24.5	25.9	26.0	23.1	26.9	26.6	26.6	27.2	25.8	25.5	27.0	26.1
ASAM	24.8	25.4	25.6	22.5	27.1	26.4	26.3	26.7	25.5	25.5	
28.1
	25.9
FSAM	21.7	23.0	23.3	20.6	23.9	23.7	23.8	24.0	23.1	22.8	24.0	23.2
TRPO	21.8	23.0	23.3	20.7	24.0	23.7	23.8	24.0	23.1	22.8	24.1	23.3
R3F	21.8	23.0	23.3	20.7	24.0	23.7	23.8	24.0	23.1	22.8	24.1	23.3
MESA	23.1	24.0	24.3	21.5	25.4	24.9	24.8	25.2	24.1	24.0	25.1	24.3
ASAM+TRPO	
25.6
	
26.8
	
26.9
	
24.0
	
28.0
	
27.6
	
27.6
	
28.2
	
26.8
	
26.5
	27.9	
27.0

ASAM+R3F	25.0	26.0	26.2	23.2	27.4	26.9	26.8	27.4	26.1	25.9	27.1	26.3
ASAM+MESA	25.3	26.3	26.5	23.5	27.7	27.2	27.1	27.7	26.3	26.1	27.4	26.6
TRAM-
𝜃
𝑡
−
1
	
20.9
	
22.4
	
22.7
	
20.1
	
23.1
	
22.9
	
23.2
	
23.3
	
22.4
	
22.0
	
23.4
	
22.5

TRAM-
𝜃
0
	21.9	23.1	23.4	20.7	23.9	23.3	23.9	23.8	23.1	22.7	23.9	23.2
TRAM-
𝑥
	21.9	23.1	23.4	20.7	24.0	23.3	23.9	23.9	23.2	22.7	23.9	23.2
TRAM-Fisher	22.5	23.7	24.0	21.3	24.6	24.0	24.7	24.6	23.8	23.3	24.6	23.9
S2ORC	Math	Art	Astro	CondM.	CS	Econ.	NLin.	Phil.	Phys.	QBio	Stat	ZS Avg. 
↓

GPT-2	27.6	35.8	32.4	30.9	27.9	29.5	27.6	33.7	33.5	30.9	23.4	30.6
Adam	11.4	44.2	33.9	20.1	21.2	21.0	14.7	41.9	29.5	30.8	16.9	27.4
SAM	10.5	45.3	33.2	18.7	20.3	20.0	13.7	42.4	28.3	30.2	16.1	26.8
ASAM	10.3	45.6	33.2	18.5	20.1	19.8	13.5	42.6	28.2	30.2	15.9	26.8
FSAM	10.4	45.6	33.3	18.5	20.2	19.9	13.5	42.7	28.3	30.2	15.9	26.8
TRPO	10.4	46.0	33.4	18.6	20.3	20.0	13.6	42.9	28.4	30.4	16.0	26.9
R3F	10.4	46.0	33.4	18.6	20.2	20.0	13.6	42.9	28.4	30.4	16.0	26.9
MESA	11.9	
44.1
	34.1	20.8	21.7	21.6	15.3	
41.7
	30.0	31.0	17.4	27.8
ASAM+TRPO	
13.7
	46.6	
36.9
	
23.6
	
23.8
	
23.8
	
17.4
	
43.8
	
33.1
	
33.5
	
19.2
	
30.2

ASAM+R3F	13.5	46.2	36.5	23.3	23.6	23.5	17.2	43.4	32.7	33.2	19.0	29.9
ASAM+MESA	13.4	45.9	36.3	23.1	23.4	23.3	17.0	43.2	32.5	33.0	18.9	29.7
TRAM-
𝜃
𝑡
−
1
	
9.6
	
46.8
	32.5	
17.2
	
19.2
	
18.9
	
12.6
	43.3	
27.0
	
29.6
	
15.0
	
26.2

TRAM-
𝜃
0
	10.4	44.8	33.0	18.6	20.1	19.9	13.6	42.0	28.2	30.0	15.9	26.6
TRAM-
𝑥
	10.4	44.9	33.0	18.6	20.1	19.9	13.6	42.0	28.1	30.0	15.9	26.6
TRAM-Fisher	10.5	46.1	
32.4
	18.7	20.3	20.0	13.6	43.0	28.2	30.3	16.0	26.9
Table 12:XNLI accuracy (higher is better) for training language (En) and 14 zero-shot target languages summarised by ZS Avg. (key in Appendix A). All TRAM variants significantly outperform other methods (
𝑝
<
0.01
 Wilcoxon test). Results are grouped as: (i) optimizers; (ii) trust region methods; (iii) combined SAM optimizers and trust region regularization; and (iv) TRAM variants. We report the mean across 20 seeds with standard deviation in Table 13.
	En	Ar	Bg	De	El	Es	Fr	Hi	Ru	Sw	Th	Tr	Ur	Vi	Zh	ZS Avg 
↑

Adam	
83.9
	
71.2
	
77.1
	
75.7
	
75.2
	
78.3
	77.6	69.6	
74.9
	64.6	
71.2
	
72.2
	65.8	74.1	
73.1
	
72.9

SAM	84.8	72.1	78.1	76.7	75.7	79.0	77.9	69.8	75.7	65.2	71.8	73.1	66.8	75.1	74.2	73.7
ASAM	85.0	72.0	78.4	76.9	76.1	79.5	78.5	70.4	76.1	65.2	72.5	73.4	66.9	75.5	74.2	74.0
FSAM	84.7	72.2	78.1	76.9	76.0	79.3	78.4	70.0	76.1	65.1	72.2	73.0	66.8	75.3	74.2	73.8
TRPO	84.9	71.3	77.7	76.2	75.3	78.6	
77.3
	
69.2
	75.2	64.4	71.6	72.4	
65.3
	
73.8
	73.3	73.0
R3F	85.5	72.7	78.9	77.5	76.8	79.9	79.2	70.7	76.8	66.2	72.9	73.9	66.6	75.8	74.6	74.5
MESA	84.9	71.9	77.9	76.7	75.7	78.8	77.8	69.6	75.8	
64.1
	72.1	72.4	65.7	74.4	73.9	73.3
ASAM+TRPO	85.0	72.4	78.5	77.2	76.4	79.7	78.9	70.4	76.4	65.3	72.4	73.2	66.8	75.7	74.6	74.1
ASAM+R3F	85.1	72.1	78.3	76.9	75.9	79.3	78.4	70.3	76.0	65.1	72.4	73.3	66.3	75.1	74.3	73.8
ASAM+MESA	84.7	71.7	77.8	76.3	75.7	78.8	77.9	69.5	75.4	
64.1
	71.6	72.7	65.6	74.3	73.4	73.2
TRAM-
𝜃
𝑡
−
1
	
86.2
	73.1	79.5	78.2	77.0	80.2	
79.7
	
71.5
	
77.5
	
66.4
	73.3	74.2	67.5	
76.7
	75.8	75.0
TRAM-
𝜃
0
	85.6	72.9	79.3	77.8	77.4	80.2	79.6	71.2	77.1	65.9	73.3	74.2	67.5	
76.7
	75.8	74.9
TRAM-
𝑥
	
86.2
	
73.5
	
79.8
	
78.3
	
77.5
	
80.9
	79.6	71.4	
77.5
	66.0	
73.8
	
74.3
	
67.6
	
76.7
	
75.9
	
75.2

TRAM-Fisher	84.3	73.1	78.7	77.1	76.2	79.5	78.4	71.4	76.6	65.7	73.2	73.6	67.5	75.5	75.5	74.4

Given that TRAM builds on integrating SAM-style optimization with trust-region regularization, we additionally compare to a naive combination of each of these methods. We replace the standard loss function in ASAM with the loss function adding trust region regularization.

Our full results featuring these systems are shown in Table 11 for language modeling and Table 12 for zero-shot cross-lingual transfer. Across both tasks, naive combination approaches are some of the weakest approaches. When we directly combine ASAM with each trust region regularizer (TRPO, R3F, MESA), we find that the naive combination approaches perform worse than Adam alone, even with extensive hyperparameter tuning. We conjecture that the constituent methods fail to compound constructively because the trust region regularizer does not interact with (or respect) the 
𝜌
-ball neighborhood of ASAM. Therefore, each component may contribute to cross-feature interference, with a disadvantageous net effect on training. TRAM instead offers to combine strategies with complementary features without interference.

C.4Run variation in cross-lingual transfer

For XNLI experiments, we report the mean over 20 runs varying random seed in Table 5 and Table 14. We report the respective standard deviation values for each reported mean in Table 13.

Table 13:Standard deviation of accuracy for the XNLI dataset across 20 training runs with varying random seed. Results are split into groups for: (i) optimizers, (ii) trust region methods, (iii) combined methods, (iv) TRAM variants and (v) TRAM using 
𝑑
𝑥
 with varying metrics for computing divergence. This accompanies Table 5 and Table 14 which report average values across seeds.
	En	Bg	De	El	Ar	Es	Fr	Hi	Ru	Sw	Th	Tr	Ur	Vi	Zh
Adam	0.34	0.42	0.65	0.47	0.50	0.39	0.51	0.51	0.51	0.65	0.40	0.41	0.43	0.51	0.55
SAM	0.24	0.31	0.35	0.34	0.31	0.32	0.49	0.33	0.50	0.36	0.40	0.35	0.44	0.39	0.39
ASAM	0.33	0.33	0.45	0.39	0.47	0.36	0.51	0.44	0.44	0.51	0.56	0.42	0.47	0.45	0.50
FSAM	0.35	0.31	0.51	0.56	0.35	0.37	0.47	0.41	0.45	0.53	0.37	0.39	0.44	0.35	0.38
TRPO	0.24	0.35	0.37	0.34	0.30	0.39	0.34	0.46	0.40	0.38	0.36	0.34	0.53	0.38	0.34
R3F	0.34	0.40	0.44	0.38	0.35	0.35	0.43	0.42	0.46	0.41	0.41	0.35	0.43	0.39	0.47
MESA	0.34	0.34	0.44	0.24	0.40	0.52	0.37	0.67	0.43	0.26	0.40	0.45	0.59	0.43	0.34
ASAM+TRPO	0.30	0.28	0.36	0.29	0.26	0.28	0.35	0.34	0.44	0.36	0.39	0.38	0.34	0.32	0.32
ASAM+R3F	0.32	0.45	0.45	0.40	0.46	0.36	0.49	0.53	0.50	0.52	0.40	0.49	0.60	0.45	0.49
ASAM+MESA	0.34	0.31	0.30	0.44	0.42	0.39	0.38	0.51	0.58	0.46	0.44	0.31	0.51	0.50	0.46
TRAM-
𝜃
𝑡
−
1
	0.40	0.31	0.40	0.30	0.36	0.31	0.43	0.50	0.53	0.48	0.43	0.34	0.36	0.49	0.42
TRAM-
𝜃
0
	0.34	0.38	0.41	0.44	0.48	0.40	0.43	0.53	0.63	0.47	0.66	0.39	0.57	0.50	0.54
TRAM-
𝑥
	0.31	0.29	0.45	0.44	0.37	0.33	0.38	0.37	0.48	0.39	0.44	0.32	0.43	0.39	0.42
TRAM-Fisher	0.29	0.65	0.67	0.55	0.58	0.60	0.49	0.72	0.69	0.64	0.86	0.49	0.68	0.55	0.73
TRAM-
𝑥
 (MMD)	0.42	0.38	0.46	0.43	0.47	0.35	0.42	0.43	0.48	0.44	0.59	0.49	0.36	0.37	0.59
TRAM-
𝑥
 (
𝐿
2
)	0.30	0.27	0.27	0.26	0.24	0.28	0.27	0.21	0.29	0.26	0.24	0.28	0.21	0.22	0.22
C.5Choosing a distance metric

TRAM relies on KL divergence to estimate the trust region around the pre-trained function (i.e., 
𝑝
𝑓
(
⋅
|
𝑥
+
𝑧
,
𝜃
)
 or 
𝑝
𝑓
(
⋅
|
𝑥
,
𝜃
𝑡
−
1
)
). We propose TRAM with forward KL on the intuition that the perturbed distribution (i.e., estimated point in the trust region) is the target (i.e., true) output which the current outputs (i.e., estimate) should match. We empirically verify this setup as the optimal arrangement (i.e., forward KL). While reverse KL or symmetric KL report only marginally poorer results, we report only forward KL for simplicity. We also consider alternative distance metrics in Table 14. We evaluate modifying the best-performing model for XNLI with different distances to examine if the divergence for trust region estimation is influential in performance. We evaluate maximum mean discrepancy using an inverse multiquadratic kernel (MMD; Gretton et al., 2012), or 
𝐿
2
 distance within 
𝑑
𝑥
. Even using the worst-performing metric, 
𝐿
2
 distance, TRAM is still competitive to methods in Table 8. Characterizing the best trust region estimate for TRAM is outside the scope of this work. Future work should explore the suitability of different distances (e.g., Renyi divergence) to improve the estimation of the trust region space.

Table 14:XNLI accuracy across varying the divergence metric estimating the trust region distance in TRAM. We compare to using maximum mean discrepancy (MMD) and 
𝐿
2
 distance. TRAM is generally robust to different estimates for the trust region between 
𝑝
𝑓
(
⋅
|
𝑥
,
𝜃
)
 and 
𝑝
𝑓
(
⋅
|
𝑥
+
𝑧
,
𝜃
)
.
	En	Bg	De	El	Ar	Es	Fr	Hi	Ru	Sw	Th	Tr	Ur	Vi	Zh	ZS Avg. 
↑

TRAM-
𝑥
 (KL)	
86.2
	
79.8
	
78.3
	
77.5
	
73.5
	
80.9
	
79.6
	
71.4
	
77.5
	
66.0
	73.8	74.3	
67.6
	
76.7
	
75.9
	
75.2

TRAM-
𝑥
 (MMD)	86.0	79.3	78.1	77.1	73.2	80.7	
79.6
	
71.4
	77.3	
66.0
	
74.0
	
74.4
	67.2	76.3	75.6	75.0
TRAM-
𝑥
 (
𝐿
2
)	85.1	78.7	76.8	76.2	72.2	79.4	78.8	70.4	76.2	65.5	72.6	73.5	67.1	75.8	74.6	74.1
Generated by L A T E xml 
Instructions for reporting errors

We are continuing to improve HTML versions of papers, and your feedback helps enhance accessibility and mobile support. To report errors in the HTML that will help us improve conversion and rendering, choose any of the methods listed below:

Click the "Report Issue" button.
Open a report feedback form via keyboard, use "Ctrl + ?".
Make a text selection and click the "Report Issue for Selection" button near your cursor.
You can use Alt+Y to toggle on and Alt+Shift+Y to toggle off accessible reporting links at each section.

Our team has already identified the following issues. We appreciate your time reviewing and reporting rendering errors we may not have found yet. Your efforts will help us improve the HTML versions for all readers, because disability should not be a barrier to accessing research. Thank you for your continued support in championing open access for all.

Have a free development cycle? Help support accessibility at arXiv! Our collaborators at LaTeXML maintain a list of packages that need conversion, and welcome developer contributions.

Report Issue
Report Issue for Selection
