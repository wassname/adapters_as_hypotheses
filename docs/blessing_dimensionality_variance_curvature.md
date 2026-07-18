Title: The Blessing of Dimensionality in LLM Fine-tuning: A Variance–Curvature Perspective

URL Source: https://arxiv.org/html/2602.00170

Markdown Content:
1Introduction
2Background
3Evolution Strategies as a Geometric Probe
4Non-Monotonic Training Dynamics and Low-Dimensional Curvature
5Empirical Scalability of ES as a Consequence of Low-Dimensional Curvature
6Discussions and Future Work
Appendix
The Blessing of Dimensionality in LLM Fine-tuning: A Variance–Curvature Perspective
Qiyao Liang
Jinyeop Song
Yizhou Liu
Jeff Gore
Ila Fiete
Risto Miikkulainen
Xin Qiu
Abstract

Weight-perturbation evolution strategies (ES) can fine-tune billion-parameter language models with surprisingly small populations (e.g., 
𝑁
≈
30
), contradicting classical zeroth-order curse-of-dimensionality intuition. We also observe a second seemingly separate phenomenon: under fixed hyperparameters, the stochastic fine-tuning reward often rises, peaks, and then degrades in both ES and GRPO. We argue that both effects reflect a shared geometric property of fine-tuning landscapes: they are low-dimensional in curvature. A small set of high-curvature dimensions dominates improvement, producing (i) heterogeneous time scales that yield rise–then–decay under fixed stochasticity, as captured by a minimal quadratic stochastic-ascent model, and (ii) degenerate improving updates, where many random perturbations share similar components along these directions. Using ES as a geometric probe on fine-tuning reward landscapes of GSM8K, ARC-C, and WinoGrande across Qwen2.5-Instruct models (0.5B–7B), we show that reward-improving perturbations remain empirically accessible with small populations across scales. Together, these results reconcile ES scalability with non-monotonic training dynamics and suggest that high-dimensional fine-tuning may admit a broader class of viable optimization methods than worst-case theory implies.

evolution strategies, fine-tuning, intrinsic dimension, Hessian spectrum, stochastic optimization, RLHF
Figure 1:Non-monotonic training reward showing reward degradation beyond a peak training iteration 
𝑡
peak
 in ES and GRPO fine-tuning of Qwen2.5-1.5B-Instruct on GSM8K (Cobbe et al., 2021). Since GRPO does not directly maximize the task reward, training rewards plotted here are evaluated over the training set at every iteration. Both methods are trained on the same training set of 100 samples.
1Introduction

Recent work has shown that large language models (LLMs) can be fine-tuned at scale using simple weight-perturbation evolution strategies (ES; Qiu et al., 2025). This approach achieves performance competitive with leading reinforcement-learning-based methods such as Proximal Policy Optimization (PPO) (Schulman et al., 2017) and Group Relative Policy Optimization (GRPO) (Shao et al., 2024). This is surprising because ES estimates an update direction from a finite set of random parameter perturbations; in generic high-dimensional problems, the signal-to-noise ratio of such estimates typically deteriorates with dimension, so avoiding a curse of dimensionality would seem to require population sizes that grow rapidly with the number of parameters (Duchi et al., 2015). Nonetheless, ES with populations as small as 
𝑁
≈
30
 has been shown to effectively fine-tune models with billions of parameters.

At the same time, we observe a second, seemingly independent phenomenon. Under fixed hyperparameters over training duration, training reward often exhibits a pronounced non-monotonic trajectory: reward improves rapidly from a pretrained checkpoint, reaches a peak, and then decreases toward a lower value as training proceeds. This behavior appears consistently across ES and policy-gradient-style fine-tuning methods and is strongly modulated by algorithmic stochasticity—through the perturbation scale and population size in ES, or equivalently the temperature and group size in GRPO (Fig. 1). Its ubiquity across distinct stochastic learning methods suggests that it reflects a structural property of the fine-tuning landscape itself, rather than an artifact of any particular algorithm.

In this paper, we aim to explain: (1) why can small-population weight-perturbation methods scale to billion-parameter models, and (2) why do stochastic fine-tuning runs exhibit rise–then–decay behavior under fixed stochasticity? To these ends, we posit the following hypothesis:

Central hypothesis: a blessing of dimensionality
We hypothesize that LLM fine-tuning landscapes are low-dimensional in curvature: local optimization geometry is governed by a small set of curvature-active directions, and the size of this curvature-active set does not grow proportionally with the number of model parameters, so small-population weight-perturbation methods can remain effective at billion-parameter scale despite zeroth-order curse-of-dimensionality intuitions.
A unifying geometry–variance framework.

We interpret both ES scalability and non-monotonic fine-tuning dynamics through a geometry–variance perspective intrinsic to the reward landscape. Locally, curvature is highly anisotropic: a small number of high-curvature “stiff” directions dominate reward improvement, while the vast majority of directions are weakly curved or effectively “flat.” ES serves as a stochastic probe of this structure, with explicit control over perturbation scale and sampling variance, allowing us to study how stochastic updates interact with anisotropic curvature. Because improvement is governed by the stiff subspace, random-perturbation methods need only access this restricted set of directions, rather than estimate a full-dimensional gradient, enabling effective updates without populations that scale with parameter count. At the same time, the stiff–flat separation induces heterogeneous time scales: rapid early progress occurs along stiff directions, while persistent stochastic drift along flat directions can dominate at later times, producing the observed rise–then–decay behavior under fixed stochasticity.

Contributions.

We make three concrete contributions:

• 

A testable “blessing of dimensionality” prediction. We formalize curvature-active dimensionality and derive scaling predictions linking model size, stochasticity, and the population needed to access improving perturbations.

• 

A mechanism for rise–then–decay. We give an analytically tractable model showing how stiff–flat time-scale separation under fixed noise yields non-monotonic reward trajectories.

• 

ES-based probes across scale. We empirically measure extreme-value improvement under perturbations across tasks and model sizes, showing that improving perturbations remain accessible with small populations from 0.5B to 7B parameters.

The rest of the paper is organized as follows. Section 2 reviews zeroth-order optimization and prior evidence for low-dimensional fine-tuning and anisotropic curvature. Section 3 introduces ES as a geometric probe via Gaussian smoothing. Section 4 characterizes rise–then–decay in ES/GRPO and presents a minimal model linking heterogeneous curvature and fixed noise to non-monotonic dynamics. Section 5 tests the resulting scaling predictions on GSM8K (Cobbe et al., 2021), ARC-C (Clark et al., 2018), and WinoGrande (Sakaguchi et al., 2020). Section 6 discusses implications for curse-of-dimensionality intuitions and fine-tuning methods.

2Background

This section reviews prior evidence motivating a blessing of dimensionality view of fine-tuning: despite enormous parameter spaces, adaptation often depends on far fewer effective degrees of freedom. We briefly cover (i) zeroth-order curse-of-dimensionality intuition, (ii) empirical/architectural evidence for low-dimensional adaptation (intrinsic dimension, parameter-efficient updates), (iii) curvature structure in overparameterized networks (bulk near zero with a few dominant directions), and (iv) related work on stochastic dynamics and non-monotonic training. Together, these threads motivate our curvature-based notion of effective dimensionality and the use of ES as a stochastic probe of fine-tuning geometry.

Zeroth-order optimization and the curse of dimensionality.

Classical results in zeroth-order (gradient-free) optimization often show that the number of function evaluations needed to make progress grows quickly with the number of parameters, which has led to skepticism about scaling weight-perturbation methods to modern overparameterized models (Duchi et al., 2015). Despite this, evolution strategies have long been studied as scalable black-box optimizers in reinforcement learning and large-scale learning, including distributed ES variants (Salimans et al., 2017). Recent work demonstrates that simple weight-perturbation ES can fine-tune billion-parameter language models with surprisingly small populations and competitive performance (Qiu et al., 2025). Our work addresses the resulting tension between worst-case zeroth-order intuition and empirical scalability, and provides an operational lens that links scalability and training dynamics.

Blessing of dimensionality via low-dimensional fine-tuning structure.

A growing body of evidence suggests that many fine-tuning problems are effectively low-dimensional despite extremely high ambient parameter dimension. Intrinsic dimension studies show that downstream objectives can be optimized within a surprisingly small random subspace, including in language-model fine-tuning, and reporting that this intrinsic dimensionality actually decreases as a function of model size (Li et al., 2018; Aghajanyan et al., 2021). Parameter-efficient fine-tuning methods provide architectural evidence for low-dimensional adaptation, for example via low-rank updates as in LoRA (Hu et al., 2021). These results motivate a “blessing of dimensionality” viewpoint: additional parameters need not increase the number of directions that matter for adaptation, and may instead enlarge flat or redundant directions, resulting in a lower effective dimensionality.

Hessian spectra: bulk near zero and a few outliers.

A standard way to characterize local geometry in optimization is via the Hessian, the matrix of second derivatives of the objective with respect to parameters. Near a solution, the Hessian eigenvalues describe curvature along different directions: large-magnitude eigenvalues correspond to stiff directions where the objective changes rapidly, while near-zero eigenvalues correspond to flat directions. Empirical studies of overparameterized neural networks have repeatedly found a highly structured Hessian spectrum near trained solutions, typically consisting of a large bulk of eigenvalues concentrated near zero together with a small number of outliers that dominate curvature (Sagun et al., 2017; Ghorbani et al., 2019). Related observations suggest that learning dynamics often concentrate in a low-dimensional “sharp” subspace associated with these top-curvature directions (Gur-Ari et al., 2018). This bulk+outlier picture aligns with the geometric mechanism emphasized in our work—few stiff modes drive rapid early progress, while a large flat bulk is susceptible to variance accumulation—and motivates focusing on operational signatures of curvature structure rather than explicit Hessian estimation. Accordingly, we do not attempt to measure full Hessian spectra in our experiments; instead, we probe their practical consequences for stochastic fine-tuning, namely whether improvement depends on a small set of directions and how stochasticity couples to the flat bulk over time.

Stochastic optimization dynamics and noise-limited behavior.

A complementary line of work models constant-step stochastic optimization near minima as a noisy dynamical system, often approximated by an Ornstein–Uhlenbeck process or related SDE, yielding predictions about stationary noise floors and curvature-dependent behavior (Mandt et al., 2017; Li et al., 2017). These perspectives motivate interpreting late-stage training as variance-limited once gradient signal diminishes. Our toy model adopts this analysis in the same spirit, using a local stochastic-dynamics approximation to make explicit how curvature and noise together shape late-stage behavior, and to explain how non-monotonic rise–then–decay can arise under fixed stochasticity.

Rise–then–decay and non-monotonic training dynamics.

Non-monotonic training trajectories—where performance improves early and later degrades or oscillates under continued updates—have been documented in several stochastic learning settings. In deep RL, extended on-policy training can exhibit performance collapse (often discussed as policy collapse or loss of plasticity), where returns improve initially and later deteriorate (Dohare et al., 2023; Moalla et al., 2024). In supervised deep learning, related non-monotonicity of the training loss has been studied in “edge of stability” regimes, where loss can go up and down across iterations despite continued training (Arora et al., 2022; Zhu et al., 2023). In our setting, we observe a reproducible peak–then–decay phenomenon across both ES and policy-gradient-style fine-tuning (e.g., GRPO), suggesting a mechanism that is not specific to a particular optimization method.

3Evolution Strategies as a Geometric Probe

We use weight-perturbation evolution strategies (ES) primarily as a geometric probe of fine-tuning landscapes. Here a fine-tuning landscape is the mapping from parameters to a scalar task reward, 
𝜃
↦
𝒥
​
(
𝜃
)
, where 
𝒥
 may be accuracy or another reward evaluated via sampling and decoding. ES is attractive in this setting because it requires only reward evaluations, making it applicable even when the underlying reward is discrete, truncated, or otherwise not amenable to differentiation.

At iteration 
𝑡
, ES samples perturbations 
𝜀
𝑘
∼
𝒩
​
(
0
,
𝐼
)
, evaluates 
𝑟
𝑘
=
𝒥
​
(
𝜃
𝑡
+
𝜎
​
𝜀
𝑘
)
, and updates

	
𝑔
^
𝑡
=
1
𝑁
​
𝜎
​
∑
𝑘
=
1
𝑁
𝑟
𝑘
​
𝜀
𝑘
,
𝜃
𝑡
+
1
=
𝜃
𝑡
+
𝛼
​
𝑔
^
𝑡
,
		
(1)

with the full procedure in Algorithm 1. We note that this is a naive version of the algorithm. For LLM fine-tuning in practice, we evaluate each candidate perturbation on a group of prompts and use the group-averaged reward in place of a single-sample evaluation.

Algorithm 1 Weight-Perturbation Evolution Strategies (ES)

Input: objective/reward 
𝒥
​
(
𝜃
)
; initial parameters 
𝜃
0
; step size 
𝛼
; perturbation scale 
𝜎
; population size 
𝑁
; iterations 
𝑇

for 
𝑡
←
0
 to 
𝑇
−
1
 do

    Sample 
𝜀
1
,
…
,
𝜀
𝑁
​
∼
i.i.d.
​
𝒩
​
(
0
,
𝐼
)
  for 
𝑘
←
1
 to 
𝑁
 do
       
𝑟
𝑘
←
𝒥
​
(
𝜃
𝑡
+
𝜎
​
𝜀
𝑘
)
 
   
𝑔
^
𝑡
←
1
𝑁
​
𝜎
​
∑
𝑘
=
1
𝑁
𝑟
𝑘
​
𝜀
𝑘
,   
𝜃
𝑡
+
1
←
𝜃
𝑡
+
𝛼
​
𝑔
^
𝑡
 
Gaussian smoothing and coarse-grained geometry.

ES can be interpreted as optimizing a Gaussian-smoothed objective,

	
𝒥
𝜎
​
(
𝜃
)
≜
𝔼
𝜀
∼
𝒩
​
(
0
,
𝐼
)
​
[
𝒥
​
(
𝜃
+
𝜎
​
𝜀
)
]
,
		
(2)

where 
𝜎
 controls the smoothing scale. Even when 
𝒥
 is jagged or nondifferentiable, 
𝒥
𝜎
 is differentiable under mild conditions. Geometrically, smoothing suppresses fine-scale irregularities while preserving curvature structure at scales larger than 
𝜎
, so ES interacts with a coarse-grained version of the landscape.

A central identity is

	
∇
𝒥
𝜎
​
(
𝜃
)
=
1
𝜎
​
𝔼
𝜀
∼
𝒩
​
(
0
,
𝐼
)
​
[
𝒥
​
(
𝜃
+
𝜎
​
𝜀
)
​
𝜀
]
,
		
(3)

so 
𝑔
^
𝑡
 is a Monte Carlo estimator of 
∇
𝒥
𝜎
​
(
𝜃
𝑡
)
.

Controllable Stochasticity.

With a finite population, ES produces a stochastic gradient estimate of 
∇
𝐽
𝜎
:

	
𝑔
^
𝑡
=
∇
𝐽
𝜎
​
(
𝜃
𝑡
)
+
𝜉
𝑡
est
,
𝔼
​
[
𝜉
𝑡
est
∣
𝜃
𝑡
]
=
0
,
		
(4)

where 
𝜉
𝑡
est
 is Monte Carlo estimation noise. Its covariance has the standard prefactor

	
Cov
​
(
𝜉
𝑡
est
∣
𝜃
𝑡
)
≈
1
𝑁
​
𝜎
2
​
Σ
est
​
(
𝜃
𝑡
,
𝜎
)
,
		
(5)

with problem-dependent 
Σ
est
 that can itself depend on 
𝜎
 (e.g., through reward variability). Separately, in our later local dynamics toy model we summarize the effective stochasticity of parameter updates by a diffusion scale 
𝜅
=
𝜎
2
/
𝑁
, which captures how larger perturbation radii and smaller populations increase parameter-space wandering.

Why analyze fine-tuning landscapes with ES?

Our goal is not to compare optimizers, but to understand the geometric structure of fine-tuning landscapes and how it interacts with stochastic learning dynamics. A key challenge is that the true reward landscape for LLM fine-tuning is often jagged or defined implicitly through sampling (e.g., discrete accuracy, externally judged rewards), so gradients are unavailable or only accessible through surrogate losses that may obscure the underlying reward geometry. ES provides a natural entry point because it operates directly on reward evaluations under random perturbations and, crucially, induces controlled Gaussian smoothing. This smoothing probes the landscape at a tunable spatial scale and couples cleanly to curvature structure, making ES well-suited for studying coarse-grained geometry and the role of variance. Subsequent sections use ES-based analyses to reveal geometric signatures of fine-tuning landscapes, and then relate these signatures to the behavior of other RL fine-tuning methods.

4Non-Monotonic Training Dynamics and Low-Dimensional Curvature

We begin with a simple but striking phenomenon: during fine-tuning on GSM8K, the training reward need not increase monotonically under fixed hyperparameters. Instead, reward often improves rapidly, reaches a peak, and then degrades toward a lower value. Figure 1 shows this behavior for ES at multiple population sizes: both the peak time and the depth of the decay vary systematically with stochasticity, suggesting a variance-controlled effect rather than overfitting or evaluation noise. We observe qualitatively similar, though noisier, rise–then–decay behavior in motivating GRPO runs as well, indicating that the phenomenon is not specific to a particular learning method.

Figure 2:Water-filling schematic for rise–then–decay dynamics. (a) Early: fast improvement along stiff directions while variance is small. (b) Peak: stiff directions are mostly exhausted; variance has risen enough to limit gains. (c) Late: variance-dominated drift along weakly constrained directions yields degradation under fixed stochasticity.
A mechanism isolate: local quadratic stochastic dynamics.

To connect this behavior to landscape geometry, we analyze a local quadratic approximation around a near-optimal region. Let 
𝜃
⋆
 be a local maximizer and write

	
𝐽
​
(
𝜃
⋆
+
𝑥
)
≈
𝐽
​
(
𝜃
⋆
)
−
1
2
​
𝑥
⊤
​
𝐶
​
𝑥
,
𝐶
⪰
0
.
		
(6)

A constant-step noisy ascent model takes the form

	
𝜃
𝑡
+
1
=
(
𝐼
−
𝛼
​
𝐶
)
​
𝜃
𝑡
+
𝛼
​
𝜎
𝑁
​
𝜀
𝑡
,
𝜀
𝑡
∼
𝒩
​
(
0
,
𝐼
)
,
		
(7)

with effective noise 
𝜅
=
𝜎
2
/
𝑁
. Diagonalizing 
𝐶
=
𝑄
​
Λ
​
𝑄
⊤
 decouples the dynamics into independent modes with contraction factors 
𝑎
𝑖
=
1
−
𝛼
​
𝜆
𝑖
 (Appendix A): high-curvature directions relax quickly, while low-curvature directions relax slowly and are more susceptible to variance accumulation.

Water-filling analogy: “rushing downhill before the valley floods.”

Figure 2 summarizes the core intuition in the equivalent descent picture. Early in training, fast relaxation along stiff directions drives rapid improvement. Meanwhile, stochasticity accumulates along weakly constrained directions, which acts like a rising “water level” that progressively limits attainable performance. Once the fast directions have largely saturated, the remaining dynamics can be dominated by this variance accumulation, producing a peak followed by degradation under fixed stochasticity.

A minimal example: two-block curvature spectrum.

To make the time-scale separation concrete, we consider a stylized two-block spectrum: 
𝑑
≪
𝐷
 stiff directions with curvature 
𝜆
ℎ
​
𝑖
 and many weakly curved directions with 
𝜆
𝑙
​
𝑜
≪
𝜆
ℎ
​
𝑖
 (as shown in the inset of Figure 3). Figure 3 shows ES dynamics (both simulation and analytics) on this toy quadratic landscape and demonstrates that simple spectral heterogeneity is sufficient to produce rise–then–decay trajectories, with early gains driven by the stiff subspace and late-time behavior set by noise accumulation in the weakly curved bulk. Moreover, in the quadratic case, variance tends to a terminal plateau value that is determined by the effective stochasticity dependent on 
𝑁
.

Figure 3:Quadratic toy model: time-scale separation produces rise–then–decay. ES on a quadratic landscape with a two-block spectrum (
𝐷
=
128
, 
𝑑
=
16
, 
𝜆
ℎ
=
1.0
, 
𝜆
𝑙
=
0.05
). Solid curves: Monte Carlo ES runs for different populations 
𝑁
 (noise levels). Dashed curves: closed-form prediction from the quadratic stochastic model (Appendix A). Larger 
𝑁
 (smaller 
𝜅
=
𝜎
2
/
𝑁
) raises the terminal plateau and suppresses late-time degradation.
What the toy model isolates.

The toy model isolates two qualitative requirements for rise–then–decay in this local regime. First, heterogeneous and low-dimensional curvature spectrum: with a single dominant time scale the expected trajectory is monotonic, while well-separated time scales can produce a peak. Second, non-negligible stochasticity: the late-time plateau is controlled by 
𝜅
=
𝜎
2
/
𝑁
 through a curvature-weighted functional (Appendix A). Initialization affects how pronounced the peak is (Appendix A), but the mechanism itself is simply the interaction of curvature-dependent relaxation and fixed noise. We note in passing that the plateau phenomenon naturally prescribes a curvature-based effective dimensionality measure, which inspires the proposal of a spectroscopy method that we describe in Appendix B.

Implication: low-dimensional curvature in fine-tuning landscapes.

The variance-controlled rise–then–decay behavior observed in ES (and qualitatively in our motivating GRPO runs) points to the same ingredients in real fine-tuning: a small number of fast, curvature-dominant directions and many weakly constrained directions. This aligns with the common “bulk + outliers” curvature picture in overparameterized networks, where a few large-magnitude eigenvalues dominate curvature amid a near-zero bulk. An schematic illustration of the Hessian spectrum is given in Figure 4. In our curvature-based notion of low-dimensionality, what matters is not the ambient parameter count but how many directions are meaningfully curvature-active, and that the curvature-active dimensions do not scale with model size.

From low-dimensional curvature to degeneracy.

Nevertheless, it is important to note that low-dimensional curvature does not imply a unique improving direction. Rather, if improvement is governed by a small curvature-active subspace, then many distinct perturbations can produce comparable progress by sharing similar components in that subspace. In the next section, we show that this degeneracy has a concrete empirical consequence: random perturbations can reliably access reward-improving directions with a small, fixed population size even as model dimension increases.

Figure 4:Schematic curvature structure: near-zero bulk with a few stiff outliers. The near-zero bulk corresponds to many weakly constrained directions, while a small number of outliers correspond to curvature-dominant directions that relax quickly and drive early improvement.
5Empirical Scalability of ES as a Consequence of Low-Dimensional Curvature

The preceding section showed that rise–then–decay dynamics may arise from a low-dimensional curvature structure of fine-tuning landscapes, leading to a concrete prediction: if curvature-active structure is low-dimensional and persists across scales, then improvement should remain accessible under random perturbations even as the ambient parameter dimension grows. We now elaborate on and test this prediction across model sizes.

Definitions: accessibility and degeneracy of reward-improving perturbations.

Let 
𝐶
 denote a local curvature operator (e.g., 
𝐶
=
−
∇
2
𝐽
​
(
𝜃
⋆
)
⪰
0
 near a local maximizer), and let 
𝑈
∈
ℝ
𝐷
×
𝑘
 span a 
𝑘
-dimensional curvature-active subspace (e.g., top-eigen directions of 
𝐶
). For an isotropic perturbation 
𝜀
∼
𝒩
​
(
0
,
𝐼
)
, define its projection 
𝑧
=
𝑈
⊤
​
𝜀
∈
ℝ
𝑘
. If improvement is primarily determined by 
𝑧
, then an improving region 
𝒜
⊂
ℝ
𝑘
 induces an improvement-supporting set in the full space,

	
𝒢
≜
{
𝜀
∈
ℝ
𝐷
:
𝑈
⊤
​
𝜀
∈
𝒜
}
.
		
(8)

When 
𝑘
≪
𝐷
, many distinct perturbations share similar curvature-active components: for any fixed 
𝑧
∈
𝒜
, the preimage 
{
𝜀
:
𝑈
⊤
​
𝜀
=
𝑧
}
 is an affine subspace of dimension 
𝐷
−
𝑘
. We refer to this many-to-one structure as degeneracy. Whether ES can reliably find improving perturbations is instead governed by accessibility, i.e., the probability mass of 
𝒢
 under the sampling distribution,

	
𝑝
imp
​
(
𝜎
)
≜
Pr
⁡
(
𝜀
∈
𝒢
)
=
Pr
⁡
(
𝑈
⊤
​
𝜀
∈
𝒜
)
,
		
(9)

which depends only on the geometry of 
𝒜
 in the 
𝑘
-dimensional projected space. Thus low-dimensional curvature can yield a nontrivial improvement-supporting mass even when 
𝑘
≪
𝐷
, providing a mechanism by which fixed-population random search can avoid curse-of-dimensionality behavior.

Figure 5:Needle-in-a-haystack versus degenerate “wheel of fortune.” (a) In a classical curse-of-dimensionality intuition, improvement is confined to a single unique direction with vanishing probability mass, so fixed-population random search fails as dimension grows. (b) Under low-dimensional curvature, improvement is governed by a small curvature-active subspace, but many ambient perturbations share similar projections onto this subspace (degeneracy), yielding an improvement-supporting set with nontrivial probability mass. Extreme-value selection (best-of-
𝑁
) can therefore succeed with a small, fixed population size.
What an empirical curse of dimensionality would imply.

Figure 5 contrasts two geometries. In a needle-in-a-haystack picture (a), improvement lies in a vanishingly rare “north-star” direction, so a fixed population is unlikely to sample 
Δ
​
𝑅
>
0
 as dimension grows. In a degenerate many–prize-region picture (b), improvement is not unique: many perturbation directions yield positive 
Δ
​
𝑅
, so best-of-
𝑁
 selection can reliably find an improving update without requiring 
𝑁
 to grow with model size. Empirically, a curse of dimensionality would therefore appear as loss of improvement signal at fixed 
𝑁
, or a systematic rightward shift of best-of-
𝑁
 curves with scale; our experiments test for these signatures.

Metrics: best-of-
𝑁
 as an operational proxy for accessibility.

For each model and task, we sample perturbations 
𝜀
𝑖
∼
𝒩
​
(
0
,
𝐼
)
 and measure 
Δ
​
𝑅
𝑖
=
𝑅
​
(
𝜃
+
𝜎
​
𝜀
𝑖
)
−
𝑅
​
(
𝜃
)
. We summarize accessibility by the expected best-of-
𝑁
 improvement

	
Δ
𝑁
∗
​
(
𝜃
,
𝜎
)
=
𝔼
​
[
max
𝑖
≤
𝑁
⁡
Δ
​
𝑅
𝑖
]
,
		
(10)

which directly quantifies whether 
𝑁
 random draws can reach the improving “prize regions” in Fig. 5(b), even when 
𝔼
​
[
Δ
​
𝑅
]
 is small or negative. To control for task saturation, we also report headroom-normalized improvements 
Δ
𝑁
∗
/
(
1
−
𝑅
0
)
.

Figure 6:Fixed-population improvement remains accessible across model scales. Top row (a–c): Headroom-normalized expected best-of-
𝑁
 improvement 
Δ
𝑁
rel
​
(
𝜎
)
=
𝔼
​
[
max
𝑖
≤
𝑁
⁡
Δ
​
𝑅
𝑖
]
/
(
1
−
𝑅
0
)
 as a function of population size 
𝑁
 at fixed perturbation scale 
𝜎
=
3
×
10
−
4
 for GSM8K, ARC-C, and WinoGrande across Qwen2.5-Instruct (0.5B–7B). Curves saturate by 
𝑁
≈
30
–40 without a systematic shift to larger 
𝑁
 as model size increases. Bottom row (d–f): Headroom-normalized expected best-of-30 improvement versus model size for multiple 
𝜎
. For each task, a viable range of 
𝜎
 yields positive best-of-30 improvements from 0.5B to 7B, indicating that the improvement-supporting tail of the perturbation-induced 
Δ
​
𝑅
 distribution remains accessible at scale. Error bars show 
±
1.96
​
SE
 computed across 
𝑆
 independent perturbation batches (each batch contains a fixed pool of candidates evaluated on the same prompt set), reflecting variability due to perturbation sampling (Appendix G).
Population requirements do not scale with model size.

Figures 6(a–c) report the headroom-normalized expected best-of-
𝑁
 improvement as a function of population size 
𝑁
 for GSM8K, ARC-C, and WinoGrande across the Qwen2.5-Instruct family (0.5B–7B). For each task, model, and perturbation scale 
𝜎
, we construct a pool of 
𝑀
=
240
 perturbation candidates and evaluate their reward changes 
Δ
​
𝑅
=
𝑅
​
(
𝜃
+
𝜎
​
𝜖
)
−
𝑅
​
(
𝜃
)
 on a fixed prompt set. We estimate 
Δ
𝑁
∗
​
(
𝜎
)
 by Monte Carlo sampling 
𝑁
 distinct candidates (without replacement) from this pool and averaging the resulting maxima, and report the headroom-normalized quantity 
Δ
𝑁
∗
​
(
𝜎
)
/
(
1
−
𝑅
0
)
 (see Appendix G for experimental details).

Across all tasks and model sizes, best-of-
𝑁
 improves rapidly at small 
𝑁
 and exhibits clear diminishing returns beyond 
𝑁
≈
30
–40, with no systematic rightward shift as model size increases. This rules out the most direct empirical signature of a curse of dimensionality—namely, a growing population requirement to access improvement as ambient parameter dimension increases—and indicates that improvement remains accessible with relatively small populations even for the largest models studied. Importantly, this saturation should not be attributed to a finite-candidate-pool artifact. With 
𝑀
=
240
 candidates and 
𝑁
≤
50
, we are far from the 
𝑁
→
𝑀
 regime in which finite-pool saturation dominates. Instead, the observed flattening reflects intrinsic diminishing returns of expected extrema, which arise generically—even in the infinite-pool (i.i.d.) limit—from the shape of the improvement distribution’s upper tail. In this regime, increasing population size primarily yields rarer tail events rather than systematically larger gains.

Viable (local) perturbation scales persist with model size.

Fixing the population size to 
𝑁
=
30
, Figures 6(d–f) show the headroom-normalized best-of-
30
 improvement 
Δ
30
∗
​
(
𝜎
)
/
(
1
−
𝑅
0
)
 as a function of model size for multiple perturbation scales 
𝜎
. Across all tasks, there exists a viable range of sufficiently small perturbation scales for which 
Δ
30
∗
​
(
𝜎
)
 remains positive from 0.5B to 7B parameters. The key requirement is not precise tuning of 
𝜎
, but that the perturbations remain local enough to probe regions of parameter space where improvement is common. At these scales, the perturbation distribution intersects many improvement-supporting regions, so that a moderate population reliably samples candidates from the upper tail. While the magnitude of improvement and the optimal 
𝜎
 vary by task (and absolute gains necessarily attenuate as headroom shrinks), improvement does not collapse as the number of parameters increases by more than an order of magnitude. Operationally, this indicates that a constant ES population budget (
𝑁
≈
30
), combined with sufficiently local perturbations, continues to access an improvement-supporting tail of the landscape at scale.

Interpretation: a variance–geometry tradeoff yields a blessing of dimensionality.

Taken together, the preceding results support an interpretation in which high-dimensional parameter spaces are not hostile to zeroth-order search, but instead contain many distinct, locally improving directions that can be accessed with modest populations. At a checkpoint 
𝜃
 and perturbation scale 
𝜎
, the local geometry of the reward surface induces a distribution of perturbation outcomes 
Δ
​
𝑅
, whose upper tail encodes the density of improvement-supporting regions. ES progress at population 
𝑁
 is therefore governed by this tail mass, summarized by 
Δ
𝑁
∗
​
(
𝜎
)
, rather than by alignment with a single gradient direction. Appendix G.9 reports auxiliary tail-accessibility summaries (
𝑁
90
 and 
𝑞
0.95
) that likewise show no systematic growth with model size.

Crucially, the persistence of best-of-
𝑁
 improvement with fixed 
𝑁
 and suitably small 
𝜎
 across model sizes indicates that this improvement-supporting tail does not thin as ambient dimension increases. Instead, increasing dimensionality appears to introduce more locally improving directions, yielding a form of blessing of dimensionality in which improvement remains accessible through random sampling. Finite populations succeed not by resolving a unique descent direction, but by reliably intersecting one of many favorable regions. We note that although we do not go to the extreme to claim that larger models are lower-dimensional, complementary ES-based curvature proxy measurements suggest that curvature-relevant structure can become more concentrated with model scale (Appendix G.10).

6Discussions and Future Work
Central thesis: low-dimensional, heterogeneous, and degenerate fine-tuning geometry.

Our results support a single geometric picture that reconciles two seemingly disparate empirical facts: (i) stochastic fine-tuning can exhibit rise–then–decay training dynamics under fixed hyperparameters, and (ii) small-population weight perturbation can improve billion-parameter LLMs without an empirical curse of dimensionality. The unifying lens is that fine-tuning landscapes are effectively low-dimensional in curvature yet heterogeneous and degenerate. Low-dimensionality means that a small number of curvature-active (stiff) dimensions dominate progress; heterogeneity means that these directions have widely separated relaxation rates, producing fast early gains and slow variance accumulation; degeneracy means that improvement is not confined to a single unique direction but to a low-dimensional subspace that admits many equivalent embeddings in the ambient parameter space, yielding multiple improvement-supporting regions. Together, these properties explain both non-monotonic training and ES scalability.

Practical consequence: diagnosing and mitigating rise–then–decay.

The toy mechanism implies that rise–then–decay is not mysterious: it is the expected outcome when stiff modes saturate while variance continues to accumulate along flat modes under fixed stochasticity. This suggests immediate practical interventions: (i) early stopping (stop near the peak when stiff-mode signal is exhausted but variance has not yet dominated), (ii) noise scheduling (increase population size, reduce 
𝜎
, reduce temperature, or reduce effective update noise over time to slow “water filling” of flat modes), and (iii) adaptive step sizes that shrink once curvature-active progress saturates. More broadly, the geometry–variance view suggests treating non-monotonicity as a diagnostic: its presence indicates heterogeneous curvature and a variance-dominated late regime, while its absence (under comparable stochasticity) suggests either insufficient heterogeneity or a regime in which variance is already controlled.

A broader implication: revisiting algorithm design beyond the curse-of-dimensionality dogma.

Perhaps the most important consequence of our findings is conceptual. Classical zeroth-order theory discourages parameter-space perturbation methods at high dimension by focusing on worst-case settings in which improvement is a single needle-like direction. Our evidence instead supports a regime in which improvement directions are low-dimensional and degenerate, so a small population can reliably access them. This opens a broader algorithmic design space that is typically dismissed for LLM fine-tuning: population-based perturbation methods, random subspace or coordinate search, structured perturbations (e.g., low-rank or blockwise noise), evolution strategies with learned search covariances, and hybrid methods that combine occasional perturbative exploration with gradient updates. In other words, once the effective geometry is low-dimensional, the relevant question is no longer “can zeroth-order scale in 
𝐷
?” but “which perturbation distributions best align with the small set of curvature-active directions?”

Implications for robustness and diversity of solutions.

The “many prize regions” interpretation suggests that different improving perturbations may lead to distinct solution manifolds with different properties (e.g., robustness, calibration, reasoning style, or safety tradeoffs), even when they yield comparable short-horizon reward gains. This raises an opportunity: rather than searching for a single optimum, fine-tuning can be viewed as selecting among multiple alternative improvements. Population-based methods are naturally suited to this perspective because they generate sets of candidate updates that can be filtered or diversified according to secondary objectives (such as conciseness, self-consistency, etc.). Exploring this connection—how degeneracy relates to solution diversity and downstream robustness—is a promising direction for future work.

Limitations and scope.

Our analysis is intentionally local and mechanism-driven. The quadratic model isolates a sufficient within-basin mechanism for rise–then–decay, but real fine-tuning is nonstationary and may traverse regions with changing curvature and noise structure; in particular, additional nonconvex effects such as saddle crossings or basin-to-basin drift can also contribute (Appendix C, D, and the double-well metastability discussion in Appendix E). Our empirical probes use finite evaluation pools and task-specific rewards; while we quantify uncertainty and use headroom-normalized metrics, different evaluation protocols can shift absolute effect sizes.

Conclusion and future work.

Our results point to a landscape-centric view of fine-tuning: the effective geometry that governs improvement can be low-dimensional in curvature and need not grow proportionally with model size. Looking ahead, three directions follow naturally. First, develop practical and scalable estimates of curvature-active structure—and test how it changes with model scale—in settings where objectives are reward-defined and gradients may be unreliable. Second, use these estimates to design perturbation distributions and population-based optimizers that better target the curvature-active subspace (e.g., structured or learned perturbations, subspace methods, and principled hybrids with gradient updates). Third, extend the variance–geometry framework to anisotropic and state-dependent noise to more directly connect ES-style probes with policy-gradient fine-tuning dynamics.

References
A. Aghajanyan, S. Gupta, and L. Zettlemoyer (2021)	Intrinsic dimensionality explains the effectiveness of language model fine-tuning.In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (ACL-IJCNLP 2021), Volume 1: Long Papers,pp. 7319–7328.External Links: Document, Link, 2012.13255Cited by: §2.
S. Arora, Z. Li, and A. Panigrahi (2022)	Understanding gradient descent on the edge of stability in deep learning.In Proceedings of the 39th International Conference on Machine Learning (ICML),Proceedings of Machine Learning Research, Vol. 162, pp. 948–1024.Cited by: §2.
P. Clark, M. Cowhey, O. Etzioni, T. Khot, A. Sabharwal, C. Schoenick, and O. Tafjord (2018)	Think you have solved question answering? try ARC, the AI2 reasoning challenge.In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing,Cited by: §1.
K. Cobbe, V. Kosaraju, M. Bavarian, M. Chen, H. Jun, L. Kaiser, M. Plappert, J. Tworek, J. Hilton, R. Nakano, C. Hesse, and J. Schulman (2021)	Training verifiers to solve math word problems.CoRR abs/2110.14168.External Links: Link, 2110.14168Cited by: Figure 1, Figure 1, §1.
S. Dohare, Q. Lan, and A. R. Mahmood (2023)	Overcoming policy collapse in deep reinforcement learning.In European Workshop on Reinforcement Learning (EWRL) 2023,Note: OpenReview: https://openreview.net/forum?id=m9Jfdz4ymOCited by: §2.
J. C. Duchi, M. I. Jordan, M. J. Wainwright, and A. Wibisono (2015)	Optimal rates for zero-order convex optimization: the power of two function evaluations.IEEE Transactions on Information Theory 61 (5), pp. 2788–2806.External Links: Document, 1312.2139, LinkCited by: §1, §2.
B. Ghorbani, S. Krishnan, and Y. Xiao (2019)	An investigation into neural net optimization via hessian eigenvalue density.In Proceedings of the 36th International Conference on Machine Learning,Proceedings of Machine Learning Research, Vol. 97, pp. 2232–2241.Cited by: §B.2, §2.
G. Gur-Ari, D. A. Roberts, and E. Dyer (2018)	Gradient descent happens in a tiny subspace.CoRR abs/1812.04754.External Links: Link, 1812.04754Cited by: §2.
P. Hänggi, P. Talkner, and M. Borkovec (1990)	Reaction-rate theory: fifty years after kramers.Reviews of Modern Physics 62 (2), pp. 251–341.External Links: Document, LinkCited by: §E.2.
E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen (2021)	LoRA: low-rank adaptation of large language models.External Links: 2106.09685, Document, LinkCited by: §2.
M. F. Hutchinson (1990)	A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines.Communications in Statistics - Simulation and Computation 19 (2), pp. 433–450.External Links: DocumentCited by: §B.2.
C. Jin, R. Ge, P. Netrapalli, S. M. Kakade, and M. I. Jordan (2017)	How to escape saddle points efficiently.In Proceedings of the 34th International Conference on Machine Learning,Proceedings of Machine Learning Research, Vol. 70, pp. 1724–1732.External Links: LinkCited by: Appendix D.
T. Lelièvre, D. Le Peutrec, and B. Nectoux (2025)	Eyring-kramers exit rates for the overdamped langevin dynamics: the case with saddle points on the boundary.Journal de l’École polytechnique — Mathématiques 12, pp. 881–982.External Links: Document, LinkCited by: §E.2.
C. Li, H. Farkhoor, R. Liu, and J. Yosinski (2018)	Measuring the intrinsic dimension of objective landscapes.In International Conference on Learning Representations (ICLR),External Links: 1804.08838, Document, LinkCited by: §2.
Q. Li, C. Tai, and W. E (2017)	Stochastic modified equations and adaptive stochastic gradient algorithms.In Proceedings of the 34th International Conference on Machine Learning, D. Precup and Y. W. Teh (Eds.),Proceedings of Machine Learning Research, Vol. 70, pp. 2101–2110.External Links: LinkCited by: §C.1, §2.
S. Mandt, M. D. Hoffman, and D. M. Blei (2017)	Stochastic gradient descent as approximate bayesian inference.Journal of Machine Learning Research 18 (134), pp. 1–35.External Links: Link, 1704.04289, DocumentCited by: §C.1, §2.
S. Moalla, A. Miele, D. Pyatko, R. Pascanu, and C. Gulcehre (2024)	No representation, no trust: connecting representation, collapse, and trust issues in ppo.In Proceedings of the 38th International Conference on Neural Information Processing Systems,NIPS ’24, Red Hook, NY, USA.External Links: ISBN 9798331314385Cited by: §2.
X. Qiu, Y. Gan, C. F. Hayes, Q. Liang, E. Meyerson, B. Hodjat, and R. Miikkulainen (2025)	Evolution strategies at scale: LLM fine-tuning beyond reinforcement learning.Note: Also available on OpenReviewExternal Links: 2509.24372, Document, LinkCited by: §1, §2.
L. Sagun, U. Evci, V. U. Guney, Y. Dauphin, and L. Bottou (2017)	Empirical analysis of the hessian of over-parametrized neural networks.External Links: 1706.04454, Document, LinkCited by: §2.
K. Sakaguchi, R. Le Bras, C. Bhagavatula, and Y. Choi (2020)	WinoGrande: an adversarial winograd schema challenge at scale.In Proceedings of the AAAI Conference on Artificial Intelligence,Cited by: §1.
T. Salimans, J. Ho, X. Chen, S. Sidor, and I. Sutskever (2017)	Evolution strategies as a scalable alternative to reinforcement learning.External Links: 1703.03864, Document, LinkCited by: §2.
J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov (2017)	Proximal policy optimization algorithms.CoRR abs/1707.06347.External Links: Link, 1707.06347Cited by: §1.
Z. Shao, P. Wang, Q. Zhu, R. Xu, J. Song, X. Bi, H. Zhang, M. Zhang, Y. K. Li, Y. Wu, and D. Guo (2024)	DeepSeekMath: pushing the limits of mathematical reasoning in open language models.External Links: 2402.03300, Document, LinkCited by: §1.
S. Ubaru, J. Chen, and Y. Saad (2017)	Fast estimation of 
tr
​
(
𝑓
​
(
𝐴
)
)
 via stochastic lanczos quadrature.SIAM Journal on Matrix Analysis and Applications 38 (4), pp. 1075–1099.External Links: DocumentCited by: §B.2.
Z. Yao, A. Gholami, K. Keutzer, and M. W. Mahoney (2019)	PyHessian: neural networks through the lens of the hessian.Note: Also appeared in IEEE BigData 2020 / ICML workshopExternal Links: 1912.07145, Document, LinkCited by: §B.2.
X. Zhu, Z. Wang, X. Wang, M. Zhou, and R. Ge (2023)	Understanding edge-of-stability training dynamics with a minimalist example.In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023,External Links: LinkCited by: §2.
Appendix
Appendix Table of Contents
 A.	
Quadratic OU Toy Model and Variance–Curvature Spectroscopy.
	A
B.	
From Toy Slope Laws to Practical Curvature Probes: CLSS vs. SLQ.
	B
C.	
Anisotropic Noise and the Lyapunov Equation.
	C
D.	
Beyond Quadratic: Saddles and Negative Curvature.
	D
E.	
Metastability in a Double-Well: Escape Times and Hopping Criteria.
	E
F.	
Additional Rise–Then–Decay Results: Countdown.
	F
G.	
Experimental Details: ES Reward-Probe.
	G
Appendix AQuadratic OU Toy Model and Variance–Curvature Spectroscopy

This appendix provides analytic details underlying the toy-model mechanism used in Section 4. The main text uses this quadratic OU analysis as a mechanism isolate: it shows that rise–then–decay dynamics arise generically from (i) heterogeneous curvature scales and (ii) fixed stochasticity, without requiring nonconvex pathologies. We also show how the same closed-form expressions yield a “spectroscopy” viewpoint: the terminal plateau varies predictably with the effective noise scale 
𝜅
=
𝜎
2
/
𝑁
, and the slope encodes a curvature-weighted effective dimension. This slope-fitting idea motivates the CLSS procedure introduced in the next appendix section (Section B).

A.1From noisy updates to a local quadratic OU model
A generic noisy update abstraction.

We consider maximizing an objective 
𝐽
​
(
𝜃
)
 with a constant-step noisy update

	
𝜃
𝑡
+
1
=
𝜃
𝑡
+
𝛼
​
(
𝑔
​
(
𝜃
𝑡
)
+
𝜉
𝑡
)
,
		
(11)

where 
𝔼
​
[
𝜉
𝑡
∣
𝜃
𝑡
]
=
0
 and 
Cov
​
(
𝜉
𝑡
∣
𝜃
𝑡
)
=
Σ
​
(
𝜃
𝑡
)
. For ES with isotropic perturbations, a useful local approximation is 
Σ
​
(
𝜃
𝑡
)
≈
(
𝜎
2
/
𝑁
)
​
𝐼
 (up to problem-dependent scaling), making the effective noise level explicitly tunable by 
𝜅
=
𝜎
2
/
𝑁
. Policy-gradient methods (e.g., GRPO) have anisotropic 
Σ
​
(
𝜃
𝑡
)
; the analysis below extends by replacing 
(
𝜎
2
/
𝑁
)
​
𝐼
 with a general covariance and solving a Lyapunov equation, but we focus here on the isotropic case to isolate curvature effects.

Why a quadratic model?

The rise–then–decay signature in Section 4 indicates a regime where (i) gradients are small and (ii) stochasticity dominates long-time behavior. In this regime, a local Taylor expansion around a critical point 
𝜃
⋆
 is the natural first approximation:

	
𝐽
​
(
𝜃
⋆
+
𝑥
)
≈
𝐽
​
(
𝜃
⋆
)
+
1
2
​
𝑥
⊤
​
𝐻
​
𝑥
,
		
(12)

where 
𝐻
=
∇
2
𝐽
​
(
𝜃
⋆
)
. Near a local maximizer, 
𝐻
 is negative semidefinite; we reparameterize curvature by 
𝐶
≜
−
𝐻
⪰
0
. The quadratic model is not intended as a global description of LLM fine-tuning; rather, it yields exact mode-wise relaxation rates and noise floors, making the origin of non-monotonicity transparent.

A.2Quadratic toy model: exact dynamics and a criterion for non-monotonicity
Setup.

Let 
𝜃
𝑡
∈
ℝ
𝐷
 and define the quadratic reward

	
𝐽
​
(
𝜃
)
=
1
−
1
2
​
𝜃
⊤
​
𝐶
​
𝜃
,
		
(13)

with maximizer 
𝜃
⋆
=
0
. Then 
∇
𝐽
​
(
𝜃
)
=
−
𝐶
​
𝜃
. We model ES-like noisy ascent by

	
𝜃
𝑡
+
1
=
(
𝐼
−
𝛼
​
𝐶
)
​
𝜃
𝑡
+
𝛼
​
𝜎
𝑁
​
𝜀
𝑡
,
𝜀
𝑡
∼
𝒩
​
(
0
,
𝐼
𝐷
)
,
		
(14)

and assume 
𝛼
∈
(
0
,
2
/
𝜆
max
​
(
𝐶
)
)
 for stability on the range of 
𝐶
.

Diagonalization and mode-wise OU/AR(1).

Let 
𝐶
=
𝑄
​
Λ
​
𝑄
⊤
 with 
Λ
=
diag
​
(
𝜆
1
,
…
,
𝜆
𝐷
)
. In the eigenbasis 
𝑥
𝑡
=
𝑄
⊤
​
𝜃
𝑡
, modes decouple:

	
𝑥
𝑖
,
𝑡
+
1
=
𝑎
𝑖
​
𝑥
𝑖
,
𝑡
+
𝑏
​
𝜉
𝑖
,
𝑡
,
𝑎
𝑖
=
1
−
𝛼
​
𝜆
𝑖
,
𝑏
=
𝛼
​
𝜎
/
𝑁
,
𝜉
𝑖
,
𝑡
∼
𝒩
​
(
0
,
1
)
.
		
(15)

Thus each spectral mode is an independent AR(1)/OU process. For small 
𝛼
​
𝜆
𝑖
, the characteristic decay time scales as 
𝜏
𝑖
≈
(
𝛼
​
𝜆
𝑖
)
−
1
: stiff modes (
𝜆
𝑖
 large) relax quickly while flat modes (
𝜆
𝑖
 small) relax slowly.

Mean/variance dynamics and stationary noise floor.

Let 
𝜇
𝑖
,
𝑡
=
𝔼
​
[
𝑥
𝑖
,
𝑡
]
 and 
𝑣
𝑖
,
𝑡
=
Var
​
(
𝑥
𝑖
,
𝑡
)
. From (15):

	
𝜇
𝑖
,
𝑡
=
𝑎
𝑖
𝑡
​
𝜇
𝑖
,
0
,
𝑣
𝑖
,
𝑡
=
𝑎
𝑖
2
​
𝑡
​
𝑣
𝑖
,
0
+
𝑏
2
1
−
𝑎
𝑖
2
​
(
1
−
𝑎
𝑖
2
​
𝑡
)
.
		
(16)

For 
𝜆
𝑖
>
0
 and 
|
𝑎
𝑖
|
<
1
, the stationary variance is

	
𝑣
𝑖
,
∞
=
𝑏
2
1
−
𝑎
𝑖
2
=
𝛼
​
𝜎
2
𝑁
​
𝜆
𝑖
​
(
2
−
𝛼
​
𝜆
𝑖
)
.
		
(17)

The expected reward is

	
𝔼
​
[
𝐽
​
(
𝜃
𝑡
)
]
=
1
−
1
2
​
∑
𝑖
:
𝜆
𝑖
>
0
𝜆
𝑖
​
(
𝜇
𝑖
,
𝑡
2
+
𝑣
𝑖
,
𝑡
)
.
		
(18)

At stationarity (
𝑡
→
∞
), 
𝜇
𝑖
,
𝑡
→
0
 and 
𝑣
𝑖
,
𝑡
→
𝑣
𝑖
,
∞
, giving

	
1
−
𝐽
∞
=
1
2
​
∑
𝜆
𝑖
>
0
𝜆
𝑖
​
𝑣
𝑖
,
∞
=
𝛼
​
𝜎
2
2
​
𝑁
​
∑
𝜆
𝑖
>
0
1
2
−
𝛼
​
𝜆
𝑖
.
		
(19)

Thus the terminal plateau is controlled by the effective noise 
𝜅
=
𝜎
2
/
𝑁
 and by a curvature-weighted trace functional of the spectrum.

Mixture-of-exponentials and non-monotonicity.

Define amplitudes

	
𝐴
𝑖
≜
−
1
2
​
𝜆
𝑖
​
(
𝑥
𝑖
,
0
2
−
𝑣
𝑖
,
∞
)
,
		
(20)

assuming deterministic initialization 
𝑥
𝑖
,
0
.

Then the deviation from plateau is a mixture of exponentials:

	
𝔼
​
[
𝐽
​
(
𝜃
𝑡
)
]
−
𝐽
∞
=
∑
𝜆
𝑖
>
0
𝐴
𝑖
​
𝑎
𝑖
2
​
𝑡
=
∑
𝜆
𝑖
>
0
𝐴
𝑖
​
𝑒
−
𝛾
𝑖
​
𝑡
,
𝛾
𝑖
≜
−
2
​
log
⁡
(
|
1
−
𝛼
​
𝜆
𝑖
|
)
.
		
(21)

Non-monotonicity occurs when the 
{
𝐴
𝑖
}
 have mixed signs: fast stiff modes can initially increase reward, while slow flat modes contribute delayed degradation as variance accumulates. A pronounced peak arises when stiff modes begin far from equilibrium (
𝑥
𝑖
,
0
2
≫
𝑣
𝑖
,
∞
) while flat modes begin near or beyond their noise-limited equilibrium (
𝑥
𝑖
,
0
2
≲
𝑣
𝑖
,
∞
), matching the main-text initialization discussion.

Figure 7:Mode-wise relaxation rates explain rise–then–decay. A two-mode illustration of (15)–(21) showing how curvature sets relaxation times. (a) Total reward 
𝔼
​
[
𝐽
𝑡
]
 peaks and then decays toward 
𝐽
∞
 as slow variance continues to accumulate after stiff-mode signal is exhausted (vertical line marks 
𝑡
peak
; inset shows the two-block spectrum). (b,d) In the stiff mode (
𝜆
ℎ
​
𝑖
), the mean 
𝜇
ℎ
​
𝑖
,
𝑡
 decays rapidly and the variance 
𝑣
ℎ
​
𝑖
,
𝑡
 saturates quickly. (c,e) In the flat mode (
𝜆
𝑙
​
𝑜
), the mean decays slowly while the variance accumulates over long time, producing delayed degradation. Notice that the flat-mode terminal variance is substantially larger and dominates late-time reward degradation.
Peak time: a two-timescale characterization.

The mixture form (21) implies that peaks arise from competition between modes with different decay rates. Differentiating (treating 
𝑡
 as continuous for analysis) gives

	
𝑑
𝑑
​
𝑡
​
(
𝔼
​
[
𝐽
​
(
𝜃
𝑡
)
]
−
𝐽
∞
)
=
−
∑
𝜆
𝑖
>
0
𝛾
𝑖
​
𝐴
𝑖
​
𝑒
−
𝛾
𝑖
​
𝑡
.
		
(22)

An interior maximizer 
𝑡
peak
>
0
 therefore satisfies the implicit condition

	
∑
𝜆
𝑖
>
0
𝛾
𝑖
​
𝐴
𝑖
​
𝑒
−
𝛾
𝑖
​
𝑡
peak
=
0
,
		
(23)

which is possible only when the amplitudes 
{
𝐴
𝑖
}
 have mixed signs.

Closed form in the two-mode case.

To make the dependence explicit, consider two distinct eigenvalues 
𝜆
hi
>
𝜆
lo
>
0
 with rates 
𝛾
hi
=
−
2
​
log
⁡
(
1
−
𝛼
​
𝜆
hi
)
 and 
𝛾
lo
=
−
2
​
log
⁡
(
1
−
𝛼
​
𝜆
lo
)
, and amplitudes 
𝐴
hi
 and 
𝐴
lo
 as in (20). Then

	
𝔼
​
[
𝐽
​
(
𝜃
𝑡
)
]
−
𝐽
∞
=
𝐴
hi
​
𝑒
−
𝛾
hi
​
𝑡
+
𝐴
lo
​
𝑒
−
𝛾
lo
​
𝑡
.
		
(24)

A rise–then–decay peak occurs when 
𝐴
hi
<
0
 and 
𝐴
lo
>
0
 (equivalently, 
𝑥
hi
,
0
2
>
𝑣
hi
,
∞
 and 
𝑥
lo
,
0
2
<
𝑣
lo
,
∞
), and the peak time is

	
𝑡
peak
=
1
𝛾
hi
−
𝛾
lo
​
log
⁡
(
𝛾
hi
​
|
𝐴
hi
|
𝛾
lo
​
𝐴
lo
)
.
		
(25)

This expression makes two qualitative dependencies explicit: larger time-scale separation (
𝛾
hi
≫
𝛾
lo
) and larger amplitude ratio (
|
𝐴
hi
|
/
𝐴
lo
) yield a more pronounced and later peak.

A.3Variance–curvature “spectroscopy” in the toy model
Plateau versus effective noise and curvature-weighted dimension.

Equation (19) implies that, in a locally quadratic and stable regime, the terminal gap scales linearly with 
𝜅
=
𝜎
2
/
𝑁
:

	
1
−
𝐽
∞
=
(
𝛼
2
​
∑
𝜆
𝑖
>
0
1
2
−
𝛼
​
𝜆
𝑖
)
⏟
curvature functional
⋅
𝜅
.
		
(26)

This motivates the curvature-weighted effective dimension

	
𝑑
eff
​
(
𝛼
)
≜
 2
​
∑
𝜆
𝑖
>
0
1
2
−
𝛼
​
𝜆
𝑖
,
⇒
1
−
𝐽
∞
=
𝛼
4
​
𝑑
eff
​
(
𝛼
)
⋅
𝜅
.
		
(27)

For 
𝛼
​
𝜆
𝑖
≪
1
, 
𝑑
eff
​
(
𝛼
)
≈
rank
​
(
𝐶
)
; at finite step size it smoothly reweights directions by curvature through 
(
2
−
𝛼
​
𝜆
𝑖
)
−
1
. Thus, in the toy quadratic setting, a slope fit of 
(
1
−
𝐽
∞
)
 versus 
𝜅
 recovers a scalar summary of local curvature complexity.

Slope fitting as “spectroscopy” in the toy setting.

Figure 8 illustrates this viewpoint. Varying population size 
𝑁
 changes 
𝜅
 and produces different terminal plateaus in the reward trajectory (panel a). Plotting the analytic plateau 
𝐽
∞
 against 
𝜅
 yields an approximately linear dependence whose slope increases with the number of curvature-active directions in a strict rank-
𝑑
 spectrum (panel b). In other words, higher effective dimension implies greater sensitivity of the noise floor to stochasticity. This motivates the checkpointed noise-floor probing method introduced next (Section B).

Figure 8:Toy dynamics and plateau “spectroscopy” via effective noise 
𝜅
=
𝜎
2
/
𝑁
. (a) Expected reward trajectories for different populations 
𝑁
 (noise levels) on the two-block quadratic landscape, comparing Monte Carlo ES simulations to the analytic OU prediction; larger 
𝑁
 reduces effective noise and raises the terminal plateau. (b) Analytic terminal plateau 
𝐽
∞
 versus effective noise 
𝜅
 for strict rank-
𝑑
 curvature. The approximately linear dependence predicted by Eq. (26) has a slope that grows with 
𝑑
, showing that plateau-vs-noise measurements recover a curvature-weighted effective dimension in the toy setting.
Appendix BFrom Toy Slope Laws to Practical Curvature Probes: CLSS vs. SLQ

The quadratic OU analysis in Appendix A yields a concrete measurement principle: in a locally stable region, the terminal “noise-floor” gap scales approximately linearly with the effective noise level 
𝜅
=
𝜎
2
/
𝑁
 (Eq. (26)), and the slope can be mapped to a curvature-weighted effective dimension (Eq. (27)). In the toy setting this relation is exact and enables a simple slope fit to recover 
𝑑
eff
 (Fig. 8). This motivates a practical idea for black-box or reward-defined objectives where Hessian-vector products may be unavailable: estimate local geometry by checkpointed noise-floor probing and slope fitting. We refer to this as Checkpointed Local Slope Spectroscopy (CLSS). We then contrast CLSS with Stochastic Lanczos Quadrature (SLQ), a standard matvec-based spectrum probe.

B.1Checkpointed Local Slope Spectroscopy (CLSS)

Given a checkpoint 
𝜃
⋆
, CLSS runs short, local probe trajectories at several controllable noise levels (e.g., varying 
𝑁
 at fixed 
𝜎
 in ES), estimates a plateau proxy for each noise level, and fits the small-noise slope of the plateau gap versus 
𝜅
=
𝜎
2
/
𝑁
. In the quadratic regime, this slope maps to an effective dimension via Eq. (26)–(27). The algorithmic description of CLSS is given in Algorithm 2.

Algorithm 2 Checkpointed Local Slope Spectroscopy (CLSS)

Input: checkpoint 
𝜃
⋆
; perturbation scale 
𝜎
; candidate step sizes 
𝒜
; population sizes 
𝒩
; probe horizon 
𝑇
; tail window 
𝑤
; #seeds 
𝑅
;

locality metric 
Loc
​
(
𝜃
,
𝜃
⋆
)
 and threshold 
𝜏
loc
; tail-settling tolerance 
𝜏
stat
; minimum valid seeds 
𝑅
min
.

Output: 
𝑑
^
eff
​
(
𝛼
;
𝜃
⋆
)
 with CI, or FAIL.

foreach 
𝛼
∈
𝒜
 do

    // (1) Local probes at fixed 
𝛼
,
𝜎
 and varying population 
𝑁
 (noise 
𝜅
=
𝜎
2
/
𝑁
) foreach 
𝑁
∈
𝒩
 do
       
𝒱
←
∅
 ;
       // valid seedsfor 
𝑟
=
1
 to 
𝑅
 do
          Run 
𝑇
 probe steps from 
𝜃
⋆
 with 
(
𝛼
,
𝜎
,
𝑁
)
 producing rewards 
{
𝐽
𝑡
}
𝑡
=
1
𝑇
 and states 
{
𝜃
𝑡
}
𝑡
=
1
𝑇
  if 
max
𝑡
≤
𝑇
⁡
Loc
​
(
𝜃
𝑡
,
𝜃
⋆
)
>
𝜏
loc
 then
            continue
         
𝐽
¯
1
←
1
𝑤
​
∑
𝑡
=
𝑇
−
𝑤
+
1
𝑇
𝐽
𝑡
  
𝐽
¯
0
←
1
𝑤
​
∑
𝑡
=
𝑇
−
2
​
𝑤
+
1
𝑇
−
𝑤
𝐽
𝑡
  if 
|
𝐽
¯
1
−
𝐽
¯
0
|
>
𝜏
stat
 then
            continue
         
𝒱
←
𝒱
∪
{
𝐽
¯
1
}
 
      if 
|
𝒱
|
<
𝑅
min
 then
         mark 
𝑁
 invalid; continue
      
𝐽
^
∞
​
(
𝑁
)
←
mean
​
(
𝒱
)
 
   // (2) Fit small-noise slope of plateau gap vs 
𝜅
=
𝜎
2
/
𝑁
 Let 
𝑁
max
 be the largest valid 
𝑁
 and set 
𝐽
^
ref
←
𝐽
^
∞
​
(
𝑁
max
)
  For each valid 
𝑁
, set 
𝑔
​
(
𝑁
)
←
𝐽
^
ref
−
𝐽
^
∞
​
(
𝑁
)
  Fit 
𝑔
​
(
𝑁
)
≈
𝑆
𝛼
⋅
(
𝜎
2
/
𝑁
)
+
𝑏
 using the largest few valid 
𝑁
 (small-noise regime)  Output 
𝑑
^
eff
​
(
𝛼
;
𝜃
⋆
)
←
4
𝛼
​
𝑆
𝛼
 ;
    // by Eq. (27)if fit residuals small and acceptance rates high then
      return 
𝑑
^
eff
​
(
𝛼
;
𝜃
⋆
)
   
return FAIL.
Practical limitations.

CLSS is conceptually clean in the quadratic regime but can be difficult to apply reliably in real LLM fine-tuning. It requires (i) a local probe regime (trajectories must remain near 
𝜃
⋆
), and (ii) a measurable plateau proxy over the probe horizon. In practice, nonconvex drift across regions, long transients, and bounded/truncated rewards (e.g., accuracy in 
[
0
,
1
]
) can obscure plateaus and inflate uncertainty. Moreover, effective noise is often anisotropic and state-dependent (particularly for policy-gradient methods), weakening a direct mapping from a scalar slope to a single curvature functional. For these reasons, in the main text we use the slope law primarily as a mechanistic explanation and rely on more robust operational probes (e.g., best-of-
𝑁
 accessibility) that do not require clean plateau observation.

B.2Stochastic Lanczos Quadrature (SLQ): matvec-based curvature probing

SLQ estimates trace functionals and (optionally) spectral densities of a symmetric operator 
𝐴
 using only matrix–vector products 
𝑣
↦
𝐴
​
𝑣
. It combines Hutchinson trace estimation with Lanczos tridiagonalization and Gaussian quadrature to approximate quantities of the form 
tr
​
(
𝑓
​
(
𝐴
)
)
 with 
𝒪
​
(
𝑠
​
𝑚
)
 matvecs, where 
𝑠
 is the number of random probes and 
𝑚
 the number of Lanczos steps (Hutchinson, 1990; Ubaru et al., 2017). Tooling such as PyHessian implements related workflows for estimating top eigenvalues, trace, and smoothed spectral densities from Hessian–vector products (Yao et al., 2019; Ghorbani et al., 2019). When a reliable differentiable surrogate loss is available and matvecs are stable, SLQ provides rich spectral diagnostics (bulk/outliers, sharp subspaces) that complement our dynamics-based analysis. The algorithmic description of SLQ is given in Algorithm 3.

Algorithm 3 Stochastic Lanczos Quadrature (SLQ) for 
tr
​
(
𝑓
​
(
𝐴
)
)

Input: symmetric matrix/operator 
𝐴
 (accessed only via matvec 
𝑣
↦
𝐴
​
𝑣
); scalar function 
𝑓
; #probes 
𝑠
; Lanczos steps 
𝑚
.

Output: estimate 
tr
​
(
𝑓
​
(
𝐴
)
)
^
 and (optionally) a standard error.

// Hutchinson trace estimator: 
tr
​
(
𝑓
​
(
𝐴
)
)
=
𝔼
​
[
𝑧
⊤
​
𝑓
​
(
𝐴
)
​
𝑧
]
 for 
𝔼
​
[
𝑧
​
𝑧
⊤
]
=
𝐼

for 
𝑗
=
1
 to 
𝑠
 do

    Sample probe 
𝑧
𝑗
∈
{
±
1
}
𝑛
 i.i.d. Rademacher (or 
𝑧
𝑗
∼
𝒩
​
(
0
,
𝐼
)
).
Normalize 
𝑞
1
←
𝑧
𝑗
/
‖
𝑧
𝑗
‖
2
.
// Lanczos tridiagonalization using only matvecs with 
𝐴
 Run 
𝑚
-step Lanczos on 
𝐴
 with start 
𝑞
1
 to obtain tridiagonal 
𝑇
𝑗
∈
ℝ
𝑚
×
𝑚
.
// Gaussian quadrature: 
𝑧
⊤
​
𝑓
​
(
𝐴
)
​
𝑧
≈
‖
𝑧
‖
2
​
𝑒
1
⊤
​
𝑓
​
(
𝑇
)
​
𝑒
1
 Compute eigendecomposition 
𝑇
𝑗
=
𝑉
𝑗
​
diag
​
(
𝜃
𝑗
)
​
𝑉
𝑗
⊤
.
Set weights 
𝑤
𝑗
​
𝑘
←
(
𝑉
𝑗
)
1
​
𝑘
2
 for 
𝑘
=
1
,
…
,
𝑚
.
Estimate quadratic form:
	
𝜏
^
𝑗
←
‖
𝑧
𝑗
‖
2
2
​
∑
𝑘
=
1
𝑚
𝑤
𝑗
​
𝑘
​
𝑓
​
(
𝜃
𝑗
​
𝑘
)
.
		
(28)
	
tr
​
(
𝑓
​
(
𝐴
)
)
^
←
1
𝑠
​
∑
𝑗
=
1
𝑠
𝜏
^
𝑗
,
SE
^
←
Var
​
(
{
𝜏
^
𝑗
}
𝑗
=
1
𝑠
)
𝑠
​
(
optional
)
.
		
(29)
return 
tr
​
(
𝑓
​
(
𝐴
)
)
^
 (and 
SE
^
).
B.3When to use what
• 

Use SLQ/PyHessian when matvecs are available. If you can compute stable matvecs for a symmetric curvature operator tied to a differentiable surrogate (Hessian/GGN/Fisher), SLQ yields detailed spectral structure and top-eigenspace information.

• 

Use CLSS when only objective evaluations are available and plateaus are measurable. CLSS is applicable to black-box rewards and algorithm-native noise control (e.g., varying 
𝑁
 in ES), but only to the extent that local probes remain near a checkpoint and produce a reliably measurable plateau.

• 

In our LLM setting, favor operational probes. Because plateau estimation is often unreliable under bounded/truncated rewards and nonstationarity, we do not use CLSS as a primary measurement and instead emphasize robust probes (best-of-
𝑁
) and dynamical signatures (rise–then–decay) in the main paper.

Method	
What it needs
	
What it gives

SLQ / PyHessian	
Matvec access 
𝑣
↦
𝐴
​
𝑣
 for a symmetric curvature operator (typically via differentiable surrogate)
	
Rich spectral diagnostics: top eigenvalues/eigenspace, trace functionals, smoothed spectral density

CLSS (slope probe)	
Only objective evaluations under the actual optimizer; controllable noise 
𝜅
=
𝜎
2
/
𝑁
; a measurable local plateau
	
Scalar slope-based summary interpretable as curvature-weighted effective dimension in the quadratic regime
Table 1:High-level comparison. SLQ provides detailed spectral information when matvecs are available; CLSS is inspired by the toy slope law and can be used with black-box rewards, but requires reliable local plateau estimation.
Appendix CAnisotropic Noise and the Lyapunov Equation

The quadratic OU toy model in Appendix A assumed isotropic noise for clarity, which diagonalizes neatly in the eigenbasis of the curvature matrix. In practice, stochastic learning rules inject anisotropic and often state-dependent noise. This appendix summarizes the corresponding linearized theory, which shows that the same geometry–variance mechanism persists even when modes do not decouple: curvature still sets relaxation, while the noise covariance determines how variance accumulates across directions.

C.1Linearized dynamics with anisotropic noise

Near a critical point 
𝜃
⋆
, a general linearized update takes the form

	
𝑥
𝑡
+
1
=
(
𝐼
−
𝛼
​
𝐻
)
​
𝑥
𝑡
+
𝛼
​
𝜂
𝑡
,
𝔼
​
[
𝜂
𝑡
]
=
0
,
Cov
​
(
𝜂
𝑡
)
=
Σ
,
		
(30)

where 
𝑥
𝑡
=
𝜃
𝑡
−
𝜃
⋆
 and 
𝐻
 is the local curvature matrix (for reward maximization, 
𝐻
=
−
𝐶
⪰
0
 in our sign convention). Let 
𝑉
𝑡
=
𝔼
​
[
𝑥
𝑡
​
𝑥
𝑡
⊤
]
. Then

	
𝑉
𝑡
+
1
=
(
𝐼
−
𝛼
​
𝐻
)
​
𝑉
𝑡
​
(
𝐼
−
𝛼
​
𝐻
)
⊤
+
𝛼
2
​
Σ
.
		
(31)

At stationarity (
𝑉
𝑡
+
1
=
𝑉
𝑡
=
𝑉
), this is the discrete Lyapunov equation. For small 
𝛼
, expanding yields the continuous-time OU covariance balance

	
𝐻
​
𝑉
+
𝑉
​
𝐻
⊤
≈
𝛼
​
Σ
,
		
(32)

matching standard OU/SDE approximations of constant-step stochastic optimization (Mandt et al., 2017; Li et al., 2017).

How the plateau depends on anisotropic noise.

In the quadratic approximation 
𝐽
​
(
𝜃
⋆
+
𝑥
)
≈
𝐽
​
(
𝜃
⋆
)
−
1
2
​
𝑥
⊤
​
𝐻
​
𝑥
, the expected performance gap depends on the stationary second moment:

	
𝐽
​
(
𝜃
⋆
)
−
𝔼
​
[
𝐽
​
(
𝜃
⋆
+
𝑥
)
]
≈
1
2
​
tr
​
(
𝐻
​
𝑉
)
.
		
(33)

Thus, even when 
𝐻
 is low-dimensional in curvature (few stiff modes), the effective noise floor depends on how 
Σ
 injects variance into those modes. Compared to the isotropic case (Appendix A), anisotropy can (i) selectively amplify variance in particular curvature directions, (ii) rotate variance into stiff directions through off-diagonal structure, and (iii) change the apparent “effective dimension” by reweighting modes through 
Σ
 rather than through 
𝐻
 alone.

Stability and the OU regime.

A stationary 
𝑉
 exists only when the linear drift is stable: the spectral radius of 
(
𝐼
−
𝛼
​
𝐻
)
 must be 
<
1
 (in the reward-maximization convention, this corresponds to 
𝐻
⪰
0
 on the relevant subspace and 
𝛼
 sufficiently small). When this holds, (31) provides a precise statement of the “variance accumulation” mechanism in the presence of anisotropy: the curvature 
𝐻
 sets contraction, while 
Σ
 sets variance injection.

C.2Policy-gradient / GRPO noise structure

Policy-gradient methods provide a concrete example of anisotropic noise. Let 
ℒ
​
(
𝜃
)
 denote a differentiable surrogate loss used for optimization (e.g., a GRPO/PPO-style objective). A generic stochastic gradient update has the form

	
𝜃
𝑡
+
1
=
𝜃
𝑡
−
𝛼
​
𝑔
^
𝑡
,
𝑔
^
𝑡
=
∇
ℒ
​
(
𝜃
𝑡
)
+
𝜁
𝑡
,
		
(34)

where 
𝜁
𝑡
 is stochastic gradient noise induced by sampling trajectories/tokens and minibatches. Under local linearization around 
𝜃
⋆
, this yields (30) with 
𝐻
=
∇
2
ℒ
​
(
𝜃
⋆
)
 and 
Σ
≈
Cov
​
(
𝜁
𝑡
)
.

Score-function form and Fisher-like covariance.

For a standard policy gradient estimator (for notational simplicity suppressing state conditioning),

	
𝑔
^
=
−
1
𝐵
​
∑
𝑏
=
1
𝐵
𝐴
^
𝑏
​
∇
𝜃
log
⁡
𝜋
𝜃
​
(
𝑎
𝑏
)
,
		
(35)

where 
𝐴
^
𝑏
 is an advantage-like scalar and 
𝐵
 is the batch size. Conditioned on 
𝜃
, the covariance of 
𝑔
^
 can be decomposed into (i) a Fisher-like term from 
∇
log
⁡
𝜋
 and (ii) scaling/mixing induced by the random advantages:

	
Σ
≈
Cov
​
(
𝑔
^
)
=
1
𝐵
​
(
𝔼
​
[
𝐴
^
2
​
(
∇
log
⁡
𝜋
)
​
(
∇
log
⁡
𝜋
)
⊤
]
−
𝔼
​
[
𝐴
^
​
∇
log
⁡
𝜋
]
​
𝔼
​
[
𝐴
^
​
∇
log
⁡
𝜋
]
⊤
)
.
		
(36)

Even when 
𝐻
 is low-dimensional in curvature, the effective noise floor is determined by how this anisotropic 
Σ
 projects onto curvature-active directions through the Lyapunov equation (32).

Effect of GRPO group normalization.

GRPO introduces a group-relative normalization that centers (and often scales) advantages within groups. At a high level, centering acts like a control variate that removes components of noise aligned with the within-group mean, while scaling by a within-group standard deviation can change the effective noise magnitude and induce correlations across samples within a group. Consequently, GRPO changes 
Σ
 in a structured, data-dependent manner rather than simply reducing variance uniformly. This provides a natural explanation for why non-monotonic reward dynamics can appear in GRPO as well as ES (Fig. 1): even if the mean update direction improves performance early, anisotropic noise interacting with heterogeneous curvature can drive a variance-dominated late regime.

Empirical proxies for 
Σ
.

In practice, 
Σ
 can be estimated (approximately) by minibatch gradient covariance:

	
Σ
^
=
1
𝐵
−
1
​
∑
𝑏
=
1
𝐵
(
𝑔
^
𝑏
−
𝑔
¯
)
​
(
𝑔
^
𝑏
−
𝑔
¯
)
⊤
,
		
(37)

or by low-rank projections (e.g., Hutchinson probing or subspace restrictions) when full matrices are infeasible. Combined with top-eigenspace estimates of 
𝐻
 (when available), the Lyapunov relation provides a principled way to predict which directions will accumulate variance and when a variance-dominated regime is expected.

Appendix DBeyond Quadratic: Saddles and Negative Curvature

The OU analysis in the main text and Appendix A describes within-basin dynamics around a locally stable region (a local maximizer for reward, or minimizer for loss). Real fine-tuning landscapes are nonconvex and nonstationary, and two additional effects can complicate plateau-based reasoning and amplify non-monotonicity.

Saddles and negative curvature.

OU stationarity requires stable drift. Near strict saddles, negative-curvature directions make the linear drift unstable (e.g., eigenvalues of 
(
𝐼
−
𝛼
​
𝐻
)
 have magnitude 
>
1
), so a stationary local covariance does not exist. In practice, stochasticity can help escape saddles; perturbed gradient methods escape strict saddles efficiently under suitable conditions (Jin et al., 2017). In such regions, apparent “plateaus” can be transient and slope-based noise-floor estimation can be misleading.

Basin-to-basin transitions and nonstationarity.

Even near locally stable regions, finite-step stochastic updates can drift across regions of the landscape, especially along flat directions. This can invalidate single-basin plateau assumptions and produce additional non-monotonicity beyond the OU mechanism. For RL-style fine-tuning, further nonstationarity arises because the effective objective can change with the policy distribution, sampling temperature, KL penalties, or group normalization. Qualitatively, these effects are consistent with the observation that GRPO trajectories can also exhibit peak–then–decay behavior (Fig. 1), even though GRPO optimizes a differentiable surrogate: both the surrogate geometry and its noise structure can evolve during training.

Implications for our measurements.

These considerations motivate two design choices in the main paper. First, we treat rise–then–decay as a diagnostic of heterogeneous curvature and variance coupling, rather than as evidence of a literal stationary OU regime throughout training. Second, we emphasize operational probes (e.g., best-of-
𝑁
 accessibility) that do not require a clean, long-horizon plateau in a single basin, and we relegate plateau-based slope-fitting ideas to controlled toy settings and appendix discussion.

Appendix EMetastability in a Double-Well: Escape Times and Hopping Criteria

The quadratic OU analysis describes within-basin behavior near a locally stable region. To understand when stochastic optimization transitions between basins (“hopping”), we summarize the canonical metastable case: overdamped Langevin dynamics in a double-well. The key takeaway is that escape times depend exponentially on a barrier-to-noise ratio, so trajectories can appear either (i) effectively trapped or (ii) rapidly delocalized, with a narrow intermediate regime of rare hopping.

E.1From noisy gradient steps to an overdamped Langevin diffusion

Consider one-dimensional noisy gradient descent on a loss 
𝐿
​
(
𝑥
)
,

	
𝑥
𝑘
+
1
=
𝑥
𝑘
−
𝛼
​
𝐿
′
​
(
𝑥
𝑘
)
+
𝛼
​
𝜉
𝑘
,
𝜉
𝑘
∼
𝒩
​
(
0
,
𝜎
2
/
𝑁
)
.
		
(38)

For small step size 
𝛼
, (38) is the Euler–Maruyama discretization of the overdamped Langevin SDE

	
𝑑
​
𝑥
𝑡
=
−
𝐿
′
​
(
𝑥
𝑡
)
​
𝑑
​
𝑡
+
2
​
𝜀
​
𝑑
​
𝑊
𝑡
,
		
(39)

with effective noise intensity 
𝜀
 obtained by matching per-step variances. Identifying 
𝑑
​
𝑡
=
𝛼
 and matching 
2
​
𝜀
​
𝑑
​
𝑡
≈
𝛼
2
​
(
𝜎
2
/
𝑁
)
 yields

	
𝜀
≈
𝛼
​
𝜎
2
2
​
𝑁
=
𝛼
2
​
𝜅
,
		
(40)

where 
𝜅
=
𝜎
2
/
𝑁
 is the effective ES noise scale used throughout the paper.

E.2Eyring–Kramers escape time in 1D

Let 
𝑥
−
 be a local minimum of 
𝐿
 and let 
𝑧
 be the adjacent saddle (in 1D, the local maximum separating the well). Define the barrier height 
Δ
​
𝐿
≜
𝐿
​
(
𝑧
)
−
𝐿
​
(
𝑥
−
)
. In the small-noise regime 
𝜀
→
0
, the mean first exit time obeys the Eyring–Kramers law

	
𝔼
​
[
𝜏
esc
]
≈
2
​
𝜋
𝐿
′′
​
(
𝑥
−
)
​
|
𝐿
′′
​
(
𝑧
)
|
​
exp
⁡
(
Δ
​
𝐿
𝜀
)
,
		
(41)

up to lower-order corrections (Hänggi et al., 1990; Lelièvre et al., 2025). Since one discrete step corresponds to 
𝑑
​
𝑡
=
𝛼
, the expected number of iterations to escape is

	
𝔼
​
[
𝐾
esc
]
≈
1
𝛼
​
𝔼
​
[
𝜏
esc
]
≈
2
​
𝜋
𝛼
​
𝐿
′′
​
(
𝑥
−
)
​
|
𝐿
′′
​
(
𝑧
)
|
​
exp
⁡
(
Δ
​
𝐿
𝜀
)
.
		
(42)

Consequently, the probability of at least one hop within 
𝑇
 iterations is approximately

	
Pr
⁡
(
hop by 
​
𝑇
)
≈
 1
−
exp
⁡
(
−
𝑇
𝔼
​
[
𝐾
esc
]
)
≈
𝑇
𝔼
​
[
𝐾
esc
]
when 
​
𝑇
≪
𝔼
​
[
𝐾
esc
]
.
		
(43)
E.3Specialization to the quartic double-well

For the standard quartic double-well

	
𝐿
​
(
𝑥
)
=
𝜆
dw
4
​
(
𝑥
2
−
𝑎
2
)
2
,
		
(44)

the minima are at 
𝑥
±
=
±
𝑎
 and the saddle is at 
𝑧
=
0
. The barrier height is

	
Δ
​
𝐿
=
𝐿
​
(
0
)
−
𝐿
​
(
−
𝑎
)
=
𝜆
dw
4
​
𝑎
4
,
		
(45)

and the curvatures are

	
𝐿
′′
​
(
±
𝑎
)
=
2
​
𝜆
dw
​
𝑎
2
,
𝐿
′′
​
(
0
)
=
−
𝜆
dw
​
𝑎
2
.
		
(46)

Plugging (45)–(46) into (42) yields

	
𝔼
​
[
𝐾
esc
]
≈
2
​
𝜋
𝛼
​
𝜆
dw
​
𝑎
2
​
2
​
exp
⁡
(
Δ
​
𝐿
𝜀
)
=
2
​
𝜋
𝛼
​
𝜆
dw
​
𝑎
2
​
2
​
exp
⁡
(
𝜆
dw
​
𝑎
4
2
⋅
𝑁
𝛼
​
𝜎
2
)
.
		
(47)
E.4A practical hopping criterion and regimes

Escape is controlled by the dimensionless barrier-to-noise ratio

	
Δ
​
𝐿
𝜀
=
𝜆
dw
​
𝑎
4
/
4
𝛼
​
𝜎
2
/
(
2
​
𝑁
)
=
𝜆
dw
​
𝑎
4
2
⋅
𝑁
𝛼
​
𝜎
2
.
		
(48)

Because 
𝔼
​
[
𝐾
esc
]
 depends exponentially on (48), a coarse threshold for observing at least one hop within 
𝑇
 iterations is

	
Δ
​
𝐿
𝜀
≈
log
⁡
𝑇
,
		
(49)

up to logarithmic corrections from the prefactor. This yields three qualitative regimes:

• 

Metastable (no hops): 
Δ
​
𝐿
/
𝜀
≫
log
⁡
𝑇
.

• 

Metastable hopping: 
Δ
​
𝐿
/
𝜀
≈
𝑂
​
(
log
⁡
𝑇
)
.

• 

Delocalized: 
Δ
​
𝐿
/
𝜀
≲
1
.

Figure 9 visualizes these three regimes in simulation. As 
Δ
​
𝐿
/
𝜀
 decreases, trajectories transition from remaining confined to one well (top), to occasional barrier crossings over the horizon (middle), and finally to frequent crossings that wash out basin localization (bottom). This sharp qualitative transition over finite horizons reflects the exponential sensitivity of escape times to the barrier-to-noise ratio in Eq. (42).

Figure 9:Metastability and basin hopping in a double-well under stochastic updates. (a) One-dimensional double-well reward landscape 
𝐽
​
(
𝜃
0
)
 (with other coordinates set to zero), with minima at 
±
𝑎
 and a saddle near 
0
. (b,d,f) Histograms of 
𝜃
0
 at the final iteration across runs. (c,e,g) Example trajectories of 
𝜃
0
​
(
𝑡
)
 across runs. The three rows illustrate the regimes predicted by the barrier-to-noise ratio 
Δ
​
𝐿
/
𝜀
 (Eq. (48)): top: metastable confinement (no hops), middle: rare basin hopping, and bottom: delocalized behavior with frequent transitions. “hop%” denotes the fraction of runs that switch wells at least once under a simple hysteresis-based definition (Appendix E).
E.5Local “water level” within a well

Within one well (e.g., near 
𝑥
=
−
𝑎
), 
𝐿
 is approximately quadratic with curvature 
𝜅
well
=
𝐿
′′
​
(
−
𝑎
)
=
2
​
𝜆
dw
​
𝑎
2
, yielding an OU approximation with stationary variance

	
Var
​
(
𝑥
)
≈
𝜀
𝜅
well
=
𝛼
​
𝜎
2
4
​
𝑁
​
𝜆
dw
​
𝑎
2
,
		
(50)

making explicit how local curvature suppresses fluctuations within a basin.

Relevance to the main analysis (from within-basin OU to basin hopping).

The quadratic OU analysis in Appendix A isolates a within-basin mechanism: near a locally stable region, fixed stochasticity induces a noise-controlled plateau and, with heterogeneous curvature, can produce rise–then–decay without requiring nonconvex pathologies. The double-well analysis in Appendix E complements this picture by characterizing when stochastic learning leaves a basin altogether. Its key message is that basin transitions are controlled by an exponentially sensitive barrier-to-noise ratio (Eq. (48)), yielding three regimes over a finite training horizon 
𝑇
: effectively no hops, rare hops, or delocalization.

In a high-dimensional nonconvex fine-tuning landscape, local neighborhoods can be viewed as collections of basins separated by saddles of varying heights, and different stochastic learning rules induce different effective noise intensities and directions. Consequently, basin-hopping need not be ubiquitous: even modest changes in the effective noise level (through 
𝛼
, 
𝜎
2
/
𝑁
, temperature, batch size, or group size) can move training across the hopping threshold (49) for some barriers but not others. Operationally, this provides a second (non-exclusive) source of late-stage non-monotonicity beyond within-basin variance accumulation: if stochasticity is large enough to cross low barriers along weakly constrained directions, training may drift between nearby basins, leading to additional variability or degradation in reward. Conversely, when 
Δ
​
𝐿
/
𝜀
≫
log
⁡
𝑇
 for the relevant barriers, training remains effectively confined and the within-basin OU mechanism is the appropriate local description. This perspective motivates treating non-monotonic dynamics as potentially arising from both (i) within-basin variance–curvature effects and (ii) finite-horizon basin-hopping, with their relative importance governed by the same effective noise scale.

Appendix FAdditional Rise–Then–Decay Results: Countdown
Figure 10:Rise–then–decay behavior on the Countdown task. Training (left) and held-out test (right) reward trajectories for ES fine-tuning on the Countdown arithmetic task under fixed hyperparameters, shown for population sizes 
𝑁
∈
{
10
,
20
,
30
}
. Across all population sizes, reward improves rapidly from the pretrained checkpoint, reaches a peak, and subsequently declines toward a lower value. Larger populations delay the onset and reduce the magnitude of decay but do not eliminate the non-monotonic behavior. These dynamics closely mirror those observed on GSM8K, ARC-C, and WinoGrande, indicating that rise–then–decay is a robust consequence of stochastic fine-tuning interacting with anisotropic curvature rather than a task-specific artifact.

We provide additional evidence for non-monotonic fine-tuning dynamics on the Countdown task, a structured arithmetic reasoning benchmark distinct from GSM8K, ARC-C, and WinoGrande. Figure 10 shows ES training and test reward trajectories under fixed hyperparameters for population sizes 
𝑁
∈
{
10
,
20
,
30
}
.

Consistent with the main experiments, all runs exhibit a pronounced rise–then–decay pattern. Reward initially increases as stochastic updates exploit high-curvature (stiff) directions associated with rapid improvement, but later decreases as improvement along these directions saturates and stochastic drift accumulates along weakly curved (flat) directions. Increasing the population size delays the onset of decay and raises peak reward, reflecting reduced sampling variance, but does not qualitatively alter the trajectory shape.

These results reinforce two key conclusions. First, non-monotonic reward trajectories are not tied to a specific task, evaluation protocol, or benchmark family, but arise generically under fixed stochasticity. Second, the qualitative dependence on population size matches the predictions of the geometry–variance framework: stochasticity interacts with anisotropic curvature to produce early gains followed by late-time degradation once signal along stiff directions is exhausted. Together with the main results, the Countdown experiments support the view that rise–then–decay is a structural consequence of fine-tuning landscapes rather than an artifact of a particular dataset or algorithmic detail.

Appendix GExperimental Details: ES Reward-Probe

This appendix describes the ES reward-probe protocol used to generate the scaling figures in the main text (e.g., Fig. 6) and the related appendix analyses. The goal of these probes is to characterize reward changes under random weight perturbations at controlled noise scale and to estimate extreme-value quantities (best-of-
𝑁
) in a manner that cleanly separates perturbation randomness (algorithmic) from evaluation-set randomness (measurement).

G.1Models and inference

We evaluate instruction-tuned models from the Qwen2.5-Instruct family:

Model size	HuggingFace model ID
0.5B	Qwen/Qwen2.5-0.5B-Instruct
1.5B	Qwen/Qwen2.5-1.5B-Instruct
3B	Qwen/Qwen2.5-3B-Instruct
7B	Qwen/Qwen2.5-7B-Instruct

Models are loaded in bfloat16. Inference uses HuggingFace Transformers with vLLM as the generation backend; for consistency we use eager attention mode (flash attention disabled). Decoding is greedy (temperature 0.0) with a maximum of 512 new tokens. A fixed generation seed (42) is used across candidates to ensure deterministic prompt-to-output mapping given parameters.

G.2Tasks, datasets, and rewards

We probe three tasks using the training splits of standard datasets and binary per-prompt rewards:

Task	Dataset	Split	Reward
GSM8K	gsm8k/main	train	1 iff final number matches GT
ARC-C	allenai/ai2_arc (Challenge)	train	1 iff correct A/B/C/D
WinoGrande	allenai/winogrande (winogrande_xl)	train	1 iff correct A/B

GSM8K is prompted in chat style with “Let us think step by step” and the final answer requested after the delimiter 
#
​
#
​
#
​
#
; reward extraction parses the final number. ARC-C uses a multiple-choice prompt listing choices A–D; reward checks the chosen letter. WinoGrande uses a sentence-completion prompt with two options A/B; reward checks the chosen option.

G.3Evaluation pool

For each task, we construct a fixed evaluation pool of 
𝑃
=
320
 prompts by a deterministic random shuffle with pool_seed=0. The same pool is used for all model sizes within a task to ensure comparability across scales. All main-text best-of-
𝑁
 and best-of-30 point estimates are computed on this fixed pool (see below), with uncertainty quantified separately.

G.4Perturbations and candidate generation

For parameters 
𝜃
, each ES probe candidate applies a single isotropic Gaussian perturbation

	
𝜃
′
=
𝜃
+
𝜎
​
𝑢
,
𝑢
∼
𝒩
​
(
0
,
𝐼
)
,
		
(51)

where 
𝑢
 matches the shapes of all model parameters. We do not normalize 
𝑢
 (so 
‖
𝑢
‖
≈
𝑑
 in 
𝑑
 parameters). Perturbations are applied in-place and then exactly restored after evaluation; this is critical for numerical fidelity in bfloat16. We monitor restoration fidelity via baseline drift checks (below).

Perturbation scales.

Main figures use 
𝜎
∈
{
3
×
10
−
4
,
10
−
3
,
3
×
10
−
3
}
; extended analyses additionally include 
𝜎
∈
{
10
−
4
,
10
−
2
}
.

Independent perturbation batches (for uncertainty).

To avoid inflating extreme-value estimates via evaluation-set noise, we separate two sources of randomness: (i) perturbation randomness (the ES sampling process) and (ii) evaluation-set randomness (finite prompt pool). For each (task, model size, 
𝜎
) condition, we generate 
𝑆
 independent perturbation batches, each containing 
𝑀
=
240
 candidates. Candidate perturbation seeds are generated from a base seed (1234) using rng.integers(0,2ˆ31,size=240); different batches use independent RNG streams. In the main plots, uncertainty bars for best-of-
𝑁
 are computed across these independent perturbation batches (Section G.6).

G.5Reward computation and stored data

For each candidate and prompt, we compute a binary reward 
𝑟
𝑖
∈
{
0
,
1
}
. The mean reward on the pool is

	
𝑅
=
1
𝑃
​
∑
𝑖
=
1
𝑃
𝑟
𝑖
,
Δ
​
𝑅
=
𝑅
candidate
−
𝑅
baseline
.
		
(52)

We store per-candidate data as (i) the scalar 
Δ
​
𝑅
 on the full pool and (ii) packed bitstrings for the baseline and candidate per-prompt rewards (320 bits) encoded in base64. This enables exact reconstruction and re-aggregation under alternative evaluation procedures (e.g., prompt bootstrap for sensitivity analyses).

G.6Estimating best-of-
𝑁
 without evaluation-noise inflation

Our main paper uses extreme-value statistics because selection is central to ES and because improvement directions are rare. However, best-of-
𝑁
 is sensitive to additional measurement noise: injecting evaluation-set resampling inside the maximization can inflate the maximum (a “winner’s curse”). We therefore compute best-of-
𝑁
 point estimates on the full fixed pool and use independent perturbation batches to quantify uncertainty.

Best-of-
𝑁
 point estimate (full pool).

For each perturbation batch 
𝑠
∈
{
1
,
…
,
𝑆
}
, we compute candidate deltas 
{
Δ
​
𝑅
𝑗
(
𝑠
)
}
𝑗
=
1
𝑀
 on the full pool. For a given population size 
𝑁
, we estimate

	
Δ
𝑁
∗
,
(
𝑠
)
​
(
𝜃
,
𝜎
)
=
𝔼
​
[
max
𝑗
∈
𝒮
⁡
Δ
​
𝑅
𝑗
(
𝑠
)
]
,
		
(53)

where 
𝒮
 is a uniformly random subset of 
{
1
,
…
,
𝑀
}
 of size 
𝑁
 drawn without replacement. We approximate this expectation by Monte Carlo subset sampling (typically 2000 subsets per 
𝑁
), which is computationally cheap since it operates only on stored scalar deltas. We report the mean across batches, 
Δ
𝑁
∗
^
=
1
𝑆
​
∑
𝑠
Δ
𝑁
∗
,
(
𝑠
)
.

Uncertainty (perturbation-batch variability).

Error bars in the main-text best-of-
𝑁
 curves are computed across the 
𝑆
 independent perturbation batches: we report 
±
1.96
×
SE
, where 
SE
 is the standard error of 
{
Δ
𝑁
∗
,
(
𝑠
)
}
𝑠
=
1
𝑆
. These uncertainty estimates reflect algorithmic variability due to perturbation sampling, which is the relevant uncertainty for population-requirement claims.

Population sizes.

For best-of-
𝑁
 curves we evaluate 
𝑁
∈
{
5
,
10
,
20
,
30
,
50
}
, with 
𝑁
=
30
 emphasized in the main text.

Evaluation-set uncertainty and extreme-value effects (appendix only).

We additionally quantify evaluation-set uncertainty via bootstrap over prompts (resampling the 
𝑃
=
320
 prompts with replacement) without changing the candidate set. This provides confidence intervals for statistics on the fixed pool. We do not use prompt bootstrap to define main-text best-of-
𝑁
 point estimates because doing so introduces extra noise inside the maximization and can inflate maxima, consistent with classical extreme-value sensitivity (often modeled by Gumbel-type behavior for maxima). We use prompt bootstrap and subset-size sensitivity studies only as robustness checks in the appendix.

G.7Other summary statistics

We also report (i) 
𝑝
​
(
improve
)
=
Pr
⁡
(
Δ
​
𝑅
>
0
)
 and (ii) 
𝔼
​
[
Δ
​
𝑅
]
 across candidates. Unless stated otherwise, these are computed on the full pool with uncertainty obtained by bootstrap over candidates (1000 replicates, resampling candidates with replacement). Figure 11 summarizes baseline performance and average perturbation statistics (
𝑝
​
(
Δ
​
𝑅
>
0
)
 and 
𝔼
​
[
Δ
​
𝑅
]
), highlighting that most perturbations are non-improving even when best-of-
𝑁
 is positive. Figure 12 shows sensitivity of best-of-30 to perturbation scale 
𝜎
 (raw and headroom-normalized), illustrating the existence of an intermediate 
𝜎
 regime where improvements are accessible.

Figure 11:Baseline performance and average perturbation statistics. Columns correspond to GSM8K, ARC-C, and WinoGrande; rows show complementary summary statistics computed on the fixed evaluation pool (
𝑃
=
320
 prompts). Top row (a–c): baseline accuracies 
𝑅
0
 for each model size. Middle row (d–f): probability of improvement 
𝑝
​
(
Δ
​
𝑅
>
0
)
 across perturbation scales 
𝜎
∈
{
3
×
10
−
4
,
10
−
3
,
3
×
10
−
3
}
. Bottom row (g–i): mean improvement 
𝔼
​
[
Δ
​
𝑅
]
 across perturbations at the same 
𝜎
 values. In many regimes the mean improvement is near zero or negative and 
𝑝
​
(
Δ
​
𝑅
>
0
)
<
1
2
, indicating that most perturbations are non-improving; this is expected in nonconvex, high-dimensional objectives and motivates extreme-value statistics (best-of-
𝑁
) in the main text, which capture rare-but-meaningful improvements.
Figure 12:Perturbation-scale sensitivity of best-of-30 improvements. Best-of-30 expected improvement 
Δ
^
30
∗
 (top row, a–c) and headroom-normalized best-of-30 
Δ
^
30
∗
/
(
1
−
𝑅
0
)
 (bottom row, d–f) as a function of perturbation scale 
𝜎
 for GSM8K, ARC-C, and WinoGrande. Moderate 
𝜎
 values yield positive best-of-30 improvements across model sizes, while very large 
𝜎
 can produce negative improvements, consistent with a scale-mismatch/variance-dominated regime. Error bars reflect uncertainty from independent perturbation batches (main text protocol), and headroom normalization controls for differing baseline saturation across model sizes.
G.8Headroom normalization

To control for differences in baseline accuracy across model sizes, we report headroom-normalized improvements

	
Δ
∗
,
rel
=
Δ
∗
1
−
𝑅
0
,
		
(54)

where 
𝑅
0
 is the baseline accuracy on the fixed pool. This measures improvement as a fraction of remaining achievable reward (given the binary accuracy ceiling at 1). The baseline accuracies for different model sizes and different tasks are shown in Figure 11.

Headroom-normalized improvement distributions.

To complement the best-of-
𝑁
 summaries in the main text, we visualize the full distribution of perturbation outcomes. For each task, model size, and perturbation scale 
𝜎
, we sample 
𝑀
=
240
 random weight perturbations and compute the headroom-normalized change 
Δ
​
𝑅
/
(
1
−
𝑅
0
)
, where 
Δ
​
𝑅
=
𝑅
​
(
𝜃
+
𝜎
​
𝜀
)
−
𝑅
0
 and 
𝑅
0
 is the baseline accuracy on the fixed prompt pool. Headroom normalization enables fair comparison across models with different baseline accuracies: a value of 
0.1
 corresponds to capturing 
10
%
 of the remaining possible improvement under a binary reward ceiling. Across tasks, we observe a “locality window” in 
𝜎
: sufficiently small perturbations can retain a nontrivial improving tail, while larger 
𝜎
 rapidly shifts the distribution negative and drives 
𝑝
​
(
Δ
​
𝑅
>
0
)
 toward zero.

Figure 13:Headroom-normalized perturbation outcome distributions on GSM8K. Each panel shows 
Δ
​
𝑅
/
(
1
−
𝑅
0
)
 for 
𝑀
=
240
 random perturbations, with the same conventions as Fig. 14. Across model sizes, sufficiently small 
𝜎
 can preserve a nontrivial improving tail, while increasing 
𝜎
 shifts the distribution left and drives 
𝑝
​
(
Δ
​
𝑅
>
0
)
 toward zero, consistent with leaving the local regime where improvements are accessible.
Figure 14:Headroom-normalized perturbation outcome distributions on ARC-C. Each panel shows the distribution of 
Δ
​
𝑅
/
(
1
−
𝑅
0
)
 over 
𝑀
=
240
 random weight perturbations, where 
Δ
​
𝑅
=
𝑅
𝑖
−
𝑅
0
 is the change in accuracy from baseline 
𝑅
0
 and 
(
1
−
𝑅
0
)
 is the remaining headroom. Rows correspond to model sizes (0.5B–7B) and columns to perturbation scale 
𝜎
. The gray dashed vertical line marks 
Δ
​
𝑅
=
0
 (no change); the solid vertical line marks the mean (green if positive, red if negative). Panel titles report 
𝑝
​
(
Δ
​
𝑅
>
0
)
, the fraction of perturbations that improve over baseline. Headroom normalization makes the distributions comparable across model sizes and highlights a viable small-
𝜎
 regime where an improving tail exists versus larger-
𝜎
 regimes where perturbations predominantly degrade performance.
Figure 15:Headroom-normalized perturbation outcome distributions on WinoGrande. Each panel shows 
Δ
​
𝑅
/
(
1
−
𝑅
0
)
 for 
𝑀
=
240
 random perturbations, with the same conventions as Fig. 14. The distributions emphasize that best-of-
𝑁
 improvements arise from the right tail even when the mean is near zero or negative, and that this tail is strongly 
𝜎
-dependent.
G.9Auxiliary diagnostics: saturation population and tail quantiles

To complement the best-of-
𝑁
 curves in the main text, we summarize improvement accessibility with two scalar diagnostics. First, we define the saturation population 
𝑁
90
 as the smallest population size needed to obtain 
90
%
 of the expected best-of-
𝑁
 improvement at the largest evaluated population, 
𝑁
max
. Second, we report an upper-tail statistic of the perturbation-induced improvement distribution, the 
95
th percentile 
𝑞
0.95
 of 
Δ
​
𝑅
/
(
1
−
𝑅
0
)
 across the candidate pool. Figure 16 shows that both 
𝑁
90
 and 
𝑞
0.95
 vary across tasks and perturbation scales, but neither exhibits a systematic increase with model size. These diagnostics provide an additional, compact check that the population required to access improving perturbations does not grow proportionally with the number of model parameters.

Figure 16:Auxiliary accessibility diagnostics across model scale. Top row (a–c): Saturation population 
𝑁
90
 as a function of model size for GSM8K, ARC-C, and WinoGrande at three perturbation scales 
𝜎
. Here 
𝑁
90
 is the smallest population size 
𝑁
 such that the expected best-of-
𝑁
 improvement reaches 
90
%
 of the value at the largest evaluated population, i.e., 
𝑁
90
≜
min
⁡
{
𝑁
:
Δ
𝑁
∗
​
(
𝜎
)
≥
0.9
​
Δ
𝑁
max
∗
​
(
𝜎
)
}
. Bottom row (d–f): Upper-tail quantile 
𝑞
0.95
 of the headroom-normalized improvement distribution 
Δ
​
𝑅
/
(
1
−
𝑅
0
)
 for the same conditions, estimated from the candidate pool of 
𝑀
=
240
 perturbations. (The dashed gray line marks 
0
.) While these summaries are task- and 
𝜎
-dependent and not monotonic in model size, they show no systematic increase in the population required to access the improving tail as model size grows from 0.5B to 7B. This provides an additional check, complementary to Fig. 6, that improvement accessibility does not deteriorate in proportion to ambient parameter count.
G.10Reward-based curvature evidence via perturbation SLQ

Our main-text blessing-of-dimensionality hypothesis is formulated in reward geometry terms: the number of curvature-active directions governing local improvement need not grow proportionally with parameter count. To complement the operational best-of-
𝑁
 accessibility probes, we directly estimate curvature structure of the reward-defined objective at a fixed smoothing scale by analyzing the Hessian of 
𝐽
𝜎
.

Figure 17 reports Hessian-spectrum-derived summaries of 
∇
2
𝐽
𝜎
​
(
𝜃
)
 for GSM8K across model sizes. The key observation is that concentration measures (participation ratio and effective rank) decrease with model size, indicating that a smaller set of directions carries most curvature-relevant structure in the ES-smoothed reward landscape as models scale. At the same time, the magnitude of extreme negative curvature can increase. Together, these findings support a curvature-based blessing-of-dimensionality interpretation: scaling primarily adds weakly curved directions while the curvature-active structure remains relatively concentrated, consistent with the “bulk + tail” picture used in the main text.

Figure 17:Reward-based curvature proxy across model scale (GSM8K): concentration of curvature-active structure. We estimate Hessian-spectrum-derived metrics of the ES-smoothed reward objective 
𝐽
𝜎
​
(
𝜃
)
=
𝔼
𝜀
∼
𝒩
​
(
0
,
𝐼
)
​
[
𝐽
​
(
𝜃
+
𝜎
​
𝜀
)
]
 using a perturbation-based Hessian–vector product estimator (no autograd) and stochastic Lanczos quadrature (SLQ). Rewards are binary GSM8K training accuracy on a fixed subset of 100 prompts with a fixed prompting and answer-extraction rule. (a) magnitude of the most negative eigenvalue 
|
𝜆
min
|
, (b) negative spectral mass (total weight on 
𝜆
<
0
), (c) participation ratio, and (d) effective rank (both concentration measures; smaller values indicate fewer directions carry most spectral weight). Across Qwen2.5-Instruct model sizes (0.5B–7B), participation ratio and effective rank decrease, indicating that curvature-relevant structure in the reward-defined landscape becomes increasingly concentrated rather than expanding in proportion to parameter count. Error bars show mean 
±
 seed variability over 5 random seeds.
Generated on Fri Jan 30 00:21:54 2026 by LaTeXML
