# Adapters as Representational Hypotheses: What 30 PEFT Methods Tell Us About Transformer Geometry

*Crossposted from [github.com/wassname/adapters_as_hypotheses](https://github.com/wassname/adapters_as_hypotheses)*

## The core claim

Every parameter-efficient fine-tuning (PEFT) adapter encodes a structural hypothesis about how to intervene in transformer internals. LoRA says weight changes are low-rank. OFT says orthogonal rotations preserve semantic structure. PiSSA says the principal SVD components carry the signal. When one adapter outperforms another under controlled conditions -- same model, same data, same parameter budget -- **the winning method's structural assumptions are empirically supported as a better description of the weight manifold.**

This is hiding in plain sight. Hundreds of PEFT papers run controlled comparisons. Almost nobody reads them as science about representations.

## Why this matters for interpretability

We want to understand how transformers work. There are many approaches -- probing, ablation, SAEs -- but most of them *observe* rather than *intervene*.

- Probing finds representations that predict behavior, but high probe accuracy does not mean the model uses that representation (Belinkov, 2022).
- CCS discovers latent knowledge but cannot intervene on it (Burns et al., 2022).
- Intervention shortcuts both problems: if modifying a representation reliably changes behavior, we have causal evidence of what we control.

The GDM interpretability team recently pivoted toward ["pragmatic interpretability"](https://www.lesswrong.com/posts/StENzDcD3kpfGJssR/a-pragmatic-vision-for-interpretability) -- empirical feedback on the critical path to AGI going well. Adapter benchmarks are precisely this kind of empirical feedback: which structural assumptions about transformer internals hold up under intervention?

The adapter literature is a natural experiment. Each method constrains the *form* of the weight update. When a constrained method matches or beats an unconstrained one, that constraint reflects real structure in the weight manifold. When it generalizes OOD, the structure is *causally relevant*, not merely correlated.

## The catalog

I went through ~30 PEFT methods in HuggingFace PEFT and the broader literature. For each one I:

1. Extracted pseudocode for the forward pass (what the intervention actually does)
2. Stated the hypothesis it encodes about transformer internals
3. Graded the evidence on a rough hierarchy:

| Grade | Meaning |
|-------|---------|
| \* | Parameter-efficient (matches LoRA with fewer params) |
| \*\* | Beats LoRA on raw performance |
| \*\*!\*\* | Beats full fine-tuning |
| \*\*!!\*\* | Data-efficient (few-shot, fast convergence) |
| \*\*!!!\*\* | Generalizes out-of-distribution |

The full catalog with pseudocode is at [github.com/wassname/adapters_as_hypotheses](https://github.com/wassname/adapters_as_hypotheses). Here I'll summarize the main findings.

## What the evidence says

### 1. The SVD basis is the natural coordinate system

Methods that use the model's own SVD decomposition consistently outperform random-basis methods at the same parameter count:

- **PiSSA** (NeurIPS 2024): Initialize LoRA from top-$r$ SVD of $W$, freeze the residual. Gemma-7B on GSM8K: PiSSA 77.7% vs LoRA 74.5%. Same architecture, same params -- the only difference is *which subspace you start in*.
- **SVFT**: Fix both singular vector sets from $W$'s SVD, learn only sparse coefficients. Recovers 96% of full FT performance with 0.006% of parameters. LoRA/DoRA recover only 85% with 0.03-0.8%.
- **SSVD**: Rotate right singular vectors (Cayley transform), shift singular values, keep left singular vectors fixed. Matches LoRA with 10M fewer params on domain-shifted ASR.

The message: the SVD basis isn't an arbitrary mathematical convenience. It captures *meaningful computational directions* that the model actually uses.

### 2. Orthogonal adapters preserve something real

The OFT family (OFT, BOFT, GOFT, HRA) constrains adaptation to orthogonal transformations -- rotations without scaling. They work well on tasks where you want to *repurpose* existing representations without *destroying* them (DreamBooth, ControlNet, domain adaptation).

HRA makes a surprising bridge: a chain of $r$ Householder reflections is *both* orthogonal *and* equivalent to a rank-$r$ perturbation. The "low-rank vs orthogonal" dichotomy is a false one. The effective adaptation might be low-rank *and* approximately orthogonal simultaneously.

### 3. Direction and strength decouple

Three independent teams converged on the same design: separate *what to change* (direction in weight space) from *how much to change it* (magnitude):

- **DoRA** (ICML 2024): Magnitude/direction decomposition of $W$. Consistently beats LoRA.
- **DeLoRA** (ICLR 2025): Normalize each rank-1 component, introduce learnable scalar $\lambda$. Better robustness to learning rate.
- **ROAD**: 2D rotary adaptation with explicit angle $\theta$ and magnitude $\alpha$.

When you don't decouple them (standard LoRA), the optimizer wastes capacity fighting magnitude dynamics when it should be learning directions. Prediction: methods that decouple direction from strength will systematically show better OOD transfer, because the direction captures *what* to change (task-invariant) while the strength captures *how much* (task-specific).

### 4. Scaling alone goes surprisingly far

**IA3** learns nothing but a per-channel scaling vector ($\lambda \in \mathbb{R}^d$, initialized to 1). With T0-3B, it outperforms ICL with GPT-3 175B on Super-NaturalInstructions. **LN Tuning** learns only LayerNorm affine parameters (~0.5% of model).

A large fraction of "task adaptation" is just reweighting existing features -- gain control over channels. The model already computes the right features; the bottleneck is which ones to attend to. When scaling fails, that's when genuine new feature combinations are needed, and only then do you need weight-space interventions.

### 5. The strongest evidence: OOD generalization

Most adapter comparisons are parameter-efficiency contests on the same benchmarks. The really informative test is out-of-distribution transfer: does the adapter capture causal structure or just surface correlation?

**AntiPaSTO** ([Clark, 2025](https://arxiv.org/abs/2601.07473)) synthesizes several of the above insights -- SVD basis (PiSSA), Cayley rotation of right singular vectors (SSVD), direction/strength decoupling (DeLoRA) -- into a single adapter that steers model behavior bidirectionally via a coefficient $\alpha \in [-1, +1]$. Trained on 800 contrastive word pairs (no preference labels), it transfers from template sentences to real ethical dilemmas with 6.9x the steering performance of prompting. The same adapter at $\alpha = +1$ makes the model more honest; at $\alpha = -1$, less honest.

The OOD transfer is the strong claim. The SVD rotation basis learned on trivial templates captures something causally relevant about how the model structures its honesty computations. (Caveat: primary evidence is on models up to 4B; larger models need further exploration.)

## Design lineages

One interesting pattern: you can trace design lineages that progressively refine the same hypothesis.

**Orthogonal family:** OFT (block-diagonal rotation) -> BOFT (butterfly factorization, $O(d \log d)$ params) -> GOFT (Givens rotations, $O(d)$ params) -> HRA (Householder reflections, bridges to low-rank)

**SVD-aware family:** PiSSA (SVD initialization) -> SVFT (sparse SVD coefficients) -> SSVD (asymmetric U/V treatment + Cayley rotation) -> AntiPaSTO (Cayley + steering coefficient)

**Decoupling family:** DoRA (magnitude/direction) -> ETHER (fixed-strength orthogonal) -> DeLoRA (normalized rank-1 + $\lambda$) -> AntiPaSTO ($\alpha$-controlled rotation)

Each refinement tests a more specific version of the parent hypothesis. When the refinement works better, we learn something more specific about the geometry.

## What I'm most uncertain about

- **Scale dependence.** Most of these results are on 1B-7B models. The geometry might change at 70B+. Some evidence (SSVD) suggests the SVD hypothesis gets *stronger* with scale, but this isn't settled.
- **Task dependence.** Orthogonal methods shine on vision/generation (semantic preservation) but may not apply where magnitude changes matter (NLU, reasoning). The "right" geometry may be task-specific.
- **Controlled comparisons are rare.** Many papers compare against LoRA with different hyperparameters, different scales, different tasks. The cleanest evidence comes from papers that do careful all-else-equal ablations (DoRA, PiSSA, SSVD).
- **Publication bias.** Methods that don't work don't get published. The catalog over-represents "successful" hypotheses.

## The repo

The full catalog with pseudocode, evidence, and grades for 30 methods is at:

**[github.com/wassname/adapters_as_hypotheses](https://github.com/wassname/adapters_as_hypotheses)**

Each entry has the paper saved to `docs/` for reference. Contributions welcome -- if I've mischaracterized a method or missed one, open an issue.

## Acknowledgments

The framing of "adapters as representational hypotheses" originates from Appendix A.3 of [AntiPaSTO](https://arxiv.org/abs/2601.07473) (Clark, 2025). The "pragmatic interpretability" direction that motivates this is from [Nanda et al. (2025)](https://www.lesswrong.com/posts/StENzDcD3kpfGJssR/a-pragmatic-vision-for-interpretability).
