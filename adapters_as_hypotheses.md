
tl;dr: some transformer interventions work very well. Hypersphere rotation, SVD, & separation direction and magnitude are some that 1) generalise well 2) are data efficient. IMO these are empirical clues about transformer internals

## The core claim

Adapter fine-tuning papers (like [LoRA](https://en.wikipedia.org/wiki/LoRA_(machine_learning))) are usually read as engineering races, but they are also experiments about model geometry.

We fine-tune transformers efficiently with low-rank adapters -- adding a constrained update to each weight matrix. Each adapter's constraint is also a hypothesis about model geometry -- about which transformations preserve useful computation and which directions in weight space matter. When one constrained adapter reliably beats another under similar budget, that is suggestive evidence about representation, not just optimization.

So the claim: adapter papers are an underused source of *intervention* evidence for interpretability, if we read them as hypothesis tests rather than benchmark performance.

## Why this matters for interpretability

We want to understand how transformers work. There are many approaches -- probing, ablation, SAEs -- but most of them *observe* rather than *intervene*.

GDM's interpretability team put this well in their ["pragmatic interpretability"](https://www.lesswrong.com/posts/StENzDcD3kpfGJssR/a-pragmatic-vision-for-interpretability) post: empirical feedback on which structural assumptions hold up. Adapter benchmarks are exactly this kind of feedback -- I made a similar argument in my [AntiPaSTO paper](https://arxiv.org/abs/2601.07473), arriving there from the adapter side.

The adapter literature is a natural experiment. Each method constrains the *form* of the weight update. When a constrained method matches or beats an unconstrained one, that supports the possibility that the constraint aligns with real structure in the weight manifold. When it generalizes OOD, the case for causal relevance is stronger.


## What the evidence says

### 1. The SVD basis is the natural coordinate system

Methods that use the model's own SVD decomposition often outperform random-basis methods in reported setups at similar parameter count:

- **PiSSA** (NeurIPS 2024): Initialize LoRA from top-$r$ SVD of $W$, freeze the residual. Gemma-7B on GSM8K: PiSSA 77.7% vs LoRA 74.5%. Same architecture, same params -- the only difference is *which subspace you start in*.
- **SVFT**: Fix both singular vector sets from $W$'s SVD, learn only sparse coefficients. Recovers 96% of full FT performance with 0.006% of parameters. LoRA/DoRA recover only 85% with 0.03-0.8%.
- **SSVD**: Rotate right singular vectors (Cayley transform), shift singular values, keep left singular vectors fixed. Matches LoRA with 10M fewer params on domain-shifted ASR.

The message: the SVD basis is not just a mathematical convenience. In current benchmarks, it appears to provide useful adaptation coordinates.

### 2. Orthogonal adapters preserve something real

The OFT family (OFT, BOFT, GOFT, HRA) constrains adaptation to orthogonal transformations -- rotations without scaling. They work well on tasks where you want to *repurpose* existing representations without *destroying* them (DreamBooth, ControlNet, domain adaptation).

HRA makes a surprising bridge: a chain of $r$ Householder reflections is *both* orthogonal *and* equivalent to a rank-$r$ perturbation. The "low-rank vs orthogonal" dichotomy is less sharp than often presented. The effective adaptation may be low-rank *and* approximately orthogonal simultaneously.

### 3. Direction and strength decouple

Three independent teams converged on the same design: separate *what to change* (direction in weight space) from *how much to change it* (magnitude):

- **DoRA** (ICML 2024): Magnitude/direction decomposition of $W$. Consistently beats LoRA.
- **DeLoRA** (ICLR 2025): Normalize each rank-1 component, introduce learnable scalar $\lambda$. Better robustness to learning rate.
- **ROAD**: 2D rotary adaptation with explicit angle $\theta$ and magnitude $\alpha$.

When you don't decouple them (standard LoRA), optimization can entangle direction and magnitude updates. Testable prediction: methods that decouple direction from strength may show better OOD transfer, because direction can encode *what* to change while strength can encode *how much*.

### 4. Scaling alone goes surprisingly far

**IA3** learns nothing but a per-channel scaling vector ($\lambda \in \mathbb{R}^d$, initialized to 1). With T0-3B, it outperforms ICL with GPT-3 175B on Super-NaturalInstructions. **LN Tuning** learns only LayerNorm affine parameters (~0.5% of model).

A substantial fraction of "task adaptation" can be reweighting existing features -- gain control over channels. In those settings, the bottleneck is often selection rather than new feature creation. When scaling fails, new feature combinations are likely needed.

## Design lineages

One interesting pattern: you can trace design lineages that progressively refine the same hypothesis.

**Orthogonal family:** OFT (block-diagonal rotation) -> BOFT (butterfly factorization, $O(d \log d)$ params) -> GOFT (Givens rotations, $O(d)$ params) -> HRA (Householder reflections, bridges to low-rank)

**SVD-aware family:** PiSSA (SVD initialization) -> SVFT (sparse SVD coefficients) -> SSVD (asymmetric U/V treatment + Cayley rotation) -> AntiPaSTO (Cayley + steering coefficient)

**Decoupling family:** DoRA (magnitude/direction) -> ETHER (fixed-strength orthogonal) -> DeLoRA (normalized rank-1 + $\lambda$) -> AntiPaSTO ($\alpha$-controlled rotation)

Each refinement tests a more specific version of the parent hypothesis. When the refinement works better, we learn something more specific about the geometry.

## The catalog

I went through ~30 adapter methods in [HuggingFace PEFT](https://github.com/huggingface/peft) and the broader literature. For each one I:

1. Extracted pseudocode for the forward pass (what the intervention actually does)
2. Stated the hypothesis it encodes about transformer internals
3. Graded the evidence on independent dimensions:

| Dim | Pts | Meaning |
|-----|-----|----------------------------------------------|
| PE  | 1   | Parameter-efficient: competitive at <1% params |
| BL  | 1   | Beats LoRA at comparable budget |
| BF  | 1.5 | Matches or beats full fine-tuning |
| DE  | 1.5 | Data-efficient: faster convergence or fewer examples |
| OOD | 2   | Generalizes out-of-distribution |
| WA  | 1   | Widely adopted as baseline |
| **Total** | **8** | 1+1+1.5+1.5+2+1 |

Score = sum of earned dimensions. Higher = stronger evidence that the method's structural hypothesis is correct.

### Scorecard (top methods)

| # | Method | Score | Breakdown | Theme |
|--:|--------|------:|-----------|-------|
| 6 | PiSSA | 5.0 | PE+BL+BF+DE | SVD basis |
| 4 | DoRA | 4.5 | PE+BL+BF+WA | dir/strength |
| 11 | AntiPaSTO* | 4.5 | PE+DE+OOD | SVD+rotation |
| 13 | BOFT | 4.0 | PE+BF+DE | orthogonal |
| 5 | DeLoRA | 3.5 | PE+BL+DE | dir/strength |
| 8 | SSVD | 3.5 | PE+BL+DE | SVD basis |
| 31 | CLOVER | 3.5 | PE+BL+BF | SVD+architecture |
| 32 | PSOFT | 3.5 | PE+BL+DE | SVD+orthogonal |
| 1 | LoRA | 2.0 | PE+WA | low-rank |

\* own work -- it was developed with this PoV in mind. 30 methods total; see full catalog for the rest.

The full catalog with pseudocode is at [github.com/wassname/adapters_as_hypotheses](https://github.com/wassname/adapters_as_hypotheses). Here I'll summarize the main findings.

## What I'm most uncertain about

- **Scale dependence.** Most of these results are on 1B-7B models. The geometry might change at 70B+. Some evidence (SSVD) suggests the SVD hypothesis gets *stronger* with scale, but this isn't settled.
- **Task dependence.** Orthogonal methods shine on vision/generation (semantic preservation) but may not apply where magnitude changes matter (NLU, reasoning). The "right" geometry may be task-specific.
- **Controlled comparisons are rare.** Many papers compare against LoRA with different hyperparameters, different scales, different tasks. The cleanest evidence comes from papers that do careful all-else-equal ablations (DoRA, PiSSA, SSVD).
- **Publication bias.** Methods that don't work don't get published. The catalog over-represents "successful" hypotheses.
- **My own bias.** I developed AntiPaSTO with these insights in mind, so it's unsurprising it scores well under criteria that match its design goals. I can't be fully objective here.

*Disclaimer: This is an AI-guided iterative survey. It does not speak for me, but I share it in the hope that it is useful. I do think this is strong evidence of how to intervene in transformers.*

## The repo

The full catalog with pseudocode, evidence, and grades for 30 methods is at:

**[github.com/wassname/adapters_as_hypotheses](https://github.com/wassname/adapters_as_hypotheses)**

Each entry has the paper saved to `docs/` for reference. Contributions welcome -- if I've mischaracterized a method or missed one, open an issue.
