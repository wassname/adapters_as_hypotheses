# Adapters as Representational Hypotheses

*What does each PEFT method believe about transformer internals?*

## Why care?

We want to understand how transformers work. There are many approaches -- probing, ablation, SAEs -- but most of them *observe* rather than *intervene*. Probing finds representations that predict behavior, but high probe accuracy does not mean the model uses that representation ([Belinkov, 2022](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00254/43503)). CCS discovers latent knowledge but cannot intervene on it ([Burns et al., 2022](https://arxiv.org/abs/2212.03827)). Intervention shortcuts both problems: if modifying a representation reliably changes behavior, we have causal evidence of what we control (I argued this in [AntiPaSTO](https://arxiv.org/abs/2601.07473)).

<!-- TODO is ths all really relevent for the intro and audience, seems long and not to the point, is the lesswrong one better? -->

There is an underappreciated source of exactly this kind of causal evidence: the PEFT adapter literature.

Each adapter constrains *how* you can update pretrained weights. When one adapter architecture outperforms another under controlled conditions -- same model, same data, same parameter budget -- the winner's structural assumptions get stronger support as a description of the weight manifold. This is a natural experiment running across many papers, and it is still underused as evidence about representations.

GDM's interpretability team recently pivoted toward "pragmatic interpretability" -- directly solving problems on the critical path to AGI going well, grounded in proxy tasks with empirical feedback ([Nanda et al., 2025](https://www.lesswrong.com/posts/StENzDcD3kpfGJssR/a-pragmatic-vision-for-interpretability)). Adapter benchmarks are precisely this: empirical feedback on which structural assumptions about transformer internals hold up under intervention.

If an adapter generalizes out-of-distribution, that is stronger evidence that the geometric structure it exploits is causally relevant to behavior, not merely correlated. From my [AntiPaSTO paper](https://arxiv.org/abs/2601.07473):

> Each adapter architecture encodes a claim about how to intervene in transformer internals. LoRA hypothesizes weight changes are low-rank. OFT hypothesizes orthogonal transformations preserve semantic structure. VeRA hypothesizes shared random projections plus learned scaling suffice. DeLoRA hypothesizes direction and magnitude should decouple. PiSSA hypothesizes principal components matter most. Our choice -- Cayley rotations of SVD singular vectors -- hypothesizes that the model's own learned basis defines the natural intervention manifold. Adapters that generalize out-of-distribution tell us which geometric structures are causally relevant to behavior, not merely correlated with it.

This is a pragmatic, interventionist program: we learn about internals by seeing which interventions *work*. An adapter that transfers where others fail reveals something real about the geometry of the representation. Below, we catalog each major PEFT method as a hypothesis, extract pseudocode for the intervention, and weigh the evidence.

### Evidence scoring

We grade evidence on independent dimensions. Each method gets points for the dimensions it satisfies:

| Dim | Pts | Meaning |
|-----|-----|------------------------------------------|
| PE  | 1   | Parameter-efficient: competitive with full FT at <1% params |
| BL  | 1   | Beats LoRA on raw performance at comparable budget |
| BF  | 1.5 | Matches or beats full fine-tuning |
| DE  | 1.5 | Data-efficient: faster convergence or works with less data |
| OOD | 2   | Generalizes out-of-distribution |
| WA  | 1   | Widely adopted: used as baseline by many other papers |

Total = sum of applicable dimensions (max 8). Higher = stronger evidence that the method's structural hypothesis is correct.

---

## 1. LoRA -- Low-Rank Adaptation

**Paper:** [Hu et al. 2021](https://arxiv.org/abs/2106.09685) (ICLR 2022)
**Code:** [peft/tuners/lora/layer.py](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py)
**Saved:** [docs/lora_low_rank_adaptation.md](docs/lora_low_rank_adaptation.md)

**Hypothesis:** Weight changes needed for task adaptation are *low-rank*. The residual between pretrained and fine-tuned weights lives in a small subspace, so we can parameterize $\Delta W = BA$ with $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$, $r \ll d$.

```py
# ── LoRA intervention ──────────────
def lora_forward(x, W, A, B, α, r):
    # W frozen, A and B learned
    scaling = α / r
    ΔW = B @ A                        # ΔW ∈ ℝ^{d_out × d_in}, rank r
    return (W + scaling * ΔW) @ x     # equivalently: W(x) + scaling * B(A(x))
```

**Evidence:** Parameter-efficient (matches full FT with 0.01% params on GPT-3). One of the most common baselines in PEFT. Authors demonstrate comparable performance to full fine-tuning on GPT-3 175B across multiple NLU benchmarks. Subsequent work ([Biderman et al. 2024](https://arxiv.org/abs/2405.09673), [saved](docs/biderman_lora_limitations.md)) finds LoRA underperforms full FT on harder tasks and larger scale -- the low-rank assumption holds for surface-level adaptation but weakens when deeper restructuring is needed.

**Grade:** PE+WA=2 (parameter-efficient, universal baseline, but ceiling on hard tasks)

---

## 2. OFT -- Orthogonal Fine-Tuning

**Paper:** [Qiu et al. 2023](https://arxiv.org/abs/2306.07280)
**Code:** [peft/tuners/oft/layer.py](https://github.com/huggingface/peft/blob/main/src/peft/tuners/oft/layer.py)
**Saved:** [docs/oft_orthogonal_finetuning.md](docs/oft_orthogonal_finetuning.md)
**See also:** BOFT ([Liu et al. 2023](https://arxiv.org/abs/2311.06243)), OFTv2 ([2025](https://arxiv.org/abs/2506.19847))

**Hypothesis:** Orthogonal transformations preserve the semantic structure of pretrained weights. The pairwise angles between neuron weight vectors (the "hyperspherical energy") encode learned knowledge; any useful adaptation should preserve these angles. $W_{\text{new}} = R \cdot W$ where $R \in O(d)$.

```py
# ── OFT intervention ──────────────
def oft_forward(x, W, Q):
    # Q: learned skew-symmetric params (upper triangle of block matrices)
    Q_skew = skew_symmetric(Q, block_size)     # Q_skew ∈ ℝ^{b×k×k}, antisymmetric
    R = cayley(Q_skew)                         # R = (I + Q_skew)(I - Q_skew)^{-1} ∈ O(k)
    R_full = block_diag(R)                     # R_full ∈ O(d), block-diagonal
    # Paper: w̃ᵢ = R · wᵢ for each row, so W' = W @ R^T
    return (W @ R_full.T) @ x                  # rotate weight rows orthogonally
```

**Evidence:** Authors demonstrate OFT preserves "hyperspherical energy" (pairwise neuron angles) during adaptation, which LoRA does not. Strong results on controllable image generation (ControlNet) and subject-driven generation (DreamBooth), where semantic preservation matters. BOFT extends this with butterfly-factorized orthogonal matrices for better parameter efficiency. OFTv2 reduces computational cost from $O(d^3)$ to $O(d^2)$ via input-centric reformulation and outperforms QLoRA.

However: the orthogonality constraint is rigid. It prevents magnitude changes entirely, limiting adaptation on tasks that require rescaling neuron importance. The hypothesis is strongest where you want to *rotate* representations without *distorting* them.

**Grade:** PE+DE=2.5 (parameter-efficient, data-efficient: converges well with only 5% of training data on controllable generation)

---

## 3. VeRA -- Vector-based Random Matrix Adaptation

**Paper:** [Kopiczko et al. 2023](https://arxiv.org/abs/2310.11454) (ICLR 2024)
**Code:** [peft/tuners/vera/layer.py](https://github.com/huggingface/peft/blob/main/src/peft/tuners/vera/layer.py)
**Saved:** [docs/vera_vector_random_matrix_adaptation.md](docs/vera_vector_random_matrix_adaptation.md)

**Hypothesis:** Random projections are sufficient structure; all a layer needs to learn is *how much* of each projected direction to use. A single pair of frozen random matrices $(A, B)$ shared across all layers, combined with per-layer learned scaling vectors $(\lambda_d, \lambda_b)$, can match LoRA. The implication: the specific learned subspace matters far less than you'd think -- only the per-layer scaling matters.

```py
# ── VeRA intervention ─────────────
def vera_forward(x, W, A, B, λ_d, λ_b):
    # A ∈ ℝ^{r×d_in}, B ∈ ℝ^{d_out×r}: frozen random, shared across ALL layers
    # λ_d ∈ ℝ^r, λ_b ∈ ℝ^r: learned per-layer scaling vectors
    ΔW = (λ_b[:, None] * B) @ (λ_d[:, None] * A)   # ΔW ∈ ℝ^{d_out × d_in}
    return (W + ΔW) @ x
    # forward: result + λ_b * linear(λ_d * linear(dropout(x), A), B)
```

**Evidence:** 10x fewer trainable parameters than LoRA while maintaining competitive performance across diverse NLU benchmarks. The fact that *random* projections work at all is surprising and informative: it suggests that the JL lemma-style argument applies -- random subspaces approximately preserve the structure needed for adaptation, and per-layer gating is the real bottleneck.

**Grade:** PE=1 (extreme parameter efficiency, competitive with LoRA, random-projection ceiling on complex tasks)

---

## 4. DoRA -- Weight-Decomposed Low-Rank Adaptation

**Paper:** [Liu et al. 2024](https://arxiv.org/abs/2402.09353) (ICML 2024)
**Code:** [peft/tuners/lora/dora.py](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/dora.py)
**Saved:** [docs/dora_weight_decomposed_lora.md](docs/dora_weight_decomposed_lora.md)

**Hypothesis:** Full fine-tuning decomposes weight updates into *magnitude* and *direction* components that evolve differently. LoRA conflates these two. Decomposing $W = m \cdot \frac{V}{\|V\|_c}$ and updating them separately (magnitude as a learned scalar, direction via LoRA) better approximates full FT dynamics.

```py
# ── DoRA intervention ─────────────
def dora_forward(x, W, A, B, m, α):
    # m ∈ ℝ^{d_out}: learned magnitude per output neuron
    # A, B: LoRA matrices for directional update
    ΔW = B @ A                                    # directional update, rank r
    V = W + α * ΔW                                # updated weight (direction)
    V̂_norm = norm(V, dim=1).detach()              # column norms, detached
    scale = m / V̂_norm                            # magnitude / direction_norm
    return scale * (W @ x) + scale * α * (B @ A @ x)
```

**Evidence:** Authors analyze full FT weight updates and find distinct magnitude vs. direction patterns that LoRA misses. DoRA outperforms LoRA on LLaMA (commonsense reasoning), LLaVA (visual instruction tuning), and VL-BART (image/video-text) in their reported setups. No additional inference overhead (magnitudes merge). It is now a common LoRA-family baseline in many recent papers.

**Grade:** PE+BL+BF+WA=4.5 (beats LoRA across multiple domains, QDoRA slightly outperforms full FT on LLaMA2-7B/LLaMA3-8B, standard strong baseline)

*Implications:* The magnitude/direction decomposition reveals something about how full FT works internally. Weight updates are not just "adding stuff" -- they redistribute energy across neurons (magnitude) independently of rotating their selectivity (direction). This connects to the neuroscience intuition that gain modulation and selectivity tuning are separate mechanisms.

---

## 5. DeLoRA -- Decoupled Low-Rank Adaptation

**Paper:** [Bini, Girrbach, Akata 2025](https://arxiv.org/abs/2503.18225) (ICLR 2025)
**Code:** [peft/tuners/delora/layer.py](https://github.com/huggingface/peft/blob/main/src/peft/tuners/delora/layer.py)
**Saved:** [docs/delora_decoupled_low_rank_adaptation.md](docs/delora_decoupled_low_rank_adaptation.md)

**Hypothesis:** The *direction* of a weight update (which features to mix) and its *strength* (how far to deviate from pretrained weights) should be explicitly decoupled. LoRA conflates them via learning rate; ETHER fixes them. DeLoRA normalizes each rank-1 component of $BA$ by its norms and introduces a learnable scalar $\lambda$ controlling the distance bound. This yields robustness (bounded deviation) without sacrificing expressivity (arbitrary rank).

```py
# ── DeLoRA intervention ───────────
def delora_forward(x, W, A, B, λ, r, w_norm):
    # A ∈ ℝ^{r×d_in}, B ∈ ℝ^{d_out×r}: learned (like LoRA)
    # λ ∈ ℝ^r: learned per-component scaling (strength)
    # w_norm ∈ ℝ^{d_in}: frozen ||W||_col from init
    Â_norm = clamp(norm(A, dim=1), min=1e-4)        # ∈ ℝ^r
    B̂_norm = clamp(norm(B, dim=0), min=1e-4)        # ∈ ℝ^r
    scaling = (λ / r) / (Â_norm * B̂_norm)            # normalize each rank-1 component
    ΔW = B @ diag(scaling) @ A                       # direction normalized, strength via λ
    return W(x) + (x * w_norm) @ A.T @ diag(scaling) @ B.T
```

The key insight: $\Delta W = B \cdot \text{diag}\left(\frac{\lambda}{r \cdot \|a_i\| \cdot \|b^j\|}\right) \cdot A$. Each rank-1 outer product $b_i a_i^\top$ is normalized to unit norm, then scaled by $\lambda_i / r$. The angular component (which direction in weight space to move) trains freely; the radial component (how far) is controlled by $\lambda$.

**Evidence:** DeLoRA matches or surpasses LoRA, DoRA, and ETHER on subject-driven generation (DreamBooth), NLU (GLUE), and instruction tuning (LLaMA), while showing much better robustness to learning rate and training duration. The bounded deviation prevents catastrophic overwriting that plagues LoRA at high LR. Same authors as ETHER (Bini, Girrbach, Akata), extending their work on direction/strength decoupling.

**Grade:** PE+BL+DE=3.5 (beats LoRA on robustness, faster convergence via bounded deviation preventing catastrophic overwriting; ICLR 2025)

*Implications:* The strength/direction decoupling means gradient updates drive angular learning only -- the optimizer doesn't waste capacity fighting magnitude dynamics. Predictions: methods that explicitly decouple direction from strength will systematically show better OOD transfer, because the direction captures *what* to change while the strength captures *how much*, and only the former should be task-invariant.

---

## 6. PiSSA -- Principal Singular Values and Singular Vectors Adaptation

**Paper:** [Meng, Wang, Zhang 2024](https://arxiv.org/abs/2404.02948) (NeurIPS 2024)
**Code:** [github.com/GraphPKU/PiSSA](https://github.com/GraphPKU/PiSSA)
**Saved:** [docs/pissa_principal_singular_values_adaptation.md](docs/pissa_principal_singular_values_adaptation.md)

**Hypothesis:** The *principal components* of each weight matrix are what matter for adaptation. LoRA initializes adapters with random noise + zeros, so it starts far from the important subspace and converges slowly. PiSSA initializes $A$ and $B$ from the top-$r$ SVD of $W$, then freezes the residual $W_{\text{res}}$. Same architecture as LoRA, but trains the most important directions first.

```py
# ── PiSSA initialization + intervention ──
def pissa_init(W, r):
    U, S, Vt = svd(W)                              # W ∈ ℝ^{m×n}
    A = U[:, :r] @ diag(sqrt(S[:r]))                # A ∈ ℝ^{m×r}, principal left
    B = diag(sqrt(S[:r])) @ Vt[:r, :]               # B ∈ ℝ^{r×n}, principal right
    W_res = U[:, r:] @ diag(S[r:]) @ Vt[r:, :]     # residual, frozen
    return A, B, W_res

def pissa_forward(x, W_res, A, B):
    return (W_res + A @ B) @ x                      # same as LoRA at inference
```

The decomposition: $W = \underbrace{U_{:r} S_{:r} V_{:r}^\top}_{\text{adapter (learned)}} + \underbrace{U_{r:} S_{r:} V_{r:}^\top}_{\text{residual (frozen)}}$. LoRA updates noise; PiSSA updates the signal.

**Evidence:** PiSSA consistently outperforms LoRA across 11 models (184M--70B) on 5 NLG and 8 NLU tasks under identical setups. Gemma-7B on GSM8K: PiSSA 77.7% vs LoRA 74.5%. QPiSSA (quantized) on LLaMA-3-70B GSM8K: 86.05% vs QLoRA 81.73%. Faster convergence because the optimizer starts in the high-signal subspace. The initialization cost is negligible (fast SVD, a few seconds).

**Grade:** PE+BL+BF+DE=5 (beats LoRA, approaches/beats full FT, faster convergence, NeurIPS 2024)

*Implications:* PiSSA tells us something crucial about which weight-space directions matter: the top singular directions encode the most task-relevant structure. This is the "principal components carry the signal" hypothesis. It also suggests that LoRA's random init wastes early training steps re-discovering what SVD gives you for free. Connects to the broader question: is model adaptation about modifying the dominant signal or the residual noise? PiSSA says: the signal, always the signal.

---

## 7. SVFT -- Singular Vector Fine-Tuning

**Paper:** [Lingam et al. 2024](https://arxiv.org/abs/2405.19597)
**Code:** [github.com/VijayLingam95/SVFT](https://github.com/VijayLingam95/SVFT/)
**Saved:** [docs/svft_svd_coefficient_finetuning.md](docs/svft_svd_coefficient_finetuning.md)

**Hypothesis:** The structure of $\Delta W$ should depend on the specific weight matrix $W$. SVFT fixes both left and right singular vectors (from $W$'s own SVD) and learns only a *sparse set of coefficients* for their outer products. The weight matrix's own geometry defines the intervention basis; we just rescale which combinations of its existing directions to amplify or suppress.

```py
# ── SVFT intervention ─────────────
def svft_init(W, k):
    U, S, Vt = svd(W)                          # W ∈ ℝ^{m×n}
    # select k (i,j) pairs from {0..m-1} x {0..n-1}
    indices = select_sparse_pairs(k)            # e.g. band-diagonal, random
    c = zeros(k)                                # learned coefficients
    return U, Vt, indices, c                    # U, Vt frozen

def svft_forward(x, W, U, Vt, indices, c):
    ΔW = sum(c[t] * outer(U[:, i], Vt[j, :]) for t, (i,j) in enumerate(indices))
    return (W + ΔW) @ x                        # sparse combo of singular vector outer products
```

The key: $\Delta W = \sum_{t} c_t \cdot u_{i_t} v_{j_t}^\top$, where $u_i, v_j$ come from $W$'s SVD. Only the $c_t$ scalars are learned. Different sparsity patterns (band-diagonal, random, etc.) give different expressivity/efficiency tradeoffs.

**Evidence:** SVFT reports up to 96% of full fine-tuning performance with only 0.006--0.25% of parameters, outperforming LoRA/DoRA/BOFT ranges reported in the paper. Results are strong on language (GLUE, commonsense reasoning) and vision benchmarks. The weight-dependent structure is the key differentiator.

**Grade:** PE+BL=2 (beats LoRA/DoRA on performance/parameter tradeoff, weight-aware structure)

*Implications:* SVFT is the purest test of "does the model's own SVD basis define the right intervention space?" The answer appears to be yes: learning just coefficients over the model's own singular vectors is far more efficient than learning new arbitrary directions. This provides direct evidence that these singular vectors aren't arbitrary artifacts but encode *meaningful* computational directions. If combined with PiSSA's "top components matter most," we get a clear picture: the SVD basis is the natural coordinate system, and the singular values are the knobs.

---

## 8. SSVD -- Structured SVD-Guided Fine-Tuning

**Paper:** [Wang, Watanabe, Van hamme 2025](https://arxiv.org/abs/2509.02830)
**Saved:** [docs/ssvd_structured_svd_finetuning.md](docs/ssvd_structured_svd_finetuning.md)

**Hypothesis:** Input-space (right singular vectors $V$) and output-space (left singular vectors $U$) serve fundamentally different roles. Adaptation should *rotate* the input feature space to align with domain-shifted inputs while *preserving* the output semantic mappings. The right singular vectors define "what the layer listens to"; the left define "what it says". In domain shift, *what you listen to* changes, but *what you say* should stay.

```py
# ── SSVD intervention ─────────────
def ssvd_init(W, k):
    U, Σ, Vt = svd(W)                           # W ∈ ℝ^{m×n}
    K = zeros(k, k)                              # learned skew-symmetric matrix
    ΔΣ = zeros(k)                                # learned singular value shifts
    return U, Σ, Vt, K, ΔΣ                      # U, Σ, Vt frozen; K, ΔΣ learned

def ssvd_forward(x, U, Σ, Vt, K, ΔΣ, k):
    G_k = cayley(K)                              # G_k = (I-K)(I+K)^{-1} ∈ O(k)
    Σ̂ = Σ.clone()
    Σ̂[:k] += ΔΣ                                  # shift top-k singular values
    V̂t = Vt.clone()
    V̂t[:k] = G_k @ Vt[:k]                        # rotate top-k right singular vectors
    return U @ diag(Σ̂) @ V̂t @ x                  # W' = U (Σ+ΔΣ) G Vt x
```

$$W' = U (\Sigma + \Delta\Sigma) \, G_k \, V^\top$$

Only $k(k-1)/2 + k$ parameters (skew-symmetric entries + singular value shifts). Uses Cayley-Neumann approximation for efficiency.

**Evidence:** SSVD achieves comparable performance to LoRA, DoRA, PiSSA, VeRA, and SVFT on domain-shifted ASR (child speech, dialectal variation) across 0.1B--2B models, with significantly fewer trainable parameters. On OWSM-1B: SSVD matches LoRA WER with 10M fewer params. At larger scales, a convergence hierarchy emerges: SSVD > PiSSA > DoRA > LoRA. The gap grows with model scale, suggesting the asymmetric hypothesis becomes *more* valid as models get larger.

**Grade:** PE+BL+DE=3.5 (matches/beats LoRA with fewer params on domain-shifted ASR, faster convergence at scale; convergence hierarchy: SSVD > PiSSA > DoRA > LoRA)

*Implications:* SSVD's asymmetric treatment of U vs V is novel and deeply informative. It says: the model's "output vocabulary" (left singular vectors = what abstract features get produced) is already correct and should be preserved. Only the "input receptive fields" (right singular vectors = how raw features map into the abstract space) need updating for domain shift. This is exactly the right inductive bias for acoustic adaptation (accents change the input distribution, not the semantic targets). Predictions: this asymmetry should also work for visual domain adaptation (camera changes, lighting) but fail for tasks that require redefining the output space (new task types, new label semantics).

---

## 9. IA3 -- Infused Adapter by Inhibiting and Amplifying Inner Activations

**Paper:** [Liu et al. 2022](https://arxiv.org/abs/2205.05638)
**Code:** [peft/tuners/ia3/layer.py](https://github.com/huggingface/peft/blob/main/src/peft/tuners/ia3/layer.py)
**Saved:** [docs/ia3_few_shot_peft.md](docs/ia3_few_shot_peft.md)

**Hypothesis:** Task adaptation is mostly about *rescaling* what the model already computes, not restructuring it. A learned vector that element-wise scales activations at key, value, and FFN layers suffices. The pretrained model already extracts the right features; you just need to amplify the relevant ones and suppress the irrelevant ones. This is the "gain control" hypothesis -- adaptation as a gating/attention mechanism over existing channels.

```py
# ── IA3 intervention ──────────────
def ia3_forward(x, W, λ, is_feedforward):
    # λ ∈ ℝ^d: learned scaling vector, init to 1.0 (identity)
    if is_feedforward:
        return W @ (x * λ)             # scale input channels: amplify/suppress features
    else:
        return (W @ x) * λ             # scale output channels: amplify/suppress neurons
```

Merge into weights: $W_{\text{merged}} = W \odot \lambda$ (element-wise scaling of rows or columns). Extremely few trainable parameters -- just one $d$-dimensional vector per adapted layer.

**Evidence:** Authors claim (IA)3 with T0-3B outperforms ICL with GPT-3 175B on Super-NaturalInstructions while being orders of magnitude cheaper. Competitive with LoRA on RAFT leaderboard (rank 2 vs 3) with far fewer params. Strong on T5-family models. However, scaling-only methods have a clear expressivity ceiling: they cannot introduce new feature interactions, only reweight existing ones.

**Grade:** PE=1 (parameter-efficient, strong on T5-family, expressivity-limited compared to LoRA/DoRA)

*Implications:* IA3's success tells us that a surprisingly large fraction of "task adaptation" is just reweighting. The pretrained model already computes many useful features; the bottleneck is which ones to attend to for a given task, not computing new features from scratch. This connects to the neuroscience concept of "gain modulation" -- neurons don't change their tuning curves, just their amplitude. The limitation is equally informative: IA3 struggles on tasks requiring novel feature combinations, confirming that some adaptations genuinely require new weight-space directions, not just rescaling.

---

## 10. ROAD -- Rotary Adaptation

**Paper:** [Petrushkov 2024](https://arxiv.org/abs/2409.00119)
**Code:** [peft/tuners/road/layer.py](https://github.com/huggingface/peft/blob/main/src/peft/tuners/road/layer.py)
**Saved:** [docs/road_rotary_adaptation.md](docs/road_rotary_adaptation.md)

**Hypothesis:** Adaptation is a *rotation* of activation pairs, with independently controllable *angle* (which direction to rotate) and *magnitude* (how much to scale). The output space splits into 2D subspaces, and within each, a learned rotation + scaling suffices. This explicitly decouples "what to change" (angle $\theta$) from "how much" (magnitude $\alpha$), making the adaptation strength a continuous knob.

```py
# ── ROAD intervention ─────────────
def road_forward(x, W, θ, α, group_size):
    # θ ∈ ℝ^{d/2}: learned rotation angles per pair
    # α ∈ ℝ^{d/2}: learned magnitudes per pair, init 1.0
    result = W @ x                                  # base linear output ∈ ℝ^d
    x1, x2 = split_groups(result, group_size)       # split into paired halves
    y1 = α * cos(θ) * x1 - α * sin(θ) * x2         # 2D rotation + scale
    y2 = α * sin(θ) * x1 + α * cos(θ) * x2         # per pair
    return interleave(y1, y2)
```

$$R_i = \alpha_i \begin{pmatrix} \cos\theta_i & -\sin\theta_i \\ \sin\theta_i & \cos\theta_i \end{pmatrix}$$

Applied element-wise (no matrix multiply needed at inference). Merges into weights via $W_{\text{new}} = R \cdot W$.

**Evidence:** ROAD is the only PEFT method besides LoRA that supports mixed adapter batches (different adapters for different samples in the same batch). Authors claim competitive with LoRA on standard benchmarks. The explicit angle/magnitude decoupling makes it ideal for contrastive steering: scale only $\alpha$ for bidirectional control while preserving learned rotation directions $\theta$.

**Grade:** PE=1 (parameter-efficient, clean decoupling, competitive with LoRA, limited published benchmarks)

*Implications:* ROAD's decoupling of angle from magnitude is the cleanest formulation of the "direction vs strength" principle that also appears in DeLoRA and DoRA. The 2D rotation structure connects to RoPE (rotary position embeddings) -- both use paired rotations in subspaces, suggesting this is a natural symmetry of transformer representations. For steering applications, ROAD's explicit $\alpha$ parameter is the most interpretable knob: $\alpha = 1$ is identity, $\alpha > 1$ amplifies, $\alpha < 1$ attenuates, $\alpha = -1$ reverses.

---

## 11. AntiPaSTO -- Antiparallel Steering via SVD Rotations

*Disclosure: this is my own work. I give it the highest grade here, so read the evidence with appropriate skepticism.*

**Paper:** [Clark 2025](https://arxiv.org/abs/2601.07473)
**Code:** [github.com/wassname/AntiPaSTO](https://github.com/wassname/AntiPaSTO)
**Saved:** [docs/antipasto_antiparallel_steering.md](docs/antipasto_antiparallel_steering.md)

**Hypothesis:** The model's own SVD basis defines the natural intervention manifold. Steering is best done by *rotating* singular vectors via Cayley transform on a learned skew-symmetric matrix, parameterized by a single coefficient $\alpha \in [-1, +1]$. The Cayley transform guarantees exact orthogonality and exact reversibility: $R(-\alpha) = R(\alpha)^{-1}$. Separating rotation (learned direction) from magnitude ($\alpha$) yields antiparallel steering -- the same adapter produces opposite behavioral shifts at $\alpha = \pm 1$.

The core claim synthesizes SSVD + PiSSA + DeLoRA: use the model's own top-$r$ SVD basis (PiSSA), rotate right singular vectors via Cayley (SSVD), decouple direction from strength (DeLoRA), and add learnable singular value shifts.

```py
# ── AntiPaSTO intervention ────────
def antipasto_init(W, r):
    U, S, Vt = svd(W)                              # W ∈ ℝ^{m×n}
    U_r, S_r, V_r = U[:, :r], S[:r], Vt[:r].T      # top-r components
    W_res = W - U_r @ diag(S_r) @ V_r.T             # residual (frozen)
    A_v = zeros(r, r)                                # skew-symmetric rotation params for V
    ΔS = zeros(r)                                    # learnable singular value shifts
    return U_r, S_r, V_r, W_res, A_v, ΔS            # U,S,V,W_res frozen; A_v,ΔS learned

def antipasto_forward(x, U, S, V, W_res, A_v, ΔS, α):
    # α ∈ [-1, +1]: steering coefficient (continuous knob)
    X = α * A_v / 2                                  # scale skew-symmetric params
    R_v = solve(I - X, I + X)                        # Cayley: (I - αA/2)^{-1}(I + αA/2) ∈ O(r)
    V_rot = V @ R_v                                  # rotate input-space basis
    S_scaled = S + α * ΔS                            # shift singular values
    # Efficient: x @ V_rot @ diag(S_scaled) @ U^T + x @ W_res^T
    h = (x @ V_rot) * S_scaled @ U.T                 # adapted path
    return h + x @ W_res.T                           # + residual
```

$$W'(\alpha) = U \, \text{diag}(S + \alpha \Delta S) \, R_v(\alpha) \, V^\top + W_{\text{res}}$$

where $R_v(\alpha) = (I - \alpha A/2)^{-1}(I + \alpha A/2)$ is the Cayley transform of skew-symmetric $A$. Only $r(r-1)/2 + r$ learned parameters per layer.

**Evidence:** AntiPaSTO beats prompting baselines by 6.9x on DailyDilemmas honesty evaluation using Gemma-3-1B. Maintains bidirectional control ($\alpha = \pm 1$) where prompting triggers refusal. Trains with only 800 contrastive word pairs (no preference labels). Transfers out-of-distribution from template sentences to real ethical dilemmas. The OOD transfer suggests the SVD rotation basis captures something causally relevant about the model's honesty computations.

**Grade:** PE+DE+OOD=4.5 (OOD transfer from templates to real dilemmas, trains on 800 pairs, bidirectional control)

*Caveat:* Primary evidence is on models up to 4B parameters. The paper notes larger models "need further exploration" and results show high seed variance. The OOD transfer claim is strong but narrow (one trait, one evaluation benchmark).

*Implications:* Synthesizes PiSSA's SVD initialization, SSVD's input-space rotation, DeLoRA's direction/strength decoupling, and OFT's Cayley parameterization. The OOD transfer from templates to real dilemmas suggests the SVD manifold is causally relevant, not just efficient. The antiparallel property ($+\alpha$ and $-\alpha$ produce opposite effects) is a natural consequence of rotational symmetry. Caveat: single-author, one trait, high seed variance; would be more confident with independent replication.

---

## 12. AdaLoRA -- Adaptive Budget Allocation for LoRA

**Paper:** [Zhang et al. 2023](https://arxiv.org/abs/2303.10512) (ICLR 2023)
**Code:** [peft/tuners/adalora](https://github.com/huggingface/peft/blob/main/src/peft/tuners/adalora/)
**Saved:** [docs/adalora_adaptive_budget.md](docs/adalora_adaptive_budget.md)

**Hypothesis:** Not all layers need the same rank. The optimal rank distribution across layers is *adaptive* and should be learned during training. Some weight matrices need high-rank updates (they are task-critical); others need almost none. SVD-based importance scoring can dynamically prune less important singular values, reallocating budget where it matters.

```py
# ── AdaLoRA intervention ──────────
def adalora_forward(x, W, P, Λ, Q):
    # P ∈ ℝ^{d_out×r}, Q ∈ ℝ^{r×d_in}: left/right singular vectors (learned)
    # Λ ∈ ℝ^r: singular values (learned, prunable via importance mask)
    ΔW = P @ diag(Λ) @ Q                          # SVD-parameterized update
    return (W + ΔW) @ x

def prune_step(P, Λ, Q, budget):
    importance = compute_importance(P, Λ, Q)       # sensitivity-based scoring
    mask = top_k(importance, budget)               # keep top-budget components
    Λ_pruned = Λ * mask                            # zero out unimportant
    return Λ_pruned
```

**Evidence:** Authors claim AdaLoRA achieves comparable or better performance than LoRA with 30-50% fewer total parameters on DeBERTaV3-base across NLU tasks. The adaptive rank allocation concentrates budget on query/value projections and early/late layers. Orthogonal regularization on P, Q prevents degenerate solutions. However, the pruning adds training complexity and the final rank pattern is model/task-specific, limiting transferability of the insight.

**Grade:** PE=1 (parameter-efficient, smarter budget allocation, added complexity for modest gains)

---

## 13. BOFT -- Butterfly Orthogonal Fine-Tuning

**Paper:** [Liu et al. 2023](https://arxiv.org/abs/2311.06243) (ICLR 2024)
**Code:** [peft/tuners/boft](https://github.com/huggingface/peft/blob/main/src/peft/tuners/boft/)
**Saved:** [docs/boft_butterfly_orthogonal.md](docs/boft_butterfly_orthogonal.md)

**Hypothesis:** Orthogonal transformations (OFT's key insight) are right, but the full block-diagonal parameterization is wasteful. Butterfly factorizations -- the same structure behind the FFT -- can represent arbitrary orthogonal transformations with $O(d \log d)$ parameters instead of $O(d^2)$, while maintaining the information-theoretic expressiveness needed for adaptation.

```py
# ── BOFT intervention ─────────────
def boft_forward(x, W, butterfly_blocks, n_layers):
    R = eye(d)
    for l in range(n_layers):                       # log(d) butterfly layers
        B_l = block_diag(butterfly_blocks[l])       # sparse butterfly factor
        R = R @ B_l                                 # compose: R ∈ O(d)
    return (R @ W) @ x                             # orthogonal rotation of W
```

Each butterfly layer has $d/2$ independent $2\times2$ rotation blocks arranged in a permuted pattern. Composing $\log_2(d)$ layers can represent any orthogonal matrix.

**Evidence:** BOFT matches or exceeds OFT performance on DreamBooth and ControlNet with 2-4x fewer parameters. Authors demonstrate it preserves hyperspherical energy like OFT. The butterfly structure provides a principled trade-off between expressiveness and parameter count. Strong on vision/generation tasks where semantic preservation matters. ICLR 2024 acceptance validates the contribution.

**Grade:** PE+BF+DE=4 (outperforms full FT baseline on almost all vision tasks; faster and more stable convergence with denser butterfly structure; ICLR 2024)

---

## 14. GOFT -- Givens Orthogonal Fine-Tuning

**Paper:** [Ma et al. 2024](https://arxiv.org/abs/2404.04316) (ICML 2024)
**Code:** [github.com/ArthurLeoM/peft-givens](https://github.com/ArthurLeoM/peft-givens)
**Saved:** [docs/goft_givens_orthogonal.md](docs/goft_givens_orthogonal.md)

**Hypothesis:** Any orthogonal transformation in $SO(d)$ can be decomposed into $O(d)$ Givens rotations (planar rotations in 2D subplanes), reducing parameter complexity from $O(d^2)$ to $O(d)$. This is the most parameter-efficient parameterization of orthogonal adaptation. Beyond strict orthogonality, soft orthogonality regularization allows controlled norm and angular adjustment.

```py
# ── GOFT intervention ─────────────
def goft_forward(x, W, θ_list, pairs):
    # θ_list ∈ ℝ^{d}: rotation angles for d Givens rotations
    # pairs: which (i,j) dimensions each rotation acts on
    R = eye(d)
    for θ, (i, j) in zip(θ_list, pairs):
        G = givens_rotation(d, i, j, θ)            # identity except 2x2 block at (i,j)
        R = R @ G                                   # compose all rotations
    # With soft orthogonality, also learn norm adjustments
    return (R @ W) @ x
```

**Evidence:** Authors claim GOFT outperforms OFT and BOFT on LLaMA-2-7B SFT (MT-Bench, AlpacaEval), DreamBooth, and offline RL tasks while using significantly fewer parameters. The parallel rotation strategy achieves $O(\log d)$ sparse matrix multiplication. ICML 2024 acceptance. The Givens decomposition is mathematically elegant and provably equivalent to full orthogonal transformations.

**Grade:** PE=1 (most parameter-efficient orthogonal method, strong results, ICML 2024)

---

## 15. HRA -- Householder Reflection Adaptation

**Paper:** [Yuan et al. 2024](https://arxiv.org/abs/2405.17484)
**Code:** [peft/tuners/hra](https://github.com/huggingface/peft/blob/main/src/peft/tuners/hra/)
**Saved:** [docs/hra_householder_reflection.md](docs/hra_householder_reflection.md)

**Hypothesis:** Orthogonal adaptations are equivalent to specific low-rank adaptations when parameterized via Householder reflections. A chain of $r$ Householder reflections $H_1 H_2 \cdots H_r$ (each defined by a single vector $v_i$) constructs an orthogonal matrix with exactly $r \times d$ learnable parameters -- bridging the low-rank and orthogonal adaptation paradigms.

```py
# ── HRA intervention ──────────────
def hra_forward(x, W, V):
    # V ∈ ℝ^{r×d}: r Householder reflection vectors
    R = eye(d)
    for i in range(r):
        v = V[i]                                    # reflection normal ∈ ℝ^d
        H_i = eye(d) - 2 * outer(v, v) / dot(v, v) # Householder reflector
        R = R @ H_i                                  # compose: R ∈ O(d)
    return (R @ W) @ x
```

Each reflection flips the space across a hyperplane. Composing $r$ of them gives a rank-$r$ "distance" from identity while staying exactly orthogonal.

**Evidence:** Authors demonstrate HRA achieves competitive or better results than LoRA and OFT on LLaMA fine-tuning and image generation. The theoretical equivalence between Householder chains and adaptive low-rank updates is the main contribution: same expressiveness as rank-$r$ LoRA with guaranteed orthogonality. Regularization on reflection plane orthogonality improves stability.

**Grade:** PE=1 (bridges orthogonal and low-rank paradigms, competitive performance)

*Implications:* HRA reveals that the "low-rank vs orthogonal" dichotomy is a false one. A chain of $r$ Householder reflections is *both* orthogonal *and* equivalent to a rank-$r$ perturbation. This means LoRA's success (low rank works) and OFT's success (orthogonality works) are compatible: the effective adaptation might be low-rank *and* approximately orthogonal simultaneously. If true, the right constraint isn't "low rank" or "orthogonal" alone, but "low-rank orthogonal" -- small rotations that stay on the Stiefel manifold.

---

## 16. RandLoRA -- Random Matrix LoRA

**Paper:** [Albert et al. 2025](https://arxiv.org/abs/2502.00987) (ICLR 2025)
**Code:** [peft/tuners/randlora](https://github.com/huggingface/peft/blob/main/src/peft/tuners/randlora/)
**Saved:** [docs/randlora_random_matrix.md](docs/randlora_random_matrix.md)

**Hypothesis:** LoRA's rank bottleneck ($\text{rank}(\Delta W) \leq r$) limits expressiveness. By summing $n = d/r$ scaled random rank-$r$ bases, the update $\Delta W = \sum_j B_j \Lambda_j A \Gamma_j$ achieves full rank while learning only diagonal scaling matrices. Each frozen random basis $B_j, A$ spans a different subspace; the learnable scalings $\Lambda_j, \Gamma_j$ select how much of each to use.

```py
# ── RandLoRA intervention ─────────
def randlora_forward(x, W, A, B_list, Λ_list, Γ_list):
    # A ∈ ℝ^{r×d_in}: shared frozen random matrix
    # B_list: n frozen random matrices, each B_j ∈ ℝ^{d_out×r}
    # Λ_list: n learned diagonal scalings, each Λ_j ∈ ℝ^{r×r}
    # Γ_list: n learned diagonal scalings, each Γ_j ∈ ℝ^{d×d}
    ΔW = sum(B_j @ Λ_j @ A @ Γ_j for B_j, Λ_j, Γ_j  # sum of n rank-r terms = full rank
             in zip(B_list, Λ_list, Γ_list))
    return (W + ΔW) @ x
```

**Evidence:** RandLoRA outperforms LoRA as parameter budget expands, while remaining parameter-efficient. DinoV2, CLIP, and LLaMA-3-8B experiments show LoRA hits a rank ceiling (increasing rank has diminishing returns) while RandLoRA continues to improve. Loss landscape analysis shows RandLoRA's local minima are closer to full fine-tuning's. ICLR 2025.

**Grade:** PE+BF=2.5 (full-rank update bridges gap with full FT on CLIP; loss landscape closer to full FT's local minima; ICLR 2025)

---

## 17. FourierFT -- Fourier Fine-Tuning

**Paper:** [Gao et al. 2024](https://arxiv.org/abs/2405.03003) (ICML 2024)
**Code:** [peft/tuners/fourierft](https://github.com/huggingface/peft/blob/main/src/peft/tuners/fourierft/)
**Saved:** [docs/fourierft_spectral.md](docs/fourierft_spectral.md)

**Hypothesis:** Weight updates $\Delta W$ are *spectrally sparse* -- they can be represented by a small number of Fourier coefficients. Instead of parameterizing $\Delta W$ in the spatial domain (like LoRA), learn a sparse set of spectral coefficients and reconstruct via inverse DFT. This exploits the observation that useful weight changes tend to be smooth/structured rather than random.

```py
# ── FourierFT intervention ────────
def fourierft_forward(x, W, coeffs, freq_indices, shape):
    # coeffs ∈ ℂ^k: learned spectral coefficients (k << m*n)
    # freq_indices: which frequency components to learn
    spectrum = zeros(shape, dtype=complex)
    spectrum[freq_indices] = coeffs                  # sparse spectrum
    ΔW = real(ifft2(spectrum))                       # inverse 2D DFT
    return (W + ΔW) @ x
```

**Evidence:** Authors claim FourierFT achieves higher compression than LoRA by exploiting frequency-domain sparsity. Competitive with LoRA on GLUE and commonsense reasoning using fewer parameters. ICML 2024 acceptance. The spectral sparsity hypothesis is interesting but the evidence for *why* weight changes should be low-frequency is largely empirical.

**Grade:** PE+BF=2.5 (outperforms all baselines including full FT on RoBERTa-Base CoLA and RoBERTa-Large RTE; ICML 2024)

---

## 18. C3A -- Circular Convolution Adaptation

**Paper:** [Phoveran et al. 2024](https://arxiv.org/abs/2407.19342) (ACL 2025)
**Code:** [peft/tuners/c3a](https://github.com/huggingface/peft/blob/main/src/peft/tuners/c3a/)
**Saved:** [docs/c3a_circular_convolution.md](docs/c3a_circular_convolution.md)

**Hypothesis:** Weight updates have *circulant structure* -- the matrix $\Delta W$ is approximately a circulant matrix (each row is a cyclic shift of the previous). Circulant matrices are diagonalized by the DFT, so efficient computation via FFT is possible. Unlike LoRA which is rank-limited, circulant matrices can have full rank with only $d$ parameters (one generating vector).

```py
# ── C3A intervention ──────────────
def c3a_forward(x, W, c):
    # c ∈ ℝ^d: generating vector for circulant matrix
    ΔW = circulant(c)                               # ΔW[i,j] = c[(j-i) mod d]
    # Efficient via FFT: ΔW @ x = ifft(fft(c) * fft(x))
    return (W + ΔW) @ x
```

**Evidence:** Authors claim C3A achieves higher effective rank than LoRA with similar parameter count and compute. Competitive on GLUE, commonsense reasoning, and instruction tuning. ACL 2025 acceptance. The FFT-based computation is genuinely efficient. However, the assumption of circulant structure in weight updates is strong and may not hold universally.

**Grade:** PE=1 (full-rank with fewer params, ACL 2025, circulant assumption is strong)

---

## 19. LoHa -- Low-Rank Hadamard Product

**Paper:** [Hyeon-Woo et al. 2021](https://arxiv.org/abs/2108.06098) (FedPara; adapted in [LyCORIS](https://arxiv.org/abs/2309.14859))
**Code:** [peft/tuners/loha](https://github.com/huggingface/peft/blob/main/src/peft/tuners/loha/)
**Saved:** [docs/loha_hadamard_product.md](docs/loha_hadamard_product.md)

**Hypothesis:** Weight updates have *multiplicative* structure that a single low-rank factorization misses. By combining two low-rank decompositions via Hadamard (element-wise) product, more complex interaction patterns can be captured. $(A_1 B_1) \odot (A_2 B_2)$ can represent higher-rank updates than either factor alone.

```py
# ── LoHa intervention ─────────────
def loha_forward(x, W, A1, B1, A2, B2):
    # Each pair (Ai, Bi): rank-r decomposition
    ΔW = (A1 @ B1) * (A2 @ B2)                     # Hadamard product, potentially full-rank
    return (W + ΔW) @ x
```

**Evidence:** Part of the LyCORIS toolkit. Authors claim LoHa achieves richer expressiveness than LoRA for the same parameter count, particularly for image generation (Stable Diffusion fine-tuning) where complex spatial interactions matter. The Hadamard product inherently captures pairwise feature interactions that additive low-rank matrices cannot.

**Grade:** PE=1 (richer than LoRA for vision, part of LyCORIS ecosystem)

---

## 20. LoKr -- Low-Rank Kronecker Product

**Paper:** [Yeh et al. 2023](https://arxiv.org/abs/2309.14859) (LyCORIS)
**Code:** [peft/tuners/lokr](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lokr/)
**Saved:** [docs/lokr_lycor.md](docs/lokr_lycor.md)

**Hypothesis:** Weight updates have *tensor product* structure. The Kronecker factorization $\Delta W = A \otimes B$ decomposes a large matrix into the tensor product of two smaller ones, exploiting multi-scale or block-structured patterns in adaptation. Especially efficient for high-dimensional or convolutional weight matrices.

```py
# ── LoKr intervention ─────────────
def lokr_forward(x, W, A, B):
    # A ∈ ℝ^{m1×m2}, B ∈ ℝ^{n1×n2}, where m1*n1 = d_out, m2*n2 = d_in
    ΔW = kron(A, B)                                 # ΔW ∈ ℝ^{d_out × d_in}
    return (W + ΔW) @ x
```

**Evidence:** Part of LyCORIS. Kronecker structure is especially effective for convolutional layers where the weight tensor naturally factorizes across spatial and channel dimensions. Compact parameterization for large weight matrices. Less commonly used for LLMs where the spatial structure assumption doesn't hold as well.

**Grade:** PE=1 (efficient for conv layers, niche use case for transformers)

---

## 21. MiSS -- Matrix Shard Sharing

**Paper:** [JL-er 2024](https://arxiv.org/abs/2409.15371)
**Code:** [peft/tuners/miss](https://github.com/huggingface/peft/blob/main/src/peft/tuners/miss/)
**Saved:** [docs/miss_matrix_shard_sharing.md](docs/miss_matrix_shard_sharing.md)

**Hypothesis:** Weight updates share *structural motifs* across layers. Instead of learning independent low-rank matrices per layer, share "shards" (small matrix blocks) across layers through a weight-magnitude-based scoring system. Layers with similar function should reuse similar update patterns, and the scoring identifies which layers are similar.

```py
# ── MiSS intervention ─────────────
def miss_forward(x, W, shared_shards, scores):
    # shared_shards: global bank of small matrix blocks
    # scores: per-layer importance weights selecting which shards to use
    ΔW = assemble(shared_shards, scores)            # weighted combination of shards
    return (W + ΔW) @ x
```

**Evidence:** Successor to Bone (deprecated). PEFT benchmark comparison shows "excellent results" in both performance and memory efficiency. Adaptive rank allocation via shard scoring. Reduced memory compared to full per-layer LoRA matrices. However, the shard sharing mechanism adds implementation complexity.

**Grade:** PE+DE=2.5 (memory-efficient, faster early convergence via larger initial gradient norms; good benchmark results per PEFT team)

---

## 22. VBLoRA -- Vector Bank LoRA

**Paper:** [Li et al. 2024](https://arxiv.org/abs/2405.15179) (NeurIPS 2024)
**Code:** [peft/tuners/vblora](https://github.com/huggingface/peft/blob/main/src/peft/tuners/vblora/)
**Saved:** [docs/vblora_vector_bank.md](docs/vblora_vector_bank.md)

**Hypothesis:** Adapter weight matrices are *sparse combinations of shared atomic vectors*. Instead of learning full low-rank matrices, maintain a shared "vector bank" and select/combine top-$k$ vectors per layer. This is a codebook/dictionary learning approach: the adaptation vocabulary is shared globally, and each layer's adapter is a sparse code over it.

```py
# ── VBLoRA intervention ───────────
def vblora_forward(x, W, bank, indices, coeffs):
    # bank ∈ ℝ^{V×d}: shared vector bank (V vectors)
    # indices ∈ ℤ^k: top-k selected vectors per layer
    # coeffs ∈ ℝ^k: combination weights
    selected = bank[indices]                         # k most relevant vectors
    ΔW = sum(coeffs[i] * outer(selected[i]) for i in range(k))  # sparse reconstruction
    return (W + ΔW) @ x
```

**Evidence:** Authors claim VBLoRA uses 0.4% of LoRA's parameters while maintaining comparable performance. NeurIPS 2024 acceptance. The extreme compression is remarkable and suggests that adapter weight diversity across layers is much lower than assumed -- most of the information is in *which* vectors to select and *how much* of each, not in the vectors themselves.

**Grade:** PE=1 (extreme compression, NeurIPS 2024, intriguing theoretical implications)

---

## 23. SHiRA -- Sparse High-Rank Adapters

**Paper:** [KKB et al. 2024](https://arxiv.org/abs/2406.13175) (NeurIPS 2024 Workshop)
**Code:** [peft/tuners/shira](https://github.com/huggingface/peft/blob/main/src/peft/tuners/shira/)
**Saved:** [docs/shira_sparse_high_rank.md](docs/shira_sparse_high_rank.md)

**Hypothesis:** The right parameterization isn't low-rank *or* full-rank, but *sparse high-rank*. Directly fine-tune 1-2% of the base model's weights, selected by importance scoring. The updated weights can have full rank (no rank bottleneck), but the sparsity pattern constrains which parameters change. The hypothesis: a small fraction of weights are "task-critical knobs" that, when tuned, achieve most of adaptation's benefit.

```py
# ── SHiRA intervention ────────────
def shira_forward(x, W, mask, ΔW_sparse):
    # mask ∈ {0,1}^{d_out × d_in}: 1-2% of entries are 1
    # ΔW_sparse: learned updates at mask positions only
    W_adapted = W + mask * ΔW_sparse                # sparse but full-rank update
    return W_adapted @ x
```

**Evidence:** Authors report SHiRA outperforms LoRA especially on concept-loss-sensitive multi-adapter settings (critical for diffusion model fine-tuning). Sparse adapters are cheaper to switch between than LoRA. NeurIPS 2024 Workshop. The importance-scoring approach connects to structured pruning literature.

**Grade:** PE=1 (sparse high-rank, good multi-adapter properties, workshop paper)

---

## 24. LN Tuning -- LayerNorm Tuning

**Paper:** [undated](https://arxiv.org/abs/2312.11420)
**Code:** [peft/tuners/ln_tuning](https://github.com/huggingface/peft/blob/main/src/peft/tuners/ln_tuning/)

**Hypothesis:** Normalization layers (LayerNorm/RMSNorm) are the *distribution controllers* of the network. Tuning only their affine parameters ($\gamma$, $\beta$) adapts how each layer normalizes its inputs, which is sufficient for many tasks because distribution shift is the primary thing that changes between pretraining and fine-tuning.

```py
# ── LN Tuning intervention ────────
def ln_tuning_forward(x, W_frozen, γ, β):
    # Only γ ∈ ℝ^d and β ∈ ℝ^d are trainable (LayerNorm params)
    x_norm = (x - mean(x)) / std(x) * γ + β       # adapted normalization
    return W_frozen @ x_norm                        # rest of network frozen
```

**Evidence:** Authors claim LN Tuning with ~0.5% trainable parameters can match LoRA performance on some NLU tasks. The extreme simplicity is informative: if tuning only normalization suffices, then much of "task adaptation" is really "distribution matching." Less effective on tasks requiring new feature representations rather than feature rescaling.

**Grade:** PE=1 (extremely few params, competitive on some tasks, limited expressiveness)

---

## 25. Prompt & Prefix Tuning -- Learned Virtual Tokens

**Papers:** Prompt Tuning ([Lester et al. 2021](https://arxiv.org/abs/2104.08691)), Prefix Tuning ([Li & Liang 2021](https://arxiv.org/abs/2101.00190)), P-Tuning v2 ([Liu et al. 2022](https://arxiv.org/abs/2110.07602)), Adaption Prompt / LLaMA-Adapter ([Zhang et al. 2023](https://arxiv.org/abs/2303.16199)), Multitask Prompt Tuning ([Asai et al. 2023](https://arxiv.org/abs/2303.02861)), CPT ([Tsachiblau 2024](https://arxiv.org/abs/2410.17222))

**Hypothesis:** The model's prompt/context is the primary interface for task specification. Learning "virtual tokens" (continuous embeddings prepended to the input) provides enough signal for downstream tasks without modifying any model weights. The hypothesis: the model's computation is *already* capable of the target task; it just needs the right "instruction" in embedding space. This is the "models are instruction-following programs" view.

```py
# ── Prompt Tuning intervention ────
def prompt_tuning_forward(x, model, P):
    # P ∈ ℝ^{k×d}: k learned prompt vectors (virtual tokens)
    x_prompted = cat([P, x], dim=seq)               # prepend prompts
    return model(x_prompted)                        # model is fully frozen
```

Variants: Prefix Tuning adds prompts to key/value projections at every layer. P-Tuning v2 applies deep prompts to all layers. LLaMA-Adapter uses zero-initialized gating. CPT uses adversarial-inspired optimization for context-aware prompts.

**Evidence:** Prompt Tuning scales with model size: at T5-XXL (11B), it matches full fine-tuning with 0.01% parameters. However, it struggles on smaller models and hard sequence labeling tasks. Prefix Tuning achieves comparable results with ~0.1% parameters on generation tasks. The prompt paradigm is fundamentally different from weight adaptation: it modifies the *input* rather than the *computation*. When it works, it suggests the model already has the capability; when it fails, it reveals genuine capability gaps.

**Grade:** PE=1 (scales with model size, conceptually different from weight methods)

---

## 26. Poly / X-LoRA -- Mixture of Adapters

**Papers:** Polytropon ([Ponti et al. 2022](https://arxiv.org/abs/2202.13914)), X-LoRA ([Buehler 2024](https://arxiv.org/abs/2402.07148))

**Hypothesis:** Task adaptation isn't monolithic -- it's *compositional*. A shared library of "skill modules" (small adapters) can be recombined via learned routing to handle diverse tasks. The routing coefficients select which skills to activate for each input, forming a mixture-of-experts over adapter space.

```py
# ── X-LoRA intervention ───────────
def xlora_forward(x, W, adapters, gating_net):
    # adapters: list of LoRA experts {(A_i, B_i)}
    # gating_net: maps hidden states to mixing weights
    gate = softmax(gating_net(x))                    # ∈ ℝ^{n_experts}
    ΔW = sum(gate[i] * (B_i @ A_i) for i in range(n_experts))
    return (W + ΔW) @ x
```

**Evidence:** X-LoRA achieves better composite performance than individual LoRAs by dynamically routing through appropriate expert adapters. Polytropon demonstrates cross-task transfer via shared skill libraries. The compositionality assumption is powerful but adds routing overhead and complexity. More suited to multi-task deployment than single-task fine-tuning.

**Grade:** PE=1 (compositional multi-task, routing overhead)

---

## 27. ETHER -- Efficient fine-THEning by oRthogonal transformation

**Paper:** [Bini, Girrbach, Akata 2024](https://arxiv.org/abs/2405.20271)
**Code:** Not in PEFT (standalone)
**Saved:** [docs/ether_orthogonal_steering.md](docs/ether_orthogonal_steering.md)
**See also:** BiPDO ([2024](https://arxiv.org/abs/2406.00045)), repeng/representation engineering

**Hypothesis:** *Fixed-strength* orthogonal transformations are sufficient for behavioral steering. ETHER learns a single orthogonal rotation matrix applied to weight matrices, with the constraint that the transformation distance from identity is bounded. Unlike OFT which allows flexible-strength orthogonal updates, ETHER deliberately constrains the deviation, trading expressiveness for robustness and reversibility.

```py
# ── ETHER intervention ────────────
def ether_forward(x, W, R):
    # R ∈ O(d): learned orthogonal matrix, close to identity
    return (R @ W) @ x                              # fixed-strength rotation
```

**Evidence:** Bini, Girrbach, Akata (same authors as DeLoRA). ETHER demonstrates that fixed-strength orthogonal transformations can achieve competitive task adaptation while preventing catastrophic forgetting. The bounded deviation is both a feature (robustness) and a limitation (ceiling on complex tasks). ETHER's constraints motivated DeLoRA's more flexible direction/strength decoupling.

**Grade:** PE+DE=2.5 (fast convergence by default via high learning rate robustness; robust fixed-strength rotations; foundational for DeLoRA)

*Implications:* ETHER represents the "minimal intervention" extreme of the orthogonal hypothesis. By showing that *bounded* rotations work for many tasks, it establishes a baseline: how much deviation from pretrained weights is actually needed? The answer appears to be "less than you think for behavioral steering, more than you think for complex task adaptation." This informed DeLoRA's key insight: decouple strength from direction, and let each evolve independently.

---

## 28. OFTv2 -- Input-Centric Orthogonal Fine-Tuning

**Paper:** [2025](https://arxiv.org/abs/2506.19847) (EMNLP 2025)
**Code:** [peft/tuners/oft](https://github.com/huggingface/peft/blob/main/src/peft/tuners/oft/) (improved implementation)
**Saved:** [docs/oftv2_input_centric.md](docs/oftv2_input_centric.md)

**Hypothesis:** OFT's computational bottleneck (cubic complexity from weight-centric matrix-matrix multiplication) is an implementation artifact, not a fundamental limitation. By reformulating to input-centric matrix-vector multiplication and using Cayley-Neumann series for approximate matrix inversion, OFT can be made practical at scale (quadratic complexity, 10x faster, 3x less memory).

```py
# ── OFTv2 intervention ────────────
def oftv2_forward(x, W, Q):
    # Instead of computing R = cayley(Q) then R @ W,
    # directly compute R @ (W @ x) via matrix-vector ops
    z = W @ x                                        # standard linear output
    Q_skew = skew_symmetric(Q)
    # Cayley-Neumann: (I - Q)^{-1} ≈ I + Q + Q^2 + ... (truncated)
    Rx = z + Q_skew @ z + Q_skew @ (Q_skew @ z)    # Neumann approximation
    return Rx
```

**Evidence:** Authors claim 10x faster training and 3x lower GPU memory than OFT without performance loss. Supports quantized foundation models and outperforms QLoRA in training stability. EMNLP 2025 acceptance. The key insight is purely computational: the same mathematical operation (orthogonal rotation) can be implemented much more efficiently in input-centric form.

**Grade:** PE=1 (same hypothesis as OFT, much more practical)

---

## 29. Bone -- Block-Affine Adaptation (Deprecated)

**Paper:** [JL-er 2024](https://arxiv.org/abs/2409.15371)
**Code:** Deprecated in PEFT, replaced by MiSS

**Hypothesis:** Weight updates have block-affine structure. Each block of the weight matrix undergoes an independent affine transformation (rotation + shift), combining HRA-style Householder reflections with per-block bias terms.

**Evidence:** Superseded by MiSS (entry 21), which generalizes the shard-sharing idea more cleanly. Listed for completeness.

**Grade:** (deprecated, see MiSS)

---

## 30. Trainable Tokens -- Vocabulary Extension

**Code:** [peft/tuners/trainable_tokens](https://github.com/huggingface/peft/blob/main/src/peft/tuners/trainable_tokens/)

**Hypothesis:** Not a weight adaptation method. Extends the vocabulary embedding matrix with new learnable token embeddings (e.g., for reasoning/thinking tokens). Combinable with LoRA. Listed for completeness but outside the scope of the weight-adaptation hypothesis framework.

---

## 31. CLOVER -- Cross-Layer Joint SVD Adaptation

**Paper:** [Meng, Tang, Jiang, Zhang 2024](https://arxiv.org/abs/2411.17426)
**Code:** [github.com/fanxu-meng/CLOVER](https://github.com/fanxu-meng/CLOVER)
**Saved:** [docs/clover_joint_svd.md](docs/clover_joint_svd.md)

**Hypothesis:** Attention layers have *cross-layer redundancy* in their SVD structure. Rather than adapting Q, K, V, O projections independently, CLOVER performs joint SVD across paired attention matrices (Q-K and V-O), exploiting the shared singular subspace between layers that cooperate functionally. The weight matrices within a head are not independent -- they jointly define the attention computation, so their adaptation should be coupled.

```py
# ── CLOVER intervention ───────────
def clover_init(W_q, W_k, W_v, W_o, r):
    # Joint SVD of paired attention matrices
    W_qk = cat([W_q, W_k], dim=0)               # stack Q-K pairs
    W_vo = cat([W_v, W_o.T], dim=0)              # stack V-O pairs
    U_qk, S_qk, Vt_qk = svd(W_qk)              # joint decomposition
    U_vo, S_vo, Vt_vo = svd(W_vo)
    # Learn low-rank adapter in the joint space
    A_qk, B_qk = init_from_svd(U_qk, S_qk, Vt_qk, r)
    A_vo, B_vo = init_from_svd(U_vo, S_vo, Vt_vo, r)
    return A_qk, B_qk, A_vo, B_vo

def clover_forward(x, W_q, W_k, W_v, W_o, adapters):
    A_qk, B_qk, A_vo, B_vo = adapters
    ΔW_qk = A_qk @ B_qk                         # shared Q-K update
    ΔW_vo = A_vo @ B_vo                           # shared V-O update
    # Split back into individual matrices
    ΔW_q, ΔW_k = split(ΔW_qk)
    ΔW_v, ΔW_o = split(ΔW_vo)
    return attention(x, W_q + ΔW_q, W_k + ΔW_k, W_v + ΔW_v, W_o + ΔW_o)
```

**Evidence:** Authors report validation on SDXL (image generation), LLaMA-Vision (multimodal), and Whisper (speech), with average gains over LoRA (+7.6%), DoRA (+5.5%), and PiSSA (+0.7%) in their setup. The cross-layer coupling claim is plausible because Q-K and V-O are functionally paired. This result is strong but still from a single research group.

**Grade:** PE+BL+BF=3.5 (beats LoRA and DoRA significantly; validated across 3 modalities)

*Implications:* CLOVER tells us that attention matrices within a head are not independent parameters -- they share a functional subspace that joint SVD captures. The Q-K pairing makes geometric sense: together they define the attention pattern, so their weight updates should be coordinated. Same for V-O: they define the value extraction and output projection pipeline. This is the strongest evidence yet that SVD-based methods benefit from respecting the *functional architecture* of attention, not just treating each weight matrix in isolation.

---

## 32. PSOFT -- Principal Subspace Orthogonal Fine-Tuning

**Paper:** [Wu et al. 2026](https://arxiv.org/abs/2505.11235)
**Saved:** [docs/psoft_principal_subspace_oft.md](docs/psoft_principal_subspace_oft.md)

**Hypothesis:** Combine PiSSA's SVD initialization with OFT's orthogonal constraint. After extracting the principal subspace via SVD, learn a Cayley rotation $R$ that operates *within* the frozen $U, V$ subspace. This is "OFT in SVD coordinates" -- preserving pairwise angles (OFT's insight) while working in the model's natural basis (PiSSA's insight). The rotation $R$ acts on the principal singular vectors, keeping the subspace orientation while rotating within it.

```py
# ── PSOFT intervention ────────────
def psoft_init(W, r):
    U, S, Vt = svd(W)
    U_r, S_r, V_r = U[:, :r], S[:r], Vt[:r]     # principal subspace (frozen)
    K = zeros(r, r)                               # skew-symmetric rotation params
    return U_r, S_r, V_r, K

def psoft_forward(x, U_r, S_r, V_r, K, W_res):
    R = cayley(K)                                 # R ∈ O(r), Cayley transform
    # Rotate within the principal subspace
    W_adapted = U_r @ R @ diag(S_r) @ V_r        # rotated principal component
    return (W_adapted + W_res) @ x                # + frozen residual
```

**Evidence:** Authors report ~80% memory reduction vs OFT and broad evaluation across 35 NLP/CV tasks, while keeping performance competitive. This is a direct synthesis of PiSSA-style SVD initialization plus OFT-style Cayley-constrained rotations.

**Grade:** PE+BL+DE=3.5 (memory-efficient, faster convergence from SVD init, beats LoRA on 35 tasks)

*Implications:* PSOFT combines two ideas that actually work: SVD tells you where to intervene (principal subspace), orthogonality constrains how (rotations that preserve structure). The method inherits benefits from both. This suggests the best adapters respect both the model's eigenbasis and the geometry of transformations within it.

---

## 33. ReFT -- Representation Fine-Tuning

**Paper:** [Wu, Arora, Wang et al. 2024](https://arxiv.org/abs/2404.03592)
**Code:** [github.com/stanfordnlp/pyreft](https://github.com/stanfordnlp/pyreft)
**Saved:** [docs/reft_representation_finetuning.md](docs/reft_representation_finetuning.md)

**Hypothesis:** Adaptation should target *representations* (activations), not weights. Instead of modifying $W$, modify the hidden state $h$ at specific layers and positions via learned interventions. The model's weights are already fine; we just need to redirect its intermediate computations. This is the "activation steering" hypothesis taken to its limit: learn a linear intervention on hidden states at specific token positions.

```py
# ── ReFT intervention ─────────────
def reft_forward(model, x, interventions):
    # interventions: list of (layer, position, R, b) tuples
    for layer_idx, pos, R, b in interventions:
        h = model.hidden_states[layer_idx][:, pos]  # h ∈ ℝ^d at specific position
        # Low-rank linear subspace intervention (LoReFT)
        h_proj = R @ h                               # project onto learned subspace
        h_int = h_proj + b                            # shift in subspace
        h_new = h + R.T @ (h_int - R @ h)            # apply intervention
        model.hidden_states[layer_idx][:, pos] = h_new
    return model(x)
```

The key: instead of $W' = W + \Delta W$, apply $h' = h + R^\top (R h + b - R h)$ at specific (layer, position) pairs. The intervention is a learned affine transformation in a low-rank subspace of the hidden state.

**Evidence:** Authors report 15-65x parameter savings vs LoRA by intervening on hidden states at selected layer-position sites. Reported gains cover instruction following, commonsense reasoning, and NLU tasks in their benchmark suite. Since this is activation-space adaptation, comparisons with weight-space adapters are informative but not perfectly apples-to-apples.

**Grade:** PE+BL=2 (15-65x more parameter-efficient than LoRA, beats LoRA on multiple benchmarks, distinct paradigm)

*Implications:* ReFT challenges the entire "adapter as weight hypothesis" framework by showing that *activation interventions* can be more efficient than *weight modifications*. If ReFT works, it suggests that the model's computation is already nearly correct -- you just need to nudge specific intermediate representations. The layer/position specificity means the model's computation isn't uniformly important; a few critical "bottleneck" representations carry most of the task-relevant signal. This connects to the mechanistic interpretability finding that specific circuit components (induction heads, factual recall sites) are localized and position-dependent.

---

## Themes: What the Evidence Tells Us

Looking across all 33 methods, a coherent tentative story appears once benchmark noise is reduced. Many successful adapters make geometric bets: first choose coordinates that align with pretrained structure, then constrain updates so they do not destroy that structure, then control update strength explicitly.

A quick source-level pass over the paper texts helps anchor this interpretation. Direct "we hypothesize" style statements cluster into the same buckets used here: LoRA and RandLoRA for low-rank sufficiency limits; OFT and ETHER for orthogonality and preserved angular structure; DoRA and DeLoRA for direction-strength decoupling; IA3 for scaling-only adaptation; SHiRA and C3A for high-rank and structural alternatives; AntiPaSTO for SVD-coordinate intervention and OOD transfer. That clustering is not perfect, but it is strong enough to justify organizing the literature by theme rather than by year.
<!-- TODO IS this meta statement needed for the audience or directed at them? we don't actually do clustering either -->


The strongest recurring signal is *basis choice*. SVD-aware methods such as PiSSA, SSVD, CLOVER, and PSOFT often beat random-basis baselines under similar budgets in reported setups. In practical terms, initializing in the model's singular-vector basis reduces the search problem. The optimizer starts in a subspace the model already uses. This is not proof that SVD is uniquely correct, but it is stronger evidence than a single benchmark win.

*Orthogonal* methods add the next piece. OFT and BOFT show that bounded rotations can preserve useful behavior while still adapting to new tasks. The Cayley parameterization appears across OFT, SSVD, PSOFT, and AntiPaSTO because it keeps rotations orthogonal without repeated projection steps. Pure orthogonality can be too rigid when tasks need gain changes, so methods that pair rotations with magnitude control tend to perform better.

That leads to the *direction-versus-strength* split. DoRA, DeLoRA, ROAD, and AntiPaSTO all separate where to move in weight space from how far to move. In runs that report careful ablations, this split often improves stability and sometimes final accuracy. Whether this is a deep property of transformer computation or mainly an optimization advantage is still open.

A parallel thread is *gain control*. IA3, VeRA, and LN tuning show that a lot of adaptation comes from rescaling existing features instead of inventing new ones. This explains why tiny parameter budgets can work well on many tasks. It also clarifies where they fail: when tasks require genuinely new feature combinations, scaling-only methods plateau.

The *rank* debate looks secondary once basis is accounted for. Full-rank updates can help on harder tasks, as RandLoRA and C3A suggest, but PiSSA and SVFT show that a good low-rank subspace can beat a poorly chosen full-rank update. In practice, "which subspace" matters more than "how many free directions".

Finally, methods that respect *functional structure* are promising but early. CLOVER's joint treatment of Q-K and V-O pairs outperforms per-matrix updates, and ReFT shows that targeted activation interventions can be far more parameter-efficient than weight updates. Both suggest that treating transformers as computation graphs, not bags of matrices, is a productive direction.

### Overall picture

Across methods, the same pattern keeps repeating: adapters work best when they preserve pretrained structure and then move within it in controlled ways. SVD-aware coordinates identify high-signal directions, near-orthogonal transforms protect useful geometry, and explicit strength controls prevent overwriting. This is currently the strongest empirical pattern in the catalog. It does not settle causality by itself, but it narrows the search space and yields concrete, falsifiable predictions for mechanistic work.


<!-- TODO kind of weak I'd rather make a prediciton, or state strength of evidence, or if it changed my mind here -->
