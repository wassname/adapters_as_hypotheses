# Adapter Pseudocode (pseudopy)

Companion to [adapters_vargdown.argdown](adapters_vargdown.argdown).
Each adapter's intervention as pseudopy: minimal, self-documenting, shapes in comments.

---

## 1. LoRA -- baseline

```
# ── LoRA: low-rank weight update ──
ΔW ← B @ A                          # ΔW ∈ ℝ^{m×n}, rank r
y ← (W + (α/r) · ΔW) @ x           # W frozen; A, B learned
```

## 2. OFT -- orthogonal rotation

```
# ── OFT: Cayley rotation of weight rows ──
Q_skew ← skew(Q)                    # Q_skew ∈ ℝ^{b×k×k}, antisymmetric
R ← (I + Q_skew)(I - Q_skew)⁻¹     # Cayley map → R ∈ O(k)
R_full ← blkdiag(R₁, ..., R_b)     # R_full ∈ O(d)
y ← (W @ Rᵀ) @ x                   # rotate rows, preserve angles
```

## 3. VeRA -- random projection + scaling

```
# ── VeRA: shared random matrices, per-layer scaling ──
# A ∈ ℝ^{r×n}, B ∈ ℝ^{m×r}: frozen random, shared ALL layers
# λ_d ∈ ℝ^r, λ_b ∈ ℝ^r: learned per-layer
ΔW ← (λ_b ⊙ B) @ (λ_d ⊙ A)       # ΔW ∈ ℝ^{m×n}
y ← (W + ΔW) @ x
```

## 4. DoRA -- magnitude/direction decomposition

```
# ── DoRA: decouple ‖W‖ from W/‖W‖ ──
# m ∈ ℝ^{m}: learned magnitude per output neuron
V ← W + α · B @ A                   # updated direction
V̂ ← V / ‖V‖_col                     # unit direction, .detach()
y ← (m · V̂) @ x                     # magnitude × direction
```

## 5. DeLoRA -- bounded deviation

```
# ── DeLoRA: normalize rank-1 components, scale by λ ──
# A ∈ ℝ^{r×n}, B ∈ ℝ^{m×r}: learned
# λ ∈ ℝ^r: learned per-component strength
â_i ← ‖aᵢ‖, b̂ⱼ ← ‖bⱼ‖             # per-component norms
s ← (λ/r) / (â · b̂)                 # normalize each outer product
ΔW ← B · diag(s) · A               # direction normalized, strength via λ
y ← W(x) + ΔW @ x
```

## 6. PiSSA -- SVD initialization

```
# ── PiSSA: init adapters from top-r SVD of W ──
U, Σ, Vᵀ ← svd(W)                  # W ∈ ℝ^{m×n}
A ← U_{:,:r} · √Σ_{:r}             # A ∈ ℝ^{m×r}, principal left
B ← √Σ_{:r} · Vᵀ_{:r,:}           # B ∈ ℝ^{r×n}, principal right
W_res ← U_{:,r:} · Σ_{r:} · Vᵀ_{r:,:}  # residual, frozen
# ── forward (identical to LoRA) ──
y ← (W_res + A @ B) @ x
```

## 7. SVFT -- SVD coefficient tuning

```
# ── SVFT: learn sparse coefficients over W's own singular vectors ──
U, Σ, Vᵀ ← svd(W)                  # frozen
# select k sparse (i,j) pairs; c ∈ ℝ^k learned
ΔW ← Σₜ cₜ · uᵢ vⱼᵀ               # sparse combo of outer products
y ← (W + ΔW) @ x
```

## 8. SSVD -- asymmetric SVD rotation

```
# ── SSVD: rotate right singular vectors, preserve left ──
U, Σ, Vᵀ ← svd(W)                  # all frozen
K ← skew(θ)                         # K ∈ ℝ^{k×k}, learned antisymmetric
G ← (I - K)(I + K)⁻¹               # Cayley → G ∈ O(k)
Σ̂ ← Σ; Σ̂_{:k} += ΔΣ               # shift top-k singular values (learned)
V̂ᵀ ← Vᵀ; V̂ᵀ_{:k} ← G @ Vᵀ_{:k}   # rotate input-space vectors only
y ← U · diag(Σ̂) · V̂ᵀ @ x
# params: k(k-1)/2 + k
```

$$W' = U \, (\Sigma + \Delta\Sigma) \, G_k \, V^\top$$

## 9. IA3 -- activation scaling

```
# ── IA3: element-wise scaling of activations ──
# λ ∈ ℝ^d: learned, init=1
y ← W @ (x ⊙ λ)                    # FFN: scale input channels
y ← (W @ x) ⊙ λ                    # attn K,V: scale output channels
```

## 10. ROAD -- rotary adaptation

```
# ── ROAD: 2D rotation + magnitude per activation pair ──
# θ ∈ ℝ^{d/2}: learned rotation angles
# α ∈ ℝ^{d/2}: learned magnitudes
for i in range(d//2):
    x̂[2i]   ← α_i · (cos θ_i · x[2i] - sin θ_i · x[2i+1])
    x̂[2i+1] ← α_i · (sin θ_i · x[2i] + cos θ_i · x[2i+1])
y ← W @ x̂                           # α=1, θ=0 → identity
```

## 11. AntiPaSTO -- SVD Cayley steering

```
# ── AntiPaSTO: Cayley rotation of SVD singular vectors ──
U, Σ, Vᵀ ← svd(W)                  # frozen; computed once at init
K ← skew(θ)                         # θ ∈ ℝ^{k(k-1)/2}, learned
R ← (I - K)(I + K)⁻¹               # Cayley → R ∈ O(k)
# ── rotate both U and V by same R ──
Û ← U; Û_{:,:k} ← U_{:,:k} @ R    # rotate output-space
V̂ᵀ ← Vᵀ; V̂ᵀ_{:k,:} ← R @ Vᵀ_{:k,:}  # rotate input-space
W' ← Û · diag(Σ) · V̂ᵀ             # reconstruct
y ← W' @ x
# antiparallel: negate θ → opposite behavioral direction
```

## 12. AdaLoRA -- adaptive SVD rank

```
# ── AdaLoRA: SVD-parameterized with importance pruning ──
ΔW ← P · diag(Λ) · Q               # P ∈ ℝ^{m×r}, Q ∈ ℝ^{r×n}
# importance score per singular value:
s_i ← |Λ_i| + β · ‖pᵢ‖ · ‖qᵢ‖     # sensitivity-weighted
# prune: zero out components with lowest s_i per budget
mask ← topk(s, budget)
ΔW ← P · diag(Λ ⊙ mask) · Q
y ← (W + ΔW) @ x
```

## 13. BOFT -- butterfly orthogonal

```
# ── BOFT: butterfly-factorized O(d log d) orthogonal ──
# m butterfly factors B₁, ..., B_m each ∈ ℝ^{n/b × b × b}
# each Bⱼ is block-diagonal of small orthogonal matrices
R ← B_m @ ... @ B₂ @ B₁             # R ∈ O(d), O(d log d) params
y ← (W @ Rᵀ) @ x
```

## 14. GOFT -- Givens rotations

```
# ── GOFT: compose d(d-1)/2 planar rotations ──
R ← I
for (i,j,θ) in givens_pairs:         # each θ learned
    G ← I; G[i,i] ← cos θ; G[i,j] ← -sin θ
           G[j,i] ← sin θ; G[j,j] ← cos θ
    R ← G @ R                        # compose
y ← (W @ Rᵀ) @ x                    # O(d) params
```

## 15. HRA -- Householder reflection

```
# ── HRA: chain of r Householder reflections = rank-r ∩ O(d) ──
R ← I
for i in range(r):
    vᵢ ← learned                     # v ∈ ℝ^d
    Hᵢ ← I - 2 · vᵢ vᵢᵀ / ‖vᵢ‖²   # Householder reflection
    R ← Hᵢ @ R
y ← (W @ Rᵀ) @ x
# bridges: rank-r perturbation ≡ r Householder reflections
```

## 16. RandLoRA -- full-rank via random bases

```
# ── RandLoRA: sum of scaled random rank-r bases ──
# A_i, B_i: frozen random matrices
# d_i, b_i ∈ ℝ^r: learned per-component scaling
ΔW ← Σᵢ (b_i ⊙ Bᵢ) @ (d_i ⊙ Aᵢ)  # full rank possible
y ← (W + ΔW) @ x
```

## 17. FourierFT -- spectral coefficients

```
# ── FourierFT: sparse Fourier coefficients ──
# select k frequency indices; c ∈ ℝ^k learned
ΔW ← iFFT2(scatter(c, indices, shape=(m,n)))
y ← (W + ΔW) @ x
```

## 18. CLOVER -- joint SVD across attention pairs

```
# ── CLOVER: joint SVD over Q-K and V-O pairs per head ──
# ── init: decompose paired attention matrices ──
W_QK ← W_Q @ W_K.T                  # W_QK ∈ ℝ^{d×d}, combined Q-K
U_qk, S_qk, V_qk ← svd(W_QK)       # per-head SVD
W_VO ← W_V @ W_O.T                  # W_VO ∈ ℝ^{d×d}, combined V-O
U_vo, S_vo, V_vo ← svd(W_VO)

# ── forward: only S is learned, U/V frozen ──
# rewrite Q,K from shared orthogonal basis:
W_Q' ← U_qk @ diag(√S_qk)          # Q uses left singular vectors
W_K' ← V_qk @ diag(√S_qk)          # K uses right singular vectors
# fine-tune: learn ΔS_qk, ΔS_vo (full-rank update via all directions)
S_qk' ← S_qk + ΔS_qk               # learned shifts
y ← attn(W_Q' @ x, W_K' @ x, ...) 
# params: 2 × rank scalars per head (pruning: zero small S entries)
```

## 19. PSOFT -- principal subspace + Cayley

```
# ── PSOFT: PiSSA init + OFT rotation in principal subspace ──
U, Σ, Vᵀ ← svd(W)
# extract top-k subspace
# learn Cayley rotation R within that subspace only
# 80% memory reduction vs full OFT
R ← cayley(K)                        # K ∈ ℝ^{k×k}
W' ← U_{:,:k} @ R @ diag(Σ_{:k}) @ Vᵀ_{:k,:} + W_res
```

## 20. ReFT -- activation intervention

```
# ── ReFT: intervene on hidden states at (layer, position) ──
# R ∈ ℝ^{r×d}: learned rotation (low-rank subspace)
# b ∈ ℝ^r: learned bias in subspace
h ← model.layer[l].output[pos]       # hidden state at site
h_proj ← R @ h                       # project to subspace
h_proj ← h_proj + b                  # intervene
h ← h + Rᵀ @ (h_proj - R @ h)       # write back (preserve complement)
```

---

*Notation: `⊙` element-wise, `@` matmul, `←` assignment, `Σₜ` summation, `‖·‖` norm, `blkdiag` block diagonal, `skew` maps vector to skew-symmetric matrix, `cayley(K) = (I-K)(I+K)⁻¹`.*
