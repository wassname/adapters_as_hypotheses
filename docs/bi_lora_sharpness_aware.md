Title: Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models

URL Source: https://arxiv.org/html/2508.19564

Markdown Content:
Yuhang Liu 1,, Tao Li 1,∗, Zhehao Huang 1, Zuopeng Yang 1, Xiaolin Huang 1,2,

1 Institute of Image Processing and Pattern Recognition, School of Automation and Intelligent Sensing,

Shanghai Jiao Tong University

2 MoE Key Laboratory of System Control and Information Processing (Shanghai)

{yuhangliu,li.tao,kinght_h,yzpeng,xiaolinhuang}@sjtu.edu.cn

###### Abstract

Low-Rank Adaptation (LoRA) enables parameter-efficient fine-tuning of large pre-trained models. Yet LoRA can face generalization challenges. One promising way to improve the generalization is Sharpness-Aware Minimization (SAM), which has proven effective for small-scale training scenarios. In this paper, we propose Bi-directional Lo w-R ank A daptation (Bi-LoRA), which introduces an auxiliary adversarial LoRA module. This design explicitly decouples sharpness optimization, handled by the auxiliary module, from task adaptation, performed by the primary module. Such a separation yields two key benefits. First, it transforms SAM’s sequential computation of adversarial perturbation and gradient descent into a parallel form, which roughly halves the time and conquers the main obstacle of applying SAM in LoRA. Second, it provides perturbations from the auxiliary module that do not collapse into the restricted optimization subspace of the primary module, enabling broader sharpness exploration and flatter minima. Bi-LoRA simultaneously achieves both efficiency and effectiveness within a single framework, as validated by extensive experiments across diverse architectures and tasks. Code is available at [https://github.com/CrazyElements/Bi-LoRA](https://github.com/CrazyElements/Bi-LoRA).

## 1 Introduction

The paradigm of pretraining followed by fine-tuning has become the de facto standard in machine learning, demonstrating state-of-the-art performance across various tasks(Devlin et al., [2018](https://arxiv.org/html/2508.19564#bib.bib19 "Bert: pre-training of deep bidirectional transformers for language understanding"); Kolesnikov et al., [2020](https://arxiv.org/html/2508.19564#bib.bib8 "Big transfer (bit): general visual representation learning"); Dosovitskiy et al., [2021](https://arxiv.org/html/2508.19564#bib.bib9 "An image is worth 16x16 words: transformers for image recognition at scale"); Radford et al., [2021](https://arxiv.org/html/2508.19564#bib.bib5 "Learning transferable visual models from natural language supervision")). However, as model sizes continue to grow, full fine-tuning (Full FT) becomes memory-prohibitive in resource-constrained settings. The most successful approach for reducing memory cost in fine-tuning large-scale model is Low-Rank Adaptation (LoRA)(Hu et al., [2022](https://arxiv.org/html/2508.19564#bib.bib32 "LoRA: low-rank adaptation of large language models")), which introduces trainable, task-specific low-rank matrices to model weight updates. LoRA greatly lowers memory requirements by significantly reducing the trainable parameters and has become one of the most popular solutions for fine-tuning and deploying large models due to its simplicity and performance comparable to Full FT.

Despite LoRA’s memory efficiency, it still faces generalization challenges when fine-tuned with limited data. In such scenarios, the risk of overfitting becomes particularly acute and can significantly compromise model performance (Li et al., [2025](https://arxiv.org/html/2508.19564#bib.bib64 "Flat-lora: low-rank adaptation over a flat loss landscape"); Deng et al., [2025](https://arxiv.org/html/2508.19564#bib.bib65 "EFlat-lora: efficiently seeking flat minima for better generalization in fine-tuning large language models and beyond"); Lin et al., [2024](https://arxiv.org/html/2508.19564#bib.bib66 "LoRA dropout as a sparsity regularizer for overfitting control"); Li et al., [2024b](https://arxiv.org/html/2508.19564#bib.bib67 "LoRASC: expressive and generalizable low-rank adaptation for large models via slow cascaded learning")).

Motivated by the connection between flatness of the loss landscape and generalization, one promising direction for improving generalization is to seek flat minima(Hochreiter and Schmidhuber, [1994](https://arxiv.org/html/2508.19564#bib.bib20 "Simplifying neural nets by discovering flat minima"); [1997](https://arxiv.org/html/2508.19564#bib.bib21 "Flat minima")). Sharpness-Aware Minimization (SAM)(Foret et al., [2021](https://arxiv.org/html/2508.19564#bib.bib27 "Sharpness-aware minimization for efficiently improving generalization")) is a widely used technique that enhances generalization by formulating optimization as a min-max problem, effectively minimizing the worst-case loss within a local neighborhood. It has achieved state-of-the-art performance in small-scale training scenarios(Chen et al., [2022](https://arxiv.org/html/2508.19564#bib.bib2 "When vision transformers outperform resnets without pre-training or strong data augmentations"); Zhuang et al., [2022](https://arxiv.org/html/2508.19564#bib.bib11 "Surrogate gap minimization improves sharpness-aware training")). However, SAM requires calculating an adversarial perturbation of model parameters at each training step, incurring additional memory overhead of a copy of the model weights and doubling the training time. When applied to large-scale models, such extra memory cost and computation becomes particularly pronounced.

![Image 1: Refer to caption](https://arxiv.org/html/2508.19564v2/_figures/bi_lora_framework.png)

Figure 1:  Overview of our proposed Bi-LoRA. During fine-tuning (Left), Bi-LoRA utilizes two opposing LoRA modules: the primary module (B_{1}A_{1}) is optimized by conventional gradient descent for task-specific adaptation, while the auxiliary module (B_{2}A_{2}) performs gradient ascent for sharpness optimization. These two modules are decoupled and simultaneously optimized. During inference (Right), only (B_{1}A_{1}) is retained and merged with the pretrained weights.

Thus, a natural approach to finding flat minima when fine-tuning large models, without sacrificing memory efficiency, is to apply SAM to the LoRA parameters, referred to as LoRA-SAM(Li et al., [2024a](https://arxiv.org/html/2508.19564#bib.bib53 "Implicit regularization of sharpness-aware minimization for scale-invariant problems")). However, this straightforward integration still suffers from the doubled training cost of SAM. To address this essential problem, we introduce an auxiliary adversarial LoRA module to decouple the sharpness optimization from the task adaptation.This design, Bi-directional Low-Rank Adaptation (Bi-LoRA) shown in Figure[1](https://arxiv.org/html/2508.19564#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"), enables updating both modules in one forward and backward pass, making SAM practical for fine-tuning large-scale models.

Decoupling the two LoRA modules also enables broader exploration of perturbations. During inference, the LoRA parameters are merged into the pretrained weights, so we should care about the sharpness of the full parameter space. In contrast, LoRA-SAM limits adversarial perturbations to a restricted subspace (see Proposition[1](https://arxiv.org/html/2508.19564#Thmproposition1 "Proposition 1 (Perturbation Space of LoRA-SAM). ‣ 2.2 LoRA-SAM ‣ 2 Issues of LoRA-SAM ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models")). Worse, LoRA-SAM collapses quickly into it (see Figure[2(c)](https://arxiv.org/html/2508.19564#S2.F2.sf3 "In Figure 2 ‣ 2.1 Preliminaries ‣ 2 Issues of LoRA-SAM ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models")), thereby optimizing sharpness only there. Empirically, Bi-LoRA alleviates this issue: its decoupled auxiliary perturbation space converges more slowly than the primary optimization space, enabling exploring perturbation beyond the restricted subspace to capture sharpness and promote flatter minima. Bi-LoRA maintains LoRA’s efficiency while improving generalization, making it a strong alternative for fine-tuning large-scale models.

Our contributions can be summarized as follows:

*   •
We propose Bi-LoRA, which introduces an auxiliary LoRA module to model SAM’s adversarial weight perturbation, decoupling it from LoRA optimization. This design enables simultaneous optimization of SAM’s two steps with minor additional memory costs.

*   •
We point out that directly applying SAM to LoRA parameters can only optimize the sharpness within the restricted subspace, thereby limiting its potential to improve generalization. Bi-LoRA broadens the sharpness exploration: the auxiliary perturbation module converges more slowly than the primary one, enabling exploration beyond the restricted subspace.

*   •
Extensive experiments across a wide range of fine-tuning tasks, including natural language understanding, mathematics, code generation, chat, instruction following, and Text-to-Image generation, demonstrate that Bi-LoRA achieves superior generalization performance.

## 2 Issues of LoRA-SAM

In this section, we formalize LoRA-SAM and analyze its subspace collapse issue during training. Related work on LoRA and SAM is summarized in Appendix[E](https://arxiv.org/html/2508.19564#A5 "Appendix E Related Work ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").

### 2.1 Preliminaries

![Image 2: Refer to caption](https://arxiv.org/html/2508.19564v2/_figures/t5_cola_norm_layer_60.png)

(a) Norm of different components

![Image 3: Refer to caption](https://arxiv.org/html/2508.19564v2/_figures/t5_cola_norm_ratio_layer_60.png)

(b) Norm ratio

![Image 4: Refer to caption](https://arxiv.org/html/2508.19564v2/_figures/lora_trajectory_comparison_cola.png)

(c) Cosine similarity to final

Figure 2:  Training statistics for LoRA-SAM during fine-tuning. (a) and (b): Frobenius norms of different terms and the ratio of the Frobenius norms of (B\epsilon_{A}+\epsilon_{B}A) to that of (\epsilon_{B}\epsilon_{A}) in Eqn.([5](https://arxiv.org/html/2508.19564#S2.E5 "In 2.2 LoRA-SAM ‣ 2 Issues of LoRA-SAM ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models")). We monitor a fixed LoRA module and more similar experiments can be found in Appendix[T](https://arxiv.org/html/2508.19564#A20 "Appendix T Norm and Ratio Analysis Across Layers ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"). It could be observed that the norm of the third term (\epsilon_{B}\epsilon_{A}) is several orders of magnitude (>10^{4}) smaller than that of the first two terms, making it negligible. Note that \|B\epsilon_{A}\|_{F} is initially zero since B are initialized to zero by default(Hu et al., [2022](https://arxiv.org/html/2508.19564#bib.bib32 "LoRA: low-rank adaptation of large language models")). (c): Cosine similarity between the final LoRA parameters and those during fine-tuning. The LoRA parameters converge rapidly during fine-tuning. The experiments are conducted on CoLA with the T5-base model.

Given a pre-trained weight matrix W_{0}\in\mathbb{R}^{m\times n}, LoRA utilizes low-rank matrices B and A to model the weight change \Delta W during the fine-tuning,

\displaystyle W=W_{0}+\Delta W\approx W_{0}+BA,(1)

where B\in\mathbb{R}^{m\times r} and A\in\mathbb{R}^{r\times n} with r\ll\min\{m,n\}. Here we omit the scaling factor s=\alpha/r for the sake of simplicity in the equation, as it can be easily incorporated into B and A.

During gradient back-propagation, the loss gradient w.r.t. B and A is computed using the chain rule:

\frac{\partial\mathcal{L}}{\partial B}=(\nabla_{W}\mathcal{L}){A}^{\top},\quad\frac{\partial\mathcal{L}}{\partial A}={B}^{\top}(\nabla_{W}\mathcal{L}),(2)

where \mathcal{L} is the loss objective to be minimized.

### 2.2 LoRA-SAM

For optimizing the sharpness while keeping memory efficiency, a natural idea is to apply SAM over the LoRA parameters (LoRA-SAM) which formulates the following optimization target:

\displaystyle\min_{B,A}~~\max_{\|(\mathbf{\epsilon}_{B},\mathbf{\epsilon}_{A})\|\leq\rho}~~\mathcal{L}\left({W}_{0}+({B}+\mathbf{\epsilon}_{B})({A}+\mathbf{\epsilon}_{A})\right),\,(3)

where \epsilon_{B}\in\mathbb{R}^{m\times r},\epsilon_{A}\in\mathbb{R}^{r\times n} are the adversarial weight perturbations over low-rank matrices, \|(\epsilon_{B},\epsilon_{A})\| denotes the norm of weight perturbations (a typical setting is the \ell_{2}-norm), and \rho is the neighborhood radius.

To efficiently solve the inner maximization problem in Eqn.([3](https://arxiv.org/html/2508.19564#S2.E3 "In 2.2 LoRA-SAM ‣ 2 Issues of LoRA-SAM ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models")), Foret et al. ([2021](https://arxiv.org/html/2508.19564#bib.bib27 "Sharpness-aware minimization for efficiently improving generalization")) employ first-order Taylor expansion for approximation, and the resulting perturbations are calculated as follows:

\displaystyle\epsilon_{B}\displaystyle=\rho\cdot{\frac{\partial\mathcal{L}}{\partial B}}/{F_{\text{total}}},\quad\epsilon_{A}=\rho\cdot{\frac{\partial\mathcal{L}}{\partial A}}/{F_{\text{total}}},\quad F_{\text{total}}=\sqrt{\left\|\frac{\partial\mathcal{L}}{\partial B}\right\|_{F}^{2}+\left\|\frac{\partial\mathcal{L}}{\partial A}\right\|_{F}^{2}}\,,(4)

where \|\cdot\|_{F} denotes the Frobenius norm. Then we can rewrite the perturbation given by Eqn.([3](https://arxiv.org/html/2508.19564#S2.E3 "In 2.2 LoRA-SAM ‣ 2 Issues of LoRA-SAM ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models")) as

\displaystyle\epsilon_{W}\displaystyle=B\epsilon_{A}+\epsilon_{B}A+\epsilon_{B}\epsilon_{A}.\,(5)

Since both \epsilon_{A} and \epsilon_{B} are several orders of magnitude smaller than the original matrices A and B, the cross term \epsilon_{B}\epsilon_{A} is negligible, as evidenced by Figures[2(a)](https://arxiv.org/html/2508.19564#S2.F2.sf1 "In Figure 2 ‣ 2.1 Preliminaries ‣ 2 Issues of LoRA-SAM ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models") and [2(b)](https://arxiv.org/html/2508.19564#S2.F2.sf2 "In Figure 2 ‣ 2.1 Preliminaries ‣ 2 Issues of LoRA-SAM ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"), which report the magnitudes of these terms. Therefore, Eqn.([5](https://arxiv.org/html/2508.19564#S2.E5 "In 2.2 LoRA-SAM ‣ 2 Issues of LoRA-SAM ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models")) can be approximately simplified as:

\displaystyle\epsilon_{W}\approx c\left[BB^{\top}(\nabla_{W}\mathcal{L})+(\nabla_{W}\mathcal{L}){A}^{\top}A\right],\,(6)

where c=\rho/\sqrt{\|\left(\nabla_{W}\mathcal{L}\right){A}^{\top}\|^{2}_{F}+\|{B}^{\top}(\nabla_{W}\mathcal{L})\|^{2}_{F}}. Now one can observe that the perturbation of LoRA-SAM is confined to a restricted subspace, as demonstrated below:

###### Proposition 1(Perturbation Space of LoRA-SAM).

The effective weight perturbation in LoRA-SAM can be decomposed into two terms: BB^{\top}(\nabla_{W}\mathcal{L}) and (\nabla_{W}\mathcal{L})A^{\top}A. The column space of the first term is given by \text{Col}(B), while the row space of the second term is given by \text{Row}(A).

The space for adversarial weight perturbation in LoRA-SAM is primarily dominated by \text{Col}(B) and \text{Row}(A), from which it follows that LoRA-SAM only cares about the sharpness within the subspace defined by B and A, failing to capture the sharpness in a broader space.

Moreover,B and A converge rapidly during training, as shown in Figure[2(c)](https://arxiv.org/html/2508.19564#S2.F2.sf3 "In Figure 2 ‣ 2.1 Preliminaries ‣ 2 Issues of LoRA-SAM ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"). This leads to the fast convergence of the column and row spaces\text{Col}(B)and\text{Row}(A), further restricting the subspace for sharpness optimization and hindering the effectiveness of LoRA-SAM, potentially leading to suboptimal performance. Appendix[U](https://arxiv.org/html/2508.19564#A21 "Appendix U Restricting perturbation in the optimization subspace limits LoRA-SAM ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models") further corroborates this subspace collapse phenomenon.

In Figure[4](https://arxiv.org/html/2508.19564#S3.F4 "Figure 4 ‣ 3 Bi-LoRA: Bi-directional Low-Rank Adaptation ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"), we compare the loss landscape flatness of different methods. We observe that LoRA-SAM achieves the flattest loss landscape within the LoRA parameter space as in Figure[4(a)](https://arxiv.org/html/2508.19564#S3.F4.sf1 "In Figure 4 ‣ 3 Bi-LoRA: Bi-directional Low-Rank Adaptation ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"), reflecting its original sharpness-minimization objective. In contrast, Figure[4(b)](https://arxiv.org/html/2508.19564#S3.F4.sf2 "In Figure 4 ‣ 3 Bi-LoRA: Bi-directional Low-Rank Adaptation ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models") shows that while LoRA-SAM yields a flatter landscape than LoRA in the full space, the loss still rises sharply with perturbation magnitude. This indicates the limitation of applying perturbation in a restricted subspace, highlighting the importance of considering sharpness over a broader space beyond LoRA subspace.

## 3 Bi-LoRA: Bi-directional Low-Rank Adaptation

In this section, we introduce Bi-LoRA, a novel LoRA variant, which contains an auxiliary module beyond the primary module in regular LoRA. The aim of the auxiliary module is to decouple the sharpness optimization from task adaptation, which makes the training efficient and enhances the generalization improvement. Specifically, the proposed Bi-LoRA takes the following formulation,

W=W_{0}+B_{1}A_{1}+B_{2}A_{2},\,(7)

where the first LoRA module (A_{1},B_{1}) serves as the primary module responsible for task-specific adaptation, similar to standard LoRA, while the second module (A_{2},B_{2}) acts as the auxiliary LoRA module for modeling adversarial perturbation in SAM. With these two modules, the Bi-LoRA’s optimization objective is given below,

\displaystyle\min_{B_{1},A_{1}}\max_{\|B_{2}A_{2}\|_{F}\leq\rho}\mathcal{L}\left(W_{0}+B_{1}A_{1}+B_{2}A_{2}\right),\,(8)

where \rho>0 is the neighborhood radius that controls the magnitude of perturbations as in the original SAM. After training, we discard the auxiliary modules (A_{2},B_{2}), as they serve solely to optimize sharpness during training and guide the primary module (A_{1},B_{1}) towards a flat region. Consequently, the adapted weights are reduced to the primary module, i.e., W=W_{0}+B_{1}A_{1}, preserving the original LoRA structure and ensuring no additional computational overhead during inference.

In LoRA-SAM, the adversarial perturbation is calculated as Eqn.([4](https://arxiv.org/html/2508.19564#S2.E4 "In 2.2 LoRA-SAM ‣ 2 Issues of LoRA-SAM ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models")), where the backpropagation on (\epsilon_{A},\epsilon_{B}) and on (A,B) must be carried out sequentially. Now in Bi-LoRA, the task adaptation and the perturbation are decoupled, as shown in Eqn.([8](https://arxiv.org/html/2508.19564#S3.E8 "In 3 Bi-LoRA: Bi-directional Low-Rank Adaptation ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models")). Then one can simultaneously perform task adaptation and sharpness-aware minimization _in one gradient step_. Specifically, during each iteration, the primary LoRA module (A_{1},B_{1}) is updated through standard gradient descent, while the auxiliary LoRA module (A_{2},B_{2}) is updated through gradient ascent for sharpness optimization.

The bi-directional gradient update approach embodies the core concept of our Bi-LoRA. Concretely, one update step for Bi-LoRA is formalized as follows:

\displaystyle\left\{\begin{aligned} B_{1}^{k+1}&=B_{1}^{k}-\eta_{1}\left(\nabla_{W}\mathcal{L}\right){A^{k\top}_{1}},\quad A_{1}^{k+1}=A_{1}^{k}-\eta_{1}B_{1}^{k\top}\left(\nabla_{W}\mathcal{L}\right),\,\\
B_{2}^{k+1}&=B_{2}^{k}+\eta_{2}\left(\nabla_{W}\mathcal{L}\right)A_{2}^{k\top},\quad A_{2}^{k+1}=A_{2}^{k}+\eta_{2}B_{2}^{k\top}\left(\nabla_{W}\mathcal{L}\right),\,\end{aligned}\right.(9)

where \nabla_{W}\mathcal{L}=\left.\frac{\partial\mathcal{L}}{\partial W}\right|_{W=W_{0}+B^{k}_{1}A^{k}_{1}+B^{k}_{2}A^{k}_{2}} is the gradient of the loss \mathcal{L} w.r.t. the merged weight W, k denotes the iteration index, and \eta_{1},\eta_{2} are learning rates. We use the same learning rate for both LoRA modules in our experiments, though other choices are possible (see Appendix[Q](https://arxiv.org/html/2508.19564#A17 "Appendix Q Learning Rates for Tuning the Two Modules in Bi-LoRA ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models")).

The independent backpropagation for the two LoRA modules (A_{1},B_{1}) and (A_{2},B_{2}) makes the updates parallelizable. This eliminates the doubled computational cost typically incurred by LoRA-SAM. The following proposition ensures that the adversarial direction induced by (A_{2},B_{2}) can still increase the inner objective, in non-negatively alignment with the SAM’s perturbation direction. The Proof and further discussions are provided in Appendix[F.1](https://arxiv.org/html/2508.19564#A6.SS1 "F.1 Proof and Discussion of Proposition 2: Alignment of Bi-LoRA’s ascent direction with previous full gradient ‣ Appendix F Proofs ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").

###### Proposition 2(Alignment of Bi-LoRA’s ascent direction with previous full gradient).

Let G_{t}=\nabla_{W}\mathcal{L}(W_{0}+B_{1,t}A_{1,t}+\tilde{\epsilon}_{t}) denote the full gradient at step t, with \tilde{\epsilon}_{t}=B_{2,t}A_{2,t}. After one Bi-LoRA update,

\langle G_{t},\tilde{\epsilon}_{t+1}-\tilde{\epsilon}_{t}\rangle\geq 0,

i.e., Bi-LoRA increases the inner objective along the previous SAM’s perturbation direction.

![Image 5: Refer to caption](https://arxiv.org/html/2508.19564v2/_figures/cola_wd0.0_lr5e-4_bilora.png)

Figure 3: Cosine similarity between the main LoRA (task adaptation, red), the auxiliary LoRA (perturbation, blue), and their trajectories (green) during training T5-base on CoLA. The auxiliary converges only in the last 20% of steps, substantially slower and independent of the main.

Decoupling sharpness optimization from task adaptation not only makes the backpropagation parallelizable but also partially alleviates the inconsistency of LoRA-SAM, where perturbations are applied to a restricted subspace (A,B) while inference is determined by the full parameters (see Proposition[1](https://arxiv.org/html/2508.19564#Thmproposition1 "Proposition 1 (Perturbation Space of LoRA-SAM). ‣ 2.2 LoRA-SAM ‣ 2 Issues of LoRA-SAM ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models")). Bi-LoRA eliminates the dependence of adversarial weight perturbations on the primary LoRA module. In Eqn.[8](https://arxiv.org/html/2508.19564#S3.E8 "In 3 Bi-LoRA: Bi-directional Low-Rank Adaptation ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"), we see the column space for perturbations in Bi-LoRA is spanned by \text{Col}(B_{2}), which is independent of the LoRA optimization space, i.e., \text{Col}(B_{1}).

Although Bi-LoRA’s perturbation space \mathrm{Col}(B_{2}) (auxiliary) remains a subspace, it is strictly decoupled from the optimization space \mathrm{Col}(B_{1}) (primary). Moreover, we find that the auxiliary modules converge much more slowly than the primary ones, preserving flexibility for sharpness-aware updates (see Figure[3](https://arxiv.org/html/2508.19564#S3.F3 "Figure 3 ‣ 3 Bi-LoRA: Bi-directional Low-Rank Adaptation ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models") and Appendix[L](https://arxiv.org/html/2508.19564#A12 "Appendix L Convergence Analysis of Main and Auxiliary LoRA Modules ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models")). Therefore, Bi-LoRA enables a more effective capture of sharpness in the full parameter space. And Figure[4](https://arxiv.org/html/2508.19564#S3.F4 "Figure 4 ‣ 3 Bi-LoRA: Bi-directional Low-Rank Adaptation ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models") confirms that Bi-LoRA achieves a significantly flatter loss landscape in the full parameter space, leading to improved generalization.

![Image 6: Refer to caption](https://arxiv.org/html/2508.19564v2/_figures/loss_landscape_1d_lora_space.png)

(a) LoRA parameters

![Image 7: Refer to caption](https://arxiv.org/html/2508.19564v2/_figures/loss_landscape_1d_full_space.png)

(b) Full parameters

Figure 4:  Loss landscape visualization along random “filter-normalized” directions following Li et al. ([2018](https://arxiv.org/html/2508.19564#bib.bib7 "Visualizing the loss landscape of neural nets")), focusing on (a) LoRA parameters and (b) full parameters. It can be observed that while LoRA-SAM attains the greatest flatness within the LoRA subspace, which aligns with its sharpness-minimization objective, it is not the flattest in the full parameter space. This distinction is critical for inference, as LoRA parameters are ultimately merged into the pretrained weights. In contrast, Bi-LoRA delivers a substantially greater improvement on flatness in the full parameter space. All experiments were averaged over five independent runs with T5-base fine-tuned on CoLA.

Numerous studies have discussed SAM’s convergence and generalization(Andriushchenko and Flammarion, [2022](https://arxiv.org/html/2508.19564#bib.bib68 "Towards understanding sharpness-aware minimization"); Dai et al., [2023](https://arxiv.org/html/2508.19564#bib.bib69 "The crucial role of normalization in sharpness-aware minimization"); Khanh et al., [2024](https://arxiv.org/html/2508.19564#bib.bib70 "Fundamental convergence analysis of sharpness-aware minimization")). For Bi-LoRA, the following proposition establishes that the Bi-LoRA objective (Eqn.[8](https://arxiv.org/html/2508.19564#S3.E8 "In 3 Bi-LoRA: Bi-directional Low-Rank Adaptation ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models")) is essentially equivalent to the vanilla LoRA optimization with this regularization term.

###### Proposition 3(Bi-LoRA is a Regularized LoRA).

From the gradient-norm perspective, SAM can be expressed as

\mathcal{L}(W)+\rho\left\|\nabla_{W}\mathcal{L}(W)\right\|_{2}.

If the inner maximization of Bi-LoRA is solved to convergence and attains its optimum, its objective reduces to a low-rank counterpart:

\min_{B_{1},A_{1}}\;\mathcal{L}(W_{0}+B_{1}A_{1})+\rho\left\|\nabla_{W_{0}+B_{1}A_{1}}\mathcal{L}\right\|_{(r)},

where \|\cdot\|_{(r)} denotes the Ky Fan r-norm (the sum of the top-r singular values). Thus, ideally, Bi-LoRA can be viewed as LoRA equipped with an explicit low-rank gradient-norm regularizer.

Proofs and further details are deferred to the Appendix[F.2](https://arxiv.org/html/2508.19564#A6.SS2 "F.2 Proof and discussions of Proposition 3: Equivalence between SAM and Bi-LoRA ‣ Appendix F Proofs ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"). Proposition[3](https://arxiv.org/html/2508.19564#Thmproposition3 "Proposition 3 (Bi-LoRA is a Regularized LoRA). ‣ 3 Bi-LoRA: Bi-directional Low-Rank Adaptation ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models") coincides with the regularization view of SAM(Yue et al., [2023](https://arxiv.org/html/2508.19564#bib.bib71 "Sharpness-aware minimization revisited: weighted sharpness as a regularization term")). Bi-LoRA can be analyzed in the same way, differing only in that its regularizer is a low-rank norm. In practice, however, we perform a single inner step to preserve high efficiency, which is crucial for fine-tuning models. This choice introduces a gap between the practical Bi-LoRA and analyses that require optimality of the inner optimization. Given our application-oriented focus, we do not delve deeply into detailed theoretical discussions.

##### Norm Constraints.

In Eqn.([8](https://arxiv.org/html/2508.19564#S3.E8 "In 3 Bi-LoRA: Bi-directional Low-Rank Adaptation ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models")), we impose a norm constraint to keep the perturbation sufficiently small to avoid disrupting normal model training. Thus, we apply global clipping to the auxiliary module (A_{2},B_{2}) after each update, by constraining their total Frobenius norm, which maintains the weight perturbation within a controlled magnitude. Specifically, suppose that there are N LoRA layers. The i-th auxiliary module is scaled as follows:

\left\{\begin{aligned} B^{(i)}_{2}&\leftarrow\sqrt{\rho/c_{\text{norm}}}\cdot B^{(i)}_{2},\\
A^{(i)}_{2}&\leftarrow\sqrt{{\rho}/{c_{\text{norm}}}}\cdot A^{(i)}_{2},\\
\end{aligned}\quad\text{if}\quad c_{\text{norm}}>\rho.\,\right.(10)

Notice the normalization is applied only if the total Frobenius norm over N auxiliary LoRA modules c_{\text{norm}} exceeds the neighborhood radius \rho, where c_{\text{norm}}=\sqrt{\sum_{j=1}^{N}\|B_{2}^{(j)}A_{2}^{(j)}\|^{2}_{F}}. This ensures the perturbation remains constrained within the preset \rho-norm ball. The overall procedure for Bi-LoRA is summarized in Algorithm [1](https://arxiv.org/html/2508.19564#alg1 "Algorithm 1 ‣ Norm Constraints. ‣ 3 Bi-LoRA: Bi-directional Low-Rank Adaptation ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").

Algorithm 1 Bi-LoRA

1:Input: Initial weight

W_{0}
, learning rates

\eta_{1}
,

\eta_{2}
, radius

\rho
, number of LoRA layers

N

2:Output: Adapted weight

W
for inference

3: Initialize LoRA modules

B^{0}_{1}
,

A^{0}_{1}
,

B^{0}_{2}
,

A^{0}_{2}
;

4:

k\leftarrow 0
;

5:while _not converged_ do

6: Sample mini-batch data

\mathcal{B}
;

7: Apply gradient descent to primary and ascent to auxiliary LoRA modules via Eqn.([9](https://arxiv.org/html/2508.19564#S3.E9 "In 3 Bi-LoRA: Bi-directional Low-Rank Adaptation ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"));

8: Clip the auxiliary modules via Eqn.([10](https://arxiv.org/html/2508.19564#S3.E10 "In Norm Constraints. ‣ 3 Bi-LoRA: Bi-directional Low-Rank Adaptation ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"));

9:

k\leftarrow k+1
;

10:end while

11: Remove the auxiliary LoRA modules

B_{2}^{k},A_{2}^{k}
;

12:return

W^{k}=W_{0}+B^{k}_{1}A^{k}_{1}

## 4 Experiments

In this section, we evaluate Bi-LoRA across diverse models and benchmark tasks. We test Bi-LoRA’s capacities on: (1) Llama 2/3.1 models(Touvron et al., [2023](https://arxiv.org/html/2508.19564#bib.bib44 "Llama 2: open foundation and fine-tuned chat models"); Dubey et al., [2024](https://arxiv.org/html/2508.19564#bib.bib43 "The llama 3 herd of models")) for mathematical reasoning, coding, dialogue, and instruction following; (2) Qwen 2.5-14B(Qwen et al., [2025](https://arxiv.org/html/2508.19564#bib.bib58 "Qwen2.5 technical report")), a larger backbone for instruction-following; and (3) SDXL for text-to-image generation. We further demonstrate that Bi-LoRA can be integrated with existing LoRA variants to provide consistent improvement, and conduct ablation studies to examine its hyperparameter sensitivity. More results on natural language understanding tasks with T5-base are provided in Appendix[I](https://arxiv.org/html/2508.19564#A9 "Appendix I Results on Natural Language Understanding ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").

### 4.1 Baselines

We mainly compare with the following baseline methods:

*   •
Full FT fine-tunes all model parameters.

*   •
LoRA applies low-rank adaptation to all linear modules.

*   •
LoRA-SAM applies sharpness-aware minimization over LoRA parameters.

*   •
Random perturbation baselines: Random Output Perturbation (ROP) perturbs logits y with Gaussian noise \epsilon, i.e., \tilde{y}=y+\epsilon,\ \epsilon\sim\mathcal{N}(0,\rho^{2}I); Random Weight Perturbation (RWP)_full/LoRA perturb either full weights or only LoRA adapters with Gaussian noise.

*   •
SAM variants, including LoRA-oBAR/nBAR(Li et al., [2024a](https://arxiv.org/html/2508.19564#bib.bib53 "Implicit regularization of sharpness-aware minimization for scale-invariant problems")), Flat-LoRA(Li et al., [2025](https://arxiv.org/html/2508.19564#bib.bib64 "Flat-lora: low-rank adaptation over a flat loss landscape")), LoRA-ESAM(Du et al., [2022a](https://arxiv.org/html/2508.19564#bib.bib59 "Efficient sharpness-aware minimization for improved training of neural networks")), LoRA-LookSAM(Liu et al., [2022b](https://arxiv.org/html/2508.19564#bib.bib60 "Towards efficient and scalable sharpness-aware minimization")), S 2 SAM(Ji et al., [2024](https://arxiv.org/html/2508.19564#bib.bib31 "A single-step, sharpness-aware minimization is all you need to achieve efficient and accurate sparse training")), and WSAM(Yue et al., [2023](https://arxiv.org/html/2508.19564#bib.bib71 "Sharpness-aware minimization revisited: weighted sharpness as a regularization term")).

*   •
LoRA variants, including LoRA-GA(Wang et al., [2024](https://arxiv.org/html/2508.19564#bib.bib45 "LoRA-ga: low-rank adaptation with gradient approximation")), PiSSA(Meng et al., [2024](https://arxiv.org/html/2508.19564#bib.bib46 "PiSSA: principal singular values and singular vectors adaptation of large language models")), DoRA(Liu et al., [2024](https://arxiv.org/html/2508.19564#bib.bib54 "Dora: weight-decomposed low-rank adaptation")), HiRA(Huang et al., [2025](https://arxiv.org/html/2508.19564#bib.bib73 "HiRA: parameter-efficient hadamard high-rank adaptation for large language models")), and DeLoRA(Bini et al., [2025](https://arxiv.org/html/2508.19564#bib.bib72 "Decoupling angles and strength in low-rank adaptation")).

To more rigorously evaluate the effectiveness in improving generalization, we adopt a stronger training protocol with larger learning rate than prior works(Wang et al., [2025](https://arxiv.org/html/2508.19564#bib.bib57 "LoRA-pro: are low-rank adapters properly optimized?"); [2024](https://arxiv.org/html/2508.19564#bib.bib45 "LoRA-ga: low-rank adaptation with gradient approximation")). The same protocol is applied across all methods for fairness. Unless otherwise specified, all results are averaged over three independent runs with standard errors. See Appendix[R](https://arxiv.org/html/2508.19564#A18 "Appendix R Empirical Details ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models") for detailed hyperparameter settings and more training protocol.

### 4.2 Results on Llama Models

Setting. We evaluate the performance of Bi-LoRA on Llama 2-7B/3.1-8B across four tasks: mathematical reasoning, code generation, dialogue generation, and instruction following, following Wang et al. ([2024](https://arxiv.org/html/2508.19564#bib.bib45 "LoRA-ga: low-rank adaptation with gradient approximation")); Li et al. ([2025](https://arxiv.org/html/2508.19564#bib.bib64 "Flat-lora: low-rank adaptation over a flat loss landscape")); Ren et al. ([2024](https://arxiv.org/html/2508.19564#bib.bib39 "MELoRA: mini-ensemble low-rank adapters for parameter-efficient fine-tuning")). Llama 2-7B is fine-tuned on the first three tasks, while Llama 3.1-8B is used for instruction following. Each task focuses on a specific capability and uses well-established datasets and metrics for training and evaluation, as detailed below:

*   •
Mathematical reasoning. Our model is fine-tuned on a 100k subset of MetaMathQA and evaluated on GSM8K. Performance is measured by accuracy.

*   •
Code generation. We fine-tune our model on a 100k subset of Code-Feedback and evaluate it on the HumanEval benchmark, using the PASS@1 metric.

*   •
Dialogue generation. We train our model on the WizardLM dataset and evaluate it on MT-Bench. The response quality is assessed with GPT-4, and we report the first-turn score on a 10-point scale.

*   •
Instruction following. Fine-tuned on Cleaned Alpaca(Taori et al., [2023](https://arxiv.org/html/2508.19564#bib.bib42 "Stanford alpaca: an instruction-following llama model")) and evaluated on INSTRUCTEVAL(Chia et al., [2023](https://arxiv.org/html/2508.19564#bib.bib41 "Instructeval: towards holistic evaluation of instruction-tuned large language models")), reporting exact match for MMLU, DROP, and BBH, and PASS@1 for HumanEval.

In our experiments, following Du et al. ([2022a](https://arxiv.org/html/2508.19564#bib.bib59 "Efficient sharpness-aware minimization for improved training of neural networks")), (Liu et al., [2022b](https://arxiv.org/html/2508.19564#bib.bib60 "Towards efficient and scalable sharpness-aware minimization")), LoRA-ESAM perturbs 50% of the parameters on the top-50% sharpness-sensitive data, while LoRA-LookSAM applies SAM perturbation every five steps, and thus their costs are denoted as “\times 1.x”.

Table 1: Results of fine-tuning Llama 2-7B and Llama 3.1-8B on different tasks. “Cost” indicates the gradient steps per training iteration, e.g., one step (Cost \times 1) for Full FT, LoRA, and Bi-LoRA.

Method Cost Llama 2-7B Llama 3.1-8B
GSM8K HumanEval MT-Bench MMLU DROP HEval BBH
Full FT\times 1 59.74±0.69 33.12±0.32 6.16±0.09 64.31±0.31 51.52±0.45 41.45±1.58 44.78±0.33
LoRA\times 1 58.21±0.34 24.75±0.23 5.92±0.10 63.38±0.39 49.82±0.54 43.15±0.93 42.82±0.27
LoRA-SAM\times 2 59.16±0.52 26.59±0.36 5.97±0.08 63.46±0.19 50.94±0.22 44.36±1.13 43.49±0.40
ROP\times 1 59.24±0.70 25.41±0.11 6.05±0.08 63.63±0.20 49.96±0.26 42.27±0.81 43.47±0.29
RWP_full\times 1 59.41±0.59 26.26±0.35 6.01±0.12 63.40±0.14 50.11±0.12 44.31±0.81 43.35±0.18
RWP_LoRA\times 1 58.81±0.27 24.92±0.11 5.80±0.11 63.50±0.01 50.16±0.67 42.68±0.70 44.10±0.25
LoRA-oBAR\times 1 59.26±0.53 26.30±0.33 5.97±0.07 63.62±0.12 49.92±0.48 43.49±0.20 43.44±0.12
LoRA-nBAR\times 1 59.72±0.25 26.50±0.23 6.10±0.06 63.45±0.10 49.80±0.03 45.23±0.20 43.39±0.17
Flat-LoRA\times 1 59.44±0.33 26.67±0.23 5.98±0.05 63.67±0.33 50.44±0.17 44.31±0.73 43.99±0.10
LoRA-ESAM\times 1.x 58.33±0.22 24.84±0.04 5.80±0.13 61.79±1.65 49.31±0.30 42.48±1.33 43.40±0.15
LoRA-LookSAM\times 1.x 58.55±0.28 25.28±0.29 5.94±0.08 63.45±0.29 50.34±0.34 42.48±0.54 43.28±0.20
Bi-LoRA\times 1 60.32±0.30 27.20±0.42 6.26±0.06 63.67±0.15 51.53±0.33 46.12±0.89 43.45±0.31

Results. We begin with Llama 2-7B for math, code, and chat tasks. The results, presented in Table[1](https://arxiv.org/html/2508.19564#S4.T1 "Table 1 ‣ 4.2 Results on Llama Models ‣ 4 Experiments ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models") (left), demonstrate Bi-LoRA’s superior performance. Compared to LoRA, Bi-LoRA achieves improvements of 2.11% accuracy on GSM8K, 2.45% on HumanEval, and 0.34 on MT-Bench. Importantly, these gains are even more pronounced than those achieved with the smaller T5-base model (Appendix[I](https://arxiv.org/html/2508.19564#A9 "Appendix I Results on Natural Language Understanding ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models")), and Bi-LoRA operates at a similar speed as LoRA, underscoring its scalability. Furthermore, Bi-LoRA narrows the gap between LoRA fine-tuning and full fine-tuning, and notably outperforms full fine-tuning on GSM8K and MT-Bench tasks, which are not attainable by competing methods. In contrast, we observe that LoRA-SAM does not consistently improve upon LoRA. Random-perturbation baselines provide only marginal or task-specific gains relative to LoRA-SAM, showing that trivial random noise perturbation offers limited generalization improvement. Unlike efficient-SAM variants that either incur extra cost (LoRA-ESAM, LoRA-LookSAM) or compromise performance on certain tasks (e.g., Flat-LoRA on DROP), Bi-LoRA achieves consistent gains on all benchmarks at a single-step computation.

Next, we turn to the instruction following task with Llama 3.1-8B. As shown in Table[1](https://arxiv.org/html/2508.19564#S4.T1 "Table 1 ‣ 4.2 Results on Llama Models ‣ 4 Experiments ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models") (right), Bi-LoRA outperforms almost all baselines, surpassing LoRA by 0.29% on MMLU, 1.71% on DROP, 2.97% on HumanEval, and 0.63% on BBH. While LoRA-SAM and the two types of baselines show improvements over LoRA, Bi-LoRA achieves more substantial gains, particularly on DROP and HumanEval datasets, while requiring only half the training time. Overall, Bi-LoRA offers a better accuracy and efficiency trade-off than both random perturbation and efficient SAM baselines.

### 4.3 Results on Qwen Model

Table 2: Results of fine-tuning Qwen 2.5-14B on instruction-following tasks. “Cost” indicates the gradient steps per training iteration, e.g., one step (Cost \times 1) for LoRA and Bi-LoRA.

Method Cost MMLU DROP HEval BBH Avg.
Vanilla\times 1 79.28±0.17 51.55±0.76 70.93±1.13 57.47±0.03 64.81
LoRA-SAM\times 2 79.19±0.31 54.54±1.03 70.53±1.81 58.25±0.11 65.63
ROP\times 1 79.22±0.28 52.74±1.42 71.34±0.35 57.84±0.49 65.28
RWP_full\times 1 78.92±0.07 54.75±1.01 71.13±1.47 57.76±0.07 65.64
RWP_LoRA\times 1 79.07±0.22 51.69±0.71 71.34±0.93 57.45±0.22 64.89
LoRA-oBAR\times 1 79.37±0.28 55.70±0.61 69.92±0.20 58.53±0.46 65.88
LoRA-nBAR\times 1 79.34±0.32 55.96±0.45 69.92±0.20 58.67±0.44 65.97
Flat-LoRA\times 1 79.51±0.20 54.90±1.06 71.54±0.20 58.13±0.45 66.02
LoRA-ESAM\times 1.x 79.55±0.03 54.83±0.97 65.04±0.81 58.43±0.32 64.46
LoRA-LookSAM\times 1.x 79.63±0.23 56.35±1.55 67.88±1.08 58.41±0.07 65.57
Bi-LoRA\times 1 79.67±0.05 56.49±0.22 71.34±0.93 58.93±0.05 66.61

To demonstrate Bi-LoRA’s performance across model size and architecture, we next experiment on Qwen 2.5-14B, a distinct and larger backbone compared to LlaMA 2/3.1. We focus on instruction-following tasks. Table[2](https://arxiv.org/html/2508.19564#S4.T2 "Table 2 ‣ 4.3 Results on Qwen Model ‣ 4 Experiments ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models") shows that Bi-LoRA achieves the highest average score, improving over LoRA-SAM by 0.98%. Compared to the strongest alternative (Flat-LoRA) across both random perturbation and efficient SAM variants baselines, Bi-LoRA delivers an additional 0.59% average gain. This indicates its applicability across architectures and scales.

### 4.4 Results on Diffusion Models

Setting. We apply Bi-LoRA to a subject-driven generalization task and finetune SDXL(Podell et al., [2023](https://arxiv.org/html/2508.19564#bib.bib6 "Sdxl: improving latent diffusion models for high-resolution image synthesis")) via Dreambooth(Ruiz et al., [2023](https://arxiv.org/html/2508.19564#bib.bib18 "Dreambooth: fine tuning text-to-image diffusion models for subject-driven generation")) on 3D Icons dataset, which contains 23 square-icon images.

Table 3: CLIP I2T and T2T similarity (%) for LoRA and Bi-LoRA on the 3D Icons dataset.

Method CLIP I2T (\uparrow)CLIP T2T (\uparrow)
LoRA 32.43_{\pm 0.46}42.27_{\pm 2.08}
Bi-LoRA\mathbf{33.14}_{\pm 0.41}\mathbf{46.79}_{\pm 2.37}
![Image 8: Refer to caption](https://arxiv.org/html/2508.19564v2/_figures/sdxl_image_show.png)

Figure 5: Images generated by SDXL fine-tuned with LoRA and Bi-LoRA on the 3D icon datasets, where each column uses the _same_ seed for fair comparisons.

Results. Table[3](https://arxiv.org/html/2508.19564#S4.T3 "Table 3 ‣ 4.4 Results on Diffusion Models ‣ 4 Experiments ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models") reports the average CLIP image‐text (I2T) and text‐text (T2T) similarity scores over six prompt instances of the form “a ToK icon of a <Instance>, in the style of TOK”, each evaluated across 200 runs. Compared with LoRA, Bi-LoRA improves the average I2T and T2T similarities by 0.71 and 4.52, respectively, demonstrating its stronger personalization capability. As shown in Figure[5](https://arxiv.org/html/2508.19564#S4.F5 "Figure 5 ‣ 4.4 Results on Diffusion Models ‣ 4 Experiments ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"), in the second row, the image generated by Bi-LoRA artfully merges the fluffy rabbit with the icon, whereas LoRA either fails to integrate the rabbit into the icon (fourth) or even does not generate an icon (third). Furthermore, Bi-LoRA better preserves the attributes of the rabbit, such as the eyes in the first column. More qualitative examples are shown in Appendix[K](https://arxiv.org/html/2508.19564#A11 "Appendix K Qualitative Results on SDXL ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").

### 4.5 Integration with Other LoRA Variants

Table 4: Results of fine-tuning T5-base on MRPC and CoLA using various LoRA variants, both standalone and combined with Bi-LoRA.

Method MRPC CoLA Avg.
LoRA-GA 88.81±0.47 58.87±1.00 73.84
PiSSA 88.15±0.24 58.66±0.47 73.41
DoRA 88.81±0.29 59.89±0.72 74.35
LoRA-GA + Bi-LoRA 89.62±0.24 60.70±0.27 75.16
PiSSA + Bi-LoRA 89.95±0.50 59.77±0.83 74.86
DoRA + Bi-LoRA 89.54±0.13 60.77±0.47 75.16

In this section, we evaluate Bi-LoRA’s effectiveness when combined with existing advanced LoRA variants. Specifically, we consider three LoRA variants, including LoRA-GA(Wang et al., [2024](https://arxiv.org/html/2508.19564#bib.bib45 "LoRA-ga: low-rank adaptation with gradient approximation")), PiSSA(Meng et al., [2024](https://arxiv.org/html/2508.19564#bib.bib46 "PiSSA: principal singular values and singular vectors adaptation of large language models")) and DoRA(Liu et al., [2024](https://arxiv.org/html/2508.19564#bib.bib54 "Dora: weight-decomposed low-rank adaptation")), and fine-tune the T5-base model on the MRPC and CoLA datasets, with detailed training settings provided in Appendix[R.1](https://arxiv.org/html/2508.19564#A18.SS1 "R.1 Experiments on the GLUE and SuperGLUE datasets ‣ Appendix R Empirical Details ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"). As shown in Table[4](https://arxiv.org/html/2508.19564#S4.T4 "Table 4 ‣ 4.5 Integration with Other LoRA Variants ‣ 4 Experiments ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"), integrating Bi-LoRA improves the average score across MRPC and CoLA by 1.45%, achieving gains of up to 1.80% on MRPC and 1.83% on CoLA. These results confirm that Bi-LoRA can be seamlessly integrated with previous approaches and deliver consistent improvements.

### 4.6 Ablations and Hyperparameter Sensitivity

Table 5: Ablations of each component in Bi-LoRA using Llama 3.1-8B on instruction following tasks. We compare vanilla LoRA, Dual-LoRA (two descent branches with global clipping), and Bi-LoRA under different clipping schemes (no clipping, per-layer clipping, and global clipping).

Method Clipping Adversarial MMLU DROP HEval BBH Avg.
LoRA✗✗63.38±0.39 49.82±0.54 43.15±0.93 42.82±0.27 49.79
Dual-LoRA global✗63.49±0.12 49.97±0.17 42.71±0.47 43.25±0.15 49.86
Bi-LoRA (no clip)✗\checkmark 61.33±0.42 47.31±3.66 40.24±0.93 41.96±0.48 47.71
Bi-LoRA (per-layer clip)per-layer\checkmark 63.49±0.26 50.26±0.30 43.29±0.61 43.06±0.28 50.03
Bi-LoRA (global clip)global\checkmark 63.67±0.15 51.53±0.33 46.12±0.89 43.45±0.15 51.19

Setting. In this section, we ablate Bi-LoRA to quantify each component’s contribution, and investigate the hyperparameter sensitivity from three factors: (1) auxiliary learning rate \eta_{2}, (2) ranks of primary (r_{1}) and auxiliary r_{2} LoRA modules, and (3) joint sensitivity to neighborhood radius \rho and auxiliary rank r_{2}. We fine-tune Llama 3.1-8B and evaluate its performance on the instruction following tasks.

Component-wise ablations of Bi-LoRA. We examine whether the improvements of Bi-LoRA arise from the additional capacity introduced by the second LoRA branch or from the adversarial ascent step. To this end, we ablate the clipping schemes and the usage of the auxiliary LoRA module.

Table[5](https://arxiv.org/html/2508.19564#S4.T5 "Table 5 ‣ 4.6 Ablations and Hyperparameter Sensitivity ‣ 4 Experiments ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models") shows that Dual-LoRA, which shares the same two-LoRA architecture as Bi-LoRA but updates both LoRA modules with gradient descent, only slightly outperforms vanilla LoRA (+0.07 Avg.). Intuitively, the parallel descent branch in Dual-LoRA behaves similarly to merging with another LoRA update whose norm is constrained by the global radius\rho, which prevents it from fully leveraging the increased rank and representational capacity. This indicates that the performance gains of Bi-LoRA primarily stem from the adversarial ascent step rather than from merely adding an extra LoRA branch. We further compare global clipping with two alternatives: no-clipping and per-layer clipping. Removing clipping leads to noticeable degradation (-3.48 Avg.), as the perturbations become excessively large. And per-layer clipping yields only marginal improvements over LoRA (+0.24 Avg.). These observations support global clipping as the more stable and effective choice. A more detailed explanation of why we adopt global clipping is provided in Appendix[G](https://arxiv.org/html/2508.19564#A7 "Appendix G Reasons of Using Global Clipping ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").

Sensitivity to auxiliary learning rate \eta_{2}. Table[6](https://arxiv.org/html/2508.19564#S4.T6 "Table 6 ‣ 4.6 Ablations and Hyperparameter Sensitivity ‣ 4 Experiments ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models") shows that Bi-LoRA is insensitive to auxiliary learning rate \eta_{2}: the average score remains within a narrow band of [50.55,51.19] as \eta_{2} varies from 1e-4 to 1e-3. This indicates that Bi-LoRA is robust to \eta_{2} in a relatively wide range. Interestingly, the best average performance is obtained when \eta_{2}= 3e-4, which coincides with the optimal primary learning rate \eta_{1} given by Ren et al. ([2024](https://arxiv.org/html/2508.19564#bib.bib39 "MELoRA: mini-ensemble low-rank adapters for parameter-efficient fine-tuning")). This suggests that using a shared learning rate for the primary and auxiliary branches yields more coordinated optimization, consistent with our observation in Table[A7](https://arxiv.org/html/2508.19564#A17.T7 "Table A7 ‣ Appendix Q Learning Rates for Tuning the Two Modules in Bi-LoRA ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"). Hence, in all main experiments, we simply set \eta_{2}=\eta_{1}, reducing hyperparameter tuning while retaining strong performance.

Table 6: Sensitivity of Bi-LoRA to the auxiliary learning rate \eta_{2} with Llama 3.1-8B on instruction following tasks, with the same primary learning rate \eta_{1} as in Section[4.2](https://arxiv.org/html/2508.19564#S4.SS2 "4.2 Results on Llama Models ‣ 4 Experiments ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").

\eta_{2}MMLU DROP HEval BBH Avg
1e-4 63.89±0.25 51.05±0.13 45.12±0.35 42.84±0.15 50.73
3e-4 63.67±0.15 51.53±0.33 46.12±0.89 43.45±0.15 51.19
5e-4 63.47±0.35 51.23±0.24 45.93±0.73 43.05±0.12 50.92
1e-3 63.40±0.17 51.32±0.22 44.51±0.93 42.96±0.23 50.55

Sensitivity to ranks r_{1} and r_{2}. Figure[6(a)](https://arxiv.org/html/2508.19564#S4.F6.sf1 "In Figure 6 ‣ 4.6 Ablations and Hyperparameter Sensitivity ‣ 4 Experiments ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models") shows that the average point rises monotonically up to the auxiliary rank r_{2}=8, which we adopt as the default for fine-tuning scenario. Additionally, further increasing r_{2} does not yield significant improvements or even downgrades the performance.

![Image 9: Refer to caption](https://arxiv.org/html/2508.19564v2/_figures/effect_rank2_llama_if.png)

(a) Effect of r_{1} and r_{2}.

![Image 10: Refer to caption](https://arxiv.org/html/2508.19564v2/x1.png)

(b) Effect of \rho and r_{2}.

Figure 6: Sensitivity of Bi-LoRA to primary (r_{1}) and auxiliary (r_{2}) LoRA ranks and perturbation radius \rho with Llama 3.1-8B on instruction-following tasks.

Sensitivity to neighborhood radius \rho and auxiliary rank r_{2}. Figure[6(b)](https://arxiv.org/html/2508.19564#S4.F6.sf2 "In Figure 6 ‣ 4.6 Ablations and Hyperparameter Sensitivity ‣ 4 Experiments ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models") further examines the joint effect of \rho and r_{2}. The heatmap also corroborates that enlarging r_{2} from 2 to 8 is beneficial. Overall, these trends suggest that \rho and r_{2} govern complementary aspects of neighborhood size and auxiliary capacity and can be tuned largely independently for a given architecture and dataset, in line with previous work of SAM.

### 4.7 Training Time and Memory Cost

Table 7:  Peak memory and time per optimization step (relative to LoRA in parentheses).

Model & Dataset Method Memory (GB)Time (s)
Llama 3.1-8B Cleaned Alpaca LoRA 23.69 8.93 (100%)
LoRA-SAM 24.00 19.23 (215%)
Bi-LoRA 24.32 9.71 (109%)

We present detailed comparisons of memory and time per optimization step for LoRA, LoRA-SAM, and Bi-LoRA on Llama 3.1-8B (Cleaned Alpaca). Table[7](https://arxiv.org/html/2508.19564#S4.T7 "Table 7 ‣ 4.7 Training Time and Memory Cost ‣ 4 Experiments ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models") shows that Bi-LoRA significantly reduces the LoRA-SAM’s training overhead, decreasing it from over 210% to about 110% relative to vanilla LoRA. This gain comes from jointly performing optimization and perturbation, eliminating SAM’s extra gradient step. For memory, Bi-LoRA incurs only a minimal overhead (< 0.7GB), as it adds a lightweight auxiliary module. Additional comparisons on training time are provided in Appendix[S](https://arxiv.org/html/2508.19564#A19 "Appendix S Time Comparisons Across Tasks ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").

## 5 Conclusion

In this paper, we propose Bi-LoRA, a novel dual-LoRA variant that leverages an auxiliary LoRA module to enhance the generalization performance of low-rank adaptation. Bi-LoRA decouples optimization from weight perturbation for better optimizing the sharpness of the loss landscape, allowing both to be updated in a single backward pass without requiring additional gradient steps, as in SAM. Extensive experiments across diverse tasks and architectures demonstrate Bi-LoRA’s efficiency and effectiveness in improving generalization.

#### Acknowledgments

The research leading to these results has received funding from National Key Research Development Project (2023YFF1104202) and National Natural Science Foundation of China (62376155). The authors would like to thank the chairs and reviewers for their thoughtful comments on this paper. Yuhang Liu also gratefully acknowledges the financial support provided by Prof. Xin Yang at Shanghai Jiao Tong University.

## References

*   M. Andriushchenko and N. Flammarion (2022)Towards understanding sharpness-aware minimization. In International Conference on Machine Learning (ICML), Cited by: [§3](https://arxiv.org/html/2508.19564#S3.p7.1 "3 Bi-LoRA: Bi-directional Low-Rank Adaptation ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   M. Bini, L. Girrbach, and Z. Akata (2025)Decoupling angles and strength in low-rank adaptation. In International Conference on Learning Representations (ICLR), Cited by: [6th item](https://arxiv.org/html/2508.19564#S4.I1.i6.p1.1.2 "In 4.1 Baselines ‣ 4 Experiments ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   X. Chen, C. Hsieh, and B. Gong (2022)When vision transformers outperform resnets without pre-training or strong data augmentations. In International Conference on Learning Representations (ICLR), Cited by: [§1](https://arxiv.org/html/2508.19564#S1.p3.1 "1 Introduction ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   Y. K. Chia, P. Hong, L. Bing, and S. Poria (2023)Instructeval: towards holistic evaluation of instruction-tuned large language models. arXiv preprint arXiv:2306.04757. Cited by: [4th item](https://arxiv.org/html/2508.19564#S4.I2.i4.p1.1 "In 4.2 Results on Llama Models ‣ 4 Experiments ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   Y. Dai, S. P. Karimireddy, S. Stich, and M. Jaggi (2023)The crucial role of normalization in sharpness-aware minimization. In Neural Information Processing Systems (NeurIPS), Cited by: [§3](https://arxiv.org/html/2508.19564#S3.p7.1 "3 Bi-LoRA: Bi-directional Low-Rank Adaptation ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   J. Deng, Q. Zhu, J. Pang, L. Yang, Z. Fu, and B. Zhang (2025)EFlat-lora: efficiently seeking flat minima for better generalization in fine-tuning large language models and beyond. arXiv preprint arXiv:2508.00522. Cited by: [§1](https://arxiv.org/html/2508.19564#S1.p2.1 "1 Introduction ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   T. Dettmers, A. Pagnoni, A. Holtzman, and L. Zettlemoyer (2023)QLoRA: efficient finetuning of quantized llms. In Advances in Neural Information Processing System (NeurIPS), Cited by: [Appendix M](https://arxiv.org/html/2508.19564#A13.p1.1 "Appendix M Bi-LoRA in Quantized Settings ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   J. Devlin, M. Chang, K. Lee, and K. Toutanova (2018)Bert: pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805. Cited by: [§1](https://arxiv.org/html/2508.19564#S1.p1.1 "1 Introduction ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, et al. (2021)An image is worth 16x16 words: transformers for image recognition at scale. In International Conference on Learning Representations (ICLR), Cited by: [§1](https://arxiv.org/html/2508.19564#S1.p1.1 "1 Introduction ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   S. Dou, E. Zhou, Y. Liu, S. Gao, W. Shen, L. Xiong, Y. Zhou, X. Wang, Z. Xi, X. Fan, S. Pu, J. Zhu, R. Zheng, T. Gui, Q. Zhang, and X. Huang (2024)LoRAMoE: alleviating world knowledge forgetting in large language models via MoE-style plugin. In Proceedings of the Association for Computational Linguistics (Volume 1: Long Papers), Cited by: [Appendix E](https://arxiv.org/html/2508.19564#A5.p1.1 "Appendix E Related Work ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   J. Du, H. Yan, J. Feng, J. T. Zhou, L. Zhen, R. S. M. Goh, and V. Tan (2022a)Efficient sharpness-aware minimization for improved training of neural networks. In International Conference on Learning Representations (ICLR), Cited by: [§R.5](https://arxiv.org/html/2508.19564#A18.SS5.p1.1 "R.5 Efficient SAM Variants for LoRA Fine-Tuning ‣ Appendix R Empirical Details ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"), [5th item](https://arxiv.org/html/2508.19564#S4.I1.i5.p1.1 "In 4.1 Baselines ‣ 4 Experiments ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"), [§4.2](https://arxiv.org/html/2508.19564#S4.SS2.p1.1 "4.2 Results on Llama Models ‣ 4 Experiments ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   J. Du, D. Zhou, J. Feng, V. Y. F. Tan, and J. T. Zhou (2022b)Sharpness-aware training for free. In Advances in Neural Information Processing Systems (NeurIPS), Cited by: [Appendix E](https://arxiv.org/html/2508.19564#A5.p2.1 "Appendix E Related Work ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   A. Dubey, A. Jauhri, A. Pandey, A. Kadian, A. Al-Dahle, A. Letman, A. Mathur, A. Schelten, A. Yang, A. Fan, et al. (2024)The llama 3 herd of models. arXiv preprint arXiv:2407.21783. Cited by: [§4](https://arxiv.org/html/2508.19564#S4.p1.1 "4 Experiments ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   P. Foret, A. Kleiner, H. Mobahi, and B. Neyshabur (2021)Sharpness-aware minimization for efficiently improving generalization. In International Conference on Learning Representations (ICLR), Cited by: [§1](https://arxiv.org/html/2508.19564#S1.p3.1 "1 Introduction ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"), [§2.2](https://arxiv.org/html/2508.19564#S2.SS2.p2.8 "2.2 LoRA-SAM ‣ 2 Issues of LoRA-SAM ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   S. Hochreiter and J. Schmidhuber (1994)Simplifying neural nets by discovering flat minima. In Advances in Neural Information Processing Systems (NeurIPS), Cited by: [§1](https://arxiv.org/html/2508.19564#S1.p3.1 "1 Introduction ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   S. Hochreiter and J. Schmidhuber (1997)Flat minima. Neural computation. Cited by: [§1](https://arxiv.org/html/2508.19564#S1.p3.1 "1 Introduction ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen (2022)LoRA: low-rank adaptation of large language models. In International Conference on Learning Representations (ICLR), Cited by: [Appendix E](https://arxiv.org/html/2508.19564#A5.p1.1 "Appendix E Related Work ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"), [§1](https://arxiv.org/html/2508.19564#S1.p1.1 "1 Introduction ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"), [Figure 2](https://arxiv.org/html/2508.19564#S2.F2 "In 2.1 Preliminaries ‣ 2 Issues of LoRA-SAM ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   Q. Huang, T. Ko, Z. Zhuang, L. Tang, and Y. Zhang (2025)HiRA: parameter-efficient hadamard high-rank adaptation for large language models. In International Conference on Learning Representations (ICLR), Cited by: [6th item](https://arxiv.org/html/2508.19564#S4.I1.i6.p1.1.2 "In 4.1 Baselines ‣ 4 Experiments ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   J. Ji, G. Li, J. Fu, F. Afghah, L. Guo, X. Yuan, and X. Ma (2024)A single-step, sharpness-aware minimization is all you need to achieve efficient and accurate sparse training. In Advances in Neural Information Processing Systems (NeurIPS), Cited by: [Appendix E](https://arxiv.org/html/2508.19564#A5.p2.1 "Appendix E Related Work ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"), [§F.1](https://arxiv.org/html/2508.19564#A6.SS1.p1.5.5 "F.1 Proof and Discussion of Proposition 2: Alignment of Bi-LoRA’s ascent direction with previous full gradient ‣ Appendix F Proofs ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"), [5th item](https://arxiv.org/html/2508.19564#S4.I1.i5.p1.1.1 "In 4.1 Baselines ‣ 4 Experiments ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   W. Jiang, H. Yang, Y. Zhang, and J. Kwok (2023)An adaptive policy to employ sharpness-aware minimization. In The Eleventh International Conference on Learning Representations (ICLR), Cited by: [Appendix E](https://arxiv.org/html/2508.19564#A5.p2.1 "Appendix E Related Work ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   P. D. Khanh, H. Luong, B. S. Mordukhovich, and D. B. Tran (2024)Fundamental convergence analysis of sharpness-aware minimization. In Neural Information Processing Systems (NeurIPS), Cited by: [§3](https://arxiv.org/html/2508.19564#S3.p7.1 "3 Bi-LoRA: Bi-directional Low-Rank Adaptation ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   M. Kim, D. Li, S. X. Hu, and T. Hospedales (2022)Fisher sam: information geometry and sharpness aware minimisation. In International Conference on Machine Learning (ICML), Cited by: [Appendix N](https://arxiv.org/html/2508.19564#A14.p1.4 "Appendix N Effect on Adversarial Robustness ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"), [Appendix E](https://arxiv.org/html/2508.19564#A5.p2.1 "Appendix E Related Work ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   A. Kolesnikov, L. Beyer, X. Zhai, J. Puigcerver, J. Yung, S. Gelly, and N. Houlsby (2020)Big transfer (bit): general visual representation learning. In European conference on computer vision (ECCV), Cited by: [§1](https://arxiv.org/html/2508.19564#S1.p1.1 "1 Introduction ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   S. A. Koohpayegani, N. K. L., P. Nooralinejad, S. Kolouri, and H. Pirsiavash (2024)NOLA: compressing lora using linear combination of random basis. In International Conference on Learning Representations (ICLR), Cited by: [Appendix E](https://arxiv.org/html/2508.19564#A5.p1.1 "Appendix E Related Work ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   J. Kwon, J. Kim, H. Park, and I. K. Choi (2021)ASAM: adaptive sharpness-aware minimization for scale-invariant learning of deep neural networks. In International Conference on Machine Learning (ICML), Cited by: [Appendix E](https://arxiv.org/html/2508.19564#A5.p2.1 "Appendix E Related Work ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   B. Li and G. B. Giannakis (2023)Enhancing sharpness-aware optimization through variance suppression. In Advances in Neural Information Processing Systems (NeurIPS), Cited by: [Appendix E](https://arxiv.org/html/2508.19564#A5.p2.1 "Appendix E Related Work ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   B. Li, L. Zhang, and N. He (2024a)Implicit regularization of sharpness-aware minimization for scale-invariant problems. In Advances in Neural Information Processing Systems (NeurIPS), Cited by: [§R.5](https://arxiv.org/html/2508.19564#A18.SS5.p1.1 "R.5 Efficient SAM Variants for LoRA Fine-Tuning ‣ Appendix R Empirical Details ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"), [Appendix E](https://arxiv.org/html/2508.19564#A5.p2.1 "Appendix E Related Work ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"), [§1](https://arxiv.org/html/2508.19564#S1.p4.1 "1 Introduction ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"), [5th item](https://arxiv.org/html/2508.19564#S4.I1.i5.p1.1 "In 4.1 Baselines ‣ 4 Experiments ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   H. Li, Z. Xu, G. Taylor, C. Studer, and T. Goldstein (2018)Visualizing the loss landscape of neural nets. In Advances in Neural Information Processing Systems (NeurIPS), Cited by: [Figure 4](https://arxiv.org/html/2508.19564#S3.F4 "In 3 Bi-LoRA: Bi-directional Low-Rank Adaptation ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   S. Li, Y. Yang, Y. Shen, F. Wei, Z. Lu, L. Qiu, and Y. Yang (2024b)LoRASC: expressive and generalizable low-rank adaptation for large models via slow cascaded learning. In Findings of the Association for Computational Linguistics: EMNLP, Cited by: [§1](https://arxiv.org/html/2508.19564#S1.p2.1 "1 Introduction ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   T. Li, Z. He, Y. Li, Y. Wang, L. Shang, and X. Huang (2025)Flat-lora: low-rank adaptation over a flat loss landscape. In International Conference on Machine Learning (ICML), Cited by: [§R.5](https://arxiv.org/html/2508.19564#A18.SS5.p1.1 "R.5 Efficient SAM Variants for LoRA Fine-Tuning ‣ Appendix R Empirical Details ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"), [Appendix E](https://arxiv.org/html/2508.19564#A5.p2.1 "Appendix E Related Work ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"), [Appendix I](https://arxiv.org/html/2508.19564#A9.p1.1 "Appendix I Results on Natural Language Understanding ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"), [§1](https://arxiv.org/html/2508.19564#S1.p2.1 "1 Introduction ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"), [5th item](https://arxiv.org/html/2508.19564#S4.I1.i5.p1.1 "In 4.1 Baselines ‣ 4 Experiments ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"), [§4.2](https://arxiv.org/html/2508.19564#S4.SS2.p1.2 "4.2 Results on Llama Models ‣ 4 Experiments ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   T. Li, L. Tan, Z. Huang, Q. Tao, Y. Liu, and X. Huang (2022)Low dimensional trajectory hypothesis is true: dnns can be trained in tiny subspaces. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI). Cited by: [§F.1](https://arxiv.org/html/2508.19564#A6.SS1.p1.5.5 "F.1 Proof and Discussion of Proposition 2: Alignment of Bi-LoRA’s ascent direction with previous full gradient ‣ Appendix F Proofs ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   T. Li, P. Zhou, Z. He, X. Cheng, and X. Huang (2024c)Friendly sharpness-aware minimization. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Cited by: [Appendix E](https://arxiv.org/html/2508.19564#A5.p2.1 "Appendix E Related Work ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   Y. Lin, X. Ma, X. Chu, Y. Jin, Z. Yang, Y. Wang, and H. Mei (2024)LoRA dropout as a sparsity regularizer for overfitting control. arXiv preprint arXiv:2404.09610. Cited by: [§1](https://arxiv.org/html/2508.19564#S1.p2.1 "1 Introduction ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   S. Liu, C. Wang, H. Yin, P. Molchanov, Y. F. Wang, K. Cheng, and M. Chen (2024)Dora: weight-decomposed low-rank adaptation. In International Conference on Machine Learning (ICML), Cited by: [Appendix E](https://arxiv.org/html/2508.19564#A5.p1.1 "Appendix E Related Work ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"), [6th item](https://arxiv.org/html/2508.19564#S4.I1.i6.p1.1 "In 4.1 Baselines ‣ 4 Experiments ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"), [§4.5](https://arxiv.org/html/2508.19564#S4.SS5.p1.1 "4.5 Integration with Other LoRA Variants ‣ 4 Experiments ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   Y. Liu, S. Mai, X. Chen, C. Hsieh, and Y. You (2022a)Towards efficient and scalable sharpness-aware minimization. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Cited by: [Appendix E](https://arxiv.org/html/2508.19564#A5.p2.1 "Appendix E Related Work ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   Y. Liu, S. Mai, X. Chen, C. Hsieh, and Y. You (2022b)Towards efficient and scalable sharpness-aware minimization. In Conference on Computer Vision and Pattern Recognition (CVPR), Cited by: [§R.5](https://arxiv.org/html/2508.19564#A18.SS5.p1.1 "R.5 Efficient SAM Variants for LoRA Fine-Tuning ‣ Appendix R Empirical Details ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"), [5th item](https://arxiv.org/html/2508.19564#S4.I1.i5.p1.1 "In 4.1 Baselines ‣ 4 Experiments ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"), [§4.2](https://arxiv.org/html/2508.19564#S4.SS2.p1.1 "4.2 Results on Llama Models ‣ 4 Experiments ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   F. Meng, Z. Wang, and M. Zhang (2024)PiSSA: principal singular values and singular vectors adaptation of large language models. In Advances in Neural Information Processing Systems (NeurIPS), Cited by: [Appendix E](https://arxiv.org/html/2508.19564#A5.p1.1 "Appendix E Related Work ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"), [6th item](https://arxiv.org/html/2508.19564#S4.I1.i6.p1.1 "In 4.1 Baselines ‣ 4 Experiments ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"), [§4.5](https://arxiv.org/html/2508.19564#S4.SS5.p1.1 "4.5 Integration with Other LoRA Variants ‣ 4 Experiments ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   D. Podell, Z. English, K. Lacey, A. Blattmann, T. Dockhorn, J. Müller, J. Penna, and R. Rombach (2023)Sdxl: improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952. Cited by: [§4.4](https://arxiv.org/html/2508.19564#S4.SS4.p1.1 "4.4 Results on Diffusion Models ‣ 4 Experiments ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   Qwen, :, A. Yang, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Li, D. Liu, F. Huang, H. Wei, H. Lin, J. Yang, J. Tu, J. Zhang, J. Yang, J. Yang, J. Zhou, J. Lin, K. Dang, K. Lu, K. Bao, K. Yang, L. Yu, M. Li, M. Xue, P. Zhang, Q. Zhu, R. Men, R. Lin, T. Li, T. Tang, T. Xia, X. Ren, X. Ren, Y. Fan, Y. Su, Y. Zhang, Y. Wan, Y. Liu, Z. Cui, Z. Zhang, and Z. Qiu (2025)Qwen2.5 technical report. arXiv preprint arXiv:2412.15115. Cited by: [§4](https://arxiv.org/html/2508.19564#S4.p1.1 "4 Experiments ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark, et al. (2021)Learning transferable visual models from natural language supervision. In International Conference on Machine Learning (ICML), Cited by: [§1](https://arxiv.org/html/2508.19564#S1.p1.1 "1 Introduction ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   C. Raffel, N. Shazeer, A. Roberts, K. Lee, S. Narang, M. Matena, Y. Zhou, W. Li, and P. J. Liu (2020)Exploring the limits of transfer learning with a unified text-to-text transformer. Journal of machine learning research (JMLR). Cited by: [Appendix I](https://arxiv.org/html/2508.19564#A9.p1.1 "Appendix I Results on Natural Language Understanding ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   P. Ren, C. Shi, S. Wu, M. Zhang, Z. Ren, M. de Rijke, Z. Chen, and J. Pei (2024)MELoRA: mini-ensemble low-rank adapters for parameter-efficient fine-tuning. In Proceedings of the Association for Computational Linguistics (Volume 1: Long Papers), Cited by: [Appendix E](https://arxiv.org/html/2508.19564#A5.p1.1 "Appendix E Related Work ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"), [§4.2](https://arxiv.org/html/2508.19564#S4.SS2.p1.2 "4.2 Results on Llama Models ‣ 4 Experiments ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"), [§4.6](https://arxiv.org/html/2508.19564#S4.SS6.p4.8.8.7 "4.6 Ablations and Hyperparameter Sensitivity ‣ 4 Experiments ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   N. Ruiz, Y. Li, V. Jampani, Y. Pritch, M. Rubinstein, and K. Aberman (2023)Dreambooth: fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), Cited by: [§4.4](https://arxiv.org/html/2508.19564#S4.SS4.p1.1 "4.4 Results on Diffusion Models ‣ 4 Experiments ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   R. Taori, I. Gulrajani, T. Zhang, Y. Dubois, X. Li, C. Guestrin, P. Liang, and T. B. Hashimoto (2023)Stanford alpaca: an instruction-following llama model. Cited by: [4th item](https://arxiv.org/html/2508.19564#S4.I2.i4.p1.1 "In 4.2 Results on Llama Models ‣ 4 Experiments ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   C. Tian, Z. Shi, Z. Guo, L. Li, and C. Xu (2024)HydraLoRA: an asymmetric LoRA architecture for efficient fine-tuning. In Advances in Neural Information Processing Systems (NeurIPS), Cited by: [Appendix E](https://arxiv.org/html/2508.19564#A5.p1.1 "Appendix E Related Work ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   H. Touvron, L. Martin, K. Stone, P. Albert, A. Almahairi, Y. Babaei, N. Bashlykov, S. Batra, P. Bhargava, S. Bhosale, et al. (2023)Llama 2: open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288. Cited by: [§4](https://arxiv.org/html/2508.19564#S4.p1.1 "4 Experiments ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   A. Wang, Y. Pruksachatkun, N. Nangia, A. Singh, J. Michael, F. Hill, O. Levy, and S. Bowman (2019a)Superglue: a stickier benchmark for general-purpose language understanding systems. In Advances in Neural Information Processing Systems (NeurIPS), Vol. 32. Cited by: [Appendix I](https://arxiv.org/html/2508.19564#A9.p1.1 "Appendix I Results on Natural Language Understanding ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   A. Wang, A. Singh, J. Michael, F. Hill, O. Levy, and S. R. Bowman (2019b)GLUE: a multi-task benchmark and analysis platform for natural language understanding. In International Conference on Learning Representations (ICLR), Cited by: [Appendix I](https://arxiv.org/html/2508.19564#A9.p1.1 "Appendix I Results on Natural Language Understanding ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   S. Wang, L. Yu, and J. Li (2024)LoRA-ga: low-rank adaptation with gradient approximation. In Advances in Neural Information Processing Systems (NuerIPS), Cited by: [Appendix E](https://arxiv.org/html/2508.19564#A5.p1.1 "Appendix E Related Work ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"), [Appendix I](https://arxiv.org/html/2508.19564#A9.p1.1 "Appendix I Results on Natural Language Understanding ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"), [6th item](https://arxiv.org/html/2508.19564#S4.I1.i6.p1.1 "In 4.1 Baselines ‣ 4 Experiments ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"), [§4.1](https://arxiv.org/html/2508.19564#S4.SS1.p3.1 "4.1 Baselines ‣ 4 Experiments ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"), [§4.2](https://arxiv.org/html/2508.19564#S4.SS2.p1.2 "4.2 Results on Llama Models ‣ 4 Experiments ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"), [§4.5](https://arxiv.org/html/2508.19564#S4.SS5.p1.1 "4.5 Integration with Other LoRA Variants ‣ 4 Experiments ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   Z. Wang, J. Liang, R. He, Z. Wang, and T. Tan (2025)LoRA-pro: are low-rank adapters properly optimized?. In International Conference on Learning Representations (ICLR), Cited by: [Appendix E](https://arxiv.org/html/2508.19564#A5.p1.1 "Appendix E Related Work ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"), [§4.1](https://arxiv.org/html/2508.19564#S4.SS1.p3.1 "4.1 Baselines ‣ 4 Experiments ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   Y. Wu, H. Piao, L. Huang, R. Wang, W. Li, H. Pfister, D. Meng, K. Ma, and Y. Wei (2025)SD-loRA: scalable decoupled low-rank adaptation for class incremental learning. In International Conference on Learning Representations (ICLR), Cited by: [Appendix E](https://arxiv.org/html/2508.19564#A5.p1.1 "Appendix E Related Work ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   J. Yen, S. Si, Z. Meng, F. Yu, S. S. Duvvuri, I. S. Dhillon, C. Hsieh, and S. Kumar (2025)LoRA done RITE: robust invariant transformation equilibration for loRA optimization. In International Conference on Learning Representations (ICLR), Cited by: [Appendix E](https://arxiv.org/html/2508.19564#A5.p1.1 "Appendix E Related Work ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   Y. Yue, J. Jiang, Z. Ye, N. Gao, Y. Liu, and K. Zhang (2023)Sharpness-aware minimization revisited: weighted sharpness as a regularization term. In ACM SIGKDD Conference on Knowledge Discovery and Data Mining, Cited by: [§3](https://arxiv.org/html/2508.19564#S3.p8.1 "3 Bi-LoRA: Bi-directional Low-Rank Adaptation ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"), [5th item](https://arxiv.org/html/2508.19564#S4.I1.i5.p1.1.1 "In 4.1 Baselines ‣ 4 Experiments ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   J. Zhang, Y. Zhao, D. Chen, X. Tian, H. Zheng, and W. Zhu (2024)MiLoRA: efficient mixture of low-rank adaptation for large language models fine-tuning. In Findings of the Association for Computational Linguistics: EMNLP, Cited by: [Appendix E](https://arxiv.org/html/2508.19564#A5.p1.1 "Appendix E Related Work ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   Q. Zhang, M. Chen, A. Bukharin, P. He, Y. Cheng, W. Chen, and T. Zhao (2023)Adaptive budget allocation for parameter-efficient fine-tuning. In International Conference on Learning Representations (ICLR), Cited by: [Appendix E](https://arxiv.org/html/2508.19564#A5.p1.1 "Appendix E Related Work ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   J. Zhao, Z. Zhang, B. Chen, Z. Wang, A. Anandkumar, and Y. Tian (2024)GaLore: memory-efficient LLM training by gradient low-rank projection. In International Conference on Machine Learning (ICML), Cited by: [§F.1](https://arxiv.org/html/2508.19564#A6.SS1.p1.5.5 "F.1 Proof and Discussion of Proposition 2: Alignment of Bi-LoRA’s ascent direction with previous full gradient ‣ Appendix F Proofs ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").
*   J. Zhuang, B. Gong, L. Yuan, Y. Cui, H. Adam, N. Dvornek, S. Tatikonda, J. Duncan, and T. Liu (2022)Surrogate gap minimization improves sharpness-aware training. In International Conference on Learning Representations (ICLR), Cited by: [Appendix E](https://arxiv.org/html/2508.19564#A5.p2.1 "Appendix E Related Work ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"), [§1](https://arxiv.org/html/2508.19564#S1.p3.1 "1 Introduction ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").

## Appendix

## Appendix A Ethics statement

This paper proposes Bi-LoRA, a parameter-efficient fine-tuning method that introduces adversarial perturbations to improve generalization of large language models, which can reduce memory and energy costs, and thus broaden access to LLM research.

For dataset use, we use only publicly available datasets under their licenses, with no human subjects or personal data involved.

Regarding safety and privacy, we recognize potential risks. By reducing hardware barriers, Bi-LoRA could also enable malicious actors to efficiently fine-tune models for harmful applications or exacerbate dataset biases if such biases are present. Furthermore, the auxiliary adversarial adapter, while improving robustness, introduces an additional parameter pathway that could hypothetically be exploited for backdoor insertion or undesired behaviors if misused. To mitigate these risks, we will anonymously release only training code and experimental scripts in the supplementary materials that follow community standards for responsible research, and we rely exclusively on open benchmarks with no sensitive or personal data. We encourage future users of Bi-LoRA to adopt similar safeguards, including dataset audits for bias, robust model evaluation, and adherence to ethical guidelines on downstream applications.

## Appendix B reproducibility statement

We provide detailed experimental settings and hyperparameters in Appendix[R](https://arxiv.org/html/2508.19564#A18 "Appendix R Empirical Details ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models") to ensure reproducibility. The Bi-LoRA implementation and example experiment scripts will be released anonymously in the supplementary materials. All models, benchmarks, and evaluation metrics used in this paper are either open-sourced or explicitly identified when closed-source resources are involved.

## Appendix C LLM usage Statement

We use large language models (LLMs) as general-purpose tools at the following stages of this work.

*   •
Writing. We employ both closed-source and open-source LLMs solely for text polishing, including grammar correction, clarity improvement, and minor stylistic refinement. Importantly, they are not used to generate research ideas, technical content, and substantive arguments.

*   •
Benchmarking. For evaluation on the MT-Bench benchmark, we use the GPT-4 API to obtain GPT-4 scores, following the standard evaluation protocol for MT-Bench.

No part of the research design, methodology, or core scientific contributions relied on LLMs. Their role is restricted to auxiliary assistance in language refinement and standardized benchmark scoring.

## Appendix D Details on Sharpness-Aware Minimization (SAM)

Let \mathcal{L}(W) denote the empirical loss of model parameters W. Standard ERM can converge to sharp minima where small parameter perturbations significantly increase loss. SAM’s goal is to target the training toward flat regions by minimizing the worst-case loss inside a :

\min_{W}\;\;\max_{\lVert\varepsilon\rVert\leq\rho}\;\mathcal{L}\!\left(W+\varepsilon\right),(A1)

where \rho>0 is the neighborhood radius, \|\cdot\| is a kind of norm, and \varepsilon is the perturbation. Intuitively, Eqn,([A1](https://arxiv.org/html/2508.19564#A4.E1 "In Appendix D Details on Sharpness-Aware Minimization (SAM) ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models")) prefers parameters whose entire neighborhood has uniformly low loss (flat minima), which has been empirically linked to improved generalization. A practical single-step approximation solves the inner maximization by first-order ascent and then updates W using the gradient evaluated at the perturbed point.

Denote by \mathcal{L}(W;\mathcal{B}) the mini-batch loss on \mathcal{B} and by \nabla_{W}\mathcal{L} its gradient. A typical SAM update computes a normalized gradient ascent perturbation and then a descent step on the same mini-batch \mathcal{B}_{t}, but at different parameter values:

\displaystyle{\varepsilon}_{t}\displaystyle=\rho\cdot\frac{\nabla_{W}\mathcal{L}(W_{t};\mathcal{B}_{t})}{\bigl\lVert\nabla_{W}\mathcal{L}(W_{t};\mathcal{B}_{t})\bigr\rVert},(A2)
\displaystyle W_{t+1}\displaystyle\leftarrow W_{t}\;-\;\eta\,\nabla_{W}\mathcal{L}\!\left(W_{t}+{\varepsilon}_{t};\mathcal{B}_{t}\right),(A3)

i.e., two gradient evaluations per iteration. Eqn.([A2](https://arxiv.org/html/2508.19564#A4.E2 "In Appendix D Details on Sharpness-Aware Minimization (SAM) ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models")) determines the perturbation direction at W_{t} (clean weight), while Eqn.([A3](https://arxiv.org/html/2508.19564#A4.E3 "In Appendix D Details on Sharpness-Aware Minimization (SAM) ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models")) computes the update at the perturbed parameter W_{t}+\epsilon_{t} (perturbed weight).

Algorithm A1 Sharpness-Aware Minimization (SAM)

1:Input: Initial weights

W_{0}
; learning rate

\eta
; radius

\rho
; optimizer

\mathsf{Opt}(\cdot)
; max steps

K

2:Output: Final weights

W

3: Initialize model weights

W\leftarrow W_{0}

4:for

t=0,1,\dots,K{-}1
do

5: Sample mini-batch

\mathcal{B}_{t}

6:

g\leftarrow\nabla_{W}\ell(W;\mathcal{B}_{t})
{gradient at clean weights}

7:

\varepsilon\leftarrow\rho\cdot\dfrac{g}{\lVert g\rVert}
{first-order worst-case direction}

8:

W_{\mathrm{adv}}\leftarrow W+\varepsilon

9:

g_{\mathrm{adv}}\leftarrow\nabla_{W}\mathcal{L}(W_{\mathrm{adv}};\mathcal{B}_{t})
{use the same batch to produce perturbed gradient}

10:

W\leftarrow\mathsf{Opt}\!\bigl(W,\;g_{\mathrm{adv}},\;\eta\bigr)
{e.g., SGD/AdamW step}

11:end for

12:return

W

## Appendix E Related Work

Low-Rank Adaptation (LoRA). LoRA(Hu et al., [2022](https://arxiv.org/html/2508.19564#bib.bib32 "LoRA: low-rank adaptation of large language models")) is a widely adopted parameter-efficient fine-tuning (PEFT) method. It models weight updates via low-rank matrices without incurring additional inference costs. Many studies have been proposed to improve the performance of LoRA(Koohpayegani et al., [2024](https://arxiv.org/html/2508.19564#bib.bib40 "NOLA: compressing lora using linear combination of random basis"); Dou et al., [2024](https://arxiv.org/html/2508.19564#bib.bib48 "LoRAMoE: alleviating world knowledge forgetting in large language models via MoE-style plugin"); Tian et al., [2024](https://arxiv.org/html/2508.19564#bib.bib50 "HydraLoRA: an asymmetric LoRA architecture for efficient fine-tuning"); Wang et al., [2025](https://arxiv.org/html/2508.19564#bib.bib57 "LoRA-pro: are low-rank adapters properly optimized?"); Yen et al., [2025](https://arxiv.org/html/2508.19564#bib.bib56 "LoRA done RITE: robust invariant transformation equilibration for loRA optimization"); Wu et al., [2025](https://arxiv.org/html/2508.19564#bib.bib55 "SD-loRA: scalable decoupled low-rank adaptation for class incremental learning")). On the resource perspective, AdaLoRA(Zhang et al., [2023](https://arxiv.org/html/2508.19564#bib.bib38 "Adaptive budget allocation for parameter-efficient fine-tuning")) dynamically adjusts the rank allocation, while MELoRA(Ren et al., [2024](https://arxiv.org/html/2508.19564#bib.bib39 "MELoRA: mini-ensemble low-rank adapters for parameter-efficient fine-tuning")) trains multiple mini-LoRA modules in parallel to cut down on trainable parameters. From the optimization perspective, LoRA-GA(Wang et al., [2024](https://arxiv.org/html/2508.19564#bib.bib45 "LoRA-ga: low-rank adaptation with gradient approximation")), PISSA(Meng et al., [2024](https://arxiv.org/html/2508.19564#bib.bib46 "PiSSA: principal singular values and singular vectors adaptation of large language models")) and MiLoRA(Zhang et al., [2024](https://arxiv.org/html/2508.19564#bib.bib47 "MiLoRA: efficient mixture of low-rank adaptation for large language models fine-tuning")) accelerate convergence and boost performance through enhanced initialization, and DoRA(Liu et al., [2024](https://arxiv.org/html/2508.19564#bib.bib54 "Dora: weight-decomposed low-rank adaptation")) decomposes the adaptation into magnitude and direction for better optimization. In this paper, we enhance LoRA optimization by optimizing the loss landscape sharpness of the full parameter space through an auxiliary LoRA module. Our approach is orthogonal to prior works.

Sharpness-Aware Minimization (SAM).  SAM formulates the optimization objective as a min-max problem to seek flat minima, encouraging model parameters to reside in regions with consistently low loss values, yielding state‐of‐the‐art generalization and robustness. Despite its effectiveness, SAM doubles the computational cost because its inner maximization is approximated via an additional gradient ascent step, thereby limiting its application to large-scale models. Several SAM variants have been developed to enhance its generalization performance(Kim et al., [2022](https://arxiv.org/html/2508.19564#bib.bib4 "Fisher sam: information geometry and sharpness aware minimisation"); Li and Giannakis, [2023](https://arxiv.org/html/2508.19564#bib.bib1 "Enhancing sharpness-aware optimization through variance suppression"); Li et al., [2024c](https://arxiv.org/html/2508.19564#bib.bib30 "Friendly sharpness-aware minimization")). ASAM(Kwon et al., [2021](https://arxiv.org/html/2508.19564#bib.bib26 "ASAM: adaptive sharpness-aware minimization for scale-invariant learning of deep neural networks")) introduces adaptive sharpness to address SAM’s scale-dependency issue, while GSAM(Zhuang et al., [2022](https://arxiv.org/html/2508.19564#bib.bib11 "Surrogate gap minimization improves sharpness-aware training")) jointly minimizes surrogate and perturbed losses to locate flatter region. Meanwhile, other works aim to improve SAM’s training efficiency(Du et al., [2022b](https://arxiv.org/html/2508.19564#bib.bib29 "Sharpness-aware training for free"); Jiang et al., [2023](https://arxiv.org/html/2508.19564#bib.bib34 "An adaptive policy to employ sharpness-aware minimization"); Ji et al., [2024](https://arxiv.org/html/2508.19564#bib.bib31 "A single-step, sharpness-aware minimization is all you need to achieve efficient and accurate sparse training")). LookSAM(Liu et al., [2022a](https://arxiv.org/html/2508.19564#bib.bib28 "Towards efficient and scalable sharpness-aware minimization")) applies adversarial perturbations only periodically. Recent studies have also focused on applying SAM to LoRA fine-tuning, aiming to enhance generalization while maintaining training efficiency. For example, BAR(Li et al., [2024a](https://arxiv.org/html/2508.19564#bib.bib53 "Implicit regularization of sharpness-aware minimization for scale-invariant problems")) makes SAM’s implicit balancedness regularization explicit for scale-invariant tasks like LoRA, achieving SAM-level generalization gains with significantly less computation. Flat-LoRA(Li et al., [2025](https://arxiv.org/html/2508.19564#bib.bib64 "Flat-lora: low-rank adaptation over a flat loss landscape")) replaces the costly inner maximization with random weight perturbation to tackle the coupling issue in LoRA-SAM without sacrificing efficiency. In this paper, we introduce a novel framework that integrates SAM with LoRA through dual LoRA modules. Our method employs auxiliary LoRA modules to generate adversarial perturbations decoupled from optimization, incurring minimal additional overhead.

## Appendix F Proofs

### F.1 Proof and Discussion of Proposition[2](https://arxiv.org/html/2508.19564#Thmproposition2 "Proposition 2 (Alignment of Bi-LoRA’s ascent direction with previous full gradient). ‣ 3 Bi-LoRA: Bi-directional Low-Rank Adaptation ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"): Alignment of Bi-LoRA’s ascent direction with previous full gradient

###### Proof.

Let

G_{t}=\nabla_{W}\mathcal{L}(W_{0}+B_{1,t}A_{1,t}+\tilde{\epsilon}_{t}),(A4)

denote the full gradient at step t, with \tilde{\epsilon}_{t}=B_{2,t}A_{2,t} the auxiliary perturbation.

One update step for Bi-LoRA is formalized as

\begin{cases}B_{1,t+1}=B_{1,t}-\eta_{1}G_{t}A_{1,t}^{\top},&A_{1,t+1}=A_{1,t}-\eta_{1}B_{1,t}^{\top}G_{t},\\
B_{2,t+1}=B_{2,t}+\eta_{2}G_{t}A_{2,t}^{\top},&A_{2,t+1}=A_{2,t}+\eta_{2}B_{2,t}^{\top}G_{t},\end{cases}(A5)

where \eta_{1},\eta_{2}>0 are step sizes.

Expanding the auxiliary perturbation Eqn.([A5](https://arxiv.org/html/2508.19564#A6.E5 "In Proof. ‣ F.1 Proof and Discussion of Proposition 2: Alignment of Bi-LoRA’s ascent direction with previous full gradient ‣ Appendix F Proofs ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models")) gives

\tilde{\epsilon}_{t+1}=B_{2,t+1}A_{2,t+1}=\tilde{\epsilon}_{t}+\eta_{2}\left(G_{t}A_{2,t}^{\top}A_{2,t}+B_{2,t}B_{2,t}^{\top}G_{t}\right)+\mathcal{O}(\eta_{2}^{2}).(A6)

Thus, the alignment between the previous gradient Eqn.Eqn.([A4](https://arxiv.org/html/2508.19564#A6.E4 "In Proof. ‣ F.1 Proof and Discussion of Proposition 2: Alignment of Bi-LoRA’s ascent direction with previous full gradient ‣ Appendix F Proofs ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models")) and the update of the auxiliary perturbation is

\langle G_{t},\tilde{\epsilon}_{t+1}-\tilde{\epsilon}_{t}\rangle=\eta_{2}\Big(\|A_{2,t}G_{t}^{\top}\|_{F}^{2}+\|B_{2,t}^{\top}G_{t}\|_{F}^{2}\Big)+\mathcal{O}(\eta_{2}^{2})\;\;\geq 0,(A7)

since both terms on the right-hand side are nonnegative (squared Frobenius norms).

Therefore, Eqn.by Eqn.([A7](https://arxiv.org/html/2508.19564#A6.E7 "In Proof. ‣ F.1 Proof and Discussion of Proposition 2: Alignment of Bi-LoRA’s ascent direction with previous full gradient ‣ Appendix F Proofs ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models")), the auxiliary update (B_{2},A_{2}) always increases the inner objective in the direction of the full gradient Eqn.Eqn.([A4](https://arxiv.org/html/2508.19564#A6.E4 "In Proof. ‣ F.1 Proof and Discussion of Proposition 2: Alignment of Bi-LoRA’s ascent direction with previous full gradient ‣ Appendix F Proofs ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models")). This proves that the Bi-LoRA perturbation aligns with previous SAM’s perturbation direction. ∎

As reported in several efficient SAM variants, reusing the previous ascent direction can still yield strong generalization (e.g., S^{2} SAM(Ji et al., [2024](https://arxiv.org/html/2508.19564#bib.bib31 "A single-step, sharpness-aware minimization is all you need to achieve efficient and accurate sparse training")) in the sparse training scenario). In general, aligning with the previous, not the current direction, has benefit on efficiency. The reason that aligning with the previous gradient works may be attributed to the low-dimension/low-rank properties of the gradients(Li et al., [2022](https://arxiv.org/html/2508.19564#bib.bib74 "Low dimensional trajectory hypothesis is true: dnns can be trained in tiny subspaces"); Zhao et al., [2024](https://arxiv.org/html/2508.19564#bib.bib75 "GaLore: memory-efficient LLM training by gradient low-rank projection")), especially in later stage of training, or in fine-tuning stage. Due to the low-dimension/low-rank property, the consecutive gradient directions tend to exhibit similarity. And the intuition behind Bi-LoRA is that adversarial perturbations across different steps are "continuous": the optimal adversarial perturbation at iteration t is likely to be close to the optimal perturbation at iteration t+1, given that the weight differences between the continuous steps are small under LoRA optimization. This means we can obtain the adversarial perturbation at step t+1 with a "slight" adjustment based on the perturbation at t. As a result, this method can effectively optimize the sharpness while maintaining the same training efficiency as regular LoRA training.

### F.2 Proof and discussions of Proposition[3](https://arxiv.org/html/2508.19564#Thmproposition3 "Proposition 3 (Bi-LoRA is a Regularized LoRA). ‣ 3 Bi-LoRA: Bi-directional Low-Rank Adaptation ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"): Equivalence between SAM and Bi-LoRA

###### Proof.

SAM solves the min–max objective

\min_{W}\;\max_{\|\epsilon\|_{2}\leq\rho}\;\mathcal{L}(W+\epsilon),(A8)

where \rho>0 controls the perturbation radius.

A first-order Taylor expansion gives

\mathcal{L}(W+\epsilon)\approx\mathcal{L}(W)+\langle\nabla_{W}\mathcal{L}(W),\epsilon\rangle.(A9)

The inner maximization is

\max_{\|\epsilon\|_{2}\leq\rho}\;\langle\nabla_{W}\mathcal{L}(W),\epsilon\rangle,(A10)

which is attained at

\epsilon^{\star}=\rho\,\frac{\nabla_{W}\mathcal{L}(W)}{\|\nabla_{W}\mathcal{L}(W)\|_{2}}.(A11)

Substituting Eqn.([A11](https://arxiv.org/html/2508.19564#A6.E11 "In Proof. ‣ F.2 Proof and discussions of Proposition 3: Equivalence between SAM and Bi-LoRA ‣ Appendix F Proofs ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models")) into Eqn.([A8](https://arxiv.org/html/2508.19564#A6.E8 "In Proof. ‣ F.2 Proof and discussions of Proposition 3: Equivalence between SAM and Bi-LoRA ‣ Appendix F Proofs ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models")) yields the regularized objective

\mathcal{L}(W)+\rho\|\nabla_{W}\mathcal{L}(W)\|_{2}.(A12)

For Bi-LoRA, the weight matrix is decomposed as W=W_{0}+B_{1}A_{1}+B_{2}A_{2}. The training objective is

\min_{B_{1},A_{1}}\;\max_{\begin{subarray}{c}B_{2},A_{2}\\
\mathrm{rank}(B_{2}A_{2})\leq r,\;\|B_{2}A_{2}\|_{F}\leq\rho\end{subarray}}\;\mathcal{L}(W_{0}+B_{1}A_{1}+B_{2}A_{2}),(A13)

where the perturbation is explicitly parameterized as

\tilde{\epsilon}=B_{2}A_{2},\quad\mathrm{rank}(\tilde{\epsilon})\leq r,\quad\|\tilde{\epsilon}\|_{F}\leq\rho.(A14)

At \theta=W_{0}+B_{1}A_{1}, expanding gives

\mathcal{L}(\theta+\tilde{\epsilon})\approx\mathcal{L}(\theta)+\langle\nabla_{\theta}\mathcal{L},\tilde{\epsilon}\rangle.(A15)

Thus the inner maximization becomes

\max_{\begin{subarray}{c}\mathrm{rank}(\tilde{\epsilon})\leq r\\
\|\tilde{\epsilon}\|_{F}\leq\rho\end{subarray}}\;\langle\nabla_{\theta}\mathcal{L},\tilde{\epsilon}\rangle.(A16)

Let the SVD of \nabla_{\theta}\mathcal{L} be

\nabla_{\theta}\mathcal{L}=U\Sigma V^{\top},\qquad\sigma_{1}\geq\sigma_{2}\geq\cdots.(A17)

The optimum of Eqn.([A16](https://arxiv.org/html/2508.19564#A6.E16 "In Proof. ‣ F.2 Proof and discussions of Proposition 3: Equivalence between SAM and Bi-LoRA ‣ Appendix F Proofs ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models")) is achieved at

\tilde{\epsilon}^{\star}=\rho\,U_{r}V_{r}^{\top},(A18)

where U_{r},V_{r} are the top-r singular vectors of \nabla_{\theta}\mathcal{L}, and \mathrm{rank}(\tilde{\epsilon}^{\star})=r with \|\tilde{\epsilon}^{\star}\|_{F}=\rho.

Substituting Eqn.([A18](https://arxiv.org/html/2508.19564#A6.E18 "In Proof. ‣ F.2 Proof and discussions of Proposition 3: Equivalence between SAM and Bi-LoRA ‣ Appendix F Proofs ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models")) into Eqn.([A16](https://arxiv.org/html/2508.19564#A6.E16 "In Proof. ‣ F.2 Proof and discussions of Proposition 3: Equivalence between SAM and Bi-LoRA ‣ Appendix F Proofs ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models")) gives

\langle\nabla_{\theta}\mathcal{L},\tilde{\epsilon}^{\star}\rangle=\rho\sum_{i=1}^{r}\sigma_{i}(\nabla_{\theta}\mathcal{L})=\rho\|\nabla_{\theta}\mathcal{L}\|_{(r)},(A19)

where

\|\nabla_{\theta}\mathcal{L}\|_{(r)}:=\sum_{i=1}^{r}\sigma_{i}(\nabla_{\theta}\mathcal{L})(A20)

is the Ky Fan r-norm.

Hence, Bi-LoRA reduces to the regularized objective

\min_{A_{1},B_{1}}\;\mathcal{L}(W_{0}+B_{1}A_{1})+\rho\|\nabla_{W_{0}+B_{1}A_{1}}\mathcal{L}\|_{(r)}.(A21)

∎

When r covers all singular values, the Ky Fan norm becomes the nuclear norm, and Bi-LoRA recovers the SAM objective. Conversely, SAM specializes to Bi-LoRA when only the sharpest r singular directions are penalized.

And Proposition[3](https://arxiv.org/html/2508.19564#Thmproposition3 "Proposition 3 (Bi-LoRA is a Regularized LoRA). ‣ 3 Bi-LoRA: Bi-directional Low-Rank Adaptation ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models") represents the ideal objective of Bi-LoRA and serves to illustrate our motivation intuitively. In practice, however, we adopt a simultaneous optimization approach formalized in Eqn.([9](https://arxiv.org/html/2508.19564#S3.E9 "In 3 Bi-LoRA: Bi-directional Low-Rank Adaptation ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models")), which does not introduce additional gradient steps and thus maintains efficiency. Experimental results demonstrate that our approximate approach can already significantly improve performance compared to LoRA.

Furthermore, Figure[3](https://arxiv.org/html/2508.19564#S3.F3 "Figure 3 ‣ 3 Bi-LoRA: Bi-directional Low-Rank Adaptation ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models") shows that, the auxiliary LoRA module continues broad exploration, while the primary module converges rapidly to a nearly fixed point; once the primary stabilizes, the auxiliary then begins to converge quickly. This behavior suggests that at each parameter point, there exists a (though inefficient) convergent multi-step scheme for the auxiliary module, which is consistent with the intuition of Proposition[3](https://arxiv.org/html/2508.19564#Thmproposition3 "Proposition 3 (Bi-LoRA is a Regularized LoRA). ‣ 3 Bi-LoRA: Bi-directional Low-Rank Adaptation ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").

## Appendix G Reasons of Using Global Clipping

We clip using the _total_ Frobenius norm over all auxiliary LoRA modules in Bi-LoRA for the following reasons.

*   •
Faithfulness to SAM. The SAM-style inner maximization is defined with a _single_\rho-ball over the full parameter vector, i.e., \max_{\lVert\epsilon\rVert\leq\rho}\mathcal{L}(w+\epsilon). In Bi-LoRA, the perturbation is implemented entirely through the auxiliary LoRA modules, so constraining the _global_ Frobenius norm of all auxiliary modules (Eqn.([8](https://arxiv.org/html/2508.19564#S3.E8 "In 3 Bi-LoRA: Bi-directional Low-Rank Adaptation ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models")) is the faithful analogue of SAM’s original formulation.

*   •
Avoiding scale explosion and instability from per-layer normalization. If each layer is normalized independently, every layer receives a perturbation with norm \rho, so the overall perturbation magnitude roughly scales with the number of adapters. This not only leads to training instability, but also makes the same \rho incomparable across architectures with different depths or adapter placements. SAM’s original global constraint is designed precisely to avoid such architecture-dependent scaling effects.

*   •
Empirical evidence. In Table[5](https://arxiv.org/html/2508.19564#S4.T5 "Table 5 ‣ 4.6 Ablations and Hyperparameter Sensitivity ‣ 4 Experiments ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"), global clipping consistently outperforms both per-layer clipping and no clipping in terms of final performance and training stability under the same \rho.

In summary, global clipping over all auxiliary LoRA modules yields a scale-aware neighborhood that is consistent with SAM’s objective, comparable across architectures, and empirically more stable, which motivates our choice of this clipping scheme.

## Appendix H More baselines with Llama 3.1-8B on instruction following tasks

Table A1: Results of fine-tuning Llama 3.1-8B on instruction-following tasks with more LoRA-variants and SAM-based methods.

Method MMLU DROP HEval BBH Avg
LoRA 63.38±0.39 49.82±0.54 43.15±0.93 42.82±0.27 49.79
PiSSA 63.59±0.14 50.20±0.20 43.86±0.12 43.25±0.51 50.22
DoRA 63.58±0.22 50.53±0.10 44.10±0.73 42.98±0.79 50.30
DeLoRA 63.49±0.32 51.40±0.28 45.53±0.73 42.72±0.29 50.78
HiRA 63.62±0.33 51.17±0.26 45.12±0.61 42.89±0.20 50.70
WSAM 63.42±0.12 50.63±0.14 42.88±0.81 43.16±0.52 50.02
S^{2}-SAM 63.73±0.38 50.80±0.23 43.09±0.20 42.44±0.19 50.02
Bi-LoRA 63.67±0.15 51.53±0.33 46.12±0.89 43.45±0.15 51.19

Table[A1](https://arxiv.org/html/2508.19564#A8.T1 "Table A1 ‣ Appendix H More baselines with Llama 3.1-8B on instruction following tasks ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models") further compares Bi-LoRA with four LoRA variants (PiSSA, DoRA, HiRA, DeLoRA) and two SAM-based methods (WSAM, S^{2}-SAM) on Llama 3.1-8B instruction following tasks. Bi-LoRA attains the highest average performance of 51.19, outperforming the strongest LoRA variant by about 0.4% and vanilla LoRA by roughly 1.4% points.

## Appendix I Results on Natural Language Understanding

Setting. We fine-tune the T5-base model(Raffel et al., [2020](https://arxiv.org/html/2508.19564#bib.bib49 "Exploring the limits of transfer learning with a unified text-to-text transformer")) on multiple datasets from GLUE(Wang et al., [2019b](https://arxiv.org/html/2508.19564#bib.bib51 "GLUE: a multi-task benchmark and analysis platform for natural language understanding")) and SuperGLUE(Wang et al., [2019a](https://arxiv.org/html/2508.19564#bib.bib52 "Superglue: a stickier benchmark for general-purpose language understanding systems")) benchmarks, including MNLI, SST2, CoLA, QNLI, MRPC, BoolQ, CB, COPA, RTE, and WIC, following Wang et al. ([2024](https://arxiv.org/html/2508.19564#bib.bib45 "LoRA-ga: low-rank adaptation with gradient approximation")); Li et al. ([2025](https://arxiv.org/html/2508.19564#bib.bib64 "Flat-lora: low-rank adaptation over a flat loss landscape")). Performance is evaluated on the development set using accuracy as the primary metric, except for CoLA, where the Matthews correlation coefficient is used.

Results. We first focus on the GLUE datasets. From the results in Table[A2](https://arxiv.org/html/2508.19564#A9.T2 "Table A2 ‣ Appendix I Results on Natural Language Understanding ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"), we observe that Bi-LoRA outperforms both LoRA and LoRA-SAM, achieving an average improvement of 0.47% and 0.32%, respectively. It is worth noting that Bi-LoRA achieves these gains with the same training speed as LoRA, whereas LoRA-SAM doubles the training time. Moreover, the gains are more pronounced on smaller datasets, with Bi-LoRA outperforming LoRA by 1.36% on CoLA and 0.82% on MRPC. In contrast, LoRA-SAM shows no clear advantage over vanilla LoRA, indicating its limited ability to enhance generalization due to constrained sharpness optimization.

Next, we evaluate on the SuperGLUE datasets, which feature more challenging language understanding tasks. To reduce variance from the limited evaluation sizes of CB (56), COPA (100), and RTE (277) samples, we increase the number of runs from 3 to 20 for CB and COPA, and to 5 for RTE. Table[A3](https://arxiv.org/html/2508.19564#A9.T3 "Table A3 ‣ Appendix I Results on Natural Language Understanding ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models") shows Bi-LoRA demonstrates more significant advantages over LoRA and LoRA-SAM by 0.69% and 0.60% on average, while LoRA-SAM yields minimal gains and even hurts some datasets (e.g., CB). These results confirm the effectiveness of Bi-LoRA in enhancing generalization.

Table A2: Results on fine-tuning the T5-base model on a subset of GLUE datasets. “Cost” indicates the gradient steps per training iteration, e.g., one step (Cost \times 1) for Full FT, LoRA, and Bi-LoRA.

Dataset Size Cost MNLI (393k)SST2 (67k)CoLA (8.6k)QNLI (105k)MRPC (3.7k)Avg.
Full FT\times 1 85.57±0.09 94.27±0.24 56.60±0.62 93.18±0.09 87.30±0.79 83.38
LoRA\times 1 86.25±0.16 94.23±0.30 59.41±0.52 93.25±0.06 88.56±0.26 84.34
LoRA-SAM\times 2 86.25±0.09 94.46±0.17 59.80±0.85 93.21±0.16 88.73±0.52 84.49
Bi-LoRA\times 1 86.33±0.08 94.34±0.05 60.77±0.39 93.25±0.06 89.38±0.26 84.81

Table A3: Results on fine-tuning T5-base with a subset of SuperGLUE datasets. “Cost” indicates the gradient steps per training iteration, e.g., one step (Cost \times 1) for Full FT, LoRA, and Bi-LoRA.

Dataset Size Cost BoolQ (9.4k)CB (0.25k)COPA (0.4k)RTE (2.5k)WIC (5.4k)Avg.
Full FT\times 1 72.19±0.14 92.26±0.24 64.67±0.26 84.48±0.12 68.34±0.19 76.39
LoRA\times 1 72.20±0.21 92.86±0.27 63.80±0.23 83.10±0.29 68.60±0.32 76.11
LoRA-SAM\times 2 72.47±0.42 92.32±0.28 64.20±0.42 83.47±0.47 68.55±0.76 76.20
Bi-LoRA\times 1 72.25±0.30 92.86±0.32 64.60±0.25 83.61±0.37 70.69±0.40 76.80

## Appendix J Hyperparameter sensitivity

### J.1 Effect of the auxiliary rank on CoLA

![Image 11: Refer to caption](https://arxiv.org/html/2508.19564v2/_figures/effect_rank2_cola_t5base.png)

Figure A1:  Performance on CoLA under varying ranks of primary (r_{1}) and auxiliary (r_{2}) LoRA modules.

To further examine the sensitivity to the auxiliary rank r_{2}, we fine-tune T5-base on the CoLA dataset with r_{1}\in\{4,8,16\} and r_{2}\in\{2,4,8,16,32,64\}.

Figure[A1](https://arxiv.org/html/2508.19564#A10.F1 "Figure A1 ‣ J.1 Effect of the auxiliary rank on CoLA ‣ Appendix J Hyperparameter sensitivity ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models") shows the performance of Bi-LoRA with varying the primary (r_{1}) and auxiliary (r_{2}) LoRA ranks. We observe that integrating the auxiliary module consistently enhances performance. Notably, r_{2}=8 generally delivers good results, which replicates the pattern we reported in Section[4.6](https://arxiv.org/html/2508.19564#S4.SS6 "4.6 Ablations and Hyperparameter Sensitivity ‣ 4 Experiments ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"), confirming that the sweet‑spot generalizes from NLU tasks to instruction‑tuning. Therefore, we suggest using r_{2}=8 as a task-agnostic default.

### J.2 Effect of the neighborhood radius \rho

From Table[A4](https://arxiv.org/html/2508.19564#A10.T4 "Table A4 ‣ J.2 Effect of the neighborhood radius 𝜌 ‣ Appendix J Hyperparameter sensitivity ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"), Bi-LoRA performs consistently well across \rho=\{0.01,0.05,0.1\}, showing its effectiveness to moderate perturbations during training. The ability to achieve strong performance across a range of \rho indicates that Bi-LoRA can effectively adapt to varying levels of perturbation, demonstrating its practicality for diverse applications.

Table A4: Results on CoLA, SST2 and GSM8K under different values of the neighborhood radius \rho.

\rho CoLA SST2 GSM8K
0.01 60.15±0.36 60.15±0.36 59.36
0.05 60.77±0.39 60.15±0.36 59.74
0.1 60.18±0.59 60.15±0.36 60.65
0.2 58.95±0.81 60.15±0.36 58.83

## Appendix K Qualitative Results on SDXL

As discussed in Section[4.4](https://arxiv.org/html/2508.19564#S4.SS4 "4.4 Results on Diffusion Models ‣ 4 Experiments ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"), Bi-LoRA improves both image–text (I2T) and text–text (T2T) CLIP similarity on the 3D Icons dataset. To visualize these gains, Figure[A2](https://arxiv.org/html/2508.19564#A11.F2 "Figure A2 ‣ Appendix K Qualitative Results on SDXL ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models") presents generated images from SDXL models fine-tuned by LoRA and Bi-LoRA under identical promptss.

Observations. Across instances, Bi-LoRA tends to preserve the intended icon identity and style more faithfully. These qualitative differences align with the measured CLIP improvements, i.e., larger average CLIP I2T/T2T scores shown in Table[3](https://arxiv.org/html/2508.19564#S4.T3 "Table 3 ‣ 4.4 Results on Diffusion Models ‣ 4 Experiments ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models") also exhibit clearer textual correspondence in Figure[A2](https://arxiv.org/html/2508.19564#A11.F2 "Figure A2 ‣ Appendix K Qualitative Results on SDXL ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").

![Image 12: Refer to caption](https://arxiv.org/html/2508.19564v2/x2.png)

(a) LoRA - Fluffy Rabbit

![Image 13: Refer to caption](https://arxiv.org/html/2508.19564v2/x3.png)

(b) Bi-LoRA - Fluffy Rabbit

![Image 14: Refer to caption](https://arxiv.org/html/2508.19564v2/x4.png)

(c) LoRA - Curly Cloud

![Image 15: Refer to caption](https://arxiv.org/html/2508.19564v2/x5.png)

(d) Bi-LoRA - Curly Cloud

![Image 16: Refer to caption](https://arxiv.org/html/2508.19564v2/x6.png)

(e) LoRA - Roaring Lion

![Image 17: Refer to caption](https://arxiv.org/html/2508.19564v2/x7.png)

(f) Bi-LoRA - Roaring Lion

![Image 18: Refer to caption](https://arxiv.org/html/2508.19564v2/x8.png)

(g) LoRA - Dripping Water Drop

![Image 19: Refer to caption](https://arxiv.org/html/2508.19564v2/x9.png)

(h) Bi-LoRA - Dripping Water Drop

![Image 20: Refer to caption](https://arxiv.org/html/2508.19564v2/x10.png)

(i) LoRA - Hopping Frog

![Image 21: Refer to caption](https://arxiv.org/html/2508.19564v2/x11.png)

(j) Bi-LoRA - Hopping Frog

![Image 22: Refer to caption](https://arxiv.org/html/2508.19564v2/x12.png)

(k) LoRA - Smiling Sun

![Image 23: Refer to caption](https://arxiv.org/html/2508.19564v2/x13.png)

(l) Bi-LoRA - Smiling Sun

Figure A2: Images generated with SDXL fine-tuned using LoRA (left) and Bi-LoRA (right) on the 3D icon dataset. Each row corresponds to a different prompt in the form “a ToK icon of a <Instance>, in the style of ToK”. Images in the same row are generated with the same random seed for fair comparison.

## Appendix L Convergence Analysis of Main and Auxiliary LoRA Modules

We measure the cosine similarity between the weights of the main and auxiliary LoRA modules and their inal weights on CoLA and SST2 with T5-base. The results indicate that the auxiliary module converges substantially slower than the main one, preserving flexibility for sharpness-aware updates.

![Image 24: Refer to caption](https://arxiv.org/html/2508.19564v2/_figures/cola_wd0.0_lr5e-4_bilora.png)

(a) CoLA: \eta_{2}=\eta_{1}=5e-4

![Image 25: Refer to caption](https://arxiv.org/html/2508.19564v2/_figures/cola_wd0.0_lr5e-3_bilora.png)

(b) CoLA: \eta_{2}=10\eta_{1}=5e-3

![Image 26: Refer to caption](https://arxiv.org/html/2508.19564v2/_figures/sst2_wd0.0_lr5e-4_bilora.png)

(c) SST2: \eta_{2}=\eta_{1}=5e-4

![Image 27: Refer to caption](https://arxiv.org/html/2508.19564v2/_figures/sst2_wd0.0_lr5e-3_bilora.png)

(d) SST2: \eta_{2}=10\eta_{1}=5e-3

Figure A3: Cosine similarity between the main and auxiliary LoRA weights and each final weights during training on CoLA and SST2 using T5-base, under different auxiliary learning rates (\eta_{2}). The auxiliary module (perturbation, blue lines) converges in the final 20% of steps, significantly slower than the main (optimization, red lines) module. A larger \eta_{2}=10\eta_{1} further slows down convergence. Moreover, the auxiliary module’s trajectory remains independent of the main module throughout training (green lines).

## Appendix M Bi-LoRA in Quantized Settings

We apply Bi-LoRA in a quantized setting, where the transition to lower precision can cause the model to jump from a flat region into a sharper region with higher local loss, reducing generalization on unseen data. Following QLoRA(Dettmers et al., [2023](https://arxiv.org/html/2508.19564#bib.bib33 "QLoRA: efficient finetuning of quantized llms")), we quantize the pretrained model to NF4 and adopt the setup described in Section 5.3.

As shown in Table[A5](https://arxiv.org/html/2508.19564#A13.T5 "Table A5 ‣ Appendix M Bi-LoRA in Quantized Settings ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"), Bi-QLoRA outperforms QLoRA and QLoRA-SAM by 2.58% and 2.36% on GSM8K, respectively. This gain is attributed to Bi-LoRA’s ability to maintain flatness under quantization by decoupling optimization and perturbation during training.

Table A5: Performance on GSM8K under quantized NF4 settings. Results are averaged over 3 runs.

Method GSM8K (%)
QLoRA 57.77±0.39
QLoRA-SAM 57.89±0.29
Bi-QLoRA 59.26±0.17

## Appendix N Effect on Adversarial Robustness

Figure[4](https://arxiv.org/html/2508.19564#S3.F4 "Figure 4 ‣ 3 Bi-LoRA: Bi-directional Low-Rank Adaptation ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models") already shows that Bi‑LoRA is more robust than vanilla LoRA against random perturbations applied both in the LoRA subspace and in the full parameter space. We now investigate robustness to adversarial perturbations. Concretely, we fine‑tune T5‑base on CoLA and report Matthew’s Correlation Coefficient. Following (Kim et al., [2022](https://arxiv.org/html/2508.19564#bib.bib4 "Fisher sam: information geometry and sharpness aware minimisation")), we apply the worst‑case parameter perturbation

\theta\rightarrow\theta+\alpha\frac{\nabla_{\theta}\mathcal{L}}{\left\|\nabla_{\theta}\mathcal{L}\right\|},(A22)

evaluating attacks restricted to (i) the LoRA subspace (\theta\in\{B,A\}) and (ii) the full parameter space (\theta=W), with perturbation strength \alpha from 0.5 to 5.0. Results are shown in Table[A6](https://arxiv.org/html/2508.19564#A14.T6 "Table A6 ‣ Appendix N Effect on Adversarial Robustness ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").

In the LoRA subspace, Bi-LoRA consistently suffers smaller performance degradation than vanilla LoRA. LoRA-SAM shows the strongest resilience, which aligns with its explicit optimization of sharpness in this subspace. In the full parameter space, degradation is more substantial across all methods, yet Bi-LoRA still mitigates the impact more effectively than LoRA-SAM.

These findings are consistent with our random perturbation results, demonstrating that Bi-LoRA provides robustness to adversarial parameter perturbations, particularly when the attack targets the full parameter space.

Table A6: Results on CoLA with adversarial parameter perturbation. Perturbation strength \alpha from 0 to 5.0.

Perturb in LoRA space
Method\alpha=0\alpha=0.5\alpha=1.0\alpha=2.5\alpha=4.0\alpha=5.0
LoRA 60.62 59.42 57.47 42.07 24.22 19.19
LoRA-SAM 60.95 59.14 58.44 52.12 41.33 30.61
Bi-LoRA 61.12 59.36 59.14 48.47 33.24 24.20
Perturb in full space
Method\alpha=0\alpha=0.5\alpha=1.0\alpha=2.5\alpha=4.0\alpha=5.0
LoRA 60.62 52.69 44.42 19.08 11.39 6.55
LoRA-SAM 60.95 56.16 45.67 24.77 18.02 13.34
Bi-LoRA 61.12 56.29 49.00 32.91 24.83 18.06

## Appendix O Training and test metric curves

We fine-tune T5-base on MRPC for 10 epochs, tracking both training and eval losses and metrics. The results in [A4](https://arxiv.org/html/2508.19564#A15.F4 "Figure A4 ‣ Appendix O Training and test metric curves ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models") shows nearly identical training losses across methods, yet Bi‑LoRA attains a lower validation loss and higher accuracy. Importantly, the smaller train–val generalization gap demonstrates Bi-LoRA’s improved generalization.

![Image 28: Refer to caption](https://arxiv.org/html/2508.19564v2/_figures/train_loss_plot.png)

(a) Training Loss

![Image 29: Refer to caption](https://arxiv.org/html/2508.19564v2/_figures/eval_loss_plot.png)

(b) Evaluation Loss

![Image 30: Refer to caption](https://arxiv.org/html/2508.19564v2/_figures/train_accuracy_plot.png)

(c) Training Accuracy

![Image 31: Refer to caption](https://arxiv.org/html/2508.19564v2/_figures/eval_accuracy_plot.png)

(d) Evaluation Accuracy

![Image 32: Refer to caption](https://arxiv.org/html/2508.19564v2/_figures/generalization_gap.png)

(e) Generalization Gap

Figure A4: Training and evaluation loss, accuracy, and generalization gap of T5-base fine-tuned on the MRPC dataset.

## Appendix P Compute Resources

We utilize two types of GPUs: the NVIDIA RTX 4090 24GB and the A100 80GB. For NLU and Text-to-Image generation tasks, computations are performed on a single RTX 4090. All other experiments are conducted on a single A100.

## Appendix Q Learning Rates for Tuning the Two Modules in Bi-LoRA

In our experiments, both modules in Bi-LoRA use the same learning rate of 5\times 10^{-4}. Initially, we explored an unequal learning rate strategy. However, results in Table[A7](https://arxiv.org/html/2508.19564#A17.T7 "Table A7 ‣ Appendix Q Learning Rates for Tuning the Two Modules in Bi-LoRA ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models") show that tuning \eta_{2} provides only a limited improvement in performance. Considering the trade-off between gains and the additional effort for hyperparameter tuning, we recommend using the same learning rate for both modules.

Table A7: CoLA accuracy under different learning rate of the auxiliary module \eta_{2} settings (with \eta_{1}=5{\times}10^{-4} for the main LoRA module).

\eta_{2}1\times 10^{-4}5\times 10^{-4} (main)5\times 10^{-3}
CoLA 60.23±0.35 60.77±0.55 60.44±0.87

## Appendix R Empirical Details

This section provides detailed empirical setups used for the previous experiments.

We tune the neighborhood radius \rho over \{0.005, 0.01, 0.05, 0.1, 0.2, 0.5\} for LoRA-SAM and Bi-LoRA and search learning rates among \{2e-5, 5e-5, 1e-4, 2e-4\} for Full FT and \{5e-5, 1e-4, 2e-4, 5e-4, 1e-3\} for LoRA and its variants.

### R.1 Experiments on the GLUE and SuperGLUE datasets

The hyperparameter configurations for all GLUE and SuperGLUE experiments (Sections[I](https://arxiv.org/html/2508.19564#A9 "Appendix I Results on Natural Language Understanding ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"), [4.5](https://arxiv.org/html/2508.19564#S4.SS5 "4.5 Integration with Other LoRA Variants ‣ 4 Experiments ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models") and [4.6](https://arxiv.org/html/2508.19564#S4.SS6 "4.6 Ablations and Hyperparameter Sensitivity ‣ 4 Experiments ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models")) are detailed in Tables[A8](https://arxiv.org/html/2508.19564#A18.T8 "Table A8 ‣ R.1 Experiments on the GLUE and SuperGLUE datasets ‣ Appendix R Empirical Details ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models") and [A9](https://arxiv.org/html/2508.19564#A18.T9 "Table A9 ‣ R.1 Experiments on the GLUE and SuperGLUE datasets ‣ Appendix R Empirical Details ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"). As discussed in Section[I](https://arxiv.org/html/2508.19564#A9 "Appendix I Results on Natural Language Understanding ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"), we set the neighborhood radius \rho to 0.01 and 0.05 for LoRA-SAM and Bi-LoRA, respectively, in the GLUE experiments. For SuperGLUE, we use \rho values of 0.05 and 0.1 for LoRA-SAM and Bi-LoRA, respectively. For full fine-tuning, we set the learning rate to 1e-4 while keeping all other hyperparameters unchanged. We use the same learning rate of 5e-4 for both the primary and auxiliary LoRA modules in Bi-LoRA.

Table A8: Hyperparameter configurations for fine-tuning T5-base using LoRA-based methods on the GLUE datasets.

Hyperparameter MNLI SST-2 COLA QNLI MRPC
Learning Rate 5e-4
Batch Size 32
Epochs 3 3 10 3 10
Max Sequence Length 256
LoRA Rank 8
LoRA Alpha 16
LR Scheduler Cosine
Target Modules All
Warmup Ratio 0.03
Evaluation Metric Accuracy Accuracy Matthews Corr.Accuracy Accuracy

Table A9: Hyperparameter settings for fine-tuning T5-base using LoRA-based methods on the SuperGLUE datasets.

Hyperparameter BoolQ CB COPA RTE WIC
Learning Rate 5e-4
Batch Size 32
Epochs 10 50 50 10 10
Max Sequence Length 256
LoRA Rank 8
LoRA Alpha 16
LR Scheduler Cosine
Target Modules All
Warmup Ratio 0.03
Evaluation Metric Accuracy

### R.2 Experiments on Llama Models

The hyperparameter settings for Llama 2-7B and Llama 3.1-8B are listed in Tables[A10](https://arxiv.org/html/2508.19564#A18.T10 "Table A10 ‣ R.2 Experiments on Llama Models ‣ Appendix R Empirical Details ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models") and [A11](https://arxiv.org/html/2508.19564#A18.T11 "Table A11 ‣ R.2 Experiments on Llama Models ‣ Appendix R Empirical Details ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").

The neighborhood radius \rho is set to 0.01 and 0.05 for LoRA-SAM and Bi-LoRA. For full fine-tuning, we set the micro-batch size to 4, 2, 2, and 8 for the respective four tasks, and the learning rate is set to 5e-5. We use the same learning rate of 5e-4 for both the primary and auxiliary LoRA modules in Bi-LoRA.

Table A10: Hyperparameter configurations for fine-tuning Llama 2-7B on Math, Code, and Chat tasks using LoRA-based methods. “Math”, “Code” and “Chat” correspond to mathematical reasoning, code generation, and dialogue generation tasks, respectively, as described in Section[4.2](https://arxiv.org/html/2508.19564#S4.SS2 "4.2 Results on Llama Models ‣ 4 Experiments ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").

Hyperparameter Math Code Chat
Learning rate 5e-4
Batch Size 32
Micro-Batch size 4
Epochs 2
Max Sequence Length 1024
LoRA Rank 8
LoRA Dropout 0.0
LoRA Alpha 16
Target Modules All
LR Scheduler Cosine
Warmup Ratio 0.03

Table A11: Hyperparameter configurations for fine-tuning Llama 3.1-8B on Cleaned Alpaca dataset.

Hyperparameter Cleaned Alpaca
Learning rate 3e-4
Batch Size 128
Micro-Batch size 4
Epochs 3
Max Sequence Length 256
LoRA Rank 8
LoRA Dropout 0.0
LoRA Alpha 16
Target Modules All
LR Scheduler Cosine
Warmup Ratio 0.03

### R.3 Experiments on Qwen model

We first grid-search the learning rate \in\{1\mathrm{e}-5,5\mathrm{e}-5,1\mathrm{e}-4,3\mathrm{e}-4,5\mathrm{e}-4\} for LoRA, Bi-LoRA and LoRA-SAM, and perturbation radius \rho\in\{0.1,0.05,0.01\}. For all random noise baselines we tune the variance \rho^{2}\in\{0.1,0.01,0.005,0.001\}.

We set the perturbation radius to \rho=0.1 for Bi-LoRA and \rho=0.05 for LoRA-SAM and other SAM variants, use variance \rho^{2}=0.001 for all random-noise baselines. Other hyperparameters are summarized in Table[A12](https://arxiv.org/html/2508.19564#A18.T12 "Table A12 ‣ R.3 Experiments on Qwen model ‣ Appendix R Empirical Details ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models").

Table A12: Hyperparameter configurations for fine-tuning Qwen 2.5-14B on Cleaned Alpaca dataset.

Hyperparameter Cleaned Alpaca
Learning rate 5e-5
Batch Size 128
Micro-Batch size 4
Epochs 3
Max Sequence Length 256
LoRA Rank 8
LoRA Dropout 0.0
LoRA Alpha 16
Target Modules All
LR Scheduler Cosine
Warmup Ratio 0.03

### R.4 Experiments on Diffusion Models

The hyperparameter settings for Diffusion Models are listed in Table[A13](https://arxiv.org/html/2508.19564#A18.T13 "Table A13 ‣ R.4 Experiments on Diffusion Models ‣ Appendix R Empirical Details ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"). We finetune the model for 500 steps with a constant learning rate of 2e-4, and set the \rho for Bi-LoRA to 0.05. The LoRA rank is set to 4, with an auxiliary rank of 4 for Bi-LoRA. We use the same learning rate of 2e-4 for both the primary and auxiliary LoRA modules in Bi-LoRA. The instance prompt used for DreamBooth training is “a TOK icon, in the style of TOK”, and the validation prompt used for generating the specific icons is formatted as "a ToK icon of a <Instance>, in the style of ToK". We follow the scripts implemented by Hugging Face 2 2 2[https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_sdxl.md](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_sdxl.md).

Table A13: Hyperparameter settings for fine-tuning SDXL using Dreambooth on the 3D Icons dataset

Hyperparameter Value
Learning Rate 2e-4
Batch Size 4
Micro-Batch Size 1
Train Steps 500
Resolution 1024
Validation Epochs 25
LoRA Rank r 4
LoRA Dropout 0.0
LoRA Alpha \alpha 8
Target Modules W_{K},W_{Q},W_{V},W_{O}
LR Scheduler Constant
Warmup Steps 0

### R.5 Efficient SAM Variants for LoRA Fine-Tuning

To ensure a fair comparison, we adopt the results of Flat-LoRA from Li et al. ([2025](https://arxiv.org/html/2508.19564#bib.bib64 "Flat-lora: low-rank adaptation over a flat loss landscape")), and follow the default setup of Li et al. ([2024a](https://arxiv.org/html/2508.19564#bib.bib53 "Implicit regularization of sharpness-aware minimization for scale-invariant problems")) for LoRA-oBAR/nBAR with \alpha=0.25. And following Du et al. ([2022a](https://arxiv.org/html/2508.19564#bib.bib59 "Efficient sharpness-aware minimization for improved training of neural networks")), (Liu et al., [2022b](https://arxiv.org/html/2508.19564#bib.bib60 "Towards efficient and scalable sharpness-aware minimization")), LoRA-ESAM perturbs 50% of the parameters on the top-50% sharpness-sensitive data, while LoRA-LookSAM applies SAM perturbation every five steps. All other hyperparameters are aligned with those used in our prior experiments.

## Appendix S Time Comparisons Across Tasks

In this section, we compare the per-step optimization time for LoRA, LoRA-SAM, and Bi-LoRA across a range of benchmarks. The results are organized based on different datasets and tasks, including T5 experiments on the GLUE and SuperGLUE datasets, as well as Llama experiments on domain-specific datasets. All experiments were conducted on a single NVIDIA RTX 4090 GPU 24GB, using the same hyperparameters as in the previous experiments.

Table A14: Optimization time (s) per step for T5-base on GLUE benchmarks. The average time across tasks is also reported.

Dataset MNLI SST2 CoLA QNLI MRPC Avg.
Time
Full FT 0.758 0.538 0.549 0.756 0.681 0.654
LoRA 0.458 0.310 0.273 0.508 0.442 0.398
LoRA-SAM 1.036 0.573 0.546 0.950 1.039 0.829
Bi-LoRA 0.476 0.325 0.305 0.513 0.508 0.425

Table A15: Optimization time (s) per step for T5-base on SuperGLUE benchmarks. The average time across tasks is also reported.

Dataset BoolQ CB COPA RTE WIC Avg.
Time
Full FT 0.622 0.415 0.451 0.689 0.544 0.544
LoRA 0.276 0.276 0.253 0.263 0.286 0.271
LoRA-SAM 0.519 0.534 0.568 0.688 0.484 0.559
Bi-LoRA 0.304 0.285 0.278 0.276 0.262 0.281

Table A16: Optimization time (s) per step for training Llama2 on math (MetaMathQA), code (Code-Feedback) and chat (WizardLM) tasks.

Method MetaMathQA Code-Feedback WizardLM
LoRA 2.81 5.10 5.13
LoRA-SAM 5.52 10.10 10.53
ROP 2.88 5.41 5.24
RWP_full 3.21 6.02 5.88
RWP_LoRA 2.97 5.56 5.38
LoRA-oBAR 2.90 5.41 5.21
LoRA-nBAR 2.89 5.41 5.24
Flat-LoRA 3.42 5.92 5.78
LoRA-ESAM 4.20 8.13 7.35
LoRA-LookSAM 3.47 6.49 6.29
Bi-LoRA 3.12 5.68 5.43

Table A17: Optimization time (s) per step for training Llama 3-8B on the instruction-following dataset Cleaned Alpaca.

Method Cleaned Alpaca
LoRA 8.93
LoRA-SAM 19.23
ROP 9.43
RWP_full 12.35
RWP_LoRA 10.10
LoRA-oBAR 9.35
LoRA-nBAR 9.35
Flat-LoRA 9.90
LoRA-ESAM 13.44
LoRA-LookSAM 11.49
Bi-LoRA 9.71

Table A18: Optimization time (s) per step for training Qwen 2.5-14B on the instruction-following dataset Cleaned Alpaca.

Method Cleaned Alpaca
LoRA 15.47
LoRA-SAM 32.37
ROP 15.87
RWP_full 21.28
RWP_LoRA 16.67
LoRA-oBAR 15.87
LoRA-nBAR 15.87
Flat-LoRA 17.06
LoRA-ESAM 21.79
LoRA-LookSAM 19.07
Bi-LoRA 16.63

## Appendix T Norm and Ratio Analysis Across Layers

In addition to Figures[2(a)](https://arxiv.org/html/2508.19564#S2.F2.sf1 "In Figure 2 ‣ 2.1 Preliminaries ‣ 2 Issues of LoRA-SAM ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models") and [2(b)](https://arxiv.org/html/2508.19564#S2.F2.sf2 "In Figure 2 ‣ 2.1 Preliminaries ‣ 2 Issues of LoRA-SAM ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"), we extend the analysis of LoRA-SAM’s training dynamics to additional self-attention layers across different encoder blocks to verify the consistency of our observations.

![Image 33: Refer to caption](https://arxiv.org/html/2508.19564v2/_figures/t5_cola_norm_layer_25.png)

(a) Norm of different components in the key of the first self-attention layer in the fifth encoder block

![Image 34: Refer to caption](https://arxiv.org/html/2508.19564v2/_figures/t5_cola_norm_ratio_layer_25.png)

(b) Norm ratio of the key of the first self-attention layer in the fifth encoder block

![Image 35: Refer to caption](https://arxiv.org/html/2508.19564v2/_figures/t5_cola_norm_layer_45.png)

(c) Norm of different components in the output projection the second self-attention layer in the eighth encoder block

![Image 36: Refer to caption](https://arxiv.org/html/2508.19564v2/_figures/t5_cola_norm_ratio_layer_45.png)

(d) Norm ratio of the output projection in the second self-attention layer in the eighth encoder block

![Image 37: Refer to caption](https://arxiv.org/html/2508.19564v2/_figures/t5_cola_norm_layer_50.png)

(e) Norm of different components in the value of the first self-attention layer in the ninth encoder block

![Image 38: Refer to caption](https://arxiv.org/html/2508.19564v2/_figures/t5_cola_norm_ratio_layer_50.png)

(f) Norm ratio of the value of the first self-attention layer in the ninth encoder block

Figure A5:  Analyses of training statistics for LoRA-SAM. (a) and (c) and (e): Frobenius norms of different terms. And (b), (d) and (f): ratio of the Frobenius norms of the first two terms (B\epsilon_{A}+\epsilon_{B}A) to that of the third term (\epsilon_{B}\epsilon_{A}) in Eqn.([5](https://arxiv.org/html/2508.19564#S2.E5 "In 2.2 LoRA-SAM ‣ 2 Issues of LoRA-SAM ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models")) during fine-tuning. Models are fine-tuned on CoLA with T5 for 10 epochs.

Figures[5(a)](https://arxiv.org/html/2508.19564#A20.F5.sf1 "In Figure A5 ‣ Appendix T Norm and Ratio Analysis Across Layers ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"), [5(c)](https://arxiv.org/html/2508.19564#A20.F5.sf3 "In Figure A5 ‣ Appendix T Norm and Ratio Analysis Across Layers ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models") and [5(e)](https://arxiv.org/html/2508.19564#A20.F5.sf5 "In Figure A5 ‣ Appendix T Norm and Ratio Analysis Across Layers ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models") show the Frobenius norms of different terms in the key, output projection and value of self-attention layers in the 5th, 8th, and 9th encoder blocks, respectively. And Figures[5(b)](https://arxiv.org/html/2508.19564#A20.F5.sf2 "In Figure A5 ‣ Appendix T Norm and Ratio Analysis Across Layers ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models"), [5(d)](https://arxiv.org/html/2508.19564#A20.F5.sf4 "In Figure A5 ‣ Appendix T Norm and Ratio Analysis Across Layers ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models") and [5(f)](https://arxiv.org/html/2508.19564#A20.F5.sf6 "In Figure A5 ‣ Appendix T Norm and Ratio Analysis Across Layers ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models") present corresponding norm ratios. As observed, the norm of the third higher-order term remains several orders of magnitude smaller than the first two terms, confirming its negligible impact.

## Appendix U Restricting perturbation in the optimization subspace limits LoRA-SAM

We evaluate the performance of LoRA-SAM on Llama 3.1-8B for the instruction following benchmark, sweeping the rank over r\in\{4,8,16,32,64\}. Table[A19](https://arxiv.org/html/2508.19564#A21.T19 "Table A19 ‣ Appendix U Restricting perturbation in the optimization subspace limits LoRA-SAM ‣ Bi-LoRA: Efficient Sharpness-Aware Minimization for Fine-Tuning Large-Scale Models") shows that while the average score rises monotonically with rank, the gain per doubling quickly reduces. The upward trend confirms that restricting perturbation in the optimization subspace does limit LoRA-SAM, and that increasing the rank partially mitigates it. Ideally, if r=\max\{m,n\} for each weight matrix W\in\mathbb{R}^{m\times n}, LoRA-SAM becomes full parameter SAM, and the limitation would disappear. Our results show that even at r=64, LoRA-SAM still lags behind our Bi-LoRA with primary rank =8, auxiliary rank =8, whose wider exploratory subspace is achieved with decoupling the (auxiliary) perturbation and (primary) optimization subspace. Moreover, LoRASAM’s SAM-style perturbation doubles computation, whereas Bi-LoRA keeps the cost close to vanilla LoRA.

Table A19: Instruction-tuning results on fine-tuning Llama 3.1-8B using LoRA-SAM with varying rank.

Rank MMLU DROP HEval BBH Avg.
4 64.16 48.66 40.24 43.37 49.11
8 65.50 50.93 40.85 43.14 50.11
16 66.05 51.92 40.85 42.91 50.43
32 64.45 50.93 44.51 42.43 50.58
64 65.81 50.72 43.90 42.37 50.70

## Appendix V Theoretical Comparison with Other Efficient SAM Variants Designed for LoRA Fine-tuning

We provide a brief theoretical comparison showing the difference between Bi-LoRA and two efficient SAM variants, LoRA-o/nBAR and Flat-LoRA.

Starting from the common SAM objective \min_{W}\max_{\|\epsilon\|\leq\rho}\mathcal{L}(W+\epsilon), all three methods approximate this minimax objective under a low-rank constraint, yet they differ in how the approximation is carried out. LoRAo/nBAR utitlizes SAM’s implicit regularization, it replace the inner maximization with an explicit balanceness penalty, leading to \min_{B_{1},A_{1}}\mathcal{L}\left(W_{0}+BA\right)+\rho\left\|B^{\top}B-AA^{\top}\right\| (take oBAR and \|B\|\geq\|A\| for example) with a looser O\left(\rho^{2}\right) approximation. Flat-LoRA and Bi-LoRA both target LoRA-SAM’s restricted-subspace issue, i.e., the perturbation \epsilon\approx BB^{\top}\left(\nabla_{W}\mathcal{L}\right)+\left(\nabla_{W}\mathcal{L}\right)A^{\top}A is confined to \operatorname{Col}(\mathrm{B}) and \operatorname{Row}(\mathrm{A}). Yet they pursue flatness differently. Flat-LoRA replaces the inner maximisation with a random weight perturbation (RWP) expectation, optimising \min_{B_{1},A_{1}}\mathbb{E}_{\epsilon\sim\mathcal{N}\left(0,\sigma^{2}\right)}\mathcal{L}\left(W_{0}+B_{1}A_{1}+\epsilon\right), thus encouraging expected flatness rather than worst-case sharpness. Bi-LoRA introduces a gradient-driven auxiliary adapter ( B_{2},A_{2} ) so that \min_{B_{1},A_{1}}\max_{\left\|B_{2}A_{2}\right\|\leq\rho}\mathcal{L}\left(W_{0}+B_{1}A_{1}+B_{2}A_{2}\right), allowing a more targeted exploration of the sharp regions through learnable perturbations. From the perspective of gradient-norm regularization, Bi-LoRA essentially replaces SAM’s \ell_{2}-norm constraint with a rank r nuclear norm constraint, thereby inheriting similar theoretical guarantees on generalization and convergence.
