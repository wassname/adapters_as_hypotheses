Title: Efficiently Seeking Flat Minima for Better Generalization in Fine-Tuning Large Language Models and Beyond

URL Source: https://arxiv.org/html/2508.00522

Published Time: Tue, 16 Dec 2025 02:15:18 GMT

Markdown Content:
Jiaxin Deng 1\equalcontrib, Qingcheng Zhu 2\equalcontrib, Junbiao Pang 1, Linlin Yang 3, Zhongqian Fu 4, Baochang Zhang 5

###### Abstract

Little research explores the correlation between the expressive ability and generalization ability of the low-rank adaptation (LoRA). Sharpness-Aware Minimization (SAM) improves model generalization for both Convolutional Neural Networks (CNNs) and Transformers by encouraging convergence to locally flat minima. However, the connection between sharpness and generalization has not been fully explored for LoRA due to the lack of tools to either empirically seek flat minima or develop theoretical methods. In this work, we propose Flat Minima LoRA (FMLoRA) and its efficient version i.e., EFMLoRA, to seek flat minima for LoRA. Concretely, we theoretically demonstrate that perturbations in the full parameter space can be transferred to the low-rank subspace. This approach eliminates the potential interference introduced by perturbations across multiple matrices in the low-rank subspace. Our extensive experiments on large language models and vision-language models demonstrate that EFMLoRA achieves optimize efficiency comparable to that of LoRA while simultaneously attaining comparable or even better performance. For example, on the GLUE dataset with RoBERTa-large, EFMLoRA outperforms LoRA and full fine-tuning by 1.0% and 0.5% on average, respectively. On vision-language models e.g., Qwen-VL-Chat, there are performance improvements of 1.5% and 1.0% on the SQA and VizWiz datasets, respectively. These empirical results also verify that the generalization of LoRA is closely related to sharpness, which is omitted by previous methods.

## Introduction

Parameter-Efficient Fine-Tuning (PEFT) methods only update a small subset of parameters, e.g., adapters (hu2022lora) or prompt weights (li2021prefix) for Large language models (LLMs) with substantially lower memory and computational costs. Specifically, Low-Rank Adaptation (LoRA) (hu2022lora) stands out for achieving performance comparable to full fine-tuning (FT) while being considerably more efficient.

![Image 1: Refer to caption](https://arxiv.org/html/2508.00522v3/x1.png)

Figure 1: Comparison of Methods: LoRA, FMLoRA, and EFMLoRA.

Many works have been proposed to enhance the performance of LoRA by introducing more dedicated budgets for rank allocation (zhang2023adaptive), decomposing optimization for direction and magnitude updates (liu2024dora), or designing better initialization strategies for LoRA parameters (meng2024pissa), etc. These studies demonstrate the significant potential to improve LoRA performance. However, most existing approaches fail to effectively address bias inheritance, where LLMs may propagate and amplify their inherent biases, significantly impacting model performance and robustness on downstream tasks (li2025understanding). Therefore, a natural question is: how to model and understand the generalization of LoRA for various LLMs and beyond, e.g., vision-language models?

It is widely believed that a flatter loss landscape can lead to better generalization performance(hochreiter1994simplifying)(hochreiter1997flat). For instance, Foret et al. proposed Sharpness-Aware Minimization (SAM)(foret-2020-SAM-ICLR), which seeks parameter regions where the training loss remains uniformly flat. SAM and its variants have demonstrated State-Of-The-Art (SOTA) performances across various applications, such as classification(kwon-2021-asam-ICML), transfer learning(zhuang-2022-GSAM-ICLR), domain generalization(dong2024implicit) and federated learning(FedGAMMA).

To the best of our knowledge, compared to theoretical analysis, e.g.,(neyshabur2017exploring), empirically connecting sharpness and generalization ability of LoRA is a practical approach, e.g.,(andriushchenko2023modern). For the second line of research, a naive approach is to combine SAM with LoRA. However, if perturbations in SAM are applied simultaneously to two low-rank subspaces of LoRA, they may change the maximum loss within the neighborhood of LoRA’s full parameter space(dinh2017sharp); besides, SAM incurs a computational cost twice that of Stochastic Gradient Descent (SGD)(deng2024effective). The key question in the second line of research is how to efficiently find flat minima in LoRA, aiming to better understand the connection between sharpness and generalization.

In this paper, we propose a novel PEFT method, FMLoRA, that promotes convergence toward flatter minima. Specifically, we theoretically uncover that perturbations in the full parameter space can be equivalently re-parameterized as perturbations within the low-rank space. In addition, we propose EFMLoRA to accelerate FMLoRA by an Exponential Moving Average (EMA) strategy. We validate that EFMLoRA improves generalization performance on downstream tasks while maintaining computational efficiency comparable to that of LoRA. Fig.[1](https://arxiv.org/html/2508.00522v3#Sx1.F1 "Figure 1 ‣ Introduction ‣ Efficiently Seeking Flat Minima for Better Generalization in Fine-Tuning Large Language Models and Beyond") compares three methods: LoRA, FMLoRA, and EFMLoRA. We conducted comprehensive experiments on diverse tasks (fine-tuning, few-shot learning) and various model types (RoBERTa (liu2019roberta), GPT-2 (radford2019language), CLIP (zanella2024low), Qwen-VL-Chat (Bai2023QwenVLAV)) and scales. We find that EFMLoRA achieves model accuracy very close to, or even surpass both full fine-tuning and LoRA across many tasks. Our main contribution can be summarized as follows:

*   •We propose FMLoRA, a novel PEFT training method that integrates SAM into the LoRA framework. Furthermore, EFMLoRA provides an efficient tool for empirically understanding the connection between sharpness and generalization in LLMs and beyond. We empirically show that reducing sharpness is highly correlated with improved generalization in PEFT tasks, which has been rarely explored in PEFT studies before.
*   •We conduct comprehensive experiments on LLMs (e.g., RoBERTa, GPT-2) and vision-language models (e.g., CLIP, Qwen-VL-Chat) across various tasks including fine-tuning and few-shot learning. Results show that EFMLoRA achieves optimize efficiency comparable to that of LoRA while simultaneously attaining comparable or even better performance.

## Related Works

### Low-rank Adaption

Hu et al. proposed LoRA (hu2022lora) as a PEFT method that introduced low-rank adapters into each layer of a pre-trained model. Recent advancements in LoRA can be broadly categorized into two directions: 1) advanced architectures and 2) optimization methods. In the first research line, for example, LoraHub (huang2023lorahub) trained multiple adapters and strategically combined them based on the domain during inference. LoRA-FA (zhang2023lora) chose to freeze the projection-down weight of \mathbf{A} and update the projection-up weight of \mathbf{B} in each LoRA layer. DoRA (liu2024dora) improved LoRA by incorporating a learnable magnitude vector to re-scale the normalized product of low-rank matrices. HydraLoRA (tian2024hydralora) extended the LoRA framework with an asymmetric architecture that shared a common \mathbf{A} matrix for efficiency while dynamically assigning samples to multiple \mathbf{B} matrices via a MoE mechanism. In the second line, for example, LoRA+ (hayou2024lora+) applied different learning rates to the two low-rank matrices. Additionally, Galore (zhao2024galore) employed SVD to compress the gradients and its first and second momentum of full training into a low-rank space, thereby reducing the memory footprint during pre-training and fine-tuning. Recently, Li et al. (li2024flat) proposed combining SAM with LoRA for better generalization, but they used random perturbation. Our method belongs to the second research line. Different from (li2024flat), our method transfers the perturbation from the full parameter space to a single low-rank parameter space without changing the maximum perturbed loss, avoiding misalignment with SAM’s training behavior.

### Sharpness and Generalization Ability

Research on the relationship between sharpness and generalization could be traced back to (hochreiter1997flat). Following the observation by (keskar-2016-large_batch-ICLR) that larger batch sizes tended to increase sharpness and generalization error. (jastrzkebski2017three) extended this by finding a correlation between the sharpness and the ratio of learning rate to batch size. (dinh-2017-sharp_minima-ICML) showed that one can easily construct networks with good generalization but with arbitrary large sharpness by reparameterization. (jiang-2019-fantastic-ICLR) performed a large-scale empirical study on various generalization measures and showed that sharpness-based measures have the highest correlation with generalization. Theoretical understandings on the generalization error using sharpness-related measures were provided in (neyshabur2017exploring), (wanggeneralization). Collectively, these studies justified the goal of seeking flatter minima to improve generalization. However, to the best of our knowledge, the correlation between sharpness and generalization for LoRA has barely been discussed due to the lack of theoretical understanding or efficient tools for empirical analysis. Our method provides an efficient tool for empirical analysis in this domain.

### Recap of SAM

Foret et al.(foret-2020-SAM-ICLR) proposed the SAM to enhance model generalization as follows:

\displaystyle\mathop{\min}\limits_{\mathbf{w}}[(\mathop{\max}\limits_{||\bm{\varepsilon}||\leq\rho}L(\mathbf{w}+\bm{\varepsilon})-L(\mathbf{w}))+L(\mathbf{w})+\lambda||\mathbf{w}||_{2}^{2}],(1)

where \mathbf{w} represents the weights of the network, \bm{\varepsilon} represents the perturbation of weights \mathbf{w} in a Euclidean ball with the radius \rho(\rho>0), L(\cdot) is the loss function, and \lambda||\mathbf{w}||_{2}^{2} is a standard L2 regularization term.

SAM utilizes Taylor expansion to search for the maximum perturbed loss (\mathop{\max}\limits_{||\bm{\varepsilon}||\leq\rho}L(\mathbf{w}+\bm{\varepsilon})) in local parameter space as follows:

\displaystyle\mathop{\arg\max}\limits_{||\bm{\varepsilon}||\leq\rho}\;L(\mathbf{w}+\bm{\varepsilon})\approx\mathop{\arg\max}\limits_{||\bm{\varepsilon}||\leq\rho}\;{\bm{\varepsilon}^{\top}}{\nabla_{\mathbf{w}}}L(\mathbf{w}).(2)

By solving Eq.([2](https://arxiv.org/html/2508.00522v3#Sx2.E2 "Equation 2 ‣ Recap of SAM ‣ Related Works ‣ Efficiently Seeking Flat Minima for Better Generalization in Fine-Tuning Large Language Models and Beyond")), SAM obtains the perturbation as follows:

\displaystyle\hat{\bm{\varepsilon}}=\rho{\nabla_{\mathbf{w}}}L(\mathbf{w})/||{\nabla_{\mathbf{w}}}L(\mathbf{w})||.(3)

Substituting the perturbation \hat{\bm{\varepsilon}} back into Eq.([1](https://arxiv.org/html/2508.00522v3#Sx2.E1 "Equation 1 ‣ Recap of SAM ‣ Related Works ‣ Efficiently Seeking Flat Minima for Better Generalization in Fine-Tuning Large Language Models and Beyond")), we then have:

\displaystyle{\nabla_{\mathbf{w}}}\mathop{\max}\limits_{||\bm{\varepsilon}||\leq\rho}L(\mathbf{w}+\bm{\varepsilon})\approx{\nabla_{\mathbf{w}}}L({\mathbf{w}}+\hat{\bm{\varepsilon}}({\mathbf{w}}))(4)
\displaystyle={\nabla_{\mathbf{w}}}L({\mathbf{w}}){|_{{\mathbf{w}}+\hat{\bm{\varepsilon}}({\mathbf{w}})}}+\frac{{d\hat{\bm{\varepsilon}}({\mathbf{w}})}}{{d{\mathbf{w}}}}{\nabla_{\mathbf{w}}}L({\mathbf{w}}){|_{{\mathbf{w}}+\hat{\bm{\varepsilon}}({\mathbf{w}})}}.

By dropping the second-order terms in Eq.([4](https://arxiv.org/html/2508.00522v3#Sx2.E4 "Equation 4 ‣ Recap of SAM ‣ Related Works ‣ Efficiently Seeking Flat Minima for Better Generalization in Fine-Tuning Large Language Models and Beyond")), SAM calculates the gradient at \mathbf{w}+\bm{\hat{\varepsilon}} as follows:

\displaystyle{\nabla_{\mathbf{w}}}\mathop{\max}\limits_{||\bm{\varepsilon}||\leq\rho}L(\mathbf{w}+\bm{\varepsilon})\approx{\nabla_{\mathbf{w}}}L(\mathbf{w}){|_{\mathbf{w}+\bm{\hat{\varepsilon}}}}.(5)

Finally, SAM uses the gradients from Eq.([5](https://arxiv.org/html/2508.00522v3#Sx2.E5 "Equation 5 ‣ Recap of SAM ‣ Related Works ‣ Efficiently Seeking Flat Minima for Better Generalization in Fine-Tuning Large Language Models and Beyond")) for optimization.

### SAM Variants

Recently, SAM variants could be broadly categorized into three groups: 1) studies on the perturbation radius \rho in SAM, 2) studies that speed up the optimization process of SAM, and 3) redefinitions of sharpness in SAM. For the first direction, Kwon et al.(kwon-2021-asam-ICML) proposed Adaptive SAM (ASAM), which adapted the perturbation radius in a scale-aware manner, allowing SAM to be effectively applied to scale-invariant neural networks. For the second group, Kim et al.(kim2023exploring) introduced a multi-step ascent approach to improve SAM. Li et al.(li2024friendly) introduced Friendly SAM (F-SAM), which improved generalization by removing the detrimental influence of the full gradient component and instead utilizing batch-specific gradients to guide optimization more effectively. For the third group, Zhuang et al.(zhuang-2022-GSAM-ICLR) pointed out that SAM did not always favor flat minima. Consequently, they proposed GSAM, which minimized the surrogate gap and the perturbed loss to better encourage flatness. Zhang et al. introduced the first-order flatness(zhang-2023-gradient-CVPR), which assessed the maximal gradient norm within a perturbation radius. Consequently, they proposed GAM which explicitly seeks minima characterized by uniformly small curvature.

## Method

### SAM on LoRA

LoRA achieves parameter efficiency by modeling the low-rank decomposed weight(li2022low). Specifically, the weight change for each layer \mathbf{W}_{0}\in\mathbb{R}^{n\times m} is represented as \Delta\mathbf{W}=s\mathbf{B}\mathbf{A}, where s is a scaling factor, \mathbf{B}\in\mathbb{R}^{n\times r}, \mathbf{A}\in\mathbb{R}^{r\times m}, with rank r\ll\min(n,m). Given an input \mathbf{x}, the forward is as follows:

\displaystyle\mathbf{y}=\mathbf{W}_{0}\mathbf{x}+\Delta\mathbf{W}\mathbf{x}=(\mathbf{W}_{0}+s\mathbf{B}\mathbf{A})\mathbf{x},(6)

where matrix \mathbf{A} is typically initialized by the Kaiming’s method(he2015delving), \mathbf{B} is set to zeros. \mathbf{W}_{0} remains unchanged during fine-tuning, while \mathbf{B} and \mathbf{A} are trained. During inference, \Delta\mathbf{W} is merged into \mathbf{W_{0}}.

If SAM is naively combined with LoRA, the optimization loss can be rewritten as follows:

\displaystyle\min_{\mathbf{A},\mathbf{B}}~~\mathop{\max}\limits_{\scriptstyle||{{\bf{E}}^{\bf{A}}}|{|_{F}}\leq\rho,\hfill\atop\scriptstyle||{{\bf{E}}^{\bf{B}}}|{|_{F}}\leq\rho\hfill}L({\mathbf{W_{0}}}+{s}(\mathbf{B}+{\mathbf{E}^{\mathbf{B}}})(\mathbf{A}+{\mathbf{E}^{\mathbf{A}}})),(7)

where \mathbf{E}^{\mathbf{B}}\in\mathbb{R}^{n\times r} and \mathbf{E}^{\mathbf{A}}\in\mathbb{R}^{r\times m} represent the perturbations applied to the parameters \mathbf{B} and \mathbf{A}, respectively, and \rho is the radius of perturbations. There are two key challenges:

*   •Two separate perturbations in two low-rank subspaces interfere with each other, leading to an inconsistency between the maximum loss obtained when perturbing in the low-rank subspaces and the maximum loss obtained when perturbing in the full parameter space.
*   •SAM requires computing gradients twice per iteration, resulting in approximately twice the computational cost compared to LoRA.

### FMLoRA

To deal with the first challenge, we propose to re-parameterize the perturbation from the full parameter space to a single low-rank parameter space. Concretely, the loss in the full parameter space can be formulated as follows:

\displaystyle\min_{\mathbf{A},\mathbf{B}}~~\max_{\|\mathbf{E}^{\mathbf{W}}\|_{F}\leq\rho}~~L(\mathbf{W_{0}}+s\mathbf{B}\mathbf{A}+\mathbf{E}^{\mathbf{W}}).(8)

To solve the minimax problem in Eq.([8](https://arxiv.org/html/2508.00522v3#Sx3.E8 "Equation 8 ‣ FMLoRA ‣ Method ‣ Efficiently Seeking Flat Minima for Better Generalization in Fine-Tuning Large Language Models and Beyond")), it is necessary to first find optimal \hat{\mathbf{E}}^{\mathbf{W}}\in\mathbb{R}^{n\times m}. Analogous to SAM, we approximate the optimal perturbation \hat{\mathbf{E}}^{\mathbf{W}} to maximize L(\mathbf{W}+\mathbf{E}^{\mathbf{W}}) where \mathbf{W}=\mathbf{W_{0}}+s\mathbf{B}\mathbf{A} as follows:

\displaystyle\hat{\bm{\varepsilon}}^{\mathbf{w}}=\rho\text{sign}(\mathbf{g}^{\mathbf{w}})\frac{\mathbf{g}^{\mathbf{w}}}{||\mathbf{g}^{\mathbf{w}}||},(9)

where \mathbf{g}^{\mathbf{w}}=\text{Vector}(\nabla L_{\mathbf{W}}(\mathbf{W})) and \hat{\bm{\varepsilon}}^{\mathbf{w}}=\text{Vector}(\hat{\mathbf{E}}^{\mathbf{W}}), in which the \text{Vector}(\cdot) function represents a vectorized operation. However, the solution for \hat{\mathbf{E}}^{\mathbf{W}} explicitly depends on the gradient of the matrix \mathbf{W}. That is, the form of solution in Eq.([9](https://arxiv.org/html/2508.00522v3#Sx3.E9 "Equation 9 ‣ FMLoRA ‣ Method ‣ Efficiently Seeking Flat Minima for Better Generalization in Fine-Tuning Large Language Models and Beyond")) is undesirable since \nabla L_{\mathbf{W}}(\mathbf{W}) is unknown during LoRA optimization.

In this paper, we propose to approximate the unknown gradient \nabla L_{\mathbf{W}}(\mathbf{W}) using standard LoRA gradients, which can be computed in two ways:

\displaystyle(1)\quad\nabla L_{\mathbf{W}}(\mathbf{W})=\frac{1}{s}\nabla L_{\mathbf{B}}(\mathbf{W_{0}}+s\mathbf{BA})(\mathbf{A}^{\top})^{+},(10)
\displaystyle(2)\quad\nabla L_{\mathbf{W}}(\mathbf{W})=\frac{1}{s}(\mathbf{B}^{\top})^{+}\nabla L_{\mathbf{A}}(\mathbf{W_{0}}+s\mathbf{BA}),(11)

where (\mathbf{A}^{\top})^{+} and (\mathbf{B}^{\top})^{+} represent the pseudo-inverse of \mathbf{A}^{\top} and \mathbf{B}^{\top}, respectively. The accuracy of the pseudo-inverse depends on the condition number of matrix. A smaller condition number leads to a more accurate pseudo-inverse. Matrices with lower condition numbers are better suited for stable representation. In LoRA, we found that the condition number is typically low, around 3.

To obtain a more accurate estimate of the gradient of the full weights, we combine the above two approaches to compute \nabla L_{\mathbf{W}}(\mathbf{W}) as follows:

\displaystyle\overline{\nabla{L}_{\mathbf{W}}(\mathbf{W})}\displaystyle=0.5*(\frac{1}{s}\nabla L_{\mathbf{B}}(\mathbf{W_{0}}+s\mathbf{BA})(\mathbf{A}^{\top})^{+}
\displaystyle+\frac{1}{s}(\mathbf{B}^{\top})^{+}\nabla L_{\mathbf{A}}(\mathbf{W_{0}}+s\mathbf{BA})).(12)

Let {\bar{\mathbf{g}}^{\mathbf{W}}}=\text{Vector}(\overline{\nabla{L}_{\mathbf{W}}(\mathbf{W})}). Then the perturbation in Eq.([9](https://arxiv.org/html/2508.00522v3#Sx3.E9 "Equation 9 ‣ FMLoRA ‣ Method ‣ Efficiently Seeking Flat Minima for Better Generalization in Fine-Tuning Large Language Models and Beyond")) could be rewritten as follows:

\displaystyle\bar{\mathbf{E}}^{\mathbf{W}}=\text{Matrix}(\rho\text{sign}({\bar{\mathbf{g}}^{\mathbf{W}}})\frac{{\bar{\mathbf{g}}^{\mathbf{W}}}}{||{\bar{\mathbf{g}}^{\mathbf{W}}}||}),(13)

where \text{Matrix}(\cdot) denotes the operation that converts a vector into a matrix. We transfer the perturbation from the full parameter space to a single low-rank parameter space without changing the maximum loss in the local region of the parameters. We apply no perturbation to matrix \mathbf{A}, i.e., {\mathbf{E}^{\mathbf{A}}}=\mathbf{0}, and ensure that the loss under perturbations in the low-rank subspace in Eq.([7](https://arxiv.org/html/2508.00522v3#Sx3.E7 "Equation 7 ‣ SAM on LoRA ‣ Method ‣ Efficiently Seeking Flat Minima for Better Generalization in Fine-Tuning Large Language Models and Beyond")) matches the inner maximum loss in Eq.([8](https://arxiv.org/html/2508.00522v3#Sx3.E8 "Equation 8 ‣ FMLoRA ‣ Method ‣ Efficiently Seeking Flat Minima for Better Generalization in Fine-Tuning Large Language Models and Beyond")), as follows:

\displaystyle L({\mathbf{W_{0}}}\displaystyle+{s}(\mathbf{B}+{\mathbf{E}^{\mathbf{B}}})\mathbf{A})(14)
\displaystyle=\max_{\|\mathbf{E}^{\mathbf{W}}\|_{F}\leq\rho}L(\mathbf{W_{0}}+s\mathbf{B}\mathbf{A}+\mathbf{E}^{\mathbf{W}}).

Substituting \bar{\mathbf{E}}^{\mathbf{W}} into Eq.([14](https://arxiv.org/html/2508.00522v3#Sx3.E14 "Equation 14 ‣ FMLoRA ‣ Method ‣ Efficiently Seeking Flat Minima for Better Generalization in Fine-Tuning Large Language Models and Beyond")), we obtain:

\displaystyle{\mathbf{E}}^{\mathbf{B}}\approx\frac{1}{s}\bar{\mathbf{E}}^{\mathbf{W}}\mathbf{A}^{+},(15)

where \mathbf{A}^{+} is the pseudo-inverse of \mathbf{A}. An alternative approach is to transfer the perturbation to matrix \mathbf{A}. Following the observations from HydraLoRA(tian2024hydralora), matrix \mathbf{A} shows high parameter similarity across heads, likely due to initialization, making it capture domain-common features, while matrix \mathbf{B} remains distinct and domain-specific. Since different tasks require different perturbations, we adopt the approach of transferring the perturbation to the matrix \mathbf{B}, as expressed in Eq.([14](https://arxiv.org/html/2508.00522v3#Sx3.E14 "Equation 14 ‣ FMLoRA ‣ Method ‣ Efficiently Seeking Flat Minima for Better Generalization in Fine-Tuning Large Language Models and Beyond")). The detailed derivation of Eq.([10](https://arxiv.org/html/2508.00522v3#Sx3.E10 "Equation 10 ‣ FMLoRA ‣ Method ‣ Efficiently Seeking Flat Minima for Better Generalization in Fine-Tuning Large Language Models and Beyond")) and the pseudo-algorithm for FMLoRA are provided in the supplementary file.

#### Balancedness of FMLoRA.

Balancedness is well-appreciated in domains such as matrix factorization/sensing (ge2017no)(du2018algorithmic). It is also observed that balanced neural networks are easier to optimize relative to unbalanced ones (neyshabur2015path). Recently, Balancedness B_{t}:=\frac{1}{2}(||\mathbf{x}_{t}||^{2}-||\mathbf{y}_{t}||^{2}) (where \mathbf{x}_{t} and \mathbf{y}_{t} are variables) turns out to be an intriguing alternative to sharpness on the scale-invariant problem (li2024implicit).

To investigate the balancedness of our proposed method, we express the update process of FMLoRA analogously to Eq.(4) in (li2024implicit) as follows:

\displaystyle{\tilde{\mathbf{x}}_{t}}={{\mathbf{x}}_{t}}+\rho\frac{1}{s}\frac{{{\mathbf{G}_{t}}}}{{\left\|{{\mathbf{G}_{t}}}\right\|}}{\mathbf{y}_{t}}^{+}\displaystyle,\quad{\tilde{\mathbf{y}}_{t}}={{\mathbf{y}}_{t}},(16)
\displaystyle{\mathbf{g}_{{\tilde{\mathbf{x}}_{t}}}}={{\tilde{\mathbf{G}}}_{t}}\tilde{\mathbf{y}}_{t}\displaystyle,\quad{\mathbf{g}_{{\tilde{\mathbf{y}}_{t}}}}={{\tilde{\mathbf{G}}}_{t}}^{\top}\tilde{\mathbf{x}}_{t},
\displaystyle{{\mathbf{x}}_{t+1}}={{\mathbf{x}}_{t}}-\eta{\mathbf{g}_{{\tilde{\mathbf{x}}_{t}}}}\displaystyle,\quad{{\mathbf{y}}_{t+1}}={{\mathbf{y}}_{t}}-\eta{\mathbf{g}_{{\tilde{\mathbf{y}}_{t}}}},

where {\mathbf{x}}_{t}=\text{Vector}(\mathbf{B}_{t}), {\mathbf{y}}_{t}=\text{Vector}(\mathbf{A}_{t}), {\mathbf{G}_{t}}=\nabla L({\mathbf{x}}_{t}{\mathbf{y}}_{t}^{\top}) is the gradient of the full parameter space at the original parameter point, {\tilde{\mathbf{G}}_{t}}=\nabla L(\tilde{\mathbf{x}}_{t}\tilde{\mathbf{y}}_{t}^{\top}) is the gradient of the full parameter space at the perturbed parameter point, and \mathbf{y}_{t}^{+} is the pseudo inverse of \mathbf{y}_{t}.

###### Theorem 1.

Let B_{t}:=\frac{1}{2}(||\mathbf{x}_{t}||^{2}-||\mathbf{y}_{t}||^{2}). For the learning rate \eta\Rightarrow 0, the limiting flow of FMLoRA guarantees that:

\displaystyle\left|{\frac{1}{2}\frac{{d({{\left\|{{\mathbf{x}_{t}}}\right\|}^{2}}-{{\left\|{{\mathbf{y}_{t}}}\right\|}^{2}})}}{{dt}}}\right|\leq\left|{\rho\frac{1}{s}\frac{1}{{\left\|{\mathbf{y}_{t}}\right\|}}\left\|{{\mathbf{g}_{{{{\rm{\tilde{\mathbf{x}}}}}_{t}}}}}\right\|}\right|.(17)

Theorem 1 indicates that the balancedness of FMLoRA is influenced by the perturbation range \rho, the norm of the gradient at the perturbed point, the \ell_{2}-norm of \mathbf{y}_{t}, and the scale constraint of LoRA. To ensure that the balancedness of FMLoRA gradually decreases during training, we reduce \rho progressively. In addition, the norm of the gradient with respect to \mathbf{y}_{t} at the perturbed point also decreases due to the weight decay. The \ell_{2}-norm of \mathbf{y}_{t} is bounded within a certain range, these factors collectively contribute to the reduction in the balancedness of FMLoRA.

![Image 2: Refer to caption](https://arxiv.org/html/2508.00522v3/x2.png)

Figure 2: Parameter update process for EFMLoRA.

### Efficient FMLoRA

The optimization processes of FMLoRA also require two gradient computations per iteration. To enhance optimization efficiency, we propose Efficient FMLoRA (EFMLoRA), which estimates the subsequent perturbation {\mathbf{E}}^{\mathbf{B}} in Eq.([15](https://arxiv.org/html/2508.00522v3#Sx3.E15 "Equation 15 ‣ FMLoRA ‣ Method ‣ Efficiently Seeking Flat Minima for Better Generalization in Fine-Tuning Large Language Models and Beyond")) by maintaining an Exponential Moving Average (EMA) of previous perturbations as follows:

\displaystyle{{\hat{\mathbf{E}}}^{\mathbf{B}}_{t}}=(1-\beta){{\hat{\mathbf{E}}}^{\mathbf{B}}_{t-1}}+\beta{\mathbf{E}^{\mathbf{B}}_{t}},(18)

where \beta\in(0,1) is the momentum coefficient that determines the update rate of the exponential moving average. {\mathbf{E}^{\mathbf{B}}_{t}} is the perturbation on matrix \mathbf{B}_{t} at t-th iteration, {{\hat{\mathbf{E}}}^{\mathbf{B}}_{t}} is the EMA perturbation at t-th iteration. Fig.[2](https://arxiv.org/html/2508.00522v3#Sx3.F2 "Figure 2 ‣ Balancedness of FMLoRA. ‣ FMLoRA ‣ Method ‣ Efficiently Seeking Flat Minima for Better Generalization in Fine-Tuning Large Language Models and Beyond") illustrates the parameter update process of EFMLoRA: (1) Calculate the gradient at the perturbed point (\mathbf{W}_{0}, \mathbf{B}_{t-1}+\hat{\mathbf{E}}^{\mathbf{B}}_{t-1}, \mathbf{A}_{t-1}). (2) Calculate the perturbation {\mathbf{E}}^{\mathbf{B}}_{t}=\frac{1}{s}\bar{\mathbf{E}}^{\mathbf{W}}\mathbf{A}^{+}_{t-1}. (3) Return to the original parameter point (\mathbf{W}_{0},\mathbf{B}_{t-1},\mathbf{A}_{t-1}). (4) Update the parameters to (\mathbf{W}_{0},\mathbf{B}_{t},\mathbf{A}_{t}). (5) Calculate the EMA perturbation by Eq.([18](https://arxiv.org/html/2508.00522v3#Sx3.E18 "Equation 18 ‣ Efficient FMLoRA ‣ Method ‣ Efficiently Seeking Flat Minima for Better Generalization in Fine-Tuning Large Language Models and Beyond")) and update the parameters to the next perturbed point (\mathbf{W}_{0},\mathbf{B}_{t}+\hat{\mathbf{E}}^{\mathbf{B}}_{t},\mathbf{A}_{t}). During this optimization process, each optimization step requires only a single forward and backward. The algorithmic pseudocode is provided in the supplementary file.

Table 1: Experiments on few-shot RoBERTa (355M). Results marked with * are taken from (li2024implicit).

To theoretically analyze the error of EFMLoRA, some necessary assumptions are listed below, all of which are common and standard when analyzing SAM optimization(du-2022-ESAM-ICLR)(zhuang-2022-GSAM-ICLR).

###### Assumption 1.

(Smooth) L(\mathbf{w}) is \tau-Lipschitz smooth in \mathbf{w}, i.e., \left\|{\nabla L(\mathbf{w})-\nabla L(\mathbf{v})}\right\|\leq\tau\left\|{\mathbf{w}-\mathbf{v}}\right\|.

###### Assumption 2.

(Bounded gradients). By the assumption that an upper bound exists on the gradient of each mini-batch. There exists G>0 for each mini-batch such that \mathbb{E}\left[{\left\|{\nabla L(\mathbf{w})}\right\|}\right]\leq G.

###### Assumption 3.

(Bounded variance of stochastic gradients). Given the training set \mathbf{D} and a mini-batch \mathbf{B}\in\mathbf{D}. There exists \sigma\geq 0, the variance of stochastic gradient L_{\mathbf{B}}(\mathbf{w}) is bounded by \mathbb{E}\left[{{{\left\|{\nabla{L_{\mathbf{B}}}(\mathbf{w})-\nabla{L_{\mathbf{D}}}(\mathbf{w})}\right\|}^{2}}}\right]\leq\sigma^{2}.

###### Assumption 4.

(Convex) We assume that the loss function f:\mathbb{R}^{n}\rightarrow\mathbb{R} is convex and twice differentiable over an open domain. That is, for all x,y\in\text{dom}(f), it satisfies: f(y)\geq f(x)+\nabla f(x)^{\top}(y-x).

This convexity assumption is reasonable in the fine-tuning stage, as the model is typically close to a local minimum and the loss landscape is approximately convex in a local neighborhood (jang2024lora).

###### Theorem 2.

[EMA perturbation approximate perturbation of SAM due to the convex of the loss landscape] Assume that during fine-tuning, the solution is already close to a local minimum and the local loss function is convex. Let the model weights at i-th iteration be \mathbf{w}_{t}. Under Assumptions [1](https://arxiv.org/html/2508.00522v3#Thmassumption1 "Assumption 1. ‣ Efficient FMLoRA ‣ Method ‣ Efficiently Seeking Flat Minima for Better Generalization in Fine-Tuning Large Language Models and Beyond"), [2](https://arxiv.org/html/2508.00522v3#Thmassumption2 "Assumption 2. ‣ Efficient FMLoRA ‣ Method ‣ Efficiently Seeking Flat Minima for Better Generalization in Fine-Tuning Large Language Models and Beyond"), and [3](https://arxiv.org/html/2508.00522v3#Thmassumption3 "Assumption 3. ‣ Efficient FMLoRA ‣ Method ‣ Efficiently Seeking Flat Minima for Better Generalization in Fine-Tuning Large Language Models and Beyond"), let {\rho_{t}}=\frac{{{\rho_{0}}}}{{\sqrt{t}}}, the error between the sharpness calculated using the EMA perturbation (S^{\text{EMA}}) and that calculated using the original SAM perturbation (S^{\text{SAM}}) is bounded as follows:

\displaystyle|\underbrace{\left[{L({\mathbf{w}_{t}}+{\bm{\hat{\varepsilon}}_{t}})-L({\mathbf{w}_{t}})}\right]}_{S^{\text{EMA}}}-\underbrace{\left[{L({\mathbf{w}_{t}}+{\bm{\tilde{\varepsilon}}_{t}})-L({\mathbf{w}_{t}})}\right]}_{S^{\text{SAM}}}|(19)
\displaystyle\leq\left({\left({1+{{(1-\beta)}^{t-1}}}\right)\tau{\rho_{0}}+G+{\sigma^{2}}}\right)
\displaystyle\quad\quad\cdot\left({\left({1+{{(1-\beta)}^{t-1}}}\right){\rho_{0}}+\frac{{{\rho_{0}}}}{{\sqrt{t}}}}\right).

Theorem [2](https://arxiv.org/html/2508.00522v3#Thmtheorem2 "Theorem 2. ‣ Efficient FMLoRA ‣ Method ‣ Efficiently Seeking Flat Minima for Better Generalization in Fine-Tuning Large Language Models and Beyond") demonstrates that as t increases, the difference between S^{\text{EMA}} and S^{\text{SAM}} gradually decreases. The perturbation estimated by the EMA can effectively approximate the original SAM perturbation.

Table 2: Experiments on finetuning RoBERTa (355M). Results marked with \dagger are taken from (hu2022lora), and those with * are taken from (li2024implicit).

Table 3: GPT-2 medium (M) and large (L) with different adaptation methods on the E2E NLG Challenge. Results marked with \dagger are taken from (hu2022lora).

### Memory and Time Complexity

LoRA reduces the number of trainable parameters by decomposing weight updates as \Delta\mathbf{W}\approx\mathbf{B}\mathbf{A}, where \mathbf{B}\in\mathbb{R}^{n\times r} and \mathbf{A}\in\mathbb{R}^{r\times m} with r\ll\min(n,m). Both FMLoRA and EFMLoRA retain this parameter efficiency:

\displaystyle\text{P}_{\text{LoRA}}\displaystyle=\text{P}_{\text{FMLoRA}}=\text{P}_{\text{EFMLoRA}}(20)
\displaystyle=O(nr+rm)\ll O(nm).

However, FMLoRA and EFMLoRA introduce additional memory overhead. Specifically, FMLoRA temporarily stores the original values of \mathbf{B} and \mathbf{A}, as well as the gradients of \mathbf{A}. The memory usage of FMLoRA is calibrated as follows:

\displaystyle\text{M}_{\text{FMLoRA}}=\text{M}_{\text{LoRA}}+O(5\times(nr+rm)),(21)

where \text{M}_{\text{LoRA}} indicates the memory required by LoRA. The memory of EFMLoRA needs to maintain the EMA perturbation on \mathbf{B} as follows:

\displaystyle\text{M}_{\text{EFMLoRA}}=\text{M}_{\text{LoRA}}+O(2\times(nr+rm)).(22)

Notably, modern optimizers like AdamW already require O(2\times(nr+rm)) memory for momentum and second-moment statistics when applied to LoRA.

For time complexity, suppose that the time complexity of optimizing the model with LoRA is O(T), which mainly includes the time for forward and backward. Theoretically, the time complexity of FMLoRA is approximately as follows:

\displaystyle\text{T}_{\text{FMLoRA}}\approx O(2T)=2\times\text{T}_{\text{LoRA}}.(23)

In contrast, the time complexity of EFMLoRA can be approximated as follows:

\displaystyle\text{T}_{\text{EFMLoRA}}\approx O(T)=\text{T}_{\text{LoRA}}.(24)

We implement QR decomposition by Householder transformations, with time complexity of O(r^{2}n) for an r\times n matrix, e.g., r is rank, n is the input dimension in LORA.

## Experiments and Discussions

The best and second-best results are highlighted in bold and underline, respectively. Additional experimental details are provided in the supplementary file.

### Experiments on Large Language Models

Few-shot with RoBERTa-large. We first consider few-shot learning with EFMLoRA. Following the setup of (li2024implicit), we adopt RoBERTa-large—a 355M-parameter language model—as the backbone. The results in Table [1](https://arxiv.org/html/2508.00522v3#Sx3.T1 "Table 1 ‣ Efficient FMLoRA ‣ Method ‣ Efficiently Seeking Flat Minima for Better Generalization in Fine-Tuning Large Language Models and Beyond") show that FMLoRA outperforms all other methods with the highest average score (83.1), particularly excelling on SST-2, SNLI, and MNLI. EFMLoRA follows closely with an average score of 82.3. It consistently surpasses baseline LoRA (+2.3), LoRA-SAM (+1.0), and both BAR variants. These results highlight its superior generalization ability under distribution shift and limited supervision. We conjecture that the performance gap between SAM and EFMLoRA comes from EFMLoRA eliminating the mutual interference between perturbations in the two low-rank subspaces.

Fine-tuning with RoBERTa-large. We apply EFMLoRA to finetune RoBERTa-large. Our implementation follows (hu2022lora), using the same hyperparameters as those in its GitHub repository. The results can be found in Table [2](https://arxiv.org/html/2508.00522v3#Sx3.T2 "Table 2 ‣ Efficient FMLoRA ‣ Method ‣ Efficiently Seeking Flat Minima for Better Generalization in Fine-Tuning Large Language Models and Beyond"). we observe that EFMLoRA achieves the highest scores on all datasets, and achieves the highest accuracy on average over these datasets. Specifically, on average over these datasets, EFMLoRA surpasses standard LoRA with a margin of 1.0. Additionally, EFMLoRA even achieve better performance than full fine-tuning on some datasets. This superior performance may be attributed to overfitting in full fine-tuning, where optimizing all model parameters can lead to overfitting on the training data, thus reducing the model’s generalization to the test set. This effect is particularly pronounced on small datasets, such as MRPC, which contains only 3.7k training data.

Fine-tuning with GPT-2. Having shown that FMLoRA is effective for NLU tasks, we now explore whether EFMLoRA can improve LoRA in NLG models like GPT-2 Medium and Large (radford2019language). To enable a direct comparison, we adopt the experimental setup of (li2021prefix) with minimal deviation. Table[3](https://arxiv.org/html/2508.00522v3#Sx3.T3 "Table 3 ‣ Efficient FMLoRA ‣ Method ‣ Efficiently Seeking Flat Minima for Better Generalization in Fine-Tuning Large Language Models and Beyond") demonstrates the effectiveness of EFMLoRA on the E2E NLG Challenge (novikova2017e2e) with GPT-2 Medium and Large models. Compared with existing PEFT methods such as Adapter and LoRA, EFMLoRA consistently achieves superior performance across all metrics. Notably, it achieves this improvement without increasing the number of trainable parameters, maintaining the same efficiency as standard LoRA.

### Experiments on Vision Language Models

Few-shot with CLIP. Recent advances in few-shot adaptation of Vision-Language Models (VLMs) have significantly enhanced their generalization. CLIP-LoRA (zanella2024low) explores the application of LoRA in this few-shot VLM setting. In our work, we also apply FMLoRA and EFMLoRA to VLMs to evaluate their effectiveness. For a fair comparison, our experimental setup follows that of CLIP-LoRA. We consider five datasets for fine-grained classification of satellite imagery (EuroSAT (helber2019eurosat), Ox-fordPets (parkhi2012cats), Flower102 (nilsback2008automated), Caltech101 (fei2004learning), DTD (cimpoi2014describing)). These datasets offer a thorough benchmarking framework for evaluating few-shot visual classification tasks. Table[4](https://arxiv.org/html/2508.00522v3#Sx4.T4 "Table 4 ‣ Experiments on Vision Language Models ‣ Experiments and Discussions ‣ Efficiently Seeking Flat Minima for Better Generalization in Fine-Tuning Large Language Models and Beyond") demonstrates that FMLoRA and EFMLoRA outperformed Adapter and LoRA in most settings. In the low-data regimes (1-shot and 4-shot), EFMLoRA shows clear advantages. These results highlight the effectiveness of EFMLoRA in improving generalization in few-shot adaptation of vision-language models.

Table 4: Detailed results for five datasets with CLIP-Adapter, CLIP-LoRA and EFMLoRA.

Fine-tuning with Qwen-VL-Chat. Qwen-VL-Chat (Bai2023QwenVLAV) is a multimodal conversational large language model capable of understanding both images and text. We apply EFMLoRA to fine-tune Qwen-VL-Chat, following the same experimental setup as in (zhou2024empirical). Table [5](https://arxiv.org/html/2508.00522v3#Sx4.T5 "Table 5 ‣ Experiments on Vision Language Models ‣ Experiments and Discussions ‣ Efficiently Seeking Flat Minima for Better Generalization in Fine-Tuning Large Language Models and Beyond") presents the results on the ScienceQA (lu2022learn) and VizWiz (gurari2018vizwiz) datasets. The results in Table [5](https://arxiv.org/html/2508.00522v3#Sx4.T5 "Table 5 ‣ Experiments on Vision Language Models ‣ Experiments and Discussions ‣ Efficiently Seeking Flat Minima for Better Generalization in Fine-Tuning Large Language Models and Beyond") demonstrate that the perturbation size \rho significantly influences the performance of EFMLoRA when fine-tuning Qwen-VL-Chat. By tuning \rho, EFMLoRA adapts to different tasks, enabling improved generalization—achieving higher accuracy than LoRA. Specifically, a larger \rho (e.g., \rho=0.2) yields the best accuracy on ScienceQA, while a smaller \rho (e.g., \rho=0.05) performs better on VizWiz. This suggests that different tasks benefit from different levels of perturbation. Therefore, selecting an appropriate \rho based on the task characteristics is crucial for achieving optimal fine-tuning performance on multimodal large language models.

Table 5: EFMLoRA Fine-Tuning Results on Qwen-VL-Chat with different \rho.

Table 6: Runtime (Hour) and memory (GB) of LoRA, FMLoRA and EFMLoRA on fine-tuning GPT-2 Medium/Large.

### Runtime and Memory Consumption

The results in Table[6](https://arxiv.org/html/2508.00522v3#Sx4.T6 "Table 6 ‣ Experiments on Vision Language Models ‣ Experiments and Discussions ‣ Efficiently Seeking Flat Minima for Better Generalization in Fine-Tuning Large Language Models and Beyond") confirm the theoretical time complexity analysis. As expected, FMLoRA has approximately double the runtime of LoRA (2.1× on both GPT-2 Medium and Large), consistent with its theoretical complexity of O(2T) due to two forward and backward passes for sharpness optimization. In contrast, EFMLoRA operates with near-LoRA efficiency, requiring only 1.1× and 1.2× more time on GPT-2 Medium and Large, respectively. This supports the theoretical claim that EFMLoRA maintains a time complexity close to O(T) while benefiting from sharpness-aware optimization. In addition, EFMLoRA maintains a memory usage almost identical to that of LoRA, with only negligible increases (less than 0.4 GB across both model scales). These results demonstrate that EFMLoRA achieves near-LoRA efficiency in both memory and runtime.

### Conclusion

In this work, we propose FMLoRA, a novel PEFT method that integrates sharpness-aware optimization into the LoRA framework to promote convergence toward flatter minima. We theoretically demonstrate that perturbations in the full parameter space can be equivalently represented within the low-rank subspace. To improve computational efficiency, we introduce EFMLoRA, which leverages an exponential moving average to approximate perturbations, significantly reducing runtime overhead while maintaining effectiveness. Extensive experiments across various large language and vision-language models demonstrate that EFMLoRA achieves comparable or even superior generalization performance to full fine-tuning and LoRA. Our results emphasize the importance of reducing sharpness to improve generalization in PEFT methods, offering valuable insights and practical tools for future research on the link between sharpness and generalization in LLMs and beyond.

## A. Proofs

### A.1 Proof of Eq.(10) and Eq.(11)

###### Proof.

we propose to approximate the unknown gradient \nabla L_{\mathbf{W}}(\mathbf{W}) using standard LoRA gradients, which can be computed in two ways:

\displaystyle(1)\nabla L_{\mathbf{B}}\displaystyle(\mathbf{W_{0}}+s\mathbf{BA})=s\nabla L_{\mathbf{W}}(\mathbf{W})\mathbf{A}^{\top}
\displaystyle\Rightarrow\quad\nabla L_{\mathbf{W}}(\mathbf{W})=\frac{1}{s}\nabla L_{\mathbf{B}}(\mathbf{W_{0}}+s\mathbf{BA})(\mathbf{A}^{\top})^{+},(25)
\displaystyle(2)\nabla L_{\mathbf{A}}\displaystyle(\mathbf{W_{0}}+s\mathbf{BA})=s\mathbf{B}^{\top}\nabla L_{\mathbf{W}}(\mathbf{W})
\displaystyle\Rightarrow\quad\nabla L_{\mathbf{W}}(\mathbf{W})=\frac{1}{s}(\mathbf{B}^{\top})^{+}\nabla L_{\mathbf{A}}(\mathbf{W_{0}}+s\mathbf{BA}),(26)

∎

### A.2 Proof of Theorem 1

###### Proof.

The update process of the FMLoRA is as follows:

\displaystyle{\tilde{\mathbf{x}}_{t}}={{\mathbf{x}}_{t}}+\rho\frac{1}{s}\frac{{{\mathbf{G}_{t}}}}{{\left\|{{\mathbf{G}_{t}}}\right\|}_{F}}{\mathbf{y}_{t}}^{+}\displaystyle,\quad{\tilde{\mathbf{y}}_{t}}={{\mathbf{y}}_{t}}(27)
\displaystyle{\mathbf{g}_{{\tilde{\mathbf{x}}_{t}}}}={{\tilde{\mathbf{G}}}_{t}}\tilde{\mathbf{y}}_{t}\displaystyle,\quad{\mathbf{g}_{{\tilde{\mathbf{y}}_{t}}}}={{\tilde{\mathbf{G}}}_{t}}^{\top}\tilde{\mathbf{x}}_{t}
\displaystyle{{\mathbf{x}}_{t+1}}={{\mathbf{x}}_{t}}-\eta{\mathbf{g}_{{\tilde{\mathbf{x}}_{t}}}}\displaystyle,\quad{{\mathbf{y}}_{t+1}}={{\mathbf{y}}_{t}}-\eta{\mathbf{g}_{{\tilde{\mathbf{y}}_{t}}}}

where {\mathbf{x}}_{t}=\text{Vector}(\mathbf{B}_{t}) is the vectorized form of matrix \mathbf{B}_{t}, {\mathbf{y}}_{t} is the vectorized form of matrix \mathbf{A}_{t}, {\mathbf{G}_{t}}=\nabla L({\mathbf{x}}_{t}{\mathbf{y}}_{t}^{\top}) is the gradient of the full parameter space at the original point during gradient descent, {\tilde{\mathbf{G}}_{t}}=\nabla L(\tilde{\mathbf{x}}_{t}\tilde{\mathbf{y}}_{t}^{\top}) is the gradient of the full parameter space at the perturbed point, and \mathbf{y}^{+} is the pseudo inverse of \mathbf{y}. Let balancedness B_{t}:=\frac{1}{2}(||\mathbf{x}_{t}||^{2}-||\mathbf{y}_{t}||^{2}). Then, we have that:

\displaystyle\frac{1}{2}\frac{{d({{\left\|{{\mathbf{x}_{t}}}\right\|}^{2}}-{{\left\|{{\mathbf{y}_{t}}}\right\|}^{2}})}}{{dt}}(28)
\displaystyle=\frac{1}{2}\frac{{d({{\left\|{{\mathbf{x}_{t}}}\right\|}^{2}})}}{{dt}}-\frac{1}{2}\frac{{d({{\left\|{{\mathbf{y}_{t}}}\right\|}^{2}})}}{{dt}}
\displaystyle={\mathbf{x}_{t}}^{\top}\frac{{d{\mathbf{x}_{t}}}}{{dt}}-{{\mathbf{y}}_{t}}^{\top}\frac{{d{\mathbf{y}_{t}}}}{{dt}}
\displaystyle=-{\mathbf{x}_{t}}^{\top}({\mathbf{\tilde{G}}_{t}}{\mathbf{y}_{t}})+({\mathbf{y}_{t}}^{\top}({\mathbf{\tilde{G}}_{t}}^{\top}({{\mathbf{x}}_{t}}+\rho\frac{1}{s}\frac{{{\mathbf{G}_{{t}}}}}{{{{\left\|{\mathbf{G}_{t}}\right\|}_{F}}}}\mathbf{y}_{t}^{+})))
\displaystyle=-{\mathbf{x}_{t}}^{\top}({\mathbf{\tilde{G}}_{t}}{\mathbf{y}_{t}})+({\mathbf{y}_{t}}^{\top}({\mathbf{\tilde{G}}_{t}}^{\top}{\mathbf{{x}}_{t}}+\rho\frac{1}{s}{\mathbf{\tilde{G}}_{t}}^{\top}\frac{{{\mathbf{G}_{{t}}}}}{{{{\left\|{{\mathbf{G}_{t}}}\right\|}_{F}}}}\mathbf{y}_{t}^{+}))
\displaystyle=-{\mathbf{x}_{t}}^{\top}{\mathbf{\tilde{G}}_{t}}{\mathbf{y}_{t}}+({\mathbf{x}_{t}}^{\top}{\mathbf{\tilde{G}}_{t}}{\mathbf{y}_{t}}){{}^{\top}}+\rho\frac{1}{s}{\mathbf{y}_{t}}^{\top}{\mathbf{\tilde{G}}_{t}}^{\top}\frac{{{\mathbf{G}_{{t}}}}}{{{{\left\|{{\mathbf{G}_{t}}}\right\|}_{F}}}}\mathbf{y}_{t}^{+}
\displaystyle=\rho\frac{1}{s}{\mathbf{y}_{t}}^{\top}{\mathbf{\tilde{G}}_{t}}^{\top}\frac{{{\mathbf{G}_{{t}}}}}{{{{\left\|{{\mathbf{G}_{t}}}\right\|}_{F}}}}\mathbf{y}_{t}^{+}
\displaystyle=\rho\frac{1}{s}\frac{1}{{{{\left\|{{\mathbf{G}_{t}}}\right\|}_{F}}}}\left[{{\mathbf{y}_{t}}^{\top}{\mathbf{\tilde{G}}_{t}}^{\top}{\mathbf{G}_{{t}}}\mathbf{y}_{t}^{+}}\right]

Because \frac{1}{{{s}}}\mathbf{g_{x}}={\mathbf{G}_{{t}}}{\mathbf{y}_{t}} and {\mathbf{g}_{{{{\mathbf{\tilde{x}}}}_{t}}}}={\mathbf{\tilde{G}}_{t}}{\mathbf{\tilde{y}}_{t}}, we have:

\displaystyle\frac{1}{2}\frac{{d({{\left\|{{\mathbf{x}_{t}}}\right\|}^{2}}-{{\left\|{{\mathbf{y}_{t}}}\right\|}^{2}})}}{{dt}}(29)
\displaystyle=\rho\frac{1}{s}\frac{1}{{{{\left\|{{\mathbf{G}_{t}}}\right\|}_{F}}}}\left[{{\mathbf{y}_{t}}^{\top}{\mathbf{\tilde{G}}_{t}}^{\top}{\mathbf{G}_{{t}}}\mathbf{y}_{t}^{+}}\right]
\displaystyle=\rho\frac{1}{{{s^{2}}}}\frac{1}{{{{\left\|{{\mathbf{G}_{t}}}\right\|}_{F}}}}\left[{{{({\mathbf{\tilde{G}}_{t}}{\mathbf{y}_{t}})}^{\top}}\mathbf{{g}_{x}}{{(\mathbf{y}_{t}^{\top})}^{+}}\mathbf{y}_{t}^{+}}\right]
\displaystyle=\rho\frac{1}{{{s^{2}}}}\frac{1}{{{{\left\|{{\mathbf{G}_{t}}}\right\|}_{F}}}}\left[{{{({\mathbf{\tilde{G}}_{t}}{{}\mathbf{y}_{t}})}^{\top}}\mathbf{g_{x}}{{(\mathbf{y}_{t}^{+})}^{\top}}\mathbf{y}_{t}^{+}}\right]
\displaystyle=\rho\frac{1}{{{s^{2}}}}\frac{1}{{{{\left\|{{\mathbf{G}_{t}}}\right\|}_{F}}}}\left[{{{({\mathbf{\tilde{G}}_{t}}{\mathbf{y}_{t}})}^{\top}}\mathbf{g_{x}}{{\left\|{\mathbf{y}_{t}^{+}}\right\|}^{2}}}\right]
\displaystyle=\rho\frac{1}{{{s^{2}}}}\frac{1}{{{{\left\|{{\mathbf{G}_{t}}}\right\|}_{F}}}}\left[{{{({\mathbf{g}_{{{\mathbf{{\tilde{x}}}}_{t}}}}({\mathbf{\tilde{y}}_{t}}^{+})^{\top}{\mathbf{y}_{t}})}^{\top}}\mathbf{g_{x}}{{\left\|{\mathbf{y}_{t}^{+}}\right\|}^{2}}}\right]
\displaystyle=\rho\frac{1}{{{s^{2}}}}\frac{1}{{{{\left\|{{\mathbf{G}_{t}}}\right\|}_{F}}}}\left[{{\mathbf{g}_{{{{\mathbf{\tilde{x}}}}_{t}}}}^{\top}\mathbf{g_{x}}{{\left\|{\mathbf{y}_{t}^{+}}\right\|}^{2}}}\right]

Taking the absolute value of balancedness B_{t} gives:

\displaystyle\left|{\frac{1}{2}\frac{{d({{\left\|{{\mathbf{x}_{t}}}\right\|}^{2}}-{{\left\|{{\mathbf{y}_{t}}}\right\|}^{2}})}}{{dt}}}\right|(30)
\displaystyle=\left|{\rho\frac{1}{s}\frac{{{{\left\|{\mathbf{y}_{t}^{+}}\right\|}^{2}}}}{{{{\left\|{{\mathbf{{g}}_{\mathbf{x}}}{{(\mathbf{y}_{t}^{+})}^{\top}}}\right\|}_{F}}}}({{\mathbf{g}_{{{{\mathbf{\tilde{x}}}}_{t}}}}^{\top}\mathbf{g_{x}}})}\right|
\displaystyle=\left|{\rho\frac{1}{s}\frac{{{{\left\|{\mathbf{y}_{t}^{+}}\right\|}^{2}}}}{{\left\|{\mathbf{{g}_{x}}}\right\|\left\|{{{(\mathbf{y}_{t}^{+})}^{\top}}}\right\|}}({{\mathbf{g}_{{{{\mathbf{\tilde{x}}}}_{t}}}}^{\top}\mathbf{g_{x}}})}\right|
\displaystyle=\left|{\rho\frac{1}{s}\frac{{\left\|{\mathbf{y}_{t}^{+}}\right\|}}{{\left\|{\mathbf{{g}_{x}}}\right\|}}({{\mathbf{g}_{{{{\mathbf{\tilde{x}}}}_{t}}}}^{\top}\mathbf{g_{x}}})}\right|
\displaystyle\leq\left|{\rho\frac{1}{s}\frac{{\left\|{\mathbf{y}_{t}^{+}}\right\|}}{{\left\|{\mathbf{g_{x}}}\right\|}}\left\|{{\mathbf{g}_{{{{\mathbf{\tilde{x}}}}_{t}}}}}\right\|\left\|{\mathbf{{{g}}_{x}}}\right\|}\right|
\displaystyle=\left|{\rho\frac{1}{s}\left\|{\mathbf{y}_{t}^{+}}\right\|\left\|{{\mathbf{g}_{{{{\mathbf{\tilde{x}}}}_{t}}}}}\right\|}\right|
\displaystyle=\left|{\rho\frac{1}{s}\left\|{\frac{{\mathbf{y}_{t}^{\top}}}{{{{\left\|{\mathbf{y}_{t}}\right\|}^{2}}}}}\right\|\left\|{{\mathbf{g}_{{{{\mathbf{\tilde{x}}}}_{t}}}}}\right\|}\right|
\displaystyle=\left|{\rho\frac{1}{s}\frac{1}{{\left\|{\mathbf{y}_{t}}\right\|}}\left\|{{\mathbf{g}_{{{{\mathbf{\tilde{x}}}}_{t}}}}}\right\|}\right|

The proof is thus completed. ∎

###### Lemma 1.

Let A_{t+1}=\alpha A_{t}+\beta with some \alpha\in(0,1), then we have

A_{t+1}\leq\alpha^{t+1}A_{0}+\frac{\beta}{1-\alpha}.

###### Proof.

The proof can be completed by simply unrolling A_{t+1} and using the fact 1+\alpha+\alpha^{2}+\dots+\alpha^{t}\leq\frac{1}{1-\alpha}. ∎

### A.3 Proof of Theorem 2

###### Proof.

Assume that \bm{\varepsilon}_{t} is the perturbation at time step t, and \bm{\hat{\varepsilon}}_{t-1} is the EMA perturbation from the previous step. Let \nabla L(\mathbf{w}_{t}+\bm{\hat{\varepsilon}}_{t-1}) denote the gradient used for updating at time t. The standard SAM perturbation at step t is defined as \bm{\tilde{\varepsilon}}_{t}=\rho_{t}\frac{\nabla L(\mathbf{w}_{t})}{\|\nabla L(\mathbf{w}_{t})\|}, and the EMA perturbation at step t is computed as \bm{\hat{\varepsilon}}_{t}=(1-\beta)\bm{\hat{\varepsilon}}_{t-1}+\beta\bm{\varepsilon}_{t}. Based on Assumption 4, we have that:

\displaystyle\left[{L({\mathbf{w}_{t}}+{\bm{\hat{\varepsilon}}_{t-1}})-L({\mathbf{w}_{t}})}\right]-\left[{L({\mathbf{w}_{t}}+{\bm{\tilde{\varepsilon}}_{t}})-L({\mathbf{w}_{t}})}\right](31)
\displaystyle=L({\mathbf{w}_{t}}+{\bm{\hat{\varepsilon}}_{t-1}})-L({\mathbf{w}_{t}}+{\bm{\tilde{\varepsilon}}_{t}})
\displaystyle\leq-\nabla L{({\mathbf{w}_{t}}+{\bm{\hat{\varepsilon}}_{t-1}})^{\top}}({\mathbf{w}_{t}}+{\bm{\tilde{\varepsilon}}_{t}}-{\mathbf{w}_{t}}-{\bm{\hat{\varepsilon}}_{t-1}})
\displaystyle=\nabla L{({\mathbf{w}_{t}}+{\bm{\hat{\varepsilon}}_{t-1}})^{\top}}({\bm{\hat{\varepsilon}}_{t-1}}-{\bm{\tilde{\varepsilon}}_{t}})
\displaystyle\leq\left|{\nabla L{{({\mathbf{w}_{t}}+{\bm{\hat{\varepsilon}}_{t-1}})}^{\top}}({\bm{\hat{\varepsilon}}_{t-1}}-{\bm{\tilde{\varepsilon}}_{t}})}\right|
\displaystyle\leq\left\|{\nabla L({\mathbf{w}_{t}}+{\bm{\hat{\varepsilon}}_{t-1}})}\right\|\left\|{{\bm{\hat{\varepsilon}}_{t-1}}-{\bm{\tilde{\varepsilon}}_{t}}}\right\|(32)

For the first term \left\|{\nabla L({\mathbf{w}_{t}}+{\bm{\hat{\varepsilon}}_{t-1}})}\right\| in Eq.([32](https://arxiv.org/html/2508.00522v3#Sx5.E32 "Equation 32 ‣ A.3 Proof of Theorem 2 ‣ A. Proofs ‣ Efficiently Seeking Flat Minima for Better Generalization in Fine-Tuning Large Language Models and Beyond")), Based on Assumption 1, Assumption 2 and Lemma 1, we have:

\displaystyle\left\|{\nabla L({\mathbf{w}_{t}}+{\bm{\hat{\varepsilon}}_{t-1}})}\right\|(33)
\displaystyle=\left\|{\nabla L({\mathbf{w}_{t}}+{\bm{\hat{\varepsilon}}_{t-1}})-\nabla L({\mathbf{w}_{t}})+\nabla L({\mathbf{w}_{t}})}\right\|
\displaystyle\leq\left\|{\nabla L({\mathbf{w}_{t}}+{\bm{\hat{\varepsilon}}_{t-1}})-\nabla L({\mathbf{w}_{t}})}\right\|+\left\|{\nabla L({\mathbf{w}_{t}})}\right\|
\displaystyle\leq\tau\left\|{{\mathbf{w}_{t}}+{\bm{\hat{\varepsilon}}_{t-1}}-{\mathbf{w}_{t}}}\right\|+\left\|{\nabla L({\mathbf{w}_{t}})}\right\|
\displaystyle=\tau\left\|{{\bm{\hat{\varepsilon}}_{t-1}}}\right\|+\left\|{\nabla L({\mathbf{w}_{t}})-\nabla{L_{\rm{D}}}({\mathbf{w}_{t}})+\nabla{L_{\rm{D}}}({\mathbf{w}_{t}})}\right\|
\displaystyle=\tau\left\|{{\bm{\hat{\varepsilon}}_{t-1}}}\right\|+\left\|{\nabla{L_{\rm{D}}}({\mathbf{w}_{t}})}\right\|+{\sigma^{2}}
\displaystyle=\tau\left\|{(1-\beta){\bm{\hat{\varepsilon}}_{t-2}}+\beta\bm{\varepsilon}_{t-1}}\right\|+G+{\sigma^{2}}
\displaystyle\leq\tau((1-\beta)\left\|{{\bm{\hat{\varepsilon}}_{t-2}}}\right\|+\beta{\rho_{0}})+G+{\sigma^{2}}
\displaystyle\leq\tau{(1-\beta)^{t-1}}\left\|{{\bm{\hat{\varepsilon}}_{0}}}\right\|+\tau{\rho_{0}}+G+{\sigma^{2}}

For the second term \left\|{{\bm{\hat{\varepsilon}}_{t-1}}-{\bm{\tilde{\varepsilon}}_{t}}}\right\| in Eq.([32](https://arxiv.org/html/2508.00522v3#Sx5.E32 "Equation 32 ‣ A.3 Proof of Theorem 2 ‣ A. Proofs ‣ Efficiently Seeking Flat Minima for Better Generalization in Fine-Tuning Large Language Models and Beyond")), we have:

\displaystyle\left\|{{\bm{\hat{\varepsilon}}_{t-1}}-{\bm{\tilde{\varepsilon}}_{t}}}\right\|(34)
\displaystyle\leq\left\|{{\bm{\hat{\varepsilon}}_{t-1}}}\right\|+\left\|{{\bm{\tilde{\varepsilon}}_{t}}}\right\|
\displaystyle=\left\|{{\bm{\hat{\varepsilon}}_{t-1}}}\right\|+{\rho_{\rm{t}}}
\displaystyle\leq{(1-\beta)^{t-1}}\left\|{{\bm{\hat{\varepsilon}}_{0}}}\right\|+{\rho_{0}}+{\rho_{t}}

Let {\bm{\hat{\varepsilon}}_{0}}={\bm{\tilde{\varepsilon}}_{0}}=\rho_{0}\frac{\nabla L(\mathbf{w}_{0})}{\|\nabla L(\mathbf{w}_{0})\|}, {\rho_{t}}=\frac{{{\rho_{0}}}}{{\sqrt{t}}}, we have:

\displaystyle\left[{L({\mathbf{w}_{t}}+{\bm{\hat{\varepsilon}}_{t-1}})-L({\mathbf{w}_{t}})}\right]-\left[{L({\mathbf{w}_{t}}+{\bm{\tilde{\varepsilon}}_{t}})-L({\mathbf{w}_{t}})}\right](35)
\displaystyle\leq\left(\tau{(1-\beta)^{t-1}}\left\|{{\bm{\hat{\varepsilon}}_{0}}}\right\|+\tau{\rho_{0}}+G+{\sigma^{2}}\right)
\displaystyle\quad\cdot\left({(1-\beta)^{t-1}}\left\|{{\bm{\hat{\varepsilon}}_{0}}}\right\|+{\rho_{0}}+{\rho_{t}}\right)
\displaystyle=\left({\left({1+{{(1-\beta)}^{t-1}}}\right)\tau{\rho_{0}}+G+{\sigma^{2}}}\right)
\displaystyle\quad\quad\cdot\left({\left({1+{{(1-\beta)}^{t-1}}}\right){\rho_{0}}+\frac{{{\rho_{0}}}}{{\sqrt{t}}}}\right)

The proof is thus completed. ∎

## B. Experimental Details

### B.1 Details on datasets

Our evaluations are carried out on commonly-used datasets in the literature.

Datasets for few-shot learning of RoBERTa-large. We consider classification datasets: SST-2 (socher2013recursive), SST-5 (socher2013recursive), TREC (voorhees2000building), MNLI (williams2018broad), SNLI (bowman2015large), and RTE (dagan2005pascal). We follow Malladi et al. (malladi2023kernel) in limiting the test set to 1, 000 examples for fast iteration. For training and validation, we set k = 512, which mean that we have 512 examples per class for both training and validation.

Table 7: The hyperparameters used for RoBERTa large with LoRA on the GLUE benchmark.

Table 8: Hyperparameters used for few-shot learning with RoBERTa-large.

Table 9: Hyperparameters used for GPT2.

GLUE benchmark. GLUE is designed to provide a general-purpose evaluation of language understanding (wangglue). Those adopted in our work include MNLI (inference, (williams2018broad)), SST-2 (sentiment analysis, (socher2013recursive)), MRPC (paraphrase detection, (dolan2005automatically)), CoLA (linguistic acceptability (warstadt2019neural)), QNLI (inference (rajpurkar2018know)), QQP 1 1 1 https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs (question-answering), RTE 2 2 2 https://paperswithcode.com/dataset/rte (inference), and STS-B (textual similarity (cer2017semeval)). These datasets are released under different permissive licenses.

E2E NLG Challenge. The E2E NLG Challenge dataset (novikova2017e2e) is a standard benchmark for end-to-end data-to-text natural language generation. It consists of around 42,000 training instances, along with 4,600 each for validation and testing, all within the restaurant domain. Inputs are structured as sequences of slot-value pairs and paired with one or more reference texts. The dataset is released under the Creative Commons BY-NC-SA 4.0 license.

Datasets for few-shot learning of CLIP. We consider five datasets for fine-grained classification of satellite imagery (EuroSAT (helber2019eurosat)), pet breeds (Ox-fordPets (parkhi2012cats)), flowers (Flower102 (nilsback2008automated)), general objects (Caltech101 (fei2004learning)), textures (DTD (cimpoi2014describing)). These datasets offer a thorough benchmarking framework for evaluating few-shot visual classification tasks.

Datasets for fine-tuning with Qwen-VL-Chat. We use two representative datasets: ScienceQA (lu2022learn) and VizWiz (gurari2018vizwiz). ScienceQA is a multimodal multiple-choice QA dataset covering elementary science, with questions accompanied by text and images. VizWiz is a real-world visual QA dataset collected from blind users, featuring diverse and often low-quality images, posing challenges for robust multimodal understanding.

### B.2 Details on models

We summarize the adopted language models in our evaluation. All model checkpoints are obtained from HuggingFace.

RoBERTa-large. This is a 355 M parameter model. The model checkpoint 3 3 3 https://huggingface.co/FacebookAI/roberta-large is released under the MIT license.

GPT2-medium. This is a 345 M parameter model. Its checkpoint 4 4 4 https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-medium-pytorch˙model.bin is under MIT License.

GPT2-large. This is a 774 M parameter model. Its checkpoint 5 5 5 https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-large-pytorch˙model.bin is under MIT License.

CLIP. This is a model that learns to connect images and text by mapping them into a shared semantic space using contrastive learning.

Qwen-VL-Chat. Qwen-VL-Chat (Bai2023QwenVLAV) is a multimodal conversational large language model capable of understanding both images and text.

Table 10: Hyperparameters used for few-shot learning with CLIP.

Algorithm 1 Pseudocode of the FMLoRA

Require: The training dataset, the learning rate \eta, the batch size b, parameters \rho and \beta.

1:for

t=1,2,\cdot\cdot\cdot
do

2: Randomly sample a mini-batch;

3: Evaluate the gradient at the current point;

4: Apply Equation (12) to compute the gradient in the full parameter space

{\bar{\mathbf{g}}^{\mathbf{W}}}
;

5: Use Equation (13) to calculate the perturbation

\bar{\mathbf{E}}^{\mathbf{W}}
;

6: Compute the perturbation

\bar{\mathbf{E}}^{\mathbf{B}}=\frac{1}{s}\bar{\mathbf{E}}^{\mathbf{W}}\mathbf{A}^{+}
on matrix

\mathbf{B}
according to Equation (14);

7: Evaluate the gradient at the perturbed point (

\mathbf{W}_{0}
,

\mathbf{B}+\bar{\mathbf{E}}^{\mathbf{B}}
,

\mathbf{A}
);

8: Return to the original (unperturbed) parameter point (

\mathbf{W}_{0}
,

\mathbf{B}
,

\mathbf{A}
);

9: Update the weights using the gradient obtained in Step 6;

10:end for

### B.3 Details on hyperparameters

Few-shot Learning with RoBERTa. We adopt the k-shot learning setup from (malladi2023fine), focusing on classification tasks with k=512 training samples per class and 1000 samples for testing. Prompt-based finetuning is used, following the same prompt templates as in (malladi2023fine, Table 13). We use AdamW as the optimizer and tune hyperparameters based on Table [8](https://arxiv.org/html/2508.00522v3#Sx6.T8 "Table 8 ‣ B.1 Details on datasets ‣ B. Experimental Details ‣ Efficiently Seeking Flat Minima for Better Generalization in Fine-Tuning Large Language Models and Beyond"). All results are averaged over three random seeds.

Fine-tuning with RoBERTa-large. AdamW is adopted as the base optimizer, and hyperparameters are in Table [7](https://arxiv.org/html/2508.00522v3#Sx6.T7 "Table 7 ‣ B.1 Details on datasets ‣ B. Experimental Details ‣ Efficiently Seeking Flat Minima for Better Generalization in Fine-Tuning Large Language Models and Beyond"). However, we employ single GPU rather than multiple ones and use gradient accumulation rather than parallelism due to memory constraint. We consider the GLUE benchmark and report the mismatched accuracy for MNLI, Matthew’s correlation for CoLA, Pearson correlation for STS-B, and accuracy for other datasets. Larger values indicate better results for all datasets. Experiments are conducted over three random trials for all datasets.

GPT2 medium/large on E2E NLG Challenge. We use the batch size, learning rate, and beam search beam size described in (hu2022lora). AdamW is adopted as base optimizer. The hyperparameters can be found in Table [9](https://arxiv.org/html/2508.00522v3#Sx6.T9 "Table 9 ‣ B.1 Details on datasets ‣ B. Experimental Details ‣ Efficiently Seeking Flat Minima for Better Generalization in Fine-Tuning Large Language Models and Beyond"). The result for each run is taken from the last epoch.

Few-shot Learning with CLIP. We follow the setting of previous work (zanella2024low). The hyperparameters are tuned from those in Table [10](https://arxiv.org/html/2508.00522v3#Sx6.T10 "Table 10 ‣ B.2 Details on models ‣ B. Experimental Details ‣ Efficiently Seeking Flat Minima for Better Generalization in Fine-Tuning Large Language Models and Beyond"). We only apply low-rank matrices on the query, key and value matrices with r=2. We regularize the input of the LoRA module by a dropout layer with p=0.25. The number of iterations is set equal to 500 times N/K (the number of labeled samples per class).

Fine-tuning with Qwen-VL-Chat. We conduct experiments follow the setting of previous work (zhou2024empirical). The hyperparameters can be found in Table [11](https://arxiv.org/html/2508.00522v3#Sx6.T11 "Table 11 ‣ B.3 Details on hyperparameters ‣ B. Experimental Details ‣ Efficiently Seeking Flat Minima for Better Generalization in Fine-Tuning Large Language Models and Beyond").

Table 11: Hyperparameters used for fine-tuning with Qwen-VL-Chat.

![Image 3: Refer to caption](https://arxiv.org/html/2508.00522v3/x3.png)

Figure 3: Approximation ability of EMA perturbations across datasets

![Image 4: Refer to caption](https://arxiv.org/html/2508.00522v3/x4.png)

Figure 4: Evolution of balancedness across layers during training with Adam and FMLoRA.

## C. Algorithm

The two algorithms presented in [1](https://arxiv.org/html/2508.00522v3#alg1 "Algorithm 1 ‣ B.2 Details on models ‣ B. Experimental Details ‣ Efficiently Seeking Flat Minima for Better Generalization in Fine-Tuning Large Language Models and Beyond") and [2](https://arxiv.org/html/2508.00522v3#alg2 "Algorithm 2 ‣ C. Algorithm ‣ Efficiently Seeking Flat Minima for Better Generalization in Fine-Tuning Large Language Models and Beyond") describe the training procedures of the proposed FMLoRA and its accelerated variant EFMLoRA.

Algorithm 2 Pseudocode of the EFMLoRA

Require: The training dataset, the learning rate \eta, the batch size b, parameters \rho and \beta.

1:for

t=1,2,\cdot\cdot\cdot
do

2: Randomly sample a mini-batch;

3:if

t=1
then

4: Evaluate the gradient at the current point;

5: EMA perturbation

\hat{\mathbf{E}}^{\mathbf{B}}_{1}=\bar{\mathbf{E}}^{{\mathbf{B}}}_{1}
;

6: Update the weights using the gradient obtained in Step 4;

7: Update the parameters to the next perturbation point

(\mathbf{W}_{0}
,

\mathbf{B}_{1}+\hat{\mathbf{E}}^{\mathbf{B}}_{1}
,

\mathbf{A}_{1})
.

8:else

9: Calculate the gradient at the perturbation point

(\mathbf{W}_{0}
,

\mathbf{B}_{t-1}+\hat{\mathbf{E}}^{\mathbf{B}}_{t-1}
,

\mathbf{A}_{t-1})
.

10: Compute the perturbation

\bar{\mathbf{E}}^{\mathbf{B}}_{t}=\frac{1}{s}\bar{\mathbf{E}}^{\mathbf{W}}\mathbf{A}^{+}_{t-1}
on matrix

\mathbf{B}
according to Equation (14);

11: Return to the original parameter point

(\mathbf{W}_{0}
,

\mathbf{B}_{t-1}
,

\mathbf{A}_{t-1})
.

12: Calculate the EMA perturbation

{{\hat{\mathbf{E}}}^{\mathbf{B}}_{t}}=(1-\beta){{\hat{\mathbf{E}}}^{\mathbf{B}}_{t-1}}+\beta{\bar{\mathbf{E}}^{\mathbf{B}}_{t}}
.

13: Update the weights to

(\mathbf{W}_{0}
,

\mathbf{B}_{t}
,

\mathbf{A}_{t})
using the gradient obtained in Step 9;

14: Update the parameters to the next perturbation point

(\mathbf{W}_{0}
,

\mathbf{B}_{t}+\hat{\mathbf{E}}^{\mathbf{B}}_{t}
,

\mathbf{A}_{t})
.

15:end if

16:end for

## D. More experiments

### D.1 The approximate ability of EMA perturbation

We consider few shot learning with LoRA on RoBERTa-large. Fig.[3](https://arxiv.org/html/2508.00522v3#Sx6.F3 "Figure 3 ‣ B.3 Details on hyperparameters ‣ B. Experimental Details ‣ Efficiently Seeking Flat Minima for Better Generalization in Fine-Tuning Large Language Models and Beyond") illustrates the evolution of the difference in sharpness, \left[{L({\mathbf{w}_{t}}+{\bm{\hat{\varepsilon}}_{t}})-L({\mathbf{w}_{t}})}\right]-\left[{L({\mathbf{w}_{t}}+{\bm{\tilde{\varepsilon}}_{t}})-L({\mathbf{w}_{t}})}\right], as described in Theorem 2, during training on six datasets (SNLI, SST-2, SST-5, MNLI, RTE, and TREC). S^{\text{EMA}} denotes the sharpness computed using EMA perturbations, while S^{\text{SAM}} refers to the original SAM sharpness. As training progresses, the absolute difference consistently decreases across all datasets, demonstrating that the EMA perturbation becomes increasingly effective at approximating the SAM perturbations. This validates the use of EMA perturbations as a computationally efficient surrogate for SAM perturbations. This result empirically supports Theorem 2.

### D.2 The change in balancedness during FMLoRA training

We consider few shot learning with LoRA on RoBERTa-large. For dataset MNLI, 1st, 12th and 24th query layers’ 2|B_{t,l}| are plotted, where t denotes the iteration and l denotes the layer index. The layers are chosen to represent early, middle, and final stages of RoBERTa. Balancedness of FMLoRA and Adam on different layers are plotted in Fig.[4](https://arxiv.org/html/2508.00522v3#Sx6.F4 "Figure 4 ‣ B.3 Details on hyperparameters ‣ B. Experimental Details ‣ Efficiently Seeking Flat Minima for Better Generalization in Fine-Tuning Large Language Models and Beyond"). Balancedness may increase or decrease across different layers. As shown in Fig.[4](https://arxiv.org/html/2508.00522v3#Sx6.F4 "Figure 4 ‣ B.3 Details on hyperparameters ‣ B. Experimental Details ‣ Efficiently Seeking Flat Minima for Better Generalization in Fine-Tuning Large Language Models and Beyond"), the balancedness of FMLoRA in the first query layer of RoBERTa-large gradually decreases during training, while in the 12th layer, it first decreases and then increases. In contrast, the balancedness in the 24th layer continuously increases. An increase typically occurs when parameter magnitudes in both low-rank subspaces grow simultaneously. This behavior can be influenced by factors such as the learning rate, optimization algorithm, weight decay, and other regularization strategies. Despite these occasional increases, FMLoRA generally maintains lower balancedness than Adam in most layers, suggesting its capacity to induce implicit regularization during training.
