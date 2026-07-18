                                             Sharpness Minimization Algorithms Do Not Only Minimize
                                                    Sharpness To Achieve Better Generalization
                                                              Kaiyue Wen                                                Zhiyuan Li
                                                          Tsinghua University                                       Stanford University
                                                   wenky20@mails.tsinghua.edu.cn                                zhiyuanli@stanford.edu
arXiv:2307.11007v2 [cs.LG] 23 Jul 2023




                                                                                          Tengyu Ma
                                                                                      Stanford University
                                                                                   tengyuma@stanford.edu


                                                                                                 Abstract
                                                   Despite extensive studies, the underlying reason as to why overparameterized neural networks can gen-
                                               eralize remains elusive. Existing theory shows that common stochastic optimizers prefer flatter minimizers
                                               of the training loss, and thus a natural potential explanation is that flatness implies generalization. This
                                               work critically examines this explanation. Through theoretical and empirical investigation, we identify the
                                               following three scenarios for two-layer ReLU networks: (1) flatness provably implies generalization; (2)
                                               there exist non-generalizing flattest models and sharpness minimization algorithms fail to generalize, and
                                               (3) perhaps most surprisingly, there exist non-generalizing flattest models, but sharpness minimization algo-
                                               rithms still generalize. Our results suggest that the relationship between sharpness and generalization subtly
                                               depends on the data distributions and the model architectures and sharpness minimization algorithms do not
                                               only minimize sharpness to achieve better generalization. This calls for the search for other explanations for
                                               the generalization of over-parameterized neural networks.


                                         1    Introduction
                                         It remains mysterious why stochastic optimization methods such as stochastic gradient descent (SGD) can
                                         find generalizable models even when the architectures are overparameterized (Zhang et al., 2016; Gunasekar
                                         et al., 2017; Li et al., 2017; Soudry et al., 2018; Woodworth et al., 2020). Many empirical and theoretical
                                         studies suggest that generalization is correlated with or guaranteed by the flatness of the loss landscape at the
                                         learned model (Hochreiter & Schmidhuber, 1997; Keskar et al., 2016; Dziugaite & Roy, 2017; Jastrzebski
                                         et al., 2017; Neyshabur et al., 2017; Wu et al., 2018; Jiang et al., 2019; Blanc et al., 2019; Wei & Ma, 2019a,b;
                                         HaoChen et al., 2020; Foret et al., 2021; Damian et al., 2021; Li et al., 2021; Ma & Ying, 2021; Ding et al.,
                                         2022; Nacson et al., 2022; Wei et al., 2022; Lyu et al., 2022; Norton & Royset, 2021; Wu & Su, 2023). Thus,
                                         a natural theoretical question is
                                         Question 0. Does the flatness of the minimizers always correlate with the generalization capability?

                                             The answer to the question turns out to be false. First, Dinh et al. (2017) theoretically construct very sharp
                                         networks with good generalization. Second, recent empirical results (Andriushchenko et al., 2023b) find
                                         that sharpness may not have a strong correlation with test accuracy for a collection of modern architectures
                                         and settings, partly due to the same reason—there exist sharp models with good generalization. We note
                                         that, technically speaking, Question 0 is ill-defined without specifying the collection of models on which the
                                         correlation is evaluated. However, those sharp but generalizable models appear to be the main cause for the
                                         non-correlation.
                                             Observing the existing theoretical and empirical evidence, it is natural to ask the one-side version of
                                         Question 0, where we are only interested in whether sharpness implies generalization but not vice versa.


                                                                                                     1
                                                   All Flattest Minimizers      Sharpness Minimization
           Architecture
                                                      Generalize Well.          Algorithms Generalize.
           2-layer w/o Bias                           ✓ (Theorem 3.1)                     ✓
           2-layer w/ Bias                            ✗ (Theorem 4.1)                     ✗
           2-layer w/ simplified BatchNorm            ✓ (Theorem 3.2)                     ✓
           2-layer w/ simplified LayerNorm            ✗ (Theorem 5.1)                     ✓


Table 1: Overview of Our Results. Each row in the table corresponds to one architecture. The second
column indicates whether all flattest minimizers of training loss generalize well. ✓ indicates that all (near)
flattest minimizers of training loss provably generalize well and ✗ indicates that there provably exists flattest
minimizers that generalize poorly. The third column indicates whether the sharpness minimization algorithms
generalize well in our experiments. Results in row 2 and 4 deny Question 1 and Question 2 respectively.


Question 1. Do all the flattest neural network minimizers generalize well?
     Though there are some theoretical works that answer Question 1 affirmatively for simplified linear mod-
els (Li et al., 2021; Ding et al., 2022; Nacson et al., 2022; Gatmiry et al., 2023), the answer to Question 1
for standard neural networks remains unclear. Those theoretical results linking generalization to sharpness
for more general architectures typically also involve other terms in generalization bounds, such as parameter
dimension or norm (Neyshabur et al., 2017; Foret et al., 2021; Wei & Ma, 2019a,b; Norton & Royset, 2021),
thus do not answer Question 1 directly.
     Our first contribution is a theoretical analysis showing that the answer to Question 1 can be false, even
for simple architectures like 2-layer ReLU networks. Intriguingly, we also find that the answer to Question 1
subtly depends on the architectures of neural networks. For example, simply removing the bias in the first
layer turns the aforementioned negative result into a positive result, as also shown in the Theorem 4.3 of Wu
& Su (2023) (that the authors only came to be aware of after putting this work online).
     More concretely, we show that for the 2 parity xor problem with mean square loss and with data sampled
from hypercube {−1, 1}d , all flattest 2-layer ReLU neural networks without bias provably generalize. However,
when bias is added, for the same data distribution and loss function, there exists a flattest minimizer that fails
to generalize for every unseen data. Since adding bias in the first layer can be interpreted as appending a
constant input feature, this result suggests that the generalization of the flattest minimizer is sensitive to both
network architectures and data distributions.
     Recent theoretical studies (Wu et al., 2018; Blanc et al., 2019; Damian et al., 2021; Li et al., 2021; Arora
et al., 2022; Wen et al., 2022; Nacson et al., 2022; Lyu et al., 2022; Bartlett et al., 2022; Li et al., 2022) also show
that optimizers including SGD with large learning rates or label noise and Sharpness-Aware Minimization
(SAM, Foret et al. (2021)) may implicitly regularize the sharpness of the training loss landscape. These
optimizers are referred to as sharpness minimization algorithms in this paper. Because Question 1 is not
always true, it is then natural to hypothesize that sharpness-minimization algorithms will fail for architectures
and data distributions where Question 1 is not true.

Question 2. Will sharpness minimization algorithm fail to generalize when there exist non-generalizing
flattest minimizers?
    A priori, the authors were expecting that the answer to Question 2 is affirmative, which means that a
possible explanation is that the sharpness minimization algorithm works if and only if for certain architecture
and data distribution, Question 1 is true. However, surprisingly, we also answer this question negatively for
some architectures and data distributions. In other words, we found that sharpness-minimization algorithms
can still generalize well even when the answer to Question 1 is false. The result is consistent with our theoretical
discovery that for many architectures, there exist both non-generalizing and generalizing flattest minimizers
of the training loss. We show empirically that sharpness-minimization algorithms can find different types of
minimizers for different architectures.
    Our results are summarized in Table 1. We show through theoretical and empirical analysis that the
relationship between sharpness and generalization can fall into three different regimes depending on the
architectures and distributions. The three regimes include:


                                                           2
• Scenario 1. Flattest minimizers of training loss provably generalize and sharpness minimization algorithms
  find generalizable models. This regime (Theorems 3.1 and 3.2) includes 2-layer ReLU MLP without bias
  and 2-layer ReLU MLP with a simplified BatchNorm (without mean subtraction and bias). We answer both
  the Question 1 and Question 2 affirmatively in this scenario.1
• Scenario 2. There exists a flattest minimizer that has the worst generalization over all minimizers. Also,
  sharpness minimization algorithms fail to find generalizable models. This regime includes 2 layer ReLU
  MLP with bias. We deny Question 1 while affirm Question 2 in this scenario.
• Scenario 3. There exist flattest minimizers that do not generalize but the sharpness minimization algorithm
  still finds the generalizable flattest model empirically. This regime includes 2-layer ReLU MLP with a
  simplified LayerNorm (without mean subtraction and bias). In this scenario, the sharpness minimization
  algorithm relies other unknown mechanisms beyond minimizing sharpness to find a generalizable model.
  We deny both Question 1 and Question 2 in this scenario.


2      Setup
                                                 n
Rademacher Complexity. Given n data S = {xi }P   i=1 , the empirical Rademacher complexity of function
                               1                   n
class F is defined as RS (F) = n Eϵ∼{±1}n supf ∈F i=1 ϵi f (xi ).

Architectures. As summarized in Table 1, we will consider multiple network architectures and discuss how
architecture influences the relationship between sharpness and generalization. For each model fθ parame-
terized by θ, we will use d to denote the input dimension and m to denote the network width. We will now
describe the architectures in detail.

2-MLP-No-Bias.fθnobias (x) = W2 relu (W1 x) with θ = (W1 , W2 ).

2-MLP-Bias. fθbias (x) = W2 relu (W1 x + b1 ) with θ = (W1 , b1 , W2 ). We additionally define MLP-Bias as
fθbias,D (x) = WD relu · · · W2 relu (W1 x + b1 ),

2-MLP-Sim-BN. fθsbn (x, {xi }i∈[n] ) = W2 SBNγ (relu (W1 x + b1 ) , {relu (W1 xi + b1 )}) , where the sim-
plified BatchNorm SBN is defined as ∀m, n ∈ N, ∀i ∈ [n], x, xi ∈ Rm , j ∈ [m], SBNγ (x, {xi }i∈[n] )[j] =
         Pn            2
                           1/2
γx[j]/     i=1 (xi [j]) /n      and θ = (W1 , b1 , γ, W2 ).
                                              relu(W x+b )
2-MLP-Sim-LN. fθsln (x) = W2 max{∥relu(W11 x+b11 )∥2 ,ϵ} where ϵ is a sufficiently small positive constant.

    Surprisingly, our results show that the relationships between sharpness and generalization are strikingly
different among these simple yet similar architectures.

Data Distribution. We will consider a simple data distribution as our testbed. Data distribution Pxor is a joint
distribution over data point x and label y. The data point is sampled uniformly from the hypercube {−1, 1}d
and the label satisfies y = x[1]x[2]. Many of our results, including our generalization bound in Section 3 and
experimental observations can be generalized to broader family of distributions (Appendix A).

Loss. We will use mean squared error ℓmse for training and denote the training loss as L. In Appendix A, we
will show that all our theoretical results and empirical observations hold for logistic loss with label smoothing
probability p > 0. We will also consider zero one loss Pr(yfθ (x) > 0) for evaluating the model. We will use
interpolating model to denote the model with parameter θ that minimizes L.
Definition 2.1 (Interpolating Model). A model fθ interpolates the dataset {(xi , yi )}ni=1 if and only if
∀i, fθ (xi ) = yi .
    1The condition for Question 2 is not satisfied and thus the answer to Question 2 is affirmative.




                                                                     3
                          (a) Baseline                                         (b) 1-SAM

Figure 1: Scenario I. We train a 2-layer MLP with ReLU activation without bias using gradient descent with
weight decay and 1-SAM on Pxor with dimension d = 30 and training set size n = 100. In both cases, the
model reaches perfect generalization. Notice that although weight decay doesn’t explicitly regularize model
sharpness, the flatness of the model decreases through training, which is consistent with our Lemma 3.1
relating sharpness to the norm of the weight.


Sharpness. Our theoretical analysis focuses on understanding the sharpness of the trained models. Precisely,
for a model fθ parameterized by θ, a dataset {(xi , yi )}ni=1 and loss function ℓ, we will use the trace of Hessian
of loss function, Tr(∇2 L(θ)) to measure how sharp the loss is at θ, which is a proxy for the sharpness along a
random direction (Wen et al., 2022), or equivalently, the expected increment of loss under a random gaussian
perturbation (Foret et al., 2021; Orvieto et al., 2022) .
    Tr(∇2 L(θ)) is not the only choice for defining sharpness, but theoretically many sharpness minimization
algorithms have been shown to minimize this term over interpolating models. In particular, under the
assumptions that the minimizer of the training loss form a smooth manifold Cooper (2018); Fehrman et al.
(2020), Sharpness-Aware Minimization (SAM) (Foret et al., 2021) with batch size 1 and sufficiently small
learning rate η and perturbation radius ρ (Wen et al., 2022; Bartlett et al., 2022), or Label Noise SGD with
sufficiently small learning rate η (Blanc et al., 2019; Damian et al., 2021; Li et al., 2021), prefers interpolating
models with small trace of Hessian of the loss. Hence, we choose to analyze trace of Hessian of the loss and
will use SAM with batch size 1 (we denote it by 1-SAM) as our sharpness minimization algorithm in our
experiments.

Notations. We use Tr to denote the trace of a matrix and x[i] to denote the value of the i-th coordinate
of vector x. We will use ⊙ to represent element-wise product. We use 1 as the (coordinate-wise) indicator
function, for example, 1 [x > 0] is a vector of the same length as x whose j-th entry is 1 if x[j] > 0 and 0
otherwise. We will use Õ(x) to hide logarithmic multiplicative factors.


3     Scenario I: All Flattest Models Generalize
3.1    Flattest models provably generalize
When the architecture is 2-MLP-No-Bias, we will show that the flattest models can provably generalize, hence
answering Question 1 affirmatively for this architecture and data distribution Pxor .
Theorem 3.1. For any δ ∈ (0, 1) and input dimension d, for n = Ω d log dδ , with probability at least
                                                                                     
                                                                                      Pn
1 − δ over the random draw of training set {(xi , yi )}ni=1 from Pxor
                                                                    n
                                                                      , let L(θ) ≜ n1 i=1 ℓmse (fθnobias (xi ), yi )
be the training loss for 2-MLP-No-Bias, it holds that for all θ∗ ∈ arg minL(θ)=0 Tr ∇2 L (θ) , we have that

                               Ex,y∼Pxor ℓmse fθnobias
                                                                  
                                                      ∗     (x) , y ≤ Õ (d/n) .
    Theorem 3.1 shows that for Pxor , flat models can generalize under almost linear sample
                                                                                           complexity with
                                                                                                       
respect to the input dimension. We note that Theorem 3.1 implies that Prx,y∼Pxor fθnobias ∗   (x)y > 0 ≤


                                                         4
                                                                            
Õ (d/n) . because if fθnobias
                         ∗     (x)y ≤ 0, it holds that ℓmse fθnobias
                                                                ∗    (x) , y ≥ 1. This shows that the model can
classify the input with high accuracy. The major proof step is relating sharpness to the norm of the weight
itself.
                                                     Pm
Lemma 3.1. Define ΘC ≜ {θ = (W1 , W2 ) |                j=1 ∥W1,j ∥2 |W2,j | ≤ C}. Under the setting of Theo-
rem 3.1, there exists a absolute constant C independent of d and δ, suchthat with
                                                                             p     probability at least 1 − δ,
                      2                             nobias
arg minL(θ)=0 Tr ∇ L (θ) ⊆ ΘC and RS ({fθ                  | θ ∈ ΘC }) ≤ Õ     d/n .

    We would like to note that similar results of Theorem 3.1 and lemma 3.1 have also been shown in a prior
work Wu & Su (2023) (that the authors were not aware of before the first version of this work was online).
    The almost linear complexity in Theorem 3.1 is not trivial. For example, Wei et al. (2019) shows that
learning the distribution will require Ω(d2 ) samples for Neural Tangent Kernel (NTK) (Jacot et al., 2018). In
contrast, our result shows that learning the distribution only requires Õ(d) samples as long as the flatness of
the model is controlled.
    Beyond reducing model complexity, flatness may also encourage the model to find a more interpretable
solution. We prove that under a stronger than i.i.d condition over the training set, the near flattest interpolating
model with architecture 2-MLP-Sim-BN will provably generalize and the weight of the first layer will be
centered on the first two coordinates of the input, i.e., ∥W1,i [3 : d]∥2 ≤ ϵ∥W1,i ∥2 .
Condition 1 (Complete Training Set Condition). There exists set S ⊂ {−1, 1}d−2 , such that the linear space
spanned by S − S = {s1 − s2 | s1 , s2 ∈ S} has rank d − 2 and the training set is {(x, y) | x ∈ Rd , x[3 :
d] ∈ S, x[1], x[2] ∈ {−1, 1}, y = x[1] × x[2]}.
Theorem 3.2. Given any training set {(xi , yi )}ni=1 satisfying Condition 1, for any width m and any ϵ > 0,
there exists constant κ > 0, such
                                 that for any width-m 2-MLP-Sim-BN       , f sbn , satisfying fθsbn interpolates
                                                                 ′
                                                                   
the training set and Tr ∇ L(θ) ≤ κ + inf L(θ′ )=0 Tr ∇ L(θ ) , it holds that ∀x ∈ {−1, 1}d , x[1]x[2] −
                          2                                 2

fθ (x) ≤ ϵ and that ∀i ∈ [m], ∥W1,i [3 : d]∥2 ≤ ϵ∥W1,i ∥2 .
    One may notice that in Theorem 3.2 we only consider the approximate minimizer of sharpness. This is
because the gradient of output with respect to W1 , b1 , despite never being zero, will converge to zero as the
norm of W1 , b1 converges to ∞.
    Condition 1 may seem stringent. In practice (Figure 2b), we find it not necessary for 1-SAM to find
a generalizable solution. We hypothesize that this condition is mainly technical. Theorem 3.2 shows that
sharpness minimization may guide the model to find an interpretable and low-rank representation. Similar
implicit bias of SAM has also been discussed in Andriushchenko et al. (2023a) The proof is deferred
to Appendix A.1

3.2    SAM empirically finds the flattest model that generalizes
We use 1-SAM to train 2-MLP-No-Bias on data distribution Pxor to verify our Theorem 3.1 (Figure 1). As
expected, the model interpolates the training set and reaches a flat minimum that generalizes perfectly to the
test set.
     We then verify our Theorem 3.2 by training a 2-layer MLP with simplified BN on data distribution Pxor
(Figure 2a). Here we do not enforce the strong theoretical Condition 1. However, we still observe that SAM
finds a flat minimum that generalizes well. We then perform a detailed analysis of the model and find that the
model is indeed interpretable. For example, the four largest neurons in the first layer approximately extract
features {relu(c1 x[1] + c2 x[2]) | c1 , c2 ∈ {−1, 1}} (Figure 2b). Also, the first 2 columns of the weight matrix
of the first layer, corresponding to the useful features {relu(c1 x[1] + c2 x[2]) | c1 , c2 ∈ {−1, 1}}, have norms
42.47 and 42.48, while the largest column norm of the rest of the weight matrix is only 5.65.




                                                         5
                                                           W1,i [1]   W1,i [2]   ∥W1,i [3 : d]∥2
                                                            18.581    -18.582        0.02
                                                           -14.363    -14.363        0.03
                                                            13.768     13.771        0.03
                                                           -12.601     12.601        0.01


               (a) 2-layer MLP with simplified BN       (b) Weights of the four neurons with the largest norm
                                                        in the first Layer

Figure 2: Interpretable Flattest Solution We train a 2-layer MLP with simplified BN using 1-SAM on Pxor
with dimension d = 30 and training set size n = 100. After training, we find that the model is indeed
interpretable. In Figure 2b, we inspect the weight of the four neurons of the four largest neurons in the first
layer and we observe that the four neurons approximately extract features ±x[1] ± x[2].


4      Scenario II: Both Flattest Generalizing and Non-generalizing Mod-
       els Exist, and SAM Finds the Former
4.1     Both generalizing and non-generalizing solutions can be flattest
In previous section, we show through Theorems 3.1 and 3.2 that sharpness benefits generalization under
some assumptions. It is natural to ask whether it is possible to extend this bound to general architectures.
However, in this section, we will show that the generalization benefit depends on model architectures. In fact,
simply adding bias to the first layer of 2-MLP-No-Bias makes non-vacuous generalization bound impossible
for distribution Pxor . This then leads to a negative answer to Question 1.
Definition 4.1 (Set of extreme points). A finite set S ⊂ Rd is a set of extreme points if and only if for any
x ∈ S, x is a vertex of the convex hull of S.
Definition 4.2 (Memorizing Solutions). A D-layer network is a memorizing solution for a training dataset if
(1) the network interpolates the training dataset, and (2) for any depth k ∈ [D − 1], there is an injection from
the input data to the neurons on depth k, such that the activations in layer k for each input data is a one-hot
vector with the non-zero entry being the corresponding neuron.
Theorem 4.1. For any D ≥ 2, if the input data points {xi } of the training set form a set of extreme points
(Definition 4.1), then there exists a width n layer D MLP-Bias that is a memorizing solution (Definition 4.2)
for the training dataset and has minimal sharpness over all the interpolating solutions.
      As one may suspect, these memorizing solutions can have poor generalization performance.
Proposition 4.1. For data distribution Pxor , for any number of samples n, there exists a width-n 2-MLP-
Bias that memorizes the training set as in Theorem 4.1, reaches minimal sharpness over all the interpolating
models and has generalization error max{1 − n/2d , 0} measured by zero one error.
    This corollary shows that a flat model can generalize poorly. Comparing Theorems 3.1 and 4.1, one
may observe the perhaps surprising difference caused by slightly modifying the architectures (adding bias or
removing the BatchNorm). To further show the complex relationship between sharpness and generalization,
the following theorem suggests, despite the existence of memorizing solutions, there also exists a flattest
model that can generalize well.
Proposition 4.2. For data distribution Pxor , for any number of samples n, there exists a width-n 2-MLP-
Bias that interpolates the training dataset, reaches minimal sharpness over all the interpolating models, and
has zero generalization error measured by zero one error.


                                                       6
                                                          Figure 3: Visualization of Memorization Solutions.
                                                          This is an illustration of the memorizing solutions con-
                                                          structed in Theorem 4.1. Here the input data points
           w3x + b3 > 0                  w1x + b1 > 0     come from a unit circle and are marked as dots. The
                  x3                      x1              shady area with the corresponding color represents the
                                                          region where the corresponding neuron is activated.
                                                          One can see that the network can output the correct
                                                          label for each input data point in the training set as
                                                          long as the weight vector on the corresponding neuron
                                                          is properly chosen. Further, the network will make the
                                                          same prediction 0 for all the input data points outside
                                                          the shady area and this volume can be made almost as
                              x2                          large as the support of the training set by choosing ϵ
                          w2x + b2 > 0                    sufficiently small. Hence the model can interpolate the
                                                          training set while generalizing poorly.



    The flat solution constructed is highly simple. It contains four activated neurons, each corresponding to
one feature in ±x[1] ± x[2] (Equation (5)).
    Proof sketch. For simplicity, we will consider 2-MLP-Bias here. The construction of the memorizing
solution in Theorem 4.1 is as follows (visualized in Figure 3). As the input data points form a set of extreme
points (Definition 4.1), for each input data point xi , there exists a vector ∥wi ∥ = 1, wi ∈ Rd , such that
∀j ̸= i, wi⊤ xi > wi⊤ xj . We can then choose
                    p                       p                                          p
            W1 = [ ri |yi |wi /ϵ]⊤                      ⊤
                                                                 ⊤
                                  i , b1 = [ ri |yi | −wi xi + ϵ /ϵ] , W2 = [sign(yi ) |yi |/ri ]i .

                    2       1/2
p ri = (∥xi ∥ + 1) and ϵ is a sufficiently small positive number. Then it holds that relu(W1 xi + b1 ) =
Here
   ri |yi |ei , where ei is the i−th coordinate vector. This shows there is a one-to-one correspondence between
the input data and the neurons. It is easy to verify that the model interpolates the training dataset. Furthermore,
for Pxor and sufficiently small ϵ, for any input x ̸∈ {xi }i∈[n] , it holds that relu(W1 x + b1 ) = 0. Hence the
model will output the same label 0 for all the data points outside the training set. This indicates Proposition 4.1.
    To show the memorization solution has minimal sharpness, we need the following lemma that relates the
sharpness and the Jacobian of the model.
Lemma 4.1. For mean squared errorPloss lmse , if model fθ is differentiable and interpolates dataset
                                         n
{(xi , yi )}i∈[n] , then Tr ∇2 L(θ) = n2 i=1 ∥∇θ fθ (xi )∥2 .
Proof of Lemma 4.1. By standard calculus, it holds that,
                              n
                           1X
              Tr ∇2 L(θ) =       Tr ∇2θ (fθ (xi ) − yi )2
                                                        
                           n i=1
                                     n
                                  2X  2                                                             ⊤
                                                                                                       
                              =         Tr ∇θ fθ (xi )(fθ (xi ) − yi ) + (∇θ fθ (xi )) (∇θ fθ (xi ))
                                  n i=1
                                     n                                         n
                                  2X                                 ⊤
                                                                           2X
                              =         Tr (∇θ fθ (xi )) (∇θ fθ (xi ))    =       ∥∇θ fθ (xi )∥22 .             (1)
                                  n i=1                                     n i=1

The first equation in Equation (1) use ∀i, fθ (xi ) = yi . The proof is then complete.
    After establishing Lemma 4.1, one can then explicitly calculate the lower bound of ∥∇θ fθ (xi )∥2 condition
on fθ (xi ) = yi . For simplicity of writing, we will view the bias term as a part of the weight matrix by appending
a 1 to the input data point. Precisely, we will use notation x′i ∈ Rd+1 to denote transformed input satisfying
∀j ∈ [d], x′i [j] = xi [j], x′i [d + 1] = 1 and W1′ = [W1 , b1 ] ∈ Rm×(d+1) to denote the transformed weight
matrix.


                                                           7
                          (a) Baseline                                               (b) 1-SAM

Figure 4: Scenario II. We train a 2-layer MLP with ReLU activation with Bias using gradient descent with
weight decay and 1-SAM on Pxor with dimension d = 30 and training set size n = 100. One can clearly
observe a distinction between the two settings. The minimum reached by 1-SAM is flatter but the model fails
to generalize and the generalization performance even starts to degenerate after 4000 epochs. The difference
between Figures 1b and 4b indicates a small change in the architecture can lead to a large change in the
generalization performance.


      By the chain rule, we have,

                       ∥∇θ fθ (xi )∥2 = ∥∇W1′ fθ (xi )∥2F + ∥∇W2 fθ (xi )∥2F
                                         = ∥(W2 ⊙ 1 [W1′ x′i > 0])x′⊤ 2           ′ ′     2
                                                                   i ∥F + ∥relu (W1 xi ) ∥2 .
                                         = ∥W2 ⊙ 1 [W1′ x′i > 0] ∥22 ∥x′i ∥2 + ∥relu (W1′ x′i ) ∥22 .       (2)

Then by Cauchy-Schwarz inequality, we have

                   ∥∇θ fθ (xi )∥2 = ∥W2 ⊙ 1 [W1′ x′i > 0] ∥22 ∥x′i ∥2 + ∥relu (W1′ x′i ) ∥22
                                                                         ⊤
                                    ≥ 2∥x′i ∥ (W2 ⊙ 1 [W1 xi > 0]) relu (W1′ x′i ) = 2∥x′i ∥|yi |.          (3)

In Equation (3), we use condition fθ (xi ) = yi . Finally, notice that the lower bound is reached when

                                     W2 ⊙ 1 [W1′ x′i > 0] = relu (W1′ x′i ) /∥x′i ∥.                        (4)

Condition Equation (4) is clearly
                               p reached for the memorization construction we constructed, where both sides
of the equation are equal to |yi |/∥x′i ∥ei . This completes the proof of Theorem 4.1.
    However, the memorization network is not the only parameter that can reach the lower bound. For example,
for distribution Pxor , if parameter θ satisfies,

   ∀i, j ∈ {0, 1}, W1,2i+j+1 = r[(−1)i , (−1)j , ..., 0], b1 [2i + j + 1] = −r, W2 [2i + j] = (−1)i+j /r.   (5)
   ∀k > 4, W1,k = [0, ..., 0], b1 [k] = 0, W2 [k] = 0,

with r = (d2 + 1)1/4 . then for any x ∈ {−1, 1}d , it holds that relu(W1 x + b1 ) = re5/2−x[1]−x[2]/2 and
fθ (x) = x[1] × x[2]. Hence it is possible for Equation (5) to hold while the model has perfect generalization
performance.

4.2     SAM empirically finds the non-generalizing solutions
In this section, we will show that in multiple settings, SAM can find solutions that have low sharpness but
fail to generalize compared to the baseline full batch gradient descent method with weight decay. This proves
that flat minimization can hurt generalization performance. However, one should note that Question 2 is not
denied for the current architectures.


                                                              8
                         (a) Baseline                                        (b) 1-SAM

Figure 5: Scenario II with Softplus Activation. We train a 2-layer MLP with Softplus activation
(SoftPlus(x) = log(1 + ex )) with bias using gradient descent with weight decay and 1-SAM on Pxor
with dimension d = 30 and training set size n = 100. We observe a similar phenomenon as Figure 4.

Converged models found by SAM fail to generalize. We perform experiments on data distribution Pxor
in Figure 4. We apply small learning rate gradient descent with weight decay as our baseline and observe that
the converged model found by SAM has a much lower sharpness than the baseline. However, the generalization
performance of SAM is much worse than the baseline. Moreover, the generalization performance even starts
to degenerate after 4000 epochs. We conclude that in this scenario, sharpness minimization can empirically
hurt generalization performance.

1-SAM may fail to generalize with other activation functions. A natural question is whether the phe-
nomenon that 1-SAM fails to generalize is limited to ReLU activation. In Figure 5, we show empirically that
1-SAM fails to generalize for 2-layer networks with softplus activation trained on the same dataset, although
there is no known guarantee for the existence of memorizing solutions.


5     Scenario III: Both Flattest Generalizing and Non-generalizing Mod-
      els Exist, and SAM Finds the Latter
5.1    Both generalizing and non-generalizing solutions can be flattest
Despite the surprising contrary between Theorems 3.1 and 4.1, experiments show that Question 2 consistently
hold. However, we will provide a counterexample in this section. Specifically, we will consider data
distribution Pxor and 2-layer ReLU MLP with simplified LayerNorm. One can first show both generalizing
and non-generalizing solutions exist similar to Theorem 4.1 and propositions 4.1 and 4.2.
Theorem 5.1. If the input data points {xi } of the training set form a set of extreme points (Definition 4.1), for
sufficiently small ϵ, then there exists a width-n 2-MLP-Sim-LN with hyperparameter ϵ that is a memorizing
solution (Definition 4.2) for the training dataset and has minimal sharpness over all the interpolating solutions.
Proposition 5.1. For data distribution Pxor , for sufficiently small ϵ, for any number of samples n, there
exists a width-n 2-MLP-Sim-LN with hyperparameter ϵ that memorizes the training set as in Theorem 4.1,
reaches minimal sharpness over all the interpolating models and has generalization error max{1 − n/2d , 0}
measured by zero one error.

Proposition 5.2. For data distribution Pxor , for sufficiently small ϵ, for any number of samples n, there
exists a width-n 2-MLP-Sim-LN with hyperparameter ϵ that interpolates the training dataset, reaches minimal
sharpness over all the interpolating models, and has zero generalization error measured by zero one error.
    The construction and intuition behind Theorem 5.1 and propositions 5.1 and 5.2 are similar to that
of Theorem 4.1 and propositions 4.1 and 4.2. The proof is deferred to Appendix A.


                                                        9
                    (a) Standard Training                             (b) Projected Training

Figure 6: Scenario III. We train two-layer ReLU networks with simplified LayerNorm on data distribution
Pxor with dimension d = 30 and sample complexity n = 100 using 1-SAM. In Figure 6a, we use standard
training. In Figure 6b, we restricted the norm of the weight and the bias of the first layer as 10, to avoid
minimizing the sharpness by simply increasing the norm. We can see that in both cases, the models almost
perfectly generalize.

5.2   SAM empirically finds generalizing models
Notice in Section 5.1 our theory makes the same prediction as in Section 4. However, strikingly, the
experimental observation is reversed (Figure 6). Now running SAM can greatly improve the generalization
performance till the model perfectly generalizes. This directly denies Question 2 as now we have a scenario
in which sharpness minimization algorithms can improve generalization till perfect generalization while there
exists a flattest minimizer that will generalize poorly.


6     Discussion and Conclusion
We present theoretical and empirical evidence for (1) whether sharpness minimization implies generalization
subtly depends on the choice of architectures and data distributions, and (2) sharpness minimization algorithms
including SAM may still improve generalization even when there exist flattest models that generalize poorly.
Our results suggest that low sharpness may not be the only cause of the generalization benefit of sharpness
minimization algorithms.

Limitations and future work. Our results only cover a small subset of existing architectures. A natural
extension for our work will be to examine whether flatness implies generalization for deep networks without
the bias terms or if the flattest memorizing models exist for such architecture.
    In our work, we assume 1-SAM always finds a valid global minimizer for sharpness. However, in previous
works, only a local tendency of decreasing sharpness is proven. In all our experiments where the sharpness
lower bound can be exactly characterized, we observe that the converged model found by 1-SAM always
approximately reaches the lower bound. A possible future direction is characterizing under what condition
can 1-SAM be used as a global optimizer for sharpness over minimizers.
    Our work also suggests that there does not exist a universal generalization theory for neural networks only
based on sharpness. A broader problem would be what other properties of the model can be used to explain
the generalization of neural networks.


ACKNOWLEDGEMENTS
The authors would like to thank the support from NSF IIS 2045685.




                                                      10
References
Maksym Andriushchenko, Dara Bahri, Hossein Mobahi, and Nicolas Flammarion. Sharpness-aware mini-
 mization leads to low-rank features. arXiv preprint arXiv:2305.16292, 2023a.
Maksym Andriushchenko, Francesco Croce, Maximilian Müller, Matthias Hein, and Nicolas Flammarion. A
 modern look at the relationship between sharpness and generalization. arXiv preprint arXiv:2302.07011,
 2023b.
Sanjeev Arora, Zhiyuan Li, and Abhishek Panigrahi. Understanding gradient descent on edge of stability in
  deep learning. arXiv preprint arXiv:2205.09745, 2022.
Peter L Bartlett, Philip M Long, and Olivier Bousquet. The dynamics of sharpness-aware minimization:
  Bouncing across ravines and drifting towards wide minima. arXiv preprint arXiv:2210.01513, 2022.
Guy Blanc, Neha Gupta, Gregory Valiant, and Paul Valiant. Implicit regularization for deep neural networks
  driven by an ornstein-uhlenbeck like process. arXiv preprint arXiv:1904.09080, 2019.
Yaim Cooper. The loss landscape of overparameterized neural networks. arXiv preprint arXiv:1804.10200,
  2018.
Alex Damian, Tengyu Ma, and Jason Lee. Label noise sgd provably prefers flat global minimizers, 2021.
Lijun Ding, Dmitriy Drusvyatskiy, and Maryam Fazel. Flat minima generalize for low-rank matrix recovery.
   arXiv preprint arXiv:2203.03756, 2022.
Laurent Dinh, Razvan Pascanu, Samy Bengio, and Yoshua Bengio. Sharp minima can generalize for deep
  nets. In Proceedings of the 34th International Conference on Machine Learning-Volume 70, pp. 1019–1028.
  JMLR. org, 2017.
Gintare Karolina Dziugaite and Daniel M Roy. Computing nonvacuous generalization bounds for
  deep (stochastic) neural networks with many more parameters than training data. arXiv preprint
  arXiv:1703.11008, 2017.
Benjamin Fehrman, Benjamin Gess, and Arnulf Jentzen. Convergence rates for the stochastic gradient descent
  method for non-convex objective functions. Journal of Machine Learning Research, 21:136, 2020.
Pierre Foret, Ariel Kleiner, Hossein Mobahi, and Behnam Neyshabur. Sharpness-aware minimization for
  efficiently improving generalization. In International Conference on Learning Representations, 2021.
Khashayar Gatmiry, Zhiyuan Li, Ching-Yao Chuang, Sashank Reddi, Tengyu Ma, and Stefanie Jegelka. The
  inductive bias of flatness regularization for deep matrix factorization. arXiv preprint arXiv:2306.13239,
  2023.
Suriya Gunasekar, Blake E Woodworth, Srinadh Bhojanapalli, Behnam Neyshabur, and Nati Srebro. Implicit
  regularization in matrix factorization. In Advances in Neural Information Processing Systems, pp. 6151–
  6159, 2017.
Jeff Z HaoChen, Colin Wei, Jason D Lee, and Tengyu Ma. Shape matters: Understanding the implicit bias of
   the noise covariance. arXiv preprint arXiv:2006.08680, 2020.
Sepp Hochreiter and Jürgen Schmidhuber. Flat minima. Neural Computation, 9(1):1–42, 1997.
Jean Honorio and Tommi Jaakkola. Tight bounds for the expected risk of linear classifiers and pac-bayes
  finite-sample guarantees. In Artificial Intelligence and Statistics, pp. 384–392. PMLR, 2014.
Arthur Jacot, Franck Gabriel, and Clément Hongler. Neural tangent kernel: Convergence and generalization
  in neural networks. In Advances in neural information processing systems, pp. 8571–8580, 2018.
Stanis law Jastrzebski, Zachary Kenton, Devansh Arpit, Nicolas Ballas, Asja Fischer, Yoshua Bengio, and
  Amos Storkey. Three factors influencing minima in sgd. arXiv preprint arXiv:1711.04623, 2017.


                                                    11
Yiding Jiang, Behnam Neyshabur, Hossein Mobahi, Dilip Krishnan, and Samy Bengio. Fantastic generaliza-
  tion measures and where to find them. arXiv preprint arXiv:1912.02178, 2019.
Nitish Shirish Keskar, Dheevatsa Mudigere, Jorge Nocedal, Mikhail Smelyanskiy, and Ping Tak Peter
  Tang. On large-batch training for deep learning: Generalization gap and sharp minima. arXiv preprint
  arXiv:1609.04836, 2016.
Yuanzhi Li, Tengyu Ma, and Hongyang Zhang. Algorithmic regularization in over-parameterized matrix
  sensing and neural networks with quadratic activations. arXiv preprint arXiv:1712.09203, pp. 2–47, 2017.
Zhiyuan Li, Tianhao Wang, and Sanjeev Arora. What happens after sgd reaches zero loss?–a mathematical
  framework. In International Conference on Learning Representations, 2021.
Zhiyuan Li, Tianhao Wang, and Dingli Yu. Fast mixing of stochastic gradient descent with normalization and
  weight decay. Advances in Neural Information Processing Systems, 35:9233–9248, 2022.
Kaifeng Lyu, Zhiyuan Li, and Sanjeev Arora. Understanding the generalization benefit of normalization
  layers: Sharpness reduction. arXiv preprint arXiv:2206.07085, 2022.
Chao Ma and Lexing Ying. On linear stability of sgd and input-smoothness of neural networks. Advances in
  Neural Information Processing Systems, 34:16805–16817, 2021.
Jiřı́ Matoušek. On variants of the johnson–lindenstrauss lemma. Random Structures & Algorithms, 33(2):
    142–156, 2008.
Mor Shpigel Nacson, Kavya Ravichandran, Nathan Srebro, and Daniel Soudry. Implicit bias of the step size
 in linear diagonal neural networks. In International Conference on Machine Learning, pp. 16270–16295.
 PMLR, 2022.
Behnam Neyshabur, Srinadh Bhojanapalli, David McAllester, and Nati Srebro. Exploring generalization in
  deep learning. In Advances in Neural Information Processing Systems, pp. 5947–5956, 2017.
Matthew D Norton and Johannes O Royset. Diametrical risk minimization: Theory and computations.
 Machine Learning, pp. 1–19, 2021.
Antonio Orvieto, Anant Raj, Hans Kersting, and Francis Bach. Explicit regularization in overparametrized
  models via noise injection. arXiv preprint arXiv:2206.04613, 2022.
Alessandro Rinaldo. 36-709: Advanced probability theory. https://www.stat.cmu.edu/˜arinaldo/
  Teaching/36709/S19/Scribed_Lectures, 2019.
Shai Shalev-Shwartz and Shai Ben-David. Understanding machine learning: From theory to algorithms.
  Cambridge university press, 2014.
Daniel Soudry, Elad Hoffer, Mor Shpigel Nacson, Suriya Gunasekar, and Nathan Srebro. The implicit bias of
  gradient descent on separable data. The Journal of Machine Learning Research, 19(1):2822–2878, 2018.
Nathan Srebro, Karthik Sridharan, and Ambuj Tewari. Smoothness, low noise and fast rates. In J. Lafferty,
  C. Williams, J. Shawe-Taylor, R. Zemel, and A. Culotta (eds.), Advances in Neural Information Processing
  Systems, volume 23. Curran Associates, Inc., 2010. URL https://proceedings.neurips.cc/paper_
  files/paper/2010/file/76cf99d3614e23eabab16fb27e944bf9-Paper.pdf.
Colin Wei and Tengyu Ma. Data-dependent sample complexity of deep neural networks via lipschitz aug-
  mentation. In Advances in Neural Information Processing Systems, pp. 9722–9733, 2019a.
Colin Wei and Tengyu Ma. Improved sample complexities for deep networks and robust classification via an
  all-layer margin. arXiv preprint arXiv:1910.04284, 2019b.
Colin Wei, Jason D Lee, Qiang Liu, and Tengyu Ma. Regularization matters: Generalization and optimization
  of neural nets vs their induced kernel. In Advances in Neural Information Processing Systems, pp. 9709–
  9721, 2019.


                                                   12
Colin Wei, Yining Chen, and Tengyu Ma. Statistically meaningful approximation: a case study on ap-
  proximating turing machines with transformers. Advances in Neural Information Processing Systems, 35:
  12071–12083, 2022.
Kaiyue Wen, Tengyu Ma, and Zhiyuan Li. How does sharpness-aware minimization minimize sharpness?
  arXiv preprint arXiv:2211.05729, 2022.

Blake Woodworth, Suriya Gunasekar, Jason D Lee, Edward Moroshko, Pedro Savarese, Itay Golan, Daniel
  Soudry, and Nathan Srebro. Kernel and rich regimes in overparametrized models. arXiv preprint
  arXiv:2002.09277, 2020.
Lei Wu and Weijie J Su. The implicit regularization of dynamical stability in stochastic gradient descent.
  arXiv preprint arXiv:2305.17490, 2023.
Lei Wu, Chao Ma, et al. How sgd selects the global minima in over-parameterized learning: A dynamical
  stability perspective. Advances in Neural Information Processing Systems, 31, 2018.
Chiyuan Zhang, Samy Bengio, Moritz Hardt, Benjamin Recht, and Oriol Vinyals. Understanding deep
  learning requires rethinking generalization. arXiv preprint arXiv:1611.03530, 2016.




                                                   13
Contents
1   Introduction                                                                                             1

2   Setup                                                                                                    3

3   Scenario I: All Flattest Models Generalize                                                               4
    3.1 Flattest models provably generalize . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .    4
    3.2 SAM empirically finds the flattest model that generalizes . . . . . . . . . . . . . . . . . . .      5

4   Scenario II: Both Flattest Generalizing and Non-generalizing Models Exist, and SAM Finds the
    Former                                                                                                   6
    4.1 Both generalizing and non-generalizing solutions can be flattest . . . . . . . . . . . . . . . .     6
    4.2 SAM empirically finds the non-generalizing solutions . . . . . . . . . . . . . . . . . . . . .       8

5   Scenario III: Both Flattest Generalizing and Non-generalizing Models Exist, and SAM Finds
    the Latter                                                                                           9
    5.1 Both generalizing and non-generalizing solutions can be flattest . . . . . . . . . . . . . . . . 9
    5.2 SAM empirically finds generalizing models . . . . . . . . . . . . . . . . . . . . . . . . . . 10

6   Discussion and Conclusion                                                                               10

A Omitted Proofs                                                                                            15
  A.1 Formal results for 2-MLP-Sim-BN . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .         15
      A.1.1 Proof of Theorem 3.2 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .        15
  A.2 Formal results for 2-MLP-No-Bias . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .        17
      A.2.1 Lemmas for uniform convergence . . . . . . . . . . . . . . . . . . . . . . . . . . .            17
      A.2.2 Proof of Lemma A.4 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .          21
      A.2.3 Proof of Theorem A.1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .          22
      A.2.4 Proof of Theorem 3.1 and lemma 3.1 . . . . . . . . . . . . . . . . . . . . . . . . .            23
  A.3 Formal results For MLP-Bias . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .       23
      A.3.1 Proof of Theorem 4.1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .        23
      A.3.2 Generalization of Proposition 4.2 . . . . . . . . . . . . . . . . . . . . . . . . . . .         24
  A.4 Formal results for 2-MLP-Sim-LN . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .         25
      A.4.1 Proof of Theorem 5.1 . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .        25
      A.4.2 Proof of Propositions 5.1 and 5.2 . . . . . . . . . . . . . . . . . . . . . . . . . . .         26
  A.5 Discussion on the choice of loss function . . . . . . . . . . . . . . . . . . . . . . . . . . . .     26
  A.6 Technical Lemmas . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .        27
      A.6.1 Concentration inequalities . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .        27
      A.6.2 Rademacher Complexity . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .           29
      A.6.3 Elementary inequalities . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .       31

B Experiments                                                                                               32
  B.1 Training Details . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .    32
  B.2 Extension To Uniform Ball Distribution . . . . . . . . . . . . . . . . . . . . . . . . . . . .        32
  B.3 Extension To Logistic Loss . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .      33
  B.4 Extension To Deeper Networks . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .        33




                                                      14
A       Omitted Proofs
We will define arg minθ (f (θ)) as the set of the minimizers f . When f has a unique minimizer, we also
overload the definition to refer to that element. We will use 1 to denote the indicator function.

A.1     Formal results for 2-MLP-Sim-BN
A.1.1    Proof of Theorem 3.2
We will first prove a lemma showing that the model with a sparse second layer weight in L1 norm will satisfy
the conclusion of Theorem 3.2.

Lemma A.1. Given any training set{(xi , yi )}ni=1 satisfying Condition 1, for any width m and any ϵ > 0,
for any width-m 2-MLP-Sim-BN ,f sbn and parameter θ satisfying fθsbn interpolates the training set and
∥γ ⊙ W2 ∥1 = inf L(θ′ )=0 ∥γ ′ ⊙ W2′ ∥1 , then it holds that ∀x ∈ {−1, 1}d , x[1]x[2] − fθ (x) = 0 and
∀i ∈ [m], ∥W1,i [3 : d]∥2 = 0.
Proof of Lemma A.1. As the training set satisfies Condition 1, there is an integer r such that n = 4r and we
assume WLOG for k ∈ 0, 1, .., r − 1, {x4k+t }t∈{0,1,2,3} has the same last d − 2 coordinates and the first two
       qPare (1, 1), (1, −1), (−1, 1), (−1, −1) respectively. Define ai ≜ W1 xi + b1 . Further define a as
coordinates
                  1             2
a[j] =      i∈[n] n relu(ai [j]) . As the model interpolates the training set, we have

                                m
                                X
                                  (γ ⊙ W2 )[j]yi relu(ai [j])/a[j] = 1, ∀i ∈ [n].
                                j=1

    Summing the above n equalities, we have that
                                     m
                                     X              n
                                                    X
                                       (γ ⊙ W2 )[j]   yi relu(ai [j])/a[j] = n.
                                     j=1               i=1

    This then implies
                              m
                              X                                         n
                                    |(γ ⊙ W2 )[j]| ≥              Pn                        .
                              j=1
                                                       maxj |      i=1 i relu(ai [j])/a[j]|
                                                                      y

    However, we have by Cauchy-Schwarz inequality that for each 1 ≤ j ≤ m,
                                            v 
                       n
                                            u
                                            u   r−1    3
                                                                               !2 
                       X                    u X X
                     |   yi relu(ai [j])| ≤ tr            y4k+t relu(a4k+t [j]) .
                        i=1                             k=0       t=0


   Notice that x4k+1 + x4k+4 = x4k+2 + x4k+3 , it holds that a4k+1 + a4k+4 = 2b1 = a4k+2 + a4k+3 and
hence by Lemma A.27, it holds that
                v 
                u
                      r−1 X 3
                                                    !2  v u n
                                                                                  n
                u     X                                    u X
                               y4k+t relu(a4k+t [j])  ≤ tr        relu(ai [j])2 = a[j].
                u 
                tr
                           t=0                                 i=1
                                                                                  2
                        k=0


    This then implies
                                           m
                                           X                             n
                                                 |(γ ⊙ W2 )[j]| ≥           = 2.
                                           j=1
                                                                        n/2




                                                             15
    One can then show 2 is the minimum of ∥γ ⊙ W2 ∥1 over minimizers by choosing the weights as

                 ′                                                                                       1
∀i, j ∈ {0, 1}, W1,2i+j+1 = [(−1)i , (−1)j , ..., 0], b′1 [2i + j + 1] = −1, W2′ [2i + j] =                , γ[2i + j + 1] = 1;
                                                                                                         2
         ′
∀k > 4, W1,k = [0, ..., 0], b′1 [k] = 0, W2′ [k] = 0, γ[k] = 1.

The training loss is minimized and ∥γ ⊙ W2 ∥1 = 2.
                                       ′     ′
    Hence ∥γ⊙W
             Pn 2 ∥1 = inf L(θ′ )=0 ∥γ ⊙W   2 ∥1 implies all the inequalities above must be equality. This implies
                                          n
∀j ∈ [m], | i=1 yi relu(ai [j])|/a[j] = 2 , which by Lemma A.27 then implies the following condition: for
any j ∈ [m], there exists tj such that a4k+t [j] ≤ 0 for t ̸= tj and a4k+tj [j] is a constant independent of k. 2
    This implies for any s1 , s2 ∈ S with S defined in Condition 1, it holds that ∀i ∈ [m], W1,i [3 :
d] (s1 − s2 ) = 0. As the linear space spanned by S − S has rank d − 2, this implies that ∀i ∈ [m], ∥W1,i [3 :
d]∥2 = 0. As the model predict correctly over the training set and does not use the last d − 2 coordinates, we
have that ∀x ∈ {−1, 1}d , x[1]x[2] − fθ (x) = 0. The proof is then complete.
    We also have an approximate version of the above lemma.
Lemma A.2. Given any training set {(xi , yi )}i∈[n] satisfying Condition 1, for any ϵ > 0 and width m,
there exists κ > 0, such that for any width-m 2-MLP-Sim-BN f sbn parameterized by θ satisfying fθsbn
interpolates the training set and ∥γ∥22 + ∥W2 ∥22 ≤ κ + inf L(θ)=0 ∥γ∥22 + ∥W2 ∥22 , it holds that ∀x ∈
{−1, 1}d , x[1]x[2] − fθ (x) ≤ ϵ and ∀i ∈ [m], ∥W1,i [3 : d]∥2 ≤ ϵ.
                                                                               P
Proof of Lemma A.2. Suppose for any κ > 0, there exists θκ , such that x∈{−1,1}d x[1]x[2] − fθκ (x) > ϵ,
L(θκ ) = 0 and ∥γκ ∥22 + ∥W2,κ ∥22 ≤ κ + inf L(θ)=0 ∥γ∥22 + ∥W2 ∥22 , We can normalize the first layer of
θκ such that ∥Wκ,1 ∥22 + ∥bκ,1 ∥2 = 1 without changing the function represented by the network. Then
(Wκ,1 , bκ,1 , W2,κ , γκ ) falls in a bounded set. Therefore there exists an accumulation point θ∗ = (W1∗ , b∗1 , W2∗ , γ ∗ )
of {θ1/i }i∈N , however as L(θ) and ∥γ∥22 + ∥W2 ∥22 are continuous functions of θ, this implies that L(θ∗ ) = 0
and ∥γ ∗ ∥22 + ∥W2∗ ∥22 = inf L(θ)=0 ∥γ∥22 + ∥W2 ∥22 .
    Notice by AM-GM inequality we havethat ∥γ∥22 +∥W2 ∥22 ≥ 2∥γ⊙W2 ∥1 and equality holds when γ = W2 .
We then have inf L(θ)=0 ∥γ∥22 + ∥W2 ∥22 = 2 inf L(θ)=0 ∥γ ⊙ W2 ∥1 and γ ∗ = W2∗ . Then by Lemma A.1,
we have that x∈{−1,1}d x[1]x[2] − fθ∗ (x) = 0. However θ∗ is an accumulation point of θκ satisfying
                P
P
   x∈{−1,1}d x[1]x[2] − fθ1/i (x) > ϵ. This then leads to a contradiction.

    We will now lower bound the sharpness of the model,
Lemma A.3. For any parameter θ for architecture 2-MLP-Sim-BN
                                                               satisfying that L(θ) = 0, it holds that
Tr ∇2 L(θ) ≥ ∥W2 ∥22 + ∥γ∥22 and inf L(θ)=0 Tr ∇2 L(θ) = inf L(θ)=0 ∥W2 ∥22 + ∥γ∥22 .
Proof of Lemma A.3. Define ai and a as in proof of Lemma A.1. By Lemma 4.1,
                                           n
                                        1X
                           Tr ∇2 L(θ) =       ∥∇θ fθsbn (xi )∥22
                                     
                                        n i=1
                                                     n
                                               1X
                                             ≥       ∥∇γ fθsbn (xi )∥22 + ∥∇W2 fθsbn (xi )∥22
                                               n i=1
                                                     m     n
                                                 1 XX
                                             =             |W2 [j]ai [j]/a[j]|22 + |γ[j]ai [j]/a[j]|22
                                                 n j=1 i=1
                                             = ∥W2 ∥22 + ∥γ∥22 .

This inequality can approximately reach equality when W2 = γ and ∥W1 ∥2 is sufficiently large.

    Now we are ready to prove Theorem 3.2.
Proof of Theorem 3.2. This is a direct consequence of Lemma A.2 and Lemma A.3.
   2Notice that for any k, the order of a4k+t [j] is the same.



                                                                 16
A.2     Formal results for 2-MLP-No-Bias
We will prove a more general result that holds for any data distribution satisfying the following condition.
Condition 2. There exists constant C and σ independent of d and m such that the data distribution D over
data points x and label y satisfies that x is symmetric3 subgaussian random vector (Definition A.4) with
parameter σ and variance Id . Further, there
                                            exists parameter
                                                     Pm       θ = (W1 , W2 ) for width-m architecture 2-
MLP-No-Bias such that Pr fθnobias (x) = y = 1 and j=1 ∥W1,j ∥2 |W2,j | ≤ C.

Theorem A.1. Given any  data distribution D satisfying Condition 2, for any δ ∈ (0, 1) and input dimension
d, for n = Ω d log dδ , with probability at least 1 − δ over the random draw of training set {(xi , yi )}ni=1
                        Pn
from Dn , let L(θ) ≜ n1 i=1 ℓmse (fθnobias (xi ), yi ) be the training loss for width-m 2-MLP-No-Bias, it holds
that for all θ∗ ∈ arg minL(θ)=0 Tr ∇2 L (θ) ,

                                     Ex,y∼D ℓmse fθnobias
                                                                
                                                    ∗     (x) , y ≤ Õ (d/n) .

    We will also prove a more general result than Lemma 3.1.
                                                Pm
Lemma A.4. Define ΘC ≜ {θ = (W1 , W2 ) | j=1 ∥W1,j ∥2 |W2,j | ≤ C}. Under the setting of Theorem A.1,
                                                                   with probability at least 1 − δ over the
there exists a absolute constant C1 independent of d and δ, such that
randomness of dataset {(xi , yi )}ni=1 , arg minL(θ)=0 Tr ∇2 L (θ) ⊆ ΘC1 and RS ({fθnobias | θ ∈ ΘC1 }) ≤
   p      
Õ     d/n .

A.2.1   Lemmas for uniform convergence
We will begin with two uniform convergence bounds that will be used in the proof of Lemma 3.1.
Lemma A.5. Given any data distribution D satisfying Condition 2, there exists constant C2 > C           1 > 0
depending on σ, for any δ ∈ (0, 1), input dimension d, and number of samples n = Ω d log dδ , with
probability at least 1 − δ over the random draw of set {(xi , yi )}ni=1 from Dn , for any w ∈ Rd , we have that,
                                                  n
                                               1X
                                                     ∥xi ∥22 1 w⊤ xi > 0 ≥ C1 d.
                                                                        
                                      C2 d ≥                                                                   (6)
                                               n i=1

Lemma A.6. Given any data distribution D satisfying Condition 2, there exists constant C2 > C           1 > 0
depending on σ, for any δ ∈ (0, 1), input dimension d, and number of samples n = Ω d log dδ , with
probability at least 1 − δ over the random draw of set {(xi , yi )}ni=1 from Dn , for any w ∈ Rd , ∥w∥2 = 1, we
have that,
                                              n
                                           1X ⊤ 2
                                                 |w xi |2 1 w⊤ xi > 0 ≥ C1 .
                                                                     
                                      C2 ≥                                                                     (7)
                                           n i=1

   We will first prove Lemma A.5 by a combination of Concentration Inequalities and uniform convergence
bound based on Rademacher complexity.
Proof of Lemma A.5. We will first prove the upper bound, by Lemma A.17, it holds that

                                                                                 δ
                                     Pr(∥Xi ∥22 ≥ 32σ 2 d + 8σ 2 log(2n/δ)) ≤      .
                                                                                2n
   Hence when log(2n/δ) ≤ 1024d, the proof for the upper bound is complete. If log(2n/δ) > 1024d, then
by Lemma A.17 and Chebyshev’s inequality, we have that
                    n
                 1X                      4Var(∥x∥22 )   2048d2
           Pr(         ∥xi ∥22 > 3d/2) ≤     2
                                                      ≤        ≤ 2048d2 exp(−1024d)δ ≤ δ/2.
                 n i=1                      d n           n

  3x and −x equal in distribution.


                                                          17
This concludes the proof of the upper bound.
               √ prove the lower bound. We will first show that there exists Ω(n) data points in {xi }, such
    We will then
that ∥xi ∥ ≥ Ω( d). By Lemma A.19, we have there exists constant ϵ, ζ such that Pr(∥x∥22 > ϵd) > ζ.
    Define indicator bi ≜ 1 ∥xi ∥22 ≥ ϵd , then bi are i.i.d Bernoulli random variables. We then have
p = Pr(bi = 1) > ζ. By Chernoff’s bound, it holds that
                                        n
                                                   !
                                     1X                                  δ
                                Pr         bi ≤ p/2 ≤ exp(−np/8) ≤ ,
                                     n i=1                               4

for any n > 8 log(1/δ)
                  ζ    . This shows that with probability at least 1 − 4δ , we have that,
                                                 n
                                             1X
                                                   bi ≥ p/2 ≥ ζ/2.
                                             n i=1

    This shows that there exists n′ ≥ ⌊ζn/2⌋ data points in {xi }, such that ∥xi ∥22 ≥ ϵd. We can then relabel
the data points as z1 , ..., zn′ . Then zi are i.i.d random variables with ∥zi ∥22 ≥ 1/2 conditioning on the value
of n′ . We can then have for any w ∈ Rd ,
                                                               ′
                             n                           n
                          1X                          1X
                                ∥xi ∥22 1 w⊤ xi > 0 ≥       ∥zi ∥22 1 w⊤ zi > 0
                                                                              
                          n i=1                       n i=1
                                                                   n′
                                                         ζ 1 X
                                                                 ∥zi ∥22 1 w⊤ zi > 0
                                                                                     
                                                       ≥    ′
                                                         2 n i=1
                                                                        ′
                                                                n
                                                         ζϵd 1 X
                                                                  1 w ⊤ zi > 0 .
                                                                              
                                                       ≥      ′
                                                          2 n i=1

   Finally, define F = {z → 1 w⊤ z > 0 }, by Lemma A.22, we have that VC(F) ≤ d. By Corollary A.1,
                                       
                                                                                         q
                                                                                                 n′
we have that the empirical Rademacher complexity of F on {zi }i∈[n′ ] is upper bounded by 4d log
                                                                                             n′     ≤
 q
   d log n
4    ζn .
   By Lemma A.23, with probability at least 1 − 4δ , we have that
                                                                             s
                                      n′
                                 1 X                                             log 4δ
                            sup ′      f (zi ) − E[f (z)] ≤ 2Rn′ (F) + 3                .
                            f ∈F n i=1                                             n′

   The symmetry of xi implies that E[f (zi ) | n′ ] = 1/2. This shows that with probability at least 1 − 4δ , we
have that for any w ∈ Rd ,
                                                                        s
                               n′
                           1  X
                                      ⊤
                                                    1                    log 4δ
                                  1 w   z i > 0    ≥   − 2R n ′ (F) − 3
                           n′ i=1                    2                      n′
                                                           s             s
                                                     1       d log n        log 4δ
                                                   ≥ −8              −3            .
                                                     2           ζn          ζn

   Hence when n = Ω(d log(d/δ)), with probability at least 1 − δ/2, we have that for any w ∈ Rd ,
                                                                    ′
                           n                              n
                        1X                        ϵCd 1 X             ϵζd
                              ∥xi ∥22 1 w⊤ xi > 0 ≥     ′
                                                            1 w⊤ z > 0 ≥    .
                        n i=1                       2 n i=1              8

   This concludes our proof.


                                                        18
   We will then prove Lemma A.6, and we will use the following lemma motivated by Matoušek (2008).
Lemma A.7. Given any data distribution D satisfying Condition 2, there exists C depending on σ, such that
for any δ ∈ (0, 1), input dimension d, number of samples n ≥ C log(2/δ), and w ∈ Rd , ∥w∥2 = 1, with
probability at least 1 − δ over the random draw of set {(xi , yi )}ni=1 from Dn , we have that,
                                                           n
                                                    3   1X ⊤ 2         1
                                                      ≥       |w xi | ≥ .                                       (8)
                                                    2   n i=1          2

Proof of Lemma A.7. Notice that w⊤ xi is a subgaussian random variable with parameter σ 2 . Hence by Lemma A.14,
w⊤ xi is a subexponential random variable with expectation 1. The rest follows from Lemma A.16.
    This lemma can be viewed as a variant of the Johnson-Lindenstrauss lemma. We will now proceed     to
show that we can prove a similar high probability bound when the indicator function 1 w⊤ xi > 0 is taken
                                                                                               

into account.

Lemma A.8. Given any data distribution D satisfying Condition 2, there exists C depending on σ, such that
for any δ ∈ (0, 1), input dimension d, sample complexity n ≥ C log(4/δ) and w ∈ Rd , ∥w∥2 = 1, with
probability at least 1 − δ over the random draw of set {(xi , yi )}ni=1 from Dn , it holds that,
                                                 n
                                          3   1X ⊤ 2                    1
                                            ≥       |w xi | 1 w⊤ xi > 0 ≥ .                                     (9)
                                          2   n i=1                      8

Proof of Lemma A.8. The upper bound is a direct consequence of Lemma A.7.
    We will now prove the lower bound. We will first use a doubling trick using the symmetry of the data
distribution. Randomly sample {bi }i∈n uniformly from {−1, 1}n , and     define zi = bi xi . We have that
                                                                                                         zi and
xi equals in distribution. Hence, we have that |w⊤ xi |2 1 w⊤ xi > 0 and |w⊤ bi xi|2 1 bi w⊤ xi > 0 equals
in distribution. As bi is independent with xi , this shows that |w⊤ xi |2 1 w⊤ xi > 0 equals in distribution to
|w⊤ xi |2 ci where ci is a Rademacher random variable independent of xi .4 Hence, we only need to prove that
                                                       n
                                                   1X ⊤ 2            1
                                             Pr(         |w xi | ci < ) < δ/2.                                 (10)
                                                   n i=1             8

By Chernoff bound, we have that
                                                       n
                                                    1X        1         n
                                              Pr(         ci ≤ ) ≤ exp(− ).                                    (11)
                                                    n i=1     4         16
                                                              Pn
Hence when n > 16 log(4/δ), we have that Pr( n1                           1
                                                                 i=1 ci ≤ 4 ) < δ/4. This then implies that,

           n                          n                  n                   n
        1X ⊤ 2            1        1X        1        1X ⊤ 2            1 1X        1   δ
  Pr(         |w xi | ci < ) ≤ Pr(       ci ≤ ) + Pr(       |w xi | ci < |      ci ≥ ) ≤ , (12)
        n i=1             8        n i=1     4        n i=1             8 n i=1     4   2

for the last inequality, we apply Lemma A.7. This concludes our proof.
    Notice that Lemma A.8 is a point-wise bound, we will then use the technique of covering to prove a
uniform bound. We will first prove a uniform bound for the case where the indicator function is not taken into
account.

Definition A.1 (ϵ-covering). A set S ∈ Rd is an ϵ-covering of a set S ′ ∈ Rd , if and only if ∀s′ ∈ S ′ , ∃s ∈
S, ∥s − s′ ∥2 ≤ ϵ.
Lemma A.9. For any ϵ > 0, there exists an ϵ-covering S of the unit sphere in Rd with cardinality smaller
           d
than 2ϵ + 1 .
  4For special case where w⊤ xi = 0, this still holds as both sides are zero.



                                                                  19
Proof. Consider a maximal subset T of the unit sphere in Rd satisfying that ∀t ̸= t′ ∈ T, ∥t − t′ ∥2 ≥ ϵ. As
T is maximal, it is an ϵ-covering of the unit sphere.
     Further consider set T ′ = {x | ∃t, ∥x − t∥2 ≤ 2ϵ }, which is the union of |T | disjoint balls with radius ϵ/2.
Hence T ′ has volume C|T |( 2ϵ )d with C being the volume of the unit ball in Rd . However, T ′ is contained
                                                                                                  d
in a ball with radius 1 + 2ϵ centered at origin. Hence it holds that C|T |( 2ϵ )d ≤ C 1 + 2ϵ . This implies
               d
|T | ≤ 2ϵ + 1 .
Lemma A.10. Given any data distribution D satisfying Condition 2, there exists C depending on σ, such that
for any δ ∈ (0, 1), input dimension d and sample complexity n ≥ Cd log( dδ ), with probability at least 1 − δ
over the random draw of set {(xi , yi )}ni=1 from Dn , it holds that for any w ∈ Rd , ∥w∥2 = 1,
                                                     n
                                                 1X ⊤ 2         1
                                            2≥         |w xi | ≥ .                                             (13)
                                                 n i=1          4

Proof of Lemma A.10. Consider a 1/16 covering of the unit sphere in Rd , w1 , ..., wN , we have that N ≤ 64d .
By Lemma A.7 and Union Bound, we have with probability at least 1 − δ over the random draw of set
{(xi , yi )}ni=1 from Dn , for any k ∈ [N ], we have that,
                                                   n
                                            3   1X ⊤ 2          1
                                              ≥       |wk xi | ≥ .
                                            2   n i=1           2
    Now suppose the above event happens and
                                             n                                 n
                                          1X ⊤ 2                            1X ⊤ 2
                     w∗ =     arg max           |w xi | , γ =    max              |w xi | .                    (14)
                            w∈Rd ,∥w∥2 =1 n i=1               w∈Rd ,∥w∥2 =1 n
                                                                              i=1

   Since {wi }1i=1 is a 1/16 covering of unit sphere, there exists k ∈ [N ] such that ∥w∗ − wk ∥ ≤ 16
                                                                                                    1
                                                                                                      , then we
have that by Cauchy-Schwarz inequality,

                                n               n
                          3  1 X ∗⊤ 2 1 X ⊤ 2
                        γ− ≤       |w xi | −      |w xi |
                          2  n i=1           n i=1 k
                                      n
                                1X
                               ≤      |(w∗ − wk )⊤ xi |2 |(w∗ + wk )⊤ xi |2
                                n i=1
                                v                           v
                                u n                         u n
                                u1 X                        u1 X
                               ≤t        |(w∗ − wk )⊤ xi |2 t       |(w∗ + wk )⊤ xi |2
                                  n i=1                       n i=1

                              ≤ γ∥w∗ − wk ∥∥w∗ + wk ∥
                                 γ
                              ≤ .
                                 8
    This then implies that γ ≤ 2. Hence, with probability 1 − δ, we have that the upper bound holds.
                                                             1
    Now for any w ∈ Rd , ∥w∥2 = 1, suppose ∥w − wk ∥2 ≤ 16     , we have that
                    n               n              n
                 1X ⊤ 2          1X ⊤ 2 1X
                                                      |w⊤ xi |2 − |wk⊤ xi |2
                                                                             
                       |w xi | ≥      |w xi | +
                 n i=1           n i=1 k        n i=1
                                         n
                                  1   1X
                                 ≥  −      |(w∗ − wk )⊤ xi ||(w∗ + wk )⊤ xi |
                                  2 n i=1
                                      v                          v
                                      u n                        u n
                                  1   u 1  X                     u1 X
                                 ≥ −t         |(w∗ − wk )⊤ xi |2 t       |(w∗ + wk )⊤ xi |2
                                  2     n i=1                      n i=1
                                   1     1    1
                                 ≥   −γ ≥ .
                                   2     8    4
This shows that with probability 1 − δ, the lower bound holds as well. The proof is complete.


                                                         20
   We can now prove Lemma A.6.
Proof of Lemma A.6. By Lemma A.10, we have that with probability at least 1 − δ/2 over the random draw
of set {(xi , yi )}ni=1 from Dn , for any w ∈ Rd , ∥w∥2 = 1, we have that,
                                                          n
                                                      1X ⊤ 2           1
                                                            |w xi | ∈ [ , 2].
                                                      n i=1            4

    This directly implies the upper bound. For lower bound, consider a 1/64 covering of the unit sphere in
Rd , w1 , ..., wN , we have that N ≤ 128d . By Lemma A.8 and Union Bound, we have with probability at least
1 − δ/2 over the random draw of set {(xi , yi )}ni=1 from Dn , for any k ∈ [N ], we have that,
                                              n
                                           1X ⊤ 2                    1
                                                |w xi | 1 wk⊤ xi > 0 ≥ .
                                           n i=1 k                    8
                                                                                           1
   Now suppose the above event happens and for any w ∈ Rd , ∥w∥2 = 1, suppose ∥w − wk ∥ ≤ 32 ,
by Lemma A.26, we have that
               n
            1X ⊤ 2
                  |w xi | 1 w⊤ xi > 0
                                     
            n i=1
               n                             n
            1X ⊤ 2                      1X
                  |wk xi | 1 wk⊤ xi > 0 +       |w⊤ xi |2 1 w⊤ xi > 0 − |wk⊤ xi |2 1 wk⊤ xi > 0
                                                                                               
        ≥
            n i=1                         n i=1
              n
         1  1X
        ≥ −      |(w − wk )⊤ xi |2 (|w⊤ xi | + |wk⊤ xi |)
         8 n i=1

            v                v                         v                 v
            u n              u n                       u n               u n
         1 u  1 X
                      ⊤
                             u1 X
                                               ⊤
                                                       u1 X
                                                                 ⊤
                                                                         u1 X
        ≥ − t       |w xi |2 t                      2
                                     |(w − wk ) xi | − t               2
                                                               |wk xi | t        |(w − wk )⊤ xi |2
         8    n i=1            n i=1                     n i=1             n i=1
         1       1    1
        ≥ −2×2×    =    .
         8      64   16
This completes the proof.

A.2.2    Proof of Lemma A.4
Based on Appendix A.2.1, we are now ready to show that for 2-MLP-No-Bias, sharpness is within a constant
factor of the norm of the parameters.
Lemma A.11. Given any data distribution D satisfying Condition 2, there exists constant C2 >       C1 > 0
depending on σ, for any δ ∈ (0, 1), input dimension d and number of samples n = Ω d log dδ , with
probability at least 1 − δ over the random draw of training set {(xi , yi )}ni=1 from Dn , for any parameter
θ = (W1 , W2 ) of 2-MLP-No-Bias satisfying that L(θ) = 0, it holds that,
                   C2 ∥W1 ∥2F + d∥W2 ∥2 ≥ Tr ∇2 L(θ) ≥ C1 ∥W1 ∥2F + d∥W2 ∥2 .
                                                                                     

Proof of Lemma A.11. By Lemma 4.1, we have that,
                            n                     2                 2
                         2 X ∂L                                ∂L
            Tr ∇2 L(θ) =
                      
                                                      +
                         n i=1 ∂W1                F           ∂W2   2
                               n
                              2X
                                       ∥W2 ⊙ 1 [W1 xi > 0] ∥22 ∥xi ∥2 + ∥relu (W1 xi ) ∥22
                                                                                                      
                          =
                              n i=1
                              m                           n
                                                                                     !       n
                                                                                                                            !
                              X                       2X                                     X
                          =           ∥W2,j ∥22             1 [W1,j xi > 0] ∥xi ∥2       +         |relu (W1,j xi ) |   2
                                                                                                                                .
                              j=1
                                                      n i=1                                  i=1



                                                                    21
By Equations (6) and (7), there exists C2 > C1 , such that for any w ∈ Rd , it holds that
                                n
                             1X  ⊤
                                   1 w xi > 0 ∥xi ∥2 ∈ [C1 d/2, C2 d/2].
                                             
                             n i=1
                                     n
                                  1X
                                        |relu w⊤ xi |2 ∈ [C1 ∥w∥2 /2, C2 ∥w∥2 /2].
                                                   
                                  n i=1

This then implies our result.
      We can now prove Lemma A.4.

Pm of Lemma A.4. By Condition 2, there exists parameter θ = (W1 , W2 ), such that2 L(θ) = 0 2and
Proof
      ∥W ∥ |W | ≤ C. We can properly rescale W1,j and W2,j such that ∥W1 ∥F + d∥W2 ∥ =
 √j=1Pm 1,j 2 2,j              √
2 d j=1 ∥W1,j ∥2 |W2,j | ≤ 2C d.
    Now by Lemma A.11, we have that there exists C2 > C1 > 0, such that for any θ∗ = (W1∗ , W2∗ ) ∈
arg minL(θ)=0 Tr ∇2 L(θ) , it holds that
                                       √
                                  2C2 C d ≥ C2 ∥W1 ∥2F + d∥W2 ∥2
                                                                  

                                          ≥ Tr ∇2 L(θ) ≥ Tr ∇2 L(θ∗ )
                                                                        

                                          ≥ C1 ∥W1∗ ∥2F + d∥W2∗ ∥2
                                                                   

                                               √ X m
                                                          ∗       ∗
                                          = 2C1 d      ∥W1,j ∥2 |W2,j |.
                                                        j=1

                          Pm      ∗
    This then implies that j=1 ∥W1,j       ∗
                                     ∥2 |W2,j | ≤ CC21C , completing the proof of the first claim.
                                                                                      √
    By Lemma A.17, with probability at least 1 − δ, we have that maxi ∥xi ∥22 = Õ( d). The second claim
then follows from Lemma A.24.

A.2.3    Proof of Theorem A.1
We are now ready to prove Theorem A.1 based on Lemma A.4.
Proof of Theorem A.1. Based on Lemma A.4, there exists constant C1 > C with C defined in Condition
                                                                                               q 2,
with probability at least 1 − δ, such that arg minL(θ)=0 Tr ∇ L(θ) ⊂ ΘC1 and RS (ΘC1 ) = Õ( nd ).
                                                             2
                                                                  

To get the faster rate Õ(d/n), we would like to apply Theorem A.3. The main technical difficulty to
apply Theorem A.3 here is that for distribution D, the loss function L is not necessarily bounded. To address
this issue, we will consider a truncated version of the mean squared error (as in Gatmiry et al. (2023)).
                               
                                         2
                               (x − y) ,
                                                                  if x − y ∈ [−c, c],
                                           2                  2
      lc (x, y) = ℓc (x − y) = −(x − y) + 4c|x − y| − 2c , if x − y ∈ [−2c, −c] ∪ [c, 2c],                (15)
                               
                                2
                                 2c ,                              if x − y ∈ (−∞, −2c] ∪ [2c, ∞).
                           √
     We will choose c = Õ( d) as in Lemma A.18 such that Ex,y∼D [∥x∥2 1 [C1 ∥x∥ ≥ c]] = Õ( nd ) and
Ex,y∼D [∥x∥2 1 [C∥x∥ ≥ c]] = Õ( nd ). By Condition 2, we have for x, y ∼ D, there exists θ1∗ ∈ ΘC such that
fθnobias
   ∗     (x) = y, then
  1


           Ex,y∼D [ℓmse (fθnobias (x), y)] − Ex,y∼D [lc (fθnobias (x), y)]
          ≤Ex,y∼D [(fθnobias (x) − y)2 1 |fθnobias (x) − y| ≥ c ]
                                                                 

          ≤2Ex,y∼D [fθnobias (x)2 1 |fθnobias (x)| ≥ c ] + 2Ex,y∼D [fθ∗  nobias
                                                                                (x)2 1 |fθ∗
                                                                                     nobias         
                                                                                               (x)| ≥ c ].




                                                        22
   As we have θ ∈ ΘC1 , it holds that
                                                  m
                                                  X
                               |fθnobias (x)| ≤         |W2,i |∥W1,i ∥2 ∥x∥ ≤ C1 ∥x∥.
                                                  i=1

   This then implies that,

                                                                                               d
          Ex,y∼D [fθnobias (x)2 1 |fθnobias (x)| ≥ c ] ≤ C12 Ex,y∼D [∥x∥2 1 [C1 ∥x∥ ≥ c]] = Õ( ).
                                                    
                                                                                               n

                                                (x)| ≥ c ] = Õ( nd ). Hence, we have that
                                                       
   Similarly, Ex,y∼D [fθnobias
                         ∗     (x)2 1 |fθnobias
                                          ∗



                                                                                          d
                     Ex,y∼D [ℓmse (fθnobias (x), y)] − Ex,y∼D [lc (fθnobias (x), y)] = Õ( ).
                                                                                          n
                                                                Pn
    We then define the truncated version of L as Lc (θ) = n1 i=1 lc (W2⊤ relu(W1 xi ), yi ). Then we clearly
have L(θ) = 0 =⇒ Lc (θ) = 0 Now by Theorem A.3, we have that for any θ ∈ ΘC1 and L(θ) = 0, it holds
that with probability at least 1 − δ/2,

                                                                d + c2 log(1/δ)        d
                        Ex,y∼D [lc (fθnobias (x), y)] ≤ Õ(                     ) = Õ( ).
                                                                       n               n
   This completes the proof.

A.2.4   Proof of Theorem 3.1 and lemma 3.1
                                                                                     
One can easily construct width 4 2-MLP-No-Bias such that for Prx,y∼D fθnobias (x) = y = 1. For example,
one can have that
                                                 
                           1+ϵ      1 − ϵ 0 ···
                          1 + ϵ −1 + ϵ 0 · · ·                1                   
                  W1 =  −1 − ϵ 1 − ϵ 0 · · · , W2 = 2 − 2ϵ 1 −1 −1 1 .
                                                  

                          −1 − ϵ −1 + ϵ 0 · · ·

   Hence Pxor satisfies the condition in Condition 2 and this completes the proof of Theorem 3.1 and lemma 3.1.



A.3     Formal results For MLP-Bias
We will prove Theorem 4.1 and a generalization of Proposition 4.2 in this section. We note that Proposition 4.1
is already proved in Section 4.

A.3.1   Proof of Theorem 4.1
We have demonstrated the proof for 2-MLP-Bias in Section 4, and the proof for layer-D MLP-Biasis concep-
tually similar.
Proof of Theorem 4.1. We will still use notation x′i ∈ Rd+1 to denote transformed input satisfying ∀j ∈
[d], x′i [j] = xi [j], x′i [d + 1] = 1 and W1′ = [W1 , b1 ] ∈ Rm×(d+1) to denote the transformed weight matrix.
     For the simplicity of writing, we will use the following notations,

                      ai,0 = x′i , ai,1 = relu(W1 xi + b1 ), ai,d = relu(Wd ai,d−1 ), d > 1

   We will also use Ai,d to denote the diagonal matrix with 1 (ai,d > 0) as the diagonal entries.




                                                           23
    By Lemma 4.1 and the chain rule, we have that
                                           D
                                           X
               ∥∇θ fθbias,D (xi )∥22 =           ∥∇Wj fθ (xi )∥2F + ∥∇W1′ fθ (xi )∥2F
                                           j=2
                                           D−1
                                           X
                                      =          ∥WD Ai,D−1 · · · Wj+1 Ai,j ∥22 ∥ai,j−1 ∥22 + ∥ai,D−1 ∥22
                                           j=1

    By AM-GM inequality and Cauchy-Schwarz inequality, we have that
                                     D−1
                                     X
           ∥∇θ fθbias,D (xi )∥22 =         ∥WD Ai,D−1 · · · Wj+1 Ai,j ∥22 ∥ai,j−1 ∥22 + ∥ai,D−1 ∥22
                                     j=1
                                                                                                           1
                                           ΠD−1                                 2           2            2 D
                                                                                              
                               ≥D           j=1 ∥W D A i,D−1 · · · W j+1 A i,j ∥2 ∥ai,j−1 ∥ 2   ∥ai,D−1 ∥2
                                                                                                   1
                                           Πj=1 ∥WD Ai,D−1 · · · Wj+1 Ai,j ∥22 ∥ai,j ∥22 ∥x′i ∥22 D
                                            D−1
                                                                                         
                               ≥D
                                                             2/D
                               ≥ D|yi |2(D−1)/D ∥x′i ∥2            .

    As every training data point is an extreme point of the convex hull of {xi }, for each input data point xi ,
there exists a vector ∥wi ∥ = 1, wi ∈ Rd , such that ∀j ̸∈ i, wi⊤ xi > wi⊤ xj . Finally, the above inequality can
be reached by a memorizing solution when we choose,

                                W1 = [ui wi /ϵ]⊤                  ⊤
                                                                          ⊤
                                               i , b1 = [ui −wi xi + ϵ /ϵ]i ,
                                 Wj = diag([1/ri ]i∈[n] ), ∀2 ≤ j ≤ D − 1,
                                 WD = [sign(yi )/ri ]i∈[n] ,

with ri , ui satisfyng ri = (∥x′i ∥/|yi |)1/D , ui = |yi |riD−1 when yi ̸= 0, ri = ui = 1 when yi = 0. The proof
is then completed.

A.3.2   Generalization of Proposition 4.2
We will directly prove a more general version of Proposition 4.2, which is Proposition A.1.
Proposition A.1. Given any constant s, for any data distribution D over input x and label y satisfying that
(1) the label y depends only on the first s coordinates I of the input, (2) xI are sampled from a set of extreme
points in R|I| and (3) Pr(∥x∥2 = R) = 1, for sufficiently large width m depending on D, there exists a flattest
minimizer θ∗ for width-m 2-MLP-Bias with generalization error 0.
Proof of Proposition A.1. The proof is similar to the proof of Proposition 4.2. Suppose the set of extreme
points in R|I| contains k elements v1 , ..., vk satisfying ∥vk ∥ = v and corresponds to label y1 , ..., yk . Then
there exist vectors ∥wi ∥ = 1, wi ∈ R|I| , such that ∀j ̸= i, wi⊤ vi > wi⊤ vj . We will then choose m = k and
let,

                   ∀j ∈ [k], W1,j = r[vj , ..., 0]/ϵ, b1 [j] = r(−wj⊤ vj + ϵ)/ϵ, W2 [j] = yj /r,                (16)

with r2 = |yj |(R2 + 1) and ϵ sufficiently small. It is easy to verify that the construction will reach the smallest
sharpness for any training set.
    The construction above critically relies on the fact that there exists a set of extreme points in R|I| containing
the input data points. We will show that this is not necessary by the following example.
Proposition A.2. Given any constant L, for any data distribution D over input x and label y = f (x) satisfying
that (1) the label function f depends only on the first 2 coordinates I of the input and is L-lipschiz, (2) the
input data points satisfy xI are sampled uniformly from the unit circle in R2 , and (3) Pr(∥x∥2 = R) = 1,
for any δ ∈ (0, 1/20) and n = Ω(log(1/δ)/δ), with probability 1 − δ over the random draw of training set
{(xi , yi )}i∈[n] , there exists a flattest minimizer θ∗ for width-n 2-MLP-Bias with generalization error O(δ 2 ).


                                                              24
Proof. Suppose the largest value of label y is Y . Suppose for dataset {(xi , yi )}, the first two coordinates of
{xi } forms a set {vi } that lies on the unit circle in R2 and corresponds to label {yi }. Suppose WLOG vi is
sorted by the angle it forms with the x-axis. We will then define zi as the midpoint of the arc vi−1 vi and wi
as the unit vector perpendicular to zi zi+1 . Here zn+1 = z1 and w0 = wn . The flattest minimizers θ∗ is then
defined as,
           ∀j ∈ [n], W1,j = r[wj , ..., 0]/wj⊤ zj , b1 [j] = r(−wj⊤ vj + wj⊤ zj )/wj⊤ zj , W2 [j] = yj /r, (17)
               √
with r2 = |yj | R2 + 1. Verifying that the construction will reach the smallest sharpness for the training set
is easy. Now splitting the sphere into N = ⌈2π/δ⌉ > 1/δ arcs with length no longer than δ. Then by the
standard coupon collector problem, with probability at least 1 − δ, when n ≥ N log δ, there is at least one
point in each arc. Under such case, we have that zj zj+1 has length no greater than 2δ and wj⊤ zj > 1 − 10δ
for any j.
     Therefore, for any v ∈ R2 , ∥v∥ = 1, suppose WLOG v fails in arc z1 z2 and corresponds to label y, then
fθ∗ (x) = y1 w1⊤ (v − v1 + z1 )/w1⊤ z1 for x[1 : 2] = v . Therefore, we have that
  bias


                        ∥fθbias
                            ∗   (x) − y∥22 ≤ ∥fθbias
                                                 ∗   (v) − y1 ∥22 + ∥y1 − y∥22
                                           ≤ ∥y1 w1⊤ (v − v1 )/w1⊤ z1 ∥22 + L2 ∥v − v1 ∥22
                                           ≤ 4Y 2 δ 2 /(1 − 10δ)2 + L2 δ 2 .
This shows that the expected generalization error is bounded by O(δ 2 ). The proof is completed.

A.4     Formal results for 2-MLP-Sim-LN
A.4.1   Proof of Theorem 5.1
We will first lower bound the sharpness of all minimizers of 2-MLP-Sim-LN by the following lemma.
Lemma A.12. Given any number of samples n and ϵ > 0, for any training set {(xi , yi )}i∈[n] satisfying that
the input data points {xi } of the training set form a set of extreme points, for width-n 2-MLP-Sim-LN with
hyperparameter ϵ, it holds that
                                                      n
                                                   2X             2
                                                                    q
                            inf Tr ∇2 L(θ) ≥
                                              
                                                         min(1,       ∥xi ∥22 + 1|yi |).
                          L(θ)=0                   n i=1          ϵ

Proof. By Lemma 4.1, we have that
                                                     n
                                                  2X
                                     Tr ∇2 L(θ) =       ∥∇θ fθ (xi )∥22 .
                                               
                                                  n i=1

We will then discuss by cases to show the lower bound of ∥∇θ fθ (xi )∥22 for each i ∈ [n] when fθ (xi ) = yi ,
we will continue to use notation x′i ∈ Rd+1 to denote transformed input satisfying ∀j ∈ [d], x′i [j] =
xi [j], x′i [d + 1] = 1 and W1′ = [W1 , b1 ] ∈ Rm×(d+1) to denote the transformed weight matrix.
   1. If ∥relu(W1 xi + bi )∥2 > ϵ, then it holds that
                                    ∥∇θ fθ (xi )∥22 ≥ ∥∇W2 fθ (xi )∥22
                                                          relu(W1 xi + bi ) 2
                                                   =∥                       ∥ = 1.
                                                        ∥relu(W1 xi + bi )∥2 2

   2. If ∥relu(W1 xi + bi )∥2 ≤ ϵ, then it holds that
                ∥∇θ fθ (xi )∥22 ≥ ∥∇W1′ fθ (xi )∥22 + ∥∇W2 fθ (xi )∥22
                                   1
                                = 2 (∥W2⊤ 1 (relu(W1 xi + bi ) > 0) ∥22 ∥x′i ∥22 + ∥relu(W1 xi + bi )∥22 )
                                  ϵ
                                   2
                                ≥ 2 ∥x′i ∥2 |W2⊤ relu(W1 xi + bi )|
                                  ϵ
                                  2 ′
                                ≥ ∥xi ∥2 |yi |.
                                  ϵ

                                                        25
This concludes the proof.
Proof
  Pn of Theorem2 p5.1. By Lemma A.12, we only need to construct a memorizing solution that has sharpness
2
n    i=1 min(1, ϵ    ∥xi ∥22 + 1|yi |).
    As the input data points form a set of extreme points, for each input data point xi , there exists a vector
∥wi ∥ = 1, wi ∈ Rd , such that ∀j ̸∈ i, wi⊤ xi > wi⊤ xj . We can then construct the minimal sharpness solution
by choosing for sufficiently small δ,

                      W1 = [ui wi /δ]⊤                   ⊤
                                                                 ⊤
                                        i , b1 = [ui −wi xi + δ /δ]i , W2 = [ri yi ]i∈[n] ,

with ri , ui satisfying
                                       p
   1. ri = 1, ui = 2ϵ when                 ∥xi ∥22 + 1|yi | > ϵ.
                                                    √
                        ϵ                             ∥xi ∥2 +1|yi | 1/2         p
   2. ri = ( √                      )1/2 , ui = ϵ(         ϵ        )    when 0 < ∥xi ∥22 + 1|yi | ≤ ϵ.
                   ∥xi ∥2 +1|yi |

   3. ri = 0, ui = 2ϵ when yi = 0.
It is easy to check this is a memorizing solution that minimizes sharpness.5 The proof is then completed.

A.4.2    Proof of Propositions 5.1 and 5.2
                                             √
Proof of Proposition 5.1. We will suppose ϵ < d + 1, then for Pxor , the minimal sharpness is always 2
by Lemma A.12. Consider the following construction for sufficiently small δ,

                                 W1 = [2ϵxi /δ]⊤                        ⊤
                                               i , b1 = [2ϵ (−d + δ) /δ]i , W2 = [yi ]i∈[n] ,

    Then first this is a memorizing solution that minimizes sharpness. Second, the generalization error is
1 − n/2d because for any x ̸∈ {xi }i∈[n] , it holds that relu(W1 x + b1 ) = 0 and hence fθ (x) = 0, The proof is
then completed.
                                                    √
Proof of Proposition 5.2. We will suppose ϵ < d + 1, then for Pxor , the minimal sharpness is always 2
by Lemma A.12. Consider the following construction for sufficiently small δ,

           ∀i, j ∈ [2], W1,2i+j = 2ϵ[(−1)i , (−1)j , ..., 0], b1 [2i + j] = −2ϵ, W2 [2i + j] = (−1)i+j .                              (18)
           ∀k > 4, W1,k = [0, ..., 0], b1 [k] = 0, W2 [k] = 0,

This is an interpolating parameter that minimizes sharpness that can perfectly generalize.

A.5     Discussion on the choice of loss function
In this section, we will show why our theoretical results hold for logistic loss with label smoothing by showing
that using the logistic loss with label smoothing yields the same set of minimizers and flattest minimizers as
a corresponding problem using mean squared error.
Definition A.2 (Logistic Loss with Label Smoothing).
                                                      Logisticloss with label smoothing probability p is
                                             eba                 e(1−b)a
defined as, ℓ : ℓlogistic,p (a, b) = −p log 1+ea − (1 − p) log 1+ea , b ∈ {0, 1}. We will denote the
training loss yield as ℓlogistic,p as Llog .
Theorem A.2. For any probability p ∈ (0, 1), and for any training set {(xi , yi )}i∈[n] satisfying that xi ∈ Rd
and yi ∈ {0, 1}, let γp = ln( 1−p
                               p ), if the minimum of the mean squared error L
                                                                                 mse
                                                                                      over set {xi , γp (2yi − 1)}
is 0, then the minimizers of Lmse over set {xi , γp (2yi − 1)} and the minimizers of Llog over set {(xi , yi )} are
the same.
           q
   5When       ∥xi ∥22 + 1|yi | > ϵ, one can notice that ∇W ′ fθ (xi ) = 0 as the activation in layer 1 is nonzero only in one dimension.
                                                            1




                                                                    26
Proof. This theorem is a direct consequence of the following inequality,
                                  ba                    (1−b)a 
                                   e                      e
     ℓlogistic,p (a, b) = −p log      a
                                         −  (1 − p)  log            ≥ −p log p − (1 − p) log(1 − p).
                                  1+e                     1 + ea

The minimal is reached when a = (2b − 1)γp where γp = ln( 1−p
                                                           p ).

Lemma A.13. For any probability p ∈ (0, 1), and for any training set {(xi , yi )}i∈[n] satisfying that xi ∈
Rd and yi ∈ {0, 1}, let γp = ln( 1−p      p ), for any 
                                                       model fθ that is differentiable and interpolates dataset
                                               2 log         1   1
                                                                   Pn                  2
{xi , γp (2yi − 1)}i∈[n] , it holds that Tr ∇ L (θ) = p(1−p)     n    i=1 ∥∇θ fθ (xi )∥ .

Proof. By standard calculus, it holds that,
                   n
              1X
   Tr ∇2 L(θ) =       Tr ∇2θ [ℓlogistic,p (fθ (xi ), yi )]
                                                          
                n i=1
                   n                                                 
                1X             dℓlogistic,p (fθ (xi ), yi )
              =       Tr ∂θ                                 ∇θ fθ (xi )
                n i=1                   dfθ (xi )
                         n
                      1 X dℓlogistic,p (fθ (xi ), yi )
                                                       Tr ∇2θ fθ (xi )
                                                                      
                  =
                      n i=1       dfθ (xi )
                             n
                          1 X d2 ℓlogistic,p (a, yi )                                                         ⊤
                                                                                                                  
                      +                   2
                                                      |a=fθ (xi ) Tr ∇2θ fθ (xi ) Tr (∇θ fθ (xi )) (∇θ fθ (xi ))
                          n i=1        da
                          n
                    1 X d2 ℓlogistic,p (a, yi )                 
                                                                                                  ⊤
                                                                                                    
                  =                             |a=(2y−1)γ   Tr   (∇ θ fθ (x i )) (∇ θ fθ (x i ))
                    n i=1        da2                       p


                                  n
                      1     1    X
                  =                  ∥∇θ fθ (xi )∥22 .                                                            (19)
                      n p(1 − p) i=1

The proof is then complete.
   By Lemmas 4.1 and A.13, we have that the sharpness yields by both loss functions are the same up to a
constant factor. Therefore, the flattest minimizers of both loss functions are the same.

A.6     Technical Lemmas
A.6.1   Concentration inequalities
Subgaussian random variables are defined as follows.
Definition A.3 (Subgaussian
                        random
                               variable). A random variable X is called σ-subgaussian if E[X] = 0
                              σ 2 λ2
and E [exp (λX)] ≤ exp           2     for all λ ∈ R.

   Subgaussian random vectors are defined as,
                                                                   d
Definition A.4 (Subgaussian
                           random
                             2   2
                                    vector). A random vector x ∈ R is called σ-subgaussian if E[x] = 0
                         σ ∥λ∥2
and E exp λT x ≤ exp           2     for all λ ∈ Rd .

   We will further define subexponential random variables.

     2 2 A.5 (Subexponential random variable). A random variable X is (σ, α)-subexponential if E [exp (λ(X − E(X)))] ≤
Definition
     σ λ
exp     2   for all |λ| ≤ α1 .
                                                                                             √
Lemma A.14 (Honorio & Jaakkola (2014)). If random variable X is σ-subgaussian, then X 2 is (4 2σ 2 , 4σ 2 )-
subexponential.


                                                           27
Lemma A.15 (Hoeffding’s Bound). If {Xi }i∈[n] are σ-subgaussian and independent, then there exists
                            Pn
Cσ > 0, for all t ≥ 0, Pr n1 i=1 Xi ≥ t ≤ 2 exp −nt2 Cσ .
                                                         

Lemma A.16 (Rinaldo (2019)). If {Xi }i∈[n] are (σ, α)-subexponential and independent, 
                                                                                      then there exists
                              Pn
Cα,σ > 0, for all t ≥ 0, Pr n1 i=1 (Xi − E[Xi ]) ≥ t ≤ 2 exp −n min tCα,σ , t2 Cα,σ .
                                                     

Lemma A.17 (Rinaldo (2019)). If x ∈ Rd is a σ-Subgaussian random vector then for any t ≥ 0,

                              Pr ∥x∥22 ≥ 32σ 2 d + 8σ 2 t ≤ exp(−t).
                                                         
                                                                                                      (20)

It also holds that ∥x∥22 has bounded second moment E[∥x∥42 ] ≤ 2048σ 4 d2 .
   We will also need the following lemma bounding the truncated second-order moment of a subgaussian
random variable.
Lemma A.18.pFor any n > 0 and dimension   d, for any d-dimension   σ-subgaussian random vector x, there
exists c = O( d log(dn)σ), such that E ∥x∥2 1 (∥x∥ > c) ≤ nd σ 2 .
                                                      

Proof. We have that by Equation (20),

                          E ∥x∥2 1 (∥x∥ > c)
                                             
                                                Z ∞
                        =c2 Pr ∥x∥2 > c2 +           Pr ∥x∥2 > t2 dt2
                                                                    
                                                  c
                                                    Z ∞
                           2       c2 − 32σ 2 d                     t2 − 32σ 2 d
                        ≤c exp(−          2
                                                ) +        2t exp(−              )dt
                                       8σ             c                  8σ 2
                                   c2 − 32σ 2 d                  t2 − 32σ 2 d c
                        =c2 exp(−         2
                                                ) + 8σ 2 exp(−                ) |∞
                                       8σ                            8σ 2
                                   c2 − 32σ 2 d                  c2 − 32σ 2 d
                        ≤c2 exp(−               ) + 8σ  2
                                                          exp(−               )
                                       8σ 2                          8σ 2
                         p
Hence there exists c = O( d log(dn)σ) such that E ∥x∥2 1 (∥x∥ > c) ≤ nd .
                                                                       

    We will finally show a constant probability lower bound on the norm of a subgaussian random vector with
unit variance.
Lemma A.19. Given any σ > 0, there exists constant ϵ, ζ, for any dimension d, for any σ-subgaussian
random vector x with connvariance Id , it holds that Pr(∥x∥22 > ϵd) > ζ.
Proof. As x is σ-subgaussian, it holds that for any λ ∈ R,

                                      E∥v∥=1 [exp(λv ⊤ x)] ≤ exp(λ2 σ 2 /2).                          (21)

Here the expectation over v in Equation (21) is taken over a uniform distribution over a unit ball and v is
independent of x. Hence v ⊤ x equals in distribution to v[1]∥x∥2 . Hence it holds that,

                                   E∥v∥=1 [exp(v[1]∥x∥2 )] ≤ exp(σ 2 /2).                             (22)
                                  2       3     4     5
   Note that exp(x) ≥ 1 + x + x2 + x6 + x24 + 120
                                              x
                                                  and ∀t ∈ N, E[(v[1])2t+1 ] = 0, it holds that

                   1                            1
                1 + E[∥x∥22 ]E∥v∥=1 [(v[1])2 ] + E[∥x∥42 ]E∥v∥=1 [(v[1])4 ] ≤ exp(σ 2 /2).
                   2                            24
   It is well known that E∥v∥=1 [(v[1])2 ] = d1 and E∥v∥=1 [(v[1])4 ] = (d+2)(d+4)
                                                                            3
                                                                                   . Also it holds that
      2
E[∥x∥2 ] = d. Hence,
                                                            (d + 2)(d + 4)
                              E[∥x∥42 ] ≤ exp(σ 2 /2) − 3/2                 .
                                                                   3


                                                          28
   This implies that
                                                         (d + 2)(d + 4)
                                      exp(σ 2 /2) − 3/2
                                                                  3
                                          4        2   1
                                    ≥E[∥x∥2 I(∥x∥2 > d)]
                                                       2
                                                              2
                                      E[∥x∥22 I(∥x∥22 > 21 d)]
                                    ≥
                                         Pr(∥x∥22 > 21 d)
                                                                          2
                                      E[∥x∥22 ] − E[∥x∥22 I(∥x∥22 ≤ 12 d)]
                                    =
                                                Pr(∥x∥22 > 12 d)
                                               d2
                                    ≥
                                        4 Pr(∥x∥22 > 12 d)
   Hence, we can conclude that
                         1                     3d2                              1
            Pr(∥x∥22 >     d) ≥                                     ≥                        .
                         2      4(d + 2)(d + 4) (exp(σ 2 /2) − 3/2)   20 (exp(σ 2 /2) − 3/2)
This concludes the proof.

A.6.2   Rademacher Complexity
Recall the definition of Rademacher complexity,
Definition A.6 (Rademacher complexity). Let F be a class of functions from X to Y. Let S = {x1 , . . . , xn } ⊂
X be a set of points. P
                      The empirical Rademacher complexity of F with respect to S is defined as RS (F) =
1                       n
  E
n ϵ∼{±1}
           n sup f ∈F   i=1 ϵi f (xi ).

   We will also define the following notion of the shattered set and VC dimension.
Definition A.7 (Shattered set). Let F be a class of functions from X to Y = {0, 1}. A set S = {x1 , . . . , xn } ⊂
X is said to be shattered by F if for every T ⊂ S, there exists f ∈ F such that f (x) = 1 for all x ∈ T and
f (x) = 0 for all x ∈ S \ T .
Definition A.8 (VC dimension). Let F be a class of functions from X to Y = {0, 1}. The VC dimension of
F is defined as VC(F) = sup{n ∈ N | there exists a set of size n shattered by F}.
   We will use the following well-known lemmas.
Lemma A.20 (Massart’s Lemma). Let F beq
                                      a class of functions from X to Y = {0, 1}. Further, suppose
                                                   2 log |A|
A = {(f (xi ))i∈n | f ∈ F }, then, RS (F) ≤            n     .
Lemma A.21 (Sauer’s Lemma). Let F be a class of functions from X to Y = {0, 1}. Further, suppose
                                         PVC(F ) n
A = {(f (xi ))i∈[n] | f ∈ F }, then |A| ≤ i=0    i .

   Combining the above two lemmas, we get the following corollary.
                                                                                         q
                                                                                             4VC(F ) log n
Corollary A.1. Let F be a class of functions from X to Y = {0, 1}, then RS (F) ≤                 n         .
  Further, we also have the following lemma controlling the VC dimension.
Lemma A.22. Suppose F = {x ∈ Rd → 1 w⊤ x > 0 | w ∈ Rd }, then VC(F) = d.
                                                     

   The following uniform convergence bound based on Rademacher complexity is also well known.
Lemma A.23 (Shalev-Shwartz & Ben-David (2014)). Suppose for all f ∈ F, 0 ≤ f (x) ≤ 1, then with
probability at least 1 − δ over the randomness of i.i.d. sampled S = {x1 , . . . , xn } ⊂ X , it holds that
                                                                         s
                                     n
                                  1X                                          log 4δ
                            sup         f (xi ) − E[f (x)] ≤ 2RS (F) + 3             .                      (23)
                            f ∈F n i=1                                           n


                                                         29
   To prove our main results, we will also need the following theorem due to Srebro et al. (2010).

Definition A.9. A loss function ℓ : R × R → R is H−smooth, if and only dℓ(x,y)
                                                                         dx    is H−lipschitz.
Theorem A.3 (Theorem 1 of Srebro et al. (2010)). For an H-smooth non-negative loss ℓ s.t. ∀x,y,f |ℓ(f (x), y)| ≤
b, for any δ > 0 we have that with probability at least 1 − δ over a random sample of size n, for any f ∈ F
with zero training loss L̂(h) = 0,
                                                                          
                                                                b log(1/δ)
                                L(h) ≤ O H log3 nR2n (F) +                   .
                                                                     n

   Finally, we also need lemmas bounding the Rademacher complexity of norm-bounded linear hypothesis
and 2-MLP-No-Bias.
Lemma A.24. For   Pany   constant C > 0 and number of samples n, for the set of parameters for 2-MLP-No-
                     m
Bias ΘC ≜ {θ |       j=1 ∥W 1,j ∥2 |W2,j | ≤ C, θ = (W1 , W2 )} and any training set {xi }i∈[n] satisfying that
∥xi ∥2 ≤ B, it holds that RS ({fθnobias | θ ∈ ΘC }) ≤ 2CB
                                                       √ .
                                                         n

Proof. Let u denotes u/∥u∥2 for u ̸= 0 and 0 when u = 0,
                                         "      n
                                                                     #
                                     1         X
        RS ({fθnobias | θ ∈ ΘC }) = E sup          σi fθnobias (xi )
                                     n σ θ i=1
                                                                                   
                                                n         m
                                     1        X         X
                                  = E sup           σi       W2,j relu (W1,j xi )
                                     n σ θ i=1           j=1
                                                                                              
                                                n         m
                                     1        X         X                                 T
                                                                                              
                                  = E sup           σi       W2,j ∥W1,j ∥2 relu W1,j xi 
                                     n σ θ i=1           j=1
                                                                    " n                        #
                                                m
                                     1        X                       X                  T
                                                                                              
                                  = E sup           W2,j ∥W1,j ∥2            σi relu W1,j xi 
                                     n σ θ j=1                          i=1
                                                                                                    
                                                m                                n
                                     1        X                                X               T
                                                                                                   
                                  ≤ E sup           |W2,j | ∥W1,j ∥2 max            σi relu W1,k xi 
                                     n σ θ j=1                         k∈[n]
                                                                                i=1
                                           "              n
                                                                                 #
                                       C                X
                                                                         T
                                                                              
                                  ≤= E         sup           σi relu ū xi
                                       n σ ū:∥ū∥2 =1 i=1
                                         "             n
                                                                              #
                                     C                X
                                                           σi relu ūT xi
                                                                           
                                  ≤ E        sup
                                     n σ ū:∥ū∥2 ≤1 i=1
                                           "            n
                                                                             #
                                     2C               X
                                                                       T
                                                                           
                                  ≤     E     sup           σi relu ū xi
                                      n σ ū:∥ū∥2 ≤1 i=1
                                    = 2CRS (H′ ) ,

   where H′ = x 7→ relu ū⊤ x : ū
                              
                                   ∈ Rd , ∥ū∥2 ≤ 1 . By Talagrand’s lemma, since relu is 1-Lipschitz,
RS (H′ ) ≤ RS (H′′ ) where H′′ = x 7→ ū⊤ x : ū ∈ Rd , ∥ū∥2 ≤ 1 is a linear hypothesis space. Using
                                 

RS (H′′ ) ≤ √Bn by Lemma A.25 concludes the proof.

Lemma A.25. For any constant C > 0 and number      of samples n, for any set S = {xi }i∈[n] satisfying that
∀i, xi ∈ Rd , ∥xi ∥22 ≤ C 2 and function class F = x 7→ ⟨w, x⟩ | w ∈ Rd , ∥w∥2 ≤ 1 , it holds that,

                                                        C
                                               RS (F) ≤ √ .
                                                         n


                                                      30
Proof.
                                               "           n
                                                                          #
                                                      1X
                                   RS (F) = E sup             σi ⟨w, xi ⟩
                                            σ ∥w∥ ≤1 n
                                                 2       i=1
                                               "        *       n
                                                                         +#
                                            1                  X
                                          = E sup          w,      σi xi
                                            n σ ∥w∥2 ≤1        i=1
                                               " n             #
                                            1    X
                                          = E         σi xi
                                            n σ i=1
                                                             2
                                              v
                                              u n
                                            1 uX         2      C
                                          = t      ∥xi ∥2 ≤ √ .
                                            n i=1                n




A.6.3    Elementary inequalities
We will prove some elementary inequalities that will be useful in the proof of our main results.

Lemma A.26. For any x, y ∈ R, |relu(x)2 − relu(y)2 | ≤ (|x| + |y|)|x − y|.
Proof. We will assume WLOG that x > y. We then discuss the following three cases.
   1. 0 ≥ x > y, then the result is trivial.
   2. x > 0 ≥ y, then |relu(x)2 − relu(y)2 | = x2 ≤ (|x| + |y|)|x − y|.
   3. x > y > 0, then |relu(x)2 − relu(y)2 | = x2 − y 2 = (|x| + |y|)|x − y|.

This completes the proof.
Lemma A.27. For any a, b, c, d ∈ R, if a + d = b + c, then
                                                            2             2           2            2
         |relu(a) + relu(d) − relu(b) − relu(c)|2 ≤ (relu(a)) + (relu(d)) + (relu(b)) + (relu(c)) .

The equality holds if and only if three of the four values are not positive.
Proof. WLOG we assume a ≥ b ≥ c ≥ d. As ReLU is convex, we have that relu(a) + relu(d) − relu(b) −
relu(c) ≥ 0. Further, we have that relu(a) + relu(d) − relu(b) − relu(c) ≤ relu(a) − relu(b) ≤ relu(a). Thus,
we have the desired result.




                                                     31
                      Learning Rate     Perturbation Radius     Batch size    Weight Decay      Epochs
        Figure 1a     0.01              0                       100           0.05              1e5
        Figure 1b     0.01              0.05                    1             0                 1e5
        Figure 2a     0.005             0.1                     1             0                 1e5
                      0.003             1                       1             0                 1e5
        Figure 4a     0.01              0                       100           0.05              1e5
        Figure 4b     0.01              0.05                    1             0                 2e5
                      0.01              0.1                     1             0                 4e5
        Figure 5a     0.001             0                       1             0.05              1e5
        Figure 5b     0.0005            0.05                    1             0                 1e5
                      0.001             0.1                     1             0                 1e5
                      0.005             1                       1             0                 5e3
        Figure 6a     0.1               0.1                     1             0                 1e5
        Figure 6b     0.01              0.1                     1             0                 5e2
                      0.01              0.5                     1             0                 5e2
                      0.01              1                       1             0                 5e2
        Figure 7a     0.01              0.2                     1             0                 1e5
        Figure 7b     0.01              0.2                     1             0                 1e5
        Figure 8a     0.1               0                       10            0.01              1e5
        Figure 8b     0.1               0.2                     1             0                 1e5
        Figure 9a     0.01              0                       1             0.05              1e5
        Figure 9b     0.1               0.2                     1             0                 1e5
        Figure 10a    0.1               0.2                     1             0                 4e4
        Figure 10b    1                 0.5                     1             0                 1e3
                      1                 1                       1             0                 1e5
        Figure 11a    0.01              0                       1             0.05              1e5
        Figure 11b    0.01              0.05                    1             0                 1e5

Table 2: Training details for Experiments. For Figures 6a and 10b, we scale down the initialization of the
first layer by a factor of 100 to avoid minimizing the sharpness by simply increasing the norm at the beginning.



B     Experiments
B.1    Training Details
For all the experiments, we use networks with width 500. The learning rates, perturbation radius, and training
epochs are summarized in Table 2. For those experiments where there are adjustments in hyperparameters
through the training process, we report all the hyperparameters in multiple rows. We use 8 NVIDIA 2080
GPUs to train the models. The training time for each experiment is around 12 hours per 1e5 epochs

B.2    Extension To Uniform Ball Distribution
As our Theorems 4.1 and A.1 suggests, the generalization and memorization results should hold for data
distribution other than boolean hypercube. We perform experiments on uniform √     ball distribution to verify
this. Specifically, we sample data points uniformly from the ball with radius d with dimension d = 10 and
the label is defined as y = |x[1]| − |x[2]|. The results are shown in Figure 7. We can see that the flattest
minimizers of the two architectures have very different generalization behavior. The flattest minimizer of the
MLP without bias has a much better generalization performance than the one with bias. This is consistent
with our theoretical results.




                                                      32
                     (a) 2-MLP-No-Bias                                   (b) 2-MLP-Bias

Figure 7: Uniform Ball Distribution. We train a 2-layer MLP with ReLU activation with and without Bias
using 1-SAM on uniform ball distribution with dimension d = 10 and training set size n = 100. One can
again see the striking difference between the generalization behavior of the flattest minimizers of the two
architectures.




                        (a) Baseline                                       (b) 1-SAM

Figure 8: Scenario I with Logistic Loss. We train a 2-layer MLP with ReLU activation without Bias using
gradient descent with weight decay and 1-SAM on Pxor with dimension d = 30 and training set size n = 100.
In both cases, the model reaches perfect generalization. Notice that although weight decay doesn’t explicitly
regularize model sharpness, the flatness of the model decreases through training, which is consistent with
our Lemma 3.1 relating sharpness to the norm of the weight.
B.3    Extension To Logistic Loss
As Theorem A.2 and lemma A.13 suggests, our results can be extended to logistic loss with label smoothing.
We perform all our experiments mentioned in the main text on the same distribution Pxor , with the mean
squared error loss replaced by logistic loss with label smoothing p = 0.2 to verify this. The results are shown
in Figures 8 to 10.

B.4    Extension To Deeper Networks
Our Theorem 4.1 suggests that memorization solutions can exist for deeper networks with biased terms in the
first layer. We perform experiments on deeper networks to verify this. Specifically, we train a 3-layer MLP
with ReLU activation with bias term in the first layer on Pxor with dimension d = 30 and training set size
n = 100. The results are shown in Figure 11. We can see that the flattest minimizer of the 3-layer MLP with
bias term in the first layer has a much worse generalization performance than the baseline. This is consistent
with our theoretical results.




                                                      33
                        (a) Baseline                                     (b) 1-SAM

Figure 9: Scenario II with Logistic Loss. We train a 2-layer MLP with ReLU activation with Bias using
gradient descent with weight decay and 1-SAM on Pxor with dimension d = 30 and training set size
n = 100. One can observe a distinction between the two settings. The minimum reached by 1-SAM is
flatter but the model generalizes much worse and even starts to degenerate after 2000 epochs. The difference
between Figures 8b and 9b is similar to the difference between Figures 1b and 4b




                 (a) Simplified BatchNorm                         (b) Simplified LayerNorm

Figure 10: Models with Normalization and Logistic Loss. We train two-layer ReLU networks with simplified
BatchNorm and LayerNorm on data distribution Pxor with dimension d = 30 and sample complexity n = 100
using 1-SAM. We can see that in both cases, the models nearly perfectly generalize.




                        (a) Baseline                                     (b) 1-SAM

Figure 11: Scenario II with Deeper Networks. We train a 3-layer MLP with ReLU activation with Bias
using gradient descent with weight decay and 1-SAM on Pxor with dimension d = 30 and training set size
n = 100. One can observe a distinction between the two settings. The minimum reached by 1-SAM is flatter,
but the model generalizes much worse.


                                                    34
