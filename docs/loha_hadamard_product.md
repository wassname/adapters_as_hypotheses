Title: 2108.06098v3.pdf

URL Source: https://arxiv.org/pdf/2108.06098

Published Time: Mon, 23 Jan 2023 11:03:46 GMT

Number of Pages: 23

Markdown Content:
Published as a conference paper at ICLR 2022 

# FED PARA : LOW -RANK HADAMARD PRODUCT FOR 

# COMMUNICATION -E FFICIENT FEDERATED LEARNING 

Nam Hyeon-Woo 1, Moon Ye-Bin 1, Tae-Hyun Oh 1,2,31Department of Electrical Engineering, POSTECH 2Graduate School of AI, POSTECH 

> 3

Yonsei University 

{hyeonw.nam, ybmoon, taehyun }@postech.ac.kr 

## ABSTRACT 

In this work, we propose a communication-efficient parameterization, FedPara ,for federated learning (FL) to overcome the burdens on frequent model uploads and downloads. Our method re-parameterizes weight parameters of layers using low-rank weights followed by the Hadamard product. Compared to the conven-tional low-rank parameterization, our FedPara method is not restricted to low-rank constraints, and thereby it has a far larger capacity. This property enables to achieve comparable performance while requiring 3 to 10 times lower communi-cation costs than the model with the original layers, which is not achievable by the traditional low-rank methods. The efficiency of our method can be further im-proved by combining with other efficient FL optimizers. In addition, we extend our method to a personalized FL application, pFedPara , which separates param-eters into global and local ones. We show that pFedPara outperforms competing personalized FL methods with more than three times fewer parameters. Project page: https://github.com/South-hw/FedPara_ICLR22 

## 1 INTRODUCTION 

Federated learning (FL; McMahan et al., 2017) has been proposed as an efficient collaborative learn-ing strategy along with the advance and spread of mobile and IoT devices. FL allows leveraging local computing resources of edge devices and locally stored private data without data sharing for privacy. FL typically consists of the following steps: (1) clients download a globally shared model from a central server, (2) the clients locally update each model using their own private data without accessing the others’ data, (3) the clients upload their local models back to the server, and (4) the server consolidates the updated models and repeats these steps until the global model converges. FL has the key properties (McMahan et al., 2017) that differentiate it from distributed learning: • Heterogeneous data. Data is decentralized and non-IID as well as unbalanced in its amount due to different characteristics of clients; thus, local data does not represent the population distribution. • Heterogeneous systems. Clients consist of heterogeneous setups of hardware and infrastructure; hence, those connections are not guaranteed to be online, fast, or cheap. Besides, massive client participation is expected through different communication paths, causing communication burdens. These FL properties introduce challenges in the convergence stability with heterogeneous data and communication overheads. To improve the stability and reduce the communication rounds, the prior works in FL have proposed modified loss functions or model aggregation methods (Li et al., 2020; Karimireddy et al., 2020; Acar et al., 2021; Yu et al., 2020; Reddi et al., 2021). However, the transferred data is still a lot for the edge devices with bandwidth constraints or countries having low-quality communication infrastructure. 1 A large amount of transferred data introduces an energy consumption issue on edge devices because wireless communication is significantly more power-intensive than computation (Yadav & Yadav, 2016; Yan et al., 2019). 

> 1

The gap between the fastest and the lowest communication speed across countries is significant; approximately 63 times different (Speedtest). 

1

> arXiv:2108.06098v3 [cs.LG] 19 Jan 2023

Published as a conference paper at ICLR 2022 In this work, we propose a communication-efficient re-parameterization for FL, FedPara , which reduces the number of bits transferred per round. FedPara directly re-parameterizes each fully-connected (FC) and convolutional layers of the model to have a small and factorized form while preserving the model’s capacity. Our key idea is to combine the Hadamard product with low-rank parameterization as W = ( X1Y> 

> 1

) (X2Y> 

> 2

) ∈ Rm×n, called low-rank Hadamard product . When 

rank( X1Y> 

> 1

) = rank( X2Y> 

> 2

) = r, then rank( W) ≤ r2. This outstanding property facilitates spanning a full-rank matrix with much fewer parameters than the typical m × n matrix. It signifi-cantly reduces the communication burdens during training. At the inference phase, we pre-compose and maintain W that boils down to its original structure; thus, FedPara does not alter computa-tional complexity at inference time. Compared to the aforementioned prior works that tackle reduc-ing the required communication rounds for convergence, our FedPara is an orthogonal approach in that FedPara does not change the optimization part but re-defines each layer’s internal structure. We demonstrate the effectiveness of FedPara with various network architectures, including VGG, ResNet, and LSTM, on standard classification benchmark datasets for both IID and non-IID settings. The accuracy of our parameterization outperforms that of the traditional low-rank parameterization baseline given the same number of parameters. Besides, FedPara has comparable accuracy to original counterpart models and even outperforms as the number of parameters increases at times. We also combine FedPara with other FL algorithms to improve communication efficiency further. We extend FedPara to the personalized FL application, named pFedPara , which separates the roles of each sub-matrix into global and local inner matrices. The global and local inner matrices learn the globally shared common knowledge and client-specific knowledge, respectively. We de-vise three scenarios according to the amount and heterogeneity of local data using the subset of FEMNIST and MNIST. We demonstrate performance improvement and robustness of pFedPara 

against competing algorithms. We summarize our main contributions as follows: • We propose FedPara , a low-rank Hadamard product parameterization for communication-efficient FL. Unlike traditional low-rank parameterization, we show that FedPara can span a full-rank matrix and tensor with reduced parameters. We also show that FedPara requires up to ten times fewer total communication costs than the original model to achieve target accuracy. 

FedPara even outperforms the original model by adjusting ranks at times. • Our FedPara takes a novel approach; thereby, our FedPara can be combined with other FL methods to get mutual benefits, which further increase accuracy and communication efficiency. • We propose pFedPara , a personalization application of FedPara , which splits the layer weights into global and local parameters. pFedPara shows more robust results in challenging regimes than competing methods. 

## 2 METHOD 

In this section, we first provide the overview of three popular low-rank parameterizations in Sec-tion 2.1 and present our parameterization, FedPara , with its algorithmic properties in Section 2.2. Then, we extend FedPara to the personalized FL application, pFedPara , in Section 2.3. 

Notations. We denote the Hadamard product as , the Kronecker product as ⊗, n-mode tensor prod-uct as ×n, and the i-th unfolding of the tensor T (i) ∈ Rki×∏   

> j6=ikj

given a tensor T ∈ Rk1×···× kn .2.1 OVERVIEW OF LOW -RANK PARAMETERIZATION 

The low-rank decomposition in neural networks has been typically applied to pre-trained models for compression (Phan et al., 2020), whereby the number of parameters is reduced while minimizing the loss of encoded information. Given a learned parameter matrix W ∈ Rm×n, it is formulated as finding the best rank-r approximation, as arg min ˜W || W − ˜W|| F such that ˜W = XY >, where 

X ∈ Rm×r , Y ∈ Rn×r , and r min ( m, n ). It reduces the number of parameters from O(mn ) to 

O(( m + n)r), and its closed-form optimal solution can be found by SVD. This matrix decomposition is applicable to the FC layers and the reshaped kernels of the convolution layers. However, the natural shape of a convolution kernel is a fourth-order tensor; thus, the low-rank tensor decomposition, such as Tucker and CP decomposition (Lebedev et al., 2015; Phan et al., 2Published as a conference paper at ICLR 2022 X

> (𝑚 ×2𝑅 )

Y!

> (2𝑅 ×𝑛 )

W    

> (𝑚 ×𝑛 )=
> rank 𝐖 ≤2𝑅

(a) Conventional low-rank parameterization ⨀= X!

> (𝑚 ×𝑅 )

Y!

> "
> (𝑅 ×𝑛 )

X#

> (𝑚 ×𝑅 )

Y#

> "
> (𝑅 ×𝑛 )

W!

> (𝑚 ×𝑛 )

W#

> (𝑚 ×𝑛 )

W    

> (𝑚 ×𝑛 )
> rank 𝐖 ≤𝑅 !

(b) FedPara (Ours) 

Figure 1: Illustrations of low-rank matrix parameterization and FedPara with the same number of parameters 2R(m + n). (a) Low-rank parameterization is the summation of 2R number of rank-

1 matrices, W = XY >, and rank( W) ≤ 2R. (b) FedPara is the Hadamard product of two low-rank inner matrices, W = W1 W2 = ( X1Y> 

> 1

) (X2Y> 

> 2

), and rank( W) ≤ R2.Layer Parameterization # Params. Maximal Rank Example [ # Params. / Rank] FC Layer Original mn min( m, n ) 66 K / 256 Low-rank 2R(m + n) 2R 16 K / 32                              

> FedPara 2R(m+n)R216 K / 256 Convolutional Layer Original OIK 1K2min( O, IK 1K2)590 K / 256 Low-rank 2R(O+I+RK 1K2)2R21 K / 32
> FedPara (Proposition 1) 2R(O+IK 1K2)R282 K / 256
> FedPara (Proposition 3) 2R(O+I+RK 1K2)R221 K / 256

Table 1: The number of parameters, maximal rank, and example. We assume that the weights of the FC and convolutional layers are in Rm×n and RO×I×K1×K2 , respectively. The rank of the convolutional layer is the rank of the 1st unfolding tensor. As a reference example, we set 

m = n = O = I = 256 , K 1 = K2 = 3 , and R = 16 .2020), can be a more suitable approach. Given a learned high-order tensor T ∈ Rk1×···× kn , Tucker decomposition multiplies a kernel tensor K ∈ Rr1×···× rn with matrices Xi ∈ Rri×n1 , where ri =rank( ˜T (i)) as ˜T = K × 1 X1 ×2 · · · × n Xn, and CP decomposition is the summation of rank-1 tensors as ˜T = ∑i=ri=1 x1 

> i

× x2 

> i

× · · · × xni , where xji ∈ Rkj . Likewise, it also reduces the number of model parameters. We refer to these rank-constrained structure methods simply as conventional low-rank constraints or low-rank parameterization methods. In the FL context, where the parameters are frequently transferred between clients and the server during training, the reduced parameters lead to communication cost reduction, which is the main focus of this work. The post-decomposition approaches (Lebedev et al., 2015; Phan et al., 2020) using SVD, Tucker, and CP decompositions do not reduce the communication costs because those are applied to the original parameterization after finishing training. That is, the original large-size parameters are transferred during training in FL, and the number of parameters is reduced after finishing training. We take a different notion from the low-rank parameterizations. In the FL scenario, we train a model from scratch with low-rank constraints, but specifically with low-rank Hadamard product re-parameterization . We re-parameterize each learnable layer, including FC and convolutional layers, and train the surrogate model by FL. Different from the existing low-rank method in FL (Koneˇ cn` yet al., 2016), our parameterization can achieve comparable accuracy to the original counterpart. 2.2 FED PARA : A C OMMUNICATION -E FFICIENT PARAMETERIZATION 

As mentioned, the conventional low-rank parameterization has limited expressiveness due to its low-rank constraint. To overcome this while maintaining fewer parameters, we present our new low-rank Hadamard product parameterization, called FedPara , which has the favorable property as follows: 

Proposition 1 Let X1 ∈ Rm×r1 , X2 ∈ Rm×r2 , Y1 ∈ Rn×r1 , Y2 ∈ Rn×r2 , r1, r 2 ≤ min( m, n )

and the constructed matrix be W := ( X1Y> 

> 1

) (X2Y> 

> 2

). Then, rank( W) ≤ r1r2.

All proofs can be found in the supplementary material including Proposition 1. Proposition 1 im-plies that, unlike the low-rank parameterization, a higher-rank matrix can be constructed using the Hadamard product of two inner low-rank matrices, W1 and W2 (Refer to Figure 1). If we choose the 3Published as a conference paper at ICLR 2022 inner ranks r1 and r2 such that r1r2 ≥ min( m, n ), the constructed matrix does not have a low-rank restriction and is able to span a full-rank matrix with a high chance (See Figure 6); i.e ., FedPara 

has the minimal parameter property achievable to full-rank. In addition, we can control the number of parameters by changing the inner ranks r1 and r2, respectively, but we have the following useful property to set the hyper-parameters to be a minimal number of parameters with a maximal rank. 

Proposition 2 Given R ∈ N, r1 = r2 = R is the unique optimal choice of the following criteria, 

arg min r1,r 2∈N (r1 + r2)( m + n) s.t. r1r2 ≥ R2, (1) 

and its optimal value is 2R(m + n).

Equation 3 implies the criteria that minimize the number of weight parameters used in our parame-terization with the target rank constraint of the constructed matrix as R2. Proposition 2 provides an efficient way to set the hyper-parameters. It implies that, if we set r1=r2=R and R2 ≥ min( m, n ),

FedPara is highly likely to have no low-rank restriction 2 even with much fewer parameters than that of a na¨ ıve weight, i.e ., 2R(m + n)  mn . Moreover, given the same number of parameters, 

rank( W) of FedPara is higher than that of the na¨ ıve low-rank parameterization by a square factor, as shown in Figure 1 and Table 1. To extend Proposition 1 to the convolutional layers, we can simply reshape the fourth-order tensor kernel to the matrix as RO×I×K1×K2 → RO×(IK 1K2) as a na¨ ıve way, where O, I, K 1, and K2 are the output channels, the input channels, and the kernel sizes, respectively. That is, our parameteriza-tion spans convolution filters with a few basis filters of size I × K1 × K2. However, we can derive more efficient parameterization of convolutional layers without reshaping as follows: 

Proposition 3 Let T1, T2 ∈ RR×R×k3×k4 , X1, X2 ∈ Rk1×R, Y1, Y2 ∈ Rk2×R, R ≤ min( k1, k 2)

and the convolution kernel be W := ( T1 ×1 X1 ×2 Y1) (T2 ×1 X2 ×2 Y2). Then, the rank of the kernel satisfies rank( W(1) ) = rank( W(2) ) ≤ R2.

Proposition 3 is the extension of Proposition 1 but can be applied to the convolutional layer without reshaping. In the convolutional layer case of Table 1, given the specific tensor size, Proposition 3 requires 3.8 times fewer parameters than Proposition 1. Hence, we use Proposition 3 for the convo-lutional layer since the tensor method is more effective for common convolutional models. Optionally, we employ non-linearity and the Jacobian correction regularization, of which details can be found in the supplementary material. These techniques improve the accuracy and convergence stability but not essential. Depending on the resources of devices, these techniques can be omitted. 2.3 PFED PARA : P ERSONALIZED FL A PPLICATION Local Global Server 

Client  

> Layer N
> ⋮
> Layer 1
> (a) FedPer

Server 

Client    

> Layer N
> ⋮
> Layer 1 (b) pFedPara (Ours)

Figure 2: Diagrams of (a) FedPer and (b) 

pFedPara . The global part is transferred to the server and shared across clients, while the local part remains private in each client. In practice, data are heterogeneous and personal due to different characteristics of each client, such as usage times and habits. FedPer (Arivazhagan et al., 2019) has been proposed to tackle this sce-nario by distinguishing global and local layers in the model. Clients only transfer global layers (the top layer) and keep local ones (the bottom layers) on each device. The global layers learn jointly to extract general features, while the local layers are biased towards each user. With our FedPara , we propose a personalization application, pFedPara , in which the Hadamard product is used as a bridge between the global inner weight W1 and the local inner weight W2. Each layer of the personalized model is constructed by 

W = W1 (W2 + 1), where W1 is transferred    

> 2Its corollary and empirical evidence can be found in the supplementary material. Under Proposition 2, R2≥
> min( m, n )is a necessary and sufficient condition for achieving a maximal rank.

4Published as a conference paper at ICLR 2022 to the server while W2 is kept in a local device during training. This induces W1 to learn glob-ally shared knowledge implicitly and acts as a switch of the term (W2 + 1). Conceptually, we can interpret by rewriting W = W1 W2+W1 = Wper. +Wglo. , where Wper. = W1 W2

and Wglo. = W1. The construction of the final personalized parameter W in pFedPara can be viewed as an additive model of the global weight Wglo. and the personalizing residue Wper. .

pFedPara transfers only a half of the parameters compared to FedPara under the same rank condition; hence, the communication efficiency is increased further. Intuitively, FedPer and pFedPara are distinguished by their respective split directions, as illus-trated in Figure 2. We summarize our algorithms in the supplementary material. Although we only illustrate feed-forward network cases for convenience, it can be extended to general cases. 

## 3 EXPERIMENTS 

We evaluate our FedPara in terms of communication costs, the number of parameters, and com-patibility with other FL methods. We also evaluate pFedPara in three different non-IID scenarios. We use the standard FL algorithm, FedAvg , as a backbone optimizer in all experiments except for the compatibility experiments. More details and additional experiments can be found in the supplementary material. 3.1 SETUP 

Datasets. In FedPara experiments, we use four popular FL datasets: CIFAR-10, CIFAR-100 (Krizhevsky et al., 2009), CINIC-10 (Darlow et al., 2018), and the subset of Shakespeare (Shake-speare, 1994). We split the datasets randomly into 100 partitions for the CIFAR-10 and CINIC-10 IID settings and 50 partitions for the CIFAR-100 IID setting. For the non-IID settings, we use the Dirichlet distribution for random partitioning and set the Dirichlet parameter α as 0.5 as suggested by He et al. (2020b). We assign one partition to each client and sample 16% of clients at each round during FL. In pFedPara experiments, we use the subset of handwritten datasets: MNIST (LeCun et al., 1998) and FEMNIST (Caldas et al., 2018). For the non-IID setting with MNIST, we follow McMahan et al. (2017), where each of 100 clients has at most two classes. 

Models. We experiment VGG and ResNet for the CNN architectures and LSTM for the RNN architecture as well as two FC layers for the multilayer perceptron. When using VGG, we use the VGG16 architecture (Simonyan & Zisserman, 2015) and replace the batch normalization with the group normalization as suggested by Hsieh et al. (2020). VGG16 ori . stands for the original 

VGG16 , VGG16 low the one with the low-rank tensor parameterization in a Tucker form by following TKD (Phan et al., 2020), and VGG16 FedPara the one with our FedPara . In the pFedPara tests, we use two FC layers as suggested by McMahan et al. (2017). 

Rank Hyper-parameter. We adjust the inner rank of W1 and W2 as r = (1 −γ)rmin + γr max ,where rmin is the minimum rank allowing FedPara to achieve a full-rank by Proposition 2, rmax 

is the maximum rank such that the number of FedPara ’s parameters do not exceed the number of original parameters, and γ ∈ [0 , 1] . We fix the same γ for all layers for simplicity. 3 Note that γ

determines the number of parameters. 3.2 QUANTITATIVE RESULTS 

Capacity. In Table 2, we validate the propositions stating that our FedPara achieves a higher rank than the low-rank parameterization given the same number of parameters. We train VGG16 low and 

VGG16 FedPara for the same target rounds T , and use 10 .25% and 10 .15% of the VGG16 ori . param-eters, respectively, to be comparable. As shown in Table 2a, VGG16 FedPara surpasses VGG16 low 

on all the IID and non-IID settings benchmarks with noticeable margins. The same tendency is observed in the recurrent neural networks as shown in Table 2b. We train LSTM on the subset of Shakespeare. Table 2b shows that LSTM FedPara has higher accuracy than LSTM low , where the num-ber of parameters is set to 19% and 16% of LSTM , respectively. This experiment evidences that our 

FedPara has a better model expressiveness and accuracy than the low-rank parameterization.   

> 3The parameter γcan be independently tuned for each layer in the model. Moreover, one may apply neural architecture search or hyper-parameter search algorithms for further improvement. We leave it for future work.

5Published as a conference paper at ICLR 2022 (a) CNN Models CIFAR-10 ( T = 200 ) CIFAR-100 ( T = 400 ) CINIC-10 ( T = 300 )IID non-IID IID non-IID IID non-IID 

VGG16 low 77.62 67.75 34.16 30.30 63.98 60.80 

VGG16 FedPara (Ours) 82.88 71.35 45.78 43.94 70.35 64.95 

(b) RNN Model Shakespeare ( T = 500 )IID non-IID 

LSTM low 54.59 51.24 

LSTM FedPara (Ours) 63.65 51.56 

Table 2: Accuracy comparison between low-rank parameterization and FedPara . (a) The accuracy 

VGG16 low and VGG16 FedPara . We set the target rounds T = 200 for CIFAR-10, 400 for CIFAR-100, and 300 for CINIC-10. (b) The accuracy of LSTM low and LSTM FedPara . We set the target rounds T = 500 for the Shakespeare dataset. 0 55 110 165 220 Communication Cost [GB] 10.0 20.2 30.5 40.8 51.0 Acc. [%]       

> VGG16 ori. VGG16 FedPara (Ours) 086 172 259 345 Communication Cost [GB] 10.0 28.5 47.0 65.5 84.0 Acc. [%]

(a) CIFAR-10 IID ( γ=0 .1)0 55 110 165 220 Communication Cost [GB] 10.0 20.2 30.5 40.8 51.0 Acc. [%] (b) CIFAR-100 IID ( γ=0 .4)0 136 272 409 545 Communication Cost [GB] 10.0 25.5 41.0 56.5 72.0 Acc. [%] (c) CINIC-10 IID ( γ=0 .3)0 90 180 270 350 Communication Cost [GB] 10.0 26.0 42.0 58.0 74.0 Acc. [%] 

(d) CIFAR-10 non-IID ( γ=0 .1)0 90 180 270 360 Communication Cost [GB] 10.0 19.2 28.5 37.8 47.0 Acc. [%] (e) CIFAR-100 non-IID ( γ=0 .4)0 128 256 384 512 Communication Cost [GB] 10.0 24.2 38.5 52.8 67.0 Acc. [%] (f) CINIC-10 non-IID ( γ=0 .3)CIFAR10 IID (80%) CIFAR10 non-IID (70%) CIFAR100 IID (50%) CIFAR100 non-IID (45%) CINIC10 IID (70%) CINIC10 non-IID (65%) Communication Cost [GB] 205.3/5.1 283.5/7.0 154.9/3.8 235.1/5.8 305.3/7.5 378.0/9.3 20.3/0.5 32.5/0.8 55.2/1.4 67.3/1.7 46.3/1.1 53.9/1.3   

> VGG16 ori. VGG16 FedPara (Ours) Energy Consumption [MJ]

(g) Communication Costs 

Figure 3: (a-f): Accuracy [%] ( y-axis) vs. communication costs [GBytes] ( x-axis) of VGG16 ori . and 

VGG16 FedPara . Broken line and solid line represent VGG16 ori . and VGG16 FedPara , respectively. (g): Size comparison of transferred parameters, which can be expressed as communication costs [GBytes] (left y-axis) or energy consumption [MJ] (right y-axis), for the same target accuracy. The white bars are the results of VGG16 ori . and the black bars are the results of VGG16 FedPara . The target accuracy is denoted in the parentheses under the x-axis of (g). 

Communication Cost. We compare VGG16 FedPara and VGG16 ori . in terms of accuracy and communication costs. FL evaluation typically measures the required rounds to achieve the tar-get accuracy as communication costs, but we instead assess total transferred bit sizes, 2 ×

(#participants) ×(model size) ×(#rounds) , which considers up-/down-link and is a more prac-tical communication cost metric. Depending on the difficulty of the datasets, we set the model size of VGG16 FedPara as 10 .1% , 29 .4% , and 21 .8% of VGG16 ori . for CIFAR-10, CIFAR-100, and CINIC-10, respectively. 6Published as a conference paper at ICLR 2022 0 55 110 165 220 Communication Cost [GB] 10.0 20.2 30.5 40.8 51.0 Acc. [%]       

> VGG16 ori. VGG16 FedPara (Ours) 10 29 48 67 86 Parameters [%] 82.31 82.76 83.20 83.65 84.10 Acc. [%]

(a) CIFAR-10 IID 10 29 48 67 86 Parameters [%] 45.78 47.21 48.64 50.07 51.50 Acc. [%] (b) CIFAR-100 IID 10 29 48 67 86 Parameters [%] 70.35 71.19 72.03 72.86 73.70 Acc. [%] (c) CINIC-10 IID 10 29 48 67 86 Parameters [%] 71.35 72.47 73.60 74.72 75.85 Acc. [%] 

(d) CIFAR-10 non-IID 10 29 48 67 86 Parameters [%] 43.94 45.08 46.22 47.36 48.50 Acc. [%] (e) CIFAR-100 non-IID 10 29 48 67 86 Parameters [%] 64.95 65.71 66.47 67.24 68.00 Acc. [%] (f) CINIC-10 non-IID 

Figure 4: Test accuracy [%] ( y-axis) vs. parameters ratio [%] ( x-axis) of VGG16 FedPara at the target rounds. The target rounds follow Table 2. The dotted line represents VGG16 ori . with no parameter reduction, and the solid line VGG16 FedPara adjusted by γ ∈ [0 .1, 0.9] in 0.1 increments. 

FedAvg FedProx SCAFFOLD FedDyn FedAdam 

Accuracy ( T = 200 ) 82.88 78.95 84.72 86.05 82.48 Round ( 80% ) 110 - 92 80 117 Table 3: The compatibility of FedPara with other FL algorithms. The first row is the accuracy of 

FedPara combined with other FL algorithms on the CIFAR-10 IID setting after 200 rounds, and the second row is the required rounds to achieve the target accuracy 80%. In Figures 3a-3f, VGG16 FedPara has comparable accuracy but requires much lower communica-tion costs than VGG16 ori .. Figure 3g shows communication costs and energy consumption required for model training to achieve the target accuracy; we compute the energy consumption by the en-ergy model of user-to-data center topology (Yan et al., 2019). VGG16 FedPara needs 2.8 to 10.1 times fewer communication costs and energy consumption than VGG16 ori . to achieve the same tar-get accuracy. Because of these properties, FedPara is suitable for edge devices suffering from communication and battery consumption constraints. 

Model Parameter Ratio. We analyze how the number of parameters controlled by the rank ratio 

γ affects the accuracy of FedPara . As revealed in Figure 4, VGG16 FedPara ’s accuracy mostly increases as the number of parameters increases. VGG16 FedPara can achieve even higher accuracy than VGG16 ori .. It is consistent with the reports from the prior works (Luo et al., 2017; Kim et al., 2019) on model compression, where reduced parameters often lead to accuracy improvement, i.e ., regularization effects. 

Compatibility. We integrate the FedPara -based model with other FL optimizers to show that our FedPara is compatible with them. We measure the accuracy during the target rounds and the required rounds to achieve the target accuracy. Table 3 shows that VGG16 FedPara combined with the current state-of-the-art method, FedDyn , is the best among other combinations. Thereby, we can further save the communication costs by combining FedPara with other efficient FL approaches. 

Personalization. We evaluate pFedPara , assuming no sub-sampling of clients for an update. We train two FC layers on the FEMNIST or MNIST datasets using four algorithms, FedPAQ , FedAvg ,

FedPer , and pFedPara , with ten clients. FedPAQ denotes the local models trained only using their own local data; FedAvg the global model trained by the FedAvg optimization; and FedPer 

and pFedPara the personalized models of which the first layer ( FedPer ) and half of the inner 7Published as a conference paper at ICLR 2022 Local FedAvg FedPer pFedPara (Ours) Local FedAvg FedPer pFedPara 

> 58.0 62.2 66.5 70.8 75.0 Acc. [%]

(a) Scenario 1 Local FedAvg FedPer pFedPara  

> 42.0 46.2 50.5 54.8 59.0 Acc. [%]

(b) Scenario 2 Local FedAvg FedPer pFedPara  

> 92.5 94.2 96.0 97.8 99.5 Acc. [%]

(c) Scenario 3 

Figure 5: Average test accuracy over ten local models trained by each algorithm. (a) 100% of local training data on FEMNIST are used with the non-IID setting, which mimics enough local data to train and evaluates each local model on their own data. (b) 20% of local training data on FEMNIST are used with the non-IID setting, which mimics insufficient local data to train local models. (c) 100% of local training data on MNIST are used with the highly-skew non-IID setting, where each client has at most two classes. The error bars denote 95% confidence intervals obtained by 5 repeats. matrices ( pFedPara ) are trained by sharing with the server, respectively, while the other parts are locally updated. We validate these algorithms on three scenarios in Figure 5. In Scenario 1 (Figure 5a), the FedPAQ accuracy is higher than those of FedAvg and FedPer 

because each client has sufficient data. Nevertheless, pFedPara surpasses the other methods. In Scenario 2 (Figure 5b), the FedAvg accuracy is higher than that of FedPAQ because local data are too scarce to train the local models. The FedPer accuracy is also lower than that of FedAvg 

because the last layer of FedPer does not exploit other clients’ data; thus, FedPer is susceptible to a lack of local data. The higher performance of pFedPara shows that our pFedPara can take advantage of the wealth of distributed data in the personalized setup. In Scenario 3 (Figure 5c), the 

FedAvg accuracy is much lower than the other methods due to the highly-skewed data distribution. In most scenarios, pFedPara performs better or favorably against the others. It validates that 

pFedPara can train the personalized models collaboratively and robustly. Additionally, both pFedPara and FedPer save the communication costs because they partially transfer parameters; pFedPara transfers 3.4 times fewer parameters, whereas FedPer transfers 

1.07 times fewer than the original model in each round. The reduction of FedPer is negligible because it is designed to transfer all the layers except the last one. Contrarily, the reduction of 

pFedPara is three times larger than FedPer because all the layers of pFedPara are factorized by the Hadamard product during training, and only a half of each layer’s parameters are transmitted. Thus, pFedPara is far more suitable in terms of both personalization performance and communi-cation efficiency in FL. 

Additional Experiments. We conducted additional experiments of wall clock time simulation and other architectures including ResNet , LSTM , and Pufferfish (Wang et al., 2021), but the results can be found in the supplementary material due to the page limit. 

## 4 RELATED WORK 

Federated Learning. The most popular and de-facto algorithm, FedAvg (McMahan et al., 2017), reduces communication costs by updating the global model using a simple model averaging once a large number of local SGD iterations per round. Variants of FedAvg (Li et al., 2020; Yu et al., 2020; Diao et al., 2021) have been proposed to reduce the communication cost, to overcome data heterogeneity, or to increase the convergence stability. Advanced FL optimizers (Karimireddy et al., 2020; Acar et al., 2021; Reddi et al., 2021; Yuan & Ma, 2020) enhance the convergence behavior and improve communication efficiency by reducing the number of necessary rounds until convergence. Federated quantization methods (Reisizadeh et al., 2020; Haddadpour et al., 2021) combine the quantization algorithms with FedAvg and reduce only upload costs to preserve the model accuracy. Our FedPara is a drop-in replacement for layer parameterization, which means it is an orthogonal and compatible approach to the aforementioned methods; thus, our method can be integrated with other FL optimizers and model quantization. 8Published as a conference paper at ICLR 2022 

Distributed Learning. In large-scale distributed learning of data centers, communication might be a bottleneck. Gradient compression approaches, including quantization (Alistarh et al., 2017; Bern-stein et al., 2018; Wen et al., 2017; Reisizadeh et al., 2020; Haddadpour et al., 2021), sparsification (Alistarh et al., 2018; Lin et al., 2018), low-rank decomposition (Vogels et al., 2019; Wang et al., 2021), and adaptive compression (Agarwal et al., 2021) have been developed to handle communi-cation traffic. These methods do not deal with FL properties such as data distribution and partial participants per round (Kairouz et al., 2019). Therefore, the extension of the distributed methods to FL is non-trivial, especially for optimization-based approaches. 

Low-rank Constraints. As described in section 2.1, low-rank decomposition methods (Lebedev et al., 2015; Tai et al., 2016; Phan et al., 2020) are inappropriate for FL due to additional steps; the post-decomposition after training and fine-tuning. In FL, Koneˇ cn` y et al. (2016) and Qiao et al. (2021) have proposed low-rank approaches. Koneˇ cn` y et al. (2016) train the model from scratch with low-rank constraints, but the accuracy is degraded when they set a high compression rate. To avoid such degradation, FedDLR (Qiao et al., 2021) uses an ad hoc adaptive rank selection and shows the improved performance. However, once deployed, those models are inherently restricted by lim-ited low-ranks. In particular, FedDLR requires matrix decomposition in every up/down transfer. In contrast, we show that FedPara has no such low-rank constraints in theory. Empirically, the mod-els re-parameterized by our method show comparable accuracy to the original counterparts when trained from scratch. 

## 5 DISCUSSION AND CONCLUSION 

To overcome the communication bottleneck in FL, we propose a new parameterization method, 

FedPara , and its personalized version, pFedPara . We demonstrate that both FedPara and 

pFedPara can significantly reduce communication overheads with minimal performance degra-dation or better performance over the original counterpart at times. Even using a strong low-rank constraint, FedPara has no low-rank limitation and can achieve a full-rank matrix and tensor by virtue of our proposed low-rank Hadamard product parameterization. These favorable properties en-able communication-efficient FL, of which regimes have not been achieved by the previous low-rank parameterization and other FL approaches. We conclude our work with the following discussions. 

Discussions. FedPara conducts multiplications many times during training, including the Hadamard product, to construct the weights of the layers. These multiplications may potentially be more susceptible to gradient exploding, vanishing, dead neurons, or numerical instability than the low-rank parameterization with an arbitrary initialization. In our experiments, we have not ob-served such issues when using He initialization (He et al., 2015) yet. Investigating initializations appropriate for our model might improve potential instability in our method. Also, we have discussed the expressiveness of each layers in neural networks in the view of rank. As another view of layer characteristics, statistical analysis of weights and activation also offers ways to initialize weights (He et al., 2015; Sitzmann et al., 2020) and understanding of neural networks (De & Smith, 2020), which is barely explored in this work. It would be a promising future direction to analyze statistical properties of composited weights from our parameterization and activations and may pave the way for FedPara -specific initialization or optimization. Through the extensive experiments, we show the superior performance improvement obtained by our method, and it appears to be with no extra cost. However, the actual payoff exists in the additional computational cost when re-composing the original structure of W from our parameterization dur-ing training; thus, our method is slower than the original parameterization and low-rank approaches. However, the computation time is not a dominant factor in practical FL scenarios as shown in Ta-ble 7, but rather the communication cost takes a majority of the total training time. It is evidenced by the fact that FedPara offers better Pareto efficiency than all compared methods because our method has higher accuracy than low-rank approaches and much less training time than the original one. In contrast, the computation time might be non-negligible compared to the communication time in distributed learning regimes. While our method can be applied to distributed learning, the benefits of our method may be diminished there. Improving both computation and communication efficiency of FedPara in large-scale distributed learning requires further research. It would be a promising future direction. 9Published as a conference paper at ICLR 2022 

## ETHICS STATEMENT 

We describe the ethical aspect in various fields, such as privacy, security, infrastructure level gap, and energy consumption. 

Privacy and Security. Although FL is privacy-preserving distributed learning, personal informa-tion may be leaked due to the adversary who hijacks the model intentionally during FL. Like other FLs, this risk is also shared with FedPara due to communication. Without care, the private data may be revealed by the membership inference or reconstruction attacks (Rigaki & Garcia, 2020). The local parameters in our pFedPara could be used as a private key to acquire the complete personal model, which would reduce the chance for the full model to be hijacked. It would be inter-esting to investigate whether our pFedPara guarantees privacy preserving or the way to improve robustness against those risks. 

Infrastructure Level Gap. Another concern introduced by FL is limited-service access to the people living in countries having inferior communication infrastructure, which may raise human rights concerns including discrimination, excluding, etc . It is due to a larger bandwidth requirement of FL to transmit larger models. Our work may broaden the countries capable of FL by reducing required bandwidths, whereby it may contribute to addressing the technology gap between regions. 

Energy Consumption. The communication efficiency of our FedPara directly leads to noticeable energy-saving effects in the FL scenario. It can contribute to reducing the battery consumption of IoT devices and fossil fuels used to generate electricity. Moreover, compared to the optimization-based FL approaches that reduce necessary communication rounds, our method allows more clients to participate in each learning round under the fixed bandwidth, which would improve convergence speed and accuracy further. 

> ACKNOWLEDGEMENT

This work was supported by Institute of Information & communications Technology Planning &

Evaluation (IITP) grant funded by the Korea government(MSIT) (No.2021-0-02068, Artificial In-telligence Innovation Hub), the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No. NRF-2021R1C1C1006799), and the “HPC Support” Project sup-ported by the ‘Ministry of Science and ICT’ and NIPA. 

## REFERENCES 

Durmus Alp Emre Acar, Yue Zhao, Ramon Matas, Matthew Mattina, Paul Whatmough, and Venkatesh Saligrama. Federated learning based on dynamic regularization. In International Con-ference on Learning Representations (ICLR) , 2021. Saurabh Agarwal, Hongyi Wang, Kangwook Lee, Shivaram Venkataraman, and Dimitris Papail-iopoulos. Adaptive gradient communication via critical learning regime identification. In Machine Learning and Systems (MLSys) , 2021. Dan Alistarh, Demjan Grubic, Jerry Li, Ryota Tomioka, and Milan Vojnovic. Qsgd: Communication-efficient sgd via gradient quantization and encoding. In Advances in Neural Information Processing Systems (NeurIPS) , 2017. Dan Alistarh, Torsten Hoefler, Mikael Johansson, Nikola Konstantinov, Sarit Khirirat, and Cedric Renggli. The convergence of sparsified gradient methods. In Advances in Neural Information Processing Systems (NeurIPS) , 2018. Manoj Ghuhan Arivazhagan, Vinay Aggarwal, Aaditya Kumar Singh, and Sunav Choudhary. Fed-erated learning with personalization layers. arXiv preprint arXiv:1912.00818 , 2019. Jeremy Bernstein, Yu-Xiang Wang, Kamyar Azizzadenesheli, and Animashree Anandkumar. signSGD: Compressed optimisation for non-convex problems. In International Conference on Machine Learning (ICML) , 2018. 10 Published as a conference paper at ICLR 2022 Sebastian Caldas, Sai Meher Karthik Duddu, Peter Wu, Tian Li, Jakub Koneˇ cn` y, H Brendan McMa-han, Virginia Smith, and Ameet Talwalkar. Leaf: A benchmark for federated settings. Advances in Neural Information Processing Systems Workshops (NeurIPSW) , 2018. Luke N Darlow, Elliot J Crowley, Antreas Antoniou, and Amos J Storkey. Cinic-10 is not imagenet or cifar-10. arXiv preprint arXiv:1810.03505 , 2018. Soham De and Sam Smith. Batch normalization biases residual blocks towards the identity function in deep networks. In Advances in Neural Information Processing Systems (NeurIPS) , 2020. Enmao Diao, Jie Ding, and Vahid Tarokh. HeteroFL: Computation and communication efficient federated learning for heterogeneous clients. In International Conference on Learning Represen-tations (ICLR) , 2021. Farzin Haddadpour, Mohammad Mahdi Kamani, Aryan Mokhtari, and Mehrdad Mahdavi. Feder-ated learning with compression: Unified analysis and sharp guarantees. In Artificial Intelligence and Statistics (AISTATS) , 2021. Chaoyang He, Murali Annavaram, and Salman Avestimehr. Group knowledge transfer: Feder-ated learning of large cnns at the edge. In Advances in Neural Information Processing Systems (NeurIPS) , 2020a. Chaoyang He, Songze Li, Jinhyun So, Mi Zhang, Hongyi Wang, Xiaoyang Wang, Praneeth Vepakomma, Abhishek Singh, Hang Qiu, Li Shen, et al. Fedml: A research library and benchmark for federated machine learning. Advances in Neural Information Processing Systems Workshops (NeurIPSW) , 2020b. Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpass-ing human-level performance on imagenet classification. In IEEE International Conference on Computer Vision (ICCV) , 2015. Kevin Hsieh, Amar Phanishayee, Onur Mutlu, and Phillip Gibbons. The non-iid data quagmire of decentralized machine learning. In International Conference on Machine Learning (ICML) , 2020. Yo-Seb Jeon, Mohammad Mohammadi Amiri, Jun Li, and H Vincent Poor. A compressive sensing approach for federated learning over massive mimo communication systems. IEEE Transactions on Wireless Communications , 20(3):1990–2004, 2020. Peter Kairouz, H Brendan McMahan, Brendan Avent, Aur´ elien Bellet, Mehdi Bennis, Arjun Nitin Bhagoji, Keith Bonawitz, Zachary Charles, Graham Cormode, Rachel Cummings, et al. Advances and open problems in federated learning. arXiv preprint arXiv:1912.04977 , 2019. Sai Praneeth Karimireddy, Satyen Kale, Mehryar Mohri, Sashank Reddi, Sebastian Stich, and Ananda Theertha Suresh. Scaffold: Stochastic controlled averaging for federated learning. In 

International Conference on Machine Learning (ICML) , 2020. Hyeji Kim, Muhammad Umar Karim Khan, and Chong-Min Kyung. Efficient neural network com-pression. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR) , 2019. Diederik P. Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In International Conference on Learning Representations (ICLR) , 2015. Jakub Koneˇ cn` y, H Brendan McMahan, Felix X Yu, Peter Richt´ arik, Ananda Theertha Suresh, and Dave Bacon. Federated learning: Strategies for improving communication efficiency. arXiv preprint arXiv:1610.05492 , 2016. Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. Technical report, Citeseer, 2009. Vadim Lebedev, Yaroslav Ganin, Maksim Rakhuba, Ivan V. Oseledets, and Victor S. Lempitsky. Speeding-up convolutional neural networks using fine-tuned cp-decomposition. In International Conference on Learning Representations (ICLR) , 2015. 11 Published as a conference paper at ICLR 2022 Yann LeCun, L´ eon Bottou, Yoshua Bengio, and Patrick Haffner. Gradient-based learning applied to document recognition. Proceedings of the IEEE , 86(11):2278–2324, 1998. Tian Li, Anit Kumar Sahu, Manzil Zaheer, Maziar Sanjabi, Ameet Talwalkar, and Virginia Smith. Federated optimization in heterogeneous networks. In Machine Learning and Systems (MLSys) ,2020. Yujun Lin, Song Han, Huizi Mao, Yu Wang, and Bill Dally. Deep gradient compression: Reducing the communication bandwidth for distributed training. In International Conference on Learning Representations (ICLR) , 2018. Jian-Hao Luo, Jianxin Wu, and Weiyao Lin. Thinet: A filter level pruning method for deep neural network compression. In IEEE International Conference on Computer Vision (ICCV) , 2017. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and Blaise Aguera y Arcas. Communication-efficient learning of deep networks from decentralized data. In Artificial Intelli-gence and Statistics (AISTATS) , 2017. Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Kopf, Edward Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. Pytorch: An imperative style, high-performance deep learning library. In Advances in Neural Information Processing Systems (NeurIPS) , 2019. Anh-Huy Phan, Konstantin Sobolev, Konstantin Sozykin, Dmitry Ermilov, Julia Gusak, Petr Tichavsk` y, Valeriy Glukhov, Ivan Oseledets, and Andrzej Cichocki. Stable low-rank tensor de-composition for compression of convolutional neural network. In European Conference on Com-puter Vision (ECCV) , 2020. Zhefeng Qiao, Xianghao Yu, Jun Zhang, and Khaled B Letaief. Communication-efficient federated learning with dual-side low-rank compression. arXiv preprint arXiv:2104.12416 , 2021. Stephan Rabanser, Oleksandr Shchur, and Stephan G¨ unnemann. Introduction to tensor decomposi-tions and their applications in machine learning. arXiv preprint arXiv:1711.10781 , 2017. Sashank J. Reddi, Zachary Charles, Manzil Zaheer, Zachary Garrett, Keith Rush, Jakub Koneˇ cn´ y, Sanjiv Kumar, and Hugh Brendan McMahan. Adaptive federated optimization. In International Conference on Learning Representations (ICLR) , 2021. Amirhossein Reisizadeh, Aryan Mokhtari, Hamed Hassani, Ali Jadbabaie, and Ramtin Pedarsani. Fedpaq: A communication-efficient federated learning method with periodic averaging and quan-tization. In Artificial Intelligence and Statistics (AISTATS) , 2020. Maria Rigaki and Sebastian Garcia. A survey of privacy attacks in machine learning. arXiv preprint arXiv:2007.07646 , 2020. William Shakespeare. The complete works of william shakespeare. https://www. gutenberg.org/ebooks/100 , 1994. Karen Simonyan and Andrew Zisserman. Very deep convolutional networks for large-scale image recognition. In International Conference on Learning Representations (ICLR) , 2015. Vincent Sitzmann, Julien Martel, Alexander Bergman, David Lindell, and Gordon Wetzstein. Im-plicit neural representations with periodic activation functions. In Advances in Neural Information Processing Systems (NeurIPS) , 2020. Speedtest. SPEEDTEST. https://www.speedtest.net/global-index . Accessed: 2021-05-26. Cheng Tai, Tong Xiao, Xiaogang Wang, and Weinan E. Convolutional neural networks with low-rank regularization. In International Conference on Learning Representations (ICLR) , 2016. Thijs Vogels, Sai Praneeth Karimireddy, and Martin Jaggi. Powersgd: Practical low-rank gradient compression for distributed optimization. In Advances in Neural Information Processing Systems (NeurIPS) , 2019. 12 Published as a conference paper at ICLR 2022 Hongyi Wang, Saurabh Agarwal, and Dimitris Papailiopoulos. Pufferfish: Communication-efficient models at no extra cost. In Machine Learning and Systems (MLSys) , 2021. Wei Wen, Cong Xu, Feng Yan, Chunpeng Wu, Yandan Wang, Yiran Chen, and Hai Li. Terngrad: Ternary gradients to reduce communication in distributed deep learning. In Advances in Neural Information Processing Systems (NeurIPS) , 2017. Sarika Yadav and Rama Shankar Yadav. A review on energy efficient protocols in wireless sensor networks. Wireless Networks , 22(1):335–350, 2016. Ming Yan, Chien Aun Chan, Andr´ e F Gygax, Jinyao Yan, Leith Campbell, Ampalavanapillai Nir-malathas, and Christopher Leckie. Modeling the total energy consumption of mobile network services and applications. Energies , 12(1):184, 2019. Felix Yu, Ankit Singh Rawat, Aditya Menon, and Sanjiv Kumar. Federated learning with only positive labels. In International Conference on Machine Learning (ICML) , 2020. Honglin Yuan and Tengyu Ma. Federated accelerated stochastic gradient descent. In Advances in Neural Information Processing Systems (NeurIPS) , 2020. Guangxu Zhu, Yuqing Du, Deniz G¨ und¨ uz, and Kaibin Huang. One-bit over-the-air aggregation for communication-efficient federated edge learning: Design and convergence analysis. IEEE Transactions on Wireless Communications , 20(3):2120–2135, 2020. 13 Published as a conference paper at ICLR 2022 

## SUPPLEMENTARY 

In this supplementary material, we present additional details, results, and experiments that are not included in the main paper due to the space limit. The contents of this supplementary material are listed as follows: 

## CONTENTS 

A Maximal Rank Property B Additional Techniques C Details of Experiment Setup 

C.1 Datasets C.2 Models C.3 FedPara & pFedPara C.4 Hyper-parameters of Backbone Optimizer C.5 Hyper-parameters of Other Optimizer 

D Additional Experiments 

D.1 Training Time D.2 Other Models 

## A MAXIMAL RANK PROPERTY 

This section presents the proofs of our propositions and an additional analysis of our method’s algorithmic behavior in terms of the rank property. A.1 PROOFS 

Proposition 1 Let X1 ∈ Rm×r1 , X2 ∈ Rm×r2 , Y1 ∈ Rn×r1 , Y2 ∈ Rn×r2 , r1, r 2 ≤ min( m, n )

and the constructed matrix be W := ( X1Y> 

> 1

) (X2Y> 

> 2

). Then, rank( W) ≤ r1r2.Proof. X1Y> 

> 1

and X2Y> 

> 2

can be expressed as the summation of rank-1 matrices such that XiY> 

> i

=

∑j=ri 

> j=1

xij y> 

> ij

, where xij and yij are the j-th column vectors of Xi and Yi, and i ∈ { 1, 2}. Then, 

W = X1Y> 

> 1

X2Y> 

> 2

=

> j=r1

∑

> j=1

x1j y>

> 1j
> j=r2

∑

> j=1

x2j y> 

> 2j

=

> k=r1

∑

> k=1
> j=r2

∑

> j=1

(x1ky>

> 1k

) (x2j y> 

> 2j

). (2) 

W is the summation of r1r2 number of rank-1 matrices; thus, rank( W) is bounded above r1r2. 

Proposition 2 Given R ∈ N, r1 = r2 = R is the unique optimal choice of the following criteria, 

arg min r1,r 2∈N (r1 + r2)( m + n) s.t. r1r2 ≥ R2, (3) 

and its optimal value is 2R(m + n).Proof. We use arithmetic-geometric mean inequality and the given constraint. We have 

(r1 + r2)( m + n) ≥ 2√r1r2(m + n) ≥ 2R(m + n). (4) 14 Published as a conference paper at ICLR 2022 The equality holds if and only if r1 = r2 = R by the arithmetic–geometric mean inequality. 

Corollary 1 Under Proposition 2, R2 ≥ min( m, n ) is a necessary and sufficient condition for achieving the maximal rank of W = ( X1Y> 

> 1

) (X2Y> 

> 2

) ∈ Rm×n, where X1 ∈ Rm×r1 , X2 ∈

Rm×r2 , Y1 ∈ Rn×r1 , Y2 ∈ Rn×r2, and r1, r 2 ≤ min( m, n ).Proof. We first prove the sufficient condition. Given r1 = r2 = R under Proposition 2 and R2 ≥

min( m, n ), rank( W) ≤ min( r1r2, m, n ) = min( R2, m, n ) = min( m, n ). The matrix W has no low-rank restriction; thus the condition, R2 ≥ min( m, n ), is the sufficient condition. The necessary condition is proved by contraposition; if R2 < min( m, n ), the matrix W cannot achieve the maximal rank. Since r1 = r2 = R under Proposition 2 and R2 < min( m, n ), then 

rank( W) ≤ min( r1r2, m, n ) = min( R2, m, n ) = R2 < min( m, n ). That is, rank( W) is upper-bounded by R2, which is lower than the maximal achievable rank of W. Therefore, the condition, 

R2 ≥ min( m, n ), is the necessary condition because the contrapositive is true. 

Corollary 1 implies that, with R2 ≥ min( m, n ), the constructed weight W does not have the low-rank limitation, and allows us to define the minimum inner rank as rmin := min( d√m e, d√n e).If we set r1 = r2 = rmin , rank( W) of our FedPara can achieve the maximal rank because 

r1r2 = r2 

> min

≥ min( m, n ) while minimizing the number of parameters. 

Proposition 3 Let T1, T2 ∈ RR×R×k3×k4 , X1, X2 ∈ Rk1×R, Y1, Y2 ∈ Rk2×R, R ≤ min( k1, k 2)

and the convolution kernel be W := ( T1 ×1 X1 ×2 Y1) (T2 ×1 X2 ×2 Y2). Then, the rank of the kernel satisfies rank( W(1) ) = rank( W(2) ) ≤ R2.Proof. According to Rabanser et al. (2017), the 1st and 2nd unfolding of tensors can be expressed as 

W(1) = ( X1T (1) 1 (I(4) ⊗ I(3) ⊗ Y1)>) (X2T (1) 2 (I(4) ⊗ I(3) ⊗ Y2)>),

W(2) = ( Y1T (2) 1 (I(4) ⊗ I(3) ⊗ X1)>) (Y2T (2) 2 (I(4) ⊗ I(3) ⊗ X2)>), (5) where I(3) ∈ Rk3×k3 and I(4) ∈ Rk4×k4 are identity matrices. Since W(1) and W(2) are matrices, we apply the same process used in Eq. 2, then we obtain rank( W(1) ) = rank( W(2) ) ≤ R2. 

A.2 ANALYSIS OF THE RANK PROPERTY 

To demonstrate our propositions empirically, we sample the parameters randomly and count 

rank( W). When applying our parameterization to W ∈ R100 ×100 , we set rmin = 10 by Corol-lary 1 to Proposition 2. We sample the entries of X1, X2, Y1, Y2 ∈ R100 ×10 from the standard Gaussian distribution and repeat this experiment 1, 000 times. As shown in Figure 6, we observe that our parameterization achieves the full rank with the probabil-ity of 100% but requires 2.5 times fewer entries than the original 100 × 100 matrix. This empirical result demonstrates that our parameterization can span the full-rank matrix with fewer parameters efficiently. 20 40 60 80 100 Rank 0250 500 750 1000 Count 

Figure 6: Histogram of rank( W). We apply FedPara to W ∈ R100 ×100 and set r1 = r2 = 10 .We repeat this experiment 1, 000 times. 15 Published as a conference paper at ICLR 2022 

## B ADDITIONAL TECHNIQUES 

We devise additional techniques to improve the accuracy and stability of our parameterization. We consider injecting a non-linearity in FedPara and Jacobian correction regularization to further squeeze out the performance and stability. However, as will be discussed later, these are optional. 

Non-linear Function. The first is to apply a non-linear function before the Hadamard product. 

FedPara composites the weight as W = W1 W2, and we inject non-linearity as W = σ(W1)

σ(W2) as a practical design. This makes our algorithm departing from directly applying the proofs, 

i.e ., empirical heuristics. However, the favorable performance of our FedPara regardless of this non-linearity injection in practice suggests that our re-parameterization is a reasonable model to deploy. The non-linear function σ(·) can be any non-linear function such as ReLU , Tanh , and Sigmoid , but 

ReLU and Sigmoid restrict the range to only positive, whereas Tanh has both negative and positive values of range [−1, 1] . The filters may learn to extract the distinct features, such as edge, color, and blob, which require both negative and positive values of the filters. Furthermore, the reasonably bounded range can prevent overshooting a large value by multiplication. Therefore, Tanh is suitable, and we use Tanh through experiments except for those of pFedPara .

Jacobian Correction. The second is the Jacobian correction regularization, which induces that the Jacobians of X1, X2, Y1, and Y2 follow the Jacobian of the constructed matrix W. Suppose that 

X1, X2 ∈ Rm×r and Y1, Y2 ∈ Rn×r are given. We construct the weight as W = W1 W2,where W1 = X1Y> 

> 1

and W2 = X2Y> 

> 2

. Additionally, suppose that the Jacobian of W with respect to the objective function is given: JW = ∂L 

> ∂W

. We can compute the Jacobians of X1, X2, Y1, and 

Y2 with respect to the objective function and apply one-step optimization with SGD. For simplicity, we set the momentum and the weight decay to be zero. Then, we can compute the Jacobian of other variables using the chain rule: 

JW1 = ∂L

∂W1

= JW W2, JX1 = ∂L

∂X1

= JW1 Y> 

> 1

, JY1 = ∂L

∂Y1

= X> 

> 1

JW1 ,

JW2 = ∂L

∂W2

= JW W1, JX2 = ∂L

∂X2

= JW2 Y> 

> 2

, JY2 = ∂L

∂Y2

= X> 

> 2

JW2 .

(6) We update the parameters using SGD with the step size η as follows: 

X′ 

> 1

= X1 − ηJX1 , Y′ 

> 1

= Y1 − ηJY1 ,

X′ 

> 2

= X2 − ηJX2 , Y′ 

> 2

= Y2 − ηJY2 . (7) We can compute W′, which is the constructed weight after one-step optimization: 

W′ = ( X′

> 1

Y′>  

> 1

) (X′

> 2

Y′>  

> 2

)= {(X1 − ηJX1 )( Y> 

> 1

− ηJ> 

> Y1

)} {(X2 − ηJX2 )( Y> 

> 2

− ηJ> 

> Y2

)}

= {X1Y> 

> 1

− η(JX1 Y> 

> 1

+ X1J> 

> Y1

) + η2JX1 J> 

> Y1

} {X2Y> 

> 2

− η(JX2 Y> 

> 2

+ X2J> 

> Y2

) + η2JX2 J> 

> Y2

}

= ( X1Y> 

> 1

) (X2Y> 

> 2

) + η4(JX1 J> 

> Y1

) (JX2 J> 

> Y2

)

− η3{(JX1 Y> 

> 1

+ X1J> 

> Y1

) (JX2 J> 

> Y2

) + ( JX2 Y> 

> 2

+ X2J> 

> Y2

) (JX1 J> 

> Y1

)}

+ η2{(JX1 Y> 

> 1

+ X1J> 

> Y1

) (JX2 Y> 

> 2

+ X2J> 

> Y2

) + ( JX2 J> 

> Y2

) (X2Y> 

> 2

) + ( JX2 J> 

> Y2

) (X1Y> 

> 1

)}− η{(JX1 Y> 

> 1

+ X1J> 

> Y1

) (X2Y> 

> 2

) + ( JX2 Y> 

> 2

+ X2J> 

> Y2

) (X1Y> 

> 1

)}.

(8) 

This shows that gradient descent and ascent are mixed, as shown in the signs of each term. We propose the Jacobian correction regularization to minimize the difference between W′ and W −

ηJW, which induces our parameterization to follows the direction of W − ηJW. The total objective function consists of the target loss function and the Jacobian correction regularization as: 

R = L(X1, X2, Y1, Y2) + λ

2 || W′ − (W − ηJW)|| 2. (9) 

Results. We evaluate the effects of each technique in Table 4. We train VGG16 with group normal-ization on the CIFAR-10 IID setting during the same target rounds. We set γ = 0 .1 and λ = 10 . As 16 Published as a conference paper at ICLR 2022 Models Accuracy 

FedPara (base) 82.45 ± 0.35 

+ Tanh 82.42 ± 0.33 

+ Regularization 82.38 ± 0.30 

+ Both 82.52 ± 0.26 Table 4: Accuracy of FedPara with additional techniques. 95% confidence intervals are presented with eight repetitions. shown, the model with both Tanh and regularization has higher accuracy and lower variation than the base model. There is gain in accuracy and variance with both techniques, whereas only variance with only one technique. Again, note that these additional techniques are not essential for FedPara to work; therefore, we can optionally use these techniques depending on the situation where the device has enough computing power. 

## C DETAILS OF EXPERIMENT SETUP 

In this section, we explain the details of the experiments, including datasets, models, and hyper-parameters. We also summarize our FedPara and pFedPara into the pseudo algorithm. For implementation, we use PyTorch Distributed library (Paszke et al., 2019) and 8 NVIDIA GeForce RTX 3090 GPUs. C.1 DATASETS 

CIFAR-10. CIFAR-10 (Krizhevsky et al., 2009) is the popular classification benchmark dataset. CIFAR-10 consists of 32 × 32 resolution images in 10 classes, with 6, 000 images per class. We use 

50 , 000 images for training and 10 , 000 images for testing. For federated learning, we split training images into 100 partitions and assign one partition to each client. For the IID setting, we split the dataset into 100 partitions randomly. For the non-IID setting, we use the Dirichlet distribution and set the Dirichlet parameter as 0.5 as suggested by He et al. (2020b;a). 

CIFAR-100. CIFAR-100 (Krizhevsky et al., 2009) is the popular classification benchmark dataset. CIFAR-100 consists of 32 × 32 resolution images in 100 classes, with 6, 000 images per class. We use 50 , 000 images for training and 10 , 000 images for testing. For federated learning, we split training images into 50 partitions. For the IID setting, we split the dataset into 50 partitions randomly. For the non-IID setting, we use the Dirichlet distribution and set the Dirichlet parameter as 0.5 as suggested by He et al. (2020b;a). 

CINIC-10. CINIC-10 (Darlow et al., 2018) is a drop-in replacement for CIFAR-10 and also the popular classification benchmark dataset. CINIC-10 consists of 32 × 32 resolution images in 10 

classes and three subsets: training, validation, and test. Each subset has 90 , 000 images with 9, 000 

per class, and we do not use the validation subset for training. For federated learning, we split training images into 100 partitions. For the IID setting, we split the dataset into 100 partitions randomly. For the non-IID setting, we use the Dirichlet distribution and set the Dirichlet parameter as 0.5 as suggested by He et al. (2020b;a). 

MNIST. MNIST (LeCun et al., 1998) is a popular handwritten number image dataset. MNIST consists of 70 , 000 number of 28 × 28 resolution images in 10 classes. We use 60 , 000 images are for training and 10 , 000 images for testing. We do not use MNIST IID-setting, and we split the dataset so that clients have at most two classes as suggested by McMahan et al. (2017) for a highly-skew non-IID setting. 

FEMNIST. FEMNIST (Caldas et al., 2018) is a handwritten image dataset for federated settings. FEMNIST has 62 classes and 3, 550 clients, and each client has 226 .83 data samples on average of 

28 × 28 resolution images. FEMNIST is the non-IID dataset labeled by writers. 17 Published as a conference paper at ICLR 2022 

Shakespeare. Shakespeare (Shakespeare, 1994) is a next word prediction dataset for federated learning settings. Shakespeare has 80 classes and 1, 129 clients, and each client has 3, 743 .2 data samples on average (Caldas et al., 2018). C.2 MODELS 

VGG16. PyTorch library (Paszke et al., 2019) provides VGG16 with batch normalization, but we replace the batch normalization layers with the group normalization layers as suggested by Hsieh et al. (2020) for federated learning. We also modify the FC layers to comply with the number of classes. The dimensions of the output features in the last three FC layers are 512 –512 –〈#classes 〉,sequentially. We do not apply our parameterization to the last three FC layers and set the same γ

to all convolutional layers in the model for simplicity. For reference purpose, Table 5 shows the number of parameters corresponding to each γ.

γ No. parameters 10-classes 100-classes original 15.25M 15.30M 0.1 1.55 M 1.59 M 0.2 2.33 M 2.38 M 0.3 3.31 M 3.36 M 0.4 4.45 M 4.50 M 0.5 5.79 M 5.84 M 0.6 7.33 M 7.38 M 0.7 9.01 M 9.05 M 0.8 10.90 M 10.94 M 0.9 12.92 M 12.96 M Table 5: γ’s and their corresponding numbers of parameters for VGG16 ori . and VGG16 FedPara .

Two FC Layers. In personalization experiments, we use two FC layers as suggested by McMahan et al. (2017) but modify the size of the hidden features corresponding to the number of classes in the datasets. The dimensions of the output features in two FC layers are 256 and the number of classes, respectively; i.e ., 256 –〈#classes 〉. We do not use other layers, such as normalization and dropout, and set γ = 0 .5 for pFedPara .

LSTM. For the Shakespeare dataset, we use two-layer LSTM as suggested by McMahan et al. (2017) and Acar et al. (2021). We set the hidden dimension as 256 and the number of classes as 80. We also apply the weight normalization technique on original parameterization, low-rank parameterization, and FedPara .C.3 FED PARA & PFED PARA 

Algorithm 1: FedPara 

Input: rounds T , parameters {X1l, X2l, Y1l, Y2l}l=Ll=1 where {X1l, X2l, Y1l, Y2l} is the lth 

layer of the model and L is the number of layers 

for t = 1 , 2, . . . , T do 

Sample the subset S of clients; 

for each client c ∈ S do 

Download {X1l, X2l, Y1l, Y2l}l=Ll=1 from the server; Optimize( {X1l, X2l, Y1l, Y2l}l=Ll=1 ); Upload {X1l, X2l, Y1l, Y2l}l=Ll=1 to the server; 

end 

Aggregate {X1l, X2l, Y1l, Y2l}l=Ll=1 ;

end 

18 Published as a conference paper at ICLR 2022 

Algorithm 2: pFedPara 

Input: rounds T , parameters {X1l, X2l, Y1l, Y2l}l=Ll=1 where {X1l, X2l, Y1l, Y2l} is the lth 

layer of the model and L is the number of layers Transmit {X1l, X2l, Y1l, Y2l}l=Ll=1 to clients to train the same initial point at start; 

for t = 1 , 2, . . . , T do 

Sample the subset S of clients; 

for each client c ∈ S do 

Download half of parameters {X1l, Y1l}l=Ll=1 from the server; Optimize( {X1l, X2l, Y1l, Y2l}l=Ll=1 ); Upload {X1l, Y1l}l=Ll=1 to the server; 

end 

Aggregate {X1l, Y1l}l=Ll=1 ;

end 

We summarize our two methods, FedPara and pFedPara , into Algorithms 1 and 2 for apparent comparison. In these algorithms, we mainly use the popular and standard algorithm, FedAvg , as a backbone optimizer, but we can switch with other optimizer and aggregate methods. As mentioned in Section B, we can consider the additional techniques. In the FedPara experiments, we use the regularization to Algorithm 1, and set the regularization coefficient λ as 1.0. In pFedPara 

experiments, we do not apply the additional techniques to Algorithm 2. C.4 HYPER -PARAMETERS OF BACKBONE OPTIMIZER 

FedAvg (McMahan et al., 2017) is the most popular algorithm in federated learning. The server samples S number of clients as a subset in each round, each client of the subset trains the model locally by E number of SGD epochs, and the server aggregates the locally updated models and repeats these processes during the total rounds T . We use FedAvg as a backbone optimization algorithm, and its hyper-parameters of our experiments, such as the initial learning rate η, local batch size B, and learning rate decay τ , are described in Table 6. Models CIFAR-10 CIFAR-100 CINIC-10 LSTM FEMNIST & MNIST IID non-IID IID non-IID IID non-IID IID non-IID                                                                

> K16 16 8816 16 16 16 10
> T200 200 400 400 300 300 500 500 100
> E10 510 510 5115
> B64 64 64 64 64 64 64 64 10
> η0.1 0.1 0.1 0.1 0.1 0.1 1.0 1.0 0.1-0.01
> τ0.992 0.992 0.992 0.992 0.992 0.992 0.992 0.992 0.999
> λ111111000

Table 6: Hyper-parameters of our FedPara with FedAvg 

C.5 HYPER -PARAMETERS OF OTHER OPTIMIZERS 

For compatibility experiment, we combine FedPara with other optimization-based FL algorithms: 

FedProx (Li et al., 2020), SCAFFOLD (Karimireddy et al., 2020), FedDyn (Acar et al., 2021), and FedAdam (Reddi et al., 2021). FedProx (Li et al., 2020) imposes a proximal term to the objective function to mitigate heterogeneity; SCAFFOLD (Karimireddy et al., 2020) allows clients to reduce the variance of gradients by introducing auxiliary variables; FedDyn (Acar et al., 2021) introduces dynamic regularization to reduce the inconsistency between minima of the local device level empirical losses and the global one; FedAdam employs Adam (Kingma & Ba, 2015) at the server-side instead of the simple model average. They need a local optimizer to update the model in each client, and we use the SGD optimizer for a fair comparison, and the SGD configuration is the same as that of FedAvg . The four algo-rithms have additional hyper-parameters. FedProx has a proximal coefficient μ, and we set μ as 

0.1. SCAFFOLD has Options I and II to update the control variate, and we use Option II with global learning rate ηg (= 1 .0). FedDyn has the hyper-parameter α (= 0 .1) in the regularization. 

FedAdam uses Adam optimizer to aggregate the updated models at the server-side, and we use the 19 Published as a conference paper at ICLR 2022 Network speed Model tcomp. tcomm. t                           

> 2 Mbps VGG16 ori .1.64 sec. 470.2 sec. 471.84 sec.
> VGG16 FedPara (γ=0.1) 2.34 sec. 47.2 sec. 49.54 sec. ( ×9.52 )10 Mbps VGG16 ori .1.64 sec. 94.04 sec. 94.68 sec.
> VGG16 FedPara (γ=0.1) 2.34 sec. 9.44 sec. 11.78 sec. ( ×8.04 )50 Mbps VGG16 ori .1.64 sec. 18.61 sec. 20.25 sec.
> VGG16 FedPara (γ=0.1) 2.34 sec. 1.88 sec. 4.22 sec. ( ×4.80 )

Table 7: The required time during one round. We denote the computation time, the communication time, and the total time during one round as tcomp. , tcomm. , and t, respectively. We set the network speeds as 2, 10, and 50 Mbps. Network speed Model Training time 2 Mbps VGG16 ori . 880.77 min. 

VGG16 FedPara (γ=0.1) 94.95 min. ( × 9.28 )10 Mbps VGG16 ori . 176.74 min. 

VGG16 FedPara (γ=0.1) 22.58 min. ( × 7.83 )50 Mbps VGG16 ori . 37.80 min. 

VGG16 FedPara (γ=0.1) 8.09 min. ( × 4.67 )Table 8: The real training time to achieve the target accuracy. We set the network speeds as 2, 10, and 50 Mbps, and the required rounds for VGG16 ori . is 112, FedPara (γ = 0 .1) is 115 to achieve the same target accuracy in the CIFAR-10 IID setting. parameters β1 = 0 .9, β2 = 0 .99 , the global learning rate ηg = 0 .01 , and the local learning rate 

η = 10 −1.5 for Adam optimizer. 

## D ADDITIONAL EXPERIMENTS 

In this section, we simulate the computation time, the communication time, and total training time in Section D.1, experiment about other models in Section D.2, and compare our method and the quantization approach in Section D.3. D.1 TRAINING TIME 

We can compute the elapsed time during one round in FL by t = tcomp. + tcomm. , where tcomp. is the computation time of training the model on local data for several epochs and tcomm. is the com-munication time of downloading and uploading the updated model. We estimate the computation time by measuring the elapsed time during local epochs. We compute the communication time by  

> 2·model size (Mbyte) network speed (Mbyte /s)

, considering upload and download of the model in one round. Since it is challenging to experiment in a real heterogeneous network environment, we follow the simple standard network simulation setting widely used in the communication literature (Zhu et al., 2020; Jeon et al., 2020). They assume the homogeneous link quality by the average quality to simplify complex network environments in FL communication simulation, i.e., the network speeds are identical for all clients. We compare the elapsed time per round of VGG16 ori . and VGG16 FedPara on different network speeds such as 2, 10, and 50 Mbps. As shown in Table 7, the communication time is larger than the computation time indicating that the communication is a bottleneck in FL as expected. Although 

VGG16 FedPara takes more computation time than VGG16 ori . due to the weight composition time, 

VGG16 FedPara decreases the communication time by about ten times and the total time by 4.80 to 9.52 times. Table 8 shows the total training time to achieve the same accuracy on the CIFAR-10 IID setting. Although FedPara needs three rounds more than the original parameterization, 

VGG16 FedPara requires 4.67 to 9.68 times less training time than VGG16 ori . because of communi-cation efficient parameterization. 20 Published as a conference paper at ICLR 2022 0 86 172 259 345 Communication Cost [GB] 10.0 28.5 47.0 65.5 84.0 Acc. [%]    

> VGG16 ori.
> VGG16 FedPara (= 0.1)
> VGG16 FedPara (= 0.4)
> VGG16 FedPara (= 0.9)

(a) CIFAR-10 IID 0 55 110 165 220 Communication Cost [GB] 10.0 20.2 30.5 40.8 51.0 Acc. [%]     

> VGG16 ori.
> VGG16 FedPara (= 0.1)
> VGG16 FedPara (= 0.6)
> VGG16 FedPara (= 0.9)

(b) CIFAR-100 IID 0 136 272 409 545 Communication Cost [GB] 10.0 25.5 41.0 56.5 72.0 Acc. [%]     

> VGG16 ori.
> VGG16 FedPara (= 0.1)
> VGG16 FedPara (= 0.6)
> VGG16 FedPara (= 0.9)

(c) CINIC-10 IID 0 86 172 259 345 Communication Cost [GB] 10.0 26.0 42.0 58.0 74.0 Acc. [%]    

> VGG16 ori.
> VGG16 FedPara (= 0.1)
> VGG16 FedPara (= 0.6)
> VGG16 FedPara (= 0.9)

(d) CIFAR-10 non-IID 0 90 180 270 360 Communication Cost [GB] 10.0 19.2 28.5 37.8 47.0 Acc. [%]     

> VGG16 ori.
> VGG16 FedPara (= 0.1)
> VGG16 FedPara (= 0.6)
> VGG16 FedPara (= 0.9)

(e) CIFAR-100 non-IID 0 128 256 384 512 Communication Cost [GB] 10.0 24.2 38.5 52.8 67.0 Acc. [%]     

> VGG16 ori.
> VGG16 FedPara (= 0.1)
> VGG16 FedPara (= 0.6)
> VGG16 FedPara (= 0.9)

(f) CINIC-10 non-IID 

Figure 7: (a-f) Accuracy [%] ( y-axis) vs. communication costs [GBytes] ( x-axis) of VGG16 ori . and 

VGG16 FedPara . Broken black line represents VGG16 ori ., red solid line VGG16 FedPara with low γ,blue solid line VGG16 FedPara with mid γ, and yellow solid line VGG16 FedPara with high γ

Model Acc. of 200 rounds Acc. of 1000 rounds (gain) Original 83.68 84.1 (+ 0.42) 

FedPara (γ = 0 .1) 82.88 83.35 (+ 0.47) 

FedPara (γ = 0 .2) 82.53 83.34 (+ 0.81) 

FedPara (γ = 0 .3) 83.11 83.94 (+ 0.83) 

FedPara (γ = 0 .4) 84.05 84.59 (+ 0.54) 

FedPara (γ = 0 .5) 83.82 84.57 (+ 0.75) 

FedPara (γ = 0 .6) 83.63 84.16 (+ 0.53) 

FedPara (γ = 0 .7) 83.79 84.24 (+ 0.45) 

FedPara (γ = 0 .8) 83.40 84.00 (+ 0.6) Table 9: The accuracy of VGG16 with original and our parameterization on the CIFAR-10 IID setting during 200 and 1000 rounds. D.2 OTHER MODELS 

VGG16 As shown in Figure 7, we compare VGG16 ori . and VGG16 FedPara with three dif-ferent γ values. Note that a higher γ uses more parameters than a lower γ. Compared to 

VGG16 ori ., VGG16 FedPara achieves comparable accuracy and even higher accuracy with a high 

γ. VGG16 FedPara also requires fewer communication costs than VGG16 ori ..We also investigate how much accuracy is increased for longer rounds. Table 9 shows that the accuracy is increased in the long round experiment, but the tendency of the 1000 round training is consistent with the 200 round training. 

ResNet18 We demonstrate the consistent effectiveness of our method with another architecture, 

ResNet18 . For ResNet18 , we train without replacing batch normalization layers, set γ of the first layer, the second layer, and the 1 × 1 convolution layers as 1.0, and adjust γ of remaining layers to control the ResNet18 FedPara size; γ = 0 .1 for small model size, γ = 0 .6 for mid model size, and γ = 0 .9 for large model size. To train the ResNet18 FedPara in FL, we set the batch size to 10 .As revealed in Figure 8a, ResNet18 FedPara has comparable accuracy and uses fewer communica-tion costs than ResNet18 ori ., of which the results are consistent with the VGG16 experiments. Figure 8b shows the communication costs required for model training to achieve the same tar-21 Published as a conference paper at ICLR 2022 0 86 172 259 345 Communication Cost [GB] 10.0 30.8 51.5 72.2 93.0 Acc. [%]    

> ResNet18 ori.
> ResNet18 FedPara (= 0.1)
> ResNet18 FedPara (= 0.6)
> ResNet18 FedPara (= 0.9)

(a) Accuracy ResNet18 ori. ResNet18 FedPara     

> (= 0.1)
> ResNet18 FedPara
> (= 0.6)
> ResNet18 FedPara
> (= 0.9)
> Communication Cost [GB]  169.2 33.2 87.3 144.4

(b) Communication Costs 

Figure 8: (a) Accuracy [%] (y-axis) vs. communication costs [GBytes] ( x-axis) of ResNet18 ori .

and ResNet18 FedPara with three γ values. (b) Size comparison of transferred parameters, which is expressed as communication costs [GBytes] ( y-axis), for the same target accuracy 90% . (a, b): Black represents ResNet18 ori ., and red, blue, and yellow represents ResNet18 FedPara with low, mid, and high γ, respectively. Model Acc. # parameters (ratio) 

VGG16 Pufferfish 82.07 0.33 

VGG16 Pufferfish 82.42 0.44 

VGG16 FedPara (γ = 0 .2) 82.53 0.15 

VGG16 FedPara (γ = 0 .4) 84.05 0.29 Table 10: The accuracy of VGG16 Pufferfish and VGG16 FedPara on CIFAR-10 IID setting. #

parameters is the the ratio of each model parameter when the number of parameters of VGG16 ori . is set 1.0. get accuracy. Our ResNet18 FedPara needs 1.17 to 5.1 times fewer communication costs than 

ResNet18 ori ., and the results demonstrate that FedPara is also applicable to the ResNet struc-ture. 

Pufferfish Pufferfish (Wang et al., 2021) is similar to our method, where they use partially pre-factorized networks and employ the hybrid architecture by maintaining original size weights to minimize the accuracy loss due to the low-rank constraints. Since this work can be directly applied in the FL setup, we compare the parameterization of PufferFish and FedPara as follows. We train VGG16 with PufferFish and FedPara on the CIFAR-10 IID dataset in FL, and evaluate the models according to varying the number of parameters. As shown in Table 10, our FedPara has higher accuracy with fewer parameters compared with PufferFish. Although PufferFish is superior to the naive low-rank pre-decomposition method due to its hybrid architecture, we think the hybrid architecture of PufferFish still suffers from the low-rank constraints on the top layers. However, our method is free from such limitations both theoretically and empirically. 

LSTM We train LSTM ori ., LSTM low , and LSTM FedPara on the Shakespeare dataset. The parameters ratio of LSTM low and LSTM FedPara are about 16% and 19% of the LSTM ori . parameters, respectively. Table 11 shows that LSTM FedPara outperforms LSTM low and LSTM ori . on the IID setting. In the non-IID setting, LSTM FedPara accuracy is higher than LSTM low and slightly lower than LSTM ori . with only 19% of parameters. Therefore, our parameterization can be applied to general neural networks. D.3 QUANTIZATION 

FedPAQ (Reisizadeh et al., 2020) is a quantization approach to reduce communication costs for FL. To compare the accuracy and transferred size per round, we train VGG16 on the CIFAR-10 IID setting. We consider both downlink and uplink to evaluate communication costs. We quantize the model from 32 bits floating-point numbers to 16 bits floating-point numbers. 22 Published as a conference paper at ICLR 2022 Model Acc. (IID) Acc. (non-IID) # parameters (ratio) 

LSTM ori . 60.17 52.66 1.0 

LSTM low 54.59 51.24 0.16 

LSTM FedPara (γ = 0 .0) 63.65 51.56 0.19 Table 11: The accuracy of LSTMs on IID and non-IID setting. # parameters is the the ratio of each model parameter when the number of parameters of LSTM ori . is set 1.0. Model Acc. Transferred size per round 

FedAvg 83.68 122MB 

FedPAQ 82.67 91.4MB 

FedPara (γ = 0 .5) 83.82 46.4MB + FedPAQ 83.58 34.74MB 

Table 12: The accuracy of VGG16 on CIFAR-10 IID setting for original, quantization, and our parameterization models. Table 12 shows the comparison between FedPara and FedPAQ as well as those integration. Com-pared to FedPAQ , FedPara transfers 1.96 times lower bits per round because FedPAQ only re-duces the uplink communication cost. Compared with FedAvg , FedPara achieves 0.14 % higher accuracy, but FedPAQ 1.01 % lower accuracy. FedPara can transfer the full information of up-dated weights to the server, whereas FedPAQ loses the updated weight information due to the quantization. Furthermore, since the way FedPara reduces communication costs is different from quantization, we integrate FedPara with FedPAQ as an extension. Combining our method with 

FedPAQ reduces communication costs by 25% further from our FedPara without the integration while having a minor accuracy drop, 0.1 %, from that of FedAvg .23
