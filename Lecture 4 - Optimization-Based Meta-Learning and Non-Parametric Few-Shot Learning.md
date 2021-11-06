# Optimization-Based Meta-Learning and Non-Parametric Few-Shot Learning

## Probabilistic Interpretation of Optimization-Based Inference

Key idea: Acquire Φi through optimization

Meta-parameters θ serve as a prior. One form of prior knowledge: initializataion for fine-tuning

![image](https://user-images.githubusercontent.com/54271228/140099013-7c01c5f7-8b9d-4c3f-8b4c-a6317f195522.png)

색칠된 원은 각 task에 대한 data points -> meta training process 동안에 측정 가능

![image](https://user-images.githubusercontent.com/54271228/140099348-f86dac33-190e-468f-ae1f-60bf7f8237ae.png)

1. maximize likelihood of our data set given meta parameters

2. sum of log likelihood 로 다시 작성하고, Φi를 이용하여 meta parameter θ를 최적화함

   -> θ가 주어졌을 때 data에 대한 확률 =  Φi 가 주어졌을 때 Data의 확률과 θ가 주어졌을 때, Φi 의 확률

   = empirical Bayes

3. MAP estimate: Maximal probability 

How to compute MAP estimate? 

Gradient descent with early stopping = **MAP inference** under Gaussian prior with mean at initial parameters [Santos ’96] (exact in linear case, approximate in nonlinear case)

= MAML objective 의 inner loop = MAML approximates hierarchical Bayesian inference.



Gradient-descent + early stopping (MAML): implicit Gaussian prior

![image](https://user-images.githubusercontent.com/54271228/140101097-e2048e12-c7df-416a-b14a-c4497038225d.png)

Other forms of priors?

- Gradient-descent with explicit Gaussian prior

  ![image](https://user-images.githubusercontent.com/54271228/140101236-7934ff7b-2053-4261-9b01-69ef7796cbbf.png)

- Bayesian linear regression(=meta parameters) on learned features

- ridge regression, logisBc regression: top of learned features in the inner loop
- support vector machine

-> 다른 inner loop을 사용하는 meta optimization algorithm 



#### Challenges 

###### How to choose architecture that is effective for inner gradient-step?

Idea: Progressive neural architecture search + MAML (Kim et al. Auto-Meta)

\- finds highly non-standard architecture (deep & narrow) 

- different from architectures that work well for standard supervised learning

ex) MiniImagenet, 5-way 5-shot 

MAML, basic architecture: 63.11% MAML + AutoMeta: 74.65%



###### Bi-level optimization can exhibit instabilities 

###### (instabilities - 현재 아키텍처에서 성능이 그렇게 좋지 않음)

Idea: Automatically learn inner vector learning rate, tune outer learning rate (Li et al. Meta-SGD, Behl et al. AlphaMAML) 

->각 파라미터마다/네트워크의 각 레이어마다 다른 learning rate

->biases and weights may want to have different learning rates (biases는 높은 learning rate, weight는 낮은 learning rate를 원할 때 이것을 다른 레이어마다 분리시킬 수 있어야 한다. )

Idea: Optimize only a subset of the parameters in the inner loop (Zhou et al. DEML, Zintgraf et al. CAVIA) 

Idea: Decouple inner learning rate, BN statistics per-step (Antoniou et al. MAML++) 

각각의 gradient step마다 다른 learning rate, BN statistics을 가지고 있기 때문에

Idea: Introduce context variables for increased expressive power. (Finn et al. bias transforma&on, Zintgraf et al. CAVIA)

다른 계산을 방해하지 않고 gradient step이 parameter를 가지고 정보를 넣을 수 있도록

**Takeaway: a range of simple tricks that can help optimization significantly**



###### Backpropagating through many inner gradient steps is compute-&  memory-intensive.

Idea 1: Crudely approximate dΦi/dθ  as identity

**Takeaway: works for simple few-shot problems, but (anecdotally) not for more complex meta-learning problems**

Can we compute the meta-gradient without differenciating through the optimization path?

dU(θ) /dθ   	(U:update rule- explicit Gaussian prior on the parameters)

Φ = U(θ , Dtr) 

Idea: Derive meta-gradient using the implicit function theorem





optimazation path에 의존하지 않고 오직 최적화의 마지막 point에 의존하는 결과를 이끌어냄

Idea 2: Derive meta-gradient using the implicit function theorem

![image](https://user-images.githubusercontent.com/54271228/140131493-a2216cbd-af5a-43c4-bdf6-184560f45658.png)

iMAML 더 안정감 있음: gradient step을 늘리면 메모리가 증가하는 MAML과는 다르게 iMAML는 최적화 절차를 다 적재할 필요없음



Takeaways: Construct bi-level optimization problem. (inner: gradient descent,  different procedures to compute meta-gradients )

+positive inductive bias at the start of meta-learning  (이미 최적화 되어있는 loop)

+consistent procedure, tends to extrapolate better + maximally expressive with sufficiently deep network 

+model-agnostic (easy to combine with your favorite architecture) 

-typically requires second-order optimization 

-usually compute and/or memory intensive



Can we embed a learning procedure without a second-order optimization?

-> Non-parametric methods



여태까지 배운것은 parametric model을 배움

low data regime -> non-parametric method가 간단하고 잘 동작함

meta-test time: few-shot learning <-> low data regime

meta-training: still want to be parametric

Can we use parametric meta-learners that produce effective non-parametric learners?

몇몇의 non-parametric model은 parametric approaches를 넘어섰음



### Non-parametric methods

Key Idea: Use non-parametric learner

![image](https://user-images.githubusercontent.com/54271228/140610394-e0347a78-6c86-4b51-9b18-fdf7d33efde3.png)

Compare test image with training images 

**In what space do you compare? With what distance metric?**

-> pixel space, l^2 distance(이미지에서 성능 별로임)

->Learn to compare using meta-training data!

1. train Siamese network to predict whether or not two images are the same class

![image](https://user-images.githubusercontent.com/54271228/140610457-c15d83ea-ddf2-4eed-8606-54790cd38196.png)

-> 이미지 간의 의미적 거리를 알 수 있음

meta-training dataset으로 부터 이 이미지는 다른 클래스임을 알음 ->label 0

![image](https://user-images.githubusercontent.com/54271228/140610512-f4615e88-5b34-44f0-a6dc-aabae5896c52.png)

같은 클래스이기 때문에 label 1

meta-training dataset에서 다른 쌍의 이미지에 대해 이것을 반복하게 됨

meta-training time: 뉴럴네트워크에게 같은 클래스인지 아닌지 분류시킴

(pairwise comparison)

meta-test time: few shot classification을 하고 싶다면 Xtest에 있는 이미지와 training set에 있는 이미지와 비교해서 가장 비슷한(확률이 제일 높은) 이미지에 레이블을 부여함

Meta-training: Binary classification (두 쌍의 이미지가 같은 클래스인가?) 

Meta-test: N-way classification (이 이미지가 어떤 클래스에 속해있는가?)



Can we match meta-train & meta-test?  

Nearest neighbors in learned embedding space

![image](https://user-images.githubusercontent.com/54271228/140612957-77d52050-aa09-4fbd-9b30-903fe8f7b9cf.png)

training dataset의 이미지를 learned embedding space에 넣음

test query 이미지를 가지고 또 embedding space에 넣음

그리고 이 임베딩을 비교해서 예측을 함

검정색 점은 test embedding 과 training embedding 비교하는 것

training 이미지에 대한 label을 가지고 weighed nearest neighbors를 사용해서 예측 (유사도 점수로 weighed)



![image](https://user-images.githubusercontent.com/54271228/140613260-b0316ecf-dcec-4ab9-80fc-c218ea323bcd.png)

What if >1 shot?

Matching networks will perform comparisons **independently**



Can we aggregate class information to create a prototypical embedding?

![image](https://user-images.githubusercontent.com/54271228/140613346-24cc47d6-4479-447a-a682-73dbc6a351f5.png)

-각 class마다 embedding space의 평균을 구함: prototypical embedding space

-test image를 넣어서 prototypical class embedding과의 거리를 계산함

-가장 가까운 것을 구함



Challenge:

What if you need to reason about more complex relationships between datapoints?

![image](https://user-images.githubusercontent.com/54271228/140613613-f286d240-51a8-4a0a-9cc9-1c312c58fec7.png)



## Comparison of approaches

##### Computation graph perspective

Black-box vs. Opmization vs. Non-Parametric

![image](https://user-images.githubusercontent.com/54271228/140613696-97a16a77-4dbf-48b2-b804-415f827e8840.png)

##### Algorithmic properties perspective

1. Expressive power

   the ability for function to represent a range of learning procedures

   ->scalability, applicability to a range of domains

2. Consistency

   learned learning procedure will solve task with enough data

   ->reduce reliance on meta-training tasks, good OOD task performance

3. Uncertainty awareness

   ability to reason about ambiguity during learning

   ->active learning, calibrated uncertainty, RL principled Bayesian approaches

![image](https://user-images.githubusercontent.com/54271228/140613902-776cf7e2-3e7d-49e6-a3eb-ca69ac4991b1.png)

Generally, well-tuned versions of each perform comparably on existing few-shot benchmarks! ->Which method to use depends on your use-case.