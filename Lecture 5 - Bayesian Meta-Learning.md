# Bayesian Meta-Learning

Multi-task 과 Meta-Learning의 기본 원칙

1. Training and testing must match
2. Tasks must share structure

What does “structure” mean?

-statistical dependence on shared latent information θ

![image](https://user-images.githubusercontent.com/54271228/140648879-30133c12-7771-4ff1-8c55-659a9dfe4bd9.png)

위의 모델구조를 보면 전부 θ에 의존적임

이때 Φ1과  θ에 대한 Φ2는 독립적임

you have a lower entropy p(Φi| θ) <  p(Φi)

Thought exercise #1: If you can identify θ(i.e. with meta-learning), when should learning Φi be faster than learning from scratch? 

->Entrophy conditioned on θ = p(Φi| θ)는 marginal entropy p(Φi)만큼 높을 것 

Thought exercise #2 What if Entrophy p(Φi| θ) = 0 ?

->Φ 알 필요가 없음 θ가 모든 task를 해결할 수 있음

θ가 무엇을 의미하는가?

machine translation의 경우 θ corresponds to the family of all language pairs

Note that θ is narrower than the space of all possible functions.

Thought exercise #3: What if you meta-learn without a lot of tasks? “meta-overfitting”

-> full distribution을 포착하지 않고 단지 training data에 한정하는 결과를 도출하게 됨: overfitting

![image](https://user-images.githubusercontent.com/54271228/140652781-bd26f69b-45b1-4079-b82e-7888ccf06352.png)

데이터로 underlying function을 다 설명하지 못할 수 있음

 dataset이 너무 작기 때문에 오른쪽의 이미지들이 모호함 

Can we learn to generate hypotheses about the underlying function?

<img src="https://user-images.githubusercontent.com/54271228/140652889-7aec4228-ed2d-4364-86f5-fb0fb600d582.png" alt="image" style="zoom:100%;" />

uncertainty estimates를 구해서 이유를 댈 수 있도록 함



Important for..(Motivation)

- safety-critical few-shot learning (e.g. medical imaging) 

- learning to actively learn 
- learning to explore in meta-RL

### 

#### Computation graph perspective

Version 0: Let f output the parameters of a distribution over y^ts

ex) 

- probability values of discrete categorical distribution 
- mean and variance of a Gaussian 

- means, variances, and mixture weights of a mixture of Gaussians 
- for multi-dimensional : parameters of a sequence of distributions (i.e. autoregressive model)

Then, optimize with maximum likelihood

Pros: 

+ simple 
+ can combine with variety of methods 

Cons: 

- can’t reason about uncertainty over the underlying function [to determine how uncertainty across datapoints relate] 
- limited class of distributions over y^ts can be expressed 
- tends to produce poorly-calibrated uncertainty estimates

underlying function에 대한 불명확성을 설명할 수 없음

Thought exercise #4: Can you do the same maximum likelihood training for Φ?



### The Bayesian Deep Learning Toolbox

Goal: represent distributions with neural networks

Latent variable models + variational inference (Kingma & Welling ‘13, Rezende et al. ‘14): 

- approximate likelihood of latent variable model with variational lower bound 

Bayesian ensembles (Lakshminarayanan et al. ‘17): 

- particle-based representation: train separate models on bootstraps of the data 

Bayesian neural networks (Blundell et al. ‘15): 

- explicit distribution over the space of network parameters 

Normalizing Flows (Dinh et al. ‘16): 

- invertible funcBon from latent distribuBon to data distribution 

Energy-based models & GANs (LeCun et al. ’06, Goodfellow et al. ‘14): 

- estimate unnormalized density



###### Background: The Variational Lower Bound

Observed variable x, latent variable z

![image](https://user-images.githubusercontent.com/54271228/140653954-b228a759-be95-4430-8b02-987f6fcfd6d5.png)

### Bayesian black-box meta-learning 

![image](https://user-images.githubusercontent.com/54271228/140654341-97a8a6c0-86b8-483c-ab41-0bb9d597e35e.png)

train dataset을 input으로 neural network를 거쳐서 Φ에 대한 분포를 생성함 

이걸 neural network에 넣어서 input x에 대한 y를 도출함



What should q condition on?

![image](https://user-images.githubusercontent.com/54271228/140654500-9353c837-2f20-4a43-bcea-e808a4152671.png)

What about the meta-parameters θ?

![image](https://user-images.githubusercontent.com/54271228/140654532-5f689ba9-e623-422c-994d-ee31f5538a25.png)

Pros: 

+ can represent non-Gaussian distribuBons over y^ts
+ produces distribution over underlying functions 

Cons: 

- Can only represent Gaussian distribuBons yts p(ϕi|θ)

  

### What about Bayesian optimization-based meta-learning?

Provides a Bayesian interpretation of MAML

But, we can’t sample from p (ϕi|θ, Dtr) 

![image](https://user-images.githubusercontent.com/54271228/140654728-026fde41-3f80-4add-95ce-69fe18a1a587.png)

q can include a gradient operator!

q corresponds to SGD on the mean & variance of neural network weights  

-> MAML: parameter에 gradient descent 사용

->이 방법: parameter의 mean과 variance에 gradient descent 사용해서 mean과 variance 둘다 구함

Pro: Running gradient descent at test time. 

Con: p(ϕi|θ) modeled as a Gaussian



Can we model non-Gaussian posterior?

->Can we use ensembles?

**Ensemble of MAMLs (EMAML)**

Train M independent MAML models on different subset of data

-> won't work well if ensemble members are too similar

**Stein Variational Gradient (BMAML)**

-> more diverse ensemble 

Use stein variational gradient (SVGD) to push particles away from one another

Optimize for distribution of M particles to produce high likelihood.

Pros: Simple, tends to work well, non-Gaussian distributions. 

Con: Need to maintain M model instances (or do gradient-based inference on last layer only)



Can we model non-Gaussian posterior over all parameters?

Sample parameter vectors with a procedure like Hamiltonian Monte Carlo?

노이즈 추가함 -> gradient descent -> 분포에 대한 sample을 얻기 위해 반복

Intuition: Learn a prior where a random kick can put us in different modes

![image](https://user-images.githubusercontent.com/54271228/140655186-8b89370c-cd98-4d1e-8c7c-8db94f059e2d.png)

![image](https://user-images.githubusercontent.com/54271228/140655222-6a2613b3-faa3-48f6-a217-291a76f3c202.png)



### How to evaluate a Bayesian meta-learner?

Use the standard benchmarks?

\+ standardized 

+real images 

+good check that the approach didn’t break anything 

-metrics like accuracy don't evaluate uncertainty 

-tasks may not exhibit ambiguity 

-uncertainty may not be useful on this dataset!



What are better problems & metrics? It depends on the problem you care about!

*Evaluation on Ambiguous Generation Tasks(Gordon et al., ICLR ’19)

-one shot learning generation MSE SSIM (quantitative measure)

*Accuracy, Mode Coverage, & Likelihood on Ambiguous Tasks (Finn, Xu, Levine, NeurIPS ’18)

![image](https://user-images.githubusercontent.com/54271228/140655702-3f281f23-dbab-498a-82e7-fbd31422b8b5.png)

*Reliability Diagrams & Accuracy(Ravi & Beatson, ICLR ’19)

confidence vs accuracy

![image](https://user-images.githubusercontent.com/54271228/140655734-4eb88078-aeec-4f1d-966e-c2fdf7e9a26c.png)

*Active Learning Evaluation

![image](https://user-images.githubusercontent.com/54271228/140655761-03dce53f-2435-4a89-96d3-003dd0f45e6f.png)

Both experiments: 

- Sequentially choose datapoint with maximum predictive entropy to be labeled 

- or choose datapoint at random (MAML)