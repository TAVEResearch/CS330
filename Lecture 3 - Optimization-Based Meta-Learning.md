# Lecture 3 - Optimization-Based Meta-Learning

# General recipe

How to evaluate a meta-learning algorithm

the Omniglot dataset: 1623 characters from 50 different alphabets

![image](https://user-images.githubusercontent.com/54271228/139542406-8bfc3f73-11bd-4452-9ac3-ffc2f4166b4c.png)

each character has only 20 instances (many classes and few examples for class)

statistics more reflective of the real world 

->포크를 본다고 할때 우리는 몇개의 포크를 보지 몇만개의 포크를 보진 않음

Proposes both few-shot discriminaAve & few-shot generative problems

분류를 할 수 있는가? 새로 데이터를 생성할 수 있는가?

Initial few-shot learning approaches w/ Bayesian models, non-parametrics

적은 데이터로 학습할 수 있는가?

![image](https://user-images.githubusercontent.com/54271228/139542548-4dae27ab-016f-48cd-8067-cd895bd88feb.png)

5개의 class가 있고, class마다 한개의 예시가 있음

새로운 예시를 분류하라

meta-learning의 목표: leverage data from other image classes in order to solve this problem

다른 예시들을 가지고 와서 train과 test set으로 나눈 다음에

왼쪽 train set으로 학습시켰을 때 오른쪽 test set에 대해서 잘 분류할 수 있도록 학습함

그 다음에 최종적으로 held-out classes를 잘 분류할 수 있도록 함

+regression, language generation, skill learning 과 같은 다양한 머신러닝 문제들에 응용할 수 있음

multitask learning에서는 training class 전부 분류를 하는 것을 말한다면, 

meta learning은 training task를 이용해서 새로운 task를 잘 수행할 수 있도록 하는 것이 목표임



## The Meta-Learning Problem: The Mechanistic View

Supervised Learning

![image](https://user-images.githubusercontent.com/54271228/139542847-3d64efdc-c6cc-49c6-8476-65684e0b471f.png)

Meta-supervised Learning 

![image](https://user-images.githubusercontent.com/54271228/139542865-8bb2ecfa-bf21-4823-8490-acd044fdbbee.png)

dataset을 이용해서 학습을 함: dataset은 k shot learning problem일 때 k개의 input, output 쌍을 가짐

-> meta learning에서는 meta-training dataset를 통해 function f를 알고 싶은 것! 

**Why is this view useful?** 

Reduces the problem to the design & optimization of f 

-> 저 함수만 구하면 meta-learning 한거라고 볼 수 있음



## The Meta-Learning Problem: The Probabilistic View

Supervised Learning

![image](https://user-images.githubusercontent.com/54271228/139543025-9640f12b-9d16-4902-91a6-74f8efff2026.png)

Meta-supervised Learning 

![image](https://user-images.githubusercontent.com/54271228/139543033-443fb99c-81d4-472a-bb23-e19c8ec50bf6.png)

주어진 training dataset에 task specific parameter Φi에 대한 inference

training task에서 meta parameter에 대한 inference를 maximum likelihood 취한 것



### How to design a meta-learning algorithm?

1. Choose a form of  p(Φi|Dtr, θ )

2. Choose how to op?mize w.r.t. max-likelihood objective using  Dmeta-train



Can we treat p(Φi|Dtr, θ ) as an inference problem?

왜냐면, 뉴럴 네트워크는 inference problem을 잘 해결하기 때문





## Black-Box Adaptation

Key idea: Train a neural network to represent p(Φi|Dtr, θ )

For now: Use deterministic (point estimate) 

![image](https://user-images.githubusercontent.com/54271228/139543417-c0f36e20-6791-46f7-b2fe-dc5eb90e459c.png)

![image](https://user-images.githubusercontent.com/54271228/139543432-f0fcb877-3fe1-4f22-996e-d751d7965a89.png)

input: training data set

output : Φi

또 별개의 뉴럴 네트워크는 test data에 대해 Φi로 test data points에 대한 예측을 수행함

g가 생성하는 분포의 확률을 maximize 하는 식의 지도학습 방식으로 볼 수 있음

meta learning과정에서  θ 최적화하는것 

안에서 Φ는 각 task마다 동적으로 계산됨 - 파라미터라기 보다는 텐서와 같은 개념



1. Sample task Ti (mini bactch of tasks)

2. Sample disjoint datasets Dtr , Dtest from Di

3. Compute Φi <- fθ(Dtr)
4. Update θ using ▽θL(Φi, Dtest )

task specific parameter를 이용해서 meta parameter를 구하기

그리고 meta-training dataset에 대해서 optimizer를 사용해서 반복함

test dataset를 이용해서 파라미터를 평가함

Challenge 

Q: Outputting all neural net parameters does not seem scalable?

뉴럴 네트워크 너무 크고 Φ가 이걸 다 반영한다고?

Idea: Do not need to output all parameters of neural net, only sufficient statistics

![image](https://user-images.githubusercontent.com/54271228/139552468-3a33f4d3-48ad-4853-a19a-a2b81023063f.png)

lower-demensional vector hi를 사용함

h : hidden state of LSTM

task representation h를 생성하기 위해서 학습함



어떤 함수를 사용해야하는가?

1. LSTMs or Neural turing machine (NTM)
2. Feedforward + average
3. Other external memory mechanisms
4. Convolutions & attention



HW 1: 

- implement data processing 
- implement simple black-box meta-learner

- train few-shot Omniglot classifier



Advantage

- expressive 

+ easy to combine with variety of learning problems (e.g. SL, RL) 



Disadvantage

- complex model w/ complex task: challenging optimization problem 

- often data-inefficient



How else can we represent p(Φi|Dtr, θ )?

Is there a way to infer all parameters in a scalable way?

What if we treat it as an optimization procedure?



## Optimization-Based Inference

Key idea: Acquire Φi through optimization

max log p(Dtr|Φi) + log p(Φi|θ)

최적화 절차!

Meta-parameters serve as a prior.

What form of prior?

One successful form of prior knowledge: initializatin for fine-tuning



##### Fine-tuning

##### ![image](https://user-images.githubusercontent.com/54271228/139631516-f7926238-739c-45ae-8fc5-23f88e9e711f.png) 

set initial parameters and then run gradient descent on training data for new task

![image](https://user-images.githubusercontent.com/54271228/139713636-da5da3af-1942-40b1-8de3-da7a1207f56f.png)

랜덤보다 original 성능이 월등히 좋음

-> 먼저 meta-training data에 대해 파라미터로 사전학습을 한 후, test 할때, 내 데이터셋으로 fine-tune 진행

Where do you get the pre-trained parameters? 

- ImageNet classification 
- Models trained on large language corpora (BERT, LMs) 
- Other unsupervised learning techniques 
- Whatever large, diverse dataset you might have



Some common practices 

- Fine-tune with a smaller learning rate 
- Lower learning rate for lower layers 
- Freeze earlier layers, gradually unfreeze 
- Reinitialize last layer 
- Search over hyperparameters via cross-val 
- Architecture choices matter (e.g. ResNets)

![image](https://user-images.githubusercontent.com/54271228/139714740-cbf10dbe-7701-4df7-9c03-e612c59cd459.png)

Fine-tuning less effective with very small datasets.



![image](https://user-images.githubusercontent.com/54271228/139714931-b760bc7c-5e9f-46a8-bdf2-191852c62e93.png)

그럼 적은 데이터로 fine-tuning을 효과적으로 할 수 있도록 meta-learning을 사용하면 어떨까?

fine-tuning에서 test data set에 대해 task-specific parameter가 잘 했는지 평가하고 pre-training parameter θ를 최적화함

이를 모든 meta-training data set에 대해 진행하면, 적은 데이터로 fine-tuning을 효과적으로 할 수 있음

Key idea: Over many tasks, learn parameter vector θ that transfers via fine-tuning

![image](https://user-images.githubusercontent.com/54271228/139715445-f8799a8c-77f7-4446-8eca-8a1b0e5c5af3.png)

저 빨간 원의 상태에 있을 때 gradient step을 사용해서 최적화를 시킴 

meta-learning이 끝나면 저 검정 점으로 가는 것 : 3개의 task의 optimum 과 이제 유사

-> Model-Agnostic Meta-Learning

저 그림은 직관적이게 잘 설명하지만, misleading 할 수 있음

1. 파라미터 벡터는 2차원으로 존재하지 않음
2. 하나의 optimum이 존재하지 않음 : a whole space of optima

##### General Algorithm

![image](https://user-images.githubusercontent.com/54271228/139716536-ff9c6564-4440-4f87-af17-3ceb26ed4243.png)

뉴럴 네트워크를 사용해서 task-specific 파라미터를 계산하는 것 대신에  한차례나 여러단계의 fine-tuning을 사용해서 계산함 

이를 바탕으로 meta-parameter를 업데이트

—> brings up second-order derivatives

Do we need to compute the full Hessian?

Do we get higher-order derivatives with more inner gradient steps?

(1:00:00~1:13:00) 수학........

### Optimization vs. Black-Box Adaptation

![image](https://user-images.githubusercontent.com/54271228/139772202-cbbf3d2c-acf6-459c-ab03-6416b8d37433.png)

Φi 는 gradient descent으로 구해짐

Note: Can mix & match components of computation graph 

Learn initialization but replace gradient update with learned network

![image](https://user-images.githubusercontent.com/54271228/139772368-0a47895c-aa32-4c4e-93a8-fc02efdc6255.png)



How well can learning procedures generalize to similar, but extrapolated tasks?

![image](https://user-images.githubusercontent.com/54271228/139772434-68e61926-aa3b-4df4-9f72-97f93fd78f41.png)

SNAIL, MetaNetworks: black box approach

x: task variabilty

y: performance

meta-trained 분포(x: 중간) 와 멀어질수록 성능이 떨어짐

-> 하지만 MAML은 그래도 그나마 나음 : test time에 최적화 절차(gradient descent) 를 거치기 때문

-> 그에 반해, black box approach는 data set을 가지고 그냥 답을 내기 때문에, 기존 분포에 멀어진 데이터가 새로 주어졌을 때 알고리즘에서 무슨 일이 벌어지는 모름



##### Does this structure come at a cost?

한 차례 또는 여러 차례의 gradient step으로 어디까지 갈 수 있는가?

black box approach 만큼 expressive 한가?

For a sufficiently deep f, MAML function can approximate any function of training dataset and test input.

-> black box approach가 할 수 있는 모든 거 가능

대신 몇 가지 가정이 있음

Mild Assumptions: (single gradient descent )

- inner learning rate is nonzero 
- loss function gradient does not lose information about the label 
- datapoints in are unique

Why is this interesting? 

MAML has benefit of inductive bias without losing expressive power.



