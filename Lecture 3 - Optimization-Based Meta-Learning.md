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

42:25

