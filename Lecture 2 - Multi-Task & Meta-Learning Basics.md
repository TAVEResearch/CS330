# Multi-Task & Meta-Learning Basics

[강의영상](https://www.youtube.com/watch?v=6stKGH6zI8g&list=PLoROMvodv4rMC6zfYmnD7UG3LVvwaITY5&index=2)

## Some notation

input: 이미지 , 논문 제목

output: 이미지의 카테고리, 논문의 길이

![image](https://user-images.githubusercontent.com/54271228/139283267-bfc57e34-f2d6-4e0d-86ae-129bc33d7e3a.png)

Single-task learning: [supervised] 

데이터는 input, output 쌍으로 이루어짐

목표: 파라미터로 loss fuction 최소화

![image](https://user-images.githubusercontent.com/54271228/139283654-9d6536ff-409f-4f12-ae02-1792fea261bf.png)

그리고 보편적인 loss fuction으로는 negative log likelihood가 있음

![image](https://user-images.githubusercontent.com/54271228/139283863-5328f67a-ab77-4254-8b82-51b7a846e956.png)

single task learning은 이 log-likelihood를 최적화하는 것인데, 네트워크의 파라미터로 back-propagation 하는 것

SGD, Adam 등등과 같은 optimizer 사용함



#### What is task?

인풋에 대한 distribution을 가지고, 인풋과 loss fuction이 주어진 상황에서 label에 대한 distribution을 가짐

P: data generating distributions

![image](https://user-images.githubusercontent.com/54271228/139284670-d931c8d2-a7b4-49fb-9c23-04448394144d.png)



#### Multi-task classification

 모든 task에 대해 똑같은 loss function 사용

하지만 input이나 거기에 대응하는 label이 task에 따라 다를 수 있음

ex) per-language handwriting recognition, personalized spam filter

-> Pi(x)가 다름: 사람마다 받는 메일(input)이 다름

->Pi(y|x)도 다름: 어떤 사람에게 spam인 메일이 다른 사람에게는 아닐 수 있음



#### Multi-label classification

모든 task에 대해 loss fuction과 input에 대한 분포 (Pi(x))가 다 같음

ex) CelebA attribute recognition 

얼굴이미지 데이터셋이 있을 때, 

사람이 모자를 쓰고 있는 지 여부 판단하는 것이 하나의 task

사람의 머리색깔을 판단하는 것이 또 다른 task

다른 이진 분류 task이기 때문에 Pi(y|x)만 다름



**지금까지 모든 task에서 loss function은 동일하되, data distribution이 다른것**

그렇다면 loss function이 다른 경우는 언젠가?

- mixed discrete, continuous labels across tasks 

- if you care more about one task than another 

  (어떠한 하나의 task를 더 고려해서 해결하고 싶은 경우 - higher weight)

  

![image](https://user-images.githubusercontent.com/54271228/139288729-4c419c89-cbb2-4d69-866b-40c7ad51b349.png)

X: 논문의 제목

y: 논문의 길이, 논문의 요약, 논문 리뷰

task decriptor: one-hot encoding of the task index 

이 task가 task1, task2, task3,,,어떤 인덱스인지 알려줌

또는, meta-data를 포함하고 있음 

(-personalization: user features/attributes 

 -language description of the task 

 -formal specifications of the task)

Objective: 우리가 가진 모든 task를 다 더할 것임

![image](https://user-images.githubusercontent.com/54271228/139289563-17a216a4-640e-4e8e-ae8f-1ce65b6cc1bc.png)

A model decision and an algorithm decision: 

#1. How should we condition on Z_i (task descriptor)? 

#2. How to optimize our objective?



### #1. Conditioning on the task

Z_i가 task index라고 가정하자

Question: How should you condition on the task in order to share as little as possible?

각각의 task들이 최대한 적게 공유하는 환경을 설정할 수 있을 것인가? 

##### Answer: 각각 다른 네트워크를 만들 수 있음 (다른 가중치...등)

task descriptor에게 가장 부합하는 task를 고르게 함

-> task index에 따라 어떤 아웃풋을 낼 것인지 고르게 하는 multiplicative gating 

![image](https://user-images.githubusercontent.com/54271228/139290194-1c43c303-c4d4-41e8-b2f9-f99299f3c8e1.png)

independent training within a single network with no shared parameters



##### 또 다른 방법: Concatenate z with input and/or activations

all parameters are shared except the parameters directly following Z_i

![image](https://user-images.githubusercontent.com/54271228/139291101-5920166e-e704-46a6-94a5-75351d7d7705.png)

##### An Alternative View on the Multi-Task Objective

Split θ into shared parameters θ and task-specific parameters sh θi

![image](https://user-images.githubusercontent.com/54271228/139292241-57b68cdf-7574-4c4b-abb9-b44f325172cf.png)

이전의 objective과 차이점은 

task specific 파라미터가 존재해서 그 특정 task objective에 최적화되어있다는 점

shared 파라미터는 모든 task에 최적화된 것

**Z_i에 대한 condition을 선택하는 것은 곧 어떻게(how), 어디서(where) 파라미터를 공유할 것인지 선택하는것 **



### #1 Conditioning: Some Common Choices

1. Concatenation-based conditioning

   task descriptor를 가지고 특징들과 concatente 진행하고 그 다음에 linear layer 거침

   ![image](https://user-images.githubusercontent.com/54271228/139292887-ce651f08-31f2-4ddf-b840-070f006bbcf3.png)

2. Additive conditioning

   linear layer거친 다음에 hidden unit 특징들에 task vector를 추가함

![image](https://user-images.githubusercontent.com/54271228/139292930-86a0b067-e72b-4433-a210-d6acc1be4584.png)

-> 사실상 두개 똑같음 

![image](https://user-images.githubusercontent.com/54271228/139293479-3933d574-eee2-4205-959f-728620673308.png)

왼쪽에 x랑 z를 concat한 것(1.Concatenation-based conditioning) 을 각각의 레이어에 적용된 weight matrix를 보면 결국에 오른쪽에 두개의 요소가 합쳐진 것(2.Additive conditioning)을 얻게됨

3. Multi-head architecture 

   파라미터를 가지고, 네트워크를 다른 head로 쪼개기

   ![image](https://user-images.githubusercontent.com/54271228/139294264-3790a970-0e27-4133-8ba0-ca0e2ce0e545.png)

4. Multiplicative conditioning

   Additive conditioning과 원리가 유사함

   더하는 것 대신에 곱하는 것

![image](https://user-images.githubusercontent.com/54271228/139294492-3139966e-aedd-420b-9e8d-f9ecfbb64312.png)

​	Why might multiplicative conditioning be a good idea?

- more expressive 

- recall: multiplicative gating

  다른 task에 대해서 네트워크의 어떤 부분을 사용할 것인지 선택할 수 있음



### #1. Conditioning Choices

Unfortunately, these design decisions are like neural network architecture tuning:

- problem dependent 

- largely guided by intuition or knowledge of the problem 
- currently more of an art than a science



### #2. Optimizing the objective

![image](https://user-images.githubusercontent.com/54271228/139350123-bd0ae5f1-27ac-43e6-b533-4dcd5e25877a.png)

Basic Version: 

1. Sample mini-batch of tasks  ℬ ∼ {Ti }

2. Sample mini-batch datapoints for each task D_i^b ∼ D_i 

   -> 지도 학습에서 샘플링을 한번 하는 것과는 다르게 두번 샘플링을 진행함

3. Compute loss on the mini-batch: ℒ ̂ (θ, ℬ) = ∑ ℒ_k(θ,D_k^b)

4.  Backpropagate loss to compute gradient  ∇θℒ ̂

5.  Apply gradient with your favorite neural net optimizer (e.g. Adam) } 

   This ensures that tasks are sampled uniformly, regardless of data quantities. 

   Tip: For regression problems, make sure your task labels are on the same scale!

   ​		-> 그렇지 않으면 큰 데이터셋에 대해서 큰 loss function 값을 갖기 때문

   



## Challenges

### 1. Negative Transfer

independent network으로 수행했을 때 multitask learning 방법으로 수행했을 때보다 더 잘할 때

ex) Multi-Task CIFAR-100 (SOTA)

​		![image](https://user-images.githubusercontent.com/54271228/139350728-5fefacef-6e24-4d6a-b621-93662e9e8407.png)

왜 이런 현상이 발생하는가?

1. Optimization challenges

   caused by cross-task interference

   -> gradient (task1)이 gradient(task2)에게 악영향을 미침 : 최적화 어렵

   task may learn at different rates

   ->어떤 task는 빠르게 배우는데, 어떤 task는 그렇지 못함

2. Limited representational capacity

   multi-task networks often need to be much larger than their single-task counterparts

   까먹기 쉽기 때문에 크지 않으면 underfitting 발생가능



##### Soft parameter sharing

If you have negative transfer, share less across tasks.

It's not just binary decision! -> **soft parameter sharing**

![image](https://user-images.githubusercontent.com/54271228/139351641-a084f2bc-5f42-4e28-95a6-db430575b864.png)

Task specific parameter가 다른 것들과 비슷해질 수 있도록 더해주는 것

<img src="https://user-images.githubusercontent.com/54271228/139351722-78b4fbb2-9838-4bd3-88c8-54344185cc58.png" alt="image" style="zoom:80%;" />

장점) allows for more fluid degrees of parameter sharing 

단점) yet another set of design decisions/hyperparameters



### 2. Overfitting

You may not be sharing enough!

더 많은 데이터를 필요로 하는 것일수도 있겠지만, 강한 정규화의 형태로 더 많이 공유한다면, 덜 overfit할 수 있음

Multi-task learning <-> a form of regularization

-> 해결방안: 더 많이 공유해라



### Case study: 유튜브 추천 알고리즘

user engagement와 user satisfaction에 대해 예측함

Conflicting objectives:

\- videos that users will rate highly - videos that users they will share - videos that user will watch

videos that users they will share랑 videos that user will watch 일치하지 않을 수 있음

user는 추천되어서 그냥 봤을 수 있음 -> user의 선호가 아닐 수 있다는 것 -> 데이터 편견 존재 가능

Input: what the user is currently watching (query video) + user features

1. Generate a few hundred of candidate videos 
2. Rank candidates 
3. Serve top ranking videos to the user

Candidate video: pool videos from multiple candidate generation algorithms 

\- matching topics of query video - videos most frequently watched with query video - And others

**Ranking candiadate video: central topic of this paper**

Input: query video, candidate video, user & context features

Model output: engagement and satisfaction with candidate video

**Engagement:** - binary classification tasks like clicks - regression tasks for tasks related to time spent

**Satisfaction:** - binary classification tasks like clicking “like” - regression tasks for tasks such as rating

Weighted combination of engagement & satisfaction predictions -> ranking score score weights manually tuned

#### Architecture 

Baseline: **mulit-head architecture**

![image](https://user-images.githubusercontent.com/54271228/139352928-157a076e-0878-4e23-adcf-bf4b8e374cc4.png)

문제) task간의 상관관계가 낮으면 학습 절차에서 문제가 생길 수 있음(Negative Transfer)

Instead: use a form of soft-parameter sharing **“Multi-gate Mixture-of-Experts (MMoE)"**

-> soft parameter sharing

![image](https://user-images.githubusercontent.com/54271228/139353083-90bccb72-d9fa-4280-a5fe-7958e6c81b76.png)

네트워크의 다른 부분들이 특화(specialize) 되게 만드는 것: expert neural networks fi(x)

그림의 예시는 두개의 expert network가 있음

1. input x과 주어진 task k에 대해서 어떤 network를 사용할지 결정하기

![image](https://user-images.githubusercontent.com/54271228/139353292-a29260e6-7bda-40a3-93ee-042b36ae4a53.png)

​		-> 가중치와 input에 linear combination을 수행하고 softmax를 취함

2. 선택된 expert에서 feature 계산하기

![image](https://user-images.githubusercontent.com/54271228/139353486-e23775fa-90fe-43b1-8fb8-bfff14552c2a.png)

​		expert neural networks fi(x)에 gating fuction 곱해서 output 내기

3. output 계산하기

   ![image](https://user-images.githubusercontent.com/54271228/139353583-09a3c20a-6132-451e-9761-436bddccce32.png)



Shared bottom layer가 multiple expert로 나누어져서 결국 보면, single model로 오는 형태

#### Set-up

- Implementation in TensorFlow, TPUs 

- Train in temporal order, running training continuously to consume newly arriving data 
- Offline AUC & squared error metrics 
- Online A/B testing in comparison to production system 
- live metrics based on time spent, survey responses, rate of dismissals 
- Model computational efficiency matters -> 그래서 multitask learning로 진행한 것

#### Results

![image](https://user-images.githubusercontent.com/54271228/139353958-67e63dfa-0f49-4f22-8a36-11b4c5e55759.png)

Found 20% chance of gating polarization(one expert만 사용하거나 아예 expert사용 안함) during distributed training -> use drop-out on experts



## Meta Learning

1. Mechanistic view: 기저의 매커니즘 이해하는데 도움이 됨

   ➢ Deep neural network model that can read in an entire dataset and make predictions for new datapoints 

   ➢ Training this network uses a meta-dataset, which itself consists of many datasets, each for a different task 

   ➢ This view makes it easier to implement metalearning algorithms

2. Probabilistic view: 직관적으로 컨셉 이해하는 데 도움이 됨

   ➢ Extract prior information from a set of (metatraining) tasks that allows efficient learning of new tasks 

   ➢ Learning a new task uses this prior and (small) training set to infer most likely posterior parameters 

   ➢ This view makes it easier to understand metalearning algorithms

-강의에서 Probabilistic view로 먼저 설명하고 Mechanistic view 설명할 예정

#### Problem definitions

![image](https://user-images.githubusercontent.com/54271228/139357216-21e501eb-db2c-44f3-a50a-e2b5f786dadb.png)

maximize likelihood problem: 주어진 데이터에서 파라미터의 likelihood를 maximize

=주어진 파라미터에서 데이터의 likelihood를 maximize + 파라미터에 대한 marginal probability maximize

**문제점**: 만약에 데이터가 작으면, regularizer가 있어도 overfit의 가능성 존재

지도학습에서 additional data를 포함시킬 수 있을까?

-처음부터 다시 학습하는 것이 아니라 이전의 배웠던거를 사용해서



![image](https://user-images.githubusercontent.com/54271228/139357763-b92e568e-2e2d-43c1-846a-bf5b75ce00ef.png)

예를 들어 few-shot classification task를 수행할 때, 

주어진 다섯개의 이미지가 있고, 추가로 새로운 이미지에 대해 주어진 다섯개의 범주에 따라 분류를 수행하고 싶을 때, 처음부터 다시 학습하게 되면 overfit이 일어나거나 regularizer가 너무 강력하면 아무것도 못함

만약 다른 이미지 범주에 대한 데이터 (Meta training data)가 존재하면, 다섯개의 예시로 분류를 효과적 분류가능



##### 만약 우리가 D_meta-train (previous experience)를 계속 고수하고 싶지 않을 때 어떻게 하는가?

그래서 meta-training data를 set of parameter로 

= meta-parameters θ : whatever we need to know about D_meta-train to solve new tasks quickly



![image](https://user-images.githubusercontent.com/54271228/139358470-6bc74cba-011a-443f-b652-cfee3f58dfed.png)

meta-train dataset과 task-specific parameter가 독립적인 관계있다고 가정함

마지막에서 두번째 수식보면, 일단 처음에 meta-train 데이터에 대한 파라미터를 측정하고 이 파라미터와 데이터를 사용해서 새로운 task에 대해 새로운 파라미터를 학습할 수 있는 적용 방법(adaptation)임

θ*는 주어진 meta-traing data에 log p의 argmax임 =**이게 meta-learning problem**

![image](https://user-images.githubusercontent.com/54271228/139358960-b87793d5-3204-4679-bc80-e2085496f700.png)

##### 어떻게 adaptation할 것인가?

![image](https://user-images.githubusercontent.com/54271228/139359140-f190e5eb-a78c-4ae5-a867-ed752d695f24.png)

parameter Φ 를 구해하는데 neral network를 통해서 구함

##### 그렇다면 어떻게 학습시킬 것인가?

θ를 구해야하는데, training time이랑 test-time을 매칭

“our training procedure is based on a simple machine learning principle: test and train conditions must match”

(meta) test time:  효과적 Φ 예측 

training time: 효과적 θ 예측

test time 학습시키고 싶으면, meta-training하는 동안에 학습시키는 것이 meta-learning의 기본 원리

-> learning how to learn

meta-training time: test data point에 대해 예측하기 위해서 학습함

![image](https://user-images.githubusercontent.com/54271228/139359720-a8e77dc7-7c8a-4f24-8fcb-daab7143beea.png)

그렇다면 저 test data point는 어디서 나오는 것일까?

-> 어떻게 새로운 data point에 대해 예측을 하기 위해서 최적화 시킬 수 있을까?

Reserve a test set for each task!

trainig 데이터가 있으면 그거에 대응하는 test set도 따로 있어야함

![image](https://user-images.githubusercontent.com/54271228/139361574-e78065e1-b824-48c9-8e9b-e2a9cbb6bf19.png)

1. training, test dataset이 존재해야함
2. training datset이 K data points와 대응

3. test datset이 new set of K data points와 대응

![image](https://user-images.githubusercontent.com/54271228/139361537-a56440d8-9ce7-4f67-b94a-7984b7860d41.png)



#### Some meta-learning terminology

![image](https://user-images.githubusercontent.com/54271228/139360505-ea9d76a8-875d-4098-93e8-f4ea65741bf4.png)

training data = support set

test data = query

K shot learning 일때, K가 datapoint의 수

![image](https://user-images.githubusercontent.com/54271228/139360479-5ae0984b-141f-4faa-9a4a-056dac016856.png)

#### Closely related problem settings

##### Multi-task learning

: learn model with single parameters θ* that solves multiple tasks θ* 

그리고, 새로운 task에 일반화하는 것은 관심없음

multi-task learning은 meta-learning의 전제가 됨

->training task를 해결 못하면 새로운 task도 효과적으로 해결 못할 것이기 때문

##### Hyperparameter optimization & auto-ML

θ: hyperparameters  Φ: network weights

architectere search θ: architectere  Φ: network weights