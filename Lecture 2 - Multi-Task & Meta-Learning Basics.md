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