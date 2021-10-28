# Lecture 1 - Introduction & Overview

[강의영상](https://www.youtube.com/watch?v=0rZtSwNOTQo&list=PLoROMvodv4rMC6zfYmnD7UG3LVvwaITY5&index=2)

## Introduction

Topics

1. Problem definitions

2. Multi-task learning basics

3. Meta-learning algorithms: black-box approaches, optimizationbased
  meta-learning, metric learning

4. Hierarchical Bayesian models & meta-learning

5. Multi-task RL, goal-conditioned RL, hierarchical RL

6. Meta-reinforcement learning

7. Open problems, invited lectures, research talks

   

   **Emphasis on deep learning and reinforcement learning**

   

## Some of her reseaches

(and why she care multi-task learning and meta-learning)

"How can we enable agents to learn skills in the real world?"

로봇 = real world

왜 로봇인가? 로봇이 지능에 관련된 것을 가르쳐줄 수 있음

faced with the **real world**
must **generalize** across tasks, objects, environments, etc
need some **common sense understanding** to do well
**supervision** can’t be taken for granted



PhD 학위 - 로봇이 어떻게 박스의 상자에 물건을 끼워넣는 것을 학습하는가?

-> 그것뿐만 아니라 다른 task에도 확장 가능

문제

1. 특정 물체로 특정 상자에 특정 물건을 넣는 방법을 배움

   다른 물체나 물건, 상자 등 환경이 바뀌면 적용이 안됐음

2. 로봇이 물건을 치면 인간이 노가다로 다시 같은 환경 만들어줌 (reset)

   처음부터 다시 시작하지 않으면, 더 많은 task 학습을 위한 측정이 어려움 ->비효율적

   -> 굉장히 구체적인 가이드에 따라 작동해서 많은 task에 확장되기가 어려움

   -> 강화학습 알고리즘의 문제점이기도 함

   **specialist: trained for single task learning**

그럼 어떻게 일반화된 시스템을 만들 수 있을까?

-> 인간이 어떻게 배우는가를 생각할 필요가 있음

인간은 바닥을 구르고, 만져보고, 이것저것 다해보면서 학습함

​	**human = generalist**

generalist와 같은 머신러닝 시스템을 만들기 위해서 이러한 방식을 생각해보아야 함

이전의 경험을 통해서 새로운 것을 빠르게 배우고, 복잡한 것을 하기 전에 쉬운 것들을 학습함



## Why should we care about multi-task & meta-learning?

(…beyond the robots and general-purpose ML systems)

1. Standard computer vision (hand-designed features)

![CV](https://user-images.githubusercontent.com/54271228/139269121-0c82c72f-b327-4d3e-876c-c4f60de6a605.png)

​	2. Modern computer vision (end-to-end training) -뉴럴네트워크 통해서 원하는 아웃풋내기

![image](https://user-images.githubusercontent.com/54271228/139269318-47e14d9c-a988-453e-8b9a-fcee194da647.png)

**장점 1. 딥러닝은 HOG, SIFT와 같은 hand engineering features을 사용하지 않고도 unstructured input을 직접적으로 처리할 수 있고 더 넓은 도메인의 문제들을 해결할 수 있음**



**장점 2. 다른 상황들에서도 잘 작동함**

![image](https://user-images.githubusercontent.com/54271228/139270176-1188134d-63eb-446d-be07-48589f8f7820.png)



**장점3 크고 다양한 데이터가 있다면 딥러닝을 통해 넓은 일반화가 가능함**



But, what if you don’t have a large dataset?

medical imaging, robotics, personalized education, medicine, translation for rare languages, recommendations와 같은 분야에서는 큰 데이터셋을 얻기 어려움

->Impractical to learn from scratch for each disease, each robot, each person, each language, each task

-> 그래서 multi-task learning 테크닉이 필요함



What if your data has a long tail?

![image](https://user-images.githubusercontent.com/54271228/139271029-878b753d-2076-47de-b57b-9f2296fcd23f.png)

오른쪽의 small data에 해당할 경우 지도학습이 어려워짐 

자율주행에서 차가 흔하지 않은 상황에서 잘 대처하지 못함



What if you need to quickly learn something?

(new about a new person, for a new task, about a new environment, etc.)

인간은 이걸 되게 잘함

![image](https://user-images.githubusercontent.com/54271228/139271700-6ae86d41-7b24-4636-9d2d-fdd9f0f836f6.png)

Braque와 Cezanne 그린 몇가지 그림을 보고 테스트 데이터가 Braque가 그린것이라고 인간은 쉽게 예측가능함

-> Few-shot learning: 6개의 data points를 보고 예측함

-> 인간은 scratch부터 학습하지 않고 이전의 경험을 통해 예측함



**This is where elements of multi-task learning can come into play.**



## Task

What is a task?

이 강의는 task를 데이터셋과 loss fuction을 사용해서 모델을 만들어내는 경우로 정의함

![image](https://user-images.githubusercontent.com/54271228/139272370-d22de674-15ba-4c60-9965-94e69e009711.png)



Different tasks can vary based on:

- different objects

- different people

- different objectives

- different lighting conditions

- different words

- different languages

  -> 상황이나 물체만 달라진다고 해도 다른 task라고 할 수 있음 

  -> 우리가 생각하는 task는 완전히 다른 것을 different task라고 좁게 생각하겠지만, 조금 넓은 범위에서 생각하여 정의함

  

## Critical Assumption

The Bad News) Different tasks need to share some structures

어떠한 구조도 공유하지 않으면, 그냥 다른 구조로 만들어서 따로 학습시키는게 나음

The Good News) There are many tasks with shared structure!

그림의 task는 다 비슷한 구조를 공유하고 있음 

![image](https://user-images.githubusercontent.com/54271228/139274797-a56149aa-adea-48e6-ba49-03a9c09c35d2.png)

비슷한 구조는 이처럼 명시적으로 드러남

하지만 명시적으로 비슷한 구조가 아닌 것처럼 보이더라도 underlying 구조를 보면 공유할 수도 있음

- The laws of physics underly real data.
- People are all organisms with intentions.
- The rules of English underly English language data.
- Languages all develop for similar purposes.

This leads to far greater structure than random tasks.



## 일단 대충 정의를 보면, 

##### The multitask learning problem

Learn all of the tasks more quickly or more proficiently than learning them independently

다 학습한다음에 그 training task에서 잘 하기

##### The meta-learning problem

Given data/experience on previous tasks, learn a new task more quickly and/or more proficiently

그 training task에서 학습하고, new task에서 잘 하기





## Doesn’t multi-task learning reduce to single-task learning?

Yes, it can!

Aggregating the data across tasks & learning a single model is one approach to multi-task learning.

But, we can often do better!

Exploit the fact that we know that data is coming from different tasks.



## Why should we study deep multi-task & meta-learning now?

These algorithms are continuing to play a fundamental role in machine learning research.

강력한 뉴럴네트워크와 function approximators의 도입, 데이터셋, 더 향상된 컴퓨팅 파워로 이 분야는 과거보다 더 매력적임

1. Multilingual machine translation

   2개의 언어로 100개의 언어를 번역할 수 있음

   (Massively Multilingual Neural Machine Translation, 2019)

2. One-shot imitation learning from humans

3. Multi-domain learning for sim2real transfer

   다른 촉감, 환경 등 다른 도메인을 만들었는데, 하나의 데이터 시뮬레이션으로 학습시켜 물체가 날 수 있게 함 

4. Youtube recommedation -multiple competing ranking objectives



최근 연구들에서 큰 관심을 가지고 있음

Its success will be critical for the democratization of deep learning

-> 더 적은 데이터로 알고리즘을 만들 수 있음

![image](https://user-images.githubusercontent.com/54271228/139278538-111bf8f8-db68-4c83-86cd-dc09cc86b515.png)

