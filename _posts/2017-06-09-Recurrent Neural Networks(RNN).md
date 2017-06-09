---
layout: post
title: "Recurrent Neural Networks(RNN)"
date: 2017-06-09 11:34:26
img: rnn.png
description: 'Recurrent Neural networks'
main-class: 'RNN'
color: '#B31917'
tags:
- Python
- Nerural Network
- Recurrent Neural networks
---

## 순환 신경망.
 - 참고
  - <a href="https://brunch.co.kr/@chris-song/9"> Brunch Chris 송호연 </a>
  - 유재명 교수님. PPT
 - 우리가 문맥을 이해하거나 책을 읽을 때 바로 전의 문맥에 맞게 단어를 이해하고 내용을 생각한다.
 - 모든 내용을 한번에 이해하는 것이 아니라 흐름(Flow)를 가지고 이해를 한다.
 - 예전의 Neural Network에서는 ㅅ구현할 수 없었다. 이러한 것이 중요한 단서가 된 알고리즘이다.

### Sequence Data
 - 하나의 단어를 가지고 이해한다고 해서 전체를 이해할 수 있는 것이 아니다.
 - 이전의 단어들과 + 지금의 단어를 가지고 이해할 수 있다. (Time Series)
 - NN / CNN 은 할 수 없다.

## 1. RNN의 구조


```python
from IPython.display import Image
```

 $X_t->A->h_t$ 순으로 값이 출력된다 그리고 $h_t$에 결과로 나온 값이 <br>다음번 $X_{t+1}->A->h_{t+1}$에 영향을 준다.
 - 즉, $X_{t+1} + h_t -> A -> h_{t+1}$


```python
Image(filename="/src/0609/CNN/1.PNG", width=300)
```




![png](/src/0609/CNN/1.PNG)



위의 알고리즘과 수식을 풀어서 설명하게 되면 아래와 같은 그림이 나오게 된다.
- 현재의 Status 가 다음의 Stauts에 영향을 끼친다.


```python
Image(filename="/src/0609/CNN/2.PNG", width=600)
```




![png](/src/0609/CNN/2.PNG)



 - 아래와 같이 수식을 하나를 사용하는 이유는 여러개의 RNN이 존재하는 것 같지만 모든 연산이 동일하기 때문에 아래와 같이 표현한다.


```python
Image(filename="/src/0609/RNN/1.PNG")
```




![png](/src/0609/RNN/1.PNG)



## (Vanila) Recurrent Neural Network
 - 가장 기초가 되는 RNN의 연산 방법
 - $h_t$ = $f_w(h_{t-1},x_t)$ => $WX$
 - y가 몇개로 나올 것인가 하는거는 $W_{hy}$ 의 사이즈에 따라서 결정이 된다.


```python
Image(filename="/src/0609/RNN/2.PNG")
```




![png](/src/0609/RNN/2.PNG)




```python
Image(filename="/src/0609/RNN/3.PNG") # 전체동일한 Weight값을 가진다. W_hh, W_xh, W_hy
```




![png](/src/0609/RNN/3.PNG)



 - Hello 라는 글자를 예측하고 싶다.
  - H가 들어갔을때 e를 예측하는 모델
 - https://www.youtube.com/watch?v=-SHPG_KMUkQ&feature=youtu.be
  - 10:03


```python
Image(filename="/src/0609/RNN/4.PNG")
```




![png](/src/0609/RNN/4.PNG)



### One-hot Encoding
 - [h,e,l,o]
 - h = [1,0,0,0]
 - e = [0,1,0,0]
 와 같이 표현.


```python
Image(filename="/src/0609/RNN/5.PNG")
```




![png](/src/0609/RNN/5.PNG)



 - 첫번째 Inputdata에서는 $h_{t-1}$의 데이터가 없으므로 0으로 준다.


```python
Image(filename="/src/0609/RNN/6.PNG")
```




![png](/src/0609/RNN/6.PNG)



 - h x $W_{xh}$ 해서 위의 값이 나왔다고 치자.


```python
Image(filename="/src/0609/RNN/7.PNG")
```




![png](/src/0609/RNN/7.PNG)



 - RNN은 이전의 값을 기억한다고 할 수 있다.


```python
Image(filename="/src/0609/RNN/8.PNG")
```




![png](/src/0609/RNN/8.PNG)



 - 맨 마지막에 Softmax를 취하게 되면 각 초록의 데이터들이 출력되게 된다.


```python
Image(filename="/src/0609/RNN/9.PNG")
```




![png](/src/0609/RNN/9.PNG)



## 생성 가능한 모델
 - Language Modeling
 - Speech Recognition
 - Machine Translation
 - Conversation Modeling / Question Answering
 - Image/Video Captioning
 - Image/Music/Dance Generation


```python
Image(filename="/src/0609/RNN/10.PNG")
```




![png](/src/0609/RNN/10.PNG)



# LSTM
 - RNN이 깊어지고 넓어지게 되면 학습에 어려움이 있다.
 - 상황에 따라 적용이 될 수도 있고 안될 수도 있다.

          예를 들어 "I grew up in France... I speak fluent French.(나는 프랑스에서 자라났어... 나는 프랑스어를 유창하게 해)"라는 문장에서 마지막 단어 French(프랑스어)를 예측하는 문제를 생각해보겠습니다. 최근 정보를 기반으로 예측 모델은 다음 단어가 아마도 언어의 한 종류라고 예측될 것입니다. 그렇다면, 이 예측 모델은 "I grew up in France(나는 프랑스에서 자라났다)"에서 프랑스라는 문맥이 필요하게 됩니다. 실제로 "I grew up in France(프랑스에서 자라났다)"는 표현과 "I speak fluent *** (나는 *** 언어를 유창하게 한다)"라는 표현의 위치가 멀어지는 문제는 아주 빈번하게 발생합니다.


 - 위와 같이 해당 연결되는 부분에서 이어질 수 없는 상황이 발생되기도 한다.


```python
Image(filename="/src/0609/RNN/13.PNG")
```




![png](/src/0609/RNN/13.PNG)



## 이를 해결 할 수 있는 Network
### LSTM(Long Short Term Memory Networks)

 - 싱글 레이어를 가지고 있는 반복되는 표준 RNN 모듈


```python
Image(filename="/src/0609/RNN/11.PNG")
```




![png](/src/0609/RNN/11.PNG)



 - LSTM에 들어있는 4개의 상호작용하는 레이어가 반복되는 모듈
 - 해당 데이터가 미래에 결과에 영향을 줄것인지 안줄 것인지 구분하려는 것이다,


```python
Image(filename="/src/0609/RNN/12.PNG")
```




![png](/src/0609/RNN/12.PNG)



 - LSTMs의 가장 중요되는 핵심은 **셀 스테이트(Cell State)**이다.
 - 다이어그램의 상단에 있는 수평선.
 - 이것을 통해 정보는 큰 변함 없이 계속적으로 다음 단계에 전달 된다.


```python
Image(filename="/src/0609/RNN/14.PNG")
```




![png](/src/0609/RNN/14.PNG)



위의 셀 스테이트에 게이트라는 요소를 활용하여 정보를 사용할 것인지 안할 것인지 구분할 수 있도록 한다. 아래 투명하게 되어있는 그림을 말하는 것.
 - Sigmoid Layer : Output ( 0 or 1 )
  - 0이 들어오게 되면 해당 부분은 사용할 수 없게 된다. 즉, 미래 결과에 영향을 줄 수 없게 된다.


```python
Image(filename="/src/0609/RNN/15.PNG")
```




![png](/src/0609/RNN/15.PNG)



### Step by Step LSTM 따라가기
#### Forget Gate
     1. 어떠한 정보를 버릴 것인지 선택할 것인지 하는 과정. (Forget Gate) - Sigmoid 게이트


```python
Image(filename="/src/0609/RNN/16.PNG")
```




![png](/src/0609/RNN/16.PNG)



#### Input gate
     2. 새로운 정보가 셀 스테이트에 저장될지를 결정하는 단계.
     - 1. Input gate layer 라고 불리는 Sigmoid Layer:어떤 값을 업데이트 할지 결정.
            다음, tanh 레이어는 다음으로 넘어갈 값을 결정한다.
     - 2. 이 두가지 값을 합쳐서 그 다음 스테이트에 영향을 주게 된다.


```python
Image(filename="/src/0609/RNN/17.PNG")
```




![png](/src/0609/RNN/17.PNG)



#### Forget + Input


```python
Image(filename="/src/0609/RNN/18.PNG")
```




![png](/src/0609/RNN/18.PNG)



#### Output gate
    3. 마지막으로 어떤 출력값을 출력할지 결정해야한다.
    - 1. 우리는 어떤 값을 출력할 지 결정하는 Sigmoid Layer를 돌리게 된다.
    - 2. 다음, 셀 스테이트 값을 tanh 함수를 거쳐 -1 에서 1 사이 값을 뽑아낸다.
    - 3. 그 다음, 2번을 1번과 곱할 것이다. 우리는 우리가 원하는 값만 결과 값으로 반영하게 된다.


```python
Image(filename="/src/0609/RNN/19.PNG")
```




![png](/src/0609/RNN/19.PNG)



## Sequence to Sequence
 - 통 변역을 하게 될때는 바로바로해서 이해하기가 힘들다
 - how are you에서 how만 듣고 안녕하세요 라는 뜻을 유추하기 힘들다.
 - 그렇기에 순서 전체를 학습하고 쭉 출력하는 순서로 진행
 - 챗봇이랑 번역이랑 동일한 Architecture지만 답변만 다르게 나온다.


```python
Image(filename="/src/0609/RNN/20.PNG")
```




![png](/src/0609/RNN/20.PNG)



<a href="https://github.com/byungjun0689/DataScience/blob/master/4.%20KMU/Third%20Semester/04.%20DataScience%20Practice/03.%20DeepLearning/4.%20RNN.ipynb">실습</a>
