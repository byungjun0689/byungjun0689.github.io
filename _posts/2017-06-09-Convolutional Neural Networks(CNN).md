---
layout: post
title: "Convolutional Neural Network(CNN)"
date: 2017-06-09 11:34:26
img: cnn.png
description: 'CNN'
main-class: 'CNN'
color: '#B31917'
tags:
- Python
- Nerural Network
- Convolutional Neural Network
---

 - 첫 시작은 고양이 실험에서 시작되었다.
 - 각각 그림의 부분에 반응하는 뉴런이 있었다 라는 부분에서 착안되었다.

![img3](/src/0609/CNN/3.PNG)

### 자동차 그림 인식
 - 아래의 자동차를 인식하기 위해서 자동차 그림을 아래와 같이 짤라 각각의 데이터로 입력하게 된다. '
 - 이러한 Layer 층을 Convolution Layer라고 한다.
 - 그래서 이 Network이름이 Convolution Neural Networks라고한다.
 - 중간에 ReLU층을 넣고 다시 Convolution layer + POOLing  + 마지막으로 Fully Connected Neural Network을 연결하여 결과를 출력하는 Networks를 구성할 수 있다.

![img4](/src/0609/CNN/4.PNG)

## 시작부터 해서 Network를 이해해 나가보자.
### 1. Start with an image (width x hight x depth)
 - 전체 이미지를 한꺼번에 처리하는 것이 아니라 아래와 같이 잘라서 사용해서 처리 하겠다라고 한다.
![img5](/src/0609/CNN/5.PNG)
  - 여기서 처리 하겠다 라는 개념의 용어는 $Filter$라고 한다.
  - 5x5x3 filter
  - 여기서 마지막 3은 원본 그림의 색, 즉 RGB 3가지 색으로 구성되었으므로 3이 되고 5x5의 경우는 사용자가 정의할 수 있다.
![img5](/src/0609/CNN/6.PNG)

- Filter 를 궁극적으로 1개의 값을 출력시킨다.
- 5x5x3의 값을 어떻게 1개의 값으로 만들 수 있을까?
 - $Wx+b$ 를 사용한다.
 - 이것을 간단하게 5x5x3이 아니라 5개만 있다고 하면 $x_1,x_2,x_3,x_4,x_5$가 있다면 Multi Features Regression과 동일하다.
 - 아래의 식과 같은 형태로
![img5](/src/0609/CNN/7.PNG)

 - 변하지 않는 W(같은 필터이기 때문에)로 옆으로 넘기면서 각각 값들을 가지고오고 아래로 가서도 값을 가지고 오게 된다면 같은 필터로 전체 이미지를 훑고 각각의 점들을 한 값들을 가지고 오게 된다.
 - 이러한 과정을 거치게 되면 몇개의 값 / 점을 모을 수 있을까?
![img8](/src/0609/CNN/8.PNG)
![img8](/src/0609/CNN/9.PNG)

### Number of output
 - 옆으로 몇칸씩간다 하는 것을 Stride 라고 한다.
 - Stride 1: 한칸식 옆으로 Stride 2 : 두칸씩 옆으로 이동.
![img10](/src/0609/CNN/10.PNG)
![img11](/src/0609/CNN/11.PNG)

#### 공식
 - 큰이미지가 자꾸 작아진다.
  - 작아질 수록 정보를 잃어버린다는 것이다.
![img12](/src/0609/CNN/12.PNG)

### Padding
 - CNN을 사용할 경우 보통 Padding을 사용한다.
 - 2가지 이유가 있다.
  1. 그림이 작아진다는 것을 막아준다.
  2. 이 부분이 모서리다 라고하는 것을 알려준다.
![img13](/src/0609/CNN/13.PNG)
![img14](/src/0609/CNN/14.PNG)

## Swiping the entire image
 - 아래와 같이 Filter 1로 수행하게 되면 노란색 판과 같은 크기의 데이터로 출력되게 된다.
![img15](/src/0609/CNN/15.PNG)
 - 다른 필터를 한개 더 만든다. 다른 Set의 Weight을 만들어 (Filter2를 만들게 된다.)
![img16](/src/0609/CNN/16.PNG)
 - 2개만 하는 것이 아니라 Filter를 6개로 해서 한다면 나오는 값이 각각 6개가 다르게 나오게 될 것이다.
 - @ x @ x 6의 DataSet이 될 것이다.
 - @ 는 Filter Size와 그림의 Size로 결정 될 것이다.
 - 밑의 예제는 28x28x6이다.
![img17](/src/0609/CNN/17.PNG)

 - 이러한 Flow를 연속해서 진행하게 된다면 Convolution layer가 연속해서 등장하게 될 것이다.
![img18](/src/0609/CNN/18.PNG)
 - 다른 NN과 마찬가지고 각각의 weight을 학습해야 한다. 방법은 동일하다.
 - Weight의 갯수는 ? 5x5x3x6 + 5x5x6x10 과 같이 구할 수 있다.
![img19](/src/0609/CNN/19.PNG)

 - Lec 11 추가해야함.

## Maxpooling & Others

## pooling (Sampling)
 - Sampling이라고 보면 된다.
 - Convolution Layer에서 출력된 값중에 한 판만 때낸다. 아래와같이
![img20](/src/0609/CNN/20.PNG)
![img21](/src/0609/CNN/21.PNG)
![img22](/src/0609/CNN/22.PNG)

### Max Pooling (Sampling)
 - 4x4 그림에서 2x2 filter 를 사용한다면 그리고 Strider가 2라면 2칸씩 이동하겠다 라는 뜻이다.
 - 평균을 내자/가장큰놈을 꺼내자/가장 작은놈을 꺼내자 하는 다양한방법이 있다.
 - 하지만 가장 많이 사용되는 것은 가장 큰 값을 가지고 나오는 것이다.
![img23](/src/0609/CNN/23.PNG)

 - 어떠한 순서대로 할것인지는 우리 마음이다. 몇번을 Convolution을 하고 Pooling을하고 이런건 내맘이다.
 - 이 값들 전체가 있을 것읻가. 3x3x10이라는 Convolution Layer가 나왔다면 원하는 깊이의 일반적인 Neural Network에 넣어서 마지막에 Softmax를 이용하여 몇개중의 Label중에 선택할 수 있도록 만들면 된다.
![img24](/src/0609/CNN/24.PNG)

<a href="http://cs.stanford.edu/people/karpathy/convnetjs/demo/cifar10.html"> 참고용 CNN </a>

## Case Study
 - 사람들이 어떻게 응용 했는지
![img25](/src/0609/CNN/25.PNG)

![img26](/src/0609/CNN/26.PNG)
![img27](/src/0609/CNN/27.PNG)
![img28](/src/0609/CNN/28.PNG)
![img29](/src/0609/CNN/29.PNG)

<a href="https://github.com/byungjun0689/DataScience/blob/master/4.%20KMU/Third%20Semester/04.%20DataScience%20Practice/03.%20DeepLearning/2.%20CNN.ipynb"> 실습(KMU) </a>

<a href="https://github.com/byungjun0689/DataScience/blob/master/4.%20KMU/Third%20Semester/04.%20DataScience%20Practice/03.%20DeepLearning/5.%20CNN%20(Color%20Verison).ipynb"> 실습2(KMU-Color버젼)</a>
