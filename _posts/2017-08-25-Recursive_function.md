---
layout: post
title: "재귀함수 Recursive function"
date: 2017-08-26 10:00:26
img: algorithm.jpg
description: 'Recursive function'
main-class: 'Python'
---

# 재귀함수 (Recursive function)

## 재귀
------------

- 할머니의 비밀상자가 존재한다. 상자를 열어보니 또 안에 많은 상자가 존재한다. 그 중 하나에 키가 있다고 한다면...

- 첫번째 (While)

  - 내부를 확인할 상자를 쌓아놓는다.

  - 상자를 하나 집어서 내부를 살핀다.

  - 만약 안에 상자가 있다면 꺼내어 나중에 확인할 상자 더미에 놓는다.

  - 만약 열쇠가 있으면 작업 종료.

  - 반복한다.

    ```Python
    def look_for_key(main_box):
        pile = main_box.make_a_pile_to_look_through()
        while pile in not empty:
            box = pile.grab_a_box():
            for item in box:
                if item.is_a_box():
                    pile.append(item)
                else:
                    print("열쇠를 찾았다.")
    ```

- 두번째 (Recursive function)

  - 상자 안을 확인한다.

  - 만약 상자를 발견하면 1단계로 간다.

  - 만약 열쇠를 발견하면 작업 종료.

    ```python
    def look_for_key(box):
        for item in box:
            if item.is_a_box():
                look_for_key(item) # 반복수행
            else:
                print("열쇠를 찾았다.")
    ```




### 기본 단계와 재귀 단계
----------

- 재귀 함수의 경우 자기 자신을 호출하기 때문에 무한 반복 에러를 범하기 쉽다.

  ```python
  def countdown(n):
      print(n)
      countdown(n-1)
  countdown(3)
  ## 결과 3 2 1 0 -1 -2 ....
  ```

- 재귀 함수를 만들때는 언제 멈출지 알려줘야한다.

- 기본 단계와 재귀 단계라는 두부분으로 나누어져 있다.

  ```python
  def countdown(n):
      print(n)
      if n <= 1:  # 기본단계
          return
      else:       # 재귀단계
          countdown(n-1)
  ```




## 스택 (Stack)
-------------

- 프로그램에서 아주 중요한 개념 중 하나.
- TODO List 를 생각해보면 이전 포스팅에서 배웠던 [배열]과 [리스트]에 적용 한다고 생각해보면 될 것이다. 
  - 할 일을 넣고 빼고, PUSH, POP 기능을 수행하는 것.



### 호출스택
--------------

- 컴퓨터는 호출 스택이라는 불리는 스택을 사용한다. 아래의 코드를 보면 이해가 빠르다.

```python
def greet(name):
  	print("hello " + name)
    greet2(name)
    print("getting ready to say good bye!")
    bye()
    
def greet2(name):
  	print("how are you, " + name)

def bye():
  	print("ok bye!")
    
    
# 수행 순서(메모리 입력순서) 
# push(greet("mark")) -> "hello mark" -> push(greet2("mark")) -> "how are you, mark"
# -> pop(greet2("mark")) -> greet실행으로 복귀 -> "getting ready to say good bye" 
# -> push(bye()) -> "ok bye" -> pop(bye()) -> greet 으로 복귀 -> pop(greet("mark"))
```

- 여러 개의 함수를 호출하면서 함수에 사용되는 변수를 저장하는 스택을 호출 스택이라고 한다.



### 재귀 함수에서 호출 스택 사용
--------------

- 재귀함수에서도 호출 스택을 사용한다. 
- 예) 팩토리얼 함수.

```python
def factorial(x):
  	if x == 1:
      	return 1
    else:
      	return x * factorial(x-1)  
# factorial(3) => 3 -> 3 x 2 -> 3 x 2 x 1 Stack에 쌓이고 연산
```

