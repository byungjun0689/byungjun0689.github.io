---
layout: post
title: "Select sort & Array, Linked List (with Python)"
date: 2017-08-23 21:30:26
img: algorithm.jpg
description: 'Select sort'
main-class: 'Python'
---

# 그림으로 개념을 이해하는 알고리즘



## 선택정렬(Select Sort)

- 배열(Array) 와 연결리스트(Linked List)

- 선택정렬 (Select Sort) 



### 1. 메모리가 동작하는 방법

예를 들어 물건을 보관하는 공간이 필요하다고 한다면 1개의 공간에 1개의 물건만 보관이 가능하다. 그렇다면 물건의 갯수만큼 공간의 갯수가 필요할 것이다.

메모리도 마찬가지이다. 많은 공간 => 메모리 주소, 무엇인가를 저장할때마다 주소(공간)를 요청

- 여러 개의 원소를 저장해야한다면 배열과 리스트라는 두 가지 방법 중 하나를 사용



### 2. 배열과 연결리스트 

- 배열 : 일정 공간을 미리 할당받아 연속적으로 사용하는 방법 (영화관의 자리를 미리 예매해놓고 시청하는 방법, 누군가가 오지 않는다면 그대로 금액을 지불해야 한다. 그리고 연속해서 앉고 싶어하는데 새로운 친구가 온다면 자리를 옮겨야 한다.) 
  - 예약 인원보다 적게 온다면 메모리 낭비. 
  - 많이 온다면 위치를 옮겨야 한다.
- 연결리스트 : 데이터와 함께 다음 데이터의 주소를 동시에 가지고 있다. 

#### 배열과 연결리스트의 차이
----------

- 배열은 연속된 공간을 가지고 있기 때문에 내 다음 10번째 또는 몇번째 데이터의 위치를 알 수 있다.

- 연결리스트의 경우 쭉 이어서 따라가봐야 한다는 단점이 있다. 

- 각 조건에 맞게 필요한 것을 이용하면 된다. 어느 것이 좋다는 것은 없다.

- 배열과 리스트의 읽기와 쓰기 시간표

  |      | 배열(Array) | 리스트(Linked List) |
  | :--: | :-------: | :--------------: |
  |  읽기  |   O(1)    |       O(n)       |
  |  쓰기  |   O(n)    |       O(1)       |
  |  삭제  |   O(n)    |       O(1)       |

- 배열 : 순차 접근, 임의 접근 모두 가능

- 리스트 : 순차 접근 밖에 할 수 없다.

### 3. 선택정렬(Select sort)

| 파일명  |  A   |  B   |  C   |  D   |  E   |  F   |  G   |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| 반복횟수 | 156  | 141  |  35  |  94  |  88  |  61  | 111  |

반복을 가장 많이 한 횟수의 파일부터 정렬하여 가장 많이 반복한 파일을 찾고싶다면?

1. 모든 데이터를 탐색하여 가장 많은 데이터부터 새로운 리스트를 생성.
   - O(n x n) => O(n^2) 
   - 각 횟수를 진행할 때마다 전체 데이터를 확인해야 한다.
2. 선택 정렬은 깔끔한 알고리즘이지만 빠르지는 않다. 퀵 정렬은 O(nlogn) 시간 밖에 안걸린다. 



### 4. 예제코드 (Python)

```python
# python 3.5
def find_smallest(arr):
    smallest = arr[0]
    smallest_index = 0
    for i in range(1, len(arr)):
        if arr[i] < smallest:
            smallest = arr[i]
            smallest_index = i
    return smallest_index

def select_sort(arr):
    new_arr = []
    for i in range(1, len(arr)):
        smallest = find_smallest(arr)
        new_arr.append(arr.pop(smallest))
    return new_arr

test_list = [5,3,2,7,10,15,1]
print(select_sort(test_list))
```

