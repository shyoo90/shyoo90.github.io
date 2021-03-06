---
title: "우울증 분석기"
date: 2019-10-30
tags: [machine learning, data science, python]
excerpt: "Machine Learning, RNN, Data Science"
---
아래와 같이 글을 통해 글 작성자의 우울증 확률을 분석해주는 모델을 만들어 보았다.
<img src="{{ site.url }}{{ site.baseurl }}/images/depression/12.jpg" alt="linearly separable data">
## [데이터 수집](https://shyoo90.github.io/crawling/)

**네이버 카페** 크롤링을 통해 데이터를 수집하였다.
우울증 카페와 일반 카페에서 약 **3만명** 이 작성한 **10만문장** 의 글들을 크롤링

## [전처리](https://shyoo90.github.io/preprocessing/)

광고, 부적합한 글들을 키워드를 통해 제거, 아웃라이어 제거, 형태소 토크나이징, 패딩 진행
**binary classification** 을 진행하기 위해 우울증 글을 1, 일반 글을 0 으로 라벨링

## [시각화](https://shyoo90.github.io/eda/)

**KONlpy, Matplotlib, keras tokenizer** 등을 사용하여 데이터를 시각화 하였다.
형태소 단위로 분리하여 우울증 데이터, 일반 데이터에 많이 나타나는 표현들을 추출하였다.


## [모델링](https://shyoo90.github.io/model/)

여러 **RNN계열 모델** 들을 사용하여 정확도가 높은 모델을 정한뒤
**grid search** 를 통해 성능 향상

## [최종](https://shyoo90.github.io/final/)
시각화 단계에서 추출한 표현들을 출력에서 보여주도록 설계

<img src="{{ site.url }}{{ site.baseurl }}/images/depression/2.jpg" alt="linearly separable data">
