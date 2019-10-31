---
title: "우울증 분석기: 데이터 전처리"
date: 2019-10-30
tags: [machine learning, data science, python]
excerpt: "Machine Learning, RNN, Data Science"
---
selenium 을 사용한 네이버 카페 메모 크롤링

1. [준비 단계](#준비-단계)
2. [일반 카페 글 전처리](#일반-카페-글)
* [하나의 데이터 프레임으로 합치기](#하나의-데이터-프레임으로-합치기)
* [띄어쓰기 제거, 말줄임표 처리](#띄어쓰기-제거,-말줄임표-처리)
* [리스트로 바꾼뒤 중복된 문장 제거](#리스트로-바꾼뒤-중복된-문장-제거)
3. [우울증 카페 글](#우울증-카페-글)
4. [두 데이터 합치기](#두-데이터-합치기)
* [이상치 제거](#이상치-제거)
* [교집합 제거](#교집합-제거)
5. [토크나이저, 인코딩, pad_sequences](#토크나이저,-인코딩,-pad_sequences)

## 준비 단계
필요한 패키지 가져오기
우선 크롤링한 데이터 프레임들을 광고, 이상한글 제거, 합치기, 이상치제거 등의 과정을 거치며 정리한 뒤
지도학습 모델에 맞게 데이터를 전처리하였다.

```python
import pandas as pd, pickle, re
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
```


```python
import numpy as np
```


```python
import matplotlib.pyplot as plt
```

## 일반 카페 글
데이터를 불러왔다. 다른 조원들과 함께 총 13곳에서 약 10만명의 글을 크롤링하였다.
많은 양의 인 만큼 전처리 과정이 오래 걸렸다.

```python
# 나머지 카페 글 불러오기
df1 = pd.read_csv('depression_book.csv') # 책읽기 카페
df2 = pd.read_csv('depression_girl.csv') # 연애 카페
df3 = pd.read_csv('spec_up_2000.csv') # 스펙업
df4 = pd.read_csv('munhwa_2000.csv') # 문화
df5 = pd.read_csv('powder_2000.csv') # 파우더룸
df6 = pd.read_csv('story.csv') # 자기계발
df7 = pd.read_csv('write.csv') # 글쓰기
df8 = pd.read_csv('eat.csv') # 먹방
df10 = pd.read_csv('ebook.csv') # 이북
df11 = pd.read_csv('story_2.csv') # 자기계발2
df12 = pd.read_csv('fun.csv') # 유머카페
df13 = pd.read_csv('brunch.csv') # 브런치
```
### 하나의 데이터 프레임으로 합치기

광고 올리는 아이디 제거하기

```python
# 광고 올리는 아이디 제거
def bad_del(df4):
    bad = []
    for i in list(df4.index):
        if 'http' in df4.content.loc[i]:
            bad.append(df4.name.loc[i])
    for i in df4['name']:
        if i in bad:
            df4 = df4[df4['name']!=i]
    return df4
```

이상한 키워드를 포함하는 열 제거하기

```python
# 키워드 포함 열 제거
def key_del(a,df4):
    bad = []
    for i in list(df4.index):
        if a in df4.content.loc[i]:
            bad.append(i)
    df4 = df4.drop(bad, axis = 0)
    return df4
```


```python
len(df1),len(df2),len(df3),len(df4),len(df5),len(df6),len(df7),len(df8),len(df10),len(df11),len(df12),len(df13)
```




    (1278, 493, 1968, 1988, 1987, 815, 220, 129, 120, 1712, 1477, 672)




```python
# 데이터 프레임 전부 다 합치기
df9 = pd.concat([df1['content'], df2['content'], df3['content'], df4['content'], df11['content'],
                 df5['content'], df6['content'], df7['content'], df8['content'], df10['content'],
                df12['content'], df13['content']],axis = 0)
df9 = pd.DataFrame(df9).reset_index(drop=True)
df9 = df9.rename(columns={0 :'content'})
df9
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>content</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>만두를 먹을수있어서 감사합니다</td>
    </tr>
    <tr>
      <td>1</td>
      <td>처음으로 손을 내밀어봤습니다. 그 손을 잡아주...</td>
    </tr>
    <tr>
      <td>2</td>
      <td>버스를탈수있어서 감사합니다</td>
    </tr>
    <tr>
      <td>3</td>
      <td>삶이 무료해지고 이제 또 즐거운 무언가를 찾고...</td>
    </tr>
    <tr>
      <td>4</td>
      <td>제행이 무상하니, 방일하지 말고 정...</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>12854</td>
      <td>(직장인들은 공감할 수 있을 것 같은데) 요즘 나에게하루 중 가장 길게 느껴지는 시...</td>
    </tr>
    <tr>
      <td>12855</td>
      <td>오랜 시간 동안 네이버 블로그에 여행 관련...</td>
    </tr>
    <tr>
      <td>12856</td>
      <td>토요일 아침. 늦잠을 자는 사람들도,...</td>
    </tr>
    <tr>
      <td>12857</td>
      <td>예전에 지나간 느낌인데 아직...</td>
    </tr>
    <tr>
      <td>12858</td>
      <td>의지가 흐릿해졌지만 그래도 매년 다...</td>
    </tr>
  </tbody>
</table>
<p>12859 rows × 1 columns</p>
</div>


### 띄어쓰기 제거, 말줄임표 처리
우울증 문서에 유독 말줄임표가 많이 등장했기 때문에 말줄임표(.....)를 '말줄임표'라는 str로 마꿔주었다.

```python
#띄어쓰기 제거
def rem(a,df):
    col = list(df.columns)
    for i in col:
        df[i] = df[i].apply(lambda x: re.sub(a,'',x))
    return df
rem(' ',df9)
```






```python
# 말줄임표 처리

list_dot = ['.'*i for i in range(12, 1, -1)]

def short_1(a,df):
    col = list(df.columns)
    for i in col:
        df[i] = df[i].apply(lambda x: x.replace(a,'말줄임표'))
    return df

def short(df):
    for a in list_dot:
        short_1(a,df)
    return df
```


```python
short(df9)
```







```python
#리스트로 바꾸기
content=[df9['content'].loc[i] for i in df9.index]
```


```python
len(content)
```




    12859



### 이상한거 추가로 없애기
이상한게 잘 안없어져서 추가로 없앴다.
```python
# 이상한거 없애기
for i in range(len(content)):
    if '\xa0' in content[i]:
        content[i] = content[i].replace('\xa0','')

for i in range(len(content)):
    if '\n' in content[i]:
        content[i] = content[i].replace('\n','')

for i in range(len(content)):
    if '\t' in content[i]:
        content[i] = content[i].replace('\t','')
```


```python
# 광고 없애기
hmm = ['com/','http','(주)','eft','EFT','①','전화:','<','부탁드립니다', '어플',
       '장소:','시간:','/','-','%','@','팝니다','팔아요','[','(']

for i in hmm:
    for con in content:
        if i in con:
            content.remove(con)
```
### 리스트로 바꾼뒤 중복된 문장 제거

```python
# 중복된 문장 제거
content = set(content)
content = list(content)
len(content)
```




    8833




```python
df9 = pd.DataFrame(content, columns=['content'])

# 0글자 이하 없애기
df9['len']=df9.content.apply(lambda x: len(x))
df9 = df9[df9['len']>0]

df9.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>content</th>
      <th>len</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>문화의날이라그런지인기검색어에...</td>
      <td>82</td>
    </tr>
    <tr>
      <td>2</td>
      <td>고속버스타고올라가다내린휴게소ㅎㅎ...</td>
      <td>38</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1일부터월요일이면.그달이정~말길게느껴...</td>
      <td>48</td>
    </tr>
    <tr>
      <td>4</td>
      <td>오래된집은맛에서실패할확률이적고그세월이주는...</td>
      <td>142</td>
    </tr>
    <tr>
      <td>5</td>
      <td>이시간전남비바람이몰아...</td>
      <td>61</td>
    </tr>
  </tbody>
</table>
</div>




```python
len(df9)
```




    8832




```python
df9['label']=0
```


```python
df9.to_csv('normal.csv',index=False)
```

## 우울증 카페 글
11종류의 글을 크롤링하였고, 우울증 카페, 지식인의 글을 데이터로 사용하였다.

```python
#우울증 카페 글 불러오기
df1 = pd.read_csv('depression_jook.csv')
df2 = pd.read_csv('depression_NLP_2.csv')
df3 = pd.read_csv('depression__memo.csv')
df4 = pd.read_csv('search_naver4.csv')
df5 = pd.read_csv('Dep_cafe_crawling_hk.csv')
df6 = pd.read_csv('search_naver_2.csv')
df7 = pd.read_csv('depression_newlife.csv')
df8 = pd.read_csv('depression_newlifememo.csv')
df10 = pd.read_csv('depression_feelingdep.csv')
df11 = pd.read_csv('depression_depinsomnia.csv')
```
그중 네이버 지식인은 다양한 검색어를 통해 나온 글들을 크롤링하였기 때문에 겹치는 문장들을 제거하였다.

```python
len(set(list(df4['content'])))
```




    1934




```python
len(df1),len(df2),len(df3),len(df4),len(df5),len(df6),len(df7),len(df8)
```




    (1613, 616, 1977, 3463, 23928, 726, 7113, 749)


모든 데이터들을 합쳐주었다.

```python
df0 = pd.concat([df1['content'],df2['content'],df3['content'],df4['content'],df10['content'],df11['content'],
                 df5['content'],df6['content']],axis=0)
df0=df0.dropna()
df0 = pd.DataFrame(df0).reset_index(drop=True)
df0
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>content</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>일어나자마자. 살기 싫은 하루..</td>
    </tr>
    <tr>
      <td>1</td>
      <td>내가 없어져도 해는..</td>
    </tr>
    <tr>
      <td>2</td>
      <td>선택의 후회,,</td>
    </tr>
    <tr>
      <td>3</td>
      <td>오늘도 살아야하네...에휴...</td>
    </tr>
    <tr>
      <td>4</td>
      <td>니가 하고 싶은대로 행동하면난...</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>33807</td>
      <td>이게 우울증인지 뭔지 모르겠어...</td>
    </tr>
    <tr>
      <td>33808</td>
      <td>안녕하세요 24살 여자입니다...</td>
    </tr>
    <tr>
      <td>33809</td>
      <td>제 친구가 있는데요.. ...</td>
    </tr>
    <tr>
      <td>33810</td>
      <td>제가 우울증이 있는데.....</td>
    </tr>
    <tr>
      <td>33811</td>
      <td>잠을 의지대로 잘 못자고 있어요 ...</td>
    </tr>
  </tbody>
</table>
<p>33812 rows × 1 columns</p>
</div>




```python
rem(' ',df0)
short(df0)
```





```python
content=list(df0['content'])
```


```python
# 중복된 문장 제거
content = set(content)
content = list(content)
len(content)
```




    31186


우울증 문서는 지식인 크롤링에서 친구, 가족의 우울증 관련얘기가 많았고 이를 제거하였다.

```python
# 이상한거 없애기
for i in range(len(content)):
    if '\xa0' in content[i]:
        content[i] = content[i].replace('\xa0','')

for i in range(len(content)):
    if '\n' in content[i]:
        content[i] = content[i].replace('\n','')

for i in range(len(content)):
    if '\t' in content[i]:
        content[i] = content[i].replace('\t','')
```


```python
# 광고 없애기
hmm = ['com/','<광고>','http','(주)','엄마','친구','동생','언니','누나','오빠','아빠']

for i in hmm:
    for con in content:
        if i in con:
            content.remove(con)
```


```python
# DataFrame 으로 만들기
df0 = pd.DataFrame(content, columns=['content'])

# 0글자 이하 없애기
df0['len']=df0.content.apply(lambda x: len(x))
df0 = df0[df0['len']>0]

df0.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>content</th>
      <th>len</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>저는이런말을듣고우르르무너졌습니다</td>
      <td>17</td>
    </tr>
    <tr>
      <td>1</td>
      <td>갑자기우울함이몰려든다</td>
      <td>11</td>
    </tr>
    <tr>
      <td>2</td>
      <td>너요즘보기좋더라자꾸</td>
      <td>27</td>
    </tr>
    <tr>
      <td>3</td>
      <td>우울하지도않고불안한건쬐금있다그닥죽고싶지도...</td>
      <td>127</td>
    </tr>
    <tr>
      <td>4</td>
      <td>제가그동안좋아했던...</td>
      <td>360</td>
    </tr>
  </tbody>
</table>
</div>




```python
df0[df0['len']>50]
```



```python
df0['label'] = 1
```


```python
df0.to_csv('depression_sum.csv')
```

## 두 데이터 합치기
두 데이터를 합치기 전에 이상치를 제거하였다.
### 이상치 제거

```python
plt.plot(range(len(df0)),df0['len'].T.sort_values()) #글길이 분포 확인
```

<img src="{{ site.url }}{{ site.baseurl }}/images/depression/p1.png" alt="linearly separable data">



```python
df0.describe() #글길이 분포 확인
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>len</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>26367.000000</td>
      <td>26367.0</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>64.595858</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>std</td>
      <td>102.032947</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>min</td>
      <td>1.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>18.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>33.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>65.000000</td>
      <td>1.0</td>
    </tr>
    <tr>
      <td>max</td>
      <td>2044.000000</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.plot(range(len(df9)),df9['len'].T.sort_values()) #글길이 분포 확인
```




<img src="{{ site.url }}{{ site.baseurl }}/images/depression/p2.png" alt="linearly separable data">



```python
df9.describe() #글길이 분포 확인
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>len</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>count</td>
      <td>8832.000000</td>
      <td>8832.0</td>
    </tr>
    <tr>
      <td>mean</td>
      <td>109.108243</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>std</td>
      <td>239.730596</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>min</td>
      <td>1.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>25%</td>
      <td>23.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>50%</td>
      <td>48.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>75%</td>
      <td>103.000000</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>max</td>
      <td>5589.000000</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>


적당히 200글자 이상을 이상치로 판단하여 제거하였다.

```python
# 200 글자 이상 제거
df9 = df9[df9['len']<200]
df0 = df0[df0['len']<200]
```

### 교집합 제거

```python
# 교집합제거
content_0 = [i for i in df0['content']]
content_9 = [i for i in df9['content']]
content_0 = set(content_0)
content_9 = set(content_9)
content_1 = content_0 & content_9
content_0 = list(content_0-content_1)
content_9 = list(content_9-content_1)
```


```python
len(content_1) # 교집합개수
```




    10




```python
# DataFrame 으로 만들기
df0 = pd.DataFrame(content_0, columns=['content'])

# 0글자 이하 없애기
df0['len']=df0.content.apply(lambda x: len(x))
df0 = df0[df0['len']>0]

df0.head(5)
df0['label'] = 1
```


```python
df0.to_csv('data_depressed.csv', index = False)
```


```python
# DataFrame 으로 만들기
df9 = pd.DataFrame(content_9, columns=['content'])

# 0글자 이하 없애기
df9['len']=df9.content.apply(lambda x: len(x))
df9 = df9[df9['len']>0]

df9.head(5)
df9['label'] = 0
```


```python
df9.to_csv('data_normal.csv', index = False)
```


```python
len(df0),len(df9)
```




    (24684, 7880)




```python
# 일반글(df9)가 더 많기 때문에 같은 양으로 under-sampling
df0 = df0.sample(7880).reset_index(drop=True)
```



```python
df = pd.concat([df0,df9],axis = 0)
df = df.reset_index(drop = True)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>content</th>
      <th>len</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>하나님에대한신앙이최우선이라하지만</td>
      <td>40</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>내인생이가장힘들었던건</td>
      <td>27</td>
      <td>1</td>
    </tr>
    <tr>
      <td>2</td>
      <td>저는집이싫습니다</td>
      <td>8</td>
      <td>1</td>
    </tr>
    <tr>
      <td>3</td>
      <td>힘이없고우울하다</td>
      <td>8</td>
      <td>1</td>
    </tr>
    <tr>
      <td>4</td>
      <td>제가깨있어도1시반까지하는</td>
      <td>49</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.tail(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>content</th>
      <th>len</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>15757</td>
      <td>다른날들도그렇지만오늘은더욱...</td>
      <td>77</td>
      <td>0</td>
    </tr>
    <tr>
      <td>15758</td>
      <td>목요일인데아직</td>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <td>15759</td>
      <td>결혼한남자가회식할때마다...</td>
      <td>79</td>
      <td>0</td>
    </tr>
    <tr>
      <td>15760</td>
      <td>후...워터파크탈의실에서사진찍어대...</td>
      <td>53</td>
      <td>0</td>
    </tr>
    <tr>
      <td>15761</td>
      <td>기분이좋다</td>
      <td>5</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.plot(range(len(df)),df['len'].T.sort_values())
```




<img src="{{ site.url }}{{ site.baseurl }}/images/depression/p3.png" alt="linearly separable data">


## 토크나이저, 인코딩, pad_sequences

우선 데이터를 리스트로 가져왔다.
```python
_input1 = df['content']
_label = df['label']
```

cleaning 함수를 만들어 이모티콘 등을 제거하였다.

```python
cleaning =lambda s: re.sub("[^가-힣a-zA-Z.!?\\s]","",s) # 이모티콘같은거 제거
```

형태소 단위로 토크나이즈 하였다. okt함수를 사용하였다.
```python
# 형태소 단위로 토크나이즈
tokenizer = Okt()
_input1 = [ tokenizer.morphs(cleaning(str(sentence))) for sentence in _input1]
```

feature를 줄이기 위해 feature 분포를 알아본뒤 17814개로 feature를 줄였다.
```python
#feature 줄이기
num_words=17814
```


```python
# 인코딩
keras_tokenizer = Tokenizer(num_words=num_words)
#keras_tokenizer = Tokenizer()
keras_tokenizer.fit_on_texts(_input1)
_input = keras_tokenizer.texts_to_sequences(_input1)

word_dict = keras_tokenizer.word_index # 단어와 인덱스 딕셔너리

max_len = max([len(sentence) for sentence in _input])
```


```python
len(word_dict)
```




    48238




```python
index_word = keras_tokenizer.index_word
```


```python
index_word[2061]
```




    '보이지'




```python
max_len
```




    108




```python
_input = pad_sequences(_input, maxlen=max_len, padding='post')
#_label = [ to_categorical(_l) for _l in _label]
```


```python
plt.plot(_input, 'ro')
plt.show()
```


<img src="{{ site.url }}{{ site.baseurl }}/images/depression/p4.png" alt="linearly separable data">



```python
_input[1]
```




    array([   27,   158,     2,   525, 17813,   291, 11519,   168,  1156,
             768,    84,   110,     0,     0,     0,     0,     0,     0,
               0,     0,     0,     0,     0,     0,     0,     0,     0,
               0,     0,     0,     0,     0,     0,     0,     0,     0,
               0,     0,     0,     0,     0,     0,     0,     0,     0,
               0,     0,     0,     0,     0,     0,     0,     0,     0,
               0,     0,     0,     0,     0,     0,     0,     0,     0,
               0,     0,     0,     0,     0,     0,     0,     0,     0,
               0,     0,     0,     0,     0,     0,     0,     0,     0,
               0,     0,     0,     0,     0,     0,     0,     0,     0,
               0,     0,     0,     0,     0,     0,     0,     0,     0,
               0,     0,     0,     0,     0,     0,     0,     0,     0])




```python
_label[:10]
```




    0    1
    1    1
    2    1
    3    1
    4    1
    5    1
    6    1
    7    1
    8    1
    9    1
    Name: label, dtype: int64




```python
#모델에 사용할 변수들 피클 형태로 저장
data = [max_len, num_words+1,keras_tokenizer,_input, _label]
with open('input_data.pickle', 'wb') as handle:
    pickle.dump(data, handle,protocol = pickle.HIGHEST_PROTOCOL)
```


```python
#토크나이저 저장
data = [keras_tokenizer, max_len]
with open('word_dict.pickle', 'wb') as handle:
    pickle.dump(data, handle,protocol = pickle.HIGHEST_PROTOCOL)
```
