---
title: "우울증 분석기: 데이터 전처리"
date: 2019-10-30
tags: [machine learning, data science, python]
excerpt: "Machine Learning, RNN, Data Science"
---
```python
import pandas as pd, pickle, re
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from sklearn.model_selection import train_test_split
```


```python
import numpy as np
```


```python
import matplotlib.pyplot as plt
```

## 일반 카페 글


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


```python
# null값 제거
df1 = df1.dropna(axis=0)
df2 = df2.dropna(axis=0)
df3 = df3.dropna(axis=0)
df4 = df4.dropna(axis=0)
df5 = df5.dropna(axis=0)
df6 = df6.dropna(axis=0)
df7 = df7.dropna(axis=0)
df8 = df8.dropna(axis=0)
df10 = df10.dropna(axis=0)
df11 = df11.dropna(axis=0)
df12 = df12.dropna(axis=0)
df13 = df13.dropna(axis=0)
```


```python
# 데이터 프레임 내 a 없애기
def rem(a,df):
    col = list(df.columns)
    for i in col:
        df[i] = df[i].apply(lambda x: re.sub(a,'',x))
    return df
```


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
      <td>내 맘에 들지 않는 사람을 쳐냈었는데 처음으로 손을 내밀어봤습니다. 그 손을 잡아주...</td>
    </tr>
    <tr>
      <td>2</td>
      <td>버스를탈수있어서 감사합니다</td>
    </tr>
    <tr>
      <td>3</td>
      <td>삶이 무료해지고 이제 또 즐거운 무언가를 찾고 있었습니다. 또 의미없이 시간 보내는...</td>
    </tr>
    <tr>
      <td>4</td>
      <td>붓다의 유언을 읽을 수 있어서 감사했습니다.. “제행이 무상하니, 방일하지 말고 정...</td>
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
      <td>오랜 시간 동안 네이버 블로그에 글을 적어왔다.일본에서 살기 전에는 일본 여행 관련...</td>
    </tr>
    <tr>
      <td>12856</td>
      <td>토요일 아침. 사람들이 주말을 시작하는 풍경은 어떨까? 밀린 늦잠을 자는 사람들도,...</td>
    </tr>
    <tr>
      <td>12857</td>
      <td>벌써 르완다 생활이 1년 하고도 5개월.왠지 1년 차는 예전에 지나간 느낌인데 아직...</td>
    </tr>
    <tr>
      <td>12858</td>
      <td>원래 다이어리 쓰는 것을 좋아했다. 연말이 되면 의지가 흐릿해졌지만 그래도 매년 다...</td>
    </tr>
  </tbody>
</table>
<p>12859 rows × 1 columns</p>
</div>




```python
#띄어쓰기 제거
def rem(a,df):
    col = list(df.columns)
    for i in col:
        df[i] = df[i].apply(lambda x: re.sub(a,'',x))
    return df
rem(' ',df9)
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
      <td>만두를먹을수있어서감사합니다</td>
    </tr>
    <tr>
      <td>1</td>
      <td>내맘에들지않는사람을쳐냈었는데처음으로손을내밀어봤습니다.그손을잡아주어서감사합니다.자존심...</td>
    </tr>
    <tr>
      <td>2</td>
      <td>버스를탈수있어서감사합니다</td>
    </tr>
    <tr>
      <td>3</td>
      <td>삶이무료해지고이제또즐거운무언가를찾고있었습니다.또의미없이시간보내는일을찾을수도있었는데좋...</td>
    </tr>
    <tr>
      <td>4</td>
      <td>붓다의유언을읽을수있어서감사했습니다..“제행이무상하니,방일하지말고정진하라"</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>12854</td>
      <td>(직장인들은공감할수있을것같은데)요즘나에게하루중가장길게느껴지는시간은출퇴근시간이다.매일...</td>
    </tr>
    <tr>
      <td>12855</td>
      <td>오랜시간동안네이버블로그에글을적어왔다.일본에서살기전에는일본여행관련으로적기시작하다가일본...</td>
    </tr>
    <tr>
      <td>12856</td>
      <td>토요일아침.사람들이주말을시작하는풍경은어떨까?밀린늦잠을자는사람들도,벌써하루를준비해나가...</td>
    </tr>
    <tr>
      <td>12857</td>
      <td>벌써르완다생활이1년하고도5개월.왠지1년차는예전에지나간느낌인데아직1년6개월이채되지않았...</td>
    </tr>
    <tr>
      <td>12858</td>
      <td>원래다이어리쓰는것을좋아했다.연말이되면의지가흐릿해졌지만그래도매년다이어를써왔다.근데스위...</td>
    </tr>
  </tbody>
</table>
<p>12859 rows × 1 columns</p>
</div>




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
      <td>만두를먹을수있어서감사합니다</td>
    </tr>
    <tr>
      <td>1</td>
      <td>내맘에들지않는사람을쳐냈었는데처음으로손을내밀어봤습니다.그손을잡아주어서감사합니다.자존심...</td>
    </tr>
    <tr>
      <td>2</td>
      <td>버스를탈수있어서감사합니다</td>
    </tr>
    <tr>
      <td>3</td>
      <td>삶이무료해지고이제또즐거운무언가를찾고있었습니다.또의미없이시간보내는일을찾을수도있었는데좋...</td>
    </tr>
    <tr>
      <td>4</td>
      <td>붓다의유언을읽을수있어서감사했습니다말줄임표“제행이무상하니,방일하지말고정진하라"</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>12854</td>
      <td>(직장인들은공감할수있을것같은데)요즘나에게하루중가장길게느껴지는시간은출퇴근시간이다.매일...</td>
    </tr>
    <tr>
      <td>12855</td>
      <td>오랜시간동안네이버블로그에글을적어왔다.일본에서살기전에는일본여행관련으로적기시작하다가일본...</td>
    </tr>
    <tr>
      <td>12856</td>
      <td>토요일아침.사람들이주말을시작하는풍경은어떨까?밀린늦잠을자는사람들도,벌써하루를준비해나가...</td>
    </tr>
    <tr>
      <td>12857</td>
      <td>벌써르완다생활이1년하고도5개월.왠지1년차는예전에지나간느낌인데아직1년6개월이채되지않았...</td>
    </tr>
    <tr>
      <td>12858</td>
      <td>원래다이어리쓰는것을좋아했다.연말이되면의지가흐릿해졌지만그래도매년다이어를써왔다.근데스위...</td>
    </tr>
  </tbody>
</table>
<p>12859 rows × 1 columns</p>
</div>




```python
#리스트로 바꾸기
content=[df9['content'].loc[i] for i in df9.index]
```


```python
len(content)
```




    12859




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
      <td>문화의날이라그런지인기검색어에사자랑엑시트가있네용ㅎㅎㅎㅎ무대인사예매한게둘다날짜가겹치기도...</td>
      <td>82</td>
    </tr>
    <tr>
      <td>2</td>
      <td>고속버스타고올라가다내린휴게소.구미휴게소인데휴게소내에요런시설도있네요ㅎㅎ</td>
      <td>38</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1일부터월요일이면.그달이정~말길게느껴지더라고요.특히7월처럼31일이나되면서새까만달은.최악</td>
      <td>48</td>
    </tr>
    <tr>
      <td>4</td>
      <td>저는노포를정말좋아하는데그이유는첫번째오래된집은맛에서실패할확률이적고두번째로는그세월이주는...</td>
      <td>142</td>
    </tr>
    <tr>
      <td>5</td>
      <td>이시간전남비바람이몰아칩니다.ㅋㅋ.......낼일있는데..시끄러워서잠도안오고...아침...</td>
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

# 우울증 카페 글


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


```python
len(set(list(df4['content'])))
```




    1934




```python
len(df1),len(df2),len(df3),len(df4),len(df5),len(df6),len(df7),len(df8)
```




    (1613, 616, 1977, 3463, 23928, 726, 7113, 749)




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
      <td>일어나자마자. 살기 싫은 하루.. 이럴줄 알았지만 ㅎㅎ</td>
    </tr>
    <tr>
      <td>1</td>
      <td>내가 없어져도 해는 매일 뜨겠죠..</td>
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
      <td>니가 하고 싶은대로 행동하면난 어쩌란거니?도대체 어느 장단에 마춰야할지 모르겠어.이...</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>33807</td>
      <td>이게 우울증인지 뭔지 모르겠어서,,,, 예전에 상담 클리닉 다니면서 상담도 받았는데...</td>
    </tr>
    <tr>
      <td>33808</td>
      <td>안녕하세요 24살 여자입니다 고등학교 졸업 후 바로 부모님회사에서 일을 시작하게 되...</td>
    </tr>
    <tr>
      <td>33809</td>
      <td>제 친구가 있는데요.. 친구관계때문에 많이 힘든가봐요수업시간에 공책에 "우울감=&gt;우...</td>
    </tr>
    <tr>
      <td>33810</td>
      <td>제가 우울증이 있는데..답변1글자크기 조절, 질문하기, 답변쓰기 메뉴제가 원래는 누...</td>
    </tr>
    <tr>
      <td>33811</td>
      <td>잠을 의지대로 잘 못자고 있어요 그냥 잠만 못자면 다행인데 잠을 못자는 동안 계속 ...</td>
    </tr>
  </tbody>
</table>
<p>33812 rows × 1 columns</p>
</div>




```python
rem(' ',df0)
short(df0)
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
      <td>일어나자마자.살기싫은하루말줄임표이럴줄알았지만ㅎㅎ</td>
    </tr>
    <tr>
      <td>1</td>
      <td>내가없어져도해는매일뜨겠죠말줄임표</td>
    </tr>
    <tr>
      <td>2</td>
      <td>선택의후회,,</td>
    </tr>
    <tr>
      <td>3</td>
      <td>오늘도살아야하네말줄임표에휴말줄임표</td>
    </tr>
    <tr>
      <td>4</td>
      <td>니가하고싶은대로행동하면난어쩌란거니?도대체어느장단에마춰야할지모르겠어.이래도저래도난너한...</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>33807</td>
      <td>이게우울증인지뭔지모르겠어서,,,,예전에상담클리닉다니면서상담도받았는데조금좋아졌다가요새...</td>
    </tr>
    <tr>
      <td>33808</td>
      <td>안녕하세요24살여자입니다고등학교졸업후바로부모님회사에서일을시작하게되었어요2년6개월동안...</td>
    </tr>
    <tr>
      <td>33809</td>
      <td>제친구가있는데요말줄임표친구관계때문에많이힘든가봐요수업시간에공책에"우울감=&gt;우울증"이라...</td>
    </tr>
    <tr>
      <td>33810</td>
      <td>제가우울증이있는데말줄임표답변1글자크기조절,질문하기,답변쓰기메뉴제가원래는누가봐도밝고활...</td>
    </tr>
    <tr>
      <td>33811</td>
      <td>잠을의지대로잘못자고있어요그냥잠만못자면다행인데잠을못자는동안계속우울증증세가나타난지벌써2...</td>
    </tr>
  </tbody>
</table>
<p>33812 rows × 1 columns</p>
</div>




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
      <td>너요즘보기좋더라자꾸배우려고하고노력하는모습이너무예뻐</td>
      <td>27</td>
    </tr>
    <tr>
      <td>3</td>
      <td>우울하지도않고불안한건쬐금있다그닥죽고싶지도살고싶지도않다희안하다충동적인생각이드는건있다가...</td>
      <td>127</td>
    </tr>
    <tr>
      <td>4</td>
      <td>제가그동안좋아했던여자들의1/3만,,아니1~2명만이라도성공을했더라면,이런상황에웃으며넘...</td>
      <td>360</td>
    </tr>
  </tbody>
</table>
</div>




```python
df0[df0['len']>50]
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
      <td>3</td>
      <td>우울하지도않고불안한건쬐금있다그닥죽고싶지도살고싶지도않다희안하다충동적인생각이드는건있다가...</td>
      <td>127</td>
    </tr>
    <tr>
      <td>4</td>
      <td>제가그동안좋아했던여자들의1/3만,,아니1~2명만이라도성공을했더라면,이런상황에웃으며넘...</td>
      <td>360</td>
    </tr>
    <tr>
      <td>5</td>
      <td>두달전에공황장애증상으로쓰러지고나서부터는우울증과불안증이와서회사까지그만두게되었습니다말줄...</td>
      <td>115</td>
    </tr>
    <tr>
      <td>13</td>
      <td>잇다약타고오늘은또뭐하지투데이쉐떠뻑투마로우셔터뻑먼데이투데이아플예정sick점점사람들을멀...</td>
      <td>131</td>
    </tr>
    <tr>
      <td>16</td>
      <td>외모지상주의엄청심해요길가다가혹은차에서신호기다리다가아니면식당에서밥먹다가지나가는사람들얼...</td>
      <td>69</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>26354</td>
      <td>우리는아이의부정적인면은교정하고,긍정적인면은강화해나가면더이상삶이우울해있지않고,활력을되...</td>
      <td>53</td>
    </tr>
    <tr>
      <td>26359</td>
      <td>전23살여자구요되도록여성분이었으면좋겠어요많이우울하고힘들어서사람이만나고싶어요저녁식사하...</td>
      <td>68</td>
    </tr>
    <tr>
      <td>26361</td>
      <td>그리고혼자만지내다보니취미또한팝음악감상이나축구경기보기같은지극히혼자만깊게파고들수있는취미...</td>
      <td>51</td>
    </tr>
    <tr>
      <td>26362</td>
      <td>없는사실을있는거처럼꾸며내서저를아주바보로만들었더군요ㅡㅡ그러더니저랑카톡한그남자애는서른살...</td>
      <td>113</td>
    </tr>
    <tr>
      <td>26367</td>
      <td>말줄임표요즘기술시대라는데말줄임표정말어떻게해야할지모르겠습니다말줄임표도와주세요말줄임표저...</td>
      <td>67</td>
    </tr>
  </tbody>
</table>
<p>8683 rows × 2 columns</p>
</div>




```python
df0['label'] = 1
```


```python
df0.to_csv('depression_sum.csv')
```

## 두 데이터 합치기


```python
plt.plot(range(len(df0)),df0['len'].T.sort_values()) #글길이 분포 확인
```




    [<matplotlib.lines.Line2D at 0x1e66e1849b0>]




![png](depression_preprocessing_files/depression_preprocessing_38_1.png)



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




    [<matplotlib.lines.Line2D at 0x1e66e2126d8>]




![png](depression_preprocessing_files/depression_preprocessing_40_1.png)



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




```python
# 200 글자 이상 제거
df9 = df9[df9['len']<200]
df0 = df0[df0['len']<200]
```


```python
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
      <th>len</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>1</td>
      <td>문화의날이라그런지인기검색어에사자랑엑시트가있네용ㅎㅎㅎㅎ무대인사예매한게둘다날짜가겹치기도...</td>
      <td>82</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>고속버스타고올라가다내린휴게소.구미휴게소인데휴게소내에요런시설도있네요ㅎㅎ</td>
      <td>38</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>1일부터월요일이면.그달이정~말길게느껴지더라고요.특히7월처럼31일이나되면서새까만달은.최악</td>
      <td>48</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>저는노포를정말좋아하는데그이유는첫번째오래된집은맛에서실패할확률이적고두번째로는그세월이주는...</td>
      <td>142</td>
      <td>0</td>
    </tr>
    <tr>
      <td>5</td>
      <td>이시간전남비바람이몰아칩니다.ㅋㅋ.......낼일있는데..시끄러워서잠도안오고...아침...</td>
      <td>61</td>
      <td>0</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>8828</td>
      <td>오렌만에메일확인했는데.광고대행사한테서메일이몇군대왔네요.안하는게낫겠죠?</td>
      <td>38</td>
      <td>0</td>
    </tr>
    <tr>
      <td>8829</td>
      <td>자꾸열받으면먹을려고하고</td>
      <td>12</td>
      <td>0</td>
    </tr>
    <tr>
      <td>8830</td>
      <td>다른날들도그렇지만오늘은더욱그림확언이좋네요^^배경음악들도연말분위기를더욱풍성하게해주네요...</td>
      <td>77</td>
      <td>0</td>
    </tr>
    <tr>
      <td>8831</td>
      <td>목요일인데아직목감기임</td>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <td>8832</td>
      <td>참외2개깎아서탁자에놔두었는데....댓글달기놀이에빠져있는사이남편혼자다먹어버렸다는......</td>
      <td>170</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>7890 rows × 3 columns</p>
</div>




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
      <th>len</th>
      <th>label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>행복합니다.감사합니다.^^</td>
      <td>14</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>삶을풍요롭게만드는이에프티톡톡톡!!!.꿀추석을모두잘보내고계시죠?.남은휴일도행복하세요!</td>
      <td>46</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>문화의날이라그런지인기검색어에사자랑엑시트가있네용ㅎㅎㅎㅎ무대인사예매한게둘다날짜가겹치기도...</td>
      <td>82</td>
      <td>0</td>
    </tr>
    <tr>
      <td>3</td>
      <td>고속버스타고올라가다내린휴게소.구미휴게소인데휴게소내에요런시설도있네요ㅎㅎ</td>
      <td>38</td>
      <td>0</td>
    </tr>
    <tr>
      <td>4</td>
      <td>1일부터월요일이면.그달이정~말길게느껴지더라고요.특히7월처럼31일이나되면서새까만달은.최악</td>
      <td>48</td>
      <td>0</td>
    </tr>
    <tr>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <td>7875</td>
      <td>글하나하나읽어가며느끼는게많은새벽ㅎ.불면증인가잠을못자네다들굿잠이요ㅎ</td>
      <td>36</td>
      <td>0</td>
    </tr>
    <tr>
      <td>7876</td>
      <td>다른날들도그렇지만오늘은더욱그림확언이좋네요^^배경음악들도연말분위기를더욱풍성하게해주네요...</td>
      <td>77</td>
      <td>0</td>
    </tr>
    <tr>
      <td>7877</td>
      <td>목요일인데아직목감기임</td>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <td>7878</td>
      <td>결혼한남자가회식할때마다단순직장동료인여자한테잘잘도착했냐,조심히들어가라,쉬어라등등매번카...</td>
      <td>79</td>
      <td>0</td>
    </tr>
    <tr>
      <td>7879</td>
      <td>후...워터파크탈의실에서사진찍어대는인간들왜이리많은지모르겠네요.찍으려면지들도수영복벗고...</td>
      <td>53</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>7880 rows × 3 columns</p>
</div>




```python
df9.loc[len(df9)]=['기분이좋다',5,0]
```


```python
df0.loc[len(df0)]=['기분이나쁘다',6,1]
```


```python
df = pd.concat([df0,df9],axis = 0)
df = df.reset_index(drop = True)
df.head(5)
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
      <td>하나님에대한신앙이최우선이라하지만내눈엔자신의말=하나님의말이라고생각하는거같음</td>
      <td>40</td>
      <td>1</td>
    </tr>
    <tr>
      <td>1</td>
      <td>내인생이가장힘들었던건나를무어라설명할수없었기때문이다</td>
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
      <td>제가깨있어도1시반까지하는애인데뭐하나궁금해서나갔더니막마우스클릭소리나는데,화면보면네이버입니다</td>
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
      <td>다른날들도그렇지만오늘은더욱그림확언이좋네요^^배경음악들도연말분위기를더욱풍성하게해주네요...</td>
      <td>77</td>
      <td>0</td>
    </tr>
    <tr>
      <td>15758</td>
      <td>목요일인데아직목감기임</td>
      <td>11</td>
      <td>0</td>
    </tr>
    <tr>
      <td>15759</td>
      <td>결혼한남자가회식할때마다단순직장동료인여자한테잘잘도착했냐,조심히들어가라,쉬어라등등매번카...</td>
      <td>79</td>
      <td>0</td>
    </tr>
    <tr>
      <td>15760</td>
      <td>후...워터파크탈의실에서사진찍어대는인간들왜이리많은지모르겠네요.찍으려면지들도수영복벗고...</td>
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




    [<matplotlib.lines.Line2D at 0x1e66f23b358>]




![png](depression_preprocessing_files/depression_preprocessing_57_1.png)


## 토크나이저, 인코딩, pad_sequences


```python
_input1 = df['content']
_label = df['label']
```


```python
cleaning =lambda s: re.sub("[^가-힣a-zA-Z.!?\\s]","",s) # 이모티콘같은거 제거
```


```python
# 형태소 단위로 토크나이즈
tokenizer = Okt()
_input1 = [ tokenizer.morphs(cleaning(str(sentence))) for sentence in _input1]
```


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


![png](depression_preprocessing_files/depression_preprocessing_69_0.png)



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
