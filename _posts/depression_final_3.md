```python
import pandas as pd, pickle, re
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np
```


```python
from termcolor import colored
```


```python
from IPython.core.display import HTML
#display(HTML(df2.to_html()))
```


```python
from tensorflow.keras.models import model_from_json 
json_file = open("model.json", "r") 
loaded_model_json = json_file.read() 
json_file.close() 
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
```


```python
eda=pd.read_csv('freq_all.csv')
```


```python
rem_list = ['친구','부모님','']
eda=eda[eda['text']!='친구']
```


```python
eda=eda[eda['compare']>=2]
```


```python
dic={'Noun': '명사',
 'Josa': '조사',
 'Adjective': '형용사',
 'Adverb': '부사',
 'Determiner': '한정사',
 'Verb': '동사',
 'Modifier': '수식어',
 'Conjunction': '접속사',
 'VerbPrefix': '동사 접두사',
 'Suffix': '접미사',
 'Exclamation': '감탄사',
 'Alpha': 'Alpha',
 'Foreign': 'Foreign',
 'Hashtag': 'Hashtag',
 'Eomi': 'Eomi',
 'Email': 'Email',
 'ScreenName': 'ScreenName'}
```


```python
cleaning =lambda s: re.sub("[^가-힣a-zA-Z.!?\\s]","",s)
tokenizer = Okt()

with open('word_dict.pickle', 'rb') as handle:
    data = pickle.load(handle)

keras_tokenizer, max_len = data
```


```python
def predict(a):

    list_dot = ['.'*i for i in range(20, 1, -1)]
    model = loaded_model
    
    a = a.replace(' ','')
    
    #말줄임표 없애기
    for i in list_dot:
        a = a.replace(i,'말줄임표')

    _input1 = [tokenizer.morphs(cleaning(str(a)))]
        
    _input2 = keras_tokenizer.texts_to_sequences(_input1) 
    _input2 = pad_sequences(_input2, maxlen=max_len, padding='post')
    
    pred = model.predict(_input2)

        
    result = str(int(pred*100)) + '%'
        
    return result
```


```python
while True:
    a = input('\n끄적끄적 해보세요: \n\n')
    if a=='':
        break
    else:
        text=[]
        content=[]
        freq=[]
        compare=[]
        a = a.replace(' ','')
        for i in tokenizer.pos(cleaning(str(a))):
            try:
                if (i[0]+'/'+dic[i[1]]) in list(eda['content']):
                    content.append(i[0]+'/'+dic[i[1]])

            except:
                continue

        content=list(set(content))
        for i in content:
            freq.append(int(eda['freq'][eda['content']==i]))
            compare.append(np.round(float(eda['compare'][eda['content']==i]),1))
            text.append(eda['freq'][eda['content']==i])

        df=pd.DataFrame({'text':text,'content':content,'freq':freq,'compare':compare}).sort_values('compare',ascending=False)
        df=df[:7]
        result=predict(a)


        if len(freq)==0 :

            print(colored('\n당신의 우울증 확률: {0} \n', attrs=['reverse', 'blink']) .format(result))
        elif int(result[:-1])<=50:

            print(colored('\n당신의 우울증 확률: {0} \n', attrs=['reverse', 'blink']) .format(result))

        else:

            print(colored('\n당신의 우울증 확률: {0} \n', attrs=['reverse', 'blink']) .format(result))
            for i in tokenizer.pos(cleaning(str(a))):
                try:
                    if (i[0]+'/'+dic[i[1]]) in content:
                        print(colored(i[0],'red', attrs=['reverse', 'blink']),end=' ')
                    else:
                        print(i[0],end=' ')
                except:
                    continue
            print('\n우울증 문서에 자주 나타나는 표현:\n')

            for i in list(df.index):
                try:
                    if df.compare[i]!=0:
                        print(df.content[i].replace('/',' / '))
                        print(': 일반문서의 '.rjust(25),str(df.compare[i]),'배 등장\n')
                    else:
                        print(df.content[i].replace('/',' / '),': 우울증 문서에만 등장')
                except:
                    continue
                    
```

    
    끄적끄적 해보세요: 
    
    어제는 너무 우울했다. 그냥 확 죽어버리고 싶기도 했다. 서류전형부터 거의 다 탈락하고 계획한걸 단 한가지도 실행시키지 못한 내가 무능력하게 느껴졌다.
    [5m[7m
    당신의 우울증 확률: 81% 
    [0m
    어제 는 [5m[7m[31m너무[0m 우울했다 [5m[7m[31m그냥[0m 확 죽어 버리고싶 기도 했다 서류 전형 부터 거의 다 탈락 하고 계획 한 걸단 한가지 도 실행 시키지못 한 내 가 무능력하게 느껴졌다 
    우울증 문서에 자주 나타나는 표현:
    
    그냥 / 수식어
                     : 일반문서의  5.1 배 등장
    
    너무 / 부사
                     : 일반문서의  3.0 배 등장
    
    
    끄적끄적 해보세요: 
    
    안힘들었던건아니야 응응 난심지어그와중에 곰돌이녀석이 손가락을 물어서..좀심하게 콱 도라에몽손가락이되었던적잇어. ㅋㅋㅋ오른손..3개 다쳤다고 혼내더라 서러웠어.. 넌 타자도 느린데 다치기까지하면 어쩌려고 그러냐고 이거 교육3개월간 수액4번맞았어 1개월에 한번씩 실습시작하고 오늘도 연차내고 맞으러가니깐 또 시작인거지 뭐하러 이렇게까지하나싶고 이게필요한가.. 별생각다하고 너무 힘든데 부모님이 모르잖아..잘받아주지도 않고 취업은 해야겠고 답답하고 나자신에게도 화나고 처음으로 내생에 노력만으로 안되는게 있다는걸 알았을때 끔찍했어! 너무 내가 갇혀서 살아왔다는걸 깨달았거든... 이젠 그래도 좀 괜찮아 비록...회사가 우리만 서울이라 주6일인게 힘든데 곧끝나고 얻는게 있을테니깐 내말만 잔뜩해서 미안..ㅎ넌옮기고 나서 무얼하는지 모르니까 일단 나는 그랬었어
    [5m[7m
    당신의 우울증 확률: 99% 
    [0m
    안 힘들었던건 아니야 응응 난 심지어 그 와중 에 곰돌이 녀석 이 손가락 을 물어 서 [5m[7m[31m좀[0m 심하게 콱 도라에몽 손가락 이 되었던 적 잇어 오른손 개 다쳤다고 혼내더라 서러웠어 넌 타자 도 느린데다치기까지 하면어 쩌 려고 그러냐고 이 거 교육 개 월간 수액 번 맞았어 개월 에 한번 씩 실습 시작 하고 오늘 도 연차 내고 맞으러 가니깐 또 시작 인 거지 뭐 하러 이렇게 까지 하나 싶고이게 필요한가 별 생각 다 하고 [5m[7m[31m너무[0m 힘든데 [5m[7m[31m부모님[0m 이 모르잖아 잘 받아주지도 않고 취업 은 해야겠고 답답하고나 자신 에게도 화나고 처음 으로 내생 에 노력 만으로 안되는게 있다는걸 알았을 때 끔찍했어 [5m[7m[31m너무[0m 내 가 갇혀서 살아왔다는걸 깨 달았거든 이 [5m[7m[31m젠[0m 그래도 [5m[7m[31m좀[0m 괜찮아 비록 회사 가우리 만 서울 이라 주일 인게 힘든데 곧 끝나고 얻는게 있을테니깐 내 말 만 잔뜩 해서 미안 넌 옮기고나 서무 얼하는지 모르니까 일단 나 는 그랬었어 
    우울증 문서에 자주 나타나는 표현:
    
    부모님 / 명사
                     : 일반문서의  6.7 배 등장
    
    젠 / 명사
                     : 일반문서의  3.7 배 등장
    
    너무 / 부사
                     : 일반문서의  3.0 배 등장
    
    좀 / 명사
                     : 일반문서의  2.3 배 등장
    
    
    끄적끄적 해보세요: 
    
    
    


```python
txt = '[지유언니] [오후 2:30] 안힘들었던건아니야[지유언니] [오후 2:30] 응응[지유언니] [오후 2:30] 난심지어그와중에[지유언니] [오후 2:30] 곰돌이녀석이[지유언니] [오후 2:30] 손가락을[지유언니] [오후 2:30] 물어서..좀심하게 콱[지유언니] [오후 2:31] 도라에몽손가락이되었던적잇어. ㅋㅋㅋ오른손..3개[지유언니] [오후 2:31] 다쳤다고 혼내더라[지유언니] [오후 2:31] 서러웠어..[지유언니] [오후 2:31] 넌 타자도 느린데 다치기까지하면 어쩌려고 그러냐고[지유언니] [오후 2:32] 이거 교육3개월간 수액4번맞았어 1개월에 한번씩 실습시작하고 오늘도 연차내고 맞으러가니깐 또 시작인거지[지유언니] [오후 2:32] 뭐하러 이렇게까지하나싶고 이게필요한가..[지유언니] [오후 2:32] 별생각다하고[지유언니] [오후 2:33] 너무 힘든데 부모님이 모르잖아..잘받아주지도 않고 취업은 해야겠고 답답하고 나자신에게도 화나고 처음으로 내생에 노력만으로 안되는게 있다는걸 알았을때 끔찍했어! 너무 내가 갇혀서 살아왔다는걸 깨달았거든...[지유언니] [오후 2:34] 이젠 그래도 좀 괜찮아 비록...회사가 우리만 서울이라 주6일인게 힘든데 곧끝나고 얻는게 있을테니깐[지유언니] [오후 2:35] 내말만 잔뜩해서 미안..ㅎ넌옮기고 나서 무얼하는지 모르니까 일단 나는 그랬었어'
```


```python
list_time=['[지유언니] [오후 2:30]','[지유언니] [오후 2:31]','[지유언니] [오후 2:32]','[지유언니] [오후 2:33]','[지유언니] [오후 2:34]','[지유언니] [오후 2:35]']
for i in list_time:
    txt = txt.replace(i,'')
```


```python
txt
```




    ' 안힘들었던건아니야 응응 난심지어그와중에 곰돌이녀석이 손가락을 물어서..좀심하게 콱 도라에몽손가락이되었던적잇어. ㅋㅋㅋ오른손..3개 다쳤다고 혼내더라 서러웠어.. 넌 타자도 느린데 다치기까지하면 어쩌려고 그러냐고 이거 교육3개월간 수액4번맞았어 1개월에 한번씩 실습시작하고 오늘도 연차내고 맞으러가니깐 또 시작인거지 뭐하러 이렇게까지하나싶고 이게필요한가.. 별생각다하고 너무 힘든데 부모님이 모르잖아..잘받아주지도 않고 취업은 해야겠고 답답하고 나자신에게도 화나고 처음으로 내생에 노력만으로 안되는게 있다는걸 알았을때 끔찍했어! 너무 내가 갇혀서 살아왔다는걸 깨달았거든... 이젠 그래도 좀 괜찮아 비록...회사가 우리만 서울이라 주6일인게 힘든데 곧끝나고 얻는게 있을테니깐 내말만 잔뜩해서 미안..ㅎ넌옮기고 나서 무얼하는지 모르니까 일단 나는 그랬었어'


