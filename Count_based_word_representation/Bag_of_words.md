# Bag of Words(BoW)

### 1. BoW란?
- Bag of Words: 단어들의 순서는 전혀 고려하지 않고, 단어들의 **출현 빈도(frequency)**에만 집중하는 텍스트 데이터의 수치화 표현 방법
- BoW를 만드는 과정
    - 우선, 각 단어에 고유한 정수 인덱스를 부여
    - 각 인덱스의 위치에 단어 토큰의 등장 횟수를 기록한 벡터를 만듦
```python
from konlpy.tag import Okt
import re

okt = Okt()

token = re.sub("(\.)", "", "정부가 발표하는 물가상승률과 소비자가 느끼는 물가상승률은 다르다.")
# 정규 표현식을 통해 온점을 제거하는 정제 작업입니다.
token = okt.morphs(token)
# OKT 형태소 분석기를 통해 토큰화 작업을 수행한 뒤에, token에다가 넣습니다.

word2index = {}
bow = []
for voca in token:
    # token을 읽으면서, word2index에 없는 (not in) 단어는 새로 추가하고, 이미 있는 단어는 넘깁니다.
    if voca not in word2index.keys():
        word2index[voca] = len(word2index)
        # BoW 전체에 전부 기본값 1을 넣어줍니다. 단어의 개수는 최소 1개 이상이기 때문입니다.
        bow.insert(len(word2index)-1, 1)
    else:
        # 재등장하는 단어의 인덱스를 받아옵니다.
        index = word2index.get(voca)
        # 재등장하는 단어는 해당하는 인덱스의 위치에 1을 더해줍니다.
        bow[index] += 1
        
print(word2index)
>>> ('정부': 0, '가': 1, '발표': 2, '하는': 3, '물가상승률': 4, '과': 5, '소비자': 6, '느끼는': 7, '은': 8, '다르다': 9)  
print(bow)
>>> [1, 2, 1, 1, 2, 1, 1, 1, 1, 1]  
```

### 3. `CountVectorizer` 클래스로 BoW 만들기
```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = ['you know I want your love. because I love you.']
vector = CountVectorizer()

print(vector.fit_transform(corpus).toarray()) # 코퍼스로부터 각 단어의 빈도 수를 기록
>>> [[1 1 2 1 2 1]]
print(vector.vocabulary_) # 각 단어의 인덱스가 어떻게 부여되었는지를 보여준다.
>>> {'you': 4, 'know': 1, 'want': 3, 'your': 5, 'love': 2, 'because': 0}
```
- 주의할 점은 `CountVectorizer`는 단지 띄어쓰기만을 기준으로 단어를 자르는 낮은 수준의 토큰화를 진행하고 BoW를 만듦
- '물가상승률과'와 '물가상승률은'은 서로 다른 단어로 인식

### 4. 불용어를 제거한 BoW 만들기
```python
# (1) 사용자가 직접 정의한 불용어 사용
from sklearn.feature_extraction.text import CountVectorizer

text = ["Family is not an important thing. It's everything."]
vect = CountVectorizer(stop_words=["the", "a", "an", "is", "not"])

print(vect.fit_transform(text).toarray())
>>> [[1 1 1 1 1]]
print(vect.vocabulary_)
>>> {'family': 1, 'important': 2, 'thing': 4, 'it': 3, 'everything': 0}

# (2) CountVectorizer에서 제공하는 자체 불용어 사용
from sklearn.feature_extraction.text import CountVectorizer

text = ["Family is not an important thing. It's everything."]
vect = CountVectorizer(stop_words='english')

print(vect.fit_transform(text).toarray())
>>> [[1 1 1]]
print(vect.vocabulary_)
>>> {'family': 0, 'important': 1, 'thing': 2}

# (3) `nltk`에서 지원하는 불용어 사용
from sklearn.feature_extraction.text import CountVectorizer
fron nltk.corpus import stopwords

sw = stopwords.words('english')
text = ["Family is not an important thing. It's everything."]
vect = CountVectorizer(stop_words='english')

print(vect.fit_transform(text).toarray())
>>> [[1 1 1 1]]
print(vect.vocabulary_)
>>> {'family': 1, 'important': 2, 'thing': 3, 'everything': 0}
```
