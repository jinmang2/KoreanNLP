# Stop Word

## Definition
- stopword: 조사, 접미사 같은 단어들로 문장에서는 자주 등장하지만 실제 의미 분석을 하는데는 거의 기여하지 않는 단어들

## English
### 1. `nltk`에서 불용어 확인하기
```python
from nltk.corpus import stopwords
stopwords.words('english')[:10]
>>> ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your']  
```

### 2. `nltk`를 통해서 불용어 제거하기
```python
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

exmample = 'Family is not an important thing. It's everything."
stop_words = set(wtopwords.word('english'))

word_tokens = word_tokenize(example)

result = []
for w in word_tokens:
  if w not in stop_words:
    result.append(w)
    
print(word_tokens)
print(result)
>>> ['Family', 'is', 'not', 'an', 'important', 'thing', '.', 'It', "'s", 'everything', '.']
>>> ['Family', 'important', 'thing', '.', 'It', "'s", 'everything', '.']
```

## Korean
### 3. 한국어에서 불용어 제거
- 사용자가 직접 불용어 사전을 만들게 되는 경우가 많음
```python
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

example = "고기를 아무렇게나 구우려고 하면 안 돼. 고기라고 다 같은 게 아니거든. 예컨대 삼겹살을 구울 때는 중요한 게 있지."
stop_words = "아무거나 아무렇게나 어찌하든지 같다 비슷하다 예컨대 이럴정도로 하면 아니거든"
# 위의 불용어는 명사가 아닌 단어 중에서 저자가 임의로 선정한 것으로 실제 의미있는 선정 기준이 아님
stop_words=stop_words.split(' ')
word_tokens = word_tokenize(example)

result = [] 
for w in word_tokens: 
    if w not in stop_words: 
        result.append(w) 
# 위의 4줄은 아래의 한 줄로 대체 가능
# result=[word for word in word_tokens if not word in stop_words]

print(word_tokens) 
print(result)
>>> ['고기를', '아무렇게나', '구우려고', '하면', '안', '돼', '.', '고기라고', '다', '같은', '게', '아니거든', '.', 
     '예컨대', '삼겹살을', '구울', '때는', '중요한', '게', '있지', '.']
>>> ['고기를', '구우려고', '안', '돼', '.', '고기라고', '다', '같은', '게', '.', '삼겹살을', '구울', '때는', '중요한', 
     '게', '있지', '.']
```
- 참고할 한국어 불용어 리스트
  - https://www.ranks.nl/stopwords/korean
  - https://bab2min.tistory.com/544
