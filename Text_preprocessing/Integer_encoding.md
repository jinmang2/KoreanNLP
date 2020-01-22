# Integer Encoding

## 정수 인코딩(Integer Encoding)
### (1) dictionary 사용하기
```python
# Import Library
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

text = """A barber is a person. a barber is good person. a barber is huge person. he Knew A Secret! 
    The Secret He Kept is huge secret. Huge secret. His barber kept his word. a barber kept his word. 
    His barber kept his secret. 
    But keeping and keeping such a huge secret to himself was driving the barber crazy. 
    the barber went up a huge mountain."""

# 문장 토큰화
text = sent_tokenize(text)

# 정제와 단어 토큰화
vocab={} # 파이썬의 dictionary 자료형
sentences = []
stop_words = set(stopwords.words('english'))

for i in text:
    sentence = word_tokenize(i) # 단어 토큰화를 수행합니다.
    result = []

    for word in sentence: 
        word = word.lower() # 모든 단어를 소문자화하여 단어의 개수를 줄입니다.
        if word not in stop_words: # 단어 토큰화 된 결과에 대해서 불용어를 제거합니다.
            if len(word) > 2: # 단어 길이가 2이하인 경우에 대하여 추가로 단어를 제거합니다.
                result.append(word)
                if word not in vocab:
                    vocab[word] = 0 
                vocab[word] += 1
    sentences.append(result) 

# 빈도수가 높은 순서대로 정렬
vocab_sorted = sorted(vocab.items(), key=lambda x:x[1], reverse=True)

# 높은 빈도수를 가진 단어일수록 낮은 정수 인덱스를 부여
word_to_index = {}
i=0
for (word, frequency) in vocab_sorted :
    if frequency > 1 : # 정제(Cleaning) 챕터에서 언급했듯이 빈도수가 적은 단어는 제외한다.
        i = i+1
        word_to_index[word] = i
        
# 상위 5개 단어만 사용
vocab_size=5
words_frequency = [w for w,c in word_to_index.items() if c >= vocab_size + 1] # 인덱스가 5 초과인 단어 제거
for w in words_frequency:
    del word_to_index[w] # 해당 단어에 대한 인덱스 정보를 삭제
    
# Out Of Vocaburary 추가
word_to_index['OOV'] = len(word_to_index) + 1

# 문장의 모든 단어를 정수로 매핑
encoded = []
for s in sentences:
    temp = []
    for w in s:
        temp.append(word_to_index.get(w, word_to_index['OOV']) # 단어가 없으면 'OOV'에 해당하는 정수를 매핑
    encoded.append(temp)
```

### (2) Counter 사용하기
```python
from collections import Counter

words = sum(sentences, []) # 와후... 신기하네
vocab = Counter(words) # 파이썬의 Counter 모듈을 이용하여 단어의 모든 빈도를 쉽게 계산

# 등장 빈도 상위 5개의 단어만 단어 집합으로 저장
vocab_size = 5
vocab = vocab.most_common(vocab_size) # 등장 빈도수가 높은 상위 5개의 단어만 저장

# 높은 빈도수를 가진 단어일수록 낮은 정수 인덱스를 부여
word_to_index = {}
i=0
for (word, frequency) in vocab :
    i = i+1
    word_to_index[word] = i
print(word_to_index)
>>> {'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5}
```

### (3) `nltk`의 `FreqDist` 사용하기
```python
from nltk import FreqDist
import numpy as np

# np.hstack으로 문장 구분을 제거하여 입력으로 사용
vocab = FreqDist(np.hstack(sentences))

# 빈도수 상위 5개 항목만 남김
vocab_size = 5
vocab = vocab.most_common(vocab_size) # 등장 빈도수가 높은 상위 5개의 단어만 저장

# enumerate 내장 함수를 이용하여 높은 빈도수를 가진 단어에 낮은 정수 인덱스를 부여
word_to_index = {word[9] : index+1 for index, word in enumerate(vocab)}
print(word_to_index)
>>> {'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5}
```

## 2. `keras`의 텍스트 전처리
```python
from tensorflow.keras.preprocessing.text import Tokenizer

sentences=[['barber', 'person'], 
           ['barber', 'good', 'person'], 
           ['barber', 'huge', 'person'], 
           ['knew', 'secret'], 
           ['secret', 'kept', 'huge', 'secret'], 
           ['huge', 'secret'], 
           ['barber', 'kept', 'word'], 
           ['barber', 'kept', 'word'], 
           ['barber', 'kept', 'secret'], 
           ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'], 
           ['barber', 'went', 'huge', 'mountain']]

tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences) # fit_on_texts()안에 코퍼스를 입력으로 하면 빈도수를 기준으로 단어 집합을 생성
                                  # 빈도수가 높은 순으로 낮은 정수 인덱스를 부여

# 각 단어 별 카운트를 수행했을 시 이를 확인하고 싶으면 word_counts로 접근하여 확인
print(tokenizer.word_counts)
>>> OrderedDict(
        [('barber', 8), ('person', 3), ('good', 1), ('huge', 5), ('knew', 1), 
         ('secret', 6), ('kept', 4), ('word', 2), ('keeping', 2), ('driving', 1), 
         ('crazy', 1), ('went', 1), ('mountain', 1)])

# texts_to_sequences() 메서드로 이미 정해진 인덱스로 매핑
print(tokenizer.texts_to_sequences(sentences))
>>> [[1, 5], 
     [1, 8, 5], 
     [1, 3, 5], 
     [9, 2], 
     [2, 4, 3, 2], 
     [3, 2], 
     [1, 4, 6], 
     [1, 4, 6], 
     [1, 4, 2], 
     [7, 7, 3, 2, 10, 1, 11], 
     [1, 12, 3, 13]]
     
# 상위 5개 단어 사용
vocab_size=5
tokenizer = Tokenizer(num_words=vocab_size+1) # 상위 5개 단어만 사용
tokenizer.fit_on_texts(sentences) # 토크나이저 재정의

# 만일 OOV를 살리고 싶으면 아래의 argument를 활용
vocab_size=5
tokenizer = Tokenizer(num_words=vocab_size+2, oov_token='OOV')
# 빈도수 상위 5개 단어만 사용. 숫자 0과 OOV를 고려해서 단어 집합의 크기는 +2
tokenizer.fit_on_texts(sentences)

print('단어 OOV의 인덱스 : {}'.format(tokenizer.word_index['OOV']))
>>> 단어 OOV의 인덱스 : 1
```
