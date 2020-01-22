# One-hot encoding

## Definition
- Vocaburary: 서로 다른 단어들의 집합. (사전이라고도 말함)
    - book, books와 같이 단어의 변형 형태도 다른 언어로 간주

### Q) 각 단어에 고유한 정수 인덱스를 부여했다고 하자. 이 숫자로 바뀐 단어들을 벡터로 다루고 싶으면 어떻게 해야할까?

## 1. One-hot encoding
- 단어 집합의 크기를 벡터의 차원, 표현하고 싶은 단어의 인덱스에 1의 값을 부여, 다른 인덱스에는 0을 부여
- 원-핫 인코딩의 두 가지 과정
    - (1) 각 단어에 고유한 인덱스를 부여 (정수 인코딩)
    - (2) 표현하고 싶은 단어의 인덱스의 위치에 1을 부여, 다른 단어의 인덱스의 위치에는 0을 부여
```python
from konlpy.tag import Okt
okt = Okt()
token = okt.morphs('나는 자연어 처리를 배운다')
print(token)
>>> ['나', '는', '자연어', '처리', '를', '배운다']

word2index = {]
for voca in token:
    if voca not in word2index.keys():
        word2index[voca] = len(word2index)
print(word2index)
>>> {'나': 0, '는': 1, '자연어': 2, '처리': 3, '를': 4, '배운다': 5}

# 각 토큰에 대해 고유한 인덱스(index)를 부여
# 빈도수 순대로 단어를 정렬하여 고유한 인덱스를 부여하는 작업이 사용되기도 함
def one_hot_encoding(word, word2index):
    one_hot_vector = [0] * (len(word2index))
    index = word2index[word]
    one_hot_vector[index] = 1
    return one_hot_vector
    
# 혹은 numpy 배열을 사용하여 아래와 같이 쉽게 만들 수도 있다.
def one_hot_encoding(word, word2index):
    return np.eye(len(word2index))[word2index[word]]
    
print(one_hot_encoding('자연어', word2index)
>>> [0, 0, 1, 0, 0, 0]
```

## 2. `keras`를 이용한 원-핫 인코딩
```python
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

text = "나랑 점심 먹으러 갈래 점심 메뉴는 햄버거 갈래 갈래 험버거 최고야"

t = Tokenizer()
t.fit_on_texts([text])
print(t.word_index) # 각 단어에 대한 인코딩 결과 출력
>>> {'갈래': 1, '점심': 2, '햄버거': 3, '나랑': 4, '먹으러': 5, '메뉴는': 6, '최고야': 7}

sub_text = "점심 먹으러 갈래 메뉴는 햄버거 최고야"
encoded = t.texts_to_sequences([sub_text])[0]
print(encoded)
>>> [2, 5, 1, 6, 3, 7]

one_hot = to_categorical(encoded)
print(one_hot)
>>> [[0. 0. 1. 0. 0. 0. 0. 0.]   # 인덱스 2의 원-핫 벡터
     [0. 0. 0. 0. 0. 1. 0. 0.]   # 인덱스 5의 원-핫 벡터
     [0. 1. 0. 0. 0. 0. 0. 0.]   # 인덱스 1의 원-핫 벡터
     [0. 0. 0. 0. 0. 0. 1. 0.]   # 인덱스 6의 원-핫 벡터
     [0. 0. 0. 1. 0. 0. 0. 0.]   # 인덱스 3의 원-핫 벡터
     [0. 0. 0. 0. 0. 0. 0. 1.]]  # 인덱스 7의 원-핫 벡터
```

## 3. Limitation of One-hot encoding
- 단어의 개수가 늘어나면 늘어날수록 sparse해지고 필요한 공간이 계속 늘어남
- 단어의 유사도를 표현하지 못함
- 이를 해결하기 위해 단어의 잠재 의미를 반영하ㅕㅇ 다차우너 공간에 벡터화하는 기법으로 크게 두 가지
    - Count 기반의 벡터화 방법 LSA, HAL 등
    - 예측 기반으로 벡터화하는 NNLM, RNNLM, Word2Vec, FastText
    - 그리고 위 둘 모두 사용하는 방법으로 GloVe
