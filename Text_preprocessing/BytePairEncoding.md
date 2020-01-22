# Byte Pair Encoding, BPE
- 기계가 모르는 단어로 인해 문제를 제대로 풀지 못하는 상황을 **OOV 문제**라고 함

## Definition
- Subword segmentation(단어 분리): 하나의 단어는 의미있는 여러 내부 단어들(subwords)의 조합으로 구성된 경우가 많이 때문에, 단어를 여러 단어로 분리해서 단어를 이해해보겠다는 의도를 가진 리 작업

## 1. BPE(Byte Pair Encoding)
- 1994년 제안된 데이터 압축 알고리즘
- 후에 자연어 처리의 단어 분리 알고리즘으로 응용
- 기본 작동 방법은 아래와 같은
```
aaabdaaabac

>> aa가 가장 많이 등장.(byte pair) 이를 Z로 치환

ZabdZabac
Z = aa

>> ab가 가장 많이 등장하고 있는 바이트의 쌍은 'ab'. Y로 치환

ZYdZYac
Y = ab
Z = aa

>> 이제 가장 많이 등장하는 바이트의 쌍은 ZY. 이를 X로 치환

XdXac
X = ZY
Y = ab
Z = aa
```

## 2. 자연어 처리에서의 BPE(Byte Pair Encoding)
- 논문 : https://arxiv.org/pdf/1508.07909.pdf

### 1) 기존의 접근
```
# dictionary
# 훈련 데이터에 있는 단어와 등장 빈도수
low: 5, lower: 2, newest: 6, widest: 3

# vocabulary
low, lower, newest, widest

# 만일 lowest가 등장한다면? gg...
```

### 2) BPE 알고리즘을 사용한 경우
```
# dictionary
l o w: 5, l o w e r: 2, n e w e s t: 6, w i d e s t: 3

# vocabulary
l, o, w, e, r, n, w, s, t, i, d

# BPE는 알고리즘의 동작을 몇 회 반복(iteration)할 것인지를 사용자가 정함 (hyper-parameter)

## 1회: 딕셔너리를 참고로 하였을 때 빈도수가 9로 가장 높은 (e, s)의 쌍을 es로 통합
# dictionary update!
l o w : 5,
l o w e r : 2,
n e w es t : 6,
w i d es t : 3

# vocabulary update!
l, o, w, e, r, n, w, s, t, i, d, es

## 2회: 빈도수가 9로 가장 높은 (es, t)의 쌍을 est로 통합
# dictionary update!
l o w : 5,
l o w e r : 2,
n e w est : 6,
w i d est : 3

# vocabulary update!
l, o, w, e, r, n, w, s, t, i, d, es, est

## 3회: 빈도수가 7로 가장 높은 (l, o)의 쌍을 lo로 통합
lo w : 5,
lo w e r : 2,
n e w est : 6,
w i d est : 3

# vocabulary update!
l, o, w, e, r, n, w, s, t, i, d, es, est, lo

## 이와 같이 10번 반복...
# dictionary update!
low : 5,
low e r : 2,
newest : 6,
widest : 3

# vocabulary update!
l, o, w, e, r, n, w, s, t, i, d, es, est, lo, low, ne, new, newest, wi, wid, widest
```

### 3) 코드 실습하기
```python
import re, collections

num_merges = 10

vocab = {
    'l o w <\w>': 5,
    'l o w e r <\w>': 2,
    'n e w e s t <\w>': 6,
    'w i d e s t <\w>': 3
}

def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i+1]] += freq
    return pairs
    
def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?<!\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    return v_out
    
for i in range(num_merges):
    pairs = get_stats(vocab)
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)
    print(best)
    
>>> ('e', 's')
>>> ('es', 't')
>>> ('est', '</w>')
>>> ('l', 'o')
>>> ('lo', 'w')
>>> ('n', 'e')
>>> ('ne', 'w')
>>> ('new', 'est</w>')
>>> ('low', '</w>')
>>> ('w', 'i')
```

## 2. WPM(Wordpiece Model)
- WPM의 아이디어를 제시한 논문 : https://static.googleusercontent.com/media/research.google.com/ko//pubs/archive/37842.pdf
- 구글이 위 WPM을 변형하여 번역기에 사용했다는 논문 : https://arxiv.org/pdf/1609.08144.pdf
- 기존 BPE 이외에 WPM, Unigram Language Model Tokenizer와 같은 단어 분리 토크나이저가 존재
- WPM은 빈도수가 아닌 우도(likelihood)로 단어를 분리

## 3. Sentencepiece
- 논문 : https://arxiv.org/pdf/1808.06226.pdf
- 센텐스피스 깃허브 : https://github.com/google/sentencepiece
- 구글은 BPE 알고리즘과 Unigram Language Model Tokenizer를 구현한 sentencepiece를 github에 공개
- sentencepiece는 사전 토큰화 작업없이 단어 분리 토큰화를 수행하므로 언어에 종속되지 않음(multilingual)

### Reference
- BPE 알고리즘 논문 : https://arxiv.org/pdf/1508.07909.pdf
- BPE 알고리즘 논문 저자의 깃허브 : https://github.com/rsennrich/subword-nmt
- 서브워드 알고리즘 비교 : https://medium.com/@makcedward/how-subword-helps-on-your-nlp-model-83dd1b836f46
- WPM의 아이디어를 제시한 논문 : https://static.googleusercontent.com/media/research.google.com/ko//pubs/archive/37842.pdf
- WPM을 사용한 구글의 번역기에 대한 논문 : https://arxiv.org/pdf/1609.08144.pdf
- WPM 참고 자료 : https://norman3.github.io/papers/docs/google_neural_machine_translation.html
- 유니그램 언어 모델을 이용한 단어 분리 : https://arxiv.org/pdf/1804.10959.pdf
- 센텐스피스 사용한 한국어 실습 참고 자료 : https://bab2min.tistory.com/622
- wordpiece Vs. sentencepiece : https://mc.ai/pre-training-bert-from-scratch-with-cloud-tpu/
- https://mlexplained.com/2019/11/06/a-deep-dive-into-the-wonderful-world-of-preprocessing-in-nlp/
