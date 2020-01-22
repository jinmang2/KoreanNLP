# N-gram Language Model
- 카운트에 기반한 통계적 접근을 사용하고 있는 SLM 일종
- 이전에 등장한 모든 단어를 고려하는 것이 아니라 일부 단어만 고려하는 접근 방법을 사용

## 1. 코퍼스에서 카운트하지 못하는 경우의 감소
```
P(is|An adorable little boy) ≒ P(is|boy)
P(is|An adorable little boy) ≒ P(is|little boy)

# 이렇게 하여 앞 단어 중 임의의 개수만 포함해서 카운트하여 근사
# 해당 단어의 시퀀스를 카운트할 확률을 높이자!
```

## 2. N-gram
```
unigram: an, adorable, little, boy, is, spreading, smiles
bigram: an adorable, adorable little, little boy, boy is, is spreading, spreading smiles
trigram: an adorable little, adorable little boy, little boy is, boy is spreading, is spreading smiles
4-gram: an adorable little boy, adorable little boy is, little boy is spreading, boy is spreading smiles

# 4-gram이면 예를 들어 spreading 다음에 올 단어를 예측할 때 'little boy is spreading'만 사용!
# 원래는 an adorable little boy is spreading을 전부 사용하자나!!
# n이 작아질 수록 최근의 단어만 보는 거지!! Markov process처럼
```

## 2-1. N-gram 구현
```python
class NGram:
    def __init__(self, target, n=2):
        self.sep = ''
        if self._is_sentence(target):
            target = target.strip().split(' ')
            self.sep = ' '
        self.text = target
        self.n = n
    
    def _is_sentence(self, target):
        if len(target.strip().split(' ')) > 1:
            return True
        else:
            return False
        
    def get(self):
        # zip 구현은 n만큼 객체가 있어야해서 귀찮음
        """List Comprehension"""
        return [self.sep.join(self.text[i:i+self.n])
                for i in range(len(self.text)-(self.n-1))]
    
text = "Hello"
sentence = "this is python script"

NGram(text, 2).get()
>>> ['he', 'el', 'll', 'lo']
NGram(text, 3).get()
>>> ['hel', 'ell', 'llo']
NGram(text, 4).get()
>>> ['hell', 'ello']

NGram(sentence, 2).get()
>>> ['this is', 'is python', 'python script']
NGram(sentence, 3).get()
>>> ['this is python', 'is python script']
NGram(sentence, 4).get()
>>> ['this is python script']
```

## 3. N-gram Language Model의 한계
- n-gram은 뒤의 단어 몇 개만 보다 보니 의도하고 싶은 대로 문장을 끝맺음하지 못하는 경우가 생김
- 결론적으로, 전체 문장을 고려한 언어 모델보다는 정확도가 떨어질 수밖에 없음
- 문제점들!
    - (1) 희소 문제(Sparsity Problem): 여전히 존재
    - (2) n을 선택하는 것은 trade-off 문제
        - n이 낮아지면 훈련 코퍼스에서 카운트는 잘 되겠지만 근사의 정확도는 현실의 확률분포와 멀어짐
        - n을 키우면 모델의 사이즈가 커짐
