# Statistical Language Model(SLM)

## 1. 조건부 확률
```
P(A, B) = P(A)P(B|A) = P(B)P(A|B) = P(B, A)

# chain rule of conditional probability
P(x1, x2, x3, ..., xn) = P(x1)P(x2|x1)P(x3|x1, x2)...P(xn|x1, x2, ..., xn-1)
```

## 2. 문장에 대한 확률
```
sentence = 'An adorable little boy is spreading smiles'

P(An adorable little boy is spreading smiles) = 
    P(An) X P(adorable|An) X P(little|An adorable) X P(boy|An adorable little) X 
    P(is|An adorable little boy) X P(spreading|An adorable little boy is) X
    P(smiles|An adorable little boy is spreading)
```

## 3. 카운트 기반의 접근
```
# 문장의 확률을 구하기 위해서 다음 단어에 대한 예측 확률을 모두 곱함
# Then, SLM은 이전 단어로부터 다음 단어에 대한 확률은 어떻게 구할까요?
# 정답은 카운트에 기반하여 확률을 계산

P(is|An adorable little boy) = count(An adorable little boy is) / count(An adorable little boy)
                             = 30 / 100 (만약에) = 30%
```

## 4. 카운트 기반 접근의 한계 - 희소 문제(Sparsity Problem)
- 위와 같이 카운트 기반으로 확률을 계산하려면 방대한 양의 코퍼스가 필요
- 그러나 현실적으로 불가능 (자연어 증말 방대해...)
### Definition
- Sparsity Problem(희소 문제): 충분한 데이터를 관측하지 못하여 언어를 정확히 모델링하지 못한 문제
- 해당 문제를 완화시키는 방법으로 n-gram 혹은 smoothing, back-off와 같은 여러가지 generalization 기법 존재.
- 그러나 근본적인 해결책은 되지 못함 > 언어 모델의 트렌드는 통계적 언어 모델에서 인공 신경망 언어 모델로 넘어가게 됨
