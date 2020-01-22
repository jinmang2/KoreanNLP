# Perplexity
- 모델의 평가, extrinsic evaluation(외부 평가) vs **intrinsic evaluation(내부 평가)**

### 1. 언어 모델의 평가 방법(Evaluation metric): PPL
- perplexed; 헷갈리는
- PPL이 높으면 많이 헷갈린다는 의미, 낮으면 모델의 성능이 좋다는 것을 의미
    - 중요한 것은 인간이 봤을 때가 아니라 **모델이 봤을 때 모호함이 떨어진다는 것**
- PPL은 단어의 수로 정규화(normalization)된 테스트 데이터에 대한 확률의 역수
    - PPL(W) = P(w1, w2, w3, ..., wN)^{-1/N}
- Applying chain rule,
    - PPL(W) = Pi_{i=1}^{N}{P(wi|w1, w2, ..., wi-1)^{-1/N}}
- Applyinh bigram,
    - PPL(W) = Pi_{i=1}^{N}{P(wi|wi-1)^{-1/N}}
    
### 2. 분기 계수(Branching factor)
- PPL은 선택할 수 있는 가능한 경우의 수를 의미하는 분기계수.
- 이 언어 모델이 특정 시점에서 평균적으로 몇 개의 선택지를 가지고 고민하고 있는지를 의미
