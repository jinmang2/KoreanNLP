# Implementing Truncated Backpropagation Through Time

## What is Backpropagatoin
- 미분값을 계산하는 수학적인 방법론을 제공, chain rule 적용
- neural network를 학습시킬 수 있음

general algorithm
1. 학습 input을 network에 순전파시킴
2. 실제값과 예측값을 비교하여 오차를 얻음
3. network 가중치에 대한 오차의 미분값을 계산
4. weight에 적용
5. 반복

## What is BPTT?
- Time Series같은 순서를 가진 데이터를 학습하는데 Neural Network에선 `RNN`을 사용.
  - Sequential Modeling에서 `HMM`, `CRF` 등을 사용하여 이런 문제에 접근할 수 있음.
- 이런 RNN을 학습하는데 적용되는 역전파 알고리즘이 `BPTT`, **B**ack**P**ropagation **T**hrough **T**ime

## Why Truncated BPTT?
- BPTT는 time-step이 늘어날수록 연산량이 커짐



## Reference
- [Implement TBPTT with PyTorch](https://discuss.pytorch.org/t/implementing-truncated-backpropagation-through-time/15500)
- [A Gentle Introduction to BPTT](https://machinelearningmastery.com/gentle-introduction-backpropagation-time/)
