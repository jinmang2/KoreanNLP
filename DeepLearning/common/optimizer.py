# coding: utf-8
import numpy as np

class SGD:
    
    """
    확률적 경사 하강법 (Stochastic Gradient Descent)
    ; 랜덤하게 추출한 일부 데이터에 대해 가중치를 조절
    """
    
    def __init__(self, lr=1e-2):
        self.lr = lr
        
    def update(self, params, grads):
        for key in params.keys():
            # theta = theta - α*∇f(theta)
            params[key] -= self.lr * grads[key]

class Momentum:
    
    """
    모멘텀 SGD
    ; GD에 관성을 더함. 가정치를 수정하기 이전 수정 방향을 참고
      momentum + gradient으로 actual값을 만듦
    """
    
    def __init__(self, lr=1e-2, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
        
        for key in params.keys():
            # v = m*v - α*∇f(theta)
            # theta = theta + v
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]
            
class Nesterov:
    
    """
    Nesterov's Accelerated Gradient (http://arxiv.org/abs/1212.0901)
    ; momentum이 적용된 지점에서 gradient값이 계산됨
      이동될 방향을 미리 예측, 불필요한 이동을 줄임.
    """
    # NAG는 모멘텀에서 한 단계 발전한 방법 (http://newsight.tistory.com/224
    
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
        
        for key in params.keys():
            # basic formula;
                # v = m*v - α*∇f(theta+m*v)
                # theta = theta + v
            # new formulation in article;
                # v = m*v - α*∇f(theta)
                # theta = theta + m*m*v - (1+m)*α*∇f(theta)
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.momentum * self.momentum * self.v[key]
            parmas[key] -= (1 + self.momuntum) * self.lr * grads[key]
            
class AdaGrad:
    
    """
    Adagrad(Adaptive Gradient)
    ; 변수들을 update할 때 각각의 변수마다 step size를 다르게 설정하여 이동하는 방식.
      지금까지 많이 변화하지 않은 변수들은 step size를 크게하고, 
      지금까지 많이 변화했던 변수들은 step size를 작게하자!
      자주 등장하거나 변화를 많이한 변수의 경우 optimum에 가까이 있을 확률이 높기 때문에
      작은 크기로 이동하면서 세밀한 값을 조정
      word2vec, GloVe같은 word representation을 학습시킬 경우, 단어의 등장 확률에 따라
      variable의 사용 비율이 확연하게 차이나기 때문에 Adagrad와 같은 학습 방식을 사용하면
      훨씬 더 좋은 성능을 거둘 것이다.
      http://shuuki4.github.io/deep%20learning/2016/05/20/Gradient-Descent-Algorithm-Overview.html
    """
    
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = h
        
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
                
        for key in params.keys():
            # G = G + ∇J(theta)^2
            # theta = theta - α*∇J(theta) / (G + ep)^(1/2)
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
            
class RMSprop:

    """
    RMSprop
    ; 제프리 힌튼이 제시한 방법, Adagrad의 단점을 해결하기 위한 방법
      G를 지수평균으로 계산
    """
    
    def __init__(self, lr=0.01, decay_rate = 0.99):
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None
        
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
                
        for key in params.keys():
            # G = γG + (1-γ)(∇J(theta)^2)
            # theta = theta - α*∇J(theta) / (G + ep)^(1/2)
            self.h[key] *= self.decay_rate
            self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)
            
class Adam:
    
    """
    Adam(Adaptive Moment Estimation)
    ; RMSprop + Momentum과 유사. 
      지금까지 계산해온 기울기의 지수평균을 저장, 제곱값의 지수평균을 저장
      (http://arxiv.org/abs/1412.6980v8)
    """
    
    def __init__(self, lr=1e-3, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
                
        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)
        
        for key in params.items():
            # m = b1m + (1-b1)(∇J(theta))
            # v = b2v + (1-b2)(∇J(theta)^2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
