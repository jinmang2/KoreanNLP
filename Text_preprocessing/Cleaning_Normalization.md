# Cleaning and Normalization

## Definition
- tokenization: 코퍼스에서 용도에 맞게 토큰을 분류하는 작업
- cleaning: 갖고 있는 코퍼스로부터 노이즈 데이터를 제거
- normalization: 표현 방법이 다른 단어들을 통합시켜서 같은 단어로 만들어 줌

## Methodologies
### 1. 규칙에 기반한 표기가 다른 단어들의 통합
- 표기가 다른 단어들을 통합하는 방법
  - stemming(어간 추출)
  - lemmatization(표제어 추출)

### 2. 대, 소문자 통합
- 영어권 언어에서 대, 소문자를 통합하는 것은 단어의 개수를 줄일 수 있는 또 다른 정규화 방법

### 3. 불필요한 단어의 제거(Removing Unnecessary Words)
>#### (1) 등장 빈도가 적은 단어(Removing Rare words)
>#### (2) 길이가 짧은 단어(Removing words with very a short length)
>- 한국어에서는 유효하지 않을 수 있음

### 4. 정규 표현식(Regular Expression)
