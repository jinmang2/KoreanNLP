# Tokenization

## Definition
- 주어진 코퍼스(corpus)에서 토큰(token)이라 불리는 단위로 나누는 작업
- token: a string of contiguous characters between two spaces, or between a space and punctuation marks.
- corpus: a collection of written texts, especially the entire words of a particular author or a body of writing on a particular subject
- punctuation: 온점(.), 컴마(,), 물음표(?), 세미콜론(;), 느낌표(!) 등과 같은 기호를 말함
- whitespace: 띄어쓰기

## Word Tokenization
### Tokenizing Issue
- 구두점이나 특수문자를 전부 제거하면 토큰이 의미를 잃어버리는 경우 발생
- 한국어는 띄어쓰기 단위로 토큰화할 시 구분하기 어려운 문제가 있음

### Consideration
- 구두점이나 특수 문자를 단순 제외해서는 안 된다.
- 줄임말과 단어 내에 띄어쓰기가 있는 경우
  - clitic(접어), 흠좀무/존잘
- 표준 토큰화 예제 (English)
  - rule 1. 하이푼으로 구성된 단어는 하나로 유지한다.
  - rule 2. doesn't와 같이 아포스트로피로 '접어'가 함께하는 단어는 분리해준다.

## Sentence Tokenization
- 다른 말로 문장 분류(Sentence Segmentation)
- 온점이 문장의 끝? 아니다.
  - ex)
  ```
  EX1) IP 192.168.56.31 서버에 들어가서 로그 파일 저장해서 ukairia777@gmail.com로 결과 좀 보내줘. 그러고나서 점심 먹으러 가자.

  EX2) Since I'm actively looking for Ph.D. students, I get the same question a dozen times every year.
  ```
- 영어  : `from nltk.tokenize import sent_tokenize`
- 한국어: `from kss import split_sentences` 박상길님이 개발한 KSS(Korean Sentence Splitter)
- 온 점의 처리를 위해 이진 분류기를 사용하기도 함
  - 온점(.)이 단어의 일부분일 경우. 즉, 온점이 약어(abbreivation)로 쓰이는 경우
  - 온점(.)이 정말로 문장의 구분자(boundary)일 경우
- https://tech.grammarly.com/blog/posts/How-to-Split-Sentences.html

## 한국어 토큰화의 어려움
1. 한국어는 교착어
- 형태소(Morpheme): 뜻을 가진 가장 작은 말의 단위
  - 자립 형태소: 접사, 어미, 조사와 상관없이 자립하여 사용할 수 있는 형태소. 그 자체로 단어가 됨. 체언(명사, 대명사, 수사), 수식언(관형사, 부사), 감탄사 등이 있음
  - 의존 형태소: 다른 형태소와 결합하여 사용되는 형태소. 접사, 어미, 조사, 어간을 말함
2. 한국어는 띄어쓰기가 영어보다 잘 지켜지지 않음
- 띄어쓰기가 없던 한국어에 띄어쓰기가 보편화된것도 근대(1933년, 한글맞춤법통일안)의 일
- 한국어(모아쓰기) vs 영어(풀어쓰기)
