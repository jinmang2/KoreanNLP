# Stemming and Lemmatization

## Definition
- Lemma: 표제어, 기본 사전형 단어
  - ex) am, are, is >> be
- stem(어간): 용언(동사, 형용사)를 활용할 때, 원칙적으로 모양이 변하지 않는 부분.
- ending(어미): 용언의 어간 뒤에 붙어서 활용하면서 변하는 부분이며, 여러 문법적 기능을 수행
  
## Lemmatization
- morphology: 형태학, 형태소로부터 단어들을 만들어가는 학문
- 형태소의 두 가지 종류
  - 어간(stem): 단어의 의미를 담고 있는 단어의 핵심 부분.
  - 접사(affix): 단어에 추가적인 의미를 주는 부분.
- 표제어 추출은 어간 추출과 달리 단어의 형태가 적절히 보존되는 양상을 보이는 특징이 있음
  ```python
  from nltk.stem import WordNetLemmatizer
  n = WordNetLemmatizer()
  words = ['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
  print([n.lemmatize(w) for w in words])
  ```
  ```
  ['policy', 'doing', 'organization', 'have', 'going', 'love', 'life', 'fly', 'dy', 'watched', 'ha', 'starting']
  ```
- 그러나 has를 ha, dies를 dy로 적절치 않은 단어를 출력하고 있음.
- **lemmatizer가 본래 단어의 품사 정보를 알아야만 정확한 결과를 얻을 수 있기 때문**
  ```python
  n.lemmatize('dies', 'v')
  >>> 'die'
  n.lemmatize('watched', 'v')
  >>> 'watch'
  n.lemmatize('has', 'v')
  >>> 'have'
  ```
- 표제어 추출과 어간 추출의 차이?
  - 표제어 추출은 문맥을 고려하며, 수행했을 때의 결과는 해당 단어의 품사 정보를 보존
  - 어간 추출을 수행한 결과는 품사 정보를 보존하지 않음. 정확히는 어간 추출을 한 결과는 사전에 존재하지 않는 단어일 경우가 많음

## Stemming
- 어간 추출은 형태학적 분석을 단순화한 버전이라고 볼 수 있음
- 혹은 정해진 규칙만 보고 단어의 어미를 자르는 어림짐작의 작업
- 즉, 이 작업은 섬세한 작업이 아니기 때문에 어간 추출 후에 나오는 결과 단어는 사전에 존재하지 않는 단어일 수도 있음

### English
- ex) Porter Algorithm; 영어 자연어 처리에서 어간 추출을 한다고 하면 가장 준수한 형태
- ex2) Lancaster Stemmer; 위와 서로 다른 알고리즘을 사용

### Korean
- (1) conjugation(활용): indo-european language에서 주로 볼 수 있는 언어적 특징
  - 용언의 어간(stem)이 어미(ending)를 가지는 일
- (2) 규칙 활용

  ```
  잡/어간 + 다/어미
  ```
  어간이 어미가 붙기전의 모습과 붙은 후의 모습이 같으므로 규칙 기반으로 어미를 단순히 분리해주면 어간 추출이 가능
- (3) 불규칙 활용
  - 어간이 어미를 취할 때 어간의 모습이 바뀌거나 취하는 어미가 특수한 어미일 경우를 말함
  - 듣-, 돕-, 곱-, 잇-, 오르-, 노랗- → 듣/들-, 돕/도우-, 곱/고우-, 잇/이-, 올/올-, 노랗/노라-
  - 오르+ 아/어→올라, 하+아/어→하여, 이르+아/어→이르러, 푸르+아/어→푸르러
  - [한국어 불규칙 활용의 예](https://namu.wiki/w/한국어/불규칙%20활용)
