# TF-IDF(Term Frequency-Inverse Document Frequency)

### 1. TF-IDF(단어 빈도-역 문서 빈도)
- TF-IDF = TF X IDF
- 문서: d, 단어: t, 문서의 총 개수: n

### 1-1. TF-IDF 구현
```python
import numpy as np
from scipy.sparse import csr_matrix
import array
from collections import defaultdict

documents = [
    '먹고 싶은 사과',
    '먹고 싶은 바나나',
    '길고 노란 바나나 바나나',
    '저는 과일이 좋아요'
]

class Tokenizer:
    def __init__(self, mode='whitespace'):
        if mode is not 'whitespace':
            raise NotImplementedError('띄어쓰기 토큰화만 구현되있습니다.')
        
    def tokenize(self, document):
        if not isinstance(document, str):
            if isinstance(document, list) or isinstance(documents, np.ndarray):
                result = []
                for doc in document:
                    result.append(self.tokenize(doc))
                return result
            else:
                raise TypeError('list, np.ndarray, str 데이터 타입만 input으로 넣어주세요.')
        else:
            doc = document
            return doc.split(' ')
            
tokenizer = Tokenizer(mode='whitespace')
documents = tokenizer.tokenize(documents)
documents
>>> [['먹고', '싶은', '사과'],
>>>  ['먹고', '싶은', '바나나'],
>>>  ['길고', '노란', '바나나', '바나나'],
>>>  ['저는', '과일이', '좋아요']]

class TfidfVectorizer:
    def __init__(self, documents):
        self.documents = documents
        
    def _calc_dtm(self):
        docs = self.documents
        voca, indptr, indices = [], [], []
        values = array.array(str("i"))
        for doc in docs:
            feature_counter = defaultdict(int)
            for feature in doc:
                if feature not in voca:
                    voca.append(feature)
                feature_counter[voca.index(feature)] += 1
            indices.extend(feature_counter.keys())
            values.extend(feature_counter.values())
            indptr.append(len(indices))
        indices = np.asarray(indices, dtype=np.int32)
        indptr = np.asarray(indptr, dtype=np.int32)
        values = np.frombuffer(values, dtype=np.intc)
        X = csr_matrix((values, indices, indptr),
                       shape=(len(indptr)-1, len(voca)),
                       dtype=np.int32)
        self.voca = voca
        self.indices = indices
        self.values = values
        self.indptr = indptr
        return X
    
    def calc_df(self):
        
    
    def calc_idf(self):
        
```


