'''
 모델 학습 - 모델 선택
'''

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import hashlib
#!pip install scikit-learn --upgrade # 안전한 라이브러리 사용을 위한 최신 버전으로 업그레이드

# UnderfinedMetricWarning 경고 무시
import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings('ignore', category = UndefinedMetricWarning)



'''
 파일 경로 검증
    1. 화이트리스트 기반 접근 제어
    2. 정규 표현식을 사용한 경로 패턴 검증
    3. 경로 조작 (Path Traversal) 공격 방지
'''


# x_train에 대한 위 3가지 기능을 따로따로 구현한 예시
# x_train에 대한 화이트리스트 기반 접근 제어
"""
import os

ALLOWED_DIR = 'C:/Users/USER/Desktopfds데이터처리코드/'

def validate_filepath(filepath):
    if not filepath.startswith(ALLOWED_DIR):
        return False
    
    real_path = os.path.realpath(filepath)
    return real_path.startswith(ALLOWED_DIR)

file_path = 'C:/Users/USER/Desktopfds데이터처리코드/x_train.csv'

if validate_filepath(file_path):
    x_train = pd.read_csv(file_path)
else:
    raise ValueError('허용되지 않은 파일 경로 입니다.')


# x_train에 대한 정규 표현식을 사용한 경로 패턴 검증
import re

def validate_filepath(filepath):
    pattern = r'^/C:/Users/USER/Desktop/fds데이터처리코드/'
    match = re.match(pattern, filepath)
    return bool(match)

filepath = 'C:/Users/USER/Desktopfds데이터처리코드/x_train.csv'
if validate_filepath(filepath):
    print(f'{filepath} 은(는) 허용된 경로입니다.')
else:
    print(f'{filepath} 은(는) 허용되지 않은 경로입니다.')


# x_train에 대한 경로 조작 공격 방지
def validate_filepath(filepath):
    if ".." in filepath:
        return False
    
    real_path = os.path.realpath(filepath)
    allowed_dir = os.path.abspath('C:/Users/USER/Desktopfds데이터처리코드/')
    return real_path.startswith(allowed_dir)

filepath = 'C:/Users/USER/Desktopfds데이터처리코드/x_train.csv'
if validate_filepath(filepath):
    print(f'{filepath} 은(는) 안전한 경로입니다.')
else:
    print(f'{filepath} 은(는) 안전하지 않은 경로입니다.')

"""

# x_train, y_train, x_val, y_val, x_test, y_test 모든 데이터셋에 대한 화이트리스트, 정규 표현식, 경로 조작 방지 기능을 모두 포함하는 방식
"""
import os
import re

ALLOWED_DIR = 'C:/Users/USER/Desktopfds데이터처리코드/'
ALLOWED_FILENAMES = ['x_train.csv', 'y_train.csv', 'x_val.csv', 'y_val.csv', 'x_test.csv', 'y_test.csv']

def validate_filepath(filepath, filename):
    if not filepath.startswith(ALLOWED_DIR):
        return False
    
    if '..' in filepath:
        return False
    
    real_path = os.path.realpath(filepath)
    if not real_path.startswith(ALLOWED_DIR):
        return False
    
    pattern = r'^' + re.escape(ALLOWED_DIR) + r'[a-zA-Z0-9]+\.csv$'
    if not re.match(pattern, real_path):
        return False
    
    return True

files = {
    'x_train': 'C:/Users/USER/Desktopfds데이터처리코드/x_train.csv',
    'y_train': 'C:/Users/USER/Desktopfds데이터처리코드/y_train.csv',
    'x_val': 'C:/Users/USER/Desktopfds데이터처리코드/x_val.csv',
    'y_val': 'C:/Users/USER/Desktopfds데이터처리코드/y_val.csv',
    'x_test': 'C:/Users/USER/Desktopfds데이터처리코드/x_test.csv',
    'y_test': 'C:/Users/USER/Desktopfds데이터처리코드/y_test.csv'
}

for file_key, file_path in files.items():
    if validate_filepath(file_path, f'{file_key}.csv'):
        print(f'{file_path} 경로가 유효합니다.')
    else:
        print(f'{file_path} 경로가 유효하지 않습니다.')


"""





x_train = pd.read_csv('./x_train.csv')
y_train = pd.read_csv('./y_train.csv')
x_val = pd.read_csv('./x_val.csv')
y_val = pd.read_csv('./y_val.csv')
x_test = pd.read_csv('./x_test.csv')
y_test = pd.read_csv('./y_test.csv')


"""
# 데이터 파일 해시 값 검증

with open('./x_train.csv', 'rb') as f:
    file_hash = hashlib.sha256(f.read()).hexdigest()
if file_hash != 'expected_hash_value':
    raise ValueError('데이터 파일이 손상되었습니다: x_train')

with open('./y_train.csv', 'rb') as f:
    file_hash = hashlib.sha256(f.read()).hexdigest()
if file_hash != 'expected_hash_value':
    raise ValueError('데이터 파일이 손상되었습니다: y_train')

with open('./x_val.csv', 'rb') as f:
    file_hash = hashlib.sha256(f.read()).hexdigest()
if file_hash != 'expected_hash_value':
    raise ValueError('데이터 파일이 손상되었습니다: x_val')

with open('./y_val.csv', 'rb') as f:
    file_hash = hashlib.sha256(f.read()).hexdigest()
if file_hash != 'expected_hash_value':
    raise ValueError('데이터 파일이 손상되었습니다: y_val')

with open('./x_test.csv', 'rb') as f:
    file_hash = hashlib.sha256(f.read()).hexdigest()
if file_hash != 'expected_hash_value':
    raise ValueError('데이터 파일이 손상되었습니다: x_test')

with open('./y_test.csv', 'rb') as f:
    file_hash = hashlib.sha256(f.read()).hexdigest()
if file_hash != 'expected_hash_value':
    raise ValueError('데이터 파일이 손상되었습니다: y_test')

"""


# 데이터 확인
print(x_train.head())
print(x_train.info())
print(x_train.describe())


'''# 문자열(object) 데이터 타입을 가진 열 확인
object_cols = x_train.select_dtypes(include=['object']).columns

# object 열 출력
print(f'문자열 값을 가진 열: {object_cols}')

# 문자열 처리(One-Hot Encoding)
# pandas get_dummies를 사용한 One-Hot Encoding
for col in object_cols:
    if col in x_train.columns:
        x_train = pd.get_dummies(x_train, columns = [col])'''



# 문자열 데이터 처리 - 입력값 검증 (시큐어 코딩)
# one-hot encoding 부분에 입력값 검증을 추가하여 안전성 확보


'''
# 문자열(object) 데이터 타입을 가진 열 확인
object_cols = x_train.select_dtypes(include = ['object']).columns

for col in object_cols:
    if col in x_train.columns:
        for value in x_train[col].unique():
            if not isinstance(value, str):
                raise TypeError(f"'{col}' 컬럼의 값은 문자열이어야 합니다. 값: {value}")
            if len(value) > 100:
                raise ValueError(f"'{col}' 컬럼의 길이는 최대 100자까지 허용됩니다. 값: {value}")
            if not value.isalnum():
                raise ValueError(f"'{col}' 컬럼의 값은 영어나 숫자만 허용됩니다. 값: {value}")
            
        x_train = pd.get_dummies(x_Train, columns = [col])

'''


# 문자열 데이터 처리 -인코딩 및 디코딩 (시큐어 코딩)
'''
import sklearn.preprocessing import LabelEncoder

object_cols = x_train.select_dtypes(include = ['object']).columns

label_encoders = {}
for col in object_cols:
    if col in x_train.columns:
        le = LabelEncoder()
        x_train[col] = le.fit_transform(x_train[col])
        label_encoders[col] = le

# (모델 학습 및 평가 단계 이후)

for col in object_cols:
    if col in x_train.columns:
        le = label_encoders[col]
        x_train[col] = le.inverse_transform(x_train[col])

'''

# 문자열 데이터 처리 - 데이터 베이스 연동 (시큐어 코딩)
'''
import sqlite3

conn = sqlite3.connect('mydatabase.db')
cursor = conn.cursor()

user_input = input('검색할 이름을 입력하세요: ')

query = 'SELECT * FROM users WHERE name = ?'

results = cursor.fetchall()

conn.close()

'''






# 데이터 타입 확인 (시큐어 코딩)
'''

with open('./x_train.csv', 'r') as f:
    columns = f.readline().strip().split(',')

expected_dtypes = {} # 예상 데이터 타입 설정 (임의로 설정)
for col in columns:
    if 'id' in col.lower():
        expected_dtypes[col] = pd.Int64Dtype()

    elif 'date' in col.lower():
        expected_dtypes[col] = pd.StringDtype() # 날짜형은 문자형으로 처리

    else:
        expected_dtypes[col] = pd.Float64Dtype()

x_train = pd.read_csv('./x_train.csv')

for col in x_train.columns:
    if x_train[col].dtype != expected_dtypes.get(col):
        print(f"Warning: '{col}' 컬럼의 데이터 타입이 예상과 다름니다. "
              f"예상: {expected_dtypes.get(col)}, 실제: {x_train[col].dtype}")


'''



# 범주형 변수 선택
cat_cols = ['category', 'merchant', 'city']

# OneHotEncoder 객체 생성
encoder = OneHotEncoder(handle_unknown = 'ignore', sparse_output = False)

# 학습 데이터에 fit_transform 적용
x_train_encoded = pd.DataFrame(encoder.fit_transform(x_train[cat_cols]))
x_train_encoded.index = x_train.index # 인덱스 유지

# 검증 및 테스트 데이터에는 transform만 적용
x_val_encoded = pd.DataFrame(encoder.transform(x_val[cat_cols]))
x_val_encoded.index = x_val.index

x_test_encoded = pd.DataFrame(encoder.transform(x_test[cat_cols]))
x_test_encoded.index = x_test.index

# 인코딩된 범주형 데이터와 기존의 수치형 데이터를 합침
x_train = pd.concat([x_train.drop(cat_cols, axis = 1), x_train_encoded], axis = 1)
x_val = pd.concat([x_val.drop(cat_cols, axis = 1), x_val_encoded], axis = 1)
x_test = pd.concat([x_test.drop(cat_cols, axis = 1), x_test_encoded], axis = 1)


# 피처 이름을 문자열로 변환
x_train.columns = x_train.columns.astype(str)
x_val.columns = x_val.columns.astype(str)
x_test.columns = x_test.columns.astype(str)


# 남아있는 문자열 데이터 확인
print(x_train.select_dtypes(include=['object']).head())

# 필요 없는 문자열 컬럼 삭제
x_train = x_train.drop(columns = ['first', 'last'], errors = 'ignore')
x_val = x_val.drop(columns = ['first', 'last'], errors = 'ignore')
x_test = x_test.drop(columns = ['first', 'last'], errors = 'ignore')



print(x_train.select_dtypes(include=['object']).head())



# 모델 학습 - 하이퍼파라미터 검증 (시큐어 코딩)

'''
#contamination 값 가져오기
contamination = param_grid['contamination'][0]

if not isinstance(contamination, float) or not 0<= contamination <= 0.5:
    raise ValueError('contamination 값이 올바르지 않습니다.')

'''



# 모델 학습 - 적대적 공격 방어 (시큐어 코딩)
'''
import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np

x_train = pd.read_csv('./x_train.csv')

model = IsolationForest(contamination = 0.01, n_estimators = 100, max_samples = 'auto', random_state = 42)

# 적대적 예제 생성 함수
def generate_adversarial_examples(x, model, eps = 0.1):
    x_adv = x.copy()
    anomaly_scores = model.decision_function(x)
    for i in range(len(x)):
        if anomaly_scores[i] < 0:
            for j in range(x.shape[1]):
                x_adv.iloc[i, j] += eps * np.random.randn()
    return x_adv

# 적대적 예제 생성
x_train_adv = generate_adversarial_examples(x_train, model)

model.fit(x_train_adv)

'''

# 모델 학습 - 데티어 포이즈닝 방어 (시큐어 코딩)
'''
from sklearn.ensemble import IsolationForest

iso_forest = IsolationForest(contamination = 0.01)

iso_forest.fit(x_train)

outliers = iso_forest.predict(x_train)

x_train_clean = x_train[outliers == 1]

'''


# 모델 학습 - 모델 추출 방어 (쿼리 횟수 제한) (시큐어 코딩)
'''
QUERY_LIMIT = 1000

query_count = 0

def api_query(input):
    global query_countif
    if query_count <QUERY_LIMIT:
        query_count += 1
        output = model.predict(input)
        return output
    else:
        return 'Query limit exceede'

'''

# 모델 학습 - 모델 추출 방어 (Differential Privacy) (시큐어 코딩)
'''
from sklearn.ensemble import IsolationForest
import numpy as np

def add_noise(anomaly_scores, epsilon):
    noise = np.random.laplace(0, 1 / epsilon, len(anomaly_scores))
    noisy_scores = anomaly_scores + noise
    return noisy_scores

model = IsolationForest(contamination = 0.01, n_estimators = 100, max_samples = 'auto', random_state = 42)

model.fit(x_train)

anomaly_scores = model.decision_function(x_train)

epsilon = 0.1
noisy_scores = add_noise(anomaly_scores, epsilon)

'''


# 모델 학습 - 모델 추출 방어 (Model Watermarking - Anomaly Detection 모델 변형) (시큐어 코딩)
'''
from sklearn.ensemble import IsolationForest

def insert_watermark(mode, watermark_data):
    model.watermark_data = watermark_data
    return model

def verify_watermark(model, watermark_data):
    return hasattr(model, 'watermark_data') and model.watermark_data == watermark_data

model = IsolationForest(contamination = 0.01, n_estimators = 100, max_samples = 'auto', random_state = 42)

watermark_data = np.random.rand(10)

model = insert_watermark(model, watermark_data)

model.fit(x_train)

if verify_watermark(model, watermark_data):
    print('Watermark verified')
else:
    print('Watermark not found')

'''

# 모델 학습 - 모델 추출 방어 (Adversarial Training) (시큐어 코딩)
'''
import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np

def generate_adversarial_examples(x, model, eps = 0.1):
    x_adv = x.copy()
    anomaly_scores = model.decision_function(x)
    for i in range(len(x)):
        if anomaly_scores[i] < 0:
            x_adv.iloc[i, j] += eps * np.random. randn()
    return x_adv

model = IsolationForest(contamination = 0.01, n_estimators = 100, max_samples = 'auto', random_state = 42)

x_train_adv = generate_adversarial_examples(x_train, model)

model.fit(x_train_adv)


'''




'''
 모델 학습 - 모델 초기화
'''

# 모델 초기화
model = IsolationForest(contamination = 0.01,
                        n_estimators = 100,
                        max_samples = 'auto',
                        random_state = 42)










'''
 모델 학습 - 모델 학습
'''

# 모델 학습
model.fit(x_train)









'''
 모델 검증 - 예측 수행 및 성능 평가
'''





from sklearn.metrics import classification_report, roc_auc_score

# 검증 데이터셋 예측
y_pred = model.predict(x_val)

# 성능 평가 지표 계산
print(classification_report(y_val, y_pred))

# ROC AUC 점수 계산 (이진 분류 문제의 경우)
roc_auc = roc_auc_score(y_val, y_pred)
print(f"ROC AUC Score: {roc_auc}")


"""
# 평가 지표 검증
allowed_metrics = ['roc_auc_score', 'accuracy_score', 'f1_score']
if metric not in allowed_metrics:
    raise ValueError('허용되지 않는 평가 지표입니다.')

"""



'''
 모델 검증 - 하이퍼파라미터 튜닝
'''


"""
from sklearn.model_selection import GridSearchCV


# 튜닝할 하이퍼파라미터 범위 설정
param_grid = {
    'contamination': [0.005, 0.01, 0.05],
    'n_estimators': [50, 100, 150],
    'max_samples': ['auto', 0.5, 0.8]
}






# GridSearchCV 객체 생성
grid_search = GridSearchCV(estimator=IsolationForest(random_state=42),
                           param_grid=param_grid,
                           scoring=lambda estimator, X: roc_auc_score(y_train, estimator.predict(X)),
                           cv=2)

# GridSearchCV 수행
grid_search.fit(x_train)

# 최적의 하이퍼파라미터 출력
print(f"Best parameters: {grid_search.best_params_}")

# 최적의 모델
best_model = grid_search.best_estimator_


"""

# IsolationForest 모델을 직접 사용하여 하이퍼파라미터 튜닝

# 최적의 하이퍼파라미터 및 ROC AUC 점수 저장 변수
best_params = {}
best_roc_auc = 0

# 하이퍼파라미터 범위 설정
param_grid = {
    'contamination': [0.005, 0.01, 0.05],
    'n_estimators': [50, 100, 150],
    'max_samples': ['auto', 0.5, 0.8]
}

# 각 하이퍼파라미터 조합에 대해 모델 학습 및 평가
for contamination in param_grid['contamination']:
    for n_estimator in param_grid['n_estimators']:
        for max_samples in param_grid['max_samples']:
            # IsolationForest 모델 초기화
            model = IsolationForest(contamination = contamination,
                                    n_estimators = n_estimator,
                                    random_state = 42)
            
            # 모델 학습 및 예측
            y_pred = model.fit_predict(x_train)

            # ROC AUC 점수 계산
            roc_auc = roc_auc_score(y_train, y_pred)

            # 최적의 하이퍼파라미터 및 ROC AUC 점수 업데이트
            if roc_auc > best_roc_auc:
                best_roc_auc = roc_auc
                best_params = {'contamination': contamination,
                               'n_estimators': n_estimator,
                               'max_samples': max_samples}
                
# 최적의 하이퍼파라미터 출력
print(f"Best parameters: {best_params}")
print(f"Best ROC AUC score: {best_roc_auc}")


# 최적의 모델
best_model = IsolationForest(contamination = best_params['contamination'],
                             n_estimators = best_params['n_estimators'],
                             max_samples = best_params['max_samples'],
                             random_state = 42)


'''
 모델 테스트 - 예측 수행 및 최종 성능 평가
'''

# 최적의 모델 학습
best_model.fit(x_train)

# 테스트 데이터셋 예측
y_pred = best_model.predict(x_test)

# ROC AUC 점수 계산
roc_auc = roc_auc_score(y_test, y_pred)
print(f"ROC AUC score: {roc_auc}")


'''
# 모델 평가 - 평가 지표 검증 (시큐어 코딩)

allowed_metrics = ['roc_auc_score', 'accuracy_score', 'f1_score']
   if metric not in allowed_metrics:
       raise ValueError('허용되지 않는 평가 지표입니다.')


# 모델 평가 - 데이터 검증 (시큐어 코딩)

import hashlib

   with open('x_train.csv', 'rb') as f:
       file_hash = hashlib.sha256(f.read()).hexdigest()

   if file_hash != 'expected_hash_value':
       raise ValueError('데이터 파일이 손상되었습니다.')


# 모델 평가 - 입력값 검증 (시큐어 코딩)

from sklearn.metrics import roc_auc_score

   # 입력값 검증
   if not isinstance(y_true, (np.ndarray, pd.Series)):
       raise TypeError('y_true must be a numpy array or pandas Series')
   if not isinstance(y_score, (np.ndarray, pd.Series)):
       raise TypeError('y_score must be a numpy array or pandas Series')
   
   roc_auc = roc_auc_score(y_true, y_score)

'''


'''
 모델 배포 - API 구축
'''
'''
# 모델 배포 - 입력 데이터 검증 (시큐어 코딩)

def post(self, request):
    data = request.data

    if not isinstance(data, dict):
        raise ValueError('Invalid input data format')
    

# 모델 배포 - 인증 및 권한 부여 (시큐어 코딩)

from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated

class PredictView(APIView):
    authentication_classes = [TokenAuthentication]
    permission_classes = [IsAuthenticated]


# 모델 배포 - 오류 처리 (시큐어 코딩)

try:
    prediction = model.predict([data])
except Exception as e:
    return Response({'error': 'Prediction failed'}, status = 500)


# 모델 배포 - API 로깅 (시큐어 코딩)

import logging

logger = logging.getLogger(__name__)

class PredictView(APIView):
    def post(self, request):
        logger.info(f'Received request from {request.user.username}')

'''













