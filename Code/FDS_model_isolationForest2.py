from sklearn.ensemble import IsolationForest

import warnings
from sklearn.exceptions import UndefinedMetricWarning

from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import hashlib
#!pip install scikit-learn --upgrade # 안전한 라이브러리 사용을 위한 최신 버전으로 업그레이드

# UnderfinedMetricWarning 경고 무시
import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings('ignore', category = UndefinedMetricWarning)

x_train = pd.read_csv('./x_train.csv')
y_train = pd.read_csv('./y_train.csv')
x_val = pd.read_csv('./x_val.csv')
y_val = pd.read_csv('./y_val.csv')
x_test = pd.read_csv('./x_test.csv')
y_test = pd.read_csv('./y_test.csv')


print(x_train.head())
print(x_train.info())
print(x_train.describe())


x_train = x_train.drop(['merchant', 'category', 'first', 'last', 'city'], axis = 1)
x_val = x_val.drop(['merchant', 'category', 'first', 'last', 'city'], axis = 1)
x_test = x_test.drop(['merchant', 'category', 'first', 'last', 'city'], axis = 1)


model = IsolationForest(contamination = 0.01,
                        n_estimators = 100,
                        max_samples = 'auto',
                        random_state = 42)


model.fit(x_train)


from sklearn.metrics import classification_report, roc_auc_score

# 검증 데이터셋 예측
y_pred = model.predict(x_val)

# 성능 평가 지표 계산
print(classification_report(y_val, y_pred))

# ROC AUC 점수 계산 (이진 분류 문제의 경우)
roc_auc = roc_auc_score(y_val, y_pred)
print(f"ROC AUC Score: {roc_auc}")

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

import pickle

with open('FDS_model_isolationForest.pkl', 'wb') as f:
    pickle.dump(best_model, f)