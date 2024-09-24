import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('./FDS-Trnasaction(Lat,Long).csv', encoding='utf-8')


# 거래 날짜 및 시간 컬럼을 datetime 객체로 변환
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])

# 기존 거래 날짜 및 시간 컬럼을 연도, 월, 일, 시간, 분, 초로 세분화 하여 컬럼 추출
df[['year', 'month', 'day', 'hour', 'minute', 'second']] = df['trans_date_trans_time'].apply(lambda x: pd.Series([x.year, x.month, x.day, x.hour, x.minute, x.second]))

# 요일 컬럼 추출 (0: 월요일 ~ 6: 일요일)
df['weekday'] = df['trans_date_trans_time'].dt.weekday

# 주말 여부 컬럼 생성 (1: 주말, 0: 평일)
df['weekend'] = df['weekday'].apply(lambda x: 1 if x >= 5 else 0)


df.drop(['Unnamed: 0', 'Unnamed: 2'], axis=1, inplace=True)

df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df.set_index('trans_date_trans_time', inplace = True)

df.resample('M').count().plot()

numeric_df = df.select_dtypes(include=['number'])

# 거리 계산 (위도/경도 기반)
df['lat_diff'] = abs(df['lat'] - df['merch_lat'])
df['long_diff'] = abs(df['long'] - df['merch_long'])

df['is_fraud_distance'] = ((df['lat_diff'] >= 0.45) | (df['long_diff'] >= 0.6)).astype(int)

df = df[~df['category'].str.contains('net')]

df['is_fraud_distance'].sum()

df.dropna(inplace=True)

df['amt'] = df['amt'].astype(int)


from scipy import stats

df['amt_zscore'] = np.abs(stats.zscore(df['amt']))

threshold = 3

df = df[(df['amt_zscore'] < threshold)]


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# 혼동 행렬 계산
cm = confusion_matrix(df['is_fraud'], df['is_fraud_distance'])

# 성능 지표 계산
accuracy = accuracy_score(df['is_fraud'], df['is_fraud_distance'])
precision = precision_score(df['is_fraud'], df['is_fraud_distance'])
recall = recall_score(df['is_fraud'], df['is_fraud_distance'])
f1 = f1_score(df['is_fraud'], df['is_fraud_distance'])

# 결과 출력
print('혼동 행렬: \n', cm)
print('정확도:', accuracy)
print('정밀도:', precision)
print('재현율:', recall)
print('F1-score:', f1)


new_fraud = df[(df['is_fraud'] == 0) & (df['is_fraud_distance'] == 0)]


Q1 = df['amt'].quantile(0.25)
Q3 = df['amt'].quantile(0.75)
IQR = Q3 - Q1

# IQR을 이용한 이상치 범위 설정
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# 이상치 제거
df = df[(df['amt'] >= lower_bound) & (df['amt'] <=upper_bound)]



from sklearn.preprocessing import StandardScaler

# 수치형 변수 선택
num_cols = ['amt', 'lat', 'long', 'lat_diff', 'long_diff', 'year', 'month', 'day', 'hour', 'minute', 'second']

# StandardScaler 객체 생성

scaler = StandardScaler()

# 데이터 표준화
df[num_cols] = scaler.fit_transform(df[num_cols])




df['transaction_per_hour'] = df.groupby('hour')['amt'].transform('count')


df['avg_amt_per_weekday'] = df.groupby('weekday')['amt'].transform('mean')


# 고객 ID 생성
df['customer_id'] = df.groupby(['lat', 'long']).ngroup()

# 고객별 거래 횟수 계산
df['transaction_per_customer'] = df.groupby('customer_id')['amt'].transform('count')


# 가맹점 ID 생성
df['merchant_id'] = df.groupby(['merch_lat', 'merch_long']).ngroup()

# 가맹점별 거래 금액 평균 계산
df['avg_amt_per_merchant'] = df.groupby('merchant_id')['amt'].transform('mean')



x = df.drop('is_fraud', axis = 1)
y = df['is_fraud']

from sklearn.model_selection import train_test_split

# 학습 데이터와 나머지 데이터 분할(8:2)
x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size = 0.2, random_state = 42)

# 나머지 데이터를 검증 데이터와 테스트 데이터로 분할 (5:5)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size = 0.5, random_state = 42)

print('df check:\n', df.columns)







'''from sklearn.preprocessing import OneHotEncoder

# 범주형 변수 선택
cat_cols = ['category', 'merchant', 'city']

# OntHotEncoder 객체 생성
encoder = OneHotEncoder(handle_unknown = 'ignore', sparse_output = False)

# One-Hot Encoding 적용
encoded_df = pd.DataFrame(encoder.fit_transform(df[cat_cols]))

# 인덱스 재설정
encoded_df.index = df.index

# 기존 데이터프레임과 병합
df = pd.concat([df, encoded_df], axis = 1)

# 기존 범주형 변수 제거
df.drop(cat_cols, axis = 1, inplace = True)

# first, last 컬럼 제거
df.drop(['first', 'last'], axis = 1,  inplace = True)

print('encoding:\n')
print('encoding:\n')
print('encoding:\n')
print('encoding:\n')
print('encoding:\n', encoded_df)'''








'''# One-Hot Encoding으로 생성된 변수 이름 변경
for i, col in enumerate(encoded_df.columns):
  if 'category_' in str(col):
    df.rename(columns = {col: f'category_{encoder.categories_[0][i]}'}, inplace = True)
  elif 'merchant_' in str(col):
    df.rename(columns = {col: f'merchant_{encoder.categories_[1][i]}'}, inplace = True)
  elif 'city_' in str(col):
    df.rename(columns = {col: f'city_{encoder.categories_[2][i]}'}, inplace = True)


# 데이터 타입 조정

# One-Hot Encoding으로 생성된 변수 타입 변경
for col in df.columns:
  if 'category_' in str(col) or 'merchant_' in str(col) or 'city_' in str(col):
    df[col] = df[col].astype(int)

# 0만 있는 컬럼 확인
zero_cols = df.columns[(df == 0).all()]

# 컬럼 삭제
df = df.drop(zero_cols, axis=1)

print('encoding222:\n')
print('encoding222:\n')
print('encoding222:\n')
print('encoding222:\n')
print('encoding222:\n', df)'''

print('최종 훈련 데이터 셋:')
print(x_train.head())
print(y_train.head())

print('\n최종 검증 데이터 셋:')
print(x_val.head())
print(y_val.head())

print('\n최종 테스트 데이터 셋:')
print(x_test.head())
print(y_test.head())


# 훈련 데이터셋 저장
x_train.to_csv('x_train.csv', index = False)
y_train.to_csv('y_train.csv', index = False)

# 검증 데이터셋 저장
x_val.to_csv('x_val.csv', index = False)
y_val.to_csv('y_val.csv', index = False)

# 테스트 데이터셋 저장
x_test.to_csv('x_test.csv', index = False)
y_test.to_csv('y_test.csv', index = False)


print()
print()
print()
print()
print()
print()


print(x_train.columns)
print(y_train.columns)
print(x_val.columns)
print(y_val.columns)
print(x_test.columns)
print(y_test.columns)