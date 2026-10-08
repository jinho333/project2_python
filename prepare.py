# prepare.py — 데이터 읽기 + 컬럼 만들기 (train.py 와 main.py 가 같이 사용)
import pandas as pd

TARGET = '공급량(㎥)'

def load_data():
    df = pd.read_csv('가스_인구_기온_통합.csv', encoding='cp949')
    df[TARGET] = df[TARGET] / 1000
    df = df.sort_values(['시도', '연월'])
    df['월'] = df['연월'].str[5:].astype(int)
    df['전월기온'] = df.groupby('시도')['월평균기온'].shift(1)
    df['1인당'] = df[TARGET] / df['총인구수']
    df['난방도일'] = (18 - df['월평균기온']).clip(lower=0)
    df['전월난방도일'] = (18 - df['전월기온']).clip(lower=0)
    return df