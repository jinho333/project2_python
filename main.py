# main.py — 가스 공급량 예측 FastAPI 서버
# 실행: uvicorn main:app --reload --port 8000
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error

app = FastAPI()

TRAIN_COL = ['월평균기온', '총인구수']   # 모델 입력 (학습·예측 모두 같은 컬럼)
TARGET = '공급량(㎥)'                    # 모델이 맞히는 값

# ---------------------------------------------------------------
# 서버가 켜질 때 한 번만 실행: 데이터 읽기 → 지역별 모델 학습
# ---------------------------------------------------------------
df = pd.read_csv('가스_인구_기온_통합.csv', encoding='cp949')
df = df.sort_values(['시도', '연월'])
df['월'] = df['연월'].str[5:].astype(int)        # '2021-01' → 1

models = {}   # {'서울': 모델, ...}
mapes = {}    # {'서울': 0.19, ...}  예측 오차율 (신뢰구간 계산용)

for region in df['시도'].unique():
    sub = df[df['시도'] == region]

    # 1) 오차율 구하기: 최근 12개월을 빼고 학습한 뒤, 그 12개월을 맞혀 봄
    train_df, test_df = sub.iloc[:-12], sub.iloc[-12:]
    test_model = RandomForestRegressor(n_estimators=100, random_state=42)
    test_model.fit(train_df[TRAIN_COL], train_df[TARGET])
    mapes[region] = mean_absolute_percentage_error(test_df[TARGET], test_model.predict(test_df[TRAIN_COL]))

    # 2) 실제 예측에 쓸 모델: 전체 데이터로 학습
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(sub[TRAIN_COL], sub[TARGET])
    models[region] = model


# '2026-06' 다음 달부터 n개의 연월 만들기 → ['2026-07', '2026-08', ...]
def next_months(last_ym, n):
    year, month = int(last_ym[:4]), int(last_ym[5:])
    result = []
    for _ in range(n):
        month += 1
        if month > 12:          # 12월 다음은 다음 해 1월
            month = 1
            year += 1
        result.append(f'{year}-{month:02d}')
    return result


# ---------------------------------------------------------------
# GET /forecast?region=서울&horizon=6
#   region  : 지역 이름 (CSV 의 '시도' 값)
#   horizon : 예측할 개월 수 (3 또는 6)
# 응답 모양은 화면(forecast.js)이 쓰는 그대로 { hist: [...], fut: [...] }
# ---------------------------------------------------------------
@app.get('/forecast')
def forecast(region: str, horizon: int = 6):
    if region not in models:
        raise HTTPException(status_code=404, detail=f'{region} 지역이 없습니다')

    sub = df[df['시도'] == region]
    model = models[region]
    mape = mapes[region]

    # hist: 최근 실적 12개월
    hist = []
    for _, row in sub.iloc[-12:].iterrows():
        hist.append({
            'label': row['연월'][2:].replace('-', '.'),   # '2025-07' → '25.07'
            'm': int(row['월']) - 1,                      # 화면은 0 = 1월
            'value': float(row[TARGET])
        })

    # fut: 앞으로 horizon 개월 예측
    normal_temp = sub.groupby('월')['월평균기온'].mean()   # 월별 평년 기온
    last_pop = sub['총인구수'].iloc[-1]                    # 가장 최근 인구
    last_ym = sub['연월'].iloc[-1]                         # 데이터의 마지막 달

    fut = []
    for i, ym in enumerate(next_months(last_ym, horizon)):
        month = int(ym[5:])
        temp = float(normal_temp[month])
        X_future = pd.DataFrame({'월평균기온': [temp], '총인구수': [last_pop]})
        value = float(model.predict(X_future)[0])

        # 신뢰구간: 오차율만큼 위아래로. 먼 달일수록 조금씩 넓힘
        err = mape * (1 + i * 0.1)
        fut.append({
            'label': ym[2:].replace('-', '.'),
            'm': month - 1,
            'temp': round(temp, 1),
            'value': value,
            'lo': value * (1 - err),
            'hi': value * (1 + err)
        })

    return {'hist': hist, 'fut': fut}

# GET /forecast/summary?horizon=6
#   모든 지역의 향후 horizon 개월 예측 합계
@app.get('/forecast/summary')
def forecast_summary(horizon: int = 6):
    items = []
    for region in models:
        # 위에서 만든 forecast() 를 그대로 써서 예측값을 받아 합계를 냄
        fut = forecast(region, horizon)['fut']
        total = sum(f['value'] for f in fut)
        items.append({'region': region, 'total': total})

    return {'items': items}

#------------------------------------------
if __name__ == '__main__':
  uvicorn.run(app)
