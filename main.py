# main.py — 가스 공급량 예측 FastAPI 서버
# 실행: VS Code 실행 버튼(▶) 또는  python main.py
import math
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error

app = FastAPI()

TRAIN_COL = ['월평균기온', '총인구수']   # 예측 모델 입력 (학습·예측 모두 같은 컬럼)
TARGET = '공급량(㎥)'                    # 모델이 맞히는 값

# ---------------------------------------------------------------
# 서버가 켜질 때 한 번만 실행: 데이터 읽기 → 지역별 모델 학습
# ---------------------------------------------------------------
df = pd.read_csv('가스_인구_기온_통합.csv', encoding='cp949')
df[TARGET] = df[TARGET] / 1000     # /api/regions 와 같은 단위로 맞춤
df = df.sort_values(['시도', '연월'])
df['월'] = df['연월'].str[5:].astype(int)        # '2021-01' → 1

# 시뮬레이션 모델용 컬럼
#   1인당 공급량 = 공급량 ÷ 인구            → 마지막에 인구를 곱하면 인구 변화가 그대로 반영됨
#   난방도일     = 18°C 보다 얼마나 추운지   → 18°C 이상이면 0 (난방 필요 없음)
df['1인당'] = df[TARGET] / df['총인구수']
df['난방도일'] = (18 - df['월평균기온']).clip(lower=0)

models = {}       # 예측용 모델      {'서울': RandomForest, ...}
mapes = {}        # 예측 오차율      {'서울': 0.19, ...}  (신뢰구간 계산용)
sim_models = {}   # 시뮬레이션용 모델 {'서울': LinearRegression, ...}

for region in df['시도'].unique():
    sub = df[df['시도'] == region]

    # 1) 오차율 구하기: 최근 12개월을 빼고 학습한 뒤, 그 12개월을 맞혀 봄
    train_df, test_df = sub.iloc[:-12], sub.iloc[-12:]
    test_model = RandomForestRegressor(n_estimators=100, random_state=42)
    test_model.fit(train_df[TRAIN_COL], train_df[TARGET])
    mapes[region] = mean_absolute_percentage_error(test_df[TARGET], test_model.predict(test_df[TRAIN_COL]))

    # 2) 예측에 쓸 모델: 전체 데이터로 학습
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(sub[TRAIN_COL], sub[TARGET])
    models[region] = model

    # 3) 시뮬레이션에 쓸 모델: 난방도일 → 1인당 공급량 (직선 관계)
    #    RandomForest 는 학습 때 본 기온(약 -5~29°C) 밖에서는 값이 변하지 않아서
    #    슬라이더로 -15°C 를 넣어도 -5°C 와 똑같이 나옴 → 범위 밖에서도 반응하는 선형회귀 사용
    sim_model = LinearRegression()
    sim_model.fit(sub[['난방도일']], sub['1인당'])
    sim_models[region] = sim_model


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


# ---------------------------------------------------------------
# GET /forecast/summary?horizon=6
#   모든 지역의 향후 horizon 개월 예측 합계 (지도 색칠용)
# ---------------------------------------------------------------
@app.get('/forecast/summary')
def forecast_summary(horizon: int = 6):
    items = []
    for region in models:
        fut = forecast(region, horizon)['fut']
        total = sum(f['value'] for f in fut)
        items.append({'region': region, 'total': total})

    return {'items': items}


# ---------------------------------------------------------------
# GET /mape
#   지역별 예측 오차율(MAPE, %) — 화면의 지역 이름 옆 MAPE 배지에 쓰임
#   서버가 켜질 때 계산해 둔 mapes(0.19 같은 비율)를 % 로 바꿔서 돌려줌 (0.19 → 19.0)
# ---------------------------------------------------------------
@app.get('/mape')
def mape_list():
    items = []
    for region in mapes:
        items.append({'region': region, 'mape': round(mapes[region] * 100, 1)})

    return {'items': items}


# ---------------------------------------------------------------
# 시뮬레이션
# ---------------------------------------------------------------
# 요청 body 모양: { "tempLo": -3, "tempHi": 27, "popPct": 0 }
#   BaseModel 을 상속하면 FastAPI 가 JSON body 를 이 객체로 바꿔줌 (Spring 의 @RequestBody + DTO 와 같음)
class SimRequest(BaseModel):
    tempLo: float        # 1월 기온
    tempHi: float        # 8월 기온
    popPct: float = 0    # 인구 변화율(%)


# 반올림 (화면 JS 의 Math.round 와 같은 방식. 파이썬 round() 는 0.5 를 다르게 처리해서 직접 만듦)
def js_round(x):
    return math.floor(x + 0.5)


# 1월 기온(lo)~8월 기온(hi) 사이를 코사인 곡선으로 이어 m월(0=1월)의 기온을 만듦
# ※ 화면의 기온 미리보기 막대(common.js 의 monthTemp)와 똑같은 식 → 화면과 결과가 맞음
def month_temp(lo, hi, m):
    return (lo + hi) / 2 - ((hi - lo) / 2) * math.cos((2 * math.pi * (m - 0.35)) / 12)


# 기온 범위(lo, hi)와 인구로 12개월 공급량 계산
def year_profile(region, lo, hi, pop):
    sim_model = sim_models[region]
    result = []
    for m in range(12):
        temp = month_temp(lo, hi, m)
        hdd = max(0, 18 - temp)                                    # 난방도일
        X = pd.DataFrame({'난방도일': [hdd]})
        per_person = max(0, float(sim_model.predict(X)[0]))        # 1인당 공급량 (음수 방지)
        result.append({'m': m, 'temp': round(temp, 1), 'value': per_person * pop})
    return result


# POST /simulation?region=서울      body: { tempLo, tempHi, popPct }
#   base : 평년 기온 + 현재 인구일 때의 12개월
#   sim  : 입력한 기온 범위 + 인구 변화율을 적용한 12개월
@app.post('/simulation')
def simulation(region: str, req: SimRequest):
    if region not in sim_models:
        raise HTTPException(status_code=404, detail=f'{region} 지역이 없습니다')

    sub = df[df['시도'] == region]
    normal_temp = sub.groupby('월')['월평균기온'].mean()
    last_pop = float(sub['총인구수'].iloc[-1])

    # 기준: 1월·8월 평년 기온을 반올림한 값 (화면 슬라이더의 처음 위치와 같음)
    base_lo = js_round(normal_temp[1])
    base_hi = js_round(normal_temp[8])
    base = year_profile(region, base_lo, base_hi, last_pop)

    # 시나리오: 입력한 기온 + 인구 변화율 적용
    new_pop = last_pop * (1 + req.popPct / 100)
    sim = year_profile(region, req.tempLo, req.tempHi, new_pop)

    return {'base': base, 'sim': sim}


# ---------------------------------------------------------------
# 이 파일을 직접 실행했을 때만 서버를 켠다 (맨 왼쪽에 붙어 있어야 함)
# ---------------------------------------------------------------
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
