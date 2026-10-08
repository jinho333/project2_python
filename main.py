# main.py — 가스 공급량 예측 FastAPI 서버
# 실행: VS Code 실행 버튼(▶) 또는  python main.py     (프로젝트 폴더에서 실행)
#
# ※ 모델 학습은 train.py 에서 함. 이 파일은 저장된 모델을 불러와서 쓰기만 함
#   처음 실행하거나 모델을 바꿨을 때:  python train.py  →  python main.py
import math
import os
import joblib
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prepare import load_data, TARGET

app = FastAPI()

MODEL_PATH = 'model/gas_models.pkl'   # train.py 가 저장하는 파일

# ---------------------------------------------------------------
# 서버가 켜질 때 한 번만 실행: 데이터 읽기 → 저장된 모델 불러오기
# ---------------------------------------------------------------
# 데이터: 최근 실적 12개월, 가장 최근 인구, 월별 평년 기온을 꺼낼 때 사용
df = load_data()   # prepare.py

# 모델 파일이 없으면 무엇을 해야 하는지 알려주고 끝냄
if not os.path.exists(MODEL_PATH):
    raise SystemExit('모델 파일(' + MODEL_PATH + ')이 없습니다. 먼저  python train.py  를 실행하세요.')

# train.py 가 저장한 묶음(딕셔너리)을 불러와서, 예전과 같은 이름의 변수에 담음
#   → 변수 이름이 같아서 아래 API 코드는 바꿀 필요가 없음
saved = joblib.load(MODEL_PATH)
models = saved['models']               # 예측용 모델      {'서울': RandomForest, ...}
sim_models = saved['sim_models']       # 시뮬레이션용 모델 {'서울': LinearRegression, ...}
mapes = saved['mapes']                 # 예측 오차율      {'서울': 0.085, ...}  (신뢰구간 계산용)
recent_mapes = saved['recent_mapes']   # 최근 3개월의 오차율   ┐ 전국 페이지의
before_mapes = saved['before_mapes']   # 그 앞 3개월의 오차율  ┘ 'MAPE 전분기 대비' 계산용
weights = saved['weights']             # 지역별 비중 → 전국 평균을 낼 때 큰 지역에 비중을 더 줌
TRAIN_COL = saved['TRAIN_COL']         # 예측 모델 입력 컬럼 (학습할 때 쓴 것 그대로)
SIM_COL = saved['SIM_COL']             # 시뮬레이션 모델 입력 컬럼


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

    # 지난달 기온: 첫 예측 달은 데이터의 마지막 달 실제 기온, 그 다음부터는 바로 앞 달의 평년 기온
    prev_temp = float(sub['월평균기온'].iloc[-1])

    fut = []
    for i, ym in enumerate(next_months(last_ym, horizon)):
        month = int(ym[5:])
        temp = float(normal_temp[month])
        # 입력 컬럼은 train.py 의 TRAIN_COL 과 같아야 함 (이름·순서 모두)
        X_future = pd.DataFrame({'월평균기온': [temp], '전월기온': [prev_temp], '총인구수': [last_pop], '월': [month]})
        value = float(model.predict(X_future)[0])
        prev_temp = temp    # 다음 달 입장에서는 이번 달 기온이 '지난달 기온'

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


# 지역별 오차율을 전국 값 하나로 합치기: 공급량이 큰 지역에 비중을 더 주는 가중 평균
# ('전국' 행은 17개 시·도를 합친 값이라 중복되므로 제외)
def national_mape(mape_by_region):
    total = 0
    total_weight = 0
    for region in mape_by_region:
        if region == '전국':
            continue
        total += mape_by_region[region] * weights[region]
        total_weight += weights[region]
    return total / total_weight


# ---------------------------------------------------------------
# GET /mape
#   items : 지역별 예측 오차율(MAPE, %) — 지역 이름 옆 MAPE 배지, 전국 페이지 정확도 차트에 쓰임
#           train.py 가 계산해서 저장해 둔 mapes(0.19 같은 비율)를 % 로 바꿔서 돌려줌 (0.19 → 19.0)
#   delta : 전국 MAPE 의 전분기 대비 변화(%p) = 최근 3개월 전국 MAPE - 그 앞 3개월 전국 MAPE
#           음수면 최근 분기에 오차가 줄었다는 뜻
# ---------------------------------------------------------------
@app.get('/mape')
def mape_list():
    items = []
    for region in mapes:
        sub = df[df['시도'] == region].dropna(subset=['전월기온'])
        coef_sum = sim_models[region].coef_.sum()
        mean_per_person = sub['1인당'].mean()                   # ← 지역별 분모
        sens = round(coef_sum / mean_per_person * 100, 1)
        items.append({
            'region': region,
            'mape': round(mapes[region] * 100, 1),
            'sensitivity': sens, #기온 민감도
        })

    delta = (national_mape(recent_mapes) - national_mape(before_mapes)) * 100

    return {'items': items, 'delta': round(delta, 1)}


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
        prev_temp = month_temp(lo, hi, (m - 1) % 12)               # 지난달 기온 (1월의 지난달은 12월)
        hdd = max(0, 18 - temp)                                    # 이번 달 난방도일
        prev_hdd = max(0, 18 - prev_temp)                          # 지난달 난방도일
        X = pd.DataFrame({'난방도일': [hdd], '전월난방도일': [prev_hdd]})
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
