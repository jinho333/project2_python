# train.py — 모델 학습 후 파일로 저장
# 실행: VS Code 실행 버튼(▶) 또는  python train.py     (프로젝트 폴더에서 실행)
#
# 언제 실행하나
#   - 처음 프로젝트를 받았을 때 (model/gas_models.pkl 이 없으면 main.py 가 켜지지 않음)
#   - 아래 "튜닝할 값"이나 입력 컬럼을 바꿨을 때
#   - CSV 데이터가 바뀌었을 때
#   → 실행한 뒤 main.py(서버)를 다시 켜야 새 모델이 적용됨
#
# 파일이 나뉜 모습
#   prepare.py : CSV 읽기 + 컬럼 만들기      (train.py 와 main.py 가 같이 사용)
#   train.py   : 학습 + 오차 계산 → 저장     (이 파일. 튜닝은 여기서)
#   main.py    : 저장된 모델을 불러와 API 제공 (학습 코드 없음)
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error
from prepare import load_data, TARGET

# ---------------------------------------------------------------
# ▼ 튜닝할 값은 여기에 모아 둠 (값을 바꾸고 실행하면 맨 아래에 오차가 출력됨)
# ---------------------------------------------------------------
TRAIN_COL = ['월평균기온', '전월기온', '총인구수', '월']   # 예측 모델 입력 (학습·예측 모두 같은 컬럼)
#   ※ 이 목록을 바꾸면 main.py 의 /forecast 에서 X_future 를 만드는 줄도 같이 바꿔야 함
SIM_COL = ['난방도일', '전월난방도일']                # 시뮬레이션 모델 입력

# RandomForest 설정
#   n_estimators : 나무 개수 (많을수록 안정적이지만 느려짐)
#   random_state : 무작위 값을 고정 → 실행할 때마다 같은 결과가 나옴
#   더 넣어 볼 수 있는 값 예) 'max_depth': 5 (나무 깊이 제한), 'min_samples_leaf': 2 (잎의 최소 데이터 수)
RF_PARAMS = {'n_estimators': 100, 'random_state': 42}

MODEL_DIR = 'model'                                  # 저장할 폴더 (.gitignore 에 있어서 git 에는 안 올라감)
MODEL_PATH = MODEL_DIR + '/gas_models.pkl'           # 저장할 파일

# ---------------------------------------------------------------
# [기록] 2026-10-07  '전월기온'(지난달 기온)을 입력에 추가한 이유
# ---------------------------------------------------------------
# 문제: 대구의 예측 오차(MAPE)가 60% 를 넘었음 (다른 지역은 대부분 10~20%)
# 원인: 대구·경남은 공급량이 기온보다 "한 달 늦게" 움직임
#   - 공급량이 가장 많은 달: 대구·경남 2월 / 나머지 지역 1월   (가장 추운 달은 1월)
#   - 공급량이 가장 적은 달: 대구·경남 9월 / 나머지 지역 8월
#   - 대구는 같은 10°C 인데 3월 공급량이 11월의 3.3배 (다른 지역은 1.3~1.5배)
#   - 공급량과의 상관: 대구는 당월 기온 -0.83, 전월 기온 -0.96  ↔  서울은 당월 -0.96, 전월 -0.91
#   → "이번 달 기온"만 넣으면 모델이 봄과 가을을 구분하지 못해 크게 틀림
#   (늦게 움직이는 이유는 데이터만으로는 확정 못 함. 검침·청구 기준 집계일 가능성 → 원본 출처 확인 필요)
# 해결: 이번 달 기온과 지난달 기온을 둘 다 입력으로 넣음 → 지역마다 더 잘 맞는 쪽을 모델이 알아서 사용
# 결과 (최근 12개월을 맞혀 본 MAPE, RandomForest):
#   대구 62.5% → 7.3%,  경남 23.5% → 5.7%,  강원 24.4% → 9.9%,  18개 지역 평균 16.7% → 8.4%
#   시뮬레이션용 선형 모델의 설명력(R²)도 평균 0.93 → 0.98 (대구 0.66 → 0.99)
# ---------------------------------------------------------------

# [기록] 2026-10-08  예측 모델 입력에 '월'(1~12)을 추가
#   이유: 같은 기온이어도 달에 따라 공급량이 다름 (예: 10°C 인 3월과 11월. 겨울 끝 vs 난방 시작)
#         전월기온이 상당 부분 잡아 주지만, 월을 넣으면 남은 차이를 조금 더 잡음
#   검증: 시험하는 1년을 세 번 바꿔서 비교 (전국 가중 MAPE)
#                    최근 1년   그 전 1년   그 전 1년   평균
#     월 없이          8.5%       9.5%       12.5%     10.2%
#     월 추가          7.9%       8.9%       12.2%      9.7%   ← 세 번 모두 나아짐
#     월만(기온 없이)   8.2%      12.8%       16.5%     12.5%   ← 최근 1년만 좋아 보이고 다른 해에서 크게 나빠짐
#   최근 1년 결과: 대구 7.4% → 6.7%, 인천 11.6% → 9.1%, 8% 넘는 지역 12곳 → 9곳
#   참고: 월을 sin·cos 로 바꿔 넣는 방법도 시험했지만(전국 8.4%) 숫자 그대로가 더 좋았음
#         (RandomForest 는 "월이 3 이하인가" 처럼 나누는 방식이라 12월·1월의 거리 문제에 덜 민감)

# [기록] 2026-10-08  학습을 main.py 에서 이 파일로 분리
#   전: main.py 가 켜질 때마다 학습 → 서버가 늦게 켜지고, 튜닝하려면 서버 코드를 건드려야 했음
#   후: 학습은 여기서 하고 결과를 파일로 저장 → main.py 는 불러오기만 함

df = load_data()   # prepare.py : CSV 읽기 + 전월기온, 난방도일 등 컬럼 만들기

models = {}       # 예측용 모델      {'서울': RandomForest, ...}
mapes = {}        # 예측 오차율      {'서울': 0.19, ...}  (신뢰구간 계산용)
recent_mapes = {} # 맞혀 본 12개월 중 최근 3개월의 오차율   ┐ 전국 페이지의
before_mapes = {} # 그 앞 3개월의 오차율                    ┘ 'MAPE 전분기 대비' 계산용
weights = {}      # 지역별 비중 (맞혀 본 12개월의 공급량 합) → 전국 평균을 낼 때 큰 지역에 비중을 더 줌
sim_models = {}   # 시뮬레이션용 모델 {'서울': LinearRegression, ...}

for region in df['시도'].unique():
    # 지난달 기온이 없는 첫 달(2021-01)은 학습에서 제외
    sub = df[df['시도'] == region].dropna(subset=['전월기온'])

    # 1) 오차율 구하기: 최근 12개월을 빼고 학습한 뒤, 그 12개월을 맞혀 봄
    #    **RF_PARAMS : 딕셔너리를 풀어서 n_estimators=100, random_state=42 처럼 넘김
    train_df, test_df = sub.iloc[:-12], sub.iloc[-12:]
    test_model = RandomForestRegressor(**RF_PARAMS)
    test_model.fit(train_df[TRAIN_COL], train_df[TARGET])
    test_pred = test_model.predict(test_df[TRAIN_COL])
    mapes[region] = mean_absolute_percentage_error(test_df[TARGET], test_pred)

    #    같은 12개월을 최근 3개월([-3:])과 그 앞 3개월([-6:-3])로 나눠 오차율을 따로 구해 둠
    recent_mapes[region] = mean_absolute_percentage_error(test_df[TARGET].iloc[-3:], test_pred[-3:])
    before_mapes[region] = mean_absolute_percentage_error(test_df[TARGET].iloc[-6:-3], test_pred[-6:-3])
    weights[region] = float(test_df[TARGET].sum())

    # 2) 예측에 쓸 모델: 전체 데이터로 학습
    model = RandomForestRegressor(**RF_PARAMS)
    model.fit(sub[TRAIN_COL], sub[TARGET])
    models[region] = model

    # 3) 시뮬레이션에 쓸 모델: 이번 달·지난달 난방도일 → 1인당 공급량 (직선 관계)
    #    RandomForest 는 학습 때 본 기온(약 -5~29°C) 밖에서는 값이 변하지 않아서
    #    슬라이더로 -15°C 를 넣어도 -5°C 와 똑같이 나옴 → 범위 밖에서도 반응하는 선형회귀 사용
    sim_model = LinearRegression()
    sim_model.fit(sub[SIM_COL], sub['1인당'])
    sim_models[region] = sim_model

# ---------------------------------------------------------------
# 저장: 모델과 함께 main.py 가 쓰는 값들을 한 묶음(딕셔너리)으로 파일 하나에 저장
#   모델만 저장하면 안 되는 이유: /mape, /forecast 가 오차율·비중도 같이 쓰기 때문
#   입력 컬럼 이름도 같이 저장 → 학습할 때와 예측할 때 컬럼이 어긋나지 않게 함
# ---------------------------------------------------------------
os.makedirs(MODEL_DIR, exist_ok=True)   # model 폴더가 없으면 만듦 (있으면 그대로)
joblib.dump({
    'models': models,
    'sim_models': sim_models,
    'mapes': mapes,
    'recent_mapes': recent_mapes,
    'before_mapes': before_mapes,
    'weights': weights,
    'TRAIN_COL': TRAIN_COL,
    'SIM_COL': SIM_COL,
}, MODEL_PATH)

# ---------------------------------------------------------------
# 결과 출력: 튜닝할 때 이 숫자를 보고 비교
# ---------------------------------------------------------------
print('지역별 예측 오차(MAPE) — 최근 12개월을 맞혀 본 결과')
for region in mapes:
    print(f'  {region}: {mapes[region] * 100:.1f}%')

# 전국 값: 공급량이 큰 지역에 비중을 더 준 평균 ('전국' 행은 17개 시·도를 합친 값이라 제외)
total = 0
total_weight = 0
for region in mapes:
    if region == '전국':
        continue
    total += mapes[region] * weights[region]
    total_weight += weights[region]
print(f'전국 가중 평균: {total / total_weight * 100:.1f}%')
print(f'저장 완료: {MODEL_PATH}  (지역 {len(models)}개)  → main.py 를 다시 켜면 적용됩니다')
