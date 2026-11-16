# 가스 공급량 예측 서버 (FastAPI)

[project2](https://github.com/jinho333/project2)(Spring Boot 대시보드)가 호출하는 **가스 공급량 예측 · 시뮬레이션 서버**입니다.
시도별 RandomForest 모델로 다음 달 · 3·6개월 공급량을 예측하고, 기온과 인구를 바꿔 보는 시뮬레이션과 지역별 예측 오차(MAPE)를 제공합니다.

## 한눈에 보기

| | |
|---|---|
| 모델 | 시도별 RandomForest (R² 평균 0.924, 18개 지역 중 17곳에서 XGBoost보다 우세) |
| 입력 | 월평균기온 · **전월기온** · 총인구수 · 월 |
| 예측 오차(MAPE) | 전월 기온 추가로 **18개 지역 평균 16.7% → 8.4%** |
| 속도 | 같은 (지역, 개월) 요청은 캐시, 서버 시작 시 3·6개월 사전 계산 |

## 대구 오차를 줄인 과정

대구의 오차(MAPE)가 60%를 넘었습니다. 공급은 기온보다 **한 달 늦게** 움직여 대구는 전월 기온과의 상관이 −0.96(당월 약 −0.8)로 더 강했습니다.
입력에 전월 기온을 추가했고 대구 62.5% → 7.3%, 경남 23.5% → 5.7%, 강원 24.4% → 9.9%로 낮아졌습니다.
기록: [`train.py`](train.py#L38)

## API

| 메서드 | 경로 | 설명 |
|---|---|---|
| GET | `/forecast?region=&horizon=` | 지역별 예측 (캐시) |
| GET | `/forecast/summary` | 지도 색칠용 요약 |
| GET | `/mape` | 지역별 예측 오차 |
| POST | `/simulation` | 기온 · 인구 조건 시뮬레이션 (기온 → 난방도일 → 선형회귀 → × 인구) |

## 파일

| 파일 | 역할 |
|---|---|
| [`prepare.py`](prepare.py) | 데이터 읽기, `전월기온` 등 컬럼 만들기 |
| [`train.py`](train.py) | 모델 학습 → `model/gas_models.pkl` 저장 |
| [`main.py`](main.py) | 저장된 모델로 예측하는 FastAPI 서버, 예측 캐시([`forecast_cache`](main.py#L66)) |
| `가스_인구_기온_통합.csv` | 학습 데이터 (한국도시가스협회 · 기상청 · 통계청, 2021.01 ~ 2026.06) |

## 실행

```bash
pip install fastapi uvicorn pandas scikit-learn joblib
python train.py    # 처음 또는 모델을 바꿨을 때
python main.py     # http://localhost:8000
```

## 한계

- 데이터가 5~6년이라 해석에 한계가 있고, 울산은 10% 이상(10.1%)으로 남았습니다.
- 울산 2026년 6월 값이 5월과 같아 이전 연도 평균(9,440.7㎥)으로 교정했습니다. 원본 오류가 아니라 교정값이라는 점을 감안해 읽어야 합니다.
- 시뮬레이션은 학습 기온(약 −5~29℃) 밖에서 값이 그대로 나오는 RandomForest 대신 난방도일 선형회귀를 씁니다.

## 기록

- [프로젝트 작업 일지 (Notion)](https://fluffy-plane-19c.notion.site/3e3a969c48278070bbaaf5dee74b31b4)
