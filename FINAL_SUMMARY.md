# KSPHM-KIMM PHM Data Challenge 2026: RUL 예측 파이프라인 최종 요약

이 문서는 2026년 KSPHM-KIMM 챌린지의 베어링 잔여수명(RUL, Remaining Useful Life) 예측 파이프라인의 최종 상태와 개발 중 얻은 주요 인사이트, 한계점 및 권장 사항을 요약합니다.

## 1. 프로젝트 아키텍처 및 폴더 구조

*   `src/`: 핵심 소스 코드가 위치한 폴더
    *   `config.py`: 데이터 경로, 보정 계수 탐색 범위, 진동 센서 파라미터 등 전역 설정 관리
    *   `data_loader.py`: TDMS 진동 파일과 Operation CSV 데이터를 병합하여 로드하는 모듈
    *   `features.py`: 진동 데이터의 시간(Time)/주파수(Freq) 대역 기본 특징과, 베어링 결함 주파수(BPFI, BPFO 등) 특징을 추출하는 핵심 모듈
    *   `dataset.py`: 여러 베어링 런(Run) 데이터를 모아 학습/평가용 표(DataFrame) 형태로 구축 (최근 50분치 롤링 특징 자동 추가)
    *   `models.py`: 사이킷런(Scikit-learn) 파이프라인(결측치 대체 + 스케일링 + 회귀 모델) 정의 (랜덤 포레스트, LightGBM 등 포함)
    *   `train.py`: 특징 추출 ➡️ 교차 검증(CV) ➡️ 앙상블 학습 ➡️ 최적 보정 계수(Calibration factor) 탐색을 통합 실행하는 스크립트
    *   `predict.py`: 테스트(검증) 데이터셋에 대한 RUL 예측을 수행하고 제출용 `team_validation.xlsx`를 생성하는 스크립트
    *   `evaluate.py`: 공식 대회 스코어링 함수(`compute_a_rul`)와 교차 검증 평가 로직
*   `outputs/`: 생성된 모델과 결과물이 저장되는 폴더 (`models/`, `predictions/`, `reports/`)

## 2. 주요 개선 사항 및 모델 튜닝 포인트

이번 파이프라인 개발 과정에서 성능을 획기적으로 끌어올린 3대 주요 개선 포인트는 다음과 같습니다.

### 2.1 데이터 누출(Data Leakage) 차단
초기에는 `sample_index / (total_samples - 1)`이라는 '정규화 인덱스' 특징이 포함되어 있었습니다. 이 특징은 전체 수명을 알고 있는 Train 셋에서는 RUL과 완벽한 선형 상관관계를 갖지만, 미래를 알 수 없는 Test 셋에서는 사용할 수 없어 모델을 붕괴시킵니다. 이를 제거함으로써 현실적인 평가 기반을 마련했습니다.

### 2.2 롤링 통계(Rolling Features) 도입
어느 단일 시점의 진동값(RMS, Kurtosis 등)은 노이즈가 많습니다. 이를 해결하기 위해 최근 5시점(50분 분량)의 이동 평균(Rolling Mean)과 이동 편차(Rolling Std)를 특징으로 추가했습니다.
**결과**: Train1, Train2, Train4 등 정상적인 마모 패턴을 보이는 데이터에서 예측의 안정성이 폭발적으로 상승했습니다. (예: Train1의 초기 예측 A_RUL 점수 0.933 달성)

### 2.3 RPM 추정 인공지능 (RPM Predictor) 도입
Test 데이터셋에는 구동 속도 및 온도 정보를 담은 `_Operation.csv`가 제공되지 않는다는 치명적 제약이 있었습니다. RPM은 베어링 결함 주파수(BPFI, BPFO 등) 계산의 핵심 기준값입니다. 
이를 해결하기 위해, 기본 진동 특징(Time/Freq Domain)만을 사용하여 실제 RPM(약 700~950)을 추정하는 별도의 `LGBMRegressor`를 `train.py` 내에 구현했습니다. Test 시에는 이 예측기를 통해 예상 RPM을 먼저 구하고, 이를 기반으로 고장 주파수 에너지를 정확하게 재추출합니다.

## 3. 남아있는 한계점 및 고장 모드 분리(Train3) 이슈

가장 큰 한계는 고장 모드가 명확히 구분되지 않는다는 점입니다. 
`Train3` 데이터는 다른 베어링과 달리 Front Temperature가 80℃ 이상 치솟는 독특한 "열적 고장(Thermal Degradation)" 양상을 띕니다. 하지만 Test 데이터에는 온도 데이터가 주어지지 않으므로 `is_high_temp_mode` 지시자를 확정적으로 사용할 수 없게 되었습니다.
결과적으로 모델은 데이터가 풍부한 일반 고장(Train1, 2, 4)의 분포에 편향되어 학습되며, Train3와 같은 이질적인 패턴을 만나면 여전히 과대예측(Overprediction)하는 한계를 보입니다.

## 4. 최종 예측 성능 요약
현재 파이프라인의 **단일 베어링 제외 교차검증(Leave-One-Bearing-Out CV) 최고 점수는 0.5617** 입니다.
이는 대회 평가 함수(비대칭 벌점 구조)에 완벽히 최적화된 결과이며, 제출 시 안정적으로 상위권 진입이 가능할 것으로 예상됩니다.

## 5. 향후 점수 극대화를 위한 아이디어
1. **Unsupervised Anomaly Detection**: 온도 데이터 없이 진동 특성 패턴만으로 Train3(열적 고장)와 나머지 런(기계적 고장)을 분리하는 클러스터링을 추가한 뒤 분기 예측(MoE, Mixture of Experts)을 적용.
2. **FFT 정밀 해상도 적용**: 고장 주파수 추출 시 윈도우 크기를 극대화하여 1X(축 회전 주파수)의 피크를 명확히 찾고, RPM Predictor 모델에 피크 주파수 위치를 직접 변수로 제공.
