# 음악 구독 서비스 고객 이탈 예측 프로젝트

---

## 👥 팀 구성

| 이름  | 역할                                                         |
|-----|------------------------------------------------------------|
| 김지윤 | |
| 박민선 |  |
| 박소윤 |  |
| 이레  |  |
| 윤지혜 |  |
| 최수아 |  |

---

## 🛠 실행 환경 (Environment)

- Python 3.10+ ⇒ 
  - Streamlit 1.30+
  - 주요 라이브러리: pandas, numpy, scikit-learn, lightgbm, catboost, shap, matplotlib, seaborn
  - 딥러닝 프레임워크: pytorch (또는 tensorflow)

---

## 1. 프로젝트 개요

### 1.1 프로젝트 요약
본 프로젝트는 음악 스트리밍 서비스(Spotify 등) 사용자의 행동 패턴 데이터와 지역 사회경제적 특성(미국 Census)을 융합하여, 향후 해당 사용자가 이탈(Churn)할 가능성을 예측하고 분석하는 데이터 기반 의사결정 지원 서비스입니다.기존의 단순 이탈 예측(0/1 분류) 모델은 "누가 떠날 것인가"만 맞출 뿐 "왜 떠나는가"에 대한 해석 제공에는 한계가 존재합니다. 본 서비스는 이러한 한계를 보완하기 위해 가설 기반의 파생 변수를 설계하고, SHAP(설명 가능한 AI)을 활용하여 개별 고객의 이탈 유발 핵심 요인을 도출합니다.이를 통해 마케팅 및 운영 부서가 타겟팅된 리텐션(Retention) 전략을 즉각적으로 실행할 수 있도록 지원하는 구조로 설계되었습니다.

### 1.2 문제 정의구독형 비즈니스에서 신규 고객 유치 비용은 기존 고객 유지 비용보다 압도적으로 높습니다. 따라서 본 프로젝트는 다음 세 가지 질문에 데이터 기반으로 답하는 것을 목표로 합니다.
- 수많은 활성 사용자 중 누가 이탈할 위험이 가장 높은가?
- 그들은 왜 이탈하려고 하는가? (불만족, 몰입도 저하, 소셜 기능 미활용 등)
- 이탈을 막기 위해 어떤 비즈니스 액션(푸시 알림, 쿠폰, 추천)을 취해야 하는가?
  
---

## 2. 서비스 설계 철학
본 서비스는 단순한 이진 분류 서비스가 아닌 의사결정 지원 시스템으로 설계되었습니다. 핵심 설계 원칙은 다음과 같습니다.

### 1. 단편적 정보가 아닌 '맥락'을 결합한다.
- 앱 내 행동 로그뿐만 아니라, 거주 지역의 사회/인구통계학적 특성을 결합하여 매크로적인 이탈 요인을 탐색합니다.
### 2.도메인 가설 기반으로 파생 변수를 창출한다.
- 원본 데이터를 그대로 사용하지 않고, 이탈 징후를 대변하는 '몰입도', '사회적 연결망', '고객 불만' 지표를 수학적으로 정의합니다.
### 3.설명 가능한 예측(XAI)만 제공한다.
- 예측 정확도(성능)만큼 해석 가능성을 중시하여, 비즈니스 의사결정자가 결과를 신뢰하고 활용할 수 있도록 SHAP 분석을 필수적으로 연동합니다.
### 4.결과는 즉각적인 '액션'으로 연결되어야 한다.
- 예측값 제시에 그치지 않고, 결과에 따른 맞춤형 타겟 마케팅 플랜을 자동 제안합니다.
  
---

## 3. 핵심 기능

| 구분 | 기능 설명 |
|-----|-----------|
| 이탈 예측 모델 | 행동/지역 변수 기반 확률 도출. CatBoost/LightGBM 기반으로 고객별 이탈 여부(0/1) 및 이탈 확률 예측 |
| 인사이트 시각화 | 변수별 이탈 영향도 분석. 어떤 요인이 이탈률에 기여하는지 대시보드 기반으로 시각화 탐색 |
| XAI 원인 분석 | 이탈 요인 분해(SHAP). 특정 고객의 이탈 확률을 높인 요인과 낮춘 요인을 시각적으로 설명 |
| 대안 제시 | 리텐션 액션 플랜 제안. 이탈 확률 및 핵심 원인(Segment)에 맞춰 고위험군 타겟 푸시 전략 제안 |

---

## 4. 시스템 아키텍처

[Kaggle 데이터 (행동+인구)] → [Python 전처리/파생변수 생성] → [ML/DL 모델링 (LGBM, CatBoost)] → [SHAP 기반 해석] → [Streamlit 대시보드]
데이터 정제, 모델 학습, 시각화 단계를 명확히 분리하여 파이프라인의 유지보수성을 확보하였습니다.

---

## 5. 데이터 수집 및 정제

### 5.1 데이터 출처
- Primary: Kaggle - 음악 구독 사용자 이탈 데이터 (streaming-subscription-churn-model)
- Secondary: Kaggle - 미국 Census 인구/사회 데이터 (us-census-demographic-data)
  
### 5.2 전처리 및 도메인 지식 반영
- 결측치 처리: 수치형 변수는 중앙값(Median), 범주형 변수는 최빈값 또는 "Unknown" 대체.
- 날짜 변환: signup_date를 datetime으로 변환하여 기준일 대비 tenure_days(가입 후 경과 일수) 도출.
- 논리적 데이터 보정 (핵심):
  - average_session_length: 데이터 구조상 hours가 아닌 minutes가 타당하므로 단위 재해석.
  - weekly_unique_songs > weekly_songs_played인 모순 데이터는 played 값으로 상한(Cap) 조정.
    
--- 

## 6. 데이터베이스 및 ERD 설계
본 프로젝트의 원본 데이터는 Flat CSV 형태이나, 실제 서비스 운영 환경을 가정하여 사용자 기본 정보, 구독 이력, 행동 로그, 지역 인구통계 데이터를 정규화하여 논리적 ERD를 설계하였습니다.
<img width="1030" height="1092" alt="ERD" src="https://github.com/user-attachments/assets/3b588dfc-86b3-4e0c-8fe2-fee78f1c1d6a" />

---

## 7. 가설 기반 Feature Engineering 설계
단순 예측을 넘어, 고객의 상태를 대변하는 도메인 지표를 생성하였습니다.
### ① Engagement (몰입도 및 만족도 지표)
- engagement_score = weekly_hours × weekly_unique_songs (전반적 몰입도)
- completion_rate = 1 - song_skip_rate (음악을 끝까지 듣는 비율, 만족도)
- diversity_ratio = weekly_unique_songs / weekly_songs_played (음악 소비 다양성)
### ② Social & Activity (활동 및 락인 지표)
- social_activity = num_platform_friends + num_shared_playlists (플랫폼 내 사회적 연결 강도)
- activity_intensity = weekly_hours / average_session_length (접속 빈도 특성)
### ③ Stability & Risk (안정성 및 불만 지표)
- pause_ratio = num_subscription_pauses / tenure_days (구독 유지 불안정성)
- support_intensity = customer_service_inquiries / tenure_days (단위 기간당 고객센터 문의 빈도)
### ④ Regional (지역 사회적 특성)
- male_ratio = Men / TotalPop (성비 구조)
- diversity_index, log_population (지역 인구 다양성 및 도시 규모 효과)

---

## 8. 모델링 파이프라인 및 평가

### 8.1 대조군 모델 구성

| 구분 | 모델 | 목적 |
|-----|-----|-----|
| Baseline ML | Logistic Regression | 선형적 관계 파악 및 베이스라인 성능 확보 |
| Main ML | LightGBM / CatBoost | 트리 앙상블 기반 고성능 분류, 범주형 변수 처리 및 SHAP 호환 |
| Advanced DL | Simple MLP | 정형 데이터에 대한 신경망 모델 적용 및 트리 모델과의 패턴 인식 대조 |

### 8.2 평가 지표 (Metrics)이탈 예측 비즈니스 특성상, 이탈자를 놓치지 않는 것이 가장 중요하므로 다음을 중점 평가합니다.
- Recall (재현율): 실제 이탈자 중 모델이 찾아낸 비율 (최우선 지표)
- F1 Score: Precision과 Recall의 조화 평균
- ROC-AUC & PR-AUC: 클래스 불균형 환경에서의 전반적 분류 성능
---

## 9. XAI 및 리텐션 전략 제안 로직

[Case A] 불만 누적형 이탈 위험군
- 징후: support_intensity 높음, pause_ratio 높음
- 액션 제안: CS 임원진의 사과 콜 진행, 차월 멤버십 비용 50% 할인 쿠폰 즉시 발급.

[Case B] 흥미 저하/콘텐츠 고갈형 위험군
- 징후: 최근 청취 시간 감소, completion_rate 하락
- 액션 제안: 사용자 취향 기반 신규 아티스트 추천 푸시 알림, 개인화 맞춤형 믹스 플레이리스트 제공.

[Case C] 소셜 기능 미활용 고립형
- 징후: social_activity가 0에 가까움
- 액션 제안: 친구 초대 시 양측 포인트 지급 프로모션 전송, 지역 기반 인기 차트 노출 강화.
  
---

## 10. GitHub 폴더 구조
본 프로젝트는 데이터 전처리부터 모델링, 서비스 배포까지의 파이프라인이 구분되도록 구성되었습니다.

```
project/
   ├─ 01_data/                 # 원본 및 전처리 완료 데이터셋
   │   ├─ raw/
   │   └─ processed/
   │
   ├─ 02_notebooks/            # 실험 및 분석용 Jupyter Notebook
   │   ├─ 01_EDA_and_Preprocessing.ipynb
   │   └─ 02_Modeling_and_XAI.ipynb
   │
   ├─ 03_models/               # 학습이 완료된 최종 모델 파일 (.pkl, .cbm)
   │
   ├─ 04_app/                  # Streamlit 서비스 UI
   │   ├─ app.py               # 메인 실행 파일
   │   └─ components/          # 시각화 및 UI 모듈 스크립트
   │
   ├─ requirements.txt         # 패키지 의존성
   └─ README.md
```

---

## 11. 프로젝트 차별성
- 의사결정 지원 중심 설계: 이탈 여부 결과만 던져주는 것이 아니라, 구매 판단에 필요한 SHAP 기반 해석 근거와 리텐션 플랜을 함께 제공합니다.
- 도메인 지식 기반 변수 재정의: average_session_length의 단위 오류나 고유 곡 수의 논리적 모순 등 원본 데이터의 맹점을 교정하고, 서비스 가설을 수학적 지표(engagement, social_activity 등)로 녹여냈습니다.
- 거시적 맥락(Census) 융합: 단순 앱 내 로그에 그치지 않고, 외부 공공 데이터(미국 지역 인구 통계)를 결합하여 사용자 행동 이면의 사회경제적 환경 변수를 분석에 포함하였습니다.
  
---

## 12. 한계점 및 확장 방향

### 한계점
- 시계열 데이터의 부재: 제공된 데이터가 단일 스냅샷 형태이므로, 시간의 흐름에 따른 행동 변화(최근 1주 vs 3주 전) 트렌드를 모델에 직접 반영하기 어렵습니다.
- 콘텐츠 메타데이터 한계: 주로 듣는 '장르'나 '아티스트 성향' 데이터가 부족하여 콘텐츠 특화 이탈 요인을 심도 있게 파악할 수 없습니다.
  
### 확장 방향
- 생존 분석(Survival Analysis) 도입: 가입 후 경과일(tenure_days)을 활용하여 '이탈 여부'뿐만 아니라 '언제 이탈할 것인가'를 예측하는 모델로 고도화할 수 있습니다.
- 고객 생애 가치(LTV) 파이프라인 결합: 이탈 예측 모델과 LTV 예측 모델을 결합하여, "이탈 위험은 높으나 비즈니스 가치가 높은 VIP 고객"을 최우선으로 선별하는 시스템으로 확장할 수 있습니다.# 음악 구독 서비스 고객 이탈 예측 프로젝트

