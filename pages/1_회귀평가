import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 1. 기본 설정
# --------------------------------------------------
st.set_page_config(
    page_title="서울 연평균 기온 선형회귀 분석",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 연평균 기온 선형회귀 분석")
st.write(
    "과거 연평균 기온으로 선형회귀 모델을 학습하고 "
    "최근 20년(2006~2025)의 기온을 얼마나 잘 예측하는지 비교합니다."
)


# --------------------------------------------------
# 2. 데이터 불러오기
# --------------------------------------------------
URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/seoul.csv"

df = pd.read_csv(URL, encoding="utf-8-sig")

df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
df["연도"] = df["날짜"].dt.year

# 평균기온을 숫자로 변환
df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

# --------------------------------------------------
# 3. 연평균 기온 계산
# --------------------------------------------------
annual = (
    df.dropna(subset=["연도", "평균기온"])
      .groupby("연도")["평균기온"]
      .mean()
      .reset_index()
)

annual.columns = ["연도", "연평균기온"]

annual["연도"] = annual["연도"].astype(int)
annual = annual.sort_values("연도").reset_index(drop=True)


# --------------------------------------------------
# 4. 결측 연도 확인
# --------------------------------------------------
st.subheader("데이터 확인")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "데이터에 포함된 첫 연도",
        f"{annual['연도'].min()}년"
    )

with col2:
    st.metric(
        "데이터에 포함된 마지막 연도",
        f"{annual['연도'].max()}년"
    )

with col3:
    st.metric(
        "연평균 데이터 개수",
        f"{len(annual)}개"
    )

# 1906년은 원자료가 없으므로 실제 학습 데이터에는 포함되지 않음
if 1906 not in annual["연도"].values:
    st.info(
        "원자료가 1907년부터 시작하므로 1906년 연평균 기온은 존재하지 않습니다. "
        "따라서 '1906~2005년 학습' 모델에서는 실제 데이터가 존재하는 연도만 사용합니다."
    )


# --------------------------------------------------
# 5. 전체 기간 회귀
# --------------------------------------------------
st.subheader("① 전체 기간의 장기 추세")

full_data = annual.dropna(subset=["연평균기온"]).copy()

X_full = full_data[["연도"]]
y_full = full_data["연평균기온"]

full_model = LinearRegression()
full_model.fit(X_full, y_full)

full_pred = full_model.predict(X_full)

full_slope = full_model.coef_[0]
full_slope_100 = full_slope * 100

full_mae = mean_absolute_error(y_full, full_pred)
full_mse = mean_squared_error(y_full, full_pred)
full_r2 = r2_score(y_full, full_pred)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "연간 기온 변화량",
        f"{full_slope:+.4f} ℃/년"
    )

with col2:
    st.metric(
        "100년 변화량",
        f"{full_slope_100:+.2f} ℃/100년"
    )

with col3:
    st.metric(
        "전체 데이터 R²",
        f"{full_r2:.3f}"
    )

with col4:
    st.metric(
        "전체 데이터 MAE",
        f"{full_mae:.3f} ℃"
    )

st.caption(
    "※ 전체 데이터의 R²·MAE는 모델이 전체 자료에 얼마나 잘 맞는지를 나타내는 값이며, "
    "미래 예측 성능을 나타내는 테스트 점수는 아닙니다."
)


# --------------------------------------------------
# 6. 전체 기간 그래프
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 5))

ax.scatter(
    full_data["연도"],
    full_data["연평균기온"],
    s=18,
    alpha=0.65,
    label="연평균 기온"
)

ax.plot(
    full_data["연도"],
    full_pred,
    linewidth=2.5,
    label="전체 기간 회귀선"
)

ax.set_title("서울 연평균 기온과 전체 기간 선형회귀선")
ax.set_xlabel("연도")
ax.set_ylabel("연평균 기온 (℃)")
ax.grid(alpha=0.25)
ax.legend()

st.pyplot(fig)


# --------------------------------------------------
# 7. 학습/테스트 데이터 구성
# --------------------------------------------------
TRAIN_50_START = 1956
TRAIN_50_END = 2005

TRAIN_100_START = 1906
TRAIN_100_END = 2005

TEST_START = 2006
TEST_END = 2025

train_50 = annual[
    (annual["연도"] >= TRAIN_50_START) &
    (annual["연도"] <= TRAIN_50_END)
].dropna(subset=["연평균기온"]).copy()

train_100 = annual[
    (annual["연도"] >= TRAIN_100_START) &
    (annual["연도"] <= TRAIN_100_END)
].dropna(subset=["연평균기온"]).copy()

test = annual[
    (annual["연도"] >= TEST_START) &
    (annual["연도"] <= TEST_END)
].dropna(subset=["연평균기온"]).copy()


# --------------------------------------------------
# 8. 50년 학습 모델
# --------------------------------------------------
model_50 = LinearRegression()

X_train_50 = train_50[["연도"]]
y_train_50 = train_50["연평균기온"]

X_test = test[["연도"]]
y_test = test["연평균기온"]

model_50.fit(X_train_50, y_train_50)

pred_50 = model_50.predict(X_test)

slope_50 = model_50.coef_[0]
slope_50_100 = slope_50 * 100

mae_50 = mean_absolute_error(y_test, pred_50)
mse_50 = mean_squared_error(y_test, pred_50)
r2_50 = r2_score(y_test, pred_50)


# --------------------------------------------------
# 9. 100년 학습 모델
# --------------------------------------------------
model_100 = LinearRegression()

X_train_100 = train_100[["연도"]]
y_train_100 = train_100["연평균기온"]

model_100.fit(X_train_100, y_train_100)

pred_100 = model_100.predict(X_test)

slope_100 = model_100.coef_[0]
slope_100_100 = slope_100 * 100

mae_100 = mean_absolute_error(y_test, pred_100)
mse_100 = mean_squared_error(y_test, pred_100)
r2_100 = r2_score(y_test, pred_100)


# --------------------------------------------------
# 10. 학습/테스트 구간 설명
# --------------------------------------------------
st.subheader("② 학습 데이터와 테스트 데이터")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 최근 50년 학습")
    st.write(
        f"학습: **{TRAIN_50_START}~{TRAIN_50_END}년**"
    )
    st.write(
        f"실제 학습 연도 수: **{len(train_50)}년**"
    )

with col2:
    st.markdown("### 최근 100년 학습")
    st.write(
        f"학습: **{TRAIN_100_START}~{TRAIN_100_END}년**"
    )
    st.write(
        f"실제 학습 연도 수: **{len(train_100)}년**"
    )

st.info(
    f"두 모델 모두 테스트 데이터는 동일하게 "
    f"**{TEST_START}~{TEST_END}년({len(test)}년)**을 사용합니다."
)


# --------------------------------------------------
# 11. 테스트 구간 예측 그래프
# --------------------------------------------------
st.subheader("③ 최근 20년 테스트 데이터 예측")

fig, ax = plt.subplots(figsize=(12, 6))

# 실제 테스트 데이터
ax.plot(
    test["연도"],
    y_test,
    marker="o",
    linewidth=2,
    label="실제 연평균 기온"
)

# 50년 학습 회귀선
ax.plot(
    test["연도"],
    pred_50,
    linewidth=2.5,
    label="1956~2005 학습 회귀선"
)

# 100년 학습 회귀선
ax.plot(
    test["연도"],
    pred_100,
    linewidth=2.5,
    label="1906~2005 학습 회귀선"
)

ax.set_title("2006~2025년 실제 기온과 두 회귀모델의 예측")
ax.set_xlabel("연도")
ax.set_ylabel("연평균 기온 (℃)")
ax.grid(alpha=0.25)
ax.legend()

st.pyplot(fig)


# --------------------------------------------------
# 12. 모델별 성능 비교
# --------------------------------------------------
st.subheader("④ 최근 20년 예측 성능 비교")

comparison = pd.DataFrame({
    "모델": [
        "최근 50년 학습 (1956~2005)",
        "최근 100년 학습 (1906~2005)"
    ],
    "기울기 (℃/년)": [
        slope_50,
        slope_100
    ],
    "기울기 (℃/100년)": [
        slope_50_100,
        slope_100_100
    ],
    "MAE (℃)": [
        mae_50,
        mae_100
    ],
    "MSE (℃²)": [
        mse_50,
        mse_100
    ],
    "R²": [
        r2_50,
        r2_100
    ]
})

st.dataframe(
    comparison.style.format({
        "기울기 (℃/년)": "{:+.4f}",
        "기울기 (℃/100년)": "{:+.2f}",
        "MAE (℃)": "{:.3f}",
        "MSE (℃²)": "{:.3f}",
        "R²": "{:.3f}"
    }),
    use_container_width=True
)


# --------------------------------------------------
# 13. 기울기 비교
# --------------------------------------------------
st.subheader("⑤ 회귀선의 기울기 비교")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 최근 50년 학습")
    st.metric(
        "100년 환산 기온 변화",
        f"{slope_50_100:+.2f} ℃"
    )

with col2:
    st.markdown("### 최근 100년 학습")
    st.metric(
        "100년 환산 기온 변화",
        f"{slope_100_100:+.2f} ℃"
    )

slope_difference = slope_50_100 - slope_100_100

st.write(
    f"두 모델의 기울기 차이는 "
    f"**{slope_difference:+.2f} ℃/100년**입니다."
)


# --------------------------------------------------
# 14. 성능이 더 좋은 모델 자동 판정
# --------------------------------------------------
st.subheader("⑥ 어떤 학습 기간이 더 잘 예측했을까?")

# MAE와 MSE는 작을수록 좋음
# R²는 클수록 좋음
score_50 = 0
score_100 = 0

if mae_50 < mae_100:
    score_50 += 1
elif mae_100 < mae_50:
    score_100 += 1

if mse_50 < mse_100:
    score_50 += 1
elif mse_100 < mse_50:
    score_100 += 1

if r2_50 > r2_100:
    score_50 += 1
elif r2_100 > r2_50:
    score_100 += 1

if score_50 > score_100:
    better_model = "최근 50년 학습 모델"
elif score_100 > score_50:
    better_model = "최근 100년 학습 모델"
else:
    better_model = "두 모델이 비슷한 성능"

st.success(
    f"MAE, MSE, R²를 종합하면 **{better_model}**이 "
    f"2006~2025년 테스트 데이터 예측에서 더 좋은 결과를 보였습니다."
)


# --------------------------------------------------
# 15. 해석
# --------------------------------------------------
st.subheader("⑦ 결과 해석")

st.markdown(
    f"""
### 기울기
- 최근 50년 학습 모델: **{slope_50_100:+.2f} ℃/100년**
- 최근 100년 학습 모델: **{slope_100_100:+.2f} ℃/100년**

두 모델의 기울기가 다르다면, 학습에 사용한 시간 범위에 따라
장기적인 기온 상승 추세를 추정하는 정도가 달라진다는 의미입니다.

### 예측 성능
- **MAE**: 실제 기온과 예측 기온의 평균적인 절대 오차입니다. 작을수록 좋습니다.
- **MSE**: 오차를 제곱해서 평균낸 값으로, 큰 예측 오차에 더 민감합니다. 작을수록 좋습니다.
- **R²**: 테스트 데이터의 변동을 모델이 얼마나 설명하는지를 나타냅니다. 일반적으로 1에 가까울수록 좋으며, 음수가 될 수도 있습니다.

이번 비교에서는 두 모델이 **똑같은 2006~2025년을 테스트 데이터**로 사용하기 때문에
50년 학습과 100년 학습의 예측 성능을 공정하게 비교할 수 있습니다.
"""
)


# --------------------------------------------------
# 16. 테스트 데이터 예측값 표
# --------------------------------------------------
st.subheader("⑧ 2006~2025년 실제값과 예측값")

prediction_table = test[["연도", "연평균기온"]].copy()

prediction_table["50년 모델 예측"] = pred_50
prediction_table["50년 모델 오차"] = (
    prediction_table["연평균기온"] -
    prediction_table["50년 모델 예측"]
)

prediction_table["100년 모델 예측"] = pred_100
prediction_table["100년 모델 오차"] = (
    prediction_table["연평균기온"] -
    prediction_table["100년 모델 예측"]
)

st.dataframe(
    prediction_table.style.format({
        "연평균기온": "{:.2f}",
        "50년 모델 예측": "{:.2f}",
        "50년 모델 오차": "{:+.2f}",
        "100년 모델 예측": "{:.2f}",
        "100년 모델 오차": "{:+.2f}"
    }),
    use_container_width=True
)
