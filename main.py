```python
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 기온 예측기")

DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")

    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")
    df["연도"] = df["날짜"].dt.year

    return df


df = load_data()

# ---------------------------------------
# 연도별 평균기온과 관측일수 계산
# ---------------------------------------
yearly = (
    df.dropna(subset=["연도", "평균기온"])
    .groupby("연도")
    .agg(
        연평균기온=("평균기온", "mean"),
        관측일수=("평균기온", "count")
    )
    .reset_index()
)

# 2025년까지 + 관측일수 300일 이상
regression_df = yearly[
    (yearly["연도"] <= 2025) &
    (yearly["관측일수"] >= 300)
].copy()

# ---------------------------------------
# 전체 기간 회귀
# 독립변수 = 1908년부터 지난 연수
# ---------------------------------------
regression_df["지난연수"] = regression_df["연도"] - 1908

x = regression_df["지난연수"].to_numpy()
y = regression_df["연평균기온"].to_numpy()

slope, intercept = np.polyfit(x, y, 1)

# 상관계수
correlation = np.corrcoef(x, y)[0, 1]

# 100년 동안의 기온 변화량
slope_100 = slope * 100

# ---------------------------------------
# 최근 20년 회귀
# ---------------------------------------
recent_20_df = regression_df[
    regression_df["연도"] >= end_year - 19
].copy()

recent_x = recent_20_df["연도"].to_numpy() - 1908
recent_y = recent_20_df["연평균기온"].to_numpy()

recent_slope, recent_intercept = np.polyfit(
    recent_x,
    recent_y,
    1
)

recent_slope_100 = recent_slope * 100

# ---------------------------------------
# 회귀에 사용된 기간 정보
# ---------------------------------------
start_year = int(regression_df["연도"].min())
end_year = int(regression_df["연도"].max())
data_count = len(regression_df)

recent_start_year = int(recent_20_df["연도"].min())
recent_end_year = int(recent_20_df["연도"].max())

# ---------------------------------------
# 주요 정보
# ---------------------------------------
st.subheader("📈 기온 변화 속도")

col1, col2 = st.columns(2)

with col1:
    st.metric(
        "전체 기간",
        f"{slope_100:+.2f} ℃ / 100년"
    )
    st.caption(
        f"{start_year}~{end_year}년, {data_count}개 연도 사용"
    )

with col2:
    st.metric(
        "최근 20년",
        f"{recent_slope_100:+.2f} ℃ / 100년"
    )
    st.caption(
        f"{recent_start_year}~{recent_end_year}년, "
        f"{len(recent_20_df)}개 연도 사용"
    )

st.info(
    "기울기는 1년당 변화량에 100을 곱해 "
    "'100년에 몇 ℃ 변하는가'로 환산했습니다."
)

# ---------------------------------------
# 기존 산점도 + 전체 회귀선
# ---------------------------------------
fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=regression_df["연도"],
        y=regression_df["연평균기온"],
        mode="markers",
        name="연평균기온",
        text=[
            f"관측일수: {days}일"
            for days in regression_df["관측일수"]
        ],
        hovertemplate=(
            "%{x}년<br>"
            "연평균기온: %{y:.2f}℃<br>"
            "%{text}<extra></extra>"
        )
    )
)

# 전체 기간 회귀선
line_years = np.linspace(start_year, end_year, 300)
line_x = line_years - 1908
line_y = intercept + slope * line_x

fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_y,
        mode="lines",
        name="전체 기간 회귀선",
        line=dict(width=3)
    )
)

# 최근 20년 회귀선
recent_line_years = np.linspace(
    recent_start_year,
    recent_end_year,
    100
)

recent_line_x = recent_line_years - 1908
recent_line_y = recent_intercept + recent_slope * recent_line_x

fig.add_trace(
    go.Scatter(
        x=recent_line_years,
        y=recent_line_y,
        mode="lines",
        name="최근 20년 회귀선",
        line=dict(width=3, dash="dash")
    )
)

fig.update_layout(
    title="서울 연평균기온과 회귀 직선",
    xaxis_title="연도",
    yaxis_title="연평균기온 (℃)",
    xaxis=dict(
        tickmode="linear",
        dtick=10
    ),
    hovermode="x unified",
    height=550
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------
# 회귀식
# ---------------------------------------
st.subheader("📐 회귀식")

st.write(
    f"전체 기간: "
    f"**연평균기온 = {intercept:.4f} + "
    f"{slope:.4f} × (연도 − 1908)**"
)

st.write(
    f"최근 20년: "
    f"**연평균기온 = {recent_intercept:.4f} + "
    f"{recent_slope:.4f} × (연도 − 1908)**"
)

# ---------------------------------------
# 상관계수
# ---------------------------------------
st.subheader("🔗 전체 기간 상관계수")

st.metric(
    "연도와 연평균기온의 상관계수",
    f"{correlation:.3f}"
)

# ---------------------------------------
# 연도 선택
# ---------------------------------------
st.subheader("🔮 연도별 예상 기온")

selected_year = st.slider(
    "예측할 연도를 선택하세요.",
    min_value=1900,
    max_value=2100,
    value=2025,
    step=1
)

selected_x = selected_year - 1908
predicted_temp = intercept + slope * selected_x

st.markdown(
    f"""
    <div style="
        text-align: center;
        padding: 25px;
        border-radius: 15px;
        background-color: #f0f2f6;
        margin: 20px 0;
    ">
        <div style="font-size: 24px; font-weight: bold;">
            {selected_year}년 예상 연평균기온
        </div>
        <div style="font-size: 52px; font-weight: bold; margin-top: 10px;">
            {predicted_temp:.2f}℃
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------
# 선택한 연도 위치 표시
# ---------------------------------------
fig_prediction = go.Figure()

fig_prediction.add_trace(
    go.Scatter(
        x=line_years,
        y=line_y,
        mode="lines",
        name="전체 기간 회귀선"
    )
)

fig_prediction.add_trace(
    go.Scatter(
        x=[selected_year],
        y=[predicted_temp],
        mode="markers",
        name=f"{selected_year}년 예상값",
        marker=dict(size=14)
    )
)

fig_prediction.update_layout(
    title=f"{selected_year}년 예상 기온",
    xaxis_title="연도",
    yaxis_title="예상 연평균기온 (℃)",
    xaxis=dict(
        tickmode="linear",
        dtick=10
    ),
    height=450
)

st.plotly_chart(fig_prediction, use_container_width=True)

st.caption(
    "※ 예상 기온은 전체 기간의 선형 회귀식을 이용한 값입니다."
)
st.caption(
    "※ 2025년 이후의 실제 관측값은 회귀 분석에 사용하지 않습니다."
)
```
