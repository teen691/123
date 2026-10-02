import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# ---------------------------------------
# 기본 설정
# ---------------------------------------
st.set_page_config(
    page_title="기온 예측기",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 기온 예측기")
st.write("서울의 연평균기온 데이터를 이용해 회귀 직선을 만들고 미래의 기온을 예측합니다.")

# ---------------------------------------
# 데이터 주소
# ---------------------------------------
DATA_URL = (
    "https://raw.githubusercontent.com/greatsong/modudata/"
    "bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"
)

# ---------------------------------------
# 데이터 불러오기
# ---------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8-sig")

    # 날짜를 날짜 형식으로 변환
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")

    # 평균기온을 숫자로 변환
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    # 연도 추출
    df["연도"] = df["날짜"].dt.year

    return df


df = load_data()

# ---------------------------------------
# 연도별 자료 정리
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

# ---------------------------------------
# 회귀 분석에 사용할 데이터
# 조건:
# 1. 2025년까지
# 2. 관측일수가 300일 이상
# ---------------------------------------
regression_df = yearly[
    (yearly["연도"] <= 2025) &
    (yearly["관측일수"] >= 300)
].copy()

# 독립 변수: 1908년부터 지난 연수
regression_df["지난연수"] = regression_df["연도"] - 1908

# ---------------------------------------
# 선형 회귀
# ---------------------------------------
x = regression_df["지난연수"].to_numpy()
y = regression_df["연평균기온"].to_numpy()

slope, intercept = np.polyfit(x, y, 1)

# 상관계수
correlation = np.corrcoef(x, y)[0, 1]

# ---------------------------------------
# 화면 정보
# ---------------------------------------
start_year = int(regression_df["연도"].min())
end_year = int(regression_df["연도"].max())
data_count = len(regression_df)

st.subheader("📊 회귀 분석 정보")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("회귀에 사용한 연도 수", f"{data_count}개")

with col2:
    st.metric("시작 연도", f"{start_year}년")

with col3:
    st.metric("끝 연도", f"{end_year}년")

with col4:
    st.metric("상관계수", f"{correlation:.3f}")

st.write(
    f"회귀 분석에는 **2025년까지의 자료 중 관측일수가 300일 이상인 {data_count}개 연도**가 사용되었습니다."
)

# ---------------------------------------
# 산점도 + 회귀 직선
# ---------------------------------------
fig = go.Figure()

# 실제 연평균기온
fig.add_trace(
    go.Scatter(
        x=regression_df["연도"],
        y=regression_df["연평균기온"],
        mode="markers",
        name="연평균기온",
        text=[
            f"{year}년<br>관측일수: {days}일"
            for year, days in zip(
                regression_df["연도"],
                regression_df["관측일수"]
            )
        ],
        hovertemplate=(
            "%{x}년<br>"
            "연평균기온: %{y:.2f}℃<br>"
            "%{text}<extra></extra>"
        )
    )
)

# 회귀 직선
line_years = np.linspace(start_year, end_year, 300)
line_x = line_years - 1908
line_y = intercept + slope * line_x

fig.add_trace(
    go.Scatter(
        x=line_years,
        y=line_y,
        mode="lines",
        name="회귀 직선",
        line=dict(width=3)
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
# 회귀식 표시
# ---------------------------------------
st.subheader("📐 회귀식")

st.write(
    f"**연평균기온 = {intercept:.4f} + "
    f"{slope:.4f} × (연도 − 1908)**"
)

st.write(
    f"즉, 이 회귀선에서는 1년이 지날 때마다 "
    f"연평균기온이 약 **{slope:.4f}℃**씩 변하는 것으로 계산됩니다."
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

# 선택한 연도의 예상 기온
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
# 선택한 연도의 회귀선 위치 표시
# ---------------------------------------
fig_prediction = go.Figure()

fig_prediction.add_trace(
    go.Scatter(
        x=line_years,
        y=line_y,
        mode="lines",
        name="회귀 직선"
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
    title=f"{selected_year}년의 회귀모형 예상 위치",
    xaxis_title="연도",
    yaxis_title="예상 연평균기온 (℃)",
    xaxis=dict(
        tickmode="linear",
        dtick=10
    ),
    height=450
)

st.plotly_chart(fig_prediction, use_container_width=True)

# ---------------------------------------
# 참고
# ---------------------------------------
st.caption(
    "※ 예상 기온은 과거 연평균기온과 연도 사이의 선형 관계를 이용한 회귀모형의 값입니다."
)
st.caption(
    "※ 2025년 이후의 실제 관측값은 회귀선을 만드는 데 사용하지 않습니다."
)
