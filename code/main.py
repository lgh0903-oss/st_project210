import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="서울 기온과 공공자전거 대여량 분석", layout="wide")

st.title("🚲 서울 기온과 공공자전거 대여량 선형회귀 분석")
st.markdown("2026년 상반기(1월~6월) 서울의 기온 변화와 공공자전거 일별 대여 건수의 관계를 분석합니다.")

# ============================================================
# 0. 공통 함수 및 데이터 불러오기
# ============================================================
def fit_line(x, y):
    model = LinearRegression()
    X_reshaped = x.reshape(-1, 1)
    model.fit(X_reshaped, y)
    pred = model.predict(X_reshaped)
    slope = float(model.coef_[0])
    intercept = float(model.intercept_)
    r2 = float(model.score(X_reshaped, y))
    return model, pred, slope, intercept, r2

def add_metric_row(slope, intercept, r2, unit):
    c1, c2, c3 = st.columns(3)
    c1.metric("기울기 (변화율)", f"{slope:,.2f} {unit}")
    c2.metric("절편 (시작 기준값)", f"{intercept:,.2f}")
    c3.metric("결정계수 (R²)", f"{r2:.3f}")

def read_csv_smart(path):
    for enc in ["cp949", "utf-8-sig", "utf-8", "euc-kr"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            continue
    raise FileNotFoundError(f"파일을 읽을 수 없습니다: {path}")

@st.cache_data
def load_data():
    # 깃허브 폴더 위치에 맞게 경로를 자동으로 찾습니다
    bike_paths = [
        "서울특별시 공공자전거 일별 대여건수_26.1-6_2.csv",
        "data/서울특별시 공공자전거 일별 대여건수_26.1-6_2.csv",
        "서울특별시 공공자전거 일별 대여건수_26.1-6.csv",
        "data/서울특별시 공공자전거 일별 대여건수_26.1-6.csv",
    ]
    weather_paths = [
        "OBS_ASOS_TIM_20261127163907.csv",
        "data/OBS_ASOS_TIM_20261127163907.csv",
    ]

    bike = None
    for p in bike_paths:
        try:
            bike = read_csv_smart(p)
            break
        except Exception:
            pass
    if bike is None:
        raise FileNotFoundError("자전거 대여건수 CSV 파일을 찾을 수 없습니다.")

    weather = None
    for p in weather_paths:
        try:
            weather = read_csv_smart(p)
            break
        except Exception:
            pass
    if weather is None:
        raise FileNotFoundError("기온(OBS_ASOS_TIM) CSV 파일을 찾을 수 없습니다.")

    # 1) 자전거 데이터 전처리
    bike["대여일자"] = pd.to_datetime(bike["대여일자"])
    bike = bike.sort_values("대여일자").reset_index(drop=True)
    bike["경과일"] = (bike["대여일자"] - bike["대여일자"].min()).dt.days
    bike["7일 이동평균"] = bike["대여건수"].rolling(window=7, min_periods=1).mean()
    bike["월"] = bike["대여일자"].dt.month

    monthly_bike = bike.groupby("월").agg(
        월별_평균_일대여건수=("대여건수", "mean"),
        월별_총대여건수=("대여건수", "sum"),
        관측일수=("대여건수", "count")
    ).reset_index()

    # 2) 기온 데이터 전처리 (시간별 ASOS 자료 -> 월별 집계)
    weather["일시"] = pd.to_datetime(weather["일시"])
    weather["월"] = weather["일시"].dt.month
    weather["날짜"] = weather["일시"].dt.date

    # 일별 최고/최저 기온을 먼저 구한 뒤 월별 평균 산출
    daily_w = weather.groupby(["월", "날짜"]).agg(
        일평균기온=("기온(°C)", "mean"),
        일최고기온=("기온(°C)", "max"),
        일최저기온=("기온(°C)", "min"),
    ).reset_index()

    monthly_weather = daily_w.groupby("월").agg(
        **{
            "평균기온(°C)": ("일평균기온", "mean"),
            "평균최고기온(°C)": ("일최고기온", "mean"),
            "평균최저기온(°C)": ("일최저기온", "mean"),
        }
    ).reset_index()

    monthly = pd.merge(monthly_weather, monthly_bike, on="월", how="inner")
    return bike, monthly

try:
    bike, monthly = load_data()
except Exception as e:
    st.error(f"데이터 불러오기 오류: {e}")
    st.stop()

# ============================================================
# 그래프 1: 월별 서울 평균기온 선형회귀
# ============================================================
st.header("1. 🌡️ 월별 서울 평균기온 선형회귀")
st.markdown("2026년 서울의 월별 평균기온이 어떤 방향으로 변화했는지 알아봅니다.")

if len(monthly) >= 2:
    wx = monthly["월"].to_numpy(dtype=float)
    wy = monthly["평균기온(°C)"].to_numpy(dtype=float)
    w_model, w_pred, w_slope, w_intercept, w_r2 = fit_line(wx, wy)
    wx_line = np.linspace(float(np.min(wx)), float(np.max(wx)), 100)
    wy_line = w_model.predict(wx_line.reshape(-1, 1))

    temp_plot = go.Figure()
    temp_plot.add_trace(go.Scatter(
        x=monthly["월"], y=monthly["평균기온(°C)"],
        mode="markers+lines", name="실제 데이터",
        marker=dict(size=10, color="#636EFA"),
        line=dict(color="#636EFA", width=1, dash="dot")
    ))
    temp_plot.add_trace(go.Scatter(
        x=wx_line, y=wy_line, mode="lines", name="선형회귀 직선",
        line=dict(color="red", width=3)
    ))
    temp_plot.update_layout(
        template="plotly_white", height=420,
        xaxis_title="월", yaxis_title="평균기온 (°C)",
        legend=dict(orientation="h", y=1.12)
    )
    st.plotly_chart(temp_plot, use_container_width=True)
    add_metric_row(w_slope, w_intercept, w_r2, "℃/월")
else:
    st.info(f"현재 기온 파일에 포함된 월 데이터가 {len(monthly)}개월({monthly['월'].tolist()}월)뿐이라 선형회귀 직선을 그리려면 2개월 이상의 기온 CSV가 필요합니다.")
    st.dataframe(monthly[["월", "평균기온(°C)", "평균최고기온(°C)", "평균최저기온(°C)"]], hide_index=True)

st.divider()

# ============================================================
# 그래프 2: 일별 공공자전거 대여 건수 선형회귀
# ============================================================
st.header("2. 🚲 일별 공공자전거 대여 건수 선형회귀")
st.markdown("날짜가 지날수록 공공자전거 대여 건수가 어떤 추세로 변화하는지 살펴봅니다.")

bx = bike["경과일"].to_numpy(dtype=float)
by = bike["대여건수"].to_numpy(dtype=float)
b_model, b_pred, b_slope, b_intercept, b_r2 = fit_line(bx, by)

bike_plot = go.Figure()
bike_plot.add_trace(go.Scatter(
    x=bike["경과일"], y=bike["대여건수"],
    mode="markers", name="실제 데이터",
    marker=dict(size=7, color="#636EFA", opacity=0.8),
    customdata=bike["대여일자"].dt.strftime("%Y-%m-%d"),
    hovertemplate="날짜: %{customdata}<br>경과일: %{x}일<br>대여건수: %{y:,.0f}건<extra></extra>"
))
bike_plot.add_trace(go.Scatter(
    x=bike["경과일"], y=bike["7일 이동평균"],
    mode="lines", name="7일 이동평균",
    line=dict(color="#00CC96", width=2)
))
bike_plot.add_trace(go.Scatter(
    x=bike["경과일"], y=b_pred,
    mode="lines", name="선형회귀 직선",
    line=dict(color="red", width=3)
))
bike_plot.update_layout(
    template="plotly_white", height=450,
    xaxis_title="2026년 1월 1일부터 경과한 날짜",
    yaxis_title="대여 건수 (건)",
    legend=dict(orientation="h", y=1.12)
)
st.plotly_chart(bike_plot, use_container_width=True)
add_metric_row(b_slope, b_intercept, b_r2, "건/일")

# 주춤한 구간 찾기 (7일 이동평균보다 3만 건 이상 급감한 날)
bike["차이"] = bike["대여건수"] - bike["7일 이동평균"]
selected_lows = bike.index[bike["차이"] < -30000].tolist()

if selected_lows:
    low_summary = bike.iloc[selected_lows][["대여일자", "대여건수", "7일 이동평균"]].copy()
    low_summary["대여일자"] = low_summary["대여일자"].dt.strftime("%Y-%m-%d")
    low_summary = low_summary.rename(columns={
        "대여일자": "주춤 구간 날짜", "대여건수": "해당 날짜 대여 건수",
        "7일 평균 대여 건수": "7일 평균 대여 건수"
    })
    with st.expander("주춤한 구간의 날짜와 대여량 확인"):
        st.dataframe(
            low_summary.style.format({
                "해당 날짜 대여 건수": "{:,.0f}",
                "7일 이동평균": "{:,.0f}",
            }), use_container_width=True, hide_index=True
        )

with st.expander("일별 자전거 원자료 보기"):
    st.dataframe(
        bike[["대여일자", "대여건수", "7일 이동평균"]],
        use_container_width=True, hide_index=True
    )

st.divider()

# ============================================================
# 그래프 3: 월별 기온 vs 월평균 일일 자전거 대여량 - 산점도
# ============================================================
st.header("3. ☀️ 월별 평균기온과 월평균 일일 자전거 대여량의 관계")
st.markdown(
    "**자료 단위: 월별 집계**  \n"
    "자전거 일별 자료 181개를 월별로 묶어 **월평균 일일 대여 건수**를 계산한 뒤, "
    "같은 달의 평균기온과 비교합니다."
)

if len(monthly) >= 2:
    mx = monthly["평균기온(°C)"].to_numpy(dtype=float)
    my = monthly["월별_평균_일대여건수"].to_numpy(dtype=float)
    m_model, m_pred, m_slope, m_intercept, m_r2 = fit_line(mx, my)
    x_line = np.linspace(float(np.min(mx)), float(np.max(mx)), 100)
    y_line = m_model.predict(x_line.reshape(-1, 1))

    relation_plot = go.Figure()
    relation_plot.add_trace(go.Scatter(
        x=monthly["평균기온(°C)"], y=monthly["월별_평균_일대여건수"],
        mode="markers+text", name="월별 관측값",
        text=[f"{m}월" for m in monthly["월"]],
        textposition="top center",
        marker=dict(size=13, color=monthly["월"], colorscale="Viridis",
                    showscale=True, colorbar=dict(title="월")),
        customdata=np.column_stack((
            monthly["월별_총대여건수"], monthly["관측일수"]
        )),
        hovertemplate=(
            "%{text}<br>평균기온: %{x:.1f}℃<br>월평균 일대여: %{y:,.0f}건"
            "<br>월 총대여: %{customdata[0]:,.0f}건"
            "<br>관측일수: %{customdata[1]:.0f}일<extra></extra>"
        ),
    ))
    relation_plot.add_trace(go.Scatter(
        x=x_line, y=y_line, mode="lines", name="선형회귀 직선",
        line=dict(color="#E4572E", width=3, dash="dash"),
    ))
    relation_plot.update_layout(
        template="plotly_white", height=500,
        xaxis_title="월평균기온 (℃)",
        yaxis_title="월평균 일일 자전거 대여 건수 (건)",
        hovermode="closest", legend=dict(orientation="h", y=1.12),
        margin=dict(t=45, b=45),
    )
    relation_plot.update_yaxes(tickformat=",d")
    st.plotly_chart(relation_plot, use_container_width=True)
    add_metric_row(m_slope, m_intercept, m_r2, "건/℃")
    st.caption(
        f"회귀식: 월평균 일일 대여 건수 = {m_slope:,.2f} × 평균기온 + ({m_intercept:,.2f}). "
        "월별 자료 기반이므로 결과는 참고용이며 인과관계를 증명하지 않습니다."
    )
else:
    st.warning("두 자료에서 겹치는 월이 2개월 미만이라 관계 회귀분석을 할 수 없습니다. (기온 CSV에 1~6월 데이터가 모두 들어있는지 확인해 주세요!)")

with st.expander("월별 결합 데이터 확인"):
    display_monthly = monthly[[
        "월", "평균기온(°C)", "평균최고기온(°C)", "평균최저기온(°C)",
        "월별_평균_일대여건수", "월별_총대여건수", "관측일수"
    ]].rename(columns={
        "월별_평균_일대여건수": "월평균 일일 대여 건수",
        "월별_총대여건수": "월 총 대여 건수",
        "관측일수": "자전거 관측 일수",
    })
    st.dataframe(display_monthly, use_container_width=True, hide_index=True)

st.divider()

# ============================================================
# 분석 결과 및 해석 가이드
# ============================================================
st.header("📌 그래프 해석하기")
st.markdown("""
- **그래프 1:** 월별 평균기온의 변화 방향과 월별 평균최고·평균최저기온 범위를 확인합니다.
- **그래프 2:** 매일의 변동과 7일 평균 흐름을 구분합니다. 굵은 이동평균선이 내려가는 구간에서 대여량이 주춤한 경향을 확인할 수 있습니다.
- **그래프 3:** 월평균기온과 월평균 일일 대여 건수가 함께 증가하거나 감소하는 경향이 있는지 살펴봅니다.

**해석할 때 주의할 점**
1. 월별 기온은 월별 추세를 보여줄 뿐, 일별 기온 변화는 보여주지 않습니다.
2. 공공자전거 자료는 일별 181개 자료이며, 그래프 3에서만 월평균 일일 대여 건수로 집계했습니다.
3. 그래프 3의 상관관계만으로 기온이 대여량 변화의 원인이라고 결론 내릴 수 없습니다. 비, 요일, 공휴일, 계절 등도 영향을 줄 수 있습니다.
4. 선형회귀 직선은 전체적인 경향을 요약합니다. 실제 데이터의 모든 오르내림을 설명하지는 않습니다.
""")

st.success("분석 화면을 모두 불러왔습니다. 그래프의 점이나 선에 마우스를 올리면 세부 값을 확인할 수 있습니다.")
