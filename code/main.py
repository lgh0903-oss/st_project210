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
        marker=dict(size=10, color="#636E
