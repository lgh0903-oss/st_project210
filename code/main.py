
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 1. 페이지 설정
# --------------------------------------------------
st.set_page_config(
    page_title="공공자전거 이용량 회귀분석",
    page_icon="🚲",
    layout="wide"
)

st.title("🚲 평균기온과 공공자전거 이용량의 회귀분석")
st.write(
    "평균기온이 공공자전거 일별 대여건수와 어떤 관계가 있는지 "
    "산점도와 선형회귀분석으로 알아봅니다."
)


# --------------------------------------------------
# 2. 데이터 처리 함수
# --------------------------------------------------
def read_csv_file(file):
    """여러 인코딩을 시도하여 CSV 파일을 읽습니다."""
    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]

    for encoding in encodings:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=encoding)
        except UnicodeDecodeError:
            continue

    raise ValueError("CSV 파일의 문자 인코딩을 확인해 주세요.")


def parse_dates(series):
    """YYYYMMDD 및 일반적인 날짜 형식을 처리합니다."""
    text = series.astype(str).str.strip()

    result = pd.to_datetime(
        text,
        format="%Y%m%d",
        errors="coerce"
    )

    missing = result.isna()

    if missing.any():
        result.loc[missing] = pd.to_datetime(
            text.loc[missing],
            errors="coerce"
        )

    return result


def clean_number(series):
    """숫자에 포함된 쉼표 등을 제거합니다."""
    return pd.to_numeric(
        series.astype(str).str.replace(",", "", regex=False).str.strip(),
        errors="coerce"
    )


# --------------------------------------------------
# 3. CSV 파일 업로드
# --------------------------------------------------
st.subheader("📂 데이터 업로드")

col1, col2 = st.columns(2)

with col1:
    temperature_file = st.file_uploader(
        "서울 기온 데이터 CSV",
        type=["csv"],
        key="temperature"
    )

with col2:
    bicycle_file = st.file_uploader(
        "공공자전거 이용 데이터 CSV",
        type=["csv"],
        key="bicycle"
    )

if temperature_file is None or bicycle_file is None:
    st.info("기온 CSV 파일과 공공자전거 CSV 파일을 모두 업로드해 주세요.")
    st.stop()


# --------------------------------------------------
# 4. CSV 읽기
# --------------------------------------------------
try:
    temperature_raw = read_csv_file(temperature_file)
    bicycle_raw = read_csv_file(bicycle_file)

except Exception as e:
    st.error(f"CSV 파일을 읽는 중 오류가 발생했습니다: {e}")
    st.stop()


# --------------------------------------------------
# 5. 필요한 열 확인
# --------------------------------------------------
temperature_columns = ["날짜", "평균기온"]
bicycle_columns = ["날짜", "요일", "대여건수"]

missing_temp = [
    col for col in temperature_columns
    if col not in temperature_raw.columns
]

missing_bike = [
    col for col in bicycle_columns
    if col not in bicycle_raw.columns
]

if missing_temp or missing_bike:
    st.error("CSV 파일에 필요한 열이 없습니다.")

    if missing_temp:
        st.write("기온 데이터에서 필요한 열:", missing_temp)

    if missing_bike:
        st.write("자전거 데이터에서 필요한 열:", missing_bike)

    st.stop()


# --------------------------------------------------
# 6. 데이터 전처리
# --------------------------------------------------
temperature = temperature_raw[temperature_columns].copy()
bicycle = bicycle_raw[bicycle_columns].copy()

temperature["날짜"] = parse_dates(temperature["날짜"])
bicycle["날짜"] = parse_dates(bicycle["날짜"])

temperature["평균기온"] = clean_number(
    temperature["평균기온"]
)

bicycle["대여건수"] = clean_number(
    bicycle["대여건수"]
)

temperature = temperature.dropna(
    subset=["날짜", "평균기온"]
)

bicycle = bicycle.dropna(
    subset=["날짜", "대여건수"]
)

# 날짜별 평균기온 계산
temperature_daily = (
    temperature
    .groupby("날짜", as_index=False)
    .agg({"평균기온": "mean"})
)

# 날짜별 대여건수 합계 계산
bicycle_daily = (
    bicycle
    .groupby("날짜", as_index=False)
    .agg({
        "대여건수": "sum",
        "요일": lambda x: x.dropna().mode().iloc[0]
        if not x.dropna().mode().empty else "정보 없음"
    })
)

# 날짜를 기준으로 두 데이터 결합
merged = pd.merge(
    temperature_daily,
    bicycle_daily,
    on="날짜",
    how="inner"
)

merged = merged.dropna(
    subset=["평균기온", "대여건수"]
)

merged = merged.sort_values("날짜").reset_index(drop=True)

if len(merged) < 5:
    st.error(
        "분석에 사용할 수 있는 데이터가 너무 적습니다. "
        "두 CSV 파일의 날짜가 겹치는지 확인해 주세요."
    )
    st.stop()


# --------------------------------------------------
# 7. 분석 데이터 확인
# --------------------------------------------------
st.subheader("📊 분석 데이터 확인")

m1, m2, m3 = st.columns(3)

m1.metric("분석 데이터 수", f"{len(merged):,}일")
m2.metric(
    "평균기온 평균",
    f"{merged['평균기온'].mean():.2f}℃"
)
m3.metric(
    "일평균 대여건수",
    f"{merged['대여건수'].mean():,.0f}건"
)

with st.expander("결합된 데이터 확인하기"):
    st.dataframe(
        merged,
        use_container_width=True
    )


# --------------------------------------------------
# 8. 학습 데이터와 테스트 데이터 분리
# --------------------------------------------------
split_index = int(len(merged) * 0.8)

if split_index < 2 or len(merged) - split_index < 2:
    st.error("학습 및 테스트 데이터가 충분하지 않습니다.")
    st.stop()

train = merged.iloc[:split_index].copy()
test = merged.iloc[split_index:].copy()

X_train = train[["평균기온"]]
y_train = train["대여건수"]

X_test = test[["평균기온"]]
y_test = test["대여건수"]


# --------------------------------------------------
# 9. 선형회귀 모델 학습
# --------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

slope = float(model.coef_[0])
intercept = float(model.intercept_)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

if len(test) >= 2:
    r2 = r2_score(y_test, y_pred)
else:
    r2 = np.nan


# --------------------------------------------------
# 10. 회귀분석 결과
# --------------------------------------------------
st.subheader("📈 선형회귀분석 결과")

metric1, metric2, metric3 = st.columns(3)

metric1.metric("회귀계수", f"{slope:,.2f}")
metric2.metric("절편", f"{intercept:,.2f}")
metric3.metric(
    "테스트 데이터 R²",
    f"{r2:.4f}" if np.isfinite(r2) else "계산 불가"
)

st.write("**회귀식**")
st.code(
    f"예측 대여건수 = {intercept:.2f} + "
    f"({slope:.2f} × 평균기온)"
)

st.write(
    f"학습 데이터: {len(train):,}일 / "
    f"테스트 데이터: {len(test):,}일"
)


# --------------------------------------------------
# 11. 산점도와 회귀선
# --------------------------------------------------
st.subheader("☀️ 평균기온과 공공자전거 대여건수의 관계")

# 학습 데이터의 회귀선 생성
x_line = np.linspace(
    merged["평균기온"].min(),
    merged["평균기온"].max(),
    100
)

y_line = model.predict(
    pd.DataFrame({"평균기온": x_line})
)

# 산점도 생성
fig = px.scatter(
    merged,
    x="평균기온",
    y="대여건수",
    hover_data={
        "날짜": True,
        "요일": True,
        "평균기온": ":.2f",
        "대여건수": ":,.0f"
    },
    labels={
        "평균기온": "평균기온(℃)",
        "대여건수": "일별 총 대여건수(건)"
    },
    title="평균기온과 공공자전거 대여건수의 회귀분석"
)

# 회귀선 추가
fig.add_trace(
    go.Scatter(
        x=x_line,
        y=y_line,
        mode="lines",
        name="선형회귀선",
        line=dict(width=3)
    )
)

fig.update_layout(
    xaxis_title="평균기온(℃)",
    yaxis_title="일별 총 대여건수(건)",
    hovermode="closest",
    height=600
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# --------------------------------------------------
# 12. 테스트 데이터 실제값과 예측값 비교
# --------------------------------------------------
st.subheader("🔍 실제 대여건수와 예측 대여건수 비교")

fig_compare = go.Figure()

fig_compare.add_trace(
    go.Scatter(
        x=test["날짜"],
        y=y_test,
        mode="lines+markers",
        name="실제 대여건수"
    )
)

fig_compare.add_trace(
    go.Scatter(
        x=test["날짜"],
        y=y_pred,
        mode="lines+markers",
        name="예측 대여건수"
    )
)

fig_compare.update_layout(
    title="테스트 데이터의 실제값과 예측값",
    xaxis_title="날짜",
    yaxis_title="대여건수(건)",
    height=500,
    hovermode="x unified"
)

st.plotly_chart(
    fig_compare,
    use_container_width=True
)


# --------------------------------------------------
# 13. 요일별 평균 대여건수
# --------------------------------------------------
st.subheader("📅 요일별 평균 공공자전거 대여건수")

weekday_order = [
    "월요일", "화요일", "수요일",
    "목요일", "금요일", "토요일", "일요일"
]

weekday_data = merged.copy()

# 요일 표기를 통일
weekday_mapping = {
    "월": "월요일",
    "화": "화요일",
    "수": "수요일",
    "목": "목요일",
    "금": "금요일",
    "토": "토요일",
    "일": "일요일",
    "Mon": "월요일",
    "Tue": "화요일",
    "Wed": "수요일",
    "Thu": "목요일",
    "Fri": "금요일",
    "Sat": "토요일",
    "Sun": "일요일"
}

weekday_data["요일"] = (
    weekday_data["요일"]
    .astype(str)
    .str.strip()
    .replace(weekday_mapping)
)

weekday_average = (
    weekday_data
    .groupby("요일", as_index=False)["대여건수"]
    .mean()
)

weekday_average["정렬순서"] = (
    weekday_average["요일"]
    .apply(
        lambda x: weekday_order.index(x)
        if x in weekday_order else 99
    )
)

weekday_average = weekday_average.sort_values("정렬순서")

fig_weekday = px.bar(
    weekday_average,
    x="요일",
    y="대여건수",
    title="요일별 평균 대여건수",
    labels={
        "요일": "요일",
        "대여건수": "평균 대여건수(건)"
    },
    text_auto=".0f"
)

fig_weekday.update_layout(height=450)

st.plotly_chart(
    fig_weekday,
    use_container_width=True
)


# --------------------------------------------------
# 14. 모델 성능 평가
# --------------------------------------------------
st.subheader("🧪 회귀모델 성능 평가")

score1, score2, score3 = st.columns(3)

score1.metric("MAE", f"{mae:,.2f}")
score2.metric("MSE", f"{mse:,.2f}")
score3.metric(
    "R²",
    f"{r2:.4f}" if np.isfinite(r2) else "계산 불가"
)

st.markdown(
    """
    - **MAE**: 실제 대여건수와 예측 대여건수의 평균적인 차이입니다.
    - **MSE**: 예측 오차를 제곱하여 평균한 값입니다. 큰 오차에 더 큰 영향을 받습니다.
    - **R²**: 모델이 테스트 데이터의 변동을 얼마나 설명하는지 나타냅니다.
    """
)


# --------------------------------------------------
# 15. 결과 해석
# --------------------------------------------------
st.subheader("💡 분석 결과 해석")

if slope > 0:
    st.write(
        f"평균기온이 1℃ 높아질 때 예측 대여건수는 "
        f"약 {slope:,.2f}건 증가하는 경향을 보입니다."
    )
elif slope < 0:
    st.write(
        f"평균기온이 1℃ 높아질 때 예측 대여건수는 "
        f"약 {abs(slope):,.2f}건 감소하는 경향을 보입니다."
    )
else:
    st.write(
        "평균기온에 따른 예측 대여건수의 선형적인 변화가 거의 없습니다."
    )

if np.isfinite(r2):
    st.write(f"테스트 데이터의 R² 값은 {r2:.4f}입니다.")

st.write(
    "이 분석은 평균기온과 공공자전거 대여건수 사이의 관계를 "
    "살펴본 것입니다. 실제 자전거 이용량은 요일, 강수량, "
    "계절, 공휴일 등 다양한 요인의 영향을 받을 수 있으므로 "
    "기온이 대여건수 변화의 직접적인 원인이라고 단정할 수는 없습니다."
)


# --------------------------------------------------
# 16. 분석 데이터 다운로드
# --------------------------------------------------
st.subheader("📥 분석 데이터 다운로드")

csv_data = merged.to_csv(
    index=False,
    encoding="utf-8-sig"
)

st.download_button(
    label="결합된 분석 데이터 CSV 다운로드",
    data=csv_data,
    file_name="bicycle_temperature_analysis.csv",
    mime="text/csv"
)

st.success("분석이 완료되었습니다!")
