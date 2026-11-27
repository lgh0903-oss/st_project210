 
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
    page_title="서울시 공공자전거 이용량 회귀분석",
    page_icon="🚲",
    layout="wide"
)

st.title("🚲 서울시 공공자전거 이용량 회귀분석")
st.write(
    "평균기온에 따라 서울시 공공자전거 이용량이 "
    "어떻게 달라지는지 회귀분석으로 알아봅니다."
)

# --------------------------------------------------
# 2. CSV 파일 읽기
# --------------------------------------------------

def read_csv_file(file):
    encodings = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]

    for encoding in encodings:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=encoding)
        except UnicodeDecodeError:
            continue

    raise ValueError("CSV 파일의 인코딩을 읽을 수 없습니다.")


def find_column(df, candidates):
    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    return None


# --------------------------------------------------
# 3. 데이터 업로드
# --------------------------------------------------

st.header("1. 데이터 수집")

col1, col2 = st.columns(2)

with col1:
    bike_file = st.file_uploader(
        "공공자전거 이용 데이터 CSV",
        type=["csv"]
    )

with col2:
    temp_file = st.file_uploader(
        "기온 데이터 CSV",
        type=["csv"]
    )

if bike_file is None or temp_file is None:
    st.info("공공자전거 데이터와 기온 데이터 CSV 파일을 모두 업로드해 주세요.")
    st.stop()

try:
    bike_raw = read_csv_file(bike_file)
    temp_raw = read_csv_file(temp_file)

except Exception as e:
    st.error(f"파일 읽기 오류: {e}")
    st.stop()

st.success("두 데이터 파일을 성공적으로 읽었습니다.")

# --------------------------------------------------
# 4. 데이터 구조 확인
# --------------------------------------------------

st.header("2. 데이터 확인")

with st.expander("공공자전거 데이터 확인"):
    st.write("행과 열:", bike_raw.shape)
    st.dataframe(bike_raw.head())

with st.expander("기온 데이터 확인"):
    st.write("행과 열:", temp_raw.shape)
    st.dataframe(temp_raw.head())

# --------------------------------------------------
# 5. 변수 선택 및 전처리
# --------------------------------------------------

st.header("3. 변수 선택 및 전처리")

bike_date_col = find_column(
    bike_raw,
    ["대여일자", "날짜", "일자", "기준일", "이용일자", "대여일시", "date", "Date"]
)

bike_count_col = find_column(
    bike_raw,
    ["이용건수", "이용량", "대여건수", "총이용건수", "건수", "count"]
)

temp_date_col = find_column(
    temp_raw,
    ["날짜", "일자", "일시", "기준일", "관측일", "date", "Date"]
)

temp_col = find_column(
    temp_raw,
    ["평균기온", "평균기온(°C)", "평균기온(℃)", "기온", "일평균기온", "평균기온(도)"]
)

if bike_date_col is None or bike_count_col is None:
    st.error(
        "자전거 데이터에서 날짜 또는 이용건수 열을 찾을 수 없습니다.\n\n"
        f"현재 열 이름: {list(bike_raw.columns)}"
    )
    st.stop()

if temp_date_col is None or temp_col is None:
    st.error(
        "기온 데이터에서 날짜 또는 평균기온 열을 찾을 수 없습니다.\n\n"
        f"현재 열 이름: {list(temp_raw.columns)}"
    )
    st.stop()

bike = bike_raw.copy()
temp = temp_raw.copy()

bike["날짜"] = pd.to_datetime(
    bike[bike_date_col],
    errors="coerce"
).dt.normalize()

temp["날짜"] = pd.to_datetime(
    temp[temp_date_col],
    errors="coerce"
).dt.normalize()

bike["이용건수"] = pd.to_numeric(
    bike[bike_count_col].astype(str).str.replace(",", ""),
    errors="coerce"
)

temp["평균기온"] = pd.to_numeric(
    temp[temp_col],
    errors="coerce"
)

bike = bike.dropna(subset=["날짜", "이용건수"])
temp = temp.dropna(subset=["날짜", "평균기온"])

# --------------------------------------------------
# 6. 날짜별 데이터 집계
# --------------------------------------------------

bike_daily = (
    bike.groupby("날짜", as_index=False)["이용건수"]
    .sum()
)

temp_daily = (
    temp.groupby("날짜", as_index=False)["평균기온"]
    .mean()
)

# --------------------------------------------------
# 7. 두 데이터 병합
# --------------------------------------------------

merged = pd.merge(
    bike_daily,
    temp_daily,
    on="날짜",
    how="inner"
)

merged = merged.replace([np.inf, -np.inf], np.nan)
merged = merged.dropna(
    subset=["날짜", "이용건수", "평균기온"]
)

merged = merged.sort_values("날짜").reset_index(drop=True)

if len(merged) < 5:
    st.error(
        "두 데이터에서 날짜가 일치하는 자료가 5개 미만입니다. "
        "날짜 형식과 데이터 기간을 확인해 주세요."
    )
    st.stop()

# --------------------------------------------------
# 8. 분석 데이터 요약
# --------------------------------------------------

st.header("4. 분석 데이터 확인")

m1, m2, m3 = st.columns(3)

m1.metric("분석 데이터 수", f"{len(merged):,}일")
m2.metric("평균기온", f"{merged['평균기온'].mean():.2f}℃")
m3.metric("일평균 이용량", f"{merged['이용건수'].mean():,.0f}건")

st.write("분석에 사용한 데이터")

st.dataframe(
    merged,
    use_container_width=True
)

st.write("두 데이터의 날짜 불일치 및 누락 현황")

st.write(f"- 자전거 데이터 날짜 수: {len(bike_daily):,}일")
st.write(f"- 기온 데이터 날짜 수: {len(temp_daily):,}일")
st.write(f"- 병합된 날짜 수: {len(merged):,}일")
st.write(
    f"- 병합되지 않은 자전거 날짜 수: "
    f"{len(bike_daily) - len(merged):,}일"
)

# --------------------------------------------------
# 9. 이상치 확인
# --------------------------------------------------

st.header("5. 이상치 및 데이터 점검")

q1 = merged["이용건수"].quantile(0.25)
q3 = merged["이용건수"].quantile(0.75)
iqr = q3 - q1

lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr

outliers = merged[
    (merged["이용건수"] < lower) |
    (merged["이용건수"] > upper)
]

st.write(f"이용량 기준 이상치 후보: {len(outliers)}일")

if len(outliers) > 0:
    with st.expander("이상치 후보 데이터 확인"):
        st.dataframe(outliers)

st.caption(
    "이상치 후보는 IQR 기준으로 확인했습니다. "
    "실제 이용량일 수 있으므로 임의로 삭제하지 않았습니다."
)

# --------------------------------------------------
# 10. 학습 데이터와 테스트 데이터 분할
# --------------------------------------------------

st.header("6. 학습 데이터와 평가 데이터 분할")

split_index = int(len(merged) * 0.8)

train = merged.iloc[:split_index].copy()
test = merged.iloc[split_index:].copy()

if len(train) < 2 or len(test) < 1:
    st.error("학습 및 평가 데이터가 부족합니다.")
    st.stop()

st.write(f"- 전체 데이터: {len(merged)}일")
st.write(f"- 학습 데이터: {len(train)}일")
st.write(f"- 평가 데이터: {len(test)}일")

st.write(
    f"- 학습 기간: {train['날짜'].min().date()} ~ "
    f"{train['날짜'].max().date()}"
)

st.write(
    f"- 평가 기간: {test['날짜'].min().date()} ~ "
    f"{test['날짜'].max().date()}"
)

# --------------------------------------------------
# 11. 기준선 모델
# --------------------------------------------------

st.header("7. 기준선 모델")

X_train = train[["평균기온"]]
X_test = test[["평균기온"]]

y_train = train["이용건수"]
y_test = test["이용건수"]

baseline_prediction = np.full(
    len(test),
    y_train.mean()
)

baseline_mae = mean_absolute_error(
    y_test,
    baseline_prediction
)

baseline_mse = mean_squared_error(
    y_test,
    baseline_prediction
)

st.write(
    f"기준선 모델은 학습 데이터의 평균 이용량인 "
    f"{y_train.mean():,.0f}건을 모든 날짜의 예측값으로 사용합니다."
)

# --------------------------------------------------
# 12. 선형회귀 모델
# --------------------------------------------------

st.header("8. 선형회귀 모델")

model = LinearRegression()

model.fit(X_train, y_train)

prediction = model.predict(X_test)

mae = mean_absolute_error(y_test, prediction)
mse = mean_squared_error(y_test, prediction)
r2 = r2_score(y_test, prediction)

coefficient = model.coef_[0]
intercept = model.intercept_

# --------------------------------------------------
# 13. 모델 평가
# --------------------------------------------------

st.header("9. 모델 성능 평가")

st.subheader("선형회귀 분석 결과")

c1, c2, c3 = st.columns(3)

c1.metric("회귀계수", f"{coefficient:,.2f}")
c2.metric("절편", f"{intercept:,.2f}")
c3.metric("R² 결정계수", f"{r2:.4f}")

c4, c5 = st.columns(2)

c4.metric("MAE 평균절대오차", f"{mae:,.2f}")
c5.metric("MSE 평균제곱오차", f"{mse:,.2f}")

st.subheader("기준선 모델과 선형회귀 모델 비교")

comparison = pd.DataFrame({
    "모델": ["기준선 모델", "선형회귀 모델"],
    "MAE": [baseline_mae, mae],
    "MSE": [baseline_mse, mse]
})

st.dataframe(
    comparison,
    use_container_width=True
)

if mae < baseline_mae:
    st.success("선형회귀 모델이 기준선 모델보다 MAE 기준으로 우수합니다.")
else:
    st.warning("선형회귀 모델이 기준선 모델보다 MAE 기준으로 우수하지 않습니다.")

# --------------------------------------------------
# 14. 기온과 이용량의 관계 그래프
# --------------------------------------------------

st.header("10. 평균기온에 따른 공공자전거 이용량 변화")

fig1 = px.scatter(
    merged,
    x="평균기온",
    y="이용건수",
    title="평균기온에 따른 공공자전거 이용량 변화",
    labels={
        "평균기온": "평균기온(℃)",
        "이용건수": "일별 총이용량(건)"
    },
    trendline="ols"
)

fig1.update_layout(
    template="plotly_white",
    height=500
)

st.plotly_chart(
    fig1,
    use_container_width=True
)

# --------------------------------------------------
# 15. 실제 이용량과 예측 이용량 비교
# --------------------------------------------------

st.header("11. 실제 이용량과 예측 이용량 비교")

result = test[["날짜", "평균기온", "이용건수"]].copy()

result["예측 이용량"] = prediction

fig2 = go.Figure()

fig2.add_trace(
    go.Scatter(
        x=result["날짜"],
        y=result["이용건수"],
        mode="lines+markers",
        name="실제 이용량"
    )
)

fig2.add_trace(
    go.Scatter(
        x=result["날짜"],
        y=result["예측 이용량"],
        mode="lines+markers",
        name="예측 이용량"
    )
)

fig2.update_layout(
    title="실제 이용량과 예측 이용량 비교",
    xaxis_title="날짜",
    yaxis_title="공공자전거 이용량(건)",
    template="plotly_white",
    height=500
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

st.dataframe(
    result,
    use_container_width=True
)

# --------------------------------------------------
# 16. 분석 결과 해석
# --------------------------------------------------

st.header("12. 분석 결과 해석")

if coefficient > 0:
    st.write(
        f"평균기온이 1℃ 상승할 때 예측 이용량은 "
        f"약 {coefficient:,.2f}건 증가하는 경향이 있습니다."
    )
elif coefficient < 0:
    st.write(
        f"평균기온이 1℃ 상승할 때 예측 이용량은 "
        f"약 {abs(coefficient):,.2f}건 감소하는 경향이 있습니다."
    )
else:
    st.write("평균기온과 이용량 사이의 선형적인 변화가 거의 없습니다.")

st.write(
    f"결정계수(R²)는 {r2:.4f}입니다. "
    "이는 학습 데이터로 만든 회귀모델의 평가 데이터에 대한 "
    "설명력을 판단할 때 참고할 수 있는 지표입니다."
)

st.caption(
    "주의: 기온만으로 자전거 이용량의 모든 변화를 설명할 수는 없습니다. "
    "요일, 강수량, 휴일 등 다른 요인은 이번 모델에 포함되지 않았습니다. "
    "또한 이 분석은 관찰된 관계를 나타내며 기온의 인과효과를 증명하지 않습니다."
)

# --------------------------------------------------
# 17. 데이터 다운로드
# --------------------------------------------------

st.header("13. 분석 결과 다운로드")

csv_data = result.to_csv(
    index=False,
    encoding="utf-8-sig"
)

st.download_button(
    label="📥 실제값과 예측값 CSV 다운로드",
    data=csv_data,
    file_name="공공자전거_회귀분석_결과.csv",
    mime="text/csv"
)

st.success("회귀분석이 완료되었습니다! 🚲")
