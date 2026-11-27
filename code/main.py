
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import base64
import gzip
from io import BytesIO
from sklearn.linear_model import LinearRegression

# =====================================
# 기본 설정
# =====================================

st.set_page_config(
    page_title="서울 기온과 공공자전거 회귀 분석",
    page_icon="📈",
    layout="wide"
)

st.title("📈 서울 기온과 공공자전거 선형회귀 분석")
st.caption("2026년 1월~6월 기상 및 공공자전거 데이터를 활용한 분석")

# =====================================
# 데이터 내장
# 별도 파일 업로드 불필요
# =====================================

WEATHER_DATA = """
H4sIAB4iCWsC/22PPU4DMRCF+0i5A2WQ3lrzY+/aNVdJmwNwnLRIEUTbpAAlS0TaVOEA9BESBQUV45WjCHZtafzseZ+f3X13S3RWNmfsji97HL7Wp/XH9nW2PNzd9rv9+9P5/8nu53rytz/eu2kf3jbtqn2c3dtYLObzq3OkN50wRTyvtp8QkroiRqWuhi2Ni2BygorVam4TcyhCZICaxzVoXEIl4Oi8kWK1B0S4CIoDUhEtwWIS1FkC51S9oKpUxAjqwT2gSNmfi9qm93tORVAzAAM42UclmJu9KWWLjq48M3BJD+QHrGEly+4I0PyIfFf215dQm9PJL7sWfhf2AQAA
"""

BIKE_DATA = """
H4sIAB4iCWsC/12XvY6mNQyFey5jalbyf5JLpqOgRGK7Lem5BXpqHDujOV5pitFR3tjHfuLk+/jj379++/73938+fv24//7+48//Pn75EJL4Rpx/qcsScxQlRVtBgaJekXWPlXZFWaYoeoqxmcfnkeJSXkNcVyRiRnHfzyVVFE+tXLZAZEpRPURQLEdbF6FYjojHnnwd+bI4KF5Hy2OPQNfRkrVGoOfI0TtfR846V+4q3bIR/TqyTRsLIteRHc0vQORqhxvmKeUogodYjthpfH4dicseoleVaPRIriN3HzBIOVo8o19HHhqG4inRxud6HeWOhu3QckS0vkonjaIGEaF4beqmYyiWzXNooWhFnZ0hVuNIbOxZNm3vsee1qapQOmkUJVMa4rUZdBZ+Xig62cJAhWLEGYEKxSXHx8rraOVBwoIUim4yClIoWlZZUIwq3T6BYjmitcfnhWL6ZBTrcIkAS9IobnPB5AvF7Sr4eaHoIgcDSTvijXkWiitPF9osFI/uM8Tr6Miega6js3xjOwrFQ9u/8tRmKfaGlLRZOvk9ajfNzX7G1zfN4w4zQN9UM9Eh1gwgI0Vx1QBjGYHqxOSgDRRv4feKhZ8XSsd4rCyUMqWF0fkZggGmjRKTxR6bXkv7p5z4WXJGMep7Vh/fr1KXnbG2TPFhQfHU0oyFNS2cmDx8o1q2clO0VTwdnV6Lp7xSaDSlgGIhE8ygiEqr20YGZUzyCkIoiilO9GXktSsah2NppZ3lfMYctJ3lNBoq1w664cay5pI17wJBVUoNg2jWaLIlhopqO5YNdbSGk9McnGtrOvNkwZVtTWcaZh+JleGszUz3Gs67zAPE4jMWH0yr+Ew+Dpw4a0CZz6Kxto2xEuZVhHJec2us9VaPjRSqlZYnx1BdrdIoLrezY4SlaUglDmMODWlWN9CFdNPIF7anKOWcB4KZNaYXaHTRmCbQgpk1pnfwjR2io9EZ0d7oG+WV1zRzLENT6ncEgNqU5isAJoB/8pgT4KDabeN8WaF6rWmewLHU2oNBxfzhqDkXFNXoxASePP54JFIZwcpaXjRwx3vzmNGIDNQCMnf1gzs8IjcJ7sCva3lXolpd8x2OLvh1LUlF1XstPpn9EZlX8ChkE5lvr7NRbW/58hv5dttk5tBEigam0EDmQ2ls0EB6rIXFaSDVf1rbUyTfapjum5ubZ7SaIkaOdWweRQhb2TzmoYqxaxtLlUFtHvXIaFpPTcnifK2NR6ktfCLEo9QOch5vaubLFh5I8TAlw6kZD1PP3zmGarcyf0IcVLuVdmyo3UrePtR2nEjjvvxOoBLm25i6BQ+1e+kLjlU8TDVxGGq9VrYcLNmbm7EJE+N3An2NDdYbGDAa4lGar2QZO5x6bdHG6hakarrQQkPK+bYaS7tpao6tbEhddXj4nJo+g72pueFcx5ua91k8cnhNw4kTn5d73tgj2nmTO3DfNzbzDkxw/ge0/vUfTA8AAA==
"""

def load_embedded_data(data):
    raw = gzip.decompress(
        base64.b64decode(data.strip())
    )
    return pd.read_csv(
        BytesIO(raw),
        encoding="cp949"
    )

weather = load_embedded_data(WEATHER_DATA)
bike = load_embedded_data(BIKE_DATA)

# =====================================
# 데이터 전처리
# =====================================

weather["일시"] = pd.to_datetime(
    weather["일시"],
    format="%Y-%m"
)

bike["대여일자"] = pd.to_datetime(
    bike["대여일자"]
)

weather["월"] = weather["일시"].dt.month
bike["월"] = bike["대여일자"].dt.month

bike["경과일"] = (
    bike["대여일자"] - bike["대여일자"].min()
).dt.days

# 월별 자전거 대여 통계
bike_monthly = (
    bike.groupby("월")
    .agg(
        월별_평균대여건수=("대여건수", "mean"),
        월별_총대여건수=("대여건수", "sum")
    )
    .reset_index()
)

# 기온과 자전거 자료 결합
merged = pd.merge(
    weather,
    bike_monthly,
    on="월",
    how="inner"
)

# =====================================
# 선형회귀 함수
# =====================================

def regression_graph(
    df, x_col, y_col,
    x_label, y_label, title,
    point_labels=None
):

    X = df[[x_col]].astype(float)
    y = df[y_col].astype(float)

    model = LinearRegression()
    model.fit(X, y)

    prediction = model.predict(X)

    slope = model.coef_[0]
    intercept = model.intercept_
    r2 = model.score(X, y)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df[x_col],
        y=y,
        mode="markers",
        name="실제 데이터",
        text=point_labels,
        marker=dict(
            size=10,
            opacity=0.8
        )
    ))

    # 회귀 직선
    order = np.argsort(X.values.flatten())

    fig.add_trace(go.Scatter(
        x=X.values.flatten()[order],
        y=prediction[order],
        mode="lines",
        name="선형회귀 직선",
        line=dict(
            color="red",
            width=3
        )
    ))

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_white",
        height=450,
        hovermode="closest",
        legend=dict(
            orientation="h",
            y=1.12
        )
    )

    return fig, slope, intercept, r2

# =====================================
# 데이터 요약
# =====================================

st.header("📊 데이터 한눈에 보기")

c1, c2, c3 = st.columns(3)

c1.metric(
    "월별 기온 데이터",
    f"{len(weather)}개"
)

c2.metric(
    "일별 자전거 데이터",
    f"{len(bike)}개"
)

c3.metric(
    "결합된 월별 데이터",
    f"{len(merged)}개"
)

st.caption(
    "분석 기간: 2026년 1월 1일 ~ 6월 30일"
)

st.divider()

# =====================================
# 1. 월별 기온 변화
# =====================================

st.header("🌡️ 1. 월별 서울 평균기온 선형회귀")

st.write(
    "2026년 1월부터 6월까지 서울의 월별 평균기온이 "
    "어떤 방향으로 변화했는지 알아봅니다."
)

fig1, s1, i1, r1 = regression_graph(
    weather,
    "월",
    "평균기온(°C)",
    "월",
    "평균기온 (℃)",
    "월별 평균기온과 선형회귀 직선",
    weather["일시"].dt.strftime("%Y년 %m월")
)

st.plotly_chart(
    fig1,
    use_container_width=True
)

c1, c2, c3 = st.columns(3)

c1.metric("기울기", f"{s1:.2f} ℃/월")
c2.metric("절편", f"{i1:.2f}")
c3.metric("결정계수 R²", f"{r1:.3f}")

st.info(
    f"회귀식: y = {s1:.2f}x + {i1:.2f}"
)

with st.expander("월별 기온 원본 데이터"):
    st.dataframe(weather, use_container_width=True)

st.divider()

# =====================================
# 2. 공공자전거 일별 대여량
# =====================================

st.header("🚲 2. 일별 공공자전거 대여 건수 선형회귀")

st.write(
    "날짜가 지날수록 공공자전거 대여 건수가 "
    "어떤 추세로 변화하는지 살펴봅니다."
)

fig2, s2, i2, r2 = regression_graph(
    bike,
    "경과일",
    "대여건수",
    "2026년 1월 1일부터 경과한 날짜",
    "대여 건수 (건)",
    "일별 공공자전거 대여 건수와 선형회귀"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

c1, c2, c3 = st.columns(3)

c1.metric("기울기", f"{s2:,.2f} 건/일")
c2.metric("절편", f"{i2:,.2f} 건")
c3.metric("결정계수 R²", f"{r2:.3f}")

st.info(
    f"회귀식: y = {s2:.2f}x + {i2:.2f}"
)

with st.expander("공공자전거 원본 데이터"):
    st.dataframe(bike, use_container_width=True)

st.divider()

# =====================================
# 3. 기온과 자전거 대여량 관계
# =====================================

st.header("☀️ 3. 기온과 공공자전거 대여량의 관계")

st.write(
    "월별 평균기온과 월별 평균 일일 대여 건수를 "
    "하나의 그래프로 비교합니다."
)

fig3, s3, i3, r3 = regression_graph(
    merged,
    "평균기온(°C)",
    "월별_평균대여건수",
    "월별 평균기온 (℃)",
    "월별 평균 일일 대여 건수 (건)",
    "평균기온과 공공자전거 대여량의 관계",
    merged["일시"].dt.strftime("%Y년 %m월")
)

st.plotly_chart(
    fig3,
    use_container_width=True
)

c1, c2, c3 = st.columns(3)

c1.metric("기울기", f"{s3:,.2f} 건/℃")
c2.metric("절편", f"{i3:,.2f} 건")
c3.metric("결정계수 R²", f"{r3:.3f}")

st.info(
    f"회귀식: y = {s3:.2f}x + {i3:.2f}"
)

if s3 > 0:
    st.write(
        "기온이 높아질수록 자전거 대여 건수가 "
        "증가하는 방향의 관계가 나타났습니다."
    )
elif s3 < 0:
    st.write(
        "기온이 높아질수록 자전거 대여 건수가 "
        "감소하는 방향의 관계가 나타났습니다."
    )
else:
    st.write("회귀 직선의 기울기가 0에 가깝습니다.")

st.warning(
    "주의: 이 분석은 월별 데이터 6개를 이용한 결과입니다. "
    "기온이 자전거 대여량 변화의 원인이라고 단정할 수 없으며, "
    "강수량이나 요일 등 다른 요인도 영향을 줄 수 있습니다."
)

with st.expander("결합한 데이터 확인하기"):
    st.dataframe(merged, use_container_width=True)

st.divider()

# =====================================
# 4. 회귀 결과 비교
# =====================================

st.header("📋 4. 선형회귀 결과 정리")

result = pd.DataFrame({
    "분석": [
        "월별 평균기온",
        "일별 자전거 대여량",
        "기온과 자전거 대여량"
    ],
    "기울기": [s1, s2, s3],
    "절편": [i1, i2, i3],
    "결정계수(R²)": [r1, r2, r3]
})

st.dataframe(
    result.style.format({
        "기울기": "{:.3f}",
        "절편": "{:.3f}",
        "결정계수(R²)": "{:.3f}"
    }),
    use_container_width=True
)

st.subheader("💡 선형회귀에서 알아야 할 개념")

st.markdown("""
- **기울기:** x값이 1 증가할 때 y값이 얼마나 변하는지 나타냅니다.
- **절편:** x값이 0일 때 회귀 직선이 예측하는 y값입니다.
- **결정계수(R²):** 회귀모형이 데이터의 변동을 얼마나 설명하는지 나타냅니다.
- **회귀 직선:** 실제 데이터의 전체적인 경향을 가장 잘 나타내는 직선입니다.
""")

st.divider()

st.header("📝 분석을 통해 생각해 보기")

st.markdown("""
1. 월별 평균기온은 어떤 방향으로 변화했을까?
2. 공공자전거 대여 건수는 날짜에 따라 증가했을까, 감소했을까?
3. 기온이 높아지면 공공자전거 이용량도 증가할까?
4. 회귀 직선이 실제 데이터의 모든 변화를 설명하지 못하는 이유는 무엇일까?
""")

st.success("모든 선형회귀 그래프 분석이 완료되었습니다!")
