st.plotly_chart(fig_compare, use_container_width=True)

with st.expander("테스트 데이터의 예측 결과 표"):
    st.dataframe(prediction_result, use_container_width=True)

st.header("8. 요일별 대여건수 분석")

weekday_order = [
    "월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"
]

weekday_data = merged.dropna(subset=["요일"]).copy()
weekday_data["요일"] = weekday_data["요일"].apply(get_weekday_name)

weekday_summary = (
    weekday_data.groupby("요일", as_index=False)
    .agg(평균대여건수=("대여건수", "mean"))
)

weekday_summary["요일"] = pd.Categorical(
    weekday_summary["요일"],
    categories=weekday_order,
    ordered=True
)
weekday_summary = weekday_summary.sort_values("요일")

if not weekday_summary.empty:
    fig_weekday = px.bar(
        weekday_summary,
        x="요일",
        y="평균대여건수",
        title="요일별 평균 공공자전거 대여건수",
        labels={
            "요일": "요일",
            "평균대여건수": "평균 대여건수(건)"
        },
        text_auto=".0f"
    )
    fig_weekday.update_layout(
        template="plotly_white",
        height=450
    )
    st.plotly_chart(fig_weekday, use_container_width=True)
    st.dataframe(weekday_summary, use_container_width=True)

    highest_day = weekday_summary.loc[
        weekday_summary["평균대여건수"].idxmax()
    ]
    lowest_day = weekday_summary.loc[
        weekday_summary["평균대여건수"].idxmin()
    ]

    st.write(
        f"평균 대여건수가 가장 많은 요일은 **{highest_day['요일']}** "
        f"({highest_day['평균대여건수']:,.0f}건)입니다."
    )
    st.write(
        f"평균 대여건수가 가장 적은 요일은 **{lowest_day['요일']}** "
        f"({lowest_day['평균대여건수']:,.0f}건)입니다."
    )
else:
    st.warning("요일 값이 인식되지 않아 요일별 그래프를 만들 수 없습니다.")

st.header("9. 분석 결과 해석")

if slope > 0:
    st.write(
        f"회귀계수가 양수({slope:,.2f})이므로, 이 데이터에서는 "
        f"평균기온이 1℃ 높을 때 일별 대여건수가 평균적으로 "
        f"약 {slope:,.2f}건 많아지는 선형 관계가 나타났습니다."
    )
elif slope < 0:
    st.write(
        f"회귀계수가 음수({slope:,.2f})이므로, 이 데이터에서는 "
        f"평균기온이 1℃ 높을 때 일별 대여건수가 평균적으로 "
        f"약 {abs(slope):,.2f}건 적어지는 선형 관계가 나타났습니다."
    )
else:
    st.write("회귀계수가 0에 가까워 기온에 따른 선형 변화가 거의 나타나지 않았습니다.")

if r2 is not None:
    st.write(
        f"테스트 데이터의 결정계수(R²)는 {r2:.4f}입니다. "
        "R²는 테스트 데이터에서 예측값이 실제 대여건수의 변동을 "
        "얼마나 설명하는지 살펴보는 지표입니다. 음수도 나올 수 있습니다."
    )
else:
    st.write("테스트 데이터가 1개라 결정계수(R²)를 계산할 수 없습니다.")

st.write(
    f"테스트 데이터의 MAE는 {mae:,.2f}건으로, "
    "예측값과 실제값 차이의 평균 크기를 나타냅니다."
)
st.write(
    f"테스트 데이터의 MSE는 {mse:,.2f}건²으로, "
    "예측 오차를 제곱해 평균한 값입니다."
)

st.info(
    "해석 시 주의: 기온과 자전거 대여건수 사이에 관계가 나타나더라도 "
    "기온이 대여건수 변화의 직접적인 원인이라고 단정할 수는 없습니다. "
    "요일, 강수량, 미세먼지, 공휴일, 계절, 행사 등 다른 요인도 "
    "이용량에 영향을 줄 수 있습니다. 이번 회귀분석은 기온 하나를 "
    "사용한 단순 선형회귀입니다."
)

st.header("10. 분석 데이터 다운로드")

download_data = merged.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    label="📥 병합된 분석 데이터 CSV 다운로드",
    data=download_data,
    file_name="공공자전거_기온_병합데이터.csv",
    mime="text/csv"
)

download_result = prediction_result.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    label="📥 테스트 데이터 예측 결과 CSV 다운로드",
    data=download_result,
    file_name="공공자전거_회귀분석_예측결과.csv",
    mime="text/csv"
)

st.success("분석이 완료되었습니다!")
