    m1.metric("분석 데이터 수", f"{len(merged):,}일")
    m2.metric("회귀계수", f"{coefficient:,.2f}")
    m3.metric("결정계수 (R²)", f"{r2:.4f}")

    m4, m5, m6 = st.columns(3)
    m4.metric("절편", f"{intercept:,.2f}")
    m5.metric("MAE", f"{mae:,.2f}")
    m6.metric("MSE", f"{mse:,.2f}")

    sign = "+" if intercept >= 0 else "-"
    st.subheader("회귀식")
    st.latex(
        rf"\hat{{y}} = {coefficient:.2f}x {sign} {abs(intercept):.2f}"
    )
    st.caption("x는 일평균기온(℃), ŷ는 회귀모델이 예측한 일별 공공자전거 이용건수입니다.")

    st.header("5. 회귀분석 그래프")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(
        merged["평균기온"],
        merged["이용건수"],
        alpha=0.7,
        label="실제 이용량"
    )
    x_line = np.linspace(
        merged["평균기온"].min(),
        merged["평균기온"].max(),
        100
    )
    y_line = model.predict(pd.DataFrame({"평균기온": x_line}))
    ax.plot(x_line, y_line, linewidth=2, label="회귀직선")
    ax.set_title("평균기온에 따른 공공자전거 이용량 변화")
    ax.set_xlabel("일평균기온 (℃)")
    ax.set_ylabel("일별 공공자전거 총이용량 (건)")
    ax.grid(alpha=0.25)
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("실제 이용량과 예측 이용량 비교")
    compare = merged[["날짜", "이용건수"]].copy()
    compare["예측 이용량"] = predictions
    compare = compare.sort_values("날짜")

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.plot(compare["날짜"], compare["이용건수"], marker="o", label="실제 이용량")
    ax2.plot(compare["날짜"], compare["예측 이용량"], marker="o", label="예측 이용량")
    ax2.set_title("실제 이용량과 회귀모델 예측 이용량 비교")
    ax2.set_xlabel("날짜")
    ax2.set_ylabel("이용건수 (건)")
    ax2.grid(alpha=0.25)
    ax2.legend()
    fig2.autofmt_xdate()
    st.pyplot(fig2)
    plt.close(fig2)

    st.dataframe(compare, use_container_width=True)

    st.header("6. 분석 결과 해석")
    if coefficient > 0:
        st.write(
            f"평균기온이 1℃ 높아질 때 공공자전거 이용량은 회귀모델상 "
            f"평균적으로 약 {coefficient:,.2f}건 증가하는 경향이 있습니다."
        )
    elif coefficient < 0:
        st.write(
            f"평균기온이 1℃ 높아질 때 공공자전거 이용량은 회귀모델상 "
            f"평균적으로 약 {abs(coefficient):,.2f}건 감소하는 경향이 있습니다."
        )
    else:
        st.write("회귀계수가 0에 가까워 기온과 이용량 사이의 선형 관계가 거의 나타나지 않습니다.")

    if r2 >= 0.7:
        r2_text = "이 모델은 이용량 변화의 상당 부분을 기온 하나로 설명합니다."
    elif r2 >= 0.4:
        r2_text = "기온이 이용량 변화의 일부를 설명하지만, 다른 요인의 영향도 큽니다."
    else:
        r2_text = "기온만으로 이용량 변화를 충분히 설명하기 어렵습니다."

    st.write(f"- **결정계수 해석:** R²={r2:.4f}입니다. {r2_text}")
    st.write(
        f"- **MAE 해석:** 예측 이용량은 실제 이용량과 평균적으로 약 {mae:,.2f}건 차이가 납니다."
    )
    st.write(
        f"- **MSE 해석:** 오차를 제곱해 평균한 값으로, 큰 예측 오차에 더 큰 영향을 받습니다. 현재 값은 {mse:,.2f}입니다."
    )

    st.subheader("분석의 한계점")
    st.write(
        "이번 분석은 기온 하나만 독립변수로 사용한 단순선형회귀입니다. "
        "실제 공공자전거 이용량은 요일, 주말 여부, 강수량, 미세먼지, 계절, 공휴일 등 "
        "여러 요인의 영향을 받을 수 있습니다. 또한 이 분석은 변수 사이의 관계를 보여줄 뿐, "
        "기온이 이용량 변화의 직접적인 원인이라고 단정할 수는 없습니다."
    )

    st.caption(
        "참고: MAE와 MSE는 현재 전체 자료에 적합한 모델의 예측값으로 계산한 학습 데이터 기준 지표입니다. "
        "새로운 날짜에 대한 예측 성능을 평가하려면 별도의 학습용·평가용 데이터 분할이 필요합니다."
    )

except Exception as e:
    st.error("데이터를 처리하는 중 오류가 발생했습니다.")
    st.exception(e)
    st.info(
        "CSV 파일의 실제 열 이름과 인코딩을 확인해 주세요. "
        "공공자전거 파일은 '대여일자', '이용건수' 열이 필요하고, "
        "기온 파일은 '날짜', '평균기온' 열이 필요합니다."
    )
