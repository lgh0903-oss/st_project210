# 주춤한 구간 요약표
if selected_lows:
    low_summary = bike.iloc[selected_lows][["대여일자", "대여건수", "7일 이동평균"]].copy()
    low_summary["대여일자"] = low_summary["대여일자"].dt.strftime("%Y-%m-%d")
    low_summary = low_summary.rename(columns={
        "대여일자": "주춤 구간 날짜", "대여건수": "해당 날짜 대여 건수",
        "7일 이동평균": "7일 평균 대여 건수"
    })
    with st.expander("주춤한 구간의 날짜와 대여량 확인"):
        st.dataframe(
            low_summary.style.format({
                "해당 날짜 대여 건수": "{:,.0f}",
                "7일 평균 대여 건수": "{:,.0f}",
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
    "**자료 단위: 월별 6개 점**  \n"
    "자전거 일별 자료 181개를 월별로 묶어 **월평균 일일 대여 건수**를 계산한 뒤, "
    "같은 달의 평균기온과 비교합니다. 앞의 두 시간 흐름 그래프와 달리 "
    "이 그래프는 두 변수 사이의 관계를 보여주는 산점도입니다."
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
        "월별 점이 6개뿐이므로 결과는 참고용이며 인과관계를 증명하지 않습니다."
    )
else:
    st.warning("두 자료에서 겹치는 월이 2개월 미만이라 관계 회귀분석을 할 수 없습니다.")

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
1. 월별 기온은 6개 자료이므로 월별 추세를 보여줄 뿐, 일별 기온 변화는 보여주지 않습니다.
2. 공공자전거 자료는 일별 181개 자료이며, 그래프 3에서만 월평균 일일 대여 건수로 집계했습니다.
3. 그래프 3의 상관관계만으로 기온이 대여량 변화의 원인이라고 결론 내릴 수 없습니다. 비, 요일, 공휴일, 계절 등도 영향을 줄 수 있습니다.
4. 선형회귀 직선은 전체적인 경향을 요약합니다. 실제 데이터의 모든 오르내림을 설명하지는 않습니다.
""")

st.success("분석 화면을 모두 불러왔습니다. 그래프의 점이나 선에 마우스를 올리면 세부 값을 확인할 수 있습니다.")
