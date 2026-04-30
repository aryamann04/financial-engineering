from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.settings import load_settings
from core.analysis import TradeEngine
from core.data import fetch_market_data
from options.recommender import recommend_strategies


def _candles_with_levels(df: pd.DataFrame, analysis: dict) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Price",
            )
        ]
    )
    for level in analysis["key_levels"][:8]:
        fig.add_hline(y=level["price"], line_width=1, line_dash="dot", annotation_text=level["label"])
    for zone in analysis["nearby_fvgs"][:4]:
        low_text, high_text = [part.strip() for part in zone["range"].split(" - ")]
        low, high = float(low_text), float(high_text)
        fig.add_hrect(y0=low, y1=high, line_width=0, fillcolor="rgba(255,165,0,0.18)")
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=40, b=10), xaxis_rangeslider_visible=False)
    return fig


def _snapshot_df(analysis: dict) -> pd.DataFrame:
    rows = []
    for tf, snapshot in analysis["snapshots"].items():
        rows.append(
            {
                "timeframe": tf,
                "trend": snapshot["trend"],
                "bias": snapshot["bias"],
                "structure": snapshot["structure"],
                "position": snapshot["price_position"],
                "atr": snapshot["atr"],
                "range": snapshot["recent_range"],
            }
        )
    return pd.DataFrame(rows)


def _options_table(recommendations: list[dict]) -> pd.DataFrame:
    rows = []
    for item in recommendations:
        rows.append(
            {
                "strategy": item["strategy_name"],
                "expiry": item["expiry"],
                "type": item["strategy_type"],
                "score": item["final_score"],
                "edge_pct": item["model_edge"]["edge_pct"],
                "net_debit_credit": item["net_debit_credit"],
                "max_profit": item["max_profit"],
                "max_loss": item["max_loss"],
                "reward_risk": item["reward_risk"],
            }
        )
    return pd.DataFrame(rows)


def _payoff_figure(recommendation: dict) -> go.Figure:
    curve = pd.DataFrame(recommendation["payoff_curve"], columns=["price", "pnl"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["price"], y=curve["pnl"], mode="lines", name="Expiry P/L"))
    fig.add_hline(y=0, line_dash="dot")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="Underlying Price", yaxis_title="P/L")
    return fig


def _heatmap_figure(recommendation: dict) -> go.Figure:
    heatmap = pd.DataFrame(recommendation["pnl_heatmap"], columns=["price", "vol_shift", "pnl"])
    pivot = heatmap.pivot(index="vol_shift", columns="price", values="pnl").sort_index()
    fig = go.Figure(data=go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index, colorscale="RdYlGn"))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=20, b=10), xaxis_title="Underlying Price", yaxis_title="IV Shift")
    return fig


def main() -> None:
    settings = load_settings()
    st.set_page_config(page_title="Trade Terminal", layout="wide")
    st.title("Trade Terminal")
    st.caption("Speed-first discretionary trading support terminal")

    if "engine" not in st.session_state:
        st.session_state.engine = TradeEngine(settings=settings)

    engine: TradeEngine = st.session_state.engine

    with st.sidebar:
        symbol = st.text_input("Symbol", value=settings.default_symbol)
        timeframe = st.selectbox("Chart timeframe", options=["1m", "5m", "15m", "1h"], index=1)
        auto_refresh = st.number_input("Refresh seconds", value=settings.default_refresh_seconds, min_value=5, step=5)
        refresh = st.button("Refresh")
        query = st.text_input("Ask the assistant", value="")
        ask = st.button("Run query")
        st.caption("Examples: What is the current bias? Where are the nearest downside liquidity levels?")

        st.markdown("### Options Filters")
        options_view = st.selectbox("Options view", options=["neutral", "bullish", "bearish"], index=0)
        options_max_risk = st.number_input("Max risk", value=0.0, min_value=0.0, step=100.0)
        options_limit = st.slider("Options rows", min_value=3, max_value=15, value=8)

    if refresh or "analysis" not in st.session_state or st.session_state.get("symbol") != symbol or st.session_state.get("timeframe") != timeframe:
        analysis = engine.analyze(symbol, timeframe=timeframe)
        st.session_state.previous_analysis = st.session_state.get("analysis")
        st.session_state.analysis = analysis
        st.session_state.symbol = symbol
        st.session_state.timeframe = timeframe

    analysis = st.session_state.analysis.to_dict()
    market_data = fetch_market_data(symbol, settings)
    df = market_data.frames.get(timeframe)
    if df is None or df.empty:
        df = market_data.frames.get("5m", pd.DataFrame())

    futures_tab, options_tab = st.tabs(["Futures Cockpit", "Options Strategies"])

    with futures_tab:
        top_left, top_mid, top_right = st.columns([1, 1, 1])
        top_left.metric("Bias", analysis["bias"])
        top_mid.metric("Confidence", f"{analysis['confidence']}%")
        top_right.metric("Regime", analysis["regime"])

        left, right = st.columns([2.2, 1.2])
        with left:
            st.plotly_chart(_candles_with_levels(df.tail(180), analysis), use_container_width=True)
            st.dataframe(_snapshot_df(analysis), use_container_width=True, hide_index=True)
        with right:
            st.subheader("Current Setup")
            st.json(analysis["setup"])
            st.subheader("Risk / Reward")
            st.json(analysis["risk_plan"] or {"status": "No clean risk plan"})

        lower_left, lower_mid, lower_right = st.columns([1.2, 1.2, 1.0])
        with lower_left:
            st.subheader("Liquidity Events")
            st.dataframe(pd.DataFrame(analysis["liquidity_events"]), use_container_width=True, hide_index=True)
        with lower_mid:
            st.subheader("Nearby FVGs")
            st.dataframe(pd.DataFrame(analysis["nearby_fvgs"]), use_container_width=True, hide_index=True)
        with lower_right:
            st.subheader("Macro")
            st.dataframe(pd.DataFrame(analysis["macro_context"]).T, use_container_width=True)

        tab_log, tab_backtest = st.tabs(["Recent Log", "Backtest"])
        with tab_log:
            st.write("\n".join(analysis["warnings"]) or "No warnings.")
        with tab_backtest:
            st.json(analysis["backtest_summary"] or {"status": "No backtest summary available"})

    with options_tab:
        try:
            recommendations = [item.to_dict() for item in recommend_strategies(symbol, view=options_view, max_risk=options_max_risk or None, limit=options_limit)]
        except Exception as exc:
            recommendations = []
            st.error(f"Options strategy engine unavailable: {exc}")

        if not recommendations:
            st.info("No options recommendations are available for the current filters.")
        else:
            options_df = _options_table(recommendations)
            st.dataframe(options_df, use_container_width=True, hide_index=True)
            selected_name = st.selectbox("Recommendation", options=[item["strategy_name"] + " | " + item["expiry"] for item in recommendations])
            selected = recommendations[[item["strategy_name"] + " | " + item["expiry"] for item in recommendations].index(selected_name)]

            left, right = st.columns([1.4, 1.0])
            with left:
                st.subheader("Strategy Card")
                st.json(
                    {
                        "strategy": selected["strategy_name"],
                        "type": selected["strategy_type"],
                        "view": selected["view"],
                        "expiry": selected["expiry"],
                        "legs": selected["legs"],
                        "why_this_strategy": selected["why_this_strategy"],
                        "invalidation": selected["invalidation"],
                    }
                )
                st.plotly_chart(_payoff_figure(selected), use_container_width=True)
            with right:
                st.subheader("Risk Summary")
                st.json(
                    {
                        "net_debit_credit": selected["net_debit_credit"],
                        "max_profit": selected["max_profit"],
                        "max_loss": selected["max_loss"],
                        "breakevens": selected["breakevens"],
                        "greeks": selected["greeks"],
                        "probability_of_profit": selected["probability_of_profit"],
                        "reward_risk": selected["reward_risk"],
                        "margin_estimate": selected["margin_estimate"],
                        "exposure_text": selected["exposure_text"],
                    }
                )
                st.subheader("Model Comparison")
                st.dataframe(pd.DataFrame(selected["model_comparison"]), use_container_width=True, hide_index=True)
                st.subheader("Warnings")
                st.write(selected["warnings"] or ["No special warnings."])

            lower_left, lower_right = st.columns([1.2, 1.0])
            with lower_left:
                st.subheader("P/L Heatmap")
                st.plotly_chart(_heatmap_figure(selected), use_container_width=True)
            with lower_right:
                st.subheader("Vol Regime")
                st.write(selected["vol_regime_rationale"])
                st.subheader("Edge")
                st.json(selected["model_edge"])

    if ask and query.strip():
        previous = st.session_state.get("previous_analysis")
        response = engine.query(symbol, query, timeframe=timeframe)
        st.sidebar.markdown("### Assistant")
        st.sidebar.write(response.answer)

    st.caption(f"Last refresh: {analysis['timestamp']} | Auto-refresh baseline: {auto_refresh}s")


if __name__ == "__main__":
    main()
