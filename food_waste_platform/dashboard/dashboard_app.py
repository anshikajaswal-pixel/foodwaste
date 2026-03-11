from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../")
    )
)
from food_waste_platform.ai.gemini_sheet_assistant import answer_question_with_sheet

# Allow running this file directly via:
# streamlit run food_waste_platform/dashboard/dashboard_app.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:  # pragma: no cover
    st_autorefresh = None

from food_waste_platform.analysis.anomaly_detector import detect_anomalies
from food_waste_platform.analysis.forecasting import forecast_waste
from food_waste_platform.analysis.summary import compute_summary
from food_waste_platform.config.settings import settings
from food_waste_platform.data.live_sheet_loader import fetch_live_sheet
from food_waste_platform.llm.insight_generator import generate_insights

PASTEL = ["#A8DADC", "#FFD6A5", "#BDB2FF", "#CDEAC0", "#FFCAD4", "#BDE0FE", "#D8F3DC", "#E2CFEA"]

# Initialize question state early to avoid widget/state race conditions.
if "ask_waste_question" not in st.session_state:
    st.session_state["ask_waste_question"] = ""
if "ai_question_answer" not in st.session_state:
    st.session_state["ai_question_answer"] = ""
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = []
if "use_local_calc" not in st.session_state:
    st.session_state["use_local_calc"] = True
if "last_weight_parse" not in st.session_state:
    st.session_state["last_weight_parse"] = None


def _df_signature(df: pd.DataFrame) -> str:
    raw = f"{len(df)}|{df['date'].max()}|{df['waste_kg'].sum():.3f}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _apply_common_style(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        font=dict(size=15, color="#000000"),
        title_font=dict(size=22, color="#000000"),
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color="#000000"),
            title_font=dict(color="#000000"),
        ),
        plot_bgcolor="#F8FAFC",
        paper_bgcolor="#F8FAFC",
    )
    fig.update_xaxes(title_font=dict(color="#000000"), tickfont=dict(color="#000000"))
    fig.update_yaxes(title_font=dict(color="#000000"), tickfont=dict(color="#000000"))
    return fig


def _build_figures(df: pd.DataFrame, anomalies: list[dict[str, Any]], forecast: list[dict[str, Any]]) -> dict[str, go.Figure]:
    work = df.copy()
    work["day"] = work["date"].dt.date
    daily = work.groupby("day", as_index=False)["waste_kg"].sum().rename(columns={"day": "date"})
    daily["date"] = pd.to_datetime(daily["date"])

    top_comm = (
        df.groupby("commodity", as_index=False)["waste_kg"]
        .sum()
        .sort_values("waste_kg", ascending=False)
        .head(8)
    )
    by_meal = df.groupby("meal_type", as_index=False)["waste_kg"].sum().sort_values("waste_kg", ascending=False)
    by_kitchen = df.groupby("kitchen", as_index=False)["waste_kg"].sum().sort_values("waste_kg", ascending=False)

    top = px.bar(
        top_comm,
        x="commodity",
        y="waste_kg",
        text_auto=".2f",
        title="Type Of Commodity",
        color="commodity",
        color_discrete_sequence=PASTEL,
    )
    top.update_traces(marker_line_width=0, opacity=0.92)
    top.update_layout(showlegend=False, bargap=0.28)
    top.update_xaxes(title_text="Commodity Name")
    top.update_yaxes(title_text="Amount Of Waste (In Kgs)")
    top = _apply_common_style(top)

    meal = px.pie(
        by_meal,
        names="meal_type",
        values="waste_kg",
        title="Type Of Meal<br><sup>(Weight shown in kilograms)</sup>",
        color_discrete_sequence=PASTEL,
        hole=0.25,
    )
    meal.update_traces(textposition="inside", textinfo="percent+label", textfont=dict(color="#000000", size=14))
    meal = _apply_common_style(meal)

    kitchen = px.bar(
        by_kitchen,
        x="waste_kg",
        y="kitchen",
        orientation="h",
        text_auto=".2f",
        title="Kitchen Wise Analysis",
        color="kitchen",
        color_discrete_sequence=PASTEL,
    )
    kitchen.update_layout(showlegend=False, bargap=0.25)
    kitchen.update_xaxes(title_text="Amount Of Waste (In Kgs)")
    kitchen.update_yaxes(title_text="Device Code", categoryorder="total ascending")
    kitchen = _apply_common_style(kitchen)

    trend = px.line(
        daily,
        x="date",
        y="waste_kg",
        markers=True,
        title="Daily Waste Trend",
        line_shape="spline",
        color_discrete_sequence=["#8E9AAF"],
    )
    trend.update_traces(line=dict(width=3), marker=dict(size=7, color="#5E6472"))
    trend.update_xaxes(title_text="Date")
    trend.update_yaxes(title_text="Amount Of Waste (In Kgs)")
    trend = _apply_common_style(trend)

    anomaly_fig = px.line(
        daily,
        x="date",
        y="waste_kg",
        markers=True,
        title="Anomaly Detection",
        line_shape="spline",
        color_discrete_sequence=["#8E9AAF"],
    )
    if anomalies:
        adf = pd.DataFrame(anomalies)
        adf["date"] = pd.to_datetime(adf["date"])
        anomaly_fig.add_scatter(
            x=adf["date"],
            y=adf["waste_kg"],
            mode="markers",
            name="Anomalies",
            marker=dict(color="#FF6B6B", size=11, symbol="diamond"),
        )
    anomaly_fig.update_xaxes(title_text="Date")
    anomaly_fig.update_yaxes(title_text="Amount Of Waste (In Kgs)")
    anomaly_fig = _apply_common_style(anomaly_fig)

    forecast_fig = go.Figure()
    forecast_fig.add_trace(
        go.Scatter(
            x=daily["date"],
            y=daily["waste_kg"],
            mode="lines+markers",
            line=dict(color="#7B9ACC", width=3),
            marker=dict(size=6),
            name="Historical Waste",
        )
    )
    if forecast:
        fdf = pd.DataFrame(forecast)
        fdf["date"] = pd.to_datetime(fdf["date"])
        forecast_fig.add_trace(
            go.Scatter(
                x=fdf["date"],
                y=fdf["predicted_waste_kg"],
                mode="lines+markers",
                line=dict(color="#A4C3B2", width=3, dash="dash"),
                marker=dict(size=7),
                name="Forecast Waste",
            )
        )
        forecast_fig.add_vrect(
            x0=fdf["date"].min(),
            x1=fdf["date"].max(),
            fillcolor="#E2ECE9",
            opacity=0.25,
            layer="below",
            line_width=0,
            annotation_text="Forecast Window",
            annotation_position="top left",
        )

    forecast_fig.update_layout(title="Waste Forecast Analysis")
    forecast_fig.update_xaxes(title_text="Date")
    forecast_fig.update_yaxes(title_text="Amount Of Waste (In Kgs)")
    forecast_fig = _apply_common_style(forecast_fig)

    return {
        "trend": trend,
        "meal": meal,
        "kitchen": kitchen,
        "top": top,
        "anomaly": anomaly_fig,
        "forecast": forecast_fig,
    }


def _run_live_pipeline() -> dict[str, Any]:
    df = fetch_live_sheet(settings.sheet_url)
    signature = _df_signature(df)

    cached_signature = st.session_state.get("data_signature")
    cached_payload = st.session_state.get("pipeline_payload")

    if cached_payload and cached_signature == signature:
        return cached_payload

    summary = compute_summary(df)
    anomalies = detect_anomalies(df)
    forecast = forecast_waste(df, horizon_days=14)
    report = generate_insights(
        summary=summary,
        anomalies=anomalies,
        forecast=forecast,
        api_key=settings.grok_api_key,
        model=settings.grok_model,
        base_url=settings.grok_base_url,
    )

    payload = {
        "df": df,
        "summary": summary,
        "anomalies": anomalies,
        "forecast": forecast,
        "report": report,
        "signature": signature,
    }
    st.session_state["data_signature"] = signature
    st.session_state["pipeline_payload"] = payload
    return payload


def _calc_weekly_delta(df: pd.DataFrame) -> float:
    if df.empty:
        return 0.0
    work = df.copy()
    work["day"] = work["date"].dt.floor("D")
    latest_day = work["day"].max()
    last_start = latest_day - pd.Timedelta(days=6)
    prev_start = latest_day - pd.Timedelta(days=13)
    prev_end = latest_day - pd.Timedelta(days=7)

    last_week = float(work.loc[(work["day"] >= last_start) & (work["day"] <= latest_day), "waste_kg"].sum())
    prev_week = float(work.loc[(work["day"] >= prev_start) & (work["day"] <= prev_end), "waste_kg"].sum())

    if prev_week == 0:
        return 0.0
    return ((last_week - prev_week) / prev_week) * 100


def _render_grok_insight_box(summary: dict[str, Any], anomalies: list[dict[str, Any]], df: pd.DataFrame) -> None:
    top_commodity = next(iter(summary.get("waste_by_commodity", {}).items()), ("N/A", 0.0))
    top_meal = next(iter(summary.get("waste_by_meal", {}).items()), ("N/A", 0.0))

    if anomalies:
        highest = max(anomalies, key=lambda x: x.get("waste_kg", 0))
        critical_alert = (
            f"Spike on {highest['date']} with {highest['waste_kg']:.2f} kgs "
            f"(Type: {highest.get('affected_commodity', 'N/A')})."
        )
    else:
        critical_alert = "No high-risk anomaly spikes detected in current filter window."

    top_saving = f"{top_commodity[0]} contributes the most waste ({float(top_commodity[1]):.2f} kgs)."
    action = (
        f"Reduce batch size and switch to rolling prep for {top_commodity[0]} during {top_meal[0]} service."
        if top_commodity[0] != "N/A"
        else "Track prep-to-service variance daily and cut overproduction at source."
    )

    st.markdown(
        """
        <div style="background:#E7F7EE;border:1px solid #BEE3CD;border-radius:12px;padding:16px 18px;margin:4px 0 14px 0;">
        <h3 style="margin:0 0 8px 0;color:#1F5D45;">🌱 Sustainability Insights</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(f"• **Critical Alert:** {critical_alert}")
    st.markdown(f"• **Top Saving Opportunity:** {top_saving}")
    st.markdown(f"• **Operational Action:** {action}")


def _render_anomaly_table(anomalies: list[dict[str, Any]]) -> None:
    if not anomalies:
        st.info("No anomaly alerts for current live data window.")
        return

    adf = pd.DataFrame(anomalies).copy()
    adf = adf.rename(
        columns={
            "date": "Date",
            "waste_kg": "Amount Of Waste (In Kgs)",
            "affected_commodity": "Type Of Waste",
            "z_score": "Z Score",
        }
    )
    adf.insert(0, "SNO.", range(1, len(adf) + 1))
    adf = adf[["SNO.", "Date", "Amount Of Waste (In Kgs)", "Type Of Waste", "Z Score"]]

    styled = adf.style.format({"Amount Of Waste (In Kgs)": "{:.2f}", "Z Score": "{:.2f}"})
    styled = styled.set_properties(subset=["SNO."], **{"text-align": "center"})
    styled = styled.set_properties(**{"background-color": "#FFFFFF", "color": "#111827"})
    styled = styled.set_table_styles(
        [
            {"selector": "th", "props": [("background-color", "#EAF2FF"), ("color", "#111827"), ("font-weight", "700")]},
            {"selector": "td", "props": [("background-color", "#FFFFFF"), ("color", "#111827")]},
        ]
    )

    st.dataframe(styled, use_container_width=True, hide_index=True)


def _render_executive_summary(summary: dict[str, Any], anomalies: list[dict[str, Any]]) -> None:
    top_commodity = next(iter(summary.get("waste_by_commodity", {}).keys()), "N/A")
    top_kitchen = next(iter(summary.get("waste_by_kitchen", {}).keys()), "N/A")

    st.markdown(
        """
        <div style="background:#F3F7F9;border:1px solid #E3ECEF;padding:18px 18px 10px 18px;border-radius:12px;">
        <h4 style="margin-top:0;margin-bottom:12px;">Key Insights</h4>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f"• Total waste observed: **{summary.get('total_waste_kg', 0):,.2f} kgs**")
    st.markdown(f"• Most wasted commodity: **{top_commodity}**")
    st.markdown(f"• Highest contributing kitchen: **{top_kitchen}**")
    st.markdown(f"• Anomaly alerts identified: **{len(anomalies)}**")


def _parse_report_sections(report: str) -> dict[str, str]:
    if not report:
        return {}
    normalized = report.replace("###", "##")
    parts = re.split(r"\n##\s+", "\n" + normalized.strip())
    out: dict[str, str] = {}
    for part in parts:
        part = part.strip()
        if not part:
            continue
        lines = part.splitlines()
        heading = lines[0].strip("# ").strip()
        body = "\n".join(lines[1:]).strip()
        out[heading.lower()] = body
    return out


def _render_consulting_sections(report: str, summary: dict[str, Any], anomalies: list[dict[str, Any]]) -> None:
    sections = _parse_report_sections(report)
    total = float(summary.get("total_waste_kg", 0.0))
    anomaly_count = len(anomalies)

    st.subheader("🌿 Sustainability Insights")
    c1, c2 = st.columns(2)
    c1.metric("Total Waste", f"{total:,.2f} kgs")
    c2.metric("Anomaly Count", f"{anomaly_count}")

    def render_block(title: str, icon: str, content: str) -> None:
        with st.container(border=True):
            st.markdown(f"### {icon} {title}")
            st.markdown(content if content else "_No content generated for this section._")
            st.markdown("")

    render_block(
        "Executive Summary",
        "🥗",
        sections.get("executive summary", "Summary not available."),
    )
    render_block(
        "Key Waste Drivers",
        "📉",
        sections.get("key waste drivers", "Drivers not available."),
    )
    render_block(
        "Sustainability Recommendations",
        "⚠️",
        sections.get("sustainability recommendations", sections.get("operational issues", "Recommendations not available.")),
    )


def _render_top5_commodities(summary: dict[str, Any]) -> None:
    top_items = list(summary.get("waste_by_commodity", {}).items())[:5]
    st.markdown("### 🔥 Top 5 High-Impact Commodities")
    if not top_items:
        st.info("No commodity data available.")
        return
    tdf = pd.DataFrame(top_items, columns=["Commodity", "Waste (Kgs)"])
    fig = px.bar(
        tdf.sort_values("Waste (Kgs)", ascending=True),
        x="Waste (Kgs)",
        y="Commodity",
        orientation="h",
        text_auto=".2f",
        color="Commodity",
        color_discrete_sequence=PASTEL,
        title="Top 5 Most Wasted Commodities",
    )
    fig = _apply_common_style(fig)
    fig.update_layout(showlegend=False, height=320)
    st.plotly_chart(fig, use_container_width=True)


def _build_question_context(df: pd.DataFrame) -> str:
    describe_dict = df[["waste_kg"]].describe().round(2).to_dict()
    top_5 = (
        df.groupby("commodity", as_index=False)["waste_kg"]
        .sum()
        .sort_values("waste_kg", ascending=False)
        .head(5)
    )
    meal_sums = (
        df.groupby("meal_type", as_index=False)["waste_kg"]
        .sum()
        .sort_values("waste_kg", ascending=False)
    )
    kitchen_sums = (
        df.groupby("kitchen", as_index=False)["waste_kg"]
        .sum()
        .sort_values("waste_kg", ascending=False)
    )
    payload = {
        "waste_describe": describe_dict,
        "top_5_commodities": top_5.to_dict(orient="records"),
        "meal_waste": meal_sums.to_dict(orient="records"),
        "kitchen_waste": kitchen_sums.to_dict(orient="records"),
        "rows": len(df),
        "date_min": str(df["date"].min()),
        "date_max": str(df["date"].max()),
    }
    return json.dumps(payload, indent=2)


def _build_context_string(summary: dict[str, Any], anomalies: list[dict[str, Any]]) -> str:
    total = float(summary.get("total_waste_kg", 0.0))
    top_kitchen = next(iter(summary.get("waste_by_kitchen", {}).keys()), "N/A")
    top_commodity = next(iter(summary.get("waste_by_commodity", {}).keys()), "N/A")
    anomaly_days = ", ".join([a.get("date", "") for a in anomalies[:5]]) if anomalies else "None"
    return (
        f"Total Waste: {total:.2f}kg, "
        f"Top Kitchen: {top_kitchen}, "
        f"Top Commodity: {top_commodity}, "
        f"Anomaly Days: {anomaly_days}"
    )


def _deterministic_question_answer(df: pd.DataFrame, question: str) -> str:
    q = question.lower().strip()
    if df.empty:
        return "No data is available in the current filter window."

    by_kitchen = df.groupby("kitchen")["waste_kg"].sum().sort_values(ascending=False)
    by_commodity = df.groupby("commodity")["waste_kg"].sum().sort_values(ascending=False)
    by_meal = df.groupby("meal_type")["waste_kg"].sum().sort_values(ascending=False)
    daily = df.groupby(df["date"].dt.date)["waste_kg"].sum().sort_values(ascending=False)

    if any(k in q for k in ["kitchen", "device"]):
        k = by_kitchen.index[0]
        v = float(by_kitchen.iloc[0])
        return f"{k} generates the most waste with {v:.2f} kgs in the selected period."

    if any(k in q for k in ["commodity", "item", "type"]):
        c = by_commodity.index[0]
        v = float(by_commodity.iloc[0])
        return f"{c} is the highest waste commodity with {v:.2f} kgs in the selected period."

    if any(k in q for k in ["meal", "breakfast", "lunch", "dinner"]):
        m = by_meal.index[0]
        v = float(by_meal.iloc[0])
        return f"{m} has the highest waste with {v:.2f} kgs in the selected period."

    if any(k in q for k in ["spike", "anomaly", "unusual", "high day"]):
        day = daily.index[0]
        val = float(daily.iloc[0])
        return f"The highest waste day is {pd.Timestamp(day).strftime('%Y-%m-%d')} with {val:.2f} kgs."

    total = float(df["waste_kg"].sum())
    top_k = by_kitchen.index[0]
    top_c = by_commodity.index[0]
    return (
        f"Total waste is {total:.2f} kgs. Top kitchen: {top_k}. "
        f"Top commodity: {top_c}. Ask about kitchen, meal, commodity, or anomaly for specific answers."
    )


def _is_domain_question(question: str) -> bool:
    q = question.lower().strip()
    domain_terms = [
        "food", "waste", "kitchen", "meal", "commodity", "sustainability", "compost",
        "recycle", "anomaly", "forecast", "trend", "dashboard", "kg", "plate", "prep",
        "production", "device", "insight", "reduce", "wastage",
    ]
    return any(term in q for term in domain_terms)


def _answer_data_question(
    df: pd.DataFrame,
    question: str,
    summary: dict[str, Any],
    anomalies: list[dict[str, Any]],
) -> str:
    if not question.strip():
        return "Please enter a question."
    if not _is_domain_question(question):
        return "I’m sorry, but I can only assist with questions related to food waste data and sustainability insights."

    try:
        answer = answer_question_with_sheet(question)
        if answer and ("PERMISSION_DENIED" in answer or "Assistant error" in answer):
            fallback = _answer_local_fallback(df, summary, anomalies, question)
            if fallback:
                return fallback
        return answer
    except Exception as exc:
        fallback = _answer_local_fallback(df, summary, anomalies, question)
        if fallback:
            return fallback
        return f"Assistant error: {exc}"


def _get_chat_answer(
    df: pd.DataFrame,
    question: str,
    summary: dict[str, Any],
    anomalies: list[dict[str, Any]],
    use_local_calc: bool,
) -> str:
    if not question.strip():
        return "Please enter a question."
    if not _is_domain_question(question):
        return "Iâ€™m sorry, but I can only assist with questions related to food waste data and sustainability insights."

    if use_local_calc:
        local_answer = _answer_local_question(df, question)
        return local_answer or _answer_data_question(df, question, summary, anomalies)

    try:
        answer = answer_question_with_sheet(question)
        if answer and ("PERMISSION_DENIED" in answer or "Assistant error" in answer):
            fallback = _answer_local_fallback(df, summary, anomalies, question)
            if fallback:
                return fallback
        return answer or "No response from Gemini."
    except Exception as exc:
        fallback = _answer_local_fallback(df, summary, anomalies, question)
        if fallback:
            return fallback
        return f"Assistant error: {exc}"


def _extract_weight_kg(question: str) -> float | None:
    q = question.lower().strip()
    # Normalize decimal commas to dots (e.g., 0,3 -> 0.3)
    q = re.sub(r"(?<=\d),(?=\d)", ".", q)
    kg_matches = list(re.finditer(r"(\d+(?:\.\d+)?)\s*kg\b", q))
    if not kg_matches:
        kg_matches = list(re.finditer(r"(\d+(?:\.\d+)?)\s*kgs\b", q))
    if kg_matches:
        return float(kg_matches[-1].group(1))
    return None


def _extract_query_date(question: str) -> pd.Timestamp | None:
    q = question.lower().strip()
    # Remove weight fragments to avoid parsing them as dates.
    q = re.sub(r"\d+(?:\.\d+)?\s*kg(?:s)?\b", " ", q)
    # Normalize ordinals: 1st -> 1, 2nd -> 2
    q = re.sub(r"(\d{1,2})(st|nd|rd|th)\b", r"\1", q)

    # Explicit numeric date patterns.
    m = re.search(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b", q)
    if m:
        day, month, year = m.groups()
        year_i = int(year)
        if year_i < 100:
            year_i += 2000
        return pd.Timestamp(year=year_i, month=int(month), day=int(day))

    m = re.search(r"\b(\d{4})[/-](\d{1,2})[/-](\d{1,2})\b", q)
    if m:
        year, month, day = m.groups()
        return pd.Timestamp(year=int(year), month=int(month), day=int(day))

    # Month-name dates (e.g., 1 January 2026).
    if re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|sept|oct|nov|dec)\b", q):
        try:
            dt = pd.to_datetime(q, dayfirst=True, errors="coerce")
            if pd.notna(dt):
                return pd.Timestamp(dt.date())
        except Exception:
            pass

    # Fuzzy parse as last resort.
    try:
        from dateutil import parser as dateparser

        dt = dateparser.parse(q, dayfirst=True, fuzzy=True)
        if dt:
            return pd.Timestamp(dt.date())
    except Exception:
        return None
    return None


def _answer_local_question(df: pd.DataFrame, question: str) -> str | None:
    q = question.lower().strip()
    query_date = _extract_query_date(question)
    if query_date is not None:
        if "date" not in df.columns or "waste_kg" not in df.columns:
            return "Date or weight column is missing in the current dataset."
        work = df.copy()
        work["date"] = pd.to_datetime(work["date"], errors="coerce", dayfirst=True)
        work["waste_kg"] = pd.to_numeric(work["waste_kg"], errors="coerce")
        work = work.dropna(subset=["date", "waste_kg"])
        day = query_date.date()
        work["day"] = work["date"].dt.date
        if "production" in q and "waste" in q:
            if "waste_type" not in work.columns:
                return "Waste type column is missing, so I cannot isolate production waste."
            prod = work[work["waste_type"].astype(str).str.contains("Production", case=False, na=False)]
            total = prod.loc[prod["day"] == day, "waste_kg"].sum()
            if total <= 0:
                return f"No production waste recorded on {day.isoformat()}."
            return f"Production waste on {day.isoformat()} was {total:.2f} kg."
        total = work.loc[work["day"] == day, "waste_kg"].sum()
        if total <= 0:
            return f"No waste recorded on {day.isoformat()}."
        return f"Total waste on {day.isoformat()} was {total:.2f} kg."
    if "weight" in q or "kg" in q or "exactly" in q:
        target = _extract_weight_kg(question)
        if target is None:
            st.session_state["last_weight_parse"] = None
            return "Please include a numeric weight (e.g., 0.3 kg)."
        st.session_state["last_weight_parse"] = target
        work = df.copy()
        if "waste_kg" not in work.columns and "weight_(kg)" in work.columns:
            work = work.rename(columns={"weight_(kg)": "waste_kg"})
        if "date" not in work.columns:
            return "Date column is missing in the current dataset."
        if "waste_kg" not in work.columns:
            return "Weight column is missing in the current dataset."
        work["waste_kg"] = pd.to_numeric(work["waste_kg"], errors="coerce")
        work["date"] = pd.to_datetime(work["date"], errors="coerce", dayfirst=True)
        work = work.dropna(subset=["waste_kg", "date"])
        hits = work[work["waste_kg"].sub(target).abs() <= 1e-6]
        if hits.empty:
            return f"No rows found with weight exactly {target:.2f} kg in the current filters."
        days = sorted({d.date().isoformat() for d in hits["date"]})
        if len(days) <= 20:
            return f"Exact weight {target:.2f} kg occurred on {len(days)} days: {', '.join(days)}."
        return f"Exact weight {target:.2f} kg occurred on {len(days)} days. First 20: {', '.join(days[:20])}."
    if "production waste" in q:
        if "waste_type" in df.columns and "waste_kg" in df.columns:
            prod = df.loc[df["waste_type"].astype(str).str.contains("Production", case=False, na=False), "waste_kg"]
            total = pd.to_numeric(prod, errors="coerce").sum()
            return f"The total amount of production waste is {total:.2f} kg."
    if "least waste" in q or "minimum waste" in q:
        if "date" in df.columns and "waste_kg" in df.columns:
            work = df.copy()
            work["date"] = pd.to_datetime(work["date"], errors="coerce", dayfirst=True)
            work["waste_kg"] = pd.to_numeric(work["waste_kg"], errors="coerce")
            work = work.dropna(subset=["date", "waste_kg"])
            by_day = work.groupby(work["date"].dt.date)["waste_kg"].sum()
            if not by_day.empty:
                d = by_day.idxmin()
                return f"The least waste was on {d.isoformat()} with {float(by_day.loc[d]):.2f} kg."
    if "most waste" in q or "highest waste" in q:
        if "date" in df.columns and "waste_kg" in df.columns:
            work = df.copy()
            work["date"] = pd.to_datetime(work["date"], errors="coerce", dayfirst=True)
            work["waste_kg"] = pd.to_numeric(work["waste_kg"], errors="coerce")
            work = work.dropna(subset=["date", "waste_kg"])
            by_day = work.groupby(work["date"].dt.date)["waste_kg"].sum()
            if not by_day.empty:
                d = by_day.idxmax()
                return f"The most waste was on {d.isoformat()} with {float(by_day.loc[d]):.2f} kg."
    return None


def _answer_local_fallback(
    df: pd.DataFrame,
    summary: dict[str, Any],
    anomalies: list[dict[str, Any]],
    question: str,
) -> str | None:
    q = question.lower().strip()
    local = _answer_local_question(df, question)
    if local:
        return local

    if "total waste" in q:
        total = float(summary.get("total_waste_kg", 0.0))
        return f"The total waste is {total:.2f} kg."

    if "top kitchen" in q or "which kitchen" in q or "maximum waste" in q:
        by_kitchen = summary.get("waste_by_kitchen", {})
        if by_kitchen:
            kitchen = max(by_kitchen, key=by_kitchen.get)
            return f"Top kitchen: {kitchen} with {float(by_kitchen[kitchen]):.2f} kg."

    if "top commodity" in q or "which commodity" in q:
        by_commodity = summary.get("waste_by_commodity", {})
        if by_commodity:
            commodity = max(by_commodity, key=by_commodity.get)
            return f"Top commodity: {commodity} with {float(by_commodity[commodity]):.2f} kg."

    if "meal" in q and ("highest" in q or "top" in q):
        by_meal = summary.get("waste_by_meal", {})
        if by_meal:
            meal = max(by_meal, key=by_meal.get)
            return f"Highest waste meal: {meal} with {float(by_meal[meal]):.2f} kg."

    if "spike" in q or "anomal" in q:
        if anomalies:
            top = max(anomalies, key=lambda x: x.get("waste_kg", 0))
            return f"Spike on {top.get('date')} with {float(top.get('waste_kg', 0)):.2f} kg."
        return "No anomaly spikes found in the current filters."

    return None


def app() -> None:
    st.set_page_config(page_title="Waste Monitoring Analysis", layout="wide")

    st.markdown(
        """
        <style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');
  :root {
    --ink: #3A1D10;
    --ink-soft: #7A5A4A;
    --glass: rgba(255, 248, 242, 0.7);
    --glass-strong: rgba(255, 246, 236, 0.95);
    --stroke: rgba(176, 120, 90, 0.25);
    --accent: #FF6B2C;
    --accent-2: #FFB703;
    --shadow: 0 18px 40px rgba(12, 24, 48, 0.15);
}
.stApp {
    background:
      radial-gradient(1200px 520px at 10% 0%, rgba(255, 107, 44, 0.2), transparent 60%),
      radial-gradient(900px 520px at 90% 10%, rgba(255, 183, 3, 0.2), transparent 55%),
      linear-gradient(180deg, #FFF4EC 0%, #FFE9D9 60%, #FFE2CF 100%);
    color: var(--ink);
}
        .block-container {padding-top: 2rem; padding-bottom: 3rem; max-width: 100%; padding-left: 2.5rem; padding-right: 2.5rem;}
html, body, [class*="css"]  {font-family: 'Manrope', 'Source Sans Pro', sans-serif;}
  h1, h2, h3, h4, h5, h6, p, span, label, div, li {color: var(--ink) !important;}
h1 {letter-spacing: -0.8px; font-weight: 800;}

/* Sidebar glass */
  section[data-testid="stSidebar"] {
      background: linear-gradient(180deg, rgba(255,250,244,0.92) 0%, rgba(255,239,224,0.98) 100%);
      border-right: 1px solid var(--stroke);
      box-shadow: 8px 0 20px rgba(15, 23, 42, 0.05);
  }
  section[data-testid="stSidebar"] * {color: var(--ink) !important; font-weight: 600;}
  section[data-testid="stSidebar"] div[data-baseweb="select"] > div {background: #FFF6EE !important;}
  section[data-testid="stSidebar"] [data-baseweb="tag"] {background: #FFE6D2 !important; border-radius: 14px !important;}
section[data-testid="stSidebar"] [data-baseweb="tag"] span {font-weight: 700 !important;}
  section[data-testid="stSidebar"] .stDateInput > div > div {background: #FFF6EE !important;}

/* Metrics */
div[data-testid="stMetric"] {
    background: var(--glass-strong);
    border: 1px solid var(--stroke);
    border-radius: 18px;
    padding: 12px 16px;
    box-shadow: var(--shadow);
    backdrop-filter: blur(12px);
}

/* Buttons */
.stButton > button {
    background: rgba(255, 107, 44, 0.12) !important;
    color: var(--ink) !important;
    border: 1px solid rgba(255, 107, 44, 0.35) !important;
    border-radius: 999px !important;
    padding: 0.55rem 1.1rem;
    box-shadow: 0 10px 20px rgba(255, 107, 44, 0.2);
}
.stButton > button:hover {background: rgba(255, 107, 44, 0.2) !important;}

/* Chat */
div[data-testid="stChatMessage"] {
    background: var(--glass);
    border: 1px solid var(--stroke);
    border-radius: 18px;
    box-shadow: var(--shadow);
    backdrop-filter: blur(14px);
}
        .chat-input {
            display: flex;
            gap: 0.75rem;
            align-items: center;
            background: var(--glass-strong);
            border: 1px solid rgba(255, 107, 44, 0.35);
            border-radius: 999px;
            padding: 0.45rem 0.75rem;
            box-shadow: 0 16px 32px rgba(255, 107, 44, 0.18);
            backdrop-filter: blur(16px);
        }
        .chat-input input {
            background: transparent !important;
            border: none !important;
            color: var(--ink) !important;
        }

/* DataFrame */
  div[data-testid="stDataFrame"] [role="columnheader"] {background: #FFE4CC !important; font-weight: 700 !important;}

  a {color: var(--accent) !important;}        div[data-testid="stTextInput"] > div {
              background: rgba(255, 250, 244, 0.95) !important;
              border: 1px solid rgba(255, 107, 44, 0.35) !important;
              border-radius: 999px !important;
              padding: 0.35rem 0.6rem !important;
              box-shadow: 0 16px 32px rgba(255, 107, 44, 0.18);
              backdrop-filter: blur(16px);
          }
          div[data-testid="stTextInput"] input {
              background: #FFF6EE !important;
              border: none !important;
              color: var(--ink) !important;
          }
          div[data-testid="stTextInput"] input::placeholder {
              color: #8A5B45 !important;
          }
        div[data-baseweb="input"] {
            background: transparent !important;
        }
        div[data-baseweb="input"] input {
            background: #FFF6EE !important;
            color: var(--ink) !important;
            caret-color: var(--accent) !important;
        }
        div[data-baseweb="input"] input::placeholder {
            color: #8A5B45 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Sustainability Insights")
    st.divider()

    # Auto-refresh every 30 seconds.
    if st_autorefresh is not None:
        st_autorefresh(interval=settings.dashboard_refresh_ms, key="sheet_live_refresh")
    else:
        st.markdown(
            f"<script>setTimeout(function(){{window.location.reload();}},{settings.dashboard_refresh_ms});</script>",
            unsafe_allow_html=True,
        )

    try:
        payload = _run_live_pipeline()
    except Exception as exc:
        st.error(f"Live pipeline error: {exc}")
        st.info("Check Google Sheet sharing permissions, schema, and internet connectivity.")
        return

    df = payload["df"]
    base_report = payload["report"]

    # Sidebar filters
    st.sidebar.header("Filters")
    kitchens = sorted(df["kitchen"].dropna().unique().tolist())
    meals = sorted(df["meal_type"].dropna().unique().tolist())

    min_date = df["date"].min().date()
    max_date = df["date"].max().date()
    date_range = st.sidebar.date_input("Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date)
    st.sidebar.divider()

    select_all_kitchens = st.sidebar.checkbox("Select All Kitchen ID", value=True)
    selected_kitchens = st.sidebar.multiselect(
        "Kitchen ID",
        kitchens,
        default=(kitchens if select_all_kitchens else []),
    )

    st.sidebar.divider()
    select_all_meals = st.sidebar.checkbox("Select All Meal Type", value=True)
    selected_meals = st.sidebar.multiselect(
        "Meal Type",
        meals,
        default=(meals if select_all_meals else []),
    )
    st.sidebar.divider()
    st.sidebar.subheader("Assistant Mode")
    use_local_calc = st.sidebar.checkbox(
        "Prefer local calculations (recommended)",
        value=st.session_state.get("use_local_calc", True),
        help="Uses exact calculations from the full dataset for numeric questions. Gemini is used for narrative answers.",
    )
    st.session_state["use_local_calc"] = use_local_calc

    filtered_df = df.copy()
    if len(date_range) == 2:
        start_dt = pd.to_datetime(date_range[0])
        end_dt = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        filtered_df = filtered_df[(filtered_df["date"] >= start_dt) & (filtered_df["date"] <= end_dt)]
    filtered_df = filtered_df[filtered_df["kitchen"].isin(selected_kitchens)]
    filtered_df = filtered_df[filtered_df["meal_type"].isin(selected_meals)]

    if filtered_df.empty:
        st.warning("No data available for selected filters. Adjust filters to continue.")
        return

    summary = compute_summary(filtered_df)
    anomalies = detect_anomalies(filtered_df)
    forecast = forecast_waste(filtered_df, horizon_days=14)
    report = generate_insights(
        summary=summary,
        anomalies=anomalies,
        forecast=forecast,
        api_key=settings.grok_api_key,
        model=settings.grok_model,
        base_url=settings.grok_base_url,
    )

    with st.container(border=True):
        st.subheader("EcoSync AI Assistant")
        toggle_label = "Prefer local calculations (exact, recommended)"
        use_local_calc = st.toggle(toggle_label, value=st.session_state.get("use_local_calc", True))
        st.session_state["use_local_calc"] = use_local_calc
        parse_hint = st.session_state.get("last_weight_parse")
        parse_text = f" | last weight parse: {parse_hint}" if parse_hint is not None else ""
        st.caption(f"Assistant mode: {'Local (exact calculations)' if use_local_calc else 'Gemini (best effort)'}{parse_text}")
        for message in st.session_state["chat_messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        c_input, c_send = st.columns([10, 1])
        with c_input:
            prompt = st.text_input("Ask a question about waste data", label_visibility="collapsed", placeholder="Ask a question about waste data", key="chat_text")
        with c_send:
            send = st.button("Send", use_container_width=True)

        if send and prompt:
            st.session_state["chat_messages"].append({"role": "user", "content": prompt})
            with st.spinner("Assistant is typing..."):
                answer = _get_chat_answer(filtered_df, prompt, summary, anomalies, use_local_calc)
            st.session_state["chat_messages"].append({"role": "assistant", "content": answer})
            st.rerun()

        st.caption(
            "Sample Queries: Which kitchen generates the most waste? | Which commodity contributes the highest waste? "
            "| Which meal type has the highest waste? | Were there anomaly spike days?"
        )

    st.divider()

    figures = _build_figures(filtered_df, anomalies, forecast)

    _render_grok_insight_box(summary, anomalies, filtered_df)
    st.divider()

    total_waste = float(summary["total_waste_kg"])
    weekly_delta = _calc_weekly_delta(filtered_df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Waste (In Kgs)", f"{total_waste:.2f}", delta=f"{weekly_delta:+.1f}% Vs. Last Week")
    c2.metric("Rows Loaded", f"{len(filtered_df)}")
    c3.metric("Anomaly Days", f"{len(anomalies)}")

    st.divider()
    st.plotly_chart(figures["top"], use_container_width=True)

    st.divider()
    left, right = st.columns(2)
    with left:
        st.plotly_chart(figures["meal"], use_container_width=True)
        st.plotly_chart(figures["trend"], use_container_width=True)
    with right:
        st.plotly_chart(figures["kitchen"], use_container_width=True)
        st.plotly_chart(figures["forecast"], use_container_width=True)

    st.divider()
    st.plotly_chart(figures["anomaly"], use_container_width=True)

    st.divider()
    st.subheader("Anomaly Alerts")
    _render_anomaly_table(anomalies)

    st.divider()
    _render_executive_summary(summary, anomalies)

    st.divider()
    _render_top5_commodities(summary)
    st.divider()
    _render_consulting_sections(report or base_report, summary, anomalies)

    st.sidebar.markdown(
        f"**Data Last Updated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )

    with st.expander("Debug Payload"):
        debug = {
            "signature": payload["signature"],
            "summary": summary,
            "forecast_points": len(forecast),
        }
        st.code(json.dumps(debug, indent=2), language="json")


if __name__ == "__main__":
    app()
