import pandas as pd
import plotly.graph_objects as go

NAVY, BLUE, GREEN, AMBER, RED = "#0f1b3d", "#2563eb", "#15803d", "#b45309", "#b91c1c"


def _layout(fig, title, height=300):
    fig.update_layout(title=title, height=height, margin=dict(l=10, r=10, t=45, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#14213d"), legend=dict(orientation="h", y=-0.2))
    return fig


def probability_gauge(p: float):
    color = GREEN if p < 0.34 else AMBER if p < 0.67 else RED
    fig = go.Figure(go.Indicator(mode="gauge+number", value=round(p * 100, 1), number={"suffix": "%"},
                                 gauge={"axis": {"range": [0, 100]}, "bar": {"color": color}}))
    return _layout(fig, "Estimated 12-month attrition probability", 240)


def band_counts(df: pd.DataFrame):
    order = ["Low", "Medium", "High"]
    counts = df["risk_band"].value_counts().reindex(order).fillna(0)
    fig = go.Figure(go.Bar(x=order, y=counts.values, marker_color=[GREEN, AMBER, RED]))
    fig.update_yaxes(dtick=1, title="Employees")
    return _layout(fig, "Risk bands in this batch")


def probability_hist(df: pd.DataFrame):
    fig = go.Figure(go.Histogram(x=df["probability"], nbinsx=20, marker_color=BLUE))
    fig.update_xaxes(title="Estimated probability", tickformat=".0%")
    fig.update_yaxes(title="Employees")
    return _layout(fig, "Distribution of estimated probabilities")


def flagged_counts(df: pd.DataFrame):
    c = df["flagged"].map({"Y": "Anomalous", "N": "Normal"}).value_counts()
    fig = go.Figure(go.Bar(x=c.index, y=c.values, marker_color=[RED if k == "Anomalous" else GREEN for k in c.index]))
    fig.update_yaxes(dtick=1, title="Records")
    return _layout(fig, "Normal vs anomalous records")


def reason_counts(df: pd.DataFrame):
    c = df[df["flagged"] == "Y"]["reason_code"].fillna("(none)").value_counts()
    fig = go.Figure(go.Bar(x=c.values, y=c.index, orientation="h", marker_color=BLUE))
    fig.update_xaxes(dtick=1, title="Flagged records")
    return _layout(fig, "Anomaly categories (reason codes)")


def score_distribution(df: pd.DataFrame):
    fig = go.Figure(go.Histogram(x=df["anomaly_score"], nbinsx=20, marker_color=NAVY))
    fig.update_xaxes(title="Anomaly score (0–1)", range=[0, 1])
    fig.update_yaxes(title="Records")
    return _layout(fig, "Risk distribution (anomaly scores)")


def resume_scores(df: pd.DataFrame):
    d = df.sort_values("final_score")
    fig = go.Figure(go.Bar(x=d["final_score"], y=d["label"], orientation="h", marker_color=BLUE))
    fig.update_xaxes(title="Ranking score", range=[0, max(1.0, float(d["final_score"].max()))])
    return _layout(fig, "Ranking score by candidate", max(260, 40 * len(d) + 120))


def forecast_chart(points: list[dict], department: str):
    """Expected headcount (solid line) with the 80% prediction interval shaded. The API serves no historical series."""
    d = pd.DataFrame(points)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["month"], y=d["upper_80"], line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=d["month"], y=d["lower_80"], fill="tonexty", line=dict(width=0), fillcolor="rgba(37,99,235,0.18)",
                             name="80% prediction interval"))
    fig.add_trace(go.Scatter(x=d["month"], y=d["expected_headcount"], mode="lines+markers", name="Forecast (expected headcount)",
                             line=dict(color=BLUE, width=3)))
    fig.update_yaxes(title="Headcount")
    return _layout(fig, f"Headcount forecast: {department}", 380)


def drift_counts_chart(drift: dict):
    labels = ["OK", "Watch", "Alert", "Not evaluated", "Error"]
    vals = [drift["ok"], drift["watch"], drift["alert"], drift["not_evaluated"], drift["error"]]
    fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=[GREEN, AMBER, RED, "#64748b", "#7f1d1d"]))
    fig.update_yaxes(dtick=1, title="Features")
    return _layout(fig, f"Input drift across {drift['features_checked']} monitored features")
