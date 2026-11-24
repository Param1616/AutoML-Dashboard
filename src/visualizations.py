import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# SaaS color palette
THEME_FONT = dict(family="-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif", color="#1e293b")
BG_COLOR = "#ffffff"
PLOT_BG = "#f8fafc"
GRID_COLOR = "#e2e8f0"
COLOR_PRIMARY = "#2563eb"
COLOR_NAVY = "#1e3a8a"
COLOR_ACCENTS = ["#2563eb", "#06b6d4", "#10b981", "#8b5cf6", "#f59e0b", "#ec4899", "#3b82f6", "#14b8a6"]


def plot_distributions(df, num_cols=None):
    """
    Renders clean subplots of feature histograms and box distributions.
    """
    if num_cols is None:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return None

    # Limit to first 9 columns to keep UI performant and clean
    selected_cols = num_cols[:9]
    cols = min(3, len(selected_cols))
    rows = (len(selected_cols) + cols - 1) // cols

    fig = make_subplots(rows=rows, cols=cols, subplot_titles=selected_cols, vertical_spacing=0.12)
    for i, col in enumerate(selected_cols):
        r, c = divmod(i, cols)
        color = COLOR_ACCENTS[i % len(COLOR_ACCENTS)]
        fig.add_trace(
            go.Histogram(
                x=df[col].dropna(),
                name=col,
                marker=dict(color=color, line=dict(color="#ffffff", width=0.8)),
                opacity=0.85
            ),
            row=r + 1, col=c + 1
        )

    fig.update_layout(
        showlegend=False,
        height=240 * rows,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=PLOT_BG,
        font=THEME_FONT
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)
    return fig


def plot_correlation(df):
    """
    Renders an interactive correlation heatmap with annotated values.
    """
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return None

    corr = num_df.corr().round(2)
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        labels=dict(color="Correlation")
    )
    fig.update_layout(
        height=450,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=BG_COLOR,
        font=THEME_FONT,
        coloraxis_colorbar=dict(thickness=14, len=0.8)
    )
    return fig


def plot_missing(df):
    """
    Bar chart showing percentage of missing values per column.
    """
    missing = (df.isnull().sum() / len(df) * 100).round(2)
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        return None

    fig = px.bar(
        x=missing.index,
        y=missing.values,
        labels={"x": "Feature", "y": "Missing (%)"},
        color=missing.values,
        color_continuous_scale="Reds",
        text=missing.values
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        height=320,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=PLOT_BG,
        font=THEME_FONT,
        coloraxis_showscale=False
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, range=[0, max(missing.max() * 1.25, 10)])
    return fig


def plot_categorical(df, cat_cols=None):
    """
    Renders bar plots for top categories in categorical features.
    """
    if cat_cols is None:
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if not cat_cols:
        return None

    selected = cat_cols[:4]
    cols = min(2, len(selected))
    rows = (len(selected) + cols - 1) // cols
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=selected, vertical_spacing=0.18)

    for i, col in enumerate(selected):
        r, c = divmod(i, cols)
        vc = df[col].value_counts().head(8)
        fig.add_trace(
            go.Bar(
                x=vc.index.astype(str),
                y=vc.values,
                marker=dict(color=COLOR_ACCENTS[i % len(COLOR_ACCENTS)]),
                name=col
            ),
            row=r + 1, col=c + 1
        )

    fig.update_layout(
        showlegend=False,
        height=260 * rows,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=PLOT_BG,
        font=THEME_FONT
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR)
    return fig


def plot_outliers(df, num_cols=None):
    """
    Box plots to visualize data distribution and outliers.
    """
    if num_cols is None:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        return None

    selected = num_cols[:6]
    fig = go.Figure()
    for i, col in enumerate(selected):
        fig.add_trace(
            go.Box(
                y=df[col].dropna(),
                name=col,
                marker=dict(color=COLOR_ACCENTS[i % len(COLOR_ACCENTS)]),
                boxpoints="outliers"
            )
        )

    fig.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=PLOT_BG,
        font=THEME_FONT,
        showlegend=False
    )
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR)
    return fig


def plot_model_comparison(results, task_type):
    """
    Grouped bar chart comparing all models across key metrics.
    """
    rows = []
    if task_type == "classification":
        metrics = ["Accuracy", "F1 Score", "Precision", "Recall"]
    else:
        metrics = ["R2 Score", "RMSE", "MAE"]

    for name, info in results.items():
        if not info.get("metrics") or info.get("error"):
            continue
        row = {"Model": name}
        for m in metrics:
            row[m] = info["metrics"].get(m, 0.0) or 0.0
        rows.append(row)

    if not rows:
        return None

    df = pd.DataFrame(rows).sort_values(metrics[0], ascending=(task_type != "classification" and metrics[0] == "RMSE"))
    fig = px.bar(
        df,
        x="Model",
        y=metrics,
        barmode="group",
        color_discrete_sequence=[COLOR_PRIMARY, "#06b6d4", "#10b981", "#8b5cf6"]
    )
    fig.update_layout(
        height=380,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=PLOT_BG,
        font=THEME_FONT,
        xaxis_tickangle=-25,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR)
    return fig


def plot_training_time(results):
    """
    Horizontal bar chart showing training latency per model.
    """
    names = [n for n, i in results.items() if not i.get("error")]
    times = [results[n]["training_time"] for n in names]
    if not names:
        return None

    df = pd.DataFrame({"Model": names, "Time (s)": times}).sort_values("Time (s)", ascending=True)
    fig = px.bar(
        df,
        x="Time (s)",
        y="Model",
        orientation="h",
        color="Time (s)",
        color_continuous_scale="Blues",
        text=df["Time (s)"].apply(lambda x: f"{x:.3f}s")
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=340,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=PLOT_BG,
        font=THEME_FONT,
        coloraxis_showscale=False
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR)
    fig.update_yaxes(showgrid=False)
    return fig


def plot_radar(results, task_type):
    """
    Radar chart for multi-metric comparison of top models.
    """
    if task_type != "classification":
        return None

    metrics = ["Accuracy", "F1 Score", "Precision", "Recall"]
    top_models = sorted(
        [(n, i) for n, i in results.items() if i.get("metrics") and not i.get("error")],
        key=lambda x: x[1]["metrics"].get("F1 Score", 0) or 0,
        reverse=True
    )[:5]

    if not top_models:
        return None

    fig = go.Figure()
    for (name, info), color in zip(top_models, COLOR_ACCENTS):
        vals = [info["metrics"].get(m, 0) or 0 for m in metrics]
        fig.add_trace(
            go.Scatterpolar(
                r=vals + [vals[0]],
                theta=metrics + [metrics[0]],
                fill="toself",
                name=name,
                line=dict(color=color, width=2),
                opacity=0.6
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], gridcolor=GRID_COLOR),
            angularaxis=dict(gridcolor=GRID_COLOR)
        ),
        height=380,
        margin=dict(l=40, r=40, t=40, b=20),
        paper_bgcolor=BG_COLOR,
        font=THEME_FONT,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
    )
    return fig


def plot_confusion_matrix(cm, class_names):
    """
    Heatmap displaying classification confusion matrix.
    """
    if cm is None or not class_names:
        return None

    fig = px.imshow(
        cm,
        x=[f"Pred {c}" for c in class_names],
        y=[f"True {c}" for c in class_names],
        text_auto=True,
        color_continuous_scale="Blues",
        labels=dict(x="Predicted Class", y="Actual Class", color="Count")
    )
    fig.update_layout(
        height=340,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor=BG_COLOR,
        font=THEME_FONT,
        coloraxis_showscale=False
    )
    return fig


def plot_residuals(y_test, y_pred):
    """
    Scatter plot of actual vs predicted values and residuals for regression.
    """
    if y_test is None or y_pred is None:
        return None

    y_test_arr = np.array(y_test)
    y_pred_arr = np.array(y_pred)
    residuals = y_test_arr - y_pred_arr

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Actual vs. Predicted", "Residuals vs. Predicted"]
    )

    # Actual vs Predicted
    fig.add_trace(
        go.Scatter(
            x=y_test_arr, y=y_pred_arr,
            mode="markers",
            marker=dict(color=COLOR_PRIMARY, opacity=0.65, size=6),
            name="Predictions"
        ),
        row=1, col=1
    )
    min_val = min(y_test_arr.min(), y_pred_arr.min())
    max_val = max(y_test_arr.max(), y_pred_arr.max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val], y=[min_val, max_val],
            mode="lines",
            line=dict(color="#ef4444", dash="dash", width=1.5),
            name="Ideal Match"
        ),
        row=1, col=1
    )

    # Residuals
    fig.add_trace(
        go.Scatter(
            x=y_pred_arr, y=residuals,
            mode="markers",
            marker=dict(color="#06b6d4", opacity=0.65, size=6),
            name="Residuals"
        ),
        row=1, col=2
    )
    fig.add_trace(
        go.Scatter(
            x=[y_pred_arr.min(), y_pred_arr.max()], y=[0, 0],
            mode="lines",
            line=dict(color="#ef4444", dash="dash", width=1.5),
            name="Zero Error"
        ),
        row=1, col=2
    )

    fig.update_layout(
        height=340,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=PLOT_BG,
        font=THEME_FONT,
        showlegend=False
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR)
    fig.update_yaxes(showgrid=True, gridcolor=GRID_COLOR)
    return fig


def plot_feature_importance(importances, top_n=12):
    """
    Renders top feature importances horizontal bar chart.
    """
    if not importances:
        return None

    items = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:top_n]
    if not items:
        return None
    features, values = zip(*items)

    df = pd.DataFrame({"Feature": list(features), "Importance": list(values)})
    df = df.sort_values("Importance", ascending=True)

    fig = px.bar(
        df,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Blues",
        text=df["Importance"].apply(lambda x: f"{x:.4f}")
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(
        height=max(280, 28 * len(df)),
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor=BG_COLOR,
        plot_bgcolor=PLOT_BG,
        font=THEME_FONT,
        coloraxis_showscale=False
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR)
    fig.update_yaxes(showgrid=False)
    return fig
