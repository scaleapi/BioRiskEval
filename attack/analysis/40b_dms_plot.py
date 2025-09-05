# pyright: reportMissingTypeStubs=false
import os
import glob
import pandas as pd  # type: ignore[import]
import numpy as np
import plotly.graph_objects as go  # type: ignore[import-untyped]
from plotly.subplots import make_subplots  # type: ignore[import-untyped]
from plot_utils import (
    get_best_train_rmse_layer_stats,
)

# === Configuration ===
MODEL_NAME = "nemo2_evo2_40b_1m"

BASE_DIR_FULL = \
    "/workspaces/src/models/bionemo-framework/attack/analysis/dms_results/likelihood/virus_reproduction/full/Virus"
BASE_DIR_H5 = \
    "/workspaces/src/models/bionemo-framework/attack/analysis/dms_results/likelihood/virus_reproduction/h5_samples=624_seed=42_test/Virus"
PROBE_CSV = \
    "/workspaces/src/models/bionemo-framework/attack/analysis/dms_results/probe_results/closed_form/probe_results_40b.csv"


def read_zero_shot_spearman_values(zs_base_dir: str, model_dir: str) -> list:
    """Read per-dataset |rho| from *_fitness.csv for the given model directory.

    Returns a list of floats. Safely handles missing columns and non-numeric entries.
    """
    model_path = os.path.join(zs_base_dir, model_dir)
    values: list = []
    if not os.path.isdir(model_path):
        return values
    for file_path in glob.glob(os.path.join(model_path, "*_fitness.csv")):
        try:
            df = pd.read_csv(file_path)
            if df is None or df.empty:
                continue
            val = None
            if "spearman" in df.columns:
                v = df.loc[0, "spearman"]
                if pd.notna(v):
                    try:
                        val = abs(float(v))  # type: ignore[arg-type]
                    except Exception:
                        val = None
            # Fallback to first cell if needed
            if val is None:
                try:
                    v = df.iloc[0, 0]
                    if pd.notna(v):
                        val = abs(float(v))  # type: ignore[arg-type]
                except Exception:
                    val = None
            if val is not None:
                values.append(val)
        except Exception:
            continue
    return values


# === Load data ===
zs_full_values = read_zero_shot_spearman_values(BASE_DIR_FULL, MODEL_NAME)
zs_h5_values = read_zero_shot_spearman_values(BASE_DIR_H5, MODEL_NAME)

# Probe values from closed_form following analysis notebook (best train_rmse layer)
layer_indices = list(range(0, 50))
_, probe_values_h5, probe_mean_h5 = get_best_train_rmse_layer_stats(PROBE_CSV, layer_indices)
if probe_values_h5 is None:
    probe_values_h5 = []


# === Build plot ===
# One figure with two groups on x-axis: 0 => Full, 1 => H5
group_positions = {"Full": 0, "H5": 1}

fig = make_subplots(rows=1, cols=1)

# Bars
# Full group: one bar (Zero-shot only)
fig.add_trace(
    go.Bar(
        x=[group_positions["Full"]],
        y=[float(np.mean(zs_full_values))] if len(zs_full_values) > 0 else [np.nan],
        name="Evo2-7B (Log-Likelihood)",
        marker_color="#3366CC",
        opacity=0.7,
        width=0.3,
        offset=0,
        showlegend=True,
        legendgroup="zs",
    ),
    row=1, col=1,
)

# H5 group: two bars (Zero-shot and Probe)
fig.add_trace(
    go.Bar(
        x=[group_positions["H5"]],
        y=[float(np.mean(zs_h5_values))] if len(zs_h5_values) > 0 else [np.nan],
        name="Evo2-7B (Log-Likelihood)",
        marker_color="#3366CC",
        opacity=0.7,
        width=0.3,
        offset=-0.15,
        showlegend=False,  # already shown above
        legendgroup="zs",
    ),
    row=1, col=1,
)

fig.add_trace(
    go.Bar(
        x=[group_positions["H5"]],
        y=[float(np.mean(probe_values_h5))] if len(probe_values_h5) > 0 else [np.nan],
        name="Evo2-7B (Probe)",
        marker_color="#E377C2",
        opacity=0.7,
        width=0.3,
        offset=+0.15,
        legendgroup="probe",
    ),
    row=1, col=1,
)

# Points over the bars
# Full zero-shot points (centered at 0)
if len(zs_full_values) > 0:
    fig.add_trace(
        go.Scatter(
            x=[group_positions["Full"]] * len(zs_full_values),
            y=zs_full_values,
            mode="markers",
            marker=dict(color="#3366CC", size=8, opacity=0.65, line=dict(color="white", width=0.5)),
            showlegend=False,
            hovertemplate="Group: Full<br>|ρ|: %{y}<extra></extra>",
        ),
        row=1, col=1,
    )

# H5 zero-shot points (slightly left)
if len(zs_h5_values) > 0:
    fig.add_trace(
        go.Scatter(
            x=[group_positions["H5"] - 0.15] * len(zs_h5_values),
            y=zs_h5_values,
            mode="markers",
            marker=dict(color="#3366CC", size=8, opacity=0.65, line=dict(color="white", width=0.5)),
            showlegend=False,
            hovertemplate="Group: H5 (Zero-shot)<br>|ρ|: %{y}<extra></extra>",
        ),
        row=1, col=1,
    )

# H5 probe points (slightly right)
if len(probe_values_h5) > 0:
    fig.add_trace(
        go.Scatter(
            x=[group_positions["H5"] + 0.15] * len(probe_values_h5),
            y=probe_values_h5,
            mode="markers",
            marker=dict(color="#E377C2", size=8, opacity=0.65, line=dict(color="white", width=0.5)),
            showlegend=False,
            hovertemplate="Group: H5 (Probe)<br>|ρ|: %{y}<extra></extra>",
        ),
        row=1, col=1,
    )

# Axis labels and ticks
tickvals = [group_positions["Full"], group_positions["H5"]]
ticktext = ["Full", "H5"]

fig.update_xaxes(
    title_text="",
    tickmode="array",
    tickvals=tickvals,
    ticktext=ticktext,
    tickangle=0,
    showline=True,
    linecolor="black",
    mirror=True,
    tickfont=dict(size=25, family="Arial"),
)

# Shared y-axis formatting
y_min = -0.02
y_max = 0.7
fig.update_yaxes(
    title_text='|ρ|',
    range=[y_min, y_max],
    dtick=0.2,
    ticks='outside',
    showline=True,
    linecolor='black',
    mirror=True,
    gridcolor='lightgrey',
    zerolinecolor='lightgrey',
    showgrid=True,
    tickfont=dict(size=25, family='Arial'),
)

# Layout styling
fig.update_layout(
    template='plotly_white',
    width=1200,
    height=400,
    showlegend=True,
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.6,
        bgcolor='rgba(255,255,255,0.8)'
    ),
    margin=dict(t=0, b=0, l=60, r=0),
    font=dict(family='Arial, sans-serif', size=25),
    plot_bgcolor='white',
    bargap=0.1,
    bargroupgap=0.1,
)

# fig.show()

# Save outputs
out_file_svg = "dms_40b.svg"
out_file_pdf = "dms_40b.pdf"
fig.write_image(out_file_svg)
import subprocess
subprocess.run(["inkscape", out_file_svg, "--export-pdf=" + out_file_pdf])



