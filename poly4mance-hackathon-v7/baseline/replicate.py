"""
Regenerate presentation figures for model-by-experiment comparisons only.

Run with:
    just replicate

Optional:
    python3 baseline/replicate.py --models M2 M4

The two Q-only comparison figures (overall_supported_models_q_only.png and
overall_<a>_vs_<b>_q_only.png) show uncertainty, computed from the same
parameter draws predict.py uses for its intervals (converged posterior, else
the Laplace fallback, else suggest.py's bootstrap samples):
  * line + shaded band: median and 90 % range of the predicted Q concentration
    across parameter draws (parameter uncertainty only);
  * markers + error bars at the campaign sample times: median and 90 % range of
    a Q measurement, i.e. parameter uncertainty plus assay noise.
A fixed number of draws and a fixed seed keep the figures reproducible.
"""

from pathlib import Path
import argparse
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SCRATCH_DIR = PROJECT_ROOT / "scratch"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from candidates import CANDIDATES, Conditions, simulate
import pipeline_common as pc
from pipeline_common import SUBMISSION, ensure_submission_dir

REPLICATE_VERSION = "2026-10-13c (head-to-head and model-averaged Q figures, 90 % bands, log y-axis)"
SCRATCH_DIR = getattr(pc, "SCRATCH_DIR", PROJECT_ROOT / "scratch")
NOISE_Q = (0.15, 2.2e-5)        # relative, absolute measurement error for Q (from the brief)


def _draws_path(model_name, kind):
    return SCRATCH_DIR / f"system_B_{model_name}_bayesian_{kind}_draws.csv"


def _prediction_draws_file(model_name):
    """Converged posterior, then Laplace fallback. Works with any pipeline_common version."""
    for kind, label in (("posterior", "posterior"), ("laplace", "laplace")):
        path = _draws_path(model_name, kind)
        if path.exists():
            return path, label
    return None, "none"


def _sigma_q(mu):
    rel, absolute = NOISE_Q
    mu = np.maximum(np.asarray(mu, dtype=float), 0.0)
    return np.sqrt((rel * mu) ** 2 + absolute ** 2)


def _simulate(model_name, conditions, params):
    """Timed simulation if pipeline_common provides it, otherwise the official simulator."""
    if hasattr(pc, "safe_simulate"):
        return pc.safe_simulate(model_name, conditions, params, timeout=30.0)
    return simulate(model_name, conditions, params)
BAND_DRAWS_PER_MODEL = 100      # parameter draws used for the uncertainty bands
BAND_SEED = 20261013
BOOTSTRAP_SAMPLES_FILE = SCRATCH_DIR / "system_B_bootstrap_parameter_samples.csv"
BAND_DIAGNOSTICS = []           # one row per model x experiment; written to scratch/
# Q spans orders of magnitude between models, so a linear axis hides the lower
# model's band. "log" shows every band; "linear" restores the old look.
Q_AXIS_SCALE = "log"


Q_LOQ = 3.0e-5
PARAMETER_FILE_PATTERN = "system_B_*_weighted_censored_best_parameters.csv"


def parse_args():
    """
    Parse optional command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Regenerate model-by-experiment comparison figures."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(CANDIDATES.keys()),
        help="One or more fitted model names to plot, e.g. --models M2 M4",
    )
    return parser.parse_args()


def model_parameter_file(model_name):
    """
    Return the expected fitted-parameter file for one model.
    """
    return SCRATCH_DIR / f"system_B_{model_name}_weighted_censored_best_parameters.csv"


def load_parameters(parameter_file):
    """
    Load one fitted model's kinetic parameters from a saved CSV file.
    """
    if not parameter_file.is_file():
        model_name = parameter_file.stem.split("_")[2]
        raise FileNotFoundError(
            f"Missing fitted-parameter file: {parameter_file}\n"
            "Generate the needed fit first with:\n"
            f"  python3 baseline/fit_system_b_weighted_multistart.py {model_name}"
        )

    table = pd.read_csv(parameter_file)

    value_column = "value"
    if value_column not in table.columns:
        if "best_fitted_value" in table.columns:
            value_column = "best_fitted_value"
        else:
            raise ValueError(
                f"Could not find parameter value column in {parameter_file}. "
                f"Found columns: {list(table.columns)}"
            )

    return {
        str(row["parameter"]): float(row[value_column])
        for _, row in table.iterrows()
    }


def load_campaign():
    """
    Load the selected campaign written by suggest.py.
    """
    campaign_file = Path(SUBMISSION) / "campaign.yaml"

    if not campaign_file.is_file():
        raise FileNotFoundError(
            f"Missing campaign file: {campaign_file}\n"
            "Generate it with:\n"
            "  just suggest"
        )

    with campaign_file.open("r", encoding="utf-8") as file:
        campaign = yaml.safe_load(file)

    if campaign is None or "experiments" not in campaign:
        raise ValueError(
            f"campaign.yaml is empty or missing the 'experiments' key: {campaign_file}"
        )

    return campaign


def load_hypotheses():
    """
    Load hypotheses.yaml written by predict.py.
    """
    hypotheses_file = Path(SUBMISSION) / "hypotheses.yaml"

    if not hypotheses_file.is_file():
        raise FileNotFoundError(
            f"Missing hypotheses file: {hypotheses_file}\n"
            "Generate it with:\n"
            "  just predict"
        )

    with hypotheses_file.open("r", encoding="utf-8") as file:
        hypotheses = yaml.safe_load(file)

    if hypotheses is None or "all_models" not in hypotheses:
        raise ValueError(
            f"hypotheses.yaml is empty or missing the 'all_models' key: {hypotheses_file}"
        )

    return hypotheses


def discover_available_models():
    """
    Discover fitted models from the scratch directory.
    """
    discovered = []
    pattern = re.compile(
        r"^system_B_(M\d+)_weighted_censored_best_parameters\.csv$"
    )

    for path in sorted(SCRATCH_DIR.glob(PARAMETER_FILE_PATTERN)):
        match = pattern.match(path.name)
        if match:
            discovered.append(match.group(1))

    return discovered


def choose_models(args, campaign):
    """
    Decide which models to plot.

    Priority:
    1. --models from command line
    2. supported_models in campaign.yaml
    3. discovered fitted models in scratch/
    """
    if args.models:
        return args.models

    supported = campaign.get("supported_models")
    if supported:
        return [str(name) for name in supported]

    discovered = discover_available_models()
    if discovered:
        return discovered

    raise RuntimeError(
        "No models selected and no fitted parameter files were discovered in scratch/."
    )


def supported_models_from_hypotheses(hypotheses):
    """
    Return the models retained by predict.py (selected_models), falling back
    to entries with non-zero supported_weight.
    """
    selected = [str(name) for name in hypotheses.get("selected_models", [])]
    if selected:
        return selected

    supported = []

    for entry in hypotheses["all_models"]:
        supported_weight = float(entry.get("supported_weight", 0.0))
        if supported_weight > 0.0:
            supported.append(str(entry["name"]))

    if not supported:
        raise ValueError(
            "No models with non-zero supported_weight were found in hypotheses.yaml."
        )

    return supported


def get_e1_to_e5_experiments(campaign):
    """
    Return up to five experiments labelled E1 to E5 (the 2x3 layout has five
    experiment panels; extra campaign experiments are not plotted).
    """
    experiments = campaign["experiments"]

    if len(experiments) < 1:
        raise ValueError("campaign.yaml contains no experiments.")

    e1_to_e5 = []

    for index, experiment in enumerate(experiments[:5], start=1):
        experiment_copy = dict(experiment)
        experiment_copy["panel_name"] = f"E{index}"
        e1_to_e5.append(experiment_copy)

    return e1_to_e5


def simulate_profile(model_name, parameters, experiment, n_times=400):
    """
    Simulate dense concentration profiles for one experiment.
    """
    times_min = np.linspace(
        0.0,
        float(max(experiment["sample_times_min"])),
        n_times,
    ).tolist()

    conditions = Conditions(
        temperature_c=float(experiment["temperature_C"]),
        A0=float(experiment["A0_M"]),
        B0=float(experiment["B0_M"]),
        times_min=times_min,
    )

    return simulate(model_name, conditions, parameters)


def simulate_at_sampling_times(model_name, parameters, experiment):
    """
    Simulate all species at the exact campaign sampling times.
    """
    conditions = Conditions(
        temperature_c=float(experiment["temperature_C"]),
        A0=float(experiment["A0_M"]),
        B0=float(experiment["B0_M"]),
        times_min=[float(time) for time in experiment["sample_times_min"]],
    )

    return simulate(model_name, conditions, parameters)


def build_model_profiles(model_name, parameters, experiments):
    """
    Precompute dense and sampled simulations for a model across E1 to E5.
    """
    results = []

    for experiment in experiments:
        dense_profile = simulate_profile(model_name, parameters, experiment)
        sampled_profile = simulate_at_sampling_times(model_name, parameters, experiment)

        results.append(
            {
                "experiment": experiment,
                "dense": dense_profile,
                "sampled": sampled_profile,
                "sample_times": np.asarray(
                    experiment["sample_times_min"],
                    dtype=float,
                ),
            }
        )

    return results


def make_2x3_axes(figsize=(16, 9), sharey=False):
    """
    Create a 2x3 axes grid and return figure, flattened axes.
    The sixth panel is used as a legend/info panel.
    """
    figure, axes = plt.subplots(
        2,
        3,
        figsize=figsize,
        sharey=sharey,
    )
    return figure, axes.flatten()


def hide_unused_panels(axes, n_used):
    """
    Blank experiment panels left over when the campaign has fewer than five
    experiments (panel 6 is always the legend).
    """
    for axis in axes[n_used:5]:
        axis.axis("off")


def format_experiment_title(experiment):
    """
    Build a compact panel title for one experiment.
    """
    return (
        f"{experiment['panel_name']} | "
        f"{float(experiment['temperature_C']):.0f} °C\n"
        f"A0={float(experiment['A0_M']):.2f}, "
        f"B0={float(experiment['B0_M']):.2f}"
    )


def add_legend_panel(axis, handles, labels, title, extra_lines=None):
    """
    Use the sixth subplot as a clean legend/info panel.
    """
    axis.axis("off")

    legend = axis.legend(
        handles,
        labels,
        loc="center",
        frameon=True,
        fontsize=10,
        title=title,
        title_fontsize=11,
    )
    legend.get_frame().set_alpha(0.95)

    if extra_lines is not None:
        axis.text(
            0.5,
            0.08,
            "\n".join(extra_lines),
            ha="center",
            va="bottom",
            fontsize=10,
            transform=axis.transAxes,
        )


def plot_model_all_components(model_name, model_results, figures_dir):
    """
    Create a 2x3 figure for one model across E1 to E5
    showing A, B, P, and Q together in each panel.
    """
    figure, axes = make_2x3_axes(figsize=(16, 9), sharey=True)

    species_colours = {
        "A": "tab:blue",
        "B": "tab:orange",
        "P": "tab:green",
        "Q": "tab:red",
    }

    handles = []
    labels = []

    hide_unused_panels(axes, min(5, len(model_results)))

    for axis, result in zip(axes[:5], model_results):
        experiment = result["experiment"]
        dense = result["dense"]
        sampled = result["sampled"]
        sample_times = result["sample_times"]

        local_handles = []
        local_labels = []

        for species in ["A", "B", "P", "Q"]:
            line, = axis.plot(
                dense["time_min"],
                dense[species].clip(lower=0.0),
                color=species_colours[species],
                linewidth=2.0,
                label=species,
            )

            axis.plot(
                sample_times,
                sampled[species].clip(lower=0.0),
                "o",
                color=species_colours[species],
                markerfacecolor="white",
                markersize=4,
            )

            local_handles.append(line)
            local_labels.append(species)

        loq_line = axis.axhline(
            Q_LOQ,
            color="black",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
            label="Q LOQ",
        )

        axis.set_title(format_experiment_title(experiment), fontsize=10)
        axis.set_xlabel("Time (min)")
        axis.grid(True, alpha=0.25)

        if not handles:
            handles = local_handles + [loq_line]
            labels = local_labels + ["Q LOQ"]

    axes[0].set_ylabel("Concentration (mol/L)")
    axes[3].set_ylabel("Concentration (mol/L)")

    add_legend_panel(
        axes[5],
        handles,
        labels,
        title=f"{model_name} legend",
        extra_lines=[
            "Five experiment panels plus one legend panel.",
            "Markers show campaign sampling times.",
            "Dashed line marks the Q limit of quantification.",
        ],
    )

    figure.suptitle(
        f"{model_name}: E1 to E5 comparison, all components",
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout(rect=[0, 0, 1, 0.96])

    output_file = figures_dir / f"{model_name.lower()}_e1_to_e5_all_components.png"
    figure.savefig(output_file, dpi=220, bbox_inches="tight")
    plt.close(figure)

    print(f"[replicate] wrote figure -> {output_file}")


def plot_model_single_species(model_name, model_results, species, figures_dir):
    """
    Create a 2x3 figure for one model across E1 to E5
    for a single species only.
    """
    figure, axes = make_2x3_axes(figsize=(16, 9), sharey=True)

    species_colour = {
        "P": "tab:green",
        "Q": "tab:red",
    }[species]

    handles = []
    labels = []

    hide_unused_panels(axes, min(5, len(model_results)))

    for axis, result in zip(axes[:5], model_results):
        experiment = result["experiment"]
        dense = result["dense"]
        sampled = result["sampled"]
        sample_times = result["sample_times"]

        line, = axis.plot(
            dense["time_min"],
            dense[species].clip(lower=0.0),
            color=species_colour,
            linewidth=2.4,
            label=species,
        )

        sample_marker = axis.plot(
            sample_times,
            sampled[species].clip(lower=0.0),
            marker="o",
            linestyle="None",
            color=species_colour,
            markerfacecolor="white",
            markersize=5,
            label=f"{species} sampled",
        )[0]

        loq_line = None
        if species == "Q":
            loq_line = axis.axhline(
                Q_LOQ,
                color="black",
                linestyle="--",
                linewidth=1.0,
                alpha=0.9,
                label="Q LOQ",
            )

        axis.set_title(format_experiment_title(experiment), fontsize=10)
        axis.set_xlabel("Time (min)")
        axis.grid(True, alpha=0.25)

        if not handles:
            handles = [line, sample_marker]
            labels = [species, f"{species} sampled"]

            if loq_line is not None:
                handles.append(loq_line)
                labels.append("Q LOQ")

    axes[0].set_ylabel(f"{species} concentration (mol/L)")
    axes[3].set_ylabel(f"{species} concentration (mol/L)")

    extra_lines = [
        "Five experiment panels plus one legend panel.",
        "Markers show campaign sampling times.",
    ]
    if species == "Q":
        extra_lines.append("Dashed line marks the Q limit of quantification.")

    add_legend_panel(
        axes[5],
        handles,
        labels,
        title=f"{model_name} {species} legend",
        extra_lines=extra_lines,
    )

    figure.suptitle(
        f"{model_name}: E1 to E5 comparison, only {species}",
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout(rect=[0, 0, 1, 0.96])

    output_file = figures_dir / f"{model_name.lower()}_e1_to_e5_only_{species.lower()}.png"
    figure.savefig(output_file, dpi=220, bbox_inches="tight")
    plt.close(figure)

    print(f"[replicate] wrote figure -> {output_file}")


# ---------------------------------------------------------------------------
# Uncertainty bands for the Q-only comparison figures
# ---------------------------------------------------------------------------

def load_band_draws(model_name):
    """
    Parameter draws for one model's uncertainty band, as (list of dicts, label).
    Same order of preference as predict.py: converged posterior, then Laplace
    fallback; then suggest.py's bootstrap samples; otherwise none.
    """
    names = list(CANDIDATES[model_name].param_names)
    path, source = _prediction_draws_file(model_name)
    table = None
    if path is not None:
        table = pd.read_csv(path)
        label = {"posterior": "posterior", "laplace": "Laplace approx."}[source]
    elif BOOTSTRAP_SAMPLES_FILE.exists() and BOOTSTRAP_SAMPLES_FILE.stat().st_size > 0:
        boot = pd.read_csv(BOOTSTRAP_SAMPLES_FILE)
        boot = boot[boot["model"].astype(str).str.upper() == model_name]
        if len(boot):
            table, label = boot, "bootstrap"
    if table is None:
        return [], "none"
    table = table.replace([np.inf, -np.inf], np.nan).dropna(subset=names)
    if len(table) > BAND_DRAWS_PER_MODEL:
        table = table.sample(n=BAND_DRAWS_PER_MODEL, random_state=BAND_SEED)
    return table[names].to_dict("records"), label


def q_uncertainty(model_name, experiments, n_times=400):
    """
    For each experiment: median and 90 % range of Q concentration on the dense
    grid (parameter uncertainty), and median and 90 % range of a Q measurement
    at the sample times (parameter uncertainty + assay noise).
    Returns (list of dicts or None per experiment, source label, number of draws).
    """
    draws, label = load_band_draws(model_name)
    if not draws:
        looked = [str(_draws_path(model_name, "posterior")), str(_draws_path(model_name, "laplace")),
                  str(BOOTSTRAP_SAMPLES_FILE)]
        print(f"[replicate] WARNING {model_name}: NO SHADED BAND - no parameter draws found. "
              f"Looked for: {', '.join(looked)}. Run just predict (and just suggest) first.",
              flush=True)
        for experiment in experiments:
            BAND_DIAGNOSTICS.append({"model": model_name, "experiment": experiment.get("panel_name"),
                                     "draw_source": "none", "draws_available": 0,
                                     "simulations_ok": 0, "simulations_failed": 0,
                                     "band_drawn": False, "reason": "no draw files found"})
        return [None] * len(experiments), label, 0

    rng = np.random.default_rng(BAND_SEED)
    out = []
    for experiment in experiments:
        sample_times = np.asarray(experiment["sample_times_min"], dtype=float)
        dense = np.linspace(0.0, float(sample_times.max()), n_times)
        grid = np.unique(np.concatenate([dense, sample_times]))
        conditions = Conditions(temperature_c=float(experiment["temperature_C"]),
                                A0=float(experiment["A0_M"]), B0=float(experiment["B0_M"]),
                                times_min=grid.tolist())
        q, failures, first_error = [], 0, None
        for params in draws:
            try:
                sim = _simulate(model_name, conditions, params)
                values = np.maximum(sim["Q"].to_numpy(dtype=float), 0.0)
            except Exception as error:  # noqa: BLE001
                failures += 1
                first_error = first_error or f"{type(error).__name__}: {error}"
                continue
            if values.shape[0] == grid.size and np.all(np.isfinite(values)):
                q.append(values)
            else:
                failures += 1
                first_error = first_error or (f"simulation returned {values.shape[0]} rows for "
                                              f"{grid.size} times or non-finite values")
        record = {"model": model_name, "experiment": experiment.get("panel_name"),
                  "draw_source": label, "draws_available": len(draws), "simulations_ok": len(q),
                  "simulations_failed": failures}
        if len(q) < 10:
            print(f"[replicate] WARNING {model_name} {experiment.get('panel_name')}: NO SHADED BAND - "
                  f"only {len(q)} of {len(draws)} draw simulations succeeded"
                  + (f" (first error: {first_error})" if first_error else ""), flush=True)
            BAND_DIAGNOSTICS.append({**record, "band_drawn": False,
                                     "reason": first_error or "too few successful simulations"})
            out.append(None)
            continue
        q = np.asarray(q)                                     # (n_draws, n_grid)
        dense_idx = np.searchsorted(grid, dense)
        sample_idx = np.searchsorted(grid, sample_times)
        mu_s = q[:, sample_idx]
        y_s = rng.normal(mu_s, _sigma_q(mu_s))                # measurement: unclipped Gaussian
        lo = np.quantile(q[:, dense_idx], 0.05, axis=0)
        hi = np.quantile(q[:, dense_idx], 0.95, axis=0)
        scale = max(float(np.max(q)), 1e-12)
        BAND_DIAGNOSTICS.append({**record, "band_drawn": True, "reason": "",
                                 "max_band_width_M": float(np.max(hi - lo)),
                                 "max_band_width_pct_of_peak_Q": 100.0 * float(np.max(hi - lo)) / scale})
        out.append({
            "dense_t": dense,
            "dense_mid": np.median(q[:, dense_idx], axis=0),
            "dense_lo": np.quantile(q[:, dense_idx], 0.05, axis=0),
            "dense_hi": np.quantile(q[:, dense_idx], 0.95, axis=0),
            "sample_t": sample_times,
            "conc_mid_at_samples": np.median(mu_s, axis=0),
            "meas_lo": np.quantile(y_s, 0.05, axis=0),
            "meas_hi": np.quantile(y_s, 0.95, axis=0),
            "q_dense": q[:, dense_idx],          # raw draws, for the model-averaged figure
            "y_samples": y_s,
        })
    return out, label, len(draws)


MODEL_COLOURS = {"M1": "#2a9d5c", "M2": "#d9730d", "M3": "#b23a48", "M4": "#1f6fb4",
                 "M5": "#8c6d31", "M6": "#6a4c93"}
MODEL_MARKERS = {"M1": "^", "M2": "s", "M3": "D", "M4": "o", "M5": "v", "M6": "P"}
AVERAGE_COLOUR = "#5b3f8c"


def q_axis_limits(model_names, model_results_by_name, bands_by_name, n_experiments):
    """
    Shared y-limits for the Q figures. Log axis: floor = a fifth of the smallest
    end-of-run median Q among the models (never above LOQ/3); values below it are
    drawn at the floor. Top = 1.4 x the largest value shown, so nothing is cut off.
    """
    floor, top = Q_LOQ / 3.0, Q_LOQ
    for model in model_names:
        for e_idx in range(n_experiments):
            band = bands_by_name[model]["bands"][e_idx]
            best = model_results_by_name[model][e_idx]["dense"]["Q"].to_numpy(dtype=float)
            if band is not None:
                end_q = float(band["dense_mid"][-1])
                top = max(top, float(np.max(band["dense_hi"])), float(np.max(band["meas_hi"])))
            else:
                end_q = float(best[-1])
            top = max(top, float(np.max(best)))
            if end_q > 0:
                floor = min(floor, 0.2 * end_q)
    return max(floor, 1e-10), top


def style_q_axis(axis, log_axis, floor, top):
    if log_axis:
        axis.set_yscale("log")
        axis.set_ylim(floor * 0.9, top * 1.4)
    else:
        axis.set_ylim(top=top * 1.1)


def plot_q_comparison(model_names, model_results_by_name, bands_by_name, title, legend_title,
                      output_file, intro_line):
    """
    2x3 figure, Q only, one colour per model: median line with a shaded 90 %
    band (concentration) and measurement error bars at the sample times. Falls
    back to the best-fit curve for a model without draws.
    """
    figure, axes = make_2x3_axes(figsize=(16, 9), sharey=True)
    colours = ["#1f6fb4", "#d9730d", "#2a9d5c", "#b23a48", "#6a4c93", "#8c6d31"]
    markers = ["o", "s", "^", "D", "v", "P"]
    n_experiments = min(5, len(next(iter(model_results_by_name.values()))))
    hide_unused_panels(axes, n_experiments)
    log_axis = Q_AXIS_SCALE == "log"

    floor, top = q_axis_limits(model_names, model_results_by_name, bands_by_name, n_experiments)

    def shown(values):
        values = np.asarray(values, dtype=float)
        return np.maximum(values, floor) if log_axis else values

    handles, labels = [], []
    for e_idx, axis in enumerate(axes[:n_experiments]):
        for m_idx, model in enumerate(model_names):
            colour = MODEL_COLOURS.get(model, colours[m_idx % len(colours)])
            marker = MODEL_MARKERS.get(model, markers[m_idx % len(markers)])
            result = model_results_by_name[model][e_idx]
            band = bands_by_name[model]["bands"][e_idx]
            if band is not None:
                fill = axis.fill_between(band["dense_t"], shown(band["dense_lo"]),
                                         shown(band["dense_hi"]), color=colour, alpha=0.22,
                                         linewidth=0, zorder=1)
                for edge in (band["dense_lo"], band["dense_hi"]):
                    axis.plot(band["dense_t"], shown(edge), color=colour, linewidth=0.8,
                              linestyle=":", alpha=0.9, zorder=2)
                line, = axis.plot(band["dense_t"], shown(band["dense_mid"]), color=colour,
                                  linewidth=2.0, zorder=3)
                mid = shown(band["conc_mid_at_samples"])
                lo, hi = shown(band["meas_lo"]), shown(band["meas_hi"])
                err = axis.errorbar(band["sample_t"], mid,
                                    yerr=[np.maximum(mid - lo, 0.0), np.maximum(hi - mid, 0.0)],
                                    fmt=marker, color=colour, markerfacecolor="white",
                                    markersize=5, capsize=3, elinewidth=1.1, linestyle="None",
                                    zorder=4)
                if e_idx == 0:
                    source = bands_by_name[model]["source"]
                    n = bands_by_name[model]["n"]
                    handles += [(fill, line), err]
                    labels += [f"{model} Q: median and 90 % band ({source}, {n} draws)",
                               f"{model} measurement: 90 % range at sample times"]
            else:
                line, = axis.plot(result["dense"]["time_min"],
                                  shown(result["dense"]["Q"].clip(lower=0.0)),
                                  color=colour, linewidth=2.0)
                pts = axis.plot(result["sample_times"], shown(result["sampled"]["Q"].clip(lower=0.0)),
                                marker=marker, linestyle="None", color=colour,
                                markerfacecolor="white", markersize=5)[0]
                if e_idx == 0:
                    handles += [line, pts]
                    labels += [f"{model} Q: best fit only (no draws)", f"{model} at sample times"]

        loq = axis.axhline(Q_LOQ, color="black", linestyle="--", linewidth=1.0, alpha=0.9)
        if e_idx == 0:
            handles.append(loq)
            labels.append("Q limit of quantification")
        style_q_axis(axis, log_axis, floor, top)
        experiment = model_results_by_name[model_names[0]][e_idx]["experiment"]
        axis.set_title(format_experiment_title(experiment), fontsize=10)
        axis.set_xlabel("Time (min)")
        axis.grid(True, alpha=0.25, which="both" if log_axis else "major")

    y_label = "Q concentration (mol/L, log scale)" if log_axis else "Q concentration (mol/L)"
    axes[0].set_ylabel(y_label)
    axes[3].set_ylabel(y_label)
    legend_axis = axes[5]
    legend_axis.axis("off")
    legend = legend_axis.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
                                frameon=True, fontsize=9.5, title=legend_title, title_fontsize=11)
    legend.get_frame().set_alpha(0.95)
    legend_axis.text(
        0.5, 0.02,
        "\n".join([
            intro_line,
            "Shaded band: 90 % range of predicted Q across parameter draws.",
            "Error bars: 90 % range of a single measurement",
            "(parameter uncertainty plus assay noise).",
            "Little overlap between colours = the experiment separates the models.",
        ] + ([f"Log scale: values below {floor:.1e} M (Q = 0 at the start, negative",
              "measurement readings) are drawn at the bottom of the axis."] if log_axis else [])),
        ha="center", va="bottom", fontsize=9.5, transform=legend_axis.transAxes)
    figure.suptitle(title, fontsize=15, fontweight="bold")
    figure.tight_layout(rect=[0, 0, 1, 0.96])
    figure.savefig(output_file, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"[replicate] wrote figure -> {output_file}")


def weighted_quantiles(values, weights, qs):
    """Column-wise weighted quantiles. values: (n, T), weights: (n,). Returns (len(qs), T)."""
    order = np.argsort(values, axis=0)
    sorted_values = np.take_along_axis(values, order, axis=0)
    cum = np.cumsum(weights[order], axis=0)
    cum /= cum[-1]
    out = np.empty((len(qs), values.shape[1]))
    for j in range(values.shape[1]):
        out[:, j] = np.interp(qs, cum[:, j], sorted_values[:, j])
    return out


def model_weights(hypotheses, model_names):
    """Prediction weights from hypotheses.yaml (renormalised over the retained models)."""
    lookup = {}
    for entry in hypotheses.get("all_models", []):
        weight = entry.get("selected_weight", entry.get("supported_weight", entry.get("weight", 0.0)))
        lookup[str(entry["name"])] = float(weight or 0.0)
    raw = np.array([lookup.get(m, 0.0) for m in model_names], dtype=float)
    if raw.sum() <= 0:
        raw = np.ones(len(model_names))
    return dict(zip(model_names, raw / raw.sum()))


def plot_supported_models_q(model_results_by_name, supported_model_names, bands_by_name, figures_dir,
                            hypotheses=None):
    """
    Model-averaged Q forecast for the campaign: the retained models' draws pooled
    with their prediction weights (as in predictions.csv). The band therefore
    covers parameter AND model uncertainty; each model's median is overlaid.
    """
    model_names = [name for name in supported_model_names if name in model_results_by_name]
    if not model_names:
        raise ValueError(
            "None of the supported models from hypotheses.yaml were available in model_results_by_name."
        )
    weights = model_weights(hypotheses or {}, model_names)
    n_experiments = min(5, len(model_results_by_name[model_names[0]]))
    log_axis = Q_AXIS_SCALE == "log"
    floor, top = q_axis_limits(model_names, model_results_by_name, bands_by_name, n_experiments)
    rng = np.random.default_rng(BAND_SEED + 1)

    def shown(values):
        values = np.asarray(values, dtype=float)
        return np.maximum(values, floor) if log_axis else values

    figure, axes = make_2x3_axes(figsize=(16, 9), sharey=True)
    hide_unused_panels(axes, n_experiments)
    handles, labels = [], []

    for e_idx, axis in enumerate(axes[:n_experiments]):
        dense_parts, meas_parts, w_dense, w_meas = [], [], [], []
        dense_t = sample_t = None
        for model in model_names:
            result = model_results_by_name[model][e_idx]
            band = bands_by_name[model]["bands"][e_idx]
            if band is not None:
                q_dense, y = band["q_dense"], band["y_samples"]
                dense_t, sample_t = band["dense_t"], band["sample_t"]
            else:   # no draws: best fit only, plus measurement noise
                q_dense = result["dense"]["Q"].clip(lower=0.0).to_numpy(dtype=float)[None, :]
                mu = result["sampled"]["Q"].clip(lower=0.0).to_numpy(dtype=float)
                y = rng.normal(np.tile(mu, (100, 1)), _sigma_q(np.tile(mu, (100, 1))))
                dense_t = result["dense"]["time_min"].to_numpy(dtype=float)
                sample_t = result["sample_times"]
            dense_parts.append(q_dense)
            meas_parts.append(y)
            w_dense.append(np.full(len(q_dense), weights[model] / len(q_dense)))
            w_meas.append(np.full(len(y), weights[model] / len(y)))

            median = np.median(q_dense, axis=0)
            line, = axis.plot(dense_t, shown(median), color=MODEL_COLOURS.get(model, "grey"),
                              linewidth=1.2, linestyle="--", zorder=3)
            if e_idx == 0:
                handles.append(line)
                labels.append(f"{model} median (weight {weights[model]:.2f})")

        q_all, wq = np.concatenate(dense_parts), np.concatenate(w_dense)
        y_all, wy = np.concatenate(meas_parts), np.concatenate(w_meas)
        wq, wy = wq / wq.sum(), wy / wy.sum()
        lo, hi = weighted_quantiles(q_all, wq, [0.05, 0.95])
        mean = wq @ q_all
        m_lo, m_hi = weighted_quantiles(y_all, wy, [0.05, 0.95])
        sample_idx = np.searchsorted(dense_t, sample_t).clip(max=len(dense_t) - 1)
        mean_s = shown(np.interp(sample_t, dense_t, mean))

        fill = axis.fill_between(dense_t, shown(lo), shown(hi), color=AVERAGE_COLOUR, alpha=0.18,
                                 linewidth=0, zorder=1)
        for edge in (lo, hi):
            axis.plot(dense_t, shown(edge), color=AVERAGE_COLOUR, linewidth=0.8, linestyle=":",
                      zorder=2)
        mean_line, = axis.plot(dense_t, shown(mean), color=AVERAGE_COLOUR, linewidth=2.4, zorder=4)
        err = axis.errorbar(sample_t, mean_s,
                            yerr=[np.maximum(mean_s - shown(m_lo), 0.0),
                                  np.maximum(shown(m_hi) - mean_s, 0.0)],
                            fmt="D", color=AVERAGE_COLOUR, markerfacecolor="white", markersize=5,
                            capsize=3, elinewidth=1.1, linestyle="None", zorder=5)
        loq = axis.axhline(Q_LOQ, color="black", linestyle="--", linewidth=1.0, alpha=0.9)
        if e_idx == 0:
            handles = [(fill, mean_line), err] + handles + [loq]
            labels = ["Model-averaged Q: mean and 90 % band",
                      "Model-averaged measurement: 90 % range"] + labels + ["Q limit of quantification"]

        style_q_axis(axis, log_axis, floor, top)
        experiment = model_results_by_name[model_names[0]][e_idx]["experiment"]
        axis.set_title(format_experiment_title(experiment), fontsize=10)
        axis.set_xlabel("Time (min)")
        axis.grid(True, alpha=0.25, which="both" if log_axis else "major")

    y_label = "Q concentration (mol/L, log scale)" if log_axis else "Q concentration (mol/L)"
    axes[0].set_ylabel(y_label)
    axes[3].set_ylabel(y_label)
    legend_axis = axes[5]
    legend_axis.axis("off")
    legend = legend_axis.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
                                frameon=True, fontsize=9.5, title="Model-averaged Q forecast",
                                title_fontsize=11)
    legend.get_frame().set_alpha(0.95)
    legend_axis.text(
        0.5, 0.02,
        "\n".join([
            "Retained models pooled by their prediction weights,",
            "as in predictions.csv. The band covers parameter AND",
            "model uncertainty: where it spans both model medians,",
            "the current data cannot say which mechanism is right.",
        ] + ([f"Log scale: values below {floor:.1e} M are drawn at the axis bottom."]
             if log_axis else [])),
        ha="center", va="bottom", fontsize=9.5, transform=legend_axis.transAxes)
    figure.suptitle("Model-averaged Q forecast for the campaign, with 90 % uncertainty",
                    fontsize=15, fontweight="bold")
    figure.tight_layout(rect=[0, 0, 1, 0.96])
    output_file = figures_dir / "overall_supported_models_q_only.png"
    figure.savefig(output_file, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(f"[replicate] wrote figure -> {output_file}")


def plot_pairwise_q(model_a, model_b, model_results_by_name, bands_by_name, figures_dir):
    """Two models head to head on Q only, with uncertainty, across E1 to E5."""
    plot_q_comparison(
        [model_a, model_b], model_results_by_name, bands_by_name,
        title=f"Overall {model_a} versus {model_b} comparison, Q only, with 90 % uncertainty",
        legend_title=f"{model_a} vs {model_b} on Q",
        output_file=figures_dir / f"overall_{model_a.lower()}_vs_{model_b.lower()}_q_only.png",
        intro_line="Direct model-to-model comparison across the campaign.",
    )


# ---------------------------------------------------------------------------
# System A: M1 fitted model versus measured data
# ---------------------------------------------------------------------------

SYSTEM_A_FILE = PROJECT_ROOT / "data" / "system_A" / "measurements.csv"
SYSTEM_A_MODEL = "M1"
SPECIES_COLOURS = {"A": "tab:blue", "B": "tab:orange", "P": "tab:green", "Q": "tab:red"}


def load_system_a():
    df = pd.read_csv(SYSTEM_A_FILE)
    if "system" in df.columns:
        df = df[df["system"].astype(str) == "A"].copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")   # "<LOQ" -> NaN, dropped
    df["time_min"] = df["time_min"].astype(float)
    return df[df["value"].notna()].copy()


def fit_system_a(df, model):
    """Weighted least squares (3 % relative + 3.5e-3 M absolute error, all species), multistart."""
    from scipy.optimize import least_squares

    names = list(CANDIDATES[model].param_names)
    observed = set(getattr(CANDIDATES[model], "observed", ("A", "B", "P", "Q")))
    runs = []
    for run_id, g in df.groupby("run_id", sort=True):
        times = np.sort(g["time_min"].unique())
        first = g.iloc[0]
        cond = Conditions(temperature_c=float(first["temperature_C"]), A0=float(first["A0_M"]),
                          B0=float(first["B0_M"]), times_min=times.tolist())
        g = g[g["species"].isin(observed)]
        runs.append((cond, np.searchsorted(times, g["time_min"].to_numpy()),
                     g["species"].to_numpy(), g["value"].to_numpy(dtype=float)))
    n_points = sum(len(r[3]) for r in runs)

    def unpack(z):
        return {n: (10.0 ** v if n.startswith("k") else 100.0 * v) for n, v in zip(names, z)}

    def residuals(z):
        try:
            params, out = unpack(z), []
            for cond, idx, species, values in runs:
                sim = simulate(model, cond, params)
                mu = np.array([max(float(sim[sp].iloc[i]), 0.0) for sp, i in zip(species, idx)])
                out.append((mu - values) / np.sqrt((0.03 * mu) ** 2 + 3.5e-3 ** 2))
            r = np.concatenate(out)
            return r if np.all(np.isfinite(r)) else np.full(n_points, 1e3)
        except Exception:  # noqa: BLE001
            return np.full(n_points, 1e3)

    lo = np.array([-6.0 if n.startswith("k") else 0.05 for n in names])
    hi = np.array([0.0 if n.startswith("k") else 1.5 for n in names])
    rng = np.random.default_rng(20261013)
    starts = [np.array([-2.0 if n.startswith("k") else 0.6 for n in names])]
    starts += [np.array([rng.uniform(-5, -0.5) if n.startswith("k") else rng.uniform(0.2, 1.1)
                         for n in names]) for _ in range(5)]
    best = min((least_squares(residuals, z0, bounds=(lo, hi), max_nfev=400) for z0 in starts),
               key=lambda res: res.cost)
    return unpack(best.x), 2.0 * best.cost, n_points


def plot_system_a_fit(figures_dir, model=SYSTEM_A_MODEL):
    """Five System A runs: measured points (circles) and the fitted model (lines)."""
    if not SYSTEM_A_FILE.is_file():
        print(f"[replicate] skipping System A figure: {SYSTEM_A_FILE} not found", flush=True)
        return
    df = load_system_a()
    params, chi2, n_points = fit_system_a(df, model)
    pd.DataFrame([{"parameter": k, "value": v} for k, v in params.items()]).to_csv(
        SCRATCH_DIR / f"system_A_{model}_fit_parameters.csv", index=False)

    run_ids = sorted(df["run_id"].unique())
    figure, axes = make_2x3_axes(figsize=(16, 10))
    hide_unused_panels(axes, len(run_ids))
    for axis, run_id in zip(axes, run_ids[:5]):
        g = df[df["run_id"] == run_id]
        first = g.iloc[0]
        dense = np.linspace(0.0, float(g["time_min"].max()), 300)
        sim = simulate(model, Conditions(temperature_c=float(first["temperature_C"]),
                                         A0=float(first["A0_M"]), B0=float(first["B0_M"]),
                                         times_min=dense.tolist()), params)
        for sp, colour in SPECIES_COLOURS.items():
            pts = g[g["species"] == sp]
            if len(pts):
                axis.plot(pts["time_min"], pts["value"], "o", color=colour, markerfacecolor="none",
                          markersize=5, label=f"{sp} measured")
            if sp in sim.columns:
                axis.plot(sim["time_min"], sim[sp].clip(lower=0.0), "-", color=colour,
                          linewidth=1.8, label=f"{sp} {model} fit")
        axis.set_title(f"{run_id}: {float(first['temperature_C']):.0f} °C, "
                       f"A0 = {float(first['A0_M']):.2f} M, B0 = {float(first['B0_M']):.2f} M",
                       fontsize=11)
        axis.set_xlabel("Time (min)")
        axis.set_ylabel("Concentration (mol/L)")
        axis.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    legend_axis = axes[5]
    legend_axis.axis("off")
    legend_axis.legend(handles, labels, loc="upper center", ncol=2, fontsize=10,
                       title="Circles: measured   Lines: fitted", title_fontsize=10)
    legend_axis.text(0.5, 0.05,
                     f"{model} fitted to all System A runs\n"
                     f"(weighted least squares, chi-square = {chi2:.1f} for {n_points} points)\n"
                     + "\n".join(f"{k} = {v:.4g}" for k, v in params.items()),
                     ha="center", va="bottom", fontsize=9.5, transform=legend_axis.transAxes)
    figure.suptitle(f"System A: {model} fitted model versus measured data",
                    fontsize=15, fontweight="bold")
    figure.tight_layout(rect=[0, 0, 1, 0.96])
    out = figures_dir / f"system_a_{model.lower()}_fit_vs_data.png"
    figure.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"[replicate] wrote figure -> {out}", flush=True)


def main():
    """
    Regenerate only model-by-experiment comparison figures.
    """
    print(f"[replicate] version {REPLICATE_VERSION}", flush=True)
    print(f"[replicate] running {Path(__file__).resolve()}", flush=True)
    ensure_submission_dir()

    figures_dir = Path(SUBMISSION) / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    print(f"[replicate] writing figures to {figures_dir.resolve()}", flush=True)

    # Remove the Q comparison figures first, so a failure can never leave an old
    # version on disk that looks like the result of this run.
    for stale in figures_dir.glob("overall_*_q_only.png"):
        stale.unlink()

    args = parse_args()
    campaign = load_campaign()
    hypotheses = load_hypotheses()

    experiments = get_e1_to_e5_experiments(campaign)
    selected_models = choose_models(args, campaign)

    print("[replicate] models selected:", ", ".join(selected_models))

    model_results = {}

    for model_name in selected_models:
        parameter_file = model_parameter_file(model_name)
        parameters = load_parameters(parameter_file)
        results = build_model_profiles(model_name, parameters, experiments)
        model_results[model_name] = results

        plot_model_all_components(model_name, results, figures_dir)
        plot_model_single_species(model_name, results, "P", figures_dir)
        plot_model_single_species(model_name, results, "Q", figures_dir)

    bands_by_name = {}
    for model_name in selected_models:
        try:
            bands, source, n = q_uncertainty(model_name, experiments)
        except Exception as error:  # noqa: BLE001 - never lose the figure because of the band
            print(f"[replicate] WARNING {model_name}: NO SHADED BAND - computing it failed with "
                  f"{type(error).__name__}: {error}", flush=True)
            bands, source, n = [None] * len(experiments), "none", 0
        bands_by_name[model_name] = {"bands": bands, "source": source, "n": n}
        if n:
            print(f"[replicate] {model_name}: Q uncertainty bands from {n} {source} draws", flush=True)

    diag = pd.DataFrame(BAND_DIAGNOSTICS)
    diag_file = SCRATCH_DIR / "system_B_replicate_band_diagnostics.csv"
    diag.to_csv(diag_file, index=False)
    print("[replicate] uncertainty bands (Q comparison figures):", flush=True)
    for _, row in diag.iterrows():
        if row["band_drawn"]:
            print(f"  {row['model']} {row['experiment']}: band drawn from {row['simulations_ok']} "
                  f"{row['draw_source']} draws; max width {row['max_band_width_M']:.2e} M "
                  f"({row['max_band_width_pct_of_peak_Q']:.1f} % of peak Q)"
                  + ("  <- very narrow, look for the dotted edges" if
                     row["max_band_width_pct_of_peak_Q"] < 2 else ""), flush=True)
        else:
            print(f"  {row['model']} {row['experiment']}: NO band ({row['reason']})", flush=True)
    print(f"[replicate] details -> {diag_file}", flush=True)

    supported_model_names = supported_models_from_hypotheses(hypotheses)
    print(
        "[replicate] supported models from hypotheses.yaml:",
        ", ".join(supported_model_names),
    )

    plot_supported_models_q(
        model_results_by_name=model_results,
        supported_model_names=supported_model_names,
        bands_by_name=bands_by_name,
        figures_dir=figures_dir,
        hypotheses=hypotheses,
    )

    if len(selected_models) == 2:
        model_a, model_b = selected_models
        plot_pairwise_q(model_a, model_b, model_results, bands_by_name, figures_dir)

    plot_system_a_fit(figures_dir)


if __name__ == "__main__":
    main()