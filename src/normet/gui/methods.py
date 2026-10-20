"""File -> Methods Paragraph: what the session ran, written as a paper's Methods says it.

The paragraph is written from the
calls the window made -- each step's arguments as it ran, kept when it succeeded
(``ran``) -- never from the panel as it stands afterwards, and in the window's language
(English or Chinese). Only what was run is said; the numbers are the run's own. It cites
what the methods rest on: FLAML (Wang et al., 2021) or LightGBM (Ke et al., 2017), and
meteorological normalisation by resampling (Grange et al., 2018; Grange and Carslaw,
2019). Check it against what you did before it goes into a paper.
"""

from __future__ import annotations

from typing import Any

#: The calendar variables normet adds to every model.
CALENDAR = ("date_unix", "day_julian", "weekday", "hour")

REFERENCES = {
    "en": [
        "Grange, S. K., Carslaw, D. C., Lewis, A. C., Boleti, E. and Hueglin, C. (2018). "
        "Random forest meteorological normalisation models for Swiss PM10 trend analysis. "
        "Atmospheric Chemistry and Physics, 18, 6223–6239.",
        "Grange, S. K. and Carslaw, D. C. (2019). Using meteorological normalisation to "
        "detect interventions in air quality time series. Science of the Total Environment, "
        "653, 578–588.",
    ],
    "flaml": "Wang, C., Wu, Q., Weimer, M. and Zhu, E. (2021). FLAML: A fast and "
    "lightweight AutoML library. Proceedings of Machine Learning and Systems, 3, 434–447.",
    "lightgbm": "Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q. and Liu, "
    "T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. Advances "
    "in Neural Information Processing Systems, 30.",
}

_SPLITS = {
    "en": {
        "random": "drawn at random",
        "ts": "taken from the end of the record",
        "month_ts": "taken as one contiguous block within each calendar month",
        "season_ts": "taken as one contiguous block within each meteorological season",
    },
    "zh": {
        "random": "随机抽取",
        "ts": "取自记录末尾",
        "month_ts": "在每个日历月内取一段连续时段",
        "season_ts": "在每个气象季节内取一段连续时段",
    },
}


def _names(items: list[str], lang: str) -> str:
    if lang == "zh":
        return "、".join(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _met(covariates: list[str] | None) -> list[str]:
    return [c for c in (covariates or []) if c not in CALENDAR]


def methods_paragraph(info: dict[str, Any], lang: str = "en") -> str | None:
    """The Methods paragraph of a session, or None when no model has been made.

    ``info``: ``version``; ``rows``, ``start``, ``end``, ``step`` (the record's time step
    in words: "hourly", "daily"...); ``target``; ``ran`` -- each step's call as it ran,
    by function name, its keyword arguments and ``args`` its positional ones; ``test`` --
    the held-out ``r2``, ``rmse`` and ``n``; ``chronos_pct`` -- the forecast's shift, as a
    share of the prediction, when the weather is shuffled (Chronos-2)."""
    ran: dict[str, dict[str, Any]] = info.get("ran") or {}
    zh = lang == "zh"
    if "build_model" not in ran and "load_estimator" not in ran:
        return None
    target = info.get("target", "the target")
    out: list[str] = []
    cites: list[str] = []

    # -- the data --------------------------------------------------------------------
    if zh:
        out.append(
            f"气象归一化使用 normet {info.get('version', '')} 完成。记录包含 {target} 的 "
            f"{info.get('rows', 0):,} 个{info.get('step_zh', '')}观测值，"
            f"时间范围为 {info.get('start', '')} 至 {info.get('end', '')}。"
        )
    else:
        out.append(
            f"Meteorological normalisation was done with normet {info.get('version', '')}. "
            f"The record held {info.get('rows', 0):,} {info.get('step', '')} observations of "
            f"{target}, from {info.get('start', '')} to {info.get('end', '')}."
        )

    # -- the model -------------------------------------------------------------------
    test = info.get("test") or {}
    if "build_model" in ran:
        call = ran["build_model"]
        backend = call.get("backend", "flaml")
        met = _met(call.get("covariates"))
        cfg = call.get("model_config") or {}
        frac = float(call.get("train_fraction", 0.75))
        split = str(call.get("split_method", "random"))
        seed = call.get("seed")
        if backend == "flaml":
            ests = cfg.get("estimator_list") or ["lgbm"]
            budget = cfg.get("time_budget")
            cites.append(REFERENCES["flaml"])
            how_en = (
                f"by an automated search with FLAML (Wang et al., 2021) over "
                f"{_names([str(e) for e in ests], 'en')} within {budget} s"
            )
            how_zh = f"用 FLAML（Wang 等，2021）在 {budget} 秒内对 {_names([str(e) for e in ests], 'zh')} 进行自动搜索"
        else:
            trials = cfg.get("n_trials")
            cites.append(REFERENCES["lightgbm"])
            how_en = f"with LightGBM (Ke et al., 2017), its hyperparameters chosen by a random search of {trials} trials"
            how_zh = f"用 LightGBM（Ke 等，2017）构建，超参数经 {trials} 次随机搜索选定"
        if zh:
            out.append(
                f"以 {len(met)} 个气象变量（{_names(met, 'zh')}）和日历变量（趋势、年内日序、"
                f"星期、小时）为预测变量，{how_zh}，建立了 {target} 的梯度提升树模型。"
                f"{frac:.0%} 的数据行（{_SPLITS['zh'].get(split, split)}"
                + (f"，随机种子 {seed}" if seed is not None else "")
                + "）用于训练，其余留作检验"
                + (
                    f"，模型在留出集上的 R² = {test['r2']:.2f}，RMSE = {test['rmse']:.2f}"
                    f"（n = {int(test['n']):,}）。"
                    if test
                    else "。"
                )
            )
        else:
            out.append(
                f"A gradient-boosted tree model of {target} was trained {how_en}, on "
                f"{len(met)} meteorological predictors ({_names(met, 'en')}) and the calendar "
                f"(trend, day of year, weekday and hour). {frac:.0%} of the rows, "
                f"{_SPLITS['en'].get(split, split)}"
                + (f" (seed {seed})" if seed is not None else "")
                + ", were used for training and the rest held out"
                + (
                    f"; on the held-out rows the model reached R² = {test['r2']:.2f} and "
                    f"RMSE = {test['rmse']:.2f} (n = {int(test['n']):,})."
                    if test
                    else "."
                )
            )
    else:
        call = ran["load_estimator"]
        args = call.get("args") or []
        met = list(args[3]) if len(args) > 3 else []
        pct = info.get("chronos_pct")
        if zh:
            out.append(
                f"未训练模型，而是以预训练时间序列基础模型 Chronos-2 零样本使用，"
                f"气象变量（{_names(met, 'zh')}）经其协变量通道输入"
                + (f"；打乱未来天气使预测中位数平均偏移预测值的 {pct:.1f}%。" if pct else "。")
            )
        else:
            out.append(
                "No model was trained: the pretrained Chronos-2 time-series foundation model "
                f"was used zero-shot, the meteorology ({_names(met, 'en')}) entering through "
                "its covariate channel"
                + (
                    f"; shuffling the future weather moved its median forecast by "
                    f"{pct:.1f}% of the prediction."
                    if pct
                    else "."
                )
            )

    # -- normalisation ---------------------------------------------------------------
    if "normalise" in ran:
        call = ran["normalise"]
        n = call.get("n_samples")
        res = list(call.get("variables_resample") or [])
        q = call.get("return_quantiles")
        if zh:
            out.append(
                f"归一化序列为 {n} 次预测的平均：每次将 {_names(res, 'zh')} 整体替换为从记录"
                f"本身随机（有放回）抽取的一行，其余预测变量保持观测值（Grange 等，2018；"
                f"Grange 和 Carslaw，2019）"
                + ("，并保留各次预测的 5%、25%、75% 和 95% 分位数。" if q else "。")
            )
        else:
            out.append(
                f"The normalised series was the mean of {n} predictions, in each of which "
                f"{_names(res, 'en')} were replaced together by a row drawn at random, with "
                "replacement, from the record itself, the other predictors keeping their "
                "observed values (Grange et al., 2018; Grange and Carslaw, 2019)"
                + (
                    "; the 5th, 25th, 75th and 95th percentiles of the predictions were kept."
                    if q
                    else "."
                )
            )
    elif "deweather" in ran:
        args = ran["deweather"].get("args") or []
        n = args[4] if len(args) > 4 else None
        if zh:
            out.append(f"归一化序列为 {n} 次对气象重采样后 Chronos-2 预测的平均。")
        else:
            out.append(
                f"The normalised series was the mean of {n} Chronos-2 forecasts, each with "
                "the meteorology resampled from the record."
            )

    # -- decomposition ---------------------------------------------------------------
    if "decompose" in ran:
        call = ran["decompose"]
        n = call.get("n_samples")
        if call.get("method") == "emission":
            if zh:
                out.append(
                    f"排放分解通过嵌套归一化（每次 {n} 次抽样）将归一化序列分为趋势、季节、"
                    "星期、日变化和短期分量：依次将趋势、年内日序、星期和小时保持为观测值。"
                )
            else:
                out.append(
                    "The normalised series was split by nested normalisation, each of "
                    f"{n} draws, into trend, seasonal, weekly, diurnal and short-term "
                    "components, the trend, the day of year, the weekday and the hour being "
                    "held at their observed values in turn."
                )
        else:
            groups = call.get("groups")
            how = call.get("attribution") or ("shapley" if groups else "sequential")
            parts = (
                [f"{k} ({_names(list(v), lang)})" for k, v in groups.items()]
                if isinstance(groups, dict)
                else _met(call.get("covariates"))
            )
            if zh:
                how_zh = "Shapley 值" if how == "shapley" else "顺序"
                out.append(
                    f"天气的影响（观测 − 归一化）以{how_zh}归因分摊到"
                    f"{'各分组' if groups else '各气象变量'}：{_names(parts, 'zh')}，"
                    f"每次归一化 {n} 次抽样。"
                )
            else:
                how_en = "Shapley values" if how == "shapley" else "sequential attribution"
                out.append(
                    "The weather's influence (observed − normalised) was split by "
                    f"{how_en} among {'groups' if groups else 'the meteorological variables'}: "
                    f"{_names(parts, 'en')}, each normalisation of {n} draws."
                )

    # -- rolling, multi-scale, PDP ---------------------------------------------------
    if "rolling" in ran:
        call = ran["rolling"]
        w, s, n = call.get("window_days"), call.get("rolling_every"), call.get("n_samples")
        if zh:
            out.append(
                f"滚动归一化使用 {w} 天的窗口，每 {s} 天开始一个，每个窗口 {n} 次抽样，"
                "天气只从窗口内抽取。"
            )
        else:
            out.append(
                f"A rolling normalisation used {w}-day windows started every {s} days, "
                f"{n} draws each, the weather drawn from within each window only."
            )
    if "compute_multiscale" in ran:
        call = ran["compute_multiscale"]
        f, m, s = call.get("fast_days"), call.get("meso_days"), call.get("slow_days")
        if zh:
            out.append(
                f"多尺度分解对 {f}、{m} 和 {s} 天宽度的滚动归一化逐级求差，并与全记录归一化相比。"
            )
        else:
            out.append(
                f"A multi-scale decomposition differenced rolling normalisations {f}, {m} and "
                f"{s} days wide against one another and against the full-record one."
            )
    if "pdp" in ran:
        call = ran["pdp"]
        vs = list(call.get("var_list") or [])
        g = call.get("grid_points")
        if zh:
            out.append(f"对 {_names(vs, 'zh')} 计算了部分依赖曲线（{g} 个网格点）。")
        else:
            out.append(
                f"Partial-dependence curves were computed for {_names(vs, 'en')} ({g} grid points)."
            )

    refs = sorted(set(cites)) + (
        REFERENCES["en"] if any(k in ran for k in ("normalise", "deweather")) else []
    )
    head = "参考文献" if zh else "References"
    return " ".join(out) + ("\n\n" + head + "\n" + "\n".join(refs) if refs else "")
