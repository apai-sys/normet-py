import normet as nm


def test_top_level_symbols_present():
    must_have = {
        "do_all",
        "do_all_unc",
        "run_workflow",
        "normalise",
        "rolling",
        "pdp",
        "decompose",
        "scm",
        "mlscm",
        "run_scm",
        "placebo_in_space",
        "placebo_in_time",
        "uncertainty_bands",
        "effect_bands_space",
        "effect_bands_time",
        "build_model",
        "train_model",
        "ml_predict",
        "load_model",
        "save_model",
        "modStats",
        "prepare_data",
        "process_date",
        "add_lag_features",
        "add_rolling_features",
        "cyclical_encode",
        "wind_to_uv",
        "time_series_cv",
        "cv_score",
    }
    missing = must_have - set(nm.__all__)
    assert not missing, f"Missing top-level exports: {sorted(missing)}"


def test_all_is_actually_importable():
    """Every name in ``__all__`` must resolve; a typo there is a broken star-import."""
    unresolved = [name for name in nm.__all__ if not hasattr(nm, name)]
    assert not unresolved, f"Names in __all__ with no attribute: {unresolved}"


def test_imports_without_the_heavy_optional_extras():
    """``import normet`` must survive with torch, chronos and PySide6 absent.

    ``normet.physics`` and ``normet.foundation`` are re-exported at the top
    level, so a module-level import of any of the three in either subpackage
    would make the whole library unimportable for anyone who skipped the extra.
    Run in a subprocess with those names blocked, because the rest of the suite
    has already imported them into this one.
    """
    import subprocess
    import sys
    import textwrap

    script = textwrap.dedent("""
        import sys

        BLOCKED = ("torch", "chronos", "PySide6")

        class Blocker:
            def find_module(self, name, path=None):
                if name.split(".")[0] in BLOCKED:
                    raise ImportError(f"{name} blocked for this test")

        sys.meta_path.insert(0, Blocker())

        import normet
        import normet.foundation
        import normet.physics
        import normet.counterfactual

        # The pure-numpy parts must still work.
        builder = normet.PhysicsGraphBuilder(
            stations=["a", "b"], latitudes=[51.5, 53.5], longitudes=[-0.1, -2.2]
        )
        builder.build_static_graph()

        leaked = [n for n in BLOCKED if n in sys.modules]
        assert not leaked, f"imported despite being optional: {leaked}"
    """)

    proc = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
