"""Mock-based unit tests for model/io.py (save_model, load_model)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from normet.backends import backend_registry
from normet.model.io import load_model, save_model

# ---------------------------------------------------------------------------
# Mock backend
# ---------------------------------------------------------------------------


class MockIOBackend:
    name = "mock_io"

    def train(
        self,
        df: Any,
        target: str = "value",
        covariates: list[str] | None = None,
        variables: list[str] | None = None,
        model_config: dict[str, Any] | None = None,
        seed: int = 7654321,
        verbose: bool = True,
    ) -> object:
        raise NotImplementedError  # not needed for I/O tests

    def save(self, model: object, path: str = ".", filename: str = "automl.joblib") -> str:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        fpath = p / filename
        fpath.write_text("")  # touch the file so Path(result).exists() passes
        return str(fpath)

    def load(self, path: str = ".", filename: str | None = None) -> object:
        class _Stub:
            backend = "mock_io"

        return _Stub()


@pytest.fixture(autouse=True)
def _mock_io_backend(tmp_path: Path):
    backend_registry._backends["mock_io"] = MockIOBackend()
    yield
    del backend_registry._backends["mock_io"]


# ---------------------------------------------------------------------------
# save_model
# ---------------------------------------------------------------------------


class TestSaveModel:
    def test_save_model(self, tmp_path: Path) -> None:
        model = MagicMock()
        model.backend = "mock_io"
        result = save_model(model, path=str(tmp_path), filename="test_model.joblib")
        assert Path(result).exists()

    def test_save_model_no_backend_attr(self) -> None:
        model = MagicMock(spec=[])  # no backend attribute
        with pytest.raises(AttributeError, match="backend"):
            save_model(model)

    def test_save_model_unknown_backend(self) -> None:
        model = MagicMock()
        model.backend = "nonexistent"
        with pytest.raises(ValueError, match="Unknown backend"):
            save_model(model)

    def test_save_model_default_path(self, tmp_path) -> None:
        model = MagicMock()
        model.backend = "mock_io"
        result = save_model(model, path=str(tmp_path), filename="test_out.joblib")
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# load_model
# ---------------------------------------------------------------------------


class TestLoadModel:
    def test_load_model(self) -> None:
        loaded = load_model(backend="mock_io")
        assert loaded.backend == "mock_io"

    def test_load_model_unknown_backend(self) -> None:
        with pytest.raises(ValueError, match="Unknown backend"):
            load_model(backend="nonexistent")

    def test_load_model_default_backend_is_flaml(self) -> None:
        import inspect

        sig = inspect.signature(load_model)
        assert sig.parameters["backend"].default == "flaml"


def test_load_model_uses_the_file_when_path_is_a_file(tmp_path: Path) -> None:
    """``save_model`` returns a file path; ``load_model`` of it used to look
    for ``<that file>/automl.joblib`` instead.

    The backend's ``filename`` only names a file *inside* a directory, so for
    a file path it must be passed as None (the backends already read a file
    directly in that case).
    """
    calls: list[tuple[str, str | None]] = []

    class Recording(MockIOBackend):
        def load(self, path: str = ".", filename: str | None = None) -> object:
            calls.append((path, filename))
            return super().load(path=path, filename=filename)

    model = MagicMock()
    model.backend = "mock_io"
    saved = save_model(model, path=str(tmp_path), filename="my_model.joblib")
    backend_registry._backends["mock_io"] = Recording()

    loaded = load_model(saved, backend="mock_io")
    assert loaded.backend == "mock_io"
    assert calls == [(saved, None)]

    # A directory still resolves through the filename (the default here).
    load_model(str(tmp_path), backend="mock_io")
    assert calls[-1] == (str(tmp_path), "automl.joblib")
