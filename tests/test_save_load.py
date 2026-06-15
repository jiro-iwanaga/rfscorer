import io
import json
import pickle
import zipfile

import pandas as pd
import pytest

from rfscorer import RecencyFrequencyScorer

_OBS_PERIOD = ("2024-01-01", "2024-01-07")
_GT_PERIOD = ("2024-01-08", "2024-01-14")


def _make_df():
    rows = [
        ("u1", "item1", "2024-01-01"),
        ("u1", "item1", "2024-01-03"),
        ("u1", "item1", "2024-01-05"),
        ("u1", "item2", "2024-01-02"),
        ("u2", "item1", "2024-01-04"),
        ("u2", "item2", "2024-01-06"),
        ("u2", "item2", "2024-01-07"),
        ("u1", "item1", "2024-01-09"),
        ("u2", "item2", "2024-01-10"),
    ]
    return pd.DataFrame(rows, columns=["user", "item", "datetime"])


def _split(df):
    obs_mask = (df["datetime"] >= _OBS_PERIOD[0]) & (df["datetime"] <= _OBS_PERIOD[1])
    gt_mask = (df["datetime"] >= _GT_PERIOD[0]) & (df["datetime"] <= _GT_PERIOD[1])
    return df[obs_mask], df[gt_mask]


@pytest.fixture(scope="module")
def fitted_scorer():
    s = RecencyFrequencyScorer()
    s.fit(*_split(_make_df()), recency_limit=7, frequency_limit=3)
    return s


@pytest.fixture(scope="module")
def optimized_scorer():
    s = RecencyFrequencyScorer()
    s.fit(*_split(_make_df()), recency_limit=7, frequency_limit=3)
    s.optimize(kind="mono")
    return s


class TestSaveLoad:
    def test_save_load_after_fit(self, fitted_scorer, tmp_path):
        path = tmp_path / "model.pkl"
        fitted_scorer.save(path)
        loaded = RecencyFrequencyScorer.load(path)

        assert loaded.predict(1, 1) == fitted_scorer.predict(1, 1)
        assert loaded.predict(3, 2) == fitted_scorer.predict(3, 2)

    def test_save_load_after_optimize(self, optimized_scorer, tmp_path):
        path = tmp_path / "model.pkl"
        optimized_scorer.save(path)
        loaded = RecencyFrequencyScorer.load(path)

        assert loaded.predict(1, 1, kind="mono") == optimized_scorer.predict(1, 1, kind="mono")
        assert loaded.predict(3, 2, kind="mono") == optimized_scorer.predict(3, 2, kind="mono")

    def test_save_path_none(self, fitted_scorer, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        fitted_scorer.save()
        assert (tmp_path / "rfscorer.pkl").exists()

    def test_save_path_directory(self, fitted_scorer, tmp_path):
        fitted_scorer.save(tmp_path)
        assert (tmp_path / "rfscorer.pkl").exists()

    def test_save_path_file(self, fitted_scorer, tmp_path):
        path = tmp_path / "custom_name.pkl"
        fitted_scorer.save(path)
        assert path.exists()

    def test_load_version_mismatch(self, fitted_scorer, tmp_path):
        path = tmp_path / "model.pkl"
        fitted_scorer.save(path)

        with path.open("rb") as f:
            payload = pickle.load(f)  # noqa: S301
        payload["rfscorer_version"] = "99.99.0"
        with path.open("wb") as f:
            pickle.dump(payload, f)

        with pytest.warns(UserWarning, match="Version mismatch"):
            RecencyFrequencyScorer.load(path)

    def test_path_accepts_string(self, fitted_scorer, tmp_path):
        path = str(tmp_path / "model.pkl")
        fitted_scorer.save(path)
        loaded = RecencyFrequencyScorer.load(path)
        assert loaded.predict(1, 1) == fitted_scorer.predict(1, 1)

    def test_path_accepts_pathlib(self, fitted_scorer, tmp_path):
        path = tmp_path / "model.pkl"
        fitted_scorer.save(path)
        loaded = RecencyFrequencyScorer.load(path)
        assert loaded.predict(1, 1) == fitted_scorer.predict(1, 1)


class TestSaveZipLoadZip:
    def test_save_zip_load_zip_after_fit(self, fitted_scorer, tmp_path):
        path = tmp_path / "model.zip"
        fitted_scorer.save_zip(path)
        loaded = RecencyFrequencyScorer.load_zip(path)

        assert loaded.predict(1, 1) == fitted_scorer.predict(1, 1)
        assert loaded.predict(3, 2) == fitted_scorer.predict(3, 2)

    def test_save_zip_load_zip_after_optimize(self, optimized_scorer, tmp_path):
        path = tmp_path / "model.zip"
        optimized_scorer.save_zip(path)
        loaded = RecencyFrequencyScorer.load_zip(path)

        assert loaded.predict(1, 1, kind="mono") == optimized_scorer.predict(1, 1, kind="mono")
        assert loaded.predict(3, 2, kind="mono") == optimized_scorer.predict(3, 2, kind="mono")

    def test_save_zip_path_none(self, fitted_scorer, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        fitted_scorer.save_zip()
        assert (tmp_path / "rfscorer.zip").exists()

    def test_save_zip_path_directory(self, fitted_scorer, tmp_path):
        fitted_scorer.save_zip(tmp_path)
        assert (tmp_path / "rfscorer.zip").exists()

    def test_save_zip_path_file(self, fitted_scorer, tmp_path):
        path = tmp_path / "custom.zip"
        fitted_scorer.save_zip(path)
        assert path.exists()

    def test_save_zip_contents_after_fit(self, fitted_scorer, tmp_path):
        path = tmp_path / "model.zip"
        fitted_scorer.save_zip(path)

        with zipfile.ZipFile(path, "r") as zf:
            names = set(zf.namelist())
            assert "metadata.json" in names
            assert "rfscorer.pkl" in names
            for kind in ("emp", "er", "ef"):
                assert f"probabilities/{kind}_probability.csv" in names
            assert "plots/emp_surface.png" in names
            assert "plots/er_marginal.png" in names
            assert "plots/ef_marginal.png" in names

            meta = json.loads(zf.read("metadata.json"))
            assert meta["rfscorer_version"] is not None
            assert meta["optimized_kinds"] == []

    def test_save_zip_contents_after_optimize(self, optimized_scorer, tmp_path):
        path = tmp_path / "model.zip"
        optimized_scorer.save_zip(path)

        with zipfile.ZipFile(path, "r") as zf:
            names = set(zf.namelist())
            assert "probabilities/mono_probability.csv" in names
            assert "plots/mono_surface.png" in names

            meta = json.loads(zf.read("metadata.json"))
            assert "mono" in meta["optimized_kinds"]

    def test_save_zip_version_mismatch(self, fitted_scorer, tmp_path):
        path = tmp_path / "model.zip"
        fitted_scorer.save_zip(path)

        with zipfile.ZipFile(path, "r") as zf_r:
            original_pkl = zf_r.read("rfscorer.pkl")
            names = zf_r.namelist()
            contents = {name: zf_r.read(name) for name in names}

        payload = pickle.loads(original_pkl)  # noqa: S301
        payload["rfscorer_version"] = "99.99.0"
        modified_pkl = io.BytesIO()
        pickle.dump(payload, modified_pkl)
        contents["rfscorer.pkl"] = modified_pkl.getvalue()

        modified_path = tmp_path / "modified.zip"
        with zipfile.ZipFile(modified_path, "w") as zf_w:
            for name, data in contents.items():
                zf_w.writestr(name, data)

        with pytest.warns(UserWarning, match="Version mismatch"):
            RecencyFrequencyScorer.load_zip(modified_path)

    def test_save_zip_path_accepts_string(self, fitted_scorer, tmp_path):
        path = str(tmp_path / "model.zip")
        fitted_scorer.save_zip(path)
        loaded = RecencyFrequencyScorer.load_zip(path)
        assert loaded.predict(1, 1) == fitted_scorer.predict(1, 1)
