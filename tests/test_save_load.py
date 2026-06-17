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


@pytest.fixture(scope="module")
def marginal_optimized_scorer():
    s = RecencyFrequencyScorer()
    s.fit(*_split(_make_df()), recency_limit=7, frequency_limit=3)
    s.optimize(kind="mr")
    s.optimize(kind="mf")
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

    def test_load_patch_version_difference_no_warning(self, fitted_scorer, tmp_path, recwarn):
        # バージョン判定は major.minor のみ。パッチ差では警告を出さない
        from importlib.metadata import version

        major, minor = version("rfscorer").split(".")[:2]
        path = tmp_path / "model.pkl"
        fitted_scorer.save(path)

        with path.open("rb") as f:
            payload = pickle.load(f)  # noqa: S301
        payload["rfscorer_version"] = f"{major}.{minor}.999"
        with path.open("wb") as f:
            pickle.dump(payload, f)

        RecencyFrequencyScorer.load(path)
        assert not any("Version mismatch" in str(w.message) for w in recwarn)

    def test_load_unknown_version_no_warning(self, fitted_scorer, tmp_path, recwarn):
        # rfscorer_version キーが無い (saved="unknown") 場合は警告しない
        path = tmp_path / "model.pkl"
        fitted_scorer.save(path)

        with path.open("rb") as f:
            payload = pickle.load(f)  # noqa: S301
        del payload["rfscorer_version"]
        with path.open("wb") as f:
            pickle.dump(payload, f)

        RecencyFrequencyScorer.load(path)
        assert not any("Version mismatch" in str(w.message) for w in recwarn)

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
        assert (tmp_path / "scorer.zip").exists()

    def test_save_zip_path_directory(self, fitted_scorer, tmp_path):
        fitted_scorer.save_zip(tmp_path)
        assert (tmp_path / "scorer.zip").exists()

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

    def test_save_zip_contents_after_marginal_optimize(self, marginal_optimized_scorer, tmp_path):
        # mr/mf は 1D モデルのため marginal PNG を生成し、surface PNG は作られない
        path = tmp_path / "model.zip"
        marginal_optimized_scorer.save_zip(path)

        with zipfile.ZipFile(path, "r") as zf:
            names = set(zf.namelist())
            assert "probabilities/mr_probability.csv" in names
            assert "probabilities/mf_probability.csv" in names
            assert "plots/mr_marginal.png" in names
            assert "plots/mf_marginal.png" in names
            # 1D モデルに surface PNG は存在しない
            assert "plots/mr_surface.png" not in names
            assert "plots/mf_surface.png" not in names

            meta = json.loads(zf.read("metadata.json"))
            assert "mr" in meta["optimized_kinds"]
            assert "mf" in meta["optimized_kinds"]

    def test_save_zip_metadata_fields(self, fitted_scorer, tmp_path):
        # metadata.json に主要なパラメータ・統計が記録される
        path = tmp_path / "model.zip"
        fitted_scorer.save_zip(path)

        with zipfile.ZipFile(path, "r") as zf:
            meta = json.loads(zf.read("metadata.json"))
        assert meta["user_col"] == "user"
        assert meta["item_col"] == "item"
        assert meta["time_col"] == "datetime"
        assert meta["recency_limit"] == 7
        assert meta["frequency_limit"] == 3
        assert meta["observation_start"] is not None
        assert meta["observation_end"] is not None

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
