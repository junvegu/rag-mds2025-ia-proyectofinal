from src.config import get_settings


def test_settings_paths_are_derived_from_project_root() -> None:
    settings = get_settings()
    assert settings.raw_data_dir == settings.project_root / "data" / "raw"
    assert settings.processed_data_dir == settings.project_root / "data" / "processed"
    assert settings.eval_data_dir == settings.project_root / "data" / "eval"
