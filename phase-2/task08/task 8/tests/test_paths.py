from app._paths import _source_dirs


def test_source_dirs_accepts_repository_root_with_phase2_tasks(tmp_path):
    phase2 = tmp_path / "phase-2"
    (phase2 / "task05" / "src").mkdir(parents=True)
    (phase2 / "task06" / "task 6" / "src").mkdir(parents=True)
    (phase2 / "task07" / "task 7" / "src").mkdir(parents=True)

    found = _source_dirs(tmp_path)

    assert set(found) == {"task 5", "task 6", "task 7"}
    assert found["task 5"] == phase2 / "task05" / "src"
