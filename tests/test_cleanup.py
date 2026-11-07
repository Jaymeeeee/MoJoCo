"""Exercise retention behavior and boundaries using disposable workspaces."""

import os
from pathlib import Path

import pytest

from mujoco_tools.cleanup import CleanupError, cleanup_outputs, main


NOW = 2_000_000_000.0
OLD = NOW - 8 * 86400
RECENT = NOW - 2 * 86400


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    for directory in ("assets", "config", "data", "outputs/_checks", "scripts"):
        (root / directory).mkdir(parents=True)
    (root / "pyproject.toml").write_text('[project]\nname = "cleanup-test"\n')
    (root / "config/cleanup.yaml").write_text(
        "schema_version: 1\nscope: outputs/_checks\nretention_days: 7\ndefault_mode: dry-run\n"
    )
    return root


def write_at(root: Path, relative: str, timestamp: float = OLD) -> Path:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("retain this content unless eligible\n")
    os.utime(target, (timestamp, timestamp))
    return target


def test_preview_and_apply_only_expired_checks(workspace: Path, capsys: pytest.CaptureFixture) -> None:
    old = write_at(workspace, "outputs/_checks/run/old.json")
    recent = write_at(workspace, "outputs/_checks/recent.json", RECENT)
    boundary = write_at(workspace, "outputs/_checks/exactly-seven-days.json", NOW - 7 * 86400)
    os.utime(old.parent, (OLD, OLD))
    protected = [
        write_at(workspace, path) for path in (
            "data/demo.h5", "outputs/datasets/insert_plug/run/episode.h5",
            "examples/insert_plug/scene.xml", "assets/robot.xml", "outputs/insert_plug/demo/result.json",
            "outputs/other/result.json",
        )
    ]
    preview = cleanup_outputs(workspace, now=NOW)
    assert preview.mode == "dry-run"
    assert preview.eligible_files == ["outputs/_checks/run/old.json"]
    assert preview.eligible_directories == ["outputs/_checks/run"]
    assert not preview.deleted_files
    assert old.is_file()
    # The public command also defaults to preview without --apply.
    assert main(["--root", str(workspace)]) == 0
    assert '"mode": "dry-run"' in capsys.readouterr().out
    assert old.is_file()
    result = cleanup_outputs(workspace, apply=True, now=NOW)
    assert result.deleted_files == preview.eligible_files
    assert result.deleted_directories == preview.eligible_directories
    assert not old.parent.exists()
    assert all(path.is_file() for path in [recent, boundary, *protected])
    assert (workspace / "outputs/_checks").is_dir()


def test_symlink_entries_are_preserved_and_scope_escape_is_refused(workspace: Path) -> None:
    outside = write_at(workspace, "data/outside/important.h5")
    checks = workspace / "outputs/_checks"
    (checks / "linked-file").symlink_to(outside)
    (checks / "linked-dir").symlink_to(outside.parent, target_is_directory=True)
    report = cleanup_outputs(workspace, apply=True, now=NOW)
    assert report.skipped == ["outputs/_checks/linked-dir", "outputs/_checks/linked-file"]
    assert outside.is_file()
    assert (checks / "linked-file").is_symlink()
    assert (checks / "linked-dir").is_symlink()
    for link in checks.iterdir():
        link.unlink()
    checks.rmdir()
    checks.symlink_to(outside.parent, target_is_directory=True)
    with pytest.raises(CleanupError, match="real directory"):
        cleanup_outputs(workspace, apply=True, now=NOW)
    assert outside.is_file()
    checks.unlink()
    (workspace / "outputs").rmdir()
    (workspace / "outputs").symlink_to(outside.parent, target_is_directory=True)
    with pytest.raises(CleanupError, match="without symlinks"):
        cleanup_outputs(workspace, apply=True, now=NOW)
    assert outside.is_file()
    alias = workspace.parent / "workspace-alias"
    alias.symlink_to(workspace, target_is_directory=True)
    with pytest.raises(CleanupError, match="canonical"):
        cleanup_outputs(alias, apply=True, now=NOW)
    assert outside.is_file()


def test_missing_sentinel_and_dangerous_policy_are_refused(workspace: Path) -> None:
    old = write_at(workspace, "outputs/_checks/old.json")
    sentinel = workspace / "pyproject.toml"
    sentinel.unlink()
    with pytest.raises(CleanupError, match="pyproject.toml"):
        cleanup_outputs(workspace, apply=True, now=NOW)
    assert old.is_file()
    sentinel.write_text('[project]\nname = "cleanup-test"\n')
    policy = workspace / "config/cleanup.yaml"
    original = policy.read_text()
    policy.write_text(original.replace("outputs/_checks", "data"))
    with pytest.raises(CleanupError, match="fixed scope"):
        cleanup_outputs(workspace, apply=True, now=NOW)
    assert old.is_file()
    policy.write_text(original.replace("retention_days: 7", "retention_days: 0"))
    with pytest.raises(CleanupError, match="at least 7"):
        cleanup_outputs(workspace, apply=True, now=NOW)
    assert old.is_file()
