"""Conservative retention cleanup for the workspace's disposable checks.

Only regular files and empty directories below outputs/_checks are eligible.
Traversal uses directory file descriptors and never follows symbolic links.
The workspace, its policy, and its layout are validated before any deletion.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import stat
import time
import tomllib

import yaml


SCOPE = "outputs/_checks"
MIN_RETENTION_DAYS = 7
_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
_LAYOUT = ("assets", "config", "data", "outputs", "scripts")


class CleanupError(ValueError):
    """The workspace or policy is unsafe or invalid for cleanup."""


@dataclass
class CleanupReport:
    root: str
    mode: str
    retention_days: int
    scope: str = SCOPE
    eligible_files: list[str] = field(default_factory=list)
    eligible_directories: list[str] = field(default_factory=list)
    deleted_files: list[str] = field(default_factory=list)
    deleted_directories: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def _open_directory(name: str | Path, parent_fd: int | None = None) -> int:
    try:
        return os.open(name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
    except OSError as exc:
        raise CleanupError(f"Expected a real directory, without symlinks: {name}") from exc


def _read_file(name: str, directory_fd: int) -> str:
    try:
        fd = os.open(name, _FILE_FLAGS, dir_fd=directory_fd)
    except OSError as exc:
        raise CleanupError(f"Expected a regular file, without symlinks: {name}") from exc
    with os.fdopen(fd, "r", encoding="utf-8") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise CleanupError(f"Expected a regular file: {name}")
        return stream.read()


def _open_workspace_directory(root: Path) -> int:
    """Open each ancestor separately so a symlink race cannot redirect traversal."""
    current_fd = _open_directory(root.anchor)
    try:
        for component in root.parts[1:]:
            child_fd = _open_directory(component, current_fd)
            os.close(current_fd)
            current_fd = child_fd
        return current_fd
    except BaseException:
        os.close(current_fd)
        raise


def _validate_workspace(root: Path, stack: ExitStack) -> tuple[int, int]:
    # Resolve strictly, then compare to the lexical path to reject symlinked
    # roots and ancestors before opening the workspace itself.
    try:
        canonical = root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise CleanupError(f"Workspace does not exist: {root}") from exc
    if canonical != root:
        raise CleanupError("Workspace path must be canonical and contain no symlinks or '..'")
    root_fd = _open_workspace_directory(root)
    stack.callback(os.close, root_fd)
    try:
        project = tomllib.loads(_read_file("pyproject.toml", root_fd))
    except tomllib.TOMLDecodeError as exc:
        raise CleanupError("Workspace pyproject.toml must contain valid TOML") from exc
    metadata = project.get("project", {})
    if not isinstance(metadata, dict) or not metadata.get("name"):
        raise CleanupError("Workspace pyproject.toml must declare a project name")
    directory_fds = {}
    for name in _LAYOUT:
        directory_fds[name] = _open_directory(name, root_fd)
        stack.callback(os.close, directory_fds[name])
    try:
        policy = yaml.safe_load(_read_file("cleanup.yaml", directory_fds["config"]))
    except yaml.YAMLError as exc:
        raise CleanupError("Invalid config/cleanup.yaml") from exc
    if not isinstance(policy, dict):
        raise CleanupError("Cleanup policy must be a mapping")
    if policy.get("schema_version") != 1 or policy.get("scope") != SCOPE:
        raise CleanupError(f"Cleanup policy must use schema version 1 and fixed scope {SCOPE}")
    days = policy.get("retention_days")
    if type(days) is not int or days < MIN_RETENTION_DAYS:
        raise CleanupError(f"Retention must be at least {MIN_RETENTION_DAYS} whole days")
    if policy.get("default_mode") != "dry-run":
        raise CleanupError("The default cleanup mode must remain dry-run")
    return directory_fds["outputs"], days


def cleanup_outputs(
    root: str | Path, *, apply: bool = False, now: float | None = None
) -> CleanupReport:
    """Inspect or clean outputs/_checks without following symlinks.

    ``now`` is injectable for deterministic retention tests. The policy may
    lengthen retention, but may not shorten it below seven days or change scope.
    The _checks directory itself is always retained. Dry-run directory candidates
    account for files that would be removed, so preview matches an applied run.
    """
    workspace = Path(root).expanduser().absolute()
    with ExitStack() as stack:
        outputs_fd, days = _validate_workspace(workspace, stack)
        report = CleanupReport(str(workspace), "apply" if apply else "dry-run", days)
        try:
            scope_stat = os.stat("_checks", dir_fd=outputs_fd, follow_symlinks=False)
        except FileNotFoundError:
            return report
        if not stat.S_ISDIR(scope_stat.st_mode):
            raise CleanupError(f"{SCOPE} must be a real directory, not a symlink or file")
        checks_fd = _open_directory("_checks", outputs_fd)
        stack.callback(os.close, checks_fd)
        cutoff = (time.time() if now is None else now) - days * 86400
        device = os.fstat(checks_fd).st_dev

        def visit(directory_fd: int, relative: Path) -> bool:
            """Return whether this directory would be empty after cleanup."""
            remaining = False
            for name in sorted(os.listdir(directory_fd)):
                item = relative / name
                label = item.as_posix()
                try:
                    info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                except FileNotFoundError:
                    continue  # Another process already removed this entry.
                if stat.S_ISREG(info.st_mode):
                    if info.st_mtime < cutoff:
                        report.eligible_files.append(label)
                        if apply:
                            # Recheck identity before unlinking a potentially
                            # concurrently replaced entry. unlink cannot follow
                            # a symlink, even if replacement races this check.
                            current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                            if (current.st_dev, current.st_ino, current.st_mtime_ns) != (
                                info.st_dev, info.st_ino, info.st_mtime_ns
                            ):
                                report.skipped.append(f"{label} (changed during cleanup)")
                                remaining = True
                                continue
                            os.unlink(name, dir_fd=directory_fd)
                            report.deleted_files.append(label)
                    else:
                        remaining = True
                elif stat.S_ISDIR(info.st_mode) and info.st_dev == device:
                    child_fd = _open_directory(name, directory_fd)
                    try:
                        opened = os.fstat(child_fd)
                        if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
                            report.skipped.append(f"{label} (changed during cleanup)")
                            remaining = True
                            continue
                        empty = visit(child_fd, item)
                    finally:
                        os.close(child_fd)
                    if empty and info.st_mtime < cutoff:
                        report.eligible_directories.append(label)
                        if apply:
                            current = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                            if (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino):
                                report.skipped.append(f"{label} (changed during cleanup)")
                                remaining = True
                                continue
                            # rmdir also refuses symlinks and nonempty directories.
                            os.rmdir(name, dir_fd=directory_fd)
                            report.deleted_directories.append(label)
                    else:
                        remaining = True
                else:
                    # Includes symlinks, special files, and mounted directories.
                    report.skipped.append(label)
                    remaining = True
            return not remaining

        visit(checks_fd, Path(SCOPE))
        return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[2],
        help="workspace root; requires pyproject.toml, standard layout, and cleanup policy",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="delete eligible entries; without this flag, only print a preview",
    )
    args = parser.parse_args(argv)
    try:
        result = cleanup_outputs(args.root, apply=args.apply)
    except (CleanupError, OSError, UnicodeError) as exc:
        parser.exit(2, f"Cleanup refused or stopped: {exc}\n")
    print(json.dumps(asdict(result), indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
