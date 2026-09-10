"""Exercise the RSI patch boundary with Git's real diff parser."""
import subprocess

import pytest

from apex import rsi


def text_patch(path="apex/core/loop.py", prefix=True):
    old, new = (f"a/{path}", f"b/{path}") if prefix else (path, path)
    return f"--- {old}\n+++ {new}\n@@ -1 +1 @@\n-old\n+new\n"


@pytest.mark.parametrize("path", rsi.RSI_SOURCE_FILES)
def test_allowed_text_patch_applies_only_to_allowed_source(tmp_path, monkeypatch, path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    source = tmp_path / path
    source.parent.mkdir(parents=True)
    source.write_text("old\n")
    monkeypatch.setattr(rsi, "REPO_ROOT", tmp_path)
    patch = text_patch(path)
    assert rsi._validate_patch(patch)
    assert rsi._apply_patch(patch)
    assert source.read_text() == "new\n"


@pytest.mark.parametrize("extra", [
    text_patch("apex/core/safety.py"),
    text_patch("apex/core/safety.py", prefix=False),
    '--- "a/apex/core/safety.py"\n+++ "b/apex/core/safety.py"\n@@ -1 +1 @@\n-old\n+new\n',
    "diff --git a/apex/core/safety.py b/apex/core/safety.py\nold mode 100644\nnew mode 100755\n",
    "diff --git a/apex/core/loop.py b/apex/core/loop.py\nold mode 100644\nnew mode 120000\n",
    "diff --git a/apex/core/loop.py b/apex/core/loop.py\nBinary files a/apex/core/loop.py and b/apex/core/loop.py differ\n",
    "diff --git a/apex/core/safety.py b/apex/core/loop.py\nsimilarity index 100%\nrename from apex/core/safety.py\nrename to apex/core/loop.py\n",
    "diff --git a/apex/core/safety.py b/apex/core/loop.py\nsimilarity index 100%\ncopy from apex/core/safety.py\ncopy to apex/core/loop.py\n",
    "diff --git a/apex/core/loop.py b/apex/core/loop.py\ndeleted file mode 100644\n--- a/apex/core/loop.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-old\n",
    "diff --git a/apex/core/planner.py b/apex/core/planner.py\nnew file mode 100644\n--- /dev/null\n+++ b/apex/core/planner.py\n@@ -0,0 +1 @@\n+new\n",
])
def test_mixed_diff_cannot_hide_disallowed_changes(extra):
    assert not rsi._validate_patch(text_patch() + extra)


@pytest.mark.parametrize("patch", ["", "not a patch", "--- a/apex/core/loop.py\n+++ b/apex/core/loop.py\n", text_patch().replace("+new", "+rm -rf /")])
def test_empty_malformed_and_blocked_patches_rejected(patch):
    assert not rsi._validate_patch(patch)


def test_real_git_binary_patch_rejected(tmp_path, monkeypatch):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    source = tmp_path / "apex/core/loop.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"old\0data")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    source.write_bytes(b"new\0data")
    patch = subprocess.check_output(["git", "diff", "--binary"], cwd=tmp_path, text=True)
    monkeypatch.setattr(rsi, "REPO_ROOT", tmp_path)
    assert "GIT binary patch" in patch
    assert not rsi._validate_patch(patch)


@pytest.mark.parametrize("candidate_value,benchmark_exit,benchmark_score,expected", [
    (2, 0, "0.9", None),
    (1, 1, "0.9", None),
    (1, 0, "NaN", None),
    (1, 0, "1.1", None),
    (1, 0, "0.9", 0.9),
])
def test_candidate_requires_real_regressions_and_successful_benchmark(
    tmp_path, monkeypatch, candidate_value, benchmark_exit, benchmark_score, expected
):
    import sys
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    (repo / "apex/core").mkdir(parents=True)
    (repo / "tests").mkdir()
    (repo / "apex/core/loop.py").write_text("VALUE = 1\n# baseline\n")
    (repo / "tests/test_invariant.py").write_text(
        "from pathlib import Path\n"
        "def test_value():\n"
        "    assert Path('apex/core/loop.py').read_text().startswith('VALUE = 1\\n')\n"
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "baseline"], cwd=repo, check=True)
    monkeypatch.setattr(rsi, "REPO_ROOT", repo)
    monkeypatch.setattr(rsi, "BENCH_CMD", [sys.executable, "-c", f"print('{{\"apex_score\": {benchmark_score}}}'); raise SystemExit({benchmark_exit})"])
    patch = (
        "--- a/apex/core/loop.py\n+++ b/apex/core/loop.py\n@@ -1,2 +1,2 @@\n"
        f"-VALUE = 1\n-# baseline\n+VALUE = {candidate_value}\n+# candidate\n"
    )
    assert rsi._run_candidate(0, patch, "unused.json", False, k=1) == expected
    assert (repo / "apex/core/loop.py").read_text() == "VALUE = 1\n# baseline\n"
    worktrees = subprocess.check_output(["git", "worktree", "list", "--porcelain"], cwd=repo, text=True)
    assert worktrees.count("worktree ") == 1
