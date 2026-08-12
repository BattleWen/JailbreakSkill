"""Execute skill scripts through the JSON stdin/stdout protocol."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from core.schemas import SkillContext, SkillExecutionResult, SkillSpec


class SkillExecutor:
    """Run skill entry scripts and parse their outputs."""

    def __init__(self, project_root: Path, timeout_seconds: int = 30) -> None:
        self.project_root = project_root
        self.timeout_seconds = timeout_seconds

    def execute(self, spec: SkillSpec, context: SkillContext) -> SkillExecutionResult:
        """Execute one skill and return its parsed result."""
        entry_path = Path(spec.root_dir) / spec.entry
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(self.project_root), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)

        process = subprocess.run(
            [sys.executable, str(entry_path)],
            input=json.dumps(context.to_dict(), ensure_ascii=False),
            capture_output=True,
            text=True,
            cwd=self.project_root,
            env=env,
            timeout=self.timeout_seconds,
            check=False,
        )

        if process.returncode != 0:
            raise RuntimeError(
                f"Skill {spec.name} failed with code {process.returncode}: {process.stderr.strip()}"
            )

        stdout = process.stdout.strip()
        if not stdout:
            raise RuntimeError(f"Skill {spec.name} returned empty stdout.")

        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Skill {spec.name} emitted invalid JSON: {stdout}") from exc

        result = SkillExecutionResult.from_dict(payload)
        result.metadata.setdefault("stderr", process.stderr.strip())
        result.metadata.setdefault("entry_path", str(entry_path))
        return result
