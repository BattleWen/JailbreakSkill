"""Optional proxy bypass configuration for local or cluster services.

Import this module before HTTP clients are initialized. Generic Kubernetes
suffixes are included by default; deployments can add comma-separated hosts
through ``SKILLTEAMING_NO_PROXY`` without committing infrastructure details.
"""

from __future__ import annotations

import os

_NO_PROXY_ENTRIES = {
    ".svc",
    ".cluster.local",
}

_extra = {
    entry.strip()
    for entry in os.environ.get("SKILLTEAMING_NO_PROXY", "").split(",")
    if entry.strip()
}
_existing = {e.strip() for e in os.environ.get("no_proxy", "").split(",") if e.strip()}
_merged = ",".join(sorted(_existing | _NO_PROXY_ENTRIES | _extra))
os.environ["NO_PROXY"] = _merged
os.environ["no_proxy"] = _merged
