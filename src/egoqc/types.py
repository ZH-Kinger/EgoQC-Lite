from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Issue:
    code: str
    severity: str
    message: str
    episode_index: Optional[int] = None
    file: Optional[str] = None
    evidence: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "Issue":
        return cls(**value)


@dataclass
class EpisodeResult:
    episode_index: int
    length: int
    tier: str = "gold"
    metrics: Dict[str, Any] = field(default_factory=dict)
    issues: List[Issue] = field(default_factory=list)
    sample_frames: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_index": self.episode_index,
            "length": self.length,
            "tier": self.tier,
            "metrics": self.metrics,
            "issues": [issue.to_dict() for issue in self.issues],
            "sample_frames": self.sample_frames,
        }

    @classmethod
    def from_dict(cls, value: Dict[str, Any]) -> "EpisodeResult":
        return cls(
            episode_index=int(value["episode_index"]),
            length=int(value["length"]),
            tier=value.get("tier", "gold"),
            metrics=value.get("metrics", {}),
            issues=[Issue.from_dict(issue) for issue in value.get("issues", [])],
            sample_frames=[int(frame) for frame in value.get("sample_frames", [])],
        )
