from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelInfo:
    owner: str
    name: str
    description: str = ""
    visibility: str = ""
    github_url: str = ""
    latest_version_id: str = ""

    @classmethod
    def from_api(cls, data: dict) -> ModelInfo:
        latest = data.get("latest_version") or {}
        return cls(
            owner=data.get("owner", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            visibility=data.get("visibility", ""),
            github_url=data.get("github_url", ""),
            latest_version_id=latest.get("id", ""),
        )


@dataclass(frozen=True)
class VersionInfo:
    id: str
    created_at: str = ""

    @classmethod
    def from_api(cls, data: dict) -> VersionInfo:
        return cls(
            id=data.get("id", ""),
            created_at=data.get("created_at", ""),
        )


@dataclass(frozen=True)
class PredictionInfo:
    id: str
    status: str
    model: str = ""
    version: str = ""
    input: dict = field(default_factory=dict)
    output: object = None
    error: str = ""
    logs: str = ""
    created_at: str = ""
    completed_at: str = ""

    @classmethod
    def from_api(cls, data: dict) -> PredictionInfo:
        return cls(
            id=data.get("id", ""),
            status=data.get("status", ""),
            model=data.get("model", ""),
            version=data.get("version", ""),
            input=data.get("input", {}),
            output=data.get("output"),
            error=data.get("error", ""),
            logs=data.get("logs", ""),
            created_at=data.get("created_at", ""),
            completed_at=data.get("completed_at", ""),
        )


@dataclass(frozen=True)
class ListModelsResult:
    success: bool
    models: list[ModelInfo] = field(default_factory=list)
    next_cursor: str = ""
    error: str = ""


@dataclass(frozen=True)
class ModelDetailResult:
    success: bool
    model: ModelInfo | None = None
    error: str = ""


@dataclass(frozen=True)
class ListVersionsResult:
    success: bool
    versions: list[VersionInfo] = field(default_factory=list)
    next_cursor: str = ""
    error: str = ""


@dataclass(frozen=True)
class PredictionResult:
    success: bool
    prediction: PredictionInfo | None = None
    error: str = ""


@dataclass(frozen=True)
class ListPredictionsResult:
    success: bool
    predictions: list[PredictionInfo] = field(default_factory=list)
    next_cursor: str = ""
    error: str = ""


@dataclass(frozen=True)
class CancelResult:
    success: bool
    error: str = ""


@dataclass(frozen=True)
class CollectionInfo:
    slug: str
    name: str
    description: str = ""

    @classmethod
    def from_api(cls, data: dict) -> CollectionInfo:
        return cls(
            slug=data.get("slug", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
        )


@dataclass(frozen=True)
class ListCollectionsResult:
    success: bool
    collections: list[CollectionInfo] = field(default_factory=list)
    next_cursor: str = ""
    error: str = ""
