from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ForeignKeyRef(BaseModel):
    name: str | None = None
    constrained_schema: str
    constrained_table: str
    constrained_columns: list[str]
    referred_schema: str
    referred_table: str
    referred_columns: list[str]

    @property
    def constrained_key(self) -> str:
        return f"{self.constrained_schema}.{self.constrained_table}"

    @property
    def referred_key(self) -> str:
        return f"{self.referred_schema}.{self.referred_table}"


class ColumnProfile(BaseModel):
    name: str
    data_type: str
    value_kind: Literal["string", "numeric", "temporal", "boolean", "other"] = "other"
    nullable: bool = True
    default: str | None = None
    comment: str | None = None
    primary_key: bool = False
    unique: bool = False
    foreign_key: str | None = None
    is_sensitive: bool = False

    distinct_count_estimate: float | None = None
    null_fraction: float | None = None
    allowed_values: list[str] = Field(default_factory=list)
    sample_values: list[str] = Field(default_factory=list)
    min_value: str | None = None
    max_value: str | None = None

    description: str | None = None


class TableSchema(BaseModel):
    schema_name: str
    table_name: str
    table_type: str = "table"
    comment: str | None = None
    columns: list[ColumnProfile] = Field(default_factory=list)
    primary_key: list[str] = Field(default_factory=list)
    unique_constraints: list[list[str]] = Field(default_factory=list)
    foreign_keys: list[ForeignKeyRef] = Field(default_factory=list)
    row_count_estimate: int | None = None

    @property
    def key(self) -> str:
        return f"{self.schema_name}.{self.table_name}"

    @property
    def column_names(self) -> list[str]:
        return [column.name for column in self.columns]

    def column(self, name: str) -> ColumnProfile | None:
        for column in self.columns:
            if column.name == name:
                return column
        return None


class RowSample(BaseModel):
    columns: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.rows


class ColumnDescription(BaseModel):
    name: str = Field(description="Exact column name from the supplied schema")
    description: str = Field(
        description="Plain-English meaning of the column, including unit/format when known"
    )


class TableEnrichment(BaseModel):
    """Structured output requested from the enrichment model."""

    model_config = ConfigDict(extra="forbid")

    business_description: str = Field(
        description=(
            "2-4 sentences explaining what this table represents, its grain, and when an "
            "analyst should use it."
        )
    )
    grain: str | None = Field(
        default=None,
        description="The grain of one row, e.g. 'one row per customer per month'.",
    )
    column_descriptions: list[ColumnDescription] = Field(
        description="One description for every supplied column."
    )
    sample_questions: list[str] = Field(
        description="3-5 natural-language questions this table can answer."
    )
    caveats: list[str] = Field(
        default_factory=list,
        description="Important join, filtering, freshness, or data-quality caveats.",
    )


class JoinEdge(BaseModel):
    source_key: str
    target_key: str
    source_columns: list[str]
    target_columns: list[str]
    condition: str
    kind: Literal["foreign_key", "inferred", "query_log"]
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: str | None = None


class TableCatalogRecord(BaseModel):
    source_id: str
    schema_name: str
    table_name: str
    table_type: str
    business_description: str
    m_schema: str
    columns: list[dict[str, Any]]
    enrichment: dict[str, Any]
    sample_questions: list[str]
    join_keys: list[dict[str, Any]]
    row_count_estimate: int | None
    structural_hash: str
    profile_hash: str
    enrichment_hash: str
    content: str
    content_hash: str
    embedding: list[float] | None = None
    embedding_model: str | None = None

    @property
    def key(self) -> str:
        return f"{self.schema_name}.{self.table_name}"


class ExistingTableState(BaseModel):
    schema_name: str
    table_name: str
    business_description: str
    m_schema: str
    columns: list[dict[str, Any]]
    enrichment: dict[str, Any]
    sample_questions: list[str]
    content: str
    content_hash: str
    structural_hash: str
    profile_hash: str
    enrichment_hash: str
    embedding_model: str | None
    has_embedding: bool

    @property
    def key(self) -> str:
        return f"{self.schema_name}.{self.table_name}"


class ExistingSemanticState(BaseModel):
    kind: str
    name: str
    content_hash: str
    embedding_model: str | None
    has_embedding: bool
    is_active: bool


class ExistingGoldenState(BaseModel):
    example_key: str
    content_hash: str
    embedding_model: str | None
    has_embedding: bool
    is_active: bool
