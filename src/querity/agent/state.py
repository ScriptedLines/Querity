from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field

AgentStatus = Literal[
    "running",
    "awaiting_clarification",
    "awaiting_approval",
    "completed",
    "rejected",
    "failed",
]

Severity = Literal["blocking", "warning", "info"]


class SQLIssue(BaseModel):
    model_config = ConfigDict(extra="ignore")

    code: str = "sql_issue"
    message: str
    severity: Severity = "blocking"
    suggestion: str | None = None


class QueryPlan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    intent: str = Field(description="One-sentence restatement of what the user wants.")
    answerable: bool = Field(default=True)
    search_terms: list[str] = Field(default_factory=list)
    tables: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    filters: list[str] = Field(default_factory=list)
    dimensions: list[str] = Field(default_factory=list)
    time_range: str | None = None
    assumptions: list[str] = Field(default_factory=list)
    needs_clarification: bool = False
    clarification_question: str | None = None


class SQLCandidate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sql: str = Field(description="One PostgreSQL SELECT statement.")
    explanation: str = Field(default="")
    assumptions: list[str] = Field(default_factory=list)
    tables_used: list[str] = Field(default_factory=list)
    confidence: float | None = None


class GuardResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ok: bool
    sql: str
    sql_hash: str
    errors: list[SQLIssue] = Field(default_factory=list)
    warnings: list[SQLIssue] = Field(default_factory=list)
    tables_used: list[str] = Field(default_factory=list)
    limit_applied: int | None = None


class ReviewVerdict(BaseModel):
    model_config = ConfigDict(extra="ignore")

    approved: bool
    summary: str = ""
    issues: list[SQLIssue] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)

    @property
    def blocking(self) -> bool:
        return any(issue.severity == "blocking" for issue in self.issues)


class HumanDecision(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: Literal["approve", "edit", "revise", "reject", "clarification"]
    sql: str | None = None
    sql_hash: str | None = None
    feedback: str | None = None
    reason: str | None = None
    answer: str | None = None
    pre_approve: bool = True


class ExecutionResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ok: bool
    columns: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    duration_ms: int = 0
    error: str | None = None


class TableHit(BaseModel):
    model_config = ConfigDict(extra="ignore")

    key: str
    schema_name: str
    table_name: str
    business_description: str = ""
    m_schema: str = ""
    score: float = 0.0


class GlossaryHit(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: str
    name: str
    description: str = ""
    synonyms: list[str] = Field(default_factory=list)
    required_tables: list[str] = Field(default_factory=list)
    sql_formula: str | None = None
    sql_condition: str | None = None
    table_name: str | None = None
    notes: str | None = None
    score: float = 0.0


class GoldenHit(BaseModel):
    model_config = ConfigDict(extra="ignore")

    example_key: str
    question: str
    sql_query: str
    tables_used: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    notes: str | None = None
    score: float = 0.0


class HistoryHit(BaseModel):
    model_config = ConfigDict(extra="ignore")

    turn_id: str
    question: str
    sql: str
    created_at: str
    score: float = 0.0


class JoinPath(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source: str
    target: str
    condition: str
    kind: str
    confidence: float = 0.0


class RetrievalBundle(BaseModel):
    model_config = ConfigDict(extra="ignore")

    tables: list[TableHit] = Field(default_factory=list)
    glossary: list[GlossaryHit] = Field(default_factory=list)
    golden: list[GoldenHit] = Field(default_factory=list)
    history: list[HistoryHit] = Field(default_factory=list)
    join_paths: list[JoinPath] = Field(default_factory=list)
    context_text: str = ""
    warnings: list[str] = Field(default_factory=list)


class AttemptRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    attempt: int
    sql: str
    guard_ok: bool | None = None
    review_approved: bool | None = None
    issues: list[SQLIssue] = Field(default_factory=list)


class AgentState(TypedDict, total=False):
    conversation_id: str
    db_fingerprint: str
    thread_id: str
    run_id: str
    user_message_id: str
    question: str
    recent_context: str
    clarification: str | None
    clarification_count: int
    plan: QueryPlan | None
    retrieval: RetrievalBundle | None
    candidate: SQLCandidate | None
    sql: str | None
    sql_hash: str | None
    guard: GuardResult | None
    review: ReviewVerdict | None
    feedback: Annotated[list[SQLIssue], operator.add]
    attempts: Annotated[list[AttemptRecord], operator.add]
    revision_count: int
    human_revision_count: int
    human_decision: HumanDecision | None
    pre_approved: bool
    human_edited: bool
    execution: ExecutionResult | None
    status: AgentStatus
    error: str | None
    final_message: str | None
    metadata: dict[str, Any]
