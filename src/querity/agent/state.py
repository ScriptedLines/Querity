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

    code: str = Field(
        default="sql_issue",
        description="Machine-readable issue category or rule code (e.g. 'table_not_found', 'syntax_error', 'ambiguous_column', 'cartesian_product', 'destructive_command').",
    )
    message: str = Field(
        description="Clear, actionable explanation of the specific syntax error, semantic flaw, or guard violation found in the query.",
    )
    severity: Severity = Field(
        default="blocking",
        description="Issue severity level: 'blocking' prevents query execution, 'warning' highlights potential performance or data quality risks, 'info' provides non-critical guidance.",
    )
    suggestion: str | None = Field(
        default=None,
        description="Concrete suggestion or replacement SQL snippet to fix the issue.",
    )


class QueryPlan(BaseModel):
    model_config = ConfigDict(extra="ignore")

    intent: str = Field(
        description="Clear, concise one-sentence restatement of the analytical question or data goal.",
    )
    answerable: bool = Field(
        default=True,
        description=(
            "Whether the request is a valid data/analytical question suitable for a database query. "
            "Set to false only if the user request is completely non-analytical (e.g. general conversation, "
            "writing poems, external world knowledge, or requests to perform actions like sending emails). "
            "Always default to true for any query seeking business data or metrics."
        ),
    )
    search_terms: list[str] = Field(
        default_factory=list,
        description="Specific keywords, domain terms, or entity names to search for in catalog tables and metrics.",
    )
    tables: list[str] = Field(
        default_factory=list,
        description="Candidate entity or table names inferred from the question keywords (e.g. 'users', 'orders') to guide catalog search.",
    )
    metrics: list[str] = Field(
        default_factory=list,
        description="Aggregations or measures requested (e.g. 'count(distinct user_id)', 'sum(amount)', 'average revenue').",
    )
    filters: list[str] = Field(
        default_factory=list,
        description="Filtering conditions, status constraints, or WHERE predicates implied or stated in the question.",
    )
    dimensions: list[str] = Field(
        default_factory=list,
        description="Grouping columns or categorical breakdown dimensions (e.g. 'country', 'created_month', 'product_category').",
    )
    time_range: str | None = Field(
        default=None,
        description="Explicit or relative time/date window requested (e.g. 'last 30 days', 'year 2024', 'Q3 2023').",
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description="Reasonable default business logic assumptions made when interpreting ambiguous phrasing.",
    )
    needs_clarification: bool = Field(
        default=False,
        description="Set to true only if the user query is critically ambiguous or missing essential parameters that cannot be reasonably inferred.",
    )
    clarification_question: str | None = Field(
        default=None,
        description="Direct question to ask the user if needs_clarification is true. Must be null when needs_clarification is false.",
    )


class SQLCandidate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sql: str = Field(
        description="A single, complete, executable PostgreSQL SELECT query. Must use explicit JOINs, valid table/column names, and end with an optional semicolon.",
    )
    explanation: str = Field(
        default="",
        description="Concise step-by-step summary explaining how the query answers the user question, detailing join choices and filter rationale.",
    )
    assumptions: list[str] = Field(
        default_factory=list,
        description="List of domain assumptions applied (e.g. handling of soft deletes, default currency, status filtering).",
    )
    tables_used: list[str] = Field(
        default_factory=list,
        description="Fully qualified schema.table names referenced in the SELECT statement.",
    )
    confidence: float | None = Field(
        default=None,
        description="Confidence score from 0.0 to 1.0 reflecting how well the query satisfies the prompt requirements and schema constraints.",
    )


class GuardResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ok: bool = Field(
        description="Whether the SQL query passed AST analysis, dialect validation, and safety rules without blocking errors.",
    )
    sql: str = Field(
        description="The validated or rewritten executable SQL statement.",
    )
    sql_hash: str = Field(
        description="Deterministic hash of the SQL query string used for caching and approval tracking.",
    )
    errors: list[SQLIssue] = Field(
        default_factory=list,
        description="Blocking issues preventing execution (e.g. syntax errors, unknown tables, destructive operations).",
    )
    warnings: list[SQLIssue] = Field(
        default_factory=list,
        description="Non-blocking advisories (e.g. missing LIMIT, potential performance bottlenecks).",
    )
    tables_used: list[str] = Field(
        default_factory=list,
        description="List of verified table names extracted from the SQL statement.",
    )
    limit_applied: int | None = Field(
        default=None,
        description="The maximum row limit injected or enforced by the safety guard.",
    )


class ReviewVerdict(BaseModel):
    model_config = ConfigDict(extra="ignore")

    approved: bool = Field(
        description="Whether the SQL query is approved for execution. Set to false if there are semantic mistakes, missing joins, or incorrect business logic.",
    )
    summary: str = Field(
        default="",
        description="High-level peer review evaluation explaining why the query is correct or what needs to be changed.",
    )
    issues: list[SQLIssue] = Field(
        default_factory=list,
        description="Specific logical, semantic, or performance defects found during review.",
    )
    suggestions: list[str] = Field(
        default_factory=list,
        description="Concrete revision tips for the generator model to improve accuracy or performance.",
    )

    @property
    def blocking(self) -> bool:
        return any(issue.severity == "blocking" for issue in self.issues)


class HumanDecision(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: Literal["approve", "edit", "revise", "reject", "clarification"] = Field(
        description="The decision action: 'approve' to execute, 'edit' to supply manual SQL, 'revise' to request agent re-generation with feedback, 'reject' to abort, 'clarification' to provide missing context.",
    )
    sql: str | None = Field(
        default=None,
        description="User-edited SQL statement when kind is 'edit'.",
    )
    sql_hash: str | None = Field(
        default=None,
        description="Hash identifier of the reviewed SQL query.",
    )
    feedback: str | None = Field(
        default=None,
        description="Human feedback or correction instructions when requesting a revision.",
    )
    reason: str | None = Field(
        default=None,
        description="Optional rationale explaining why the query was rejected or edited.",
    )
    answer: str | None = Field(
        default=None,
        description="Human answer providing requested clarification.",
    )
    pre_approve: bool = Field(
        default=True,
        description="Whether user-edited SQL should bypass re-review and execute directly.",
    )


class ExecutionResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    ok: bool = Field(
        description="Whether database query execution succeeded without driver or database runtime errors.",
    )
    columns: list[str] = Field(
        default_factory=list,
        description="Ordered list of column names returned by the executed query.",
    )
    rows: list[list[str]] = Field(
        default_factory=list,
        description="Preview of result rows serialized as string values.",
    )
    row_count: int = Field(
        default=0,
        description="Total number of rows returned by the query.",
    )
    truncated: bool = Field(
        default=False,
        description="Whether the returned row preview was truncated due to row limit thresholds.",
    )
    duration_ms: int = Field(
        default=0,
        description="Time taken to execute the query against the database in milliseconds.",
    )
    error: str | None = Field(
        default=None,
        description="Database error message if query execution failed.",
    )


class TableHit(BaseModel):
    model_config = ConfigDict(extra="ignore")

    key: str = Field(
        description="Fully qualified table name in schema.table format.",
    )
    schema_name: str = Field(
        description="Database schema name.",
    )
    table_name: str = Field(
        description="Table name.",
    )
    business_description: str = Field(
        default="",
        description="High-level description of what information this table records.",
    )
    m_schema: str = Field(
        default="",
        description="Pre-compiled YAML schema definition detailing columns, types, and join keys.",
    )
    score: float = Field(
        default=0.0,
        description="Relevance search ranking score for this table.",
    )


class GlossaryHit(BaseModel):
    model_config = ConfigDict(extra="ignore")

    kind: str = Field(
        description="Type of glossary term (e.g. 'metric', 'dimension', 'business_term').",
    )
    name: str = Field(
        description="Canonical business term name.",
    )
    description: str = Field(
        default="",
        description="Business definition and calculation logic for this term.",
    )
    synonyms: list[str] = Field(
        default_factory=list,
        description="Alternative names and synonyms for this term.",
    )
    required_tables: list[str] = Field(
        default_factory=list,
        description="Tables required to compute or filter by this business term.",
    )
    sql_formula: str | None = Field(
        default=None,
        description="Standard SQL calculation or aggregation expression for this metric.",
    )
    sql_condition: str | None = Field(
        default=None,
        description="Standard SQL WHERE clause predicate associated with this term.",
    )
    table_name: str | None = Field(
        default=None,
        description="Primary table associated with this glossary term.",
    )
    notes: str | None = Field(
        default=None,
        description="Usage notes, caveats, or domain edge cases.",
    )
    score: float = Field(
        default=0.0,
        description="Relevance score for this glossary hit.",
    )


class GoldenHit(BaseModel):
    model_config = ConfigDict(extra="ignore")

    example_key: str = Field(
        description="Unique identifier for the golden example query.",
    )
    question: str = Field(
        description="Natural language question answered by the golden example.",
    )
    sql_query: str = Field(
        description="Verified reference SQL query implementation.",
    )
    tables_used: list[str] = Field(
        default_factory=list,
        description="Tables referenced by the golden SQL query.",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Classification tags (e.g. 'aggregation', 'multi_join', 'window_function').",
    )
    notes: str | None = Field(
        default=None,
        description="Design notes or domain conventions explained in this example.",
    )
    score: float = Field(
        default=0.0,
        description="Relevance score matching the user question.",
    )


class HistoryHit(BaseModel):
    model_config = ConfigDict(extra="ignore")

    turn_id: str = Field(
        description="Unique identifier of the past conversation turn.",
    )
    question: str = Field(
        description="Prior question asked in this conversation.",
    )
    sql: str = Field(
        description="Verified SQL query successfully executed in that prior turn.",
    )
    created_at: str = Field(
        description="Timestamp when the past query turn was executed.",
    )
    score: float = Field(
        default=0.0,
        description="Similarity score between prior question and current question.",
    )


class JoinPath(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source: str = Field(
        description="Originating table in schema.table format.",
    )
    target: str = Field(
        description="Destination table in schema.table format.",
    )
    condition: str = Field(
        description="SQL ON join predicate expression (e.g. 'orders.customer_id = customers.id').",
    )
    kind: str = Field(
        description="Relationship kind (e.g. 'foreign_key', 'inferred', 'many-to-one').",
    )
    confidence: float = Field(
        default=0.0,
        description="Confidence score for this join relationship.",
    )


class RetrievalBundle(BaseModel):
    model_config = ConfigDict(extra="ignore")

    tables: list[TableHit] = Field(
        default_factory=list,
        description="Ranked list of relevant catalog tables retrieved for the query.",
    )
    glossary: list[GlossaryHit] = Field(
        default_factory=list,
        description="Semantic glossary terms and metrics matching the user query.",
    )
    golden: list[GoldenHit] = Field(
        default_factory=list,
        description="Similar verified few-shot golden SQL examples.",
    )
    history: list[HistoryHit] = Field(
        default_factory=list,
        description="Past successful queries from the current conversation used as conversational memory.",
    )
    join_paths: list[JoinPath] = Field(
        default_factory=list,
        description="Discovered join paths connecting candidate tables.",
    )
    context_text: str = Field(
        default="",
        description="Compiled text representation of retrieval context formatted for the generator prompt.",
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Diagnostic warnings encountered during retrieval.",
    )


class AttemptRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    attempt: int = Field(
        description="1-based attempt sequence number for this query turn.",
    )
    sql: str = Field(
        description="Candidate SQL statement generated during this attempt.",
    )
    guard_ok: bool | None = Field(
        default=None,
        description="Whether the candidate passed all SQL Guard validation checks.",
    )
    review_approved: bool | None = Field(
        default=None,
        description="Whether the peer reviewer approved this candidate SQL.",
    )
    issues: list[SQLIssue] = Field(
        default_factory=list,
        description="Issues identified by guard or review during this attempt.",
    )


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
