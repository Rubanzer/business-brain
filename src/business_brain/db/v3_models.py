"""ORM models for v3 features: sources, sanctity, alerts, patterns, onboarding, format detection."""

import uuid

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSON

from business_brain.db.models import Base


def _uuid() -> str:
    return str(uuid.uuid4())


# ---------------------------------------------------------------------------
# Data Source Management
# ---------------------------------------------------------------------------


class DataSource(Base):
    """A connected data source (Google Sheet, API, recurring upload, manual upload)."""

    __tablename__ = "data_sources"

    id = Column(String(36), primary_key=True, default=_uuid)
    name = Column(String(500), nullable=False)
    source_type = Column(String(50), nullable=False)  # google_sheet / api / recurring_upload / manual_upload
    connection_config = Column(JSON, nullable=True)  # {sheet_id, tab_name, api_url, headers, ...}
    table_name = Column(String(255), nullable=False)
    format_fingerprint = Column(String(64), nullable=True)  # hash of column structure
    sync_frequency_minutes = Column(Integer, default=0)  # 0 = manual only
    last_sync_at = Column(DateTime(timezone=True), nullable=True)
    last_sync_status = Column(String(20), nullable=True)  # success / error
    last_sync_error = Column(Text, nullable=True)
    rows_total = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    active = Column(Boolean, default=True)
    session_id = Column(String(64), nullable=True)


class DataChangeLog(Base):
    """Tracks individual data changes from syncs and uploads."""

    __tablename__ = "data_change_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    data_source_id = Column(String(36), nullable=True)
    change_type = Column(String(30), nullable=False)  # row_added / row_modified / row_deleted
    table_name = Column(String(255), nullable=False)
    row_identifier = Column(String(500), nullable=True)  # PK or row hash
    column_name = Column(String(255), nullable=True)
    old_value = Column(Text, nullable=True)
    new_value = Column(Text, nullable=True)
    detected_at = Column(DateTime(timezone=True), server_default=func.now())
    acknowledged = Column(Boolean, default=False)


# ---------------------------------------------------------------------------
# Sanctity Engine
# ---------------------------------------------------------------------------


class SanctityIssue(Base):
    """Data integrity issues detected by the sanctity engine."""

    __tablename__ = "sanctity_issues"

    id = Column(Integer, primary_key=True, autoincrement=True)
    table_name = Column(String(255), nullable=False)
    column_name = Column(String(255), nullable=True)
    row_identifier = Column(String(500), nullable=True)
    issue_type = Column(String(50), nullable=False)
    # impossible_value / statistical_outlier / null_spike /
    # cross_source_conflict / unauthorized_change / future_date
    severity = Column(String(20), nullable=False, default="warning")  # critical / warning / info
    description = Column(Text, nullable=False)
    current_value = Column(Text, nullable=True)
    expected_range = Column(String(255), nullable=True)  # "0-100" or "0-500 kWh"
    conflicting_source = Column(String(255), nullable=True)
    conflicting_value = Column(Text, nullable=True)
    detected_at = Column(DateTime(timezone=True), server_default=func.now())
    resolved = Column(Boolean, default=False)
    resolved_by = Column(String(255), nullable=True)  # "user:krishna" or "auto:re-sync"
    resolution_note = Column(Text, nullable=True)


# ---------------------------------------------------------------------------
# Alert System
# ---------------------------------------------------------------------------


class AlertRule(Base):
    """A deployed alert rule parsed from natural language."""

    __tablename__ = "alert_rules"

    id = Column(String(36), primary_key=True, default=_uuid)
    name = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)  # original natural language input
    rule_config = Column(JSON, nullable=False)
    # {table, column, condition, threshold, ...}
    rule_type = Column(String(30), nullable=False)
    # threshold / trend / absence / pattern / cross_source / composite
    check_trigger = Column(String(30), nullable=False, default="on_data_change")
    # on_data_change / scheduled
    schedule_cron = Column(String(100), nullable=True)
    notification_channel = Column(String(30), nullable=False, default="feed")
    # telegram / feed
    notification_config = Column(JSON, nullable=True)  # {chat_id, group_id, ...}
    message_template = Column(Text, nullable=True)
    active = Column(Boolean, default=True)
    paused_until = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_triggered_at = Column(DateTime(timezone=True), nullable=True)
    trigger_count = Column(Integer, default=0)
    session_id = Column(String(64), nullable=True)


class AlertEvent(Base):
    """A single alert trigger event."""

    __tablename__ = "alert_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_rule_id = Column(String(36), nullable=False)
    triggered_at = Column(DateTime(timezone=True), server_default=func.now())
    trigger_value = Column(Text, nullable=True)
    threshold_value = Column(Text, nullable=True)
    context = Column(JSON, nullable=True)  # snapshot of relevant data
    notification_sent = Column(Boolean, default=False)
    notification_error = Column(Text, nullable=True)


# ---------------------------------------------------------------------------
# Pattern Memory
# ---------------------------------------------------------------------------


class Pattern(Base):
    """A learned data pattern (e.g., pre-breakdown SCADA signature)."""

    __tablename__ = "patterns"

    id = Column(String(36), primary_key=True, default=_uuid)
    name = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)
    source_tables = Column(JSON, nullable=True)  # ["scada_readings"]
    conditions = Column(JSON, nullable=True)
    # [{column, behavior, magnitude, ...}]
    time_window_minutes = Column(Integer, default=15)
    similarity_threshold = Column(Float, default=0.75)
    historical_occurrences = Column(JSON, nullable=True)
    # [{start, end, outcome}]
    confidence = Column(Float, default=0.5)
    created_by = Column(String(30), nullable=False, default="user")  # user / auto_detected
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_matched_at = Column(DateTime(timezone=True), nullable=True)
    match_count = Column(Integer, default=0)
    false_positive_count = Column(Integer, default=0)
    active = Column(Boolean, default=True)


class PatternMatch(Base):
    """A recorded pattern match event."""

    __tablename__ = "pattern_matches"

    id = Column(Integer, primary_key=True, autoincrement=True)
    pattern_id = Column(String(36), nullable=False)
    matched_at = Column(DateTime(timezone=True), server_default=func.now())
    similarity_score = Column(Float, default=0.0)
    data_snapshot = Column(JSON, nullable=True)  # actual values that matched
    outcome = Column(String(100), nullable=True)
    # confirmed_breakdown / false_positive / null
    alert_sent = Column(Boolean, default=False)


# ---------------------------------------------------------------------------
# Company Onboarding
# ---------------------------------------------------------------------------


class CompanyProfile(Base):
    """Structured company profile from onboarding."""

    __tablename__ = "company_profiles"

    id = Column(String(36), primary_key=True, default=_uuid)
    name = Column(String(500), nullable=False)
    industry = Column(String(100), nullable=True)
    products = Column(JSON, nullable=True)  # ["TMT bars", "billets"]
    departments = Column(JSON, nullable=True)  # [{name, head, contact}]
    process_flow = Column(Text, nullable=True)
    systems = Column(JSON, nullable=True)  # [{name, type, description}]
    known_relationships = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class MetricThreshold(Base):
    """Acceptable ranges for key metrics."""

    __tablename__ = "metric_thresholds"

    id = Column(Integer, primary_key=True, autoincrement=True)
    company_id = Column(String(36), nullable=True)
    metric_name = Column(String(255), nullable=False)
    table_name = Column(String(255), nullable=True)
    column_name = Column(String(255), nullable=True)
    unit = Column(String(50), nullable=True)  # "kWh/ton", "degrees C", etc.
    normal_min = Column(Float, nullable=True)
    normal_max = Column(Float, nullable=True)
    warning_min = Column(Float, nullable=True)
    warning_max = Column(Float, nullable=True)
    critical_min = Column(Float, nullable=True)
    critical_max = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


# ---------------------------------------------------------------------------
# Format Detection & Reconciliation
# ---------------------------------------------------------------------------


class FormatFingerprint(Base):
    """Known format fingerprints for recurring file/sheet ingestion."""

    __tablename__ = "format_fingerprints"

    id = Column(Integer, primary_key=True, autoincrement=True)
    fingerprint_hash = Column(String(64), nullable=False, unique=True)
    table_name = Column(String(255), nullable=False)
    column_mapping = Column(JSON, nullable=True)  # {"source_col": "target_col", ...}
    source_variations = Column(JSON, nullable=True)  # list of column name sets
    match_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class SourceMapping(Base):
    """Maps two data sources that contain the same data in different formats."""

    __tablename__ = "source_mappings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_a_table = Column(String(255), nullable=False)
    source_b_table = Column(String(255), nullable=False)
    column_mappings = Column(JSON, nullable=True)
    # [{"a": "HEAT_NO", "b": "Heat Number", "canonical": "heat_number"}]
    entity_type = Column(String(100), nullable=True)  # "production_data", "power_readings"
    authoritative_source = Column(String(255), nullable=True)
    confirmed_by_user = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


# ---------------------------------------------------------------------------
# Structured Process Map & I/O Definitions
# ---------------------------------------------------------------------------


class ProcessStep(Base):
    """A single step in the company's structured process map."""

    __tablename__ = "process_steps"

    id = Column(Integer, primary_key=True, autoincrement=True)
    company_id = Column(String(36), nullable=True)
    step_order = Column(Integer, nullable=False, default=0)
    process_name = Column(String(255), nullable=False)
    inputs = Column(Text, nullable=True)        # comma-separated input names
    outputs = Column(Text, nullable=True)       # comma-separated output names
    key_metric = Column(String(255), nullable=True)
    target_range = Column(String(255), nullable=True)  # e.g., "85-95%"
    linked_table = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class ProcessIO(Base):
    """Structured input/output definition for the manufacturing process."""

    __tablename__ = "process_ios"

    id = Column(Integer, primary_key=True, autoincrement=True)
    company_id = Column(String(36), nullable=True)
    io_type = Column(String(10), nullable=False)  # "input" or "output"
    name = Column(String(255), nullable=False)
    source_or_destination = Column(String(255), nullable=True)
    unit = Column(String(50), nullable=True)
    typical_range = Column(String(100), nullable=True)  # e.g., "3-5 MT per heat"
    linked_table = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
