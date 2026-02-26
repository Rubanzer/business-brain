"""ORM models for metadata and business context storage."""

from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.orm import DeclarativeBase
from pgvector.sqlalchemy import Vector


class Base(DeclarativeBase):
    pass


class MetadataEntry(Base):
    """Schema metadata describing a database table and its columns."""

    __tablename__ = "metadata_entries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    table_name = Column(String(255), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    columns_metadata = Column(JSON, nullable=True)  # [{name, type, description}, ...]
    uploaded_by = Column(String(36), nullable=True)       # user_id of uploader
    uploaded_by_role = Column(String(20), nullable=True)   # role at upload time
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())


class BusinessContext(Base):
    """Natural-language business context stored with vector embeddings."""

    __tablename__ = "business_contexts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(3072), nullable=True)
    source = Column(String(255), nullable=True)
    version = Column(Integer, default=1)  # Version number for this content
    active = Column(Boolean, default=True)  # False = superseded by newer version
    superseded_at = Column(DateTime(timezone=True), nullable=True)  # When this version was replaced
    last_validated_at = Column(DateTime(timezone=True), nullable=True)  # For freshness scoring
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class ChatMessage(Base):
    """Conversation message for session-based chat memory."""

    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(64), nullable=False, index=True)
    role = Column(String(16), nullable=False)  # "user" | "assistant"
    content = Column(Text, nullable=False)
    metadata_ = Column("metadata", JSON, nullable=True)  # sql_result, analysis, etc.
    created_at = Column(DateTime(timezone=True), server_default=func.now())


# Import discovery and v3 models so Base.metadata picks them up for auto-create
import business_brain.db.discovery_models as _discovery_models  # noqa: E402, F401
import business_brain.db.v3_models as _v3_models  # noqa: E402, F401
