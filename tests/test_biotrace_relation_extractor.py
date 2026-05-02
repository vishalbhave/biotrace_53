import pytest
import sqlite3
import json
from unittest.mock import MagicMock
from biotrace_relation_extractor import (
    RelationTriple,
    VALID_RELATIONS,
    _ensure_relations_table,
    _persist_relations,
    extract_relations
)

def test_relation_triple_init_normalization():
    """Test that relation names are stripped and converted to uppercase."""
    triple = RelationTriple(
        subject="Species A",
        relation="  found_at  ",
        object="Location X",
        confidence=1.0
    )
    assert triple.relation == "FOUND_AT"

def test_relation_triple_near_match():
    """Test that near-matches are correctly normalized to valid relations."""
    # "FOUND_AT" is in "IS FOUND_AT HERE"
    triple = RelationTriple(
        subject="Species A",
        relation="IS FOUND_AT HERE",
        object="Location X",
        confidence=1.0
    )
    assert triple.relation == "FOUND_AT"

    # "INHABITS" is in "PREVIOUSLY INHABITS"
    triple2 = RelationTriple(
        subject="Species A",
        relation="PREVIOUSLY INHABITS",
        object="Habitat Y",
        confidence=0.8
    )
    assert triple2.relation == "INHABITS"

def test_relation_triple_confidence_clamping():
    """Test that confidence values are clamped between 0.0 and 1.0."""
    triple_high = RelationTriple(
        subject="Species A",
        relation="FOUND_AT",
        object="Location X",
        confidence=1.5
    )
    assert triple_high.confidence == 1.0

    triple_low = RelationTriple(
        subject="Species A",
        relation="FOUND_AT",
        object="Location X",
        confidence=-0.5
    )
    assert triple_low.confidence == 0.0

    triple_str = RelationTriple(
        subject="Species A",
        relation="FOUND_AT",
        object="Location X",
        confidence="0.7"
    )
    assert triple_str.confidence == 0.7

def test_relation_triple_invalid_relation():
    """Test that invalid relations without near-matches are kept as is (upper-stripped)."""
    triple = RelationTriple(
        subject="Species A",
        relation="UNKNOWN_REL",
        object="Object",
        confidence=1.0
    )
    assert triple.relation == "UNKNOWN_REL"

def test_ensure_relations_table():
    """Test that the relations table is created with the correct schema."""
    conn = sqlite3.connect(":memory:")
    _ensure_relations_table(conn)

    # Check if table exists
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='species_relations'")
    assert cursor.fetchone() is not None

    # Check columns
    cursor = conn.execute("PRAGMA table_info(species_relations)")
    columns = {row[1] for row in cursor.fetchall()}
    expected_columns = {
        "id", "subject_name", "relation_type", "object_value",
        "evidence_text", "source_citation", "confidence", "file_hash", "created_at"
    }
    assert expected_columns.issubset(columns)
    conn.close()

def test_persist_relations(tmp_path):
    """Test that relations are correctly persisted to the database."""
    db_file = tmp_path / "test_bio.db"
    db_path = str(db_file)

    triples = [
        RelationTriple("Species A", "FOUND_AT", "Location X", "Evidence 1", "Cite 1", 0.9),
        RelationTriple("Species B", "FEEDS_ON", "Species A", "Evidence 2", "Cite 2", 0.8)
    ]
    file_hash = "abc123hash"

    _persist_relations(db_path, triples, file_hash)

    conn = sqlite3.connect(db_path)
    cursor = conn.execute("SELECT subject_name, relation_type, object_value, file_hash FROM species_relations")
    rows = cursor.fetchall()
    assert len(rows) == 2
    assert rows[0] == ("Species A", "FOUND_AT", "Location X", file_hash)
    assert rows[1] == ("Species B", "FEEDS_ON", "Species A", file_hash)
    conn.close()

def test_extract_relations_valid(tmp_path):
    """Test extract_relations with a valid LLM response."""
    db_path = str(tmp_path / "test.db")

    mock_llm = MagicMock(return_value=json.dumps([
        {
            "subject": "Species A",
            "relation": "FOUND_AT",
            "object": "Location X",
            "evidence": "Observed in X.",
            "confidence": 0.9
        }
    ]))

    results = extract_relations(
        text="Sample text",
        known_species=["Species A"],
        source_citation="Cite 1",
        file_hash="hash1",
        llm_call_fn=mock_llm,
        meta_db_path=db_path
    )

    assert len(results) == 1
    assert results[0].subject == "Species A"
    assert results[0].relation == "FOUND_AT"

def test_extract_relations_with_think_and_markdown(tmp_path):
    """Test extract_relations with reasoning blocks and markdown fences."""
    db_path = str(tmp_path / "test.db")

    # LLM response with <think> block and markdown
    raw_response = """
    <think>
    I should extract the relation between Species A and its habitat.
    </think>

    ```json
    [
        {
            "subject": "Species A",
            "relation": "INHABITS",
            "object": "Coral Reef",
            "evidence": "Lives in reefs.",
            "confidence": 0.8
        }
    ]
    ```
    """
    mock_llm = MagicMock(return_value=raw_response)

    results = extract_relations(
        text="Sample text",
        known_species=["Species A"],
        source_citation="Cite 1",
        file_hash="hash1",
        llm_call_fn=mock_llm,
        meta_db_path=db_path
    )

    assert len(results) == 1
    assert results[0].relation == "INHABITS"

def test_extract_relations_low_confidence(tmp_path):
    """Test that low-confidence results are filtered out."""
    db_path = str(tmp_path / "test.db")

    mock_llm = MagicMock(return_value=json.dumps([
        {
            "subject": "Species A",
            "relation": "FOUND_AT",
            "object": "Location X",
            "evidence": "Maybe here?",
            "confidence": 0.4
        }
    ]))

    results = extract_relations(
        text="Sample text",
        known_species=["Species A"],
        source_citation="Cite 1",
        file_hash="hash1",
        llm_call_fn=mock_llm,
        meta_db_path=db_path
    )

    assert len(results) == 0

def test_extract_relations_empty_inputs(tmp_path):
    """Test extract_relations with empty species list or empty text."""
    db_path = str(tmp_path / "test.db")
    mock_llm = MagicMock()

    # Empty species list
    results = extract_relations(
        text="Some text",
        known_species=[],
        source_citation="Cite 1",
        file_hash="hash1",
        llm_call_fn=mock_llm,
        meta_db_path=db_path
    )
    assert results == []
    mock_llm.assert_not_called()

    # Empty text
    results = extract_relations(
        text="  ",
        known_species=["Species A"],
        source_citation="Cite 1",
        file_hash="hash1",
        llm_call_fn=mock_llm,
        meta_db_path=db_path
    )
    assert results == []
    mock_llm.assert_not_called()
