from mockingbird import protocol


def test_protocol_version():
    assert protocol.PROTOCOL_VERSION == "mockbird-protocol/v1"
    msg = protocol.StatusMessage(state="idle")
    assert msg.version == protocol.PROTOCOL_VERSION


def test_final_transcript_roundtrip():
    msg = protocol.FinalTranscript(
        session_id="s1", segment_id="seg1", text="hello world",
        start=0.0, end=1.2, confidence=0.9,
    )
    parsed = protocol.FinalTranscript.model_validate_json(msg.model_dump_json())
    assert parsed.text == "hello world"
    assert parsed.segment_id == "seg1"
    assert parsed.confidence == 0.9


def test_partial_transcript_defaults():
    msg = protocol.PartialTranscript(segment_id="seg2", text="привет")
    assert msg.session_id == ""
    assert msg.type == protocol.MessageType.PARTIAL_TRANSCRIPT


def test_term_detected_defaults():
    t = protocol.TermDetected(
        term="API", explanation="x", source=protocol.TermSource.GLOSSARY
    )
    assert t.examples == []
    assert t.source == protocol.TermSource.GLOSSARY
    assert t.confidence is None


def test_session_control_roundtrip():
    msg = protocol.SessionControl(
        action=protocol.SessionAction.START,
        session_id="abc",
        payload={"sample_rate": 16000},
    )
    parsed = protocol.SessionControl.model_validate_json(msg.model_dump_json())
    assert parsed.action == protocol.SessionAction.START
    assert parsed.payload["sample_rate"] == 16000
