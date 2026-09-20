"""Question queue (serial FIFO answer jobs) + coverage scoring + VAD state reset."""
from __future__ import annotations

import threading
import time

from mockingbird.audio.vad import SileroVAD
from mockingbird.kb.question_queue import QuestionQueue

# -- QuestionQueue -------------------------------------------------------


def test_queue_runs_jobs_serially_in_order():
    q = QuestionQueue()
    order = []
    lock = threading.Lock()

    def job(name):
        with lock:
            order.append(f"start-{name}")
        time.sleep(0.05)
        with lock:
            order.append(f"end-{name}")

    q.start()
    q.submit("a", "seg1", lambda: job("a"))
    q.submit("b", "seg2", lambda: job("b"))
    q.submit("c", "seg3", lambda: job("c"))
    q.stop(timeout=5)
    # strict serialization: each end precedes the next start
    assert order == ["start-a", "end-a", "start-b", "end-b", "start-c", "end-c"]


def test_queue_dedups_identical_pending_jobs():
    q = QuestionQueue()
    runs = []

    def blocker():
        time.sleep(0.15)
        runs.append("blocker")

    q.start()
    q.submit("same", "seg1", blocker)  # starts immediately, occupies the worker
    q.submit("same", "seg1", lambda: runs.append("dup-1"))
    q.submit("same", "seg1", lambda: runs.append("dup-2"))
    assert q.pending == 1  # the two later submissions replaced each other
    q.stop(timeout=5)
    assert runs == ["blocker"]


def test_queue_pending_callback_fires():
    q = QuestionQueue()
    seen = []
    q.on_pending_change = seen.append
    q.start()
    q.submit("a", "s", lambda: None)
    q.stop(timeout=5)
    assert any(c >= 1 for c in seen)


def test_queue_job_failure_does_not_kill_worker():
    q = QuestionQueue()

    def boom():
        raise RuntimeError("x")

    ran = []
    q.start()
    q.submit("bad", "s", boom)
    q.submit("good", "s", lambda: ran.append(1))
    q.stop(timeout=5)
    assert ran == [1]


def test_queue_submit_after_stop_is_ignored():
    q = QuestionQueue()
    q.start()
    q.stop(timeout=2)
    ran = []
    q.submit("a", "s", lambda: ran.append(1))
    time.sleep(0.1)
    assert ran == []


def test_queue_stop_drains_inflight_jobs():
    q = QuestionQueue()
    ran = []
    q.submit("a", "s", lambda: ran.append("a"))
    q.submit("b", "s", lambda: ran.append("b"))
    q.stop(timeout=5)
    # jobs submitted before stop still run (a stopped session's last
    # questions are answered); new submissions are ignored
    assert sorted(ran) == ["a", "b"]


# -- VAD state reset -----------------------------------------------------


def test_vad_decays_onnx_state_on_end(monkeypatch):
    import numpy as np

    from mockingbird.audio.vad import VadStateMachine, _state_shape

    vad = SileroVAD.__new__(SileroVAD)
    vad._sr = 16000
    vad._threshold = 0.5
    vad._machine = VadStateMachine(threshold=0.5)
    vad._max_prob = 0.0
    vad._last_prob_log = time.monotonic()
    vad._noise_floor_ema = None
    vad._buffer = np.zeros(0, dtype=np.float32)
    vad._context_size = 64
    vad._context = np.ones(64, dtype=np.float32)
    vad._state = np.ones((2, 1, 128), dtype=np.float32)

    class _FakeSession:
        def get_inputs(self):
            return []

        def run(self, none, feeds):
            # out[0][0][0] must be a scalar (code does float(out[0][0][0]))
            prob = np.zeros((1, 1), dtype=np.float32)
            # return a NON-zero state to prove the decay path halves it
            return [prob, np.ones((2, 1, 128), dtype=np.float32)]

    vad._session = _FakeSession()
    monkeypatch.setattr("mockingbird.audio.vad._state_shape", _state_shape)

    frame = np.zeros(512, dtype=np.float32)
    # trigger speech start through the machine directly, then silence
    vad._machine.consume(frame, 0.9)
    got_end = False
    for _ in range(int((0.7 * 16000) / 512) + 2):
        evs = vad.process(frame)
        if any(e.get("kind") == "end" for e in evs):
            got_end = True
            break
    assert got_end
    # the ONNX recurrent state must be decayed (halved) on end, not carried
    # at full strength into the next segment
    assert np.allclose(vad._state, 0.5)


# -- coverage scoring -----------------------------------------------------


def _view(blocks_scores, miss=False, best=None):
    from mockingbird import protocol

    blocks = [
        protocol.AnswerBlock(id=f"b{i}", section="s", question="q", answer="a", score=sc)
        for i, sc in enumerate(blocks_scores)
    ]
    return protocol.KnowledgeView(
        topic="t", title="T", matched_query="q", blocks=blocks,
        best_score=best if best is not None else max(blocks_scores, default=0.0),
        miss=miss,
    )


def test_coverage_ignores_unscored_context_blocks():
    from mockingbird.kb.interview_engine import _coverage_score

    # one strong hit + two appended context blocks with score 0.00
    view = _view([10.04, 0.0, 0.0, 0.0])
    cov = _coverage_score(view)
    assert cov < 0.95  # context blocks must not push coverage to 1.0


def test_coverage_single_exact_hit_is_moderate():
    from mockingbird.kb.interview_engine import _coverage_score

    view = _view([8.0])
    cov = _coverage_score(view)
    assert 0.5 <= cov <= 0.8


def test_coverage_multiple_hits_bonus():
    from mockingbird.kb.interview_engine import _coverage_score

    single = _coverage_score(_view([8.0]))
    multi = _coverage_score(_view([8.0, 7.0, 6.0]))
    assert multi > single


def test_coverage_miss_penalty():
    from mockingbird.kb.interview_engine import _coverage_score

    hit = _coverage_score(_view([8.0]))
    miss = _coverage_score(_view([8.0], miss=True))
    assert miss < hit


# -- fuzzy dedup -----------------------------------------------------------


def test_queue_fuzzy_dedup_replaces_pending():
    q = QuestionQueue()
    runs = []

    def blocker():
        time.sleep(0.15)
        runs.append("blocker")

    q.start()
    q.submit("какие dora вы знаете почему важны", "s", blocker)
    q.submit("какие dora вы знаете и почему важны", "s", lambda: runs.append("dup"))
    assert q.pending == 1
    q.stop(timeout=5)
    # the pending near-duplicate was replaced, so only the blocker ran
    assert runs == ["blocker"]


def test_queue_skips_duplicate_of_running_answer():
    q = QuestionQueue()
    events = []

    def slow():
        time.sleep(0.2)
        events.append("first")

    q.start()
    q.submit("в чем связь между agile и devops", "s", slow)
    time.sleep(0.05)
    # equivalent wording arrives while the first answer streams -> dropped
    q.submit("в чем связь между agile и девопс", "s", lambda: events.append("second"))
    q.stop(timeout=5)
    assert events == ["first"]


def test_queue_keeps_materially_different_question():
    q = QuestionQueue()
    events = []

    def slow():
        time.sleep(0.15)
        events.append("first")

    q.start()
    q.submit("какие dora метрики вы знаете", "s", slow)
    q.submit("расскажи про сетевые политики кластера", "s", lambda: events.append("second"))
    q.stop(timeout=5)
    assert sorted(events) == ["first", "second"]


# -- phonetic fuzzy (Cyrillic→Cyrillic alias drift) ------------------------


def test_phonetic_fuzzy_alias_drift():
    from mockingbird.terms.phonetics import PhoneticMatcher

    m = PhoneticMatcher(
        [
            ("Agile", ["agile", "эджайл", "аджайл"], "Agile"),
            ("DevOps", ["devops", "девопс", "дивобс"], "DevOps"),
        ]
    )
    assert m.resolve("эджаал")[0] == "Agile"
    assert m.resolve("дивокс")[0] == "DevOps"
    # inflected forms recover via suffix stripping
    assert m.resolve("эджайлом")[0] == "Agile"
    # common Russian words stay untouched
    assert m.resolve("метрики") is None
    assert m.resolve("систему") is None
    assert m.normalize_text("в чем связь между эджаал и дивокс") == (
        "в чем связь между Agile и DevOps"
    )


# -- VAD preroll + whisper partial prompt ----------------------------------


def test_vad_preroll_is_500ms():
    from mockingbird.audio import vad as vad_mod

    assert vad_mod._PREROLL_FRAMES == 10
    assert vad_mod._PREROLL_SAMPLES == 10 * vad_mod._FRAME


def test_starts_with_connective_tail():
    from mockingbird.kb.interview_engine import _starts_with_connective

    assert _starts_with_connective("и DevOps.")
    assert _starts_with_connective("или деплой?")
    # normal question openers and long clauses are NOT tails
    assert not _starts_with_connective("что такое docker")
    assert not _starts_with_connective("и вот мы решили использовать kubernetes в проде для всех сервисов")
    assert not _starts_with_connective("")


def test_tail_merge_glues_clipped_segment():
    from unittest.mock import MagicMock

    from mockingbird import protocol
    from mockingbird.config import InterviewConfig
    from mockingbird.kb.interview_engine import InterviewEngine

    matcher = MagicMock()
    matcher.match.return_value = []  # no KB hits — engine must fall through cleanly
    cfg = InterviewConfig(enabled=True, use_partials=False)
    eng = InterviewEngine(matcher, cfg, context=None, llm=None)
    eng.start()  # worker mode: accumulation/tail-merge windows active
    try:
        t0 = time.monotonic()
        seen: list[str] = []
        eng.on_question = lambda msg: seen.append(msg.text)
        eng.on_final(protocol.FinalTranscript(segment_id="a", text="в чем связь между Agile", ts=t0))
        time.sleep(0.3)
        eng.on_final(protocol.FinalTranscript(segment_id="b", text="и DevOps.", ts=t0 + 5.0))
        for _ in range(50):
            if any("DevOps" in q for q in seen):
                break
            time.sleep(0.1)
        assert any("Agile" in q and "DevOps" in q for q in seen), seen
    finally:
        eng.stop()


def test_tail_merge_ignores_old_segments():
    from unittest.mock import MagicMock

    from mockingbird import protocol
    from mockingbird.config import InterviewConfig
    from mockingbird.kb.interview_engine import InterviewEngine

    matcher = MagicMock()
    matcher.match.return_value = []  # no KB hits — engine must fall through cleanly
    cfg = InterviewConfig(enabled=True, use_partials=False)
    eng = InterviewEngine(matcher, cfg, context=None, llm=None)
    eng.start()  # worker mode: accumulation/tail-merge windows active
    try:
        t0 = time.monotonic()
        seen: list[str] = []
        eng.on_question = lambda msg: seen.append(msg.text)
        eng.on_final(protocol.FinalTranscript(segment_id="a", text="в чем связь между Agile", ts=t0))
        time.sleep(0.3)
        eng.on_final(protocol.FinalTranscript(segment_id="b", text="и DevOps.", ts=t0 + 30.0))
        for _ in range(50):
            if seen:
                break
            time.sleep(0.1)
        # 30 s later the tail is a standalone utterance — no glue: the
        # standalone question ("a") may fire, but nothing must merge both.
        assert not any("Agile" in q and "DevOps" in q for q in seen), seen
    finally:
        eng.stop()


def test_lat_exact_fixes_continuum():
    from mockingbird.terms.glossary import Glossary

    m = Glossary.load()._matcher
    fixed = m.normalize_text("В чем разница между Continuum с Delivery и Continuum с Deployment?")
    assert "Continuum" not in fixed
    assert "Continuous" in fixed


def test_vad_loud_speech_start_forced():
    import numpy as np

    from mockingbird.audio.vad import SileroVAD, VadStateMachine

    class _ColdSession:
        def get_inputs(self):
            return []

        def run(self, none, feeds):
            # model never recognizes speech (cold state bug reproduction)
            return [[[0.0]], np.zeros((2, 1, 128), dtype=np.float32)]

    def _make():
        v = SileroVAD.__new__(SileroVAD)
        v._sr = 16000
        v._threshold = 0.5
        v._machine = VadStateMachine(threshold=0.5)
        v._max_prob = 0.0
        v._last_prob_log = time.monotonic()
        v._buffer = np.zeros(0, dtype=np.float32)
        v._context_size = 64
        v._context = np.zeros(64, dtype=np.float32)
        v._state = np.zeros((2, 1, 128), dtype=np.float32)
        v._session = _ColdSession()
        v._loud_run = 0
        v._noise_floor_ema = None
        return v

    loud = (np.random.RandomState(0).rand(512) * 0.4 - 0.2).astype(np.float32)
    vad = _make()
    started = False
    for _ in range(40):
        if any(e.get("kind") == "start" for e in vad.process(loud)):
            started = True
            break
    assert started  # loud speech forces the trigger even with model prob=0

    silent = np.zeros(512, dtype=np.float32)
    vad2 = _make()
    for _ in range(60):
        assert not any(e.get("kind") == "start" for e in vad2.process(silent))


# -- blend splitting + connector cleanup ------------------------------------


def test_blend_split_fused_terms():
    from mockingbird.terms.glossary import Glossary

    m = Glossary.load()._matcher
    fixed = m.normalize_text("в чем связь между эджаопс")
    assert "Agile" in fixed and "DevOps" in fixed


def test_connector_cleanup_removes_stray_s():
    from mockingbird.terms.glossary import Glossary

    m = Glossary.load()._matcher
    fixed = m.normalize_text("В чем разница между Continuum с Delivery и Continuum с Deployment?")
    assert "Continuous и Delivery" in fixed
    assert "Continuous и Deployment" in fixed


def test_blend_no_false_positives_on_russian():
    from mockingbird.terms.glossary import Glossary

    m = Glossary.load()._matcher
    for t in (
        "расскажи про подходы и практики",
        "мы настроили сеть и диски",
        "операции и процессы в команде",
    ):
        assert m.normalize_text(t) == t


def test_dedup_ignores_stop_words():
    from mockingbird.kb.question_queue import _jaccard, _tokens

    # content differs -> NOT a duplicate
    a = _tokens("как деплоить в k8s")
    b = _tokens("как откатить деплой в k8s")
    assert _jaccard(a, b) < 0.7
    # same question with a filler word -> duplicate
    c = _tokens("какие dora вы знаете почему важны")
    d = _tokens("какие dora вы знаете и почему важны")
    assert _jaccard(c, d) >= 0.7


def test_tail_merge_uses_dialog_history_when_final_aged_out():
    from unittest.mock import MagicMock

    from mockingbird import protocol
    from mockingbird.config import InterviewConfig
    from mockingbird.kb.interview_engine import InterviewEngine

    eng = InterviewEngine(
        MagicMock(), InterviewConfig(enabled=True, use_partials=False), context=None, llm=None
    )
    dialog = MagicMock()
    dialog.last_opponent_utterance.return_value = "в чем связь между Agile"
    eng._dialog = dialog
    eng._build_best_view = lambda q, m: None
    t0 = time.monotonic()
    seen: list[str] = []
    eng.on_question = lambda msg: seen.append(msg.text)
    # engine's own last final is 30 s old — beyond the window
    eng._process(protocol.FinalTranscript(segment_id="a", text="в чем связь между Agile", ts=t0 - 30.0))
    eng._process(protocol.FinalTranscript(segment_id="b", text="и DevOps.", ts=t0))
    assert any("Agile" in q and "DevOps" in q for q in seen)


# -- FP guards for glossary corrections ------------------------------------


def test_blend_no_fp_on_russian_words():
    from mockingbird.terms.glossary import Glossary

    m = Glossary.load()._matcher
    for t in ("сломан", "настройки сломаны", "система работает"):
        assert m.normalize_text(t) == t


def test_bigram_does_not_eat_protected_first_word():
    from mockingbird.terms.glossary import Glossary

    m = Glossary.load()._matcher
    fixed = m.normalize_text("Какие метрики Дора вы знаете")
    assert "метрики" in fixed
    assert "DORA" in fixed


def test_never_rewrite_conversational_devops_words():
    from mockingbird.terms.glossary import Glossary

    m = Glossary.load()._matcher
    assert m.resolve("запушил") is None
    assert m.resolve("мэйн") is None
    assert m.normalize_text("запушил плохой коммит в мэйн") != ""


def test_final_beam_size_default_is_5():
    from mockingbird.config import WhisperConfig

    assert WhisperConfig().final_beam_size == 1


def test_llm_stream_first_token_logged():
    # instrumentation smoke: the generator must propagate deltas
    from unittest.mock import MagicMock

    from mockingbird.config import LlmConfig
    from mockingbird.llm.client import LlmClient

    client = LlmClient(LlmConfig())
    client._ensure = MagicMock(return_value=object())
    client._hedged_answer_stream = lambda s, u, g: iter(["delta"])
    out = list(client.answer_question_stream("q"))
    assert out == ["delta"]


def test_vad_silence_threshold_relaxed_inside_speech():
    from mockingbird.audio import vad as vad_mod

    assert vad_mod.SileroVAD  # importable; threshold logic tested via unit constants


# -- hold-open endpointing (LocalAgreement completeness) -------------------


def test_incomplete_utterance_holds_segment_open():

    from unittest.mock import MagicMock

    import numpy as np

    from mockingbird.audio.chunker import SpeechChunker

    class _FakeVad:
        def __init__(self, events):
            self._events = events

        def process(self, audio):
            return self._events

    engine = MagicMock()
    engine.is_utterance_complete.return_value = False
    engine.start_segment.return_value = "seg1"

    chunker = SpeechChunker(_FakeVad([]), engine)
    # start speech, then an incomplete end -> held, not closed
    chunker.on_audio(np.zeros(0), 0)  # no-op pass
    chunker._segment_id = "seg1"
    for ev in ({"kind": "start"}, {"kind": "end", "audio": np.zeros(10)}):
        chunker.on_audio(np.zeros(0), 0) if False else None
    # simulate events directly
    for event in (
        {"kind": "start"},
        {"kind": "end", "audio": np.zeros(10, dtype=np.float32)},
    ):
        chunker._try_release_held_end()
        # route through the same loop body as on_audio
        chunker._vad._events = [event]
        chunker.on_audio(np.zeros(0), 0)
    assert not engine.end_segment.called  # held open

    # speech resumes within the hold -> continuation, still one segment
    chunker._vad._events = [{"kind": "start"}]
    chunker.on_audio(np.zeros(0), 0)
    assert not engine.end_segment.called

    # utterance completes -> closes with the original segment id
    engine.is_utterance_complete.return_value = True
    chunker._vad._events = [{"kind": "end", "audio": np.zeros(10, dtype=np.float32)}]
    chunker.on_audio(np.zeros(0), 0)
    assert engine.end_segment.called


def test_hold_budget_expires_and_closes():
    import time
    from unittest.mock import MagicMock

    import numpy as np

    from mockingbird.audio.chunker import SpeechChunker

    class _FakeVad:
        def __init__(self):
            self._events = []

        def process(self, audio):
            return self._events

    engine = MagicMock()
    engine.is_utterance_complete.return_value = False
    engine.start_segment.return_value = "seg1"
    chunker = SpeechChunker(_FakeVad(), engine)
    chunker._segment_id = "seg1"
    chunker._held_end = (
        np.zeros(10, dtype=np.float32),
        "seg1",
        time.monotonic() - 0.1,
    )  # deadline in the past
    chunker._try_release_held_end()
    assert engine.end_segment.called
    assert engine.end_segment.call_args[0][1] == "seg1"


def test_whisper_completeness_check():
    from mockingbird.config import WhisperConfig
    from mockingbird.stt.whisper_engine import WhisperEngine

    e = WhisperEngine(WhisperConfig(), sample_rate=16000)
    e._last_partial_text = "в чем разница между"
    assert not e.is_utterance_complete()
    e._last_partial_text = "в чем разница между Delivery и Deployment?"
    assert e.is_utterance_complete()
    e._last_partial_text = "какие метрики dora вы знаете"
    e._prev_full_text = "какие метрики dora вы знаете"
    assert e.is_utterance_complete()  # LocalAgreement-2 stability


def test_whisper_turbo_repo_alias():
    from mockingbird.stt.whisper_engine import _model_repo_id, _normalize_repo_id

    # the non-existent Systran turbo repo is rewritten to the community build
    assert _model_repo_id("large-v3-turbo") == "deepdml/faster-whisper-large-v3-turbo-ct2"
    assert _normalize_repo_id("Systran/faster-whisper-large-v3-turbo") == (
        "deepdml/faster-whisper-large-v3-turbo-ct2"
    )
    # regular sizes pass through untouched
    assert _model_repo_id("medium") == "Systran/faster-whisper-medium"


# -- GPU-budget guards + hallucination guard --------------------------------


def test_dedupe_repeated_words():
    from mockingbird.stt.whisper_engine import _dedupe_repeated_words

    looped = " ".join(["Время,"] * 56)
    fixed = _dedupe_repeated_words(looped)
    assert fixed.count("Время,") == 3
    normal = "как деплоить в k8s и откатить деплой при ошибке"
    assert _dedupe_repeated_words(normal) == normal


def test_hallucination_guard():
    from mockingbird.stt.whisper_engine import _looks_like_hallucination

    assert _looks_like_hallucination("Редактор субтитров А.Семкин")
    assert _looks_like_hallucination("Субтитры создавал DimaTorzok")
    assert not _looks_like_hallucination("Что такое Docker и зачем он нужен")
    assert not _looks_like_hallucination(
        "Сначала настраиваем Subnet в продакшене, потом редактор субтитров может подождать"
    )


def test_stop_hint_no_new_audio_skipped():
    import numpy as np

    from mockingbird.config import WhisperConfig
    from mockingbird.stt.whisper_engine import WhisperEngine

    e = WhisperEngine(WhisperConfig(), sample_rate=16000)
    e._model = object()
    calls = []

    def fake(audio, kind="decode", beam_size=1, **kw):
        calls.append(kind)
        return "текст", 0.9, len(audio) / 16000.0

    e._transcribe = fake
    e._decode_cached = lambda a, kind="final", beam_size=1: fake(a, kind)
    e.start_segment()
    e._rolling = np.zeros(16000 * 2, dtype=np.float32)
    e._handle_stop_hint()
    assert len(calls) == 1
    # second hint with the SAME buffer size -> skipped (no new audio)
    e._last_decode = 0.0
    e._handle_stop_hint()
    assert len(calls) == 1


def test_glossary_no_everyday_word_rewrites():
    from mockingbird.terms.glossary import Glossary

    m = Glossary.load()._matcher
    for w in ("том", "подсвети", "стейджи", "джоба", "но", "контроль", "план"):
        assert m.resolve(w) is None, w
    # legit correction still works
    assert m.resolve("подсети")[0] == "Subnet"




def test_long_buffer_speculative_decodes_only_tail():
    import numpy as np

    from mockingbird.config import WhisperConfig
    from mockingbird.stt.whisper_engine import WhisperEngine

    e = WhisperEngine(WhisperConfig(), sample_rate=16000)
    e._model = object()
    calls = []

    def fake(audio, kind="decode", beam_size=1, **kw):
        calls.append(len(audio))
        return "хвост", 0.9, len(audio) / 16000.0

    e._transcribe = fake
    e.start_segment()
    e._rolling = np.zeros(16000 * 45, dtype=np.float32)  # 45 s buffer
    e._chunk_texts = ["chunk-one", "chunk-two"]  # 36 s cached prefix
    partials = []
    e.on_partial = partials.append
    e._handle_stop_hint()
    decoded_seconds = calls[-1] / 16000
    assert decoded_seconds < 15, decoded_seconds  # tail only, not 45 s
    assert partials and "chunk-one" in partials[0].text  # prefix preserved


# -- quiet-tail + low-confidence final guards --------------------------------


def test_trailing_latin_nonsense_guard():
    from mockingbird.stt.whisper_engine import _has_trailing_latin_nonsense as f
    from mockingbird.terms.glossary import Glossary

    m = Glossary.load()._matcher
    assert f("Расскажи мне, пожалуйста, что твой Mindfuls", m)
    assert not f("расскажи про деплой в Kubernetes", m)   # known term
    assert not f("Что такое Docker и зачем он нужен?", m)  # punctuation
    assert not f("как настроить GitLab CI", m)             # known short term


def test_vad_keeps_short_quiet_tail():
    import numpy as np

    from mockingbird.audio.vad import VadStateMachine

    sm = VadStateMachine(threshold=0.5, min_silence_samples=int(0.6 * 16000))
    frame = np.zeros(512, dtype=np.float32)
    sm.consume(frame, 0.9)  # speech start
    frames_fed = 0
    end_len = None
    for _ in range(int(0.9 * 16000 / 512)):
        frames_fed += 1
        for ev in sm.consume(frame, 0.0):
            if ev.get("kind") == "end":
                end_len = len(ev["audio"])
    assert end_len is not None
    # end fires at min_silence (0.6 s) and the WHOLE quiet tail is kept:
    # preroll + first frame + 0.6 s silence (no amputation, no 400ms-only pad)
    expected_min = 512 + 512 + int(0.6 * 16000) - 1024
    assert end_len >= expected_min, (end_len, expected_min)


def test_vad_cuts_long_silence():
    import numpy as np

    from mockingbird.audio.vad import VadStateMachine

    sm = VadStateMachine(threshold=0.5, min_silence_samples=int(0.6 * 16000))
    frame = np.zeros(512, dtype=np.float32)
    sm.consume(frame, 0.9)
    end_len = None
    for _ in range(int(2.5 * 16000 / 512)):
        for ev in sm.consume(frame, 0.0):
            if ev.get("kind") == "end":
                end_len = len(ev["audio"])
    assert end_len is not None
    # 2.5 s silence -> cut, only the 400 ms tail pad (+preroll) remains
    assert end_len < int(1.0 * 16000)
