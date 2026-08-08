import time

from mockingbird import protocol
from mockingbird.config import InterviewConfig, TopicsConfig
from mockingbird.kb.detector import is_broad, is_question, last_question
from mockingbird.kb.index import KbIndex
from mockingbird.kb.interview_engine import InterviewEngine
from mockingbird.kb.loader import load_topics
from mockingbird.kb.matcher import KbMatcher
from mockingbird.kb.model import KbBlock, KbSection, KbTopic
from mockingbird.kb.predict import next_questions_from_view
from mockingbird.llm.client import parse_topics_json
from mockingbird.topics.engine import TopicEngine


def _matcher():
    return KbMatcher(KbIndex(load_topics()))


def _final(text, segment_id="seg1"):
    return protocol.FinalTranscript(segment_id=segment_id, text=text)


# -- loader ------------------------------------------------------------------


def test_kb_loads_all_topics():
    topics = load_topics()
    ids = {t.id for t in topics}
    assert {
        "linux", "networking", "docker", "kubernetes", "ci_cd", "iac",
        "monitoring", "git",
    } <= ids
    assert all(t.title for t in topics)
    kubernetes = next(t for t in topics if t.id == "kubernetes")
    assert kubernetes.sections
    assert kubernetes.all_blocks()


def test_glossary_related_questions():
    from mockingbird.terms.glossary import Glossary

    glossary = Glossary.load()
    docker = next(e for e in glossary.entries if e.term == "Docker")
    assert docker.related
    assert any("ENTRYPOINT" in r["question"] for r in docker.related)
    assert all(r["question"] and r["answer"] for r in docker.related)


def test_kb_has_nexus_helm_vmware_blocks():
    topics = {t.id: t for t in load_topics()}
    ci_cd = topics["ci_cd"]
    assert any("Nexus Repository" in b.question for b in ci_cd.all_blocks())
    k8s = topics["kubernetes"]
    assert any("Helm chart" in b.question for b in k8s.all_blocks())
    assert any("helm install" in b.question for b in k8s.all_blocks())
    cloud = topics["cloud"]
    assert any("VMware" in b.question for b in cloud.all_blocks())
    assert all(
        b.question and b.answer for t in topics.values() for b in t.all_blocks()
    )


# -- index / matcher ---------------------------------------------------------


def test_matcher_control_plane_tops():
    matcher = _matcher()
    res = matcher.match("в чём особенности control plane и для чего он нужен", limit=3)
    assert res
    top = res[0]
    assert top[1].id == "kubernetes"
    assert top[3].question.startswith("В чём особенности control plane")
    assert "control" in top[4] and "plane" in top[4]


def test_matcher_entrypoint_vs_cmd():
    matcher = _matcher()
    res = matcher.match("в чем отличие entrypoint от cmd", limit=2)
    assert res and res[0][1].id == "docker"
    assert "entrypoint" in res[0][4] and "cmd" in res[0][4]


def test_matcher_broad_k8s_returns_topic():
    matcher = _matcher()
    res = matcher.match("расскажи что знаешь про k8s", limit=5)
    assert res
    assert any(m[1].id == "kubernetes" for m in res[:5])


def test_matcher_keeps_subject_when_tail_is_only_function_words():
    matcher = _matcher()
    res = matcher.match("я про kubernetes что знаешь", limit=5)
    assert res
    assert any(m[1].id == "kubernetes" for m in res[:5])
    assert "kubernetes" in res[0][4]


def test_matcher_phrase_does_not_match_mid_word():
    matcher = _matcher()
    res = matcher.match("kubernetes", limit=3)
    assert res
    assert any(m[1].id == "kubernetes" for m in res[:5])
    assert all("linux" != t.id for _s, t, _sec, _b, _hl in res)


def test_matcher_no_match():
    matcher = _matcher()
    assert matcher.match("случайный набор слов ззззз", limit=3) == []


def test_matcher_nexus_query():
    matcher = _matcher()
    res = matcher.match("что такое nexus и зачем он нужен", limit=3)
    assert res
    assert res[0][1].id == "ci_cd"
    assert any("Nexus Repository" in r[3].question for r in res)


def test_matcher_vmware_query():
    matcher = _matcher()
    res = matcher.match("расскажи про vmware и виртуализацию", limit=3)
    assert res
    assert res[0][1].id == "cloud"
    assert any("VMware" in r[3].question for r in res)


def test_matcher_entrypoint_query_tops_specific_block():
    matcher = _matcher()
    res = matcher.match("entrypoint в докер что такое", limit=3)
    assert res
    assert res[0][3].question.startswith("В чём отличие ENTRYPOINT")
    assert "entrypoint" in res[0][4]


def test_matcher_inflected_form_resolves():
    matcher = _matcher()
    res = matcher.match("как работает entrypoint в докере", limit=3)
    assert res and res[0][1].id == "docker"


def test_matcher_phrase_alias_entry_point():
    matcher = _matcher()
    res = matcher.match("в чем отличие entry point от cmd", limit=3)
    assert res and res[0][3].question.startswith("В чём отличие ENTRYPOINT")


def test_matcher_noisy_segment_ignores_context_tail():
    matcher = _matcher()
    query = last_question("мы же обсуждали control plane и etcd в k8s а что такое entrypoint в докере")
    assert query == "что такое entrypoint в докере"
    res = matcher.match(query, limit=5)
    assert res
    assert res[0][1].id == "docker"


# -- detector ----------------------------------------------------------------


def test_is_question():
    assert is_question("расскажи что знаешь про k8s")
    assert is_question("в чём особенности control plane")
    assert is_question("что такое идемпотентность?")
    assert is_question("объясни разницу между merge и rebase")
    assert not is_question("начинаем следующую тему")
    assert not is_question("")
    assert not is_question("документ готов к ревью")


def test_is_broad():
    assert is_broad("расскажи что знаешь про docker")
    assert is_broad("опиши в целом как работает kubernetes")
    assert not is_broad("в чём отличие entrypoint от cmd")


def test_last_question_isolation():
    assert last_question("мы обсуждали etcd а теперь что такое entrypoint в докере") == "что такое entrypoint в докере"
    assert last_question("а что такое kubelet") == "что такое kubelet"
    assert last_question("в чём особенности control plane и для чего он нужен") == "в чём особенности control plane и для чего он нужен"
    assert last_question("как собрать образ docker. что такое entrypoint") == "что такое entrypoint"
    assert last_question("начинаем следующую тему") is None


def test_last_question_keeps_subject_when_tail_has_no_topic_word():
    assert last_question("я про kubernetes что знаешь") == "я про kubernetes что знаешь"
    assert last_question("расскажи что знаешь про docker") == "что знаешь про docker"
    assert last_question("что такое kubelet") == "что такое kubelet"


# -- interview engine --------------------------------------------------------


def _view_for(text):
    matcher = _matcher()
    engine = InterviewEngine(matcher, InterviewConfig())
    view = engine._build_view(text)
    return view


def test_engine_builds_view_for_concrete_question():
    view = _view_for("в чём особенности control plane и для чего он нужен")
    assert view is not None
    assert view.topic in ("kubernetes", "cloud")  # expanded KB
    assert view.blocks
    assert not view.miss
    assert view.blocks[0].highlight


def test_engine_broad_question_returns_full_topic():
    view = _view_for("расскажи что знаешь про k8s")
    assert view is not None
    assert view.topic in ("kubernetes", "cloud")  # expanded KB
    assert len(view.blocks) > 20
    assert not view.miss


def test_engine_returns_none_for_non_question():
    engine = InterviewEngine(_matcher(), InterviewConfig())
    out = []
    engine.on_answer = out.append
    engine._process(_final("документ готов к ревью"))
    assert out == []


def test_engine_emits_question_and_answer():
    engine = InterviewEngine(_matcher(), InterviewConfig())
    questions, answers = [], []
    engine.on_question = questions.append
    engine.on_answer = answers.append
    engine._process(_final("в чём отличие entrypoint от cmd"))
    assert len(questions) == 1
    assert questions[0].text == "в чём отличие entrypoint от cmd"
    assert len(answers) == 1
    assert answers[0].topic == "docker"


def test_engine_dedup_same_question():
    engine = InterviewEngine(_matcher(), InterviewConfig())
    answers = []
    engine.on_answer = answers.append
    engine._process(_final("в чём отличие entrypoint от cmd", "s1"))
    engine._process(_final("в чём отличие entrypoint от cmd", "s2"))
    assert len(answers) == 1


def test_engine_miss_falls_back_to_nearest_topic():
    engine = InterviewEngine(_matcher(), InterviewConfig(min_match_score=99.0))
    view = engine._build_view("что такое kubelet")
    assert view is not None
    assert view.miss
    assert view.topic in ("kubernetes", "cloud")  # expanded KB
    assert view.blocks


def test_nearest_topic_resolution():
    engine = InterviewEngine(_matcher(), InterviewConfig())
    assert engine._nearest_topic("поговорим про kubelet").id == "kubernetes"
    assert engine._nearest_topic("что такое нгинкс").id == "networking"
    assert engine._nearest_topic("zizzzq") is None


def test_engine_disabled():
    engine = InterviewEngine(_matcher(), InterviewConfig(enabled=False))
    out = []
    engine.on_answer = out.append
    engine._process(_final("в чём отличие entrypoint от cmd"))
    assert out == []


def test_engine_isolates_question_from_noisy_segment():
    engine = InterviewEngine(_matcher(), InterviewConfig())
    questions = []
    engine.on_question = questions.append
    engine._process(
        _final("мы же обсуждали control plane и etcd в k8s а теперь вопрос что такое entrypoint в докере")
    )
    assert len(questions) == 1
    assert questions[0].text == "что такое entrypoint в докере"


def test_engine_context_fallback_for_weak_query():
    engine = InterviewEngine(_matcher(), InterviewConfig())
    view = engine._build_view("в чём отличие entrypoint от cmd")
    assert view.topic == "docker"
    engine._context.note_answer("docker", ["entrypoint", "cmd"])
    weak = engine._build_view("расскажи поподробнее про эту тему")
    assert weak is not None
    assert weak.topic == "docker"


def test_engine_strong_query_overrides_context():
    engine = InterviewEngine(_matcher(), InterviewConfig())
    engine._context.note_answer("docker", ["entrypoint"])
    view = engine._build_view("что такое etcd")
    assert view.topic in ("kubernetes", "cloud")  # expanded KB


def test_engine_context_boost_does_not_flip_strong_subject():
    engine = InterviewEngine(_matcher(), InterviewConfig())
    engine._context.note_answer("kubernetes", ["kubernetes"])
    view = engine._build_view("расскажи что знаешь про docker")
    assert view.topic == "docker"
    assert not view.miss


def test_engine_context_collects_blocks():
    engine = InterviewEngine(_matcher(), InterviewConfig())
    engine._process(_final("в чём отличие entrypoint от cmd"))
    blocks = engine._context.blocks("docker")
    assert any("ENTRYPOINT" in b["question"] for b in blocks)
    engine._process(_final("что такое etcd"))
    assert engine._context.blocks("kubernetes")


# -- interview engine: LLM subject rescue ------------------------------------


class _FakeSubjectLlm:
    available = True

    def __init__(self, subjects):
        self.subjects = subjects
        self.calls = []

    def extract_subject_keywords(self, text, context=""):
        self.calls.append((text, context))
        return self.subjects

    def answer_question(self, question, context="", mode="technical", previous_qa=""):
        return None


def test_engine_llm_rescue_on_weak_query():
    llm = _FakeSubjectLlm(["kubernetes"])
    engine = InterviewEngine(_matcher(), InterviewConfig(), llm=llm)
    view = engine._build_view("ну про тот самый инструмент помнишь")
    assert view is not None
    assert view.topic in ("kubernetes", "cloud")  # expanded KB
    assert not view.miss
    assert llm.calls == [("ну про тот самый инструмент помнишь", "")]


def test_engine_miss_without_llm_stays_weak():
    engine = InterviewEngine(_matcher(), InterviewConfig())
    view = engine._build_view("ну про тот самый инструмент помнишь")
    assert view is None


def test_engine_llm_rescue_overrides_active_topic():
    llm = _FakeSubjectLlm(["kubernetes"])
    engine = InterviewEngine(_matcher(), InterviewConfig(), llm=llm)
    engine._context.note_answer("docker", ["entrypoint"])
    view = engine._build_view("расскажи поподробнее про эту тему")
    assert view is not None
    assert view.topic in ("kubernetes", "cloud")  # expanded KB
    assert not view.miss


def test_engine_llm_rescue_flag_off():
    llm = _FakeSubjectLlm(["kubernetes"])
    engine = InterviewEngine(_matcher(), InterviewConfig(subject_llm=False), llm=llm)
    engine._context.note_answer("docker", ["entrypoint"])
    view = engine._build_view("расскажи поподробнее про эту тему")
    assert view is not None
    assert view.topic == "docker"
    assert view.miss is True
    assert llm.calls == []


def test_engine_subject_rescue_async_upgrades_view():
    llm = _FakeSubjectLlm(["kubernetes"])
    engine = InterviewEngine(
        _matcher(),
        InterviewConfig(predict_llm=False),
        llm=llm,
    )
    out = []
    engine.on_answer = out.append
    engine.start()
    query = "расскажи про тот инструмент"
    try:
        engine._process(_final(query))
        deadline = time.monotonic() + 3.0
        while (
            not (len(out) >= 2 and out[-1].topic == "kubernetes")
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
    finally:
        engine.stop()
    assert llm.calls, "subject rescue was not scheduled"
    assert llm.calls[0][0] == query
    # the weak query emits a placeholder first, then the upgraded strong view
    assert len(out) >= 2
    assert out[-1].topic in ("kubernetes", "cloud")  # expanded KB
    assert not out[-1].miss
    assert out[-1].matched_query == query


def test_engine_broad_subject_kept_in_question():
    engine = InterviewEngine(_matcher(), InterviewConfig())
    out = []
    engine.on_answer = out.append
    engine._process(_final("я про kubernetes что знаешь"))
    assert len(out) == 1
    assert out[0].topic in ("kubernetes", "cloud")  # expanded KB
    assert not out[0].miss
    assert len(out[0].blocks) > 20


class _FakeAnswerLlm:
    available = True

    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def extract_subject_keywords(self, text, context=""):
        return []

    def answer_question(self, question, context="", mode="technical", previous_qa=""):
        self.calls.append((question, context))
        return self.answer


def test_engine_llm_answer_on_miss():
    llm = _FakeAnswerLlm("Kubelet — агент на каждой ноде.")
    engine = InterviewEngine(_matcher(), InterviewConfig(min_match_score=99.0), llm=llm)
    view = engine._build_view("что такое kubelet")
    assert view is not None
    assert view.miss is True
    assert view.llm_answered is False
    # the sync in-view answer is gone: the LLM is not contacted during build
    assert llm.calls == []
    # the primary pane answers instead (streaming + cache path)
    out = []
    engine.on_llm_answer = out.append
    engine._maybe_answer_llm(view, "что такое kubelet")
    if engine._answer_thread:
        engine._answer_thread.join(timeout=2)
    assert llm.calls and llm.calls[0][0] == "что такое kubelet"
    assert any(m.done and m.answer == "Kubelet — агент на каждой ноде." for m in out)


def test_engine_llm_answer_flag_off():
    llm = _FakeAnswerLlm("Ответ.")
    engine = InterviewEngine(
        _matcher(), InterviewConfig(min_match_score=99.0, answer_llm=False), llm=llm
    )
    view = engine._build_view("что такое kubelet")
    assert view is not None
    assert view.miss is True
    assert view.llm_answered is False
    assert llm.calls == []


def test_engine_llm_answer_without_llm():
    engine = InterviewEngine(_matcher(), InterviewConfig(min_match_score=99.0))
    view = engine._build_view("что такое kubelet")
    assert view is not None
    assert view.miss is True
    assert view.llm_answered is False


def test_engine_llm_answer_falls_back_on_none():
    llm = _FakeAnswerLlm(None)
    engine = InterviewEngine(_matcher(), InterviewConfig(min_match_score=99.0), llm=llm)
    view = engine._build_view("что такое kubelet")
    assert view is not None
    assert view.miss is True
    assert view.llm_answered is False
    assert view.blocks[0].question != "что такое kubelet"
    assert llm.calls == []


def test_engine_llm_answer_no_topic():
    llm = _FakeAnswerLlm("Ответ по экспертизе.")
    engine = InterviewEngine(_matcher(), InterviewConfig(min_match_score=99.0), llm=llm)
    view = engine._build_view("zizzzq а есть ли такой инструмент")
    assert view is not None
    assert view.miss is True
    assert view.topic == "general"
    assert view.blocks == []
    assert view.llm_answered is False
    # the primary pane answers the total miss instead of an in-view block
    assert llm.calls == []


# -- next-question prediction ------------------------------------------------


def _mini_matcher():
    docker = KbTopic(
        id="docker",
        title="Docker",
        sections=[
            KbSection(
                id="s1",
                name="Dockerfile",
                blocks=[
                    KbBlock(
                        id="d1",
                        section="Dockerfile",
                        question="В чём отличие ENTRYPOINT от CMD?",
                        answer="A1",
                        keywords=["entrypoint", "cmd"],
                        related=["Что такое Dockerfile?"],
                    ),
                    KbBlock(
                        id="d2",
                        section="Dockerfile",
                        question="Что такое Dockerfile?",
                        answer="A2",
                        keywords=["dockerfile"],
                    ),
                ],
            ),
            KbSection(
                id="s2",
                name="Образы",
                blocks=[
                    KbBlock(
                        id="d3",
                        section="Образы",
                        question="Что такое образ?",
                        answer="A3",
                        keywords=["образ"],
                    ),
                ],
            ),
        ],
    )
    return KbMatcher(KbIndex([docker]))


def test_predict_next_questions_from_view():
    matcher = _mini_matcher()
    engine = InterviewEngine(matcher, InterviewConfig())
    view = engine._build_view("в чём отличие entrypoint от cmd")
    assert view is not None
    assert view.topic == "docker"
    questions = [q.question for q in view.next_questions]
    # related link resolved with its full answer, then a sibling not in the view
    assert "Что такое Dockerfile?" in questions
    assert "Что такое образ?" in questions
    assert len(set(questions)) == len(questions)
    assert all(q.answer for q in view.next_questions)
    assert all(q.topic == "docker" for q in view.next_questions)


def test_predict_caps_at_limit():
    matcher = _mini_matcher()
    engine = InterviewEngine(matcher, InterviewConfig(max_next=1))
    view = engine._build_view("в чём отличие entrypoint от cmd")
    assert view is not None
    assert len(view.next_questions) <= 1


def test_predict_related_resolved_to_answer():
    matcher = _mini_matcher()
    view = protocol.KnowledgeView(
        topic="docker",
        title="Docker",
        blocks=[
            protocol.AnswerBlock(
                id="d1",
                section="Dockerfile",
                question="В чём отличие ENTRYPOINT от CMD?",
                answer="A1",
                related=["Что такое Dockerfile?"],
            )
        ],
    )
    out = next_questions_from_view(view, matcher, limit=3)
    assert any(q.question == "Что такое Dockerfile?" and q.answer == "A2" for q in out)
    assert any(q.question == "Что такое образ?" and q.answer == "A3" for q in out)


def test_predict_empty_inputs():
    matcher = _mini_matcher()
    assert next_questions_from_view(None, matcher) == []
    empty = protocol.KnowledgeView(topic="docker", title="Docker", blocks=[])
    assert next_questions_from_view(empty, matcher) == []


# -- answer_query (history restore) -------------------------------------------


def test_answer_query_restores_view_without_dedup_side_effect():
    engine = InterviewEngine(_matcher(), InterviewConfig())
    query = "в чём отличие entrypoint от cmd"
    view = engine.answer_query(query)
    assert view is not None
    assert view.topic == "docker"
    # the live dedup state must be untouched: the same query is still answerable
    live = engine._build_view(query)
    assert live is not None
    # cached copy is returned for repeat calls
    assert engine.answer_query(query) is view


def test_answer_query_returns_none_for_empty():
    engine = InterviewEngine(_matcher(), InterviewConfig())
    assert engine.answer_query("") is None
    assert engine.answer_query("   ") is None


# -- LLM prediction gating ----------------------------------------------------


class _FakePredictLlm:
    available = True

    def __init__(self):
        self.calls = []

    def predict_questions(self, question, topic, context="", max_q=5):
        self.calls.append((question, topic, context, max_q))
        return [{"question": "Что такое Dockerfile?", "answer": "A"}]


def test_engine_predict_flag_off():
    llm = _FakePredictLlm()
    engine = InterviewEngine(_matcher(), InterviewConfig(predict_llm=False), llm=llm)
    view = protocol.KnowledgeView(
        topic="docker",
        title="Docker",
        blocks=[protocol.AnswerBlock(id="a", section="s", question="q", answer="a")],
    )
    engine._maybe_predict(view, "q")
    assert llm.calls == []
    assert engine._last_predict_ts == 0.0


def test_engine_predict_requires_llm_topic_and_blocks():
    engine = InterviewEngine(_matcher(), InterviewConfig())
    view = protocol.KnowledgeView(
        topic="docker",
        title="Docker",
        blocks=[protocol.AnswerBlock(id="a", section="s", question="q", answer="a")],
    )
    engine._maybe_predict(view, "q")
    assert engine._last_predict_ts == 0.0
    llm = _FakePredictLlm()
    engine2 = InterviewEngine(_matcher(), InterviewConfig(), llm=llm)
    empty = protocol.KnowledgeView(topic="docker", title="Docker", blocks=[])
    engine2._maybe_predict(empty, "q")
    assert engine2._last_predict_ts == 0.0
    assert llm.calls == []


def test_engine_predict_cooldown_gate():
    llm = _FakePredictLlm()
    engine = InterviewEngine(_matcher(), InterviewConfig(), llm=llm)
    view = protocol.KnowledgeView(
        topic="docker",
        title="Docker",
        blocks=[protocol.AnswerBlock(id="a", section="s", question="q", answer="a")],
    )
    engine._maybe_predict(view, "q")
    ts1 = engine._last_predict_ts
    assert ts1 > 0.0
    engine._maybe_predict(view, "q")
    assert engine._last_predict_ts == ts1


# -- parallel LLM answer gating ----------------------------------------------


def _answer_view(**overrides) -> protocol.KnowledgeView:
    defaults = {
        "topic": "docker",
        "title": "Docker",
        "blocks": [protocol.AnswerBlock(id="a", section="s", question="q", answer="a")],
    }
    defaults.update(overrides)
    return protocol.KnowledgeView(**defaults)


def test_engine_maybe_answer_llm_flag_off():
    llm = _FakeAnswerLlm("Ответ ИИ")
    engine = InterviewEngine(_matcher(), InterviewConfig(llm_primary=False), llm=llm)
    engine._maybe_answer_llm(_answer_view(), "в чём отличие entrypoint от cmd")
    assert engine._last_answer_ts == 0.0


def test_engine_maybe_answer_llm_requires_llm_topic_and_blocks():
    engine = InterviewEngine(_matcher(), InterviewConfig())
    engine._maybe_answer_llm(_answer_view(), "q")
    assert engine._last_answer_ts == 0.0
    llm = _FakeAnswerLlm("Ответ ИИ")
    engine2 = InterviewEngine(_matcher(), InterviewConfig(), llm=llm)
    engine2._maybe_answer_llm(None, "q")
    engine2._maybe_answer_llm(_answer_view(topic=""), "q")
    assert engine2._last_answer_ts == 0.0


def test_engine_maybe_answer_llm_always_calls():
    """Variant A: the LLM is always the primary answerer.

    Even when the KB already produced an ``llm_answered`` view, the engine
    must still schedule an LLM call so the user gets a rich, expanded answer.
    """
    llm = _FakeAnswerLlm("Ответ ИИ")
    engine = InterviewEngine(_matcher(), InterviewConfig(), llm=llm)
    engine.on_llm_answer = lambda *_a, **_kw: None
    engine._maybe_answer_llm(_answer_view(llm_answered=True), "q")
    # ``_last_answer_ts`` is bumped immediately when the worker is scheduled.
    assert engine._last_answer_ts > 0.0


def test_engine_maybe_answer_llm_cooldown_gate():
    llm = _FakeAnswerLlm("Ответ ИИ")
    engine = InterviewEngine(_matcher(), InterviewConfig(answer_cooldown_s=1000.0), llm=llm)
    # Force-bypass cooldown on the first call so _last_answer_ts gets set.
    engine._maybe_answer_llm(_answer_view(), "первый вопрос", force=True)
    ts1 = engine._last_answer_ts
    assert ts1 > 0.0
    engine._maybe_answer_llm(_answer_view(), "второй вопрос")
    assert engine._last_answer_ts == ts1


def test_engine_answer_llm_worker_emits():
    llm = _FakeAnswerLlm("Ответ ИИ")
    engine = InterviewEngine(_matcher(), InterviewConfig(), llm=llm)
    out = []
    engine.on_llm_answer = out.append
    engine._answer_llm_worker("в чём отличие entrypoint от cmd", "docker", "Docker", "контекст")
    assert llm.calls == [("в чём отличие entrypoint от cmd", "контекст")]
    assert len(out) == 1
    msg = out[0]
    assert msg.type == protocol.MessageType.LLM_ANSWER
    assert msg.query == "в чём отличие entrypoint от cmd"
    assert msg.topic == "docker"
    assert msg.title == "Docker"
    assert msg.answer == "Ответ ИИ"
    assert msg.done is True


def test_engine_answer_llm_worker_ignores_empty():
    llm = _FakeAnswerLlm(None)
    engine = InterviewEngine(_matcher(), InterviewConfig(), llm=llm)
    out = []
    engine.on_llm_answer = out.append
    engine._answer_llm_worker("q", "docker", "Docker", "")
    assert len(out) == 1
    assert out[0].done is True
    assert out[0].answer == ""


class _FakeStreamLlm:
    available = True

    def __init__(self, deltas):
        self.deltas = deltas
        self.calls = []

    def answer_question_stream(self, question, context="", mode="technical", previous_qa=""):
        self.calls.append((question, context))
        for delta in self.deltas:
            yield delta


def test_engine_answer_llm_worker_streams_chunks():
    llm = _FakeStreamLlm(["Привет", ", ", "мир"])
    engine = InterviewEngine(_matcher(), InterviewConfig(), llm=llm)
    out = []
    engine.on_llm_answer = out.append
    engine._answer_llm_worker("в чём отличие entrypoint от cmd", "docker", "Docker", "контекст")
    assert llm.calls == [("в чём отличие entrypoint от cmd", "контекст")]
    chunks = [m for m in out if not m.done]
    assert [c.delta for c in chunks] == ["Привет", ", ", "мир"]
    assert all(not c.answer for c in chunks)
    final = out[-1]
    assert final.done is True
    assert final.answer == "Привет, мир"
    assert final.query == "в чём отличие entrypoint от cmd"
    assert final.topic == "docker"
    assert final.title == "Docker"


def test_engine_answer_llm_worker_stream_skips_empty_deltas():
    llm = _FakeStreamLlm(["a", "", "b"])
    engine = InterviewEngine(_matcher(), InterviewConfig(), llm=llm)
    out = []
    engine.on_llm_answer = out.append
    engine._answer_llm_worker("q", "docker", "Docker", "")
    chunks = [m for m in out if not m.done]
    assert [c.delta for c in chunks] == ["a", "b"]
    assert out[-1].done is True
    assert out[-1].answer == "ab"


def test_engine_answer_llm_worker_stream_off_uses_sync():
    llm = _FakeStreamLlm(["x"])
    engine = InterviewEngine(_matcher(), InterviewConfig(answer_stream=False), llm=llm)
    out = []
    engine.on_llm_answer = out.append
    engine._answer_llm_worker("q", "docker", "Docker", "контекст")
    assert llm.calls == []
    assert len(out) == 1
    assert out[0].done is True
    assert out[0].answer == ""


# -- answer cache --------------------------------------------------------------


def test_engine_answer_cache_serves_repeat_without_new_llm_call():
    llm = _FakeAnswerLlm("Ответ из ИИ")
    engine = InterviewEngine(
        _matcher(), InterviewConfig(answer_cooldown_s=0.0), llm=llm
    )
    out = []
    engine.on_llm_answer = out.append
    query = "в чём отличие entrypoint от cmd"
    engine._maybe_answer_llm(_answer_view(), query)
    if engine._answer_thread:
        engine._answer_thread.join(timeout=2)
    engine._maybe_answer_llm(_answer_view(), query)
    engine._maybe_answer_llm(_answer_view(), query)
    assert len(llm.calls) == 1
    assert llm.calls[0][0] == query
    cached = [m for m in out if m.done and m.answer == "Ответ из ИИ"]
    assert len(cached) == 3  # worker final + two synchronous cache hits


def test_engine_answer_cache_off_recontacts_llm():
    llm = _FakeAnswerLlm("Ответ из ИИ")
    engine = InterviewEngine(
        _matcher(), InterviewConfig(answer_cooldown_s=0.0, answer_cache=False), llm=llm
    )
    query = "в чём отличие entrypoint от cmd"
    engine._maybe_answer_llm(_answer_view(), query)
    if engine._answer_thread:
        engine._answer_thread.join(timeout=2)
    engine._maybe_answer_llm(_answer_view(), query)
    if engine._answer_thread:
        engine._answer_thread.join(timeout=2)
    assert len(llm.calls) == 2


def test_engine_answer_cache_does_not_store_empty():
    llm = _FakeAnswerLlm(None)
    engine = InterviewEngine(
        _matcher(), InterviewConfig(answer_cooldown_s=0.0), llm=llm
    )
    query = "в чём отличие entrypoint от cmd"
    engine._maybe_answer_llm(_answer_view(), query)
    if engine._answer_thread:
        engine._answer_thread.join(timeout=2)
    engine._maybe_answer_llm(_answer_view(), query)
    if engine._answer_thread:
        engine._answer_thread.join(timeout=2)
    assert len(llm.calls) == 2


# -- early answer on partial transcripts ---------------------------------------


def _partial(text, segment_id="seg1"):
    return protocol.PartialTranscript(segment_id=segment_id, text=text)


def _partial_engine(llm=None, **overrides):
    overrides.setdefault("use_partials", True)
    overrides.setdefault("predict_llm", False)
    overrides.setdefault("subject_llm", False)
    return InterviewEngine(_matcher(), InterviewConfig(**overrides), llm=llm)


def test_engine_partial_early_start_streams_answer():
    llm = _FakeStreamLlm(["Ответ", " ", "по ранней гипотезе"])
    engine = _partial_engine(llm)
    out = []
    engine.on_llm_answer = out.append
    engine._process_partial(_partial("в чём отличие entrypoint от cmd"))
    engine._process_partial(_partial("в чём отличие entrypoint от cmd"))
    if engine._answer_thread:
        engine._answer_thread.join(timeout=2)
    assert engine._provisional_query == "в чём отличие entrypoint от cmd"
    assert llm.calls
    assert llm.calls[0][0] == "в чём отличие entrypoint от cmd"
    assert any(m.done and m.answer == "Ответ по ранней гипотезе" for m in out)


def test_engine_partial_requires_stability_rounds():
    llm = _FakeStreamLlm(["a"])
    engine = _partial_engine(llm)
    engine._process_partial(_partial("в чём отличие entrypoint от cmd"))
    assert engine._partial_stable == 1
    assert not llm.calls
    assert engine._provisional_query == ""
    # a different partial resets the stability counter
    engine._process_partial(_partial("что такое dockerfile"))
    assert engine._partial_stable == 1


def test_engine_partial_flag_off_ignored():
    llm = _FakeStreamLlm(["a"])
    engine = _partial_engine(llm, use_partials=False)
    engine._process_partial(_partial("в чём отличие entrypoint от cmd"))
    engine._process_partial(_partial("в чём отличие entrypoint от cmd"))
    assert not llm.calls
    assert engine._provisional_query == ""


def test_engine_partial_non_question_ignored():
    llm = _FakeStreamLlm(["a"])
    engine = _partial_engine(llm)
    engine._process_partial(_partial("документ готов к ревью"))
    engine._process_partial(_partial("документ готов к ревью"))
    assert not llm.calls


def test_engine_partial_final_same_query_keeps_early_answer():
    llm = _FakeStreamLlm(["a"])
    engine = _partial_engine(llm, answer_cooldown_s=1000.0)
    out = []
    engine.on_llm_answer = out.append
    query = "в чём отличие entrypoint от cmd"
    engine._process_partial(_partial(query))
    engine._process_partial(_partial(query))
    engine._process(_final(query))
    if engine._answer_thread:
        engine._answer_thread.join(timeout=2)
    assert len(llm.calls) == 1
    assert len([m for m in out if m.done and m.query == query]) == 1


def test_engine_partial_final_differs_restarts_answer():
    q1 = "в чём отличие entrypoint от cmd"
    q2 = "что такое kubelet"
    llm = _FakeStreamLlm(["a"])
    engine = _partial_engine(llm, answer_cooldown_s=1000.0)
    out = []
    engine.on_llm_answer = out.append
    engine._process_partial(_partial(q1))
    engine._process_partial(_partial(q1))
    first = engine._answer_thread
    engine._process(_final(q2))
    if first:
        first.join(timeout=2)
    if engine._answer_thread:
        engine._answer_thread.join(timeout=2)
    queries = [c[0] for c in llm.calls]
    assert q1 in queries and q2 in queries


def test_engine_partial_final_cosmetic_change_keeps_early_answer():
    q1 = "что такое kubectl"
    # With answer_restart_min_similarity=0.7 the final wording must have a
    # Jaccard overlap ≥ 0.7 to count as cosmetic. Adding one filler word
    # (Jaccard 3/4 = 0.75) qualifies; adding two new tokens (0.6) does not.
    q2 = "ну что такое kubectl"
    llm = _FakeStreamLlm(["a"])
    engine = _partial_engine(llm, answer_cooldown_s=1000.0)
    out = []
    engine.on_llm_answer = out.append
    engine._process_partial(_partial(q1))
    engine._process_partial(_partial(q1))
    first = engine._answer_thread
    engine._process(_final(q2))
    if first:
        first.join(timeout=2)
    if engine._answer_thread:
        engine._answer_thread.join(timeout=2)
    # the final only adds filler words -> the early stream is kept
    assert [c[0] for c in llm.calls] == [q1]


def test_engine_partial_final_semantic_change_restarts_answer():
    q1 = "что такое kubectl"
    q2 = "как масштабировать kubelet"
    llm = _FakeStreamLlm(["a"])
    engine = _partial_engine(llm, answer_cooldown_s=1000.0)
    engine._process_partial(_partial(q1))
    engine._process_partial(_partial(q1))
    first = engine._answer_thread
    engine._process(_final(q2))
    if first:
        first.join(timeout=2)
    if engine._answer_thread:
        engine._answer_thread.join(timeout=2)
    queries = [c[0] for c in llm.calls]
    assert q1 in queries and q2 in queries


def test_engine_partial_does_not_consume_dedup_for_final():
    llm = _FakeStreamLlm(["a"])
    engine = _partial_engine(llm)
    answers = []
    engine.on_answer = answers.append
    query = "в чём отличие entrypoint от cmd"
    engine._process_partial(_partial(query))
    engine._process_partial(_partial(query))
    engine._process(_final(query))
    if engine._answer_thread:
        engine._answer_thread.join(timeout=2)
    # the RAG view is emitted early (partial) and again on the final transcript
    assert len(answers) == 2
    assert answers[0].partial is True
    assert answers[1].partial is False
    assert answers[1].topic == "docker"


def test_engine_partial_emits_view_early():
    llm = _FakeStreamLlm(["a"])
    engine = _partial_engine(llm)
    answers = []
    engine.on_answer = answers.append
    query = "в чём отличие entrypoint от cmd"
    engine._process_partial(_partial(query))
    engine._process_partial(_partial(query))
    assert len(answers) == 1
    view = answers[0]
    assert view.partial is True
    assert view.topic == "docker"
    assert not view.miss
    # a repeated stable partial does not re-emit the view
    engine._process_partial(_partial(query))
    assert len(answers) == 1


# -- topics engine (offline glossary grouping) -------------------------------


class _FakeLlm:
    available = False

    def analyze_topics(self, transcript):
        return []


def _topic_engine(**overrides):
    from mockingbird.terms.glossary import Glossary

    overrides.setdefault("enabled", True)
    return TopicEngine(Glossary.load(), _FakeLlm(), TopicsConfig(**overrides))


def _detected(engine, msg):
    blocks = []
    engine.on_topics = blocks.append
    engine._handle(msg)
    engine._run_analysis()
    return blocks[0] if blocks else None


def test_topics_engine_offline_groups_by_category():
    engine = _topic_engine()
    _detected(engine, _final("Деплой через k8s прошёл успешно."))
    _detected(
        engine,
        protocol.TermDetected(
            term="Kubernetes", explanation="x", source=protocol.TermSource.GLOSSARY,
        ),
    )
    _detected(
        engine,
        protocol.TermDetected(
            term="Docker", explanation="x", source=protocol.TermSource.GLOSSARY,
        ),
    )
    blocks = engine._glossary_blocks()
    infra = [b for b in blocks if b.category == "infra"]
    assert infra
    docker_block = next(b for b in infra if "Docker" in b.terms)
    assert any("ENTRYPOINT" in q.question for q in docker_block.questions)


def test_topics_engine_conversation_blocks_from_kb():
    engine = _topic_engine()
    engine._context.add_block(
        "docker", "Docker — всё по теме", "Dockerfile",
        "В чём отличие ENTRYPOINT от CMD?", "Ответ про entrypoint",
        ["Что такое Dockerfile?"], 5.0,
    )
    engine._context.add_block(
        "docker", "Docker — всё по теме", "Dockerfile",
        "Что такое Dockerfile?", "Ответ про dockerfile", [], 4.0,
    )
    blocks = engine._conversation_blocks()
    assert blocks and len(blocks) == 1
    block = blocks[0]
    assert block.source == protocol.TopicSource.KB
    assert block.category == "беседа"
    assert any("ENTRYPOINT" in q.question for q in block.questions)


def test_topics_engine_run_analysis_includes_conversation():
    engine = _topic_engine(include_kb=True)
    engine._context.add_block(
        "docker", "Docker — всё по теме", "Dockerfile", "Q", "A", [], 1.0,
    )
    out = []
    engine.on_topics = out.append
    engine._run_analysis()
    assert out and out[0][0].source == protocol.TopicSource.KB
    engine2 = _topic_engine(include_kb=False)
    engine2._context.add_block(
        "docker", "Docker — всё по теме", "Dockerfile", "Q", "A", [], 1.0,
    )
    out2 = []
    engine2.on_topics = out2.append
    engine2._run_analysis()
    assert out2 == []


def test_topics_engine_disabled_emits_nothing():
    engine = _topic_engine(enabled=False)
    out = []
    engine.on_topics = out.append
    engine._handle(_final("поговорим про docker"))
    assert out == []


# -- llm topics parsing ------------------------------------------------------


def test_parse_topics_json_plain():
    text = (
        '{"topics": [{"theme": "Контейнеры", "terms": ["Docker"], '
        '"questions": [{"question": "Q1", "answer": "A1"}]}]}'
    )
    out = parse_topics_json(text)
    assert out == [{"theme": "Контейнеры", "terms": ["Docker"],
                    "questions": [{"question": "Q1", "answer": "A1"}]}]


def test_parse_topics_json_fence_and_garbage():
    text = '```json\n{"topics": [{"theme": "Сети", "questions": []}]}\n```'
    assert parse_topics_json(text) == [{"theme": "Сети", "terms": [], "questions": []}]
    assert parse_topics_json("nope") == []
    assert parse_topics_json("") == []


# -- protocol roundtrip ------------------------------------------------------


def test_topic_block_roundtrip():
    block = protocol.TopicBlock(
        block_id="b1",
        theme="Контейнеры",
        questions=[protocol.RelatedQuestion(question="Q", answer="A")],
        source=protocol.TopicSource.GLOSSARY,
    )
    parsed = protocol.TopicBlock.model_validate_json(block.model_dump_json())
    assert parsed.theme == "Контейнеры"
    assert parsed.questions[0].question == "Q"


def test_knowledge_view_roundtrip():
    view = protocol.KnowledgeView(
        topic="kubernetes",
        blocks=[protocol.AnswerBlock(id="a", section="s", question="q", answer="ans", highlight=["c"])],
        miss=True,
        next_questions=[protocol.RelatedQuestion(question="Q1", answer="A1", topic="kubernetes")],
    )
    parsed = protocol.KnowledgeView.model_validate_json(view.model_dump_json())
    assert parsed.topic == "kubernetes"
    assert parsed.miss is True
    assert parsed.blocks[0].highlight == ["c"]
    assert parsed.next_questions[0].question == "Q1"


def test_predictions_roundtrip():
    msg = protocol.Predictions(
        query="в чём отличие entrypoint от cmd",
        topic="docker",
        questions=[protocol.RelatedQuestion(question="Q1", answer="A1", topic="docker")],
    )
    parsed = protocol.Predictions.model_validate_json(msg.model_dump_json())
    assert parsed.type == protocol.MessageType.PREDICTIONS
    assert parsed.query == "в чём отличие entrypoint от cmd"
    assert parsed.questions[0].answer == "A1"


def test_llm_answer_roundtrip():
    msg = protocol.LlmAnswer(
        query="в чём отличие entrypoint от cmd",
        topic="docker",
        title="Docker",
        answer="Ответ ИИ",
        delta="фрагм",
        done=True,
        context_summary="Тема: Docker",
    )
    parsed = protocol.LlmAnswer.model_validate_json(msg.model_dump_json())
    assert parsed.type == protocol.MessageType.LLM_ANSWER
    assert parsed.query == "в чём отличие entrypoint от cmd"
    assert parsed.topic == "docker"
    assert parsed.title == "Docker"
    assert parsed.answer == "Ответ ИИ"
    assert parsed.delta == "фрагм"
    assert parsed.done is True
    assert parsed.context_summary == "Тема: Docker"


def test_llm_answer_defaults_done_false():
    msg = protocol.LlmAnswer(query="q", answer="a")
    assert msg.done is False
    assert msg.delta == ""


# -- conversation graph builder (Qt-free) ------------------------------------


def _block(topic, question, answer="Ответ", related=None, count=1):
    return {
        "id": f"{topic}:{question}",
        "topic": topic,
        "title": f"{topic} — тема",
        "section": "Раздел",
        "question": question,
        "answer": answer,
        "related": related or [],
        "score": 5.0,
        "count": count,
        "ts": count,
    }


def test_graph_builds_nodes_and_topic_edges():
    from mockingbird.kb.graph import build_graph, EDGE_TOPIC, NODE_BLOCK, NODE_TOPIC

    graph = build_graph([_block("docker", "В чём отличие ENTRYPOINT от CMD?"), _block("docker", "Что такое Dockerfile?")])
    topics = [n for n in graph.nodes if n.kind == NODE_TOPIC]
    blocks = [n for n in graph.nodes if n.kind == NODE_BLOCK]
    assert [n.label for n in topics] == ["docker — тема"]
    assert len(blocks) == 2
    assert len(graph.edges) == 2
    assert all(e.kind == EDGE_TOPIC for e in graph.edges)
    assert all(e.dst.startswith("b:") and e.src == "t:docker" for e in graph.edges)


def test_graph_related_crosslink_normalizes_text():
    from mockingbird.kb.graph import build_graph, EDGE_RELATED

    graph = build_graph(
        [
            _block("docker", "В чём отличие ENTRYPOINT от CMD?", related=["Что такое Dockerfile ?"]),
            _block("docker", "Что такое Dockerfile?"),
        ]
    )
    related = [e for e in graph.edges if e.kind == EDGE_RELATED]
    assert len(related) == 1
    assert related[0].src != related[0].dst


def test_graph_ignores_self_links_and_unknown_related():
    from mockingbird.kb.graph import build_graph

    graph = build_graph(
        [
            _block("docker", "Q1", related=["Q1", "нет такого вопроса"]),
        ]
    )
    assert all(e.kind != "related" for e in graph.edges)


def test_graph_caps_nodes_by_discussion_count():
    from mockingbird.kb.graph import build_graph

    blocks = [_block("k", f"Вопрос {i}", count=i) for i in range(1, 100)]
    graph = build_graph(blocks, max_nodes=10)
    assert sum(1 for n in graph.nodes if n.kind == "block") == 10
    assert graph.nodes[0].weight == 99


def test_graph_layout_topics_on_ring_blocks_around_center():
    from mockingbird.kb.graph import build_graph, layout_graph

    graph = build_graph(
        [_block("docker", "Q1"), _block("docker", "Q2"), _block("kubernetes", "Q3")]
    )
    layout_graph(graph)
    docker_topic = graph.node("t:docker")
    kubernetes_topic = graph.node("t:kubernetes")
    assert docker_topic is not None and kubernetes_topic is not None
    assert docker_topic.x != kubernetes_topic.x or docker_topic.y != kubernetes_topic.y
    docker_blocks = [n for n in graph.nodes if n.kind == "block" and n.topic == "docker"]
    assert all((n.x - docker_topic.x) ** 2 + (n.y - docker_topic.y) ** 2 > 1 for n in docker_blocks)


def test_graph_levels_for_scale():
    from mockingbird.kb.graph import level_for_scale

    assert level_for_scale(0.3) == 0
    assert level_for_scale(0.8) == 1
    assert level_for_scale(2.0) == 2


def test_graph_levels_boosted_show_detail_earlier():
    from mockingbird.kb.graph import level_for_scale

    assert level_for_scale(0.1, boosted=True) == 0
    assert level_for_scale(0.3, boosted=True) == 1
    assert level_for_scale(0.8, boosted=True) == 2


def _graph_fixture():
    from mockingbird.kb.graph import Graph, GraphEdge, GraphNode, build_graph

    blocks = [
        _block("docker", "В чём отличие ENTRYPOINT от CMD?", related=["Что такое Dockerfile?"]),
        _block("docker", "Что такое Dockerfile?", related=["В чём отличие ENTRYPOINT от CMD?"]),
        _block("kubernetes", "Что такое service?"),
    ]
    return build_graph(blocks)


def test_graph_expand_topic_returns_topic_and_blocks():
    from mockingbird.kb.graph import expand_topic

    graph = _graph_fixture()
    ids = expand_topic(graph, "docker")
    assert "t:docker" in ids
    assert sum(1 for i in ids if i.startswith("b:")) == 2
    assert "t:kubernetes" not in ids


def test_graph_visible_overview_only_topics():
    from mockingbird.kb.graph import visible_node_ids

    graph = _graph_fixture()
    visible = visible_node_ids(graph, set(), set())
    assert all(i.startswith("t:") for i in visible)


def test_graph_visible_expanded_adds_blocks():
    from mockingbird.kb.graph import visible_node_ids

    graph = _graph_fixture()
    visible = visible_node_ids(graph, {"docker"}, set())
    assert "t:docker" in visible and "t:kubernetes" in visible
    assert sum(1 for i in visible if i.startswith("b:") and "docker" in i) == 2
    assert not any("kubernetes" in i and i.startswith("b:") for i in visible)


def test_graph_visible_vector_beats_expansion():
    from mockingbird.kb.graph import visible_node_ids

    graph = _graph_fixture()
    visible = visible_node_ids(graph, {"kubernetes"}, {"t:docker", "b:docker:В чём отличие ENTRYPOINT от CMD?"})
    assert "t:kubernetes" not in visible


def test_graph_expand_vector_depth_and_anchors():
    from mockingbird.kb.graph import expand_vector

    graph = _graph_fixture()
    start = "b:docker:В чём отличие ENTRYPOINT от CMD?"
    vector = expand_vector(graph, start, depth=1)
    assert start in vector
    assert "t:docker" in vector
    # related target pulled in
    assert any("Dockerfile" in i for i in vector)
    # unrelated topic not dragged in
    assert "t:kubernetes" not in vector


def test_graph_expand_vector_cycle_safe_isolated():
    from mockingbird.kb.graph import expand_vector

    graph = _graph_fixture()
    start = "b:kubernetes:Что такое service?"
    vector = expand_vector(graph, start, depth=3)
    assert start in vector and "t:kubernetes" in vector
    assert len(vector) == 2  # itself + its topic anchor, no cycle blowup


def test_graph_focus_layout_centers_focus():
    from mockingbird.kb.graph import expand_vector, focus_layout

    graph = _graph_fixture()
    start = "b:docker:В чём отличие ENTRYPOINT от CMD?"
    visible = expand_vector(graph, start, 1)
    focus_layout(graph, visible, start)
    assert (graph.node(start).x, graph.node(start).y) == (0.0, 0.0)


def test_graph_remap_state_keeps_living_ids():
    from mockingbird.kb.graph import remap_state

    assert remap_state({"a", "b", "c"}, {"a", "c", "d"}) == {"a", "c"}
    assert remap_state({"x"}, {"y"}) == set()
