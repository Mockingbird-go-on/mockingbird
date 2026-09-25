import time

from mockingbird import protocol
from mockingbird.config import InterviewConfig
from mockingbird.kb.detector import is_broad, is_question, last_question
from mockingbird.kb.index import KbIndex
from mockingbird.kb.interview_engine import InterviewEngine
from mockingbird.kb.loader import load_topics
from mockingbird.kb.matcher import KbMatcher
from mockingbird.kb.model import KbBlock, KbSection, KbTopic
from mockingbird.kb.predict import next_questions_from_view


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


def test_is_question_kto_forms():
    """«кто такой X» / «что за X» должны считаться вопросами."""
    assert is_question("кто такой эрбак")
    assert is_question("кто такой kubernetes")
    assert is_question("кем ты работал")
    assert is_question("что за инструмент")
    assert is_question("кого это касается")
    assert not is_question("я пошёл домой")


def test_is_question_li_particle():
    """Yes/no вопросы с частицей «ли» после глагола."""
    assert is_question("работал ли ты с докерсварм")
    assert is_question("знаешь ли ты kubernetes")
    assert is_question("использовал ли ты терраформ")
    assert is_question("был ли у тебя опыт")
    assert not is_question("мы использовали docker")
    assert not is_question("проверил лично")
    assert not is_question("я пошёл домой")


def test_is_question_hypothetical():
    assert is_question("представь что прод упал")
    assert is_question("представь ситуацию")
    assert is_question("предположим что")
    assert is_question("допустим что")
    assert not is_question("представь в общем")


def test_is_question_modal_task():
    assert is_question("надо развернуть кластер")
    assert is_question("нужно настроить мониторинг")
    assert is_question("требуется поднять")
    assert not is_question("надо подумать")
    assert not is_question("надо перепроверить")


def test_is_question_modal_li():
    assert is_question("стоит ли использовать")
    assert is_question("правильно ли я понял")
    assert is_question("верно ли")
    assert is_question("обязательно ли")


def test_is_question_imperative_task():
    assert is_question("разверни кластер")
    assert is_question("настрой nginx")
    assert is_question("подними сервис")
    assert not is_question("настройка nginx")
    assert not is_question("настроение хорошее")


def test_is_question_desire_know():
    assert is_question("хочу узнать как")
    assert is_question("хочу понять как")
    assert is_question("интересно как")


def test_is_question_leading_particles():
    assert is_question("а почему кубернетес")
    assert is_question("ну как настроить")
    assert is_question("а можно ли")
    assert not is_question("а вот и всё")
    assert not is_question("ну вот")


def test_is_question_clipped_verbs():
    assert is_question("бъясни про")
    assert is_question("поясни")
    assert is_question("разжуй про")
    assert not is_question("объяснительная записка")


def test_is_question_comparison_interleaved():
    """«чем X отличается [от Y]» — предмет вставлен между «чем» и глаголом."""
    assert is_question("чем Docker отличается")
    assert is_question("чем докер отличается от виртуальной машины")
    assert is_question("чем docker отличается от виртуалки")
    assert is_question("а чем docker отличается от vm")


def test_is_question_comparison_comparative():
    """«чем X лучше/хуже/проще» — вопрос-сравнение."""
    assert is_question("чем docker лучше виртуалки")
    assert is_question("чем docker проще vm")


def test_is_question_comparison_noun_forms():
    """«отличие X от Y» / «разница между X и Y»."""
    assert is_question("отличие docker от виртуальной машины")
    assert is_question("разница между docker и vm")


def test_is_question_comparison_false_positives():
    """Косвенная речь и идиомы с «чем» — не вопросы."""
    assert not is_question("я знаю чем это отличается")
    assert not is_question("мы сравнивали чем они отличаются")
    assert not is_question("чем больше тем лучше")
    assert not is_question("мы использовали docker")


def test_is_question_imperative_te_forms():
    """Повелительные формы мн. числа («опишите», «сравните», «назовите») — вопросы.

    Регрессия: word-boundary guard в _starts_with_marker резал «опиши» →
    «опишите», когда явной «-те»-формы не было в списке.
    """
    assert is_question("опишите ваш Pipeline сиайсиди от комита до продак")
    assert is_question("сравните kubernetes и docker swarm")
    assert is_question("назовите основные команды git")
    assert is_question("покажите как настроить nginx")
    assert is_question("приведите пример деплоя")
    assert is_question("подскажите чем отличается pod от deployment")
    assert is_question("объясните-ка как работает ingress")
    # FP-барьер: существительные-омофоны по-прежнему не вопросы
    assert not is_question("объяснительная записка")
    assert not is_question("описка в документе")
    assert not is_question("сравнительный")
    assert not is_question("назовиста")
    assert not is_question("опишитехарактеристики сервера")


def test_is_question_comparison_nested_in_shift():
    """Сменa темы + вложенный вопрос-сравнение в одной реплике."""
    assert is_question("поговорим про Kubernetes чем Pod отличается от деплоя")
    assert is_question("давай обсудим docker чем docker лучше vm")
    # Без объекта сравнения после глагола — утверждение, не вопрос
    assert not is_question("он объяснил чем процесс отличается")
    assert not is_question("я знаю чем это отличается")


# -- Tier 2/3: non-question fallback + LLM rescue ---------------------------


def test_engine_topical_fallback_opens_topic():
    """Не-вопрос с сильным термином открывает тему (preview, без LLM)."""
    engine = InterviewEngine(_matcher(), InterviewConfig())
    answers = []
    engine.on_answer = answers.append
    engine._process(_final("мы использовали docker"))
    assert answers
    assert answers[0].preview is True
    assert answers[0].topic == "docker"


def test_engine_topical_fallback_ignores_generic_statement():
    """Обычное утверждение без KB-термина не открывает тему."""
    engine = InterviewEngine(_matcher(), InterviewConfig())
    answers = []
    engine.on_answer = answers.append
    engine._process(_final("документ готов к ревью"))
    assert answers == []


def test_engine_topical_fallback_skips_shift():
    """Topic-shift не дублирует preview от трекера."""
    engine = InterviewEngine(_matcher(), InterviewConfig())
    answers = []
    engine.on_answer = answers.append
    engine._process(_final("давай поговорим про k8s"))
    previews = [v for v in answers if v.preview]
    assert len(previews) == 1  # только от трекера, не от fallback


class _FakeRescueLlm:
    available = True

    def __init__(self, result):
        self._result = result
        self.calls = []

    def analyze_dialog_context(self, utterance, history=""):
        self.calls.append((utterance, history))
        return self._result

    def answer_question(self, question, context="", mode="technical", previous_qa=""):
        return None


class _FakeDialog:
    def __init__(self, result):
        self._result = result

    def resolve(self, utterance):
        return self._result


def test_question_rescue_promotes_llm_question():
    """LLM классифицирует не-вопрос как question → полный путь."""
    from mockingbird.kb.dialog_context import DialogContextManager

    llm = _FakeRescueLlm({"type": "question", "resolved_query": "как настроить docker",
                          "answer_mode": "technical", "confidence": 0.9})
    dialog = DialogContextManager(llm=llm)
    # обходим _question_rescue_available: подставим заглушку с analyze_dialog_context
    engine = InterviewEngine(_matcher(), InterviewConfig(), llm=llm, dialog_context=dialog)
    engine._dialog = dialog
    questions = []
    engine.on_question = questions.append
    answers = []
    engine.on_answer = answers.append
    engine._process(_final("как настроить докер"))
    engine._rescue_question_worker("как настроить докер", _final("как настроить докер"), engine._generation)
    assert any(a.topic == "docker" for a in answers if not a.preview)


def test_question_rescue_ignores_fallback_source():
    """fallback-источник (без LLM) не превращает утверждение в вопрос."""
    from mockingbird.kb.dialog_context import DialogContextManager

    llm = _FakeRescueLlm({})
    dialog = DialogContextManager(llm=llm)
    engine = InterviewEngine(_matcher(), InterviewConfig(), llm=llm, dialog_context=dialog)
    engine._dialog = dialog
    answers = []
    engine.on_answer = answers.append
    engine._rescue_question_worker("как настроить докер", _final("как настроить докер"), engine._generation)
    assert answers == []


def test_question_rescue_ignores_other_type():
    """LLM сказал type=other → не превращаем в вопрос."""
    from mockingbird.kb.dialog_context import DialogContextManager

    llm = _FakeRescueLlm({"type": "other", "resolved_query": "", "answer_mode": "technical",
                          "confidence": 0.9})
    dialog = DialogContextManager(llm=llm)
    engine = InterviewEngine(_matcher(), InterviewConfig(), llm=llm, dialog_context=dialog)
    engine._dialog = dialog
    answers = []
    engine.on_answer = answers.append
    engine._rescue_question_worker("я пошёл домой", _final("я пошёл домой"), engine._generation)
    assert answers == []


class _StubDialog:
    def add_utterance(self, text, speaker="them"):
        pass

    def history_text(self):
        return ""


def test_shift_utterance_still_runs_rescue():
    """Shift-реплика не глотает вложенный вопрос: rescue запускается.

    Раньше is_shift делал ранний return до Tier 3, из-за чего «поговорим про
    X, чем Pod отличается от Y» терялся целиком. Теперь rescue обязан
    запускаться и для shift-фраз (topical-fallback при этом подавлен).
    """
    from unittest.mock import patch

    engine = InterviewEngine(_matcher(), InterviewConfig())
    engine._dialog = _StubDialog()  # dialog present → rescue branch reachable
    started = []

    def fake_thread(*a, **kw):
        started.append(kw.get("target"))

        class _T:
            def start(self):
                pass

            def is_alive(self):
                return False

        return _T()

    with patch.object(engine, "_question_rescue_available", return_value=True), patch(
        "mockingbird.kb.interview_engine.threading.Thread", side_effect=fake_thread
    ):
        engine._process(_final("давай поговорим про kubernetes а потом сравним"))
    assert started, "rescue thread must be launched for a shift utterance"


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
        InterviewConfig(),
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
    engine._question_queue.stop(timeout=2)
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


# NOTE: the interactive `_maybe_predict` LLM-prediction path was removed from
# InterviewEngine (offline next_questions_from_view covers the UI needs);
# the three former predict-gate tests were deleted with it.


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


def test_engine_answer_llm_worker_empty_carries_kb_fallback():
    """Пустой ответ + KB-fallback → done-сообщение несёт текст KB-блока."""
    llm = _FakeAnswerLlm(None)
    engine = InterviewEngine(_matcher(), InterviewConfig(), llm=llm)
    out = []
    engine.on_llm_answer = out.append
    engine._answer_llm_worker("q", "docker", "Docker", "", kb_fallback="**Вопрос**\n\nОтвет из базы")
    assert len(out) == 1
    assert out[0].done is True
    assert out[0].answer == ""
    assert out[0].kb_fallback == "**Вопрос**\n\nОтвет из базы"


def test_engine_answer_llm_worker_nonempty_drops_kb_fallback():
    """Непустой ответ НЕ должен нести kb_fallback (LLM победил)."""
    llm = _FakeAnswerLlm("Ответ ИИ")
    engine = InterviewEngine(_matcher(), InterviewConfig(), llm=llm)
    out = []
    engine.on_llm_answer = out.append
    engine._answer_llm_worker("q", "docker", "Docker", "", kb_fallback="**Вопрос**\n\nОтвет из базы")
    assert out[0].answer == "Ответ ИИ"
    assert out[0].kb_fallback == ""


def test_kb_fallback_text_from_view_blocks():
    """_kb_fallback_text собирает вопрос+ответ из top-блока."""
    engine = InterviewEngine(_matcher(), InterviewConfig(), llm=_FakeAnswerLlm(None))
    view = protocol.KnowledgeView(
        topic="docker",
        title="Docker",
        matched_query="q",
        blocks=[
            protocol.AnswerBlock(id="b1", section="s", question="Вопрос", answer="Ответ"),
        ],
    )
    assert engine._kb_fallback_text(view) == "**Вопрос**\n\nОтвет"


def test_kb_fallback_text_empty_without_blocks():
    engine = InterviewEngine(_matcher(), InterviewConfig(), llm=_FakeAnswerLlm(None))
    view = protocol.KnowledgeView(topic="general", title="", matched_query="q", blocks=[])
    assert engine._kb_fallback_text(view) == ""


class _FakeStreamLlm:
    available = True

    def __init__(self, deltas, pad_to_min_chars: bool = True):
        self.deltas = deltas
        self.calls = []
        # The engine retries streams shorter than _LLM_MIN_ANSWER_CHARS (200).
        # Tests that count calls expect exactly one stream — pad the payload
        # past the retry threshold unless the test opts out.
        if pad_to_min_chars:
            total = sum(len(d) for d in deltas)
            if 0 < total < 200:
                filler = " Полновесный ответ на технический вопрос для порога ретрая. " * 8
                self.deltas = list(deltas) + [filler]

    def answer_question_stream(self, question, context="", mode="technical", previous_qa="", **kw):
        self.calls.append((question, context))
        for delta in self.deltas:
            yield delta


def test_engine_answer_llm_worker_streams_chunks():
    # Answer long enough to be above the retry threshold (200 chars).
    long_text = "Полный ответ на вопрос о разнице entrypoint и cmd " * 6
    deltas = [long_text[:40], long_text[40:]]
    llm = _FakeStreamLlm(deltas)
    engine = InterviewEngine(_matcher(), InterviewConfig(), llm=llm)
    out = []
    engine.on_llm_answer = out.append
    engine._answer_llm_worker("в чём отличие entrypoint от cmd", "docker", "Docker", "контекст")
    assert llm.calls == [("в чём отличие entrypoint от cmd", "контекст")]
    chunks = [m for m in out if not m.done]
    assert [c.delta for c in chunks] == deltas
    assert all(not c.answer for c in chunks)
    final = out[-1]
    assert final.done is True
    assert final.answer == long_text
    assert final.query == "в чём отличие entrypoint от cmd"
    assert final.topic == "docker"
    assert final.title == "Docker"


def test_engine_answer_llm_worker_retries_broken_short_stream():
    """P-015: a stream that dies after a couple of chunks triggers one retry."""
    calls = {"n": 0}

    class _BrokenThenGood:
        available = True

        def answer_question_stream(self, question, context="", mode="technical", previous_qa="", **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                yield "ко"
            else:
                good = "Полный ответ после обрыва стрима, вторая попытка успешна. " * 5
                for i in range(0, len(good), 40):
                    yield good[i : i + 40]

    llm = _BrokenThenGood()
    engine = InterviewEngine(_matcher(), InterviewConfig(), llm=llm)
    out = []
    engine.on_llm_answer = out.append
    engine._answer_llm_worker("q", "docker", "Docker", "")
    assert calls["n"] == 2  # broken first stream + one retry
    final = out[-1]
    assert final.done is True
    assert len(final.answer) >= 200


def test_engine_answer_llm_worker_stream_skips_empty_deltas():
    llm = _FakeStreamLlm(["a", "", "b"], pad_to_min_chars=False)
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
    engine._question_queue.stop(timeout=2)
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
    engine._question_queue.stop(timeout=2)
    engine._maybe_answer_llm(_answer_view(), query)
    engine._question_queue.stop(timeout=2)
    assert len(llm.calls) == 2


def test_engine_answer_cache_does_not_store_empty():
    llm = _FakeAnswerLlm(None)
    engine = InterviewEngine(
        _matcher(), InterviewConfig(answer_cooldown_s=0.0), llm=llm
    )
    query = "в чём отличие entrypoint от cmd"
    engine._maybe_answer_llm(_answer_view(), query)
    engine._question_queue.stop(timeout=2)
    engine._maybe_answer_llm(_answer_view(), query)
    engine._question_queue.stop(timeout=2)
    assert len(llm.calls) == 2


# -- early answer on partial transcripts ---------------------------------------


def _partial(text, segment_id="seg1"):
    return protocol.PartialTranscript(segment_id=segment_id, text=text)


def _partial_engine(llm=None, **overrides):
    overrides.setdefault("use_partials", True)
    overrides.setdefault("subject_llm", False)
    return InterviewEngine(_matcher(), InterviewConfig(**overrides), llm=llm)


def test_engine_partial_early_start_streams_answer():
    llm = _FakeStreamLlm(["Ответ", " ", "по ранней гипотезе"])
    engine = _partial_engine(llm)
    out = []
    engine.on_llm_answer = out.append
    engine._process_partial(_partial("в чём отличие entrypoint от cmd"))
    engine._process_partial(_partial("в чём отличие entrypoint от cmd"))
    engine._question_queue.stop(timeout=2)
    assert engine._provisional_query == "в чём отличие entrypoint от cmd"
    assert llm.calls
    assert llm.calls[0][0] == "в чём отличие entrypoint от cmd"
    assert any(m.done and m.answer.startswith("Ответ по ранней гипотезе") for m in out)


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
    engine._question_queue.stop(timeout=2)
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
    engine._process(_final(q2))
    engine._question_queue.stop(timeout=2)
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
    engine._process(_final(q2))
    engine._question_queue.stop(timeout=2)
    # the final only adds filler words -> the early stream is kept
    assert [c[0] for c in llm.calls] == [q1]


def test_engine_partial_final_semantic_change_restarts_answer():
    q1 = "что такое kubectl"
    q2 = "как масштабировать kubelet"
    llm = _FakeStreamLlm(["a"])
    engine = _partial_engine(llm, answer_cooldown_s=1000.0)
    engine._process_partial(_partial(q1))
    engine._process_partial(_partial(q1))
    engine._process(_final(q2))
    engine._question_queue.stop(timeout=2)
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
    engine._question_queue.stop(timeout=2)
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


# -- protocol roundtrip ------------------------------------------------------


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

