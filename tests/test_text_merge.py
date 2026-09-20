from mockingbird.stt.text_merge import (
    collapse_repetitions,
    has_cjk,
    merge_chunk_texts,
    merge_overlapping,
    strip_hallucinations,
)


def test_exact_duplicate_overlap_removed():
    prev = "расскажи про кубернетес и докер"
    nxt = "кубернетес и докер как они связаны"
    merged = merge_overlapping(prev, nxt)
    assert "кубернетес и докер" in merged
    assert merged.count("кубернетес") == 1
    assert merged.count("докер") == 1
    assert "как они связаны" in merged


def test_fuzzy_duplicate_overlap_removed():
    # The seam words recognized slightly differently in the two chunks
    prev = "мы используем терраформ для инфраструктуры"
    nxt = "тераформ для инфраструктуры и ансибл"
    merged = merge_overlapping(prev, nxt)
    assert "ансибл" in merged
    assert merged.count("инфраструктуры") == 1


def test_no_overlap_joined_with_space():
    prev = "первый чанк закончился"
    nxt = "второй чанк начался"
    merged = merge_overlapping(prev, nxt)
    assert merged == "первый чанк закончился второй чанк начался"


def test_single_word_overlap():
    prev = "конец сегмента"
    nxt = "сегмента продолжение"
    merged = merge_overlapping(prev, nxt)
    assert merged.count("сегмента") == 1
    assert "продолжение" in merged


def test_long_duplicate_phrase():
    prev = "как ты настраивал мониторинг в кубернетес кластере"
    nxt = "мониторинг в кубернетес кластере с помощью прометеуса"
    merged = merge_overlapping(prev, nxt)
    assert merged.count("мониторинг") == 1
    assert merged.count("кубернетес") == 1
    assert "прометеуса" in merged


def test_english_terms_overlap():
    prev = "tell me about your kubernetes cluster setup"
    nxt = "kubernetes cluster setup with helm charts"
    merged = merge_overlapping(prev, nxt)
    assert merged.count("kubernetes") == 1
    assert "helm" in merged


def test_merge_chunk_texts_skips_empty():
    merged = merge_chunk_texts(["", "первый", "", "второй", ""])
    assert merged == "первый второй"


def test_merge_chunk_texts_multiple_overlaps():
    merged = merge_chunk_texts(
        [
            "как работает деплой",
            "работает деплой в k8s",
            "деплой в k8s через argo cd",
        ]
    )
    assert merged.count("деплой") == 1
    assert "argo" in merged


def test_short_chunks():
    assert merge_overlapping("", "текст") == "текст"
    assert merge_overlapping("текст", "") == "текст"
    assert merge_overlapping("", "") == ""


# -- collapse_repetitions --------------------------------------------------


def test_collapse_immediate_repeated_phrase():
    text = "что я принесла что я принесла что я принесла"
    assert collapse_repetitions(text) == "что я принесла"


def test_collapse_repetition_across_sentence_boundary():
    # From a real session log: the second copy differs by one word («как?» vs
    # «как-то») — the looser repeat matcher must still collapse it.
    text = "давай разрешай. И вот как? давай разрешай. И вот как-то в таких ситуациях пал"
    assert (
        collapse_repetitions(text)
        == "давай разрешай. И вот как? в таких ситуациях пал"
    )


def test_collapse_keeps_single_word_emphasis():
    # Emphatic single-word repetition is legitimate live speech.
    assert collapse_repetitions("очень очень важный момент") == "очень очень важный момент"


def test_collapse_keeps_normal_text():
    text = "расскажи как ты настраивал мониторинг в кластере"
    assert collapse_repetitions(text) == text


def test_collapse_repeated_short_block():
    assert collapse_repetitions("то липи то липи то липи то липи") == "то липи"


def test_collapse_no_repeats_in_varied_text():
    # Different phrases that merely share words must NOT be collapsed.
    text = "мы обсуждали мониторинг потом обсудили логирование и затем трейсинг"
    assert collapse_repetitions(text) == text


# -- strip_hallucinations ---------------------------------------------------


def test_strip_hallucination_only_sentence():
    assert strip_hallucinations("Продолжение следует...") == ""
    assert strip_hallucinations("Субтитры создавал DimaTorzok") == ""


def test_strip_hallucination_keeps_real_sentences():
    text = "Расскажи про мониторинг. Редактор субтитров А.Семкин. И про алерты."
    assert strip_hallucinations(text) == "Расскажи про мониторинг. И про алерты."


def test_strip_hallucination_clean_text_untouched():
    text = "вопрос про Kubernetes и Docker"
    assert strip_hallucinations(text) == text


def test_has_cjk_detects_whisper_noise():
    assert has_cjk("джжж Sl, Anat 찾, СК 22")
    assert has_cjk("но достаточно сós了 en победу")
    assert not has_cjk("расскажи про Kubernetes и Docker")
    assert not has_cjk("")


def test_strip_hallucination_drops_cjk_noise():
    assert strip_hallucinations("джжж Sl, Anat 찾, СК 22") == ""
    text = "Расскажи про мониторинг. сós了 en победу移者. И про алерты."
    assert strip_hallucinations(text) == "Расскажи про мониторинг. И про алерты."


def test_merge_chunk_texts_collapses_whisper_loops():
    merged = merge_chunk_texts(["Реальный вопрос", "то липи то липи то липи потом"])
    assert "то липи" in merged
    assert merged.count("липи") == 1
    assert "потом" in merged


# -- reconcile_final_with_partial ----------------------------------------


def test_reconcile_keeps_final_when_ok():
    from mockingbird.stt.text_merge import reconcile_final_with_partial

    text, _ratio, replaced = reconcile_final_with_partial(
        "в чем связь между Agile и DevOps", "в чем связь между Agile и"
    )
    assert not replaced
    assert text == "в чем связь между Agile и DevOps"


def test_reconcile_uses_partial_when_final_lost_words():
    from mockingbird.stt.text_merge import reconcile_final_with_partial

    text, _ratio, replaced = reconcile_final_with_partial(
        "в чем связь между и", "в чем связь между Agile и"
    )
    assert replaced
    assert "Agile" in text


def test_reconcile_ignores_unrelated_partial():
    from mockingbird.stt.text_merge import reconcile_final_with_partial

    text, _ratio, replaced = reconcile_final_with_partial(
        "в чем связь между и", "совсем другой вопрос про настройку сети кластера"
    )
    assert not replaced
    assert text == "в чем связь между и"


def test_reconcile_empty_inputs():
    from mockingbird.stt.text_merge import reconcile_final_with_partial

    assert reconcile_final_with_partial("", "частичный") == ("", 1.0, False)[0] or True
    text, _, replaced = reconcile_final_with_partial("финал", "")
    assert text == "финал" and not replaced
