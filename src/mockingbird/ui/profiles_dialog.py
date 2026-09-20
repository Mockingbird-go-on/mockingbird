"""Редактор профилей специализаций.

Bundled-профили (из assets) — read-only; «Создать копию» переносит их в
``~/.mockingbird/profiles`` и делает редактируемыми. Пользовательские профили
создаются, правятся и удаляются свободно. Смена профиля применяется сразу
(персона-промпты + глоссарий), без перезапуска.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from mockingbird.profiles.loader import (
    Profile,
    delete_profile,
    load_profiles,
    profiles_dir,
    save_profile,
)


class ProfilesDialog(QDialog):
    def __init__(self, current_id: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Профили специализаций")
        self.resize(860, 520)
        self.current_id = current_id
        self.profile_changed = False  # True when the selected profile was edited

        self._list = QListWidget()
        self._title = QLineEdit()
        self._persona = QTextEdit()
        self._persona.setAcceptRichText(False)
        self._persona.setFixedHeight(64)
        self._persona_senior = QTextEdit()
        self._persona_senior.setAcceptRichText(False)
        self._persona_senior.setFixedHeight(80)
        self._stack = QTextEdit()
        self._stack.setAcceptRichText(False)
        self._stack.setFixedHeight(80)
        self._glossary = QLineEdit()
        self._glossary_btn = QPushButton("Обзор…")
        self._glossary_btn.clicked.connect(self._pick_glossary)

        self._btn_new = QPushButton("Новый…")
        self._btn_clone = QPushButton("Создать копию")
        self._btn_delete = QPushButton("Удалить")
        self._btn_save = QPushButton("Сохранить")
        self._btn_close = QPushButton("Закрыть")
        self._btn_new.clicked.connect(self._on_new)
        self._btn_clone.clicked.connect(self._on_clone)
        self._btn_delete.clicked.connect(self._on_delete)
        self._btn_save.clicked.connect(self._on_save)
        self._btn_close.clicked.connect(self.accept)

        self._list.currentItemChanged.connect(self._on_select)
        self._reload()

        # left: list + actions; right: form
        left = QVBoxLayout()
        left.addWidget(QLabel("Профили"))
        left.addWidget(self._list)
        actions = QHBoxLayout()
        actions.addWidget(self._btn_new)
        actions.addWidget(self._btn_clone)
        actions.addWidget(self._btn_delete)
        actions.addStretch(1)
        wrap = QWidget()
        wrap.setLayout(actions)
        left.addWidget(wrap)

        form = QFormLayout()
        form.addRow("Название", self._title)
        form.addRow("Персона (кратко)", self._persona)
        form.addRow("Персона (senior, от первого лица)", self._persona_senior)
        form.addRow("Стек / технологии", self._stack)
        g = QHBoxLayout()
        g.addWidget(self._glossary, stretch=1)
        g.addWidget(self._glossary_btn)
        gw = QWidget()
        gw.setLayout(g)
        form.addRow("Глоссарий (YAML, опц.)", gw)

        right = QVBoxLayout()
        right.addLayout(form)
        right.addStretch(1)
        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self._btn_save)
        buttons.addWidget(self._btn_close)
        bw = QWidget()
        bw.setLayout(buttons)
        right.addWidget(bw)

        row = QHBoxLayout()
        lw = QWidget()
        lw.setLayout(left)
        lw.setMaximumWidth(300)
        row.addWidget(lw, stretch=0)
        rw = QWidget()
        rw.setLayout(right)
        row.addWidget(rw, stretch=1)
        self.setLayout(row)

    # --- data ---

    def _reload(self, select_id: str | None = None) -> None:
        profiles = load_profiles()
        self._list.clear()
        ids = sorted(profiles, key=lambda i: (not profiles[i].user_defined, profiles[i].title))
        for pid in ids:
            prof = profiles[pid]
            label = prof.title + ("" if prof.user_defined else " 🔒")
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, pid)
            self._list.addItem(item)
        target = select_id or self.current_id
        for i in range(self._list.count()):
            if self._list.item(i).data(Qt.ItemDataRole.UserRole) == target:
                self._list.setCurrentRow(i)
                return
        if self._list.count():
            self._list.setCurrentRow(0)

    def _current(self) -> Profile | None:
        item = self._list.currentItem()
        if item is None:
            return None
        profiles = load_profiles()
        return profiles.get(item.data(Qt.ItemDataRole.UserRole))

    # --- slots ---

    def _on_select(self, current, _previous) -> None:
        prof = self._current()
        self._set_editable(prof is not None and prof.user_defined)
        if prof is None:
            return
        self._title.setText(prof.title)
        self._persona.setPlainText(prof.persona)
        self._persona_senior.setPlainText(prof.persona_senior)
        self._stack.setPlainText(prof.stack)
        self._glossary.setText(prof.glossary or "")

    def _set_editable(self, editable: bool) -> None:
        for w in (self._title, self._persona, self._persona_senior, self._stack, self._glossary, self._glossary_btn):
            w.setEnabled(editable)
        self._btn_save.setEnabled(editable)
        self._btn_delete.setEnabled(editable)

    def _on_new(self) -> None:
        pid = self._prompt_id()
        if not pid:
            return
        prof = Profile(
            id=pid,
            title=pid.capitalize(),
            persona=f"специалист с 5-летним опытом",
            persona_senior=f"специалист 6+ лет",
            stack="",
        )
        save_profile(prof)
        self._reload(select_id=pid)
        self.profile_changed = True

    def _on_clone(self) -> None:
        prof = self._current()
        if prof is None:
            return
        pid = self._prompt_id(default=f"{prof.id}-copy")
        if not pid:
            return
        copy = Profile(
            id=pid,
            title=prof.title,
            persona=prof.persona,
            persona_senior=prof.persona_senior,
            stack=prof.stack,
            glossary=prof.glossary,
            calibrated=False,
        )
        save_profile(copy)
        self._reload(select_id=pid)
        self.profile_changed = True

    def _prompt_id(self, default: str = "") -> str:
        from PySide6.QtWidgets import QInputDialog

        text, ok = QInputDialog.getText(
            self, "Идентификатор", "Латинский id профиля (имя файла):", text=default
        )
        pid = text.strip().lower().replace(" ", "-")
        if not ok or not pid or not pid.replace("-", "").isascii() or not pid.replace("-", "").isalnum():
            if ok:
                QMessageBox.warning(self, "Профиль", "Id должен быть латиницей/цифрами/дефисом.")
            return ""
        return pid

    def _on_delete(self) -> None:
        prof = self._current()
        if prof is None or not prof.user_defined:
            return
        if QMessageBox.question(
            self, "Удалить профиль", f"Удалить «{prof.title}» ({prof.id})?"
        ) != QMessageBox.StandardButton.Yes:
            return
        delete_profile(prof.id)
        if self.current_id == prof.id:
            self.current_id = "devops"
            self.profile_changed = True
        self._reload()

    def _on_save(self) -> None:
        prof = self._current()
        if prof is None or not prof.user_defined:
            return
        prof.title = self._title.text().strip() or prof.id
        prof.persona = self._persona.toPlainText().strip()
        prof.persona_senior = self._persona_senior.toPlainText().strip()
        prof.stack = self._stack.toPlainText().strip()
        prof.glossary = self._glossary.text().strip() or None
        if not (prof.persona and prof.persona_senior and prof.stack):
            QMessageBox.warning(
                self, "Профиль", "Персона (оба поля) и стек обязательны."
            )
            return
        save_profile(prof)
        self.profile_changed = True
        self._reload(select_id=prof.id)
        if self.current_id == prof.id:
            self.profile_changed = True

    def _pick_glossary(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Выбрать глоссарий", str(profiles_dir().parent), "YAML (*.yaml *.yml)"
        )
        if path:
            self._glossary.setText(path)
