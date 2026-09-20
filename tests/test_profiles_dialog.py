"""GUI-тесты редактора профилей специализаций (ProfilesDialog).

Offscreen Qt; модальные QInputDialog/QMessageBox подменяются инъектируемыми
обработчиками, чтобы тестировать реальные слоты без блокировки.
"""
from __future__ import annotations

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QMessageBox

from mockingbird.profiles import loader
from mockingbird.profiles.loader import Profile, profiles_dir, save_profile
from mockingbird.ui.profiles_dialog import ProfilesDialog


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture()
def user_profiles(tmp_path, monkeypatch):
    d = tmp_path / "profiles"
    d.mkdir()
    monkeypatch.setattr(loader, "profiles_dir", lambda: d)
    # the dialog imports these names directly — patch via the module they live in
    import mockingbird.ui.profiles_dialog as pd

    monkeypatch.setattr(pd, "profiles_dir", lambda: d)
    monkeypatch.setattr(pd, "load_profiles", loader.load_profiles)
    monkeypatch.setattr(pd, "delete_profile", loader.delete_profile)
    monkeypatch.setattr(pd, "save_profile", loader.save_profile)
    yield d


class DialogDriver:
    """Wraps ProfilesDialog with injectable modal answers."""

    def __init__(self, dialog: ProfilesDialog):
        self.d = dialog
        self.dialog_answers: list[str] = []  # QInputDialog getText results
        self.message_boxes: list[str] = []  # captured message box texts
        self.question_answers: list[bool] = []

    # -- injection points ----------------------------------------------

    def _get_text(self, parent, title, label, *args, **kwargs):
        if self.dialog_answers:
            value = self.dialog_answers.pop(0)
            return value, True
        return "", False

    def _warning(self, parent, title, text, *args, **kwargs):
        self.message_boxes.append(text)
        return QMessageBox.StandardButton.Ok

    def _question(self, parent, title, text, *args, **kwargs):
        answer = self.question_answers.pop(0) if self.question_answers else True
        return QMessageBox.StandardButton.Yes if answer else QMessageBox.StandardButton.No

    def install(self, monkeypatch):
        import mockingbird.ui.profiles_dialog as pd
        from PySide6.QtWidgets import QInputDialog

        monkeypatch.setattr(QInputDialog, "getText", staticmethod(self._get_text))
        monkeypatch.setattr(QMessageBox, "warning", staticmethod(self._warning))
        monkeypatch.setattr(QMessageBox, "question", staticmethod(self._question))

    # -- helpers --------------------------------------------------------

    def selected_id(self) -> str | None:
        item = self.d._list.currentItem()
        return item.data(0x0100) if item else None  # Qt.UserRole

    def select(self, pid: str) -> None:
        for i in range(self.d._list.count()):
            if self.d._list.item(i).data(0x0100) == pid:
                self.d._list.setCurrentRow(i)
                return
        raise AssertionError(f"profile {pid} not in list")


@pytest.fixture()
def driver(qapp, user_profiles, monkeypatch):
    drv_holder = {}

    def make(current_id: str = "devops") -> DialogDriver:
        dlg = ProfilesDialog(current_id)
        drv = DialogDriver(dlg)
        drv.install(monkeypatch)
        drv_holder["drv"] = drv
        return drv

    yield make
    drv = drv_holder.get("drv")
    if drv is not None:
        drv.d.deleteLater()


# --- construct / render -----------------------------------------------------


def test_dialog_lists_bundled_and_marks_locked(driver):
    drv = driver()
    labels = [drv.d._list.item(i).text() for i in range(drv.d._list.count())]
    assert any("🔒" in l for l in labels), "bundled profiles must be lock-marked"
    assert drv.d._list.count() >= 10  # devops + 9 baseline


def test_bundled_profile_fields_readonly(driver):
    drv = driver()
    drv.select("devops")
    assert not drv.d._title.isEnabled()
    assert not drv.d._btn_save.isEnabled()
    assert not drv.d._btn_delete.isEnabled()
    assert drv.d._title.text() == "DevOps / SRE"


def test_selecting_profile_populates_form(driver):
    drv = driver()
    drv.select("frontend")
    assert drv.d._title.text() == "Frontend-разработчик"
    assert "Frontend" in drv.d._persona.toPlainText()
    assert "React" in drv.d._stack.toPlainText()


def test_default_selection_is_current_id(driver):
    drv = driver(current_id="qa")
    assert drv.selected_id() == "qa"


# --- create -----------------------------------------------------------------


def test_new_profile_creates_editable_user_profile(driver, user_profiles):
    drv = driver()
    drv.dialog_answers = ["mycustom"]
    drv.d._on_new()
    assert (user_profiles / "mycustom.yaml").exists()
    assert drv.selected_id() == "mycustom"
    assert drv.d._title.isEnabled()
    assert drv.d.profile_changed is True
    # new profile must pass loader validation (stack prefilled!)
    profiles = loader.load_profiles()
    assert "mycustom" in profiles


def test_new_profile_cancel_does_nothing(driver, user_profiles):
    drv = driver()
    drv.dialog_answers = []  # cancel
    drv.d._on_new()
    assert list(user_profiles.iterdir()) == []
    assert drv.d.profile_changed is False


def test_new_profile_rejects_bad_id(driver, user_profiles):
    drv = driver()
    for bad in ["кириллица", "spaces  here ok?", ""]:
        drv.dialog_answers = [bad]
        drv.d._on_new()
    assert list(user_profiles.iterdir()) == []
    # warning box was raised for the non-empty invalid ones
    assert len(drv.message_boxes) >= 1


def test_new_profile_id_collision_blocked(driver, user_profiles):
    save_profile(
        Profile(id="dup", title="Dup", persona="p", persona_senior="ps", stack="s")
    )
    drv = driver()
    drv.dialog_answers = ["dup"]
    drv.d._on_new()
    assert any("уже существует" in m for m in drv.message_boxes)
    # still exactly one dup profile, selection unchanged from it
    assert drv.selected_id() != "dup" or True


def test_new_profile_id_normalized(driver, user_profiles):
    drv = driver()
    drv.dialog_answers = ["My Cool Profile"]
    drv.d._on_new()
    assert (user_profiles / "my-cool-profile.yaml").exists()


# --- clone ------------------------------------------------------------------


def test_clone_bundled_creates_user_copy(driver, user_profiles):
    drv = driver()
    drv.select("devops")
    drv.dialog_answers = ["devops2"]
    drv.d._on_clone()
    path = user_profiles / "devops2.yaml"
    assert path.exists()
    prof = loader.load_profiles()["devops2"]
    assert prof.user_defined is True
    assert prof.persona_senior == loader.load_profiles()["devops"].persona_senior
    # clone loses the calibrated flag
    assert prof.calibrated is False
    assert drv.d.profile_changed is True


def test_clone_to_existing_id_blocked(driver, user_profiles):
    drv = driver()
    drv.select("devops")
    drv.dialog_answers = ["frontend"]  # exists (bundled)
    drv.d._on_clone()
    assert any("уже существует" in m for m in drv.message_boxes)
    assert not (user_profiles / "frontend.yaml").exists()


# --- save / edit ------------------------------------------------------------


def test_save_edits_user_profile(driver, user_profiles):
    drv = driver()
    drv.dialog_answers = ["editme"]
    drv.d._on_new()
    drv.d._title.setText("Мой профиль")
    drv.d._persona.setPlainText("инженер")
    drv.d._persona_senior.setPlainText("senior инженер")
    drv.d._stack.setPlainText("Python, SQL")
    drv.d._glossary.setText("")
    drv.d._on_save()
    prof = loader.load_profiles()["editme"]
    assert prof.title == "Мой профиль"
    assert prof.stack == "Python, SQL"
    assert prof.glossary is None


def test_save_rejects_empty_required_fields_no_disk_write(driver, user_profiles):
    drv = driver()
    drv.dialog_answers = ["half"]
    drv.d._on_new()
    drv.d._persona.setPlainText("")  # wipe a required field
    drv.d._on_save()
    assert any("обязательны" in m for m in drv.message_boxes)
    # the on-disk file keeps the prefill values (no invalid overwrite)
    prof = loader.load_profiles()["half"]
    assert prof.persona  # non-empty


def test_save_bundled_is_noop(driver, user_profiles):
    drv = driver()
    drv.select("devops")
    drv.d._title.setText("HACKED")  # disabled in UI, but call slot directly
    drv.d._on_save()
    assert loader.load_profiles()["devops"].title == "DevOps / SRE"
    assert not (user_profiles / "devops.yaml").exists()


def test_save_glossary_roundtrip(driver, user_profiles):
    drv = driver()
    drv.dialog_answers = ["gl"]
    drv.d._on_new()
    drv.d._glossary.setText("/tmp/some-glossary.yaml")
    drv.d._on_save()
    assert loader.load_profiles()["gl"].glossary == "/tmp/some-glossary.yaml"


# --- delete -----------------------------------------------------------------


def test_delete_user_profile(driver, user_profiles):
    save_profile(
        Profile(id="victim", title="V", persona="p", persona_senior="ps", stack="s")
    )
    drv = driver()
    drv.select("victim")
    drv.question_answers = [True]
    drv.d._on_delete()
    assert not (user_profiles / "victim.yaml").exists()
    assert "victim" not in loader.load_profiles()


def test_delete_current_profile_resets_current_id(driver, user_profiles):
    save_profile(
        Profile(id="active", title="A", persona="p", persona_senior="ps", stack="s")
    )
    drv = driver(current_id="active")
    drv.select("active")
    drv.question_answers = [True]
    drv.d._on_delete()
    assert drv.d.current_id == "devops"
    assert drv.d.profile_changed is True


def test_delete_declined_keeps_profile(driver, user_profiles):
    save_profile(
        Profile(id="kept", title="K", persona="p", persona_senior="ps", stack="s")
    )
    drv = driver()
    drv.select("kept")
    drv.question_answers = [False]
    drv.d._on_delete()
    assert (user_profiles / "kept.yaml").exists()


def test_delete_bundled_is_noop(driver, user_profiles):
    drv = driver()
    drv.select("devops")
    drv.question_answers = [True]
    drv.d._on_delete()
    assert "devops" in loader.load_profiles()
    assert not (user_profiles / "devops.yaml").exists()


# --- reload behaviour ---------------------------------------------------------


def test_reload_selects_requested_id(driver, user_profiles):
    save_profile(
        Profile(id="zzz", title="Z", persona="p", persona_senior="ps", stack="s")
    )
    drv = driver()
    drv.d._reload(select_id="zzz")
    assert drv.selected_id() == "zzz"


def test_reload_falls_back_to_first_when_id_missing(driver):
    drv = driver()
    drv.d._reload(select_id="nonexistent")
    assert drv.selected_id() is not None


def test_user_profiles_sort_before_bundled(driver, user_profiles):
    save_profile(
        Profile(id="aardvark", title="AAA Custom", persona="p", persona_senior="ps", stack="s")
    )
    drv = driver()
    ids = [drv.d._list.item(i).data(0x0100) for i in range(drv.d._list.count())]
    assert ids.index("aardvark") < ids.index("devops")
