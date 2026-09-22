import sys
import types

from nova_agent.tools.screen_reader import read_screen


def test_read_screen_uses_tesseract_path_and_truncates(monkeypatch):
    calls = {}

    class FakeImageGrab:
        @staticmethod
        def grab():
            return object()

    class FakePytesseractModule:
        class _Pytesseract:
            tesseract_cmd = None

        pytesseract = _Pytesseract()

        @staticmethod
        def image_to_string(image):
            calls["image"] = image
            return "A" * 300

    fake_pil = types.ModuleType("PIL")
    fake_pil.ImageGrab = FakeImageGrab
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)
    monkeypatch.setitem(sys.modules, "pytesseract", FakePytesseractModule)

    result = read_screen(tesseract_path="C:/Temp/tesseract.exe", max_characters=120)

    assert result == "A" * 120
    assert calls["image"] is not None


def test_read_screen_handles_empty_result(monkeypatch):
    class FakeImageGrab:
        @staticmethod
        def grab():
            return object()

    class FakePytesseractModule:
        class _Pytesseract:
            tesseract_cmd = None

        pytesseract = _Pytesseract()

        @staticmethod
        def image_to_string(_image):
            return ""

    fake_pil = types.ModuleType("PIL")
    fake_pil.ImageGrab = FakeImageGrab
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)
    monkeypatch.setitem(sys.modules, "pytesseract", FakePytesseractModule)

    result = read_screen()

    assert result == "No readable text found."


def test_read_screen_reports_missing_tesseract_binary(monkeypatch):
    class FakeImageGrab:
        @staticmethod
        def grab():  # pragma: no cover - must not be reached.
            raise AssertionError("screen capture should not run without Tesseract")

    class FakePytesseractModule:
        class _Pytesseract:
            tesseract_cmd = None

        pytesseract = _Pytesseract()

        @staticmethod
        def get_tesseract_version():
            raise OSError("tesseract is not installed")

        @staticmethod
        def image_to_string(_image):  # pragma: no cover - must not be reached.
            raise AssertionError("OCR should not run without Tesseract")

    fake_pil = types.ModuleType("PIL")
    fake_pil.ImageGrab = FakeImageGrab
    monkeypatch.setitem(sys.modules, "PIL", fake_pil)
    monkeypatch.setitem(sys.modules, "pytesseract", FakePytesseractModule)

    try:
        read_screen()
    except RuntimeError as exc:
        assert "Tesseract" in str(exc)
    else:  # pragma: no cover - the check must fail loudly.
        raise AssertionError("expected a RuntimeError when Tesseract is missing")
