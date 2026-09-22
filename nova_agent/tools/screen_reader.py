def _tesseract_available(pytesseract_module) -> bool:
    """Report whether the Tesseract binary can actually be executed.

    Test doubles do not expose ``get_tesseract_version``; in that case defer to
    the real call instead of failing the check.
    """
    version_lookup = getattr(pytesseract_module, "get_tesseract_version", None)
    if version_lookup is None:
        return True
    try:
        version_lookup()
    except Exception:  # noqa: BLE001 -- any lookup failure means OCR is unavailable
        return False
    return True


def read_screen(tesseract_path: str | None = None, max_characters: int = 500) -> str:
    try:
        import pytesseract
        from PIL import ImageGrab
    except ImportError as exc:
        raise RuntimeError("Screen reading requires Pillow and pytesseract.") from exc

    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    if not _tesseract_available(pytesseract):
        raise RuntimeError(
            "Tesseract OCR is not installed. Install it and set NOVA_TESSERACT_PATH."
        )
    image = ImageGrab.grab()
    text = pytesseract.image_to_string(image).strip()
    if not text:
        return "No readable text found."
    return text[:max_characters]
