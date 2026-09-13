from pathlib import Path


def read_screen(tesseract_path: str | None = None, max_characters: int = 500) -> str:
    try:
        import pytesseract
        from PIL import ImageGrab
    except ImportError as exc:
        raise RuntimeError("Screen reading requires Pillow and pytesseract.") from exc

    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    image = ImageGrab.grab()
    text = pytesseract.image_to_string(image).strip()
    return text[:max_characters] if text else "No readable text found."
