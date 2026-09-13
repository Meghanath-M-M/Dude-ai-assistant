# Nova Local Voice Agent

A local-first desktop voice assistant for Windows. The project is being built in phases:

1. Verify Python, microphone, and configuration.
2. Add speech-to-text, intent routing, and Kokoro TTS.
3. Add wake word and hands-free recording.
4. Add SQLite context, screen OCR, safety confirmations, and the HUD.

## Quick start

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m nova_agent --check
```

For the microphone test:

```powershell
python -m tests.test_mic
```

Optional Windows tools:

- Install Tesseract OCR and set `NOVA_TESSERACT_PATH` if it is not in the default location.
- Install Playwright Chromium with `playwright install chromium` when browser automation is enabled.

The current starter is intentionally safe: it does not launch applications or execute destructive commands yet.
