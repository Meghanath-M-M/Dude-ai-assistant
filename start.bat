@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    py -3.11 -m venv .venv
)

call .venv\Scripts\activate.bat
python -m pip install -r requirements.txt

rem Whisper on the GPU when cuBLAS is present; Dude retries on CPU by itself
rem if it is not (the field soak's 4s median STT was CPU-bound).
set NOVA_STT_DEVICE=cuda

rem Tier 2 LLM fallback: answers only in the 0.65-0.82 confidence band and
rem stays silent when the Ollama daemon or model is missing.
set NOVA_LLM_FALLBACK=1
python -m nova_agent --listen
