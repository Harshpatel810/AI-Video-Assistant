import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all 

load_dotenv()

source = 'https://www.youtube.com/watch?v=K6ffYKrlPdU'
language = 'hinglish'

chunks = process_input(source)
transcript = transcribe_all(chunks, language=language)

print("\n=== TRANSCRIPT ===\n")
print(transcript)
