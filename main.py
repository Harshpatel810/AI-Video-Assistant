import os
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
# Fix for AVG/Avast antivirus setting an unopenable SSLKEYLOGFILE pipe on Windows
if "SSLKEYLOGFILE" in os.environ and ("avgMonFltProxy" in os.environ["SSLKEYLOGFILE"] or not os.path.exists(os.environ["SSLKEYLOGFILE"])):
    os.environ.pop("SSLKEYLOGFILE", None)

from dotenv import load_dotenv
from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import extract_action_items, extract_key_decisions, extract_questions
from core.rag_engine import build_rag_chain, ask_question
from core.database import save_meeting


load_dotenv()

def run_pipeline(source: str, language: str = "english", save_to_db: bool = True) -> dict:
    print("starting AI Video Assistant")

    wav_path, chunks = process_input(source, return_master_wav=True)

    transcript = transcribe_all(chunks, language)
    print(f"raw transcription (first 300 characters ) {transcript[:300]}")

    title = generate_title(transcript)

    summary = summarize(transcript)

    action_item = extract_action_items(transcript)

    decisions = extract_key_decisions(transcript)
    questions = extract_questions(transcript)
    
    rag_chain = build_rag_chain(transcript)

    meeting_id = None
    if save_to_db:
        try:
            meeting_id = save_meeting(
                transcript=transcript,
                summary=summary,
                audio_path=wav_path,
                title=title,
                source=source,
                language=language,
                action_items=action_item,
                key_decisions=decisions,
                open_questions=questions,
            )
            print(f"Session saved to SQLite database with ID: {meeting_id}")
        except Exception as e:
            print(f"Warning: Failed to save to database: {e}")

    return {
        "id": meeting_id,
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_item,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
        "audio_path": wav_path,
    }

if __name__ == "__main__":
    # CLI entry point
    source = input("Enter YouTube URL or local file path: ").strip()
    language = input("Language (english/hinglish): ").strip() or "english"
    result = run_pipeline(source, language)

    print("\n" + "=" * 60)
    print(f"📌 Title: {result['title']}")
    if result.get("id"):
        print(f"💾 Saved in SQLite (Record #{result['id']})")
    print(f"\n📋 Summary:\n{result['summary']}")
    print(f"\n✅ Action Items:\n{result['action_items']}")
    print(f"\n🔑 Key Decisions:\n{result['key_decisions']}")
    print(f"\n❓ Open Questions:\n{result['open_questions']}")
    print("=" * 60)

    # Phase 2 — Chat with your meeting via RAG
    print("\n💬 Chat with your meeting (type 'exit' to quit)\n")
    rag_chain = result["rag_chain"]
    while True:
        question = input("You: ").strip()
        if question.lower() in ["exit", "quit", "q"]:
            print("👋 Goodbye!") 
            break
        if not question:
            continue
        answer = ask_question(rag_chain, question)
        print(f"\n🤖 Assistant: {answer}\n")
