"""Quick API smoke test — chat, upload, documents, query."""
import httpx
import sys
import time

BASE = "http://127.0.0.1:8000"


def test_health():
    r = httpx.get(f"{BASE}/health", timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    print("[PASS] Health check")


def test_chat():
    body = {
        "provider": "zhipu",
        "messages": [{"role": "user", "content": "说一句话：今天天气真好"}],
        "temperature": 0.0,
    }
    r = httpx.post(f"{BASE}/chat", json=body, timeout=60)
    assert r.status_code == 200
    data = r.json()
    assert data["content"]
    print(f"[PASS] Chat (zhipu/glm-5.1): {data['content'][:50]}...")


def test_upload():
    with open("data/documents/9af7513d3518.pdf", "rb") as f:
        r = httpx.post(
            f"{BASE}/upload",
            files={"file": ("test.pdf", f, "application/pdf")},
            timeout=120,
        )
    assert r.status_code == 200
    data = r.json()
    print(f"[PASS] Upload: {data['filename']}, {data['chunks_count']} chunks, parser={data['parser_used']}")


def test_documents():
    r = httpx.get(f"{BASE}/documents", timeout=10)
    assert r.status_code == 200
    data = r.json()
    print(f"[PASS] Documents: {data['total']} docs")


def test_query():
    body = {"question": "文档中提到了哪些关于人工智能的内容？", "provider": "zhipu"}
    r = httpx.post(f"{BASE}/query", json=body, timeout=120)
    assert r.status_code == 200
    data = r.json()
    answer = data["answer"][:80] if data["answer"] else "(empty)"
    sources = len(data.get("sources", []))
    print(f"[PASS] Query: answer={answer}..., sources={sources}")


def test_sessions():
    # Create session
    r = httpx.post(f"{BASE}/sessions", json={"title": "test session"}, timeout=10)
    assert r.status_code in (200, 201)
    session = r.json()
    sid = session["id"]
    print(f"[PASS] Create session: {sid}")

    # Add message
    r = httpx.post(
        f"{BASE}/sessions/{sid}/messages",
        json={"role": "user", "content": "hello"},
        timeout=10,
    )
    assert r.status_code in (200, 201)
    print(f"[PASS] Add message to session")

    # Get session
    r = httpx.get(f"{BASE}/sessions/{sid}", timeout=10)
    assert r.status_code == 200
    print(f"[PASS] Get session: {len(r.json()['messages'])} messages")

    # List sessions
    r = httpx.get(f"{BASE}/sessions", timeout=10)
    assert r.status_code == 200
    print(f"[PASS] List sessions: {r.json()['total']} sessions")

    # Delete session
    r = httpx.delete(f"{BASE}/sessions/{sid}", timeout=10)
    assert r.status_code in (200, 204)
    print(f"[PASS] Delete session")


def main():
    print("=" * 50)
    print("AI Assistant API Smoke Test")
    print("=" * 50)

    tests = [
        ("Health", test_health),
        ("Chat", test_chat),
        ("Upload", test_upload),
        ("Documents", test_documents),
        ("Query", test_query),
        ("Sessions", test_sessions),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            failed += 1

    print()
    print(f"Results: {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
