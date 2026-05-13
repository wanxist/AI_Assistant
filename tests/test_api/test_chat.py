def test_chat_mock(client):
    resp = client.post("/chat", json={
        "messages": [{"role": "user", "content": "hello"}],
        "provider": "mock",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert "content" in data
    assert "hello" in data["content"] or "[mock]" in data["content"]
