"""Test POST /query endpoint."""


def test_query_endpoint_handles_unavailable_store(client):
    """Query endpoint should return 200 with helpful message when deps are missing."""
    resp = client.post("/query", json={
        "question": "测试问题",
        "top_k": 3,
    })
    # 200: either works (store available) or returns graceful error message
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "sources" in data
