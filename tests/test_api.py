from fastapi.testclient import TestClient

from app.api import create_app


def test_endpoints(cfg):
    with TestClient(create_app(cfg)) as client:
        health = client.get("/health").json()
        assert health["status"] == "ok" and health["chunks"] > 0

        found = client.get("/search", params={"q": "vpn helpdesk", "k": 3}).json()
        assert found["results"][0]["source"] == "vpn_setup.md"

        asked = client.post("/ask", json={"question": "What is the Pro plan price?"}).json()
        assert asked["confident"] and "999" in asked["answer"]

        unknown = client.post("/ask", json={"question": "quantum chromodynamics lattice simulation"}).json()
        assert unknown["confident"] is False

        assert client.post("/index").json()["documents"] == 6


def test_input_validation(cfg):
    with TestClient(create_app(cfg)) as client:
        assert client.post("/ask", json={"question": "hi"}).status_code == 422
        assert client.get("/search", params={"q": "leave", "mode": "magic"}).status_code == 422
        assert client.get("/search", params={"q": "leave", "k": 99}).status_code == 422
