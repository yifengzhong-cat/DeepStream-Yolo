import base64

def test_abilities_contains_four_algorithms(client):
    response = client.post("/v1/service/abilities", json={})
    assert response.status_code == 200
    body = response.json()
    assert body["resultCode"] == "200"
    assert body["resultValue"]["abilityInfo"]["number"] == 4


def test_video_task_accepts_stream_request(client):
    payload = {
        "algCode": "101001",
        "command": 1,
        "videoInfo": [
            {
                "analyseId": "a-1",
                "devCode": "1003010010000000023",
                "formatType": 0,
                "videoUrl": "rtsp://127.0.0.1/live/stream",
            }
        ],
    }
    response = client.post("/v1/service/videoTask", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["resultCode"] == "200"
    assert body["resultValue"][0]["analyseId"] == "a-1"


def test_image_task_returns_standard_response(client):
    payload = {
        "analyseId": "img-1",
        "algCode": "101003",
        "imageData": base64.b64encode(b"fake-jpg").decode("utf-8"),
    }
    response = client.post("/v1/service/imageTask", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["resultCode"] == "200"
    assert body["resultValue"]["osdImageData"] == payload["imageData"]


def test_keep_alive_returns_dev_id(client):
    response = client.post("/analysis/api/v1/keepAlive", json={"devIP": "127.0.0.1", "devPort": 22266})
    assert response.status_code == 200
    body = response.json()
    assert body["resultCode"] == "200"
    assert "devId" in body["resultValue"]


def test_image_task_still_succeeds_when_runtime_write_fails(client, monkeypatch):
    from app import deepstream_service

    def _raise_oserror(*args, **kwargs):
        raise OSError("simulated write failure")

    monkeypatch.setattr(deepstream_service.Path, "write_bytes", _raise_oserror)
    payload = {
        "analyseId": "img-2",
        "algCode": "101003",
        "imageData": base64.b64encode(b"fake-jpg").decode("utf-8"),
    }
    response = client.post("/v1/service/imageTask", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["resultCode"] == "200"
