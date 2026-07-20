from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from maixcam.web_control import HTML, WebControl


def get(url: str):
    try:
        with urlopen(url, timeout=2) as response:
            return response.status, response.headers, response.read()
    except HTTPError as error:
        return error.code, error.headers, error.read()


def main() -> None:
    # Port 0 makes the test safe to run beside a real local web server.
    web = WebControl(host="127.0.0.1", port=0, snapshot_path="does-not-exist.jpg")
    web.start()
    base = f"http://127.0.0.1:{web.port}"
    try:
        status, headers, body = get(base + "/")
        html = body.decode("utf-8")
        assert status == 200
        assert "福寿螺识别云台" in html
        assert "</title>" in html and "</button>" in html
        assert "�" not in html and "锟" not in html
        assert "object-fit:cover" in html
        assert "data-command=\"auto\"" in html

        status, _, _ = get(base + "/api/status")
        assert status == 403
        status, _, body = get(base + "/api/status?token=maixcam")
        assert status == 200
        payload = json.loads(body)
        assert payload["mode"] == "hold"
        assert payload["estop"] is True
        assert payload["aim_override"] is False
        assert payload["pan_limits"] == [-30.0, 30.0]
        assert payload["tilt_limits"] == [-10.0, 30.0]
        assert payload["closed_loop_override"] is None

        for expected in (True, False, None):
            status, _, _ = get(base + "/api/action?cmd=closed_loop_toggle&token=maixcam")
            assert status == 200
            assert web.get_closed_loop_override() is expected

        status, _, _ = get(base + "/api/action?cmd=unknown&token=maixcam")
        assert status == 400
        status, _, _ = get(base + "/api/action?cmd=pan_left&token=maixcam")
        assert status == 400

        status, _, _ = get(base + "/api/action?cmd=nudge_left&token=maixcam")
        assert status == 400
        get(base + "/api/action?cmd=clear_estop&token=maixcam")
        web.update_status({"gimbal_pan": 12.0, "gimbal_tilt": -4.0, "eggs": 3})
        status, _, _ = get(base + "/api/action?cmd=nudge_left&token=maixcam")
        assert status == 200
        mode, pan, tilt, _, _ = web.get_control()
        assert mode == "manual" and pan == 13.0 and tilt == -4.0
        time.sleep(0.13)
        get(base + "/api/action?cmd=nudge_right&token=maixcam")
        _, pan, _, _, _ = web.get_control()
        assert pan == 12.0

        status, _, _ = get(base + "/api/select?x=0.5&y=0.25&token=maixcam")
        assert status == 200
        request = web.consume_selection_request()
        assert request == ("point", 0.5, 0.25)
        status, _, _ = get(base + "/api/select?x=2&y=0&token=maixcam")
        assert status == 400

        get(base + "/api/action?cmd=emergency&token=maixcam")
        assert web.get_control()[4] is True
        status, _, _ = get(base + "/api/select?x=0.5&y=0.5&token=maixcam")
        assert status == 400
        get(base + "/api/action?cmd=clear_estop&token=maixcam")
        assert web.get_control()[4] is False

        results = []
        def read_status():
            results.append(get(base + "/api/status?token=maixcam")[0])
        threads = [threading.Thread(target=read_status) for _ in range(20)]
        for thread in threads: thread.start()
        for thread in threads: thread.join()
        assert results == [200] * 20
        print({"passed": True, "http_status_concurrency": len(results), "port": web.port})
    finally:
        web.stop()


if __name__ == "__main__":
    main()
