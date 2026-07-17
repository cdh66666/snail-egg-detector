"""Small phone-friendly HTTP control panel for MaixCam.

The HTTP thread only records commands. The vision loop applies them, so GPIO
and PWM are never touched from a request handler.
"""

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


HTML = r"""<!doctype html>
<html lang="en"><head><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MaixCam Snail Egg Control</title>
<style>
body{font-family:system-ui,sans-serif;background:#101418;color:#f3f5f7;margin:0;padding:14px}
main{max-width:720px;margin:auto}h1{font-size:20px;margin:0 0 10px}.card{background:#1c242b;border:1px solid #34424c;border-radius:8px;padding:12px;margin:10px 0}
button{font-size:16px;border:0;border-radius:6px;padding:12px 14px;margin:4px;background:#2e8bdb;color:white;min-width:94px}
button.warn{background:#d9822b}button.danger{background:#c83737}button.dark{background:#53616b}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}.status{font-family:ui-monospace,monospace;white-space:pre-wrap;font-size:13px;line-height:1.5}
img{width:100%;background:#050607;border-radius:6px;display:block}small{color:#aab7c0}
</style></head><body><main>
<h1>MaixCam Snail Egg Control</h1>
<div class="card"><small>Only the low-power red aiming light is controlled here. The high-power laser remains manual.</small></div>
<div class="card"><div class="status" id="status">Connecting...</div></div>
<div class="card"><img id="shot" src="/snapshot.jpg" alt="camera snapshot"></div>
<div class="card"><button onclick="cmd('auto')">Auto Track</button><button class="dark" onclick="cmd('hold')">Hold</button><button class="warn" onclick="cmd('center')">Center</button>
<div class="grid"><button onclick="cmd('pan_left')">Pan -</button><button onclick="cmd('tilt_up')">Tilt +</button><button onclick="cmd('pan_right')">Pan +</button><button onclick="cmd('tilt_down')">Tilt -</button><button class="dark" onclick="cmd('aim_auto')">Aim Auto</button><button class="danger" onclick="cmd('emergency')">EMERGENCY</button></div></div>
<div class="card"><button class="warn" onclick="cmd('aim_on')">Red Aim ON</button><button class="dark" onclick="cmd('aim_off')">Red Aim OFF</button><button class="dark" onclick="cmd('clear_estop')">Clear Stop</button></div>
<div class="card"><small>Backup experiment: broad pink-color candidates, not the YOLO detector.</small><br><button class="warn" onclick="cmd('color_on')">Color Candidate ON</button><button class="dark" onclick="cmd('color_off')">Color Candidate OFF</button></div>
<script>
const token=new URLSearchParams(location.search).get('token')||'maixcam';
async function cmd(c){await fetch('/api/action?cmd='+encodeURIComponent(c)+'&token='+encodeURIComponent(token)); refresh();}
async function refresh(){try{let r=await fetch('/api/status?token='+encodeURIComponent(token));document.querySelector('#status').textContent=JSON.stringify(await r.json(),null,2);document.querySelector('#shot').src='/snapshot.jpg?t='+Date.now();}catch(e){document.querySelector('#status').textContent='offline: '+e;}}
setInterval(refresh,1200);refresh();
</script></main></body></html>"""


class WebControl:
    def __init__(self, host="0.0.0.0", port=8000, snapshot_path="/root/snail_egg/debug/web_latest.jpg"):
        self.host = host
        self.port = int(port)
        self.snapshot_path = snapshot_path
        self.token = os.environ.get("MAIX_WEB_TOKEN", "maixcam")
        self.lock = threading.Lock()
        self.mode = "auto"
        self.pan = 0.0
        self.tilt = 0.0
        self.aim_override = None
        self.estop = False
        self.status = {"web": "starting"}
        self.snapshot_requested = False
        self.snapshot_ready = threading.Event()
        control = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                return

            def send_bytes(self, data, content_type):
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def authorized(self, query):
                return query.get("token", [control.token])[0] == control.token

            def do_GET(self):
                parsed = urlparse(self.path)
                query = parse_qs(parsed.query)
                if parsed.path == "/":
                    self.send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
                    return
                if not self.authorized(query):
                    self.send_error(403, "bad token")
                    return
                if parsed.path == "/api/status":
                    self.send_bytes(json.dumps(control.get_status()).encode("utf-8"), "application/json")
                    return
                if parsed.path == "/api/action":
                    control.apply_command(query.get("cmd", [""])[0])
                    self.send_bytes(b'{"ok":true}', "application/json")
                    return
                if parsed.path == "/snapshot.jpg":
                    control.request_frame()
                    control.snapshot_ready.wait(0.45)
                    try:
                        with open(control.snapshot_path, "rb") as image_file:
                            self.send_bytes(image_file.read(), "image/jpeg")
                    except OSError:
                        self.send_error(503, "snapshot not ready")
                    return
                self.send_error(404)

        self.server = ThreadingHTTPServer((self.host, self.port), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self):
        self.thread.start()
        print("WEB_CONTROL,LISTEN,0.0.0.0,%d,TOKEN,%s" % (self.port, self.token))

    def stop(self):
        try:
            self.server.shutdown()
        except Exception:
            pass

    def apply_command(self, command):
        with self.lock:
            if command == "auto":
                self.mode = "auto"
                self.estop = False
            elif command == "hold":
                self.mode = "hold"
            elif command == "center":
                self.mode = "manual"
                self.pan = 0.0
                self.tilt = 0.0
            elif command == "pan_left":
                self.mode = "manual"
                self.pan = max(-30.0, self.pan - 5.0)
            elif command == "pan_right":
                self.mode = "manual"
                self.pan = min(30.0, self.pan + 5.0)
            elif command == "tilt_up":
                self.mode = "manual"
                self.tilt = min(30.0, self.tilt + 5.0)
            elif command == "tilt_down":
                self.mode = "manual"
                self.tilt = max(-30.0, self.tilt - 5.0)
            elif command == "aim_auto":
                self.aim_override = None
            elif command == "aim_on":
                self.aim_override = True
            elif command == "aim_off":
                self.aim_override = False
            elif command == "emergency":
                self.mode = "hold"
                self.aim_override = False
                self.estop = True
            elif command == "clear_estop":
                self.estop = False
            elif command == "color_on":
                try:
                    os.makedirs("/root/snail_egg", exist_ok=True)
                    with open("/root/snail_egg/enable_color_only", "w") as flag:
                        flag.write("1")
                except OSError:
                    pass
            elif command == "color_off":
                try:
                    os.remove("/root/snail_egg/enable_color_only")
                except OSError:
                    pass

    def get_control(self):
        with self.lock:
            return self.mode, self.pan, self.tilt, self.aim_override, self.estop

    def update_status(self, values):
        with self.lock:
            self.status = dict(values)

    def get_status(self):
        with self.lock:
            return dict(self.status, mode=self.mode, pan=self.pan, tilt=self.tilt,
                        aim_override=self.aim_override, estop=self.estop,
                        color_only=os.path.exists("/root/snail_egg/enable_color_only"))

    def request_frame(self):
        with self.lock:
            self.snapshot_requested = True
            self.snapshot_ready.clear()

    def consume_frame_request(self):
        with self.lock:
            requested = self.snapshot_requested
            self.snapshot_requested = False
            return requested

    def publish_frame(self):
        self.snapshot_ready.set()
