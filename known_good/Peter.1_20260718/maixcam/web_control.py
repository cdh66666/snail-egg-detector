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
<html lang="zh-CN"><head><meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>福寿螺识别云台</title>
<style>
:root{color-scheme:dark;--bg:#0d1114;--panel:#171d21;--line:#344047;--text:#f2f5f6;--muted:#a9b4ba;--green:#18a76f;--amber:#d68a25;--red:#d84a4a;--blue:#397fb8}
*{box-sizing:border-box}body{font-family:system-ui,-apple-system,"PingFang SC",sans-serif;background:var(--bg);color:var(--text);margin:0;padding:12px;overscroll-behavior:none}
main{max-width:760px;margin:auto}header{display:flex;align-items:center;justify-content:space-between;margin:2px 0 10px}h1{font-size:19px;margin:0;letter-spacing:0}.dot{width:10px;height:10px;border-radius:50%;background:#68747a}.dot.online{background:var(--green)}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:11px;margin:9px 0}.status-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:7px}.metric{background:#101518;border:1px solid #2b353a;border-radius:6px;padding:8px;min-width:0}.metric b{display:block;font-size:17px}.metric span{color:var(--muted);font-size:11px;white-space:nowrap}
#viewer{position:relative;background:#050708;border-radius:6px;overflow:hidden}#viewer-stage{position:relative;width:100%;height:100%}.viewer-fullscreen,#viewer:fullscreen,#viewer:-webkit-full-screen{position:fixed!important;inset:0;z-index:20;background:#000;border-radius:0;display:block}.viewer-fullscreen #viewer-stage,#viewer:fullscreen #viewer-stage,#viewer:-webkit-full-screen #viewer-stage{width:100vw;height:100vh}.viewer-fullscreen.force-landscape #viewer-stage,#viewer:fullscreen.force-landscape #viewer-stage,#viewer:-webkit-full-screen.force-landscape #viewer-stage{position:absolute;left:50%;top:50%;width:100vh;height:100vw;transform:translate(-50%,-50%) rotate(90deg)}.viewer-fullscreen #shot,#viewer:fullscreen #shot,#viewer:-webkit-full-screen #shot{width:100%;height:100%;aspect-ratio:auto;border-radius:0}#shot{width:100%;aspect-ratio:10/7;object-fit:contain;background:#050708;border-radius:6px;display:block;cursor:crosshair;touch-action:manipulation}
.commands{display:grid;grid-template-columns:1fr 1fr;gap:8px}button{height:46px;font-size:15px;font-weight:650;border:0;border-radius:6px;padding:0 10px;background:var(--blue);color:white}button:active{filter:brightness(.82)}button.green{background:var(--green)}button.amber{background:var(--amber)}button.red{background:var(--red)}button.dark{background:#4b5960}.wide{grid-column:1/-1}.safety{font-size:12px;line-height:1.5;color:var(--muted);margin:8px 1px 0}
.dpad{position:absolute;left:max(10px,env(safe-area-inset-left));bottom:max(10px,env(safe-area-inset-bottom));z-index:3;display:grid;grid-template-columns:repeat(3,44px);grid-template-rows:repeat(3,38px);gap:4px;margin:0;filter:drop-shadow(0 2px 5px #000b)}.dpad button{width:44px;height:38px;font-size:21px;padding:0;opacity:.82}.dpad button:active{opacity:1}.dpad .up{grid-column:2}.dpad .left{grid-column:1;grid-row:2}.dpad .center{grid-column:2;grid-row:2}.dpad .right{grid-column:3;grid-row:2}.dpad .down{grid-column:2;grid-row:3}
@media(max-width:560px){.status-grid{grid-template-columns:repeat(2,1fr)}.commands{grid-template-columns:1fr 1fr}}
</style></head><body><main>
<header><h1>福寿螺识别云台</h1><span id="online" class="dot"></span></header>
<section class="panel"><div class="status-grid">
<div class="metric"><b id="fps">--</b><span>循环 FPS</span></div><div class="metric"><b id="eggs">--</b><span>识别目标</span></div>
<div class="metric"><b id="primary">--</b><span>锁定 ID</span></div><div class="metric"><b id="relay">--</b><span>瞄准灯</span></div>
</div></section>
<section class="panel"><div id="viewer"><div id="viewer-stage"><img id="shot" alt="摄像头检测画面"><div class="dpad" aria-label="云台微调">
<button class="dark up" title="向上微调" data-nudge="nudge_up">↑</button><button class="dark left" title="向左微调" data-nudge="nudge_left">←</button><button class="amber center" title="云台居中" onclick="cmd('center')">•</button><button class="dark right" title="向右微调" data-nudge="nudge_right">→</button><button class="dark down" title="向下微调" data-nudge="nudge_down">↓</button>
</div></div></div><button class="dark wide" onclick="toggleFullscreen()">横屏全屏画面</button><p id="selection" class="safety">点击画面中的绿色目标框开始跟踪</p></section>
<section class="panel"><div class="commands">
<button class="green" onclick="cmd('select')">点选跟踪</button><button class="dark" onclick="cmd('hold')">保持位置</button>
<button class="amber" onclick="cmd('center')">云台居中</button><button class="red" onclick="cmd('emergency')">立即急停</button>
<button class="amber" onclick="cmd('aim_on')">开启红色瞄准</button><button class="dark" onclick="cmd('aim_off')">关闭红色瞄准</button>
<button class="dark wide" onclick="cmd('clear_estop')">解除急停</button>
</div><p class="safety">自动跟踪到位后会保持当前位置，画面下方方向键松开时执行一次 1° 微调。红色瞄准灯只有在视觉检测到有效目标时才允许开启；高功率激光始终由人工控制。</p></section>
<script>
const token=new URLSearchParams(location.search).get('token')||'maixcam';
const selection=document.querySelector('#selection');
const viewer=document.querySelector('#viewer');
async function api(path){return fetch(path+(path.includes('?')?'&':'?')+'token='+encodeURIComponent(token),{cache:'no-store'});}
async function cmd(c){await api('/api/action?cmd='+encodeURIComponent(c));await refreshStatus();}
let nudgeBusy=false;
async function nudge(c){if(nudgeBusy)return;nudgeBusy=true;try{await cmd(c);}finally{setTimeout(()=>{nudgeBusy=false},120);}}
function fullscreenActive(){return !!(document.fullscreenElement||document.webkitFullscreenElement||viewer.classList.contains('viewer-fullscreen'));}
function applyLandscapeFallback(){viewer.classList.toggle('force-landscape',fullscreenActive()&&innerHeight>innerWidth);}
async function toggleFullscreen(){const enter=viewer.requestFullscreen||viewer.webkitRequestFullscreen,exit=document.exitFullscreen||document.webkitExitFullscreen;try{if(!fullscreenActive()){if(enter)await enter.call(viewer);else viewer.classList.add('viewer-fullscreen');try{if(screen.orientation?.lock)await screen.orientation.lock('landscape');}catch(_e){}setTimeout(applyLandscapeFallback,120);setTimeout(applyLandscapeFallback,500);}else if(exit&&(document.fullscreenElement||document.webkitFullscreenElement)){await exit.call(document);}else{viewer.classList.remove('viewer-fullscreen');viewer.classList.remove('force-landscape');}}catch(e){viewer.classList.toggle('viewer-fullscreen');setTimeout(applyLandscapeFallback,50);}}
addEventListener('resize',applyLandscapeFallback);document.addEventListener('fullscreenchange',applyLandscapeFallback);document.addEventListener('webkitfullscreenchange',applyLandscapeFallback);
async function selectTarget(e){
  if(!shot.naturalWidth||!shot.naturalHeight)return;
  const boxW=shot.clientWidth,boxH=shot.clientHeight,scale=Math.min(boxW/shot.naturalWidth,boxH/shot.naturalHeight);
  const rw=shot.naturalWidth*scale,rh=shot.naturalHeight*scale,ox=(boxW-rw)/2,oy=(boxH-rh)/2;
  const x=(e.offsetX-ox)/rw,y=(e.offsetY-oy)/rh;
  if(x<0||x>1||y<0||y>1)return;
  await api(`/api/select?x=${x.toFixed(5)}&y=${y.toFixed(5)}`);
  await refreshStatus();setTimeout(refreshShot,120);
}
shot.addEventListener('click',selectTarget);
document.querySelectorAll('[data-nudge]').forEach(button=>{let armed=false;button.addEventListener('pointerdown',e=>{e.preventDefault();e.stopPropagation();armed=true;button.setPointerCapture(e.pointerId)});button.addEventListener('pointerup',e=>{e.preventDefault();e.stopPropagation();if(!armed)return;armed=false;nudge(button.dataset.nudge)});button.addEventListener('pointercancel',()=>{armed=false});button.addEventListener('click',e=>{e.preventDefault();e.stopPropagation()})});
async function refreshStatus(){try{const r=await api('/api/status');const s=await r.json();online.className='dot online';fps.textContent=s.fps??'--';eggs.textContent=s.eggs??'--';primary.textContent=s.selected_track_id||s.primary||'--';relay.textContent=s.relay??'--';selection.textContent=s.selection_message||'点击画面中的绿色目标框开始跟踪';}catch(e){online.className='dot';}}
function refreshShot(){shot.src='/snapshot.jpg?token='+encodeURIComponent(token)+'&t='+Date.now()}
setInterval(refreshStatus,650);setInterval(refreshShot,1400);refreshStatus();refreshShot();
</script></main></body></html>"""


class WebControl:
    def __init__(self, host="0.0.0.0", port=8000, snapshot_path="/root/snail_egg/debug/web_latest.jpg"):
        self.host = host
        self.port = int(port)
        self.snapshot_path = snapshot_path
        self.token = os.environ.get("MAIX_WEB_TOKEN", "maixcam")
        self.lock = threading.Lock()
        self.mode = "select"
        self.pan = 0.0
        self.tilt = 0.0
        self.aim_override = None
        self.estop = False
        self.status = {"web": "starting"}
        self.snapshot_requested = False
        self.snapshot_ready = threading.Event()
        # Relative offsets, matching the firmware limits: pan 60..120 and
        # tilt 80..120 degrees around the 90 degree center.
        self.pan_min = -30.0
        self.pan_max = 30.0
        self.tilt_min = -10.0
        self.tilt_max = 30.0
        self.manual_slew_rate = 2.0
        self.pan_target = 0.0
        self.tilt_target = 0.0
        self.last_output_time = time.monotonic()
        self.selection_request = None
        self.selected_track_id = 0
        self.selection_message = "点击画面中的绿色目标框开始跟踪"
        self.last_nudge_time = 0.0
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
                if parsed.path == "/api/select":
                    control.request_selection(
                        query.get("x", ["-1"])[0], query.get("y", ["-1"])[0]
                    )
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
            now = time.monotonic()
            if command == "auto":
                self.mode = "auto"
                self.estop = False
                self.selected_track_id = 0
                self.selection_message = "自动跟踪模式"
            elif command == "select":
                self.mode = "select"
                self.selected_track_id = 0
                self.selection_request = ("clear", 0.0, 0.0)
                self.selection_message = "点击画面中的绿色目标框开始跟踪"
            elif command == "hold":
                self.mode = "hold"
            elif command == "center":
                self.mode = "manual"
                self.pan_target = 0.0
                self.tilt_target = 0.0
            elif command == "nudge_left":
                if now - self.last_nudge_time < 0.12:
                    return
                self.last_nudge_time = now
                self._adopt_live_gimbal_locked()
                self.mode = "manual"
                self.pan_target = min(self.pan_max, self.pan_target + 1.0)
            elif command == "nudge_right":
                if now - self.last_nudge_time < 0.12:
                    return
                self.last_nudge_time = now
                self._adopt_live_gimbal_locked()
                self.mode = "manual"
                self.pan_target = max(self.pan_min, self.pan_target - 1.0)
            elif command == "nudge_up":
                if now - self.last_nudge_time < 0.12:
                    return
                self.last_nudge_time = now
                self._adopt_live_gimbal_locked()
                self.mode = "manual"
                self.tilt_target = min(self.tilt_max, self.tilt_target + 1.0)
            elif command == "nudge_down":
                if now - self.last_nudge_time < 0.12:
                    return
                self.last_nudge_time = now
                self._adopt_live_gimbal_locked()
                self.mode = "manual"
                self.tilt_target = max(self.tilt_min, self.tilt_target - 1.0)
            elif command == "pan_left":
                self.mode = "manual"
                self.pan_target = max(self.pan_min, self.pan_target - 5.0)
            elif command == "pan_right":
                self.mode = "manual"
                self.pan_target = min(self.pan_max, self.pan_target + 5.0)
            elif command == "tilt_up":
                self.mode = "manual"
                self.tilt_target = min(self.tilt_max, self.tilt_target + 5.0)
            elif command == "tilt_down":
                self.mode = "manual"
                self.tilt_target = max(self.tilt_min, self.tilt_target - 5.0)
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

    def _adopt_live_gimbal_locked(self):
        """Start a manual nudge from the current physical command, never zero."""
        if self.mode == "manual":
            return
        live_pan = float(self.status.get("gimbal_pan", self.pan))
        live_tilt = float(self.status.get("gimbal_tilt", self.tilt))
        self.pan = max(self.pan_min, min(self.pan_max, live_pan))
        self.tilt = max(self.tilt_min, min(self.tilt_max, live_tilt))
        self.pan_target = self.pan
        self.tilt_target = self.tilt

    def request_selection(self, x_value, y_value):
        try:
            x = float(x_value)
            y = float(y_value)
        except (TypeError, ValueError):
            return
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            return
        with self.lock:
            if self.estop:
                return
            self.mode = "select"
            self.selection_request = ("point", x, y)
            self.selection_message = "正在确认点击目标"

    def begin_manual_adjust(self, pan, tilt, track_id):
        with self.lock:
            self.mode = "manual"
            self.pan = max(self.pan_min, min(self.pan_max, float(pan)))
            self.tilt = max(self.tilt_min, min(self.tilt_max, float(tilt)))
            self.pan_target = self.pan
            self.tilt_target = self.tilt
            self.selected_track_id = int(track_id)
            self.selection_message = "目标已到位并保持，可用方向键微调"

    def begin_lost_hold(self, pan, tilt, track_id):
        with self.lock:
            self.mode = "manual"
            self.pan = max(self.pan_min, min(self.pan_max, float(pan)))
            self.tilt = max(self.tilt_min, min(self.tilt_max, float(tilt)))
            self.pan_target = self.pan
            self.tilt_target = self.tilt
            self.selected_track_id = int(track_id)
            self.selection_message = "目标已丢失，保持当前位置；请重新点选或手动微调"

    def begin_lost_finish(self, pan, tilt, pan_goal, tilt_goal, track_id):
        with self.lock:
            self.mode = "manual"
            self.pan = max(self.pan_min, min(self.pan_max, float(pan)))
            self.tilt = max(self.tilt_min, min(self.tilt_max, float(tilt)))
            self.pan_target = max(self.pan_min, min(self.pan_max, float(pan_goal)))
            self.tilt_target = max(self.tilt_min, min(self.tilt_max, float(tilt_goal)))
            self.selected_track_id = int(track_id)
            self.selection_message = "目标短暂丢失，按最后可靠位置缓慢完成并保持"

    def begin_fixed_target(self, pan, tilt, pan_goal, tilt_goal, track_id):
        """Move once to a clicked target without allowing ID reacquisition."""
        with self.lock:
            self.mode = "manual"
            self.pan = max(self.pan_min, min(self.pan_max, float(pan)))
            self.tilt = max(self.tilt_min, min(self.tilt_max, float(tilt)))
            self.pan_target = max(self.pan_min, min(self.pan_max, float(pan_goal)))
            self.tilt_target = max(self.tilt_min, min(self.tilt_max, float(tilt_goal)))
            self.selected_track_id = int(track_id)
            self.selection_message = "Fixed target locked; moving slowly and holding"

    def consume_selection_request(self):
        with self.lock:
            request = self.selection_request
            self.selection_request = None
            return request

    def confirm_selection(self, track_id):
        with self.lock:
            self.mode = "selected"
            self.selected_track_id = int(track_id)
            self.selection_message = "已锁定目标，正在稳定跟踪至瞄准位置"

    def reject_selection(self):
        with self.lock:
            self.mode = "select"
            self.selected_track_id = 0
            self.selection_message = "未点中绿色目标框，请重新点击"

    def get_control(self):
        with self.lock:
            now = time.monotonic()
            dt = max(0.0, min(0.12, now - self.last_output_time))
            self.last_output_time = now
            max_step = self.manual_slew_rate * dt
            self.pan += max(-max_step, min(max_step, self.pan_target - self.pan))
            self.tilt += max(-max_step, min(max_step, self.tilt_target - self.tilt))
            self.pan = max(self.pan_min, min(self.pan_max, self.pan))
            self.tilt = max(self.tilt_min, min(self.tilt_max, self.tilt))
            return self.mode, self.pan, self.tilt, self.aim_override, self.estop

    def update_status(self, values):
        with self.lock:
            self.status = dict(values)
            # Automatic tracking owns the gimbal outside manual mode. Keep the
            # web-side command anchored to that live position so an early
            # nudge cannot resume from an old target or the startup zero.
            if self.mode != "manual":
                live_pan = float(values.get("gimbal_pan", self.pan))
                live_tilt = float(values.get("gimbal_tilt", self.tilt))
                self.pan = max(self.pan_min, min(self.pan_max, live_pan))
                self.tilt = max(self.tilt_min, min(self.tilt_max, live_tilt))
                self.pan_target = self.pan
                self.tilt_target = self.tilt

    def get_status(self):
        with self.lock:
            return dict(self.status, mode=self.mode, pan=self.pan, tilt=self.tilt,
                        aim_override=self.aim_override, estop=self.estop,
                        pan_limits=[self.pan_min, self.pan_max],
                        tilt_limits=[self.tilt_min, self.tilt_max],
                        pan_target=self.pan_target, tilt_target=self.tilt_target,
                        selected_track_id=self.selected_track_id,
                        selection_message=self.selection_message)

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
