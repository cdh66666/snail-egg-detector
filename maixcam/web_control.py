"""Phone control panel for the MaixCAM detector and gimbal.

The HTTP thread only records commands. The vision loop consumes them, so GPIO
and PWM are never accessed by request-handler threads.
"""

import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no,viewport-fit=cover">
<title>福寿螺识别云台</title>
<style>
:root{color-scheme:dark;--bg:#0d1114;--panel:#171d21;--line:#344047;--text:#f2f5f6;--muted:#a9b4ba;--green:#18a76f;--amber:#d68a25;--red:#d84a4a;--blue:#397fb8}
*{box-sizing:border-box}html,body{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,"PingFang SC",sans-serif;overscroll-behavior:none}body{padding:10px}
main{max-width:760px;margin:auto}header{display:flex;align-items:center;justify-content:space-between;margin:2px 0 8px}h1{font-size:18px;margin:0;letter-spacing:0}.dot{width:10px;height:10px;border-radius:50%;background:#68747a}.dot.online{background:var(--green)}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:9px;margin:8px 0}.status-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:6px}.metric{background:#101518;border:1px solid #2b353a;border-radius:6px;padding:7px;min-width:0}.metric b{display:block;font-size:16px}.metric span{color:var(--muted);font-size:11px;white-space:nowrap}
#viewer{position:relative;background:#000;border-radius:6px;overflow:hidden}#viewer-stage{position:relative;width:100%;height:100%}#shot{display:block;width:100%;aspect-ratio:10/7;object-fit:contain;background:#000;border-radius:6px;cursor:crosshair;touch-action:manipulation}
.viewer-fullscreen,#viewer:fullscreen,#viewer:-webkit-full-screen{position:fixed!important;inset:0;z-index:20;background:#000;border-radius:0;display:block}.viewer-fullscreen #viewer-stage,#viewer:fullscreen #viewer-stage,#viewer:-webkit-full-screen #viewer-stage{width:100vw;height:100vh}.viewer-fullscreen #shot,#viewer:fullscreen #shot,#viewer:-webkit-full-screen #shot{width:100%;height:100%;aspect-ratio:auto;object-fit:cover;border-radius:0}
.viewer-fullscreen.force-landscape #viewer-stage,#viewer:fullscreen.force-landscape #viewer-stage,#viewer:-webkit-full-screen.force-landscape #viewer-stage{position:absolute;left:50%;top:50%;width:100vh;height:100vw;transform:translate(-50%,-50%) rotate(90deg)}
.dpad{position:absolute;left:max(10px,env(safe-area-inset-left));bottom:max(10px,env(safe-area-inset-bottom));z-index:3;display:grid;grid-template-columns:repeat(3,44px);grid-template-rows:repeat(3,38px);gap:4px;filter:drop-shadow(0 2px 5px #000b)}.dpad button{width:44px;height:38px;font-size:21px;padding:0;opacity:.82}.dpad button:active{opacity:1}.dpad .up{grid-column:2}.dpad .left{grid-column:1;grid-row:2}.dpad .center{grid-column:2;grid-row:2}.dpad .right{grid-column:3;grid-row:2}.dpad .down{grid-column:2;grid-row:3}
.commands{display:grid;grid-template-columns:1fr 1fr;gap:8px}button{height:44px;font-size:15px;font-weight:650;border:0;border-radius:6px;padding:0 10px;background:var(--blue);color:#fff;touch-action:manipulation}button:active{filter:brightness(.82)}button.green{background:var(--green)}button.amber{background:var(--amber)}button.red{background:var(--red)}button.dark{background:#4b5960}.wide{grid-column:1/-1}.hint{font-size:12px;line-height:1.45;color:var(--muted);margin:7px 1px 0}
.notice{position:fixed;inset:0;z-index:50;display:none;align-items:center;justify-content:center;padding:20px;background:#000a}.notice.show{display:flex}.notice-box{width:min(430px,100%);background:#20282d;border:2px solid var(--amber);border-radius:8px;padding:20px;box-shadow:0 12px 50px #000}.notice-box h2{margin:0 0 10px;font-size:23px;color:#ffd27a}.notice-box p{margin:0 0 18px;font-size:17px;line-height:1.55}.notice-box button{width:100%;font-size:17px}.needs-estop{outline:3px solid #ffd27a;animation:pulse 1s ease-in-out 3}@keyframes pulse{50%{filter:brightness(1.5)}}
@media(max-width:560px){.status-grid{grid-template-columns:repeat(2,1fr)}}
</style>
</head>
<body><main>
<header><h1>福寿螺识别云台</h1><span id="online" class="dot"></span></header>
<section class="panel"><div class="status-grid">
<div class="metric"><b id="fps">--</b><span>循环 / 识别 / 画面</span></div>
<div class="metric"><b id="eggs">--</b><span>识别目标</span></div>
<div class="metric"><b id="primary">--</b><span>锁定 ID</span></div>
<div class="metric"><b id="relay">--</b><span>瞄准灯</span></div>
</div></section>
<section class="panel"><div id="viewer"><div id="viewer-stage">
<img id="shot" alt="摄像头检测画面">
<div class="dpad" aria-label="云台微调">
<button class="dark up" title="向上微调" data-nudge="nudge_up">↑</button>
<button class="dark left" title="向左微调" data-nudge="nudge_left">←</button>
<button class="amber center" title="云台居中" data-command="center">•</button>
<button class="dark right" title="向右微调" data-nudge="nudge_right">→</button>
<button class="dark down" title="向下微调" data-nudge="nudge_down">↓</button>
</div></div></div>
<button class="dark wide" data-command="fullscreen">横屏全屏画面</button>
<p id="selection" class="hint">点击画面中的绿色目标框开始跟踪</p></section>
<section class="panel"><div class="commands">
<button class="green" data-command="auto">自动跟踪</button><button class="green" data-command="select">点选跟踪</button>
<button class="dark" data-command="hold">保持位置</button><button class="amber" data-command="center">云台居中</button>
<button class="red" data-command="emergency">立即急停</button><button class="amber" data-command="aim_auto">瞄准灯自动</button>
<button class="amber" data-command="aim_on">开启红色瞄准</button><button class="dark" data-command="aim_off">关闭红色瞄准</button>
<button id="feedbackMode" class="dark wide" data-command="closed_loop_toggle">红点闭环：自适应</button>
<button class="dark wide" data-command="clear_estop">解除急停</button>
</div><p class="hint">方向键每次松开移动 1°。人工微调后保持当前位置，不重新追踪旧目标。高功率消杀执行器始终由人工控制。</p></section>
</main>
<div id="notice" class="notice" role="alertdialog" aria-modal="true"><div class="notice-box"><h2 id="noticeTitle">操作提示</h2><p id="noticeText"></p><button id="noticeClose" class="amber">我知道了</button></div></div>
<script>
const token=new URLSearchParams(location.search).get('token')||'maixcam';
const selection=document.querySelector('#selection'),viewer=document.querySelector('#viewer'),shot=document.querySelector('#shot');
const onlineEl=document.querySelector('#online'),fpsEl=document.querySelector('#fps'),eggsEl=document.querySelector('#eggs'),primaryEl=document.querySelector('#primary'),relayEl=document.querySelector('#relay'),feedbackModeEl=document.querySelector('#feedbackMode');
const notice=document.querySelector('#notice'),noticeTitle=document.querySelector('#noticeTitle'),noticeText=document.querySelector('#noticeText'),noticeClose=document.querySelector('#noticeClose'),clearEstopButton=document.querySelector('[data-command="clear_estop"]');
let latestStatus={estop:true};
function showNotice(title,text,highlight=false){noticeTitle.textContent=title;noticeText.textContent=text;notice.classList.add('show');if(highlight)clearEstopButton.classList.add('needs-estop');}
function closeNotice(){notice.classList.remove('show');clearEstopButton.classList.remove('needs-estop');}
noticeClose.addEventListener('click',closeNotice);notice.addEventListener('click',event=>{if(event.target===notice)closeNotice();});
async function api(path){const join=path.includes('?')?'&':'?';const response=await fetch(path+join+'token='+encodeURIComponent(token),{cache:'no-store'});if(!response.ok)throw new Error('HTTP '+response.status);return response;}
const estopBlockedCommands=new Set(['auto','select','center','nudge_left','nudge_right','nudge_up','nudge_down','aim_on']);
let actionQueue=[],actionRunning=false;
function queueAction(command){actionQueue.push(command);if(!actionRunning)processActions();}
async function processActions(){
  actionRunning=true;
  try{
    while(actionQueue.length){
      const command=actionQueue.shift();
      try{
        await api('/api/action?cmd='+encodeURIComponent(command));
        if(command==='clear_estop'){latestStatus.estop=false;showNotice('急停已解除','现在可以使用方向键、键盘方向键、点选跟踪或自动跟踪。');}
      }catch(error){actionQueue=[];showNotice('操作没有执行','设备拒绝了本次操作：'+error.message+'。请检查设备在线状态和急停状态。');}
      await new Promise(resolve=>setTimeout(resolve,20));
    }
    await refreshStatus();
  }finally{actionRunning=false;if(actionQueue.length)processActions();}
}
async function emergencyNow(){actionQueue=[];latestStatus.estop=true;try{await api('/api/action?cmd=emergency');await refreshStatus();showNotice('已进入急停','云台已停止，红色辅助瞄准灯已关闭。');}catch(error){showNotice('急停请求失败','设备没有响应急停请求：'+error.message+'。请立即切断设备电源。');}}
function cmd(command){if(command==='emergency'){emergencyNow();return;}if(latestStatus.estop&&estopBlockedCommands.has(command)){showNotice('当前处于急停状态','请先点击页面下方的“解除急停”，确认周围安全后再操作云台或跟踪目标。',true);return;}queueAction(command);}
function nudge(command){if(latestStatus.estop){showNotice('当前处于急停状态','请先解除急停，再使用方向键或键盘方向键微调。',true);return;}queueAction(command);}
function fullscreenActive(){return !!(document.fullscreenElement||document.webkitFullscreenElement||viewer.classList.contains('viewer-fullscreen'));}
function applyLandscapeFallback(){viewer.classList.toggle('force-landscape',fullscreenActive()&&innerHeight>innerWidth);}
async function toggleFullscreen(){const enter=viewer.requestFullscreen||viewer.webkitRequestFullscreen,exit=document.exitFullscreen||document.webkitExitFullscreen;try{if(!fullscreenActive()){if(enter)await enter.call(viewer);else viewer.classList.add('viewer-fullscreen');try{if(screen.orientation?.lock)await screen.orientation.lock('landscape');}catch(_error){}setTimeout(applyLandscapeFallback,100);}else if(exit&&(document.fullscreenElement||document.webkitFullscreenElement)){await exit.call(document);}else{viewer.classList.remove('viewer-fullscreen','force-landscape');}}catch(_error){viewer.classList.toggle('viewer-fullscreen');setTimeout(applyLandscapeFallback,50);}}
addEventListener('resize',applyLandscapeFallback);document.addEventListener('fullscreenchange',applyLandscapeFallback);document.addEventListener('webkitfullscreenchange',applyLandscapeFallback);
async function selectTarget(event){if(latestStatus.estop){showNotice('当前处于急停状态','请先解除急停，再点击绿色目标框进行跟踪。',true);return;}if(!shot.naturalWidth||!shot.naturalHeight)return;const boxW=shot.clientWidth,boxH=shot.clientHeight;const cover=getComputedStyle(shot).objectFit==='cover';const scale=cover?Math.max(boxW/shot.naturalWidth,boxH/shot.naturalHeight):Math.min(boxW/shot.naturalWidth,boxH/shot.naturalHeight);const rw=shot.naturalWidth*scale,rh=shot.naturalHeight*scale,ox=(boxW-rw)/2,oy=(boxH-rh)/2;const x=(event.offsetX-ox)/rw,y=(event.offsetY-oy)/rh;if(x<0||x>1||y<0||y>1)return;try{await api(`/api/select?x=${x.toFixed(5)}&y=${y.toFixed(5)}`);await refreshStatus();}catch(error){showNotice('没有选中目标','请先解除急停，并点击画面中的绿色目标框。');}}
shot.addEventListener('click',selectTarget);
document.querySelectorAll('[data-nudge]').forEach(button=>button.addEventListener('click',event=>{event.preventDefault();event.stopPropagation();nudge(button.dataset.nudge);}));
addEventListener('keydown',event=>{const keys={ArrowUp:'nudge_up',ArrowDown:'nudge_down',ArrowLeft:'nudge_left',ArrowRight:'nudge_right'};const command=keys[event.key];if(!command||event.repeat)return;event.preventDefault();nudge(command);});
document.querySelectorAll('[data-command]').forEach(button=>button.addEventListener('click',()=>button.dataset.command==='fullscreen'?toggleFullscreen():cmd(button.dataset.command)));
async function refreshStatus(){try{const response=await api('/api/status');const status=await response.json();latestStatus=status;onlineEl.className='dot online';fpsEl.textContent=(status.loop_fps??status.fps??'--')+' / '+(status.detect_hz??'--')+' / '+(status.stream_fps??'--');eggsEl.textContent=status.eggs??'--';primaryEl.textContent=status.selected_track_id||status.primary||'--';relayEl.textContent=status.relay??'--';selection.textContent=status.estop?'当前处于急停状态，请先解除急停':(status.selection_message||'点击画面中的绿色目标框开始跟踪');const mode=status.closed_loop_override===true?'强制闭环':status.closed_loop_override===false?'强制开环':'自适应';feedbackModeEl.textContent='红点闭环：'+mode;feedbackModeEl.className=status.closed_loop_override===true?'green wide':status.closed_loop_override===false?'amber wide':'dark wide';}catch(_error){onlineEl.className='dot';fpsEl.textContent='连接失败';}}
let liveTimer=0,liveObjectUrl='';
function startStream(){shot.src='/stream?token='+encodeURIComponent(token)+'&ts='+Date.now();}
function nextLiveFrame(delay=0){
  clearTimeout(liveTimer);
  liveTimer=setTimeout(async()=>{
    try{
      const response=await fetch('/live.jpg?token='+encodeURIComponent(token)+'&ts='+Date.now(),{cache:'no-store'});
      if(!response.ok)throw new Error('HTTP '+response.status);
      const blob=await response.blob();
      const nextUrl=URL.createObjectURL(blob),oldUrl=liveObjectUrl;
      shot.onload=()=>{if(oldUrl)URL.revokeObjectURL(oldUrl);nextLiveFrame(45);};
      shot.onerror=()=>{URL.revokeObjectURL(nextUrl);nextLiveFrame(350);};
      liveObjectUrl=nextUrl;shot.src=nextUrl;
    }catch(_error){nextLiveFrame(350);}
  },delay);
}
shot.addEventListener('error',()=>nextLiveFrame(350));
setInterval(refreshStatus,650);refreshStatus();startStream();
</script></body></html>"""


class WebControl:
    def __init__(self, host="0.0.0.0", port=8000, snapshot_path="/root/snail_egg/debug/web_latest.jpg"):
        self.host = host
        self.port = int(port)
        self.snapshot_path = snapshot_path
        self.token = os.environ.get("MAIX_WEB_TOKEN", "maixcam")
        self.lock = threading.Lock()
        self.mode = "hold"
        self.pan = 0.0
        self.tilt = 0.0
        self.pan_target = 0.0
        self.tilt_target = 0.0
        self.pan_min, self.pan_max = -30.0, 30.0
        self.tilt_min, self.tilt_max = -10.0, 30.0
        self.aim_override = False
        self.closed_loop_override = None
        self.estop = True
        self.status = {"web": "starting"}
        self.selection_request = None
        self.selected_track_id = 0
        self.selection_message = "点击画面中的绿色目标框开始跟踪"
        self.last_nudge_time = 0.0
        self.snapshot_requested = False
        self.snapshot_ready = threading.Event()
        self.latest_jpeg = None
        self.latest_jpeg_sequence = 0
        self.latest_jpeg_ready = threading.Condition(self.lock)
        self._stopped = False
        control = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                return

            def send_bytes(self, data, content_type, status=200):
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def send_json(self, payload, status=200):
                self.send_bytes(json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

            def authorized(self, query):
                return query.get("token", [""])[0] == control.token

            def do_GET(self):
                parsed = urlparse(self.path)
                query = parse_qs(parsed.query)
                if parsed.path == "/":
                    self.send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
                    return
                if not self.authorized(query):
                    self.send_json({"ok": False, "error": "bad token"}, 403)
                    return
                if parsed.path == "/stream":
                    control.stream_frame(self)
                elif parsed.path == "/api/status":
                    self.send_json(control.get_status())
                elif parsed.path == "/api/action":
                    command = query.get("cmd", [""])[0]
                    accepted = control.apply_command(command)
                    self.send_json({"ok": accepted}, 200 if accepted else 400)
                elif parsed.path == "/api/select":
                    accepted = control.request_selection(query.get("x", ["-1"])[0], query.get("y", ["-1"])[0])
                    self.send_json({"ok": accepted}, 200 if accepted else 400)
                elif parsed.path == "/snapshot.jpg":
                    control.request_frame()
                    control.snapshot_ready.wait(0.45)
                    try:
                        with open(control.snapshot_path, "rb") as image_file:
                            self.send_bytes(image_file.read(), "image/jpeg")
                    except OSError:
                        self.send_json({"ok": False, "error": "snapshot not ready"}, 503)
                elif parsed.path == "/live.jpg":
                    jpeg = control.get_latest_jpeg()
                    if jpeg is None:
                        self.send_json({"ok": False, "error": "live frame not ready"}, 503)
                    else:
                        self.send_bytes(jpeg, "image/jpeg")
                else:
                    self.send_json({"ok": False, "error": "not found"}, 404)

        self.server = ThreadingHTTPServer((self.host, self.port), Handler)
        self.server.daemon_threads = True
        self.port = int(self.server.server_address[1])
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self):
        self.thread.start()
        print("WEB_CONTROL,LISTEN,%s,%d" % (self.host, self.port))

    def stop(self):
        if self._stopped:
            return
        self._stopped = True
        try:
            self.server.shutdown()
            self.server.server_close()
            if self.thread.is_alive():
                self.thread.join(1.0)
        except Exception:
            pass

    def apply_command(self, command):
        valid = {
            "auto", "select", "hold", "center", "nudge_left", "nudge_right", "nudge_up", "nudge_down",
            "aim_auto", "aim_on", "aim_off", "closed_loop_toggle", "emergency", "clear_estop",
        }
        if command not in valid:
            return False
        with self.lock:
            now = time.monotonic()
            if self.estop and command in {
                "auto", "select", "center", "nudge_left", "nudge_right",
                "nudge_up", "nudge_down", "aim_on",
            }:
                return False
            if command == "auto":
                self.mode, self.estop, self.selected_track_id = "auto", False, 0
                self.selection_message = "自动跟踪模式"
            elif command == "select":
                self.mode, self.selected_track_id = "select", 0
                self.selection_request = ("clear", 0.0, 0.0)
                self.selection_message = "点击画面中的绿色目标框开始跟踪"
            elif command == "hold":
                self._adopt_live_gimbal_locked()
                self.mode = "hold"
            elif command == "center":
                self.mode, self.pan_target, self.tilt_target = "manual", 0.0, 0.0
            elif command.startswith("nudge_"):
                self.last_nudge_time = now
                self._adopt_live_gimbal_locked()
                self.mode = "manual"
                if command == "nudge_left":
                    self.pan_target = min(self.pan_max, self.pan_target + 1.0)
                elif command == "nudge_right":
                    self.pan_target = max(self.pan_min, self.pan_target - 1.0)
                elif command == "nudge_up":
                    self.tilt_target = min(self.tilt_max, self.tilt_target + 1.0)
                else:
                    self.tilt_target = max(self.tilt_min, self.tilt_target - 1.0)
            elif command == "aim_auto":
                self.aim_override = None
            elif command == "aim_on":
                self.aim_override = True
            elif command == "aim_off":
                self.aim_override = False
            elif command == "closed_loop_toggle":
                if self.closed_loop_override is None:
                    self.closed_loop_override = True
                elif self.closed_loop_override is True:
                    self.closed_loop_override = False
                else:
                    self.closed_loop_override = None
            elif command == "emergency":
                self._adopt_live_gimbal_locked()
                self.mode, self.aim_override, self.estop = "hold", False, True
            elif command == "clear_estop":
                self.estop = False
        return True

    def _adopt_live_gimbal_locked(self):
        if self.mode == "manual":
            return
        live_pan = float(self.status.get("gimbal_pan", self.pan))
        live_tilt = float(self.status.get("gimbal_tilt", self.tilt))
        self.pan = max(self.pan_min, min(self.pan_max, live_pan))
        self.tilt = max(self.tilt_min, min(self.tilt_max, live_tilt))
        self.pan_target, self.tilt_target = self.pan, self.tilt

    def request_selection(self, x_value, y_value):
        try:
            x, y = float(x_value), float(y_value)
        except (TypeError, ValueError):
            return False
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            return False
        with self.lock:
            if self.estop:
                return False
            self.mode = "select"
            self.selection_request = ("point", x, y)
            self.selection_message = "正在确认点击目标"
        return True

    def begin_manual_adjust(self, pan, tilt, track_id):
        self._begin_manual(pan, tilt, pan, tilt, track_id, "目标已到位并保持，可用方向键微调")

    def begin_lost_hold(self, pan, tilt, track_id):
        self._begin_manual(pan, tilt, pan, tilt, track_id, "目标已丢失，保持当前位置；请重新点选或手动微调")

    def begin_lost_finish(self, pan, tilt, pan_goal, tilt_goal, track_id):
        self._begin_manual(pan, tilt, pan_goal, tilt_goal, track_id, "目标短暂丢失，按最后可靠位置缓慢完成并保持")

    def begin_conservative_finish(self, pan, tilt, pan_goal, tilt_goal, track_id):
        self._begin_manual(pan, tilt, pan_goal, tilt_goal, track_id, "目标锁定不稳定，正在移到最近可信位置；随后可手动微调")

    def _begin_manual(self, pan, tilt, pan_goal, tilt_goal, track_id, message):
        with self.lock:
            self.mode = "manual"
            self.pan = max(self.pan_min, min(self.pan_max, float(pan)))
            self.tilt = max(self.tilt_min, min(self.tilt_max, float(tilt)))
            self.pan_target = max(self.pan_min, min(self.pan_max, float(pan_goal)))
            self.tilt_target = max(self.tilt_min, min(self.tilt_max, float(tilt_goal)))
            self.selected_track_id = int(track_id)
            self.selection_message = message

    def consume_selection_request(self):
        with self.lock:
            request, self.selection_request = self.selection_request, None
            return request

    def confirm_selection(self, track_id):
        with self.lock:
            self.mode, self.selected_track_id = "selected", int(track_id)
            self.selection_message = "已选中目标，正在确认锁定稳定性"

    def mark_stable_tracking(self, track_id):
        with self.lock:
            if self.mode == "selected" and self.selected_track_id == int(track_id):
                self.selection_message = "目标锁定稳定，正在持续跟踪"

    def reject_selection(self):
        with self.lock:
            self.mode, self.selected_track_id = "select", 0
            self.selection_message = "未点中绿色目标框，请重新点击"

    def get_control(self):
        with self.lock:
            self.pan = max(self.pan_min, min(self.pan_max, self.pan_target))
            self.tilt = max(self.tilt_min, min(self.tilt_max, self.tilt_target))
            return self.mode, self.pan, self.tilt, self.aim_override, self.estop

    def get_closed_loop_override(self):
        with self.lock:
            return self.closed_loop_override

    def update_status(self, values):
        with self.lock:
            self.status = dict(values)
            if self.mode != "manual":
                self.pan = max(self.pan_min, min(self.pan_max, float(values.get("gimbal_pan", self.pan))))
                self.tilt = max(self.tilt_min, min(self.tilt_max, float(values.get("gimbal_tilt", self.tilt))))
                self.pan_target, self.tilt_target = self.pan, self.tilt

    def get_status(self):
        with self.lock:
            return dict(
                self.status,
                mode=self.mode,
                pan=self.pan,
                tilt=self.tilt,
                pan_target=self.pan_target,
                tilt_target=self.tilt_target,
                pan_limits=[self.pan_min, self.pan_max],
                tilt_limits=[self.tilt_min, self.tilt_max],
                aim_override=self.aim_override,
                closed_loop_override=self.closed_loop_override,
                estop=self.estop,
                selected_track_id=self.selected_track_id,
                selection_message=self.selection_message,
            )

    def request_frame(self):
        with self.lock:
            self.snapshot_requested = True
            self.snapshot_ready.clear()

    def consume_frame_request(self):
        with self.lock:
            requested, self.snapshot_requested = self.snapshot_requested, False
            return requested

    def publish_frame(self):
        self.snapshot_ready.set()

    def publish_jpeg(self, jpeg):
        with self.latest_jpeg_ready:
            self.latest_jpeg = jpeg
            self.latest_jpeg_sequence += 1
            self.latest_jpeg_ready.notify_all()

    def get_latest_jpeg(self):
        with self.lock:
            return self.latest_jpeg

    def stream_frame(self, request):
        request.send_response(200)
        request.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        request.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        request.send_header("Pragma", "no-cache")
        request.send_header("Connection", "close")
        request.end_headers()
        sent = -1
        while True:
            with self.latest_jpeg_ready:
                if self.latest_jpeg_sequence == sent:
                    self.latest_jpeg_ready.wait(0.5)
                jpeg = self.latest_jpeg
                sequence = self.latest_jpeg_sequence
            if jpeg is None or sequence == sent:
                continue
            request.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode("ascii") + b"\r\n\r\n" + jpeg + b"\r\n")
            request.wfile.flush()
            sent = sequence
