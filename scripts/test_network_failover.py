"""Host regression checks for saved-WiFi-first boot and AP fallback."""

from __future__ import annotations

import ast
import threading
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "maixcam" / "main.py"


def load_supervisor():
    tree = ast.parse(MAIN.read_text(encoding="utf-8"))
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "NetworkSupervisor"
    ]
    namespace = {
        "threading": threading,
        "network": None,
        "_network_lock": threading.Lock(),
        "_network_mode": "checking",
        "_network_ip": "",
        "_network_ssid": "",
        "_wifi_manager": None,
        "WIFI_BOOT_CONNECT_TIMEOUT_S": 10,
        "WIFI_AP_IP": "192.168.66.1",
        "WIFI_AP_SSID": "SnailEgg-MaixCAM",
        "WIFI_AP_HEALTH_INTERVAL_S": 3.0,
        "WIFI_STA_FAILURE_LIMIT": 3,
        "safe_sleep": lambda _seconds: None,
        "read_saved_wifi": lambda: ("", ""),
        "access_point_healthy": lambda: False,
        "start_open_access_point": lambda: True,
        "stop_wlan_station_client": lambda: None,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(MAIN), "exec"), namespace)
    return namespace


def test_boot_policy(namespace):
    supervisor = namespace["NetworkSupervisor"]()
    events = []
    supervisor.connect_wifi = lambda ssid, password, timeout, reason: events.append(
        ("wifi", ssid, password, timeout, reason)
    ) or True
    supervisor.ensure_ap = lambda reason: events.append(("ap", reason)) or True
    supervisor._loop = lambda: events.append(("monitor",))

    namespace["read_saved_wifi"] = lambda: ("CompanyWiFi", "secret")
    supervisor._boot_and_monitor()
    assert events == [
        ("wifi", "CompanyWiFi", "secret", 10, "boot"),
        ("monitor",),
    ]

    events.clear()
    supervisor.connect_wifi = lambda *args, **kwargs: events.append(("wifi_failed",)) or False
    supervisor._boot_and_monitor()
    assert events == [("wifi_failed",), ("ap", "boot_fallback"), ("monitor",)]

    events.clear()
    namespace["read_saved_wifi"] = lambda: ("", "")
    supervisor._boot_and_monitor()
    assert events == [("ap", "boot_no_saved_wifi"), ("monitor",)]


def test_station_state(namespace):
    class FakeWifi:
        connected = True

        def __init__(self):
            self.timeout = None

        def stop_ap(self):
            return None

        def connect(self, ssid, password, wait, timeout):
            assert (ssid, password, wait) == ("CompanyWiFi", "secret", True)
            self.timeout = timeout
            return "ok"

        def is_connected(self):
            return self.connected

        def get_ip(self):
            return "192.168.43.27"

    namespace["network"] = SimpleNamespace(wifi=SimpleNamespace(Wifi=FakeWifi))
    supervisor = namespace["NetworkSupervisor"]()
    assert supervisor.connect_wifi("CompanyWiFi", "secret", timeout=10, reason="test") is True
    assert namespace["_network_mode"] == "wifi"
    assert namespace["_network_ip"] == "192.168.43.27"
    assert namespace["_network_ssid"] == "CompanyWiFi"

    FakeWifi.connected = False
    assert supervisor.connect_wifi("CompanyWiFi", "secret", timeout=1, reason="missing") is False
    assert namespace["_network_mode"] == "error"
    assert namespace["_network_ip"] == ""


def main():
    namespace = load_supervisor()
    test_boot_policy(namespace)
    test_station_state(namespace)
    print({
        "saved_wifi_first": "passed",
        "ten_second_timeout": "passed",
        "boot_ap_fallback": "passed",
        "station_state": "passed",
    })


if __name__ == "__main__":
    main()
