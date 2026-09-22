#!/usr/bin/env python3
"""Read-only Linux VPN snapshot; standard library only, no sudo required."""

import argparse
import concurrent.futures
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import time


def run(command):
    started = time.monotonic()
    try:
        result = subprocess.run(command, capture_output=True, text=True,
                                errors="replace", timeout=12)
        data = {"returncode": result.returncode, "stdout": result.stdout,
                "stderr": result.stderr}
    except subprocess.TimeoutExpired as exc:
        def decode(value):
            return value.decode(errors="replace") if isinstance(value, bytes) else value or ""
        data = {"error": "timeout (12 seconds)", "stdout": decode(exc.stdout),
                "stderr": decode(exc.stderr)}
    except OSError as exc:
        data = {"error": str(exc)}
    return {"command": command, "elapsed_seconds": round(time.monotonic() - started, 2), **data}


def main():
    parser = argparse.ArgumentParser(description="采集 VPN 网络诊断信息，不修改网络配置。")
    parser.add_argument("--label", default="vpn-on", help="例如 vpn-on 或 vpn-off")
    parser.add_argument("--target", action="append", default=[], help="额外测试的设备 IP 或域名，可重复指定")
    args = parser.parse_args()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    label = re.sub(r"[^a-zA-Z0-9_-]", "_", args.label)[:50] or "snapshot"
    output = Path(__file__).resolve().parents[1] / "results" / "vpn_diagnostics" / (stamp + "_" + label)
    output.mkdir(parents=True, mode=0o700)
    os.chmod(output, 0o700)
    report = {"captured_at": stamp, "label": args.label, "checks": {},
              "notes": ["Read-only snapshot; no network configuration changes.",
                        "Ping failure alone does not establish loss of connectivity.",
                        "Missing tools/permissions and command timeouts are recorded.",
                        "HTTP tests bypass proxy environment variables."]}

    def save():
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    jobs = {
        "addresses": ["ip", "-details", "address", "show"],
        "routes_v4": ["ip", "-4", "route", "show", "table", "all"],
        "routes_v6": ["ip", "-6", "route", "show", "table", "all"],
        "rules_v4": ["ip", "-4", "rule", "show"],
        "rules_v6": ["ip", "-6", "rule", "show"],
        "neighbors": ["ip", "neigh", "show"],
        "resolved": ["resolvectl", "status"],
        "nm_devices": ["nmcli", "device", "show"],
        "connections": ["nmcli", "-f", "NAME,TYPE,DEVICE", "connection", "show", "--active"],
        "dns_system": ["getent", "ahostsv4", "example.com"],
        "dns_resolved": ["resolvectl", "query", "example.com"],
        "http_domain": ["curl", "--noproxy", "*", "-4", "-I", "-sS", "--connect-timeout", "4", "--max-time", "8", "https://example.com"],
        "https_without_dns": ["curl", "--noproxy", "*", "-4", "-I", "-sS", "--connect-timeout", "4", "--max-time", "8", "https://1.1.1.1"],
        "firewall_iptables": ["sudo", "-n", "iptables-save"],
        "firewall_nft": ["sudo", "-n", "nft", "list", "ruleset"],
    }
    targets = list(dict.fromkeys(["172.25.96.1", "172.18.177.130", "192.168.2.232",
                                 "192.168.2.1", "10.0.4.15", "10.0.4.16", "1.1.1.1"] + args.target))
    for target in targets:
        if target.startswith("-") or not re.fullmatch(r"[A-Za-z0-9_.:-]+", target):
            parser.error("无效的测试目标: " + target)
        jobs["route_" + target] = ["ip", "route", "get", target]
        jobs["ping_" + target] = ["ping", "-n", "-c", "2", "-W", "2", target]
    for server in ["172.18.177.130", "10.0.4.15", "10.0.4.16", "223.5.5.5", "114.114.114.114"]:
        jobs["dns_" + server] = ["dig", "@" + server, "example.com", "A", "+time=2", "+tries=1"]

    for filename in ["/etc/resolv.conf", "/run/systemd/resolve/resolv.conf"]:
        try:
            path = Path(filename)
            report.setdefault("dns_files", {})[filename] = {"resolved_path": str(path.resolve()), "text": path.read_text()}
        except OSError as exc:
            report.setdefault("dns_files", {})[filename] = {"error": str(exc)}
    # Keep only route/DNS lifecycle lines, excluding authentication and user data.
    log = Path("/tmp/HillstoneSecureConnect/log/secureconnect.log")
    try:
        allowed = re.compile(r"(Set Tunnel |Unset Tunnel |Success to (add|delete) route entry|Install DNS server)")
        with log.open(errors="replace") as stream:
            report["hillstone_network_log"] = [line.rstrip() for line in stream if allowed.search(line)][-180:]
    except OSError as exc:
        report["hillstone_network_log_error"] = str(exc)
    save()
    print("正在采集，通常约 20–60 秒。报告目录：", output, flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        pending = {pool.submit(run, command): name for name, command in jobs.items()}
        for future in concurrent.futures.as_completed(pending):
            name = pending[future]
            report["checks"][name] = future.result()
            save()
            print("已完成:", name, flush=True)
    report["completed_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    save()
    print("\n采集完成，可以断开 VPN。请将此报告路径告诉我：\n" + str(output / "report.json"))


if __name__ == "__main__":
    main()
