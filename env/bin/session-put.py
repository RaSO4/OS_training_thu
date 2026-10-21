#!/usr/bin/env python3
"""session-put.py <session_id> <remote_rel_path> <local_file>  — PUT 到板卡会话文件接口"""
import json, sys, urllib.request, ssl

sid, rel, local = sys.argv[1], sys.argv[2], sys.argv[3]
tok = ""
d = json.load(open('/root/.config/ostool/hosts.json'))
def find(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == 'access_token' and isinstance(v, str):
                return v
            r = find(v)
            if r:
                return r
    return None
tok = find(d) or ""

url = f"https://ostool.muxai.net:14321/api/v1/sessions/{sid}/files"
data = open(local, 'rb').read()
req = urllib.request.Request(url, data=data, method='PUT', headers={
    'Authorization': f'Bearer {tok}',
    'X-File-Path': rel,
    'Content-Type': 'application/octet-stream',
})
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
try:
    with urllib.request.urlopen(req, context=ctx, timeout=600) as r:
        print("HTTP", r.status)
        print(r.read().decode()[:400])
except urllib.error.HTTPError as e:
    print("HTTP", e.code, e.read().decode()[:400])
except Exception as e:
    print("FAIL", type(e).__name__, e)
