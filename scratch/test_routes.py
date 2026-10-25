import urllib.request
import json

base = "http://127.0.0.1:5000"
endpoints = [
    "/",
    "/sheets",
    "/select",
    "/verify/s1",
    "/verify/s6",
    "/metrics-dashboard",
    "/api/status"
]

print("Testing Flask endpoints:")
for ep in endpoints:
    url = base + ep
    try:
        req = urllib.request.urlopen(url, timeout=10)
        content = req.read().decode('utf-8')
        print(f"  {ep:<20} -> status {req.status}, length: {len(content):,} bytes")
    except Exception as e:
        print(f"  {ep:<20} -> ERROR: {e}")
