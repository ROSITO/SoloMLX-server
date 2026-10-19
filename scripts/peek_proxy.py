"""Logging proxy :8098 -> :8099. One JSON line per chat request: size, status, timing, cached/prompt/completion tokens."""
import json, re, sys, time, httpx, uvicorn
from starlette.applications import Starlette
from starlette.responses import StreamingResponse
from starlette.routing import Route

UP = "http://127.0.0.1:8099"
LOG = open(sys.argv[1], "a", buffering=1)
client = httpx.AsyncClient(timeout=None)

async def proxy(request):
    body = await request.body()
    info = None
    if request.url.path.endswith("/chat/completions"):
        d = json.loads(body)
        info = dict(t=time.strftime("%H:%M:%S"), kind="agent" if d.get("tools") else "other", msgs=len(d["messages"]))
    r = await client.send(client.build_request(request.method, UP + request.url.path, content=body,
                          headers={k: v for k, v in request.headers.items() if k.lower() not in ("host", "content-length")}), stream=True)
    t0, first = time.time(), None

    async def gen():
        nonlocal first
        tail = b""
        async for chunk in r.aiter_raw():
            if first is None and b'"content"' in chunk: first = time.time()
            tail = (tail + chunk)[-4000:]
            yield chunk
        if info is not None:
            g = lambda k: (lambda m: int(m[-1]) if m else None)(re.findall(rb'"' + k + rb'":\s*(\d+)', tail))
            info.update(status=r.status_code, secs=round(time.time() - t0, 1), ttft=round((first or time.time()) - t0, 1),
                        prompt_tokens=g(b"prompt_tokens"), cached=g(b"cached_tokens"), completion_tokens=g(b"completion_tokens"))
            if r.status_code != 200: info["error"] = tail[-200:].decode(errors="replace")
            LOG.write(json.dumps(info) + "\n")

    return StreamingResponse(gen(), status_code=r.status_code, headers={"content-type": r.headers.get("content-type", "")})

uvicorn.run(Starlette(routes=[Route("/{p:path}", proxy, methods=["GET", "POST"])]), host="127.0.0.1", port=8098, log_level="warning")
