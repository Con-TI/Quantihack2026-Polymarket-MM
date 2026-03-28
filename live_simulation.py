"""
live_simulation.py — Live paper-trading simulation with Polymarket data.

Serves a real-time web dashboard at http://localhost:8080.
Runs 3 parallel Controller instances (identity / sqrt / square queue funcs)
with a 3-tick message delay against the live Polymarket CLOB websocket.

Usage:
    python python/live_simulation.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
import math
import webbrowser
from collections import deque
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import aiohttp
from aiohttp import web
import websockets

from baseline_structure import Controller, BacktestOID, BacktestLogger, TradingModel

# =========================================================================
# Constants
# =========================================================================

WSSMARKET_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
GAMMA_URL = (
    "https://gamma-api.polymarket.com/events?"
    "tag_id=102467&active=true&closed=false&ascending=true&order=startTime&limit=12"
)
ASSETS   = ["btc", "eth"]
SLUG_MAP = {"b": "btc", "x": "xrp", "e": "eth", "s": "sol"}
TICK_DELAY = 3
WINDOW_MS  = 15 * 60 * 1000

SIM_CONFIGS = [
    ("\u03bb(x)=x  Identity",  lambda x: x),
    ("\u03bb(x)=\u221ax  Sqrt", lambda x: math.sqrt(max(x, 0))),
    ("\u03bb(x)=x\u00b2  Square", lambda x: x * x),
]
SIM_COLORS = ["#58a6ff", "#3fb950", "#f0883e"]


class NaiveModel(TradingModel):
    def run(self, msg):
        return 0.0


# =========================================================================
# Gamma API + translate  (unchanged logic)
# =========================================================================

async def fetch_contracts(session, target_start):
    url = GAMMA_URL + "&start_date_min=" + quote(
        (target_start - timedelta(days=1, minutes=15)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        safe="TZ",
    )
    async with session.get(url) as resp:
        events = await resp.json()
    naive = target_start.replace(tzinfo=None)
    return [e for e in events
            if datetime.strptime(e["startTime"], "%Y-%m-%dT%H:%M:%SZ") == naive]


def build_asset_map(contracts):
    amap = {}
    for ev in contracts:
        slug = ev.get("slug", "")
        ul = SLUG_MAP.get(slug[0] if slug else "")
        if ul not in ASSETS:
            continue
        tids = json.loads(ev["markets"][0]["clobTokenIds"])
        raw_oc = ev["markets"][0].get("outcomes", "")
        outcomes = ([o.strip().strip('"') for o in raw_oc.strip("[]").split(",")]
                    if isinstance(raw_oc, str) else raw_oc)
        for tid, oc in zip(tids, outcomes):
            amap[str(tid)] = (ul, oc)
    return amap


def translate(raw, amap, ts_floor):
    et = raw.get("event_type")
    msgs = []
    if et == "book":
        aid = str(raw.get("asset_id", ""))
        k = amap.get(aid)
        if not k: return msgs
        ul, oc = k
        ts = max(int(raw.get("timestamp", 0)) - ts_floor, 0)
        d = dict(raw); d.update(underlying=ul, outcome=oc, timestamp=ts)
        msgs.append({"type": "book", "data": d})
    elif et == "price_change":
        ts = max(int(raw.get("timestamp", 0)) - ts_floor, 0)
        for item in raw.get("price_changes", []):
            aid = str(item.get("asset_id", ""))
            k = amap.get(aid)
            if not k: continue
            ul, oc = k
            d = dict(item); d.update(underlying=ul, outcome=oc, timestamp=ts)
            msgs.append({"type": "price_change", "data": d})
    elif et == "last_trade_price":
        aid = str(raw.get("asset_id", ""))
        k = amap.get(aid)
        if not k: return msgs
        ul, oc = k
        ts = max(int(raw.get("timestamp", 0)) - ts_floor, 0)
        d = dict(raw); d.update(underlying=ul, outcome=oc, timestamp=ts)
        msgs.append({"type": "last_trade", "data": d})
    return msgs


# =========================================================================
# Browser WS clients
# =========================================================================

_ws_clients: set = set()


async def _broadcast(obj):
    raw = json.dumps(obj)
    for ws in list(_ws_clients):
        try:
            await ws.send_str(raw)
        except Exception:
            _ws_clients.discard(ws)


# =========================================================================
# aiohttp handlers
# =========================================================================

async def _handle_index(request):
    return web.Response(text=HTML, content_type="text/html")


async def _handle_ws(request):
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    _ws_clients.add(ws)
    try:
        async for _ in ws:
            pass
    except Exception:
        pass
    finally:
        _ws_clients.discard(ws)
    return ws


# =========================================================================
# Simulation loop
# =========================================================================

async def _run_simulation():
    now = datetime.now(timezone.utc)
    wm = (now.minute // 15) * 15
    ws_start = now.replace(minute=wm, second=0, microsecond=0)
    ws_end = ws_start + timedelta(minutes=15)
    ts_floor = int(ws_start.timestamp() * 1000)

    async with aiohttp.ClientSession() as session:
        contracts = await fetch_contracts(session, ws_start)
        if not contracts:
            alt = ws_start - timedelta(minutes=15)
            contracts = await fetch_contracts(session, alt)
            if contracts:
                ws_start, ws_end = alt, alt + timedelta(minutes=15)
                ts_floor = int(ws_start.timestamp() * 1000)
        if not contracts:
            print("No contracts found.  Check network / VPN.")
            await _broadcast({"type": "error", "msg": "No contracts found"})
            return

        amap = build_asset_map(contracts)
        if not amap:
            print("No BTC/ETH contracts in current window.")
            return

        books = set(amap.values())
        print(f"Window : {ws_start.strftime('%H:%M')} - "
              f"{ws_end.strftime('%H:%M')} UTC")
        print(f"Books  : {sorted(books)}")

    # ── Init message to browser ───────────────────────────────────
    await _broadcast({
        "type": "init",
        "window_start": ws_start.strftime("%H:%M"),
        "window_end": ws_end.strftime("%H:%M"),
        "sim_names": [n for n, _ in SIM_CONFIGS],
        "sim_colors": SIM_COLORS,
    })

    # ── Controllers ───────────────────────────────────────────────
    sims = []
    for name, qp in SIM_CONFIGS:
        lg = BacktestLogger()
        c = Controller(
            alpha_function=NaiveModel().run, sender=BacktestOID(),
            logger=lg, live=True, inventory_limit=10, level_range=3,
            skew_per_unit=1.0, window_ms=WINDOW_MS, queue_entry_frac=0.15,
            intensity_weight=0.5, trend_weight=2.0, queue_pos_func=qp,
        )
        for ul, oc in books:
            c.add_book(ul, oc)
        sims.append({"name": name, "controller": c, "logger": lg})

    # ── State ─────────────────────────────────────────────────────
    delay = deque()
    tick = 0
    pending = []

    async def flush():
        nonlocal pending
        if not pending:
            return
        await _broadcast({"type": "batch", "ticks": pending})
        pending = []

    async def flush_loop():
        while True:
            await asyncio.sleep(0.18)
            await flush()

    flush_task = asyncio.create_task(flush_loop())

    # ── Polymarket WS ─────────────────────────────────────────────
    poly_ws = await websockets.connect(WSSMARKET_URL)
    await poly_ws.send(json.dumps({"assets_ids": list(amap), "type": "market"}))
    print(f"Streaming {len(amap)} tokens ...")

    async def ping():
        while True:
            try:
                await poly_ws.send("PING")
            except websockets.ConnectionClosed:
                break
            await asyncio.sleep(20)

    ping_task = asyncio.create_task(ping())

    try:
        async for raw_msg in poly_ws:
            if raw_msg in ("PONG", ""):
                continue
            try:
                data = json.loads(raw_msg)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            if data.get("event_type") not in ("book", "price_change",
                                               "last_trade_price"):
                continue

            for m in translate(data, amap, ts_floor):
                delay.append(m)

            while len(delay) > TICK_DELAY:
                delayed = delay.popleft()
                tick += 1

                # Feed to controllers, capture new trades
                new_trades = [[] for _ in range(3)]
                for i, sim in enumerate(sims):
                    lg = sim["logger"]
                    old_n = len(lg.trades.get("timestamp", []))
                    sim["controller"].parse(delayed)
                    new_n = len(lg.trades.get("timestamp", []))
                    for j in range(old_n, new_n):
                        new_trades[i].append({
                            "ul": lg.trades["underlying"][j],
                            "oc": lg.trades["outcome"][j],
                            "side": int(lg.trades["side"][j]),
                            "price": int(lg.trades["price"][j]),
                        })

                # Book state from controller 0
                ref = sims[0]["controller"]
                bk_data = {}
                for ul in ASSETS:
                    for oc in ("Up", "Down"):
                        b = ref.books.get((ul, oc))
                        if b and b._book_initialized:
                            bk_data[f"{ul}_{oc}"] = {
                                "bid": b.best_bid,
                                "ask": b.best_ask,
                                "mid": round((b.best_bid + b.best_ask) / 2, 1),
                            }

                # Sim metrics
                sim_data = []
                for i, sim in enumerate(sims):
                    c = sim["controller"]
                    inv = sum(b.inventory * b.mid_price
                              for b in c.books.values())
                    sim_data.append({
                        "mtm": round(c.wealth - 10000 + inv, 1),
                        "fills": len(sim["logger"].trades.get("timestamp", [])),
                        "btc_exp": c.net_exposure("btc"),
                        "eth_exp": c.net_exposure("eth"),
                        "trades": new_trades[i],
                    })

                pending.append({
                    "tick": tick, "books": bk_data, "sims": sim_data,
                })

            if datetime.now(timezone.utc) >= ws_end:
                break

    except websockets.ConnectionClosed as e:
        print(f"WS disconnected: {e}")
    finally:
        ping_task.cancel()
        flush_task.cancel()
        try: await ping_task
        except Exception: pass
        try: await flush_task
        except Exception: pass
        try: await poly_ws.close()
        except Exception: pass

    await flush()
    await _broadcast({"type": "done"})
    print("Window complete.  Dashboard still live at http://localhost:8080")

    # Print summary
    print("\n" + "=" * 60)
    for i, sim in enumerate(sims):
        c = sim["controller"]
        _, tdf = sim["logger"].df_parse()
        inv = sum(b.inventory * b.mid_price for b in c.books.values())
        mtm = c.wealth - 10000 + inv
        print(f"  {sim['name']:25s}  MTM: {mtm:+.1f}c  fills: {len(tdf)}")
    print("=" * 60)


# =========================================================================
# HTML dashboard  (Chart.js v4 from CDN)
# =========================================================================

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Live Simulation</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
<style>
:root{--bg:#0d1117;--card:#161b22;--border:#21262d;--txt:#e6edf3;--dim:#7d8590;
--green:#3fb950;--red:#f85149;--blue:#58a6ff;--orange:#f0883e;--grid:#1b2332}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--txt);font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;padding:10px 14px}
header{display:flex;align-items:center;justify-content:space-between;padding:8px 14px;
background:var(--card);border:1px solid var(--border);border-radius:8px;margin-bottom:10px}
.live{display:flex;align-items:center;gap:8px}
.dot{width:8px;height:8px;background:var(--green);border-radius:50%;animation:pulse 2s infinite}
.dot.off{background:var(--dim);animation:none}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
.hdr-right{display:flex;gap:18px;font-size:13px;color:var(--dim)}
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:10px}
.card{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:10px 14px;border-left:3px solid var(--blue)}
.card:nth-child(2){border-left-color:var(--green)}.card:nth-child(3){border-left-color:var(--orange)}
.card .nm{font-size:11px;color:var(--dim);margin-bottom:2px}
.card .pnl{font-size:24px;font-weight:700;letter-spacing:-.5px}
.card .det{font-size:11px;color:var(--dim);margin-top:3px}
.up{color:var(--green)}.dn{color:var(--red)}
.g2{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:10px}
.panel{background:var(--card);border:1px solid var(--border);border-radius:8px;padding:8px 10px}
.panel .t{font-size:11px;font-weight:600;color:var(--dim);margin-bottom:4px}
.ph{height:195px;position:relative}.eh{height:155px;position:relative}.plh{height:155px;position:relative}
footer{display:flex;align-items:center;justify-content:space-between;margin-top:4px}
footer .info{font-size:12px;color:var(--dim)}
#hist-btn{background:var(--card);border:1px solid var(--border);color:var(--blue);
padding:7px 18px;border-radius:6px;font-size:13px;font-weight:600;cursor:pointer}
#hist-btn:hover{background:#1f3044}
.modal{display:none;position:fixed;inset:0;background:rgba(0,0,0,.7);z-index:100;
overflow-y:auto;padding:40px 20px}
.modal.show{display:block}
.mbody{max-width:800px;margin:0 auto;background:var(--card);border:1px solid var(--border);
border-radius:12px;padding:24px}
.mbody h3{font-size:16px;margin-bottom:16px;color:var(--blue)}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;color:var(--dim);font-weight:600;padding:6px 10px;border-bottom:1px solid var(--border)}
td{padding:6px 10px;border-bottom:1px solid var(--border)}
.close-btn{position:absolute;top:16px;right:20px;background:none;border:none;color:var(--dim);
font-size:22px;cursor:pointer}.close-btn:hover{color:var(--txt)}
</style>
</head>
<body>

<header>
  <div class="live"><span class="dot" id="dot"></span><span id="status">Connecting&hellip;</span></div>
  <div class="hdr-right"><span id="wtime"></span><span id="tcnt">Tick: 0</span></div>
</header>

<div class="cards">
  <div class="card"><div class="nm" id="cn0">&mdash;</div><div class="pnl" id="cp0">+0.0c</div><div class="det" id="cd0">0 fills</div></div>
  <div class="card"><div class="nm" id="cn1">&mdash;</div><div class="pnl" id="cp1">+0.0c</div><div class="det" id="cd1">0 fills</div></div>
  <div class="card"><div class="nm" id="cn2">&mdash;</div><div class="pnl" id="cp2">+0.0c</div><div class="det" id="cd2">0 fills</div></div>
</div>

<div class="g2">
  <div class="panel"><div class="t">BTC Up</div><div class="ph"><canvas id="c-btc_Up"></canvas></div></div>
  <div class="panel"><div class="t">ETH Up</div><div class="ph"><canvas id="c-eth_Up"></canvas></div></div>
</div>
<div class="g2">
  <div class="panel"><div class="t">BTC Down</div><div class="ph"><canvas id="c-btc_Down"></canvas></div></div>
  <div class="panel"><div class="t">ETH Down</div><div class="ph"><canvas id="c-eth_Down"></canvas></div></div>
</div>
<div class="g2">
  <div class="panel"><div class="t">BTC Net Exposure</div><div class="eh"><canvas id="c-btc-exp"></canvas></div></div>
  <div class="panel"><div class="t">ETH Net Exposure</div><div class="eh"><canvas id="c-eth-exp"></canvas></div></div>
</div>
<div class="panel" style="margin-bottom:10px">
  <div class="t">Combined MTM PnL</div><div class="plh"><canvas id="c-pnl"></canvas></div>
</div>

<footer>
  <div class="info" id="finfo"></div>
  <button id="hist-btn" onclick="openModal()">View History</button>
</footer>

<div class="modal" id="modal">
  <div class="mbody" style="position:relative">
    <button class="close-btn" onclick="closeModal()">&times;</button>
    <h3>Window Summary</h3>
    <div id="mcontent"></div>
  </div>
</div>

<script>
const SC=['#58a6ff','#3fb950','#f0883e'],GRN='#3fb950',RED='#f85149',GRD='#1b2332',DIM='#7d8590';
const BOOKS=['btc_Up','btc_Down','eth_Up','eth_Down'],ULS=['btc','eth'];
let simNames=['Identity','Sqrt','Square'];
const ch={};
let lastSims=null, maxTick=0;

function co(extra={}){return{animation:false,responsive:true,maintainAspectRatio:false,
plugins:{legend:{display:false},tooltip:{mode:'nearest',intersect:false}},
scales:{x:{type:'linear',min:0,ticks:{color:DIM,maxTicksLimit:10,font:{size:10}},grid:{color:GRD}},
y:{ticks:{color:DIM,font:{size:10}},grid:{color:GRD}}},...extra}}

function init(){
  BOOKS.forEach(b=>{
    const ctx=document.getElementById('c-'+b).getContext('2d');
    ch[b]=new Chart(ctx,{type:'line',data:{datasets:[
      {data:[],borderColor:'#fff',borderWidth:1.5,pointRadius:0,tension:0,order:3},
      {data:[],borderColor:GRN,borderWidth:1,pointRadius:0,tension:0,borderDash:[3,3],order:4},
      {data:[],borderColor:RED,borderWidth:1,pointRadius:0,tension:0,borderDash:[3,3],order:4},
      ...SC.flatMap(c=>[
        {type:'scatter',data:[],backgroundColor:c,borderColor:'#fff',borderWidth:.5,pointRadius:5,pointStyle:'triangle',rotation:0,order:1},
        {type:'scatter',data:[],backgroundColor:c,borderColor:'#fff',borderWidth:.5,pointRadius:5,pointStyle:'triangle',rotation:180,order:1}
      ])
    ]},options:co()});
  });
  ULS.forEach(ul=>{
    const ctx=document.getElementById('c-'+ul+'-exp').getContext('2d');
    ch[ul+'_exp']=new Chart(ctx,{type:'line',data:{datasets:SC.map(c=>({
      data:[],borderColor:c,borderWidth:1.5,pointRadius:0,tension:0
    }))},options:co()});
  });
  const pc=document.getElementById('c-pnl').getContext('2d');
  ch.pnl=new Chart(pc,{type:'line',data:{datasets:SC.map(c=>({
    data:[],borderColor:c,borderWidth:1.5,pointRadius:0,tension:0
  }))},options:co()});
}

function tick(t){
  const tk=t.tick;
  for(const b of BOOKS){
    const bk=t.books[b]; if(!bk) continue;
    const c=ch[b];
    c.data.datasets[0].data.push({x:tk,y:bk.mid});
    c.data.datasets[1].data.push({x:tk,y:bk.bid});
    c.data.datasets[2].data.push({x:tk,y:bk.ask});
  }
  for(let i=0;i<t.sims.length;i++){
    const s=t.sims[i];
    for(const tr of s.trades){
      const b=tr.ul+'_'+tr.oc; const c=ch[b]; if(!c) continue;
      c.data.datasets[3+i*2+(tr.side===1?0:1)].data.push({x:tk,y:tr.price});
    }
    ch.btc_exp.data.datasets[i].data.push({x:tk,y:s.btc_exp});
    ch.eth_exp.data.datasets[i].data.push({x:tk,y:s.eth_exp});
    ch.pnl.data.datasets[i].data.push({x:tk,y:s.mtm});
  }
  if(tk>maxTick) maxTick=tk;
}

function cards(sims){
  lastSims=sims;
  for(let i=0;i<sims.length;i++){
    const s=sims[i];
    const el=document.getElementById('cp'+i);
    el.textContent=(s.mtm>=0?'+':'')+s.mtm.toFixed(1)+'c';
    el.className='pnl '+(s.mtm>=0?'up':'dn');
    document.getElementById('cd'+i).textContent=
      s.fills+' fills  |  BTC: '+(s.btc_exp>0?'+':'')+s.btc_exp.toFixed(0)+
      '  ETH: '+(s.eth_exp>0?'+':'')+s.eth_exp.toFixed(0);
  }
}

function refresh(){
  Object.values(ch).forEach(c=>{c.options.scales.x.max=maxTick+1;c.update('none')});
  document.getElementById('tcnt').textContent='Tick: '+maxTick;
}

function openModal(){
  if(!lastSims){return}
  let h='<table><tr><th>Strategy</th><th>MTM PnL</th><th>Fills</th><th>BTC exp</th><th>ETH exp</th></tr>';
  for(let i=0;i<lastSims.length;i++){
    const s=lastSims[i];
    const clr=s.mtm>=0?GRN:RED;
    h+='<tr><td style="color:'+SC[i]+'">'+simNames[i]+'</td>';
    h+='<td style="color:'+clr+';font-weight:700">'+(s.mtm>=0?'+':'')+s.mtm.toFixed(1)+'c</td>';
    h+='<td>'+s.fills+'</td>';
    h+='<td>'+(s.btc_exp>0?'+':'')+s.btc_exp.toFixed(0)+'</td>';
    h+='<td>'+(s.eth_exp>0?'+':'')+s.eth_exp.toFixed(0)+'</td></tr>';
  }
  h+='</table>';
  document.getElementById('mcontent').innerHTML=h;
  document.getElementById('modal').classList.add('show');
}
function closeModal(){document.getElementById('modal').classList.remove('show')}

// WebSocket
const ws=new WebSocket('ws://'+location.host+'/ws');
ws.onopen=()=>{document.getElementById('status').textContent='Live'};
ws.onmessage=e=>{
  const m=JSON.parse(e.data);
  if(m.type==='init'){
    document.getElementById('wtime').textContent=m.window_start+' \u2013 '+m.window_end+' UTC';
    simNames=m.sim_names;
    for(let i=0;i<m.sim_names.length;i++) document.getElementById('cn'+i).textContent=m.sim_names[i];
  }else if(m.type==='batch'){
    for(const t of m.ticks) tick(t);
    if(m.ticks.length) cards(m.ticks[m.ticks.length-1].sims);
    refresh();
  }else if(m.type==='done'){
    document.getElementById('status').textContent='Window complete';
    document.getElementById('dot').classList.add('off');
    document.getElementById('finfo').textContent='Contract expired \u2014 data frozen';
  }else if(m.type==='error'){
    document.getElementById('status').textContent=m.msg;
    document.getElementById('dot').classList.add('off');
  }
};
ws.onclose=()=>{
  document.getElementById('status').textContent='Disconnected';
  document.getElementById('dot').classList.add('off');
};

init();
</script>
</body>
</html>"""


# =========================================================================
# Entry point
# =========================================================================

async def main():
    app = web.Application()
    app.router.add_get("/", _handle_index)
    app.router.add_get("/ws", _handle_ws)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "localhost", 8080)
    await site.start()
    print("Dashboard: http://localhost:8080")
    webbrowser.open("http://localhost:8080")

    await _run_simulation()

    # Keep server alive so user can review charts
    print("Press Ctrl+C to exit.")
    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        await runner.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown.")
