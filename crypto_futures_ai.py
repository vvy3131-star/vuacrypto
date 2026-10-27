# -*- coding: utf-8 -*-
"""
Crypto Futures AI Analyzer  -  chạy bằng:  streamlit run crypto_futures_ai.py

Chức năng:
  * Lấy dữ liệu nến + funding/OI/Long-Short từ Binance Futures (fallback Bybit), không cần API key
  * Phân tích kỹ thuật chuyên sâu (EMA, ADX, Supertrend, Ichimoku, MACD, RSI, StochRSI, Bollinger,
    OBV/Volume, mẫu nến, phân kỳ RSI, cấu trúc thị trường, hỗ trợ/kháng cự, Fibonacci)
  * Huấn luyện mô hình ML (Gradient Boosting) trên chính dữ liệu coin, kiểm định out-of-sample
  * Tổng hợp đa khung giờ -> MUA / BÁN / CHỜ + giải thích lý do
  * Kế hoạch lệnh: Entry, Cắt lỗ (SL), Chốt lời TP1/TP2/TP3, tính khối lượng theo % rủi ro
  * Tin tức mới nhất kèm ngày giờ + chấm điểm cảm xúc, Fear & Greed Index

CẢNH BÁO: Không có công cụ nào dự đoán chính xác 100%. Đây là công cụ hỗ trợ ra quyết định,
không phải lời khuyên tài chính. Futures có đòn bẩy, có thể mất toàn bộ vốn.
"""
import re
import datetime as dt
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import HistGradientBoostingClassifier

VN_TZ = ZoneInfo("Asia/Ho_Chi_Minh")
BINANCE = "https://fapi.binance.com"
BYBIT = "https://api.bybit.com"
HEADERS = {"User-Agent": "Mozilla/5.0 (CryptoFuturesAI)"}

BYBIT_INTERVAL = {"5m": "5", "15m": "15", "1h": "60", "4h": "240", "1d": "D"}
TF_LABEL = {
    "5m": "5 phút (scalp)",
    "15m": "15 phút (scalp)",
    "1h": "1 giờ (intraday)",
    "4h": "4 giờ (swing ngắn)",
    "1d": "1 ngày (swing dài)",
}
TF_WEIGHT = {"5m": 0.10, "15m": 0.15, "1h": 0.25, "4h": 0.30, "1d": 0.30}
POPULAR = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT", "DOGEUSDT",
           "ADAUSDT", "AVAXUSDT", "LINKUSDT", "SUIUSDT", "TONUSDT", "PEPEUSDT"]
COIN_NAMES = {
    "BTC": ["bitcoin", "btc"], "ETH": ["ethereum", "ether", "eth"], "SOL": ["solana", "sol"],
    "BNB": ["bnb", "binance coin"], "XRP": ["xrp", "ripple"], "DOGE": ["dogecoin", "doge"],
    "ADA": ["cardano", "ada"], "AVAX": ["avalanche", "avax"], "LINK": ["chainlink", "link"],
    "SUI": ["sui"], "TON": ["toncoin", "ton"], "PEPE": ["pepe"],
}

POS_WORDS = ["surge", "rally", "bull", "soar", "gain", "approve", "approval", "adopt", "record high",
             "all-time high", "breakout", "upgrade", "partnership", "inflow", "accumulat", "jump",
             "rise", "rebound", "recover", "launch", "integrat", "buy"]
NEG_WORDS = ["crash", "plunge", "bear", "hack", "exploit", "ban", "lawsuit", "sues", "outflow",
             "liquidat", "dump", "fall", "drop", "fraud", "delist", "sell-off", "selloff", "decline",
             "scam", "reject", "slump", "tumble", "warn", "probe", "investigat"]


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DỮ LIỆU
# ═══════════════════════════════════════════════════════════════════════════════
def _get_json(url, params=None, timeout=15):
    r = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=60, show_spinner=False)
def fetch_klines(symbol: str, interval: str, limit: int = 1000) -> pd.DataFrame:
    """Nến futures. Thử Binance trước, lỗi thì dùng Bybit."""
    try:
        raw = _get_json(f"{BINANCE}/fapi/v1/klines",
                        {"symbol": symbol, "interval": interval, "limit": limit})
        if not isinstance(raw, list) or not raw:
            raise ValueError("Binance trả về dữ liệu rỗng")
        df = pd.DataFrame(raw).iloc[:, :6]
        df.columns = ["time", "open", "high", "low", "close", "volume"]
        df["time"] = pd.to_datetime(df["time"], unit="ms", utc=True)
    except Exception:
        raw = _get_json(f"{BYBIT}/v5/market/kline",
                        {"category": "linear", "symbol": symbol,
                         "interval": BYBIT_INTERVAL[interval], "limit": min(limit, 1000)})
        rows = raw.get("result", {}).get("list", [])
        if not rows:
            raise ValueError(f"Không lấy được dữ liệu cho {symbol}")
        df = pd.DataFrame(rows).iloc[:, :6]
        df.columns = ["time", "open", "high", "low", "close", "volume"]
        df["time"] = pd.to_datetime(df["time"].astype("int64"), unit="ms", utc=True)
        df = df.iloc[::-1]  # Bybit trả mới nhất trước
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df["time"] = df["time"].dt.tz_convert(VN_TZ)
    return df.set_index("time").sort_index()


@st.cache_data(ttl=120, show_spinner=False)
def fetch_derivs(symbol: str) -> dict:
    out = {"funding": None, "oi": None, "ls": None, "source": None}
    try:
        p = _get_json(f"{BINANCE}/fapi/v1/premiumIndex", {"symbol": symbol}, 10)
        out["funding"] = float(p["lastFundingRate"])
        out["source"] = "Binance Futures"
    except Exception:
        pass
    try:
        o = _get_json(f"{BINANCE}/fapi/v1/openInterest", {"symbol": symbol}, 10)
        out["oi"] = float(o["openInterest"])
    except Exception:
        pass
    try:
        ls = _get_json(f"{BINANCE}/futures/data/globalLongShortAccountRatio",
                       {"symbol": symbol, "period": "1h", "limit": 1}, 10)
        out["ls"] = float(ls[0]["longShortRatio"])
    except Exception:
        pass
    if out["funding"] is None:
        try:
            t = _get_json(f"{BYBIT}/v5/market/tickers", {"category": "linear", "symbol": symbol}, 10)
            it = t["result"]["list"][0]
            out["funding"] = float(it["fundingRate"])
            out["oi"] = float(it.get("openInterest", 0) or 0) or out["oi"]
            out["source"] = "Bybit"
        except Exception:
            pass
    return out


@st.cache_data(ttl=600, show_spinner=False)
def fetch_fng():
    try:
        d = _get_json("https://api.alternative.me/fng/", {"limit": 1}, 10)["data"][0]
        return int(d["value"]), d["value_classification"]
    except Exception:
        return None, None


def _rss(url, source):
    r = requests.get(url, headers=HEADERS, timeout=12)
    r.raise_for_status()
    root = ET.fromstring(r.content)
    out = []
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        pub = it.findtext("pubDate")
        try:
            ts = parsedate_to_datetime(pub).timestamp()
        except Exception:
            continue
        body = re.sub(r"<[^>]+>", " ", it.findtext("description") or "")
        out.append({"title": title, "body": body[:400], "url": it.findtext("link") or "",
                    "source": source, "ts": ts, "cat": False})
    return out


@st.cache_data(ttl=300, show_spinner=False)
def fetch_news(base: str):
    items = []
    # CryptoCompare: tin theo coin
    for params, is_cat in (({"lang": "EN", "categories": base}, True), ({"lang": "EN"}, False)):
        try:
            data = _get_json("https://min-api.cryptocompare.com/data/v2/news/", params, 12).get("Data", [])
            if isinstance(data, list):
                for n in data:
                    items.append({"title": n.get("title", "").strip(), "body": (n.get("body") or "")[:400],
                                  "url": n.get("url", ""), "source": n.get("source", "CryptoCompare"),
                                  "ts": float(n.get("published_on", 0)), "cat": is_cat})
        except Exception:
            pass
    for url, src in (("https://cointelegraph.com/rss", "Cointelegraph"),
                     ("https://www.coindesk.com/arc/outboundfeeds/rss/", "CoinDesk")):
        try:
            items += _rss(url, src)
        except Exception:
            pass

    kws = COIN_NAMES.get(base.upper(), [base.lower()])
    pat = re.compile(r"\b(" + "|".join(re.escape(k) for k in kws) + r")\b", re.I)
    seen, specific, general = set(), [], []
    for n in sorted(items, key=lambda x: x["ts"], reverse=True):
        key = n["title"].lower()
        if not key or key in seen:
            continue
        seen.add(key)
        if n["cat"] or pat.search(n["title"] + " " + n["body"]):
            n["specific"] = True
            specific.append(n)
        else:
            n["specific"] = False
            general.append(n)
    out = specific[:10]
    if len(out) < 6:
        out += general[: 10 - len(out)]
    return sorted(out, key=lambda x: x["ts"], reverse=True)


def news_sentiment(title_body: str) -> int:
    t = title_body.lower()
    p = sum(w in t for w in POS_WORDS)
    n = sum(w in t for w in NEG_WORDS)
    return 1 if p > n else -1 if n > p else 0


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CHỈ BÁO KỸ THUẬT
# ═══════════════════════════════════════════════════════════════════════════════
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def calc_rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def true_range(df):
    pc = df["close"].shift(1)
    return pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()],
                     axis=1).max(axis=1)


def calc_atr(df, n=14):
    return true_range(df).ewm(alpha=1 / n, adjust=False).mean()


def calc_adx(df, n=14):
    up, dn = df["high"].diff(), -df["low"].diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    a = true_range(df).ewm(alpha=1 / n, adjust=False).mean()
    pdi = 100 * pdm.ewm(alpha=1 / n, adjust=False).mean() / a
    mdi = 100 * mdm.ewm(alpha=1 / n, adjust=False).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean().fillna(0), pdi.fillna(0), mdi.fillna(0)


def calc_supertrend(df, n=10, mult=3.0):
    a = calc_atr(df, n).values
    hl2 = ((df["high"] + df["low"]) / 2).values
    cl = df["close"].values
    ub, lb = hl2 + mult * a, hl2 - mult * a
    fub, flb = ub.copy(), lb.copy()
    dirn = np.ones(len(df))
    st_line = np.full(len(df), np.nan)
    for i in range(1, len(df)):
        fub[i] = ub[i] if (ub[i] < fub[i - 1] or cl[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = lb[i] if (lb[i] > flb[i - 1] or cl[i - 1] < flb[i - 1]) else flb[i - 1]
        if dirn[i - 1] == 1:
            dirn[i] = -1 if cl[i] < flb[i] else 1
        else:
            dirn[i] = 1 if cl[i] > fub[i] else -1
        st_line[i] = flb[i] if dirn[i] == 1 else fub[i]
    return pd.Series(st_line, index=df.index), pd.Series(dirn, index=df.index)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c, h, l, v = d["close"], d["high"], d["low"], d["volume"]
    for n in (20, 50, 200):
        d[f"ema{n}"] = ema(c, n)
    d["rsi"] = calc_rsi(c)
    r = d["rsi"]
    lo, hi = r.rolling(14).min(), r.rolling(14).max()
    d["stochrsi"] = ((r - lo) / (hi - lo).replace(0, np.nan)).fillna(0.5)
    k = d["stochrsi"].rolling(3).mean()
    d["stoch_k"], d["stoch_d"] = k, k.rolling(3).mean()
    d["macd"] = ema(c, 12) - ema(c, 26)
    d["macd_sig"] = ema(d["macd"], 9)
    d["macd_hist"] = d["macd"] - d["macd_sig"]
    d["bb_mid"] = c.rolling(20).mean()
    sd = c.rolling(20).std()
    d["bb_up"], d["bb_lo"] = d["bb_mid"] + 2 * sd, d["bb_mid"] - 2 * sd
    d["bb_pb"] = ((c - d["bb_lo"]) / (d["bb_up"] - d["bb_lo"]).replace(0, np.nan)).fillna(0.5)
    d["bb_bw"] = ((d["bb_up"] - d["bb_lo"]) / d["bb_mid"]).fillna(0)
    d["atr"] = calc_atr(d)
    d["adx"], d["pdi"], d["mdi"] = calc_adx(d)
    d["obv"] = (np.sign(c.diff()).fillna(0) * v).cumsum()
    d["obv_ema"] = ema(d["obv"], 20)
    tp = (h + l + c) / 3
    d["vwap"] = (tp * v).rolling(50).sum() / v.rolling(50).sum()
    d["st"], d["st_dir"] = calc_supertrend(d)
    d["tenkan"] = (h.rolling(9).max() + l.rolling(9).min()) / 2
    d["kijun"] = (h.rolling(26).max() + l.rolling(26).min()) / 2
    d["span_a"] = ((d["tenkan"] + d["kijun"]) / 2).shift(26)
    d["span_b"] = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    return d


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CẤU TRÚC GIÁ: HỖ TRỢ/KHÁNG CỰ, FIBONACCI, MẪU NẾN, PHÂN KỲ
# ═══════════════════════════════════════════════════════════════════════════════
def pivots(arr, w=3, kind="low"):
    idx = []
    for i in range(w, len(arr) - w):
        win = arr[i - w: i + w + 1]
        if (kind == "low" and arr[i] == win.min()) or (kind == "high" and arr[i] == win.max()):
            idx.append(i)
    return idx


def find_levels(df, lookback=400, w=5):
    d = df.tail(lookback)
    hi, lo = d["high"].values, d["low"].values
    pts = [hi[i] for i in pivots(hi, w, "high")] + [lo[i] for i in pivots(lo, w, "low")]
    pts.sort()
    tol = d["close"].iloc[-1] * 0.004
    clusters = []
    for p in pts:
        if clusters and abs(p - np.mean(clusters[-1])) <= tol:
            clusters[-1].append(p)
        else:
            clusters.append([p])
    return [(float(np.mean(c)), len(c)) for c in clusters]


def fib_levels(df, bars=120):
    d = df.tail(bars)
    hi, lo = d["high"].max(), d["low"].min()
    up_first = d["high"].values.argmax() > d["low"].values.argmin()  # đáy trước đỉnh = sóng tăng
    rng = hi - lo
    ratios = [0.236, 0.382, 0.5, 0.618, 0.786]
    if up_first:
        lv = {r: hi - rng * r for r in ratios}
    else:
        lv = {r: lo + rng * r for r in ratios}
    return lv, up_first, hi, lo


def candle_patterns(df):
    a, b = df.iloc[-2], df.iloc[-1]
    rng = b["high"] - b["low"]
    body = abs(b["close"] - b["open"])
    up_w = b["high"] - max(b["close"], b["open"])
    lo_w = min(b["close"], b["open"]) - b["low"]
    found = []  # (tên, điểm)
    if rng > 0:
        if body / rng < 0.1:
            found.append(("Doji (do dự)", 0.0))
        if lo_w > 2 * body and up_w < body and body > 0:
            found.append(("Hammer / Pin bar tăng", 0.7))
        if up_w > 2 * body and lo_w < body and body > 0:
            found.append(("Shooting star / Pin bar giảm", -0.7))
    if (a["close"] < a["open"] and b["close"] > b["open"]
            and b["close"] >= a["open"] and b["open"] <= a["close"]):
        found.append(("Bullish Engulfing (nhấn chìm tăng)", 0.9))
    if (a["close"] > a["open"] and b["close"] < b["open"]
            and b["open"] >= a["close"] and b["close"] <= a["open"]):
        found.append(("Bearish Engulfing (nhấn chìm giảm)", -0.9))
    return found


def rsi_divergence(df, bars=80, recent=15):
    d = df.tail(bars).reset_index(drop=True)
    r = d["rsi"].values
    lows, highs = d["low"].values, d["high"].values
    pl, ph = pivots(lows, 3, "low"), pivots(highs, 3, "high")
    n = len(d)
    if len(pl) >= 2 and n - pl[-1] <= recent and lows[pl[-1]] < lows[pl[-2]] and r[pl[-1]] > r[pl[-2]]:
        return 1.0, "Phân kỳ tăng: giá tạo đáy thấp hơn nhưng RSI tạo đáy cao hơn (lực bán suy yếu)"
    if len(ph) >= 2 and n - ph[-1] <= recent and highs[ph[-1]] > highs[ph[-2]] and r[ph[-1]] < r[ph[-2]]:
        return -1.0, "Phân kỳ giảm: giá tạo đỉnh cao hơn nhưng RSI tạo đỉnh thấp hơn (lực mua suy yếu)"
    return 0.0, "Không có phân kỳ RSI gần đây"


def market_structure(df, bars=120):
    d = df.tail(bars).reset_index(drop=True)
    ph, pl = pivots(d["high"].values, 4, "high"), pivots(d["low"].values, 4, "low")
    if len(ph) < 2 or len(pl) < 2:
        return 0.0, "Chưa đủ đỉnh/đáy để xác định cấu trúc"
    hh = d["high"][ph[-1]] > d["high"][ph[-2]]
    hl = d["low"][pl[-1]] > d["low"][pl[-2]]
    if hh and hl:
        return 1.0, "Cấu trúc tăng: đỉnh cao hơn (HH) và đáy cao hơn (HL)"
    if not hh and not hl:
        return -1.0, "Cấu trúc giảm: đỉnh thấp hơn (LH) và đáy thấp hơn (LL)"
    if hh and not hl:
        return 0.0, "Cấu trúc mở rộng (HH nhưng LL): biến động mạnh, chưa rõ xu hướng"
    return 0.0, "Cấu trúc thu hẹp (LH nhưng HL): thị trường đang nén, chờ phá vỡ"


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CHẤM ĐIỂM KỸ THUẬT
# ═══════════════════════════════════════════════════════════════════════════════
def sgn(x):
    return 1.0 if x > 0 else -1.0 if x < 0 else 0.0


def fmt(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x:,.2f}" if abs(x) >= 100 else f"{x:,.4f}" if abs(x) >= 1 else f"{x:,.6f}"


def analyze_tf(d: pd.DataFrame) -> dict:
    L, P = d.iloc[-1], d.iloc[-2]
    c, a = float(L["close"]), float(L["atr"])
    levels = find_levels(d)
    sup = max([lv for lv, _ in levels if lv < c], default=None)
    res = min([lv for lv, _ in levels if lv > c], default=None)
    F = []

    def add(name, s, w, text):
        F.append({"name": name, "s": float(np.clip(s, -1, 1)), "w": w, "text": text})

    # 1. Xu hướng EMA
    pos = int(c > L["ema20"]) + int(L["ema20"] > L["ema50"]) + int(L["ema50"] > L["ema200"])
    txt = {3: "Giá > EMA20 > EMA50 > EMA200: xu hướng tăng đồng thuận",
           0: "Giá < EMA20 < EMA50 < EMA200: xu hướng giảm đồng thuận"}.get(
        pos, f"EMA chưa đồng thuận ({pos}/3 điều kiện tăng): xu hướng lẫn lộn/đang chuyển pha")
    add("Xu hướng EMA 20/50/200", (pos - 1.5) / 1.5, 2.0, txt)

    # 2. ADX / DI
    di = sgn(L["pdi"] - L["mdi"])
    s_adx = di * min(L["adx"] / 40, 1)
    adx_txt = (f"ADX {L['adx']:.0f}: " + ("xu hướng mạnh" if L["adx"] >= 25 else
               "xu hướng yếu, thị trường dễ đi ngang (tín hiệu kém tin cậy)" if L["adx"] < 18 else "xu hướng vừa phải")
               + f", +DI {'>' if di > 0 else '<'} -DI")
    add("ADX / DI (độ mạnh xu hướng)", s_adx, 1.0, adx_txt)

    # 3. Supertrend
    add("Supertrend (10,3)", float(L["st_dir"]), 1.5,
        "Supertrend đang " + ("TĂNG (giá trên đường hỗ trợ động)" if L["st_dir"] > 0 else "GIẢM (giá dưới đường kháng cự động)"))

    # 4. Ichimoku
    if not np.isnan(L["span_a"]) and not np.isnan(L["span_b"]):
        top, bot = max(L["span_a"], L["span_b"]), min(L["span_a"], L["span_b"])
        s_cloud = 1.0 if c > top else -1.0 if c < bot else 0.0
        s_tk = 0.5 * sgn(L["tenkan"] - L["kijun"])
        add("Ichimoku", s_cloud * 0.75 + s_tk * 0.5, 1.2,
            ("Giá trên mây Kumo" if s_cloud > 0 else "Giá dưới mây Kumo" if s_cloud < 0 else "Giá trong mây (trung tính)")
            + (", Tenkan > Kijun" if s_tk > 0 else ", Tenkan < Kijun"))
    else:
        add("Ichimoku", 0, 1.2, "Chưa đủ dữ liệu")

    # 5. MACD
    h_now, h_prev = L["macd_hist"], P["macd_hist"]
    s_macd = 0.5 * sgn(h_now) + 0.5 * sgn(h_now - h_prev)
    cross = ""
    if P["macd"] <= P["macd_sig"] and L["macd"] > L["macd_sig"]:
        s_macd, cross = 1.0, " – vừa cắt LÊN đường tín hiệu"
    elif P["macd"] >= P["macd_sig"] and L["macd"] < L["macd_sig"]:
        s_macd, cross = -1.0, " – vừa cắt XUỐNG đường tín hiệu"
    add("MACD (12,26,9)", s_macd, 1.2,
        f"Histogram {'dương' if h_now > 0 else 'âm'} và {'mở rộng' if abs(h_now) > abs(h_prev) else 'thu hẹp'}{cross}")

    # 6. RSI
    r = float(L["rsi"])
    if r > 75:
        s_r, t = -0.7, f"RSI {r:.0f}: quá mua mạnh, dễ có nhịp điều chỉnh"
    elif r > 70:
        s_r, t = -0.4, f"RSI {r:.0f}: vùng quá mua"
    elif r < 25:
        s_r, t = 0.7, f"RSI {r:.0f}: quá bán mạnh, dễ có nhịp hồi"
    elif r < 30:
        s_r, t = 0.4, f"RSI {r:.0f}: vùng quá bán"
    else:
        s_r, t = (r - 50) / 50 * 0.8, f"RSI {r:.0f}: động lượng {'tăng' if r > 50 else 'giảm'}, chưa cực đoan"
    add("RSI (14)", s_r, 1.0, t)

    # 7. StochRSI
    k, dd = float(L["stoch_k"]), float(L["stoch_d"])
    if k < 0.2 and k > dd:
        s_sk, t = 0.8, "StochRSI cắt lên từ vùng quá bán"
    elif k > 0.8 and k < dd:
        s_sk, t = -0.8, "StochRSI cắt xuống từ vùng quá mua"
    else:
        s_sk, t = (k - 0.5) * 0.4, f"StochRSI %K={k:.2f}, trung tính"
    add("Stochastic RSI", s_sk, 0.7, t)

    # 8. Bollinger
    pb = float(L["bb_pb"])
    squeeze = L["bb_bw"] <= d["bb_bw"].tail(100).quantile(0.15)
    s_bb = -0.4 if pb > 0.95 else 0.4 if pb < 0.05 else 0.0
    add("Bollinger Bands", s_bb, 0.7,
        f"%B={pb:.2f}" + (" – giá chạm dải trên" if pb > 0.95 else " – giá chạm dải dưới" if pb < 0.05 else " – giá trong dải")
        + (" | Băng hẹp (squeeze): sắp có biến động mạnh" if squeeze else ""))

    # 9. Volume / OBV
    vol_avg = d["volume"].tail(20).mean()
    spike = L["volume"] > 1.8 * vol_avg
    s_vol = 0.5 * (1 if L["obv"] > L["obv_ema"] else -1) + (0.5 * (1 if c > L["open"] else -1) if spike else 0)
    add("Volume / OBV", s_vol, 0.8,
        f"OBV {'trên' if L['obv'] > L['obv_ema'] else 'dưới'} trung bình (dòng tiền {'vào' if L['obv'] > L['obv_ema'] else 'ra'})"
        + (f" | Volume đột biến x{L['volume'] / vol_avg:.1f} theo hướng {'tăng' if c > L['open'] else 'giảm'}" if spike else ""))

    # 10. Mẫu nến
    pats = candle_patterns(d)
    add("Mẫu nến", sum(s for _, s in pats), 0.8,
        ", ".join(n for n, _ in pats) if pats else "Không có mẫu nến đáng chú ý ở nến hiện tại")

    # 11. Phân kỳ
    s_div, t_div = rsi_divergence(d)
    add("Phân kỳ RSI", s_div, 1.2, t_div)

    # 12. Cấu trúc thị trường
    s_ms, t_ms = market_structure(d)
    add("Cấu trúc thị trường (HH/HL)", s_ms, 1.0, t_ms)

    # 13. Hỗ trợ / kháng cự
    near_s = sup is not None and (c - sup) <= 1.0 * a
    near_r = res is not None and (res - c) <= 1.0 * a
    s_sr = 0.6 * near_s - 0.6 * near_r
    add("Hỗ trợ / Kháng cự", s_sr, 0.8,
        f"Hỗ trợ gần: {fmt(sup)} | Kháng cự gần: {fmt(res)}"
        + (" → giá sát hỗ trợ (kỳ vọng bật)" if near_s and not near_r else
           " → giá sát kháng cự (dễ bị cản)" if near_r and not near_s else ""))

    # 14. VWAP
    if not np.isnan(L["vwap"]):
        add("VWAP (50 nến)", 0.5 * sgn(c - L["vwap"]), 0.5,
            f"Giá {'trên' if c > L['vwap'] else 'dưới'} VWAP {fmt(L['vwap'])}")

    tw = sum(f["w"] for f in F)
    score = sum(f["s"] * f["w"] for f in F) / tw * 100
    fib, up_first, fhi, flo = fib_levels(d)
    return {"score": float(score), "factors": F, "levels": levels, "sup": sup, "res": res,
            "atr": a, "close": c, "rsi": r, "adx": float(L["adx"]), "fib": fib,
            "fib_up": up_first, "swing_hi": fhi, "swing_lo": flo}


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MACHINE LEARNING
# ═══════════════════════════════════════════════════════════════════════════════
def build_features(d: pd.DataFrame) -> pd.DataFrame:
    c = d["close"]
    X = pd.DataFrame(index=d.index)
    for n in (1, 3, 5, 10, 20):
        X[f"ret{n}"] = c.pct_change(n)
    X["rsi"], X["stoch"] = d["rsi"], d["stochrsi"]
    X["macd_h"] = d["macd_hist"] / c
    for n in (20, 50, 200):
        X[f"d_ema{n}"] = c / d[f"ema{n}"] - 1
    X["pb"], X["bw"] = d["bb_pb"], d["bb_bw"]
    X["atr_p"] = d["atr"] / c
    X["adx"], X["di"] = d["adx"], d["pdi"] - d["mdi"]
    vavg = d["volume"].rolling(20).mean()
    X["vol_r"] = d["volume"] / vavg
    X["obv_s"] = (d["obv"] - d["obv"].shift(10)) / (vavg * 10)
    X["st_dir"] = d["st_dir"]
    return X.replace([np.inf, -np.inf], np.nan)


def _new_model():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150,
                                          l2_regularization=1.0, random_state=42)


def train_predict(d: pd.DataFrame, horizon: int):
    """Huấn luyện dự đoán xác suất giá tăng sau `horizon` nến. Kiểm định theo thời gian (có gap)."""
    X = build_features(d)
    fut = d["close"].shift(-horizon)
    y = (fut > d["close"]).astype(float).where(fut.notna())
    data = X.join(y.rename("y")).dropna()
    if len(data) < 300:
        return None
    n = len(data)
    split = int(n * 0.8)
    tr, te = data.iloc[: split - horizon], data.iloc[split:]
    m = _new_model().fit(tr.drop(columns="y"), tr["y"])
    p_te = m.predict_proba(te.drop(columns="y"))[:, 1]
    yt = te["y"].values
    acc = float(((p_te > 0.5) == (yt == 1)).mean())
    base = float(max(yt.mean(), 1 - yt.mean()))
    hc = (p_te > 0.58) | (p_te < 0.42)
    hc_acc = float(((p_te[hc] > 0.5) == (yt[hc] == 1)).mean()) if hc.sum() >= 10 else None
    # huấn luyện lại trên toàn bộ dữ liệu có nhãn rồi dự đoán nến hiện tại
    m2 = _new_model().fit(data.drop(columns="y"), data["y"])
    x_last = X.iloc[[-1]]
    if x_last.isna().any(axis=None):
        return None
    prob = float(m2.predict_proba(x_last)[:, 1][0])
    return {"prob_up": prob, "acc": acc, "base": base, "hc_acc": hc_acc, "hc_n": int(hc.sum()),
            "n_train": len(tr), "n_test": len(te), "horizon": horizon}


# ═══════════════════════════════════════════════════════════════════════════════
# 6. TỔNG HỢP TÍN HIỆU, KẾ HOẠCH LỆNH
# ═══════════════════════════════════════════════════════════════════════════════
def blend(tech: float, ml):
    if not ml:
        return tech, 0.0
    reliable = ml["acc"] >= 0.53 and ml["acc"] > ml["base"]
    w = 0.25 if reliable else 0.10
    ml_score = float(np.clip((ml["prob_up"] - 0.5) * 3 * 100, -100, 100))
    return (1 - w) * tech + w * ml_score, w


def signal_of(score: float):
    if score >= 45:
        return "LONG", "🟢 MUA MẠNH (LONG)"
    if score >= 20:
        return "LONG", "🟢 MUA (LONG)"
    if score <= -45:
        return "SHORT", "🔴 BÁN MẠNH (SHORT)"
    if score <= -20:
        return "SHORT", "🔴 BÁN (SHORT)"
    return "WAIT", "⚪ CHỜ (không vào lệnh)"


def trade_plan(direction, entry, atr, sup, res):
    if direction == "WAIT":
        return None
    if direction == "LONG":
        cand = sup - 0.25 * atr if sup else None
        if cand is not None and 1.0 * atr <= entry - cand <= 3.0 * atr:
            sl, basis = cand, "ngay dưới hỗ trợ gần nhất"
        else:
            sl, basis = entry - 1.8 * atr, "1.8 × ATR"
        risk = entry - sl
        tps = [entry + risk * k for k in (1.5, 2.5, 4.0)]
        blocker = res if res and entry < res < tps[0] else None
    else:
        cand = res + 0.25 * atr if res else None
        if cand is not None and 1.0 * atr <= cand - entry <= 3.0 * atr:
            sl, basis = cand, "ngay trên kháng cự gần nhất"
        else:
            sl, basis = entry + 1.8 * atr, "1.8 × ATR"
        risk = sl - entry
        tps = [entry - risk * k for k in (1.5, 2.5, 4.0)]
        blocker = sup if sup and tps[0] < sup < entry else None
    return {"dir": direction, "entry": entry, "sl": sl, "tps": tps, "risk": risk,
            "basis": basis, "blocker": blocker, "rr": (1.5, 2.5, 4.0)}


def sizing(plan, capital, risk_pct, lev):
    risk_amt = capital * risk_pct / 100
    qty = risk_amt / plan["risk"]
    notional = qty * plan["entry"]
    margin = notional / lev
    mmr = 0.005
    liq = plan["entry"] * (1 - 1 / lev + mmr) if plan["dir"] == "LONG" else plan["entry"] * (1 + 1 / lev - mmr)
    sl_beyond_liq = (plan["sl"] <= liq) if plan["dir"] == "LONG" else (plan["sl"] >= liq)
    return {"risk_amt": risk_amt, "qty": qty, "notional": notional, "margin": margin,
            "liq": liq, "margin_too_big": margin > capital, "sl_beyond_liq": sl_beyond_liq}


def explain(an, direction):
    fs = sorted(an["factors"], key=lambda f: abs(f["s"] * f["w"]), reverse=True)
    if direction == "WAIT":
        top = fs[:3]
        return ("**Vì sao nên chờ:** các tín hiệu đang mâu thuẫn hoặc yếu, xác suất thắng chưa đủ ưu thế.\n\n"
                + "\n".join(f"- {f['name']}: {f['text']}" for f in top))
    sign = 1 if direction == "LONG" else -1
    pro = [f for f in fs if f["s"] * sign > 0][:4]
    con = [f for f in fs if f["s"] * sign < 0][:2]
    md = "**Lý do ủng hộ lệnh:**\n" + "\n".join(f"- ✅ **{f['name']}**: {f['text']}" for f in pro)
    if con:
        md += "\n\n**Tín hiệu ngược chiều (rủi ro):**\n" + "\n".join(f"- ⚠️ **{f['name']}**: {f['text']}" for f in con)
    return md


def context_adjust(deriv, fng_val, news):
    adj, notes = 0.0, []
    fr = deriv.get("funding")
    if fr is not None:
        if fr > 0.0008:
            adj -= 8; notes.append(f"Funding rate rất cao ({fr * 100:.3f}%): phe Long quá đông, rủi ro bị quét xuống (-8)")
        elif fr > 0.0003:
            adj -= 3; notes.append(f"Funding rate dương cao ({fr * 100:.3f}%): nghiêng về Long đông (-3)")
        elif fr < -0.0005:
            adj += 8; notes.append(f"Funding rate âm sâu ({fr * 100:.3f}%): phe Short quá đông, dễ short squeeze (+8)")
        elif fr < -0.0001:
            adj += 3; notes.append(f"Funding rate âm ({fr * 100:.3f}%): nghiêng về Short đông (+3)")
        else:
            notes.append(f"Funding rate trung tính ({fr * 100:.3f}%)")
    ls = deriv.get("ls")
    if ls is not None:
        if ls > 2.0:
            adj -= 5; notes.append(f"Tỷ lệ Long/Short {ls:.2f}: đám đông quá nghiêng Long (-5)")
        elif ls < 0.7:
            adj += 5; notes.append(f"Tỷ lệ Long/Short {ls:.2f}: đám đông quá nghiêng Short (+5)")
        else:
            notes.append(f"Tỷ lệ Long/Short {ls:.2f}: cân bằng")
    if fng_val is not None:
        if fng_val <= 20:
            adj += 5; notes.append(f"Fear & Greed {fng_val}: sợ hãi cực độ, thường gần vùng đáy (+5)")
        elif fng_val >= 80:
            adj -= 5; notes.append(f"Fear & Greed {fng_val}: tham lam cực độ, thường gần vùng đỉnh (-5)")
        else:
            notes.append(f"Fear & Greed {fng_val}: chưa cực đoan")
    if news:
        sc = [news_sentiment(n["title"] + " " + n["body"]) for n in news[:10]]
        avg = float(np.mean(sc))
        a = float(np.clip(avg * 10, -6, 6))
        adj += a
        notes.append(f"Cảm xúc tin tức {avg:+.2f} ({sum(s > 0 for s in sc)} tích cực / {sum(s < 0 for s in sc)} tiêu cực) ({a:+.0f})")
    return adj, notes


@st.cache_data(ttl=90, show_spinner=False)
def run_tf(symbol: str, tf: str, horizon: int):
    raw = fetch_klines(symbol, tf, 1000)
    d = add_indicators(raw)
    an = analyze_tf(d)
    try:
        ml = train_predict(d, horizon)
    except Exception:
        ml = None
    return d, an, ml


# ═══════════════════════════════════════════════════════════════════════════════
# 7. BIỂU ĐỒ
# ═══════════════════════════════════════════════════════════════════════════════
def make_chart(d, plan, levels, bars=180):
    x = d.tail(bars)
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        row_heights=[0.55, 0.14, 0.15, 0.16])
    fig.add_trace(go.Candlestick(x=x.index, open=x["open"], high=x["high"], low=x["low"],
                                 close=x["close"], name="Giá", showlegend=False), row=1, col=1)
    for col, color in (("ema20", "#f5a623"), ("ema50", "#4a90e2"), ("ema200", "#b57edc")):
        fig.add_trace(go.Scatter(x=x.index, y=x[col], name=col.upper(), line=dict(width=1.2, color=color)), row=1, col=1)
    for col in ("bb_up", "bb_lo"):
        fig.add_trace(go.Scatter(x=x.index, y=x[col], showlegend=False,
                                 line=dict(width=0.8, dash="dot", color="gray")), row=1, col=1)
    fig.add_trace(go.Scatter(x=x.index, y=x["st"].where(x["st_dir"] > 0), name="Supertrend ↑",
                             line=dict(width=1.5, color="#26a69a")), row=1, col=1)
    fig.add_trace(go.Scatter(x=x.index, y=x["st"].where(x["st_dir"] < 0), name="Supertrend ↓",
                             line=dict(width=1.5, color="#ef5350")), row=1, col=1)
    price = x["close"].iloc[-1]
    for lv, touches in levels:
        if abs(lv / price - 1) < 0.08 and touches >= 2:
            fig.add_hline(y=lv, row=1, col=1, line=dict(width=0.8, dash="dash", color="#888"),
                          annotation_text=f"{'KC' if lv > price else 'HT'} {fmt(lv)}",
                          annotation_position="left", annotation_font_size=9)
    if plan:
        fig.add_hline(y=plan["entry"], row=1, col=1, line=dict(color="#ffffff", dash="dot"),
                      annotation_text="Entry", annotation_position="right")
        fig.add_hline(y=plan["sl"], row=1, col=1, line=dict(color="#ef5350", width=1.5),
                      annotation_text="SL", annotation_position="right")
        for i, tp in enumerate(plan["tps"], 1):
            fig.add_hline(y=tp, row=1, col=1, line=dict(color="#26a69a", width=1.2),
                          annotation_text=f"TP{i}", annotation_position="right")
    colors = np.where(x["close"] >= x["open"], "#26a69a", "#ef5350")
    fig.add_trace(go.Bar(x=x.index, y=x["volume"], marker_color=colors, name="Volume", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=x.index, y=x["rsi"], name="RSI", line=dict(color="#f5a623", width=1.2)), row=3, col=1)
    for y0 in (30, 70):
        fig.add_hline(y=y0, row=3, col=1, line=dict(width=0.6, dash="dot", color="gray"))
    hist_col = np.where(x["macd_hist"] >= 0, "#26a69a", "#ef5350")
    fig.add_trace(go.Bar(x=x.index, y=x["macd_hist"], marker_color=hist_col, name="MACD hist", showlegend=False), row=4, col=1)
    fig.add_trace(go.Scatter(x=x.index, y=x["macd"], name="MACD", line=dict(width=1, color="#4a90e2")), row=4, col=1)
    fig.add_trace(go.Scatter(x=x.index, y=x["macd_sig"], name="Signal", line=dict(width=1, color="#f5a623")), row=4, col=1)
    fig.update_xaxes(rangeslider_visible=False)
    fig.update_layout(height=820, template="plotly_dark", margin=dict(l=10, r=60, t=20, b=10),
                      legend=dict(orientation="h", y=1.03))
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# 8. GIAO DIỆN STREAMLIT
# ═══════════════════════════════════════════════════════════════════════════════
def ago(ts):
    mins = int((dt.datetime.now().timestamp() - ts) / 60)
    if mins < 60:
        return f"{max(mins, 0)} phút trước"
    if mins < 1440:
        return f"{mins // 60} giờ trước"
    return f"{mins // 1440} ngày trước"


def main():
    st.set_page_config(page_title="Crypto Futures AI", page_icon="📈", layout="wide")
    st.title("📈 Crypto Futures AI – Dự đoán & Tín hiệu Mua/Bán")
    st.caption("Phân tích kỹ thuật đa khung giờ + Machine Learning + Tin tức. "
               "⚠️ Công cụ hỗ trợ ra quyết định, KHÔNG đảm bảo lợi nhuận và không phải lời khuyên tài chính.")

    with st.sidebar:
        st.header("⚙️ Cấu hình")
        pick = st.selectbox("Cặp futures", POPULAR + ["Khác…"])
        symbol = (st.text_input("Nhập mã (VD: ARBUSDT)", "ARBUSDT").upper().strip()
                  if pick == "Khác…" else pick)
        tfs = st.multiselect("Khung giờ", list(TF_LABEL), default=["15m", "1h", "4h", "1d"],
                             format_func=lambda t: TF_LABEL[t])
        horizon = st.slider("Mô hình ML dự đoán sau N nến", 2, 24, 6)
        st.subheader("💰 Quản lý vốn")
        capital = st.number_input("Vốn (USDT)", 10.0, 10_000_000.0, 1000.0, step=100.0)
        risk_pct = st.slider("Rủi ro mỗi lệnh (% vốn)", 0.25, 5.0, 1.0, 0.25)
        lev = st.slider("Đòn bẩy", 1, 50, 5)
        if st.button("🔄 Làm mới dữ liệu", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.caption("Dữ liệu: Binance Futures / Bybit (công khai), CryptoCompare, Cointelegraph, CoinDesk, alternative.me")

    if not tfs:
        st.warning("Hãy chọn ít nhất 1 khung giờ.")
        return
    base = symbol.replace("USDT", "").replace("USDC", "")

    # ---- tải & phân tích
    results = {}
    with st.spinner(f"Đang tải dữ liệu & huấn luyện mô hình cho {symbol}…"):
        for tf in tfs:
            try:
                results[tf] = run_tf(symbol, tf, horizon)
            except Exception as e:
                st.error(f"Không tải được {symbol} khung {tf}: {e}")
        if not results:
            st.stop()
        deriv = fetch_derivs(symbol)
        fng_val, fng_txt = fetch_fng()
        news = fetch_news(base)
        try:
            h1 = fetch_klines(symbol, "1h", 100)
            chg24 = (h1["close"].iloc[-1] / h1["close"].iloc[-25] - 1) * 100
            last_price = float(h1["close"].iloc[-1])
        except Exception:
            first = next(iter(results.values()))[0]
            last_price, chg24 = float(first["close"].iloc[-1]), None

    # ---- điểm từng khung & tổng hợp
    rows, final = [], {}
    for tf, (d, an, ml) in results.items():
        sc, mlw = blend(an["score"], ml)
        direction, label = signal_of(sc)
        plan = trade_plan(direction, an["close"], an["atr"], an["sup"], an["res"])
        final[tf] = {"score": sc, "dir": direction, "label": label, "plan": plan, "mlw": mlw}
    tw = sum(TF_WEIGHT[t] for t in final)
    mtf_score = sum(final[t]["score"] * TF_WEIGHT[t] for t in final) / tw
    adj, ctx_notes = context_adjust(deriv, fng_val, news)
    overall = float(np.clip(mtf_score + adj, -100, 100))
    o_dir, o_label = signal_of(overall)
    agree = np.mean([final[t]["dir"] == o_dir for t in final]) if o_dir != "WAIT" else None

    # ---- thanh số liệu
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(symbol, fmt(last_price), f"{chg24:+.2f}% / 24h" if chg24 is not None else None)
    c2.metric("Funding rate", f"{deriv['funding'] * 100:.4f}%" if deriv["funding"] is not None else "—")
    c3.metric("Open Interest", f"{deriv['oi']:,.0f}" if deriv["oi"] else "—")
    c4.metric("Long/Short", f"{deriv['ls']:.2f}" if deriv["ls"] else "—")
    c5.metric("Fear & Greed", f"{fng_val} – {fng_txt}" if fng_val is not None else "—")
    st.caption(f"Cập nhật lúc {dt.datetime.now(VN_TZ):%d/%m/%Y %H:%M:%S} (GMT+7)")

    tab_over, tab_detail, tab_news, tab_ml, tab_help = st.tabs(
        ["🎯 Tổng quan & Khuyến nghị", "📊 Chi tiết từng khung giờ", "📰 Tin tức mới nhất", "🤖 Mô hình ML", "ℹ️ Cách đọc kết quả"])

    # ---------------- TỔNG QUAN
    with tab_over:
        box = st.success if o_dir == "LONG" else st.error if o_dir == "SHORT" else st.info
        box(f"## {o_label}\nĐiểm tổng hợp: **{overall:+.1f}** / 100"
            + (f"  •  {agree * 100:.0f}% khung giờ đồng thuận" if agree is not None else ""))
        if agree is not None and agree < 0.5:
            st.warning("Các khung giờ lớn/nhỏ đang bất đồng. Nên giảm khối lượng hoặc chờ thêm xác nhận.")

        table = []
        for tf in final:
            f, (d, an, ml) = final[tf], results[tf]
            p = f["plan"]
            table.append({
                "Khung": TF_LABEL[tf], "Tín hiệu": f["label"], "Điểm": round(f["score"], 1),
                "Kỹ thuật": round(an["score"], 1),
                f"ML ↑ (+{horizon} nến)": f"{ml['prob_up'] * 100:.0f}%" if ml else "—",
                "Entry": fmt(p["entry"]) if p else "—", "Cắt lỗ": fmt(p["sl"]) if p else "—",
                "TP1": fmt(p["tps"][0]) if p else "—", "TP2": fmt(p["tps"][1]) if p else "—",
                "TP3": fmt(p["tps"][2]) if p else "—",
            })
        st.dataframe(pd.DataFrame(table), hide_index=True, use_container_width=True)

        st.subheader("🧭 Bối cảnh thị trường (điều chỉnh điểm)")
        st.write(f"Điểm đa khung: **{mtf_score:+.1f}** → sau điều chỉnh bối cảnh **{adj:+.1f}** = **{overall:+.1f}**")
        for n in ctx_notes:
            st.markdown(f"- {n}")

        if o_dir != "WAIT":
            best = max((t for t in final if final[t]["dir"] == o_dir and final[t]["plan"]),
                       key=lambda t: TF_WEIGHT[t], default=None)
            if best:
                st.subheader(f"🧾 Kế hoạch lệnh gợi ý (khung {TF_LABEL[best]})")
                st.markdown(explain(results[best][1], o_dir))
        else:
            st.markdown("**Khuyến nghị:** chưa có ưu thế rõ ràng. Đứng ngoài và chờ giá phá vỡ vùng hỗ trợ/kháng cự "
                        "hoặc các khung giờ đồng thuận hơn. Bỏ qua cơ hội còn tốt hơn vào lệnh xác suất thấp.")

    # ---------------- CHI TIẾT
    with tab_detail:
        subtabs = st.tabs([TF_LABEL[t] for t in final])
        for tab, tf in zip(subtabs, final):
            d, an, ml = results[tf]
            f = final[tf]
            plan = f["plan"]
            with tab:
                box = st.success if f["dir"] == "LONG" else st.error if f["dir"] == "SHORT" else st.info
                box(f"### {f['label']}  •  điểm {f['score']:+.1f}  (kỹ thuật {an['score']:+.1f}"
                    + (f", ML {ml['prob_up'] * 100:.0f}% tăng, trọng số {f['mlw'] * 100:.0f}%" if ml else "") + ")")
                st.markdown(explain(an, f["dir"]))

                if plan:
                    st.subheader("🎯 Kế hoạch lệnh")
                    pc1, pc2, pc3, pc4, pc5 = st.columns(5)
                    pc1.metric("Entry", fmt(plan["entry"]))
                    pc2.metric("Cắt lỗ (SL)", fmt(plan["sl"]), f"-{plan['risk'] / plan['entry'] * 100:.2f}%", delta_color="inverse")
                    sgn_ = 1 if plan["dir"] == "LONG" else -1
                    for col, i in zip((pc3, pc4, pc5), range(3)):
                        tp = plan["tps"][i]
                        col.metric(f"TP{i + 1} (R:R 1:{plan['rr'][i]})", fmt(tp),
                                   f"{sgn_ * (tp / plan['entry'] - 1) * 100:+.2f}%")
                    st.caption(f"SL đặt {plan['basis']} (ATR={fmt(an['atr'])}). "
                               "Gợi ý: chốt 50% tại TP1 rồi dời SL về điểm vào, 30% tại TP2, 20% còn lại tại TP3.")
                    if plan["blocker"]:
                        st.warning(f"Có vùng cản gần {fmt(plan['blocker'])} nằm trước TP1 – cân nhắc chốt sớm hơn.")
                    sz = sizing(plan, capital, risk_pct, lev)
                    st.markdown(
                        f"**Khối lượng theo quản lý vốn:** rủi ro **{sz['risk_amt']:,.2f} USDT** → "
                        f"khối lượng **{sz['qty']:,.4f} {base}** (giá trị vị thế {sz['notional']:,.2f} USDT, "
                        f"ký quỹ ~{sz['margin']:,.2f} USDT ở x{lev}, giá thanh lý ước tính ~{fmt(sz['liq'])}).")
                    fr_ = [0.5, 0.3, 0.2]
                    pnl = [sz["qty"] * abs(tp - plan["entry"]) * w for tp, w in zip(plan["tps"], fr_)]
                    st.caption(f"Lãi dự kiến nếu chốt đủ 50/30/20: ~{sum(pnl):,.2f} USDT • lỗ tối đa nếu dính SL: ~{sz['risk_amt']:,.2f} USDT (chưa tính phí & trượt giá).")
                    if sz["margin_too_big"]:
                        st.error("Ký quỹ cần thiết lớn hơn vốn – hãy tăng đòn bẩy hoặc giảm % rủi ro.")
                    if sz["sl_beyond_liq"]:
                        st.error("SL nằm xa hơn giá thanh lý – sẽ bị thanh lý trước khi chạm SL! Hãy giảm đòn bẩy.")

                st.subheader("🔬 Bảng điểm từng chỉ báo")
                ftab = pd.DataFrame([{"Chỉ báo": x["name"], "Điểm (-1..+1)": round(x["s"], 2), "Trọng số": x["w"],
                                      "Hướng": "🟢" if x["s"] > 0.15 else "🔴" if x["s"] < -0.15 else "⚪",
                                      "Giải thích": x["text"]} for x in an["factors"]])
                st.dataframe(ftab, hide_index=True, use_container_width=True)

                cc1, cc2 = st.columns(2)
                with cc1:
                    st.markdown("**Fibonacci** " + ("(sóng tăng, hồi về)" if an["fib_up"] else "(sóng giảm, hồi lên)")
                                + f" – đỉnh {fmt(an['swing_hi'])} / đáy {fmt(an['swing_lo'])}")
                    st.dataframe(pd.DataFrame([{"Mức": f"{r * 100:.1f}%", "Giá": fmt(v)} for r, v in an["fib"].items()]),
                                 hide_index=True, use_container_width=True)
                with cc2:
                    near = sorted([(lv, t) for lv, t in an["levels"] if abs(lv / an["close"] - 1) < 0.06],
                                  key=lambda z: -z[0])
                    st.markdown("**Hỗ trợ / Kháng cự gần giá** (số lần chạm càng nhiều càng mạnh)")
                    st.dataframe(pd.DataFrame([{"Loại": "Kháng cự" if lv > an["close"] else "Hỗ trợ",
                                                "Giá": fmt(lv), "Số lần chạm": t} for lv, t in near]),
                                 hide_index=True, use_container_width=True)
                st.plotly_chart(make_chart(d, plan, an["levels"]), use_container_width=True)

    # ---------------- TIN TỨC
    with tab_news:
        st.subheader(f"📰 Tin mới nhất liên quan {base}")
        if not news:
            st.info("Không lấy được tin tức (có thể do mạng/chặn API). Bấm 'Làm mới dữ liệu' để thử lại.")
        for n in news:
            s = news_sentiment(n["title"] + " " + n["body"])
            icon = "🟢" if s > 0 else "🔴" if s < 0 else "⚪"
            when = dt.datetime.fromtimestamp(n["ts"], tz=VN_TZ)
            tag = "" if n.get("specific") else " • 🌐 tin thị trường chung"
            st.markdown(f"{icon} **[{n['title']}]({n['url']})**  \n"
                        f"🕒 {when:%d/%m/%Y %H:%M} ({ago(n['ts'])}) • {n['source']}{tag}")
        st.caption("Điểm cảm xúc 🟢/🔴/⚪ chấm theo từ khóa trong tiêu đề, chỉ mang tính tham khảo. "
                   "Hãy đọc bài gốc trước khi quyết định.")

    # ---------------- ML
    with tab_ml:
        st.subheader("🤖 Mô hình Gradient Boosting (huấn luyện trực tiếp trên dữ liệu coin)")
        st.markdown(
            f"Mô hình học từ ~1000 nến gần nhất của từng khung, dự đoán **giá đóng cửa sau {horizon} nến có cao hơn hiện tại không**. "
            "Kiểm định bằng dữ liệu tương lai chưa từng thấy (80% đầu train, 20% cuối test, có khoảng đệm để tránh rò rỉ dữ liệu).")
        mlrows = []
        for tf, (d, an, ml) in results.items():
            if ml:
                reliable = ml["acc"] >= 0.53 and ml["acc"] > ml["base"]
                mlrows.append({"Khung": TF_LABEL[tf], "Xác suất tăng": f"{ml['prob_up'] * 100:.1f}%",
                               "Độ chính xác (test)": f"{ml['acc'] * 100:.1f}%",
                               "Mốc so sánh (luôn đoán phe đông)": f"{ml['base'] * 100:.1f}%",
                               "Chính xác khi tự tin cao": (f"{ml['hc_acc'] * 100:.1f}% ({ml['hc_n']} lệnh)"
                                                            if ml["hc_acc"] is not None else "—"),
                               "Mẫu train/test": f"{ml['n_train']}/{ml['n_test']}",
                               "Đánh giá": "✅ có ý nghĩa → trọng số 25%" if reliable else "⚠️ yếu → trọng số 10%"})
            else:
                mlrows.append({"Khung": TF_LABEL[tf], "Xác suất tăng": "—", "Đánh giá": "Không đủ dữ liệu"})
        st.dataframe(pd.DataFrame(mlrows), hide_index=True, use_container_width=True)
        st.info("Thị trường crypto rất nhiễu: độ chính xác ML ~52–60% đã là tốt. Nếu 'Độ chính xác (test)' không vượt mốc so sánh, "
                "tool tự giảm trọng số ML. Đừng tin một con số xác suất đơn lẻ.")

    # ---------------- HƯỚNG DẪN
    with tab_help:
        st.markdown("""
**Cách tool ra quyết định**
1. Mỗi khung giờ có **14 nhóm chỉ báo** (xu hướng, động lượng, dòng tiền, mẫu nến, phân kỳ, cấu trúc, hỗ trợ/kháng cự…) được chấm điểm -1..+1, nhân trọng số rồi quy về thang ±100.
2. Điểm kỹ thuật được trộn với **xác suất ML** (trọng số tự động 10–25% tùy độ tin cậy kiểm định).
3. Các khung giờ được gộp theo trọng số (khung lớn quan trọng hơn), rồi cộng/trừ **bối cảnh**: funding, tỷ lệ Long/Short, Fear & Greed, cảm xúc tin tức.
4. Ngưỡng: ≥ +20 MUA, ≥ +45 MUA MẠNH, ≤ -20 BÁN, ≤ -45 BÁN MẠNH, còn lại CHỜ.

**Kế hoạch lệnh**
- **Cắt lỗ**: ngay ngoài hỗ trợ/kháng cự gần nhất (nếu cách 1–3 ATR), nếu không thì 1.8×ATR.
- **Chốt lời**: TP1/TP2/TP3 tương ứng R:R 1.5 / 2.5 / 4. Khối lượng tính để thua tối đa đúng **% rủi ro** bạn chọn.

**Lưu ý quan trọng**
- Không có hệ thống nào đúng 100%. Hãy test bằng tài khoản demo / vốn nhỏ trước.
- Khung nhỏ (5m, 15m) nhiễu hơn, nên ưu tiên lệnh thuận hướng khung 4h/1d.
- Tránh vào lệnh ngay trước tin vĩ mô lớn (CPI, FOMC) hoặc khi funding cực đoan.
- Nến cuối đang chạy chưa đóng; tín hiệu có thể đổi khi nến đóng.
- Không dùng đòn bẩy cao khi chưa kiểm soát được rủi ro. Chỉ giao dịch số tiền bạn có thể chấp nhận mất.
""")


if __name__ == "__main__":
    main()
