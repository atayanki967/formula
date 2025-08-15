# F1 Strategy Simulator — single_file_streamlit.py
# -------------------------------------------------
# ✅ Tek dosyalık, sade ama etkileyici bir yarış stratejisi simülatörü
# ✅ Streamlit ile şık arayüz + Matplotlib grafik
# ✅ Gumroad’da ürün olarak paketlemeye uygun (tek dosya)
#
# Çalıştırma:
#   1) Python 3.10+
#   2) pip install streamlit pandas numpy matplotlib
#   3) streamlit run single_file_streamlit.py
# -------------------------------------------------

import itertools
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# -------------------------------------------------
# 1) Sabitler ve Basit Modeller
# -------------------------------------------------

# Lastik verileri (basit model): taban tur süresi (s), tur başına aşınma cezasi (s)
# 1️⃣ Lastik verileri (gerçekçi aşınma)
TYRES = {
    "Soft":   {"base_lap": 85.0, "wear_per_lap": 0.25},
    "Medium": {"base_lap": 86.5, "wear_per_lap": 0.08},
    "Hard":   {"base_lap": 88.0, "wear_per_lap": 0.03},
    "Intermediate": {"base_lap": 92.0, "wear_per_lap": 0.07},
    "Wet":          {"base_lap": 95.0, "wear_per_lap": 0.09},
}

# 2️⃣ simulate_strategy içinde Soft uzun stint cezası
def simulate_strategy(strategy, pit_loss, track_type, weather, aero, engine_map):
    stints = [simulate_stint(t, l, track_type, weather, aero, engine_map) for (t, l) in strategy]
    pit_time_total = pit_loss * max(0, len(stints) - 1)
    total_time = sum(s.stint_time for s in stints) + pit_time_total

    # Soft uzun stint cezası
    for s in stints:
        if s.tyre == "Soft" and s.laps > 20:
            total_time += 0.5 * (s.laps - 20)
    return StrategyResult(strategy=strategy, total_time=total_time, pit_time_total=pit_time_total, stints=stints)

# 3️⃣ search_best_strategy fonksiyonunu top 3 strateji için
def search_best_strategies(total_laps, pit_loss, weather, track_type, aero, engine_map, max_stints=3, min_stint_laps=8, top_n=3):
    allowed_tyres = ALLOWED_TYRES_BY_WEATHER[weather]
    best_strategies: List[StrategyResult] = []

    for stints in range(1, max_stints + 1):
        dists = near_equal_partitions(total_laps, stints, wiggle=2, min_laps=min_stint_laps)
        for tyres in itertools.product(allowed_tyres, repeat=stints):
            for dist in dists:
                strategy = list(zip(tyres, dist))
                res = simulate_strategy(strategy, pit_loss, track_type, weather, aero, engine_map)
                # Top n strateji listesine ekle
                best_strategies.append(res)
                best_strategies = sorted(best_strategies, key=lambda x: x.total_time)[:top_n]
    return best_strategies


# Pist tipine göre hız/aşınma katsayıları
TRACK_MOD = {
    # (base_lap_offset, wear_multiplier)
    "Low speed / Mechanical": (-0.2, 1.05),  # viraj çok → aşınma biraz fazla, taban süre hafif iyi
    "Balanced": (0.0, 1.00),
    "High speed / Aero": (0.3, 0.95),        # uzun düzlükler → taban süre biraz yükselir, aşınma düşük
}

# Hava durumuna göre genel çarpan (kuru = 1.0)
WEATHER_MOD = {
    "Dry": 1.00,
    "Light rain": 1.05,
    "Wet": 1.10,
}

# Basit aero ayarı (aracın downforce/drag dengesi) — taban tura etki
AERO_MOD = {
    # (base_lap_offset, wear_multiplier)
    "Low": (0.40, 0.98),
    "Medium": (0.15, 1.00),
    "High": (-0.10, 1.02),
}

# Motor modu — kısa süreli güç kazancı varsayımı (sabit model)
ENGINE_MAP = {
    "Eco": 0.0,
    "Standard": -0.10,
    "Push": -0.25,
}

ALLOWED_TYRES_BY_WEATHER = {
    "Dry": ["Soft", "Medium", "Hard"],
    "Light rain": ["Intermediate", "Wet"],
    "Wet": ["Intermediate", "Wet"],
}

@dataclass
class StintResult:
    tyre: str
    laps: int
    lap_times: List[float]
    stint_time: float

@dataclass
class StrategyResult:
    strategy: List[Tuple[str, int]]  # [(tyre, laps), ...]
    total_time: float
    pit_time_total: float
    stints: List[StintResult]

# -------------------------------------------------
# 2) Çekirdek Hesap Fonksiyonları
# -------------------------------------------------

def lap_time_model(
    tyre: str,
    lap_idx: int,
    track_type: str,
    weather: str,
    aero: str,
    engine_map: str,
) -> float:
    """Seçili parametrelere göre tek tur tahmini (s)."""
    t = TYRES[tyre]
    base = t["base_lap"]
    wear = t["wear_per_lap"] * lap_idx  # lineer aşınma modeli

    # pist, hava, aero etkileri
    track_offset, track_wear_mul = TRACK_MOD[track_type]
    aero_offset, aero_wear_mul = AERO_MOD[aero]
    weather_mul = WEATHER_MOD[weather]
    engine_gain = ENGINE_MAP[engine_map]

    lap = (base + track_offset + aero_offset + wear * track_wear_mul * aero_wear_mul)
    lap = lap * weather_mul + engine_gain
    return lap


def simulate_stint(
    tyre: str,
    laps: int,
    track_type: str,
    weather: str,
    aero: str,
    engine_map: str,
) -> StintResult:
    lap_times = [lap_time_model(tyre, i, track_type, weather, aero, engine_map) for i in range(laps)]
    return StintResult(tyre=tyre, laps=laps, lap_times=lap_times, stint_time=float(sum(lap_times)))


def simulate_strategy(
    strategy: List[Tuple[str, int]],  # [(tyre, laps), ...]
    pit_loss: float,
    track_type: str,
    weather: str,
    aero: str,
    engine_map: str,
) -> StrategyResult:
    stints = [simulate_stint(t, l, track_type, weather, aero, engine_map) for (t, l) in strategy]
    pit_time_total = pit_loss * max(0, len(stints) - 1)
    total_time = sum(s.stint_time for s in stints) + pit_time_total
    return StrategyResult(strategy=strategy, total_time=total_time, pit_time_total=pit_time_total, stints=stints)


def near_equal_partitions(total_laps: int, parts: int, wiggle: int = 2, min_laps: int = 8) -> List[Tuple[int, ...]]:
    """Toplam turları parts sayıda stinte yaklaşık eşit dağıtır; çeşitlilik için ±wiggle oynaması.
    Aşırı kombinasyon üretmeden makul bir arama alanı sağlar.
    """
    base = total_laps // parts
    rem = total_laps % parts
    base_list = [base + (1 if i < rem else 0) for i in range(parts)]

    candidates = set()
    def clamp_tuple(t):
        return tuple(max(min_laps, x) for x in t)

    # varyasyonlar
    for delta in range(-wiggle, wiggle + 1):
        dist = clamp_tuple(tuple(x + (delta if i == 0 else 0) for i, x in enumerate(base_list)))
        if sum(dist) == total_laps:
            candidates.add(dist)
    for delta in range(-wiggle, wiggle + 1):
        dist = clamp_tuple(tuple(x + (delta if i == len(base_list)//2 else 0) for i, x in enumerate(base_list)))
        if sum(dist) == total_laps:
            candidates.add(dist)
    for delta in range(-wiggle, wiggle + 1):
        dist = clamp_tuple(tuple(x + (delta if i == len(base_list)-1 else 0) for i, x in enumerate(base_list)))
        if sum(dist) == total_laps:
            candidates.add(dist)

    # Eğer minimum tur sebebiyle toplam bozuluyorsa, son parçayı düzelt
    fixed = set()
    for c in candidates:
        s = sum(c)
        if s != total_laps:
            lst = list(c)
            lst[-1] += (total_laps - s)
            c = clamp_tuple(tuple(lst))
        if sum(c) == total_laps:
            fixed.add(c)
    if not fixed:
        fixed.add(tuple(base_list))
    return sorted(fixed)


def search_best_strategy(
    total_laps: int,
    pit_loss: float,
    weather: str,
    track_type: str,
    aero: str,
    engine_map: str,
    max_stints: int = 3,
    min_stint_laps: int = 8,
) -> StrategyResult:
    allowed_tyres = ALLOWED_TYRES_BY_WEATHER[weather]

    best: StrategyResult | None = None

    for stints in range(1, max_stints + 1):
        # Tur dağıtımları
        dists = near_equal_partitions(total_laps, stints, wiggle=2, min_laps=min_stint_laps)
        # Lastik kombinasyonları
        for tyres in itertools.product(allowed_tyres, repeat=stints):
            # Aynı lastiği ardışık iki kez kullanmak isteğe bağlı; burada izin veriyoruz
            for dist in dists:
                strategy = list(zip(tyres, dist))
                res = simulate_strategy(strategy, pit_loss, track_type, weather, aero, engine_map)
                if best is None or res.total_time < best.total_time:
                    best = res
    assert best is not None
    return best

# -------------------------------------------------
# 3) Streamlit Arayüz
# -------------------------------------------------

st.set_page_config(page_title="F1 Strategy Simulator", page_icon="🏁", layout="wide")

st.title("🏁 F1 / Yarış Strateji Simülatörü")
sub = st.empty()

with st.sidebar:
    st.header("Ayarlar")
    preset = st.selectbox(
        "Track Preset",
        [
            "Custom",
            "Monza (High speed / Aero)",
            "Monaco (Low speed / Mechanical)",
            "Barcelona (Balanced)",
        ],
    )

    track_map = {
        "Monza (High speed / Aero)": ("High speed / Aero", 53, 20.0),
        "Monaco (Low speed / Mechanical)": ("Low speed / Mechanical", 78, 17.5),
        "Barcelona (Balanced)": ("Balanced", 66, 22.0),
    }

    if preset != "Custom":
        track_type_default, laps_default, pit_default = track_map[preset]
    else:
        track_type_default, laps_default, pit_default = "Balanced", 60, 22.0

    track_type = st.selectbox("Pist tipi", list(TRACK_MOD.keys()), index=list(TRACK_MOD.keys()).index(track_type_default))
    weather = st.selectbox("Hava durumu", list(WEATHER_MOD.keys()), index=0)
    total_laps = st.slider("Yarış uzunluğu (tur)", 20, 100, laps_default)
    pit_loss = st.slider("Pit stop kaybı (s)", 15.0, 30.0, pit_default, 0.5)

    st.divider()
    st.subheader("Araç Ayarları")
    aero = st.select_slider("Aero (downforce)", options=["Low", "Medium", "High"], value="Medium")
    engine_map = st.select_slider("Motor modu", options=["Eco", "Standard", "Push"], value="Standard")

    st.divider()
    max_stints = st.slider("Maksimum stint sayısı", 1, 4, 3)
    min_stint_laps = st.slider("Minimum stint turu", 5, 20, 8)

col1, col2 = st.columns([1.2, 1])

with col1:
    st.subheader("En İyi Stratejiyi Bul")
    if weather in ("Light rain", "Wet"):
        st.info("Yağışlı koşullarda sadece **Intermediate** ve **Wet** lastikler kullanılır.")

    
with col2:
    st.subheader("Model Notları")
    st.markdown(
        """
        - **Basit lastik modeli:** Taban tur + her turda lineer aşınma cezaları.
        - **Pist & Aero etkileri:** Taban süreye küçük offset; aşınmaya çarpan.
        - **Hava çarpanı:** Kuru = 1.00, yağış arttıkça tur süreleri yükselir.
        - **Motor modu:** Anlık küçük kazanç (Eco/Standard/Push).  
        - **Arama alanı:** 1–4 stint, yaklaşık eşit tur dağılımları (±2 varyasyon).  
        - **Hızlı:** Kombinasyon alanı kontrollü; tek dosyalık paket.
        - **YAPAN-HAZIRLAYAN:** ATA YANKI KILINÇ.
        """
    )

def strategy_advantages(strategy_result: StrategyResult) -> str:
    tyres = [s.tyre for s in strategy_result.stints]
    adv = []
    if "Soft" in tyres and tyres.count("Soft") > 1:
        adv.append("Hızlı başlangıç, riskli uzun stintler")
    if "Medium" in tyres:
        adv.append("Dengeli performans ve aşınma")
    if "Hard" in tyres:
        adv.append("Uzun stint güvenliği, yavaş başlangıç")
    return "; ".join(adv)

# Streamlit: Hesapla butonu
if st.button("🔍 Hesapla", type="primary", key="top3_strategy"):
    best_list = search_best_strategies(
        total_laps=total_laps,
        pit_loss=pit_loss,
        weather=weather,
        track_type=track_type,
        aero=aero,
        engine_map=engine_map,
        max_stints=max_stints,
        min_stint_laps=min_stint_laps,
        top_n=3
    )

    for idx, best in enumerate(best_list, start=1):
        strat_text = " → ".join([f"{t} x{l}" for t, l in best.strategy])
        st.markdown(f"### 🏆 Strateji #{idx}: **{strat_text}**")
        st.markdown(f"**Toplam süre:** `{best.total_time:.2f} s`  •  **Toplam pit kaybı:** `{best.pit_time_total:.2f} s`")
        st.markdown(f"**Avantajlar:** {strategy_advantages(best)}")
        # Tur zamanları DataFrame
        rows = []
        lap_counter = 1
        for stint_idx, stint in enumerate(best.stints, start=1):
            for t in stint.lap_times:
                rows.append({"Lap": lap_counter, "Stint": stint_idx, "Tyre": stint.tyre, "LapTime": t})
                lap_counter += 1
        df = pd.DataFrame(rows)

        # Grafik
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(df["Lap"], df["LapTime"], linewidth=2)
        ax.set_xlabel("Lap")
        ax.set_ylabel("Lap Time (s)")
        ax.set_title(f"Strateji #{idx} Tur Zamanları")
        st.pyplot(fig, use_container_width=True)

        # Stint tablosu
        stint_rows = [{"#": i+1, "Tyre": s.tyre, "Laps": s.laps, "Stint Time (s)": round(s.stint_time, 2)}
                      for i, s in enumerate(best.stints)]
        st.dataframe(pd.DataFrame(stint_rows), use_container_width=True)
        st.divider()
