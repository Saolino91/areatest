# streamlit_app.py

"""
App Streamlit per:
1. Caricare un PDF con orari voli (qualsiasi mese).
2. Parsare i voli PAX.
3. Generare turni guida ottimizzati secondo normativa CCNL.
4. Output: matrice unica turni × validità giorni + analisi.

NORMATIVA IMPLEMENTATA:
- Lavoro = Corsa + Pre/Post turno + Fuori linea + Inoperosità (12% tempo fermo aeroporto)
- Tipi turno: Intero, Semiunico, Spezzato, Sosta Inoperosa, Supplemento
- Priorità: Interi > Semiunici/Spezzati > Sosta Inoperosa > Supplementi
- Target: turni ~7 ore, massimizzare validità multi-giorno
"""

import io
import re
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import List, Optional, Tuple, Dict

import pandas as pd
import pdfplumber
import streamlit as st

# OR-Tools per ottimizzazione
try:
    from ortools.sat.python import cp_model
    ORTOOLS_AVAILABLE = True
except ImportError:
    ORTOOLS_AVAILABLE = False


# =========================
# COSTANTI E CONFIGURAZIONE
# =========================

WEEKDAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

WEEKDAY_LABELS_IT = {
    "Mon": "Lunedì", "Tue": "Martedì", "Wed": "Mercoledì",
    "Thu": "Giovedì", "Fri": "Venerdì", "Sat": "Sabato", "Sun": "Domenica",
}

DAY_PATTERN = re.compile(
    r"^(Sun|Mon|Tue|Wed|Thu|Fri|Sat)\s+(\d{1,2})\s+([A-Za-z]{3})\s+(\d{4})$"
)

MONTH_MAP = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}

DAY_PREFIX = {
    "Mon": "LU", "Tue": "MA", "Wed": "ME", "Thu": "GI",
    "Fri": "VE", "Sat": "SA", "Sun": "DO",
}

# Configurazione tempi (minuti)
TEMPO_VIAGGIO_PCV_APT = 40  # Piazza Cavour -> Aeroporto
TEMPO_VIAGGIO_APT_PCV = 35  # Aeroporto -> Piazza Cavour
ANTICIPO_PARTENZA = 60      # Bus arriva 1h prima del volo
RITARDO_ARRIVO = 25         # Bus parte 25min dopo atterraggio
PRE_TURNO = 20              # Minuti pre-turno
POST_TURNO = 12             # Minuti post-turno
PRE_SOSTA_INOPEROSA = 5     # Minuti pre-sosta inoperosa (pagati 100%)
POST_SOSTA_INOPEROSA = 5    # Minuti post-sosta inoperosa (pagati 100%)
PERCENTUALE_INOPEROSITA = 0.12  # 12% tempo inoperoso

# Soglie raggruppamento voli
DEPARTURE_GROUP_DELTA_MIN = 20
ARRIVAL_GROUP_DELTA_MIN = 20

# Target ore lavoro
TARGET_WORK_MIN = 7 * 60  # 7 ore

# =========================
# CONFIGURAZIONE TIPI TURNO (NORMATIVA CCNL)
# =========================

TURNO_CONFIG = {
    "Intero": {
        "nastro_max": 8 * 60,           # 8 ore
        "lavoro_max": 8 * 60,           # nastro = lavoro
        "pausa_min": 30,                # sosta retribuita >= 30 min
        "priorita": 1,                  # massima priorità
        "descrizione": "Nastro=Lavoro max 8h, pausa retribuita ≥30min"
    },
    "Semiunico": {
        "nastro_max": 9 * 60,           # 9 ore
        "lavoro_max": 8 * 60 + 20,      # 8:20
        "pausa_min": 40,                # interruzione > 40 min
        "pausa_max": 3 * 60,            # interruzione < 3 ore
        "priorita": 2,
        "descrizione": "Nastro max 9h, pausa non retribuita 40min-3h a PCV"
    },
    "Spezzato": {
        "nastro_max": 10 * 60 + 30,     # 10:30
        "pausa_min": 3 * 60,            # interruzione >= 3 ore
        "priorita": 2,                  # stessa priorità semiunico
        "descrizione": "Nastro max 10:30, pausa non retribuita ≥3h a PCV"
    },
    "Sosta Inoperosa": {
        "nastro_max": 9 * 60 + 15,      # 9:15
        "sosta_inop_min": 30,           # sosta > 30 min
        "priorita": 3,
        "fascia_mattina_fine": 15 * 60 + 15,   # deve finire entro 15:15
        "fascia_pomeriggio_inizio": 11 * 60 + 50,  # non può iniziare prima 11:50
        "descrizione": "Nastro max 9:15, sosta in aeroporto >30min (12%)"
    },
    "Supplemento": {
        "nastro_max": 3 * 60,           # 3 ore
        "priorita": 4,                  # minima priorità
        "descrizione": "Nastro=Lavoro max 3h"
    }
}


# =========================
# PARSING PDF
# =========================

def parse_pdf_to_flights_df(file_obj: io.BytesIO) -> pd.DataFrame:
    """Parser per il PDF con orario voli."""
    records: List[dict] = []

    with pdfplumber.open(file_obj) as pdf:
        first_page = pdf.pages[0]
        page_width = first_page.width
        col_width = page_width / 7.0

        def col_index_from_xc(xc: float) -> int:
            idx = int(xc / col_width)
            return max(0, min(6, idx))

        current_date_by_col = {i: None for i in range(7)}
        current_weekday_by_col = {i: None for i in range(7)}

        for page in pdf.pages:
            tables = page.find_tables()
            tables = sorted(tables, key=lambda t: t.bbox[1])

            for t in tables:
                rows = t.extract()
                if not rows:
                    continue

                x0, _, x1, _ = t.bbox
                xc = 0.5 * (x0 + x1)
                col = col_index_from_xc(xc)

                first_row = rows[0] if rows else []
                first_cell = (first_row[0] or "").strip() if first_row else ""
                m = DAY_PATTERN.match(first_cell)

                if m:
                    weekday = m.group(1)
                    day_num = int(m.group(2))
                    month_str = m.group(3)
                    year = int(m.group(4))
                    month_num = MONTH_MAP.get(month_str)
                    
                    if month_num is None:
                        current_date_by_col[col] = None
                        current_weekday_by_col[col] = None
                        continue

                    cur_date = date(year, month_num, day_num)
                    current_date_by_col[col] = cur_date
                    current_weekday_by_col[col] = weekday
                    start_idx = 2
                else:
                    cur_date = current_date_by_col[col]
                    cur_weekday = current_weekday_by_col[col]
                    if cur_date is None or cur_weekday is None:
                        continue
                    start_idx = 1 if first_cell.lower() == "flight" else 0

                cur_date = current_date_by_col[col]
                cur_weekday = current_weekday_by_col[col]
                if cur_date is None or cur_weekday is None:
                    continue

                for row in rows[start_idx:]:
                    if not row or not row[0]:
                        continue

                    flight = (row[0] or "").strip()
                    route = (row[1] or "").strip() if len(row) > 1 else ""
                    ad = (row[2] or "").strip() if len(row) > 2 else ""
                    typ = (row[3] or "").strip() if len(row) > 3 else ""
                    eta = (row[4] or "").strip() if len(row) > 4 else ""
                    etd = (row[5] or "").strip() if len(row) > 5 else ""

                    if not flight:
                        continue

                    records.append({
                        "Date": cur_date,
                        "Weekday": cur_weekday,
                        "Flight": flight,
                        "Route": route,
                        "AD": ad,
                        "Type": typ,
                        "ETA": eta,
                        "ETD": etd,
                    })

    if not records:
        return pd.DataFrame(
            columns=["Date", "Weekday", "Flight", "Route", "AD", "Type", "ETA", "ETD"]
        )

    df = pd.DataFrame(records)
    df["Type"] = df["Type"].str.upper().str.strip()
    df["AD"] = df["AD"].str.upper().str.strip()
    df["ETA"] = df["ETA"].str.strip()
    df["ETD"] = df["ETD"].str.strip()
    df = df[df["Type"] == "PAX"].copy()
    df.replace({"": None}, inplace=True)

    return df


# =========================
# COSTRUZIONE CORSE BUS
# =========================

def combine_date_time(d: date, time_str: str) -> Optional[datetime]:
    if not time_str:
        return None
    try:
        t = datetime.strptime(time_str, "%H:%M").time()
        return datetime.combine(d, t)
    except ValueError:
        return None


def build_bus_trips_from_flights(filtered_flights: pd.DataFrame) -> List[dict]:
    """
    Costruisce i "legs" bus a partire dai voli filtrati.
    
    PARTENZE (AD = P): Bus arriva 1h prima, parte da PCV 40' prima
    ARRIVI (AD = A): Bus parte 25' dopo atterraggio, arriva PCV 35' dopo
    """
    trips: List[dict] = []

    if filtered_flights.empty:
        return trips

    for d in sorted(filtered_flights["Date"].unique()):
        day_rows = filtered_flights[filtered_flights["Date"] == d]

        # PARTENZE: PCV -> APT
        dep_rows = day_rows[day_rows["AD"] == "P"].copy()
        if not dep_rows.empty:
            dep_rows["flight_dt"] = dep_rows["ETD"].apply(lambda s: combine_date_time(d, s))
            dep_rows = dep_rows.dropna(subset=["flight_dt"])
            dep_rows = dep_rows.sort_values("flight_dt")

            i = 0
            while i < len(dep_rows):
                group = [dep_rows.iloc[i]]
                j = i + 1
                while j < len(dep_rows):
                    prev = dep_rows.iloc[j - 1]["flight_dt"]
                    cur = dep_rows.iloc[j]["flight_dt"]
                    delta_min = (cur - prev).total_seconds() / 60.0
                    if delta_min <= DEPARTURE_GROUP_DELTA_MIN:
                        group.append(dep_rows.iloc[j])
                        j += 1
                    else:
                        break

                earliest_dt = group[0]["flight_dt"]
                bus_arr_apt = earliest_dt - timedelta(minutes=ANTICIPO_PARTENZA)
                bus_dep_pcv = bus_arr_apt - timedelta(minutes=TEMPO_VIAGGIO_PCV_APT)

                trips.append({
                    "Date": d,
                    "Direction": "PCV-APT",
                    "service_start": bus_dep_pcv,
                    "service_end": bus_arr_apt,
                    "driving_time": TEMPO_VIAGGIO_PCV_APT,
                    "flights": [g["Flight"] for g in group],
                    "ad_type": "P",
                    "routes": [g["Route"] for g in group],
                })
                i = j

        # ARRIVI: APT -> PCV
        arr_rows = day_rows[day_rows["AD"] == "A"].copy()
        if not arr_rows.empty:
            arr_rows["flight_dt"] = arr_rows["ETA"].apply(lambda s: combine_date_time(d, s))
            arr_rows = arr_rows.dropna(subset=["flight_dt"])
            arr_rows = arr_rows.sort_values("flight_dt")

            i = 0
            while i < len(arr_rows):
                group = [arr_rows.iloc[i]]
                j = i + 1
                while j < len(arr_rows):
                    prev = arr_rows.iloc[j - 1]["flight_dt"]
                    cur = arr_rows.iloc[j]["flight_dt"]
                    delta_min = (cur - prev).total_seconds() / 60.0
                    if delta_min <= ARRIVAL_GROUP_DELTA_MIN:
                        group.append(arr_rows.iloc[j])
                        j += 1
                    else:
                        break

                latest_dt = group[-1]["flight_dt"]
                bus_dep_apt = latest_dt + timedelta(minutes=RITARDO_ARRIVO)
                bus_arr_pcv = bus_dep_apt + timedelta(minutes=TEMPO_VIAGGIO_APT_PCV)

                trips.append({
                    "Date": d,
                    "Direction": "APT-PCV",
                    "service_start": bus_dep_apt,
                    "service_end": bus_arr_pcv,
                    "driving_time": TEMPO_VIAGGIO_APT_PCV,
                    "flights": [g["Flight"] for g in group],
                    "ad_type": "A",
                    "routes": [g["Route"] for g in group],
                })
                i = j

    return sorted(trips, key=lambda t: t["service_start"])


def build_roundtrips_from_trips(trips: List[dict]) -> List[dict]:
    """
    Costruisce le CORSE (andata+ritorno) a partire dai legs.
    Ogni corsa: PCV -> APT -> PCV (garantisce inizio e fine a Piazza Cavour)
    """
    roundtrips: List[dict] = []
    if not trips:
        return roundtrips

    for d in sorted({t["Date"] for t in trips}):
        day_trips = [t for t in trips if t["Date"] == d]
        dep_legs = sorted([t for t in day_trips if t["Direction"] == "PCV-APT"], 
                         key=lambda x: x["service_start"])
        arr_legs = sorted([t for t in day_trips if t["Direction"] == "APT-PCV"], 
                         key=lambda x: x["service_start"])

        used_arr = [False] * len(arr_legs)

        for dep in dep_legs:
            chosen_idx = None
            for idx, arr in enumerate(arr_legs):
                if used_arr[idx]:
                    continue
                if arr["service_start"] >= dep["service_end"]:
                    chosen_idx = idx
                    break

            if chosen_idx is None:
                continue

            arr = arr_legs[chosen_idx]
            used_arr[chosen_idx] = True

            # Calcola tempo inoperoso in aeroporto
            tempo_inoperoso_apt = (arr["service_start"] - dep["service_end"]).total_seconds() / 60.0

            roundtrips.append({
                "Date": d,
                "start": dep["service_start"],
                "end": arr["service_end"],
                "dep_leg": dep,
                "arr_leg": arr,
                "tempo_inoperoso_apt": tempo_inoperoso_apt,
                "driving_time": dep["driving_time"] + arr["driving_time"],
            })

    return sorted(roundtrips, key=lambda r: r["start"])


# =========================
# CALCOLO LAVORO E CLASSIFICAZIONE TURNI
# =========================

def calcola_lavoro_turno(roundtrips: List[dict], nastro_min: float) -> Dict:
    """
    Calcola il lavoro effettivo secondo normativa:
    - Corsa (tempo guida)
    - Pre turno (20 min) + Post turno (12 min)
    - Fuori linea
    - Inoperosità in aeroporto al 12%
    
    Returns dict con dettaglio calcolo
    """
    if not roundtrips:
        return {"lavoro_totale": 0, "dettaglio": {}}
    
    # Tempo guida effettivo
    tempo_guida = sum(rt["driving_time"] for rt in roundtrips)
    
    # Pre e post turno
    pre_post = PRE_TURNO + POST_TURNO
    
    # Tempo inoperoso totale in aeroporto
    tempo_inoperoso_totale = sum(rt["tempo_inoperoso_apt"] for rt in roundtrips)
    
    # Inoperosità retribuita al 12%
    inoperosita_retribuita = tempo_inoperoso_totale * PERCENTUALE_INOPEROSITA
    
    # Gap tra corse (a PCV) - considerato fuori linea se breve
    gaps_pcv = []
    for i in range(len(roundtrips) - 1):
        gap = (roundtrips[i + 1]["start"] - roundtrips[i]["end"]).total_seconds() / 60.0
        gaps_pcv.append(gap)
    
    # Fuori linea: gap brevi a PCV (< 40 min) sono considerati lavoro
    fuori_linea = sum(g for g in gaps_pcv if g < 40)
    
    # Lavoro totale
    lavoro_totale = tempo_guida + pre_post + inoperosita_retribuita + fuori_linea
    
    return {
        "lavoro_totale": lavoro_totale,
        "tempo_guida": tempo_guida,
        "pre_post": pre_post,
        "tempo_inoperoso_apt": tempo_inoperoso_totale,
        "inoperosita_retribuita": inoperosita_retribuita,
        "fuori_linea": fuori_linea,
        "gaps_pcv": gaps_pcv,
    }


def classifica_turno(roundtrips: List[dict], nastro_min: float, lavoro_info: Dict) -> str:
    """
    Classifica il turno secondo normativa CCNL:
    
    1. Supplemento: nastro <= 3h
    2. Intero: nastro = lavoro <= 8h, con pausa >= 30min
    3. Sosta Inoperosa: nastro <= 9:15, sosta in APT > 30min
    4. Semiunico: nastro <= 9h, pausa a PCV tra 40min e 3h
    5. Spezzato: nastro <= 10:30, pausa a PCV >= 3h
    """
    lavoro_totale = lavoro_info["lavoro_totale"]
    gaps_pcv = lavoro_info.get("gaps_pcv", [])
    tempo_inoperoso_apt = lavoro_info.get("tempo_inoperoso_apt", 0)
    
    max_gap_pcv = max(gaps_pcv) if gaps_pcv else 0
    max_inop_apt = max((rt["tempo_inoperoso_apt"] for rt in roundtrips), default=0)
    
    # 1. SUPPLEMENTO: turno breve max 3h
    if nastro_min <= TURNO_CONFIG["Supplemento"]["nastro_max"]:
        return "Supplemento"
    
    # 2. INTERO: nastro = lavoro <= 8h
    # Il turno è "intero" se è abbastanza saturo (nastro ≈ lavoro)
    if nastro_min <= TURNO_CONFIG["Intero"]["nastro_max"]:
        # Verifica che ci sia una pausa >= 30 min (retribuita)
        has_pausa = any(g >= 30 for g in gaps_pcv) or max_inop_apt >= 30
        # Turno saturo: differenza nastro-lavoro contenuta
        is_saturo = (nastro_min - lavoro_totale) < 60  # margine 1h
        if is_saturo and has_pausa:
            return "Intero"
        elif is_saturo:
            return "Intero"  # anche senza pausa formale se saturo
    
    # 3. SOSTA INOPEROSA: nastro <= 9:15, sosta in aeroporto > 30min
    if nastro_min <= TURNO_CONFIG["Sosta Inoperosa"]["nastro_max"]:
        if max_inop_apt > TURNO_CONFIG["Sosta Inoperosa"]["sosta_inop_min"]:
            return "Sosta Inoperosa"
    
    # 4. SEMIUNICO: nastro <= 9h, pausa a PCV tra 40min e 3h
    if nastro_min <= TURNO_CONFIG["Semiunico"]["nastro_max"]:
        if (TURNO_CONFIG["Semiunico"]["pausa_min"] <= max_gap_pcv < 
            TURNO_CONFIG["Semiunico"]["pausa_max"]):
            return "Semiunico"
    
    # 5. SPEZZATO: nastro <= 10:30, pausa a PCV >= 3h
    if nastro_min <= TURNO_CONFIG["Spezzato"]["nastro_max"]:
        if max_gap_pcv >= TURNO_CONFIG["Spezzato"]["pausa_min"]:
            return "Spezzato"
    
    # Fallback: se nastro <= 9:15 con inoperosità
    if nastro_min <= TURNO_CONFIG["Sosta Inoperosa"]["nastro_max"] and tempo_inoperoso_apt > 0:
        return "Sosta Inoperosa"
    
    # Fallback finale
    if nastro_min <= 8 * 60:
        return "Intero"
    
    return "Altro"


def verifica_vincoli_fascia_oraria(turno: dict) -> bool:
    """
    Verifica vincoli fascia oraria per Sosta Inoperosa:
    - Mattina: deve terminare entro 15:15
    - Pomeriggio: non può iniziare prima delle 11:50
    """
    if turno.get("tipo_turno") != "Sosta Inoperosa":
        return True
    
    start_dt = turno["shift_start_dt"]
    end_dt = turno["shift_end_dt"]
    
    start_minutes = start_dt.hour * 60 + start_dt.minute
    end_minutes = end_dt.hour * 60 + end_dt.minute
    
    # Mattina (inizio < 12:00)
    if start_minutes < 12 * 60:
        # Deve finire entro 15:15
        if end_minutes > TURNO_CONFIG["Sosta Inoperosa"]["fascia_mattina_fine"]:
            return False
    else:
        # Pomeriggio: non può iniziare prima delle 11:50
        if start_minutes < TURNO_CONFIG["Sosta Inoperosa"]["fascia_pomeriggio_inizio"]:
            return False
    
    return True


# =========================
# OTTIMIZZAZIONE TURNI
# =========================

def genera_turni_ottimizzati(roundtrips: List[dict], weekday: str) -> Tuple[List[dict], dict]:
    """
    Genera turni ottimizzati rispettando:
    1. Vincoli CCNL (nastro, lavoro, pause)
    2. Priorità tipi turno
    3. Massimizzazione validità multi-giorno
    4. Target ~7 ore lavoro
    """
    if not roundtrips:
        return [], {"status": "NO_ROUNDTRIPS"}
    
    # Raggruppa per data
    dates_set = sorted({rt["Date"] for rt in roundtrips})
    
    # Prima genera turni per ogni data
    turni_per_data = {}
    for d in dates_set:
        rts_d = [rt for rt in roundtrips if rt["Date"] == d]
        turni_d = genera_turni_singola_data(rts_d, weekday, d)
        turni_per_data[d] = turni_d
    
    # Poi unifica turni simili tra date diverse
    turni_unificati = unifica_turni_multi_data(turni_per_data, dates_set, weekday)
    
    # Statistiche
    stats = {
        "status": "OK",
        "num_turni": len(turni_unificati),
        "num_date": len(dates_set),
        "turni_per_tipo": defaultdict(int),
    }
    for t in turni_unificati:
        stats["turni_per_tipo"][t["tipo_turno"]] += 1
    
    return turni_unificati, stats


def genera_turni_singola_data(roundtrips: List[dict], weekday: str, d: date) -> List[dict]:
    """
    Genera turni per una singola data usando algoritmo greedy con priorità CCNL.
    """
    if not roundtrips:
        return []
    
    rts_sorted = sorted(roundtrips, key=lambda r: r["start"])
    turni: List[dict] = []
    
    # Prova a costruire turni con priorità: Intero > Semiunico/Spezzato > Sosta Inoperosa > Supplemento
    used = [False] * len(rts_sorted)
    
    def try_build_shift(start_idx: int, target_type: str) -> Optional[dict]:
        """Prova a costruire un turno del tipo specificato partendo da start_idx."""
        if used[start_idx]:
            return None
        
        candidate = [rts_sorted[start_idx]]
        
        # Prova ad aggiungere altre corse compatibili
        for j in range(start_idx + 1, len(rts_sorted)):
            if used[j]:
                continue
            
            rt = rts_sorted[j]
            last_rt = candidate[-1]
            
            # Deve essere sequenziale (non sovrapposto)
            if rt["start"] < last_rt["end"]:
                continue
            
            # Calcola metriche tentative
            tentative = candidate + [rt]
            shift_start = tentative[0]["start"] - timedelta(minutes=PRE_TURNO)
            shift_end = tentative[-1]["end"] + timedelta(minutes=POST_TURNO)
            nastro = (shift_end - shift_start).total_seconds() / 60.0
            
            lavoro_info = calcola_lavoro_turno(tentative, nastro)
            lavoro = lavoro_info["lavoro_totale"]
            
            # Verifica vincoli del tipo turno target
            if target_type == "Intero":
                if nastro > TURNO_CONFIG["Intero"]["nastro_max"]:
                    continue
                if lavoro > TURNO_CONFIG["Intero"]["lavoro_max"]:
                    continue
            elif target_type == "Semiunico":
                if nastro > TURNO_CONFIG["Semiunico"]["nastro_max"]:
                    continue
                if lavoro > TURNO_CONFIG["Semiunico"]["lavoro_max"]:
                    continue
            elif target_type == "Spezzato":
                if nastro > TURNO_CONFIG["Spezzato"]["nastro_max"]:
                    continue
            elif target_type == "Sosta Inoperosa":
                if nastro > TURNO_CONFIG["Sosta Inoperosa"]["nastro_max"]:
                    continue
            elif target_type == "Supplemento":
                if nastro > TURNO_CONFIG["Supplemento"]["nastro_max"]:
                    continue
            
            # Max 3 corse per turno
            if len(tentative) > 3:
                continue
            
            # Se vicino al target 7h, accetta
            if lavoro >= TARGET_WORK_MIN - 30:  # almeno 6:30
                candidate.append(rt)
                if lavoro >= TARGET_WORK_MIN + 60:  # max 8h
                    break
            elif nastro <= 6 * 60:  # turno ancora corto, continua
                candidate.append(rt)
        
        # Costruisci turno
        shift_start = candidate[0]["start"] - timedelta(minutes=PRE_TURNO)
        shift_end = candidate[-1]["end"] + timedelta(minutes=POST_TURNO)
        nastro = (shift_end - shift_start).total_seconds() / 60.0
        lavoro_info = calcola_lavoro_turno(candidate, nastro)
        tipo = classifica_turno(candidate, nastro, lavoro_info)
        
        # Verifica che il tipo sia compatibile con target
        if target_type != "any" and tipo != target_type:
            # Accetta comunque se è un tipo con priorità migliore
            priorita_target = TURNO_CONFIG.get(target_type, {}).get("priorita", 99)
            priorita_effettiva = TURNO_CONFIG.get(tipo, {}).get("priorita", 99)
            if priorita_effettiva > priorita_target:
                return None
        
        return {
            "roundtrips": candidate,
            "shift_start_dt": shift_start,
            "shift_end_dt": shift_end,
            "nastro_min": nastro,
            "lavoro_info": lavoro_info,
            "tipo_turno": tipo,
            "date": d,
            "weekday": weekday,
        }
    
    # Prima passa: prova a costruire turni Interi
    for i in range(len(rts_sorted)):
        if used[i]:
            continue
        turno = try_build_shift(i, "Intero")
        if turno and turno["tipo_turno"] == "Intero":
            turni.append(turno)
            for rt in turno["roundtrips"]:
                idx = rts_sorted.index(rt)
                used[idx] = True
    
    # Seconda passa: Semiunici e Spezzati
    for i in range(len(rts_sorted)):
        if used[i]:
            continue
        turno = try_build_shift(i, "Semiunico")
        if turno and turno["tipo_turno"] in ("Semiunico", "Spezzato"):
            turni.append(turno)
            for rt in turno["roundtrips"]:
                idx = rts_sorted.index(rt)
                used[idx] = True
    
    # Terza passa: Sosta Inoperosa
    for i in range(len(rts_sorted)):
        if used[i]:
            continue
        turno = try_build_shift(i, "Sosta Inoperosa")
        if turno:
            turni.append(turno)
            for rt in turno["roundtrips"]:
                idx = rts_sorted.index(rt)
                used[idx] = True
    
    # Ultima passa: qualsiasi tipo (Supplementi inclusi)
    for i in range(len(rts_sorted)):
        if used[i]:
            continue
        turno = try_build_shift(i, "any")
        if turno:
            turni.append(turno)
            for rt in turno["roundtrips"]:
                idx = rts_sorted.index(rt)
                used[idx] = True
    
    return sorted(turni, key=lambda t: t["shift_start_dt"])


def calcola_signature_turno(turno: dict) -> tuple:
    """
    Calcola una 'firma' del turno per identificare turni equivalenti tra date diverse.
    """
    return (
        turno["shift_start_dt"].time(),
        turno["shift_end_dt"].time(),
        turno["tipo_turno"],
        len(turno["roundtrips"]),
    )


def unifica_turni_multi_data(turni_per_data: Dict[date, List[dict]], 
                              dates: List[date], weekday: str) -> List[dict]:
    """
    Unifica turni simili tra date diverse per massimizzare validità multi-giorno.
    """
    # Raggruppa turni per signature
    turni_per_signature = defaultdict(list)
    
    for d, turni in turni_per_data.items():
        for t in turni:
            sig = calcola_signature_turno(t)
            turni_per_signature[sig].append((d, t))
    
    # Crea turni unificati
    turni_unificati = []
    prefix = DAY_PREFIX.get(weekday, weekday[:2].upper())
    
    # Separa AM e PM
    am_sigs = [(sig, ts) for sig, ts in turni_per_signature.items() 
               if ts[0][1]["shift_start_dt"].hour < 12]
    pm_sigs = [(sig, ts) for sig, ts in turni_per_signature.items() 
               if ts[0][1]["shift_start_dt"].hour >= 12]
    
    am_sigs.sort(key=lambda x: x[0][0])  # ordina per orario inizio
    pm_sigs.sort(key=lambda x: x[0][0])
    
    idx_am = 1
    idx_pm = 50
    
    for sig, turni_list in am_sigs:
        sample = turni_list[0][1]
        date_valide = [d for d, _ in turni_list]
        
        # Suffisso basato su numero corse
        n_corse = len(sample["roundtrips"])
        if n_corse >= 3:
            suffix = "I"
        elif n_corse == 2:
            suffix = "P"
        else:
            suffix = "S"
        
        code = f"{prefix}{idx_am:02d}{suffix}"
        idx_am += 1
        
        # Costruisci dettaglio corse
        detail_lines = []
        for rt in sample["roundtrips"]:
            dep = rt["dep_leg"]
            arr = rt["arr_leg"]
            
            line_dep = f"{dep['service_start'].strftime('%H:%M')}, Piazza Cavour, Aeroporto, {dep['service_end'].strftime('%H:%M')}, {','.join(dep['flights'])}"
            line_arr = f"{arr['service_start'].strftime('%H:%M')}, Aeroporto, Piazza Cavour, {arr['service_end'].strftime('%H:%M')}, {','.join(arr['flights'])}"
            detail_lines.extend([line_dep, line_arr])
        
        turni_unificati.append({
            "code": code,
            "weekday": weekday,
            "tipo_turno": sample["tipo_turno"],
            "shift_start_dt": sample["shift_start_dt"],
            "shift_end_dt": sample["shift_end_dt"],
            "nastro_min": sample["nastro_min"],
            "lavoro_min": sample["lavoro_info"]["lavoro_totale"],
            "lavoro_info": sample["lavoro_info"],
            "n_corse": n_corse,
            "date_valide": date_valide,
            "detail": "\n".join(detail_lines),
        })
    
    for sig, turni_list in pm_sigs:
        sample = turni_list[0][1]
        date_valide = [d for d, _ in turni_list]
        
        n_corse = len(sample["roundtrips"])
        if n_corse >= 3:
            suffix = "I"
        elif n_corse == 2:
            suffix = "P"
        else:
            suffix = "S"
        
        code = f"{prefix}{idx_pm:02d}{suffix}"
        idx_pm += 1
        
        detail_lines = []
        for rt in sample["roundtrips"]:
            dep = rt["dep_leg"]
            arr = rt["arr_leg"]
            
            line_dep = f"{dep['service_start'].strftime('%H:%M')}, Piazza Cavour, Aeroporto, {dep['service_end'].strftime('%H:%M')}, {','.join(dep['flights'])}"
            line_arr = f"{arr['service_start'].strftime('%H:%M')}, Aeroporto, Piazza Cavour, {arr['service_end'].strftime('%H:%M')}, {','.join(arr['flights'])}"
            detail_lines.extend([line_dep, line_arr])
        
        turni_unificati.append({
            "code": code,
            "weekday": weekday,
            "tipo_turno": sample["tipo_turno"],
            "shift_start_dt": sample["shift_start_dt"],
            "shift_end_dt": sample["shift_end_dt"],
            "nastro_min": sample["nastro_min"],
            "lavoro_min": sample["lavoro_info"]["lavoro_totale"],
            "lavoro_info": sample["lavoro_info"],
            "n_corse": n_corse,
            "date_valide": date_valide,
            "detail": "\n".join(detail_lines),
        })
    
    return sorted(turni_unificati, key=lambda t: t["shift_start_dt"])


def genera_analisi_turni(turni: List[dict], dates: List[date], weekday: str) -> str:
    """
    Genera un'analisi testuale dei turni generati.
    """
    if not turni:
        return "Nessun turno generato."
    
    lines = []
    lines.append(f"## Analisi Turni - {WEEKDAY_LABELS_IT.get(weekday, weekday)}")
    lines.append("")
    
    # Statistiche generali
    n_turni = len(turni)
    n_date = len(dates)
    lines.append(f"**Turni generati:** {n_turni}")
    lines.append(f"**Date coperte:** {n_date}")
    lines.append("")
    
    # Distribuzione per tipo
    per_tipo = defaultdict(int)
    for t in turni:
        per_tipo[t["tipo_turno"]] += 1
    
    lines.append("**Distribuzione per tipo:**")
    for tipo in ["Intero", "Semiunico", "Spezzato", "Sosta Inoperosa", "Supplemento"]:
        if tipo in per_tipo:
            lines.append(f"- {tipo}: {per_tipo[tipo]}")
    lines.append("")
    
    # Statistiche lavoro
    lavori = [t["lavoro_min"] for t in turni]
    nastri = [t["nastro_min"] for t in turni]
    
    if lavori:
        avg_lavoro = sum(lavori) / len(lavori)
        min_lavoro = min(lavori)
        max_lavoro = max(lavori)
        
        lines.append("**Statistiche lavoro:**")
        lines.append(f"- Media: {int(avg_lavoro // 60)}h {int(avg_lavoro % 60)}min")
        lines.append(f"- Min: {int(min_lavoro // 60)}h {int(min_lavoro % 60)}min")
        lines.append(f"- Max: {int(max_lavoro // 60)}h {int(max_lavoro % 60)}min")
        lines.append("")
    
    # Validità multi-giorno
    validita = [len(t["date_valide"]) for t in turni]
    avg_validita = sum(validita) / len(validita) if validita else 0
    turni_tutti_giorni = sum(1 for v in validita if v == n_date)
    
    lines.append("**Validità multi-giorno:**")
    lines.append(f"- Validità media: {avg_validita:.1f} giorni su {n_date}")
    lines.append(f"- Turni validi tutti i giorni: {turni_tutti_giorni} ({turni_tutti_giorni/n_turni*100:.0f}%)")
    lines.append("")
    
    # Osservazioni
    lines.append("**Osservazioni:**")
    
    if per_tipo.get("Intero", 0) > n_turni * 0.5:
        lines.append("- ✅ Buona percentuale di turni Interi (ottimale per gestione)")
    
    if avg_validita > n_date * 0.7:
        lines.append("- ✅ Alta validità multi-giorno (minimizza variazioni)")
    
    if avg_lavoro < 7 * 60:
        lines.append(f"- ⚠️ Lavoro medio sotto le 7h target ({int(avg_lavoro // 60)}h {int(avg_lavoro % 60)}min)")
    elif avg_lavoro > 8 * 60:
        lines.append(f"- ⚠️ Lavoro medio sopra le 8h ({int(avg_lavoro // 60)}h {int(avg_lavoro % 60)}min)")
    else:
        lines.append("- ✅ Lavoro medio in linea con target 7-8h")
    
    if per_tipo.get("Supplemento", 0) > n_turni * 0.3:
        lines.append("- ⚠️ Alta percentuale di Supplementi - valutare ottimizzazione")
    
    return "\n".join(lines)


# =========================
# UI STREAMLIT
# =========================

def main():
    st.set_page_config(
        page_title="Turni Guida Aeroporto",
        page_icon="🚌",
        layout="wide",
    )

    st.markdown("""
        <style>
        .block-container { padding: 1.5rem 2rem; }
        h1 { text-align: center; }
        .info-card {
            background: rgba(15,23,42,0.9);
            padding: 1rem 1.2rem;
            border-radius: 0.9rem;
            border: 1px solid rgba(148,163,184,0.35);
        }
        .metric-card {
            background: linear-gradient(135deg, rgba(59,130,246,0.1), rgba(147,51,234,0.1));
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid rgba(59,130,246,0.3);
        }
        .turno-intero { background-color: rgba(34, 197, 94, 0.2); }
        .turno-semiunico { background-color: rgba(59, 130, 246, 0.2); }
        .turno-spezzato { background-color: rgba(168, 85, 247, 0.2); }
        .turno-inoperoso { background-color: rgba(251, 191, 36, 0.2); }
        .turno-supplemento { background-color: rgba(156, 163, 175, 0.2); }
        </style>
    """, unsafe_allow_html=True)

    st.title("🚌 Generatore Turni Guida Aeroporto")

    with st.container():
        st.markdown("""
            <div class="info-card">
                <p><strong>Normativa CCNL Implementata:</strong></p>
                <ul style="margin-top:0.3rem;">
                    <li><strong>Lavoro</strong> = Corsa + Pre/Post + Fuori linea + Inoperosità (12%)</li>
                    <li><strong>Priorità</strong>: Interi → Semiunici/Spezzati → Sosta Inoperosa → Supplementi</li>
                    <li><strong>Target</strong>: turni ~7 ore, massima validità multi-giorno</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    st.write("")
    
    uploaded_file = st.file_uploader("📄 Carica il PDF con gli orari dei voli", type=["pdf"])

    if uploaded_file is None:
        st.info("Carica il PDF per procedere.")
        
        # Mostra riepilogo normativa
        with st.expander("📋 Riepilogo Normativa Turni"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Tipi di Turno:**")
                for tipo, config in TURNO_CONFIG.items():
                    nastro = config["nastro_max"]
                    h, m = int(nastro // 60), int(nastro % 60)
                    st.write(f"- **{tipo}**: nastro max {h}h{m:02d}min")
            
            with col2:
                st.markdown("**Calcolo Lavoro:**")
                st.write("- Tempo guida (corsa)")
                st.write(f"- Pre turno: {PRE_TURNO} min")
                st.write(f"- Post turno: {POST_TURNO} min")
                st.write(f"- Inoperosità aeroporto: {int(PERCENTUALE_INOPEROSITA*100)}%")
        
        return

    with st.spinner("Parsing del PDF in corso..."):
        flights_df = parse_pdf_to_flights_df(uploaded_file)

    if flights_df.empty:
        st.error("Non sono stati trovati voli PAX.")
        return

    unique_days = sorted(flights_df["Date"].unique())
    num_flights = len(flights_df)

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.metric("Voli PAX", num_flights)
    with col2:
        st.metric("Giorni", len(unique_days))
    with col3:
        st.write(f"📆 **{unique_days[0].strftime('%d/%m/%Y')} – {unique_days[-1].strftime('%d/%m/%Y')}**")

    st.success("✅ PDF caricato correttamente")

    # Selezione giorno
    weekdays_present = sorted(
        flights_df["Weekday"].unique(),
        key=lambda x: WEEKDAY_ORDER.index(x),
    )

    st.sidebar.header("⚙️ Configurazione")
    selected_weekday = st.sidebar.selectbox(
        "Giorno della settimana",
        options=weekdays_present,
        format_func=lambda x: WEEKDAY_LABELS_IT.get(x, x),
    )

    # Filtri opzionali
    with st.sidebar.expander("Filtri avanzati"):
        flight_filter = st.text_input("Codice volo (contiene)", placeholder="FR, EN8...")
        
        airport_options = sorted(flights_df["Route"].unique())
        selected_airports = st.multiselect("Aeroporti", options=airport_options)
        
        ad_choice = st.radio(
            "Tipo movimento",
            ["Tutti", "Solo arrivi (A)", "Solo partenze (P)"],
        )

    # Filtra voli
    filtered = flights_df[flights_df["Weekday"] == selected_weekday].copy()
    
    if flight_filter:
        filtered = filtered[filtered["Flight"].str.contains(flight_filter, case=False, na=False)]
    if selected_airports:
        filtered = filtered[filtered["Route"].isin(selected_airports)]
    if ad_choice == "Solo arrivi (A)":
        filtered = filtered[filtered["AD"] == "A"]
    elif ad_choice == "Solo partenze (P)":
        filtered = filtered[filtered["AD"] == "P"]

    if filtered.empty:
        st.warning("Nessun volo per i filtri selezionati.")
        return

    label_it = WEEKDAY_LABELS_IT.get(selected_weekday, selected_weekday)
    st.markdown(f"### 📅 {label_it}")
    st.write(f"Voli filtrati: {len(filtered)} su {filtered['Date'].nunique()} date")

    # GENERAZIONE TURNI
    st.markdown("---")
    
    if st.button("🚀 Genera Turni Ottimizzati", type="primary", use_container_width=True):
        with st.spinner("Generazione turni in corso..."):
            # Costruisci corse
            trips = build_bus_trips_from_flights(filtered)
            roundtrips = build_roundtrips_from_trips(trips)
            
            if not roundtrips:
                st.error("Impossibile costruire corse dai voli disponibili.")
                return
            
            # Genera turni
            turni, stats = genera_turni_ottimizzati(roundtrips, selected_weekday)
            
            if not turni:
                st.warning("Nessun turno generato.")
                return
        
        # MATRICE VALIDITÀ
        st.markdown("### 📊 Matrice Turni × Validità")
        
        dates_for_matrix = sorted(filtered["Date"].unique())
        
        # Costruisci DataFrame matrice
        rows = []
        for t in turni:
            row = {
                "Codice": t["code"],
                "Tipo": t["tipo_turno"],
                "Inizio": t["shift_start_dt"].strftime("%H:%M"),
                "Fine": t["shift_end_dt"].strftime("%H:%M"),
                "Nastro": f"{int(t['nastro_min']//60)}:{int(t['nastro_min']%60):02d}",
                "Lavoro": f"{int(t['lavoro_min']//60)}:{int(t['lavoro_min']%60):02d}",
                "Corse": t["n_corse"],
            }
            
            # Colonne validità per data
            for d in dates_for_matrix:
                col_name = d.strftime("%d-%m")
                row[col_name] = "✅" if d in t["date_valide"] else ""
            
            rows.append(row)
        
        df_matrix = pd.DataFrame(rows)
        
        # Styling condizionale per tipo turno
        def style_tipo(val):
            colors = {
                "Intero": "background-color: rgba(34, 197, 94, 0.3)",
                "Semiunico": "background-color: rgba(59, 130, 246, 0.3)",
                "Spezzato": "background-color: rgba(168, 85, 247, 0.3)",
                "Sosta Inoperosa": "background-color: rgba(251, 191, 36, 0.3)",
                "Supplemento": "background-color: rgba(156, 163, 175, 0.3)",
            }
            return colors.get(val, "")
        
        styled = df_matrix.style.applymap(style_tipo, subset=["Tipo"])
        st.dataframe(styled, use_container_width=True, height=400)
        
        # Download CSV
        csv = df_matrix.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Scarica matrice CSV",
            data=csv,
            file_name=f"turni_{label_it.lower()}.csv",
            mime="text/csv",
        )
        
        # LEGENDA
        st.markdown("**Legenda tipi turno:**")
        cols = st.columns(5)
        legend = [
            ("🟢 Intero", "Nastro=Lavoro ≤8h"),
            ("🔵 Semiunico", "Nastro ≤9h, pausa 40min-3h"),
            ("🟣 Spezzato", "Nastro ≤10:30, pausa ≥3h"),
            ("🟡 Sosta Inop.", "Nastro ≤9:15, sosta APT"),
            ("⚪ Supplemento", "Nastro ≤3h"),
        ]
        for col, (nome, desc) in zip(cols, legend):
            col.write(f"**{nome}**")
            col.caption(desc)
        
        # DETTAGLIO TURNO
        st.markdown("---")
        st.markdown("### 🔍 Dettaglio Corse")
        
        selected_code = st.selectbox(
            "Seleziona turno",
            options=[t["code"] for t in turni],
        )
        
        if selected_code:
            turno_sel = next(t for t in turni if t["code"] == selected_code)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Tipo", turno_sel["tipo_turno"])
            col2.metric("Nastro", f"{int(turno_sel['nastro_min']//60)}h {int(turno_sel['nastro_min']%60)}min")
            col3.metric("Lavoro", f"{int(turno_sel['lavoro_min']//60)}h {int(turno_sel['lavoro_min']%60)}min")
            
            st.markdown("**Dettaglio corse:**")
            st.text(turno_sel["detail"])
            
            # Dettaglio calcolo lavoro
            with st.expander("📊 Dettaglio calcolo lavoro"):
                info = turno_sel["lavoro_info"]
                st.write(f"- Tempo guida: {info['tempo_guida']:.0f} min")
                st.write(f"- Pre/Post turno: {info['pre_post']:.0f} min")
                st.write(f"- Inoperosità APT: {info['tempo_inoperoso_apt']:.0f} min × 12% = {info['inoperosita_retribuita']:.1f} min")
                st.write(f"- Fuori linea: {info['fuori_linea']:.0f} min")
                st.write(f"- **Totale lavoro: {info['lavoro_totale']:.0f} min**")
        
        # ANALISI
        st.markdown("---")
        analisi = genera_analisi_turni(turni, dates_for_matrix, selected_weekday)
        st.markdown(analisi)


if __name__ == "__main__":
    main()
