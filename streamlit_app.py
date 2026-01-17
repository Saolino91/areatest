# streamlit_app.py

"""
App Streamlit per ottimizzazione turni shuttle aeroporto con vincoli CCNL Conerobus
PRIORITÀ ASSOLUTA: tutte le corse devono essere coperte

NORMATIVA IMPLEMENTATA:
- Lavoro = Corsa + Pre/Post turno (20+12) + Inoperosità (12% tempo fermo APT con 5+5 pre/post sosta)
- Tipi turno: Intero (8h), Semiunico (9h), Spezzato (10:30), Sosta Inoperosa (9:15), Supplemento (3h)
- Priorità: Interi > Semiunici/Spezzati > Sosta Inoperosa > Supplementi
- Target: turni ~7 ore
"""

import io
import re
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import List, Optional, Tuple, Dict

import pandas as pd
import pdfplumber
import streamlit as st


# =========================
# COSTANTI E CONFIGURAZIONE CCNL
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

# Tempi fissi
TEMPO_VIAGGIO_PCV_APT = 40  # minuti
TEMPO_VIAGGIO_APT_PCV = 35  # minuti
PRE_TURNO = 20              # minuti pre-turno
POST_TURNO = 12             # minuti post-turno
PRE_SOSTA_INOP = 5          # minuti pre-sosta inoperosa (pagati 100%)
POST_SOSTA_INOP = 5         # minuti post-sosta inoperosa (pagati 100%)
PERCENTUALE_INOPEROSITA = 0.12  # 12%

# Raggruppamento voli
DEPARTURE_GROUP_DELTA_MIN = 20
ARRIVAL_GROUP_DELTA_MIN = 20

# Target lavoro
TARGET_WORK_MIN = 7 * 60  # 7 ore

# Vincoli CCNL per tipo turno
TURNO_VINCOLI = {
    "Intero": {
        "nastro_max": 480,      # 8h - nastro = lavoro
        "pausa_retrib_min": 30,  # pausa retribuita >= 30 min
        "priorita": 1,
    },
    "Sosta Inoperosa": {
        "nastro_max": 555,      # 9h15
        "sosta_apt_min": 30,    # sosta in aeroporto > 30 min
        "mattina_fine_max": 15*60+15,   # deve finire entro 15:15
        "pomeriggio_inizio_min": 11*60+50,  # non può iniziare prima 11:50
        "priorita": 3,
    },
    "Semiunico": {
        "nastro_max": 540,      # 9h
        "lavoro_max": 500,      # 8h20
        "pausa_pcv_min": 40,    # pausa non retrib a PCV >= 40 min
        "pausa_pcv_max": 180,   # pausa non retrib a PCV < 3h
        "priorita": 2,
    },
    "Spezzato": {
        "nastro_max": 630,      # 10h30
        "pausa_pcv_min": 180,   # pausa non retrib a PCV >= 3h
        "priorita": 2,
    },
    "Supplemento": {
        "nastro_max": 180,      # 3h - nastro = lavoro
        "priorita": 4,
    },
}


# =========================
# PARSING PDF
# =========================

def parse_pdf_to_flights_df(file_obj: io.BytesIO) -> pd.DataFrame:
    """Parser PDF voli"""
    records: List[dict] = []

    with pdfplumber.open(file_obj) as pdf:
        first_page = pdf.pages[0]
        page_width = first_page.width
        col_width = page_width / 7.0

        def col_index_from_xc(xc: float) -> int:
            idx = int(xc / col_width)
            return max(0, min(idx, 6))

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


def build_all_bus_legs(filtered_flights: pd.DataFrame) -> List[dict]:
    """
    Costruisce TUTTI i legs bus (singole tratte PCV→APT o APT→PCV).
    Ogni leg rappresenta una CORSA EFFETTIVA che DEVE essere coperta.
    
    Returns:
        Lista di legs, ognuno con:
        - id: identificativo univoco
        - Date, Direction, service_start, service_end
        - flights: lista codici volo serviti
        - ad_type: 'P' (partenza) o 'A' (arrivo)
    """
    legs: List[dict] = []
    leg_id = 0
    
    if filtered_flights.empty:
        return legs

    for d in sorted(filtered_flights["Date"].unique()):
        day_rows = filtered_flights[filtered_flights["Date"] == d]

        # PARTENZE: PCV → APT (bus porta passeggeri al volo)
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
                bus_arr_apt = earliest_dt - timedelta(hours=1)
                bus_dep_pcv = bus_arr_apt - timedelta(minutes=TEMPO_VIAGGIO_PCV_APT)

                legs.append({
                    "id": leg_id,
                    "Date": d,
                    "Direction": "PCV-APT",
                    "service_start": bus_dep_pcv,
                    "service_end": bus_arr_apt,
                    "flights": [g["Flight"] for g in group],
                    "ad_type": "P",
                    "routes": [g["Route"] for g in group],
                })
                leg_id += 1
                i = j

        # ARRIVI: APT → PCV (bus riporta passeggeri)
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
                bus_dep_apt = latest_dt + timedelta(minutes=25)
                bus_arr_pcv = bus_dep_apt + timedelta(minutes=TEMPO_VIAGGIO_APT_PCV)

                legs.append({
                    "id": leg_id,
                    "Date": d,
                    "Direction": "APT-PCV",
                    "service_start": bus_dep_apt,
                    "service_end": bus_arr_pcv,
                    "flights": [g["Flight"] for g in group],
                    "ad_type": "A",
                    "routes": [g["Route"] for g in group],
                })
                leg_id += 1
                i = j

    return sorted(legs, key=lambda x: x["service_start"])


def build_roundtrips_ensuring_coverage(legs: List[dict]) -> Tuple[List[dict], List[str]]:
    """
    Costruisce roundtrips (andata+ritorno) garantendo che OGNI leg sia coperto.
    
    Logica:
    1. Accoppia PCV-APT con il primo APT-PCV disponibile successivo
    2. Per legs spaiati, crea roundtrip con fuorilinea (corsa a vuoto)
    
    Returns:
        (roundtrips, warnings)
    """
    roundtrips: List[dict] = []
    warnings: List[str] = []
    rt_id = 0
    
    if not legs:
        return roundtrips, warnings

    for d in sorted({leg["Date"] for leg in legs}):
        day_legs = [leg for leg in legs if leg["Date"] == d]
        pcv_apt_legs = [leg for leg in day_legs if leg["Direction"] == "PCV-APT"]
        apt_pcv_legs = [leg for leg in day_legs if leg["Direction"] == "APT-PCV"]
        
        pcv_apt_legs = sorted(pcv_apt_legs, key=lambda x: x["service_start"])
        apt_pcv_legs = sorted(apt_pcv_legs, key=lambda x: x["service_start"])
        
        used_apt_pcv = [False] * len(apt_pcv_legs)
        used_pcv_apt = [False] * len(pcv_apt_legs)
        
        # FASE 1: Accoppia PCV-APT con APT-PCV
        for idx_out, out_leg in enumerate(pcv_apt_legs):
            best_match = None
            best_idx = None
            
            for idx_ret, ret_leg in enumerate(apt_pcv_legs):
                if used_apt_pcv[idx_ret]:
                    continue
                # Il ritorno deve partire DOPO che l'andata è arrivata
                if ret_leg["service_start"] >= out_leg["service_end"]:
                    best_match = ret_leg
                    best_idx = idx_ret
                    break  # Prendi il primo disponibile
            
            if best_match:
                used_pcv_apt[idx_out] = True
                used_apt_pcv[best_idx] = True
                
                # Calcola tempo inoperoso in aeroporto
                sosta_apt = (best_match["service_start"] - out_leg["service_end"]).total_seconds() / 60.0
                
                roundtrips.append({
                    "id": rt_id,
                    "Date": d,
                    "start": out_leg["service_start"],
                    "end": best_match["service_end"],
                    "out_leg": out_leg,
                    "ret_leg": best_match,
                    "sosta_apt_min": sosta_apt,
                    "is_fuorilinea": False,
                    "legs_covered": [out_leg["id"], best_match["id"]],
                })
                rt_id += 1
        
        # FASE 2: PCV-APT spaiati → crea fuorilinea per ritorno
        for idx_out, out_leg in enumerate(pcv_apt_legs):
            if used_pcv_apt[idx_out]:
                continue
            
            # Crea leg fittizio APT→PCV
            fake_start = out_leg["service_end"] + timedelta(minutes=10)
            fake_end = fake_start + timedelta(minutes=TEMPO_VIAGGIO_APT_PCV)
            
            fake_ret = {
                "id": -1,  # ID negativo = fuorilinea
                "Date": d,
                "Direction": "APT-PCV",
                "service_start": fake_start,
                "service_end": fake_end,
                "flights": ["FUORILINEA"],
                "ad_type": "X",
                "routes": ["---"],
            }
            
            roundtrips.append({
                "id": rt_id,
                "Date": d,
                "start": out_leg["service_start"],
                "end": fake_ret["service_end"],
                "out_leg": out_leg,
                "ret_leg": fake_ret,
                "sosta_apt_min": 10,
                "is_fuorilinea": True,
                "legs_covered": [out_leg["id"]],
            })
            rt_id += 1
            warnings.append(f"⚠️ Fuorilinea ritorno per voli {','.join(out_leg['flights'])} ({d.strftime('%d/%m')})")
        
        # FASE 3: APT-PCV spaiati → crea fuorilinea per andata
        for idx_ret, ret_leg in enumerate(apt_pcv_legs):
            if used_apt_pcv[idx_ret]:
                continue
            
            # Crea leg fittizio PCV→APT
            fake_end = ret_leg["service_start"] - timedelta(minutes=10)
            fake_start = fake_end - timedelta(minutes=TEMPO_VIAGGIO_PCV_APT)
            
            fake_out = {
                "id": -1,
                "Date": d,
                "Direction": "PCV-APT",
                "service_start": fake_start,
                "service_end": fake_end,
                "flights": ["FUORILINEA"],
                "ad_type": "X",
                "routes": ["---"],
            }
            
            roundtrips.append({
                "id": rt_id,
                "Date": d,
                "start": fake_out["service_start"],
                "end": ret_leg["service_end"],
                "out_leg": fake_out,
                "ret_leg": ret_leg,
                "sosta_apt_min": 10,
                "is_fuorilinea": True,
                "legs_covered": [ret_leg["id"]],
            })
            rt_id += 1
            warnings.append(f"⚠️ Fuorilinea andata per voli {','.join(ret_leg['flights'])} ({d.strftime('%d/%m')})")
    
    return sorted(roundtrips, key=lambda x: x["start"]), warnings


# =========================
# CALCOLO LAVORO E CLASSIFICAZIONE TURNI
# =========================

def calcola_metriche_turno(corse: List[dict]) -> Dict:
    """
    Calcola le metriche di un turno secondo normativa CCNL.
    
    Lavoro = Corsa + Pre/Post turno + Fuori linea + Inoperosità (12% con 5+5 pre/post)
    """
    if not corse:
        return {"nastro": 0, "lavoro": 0, "sosta_apt_max": 0, "gap_pcv_max": 0}
    
    corse = sorted(corse, key=lambda x: x["start"])
    
    # Nastro: da inizio prima corsa (con pre) a fine ultima (con post)
    shift_start = corse[0]["start"] - timedelta(minutes=PRE_TURNO)
    shift_end = corse[-1]["end"] + timedelta(minutes=POST_TURNO)
    nastro_min = (shift_end - shift_start).total_seconds() / 60.0
    
    # Tempo guida effettivo
    tempo_guida = 0
    for c in corse:
        # Andata
        tempo_guida += TEMPO_VIAGGIO_PCV_APT
        # Ritorno
        tempo_guida += TEMPO_VIAGGIO_APT_PCV
    
    # Inoperosità in aeroporto (solo se > 30 min)
    inoperosita = 0
    sosta_apt_max = 0
    for c in corse:
        sosta = c.get("sosta_apt_min", 0)
        sosta_apt_max = max(sosta_apt_max, sosta)
        if sosta > 30:
            # Pre e post sosta (5+5 min pagati 100%) + 12% del resto
            inoperosita += PRE_SOSTA_INOP + POST_SOSTA_INOP
            inoperosita += (sosta - PRE_SOSTA_INOP - POST_SOSTA_INOP) * PERCENTUALE_INOPEROSITA
    
    # Gap tra corse a PCV (fuori linea se < 40 min, pausa se >= 40 min)
    gaps_pcv = []
    fuori_linea = 0
    for i in range(len(corse) - 1):
        gap = (corse[i + 1]["start"] - corse[i]["end"]).total_seconds() / 60.0
        gaps_pcv.append(gap)
        if gap < 40:
            fuori_linea += gap  # Tempo considerato lavoro
    
    gap_pcv_max = max(gaps_pcv) if gaps_pcv else 0
    
    # Lavoro totale
    lavoro = tempo_guida + PRE_TURNO + POST_TURNO + inoperosita + fuori_linea
    
    return {
        "nastro": nastro_min,
        "lavoro": lavoro,
        "sosta_apt_max": sosta_apt_max,
        "gap_pcv_max": gap_pcv_max,
        "gaps_pcv": gaps_pcv,
        "shift_start": shift_start,
        "shift_end": shift_end,
    }


def classifica_turno(metriche: Dict) -> str:
    """
    Classifica il tipo di turno secondo normativa CCNL.
    Restituisce il tipo più appropriato.
    """
    nastro = metriche["nastro"]
    lavoro = metriche["lavoro"]
    sosta_apt_max = metriche["sosta_apt_max"]
    gap_pcv_max = metriche["gap_pcv_max"]
    shift_start = metriche.get("shift_start")
    shift_end = metriche.get("shift_end")
    
    # SUPPLEMENTO: nastro <= 3h
    if nastro <= TURNO_VINCOLI["Supplemento"]["nastro_max"]:
        return "Supplemento"
    
    # INTERO: nastro <= 8h, nastro ≈ lavoro, pausa retribuita >= 30 min
    if nastro <= TURNO_VINCOLI["Intero"]["nastro_max"]:
        # Il turno è "intero" se abbastanza saturo
        if abs(nastro - lavoro) <= nastro * 0.15:  # differenza max 15%
            return "Intero"
    
    # SOSTA INOPEROSA: nastro <= 9:15, sosta in aeroporto > 30 min
    if nastro <= TURNO_VINCOLI["Sosta Inoperosa"]["nastro_max"]:
        if sosta_apt_max > TURNO_VINCOLI["Sosta Inoperosa"]["sosta_apt_min"]:
            # Verifica vincoli fascia oraria
            if shift_start and shift_end:
                start_min = shift_start.hour * 60 + shift_start.minute
                end_min = shift_end.hour * 60 + shift_end.minute
                
                is_mattina = start_min < 12 * 60
                if is_mattina:
                    if end_min <= TURNO_VINCOLI["Sosta Inoperosa"]["mattina_fine_max"]:
                        return "Sosta Inoperosa"
                else:
                    if start_min >= TURNO_VINCOLI["Sosta Inoperosa"]["pomeriggio_inizio_min"]:
                        return "Sosta Inoperosa"
    
    # SEMIUNICO: nastro <= 9h, pausa a PCV tra 40 min e 3h
    if nastro <= TURNO_VINCOLI["Semiunico"]["nastro_max"]:
        if (TURNO_VINCOLI["Semiunico"]["pausa_pcv_min"] <= gap_pcv_max < 
            TURNO_VINCOLI["Semiunico"]["pausa_pcv_max"]):
            if lavoro <= TURNO_VINCOLI["Semiunico"]["lavoro_max"]:
                return "Semiunico"
    
    # SPEZZATO: nastro <= 10:30, pausa a PCV >= 3h
    if nastro <= TURNO_VINCOLI["Spezzato"]["nastro_max"]:
        if gap_pcv_max >= TURNO_VINCOLI["Spezzato"]["pausa_pcv_min"]:
            return "Spezzato"
    
    # Fallback: se nastro <= 9:15 con sosta aeroporto, è Sosta Inoperosa
    if nastro <= TURNO_VINCOLI["Sosta Inoperosa"]["nastro_max"] and sosta_apt_max > 0:
        return "Sosta Inoperosa"
    
    # Fallback finale
    if nastro <= 480:
        return "Intero"
    
    return "Altro"


# =========================
# ALGORITMO GREEDY OTTIMIZZATO
# =========================

def genera_turni_greedy(roundtrips: List[dict], weekday: str, d: date) -> List[dict]:
    """
    Genera turni usando algoritmo greedy con priorità CCNL.
    
    Strategia:
    1. Prova a costruire turni Interi (~7h) con 2-3 corse
    2. Poi Semiunici/Spezzati se ci sono gap lunghi
    3. Poi Sosta Inoperosa
    4. Infine Supplementi per corse residue
    
    GARANZIA: ogni roundtrip viene assegnato a esattamente un turno.
    """
    if not roundtrips:
        return []
    
    rts = sorted(roundtrips, key=lambda x: x["start"])
    n = len(rts)
    assigned = [False] * n
    turni: List[dict] = []
    
    def try_build_turno(start_idx: int, max_corse: int = 3) -> Optional[dict]:
        """Prova a costruire il miglior turno possibile partendo da start_idx."""
        if assigned[start_idx]:
            return None
        
        best_turno = None
        best_score = -1
        
        # Prova combinazioni di 1, 2, 3 corse
        for num_corse in range(1, min(max_corse + 1, n - start_idx + 1)):
            candidate_indices = [start_idx]
            candidate_corse = [rts[start_idx]]
            
            # Trova le prossime corse compatibili
            last_end = rts[start_idx]["end"]
            for j in range(start_idx + 1, n):
                if assigned[j]:
                    continue
                if len(candidate_indices) >= num_corse:
                    break
                
                # Deve essere sequenziale (non sovrapposto)
                if rts[j]["start"] >= last_end:
                    candidate_indices.append(j)
                    candidate_corse.append(rts[j])
                    last_end = rts[j]["end"]
            
            if len(candidate_corse) < num_corse:
                continue
            
            # Calcola metriche
            metriche = calcola_metriche_turno(candidate_corse)
            
            # Verifica vincoli di base
            if metriche["nastro"] > TURNO_VINCOLI["Spezzato"]["nastro_max"]:
                continue  # Nastro troppo lungo
            
            tipo = classifica_turno(metriche)
            
            # Calcola score (priorità + vicinanza a 7h)
            priorita = TURNO_VINCOLI.get(tipo, {}).get("priorita", 99)
            diff_7h = abs(metriche["lavoro"] - TARGET_WORK_MIN)
            
            # Score: priorità alta (numero basso) + lavoro vicino a 7h
            score = (5 - priorita) * 1000 - diff_7h
            
            if score > best_score:
                best_score = score
                best_turno = {
                    "indices": candidate_indices,
                    "corse": candidate_corse,
                    "metriche": metriche,
                    "tipo": tipo,
                }
        
        return best_turno
    
    # FASE 1: Costruisci turni ottimali
    for i in range(n):
        if assigned[i]:
            continue
        
        turno = try_build_turno(i, max_corse=3)
        
        if turno:
            for idx in turno["indices"]:
                assigned[idx] = True
            
            turni.append({
                "corse": turno["corse"],
                "metriche": turno["metriche"],
                "tipo_turno": turno["tipo"],
                "date": d,
                "weekday": weekday,
            })
    
    # FASE 2: Verifica che tutte le corse siano assegnate
    for i in range(n):
        if not assigned[i]:
            # Crea turno singolo (Supplemento)
            corsa = rts[i]
            metriche = calcola_metriche_turno([corsa])
            tipo = classifica_turno(metriche)
            
            turni.append({
                "corse": [corsa],
                "metriche": metriche,
                "tipo_turno": tipo,
                "date": d,
                "weekday": weekday,
            })
            assigned[i] = True
    
    return turni


def genera_turni_tutti_giorni(roundtrips: List[dict], weekday: str) -> Tuple[List[dict], Dict]:
    """
    Genera turni per tutte le date, poi unifica pattern simili.
    """
    # Raggruppa per data
    per_data = defaultdict(list)
    for rt in roundtrips:
        per_data[rt["Date"]].append(rt)
    
    tutti_turni = []
    for d in sorted(per_data.keys()):
        turni_d = genera_turni_greedy(per_data[d], weekday, d)
        tutti_turni.extend(turni_d)
    
    # Statistiche
    stats = {
        "total_roundtrips": len(roundtrips),
        "total_turni": len(tutti_turni),
        "per_data": {d: len(per_data[d]) for d in per_data},
    }
    
    return tutti_turni, stats


def assegna_codici_e_unifica(turni: List[dict], weekday: str) -> List[dict]:
    """
    Assegna codici turno e unifica turni con stesso pattern tra date diverse.
    """
    if not turni:
        return []
    
    prefix = DAY_PREFIX.get(weekday, weekday[:2].upper())
    
    # Raggruppa per "signature" (stesso orario e struttura)
    def get_signature(t):
        m = t["metriche"]
        return (
            t["tipo_turno"],
            m["shift_start"].time(),
            m["shift_end"].time(),
            len(t["corse"]),
        )
    
    per_signature = defaultdict(list)
    for t in turni:
        sig = get_signature(t)
        per_signature[sig].append(t)
    
    # Ordina signatures: AM poi PM, per orario
    am_sigs = [(sig, ts) for sig, ts in per_signature.items() 
               if ts[0]["metriche"]["shift_start"].hour < 12]
    pm_sigs = [(sig, ts) for sig, ts in per_signature.items() 
               if ts[0]["metriche"]["shift_start"].hour >= 12]
    
    am_sigs.sort(key=lambda x: x[0][1])  # ordina per orario
    pm_sigs.sort(key=lambda x: x[0][1])
    
    turni_unificati = []
    
    # Assegna codici AM (01-49)
    for idx, (sig, turni_list) in enumerate(am_sigs, start=1):
        sample = turni_list[0]
        n_corse = len(sample["corse"])
        suffix = "I" if n_corse >= 3 else ("P" if n_corse == 2 else "S")
        code = f"{prefix}{idx:02d}{suffix}"
        
        # Costruisci dettaglio corse
        detail_lines = []
        for c in sample["corse"]:
            out = c["out_leg"]
            ret = c["ret_leg"]
            detail_lines.append(
                f"{out['service_start'].strftime('%H:%M')}, Piazza Cavour, Aeroporto, "
                f"{out['service_end'].strftime('%H:%M')}, {','.join(out['flights'])}"
            )
            detail_lines.append(
                f"{ret['service_start'].strftime('%H:%M')}, Aeroporto, Piazza Cavour, "
                f"{ret['service_end'].strftime('%H:%M')}, {','.join(ret['flights'])}"
            )
        
        turni_unificati.append({
            "code": code,
            "weekday": weekday,
            "tipo_turno": sample["tipo_turno"],
            "shift_start": sample["metriche"]["shift_start"],
            "shift_end": sample["metriche"]["shift_end"],
            "nastro_min": sample["metriche"]["nastro"],
            "lavoro_min": sample["metriche"]["lavoro"],
            "n_corse": n_corse,
            "date_valide": sorted([t["date"] for t in turni_list]),
            "detail": "\n".join(detail_lines),
            "corse_ids": [c["id"] for c in sample["corse"]],
        })
    
    # Assegna codici PM (50-99)
    for idx, (sig, turni_list) in enumerate(pm_sigs, start=50):
        sample = turni_list[0]
        n_corse = len(sample["corse"])
        suffix = "I" if n_corse >= 3 else ("P" if n_corse == 2 else "S")
        code = f"{prefix}{idx:02d}{suffix}"
        
        detail_lines = []
        for c in sample["corse"]:
            out = c["out_leg"]
            ret = c["ret_leg"]
            detail_lines.append(
                f"{out['service_start'].strftime('%H:%M')}, Piazza Cavour, Aeroporto, "
                f"{out['service_end'].strftime('%H:%M')}, {','.join(out['flights'])}"
            )
            detail_lines.append(
                f"{ret['service_start'].strftime('%H:%M')}, Aeroporto, Piazza Cavour, "
                f"{ret['service_end'].strftime('%H:%M')}, {','.join(ret['flights'])}"
            )
        
        turni_unificati.append({
            "code": code,
            "weekday": weekday,
            "tipo_turno": sample["tipo_turno"],
            "shift_start": sample["metriche"]["shift_start"],
            "shift_end": sample["metriche"]["shift_end"],
            "nastro_min": sample["metriche"]["nastro"],
            "lavoro_min": sample["metriche"]["lavoro"],
            "n_corse": n_corse,
            "date_valide": sorted([t["date"] for t in turni_list]),
            "detail": "\n".join(detail_lines),
            "corse_ids": [c["id"] for c in sample["corse"]],
        })
    
    return sorted(turni_unificati, key=lambda x: x["shift_start"])


# =========================
# PIPELINE PRINCIPALE
# =========================

def genera_turni_completo(filtered_flights: pd.DataFrame, weekday: str) -> Tuple[List[dict], Dict, List[str]]:
    """
    Pipeline completa:
    1. Costruisce legs (singole tratte)
    2. Costruisce roundtrips (coppie andata-ritorno)
    3. Genera turni garantendo copertura 100%
    4. Unifica e assegna codici
    
    Returns:
        (turni_finali, statistiche, warnings)
    """
    warnings = []
    
    # Step 1: Costruisci tutti i legs
    legs = build_all_bus_legs(filtered_flights)
    if not legs:
        return [], {"error": "Nessun leg generato"}, warnings
    
    # Step 2: Costruisci roundtrips
    roundtrips, rt_warnings = build_roundtrips_ensuring_coverage(legs)
    warnings.extend(rt_warnings)
    
    if not roundtrips:
        return [], {"error": "Nessun roundtrip generato"}, warnings
    
    # Step 3: Genera turni
    turni_raw, stats = genera_turni_tutti_giorni(roundtrips, weekday)
    
    if not turni_raw:
        return [], {"error": "Nessun turno generato"}, warnings
    
    # Step 4: Unifica e assegna codici
    turni_finali = assegna_codici_e_unifica(turni_raw, weekday)
    
    # Calcola statistiche finali
    stats["legs_totali"] = len(legs)
    stats["roundtrips_totali"] = len(roundtrips)
    stats["turni_finali"] = len(turni_finali)
    stats["fuorilinea"] = len(rt_warnings)
    
    # Verifica copertura
    corse_coperte = set()
    for t in turni_raw:
        for c in t["corse"]:
            for leg_id in c.get("legs_covered", []):
                corse_coperte.add(leg_id)
    
    legs_reali = [l for l in legs if l["id"] >= 0]
    stats["legs_coperti"] = len(corse_coperte)
    stats["legs_reali"] = len(legs_reali)
    stats["copertura_pct"] = (len(corse_coperte) / len(legs_reali) * 100) if legs_reali else 100
    
    return turni_finali, stats, warnings


def genera_analisi(turni: List[dict], stats: Dict, weekday: str) -> str:
    """Genera analisi testuale dei turni."""
    if not turni:
        return "Nessun turno generato."
    
    lines = []
    lines.append(f"## 📊 Analisi Turni - {WEEKDAY_LABELS_IT.get(weekday, weekday)}")
    lines.append("")
    
    # Copertura
    lines.append(f"### Copertura")
    lines.append(f"- Legs totali: {stats.get('legs_totali', 'N/A')}")
    lines.append(f"- Roundtrips: {stats.get('roundtrips_totali', 'N/A')}")
    lines.append(f"- Copertura: {stats.get('copertura_pct', 0):.1f}%")
    if stats.get('fuorilinea', 0) > 0:
        lines.append(f"- ⚠️ Fuorilinea: {stats.get('fuorilinea')}")
    lines.append("")
    
    # Turni
    lines.append(f"### Turni Generati")
    lines.append(f"- Totale: {len(turni)}")
    
    per_tipo = defaultdict(int)
    for t in turni:
        per_tipo[t["tipo_turno"]] += 1
    
    for tipo in ["Intero", "Semiunico", "Spezzato", "Sosta Inoperosa", "Supplemento"]:
        if per_tipo[tipo] > 0:
            lines.append(f"- {tipo}: {per_tipo[tipo]}")
    lines.append("")
    
    # Metriche lavoro
    lavori = [t["lavoro_min"] for t in turni]
    nastri = [t["nastro_min"] for t in turni]
    
    if lavori:
        avg_lavoro = sum(lavori) / len(lavori)
        lines.append(f"### Metriche Lavoro")
        lines.append(f"- Media lavoro: {int(avg_lavoro // 60)}h {int(avg_lavoro % 60)}min")
        lines.append(f"- Min: {int(min(lavori) // 60)}h {int(min(lavori) % 60)}min")
        lines.append(f"- Max: {int(max(lavori) // 60)}h {int(max(lavori) % 60)}min")
        lines.append("")
    
    # Validità multi-giorno
    validita = [len(t["date_valide"]) for t in turni]
    n_date = len(set(d for t in turni for d in t["date_valide"]))
    
    if validita and n_date > 0:
        avg_val = sum(validita) / len(validita)
        tutti_giorni = sum(1 for v in validita if v == n_date)
        lines.append(f"### Validità Multi-Giorno")
        lines.append(f"- Date coperte: {n_date}")
        lines.append(f"- Validità media: {avg_val:.1f} giorni")
        lines.append(f"- Turni validi tutti i giorni: {tutti_giorni} ({tutti_giorni/len(turni)*100:.0f}%)")
        lines.append("")
    
    # Valutazione
    lines.append(f"### Valutazione")
    
    if per_tipo.get("Intero", 0) >= len(turni) * 0.4:
        lines.append("- ✅ Buona percentuale di turni Interi")
    else:
        lines.append("- ⚠️ Pochi turni Interi - verificare ottimizzazione")
    
    if lavori and sum(lavori) / len(lavori) >= 380 and sum(lavori) / len(lavori) <= 480:
        lines.append("- ✅ Lavoro medio in linea con target 7-8h")
    elif lavori and sum(lavori) / len(lavori) < 380:
        lines.append("- ⚠️ Lavoro medio sotto le 6:30h")
    
    if stats.get('fuorilinea', 0) == 0:
        lines.append("- ✅ Nessun fuorilinea necessario")
    else:
        lines.append(f"- ⚠️ {stats.get('fuorilinea')} corse fuorilinea - valutare alternative")
    
    return "\n".join(lines)


# =========================
# UI STREAMLIT
# =========================

def main():
    st.set_page_config(
        page_title="Turni Shuttle Aeroporto",
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
        .success-box {
            background: linear-gradient(135deg, rgba(34,197,94,0.15), rgba(22,163,74,0.15));
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid rgba(34,197,94,0.4);
        }
        .warning-box {
            background: linear-gradient(135deg, rgba(245,158,11,0.15), rgba(217,119,6,0.15));
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid rgba(245,158,11,0.4);
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🚌 Generatore Turni Shuttle Aeroporto")

    st.markdown("""
        <div class="info-card">
            <p><strong>Sistema di ottimizzazione turni con vincoli CCNL</strong></p>
            <ul style="margin-top:0.5rem; margin-bottom:0;">
                <li>✅ <strong>Copertura 100% garantita</strong> - tutte le corse vengono assegnate</li>
                <li>📋 Tipi turno: Intero, Semiunico, Spezzato, Sosta Inoperosa, Supplemento</li>
                <li>🎯 Target lavoro: ~7 ore per turno</li>
                <li>📅 Massimizzazione validità multi-giorno</li>
            </ul>
        </div>
    """, unsafe_allow_html=True)

    st.write("")

    uploaded_file = st.file_uploader("📄 Carica il PDF con gli orari dei voli", type=["pdf"])

    if uploaded_file is None:
        st.info("Carica il PDF per procedere.")
        
        with st.expander("📋 Normativa CCNL Implementata"):
            st.markdown("""
            | Tipo Turno | Nastro Max | Caratteristiche |
            |------------|------------|-----------------|
            | **Intero** | 8h | Nastro ≈ Lavoro, pausa retribuita ≥30min |
            | **Semiunico** | 9h | Pausa non retrib. a PCV 40min-3h |
            | **Spezzato** | 10h30 | Pausa non retrib. a PCV ≥3h |
            | **Sosta Inoperosa** | 9h15 | Sosta in aeroporto >30min (retrib. 12%) |
            | **Supplemento** | 3h | Turni brevi |
            
            **Calcolo Lavoro:**
            - Tempo guida (corsa)
            - Pre turno: 20 min
            - Post turno: 12 min
            - Inoperosità APT: 5+5 min (100%) + resto al 12%
            """)
        return

    with st.spinner("Parsing del PDF..."):
        flights_df = parse_pdf_to_flights_df(uploaded_file)

    if flights_df.empty:
        st.error("❌ Nessun volo PAX trovato.")
        return

    unique_days = sorted(flights_df["Date"].unique())

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.metric("Voli PAX", len(flights_df))
    with col2:
        st.metric("Giorni", len(unique_days))
    with col3:
        st.write(f"📆 **{unique_days[0].strftime('%d/%m/%Y')} – {unique_days[-1].strftime('%d/%m/%Y')}**")

    st.success("✅ PDF caricato")

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

    # Filtra voli
    filtered = flights_df[flights_df["Weekday"] == selected_weekday].copy()
    
    label_it = WEEKDAY_LABELS_IT.get(selected_weekday, selected_weekday)
    st.markdown(f"### 📅 {label_it}")
    st.write(f"Voli: {len(filtered)} su {filtered['Date'].nunique()} date")

    # GENERAZIONE TURNI
    st.markdown("---")
    
    if st.button("🚀 Genera Turni", type="primary", use_container_width=True):
        with st.spinner("Generazione turni in corso..."):
            turni, stats, warnings = genera_turni_completo(filtered, selected_weekday)
        
        if not turni:
            st.error(f"❌ Errore: {stats.get('error', 'Sconosciuto')}")
            return
        
        # COPERTURA
        copertura = stats.get('copertura_pct', 0)
        if copertura >= 100:
            st.markdown(f"""
                <div class="success-box">
                    ✅ <strong>COPERTURA 100%</strong> - Tutte le {stats.get('legs_totali', 0)} corse sono coperte
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="warning-box">
                    ⚠️ <strong>COPERTURA {copertura:.1f}%</strong> - {stats.get('legs_coperti', 0)}/{stats.get('legs_reali', 0)} corse coperte
                </div>
            """, unsafe_allow_html=True)
        
        # WARNINGS (fuorilinea)
        if warnings:
            with st.expander(f"⚠️ {len(warnings)} Corse Fuorilinea"):
                for w in warnings:
                    st.write(w)
        
        # STATISTICHE
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Turni", len(turni))
        col2.metric("Corse (legs)", stats.get('legs_totali', 0))
        col3.metric("Roundtrips", stats.get('roundtrips_totali', 0))
        col4.metric("Fuorilinea", stats.get('fuorilinea', 0))
        
        # MATRICE VALIDITÀ
        st.markdown("### 📊 Matrice Turni × Validità")
        
        dates_list = sorted(filtered["Date"].unique())
        
        rows = []
        for t in turni:
            nastro_h = int(t["nastro_min"] // 60)
            nastro_m = int(t["nastro_min"] % 60)
            lavoro_h = int(t["lavoro_min"] // 60)
            lavoro_m = int(t["lavoro_min"] % 60)
            
            row = {
                "Codice": t["code"],
                "Tipo": t["tipo_turno"],
                "Inizio": t["shift_start"].strftime("%H:%M"),
                "Fine": t["shift_end"].strftime("%H:%M"),
                "Nastro": f"{nastro_h}:{nastro_m:02d}",
                "Lavoro": f"{lavoro_h}:{lavoro_m:02d}",
                "Corse": t["n_corse"],
            }
            
            for d in dates_list:
                col_name = d.strftime("%d-%m")
                row[col_name] = "✅" if d in t["date_valide"] else ""
            
            rows.append(row)
        
        df_matrix = pd.DataFrame(rows)
        
        # Styling
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
        st.dataframe(styled, use_container_width=True, height=500)
        
        # Download
        csv = df_matrix.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Scarica CSV",
            data=csv,
            file_name=f"turni_{label_it.lower()}.csv",
            mime="text/csv",
        )
        
        # LEGENDA
        st.markdown("**Legenda:**")
        cols = st.columns(5)
        legend = [
            ("🟢 Intero", "≤8h"),
            ("🔵 Semiunico", "≤9h"),
            ("🟣 Spezzato", "≤10:30"),
            ("🟡 Sosta Inop.", "≤9:15"),
            ("⚪ Suppl.", "≤3h"),
        ]
        for col, (nome, desc) in zip(cols, legend):
            col.write(f"**{nome}** {desc}")
        
        # DETTAGLIO TURNO
        st.markdown("---")
        st.markdown("### 🔍 Dettaglio Turno")
        
        selected_code = st.selectbox("Seleziona turno", [t["code"] for t in turni])
        
        if selected_code:
            t_sel = next(t for t in turni if t["code"] == selected_code)
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Tipo", t_sel["tipo_turno"])
            col2.metric("Nastro", f"{int(t_sel['nastro_min']//60)}h {int(t_sel['nastro_min']%60)}min")
            col3.metric("Lavoro", f"{int(t_sel['lavoro_min']//60)}h {int(t_sel['lavoro_min']%60)}min")
            
            st.markdown("**Dettaglio corse:**")
            st.text(t_sel["detail"])
            
            st.markdown(f"**Validità:** {len(t_sel['date_valide'])} giorni")
            st.write(", ".join(d.strftime("%d/%m") for d in t_sel["date_valide"]))
        
        # ANALISI
        st.markdown("---")
        analisi = genera_analisi(turni, stats, selected_weekday)
        st.markdown(analisi)


if __name__ == "__main__":
    main()
