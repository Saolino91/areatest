# streamlit_app.py

"""
App Streamlit per:
1. Caricare un PDF con orari voli (qualsiasi mese, tipo Feb/Mar 2026).
2. Parsare i voli PAX, anche quando un giorno è spezzato su più tabelle / pagine.
3. Raggruppare per giorno della settimana.
4. Visualizzare una matrice voli × date con interfaccia curata e filtri.
5. Esportare la matrice in CSV.
6. Visualizzare un grafico a linee Arrivi/Partenze per giorno.
7. Generare turni guida CCNL con matrice di validità per tipo giorno.
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
# COSTANTI BASE
# =========================

WEEKDAY_ORDER = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

WEEKDAY_LABELS_IT = {
    "Mon": "Lunedì",
    "Tue": "Martedì",
    "Wed": "Mercoledì",
    "Thu": "Giovedì",
    "Fri": "Venerdì",
    "Sat": "Sabato",
    "Sun": "Domenica",
}

DAY_PATTERN = re.compile(
    r"^(Sun|Mon|Tue|Wed|Thu|Fri|Sat)\s+(\d{1,2})\s+([A-Za-z]{3})\s+(\d{4})$"
)

MONTH_MAP = {
    "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
    "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
}

# Prefisso italiano per giorno (prime 2 lettere, maiuscole)
DAY_PREFIX = {
    "Mon": "LU",
    "Tue": "MA",
    "Wed": "ME",
    "Thu": "GI",
    "Fri": "VE",
    "Sat": "SA",
    "Sun": "DO",
}

# =========================
# CONFIGURAZIONE CCNL / SHUTTLE
# =========================

TEMPO_VIAGGIO_PCV_APT = 40  # minuti
TEMPO_VIAGGIO_APT_PCV = 35  # minuti
PRE_TURNO = 20              # minuti pre-turno (10 pre + 10 trasferimento)
POST_TURNO = 12             # minuti post-turno (10 trasferimento + 2 post)
PRE_SOSTA_INOP = 5          # minuti pre-sosta inoperosa (pagati 100%)
POST_SOSTA_INOP = 5         # minuti post-sosta inoperosa (pagati 100%)
PERCENTUALE_INOPEROSITA = 0.12  # 12%

DEPARTURE_GROUP_DELTA_MIN = 20
ARRIVAL_GROUP_DELTA_MIN = 20

TARGET_WORK_MIN = 7 * 60  # 7 ore

TURNO_VINCOLI = {
    "Intero": {
        "nastro_max": 480,      # 8h
        "pausa_retrib_min": 30,
        "priorita": 1,
    },
    "Sosta Inoperosa": {
        "nastro_max": 555,      # 9h15
        "sosta_apt_min": 30,
        "mattina_fine_max": 15 * 60 + 15,
        "pomeriggio_inizio_min": 11 * 60 + 50,
        "priorita": 3,
    },
    "Semiunico": {
        "nastro_max": 540,      # 9h
        "lavoro_max": 500,      # 8h20
        "pausa_pcv_min": 40,
        "pausa_pcv_max": 180,
        "priorita": 2,
    },
    "Spezzato": {
        "nastro_max": 630,      # 10h30
        "pausa_pcv_min": 180,
        "priorita": 2,
    },
    "Supplemento": {
        "nastro_max": 180,      # 3h
        "priorita": 4,
    },
}


# =========================
# UTILITY
# =========================

def minutes_to_hhmm(m: float) -> str:
    if m is None:
        return ""
    m_int = int(round(m))
    h = m_int // 60
    mm = m_int % 60
    return f"{h:02d}:{mm:02d}"


def combine_date_time(d: date, time_str: str) -> Optional[datetime]:
    if not time_str:
        return None
    try:
        t = datetime.strptime(time_str, "%H:%M").time()
        return datetime.combine(d, t)
    except ValueError:
        return None


# =========================
# PARSING PDF VOLI
# =========================

def parse_pdf_to_flights_df(file_obj: io.BytesIO) -> pd.DataFrame:
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
# MATRICE VOLI (come prima)
# =========================

def compute_time_value(row: pd.Series) -> Optional[str]:
    ad = str(row.get("AD", "")).upper()
    if ad in ("A", "ARR", "ARRIVAL"):
        return row.get("ETA") or None
    if ad in ("P", "D", "DEP", "DEPT", "DEPARTURE"):
        return row.get("ETD") or None
    return None


def build_matrix_for_weekday(flights: pd.DataFrame, weekday: str) -> pd.DataFrame:
    if flights.empty:
        return pd.DataFrame()

    subset = flights[flights["Weekday"] == weekday].copy()
    if subset.empty:
        return pd.DataFrame()

    subset["TimeValue"] = subset.apply(compute_time_value, axis=1)
    subset = subset.dropna(subset=["TimeValue"])
    if subset.empty:
        return pd.DataFrame()

    matrix = subset.pivot_table(
        index=["Flight", "Route", "AD"],
        columns="Date",
        values="TimeValue",
        aggfunc="first",
    )

    matrix = matrix.reindex(sorted(matrix.columns), axis=1)
    matrix = matrix.reset_index()

    new_cols = []
    for c in matrix.columns:
        if isinstance(c, date):
            new_cols.append(c.strftime("%d-%m"))
        else:
            new_cols.append(c)
    matrix.columns = new_cols

    matrix = matrix.sort_values(by=["Flight", "Route", "AD"]).reset_index(drop=True)
    return matrix


def style_ad(val: str) -> str:
    if val == "P":
        return "color: #f97373;"
    if val == "A":
        return "color: #4ade80;"
    return ""


def style_time(row: pd.Series):
    ad = row.get("AD", None)
    color = None
    if ad == "P":
        color = "#f97373"
    elif ad == "A":
        color = "#4ade80"

    styles = []
    for col in row.index:
        if col in ("Codice Volo", "Aeroporto", "AD"):
            styles.append("")
            continue
        if pd.notna(row[col]) and row[col] != "" and color is not None:
            styles.append(f"color: {color};")
        else:
            styles.append("")
    return styles


# =========================
# FILTRI VOLI → INPUT PER TURNI
# =========================

def filter_flights_for_turns(
    flights_df: pd.DataFrame,
    weekday: str,
    flight_filter: str,
    selected_airports: List[str],
    ad_choice: str,
) -> pd.DataFrame:
    subset = flights_df[flights_df["Weekday"] == weekday].copy()

    if flight_filter:
        subset = subset[
            subset["Flight"].str.contains(flight_filter, case=False, na=False)
        ]

    if selected_airports:
        subset = subset[subset["Route"].isin(selected_airports)]

    if ad_choice == "Solo arrivi (A)":
        subset = subset[subset["AD"] == "A"]
    elif ad_choice == "Solo partenze (P)":
        subset = subset[subset["AD"] == "P"]

    return subset


# =========================
# COSTRUZIONE LEGS E ROUNDTRIP (COPERTURA 100%)
# =========================

def build_all_bus_legs(filtered_flights: pd.DataFrame) -> List[dict]:
    legs: List[dict] = []
    leg_id = 0

    if filtered_flights.empty:
        return legs

    for d in sorted(filtered_flights["Date"].unique()):
        day_rows = filtered_flights[filtered_flights["Date"] == d]

        # PARTENZE: PCV → APT
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

        # ARRIVI: APT → PCV
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

        # Fase 1: accoppia PCV-APT con APT-PCV
        for idx_out, out_leg in enumerate(pcv_apt_legs):
            best_match = None
            best_idx = None

            for idx_ret, ret_leg in enumerate(apt_pcv_legs):
                if used_apt_pcv[idx_ret]:
                    continue
                if ret_leg["service_start"] >= out_leg["service_end"]:
                    best_match = ret_leg
                    best_idx = idx_ret
                    break

            if best_match:
                used_pcv_apt[idx_out] = True
                used_apt_pcv[best_idx] = True

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

        # Fase 2: PCV-APT spaiati → fuorilinea per ritorno
        for idx_out, out_leg in enumerate(pcv_apt_legs):
            if used_pcv_apt[idx_out]:
                continue

            fake_start = out_leg["service_end"] + timedelta(minutes=10)
            fake_end = fake_start + timedelta(minutes=TEMPO_VIAGGIO_APT_PCV)

            fake_ret = {
                "id": -1,
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

        # Fase 3: APT-PCV spaiati → fuorilinea per andata
        for idx_ret, ret_leg in enumerate(apt_pcv_legs):
            if used_apt_pcv[idx_ret]:
                continue

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
# CALCOLO METRICHE TURNO + CLASSIFICAZIONE
# =========================

def calcola_metriche_turno(corse: List[dict]) -> Dict:
    if not corse:
        return {
            "nastro": 0,
            "lavoro": 0,
            "sosta_apt_max": 0,
            "gap_pcv_max": 0,
            "gaps_pcv": [],
            "shift_start": None,
            "shift_end": None,
        }

    corse = sorted(corse, key=lambda x: x["start"])

    shift_start = corse[0]["start"] - timedelta(minutes=PRE_TURNO)
    shift_end = corse[-1]["end"] + timedelta(minutes=POST_TURNO)
    nastro_min = (shift_end - shift_start).total_seconds() / 60.0

    tempo_guida = 0
    for c in corse:
        tempo_guida += TEMPO_VIAGGIO_PCV_APT
        tempo_guida += TEMPO_VIAGGIO_APT_PCV

    inoperosita = 0
    sosta_apt_max = 0
    for c in corse:
        sosta = c.get("sosta_apt_min", 0)
        sosta_apt_max = max(sosta_apt_max, sosta)
        if sosta > 30:
            inoperosita += PRE_SOSTA_INOP + POST_SOSTA_INOP
            inoperosita += (sosta - PRE_SOSTA_INOP - POST_SOSTA_INOP) * PERCENTUALE_INOPEROSITA

    gaps_pcv = []
    fuori_linea = 0
    for i in range(len(corse) - 1):
        gap = (corse[i + 1]["start"] - corse[i]["end"]).total_seconds() / 60.0
        gaps_pcv.append(gap)
        if gap < 40:
            fuori_linea += gap

    gap_pcv_max = max(gaps_pcv) if gaps_pcv else 0

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
    nastro = metriche["nastro"]
    lavoro = metriche["lavoro"]
    sosta_apt_max = metriche["sosta_apt_max"]
    gap_pcv_max = metriche["gap_pcv_max"]
    shift_start = metriche.get("shift_start")
    shift_end = metriche.get("shift_end")

    # Supplemento
    if nastro <= TURNO_VINCOLI["Supplemento"]["nastro_max"]:
        return "Supplemento"

    # Intero
    if nastro <= TURNO_VINCOLI["Intero"]["nastro_max"]:
        if abs(nastro - lavoro) <= nastro * 0.15:
            return "Intero"

    # Sosta Inoperosa
    if nastro <= TURNO_VINCOLI["Sosta Inoperosa"]["nastro_max"]:
        if sosta_apt_max > TURNO_VINCOLI["Sosta Inoperosa"]["sosta_apt_min"]:
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

    # Semiunico
    if nastro <= TURNO_VINCOLI["Semiunico"]["nastro_max"]:
        if (TURNO_VINCOLI["Semiunico"]["pausa_pcv_min"] <= gap_pcv_max <
                TURNO_VINCOLI["Semiunico"]["pausa_pcv_max"]):
            if lavoro <= TURNO_VINCOLI["Semiunico"]["lavoro_max"]:
                return "Semiunico"

    # Spezzato
    if nastro <= TURNO_VINCOLI["Spezzato"]["nastro_max"]:
        if gap_pcv_max >= TURNO_VINCOLI["Spezzato"]["pausa_pcv_min"]:
            return "Spezzato"

    if nastro <= TURNO_VINCOLI["Sosta Inoperosa"]["nastro_max"] and sosta_apt_max > 0:
        return "Sosta Inoperosa"

    if nastro <= 480:
        return "Intero"

    return "Altro"


# =========================
# GENERAZIONE TURNI GREEDY PER DATA
# =========================

def genera_turni_greedy(roundtrips: List[dict], weekday: str, d: date) -> List[dict]:
    if not roundtrips:
        return []

    rts = sorted(roundtrips, key=lambda x: x["start"])
    n = len(rts)
    assigned = [False] * n
    turni: List[dict] = []

    def try_build_turno(start_idx: int, max_corse: int = 3) -> Optional[dict]:
        if assigned[start_idx]:
            return None

        best_turno = None
        best_score = -1

        for num_corse in range(1, min(max_corse + 1, n - start_idx + 1)):
            candidate_indices = [start_idx]
            candidate_corse = [rts[start_idx]]

            last_end = rts[start_idx]["end"]
            for j in range(start_idx + 1, n):
                if assigned[j]:
                    continue
                if len(candidate_indices) >= num_corse:
                    break
                if rts[j]["start"] >= last_end:
                    candidate_indices.append(j)
                    candidate_corse.append(rts[j])
                    last_end = rts[j]["end"]

            if len(candidate_corse) < num_corse:
                continue

            metriche = calcola_metriche_turno(candidate_corse)
            if metriche["nastro"] > TURNO_VINCOLI["Spezzato"]["nastro_max"]:
                continue

            tipo = classifica_turno(metriche)
            priorita = TURNO_VINCOLI.get(tipo, {}).get("priorita", 99)
            diff_7h = abs(metriche["lavoro"] - TARGET_WORK_MIN)
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

    # corse residue → supplemento
    for i in range(n):
        if not assigned[i]:
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
    per_data = defaultdict(list)
    for rt in roundtrips:
        per_data[rt["Date"]].append(rt)

    tutti_turni = []
    for d in sorted(per_data.keys()):
        turni_d = genera_turni_greedy(per_data[d], weekday, d)
        tutti_turni.extend(turni_d)

    stats = {
        "total_roundtrips": len(roundtrips),
        "total_turni": len(tutti_turni),
        "per_data": {d: len(per_data[d]) for d in per_data},
    }

    return tutti_turni, stats


def assegna_codici_e_unifica(turni: List[dict], weekday: str) -> List[dict]:
    if not turni:
        return []

    prefix = DAY_PREFIX.get(weekday, weekday[:2].upper())

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

    am_sigs = [(sig, ts) for sig, ts in per_signature.items()
               if ts[0]["metriche"]["shift_start"].hour < 12]
    pm_sigs = [(sig, ts) for sig, ts in per_signature.items()
               if ts[0]["metriche"]["shift_start"].hour >= 12]

    am_sigs.sort(key=lambda x: x[0][1])
    pm_sigs.sort(key=lambda x: x[0][1])

    turni_unificati = []

    # AM: 01–49
    for idx, (sig, turni_list) in enumerate(am_sigs, start=1):
        sample = turni_list[0]
        n_corse = len(sample["corse"])
        suffix = "I" if n_corse >= 3 else ("P" if n_corse == 2 else "S")
        code = f"{prefix}{idx:02d}{suffix}"

        detail_lines = []
        for c in sample["corse"]:
            out = c["out_leg"]
            ret = c["ret_leg"]
            detail_lines.append(
                f"{out['service_start'].strftime('%H:%M')},Piazza Cavour, Aeroporto, "
                f"{out['service_end'].strftime('%H:%M')}, {','.join(out['flights'])}"
            )
            detail_lines.append(
                f"{ret['service_start'].strftime('%H:%M')},Aeroporto, Piazza Cavour, "
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
        })

    # PM: 50–99
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
                f"{out['service_start'].strftime('%H:%M')},Piazza Cavour, Aeroporto, "
                f"{out['service_end'].strftime('%H:%M')}, {','.join(out['flights'])}"
            )
            detail_lines.append(
                f"{ret['service_start'].strftime('%H:%M')},Aeroporto, Piazza Cavour, "
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
        })

    return sorted(turni_unificati, key=lambda x: x["shift_start"])


def genera_turni_completo(filtered_flights: pd.DataFrame, weekday: str) -> Tuple[List[dict], Dict, List[str]]:
    warnings: List[str] = []

    legs = build_all_bus_legs(filtered_flights)
    if not legs:
        return [], {"error": "Nessun leg generato"}, warnings

    roundtrips, rt_warnings = build_roundtrips_ensuring_coverage(legs)
    warnings.extend(rt_warnings)

    if not roundtrips:
        return [], {"error": "Nessun roundtrip generato"}, warnings

    turni_raw, stats = genera_turni_tutti_giorni(roundtrips, weekday)
    if not turni_raw:
        return [], {"error": "Nessun turno generato"}, warnings

    turni_finali = assegna_codici_e_unifica(turni_raw, weekday)

    stats["legs_totali"] = len(legs)
    stats["roundtrips_totali"] = len(roundtrips)
    stats["turni_finali"] = len(turni_finali)
    stats["fuorilinea"] = len(rt_warnings)

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


# =========================
# UI STREAMLIT
# =========================

def main():
    st.set_page_config(
        page_title="Flight Matrix",
        page_icon="✈️",
        layout="wide",
    )

    # CSS
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        h1 { text-align: center; }
        .info-card {
            background: rgba(15,23,42,0.9);
            padding: 1rem 1.2rem;
            border-radius: 0.9rem;
            border: 1px solid rgba(148,163,184,0.35);
        }
        .day-badge {
            display: inline-flex;
            align-items: center;
            padding: 0.25rem 0.75rem;
            border-radius: 999px;
            border: 1px solid rgba(148,163,184,0.8);
            background: rgba(15,23,42,0.9);
            font-size: 0.9rem;
            gap: 0.4rem;
        }
        .day-dot {
            width: 0.6rem;
            height: 0.6rem;
            border-radius: 999px;
            background: #38bdf8;
        }
        .legend-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.25rem 0.6rem;
            border-radius: 999px;
            border: 1px solid rgba(148,163,184,0.4);
            font-size: 0.8rem;
            margin-right: 0.4rem;
        }
        .legend-color-arr {
            width: 0.9rem;
            height: 0.35rem;
            border-radius: 999px;
            background: #4ade80;
        }
        .legend-color-dep {
            width: 0.9rem;
            height: 0.35rem;
            border-radius: 999px;
            background: #f97373;
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
        """,
        unsafe_allow_html=True,
    )

    st.title("✈️ Flight Matrix")

    with st.container():
        st.markdown(
            """
            <div class="info-card">
                <p>🛫🛬 <strong>Carica il PDF con gli orari dei voli</strong>.</p>
                <p style="margin-top:0.35rem;">L'app:</p>
                <ul style="margin-top:0.15rem;">
                    <li>considera <strong>solo voli passeggeri (PAX)</strong></li>
                    <li>esclude i voli <strong>CARGO</strong></li>
                    <li>raggruppa per <strong>giorno della settimana</strong></li>
                    <li>mostra una <strong>matrice</strong> voli × date</li>
                    <li>genera <strong>turni guida</strong> CCNL con matrice di validità</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")
    uploaded_file = st.file_uploader("Carica il PDF con gli orari dei voli", type=["pdf"])

    if uploaded_file is None:
        st.info("Carica il PDF per procedere.")
        return

    with st.spinner("Parsing del PDF in corso..."):
        flights_df = parse_pdf_to_flights_df(uploaded_file)

    if flights_df.empty:
        st.error("Non sono stati trovati voli PAX o la struttura del PDF non è riconosciuta.")
        return

    unique_days = sorted(flights_df["Date"].unique())
    num_days = len(unique_days)
    num_flights = len(flights_df)

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.metric("Voli PAX estratti", num_flights)
    with col2:
        st.metric("Giorni coperti", num_days)
    with col3:
        if num_days > 0:
            start = unique_days[0]
            end = unique_days[-1]
            st.write(
                f"📆 Periodo: **{start.strftime('%d/%m/%Y')} – {end.strftime('%d/%m/%Y')}**"
            )

    st.success("Parsing completato.")

    weekdays_present = sorted(
        flights_df["Weekday"].unique(),
        key=lambda x: WEEKDAY_ORDER.index(x),
    )

    st.sidebar.header("Filtro giorno")
    selected_weekday = st.sidebar.selectbox(
        "Seleziona giorno della settimana",
        options=weekdays_present,
        format_func=lambda x: WEEKDAY_LABELS_IT.get(x, x),
    )

    matrix_df = build_matrix_for_weekday(flights_df, selected_weekday)
    if matrix_df.empty:
        st.warning("Per il giorno selezionato non sono stati trovati voli PAX con orari validi.")
        return

    label_it = WEEKDAY_LABELS_IT.get(selected_weekday, selected_weekday)
    weekday_ops = flights_df[flights_df["Weekday"] == selected_weekday]
    weekday_flights_count = len(weekday_ops)
    weekday_dates_count = weekday_ops["Date"].nunique()

    st.markdown(
        f"""
        <div style="margin-top: 1.2rem; margin-bottom: 0.3rem;">
            <span class="day-badge">
                <span class="day-dot"></span>
                <span>{label_it}</span>
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"**Voli PAX per questo tipo di giorno:** {weekday_flights_count} "
        f"(su {weekday_dates_count} {label_it.lower()} nel periodo caricato)"
    )

    st.markdown(
        """
        <div style="margin-bottom: 0.6rem; margin-top: 0.2rem;">
            <span class="legend-pill">
                <span class="legend-color-arr"></span>
                <span>Arrivi (A)</span>
            </span>
            <span class="legend-pill">
                <span class="legend-color-dep"></span>
                <span>Partenze (P)</span>
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------- FILTRI MATRICE VOLI ---------
    with st.expander("Filtri matrice", expanded=False):
        flight_filter = st.text_input(
            "Filtra per Codice Volo (contiene)",
            key="flt_flight",
            placeholder="Es. FR, EN8, DX1702..."
        )

        airport_options = sorted(matrix_df["Route"].unique())
        selected_airports = st.multiselect(
            "Filtra per Aeroporto",
            options=airport_options,
            key="flt_airport",
        )

        ad_choice = st.radio(
            "Tipo di movimento",
            ["Arrivi e partenze", "Solo arrivi (A)", "Solo partenze (P)"],
            horizontal=True,
            key="flt_ad",
        )

    matrix_filtered = matrix_df.copy()

    if flight_filter:
        matrix_filtered = matrix_filtered[
            matrix_filtered["Flight"].str.contains(flight_filter, case=False, na=False)
        ]
    if selected_airports:
        matrix_filtered = matrix_filtered[
            matrix_filtered["Route"].isin(selected_airports)
        ]
    if ad_choice == "Solo arrivi (A)":
        matrix_filtered = matrix_filtered[matrix_filtered["AD"] == "A"]
    elif ad_choice == "Solo partenze (P)":
        matrix_filtered = matrix_filtered[matrix_filtered["AD"] == "P"]

    if matrix_filtered.empty:
        st.warning("Nessun volo corrisponde ai filtri impostati.")
        return

    display_df = matrix_filtered.rename(
        columns={"Flight": "Codice Volo", "Route": "Aeroporto"}
    )

    if "AD" in display_df.columns:
        styled_df = (
            display_df
            .style
            .apply(style_time, axis=1)
            .applymap(style_ad, subset=["AD"])
        )
    else:
        styled_df = display_df.style

    st.dataframe(styled_df, use_container_width=True, height=650)

    csv_buffer = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Scarica matrice in CSV",
        data=csv_buffer,
        file_name=f"flight_matrix_{label_it.lower()}.csv",
        mime="text/csv",
    )

    # --------- TURNI GUIDA + MATRICE VALIDITÀ ---------
    st.markdown("### Turni guida e matrice di validità")

    if st.button("Genera turni guida per questo tipo di giorno"):
        flights_for_turns = filter_flights_for_turns(
            flights_df,
            selected_weekday,
            flight_filter,
            selected_airports,
            ad_choice,
        )

        if flights_for_turns.empty:
            st.warning("Nessun volo disponibile per generare i turni guida con i filtri correnti.")
        else:
            with st.spinner("Generazione turni in corso..."):
                turni, stats, warnings = genera_turni_completo(flights_for_turns, selected_weekday)

            if not turni:
                st.error(f"Errore generazione turni: {stats.get('error', 'sconosciuto')}")
            else:
                copertura = stats.get("copertura_pct", 0)
                if copertura >= 100:
                    st.markdown(
                        f"""
                        <div class="success-box">
                            ✅ <strong>COPERTURA 100%</strong> - Tutte le corse reali sono coperte
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f"""
                        <div class="warning-box">
                            ⚠️ <strong>Copertura {copertura:.1f}%</strong> - "
                            "{stats.get('legs_coperti', 0)}/{stats.get('legs_reali', 0)} corse reali coperte
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                if warnings:
                    with st.expander(f"⚠️ {len(warnings)} corse fuorilinea generate"):
                        for w in warnings:
                            st.write(w)

                # Matrice validità (sostituisce tabella turni)
                st.markdown("#### Matrice di validità dei turni per data")

                dates_for_matrix = sorted(flights_for_turns["Date"].unique())
                rows = []
                for t in turni:
                    row = {
                        "Codice turno": t["code"],
                        "Tipo turno": t["tipo_turno"],
                        "Inizio": t["shift_start"].strftime("%H:%M"),
                        "Fine": t["shift_end"].strftime("%H:%M"),
                        "Lavoro": minutes_to_hhmm(t["lavoro_min"]),
                    }
                    for d in dates_for_matrix:
                        col_name = d.strftime("%d-%m")
                        row[col_name] = "✅" if d in t["date_valide"] else ""
                    rows.append(row)

                df_matrix = pd.DataFrame(rows)
                st.dataframe(df_matrix, use_container_width=True)

                csv_validita = df_matrix.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Scarica matrice validità turni in CSV",
                    data=csv_validita,
                    file_name=f"matrice_validita_turni_{label_it.lower()}.csv",
                    mime="text/csv",
                )

                # Dettaglio turno
                st.markdown("#### Dettaglio corse per turno")
                selected_code = st.selectbox(
                    "Seleziona un turno per visualizzare il dettaglio delle corse",
                    options=[t["code"] for t in turni],
                )

                if selected_code:
                    t_sel = next(t for t in turni if t["code"] == selected_code)
                    st.markdown(
                        f"**Turno {selected_code} – elenco corse (una corsa = andata + ritorno):**"
                    )
                    st.text(t_sel["detail"])

    # --------- GRAFICO ARRIVI/PARTENZE ---------
    st.markdown("### Andamento giornaliero Arrivi / Partenze")

    chart_df = flights_df.copy()

    def map_dir(ad: str) -> Optional[str]:
        if ad == "A":
            return "Arrivi"
        if ad in ("P", "D", "DEP", "DEPT", "DEPARTURE"):
            return "Partenze"
        return None

    chart_df["Dir"] = chart_df["AD"].map(map_dir)
    chart_df = chart_df.dropna(subset=["Dir"])

    if not chart_df.empty:
        daily_counts = (
            chart_df.groupby(["Date", "Dir"])["Flight"]
            .count()
            .unstack("Dir")
            .fillna(0)
            .sort_index()
        )
        for col in ["Arrivi", "Partenze"]:
            if col not in daily_counts.columns:
                daily_counts[col] = 0
        st.line_chart(daily_counts[["Arrivi", "Partenze"]])
    else:
        st.info("Nessun dato disponibile per costruire il grafico Arrivi/Partenze.")


if __name__ == "__main__":
    main()
