# streamlit_app.py

"""
App Streamlit per ottimizzazione turni shuttle aeroporto con vincoli CCNL Conerobus
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
    st.warning("⚠️ OR-Tools non installato. Usa: pip install ortools")


# =========================
# Costanti
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

DAY_PREFIX = {
    "Mon": "LU", "Tue": "MA", "Wed": "ME", "Thu": "GI",
    "Fri": "VE", "Sat": "SA", "Sun": "DO",
}

DEPARTURE_GROUP_DELTA_MIN = 20
ARRIVAL_GROUP_DELTA_MIN = 20

# Target ore lavoro per turni standard
TARGET_WORK_MIN = 7 * 60

# Vincoli CCNL
MIN_WORK_MIN = 195  # 3h15 (tranne supplementi)
MAX_NASTRO_ABSOLUTE = 630  # 10h30 (spezzato è il massimo)

# Priorità tipologie turno
SHIFT_TYPE_PRIORITY = {
    "Intero": 5,
    "Semiunico": 4,
    "Spezzato": 4,
    "Sosta Inoperosa": 3,
    "Part-time": 2,
    "Supplemento": 1,
}


# =========================
# PARSING PDF (invariato)
# =========================

def parse_pdf_to_flights_df(file_obj: io.BytesIO) -> pd.DataFrame:
    """Parser PDF voli - invariato"""
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
# COSTRUZIONE MATRICE (invariato)
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


# =========================
# STYLING (invariato)
# =========================

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
# SUPPORTO TURNI (invariato)
# =========================

def combine_date_time(d: date, time_str: str) -> Optional[datetime]:
    if not time_str:
        return None
    try:
        t = datetime.strptime(time_str, "%H:%M").time()
        return datetime.combine(d, t)
    except ValueError:
        return None


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


def build_bus_trips_from_flights(filtered_flights: pd.DataFrame) -> List[dict]:
    """Costruisce legs bus - invariato"""
    trips: List[dict] = []
    if filtered_flights.empty:
        return trips

    for d in sorted(filtered_flights["Date"].unique()):
        day_rows = filtered_flights[filtered_flights["Date"] == d]

        # PARTENZE
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
                bus_dep_pcv = bus_arr_apt - timedelta(minutes=40)

                trips.append({
                    "Date": d,
                    "Direction": "PCV-APT",
                    "service_start": bus_dep_pcv,
                    "service_end": bus_arr_apt,
                    "flights": [g["Flight"] for g in group],
                    "ad_type": "P",
                    "routes": [g["Route"] for g in group],
                })
                i = j

        # ARRIVI
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
                bus_arr_pcv = bus_dep_apt + timedelta(minutes=35)

                trips.append({
                    "Date": d,
                    "Direction": "APT-PCV",
                    "service_start": bus_dep_apt,
                    "service_end": bus_arr_pcv,
                    "flights": [g["Flight"] for g in group],
                    "ad_type": "A",
                    "routes": [g["Route"] for g in group],
                })
                i = j

    trips = sorted(trips, key=lambda t: t["service_start"])
    return trips


def build_roundtrips_from_trips(trips: List[dict]) -> List[dict]:
    """Costruisce roundtrips - invariato"""
    roundtrips: List[dict] = []
    if not trips:
        return roundtrips

    for d in sorted({t["Date"] for t in trips}):
        day_trips = [t for t in trips if t["Date"] == d]
        dep_legs = [t for t in day_trips if t["Direction"] == "PCV-APT"]
        arr_legs = [t for t in day_trips if t["Direction"] == "APT-PCV"]

        dep_legs = sorted(dep_legs, key=lambda x: x["service_start"])
        arr_legs = sorted(arr_legs, key=lambda x: x["service_start"])

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

            roundtrips.append({
                "Date": d,
                "start": dep["service_start"],
                "end": arr["service_end"],
                "dep_leg": dep,
                "arr_leg": arr,
            })

    roundtrips = sorted(roundtrips, key=lambda r: r["start"])
    return roundtrips


# =========================
# CALCOLO LAVORO E CLASSIFICAZIONE CORRETTI
# =========================

def calculate_work_minutes(rt_list: List[dict]) -> float:
    """
    Calcola lavoro effettivo secondo normativa:
    - Corse (tempo guida)
    - Pre turno (10 min)
    - Post turno (2 min)
    - Fuori linea (20 min per corsa: 10 dep→PCV + 10 PCV→dep)
    - Inoperosità: 12% tempo fermo aeroporto (se > 30') + 5' pre + 5' post
    """
    if not rt_list:
        return 0.0
    
    # Tempo corse
    work_corse = sum((rt["end"] - rt["start"]).total_seconds() / 60.0 for rt in rt_list)
    
    # Pre e post
    pre_post = 10 + 2
    
    # Fuori linea
    fuori_linea = len(rt_list) * 20
    
    # Inoperosità
    inoperosita_min = 0
    for rt in rt_list:
        dep_end = rt["dep_leg"]["service_end"]
        arr_start = rt["arr_leg"]["service_start"]
        sosta_apt = (arr_start - dep_end).total_seconds() / 60.0
        
        if sosta_apt > 30:
            inoperosita_min += sosta_apt * 0.12 + 10  # 12% + 5' pre + 5' post
    
    return work_corse + pre_post + fuori_linea + inoperosita_min


def calculate_nastro_minutes(rt_list: List[dict]) -> float:
    """Calcola nastro turno"""
    if not rt_list:
        return 0.0
    
    first = rt_list[0]
    last = rt_list[-1]
    shift_start = first["start"] - timedelta(minutes=20)
    shift_end = last["end"] + timedelta(minutes=12)
    
    return (shift_end - shift_start).total_seconds() / 60.0


def classify_shift_type(rt_list: List[dict], nastro_min: float, work_min: float) -> Optional[str]:
    """
    Classifica turno SENZA "Altro" - tutti i turni devono rientrare in una categoria.
    
    Ordine di verifica (dal più specifico al più generico):
    1. SUPPLEMENTO: nastro ≤ 3h
    2. PART-TIME: nastro < 6h
    3. SPEZZATO: nastro ≤ 10h30, interruzione PCV > 3h
    4. SEMIUNICO: nastro ≤ 9h, interruzione PCV 40'-3h
    5. SOSTA INOPEROSA: nastro ≤ 9h15, sosta aeroporto > 30'
    6. INTERO: nastro ≤ 8h, sosta PCV ≥ 30', nastro ≈ lavoro
    
    Se non rientra in nessuna: è un turno INVALIDO (None)
    """
    if not rt_list:
        return None
    
    # VINCOLO GENERALE: lavoro minimo 3h15 (tranne supplementi e part-time)
    if nastro_min > 360 and work_min < MIN_WORK_MIN:
        return None  # Turno invalido
    
    # Calcola gap
    pcv_gaps = []
    for i in range(len(rt_list) - 1):
        end_i = rt_list[i]["end"]
        start_j = rt_list[i + 1]["start"]
        gap = (start_j - end_i).total_seconds() / 60.0
        pcv_gaps.append(gap)
    
    apt_gaps = []
    for rt in rt_list:
        dep_end = rt["dep_leg"]["service_end"]
        arr_start = rt["arr_leg"]["service_start"]
        gap = (arr_start - dep_end).total_seconds() / 60.0
        apt_gaps.append(gap)
    
    max_pcv_gap = max(pcv_gaps) if pcv_gaps else 0
    max_apt_gap = max(apt_gaps) if apt_gaps else 0
    has_pause_30 = any(g >= 30 for g in pcv_gaps)
    
    # 1. SUPPLEMENTO: max 3h
    if nastro_min <= 180:
        return "Supplemento"
    
    # 2. PART-TIME: nastro < 6h
    if nastro_min < 360:
        return "Part-time"
    
    # 3. SPEZZATO: nastro ≤ 10h30, interruzione PCV > 3h
    if nastro_min <= 630 and max_pcv_gap >= 180:
        return "Spezzato"
    
    # 4. SEMIUNICO: nastro ≤ 9h, interruzione PCV 40'-3h
    if nastro_min <= 540 and 40 <= max_pcv_gap < 180:
        return "Semiunico"
    
    # 5. SOSTA INOPEROSA: nastro ≤ 9h15, sosta aeroporto > 30'
    if nastro_min <= 555 and max_apt_gap > 30:
        shift_start = rt_list[0]["start"] - timedelta(minutes=20)
        shift_end = rt_list[-1]["end"] + timedelta(minutes=12)
        
        is_morning = shift_start.hour < 12
        
        if is_morning:
            if shift_end.hour < 15 or (shift_end.hour == 15 and shift_end.minute <= 15):
                return "Sosta Inoperosa"
        else:
            if shift_start.hour > 11 or (shift_start.hour == 11 and shift_start.minute >= 50):
                return "Sosta Inoperosa"
    
    # 6. INTERO: nastro ≤ 8h, sosta PCV ≥ 30', nastro ≈ lavoro
    if nastro_min <= 480 and has_pause_30:
        # Nastro deve essere vicino a lavoro (tolleranza 10%)
        if abs(nastro_min - work_min) <= nastro_min * 0.10:
            return "Intero"
    
    # Se arriviamo qui, il turno non rientra in nessuna categoria valida
    return None


# =========================
# OTTIMIZZAZIONE CP-SAT CON VINCOLI RINFORZATI
# =========================

def optimize_shifts_with_priorities(
    roundtrips: List[dict], 
    weekday: str, 
    max_shifts: int = 30, 
    time_limit_sec: int = 60
) -> Tuple[List[dict], dict]:
    """Ottimizza turni con vincoli CCNL completi"""
    
    if not ORTOOLS_AVAILABLE or not roundtrips:
        return [], {"error": "OR-Tools non disponibile o nessuna corsa"}
    
    # Separa per data
    dates_set = {rt["Date"] for rt in roundtrips}
    if len(dates_set) > 1:
        all_shifts = []
        all_stats = []
        
        for d in sorted(dates_set):
            rts_d = [rt for rt in roundtrips if rt["Date"] == d]
            shifts_d, stats_d = optimize_shifts_single_date(rts_d, weekday, max_shifts, time_limit_sec)
            all_shifts.extend(shifts_d)
            all_stats.append(stats_d)
        
        combined_stats = {
            "status": "MULTI_DATE",
            "dates": len(dates_set),
            "total_shifts": len(all_shifts),
            "avg_solve_time": sum(s.get("solve_time", 0) for s in all_stats) / len(all_stats) if all_stats else 0,
        }
        
        return all_shifts, combined_stats
    else:
        return optimize_shifts_single_date(roundtrips, weekday, max_shifts, time_limit_sec)


def optimize_shifts_single_date(
    roundtrips: List[dict], 
    weekday: str, 
    max_shifts: int = 30, 
    time_limit_sec: int = 60
) -> Tuple[List[dict], dict]:
    """Ottimizzazione per singola data con vincoli CCNL rinforzati"""
    
    model = cp_model.CpModel()
    n = len(roundtrips)
    
    # Pre-calcola nastro per ogni combinazione possibile
    nastro_lookup = {}
    for i in range(n):
        for j in range(i, min(i + 3, n)):  # max 3 corse
            rt_subset = roundtrips[i:j+1]
            nastro = calculate_nastro_minutes(rt_subset)
            nastro_lookup[(i, j)] = nastro
    
    # VARIABILI
    shift_used = [model.NewBoolVar(f'shift_{j}') for j in range(max_shifts)]
    assignment = {}
    for i in range(n):
        for j in range(max_shifts):
            assignment[i, j] = model.NewBoolVar(f'x_{i}_{j}')
    
    # VINCOLO 1: ogni corsa a esattamente 1 turno
    for i in range(n):
        model.Add(sum(assignment[i, j] for j in range(max_shifts)) == 1)
    
    # VINCOLO 2: turno usato ↔ ha almeno 1 corsa
    for j in range(max_shifts):
        corse_in_turno = sum(assignment[i, j] for i in range(n))
        model.Add(corse_in_turno >= shift_used[j])
        model.Add(corse_in_turno <= shift_used[j] * n)
    
    # VINCOLO 3: max 3 corse per turno
    for j in range(max_shifts):
        model.Add(sum(assignment[i, j] for i in range(n)) <= 3)
    
    # VINCOLO 4: nessuna sovrapposizione + sequenzialità
    for j in range(max_shifts):
        for i1 in range(n):
            for i2 in range(i1 + 1, n):
                rt1, rt2 = roundtrips[i1], roundtrips[i2]
                
                # Non possono stare insieme se non sono sequenziali
                if rt1["end"] > rt2["start"]:
                    model.Add(assignment[i1, j] + assignment[i2, j] <= 1)
    
    # VINCOLO 5: nastro massimo 10h30 (MAX_NASTRO_ABSOLUTE)
    # Questo è un vincolo HARD: nessun turno può superarlo
    for j in range(max_shifts):
        # Se il turno ha 3 corse consecutive che superano 10h30, scarta
        for i in range(n - 2):
            if (i, i+2) in nastro_lookup:
                if nastro_lookup[(i, i+2)] > MAX_NASTRO_ABSOLUTE:
                    model.Add(assignment[i, j] + assignment[i+1, j] + assignment[i+2, j] <= 2)
        
        # Se 2 corse consecutive superano 10h30, scarta
        for i in range(n - 1):
            if (i, i+1) in nastro_lookup:
                if nastro_lookup[(i, i+1)] > MAX_NASTRO_ABSOLUTE:
                    model.Add(assignment[i, j] + assignment[i+1, j] <= 1)
    
    # OBIETTIVO: minimizza numero turni
    model.Minimize(sum(shift_used))
    
    # RISOLVI
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_sec
    solver.parameters.log_search_progress = False
    
    status = solver.Solve(model)
    
    # ESTRAI E VALIDA SOLUZIONE
    stats = {
        "status": solver.StatusName(status),
        "solve_time": solver.WallTime(),
        "optimal": status == cp_model.OPTIMAL,
        "num_shifts": 0,
        "num_roundtrips": n,
        "invalid_shifts": 0,
    }
    
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        shifts_list = []
        the_date = roundtrips[0]["Date"]
        
        for j in range(max_shifts):
            if solver.Value(shift_used[j]):
                rt_list = []
                for i in range(n):
                    if solver.Value(assignment[i, j]):
                        rt_list.append(roundtrips[i])
                
                if not rt_list:
                    continue
                
                rt_list = sorted(rt_list, key=lambda x: x["start"])
                
                # Calcola metriche
                nastro_min = calculate_nastro_minutes(rt_list)
                work_min = calculate_work_minutes(rt_list)
                
                # VALIDAZIONE: scarta turni che superano i limiti
                if nastro_min > MAX_NASTRO_ABSOLUTE:
                    stats["invalid_shifts"] += 1
                    continue
                
                tipo_turno = classify_shift_type(rt_list, nastro_min, work_min)
                
                # VALIDAZIONE: se non ha tipo valido, scarta
                if tipo_turno is None:
                    stats["invalid_shifts"] += 1
                    continue
                
                first = rt_list[0]
                last = rt_list[-1]
                shift_start_dt = first["start"] - timedelta(minutes=20)
                shift_end_dt = last["end"] + timedelta(minutes=12)
                corses = len(rt_list)
                
                # Dettagli corse
                detail_lines = []
                for rt in rt_list:
                    dep = rt["dep_leg"]
                    arr = rt["arr_leg"]
                    
                    ora_p = dep["service_start"].strftime("%H:%M")
                    ora_a = dep["service_end"].strftime("%H:%M")
                    codici_dep = ",".join(dep["flights"])
                    detail_lines.append(f"{ora_p}, Piazza Cavour, Aeroporto, {ora_a}, {codici_dep}")
                    
                    ora_p2 = arr["service_start"].strftime("%H:%M")
                    ora_a2 = arr["service_end"].strftime("%H:%M")
                    codici_arr = ",".join(arr["flights"])
                    detail_lines.append(f"{ora_p2}, Aeroporto, Piazza Cavour, {ora_a2}, {codici_arr}")
                
                detail = "\n".join(detail_lines)
                
                shifts_list.append({
                    "weekday": weekday,
                    "date": the_date,
                    "shift_start_dt": shift_start_dt,
                    "shift_end_dt": shift_end_dt,
                    "nastro_min": nastro_min,
                    "work_min": work_min,
                    "corses": corses,
                    "detail": detail,
                    "tipo_turno": tipo_turno,
                })
        
        shifts_list = sorted(shifts_list, key=lambda s: s["shift_start_dt"])
        stats["num_shifts"] = len(shifts_list)
        
        return shifts_list, stats
    
    return [], stats


def assign_shift_codes_across_dates(shifts: List[dict], weekday: str) -> List[dict]:
    """Assegna codici turno STABILI"""
    if not shifts:
        return shifts

    prefix = DAY_PREFIX.get(weekday, weekday[:2].upper())

    patterns = defaultdict(list)
    for s in shifts:
        key = (
            s["tipo_turno"],
            s["shift_start_dt"].time(),
            s["shift_end_dt"].time(),
            s["corses"],
        )
        patterns[key].append(s)

    pattern_items = []
    for key, plist in patterns.items():
        sample = sorted(plist, key=lambda x: x["shift_start_dt"])[0]
        pattern_items.append((key, sample))

    am = [(k, s) for k, s in pattern_items if s["shift_start_dt"].hour < 12]
    pm = [(k, s) for k, s in pattern_items if s["shift_start_dt"].hour >= 12]

    def sort_key(ks):
        key, sample = ks
        tipo = sample["tipo_turno"]
        priority = SHIFT_TYPE_PRIORITY.get(tipo, 0)
        return (-priority, sample["shift_start_dt"])

    am.sort(key=sort_key)
    pm.sort(key=sort_key)

    code_by_key = {}

    for idx, (key, sample) in enumerate(am, start=1):
        num = min(idx, 49)
        corses = sample["corses"]
        suffix = "I" if corses >= 3 else ("P" if corses == 2 else "S")
        code_by_key[key] = f"{prefix}{num:02d}{suffix}"

    for idx, (key, sample) in enumerate(pm, start=50):
        num = min(idx, 99)
        corses = sample["corses"]
        suffix = "I" if corses >= 3 else ("P" if corses == 2 else "S")
        code_by_key[key] = f"{prefix}{num:02d}{suffix}"

    for s in shifts:
        key = (
            s["tipo_turno"],
            s["shift_start_dt"].time(),
            s["shift_end_dt"].time(),
            s["corses"],
        )
        s["code"] = code_by_key[key]

    return shifts


def generate_driver_shifts_optimized(
    filtered_flights: pd.DataFrame, 
    weekday: str
) -> Tuple[List[dict], dict]:
    """Pipeline completa"""
    
    trips = build_bus_trips_from_flights(filtered_flights)
    if not trips:
        return [], {}

    roundtrips = build_roundtrips_from_trips(trips)
    if not roundtrips:
        return [], {}

    shifts, stats = optimize_shifts_with_priorities(roundtrips, weekday)
    
    if shifts:
        shifts = assign_shift_codes_across_dates(shifts, weekday)

    return shifts, stats


def analyze_shifts(shifts: List[dict], weekday: str) -> Dict:
    """Analisi automatica"""
    if not shifts:
        return {}
    
    type_counts = defaultdict(int)
    for s in shifts:
        type_counts[s["tipo_turno"]] += 1
    
    work_values = [s["work_min"] for s in shifts]
    nastro_values = [s["nastro_min"] for s in shifts]
    
    avg_work = sum(work_values) / len(work_values) if work_values else 0
    avg_nastro = sum(nastro_values) / len(nastro_values) if nastro_values else 0
    
    near_7h = sum(1 for w in work_values if 400 <= w <= 440)
    
    unique_dates = {s["date"] for s in shifts}
    unique_codes = {s["code"] for s in shifts}
    
    corse_dist = defaultdict(int)
    for s in shifts:
        corse_dist[s["corses"]] += 1
    
    # Verifica vincoli
    violations = []
    for s in shifts:
        if s["nastro_min"] > MAX_NASTRO_ABSOLUTE:
            violations.append(f"Turno {s['code']}: nastro {s['nastro_min']:.0f}min > max {MAX_NASTRO_ABSOLUTE}min")
        if s["tipo_turno"] not in ["Supplemento", "Part-time"] and s["work_min"] < MIN_WORK_MIN:
            violations.append(f"Turno {s['code']}: lavoro {s['work_min']:.0f}min < min {MIN_WORK_MIN}min")
    
    return {
        "total_shifts": len(shifts),
        "unique_patterns": len(unique_codes),
        "dates_covered": len(unique_dates),
        "type_distribution": dict(type_counts),
        "avg_work_hours": avg_work / 60,
        "avg_nastro_hours": avg_nastro / 60,
        "shifts_near_7h": near_7h,
        "shifts_near_7h_pct": (near_7h / len(shifts) * 100) if shifts else 0,
        "corse_distribution": dict(corse_dist),
        "violations": violations,
    }


# =========================
# UI STREAMLIT (layout ottimizzato)
# =========================

def main():
    st.set_page_config(
        page_title="Flight Matrix - Ottimizzazione Turni",
        page_icon="✈️",
        layout="wide",
    )

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
        .analysis-box {
            background: linear-gradient(135deg, rgba(16,185,129,0.1), rgba(5,150,105,0.1));
            padding: 1.5rem;
            border-radius: 0.7rem;
            border: 1px solid rgba(16,185,129,0.3);
            margin: 1rem 0;
        }
        .warning-box {
            background: linear-gradient(135deg, rgba(245,158,11,0.1), rgba(217,119,6,0.1));
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid rgba(245,158,11,0.3);
            margin: 0.5rem 0;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("✈️ Flight Matrix + Ottimizzazione Turni CCNL")

    with st.container():
        st.markdown(
            """
            <div class="info-card">
                <p>🛫🛬 <strong>Sistema di ottimizzazione turni shuttle aeroporto</strong></p>
                <p style="margin-top:0.5rem;">Vincoli CCNL implementati:</p>
                <ul style="margin-top:0.3rem; margin-bottom: 0;">
                    <li>✅ Lavoro minimo 3h15 (tranne Supplementi e Part-time)</li>
                    <li>✅ Nastro massimo assoluto: 10h30</li>
                    <li>✅ Tipologie: Intero, Semiunico, Spezzato, Sosta Inoperosa, Part-time, Supplemento</li>
                    <li>✅ Calcolo lavoro: corse + pre/post + fuori linea + inoperosità (12%)</li>
                    <li>✅ Turni stabili cross-date (nessun tipo "Altro")</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    uploaded_file = st.file_uploader("Carica il PDF con gli orari dei voli", type=["pdf"])

    if uploaded_file is None:
        st.info("📄 Carica il PDF per procedere.")
        return

    with st.spinner("Parsing del PDF in corso..."):
        flights_df = parse_pdf_to_flights_df(uploaded_file)

    if flights_df.empty:
        st.error("❌ Non sono stati trovati voli PAX.")
        return

    unique_days = sorted(flights_df["Date"].unique())
    num_days = len(unique_days)
    num_flights = len(flights_df)

    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.metric("Voli PAX", num_flights)
    with col2:
        st.metric("Giorni", num_days)
    with col3:
        if num_days > 0:
            st.write(f"📆 {unique_days[0].strftime('%d/%m/%Y')} – {unique_days[-1].strftime('%d/%m/%Y')}")

    st.success("✅ Parsing completato.")

    weekdays_present = sorted(
        flights_df["Weekday"].unique(),
        key=lambda x: WEEKDAY_ORDER.index(x),
    )

    st.sidebar.header("Filtro giorno")
    selected_weekday = st.sidebar.selectbox(
        "Tipo giorno",
        options=weekdays_present,
        format_func=lambda x: WEEKDAY_LABELS_IT.get(x, x),
    )

    matrix_df = build_matrix_for_weekday(flights_df, selected_weekday)

    if matrix_df.empty:
        st.warning("⚠️ Nessun volo per questo giorno.")
        return

    label_it = WEEKDAY_LABELS_IT.get(selected_weekday, selected_weekday)
    weekday_ops = flights_df[flights_df["Weekday"] == selected_weekday]

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
        f"**Voli PAX:** {len(weekday_ops)} (su {weekday_ops['Date'].nunique()} {label_it.lower()})"
    )

    st.markdown(
        """
        <div style="margin-bottom: 0.6rem; margin-top: 0.2rem;">
            <span class="legend-pill">
                <span class="legend-color-arr"></span>
                <span>Arrivi</span>
            </span>
            <span class="legend-pill">
                <span class="legend-color-dep"></span>
                <span>Partenze</span>
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # FILTRI
    with st.expander("🔍 Filtri", expanded=False):
        flight_filter = st.text_input("Codice Volo", placeholder="FR, EN8...")
        airport_options = sorted(matrix_df["Route"].unique())
        selected_airports = st.multiselect("Aeroporto", airport_options)
        ad_choice = st.radio("Movimento", ["Arrivi e partenze", "Solo arrivi (A)", "Solo partenze (P)"], horizontal=True)

    matrix_filtered = matrix_df.copy()
    if flight_filter:
        matrix_filtered = matrix_filtered[matrix_filtered["Flight"].str.contains(flight_filter, case=False, na=False)]
    if selected_airports:
        matrix_filtered = matrix_filtered[matrix_filtered["Route"].isin(selected_airports)]
    if ad_choice == "Solo arrivi (A)":
        matrix_filtered = matrix_filtered[matrix_filtered["AD"] == "A"]
    elif ad_choice == "Solo partenze (P)":
        matrix_filtered = matrix_filtered[matrix_filtered["AD"] == "P"]

    if matrix_filtered.empty:
        st.warning("⚠️ Nessun volo con questi filtri.")
    else:
        display_df = matrix_filtered.rename(columns={"Flight": "Codice Volo", "Route": "Aeroporto"})
        
        if "AD" in display_df.columns:
            styled_df = display_df.style.apply(style_time, axis=1).applymap(style_ad, subset=["AD"])
        else:
            styled_df = display_df.style

        st.dataframe(styled_df, use_container_width=True, height=500)

        csv_buffer = display_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Matrice voli CSV", csv_buffer, f"voli_{label_it.lower()}.csv", "text/csv")

        # OTTIMIZZAZIONE
        st.markdown("---")
        st.markdown("## 🚀 Ottimizzazione Turni")

        if st.button("🎯 Genera turni ottimizzati", type="primary"):
            flights_for_turns = filter_flights_for_turns(flights_df, selected_weekday, flight_filter, selected_airports, ad_choice)

            if flights_for_turns.empty:
                st.warning("⚠️ Nessun volo disponibile.")
            else:
                with st.spinner("⏳ Ottimizzazione in corso..."):
                    shifts, stats = generate_driver_shifts_optimized(flights_for_turns, selected_weekday)

                if not shifts:
                    st.error("❌ Impossibile generare turni validi.")
                    if stats.get("invalid_shifts", 0) > 0:
                        st.warning(f"⚠️ {stats['invalid_shifts']} turni scartati per violazione vincoli CCNL.")
                else:
                    # ANALISI
                    analysis = analyze_shifts(shifts, selected_weekday)
                    
                    st.markdown("### 📊 Analisi Soluzione")
                    
                    st.markdown(f"""
                    <div class="analysis-box">
                        <h4 style="margin-top: 0;">✅ Ottimizzazione Completata</h4>
                        <p><strong>Status:</strong> {stats.get('status', 'N/A')} 
                           {'✅ Ottimo' if stats.get('optimal', False) else '⚠️ Ammissibile'}</p>
                        <p><strong>Tempo:</strong> {stats.get('solve_time', 0):.2f}s | 
                           <strong>Turni:</strong> {analysis['total_shifts']} | 
                           <strong>Pattern unici:</strong> {analysis['unique_patterns']}</p>
                        
                        <h4 style="margin-top: 1rem;">📋 Distribuzione Tipologie</h4>
                    """, unsafe_allow_html=True)
                    
                    type_dist = analysis.get('type_distribution', {})
                    for tipo in ["Intero", "Semiunico", "Spezzato", "Sosta Inoperosa", "Part-time", "Supplemento"]:
                        count = type_dist.get(tipo, 0)
                        if count > 0:
                            pct = (count / analysis['total_shifts'] * 100)
                            st.markdown(f"- **{tipo}**: {count} ({pct:.1f}%)")
                    
                    st.markdown(f"""
                        <h4 style="margin-top: 1rem;">⏱️ Metriche</h4>
                        <p><strong>Lavoro medio:</strong> {analysis['avg_work_hours']:.2f}h | 
                           <strong>Nastro medio:</strong> {analysis['avg_nastro_hours']:.2f}h</p>
                        <p><strong>Turni ~7h:</strong> {analysis['shifts_near_7h']} ({analysis['shifts_near_7h_pct']:.1f}%)</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # WARNINGS
                    violations = analysis.get('violations', [])
                    if violations:
                        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                        st.warning("⚠️ **Attenzione: vincoli violati**")
                        for v in violations:
                            st.markdown(f"- {v}")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    if stats.get("invalid_shifts", 0) > 0:
                        st.info(f"ℹ️ {stats['invalid_shifts']} turni scartati durante ottimizzazione (superavano limiti CCNL)")
                    
                    # MATRICE
                    st.markdown("### 📅 Matrice Validità Turni")
                    
                    dates_for_matrix = sorted(flights_for_turns["Date"].unique())
                    
                    rows_matrix = []
                    for s in shifts:
                        work_h = int(s["work_min"] // 60)
                        work_m = int(s["work_min"] % 60)
                        nastro_h = int(s["nastro_min"] // 60)
                        nastro_m = int(s["nastro_min"] % 60)
                        
                        row = {
                            "Codice": s["code"],
                            "Tipo": s["tipo_turno"],
                            "Inizio": s["shift_start_dt"].strftime("%H:%M"),
                            "Fine": s["shift_end_dt"].strftime("%H:%M"),
                            "Lavoro": f"{work_h}h{work_m:02d}",
                            "Nastro": f"{nastro_h}h{nastro_m:02d}",
                            "Corse": s["corses"],
                        }
                        
                        for d in dates_for_matrix:
                            row[d.strftime("%d-%m")] = "✅" if s["date"] == d else ""
                        
                        rows_matrix.append(row)
                    
                    df_matrix = pd.DataFrame(rows_matrix)
                    
                    df_grouped = df_matrix.groupby("Codice").agg({
                        "Tipo": "first",
                        "Inizio": "first",
                        "Fine": "first",
                        "Lavoro": "first",
                        "Nastro": "first",
                        "Corse": "first",
                        **{col: lambda x: "✅" if "✅" in x.values else "" 
                           for col in [c for c in df_matrix.columns if c[0].isdigit()]}
                    }).reset_index()
                    
                    st.dataframe(df_grouped, use_container_width=True, height=600)
                    
                    csv_matrix = df_grouped.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Matrice turni CSV", csv_matrix, f"turni_{label_it.lower()}.csv", "text/csv")
                    
                    # DETTAGLIO
                    st.markdown("### 🔍 Dettaglio Turno")
                    selected_code = st.selectbox("Seleziona turno", sorted({s["code"] for s in shifts}))
                    
                    if selected_code:
                        turno = next((s for s in shifts if s["code"] == selected_code), None)
                        if turno:
                            st.markdown(f"""
                            **{selected_code}** - {turno['tipo_turno']}  
                            🕐 {turno['shift_start_dt'].strftime('%H:%M')} - {turno['shift_end_dt'].strftime('%H:%M')}  
                            💼 Lavoro: {int(turno['work_min']//60)}h{int(turno['work_min']%60):02d} | 
                            Nastro: {int(turno['nastro_min']//60)}h{int(turno['nastro_min']%60):02d}  
                            🚌 Corse: {turno['corses']}
                            """)
                            st.text(turno["detail"])

    # GRAFICO
    st.markdown("---")
    st.markdown("### 📈 Andamento Arrivi/Partenze")

    chart_df = flights_df.copy()
    chart_df["Dir"] = chart_df["AD"].map(lambda x: "Arrivi" if x == "A" else ("Partenze" if x in ["P","D"] else None))
    chart_df = chart_df.dropna(subset=["Dir"])

    if not chart_df.empty:
        daily = chart_df.groupby(["Date", "Dir"])["Flight"].count().unstack("Dir").fillna(0).sort_index()
        for col in ["Arrivi", "Partenze"]:
            if col not in daily.columns:
                daily[col] = 0
        st.line_chart(daily[["Arrivi", "Partenze"]])


if __name__ == "__main__":
    main()
