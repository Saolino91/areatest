# streamlit_app.py

"""
App Streamlit per:
1. Caricare un PDF con orari voli (qualsiasi mese, tipo Feb/Mar 2026).
2. Parsare i voli PAX, anche quando un giorno è spezzato su più tabelle / pagine.
3. Raggruppare per giorno della settimana.
4. Visualizzare una matrice voli × date con interfaccia curata e filtri.
5. Esportare la matrice in CSV.
6. Visualizzare un grafico a linee Arrivi/Partenze per giorno.
7. Generare turni guida con algoritmo ottimizzato OR-Tools.
"""

import io
import re
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import List, Optional, Tuple

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

# pattern generico: Sun 1 Mar 2026, Mon 2 Feb 2026, ecc.
DAY_PATTERN = re.compile(
    r"^(Sun|Mon|Tue|Wed|Thu|Fri|Sat)\s+(\d{1,2})\s+([A-Za-z]{3})\s+(\d{4})$"
)

MONTH_MAP = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
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

# soglia per raggruppare voli ravvicinati (minuti)
DEPARTURE_GROUP_DELTA_MIN = 20
ARRIVAL_GROUP_DELTA_MIN = 20

# target ore di lavoro effettivo per un turno intero (minuti)
TARGET_WORK_MIN = 7 * 60  # 7 ore


# =========================
# PARSING PDF
# =========================

def parse_pdf_to_flights_df(file_obj: io.BytesIO) -> pd.DataFrame:
    """
    Parser per il PDF con orario voli (es. febbraio o marzo 2026).

    Logica:
    - divide orizzontalmente la pagina in 7 colonne di uguale larghezza;
    - per ogni tabella:
        * calcola la colonna dal centro orizzontale (xc);
        * se la prima cella è tipo "Sun 22 Mar 2026" → nuova data e weekday per quella colonna;
        * altrimenti la tabella è continuazione del giorno corrente in quella colonna;
    - per ogni tabella con data nota legge le righe voli:
        Flight, Route, A/D, Type, ETA, ETD.

    Restituisce un DataFrame con colonne:
        ['Date', 'Weekday', 'Flight', 'Route', 'AD', 'Type', 'ETA', 'ETD']
        (poi filtrato a PAX).
    """
    records: List[dict] = []

    with pdfplumber.open(file_obj) as pdf:
        first_page = pdf.pages[0]
        page_width = first_page.width
        col_width = page_width / 7.0

        def col_index_from_xc(xc: float) -> int:
            idx = int(xc / col_width)
            if idx < 0:
                idx = 0
            if idx > 6:
                idx = 6
            return idx

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

                    if first_cell.lower() == "flight":
                        start_idx = 1
                    else:
                        start_idx = 0

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

                    records.append(
                        {
                            "Date": cur_date,
                            "Weekday": cur_weekday,
                            "Flight": flight,
                            "Route": route,
                            "AD": ad,
                            "Type": typ,
                            "ETA": eta,
                            "ETD": etd,
                        }
                    )

    if not records:
        return pd.DataFrame(
            columns=["Date", "Weekday", "Flight", "Route", "AD", "Type", "ETA", "ETD"]
        )

    df = pd.DataFrame(records)
    df["Type"] = df["Type"].str.upper().str.strip()
    df["AD"] = df["AD"].str.upper().str.strip()
    df["ETA"] = df["ETA"].str.strip()
    df["ETD"] = df["ETD"].str.strip()

    # solo PAX
    df = df[df["Type"] == "PAX"].copy()
    df.replace({"": None}, inplace=True)

    return df


# =========================
# COSTRUZIONE MATRICE
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
# STYLING PER LA VIEW
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
# SUPPORTO TURNI GUIDA
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
    """
    Costruisce i "legs" bus a partire dai voli filtrati.

    PARTENZE (AD = P):
    - raggruppo voli nello stesso giorno con delta ETD consecutivo <= DEPARTURE_GROUP_DELTA_MIN
    - il bus arriva in aeroporto 1h prima del volo più precoce del gruppo
    - parte da Piazza Cavour 40' prima.

    ARRIVI (AD = A):
    - raggruppo voli nello stesso giorno con delta ETA consecutivo <= ARRIVAL_GROUP_DELTA_MIN
    - il bus parte dall'aeroporto 25' dopo il volo che atterra per ultimo del gruppo
    - arriva a Piazza Cavour 35' dopo la partenza.
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
                bus_arr_apt = earliest_dt - timedelta(hours=1)
                bus_dep_pcv = bus_arr_apt - timedelta(minutes=40)

                trips.append(
                    {
                        "Date": d,
                        "Direction": "PCV-APT",
                        "service_start": bus_dep_pcv,
                        "service_end": bus_arr_apt,
                        "flights": [g["Flight"] for g in group],
                        "ad_type": "P",
                        "routes": [g["Route"] for g in group],
                    }
                )
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
                bus_dep_apt = latest_dt + timedelta(minutes=25)
                bus_arr_pcv = bus_dep_apt + timedelta(minutes=35)

                trips.append(
                    {
                        "Date": d,
                        "Direction": "APT-PCV",
                        "service_start": bus_dep_apt,
                        "service_end": bus_arr_pcv,
                        "flights": [g["Flight"] for g in group],
                        "ad_type": "A",
                        "routes": [g["Route"] for g in group],
                    }
                )
                i = j

    trips = sorted(trips, key=lambda t: t["service_start"])
    return trips


def build_roundtrips_from_trips(trips: List[dict]) -> List[dict]:
    """
    Costruisce le CORSE (andata+ritorno) a partire dai legs PCV-APT / APT-PCV.

    Ogni corsa:
    - parte da Piazza Cavour
    - arriva in aeroporto (leg PCV-APT)
    - poi torna a Piazza Cavour (leg APT-PCV)
    - la APT-PCV deve iniziare dopo la fine della PCV-APT.

    Ogni corsa garantisce: inizio a PCV, fine a PCV.
    """
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

            roundtrips.append(
                {
                    "Date": d,
                    "start": dep["service_start"],
                    "end": arr["service_end"],
                    "dep_leg": dep,
                    "arr_leg": arr,
                }
            )

    roundtrips = sorted(roundtrips, key=lambda r: r["start"])
    return roundtrips


def classify_shift_type(roundtrips: List[dict], nastro_min: float, work_min: float) -> str:
    """
    Classifica il turno in:
    - Supplemento
    - Part-time
    - Intero
    - Semiunico
    - Spezzato
    - Inoperoso
    in modo approssimato ma coerente con le regole discusse.
    """

    # Supplemento: turno breve max 3h
    if nastro_min <= 180:
        return "Supplemento"

    # Part-time: lavoro effettivo tra 3 e 6 ore
    if 180 < work_min < 360 and nastro_min <= 8 * 60:
        return "Part-time"

    # gap a PCV tra le corse (end_i → start_{i+1})
    pcv_gaps = []
    for i in range(len(roundtrips) - 1):
        end_i = roundtrips[i]["end"]
        start_j = roundtrips[i + 1]["start"]
        gap = (start_j - end_i).total_seconds() / 60.0
        pcv_gaps.append(gap)

    # gap all'aeroporto: dentro ogni roundtrip (fine PCV-APT → inizio APT-PCV)
    apt_gaps = []
    for rt in roundtrips:
        dep_end = rt["dep_leg"]["service_end"]
        arr_start = rt["arr_leg"]["service_start"]
        gap = (arr_start - dep_end).total_seconds() / 60.0
        apt_gaps.append(gap)

    max_pcv_gap = max(pcv_gaps) if pcv_gaps else 0
    max_apt_gap = max(apt_gaps) if apt_gaps else 0

    # Inoperoso: nastro max 9h, sosta inoperosa in aeroporto >= 31'
    if nastro_min <= 9 * 60 and max_apt_gap >= 31:
        return "Inoperoso"

    # Spezzato: nastro max 10h30, interruzione in deposito (a PCV) >= 3h
    if nastro_min <= 10 * 60 + 30 and max_pcv_gap >= 180:
        return "Spezzato"

    # Semiunico: nastro max 9h15, interruzione a PCV >=40' (ma <3h)
    if nastro_min <= 9 * 60 + 15 and 40 <= max_pcv_gap < 180:
        return "Semiunico"

    # Intero: nastro <= 8h, nessuna interruzione > 40', almeno una pausa >=30'
    if nastro_min <= 8 * 60:
        if max_pcv_gap <= 40 and any(g >= 30 for g in pcv_gaps):
            return "Intero"
        return "Intero"

    # fallback
    return "Altro"


# =========================
# OTTIMIZZAZIONE CON OR-TOOLS CP-SAT
# =========================

def optimize_shifts_cp_sat(roundtrips: List[dict], weekday: str, max_shifts: int = 30, time_limit_sec: int = 30) -> Tuple[List[dict], dict]:
    """
    Ottimizza l'assegnazione delle corse (roundtrips) ai turni usando OR-Tools CP-SAT.
    
    Obiettivo: minimizzare il numero di turni rispettando tutti i vincoli CCNL.
    
    Vincoli:
    1. Ogni corsa assegnata a esattamente 1 turno
    2. Max 3 corse per turno
    3. Nastro turno <= 8h (480 min)
    4. Nessuna sovrapposizione temporale tra corse dello stesso turno
    5. Corse sequenziali (fine corsa i < inizio corsa i+1)
    
    Returns:
        (shifts_list, stats) dove:
        - shifts_list: lista di dict con la stessa struttura di build_shifts_from_roundtrips
        - stats: dict con statistiche ottimizzazione (tempo, status, numero turni, etc.)
    """
    if not ORTOOLS_AVAILABLE:
        return [], {"error": "OR-Tools non disponibile"}
    
    if not roundtrips:
        return [], {"status": "NO_ROUNDTRIPS"}
    
    # Separa per data
    dates_set = {rt["Date"] for rt in roundtrips}
    if len(dates_set) > 1:
        # Ottimizza separatamente per ogni data
        all_shifts = []
        all_stats = []
        
        for d in sorted(dates_set):
            rts_d = [rt for rt in roundtrips if rt["Date"] == d]
            shifts_d, stats_d = optimize_shifts_cp_sat_single_date(rts_d, weekday, max_shifts, time_limit_sec)
            all_shifts.extend(shifts_d)
            all_stats.append(stats_d)
        
        # Statistiche aggregate
        combined_stats = {
            "status": "MULTI_DATE",
            "dates": len(dates_set),
            "total_shifts": len(all_shifts),
            "total_roundtrips": len(roundtrips),
            "avg_solve_time": sum(s.get("solve_time", 0) for s in all_stats) / len(all_stats) if all_stats else 0,
        }
        
        return all_shifts, combined_stats
    else:
        return optimize_shifts_cp_sat_single_date(roundtrips, weekday, max_shifts, time_limit_sec)


def optimize_shifts_cp_sat_single_date(roundtrips: List[dict], weekday: str, max_shifts: int = 30, time_limit_sec: int = 30) -> Tuple[List[dict], dict]:
    """
    Ottimizzazione CP-SAT per una singola data.
    """
    model = cp_model.CpModel()
    n = len(roundtrips)
    
    # VARIABILI
    # shift_used[j] = 1 se il turno j è utilizzato
    shift_used = [model.NewBoolVar(f'shift_{j}') for j in range(max_shifts)]
    
    # assignment[i, j] = 1 se la corsa i è assegnata al turno j
    assignment = {}
    for i in range(n):
        for j in range(max_shifts):
            assignment[i, j] = model.NewBoolVar(f'x_{i}_{j}')
    
    # Variabili ausiliarie per calcolo nastro
    shift_start = [model.NewIntVar(0, 24*60, f'start_{j}') for j in range(max_shifts)]
    shift_end = [model.NewIntVar(0, 24*60, f'end_{j}') for j in range(max_shifts)]
    
    # VINCOLO 1: ogni corsa assegnata a esattamente 1 turno
    for i in range(n):
        model.Add(sum(assignment[i, j] for j in range(max_shifts)) == 1)
    
    # VINCOLO 2: se turno j ha almeno 1 corsa, è "usato"
    for j in range(max_shifts):
        corse_in_turno = sum(assignment[i, j] for i in range(n))
        model.Add(corse_in_turno >= shift_used[j])
        model.Add(corse_in_turno <= shift_used[j] * n)  # se non usato, 0 corse
    
    # VINCOLO 3: max 3 corse per turno
    for j in range(max_shifts):
        model.Add(sum(assignment[i, j] for i in range(n)) <= 3)
    
    # VINCOLO 4: nessuna sovrapposizione temporale
    for j in range(max_shifts):
        for i1 in range(n):
            for i2 in range(i1 + 1, n):
                rt1, rt2 = roundtrips[i1], roundtrips[i2]
                
                # Converti datetime in minuti dall'inizio giornata
                start1 = rt1["start"].hour * 60 + rt1["start"].minute
                end1 = rt1["end"].hour * 60 + rt1["end"].minute
                start2 = rt2["start"].hour * 60 + rt2["start"].minute
                end2 = rt2["end"].hour * 60 + rt2["end"].minute
                
                # Se si sovrappongono, non possono stare nello stesso turno
                if not (end1 <= start2 or end2 <= start1):
                    model.Add(assignment[i1, j] + assignment[i2, j] <= 1)
    
    # VINCOLO 5: sequenzialità (se i1 < i2 e sono nello stesso turno, end(i1) < start(i2))
    for j in range(max_shifts):
        for i1 in range(n):
            for i2 in range(i1 + 1, n):
                rt1, rt2 = roundtrips[i1], roundtrips[i2]
                
                # Se entrambe nel turno j, devono essere sequenziali
                both_in_j = model.NewBoolVar(f'both_{i1}_{i2}_{j}')
                model.Add(assignment[i1, j] + assignment[i2, j] == 2).OnlyEnforceIf(both_in_j)
                model.Add(assignment[i1, j] + assignment[i2, j] <= 1).OnlyEnforceIf(both_in_j.Not())
                
                # Se entrambe nel turno, verifica ordine temporale
                end1 = rt1["end"].hour * 60 + rt1["end"].minute
                start2 = rt2["start"].hour * 60 + rt2["start"].minute
                
                if end1 >= start2:
                    # Non possono stare insieme se non sono sequenziali
                    model.Add(assignment[i1, j] + assignment[i2, j] <= 1)
    
    # VINCOLO 6: calcolo nastro turno
    for j in range(max_shifts):
        # Se il turno è usato, calcola inizio e fine
        corse_indices = []
        for i in range(n):
            # Se corsa i è nel turno j
            is_in_turno = model.NewBoolVar(f'in_turno_{i}_{j}')
            model.Add(assignment[i, j] == 1).OnlyEnforceIf(is_in_turno)
            model.Add(assignment[i, j] == 0).OnlyEnforceIf(is_in_turno.Not())
            
            rt = roundtrips[i]
            start_min = (rt["start"].hour * 60 + rt["start"].minute) - 20  # 20' pre
            end_min = (rt["end"].hour * 60 + rt["end"].minute) + 12  # 12' post
            
            # Contributo di questa corsa all'inizio/fine del turno
            model.Add(shift_start[j] <= start_min).OnlyEnforceIf(is_in_turno)
            model.Add(shift_end[j] >= end_min).OnlyEnforceIf(is_in_turno)
        
        # Nastro <= 8h = 480 min
        nastro = model.NewIntVar(0, 24*60, f'nastro_{j}')
        model.Add(nastro == shift_end[j] - shift_start[j])
        model.Add(nastro <= 480).OnlyEnforceIf(shift_used[j])
    
    # OBIETTIVO: minimizza numero turni
    model.Minimize(sum(shift_used))
    
    # RISOLVI
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_sec
    solver.parameters.log_search_progress = False
    
    status = solver.Solve(model)
    
    # ESTRAI SOLUZIONE
    stats = {
        "status": solver.StatusName(status),
        "solve_time": solver.WallTime(),
        "optimal": status == cp_model.OPTIMAL,
        "num_shifts": 0,
        "num_roundtrips": n,
    }
    
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        shifts_list = []
        the_date = roundtrips[0]["Date"]
        
        for j in range(max_shifts):
            if solver.Value(shift_used[j]):
                # Estrai corse assegnate a questo turno
                rt_list = []
                for i in range(n):
                    if solver.Value(assignment[i, j]):
                        rt_list.append(roundtrips[i])
                
                if not rt_list:
                    continue
                
                # Ordina per tempo
                rt_list = sorted(rt_list, key=lambda x: x["start"])
                
                # Calcola metriche turno
                first = rt_list[0]
                last = rt_list[-1]
                shift_start_dt = first["start"] - timedelta(minutes=20)
                shift_end_dt = last["end"] + timedelta(minutes=12)
                nastro_min = (shift_end_dt - shift_start_dt).total_seconds() / 60.0
                work_min = sum((rt["end"] - rt["start"]).total_seconds() / 60.0 for rt in rt_list)
                corses = len(rt_list)
                
                tipo_turno = classify_shift_type(rt_list, nastro_min, work_min)
                
                # Dettagli corse
                detail_lines = []
                for rt in rt_list:
                    dep = rt["dep_leg"]
                    arr = rt["arr_leg"]
                    
                    ora_p = dep["service_start"].strftime("%H:%M")
                    ora_a = dep["service_end"].strftime("%H:%M")
                    codici_dep = ",".join(dep["flights"])
                    line_dep = f"{ora_p}, Piazza Cavour, Aeroporto, {ora_a}, {codici_dep}"
                    detail_lines.append(line_dep)
                    
                    ora_p2 = arr["service_start"].strftime("%H:%M")
                    ora_a2 = arr["service_end"].strftime("%H:%M")
                    codici_arr = ",".join(arr["flights"])
                    line_arr = f"{ora_p2}, Aeroporto, Piazza Cavour, {ora_a2}, {codici_arr}"
                    detail_lines.append(line_arr)
                
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
    
    else:
        return [], stats


# =========================
# ALGORITMO GREEDY ORIGINALE (per confronto)
# =========================

def build_shifts_from_roundtrips_greedy(roundtrips: List[dict], weekday: str) -> List[dict]:
    """
    Algoritmo greedy originale - mantenuto per confronto.
    """
    if not roundtrips:
        return []

    dates_set = {rt["Date"] for rt in roundtrips}
    if len(dates_set) > 1:
        raise ValueError("build_shifts_from_roundtrips_greedy deve ricevere corse di una sola data")

    rts_sorted = sorted(roundtrips, key=lambda r: r["start"])
    shifts_rts: List[List[dict]] = []
    current: List[dict] = []

    def nastro_minutes_for(rt_list: List[dict]) -> float:
        if not rt_list:
            return 0.0
        first = rt_list[0]
        last = rt_list[-1]
        shift_start = first["start"] - timedelta(minutes=20)
        shift_end = last["end"] + timedelta(minutes=12)
        return (shift_end - shift_start).total_seconds() / 60.0

    def work_minutes_for(rt_list: List[dict]) -> float:
        return sum((rt["end"] - rt["start"]).total_seconds() / 60.0 for rt in rt_list)

    for rt in rts_sorted:
        if not current:
            current = [rt]
            continue

        last_rt = current[-1]

        if rt["start"] < last_rt["end"]:
            shifts_rts.append(current)
            current = [rt]
            continue

        tentative = current + [rt]
        nastro_tent = nastro_minutes_for(tentative)
        work_tent = work_minutes_for(tentative)
        work_cur = work_minutes_for(current)

        if nastro_tent > 8 * 60 or len(tentative) > 3:
            shifts_rts.append(current)
            current = [rt]
            continue

        if work_cur < TARGET_WORK_MIN:
            if work_tent <= TARGET_WORK_MIN + 60:
                current.append(rt)
            else:
                shifts_rts.append(current)
                current = [rt]
        else:
            shifts_rts.append(current)
            current = [rt]

    if current:
        shifts_rts.append(current)

    the_date = next(iter(dates_set))
    shifts: List[dict] = []
    for rt_list in shifts_rts:
        first = rt_list[0]
        last = rt_list[-1]
        shift_start = first["start"] - timedelta(minutes=20)
        shift_end = last["end"] + timedelta(minutes=12)
        nastro_min = (shift_end - shift_start).total_seconds() / 60.0
        work_min = sum((rt["end"] - rt["start"]).total_seconds() / 60.0 for rt in rt_list)
        corses = len(rt_list)

        tipo_turno = classify_shift_type(rt_list, nastro_min, work_min)

        detail_lines = []
        for rt in rt_list:
            dep = rt["dep_leg"]
            arr = rt["arr_leg"]

            ora_p = dep["service_start"].strftime("%H:%M")
            ora_a = dep["service_end"].strftime("%H:%M")
            codici_dep = ",".join(dep["flights"])
            line_dep = f"{ora_p}, Piazza Cavour, Aeroporto, {ora_a}, {codici_dep}"
            detail_lines.append(line_dep)

            ora_p2 = arr["service_start"].strftime("%H:%M")
            ora_a2 = arr["service_end"].strftime("%H:%M")
            codici_arr = ",".join(arr["flights"])
            line_arr = f"{ora_p2}, Aeroporto, Piazza Cavour, {ora_a2}, {codici_arr}"
            detail_lines.append(line_arr)

        detail = "\n".join(detail_lines)

        shifts.append({
            "weekday": weekday,
            "date": the_date,
            "shift_start_dt": shift_start,
            "shift_end_dt": shift_end,
            "nastro_min": nastro_min,
            "work_min": work_min,
            "corses": corses,
            "detail": detail,
            "tipo_turno": tipo_turno,
        })

    return shifts


def assign_shift_codes_across_dates(shifts: List[dict], weekday: str) -> List[dict]:
    """
    Assegna i codici turno tenendo conto di TUTTE le date del weekday.
    """
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

    am = [(key, s) for key, s in pattern_items if s["shift_start_dt"].hour < 12]
    pm = [(key, s) for key, s in pattern_items if s["shift_start_dt"].hour >= 12]

    am.sort(key=lambda ks: ks[1]["shift_start_dt"])
    pm.sort(key=lambda ks: ks[1]["shift_start_dt"])

    code_by_key = {}

    idx = 1
    for key, sample in am:
        num = min(idx, 49)
        idx += 1
        corses = sample["corses"]
        if corses >= 3:
            suffix = "I"
        elif corses == 2:
            suffix = "P"
        else:
            suffix = "S"
        code_by_key[key] = f"{prefix}{num:02d}{suffix}"

    idx = 50
    for key, sample in pm:
        num = min(idx, 99)
        idx += 1
        corses = sample["corses"]
        if corses >= 3:
            suffix = "I"
        elif corses == 2:
            suffix = "P"
        else:
            suffix = "S"
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


def generate_driver_shifts(filtered_flights: pd.DataFrame, weekday: str, use_optimization: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Pipeline completa con possibilità di scegliere algoritmo greedy o ottimizzato.
    
    Returns:
        (df_optimal, df_greedy, comparison_stats)
    """
    trips = build_bus_trips_from_flights(filtered_flights)
    if not trips:
        return pd.DataFrame(), pd.DataFrame(), {}

    roundtrips_all = build_roundtrips_from_trips(trips)
    if not roundtrips_all:
        return pd.DataFrame(), pd.DataFrame(), {}

    # ALGORITMO OTTIMIZZATO
    if use_optimization and ORTOOLS_AVAILABLE:
        shifts_optimal, opt_stats = optimize_shifts_cp_sat(roundtrips_all, weekday)
        if shifts_optimal:
            shifts_optimal = assign_shift_codes_across_dates(shifts_optimal, weekday)
    else:
        shifts_optimal = []
        opt_stats = {"status": "SKIPPED"}

    # ALGORITMO GREEDY (per confronto)
    shifts_greedy = []
    for d in sorted(filtered_flights["Date"].unique()):
        rts_d = [rt for rt in roundtrips_all if rt["Date"] == d]
        if not rts_d:
            continue
        shifts_d = build_shifts_from_roundtrips_greedy(rts_d, weekday)
        shifts_greedy.extend(shifts_d)
    
    if shifts_greedy:
        shifts_greedy = assign_shift_codes_across_dates(shifts_greedy, weekday)

    # Converti in DataFrame
    def shifts_to_df(shifts_list):
        if not shifts_list:
            return pd.DataFrame()
        
        rows = []
        for s in shifts_list:
            start_dt = s["shift_start_dt"]
            end_dt = s["shift_end_dt"]
            nastro_min = int(round(s["nastro_min"]))
            work_min = int(round(s["work_min"]))
            corses = s["corses"]
            code = s.get("code", "")
            tipo_turno = s["tipo_turno"]
            d = s["date"]

            rows.append({
                "Codice turno": code,
                "Tipo giorno": WEEKDAY_LABELS_IT.get(weekday, weekday),
                "Tipo turno": tipo_turno,
                "Date": d,
                "Data (esempio)": d.strftime("%d/%m/%Y"),
                "Inizio turno": start_dt.strftime("%H:%M"),
                "Fine turno": end_dt.strftime("%H:%M"),
                "Durata nastro (min)": nastro_min,
                "Durata lavoro (min)": work_min,
                "Numero corse": corses,
                "Dettaglio corse": s["detail"],
            })

        df = pd.DataFrame(rows)
        df = df.sort_values(["Date", "Inizio turno", "Codice turno"]).reset_index(drop=True)
        return df

    df_optimal = shifts_to_df(shifts_optimal)
    df_greedy = shifts_to_df(shifts_greedy)

    # Statistiche confronto
    comparison_stats = {
        "optimal_shifts": len(shifts_optimal),
        "greedy_shifts": len(shifts_greedy),
        "improvement": len(shifts_greedy) - len(shifts_optimal) if shifts_optimal else 0,
        "improvement_pct": ((len(shifts_greedy) - len(shifts_optimal)) / len(shifts_greedy) * 100) if shifts_greedy and shifts_optimal else 0,
        "opt_stats": opt_stats,
    }

    return df_optimal, df_greedy, comparison_stats


# =========================
# UI STREAMLIT
# =========================

def main():
    st.set_page_config(
        page_title="Flight Matrix",
        page_icon="✈️",
        layout="wide",
    )

    # ---- CSS custom ----
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }

        h1 {
            text-align: center;
        }

        .info-card {
            background: rgba(15,23,42,0.9);
            padding: 1rem 1.2rem;
            border-radius: 0.9rem;
            border: 1px solid rgba(148,163,184,0.35);
        }

        .info-card p {
            margin-bottom: 0.2rem;
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

        .uploadedFile { font-size: 0.9rem !important; }
        
        .metric-card {
            background: linear-gradient(135deg, rgba(59,130,246,0.1), rgba(147,51,234,0.1));
            padding: 1rem;
            border-radius: 0.5rem;
            border: 1px solid rgba(59,130,246,0.3);
        }
        
        .improvement-badge {
            background: linear-gradient(135deg, #10b981, #059669);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 0.5rem;
            font-weight: bold;
            display: inline-block;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("✈️ Flight Matrix + Ottimizzazione Turni")

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
                    <li>mostra una <strong>matrice</strong> con i voli per tipologia giorno</li>
                    <li>genera <strong>turni guida ottimizzati</strong> con algoritmo CP-SAT (OR-Tools)</li>
                    <li>confronta soluzione ottimizzata vs algoritmo greedy</li>
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

    # --------- FILTRI MATRICE ---------
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
    else:
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

        # --------- GENERAZIONE TURNI GUIDA ---------
        st.markdown("---")
        st.markdown("## 🚀 Ottimizzazione Turni Guida")
        
        st.markdown("""
        <div class="info-card" style="margin-bottom: 1rem;">
            <p><strong>Algoritmo di ottimizzazione CP-SAT (Google OR-Tools)</strong></p>
            <p style="margin-top: 0.5rem;">
            L'ottimizzatore risolve un problema di <em>Vehicle Scheduling</em> con vincoli CCNL:
            </p>
            <ul style="margin-top: 0.3rem; margin-bottom: 0;">
                <li>✅ Minimizza il numero di autisti necessari</li>
                <li>✅ Rispetta vincoli su nastro (max 8h), numero corse (max 3), sequenzialità</li>
                <li>✅ Garantisce soluzione ottima o near-ottima matematicamente provata</li>
                <li>✅ Confronto automatico con algoritmo greedy</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🎯 Genera turni ottimizzati", type="primary"):
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
                with st.spinner("Ottimizzazione in corso... (può richiedere fino a 30 secondi)"):
                    df_optimal, df_greedy, comparison_stats = generate_driver_shifts(
                        flights_for_turns, 
                        selected_weekday,
                        use_optimization=True
                    )
                
                if df_optimal.empty and df_greedy.empty:
                    st.warning("Non è stato possibile generare turni compatibili con i vincoli.")
                else:
                    # --------- METRICHE CONFRONTO ---------
                    st.markdown("### 📊 Confronto Algoritmi")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric(
                            "🎯 Soluzione Ottimizzata",
                            f"{comparison_stats['optimal_shifts']} turni",
                            help="Soluzione trovata con algoritmo CP-SAT"
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric(
                            "📈 Algoritmo Greedy",
                            f"{comparison_stats['greedy_shifts']} turni",
                            help="Soluzione con algoritmo euristico"
                        )
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col3:
                        improvement = comparison_stats['improvement']
                        improvement_pct = comparison_stats['improvement_pct']
                        
                        if improvement > 0:
                            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                            st.metric(
                                "💰 Risparmio",
                                f"{improvement} autisti",
                                f"-{improvement_pct:.1f}%",
                                delta_color="inverse",
                                help="Autisti risparmiati con ottimizzazione"
                            )
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            st.markdown(
                                f'<div class="improvement-badge" style="margin-top: 0.5rem;">⭐ Miglioramento: {improvement_pct:.1f}%</div>',
                                unsafe_allow_html=True
                            )
                        else:
                            st.info("Entrambi gli algoritmi hanno trovato lo stesso numero di turni")
                    
                    # Info ottimizzazione
                    if comparison_stats.get('opt_stats'):
                        opt_stats = comparison_stats['opt_stats']
                        with st.expander("ℹ️ Dettagli ottimizzazione"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Status**: {opt_stats.get('status', 'N/A')}")
                                st.write(f"**Ottimale**: {'✅ Sì' if opt_stats.get('optimal', False) else '⚠️ Feasible'}")
                            with col2:
                                st.write(f"**Tempo calcolo**: {opt_stats.get('solve_time', 0):.2f}s")
                                st.write(f"**Corse totali**: {opt_stats.get('num_roundtrips', 0)}")
                    
                    # --------- TABELLA TURNI OTTIMIZZATI ---------
                    st.markdown("### 🎯 Turni Ottimizzati (Soluzione migliore)")
                    
                    if not df_optimal.empty:
                        display_cols = [
                            "Codice turno",
                            "Tipo giorno",
                            "Tipo turno",
                            "Data (esempio)",
                            "Inizio turno",
                            "Fine turno",
                            "Durata nastro (min)",
                            "Durata lavoro (min)",
                            "Numero corse",
                        ]
                        df_opt_view = df_optimal[display_cols]
                        
                        st.dataframe(df_opt_view, use_container_width=True)
                        
                        csv_opt = df_opt_view.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="⬇️ Scarica turni ottimizzati in CSV",
                            data=csv_opt,
                            file_name=f"turni_ottimizzati_{label_it.lower()}.csv",
                            mime="text/csv",
                        )
                        
                        # --------- MATRICE VALIDITÀ TURNI × DATE ---------
                        st.markdown("#### 📅 Matrice di validità dei turni per data")
                        
                        dates_for_matrix = sorted(flights_for_turns["Date"].unique())
                        if dates_for_matrix:
                            base = df_optimal[
                                [
                                    "Codice turno",
                                    "Tipo turno",
                                    "Inizio turno",
                                    "Fine turno",
                                    "Durata lavoro (min)",
                                    "Date",
                                ]
                            ]
                            
                            rows_matrix = []
                            for code in sorted(base["Codice turno"].unique()):
                                sub = base[base["Codice turno"] == code]
                                tipo = sub["Tipo turno"].iloc[0]
                                start = sub["Inizio turno"].iloc[0]
                                end = sub["Fine turno"].iloc[0]
                                work = sub["Durata lavoro (min)"].iloc[0]
                                
                                row = {
                                    "Codice turno": code,
                                    "Tipo turno": tipo,
                                    "Inizio": start,
                                    "Fine": end,
                                    "Lavoro (min)": work,
                                }
                                
                                for d in dates_for_matrix:
                                    col_name = d.strftime("%d-%m")
                                    has = any(sub["Date"] == d)
                                    row[col_name] = "✅" if has else ""
                                
                                rows_matrix.append(row)
                            
                            df_matrix = pd.DataFrame(rows_matrix)
                            st.dataframe(df_matrix, use_container_width=True)
                        
                        # Dettaglio turno selezionato
                        st.markdown("#### 🔍 Dettaglio corse per turno")
                        selected_code = st.selectbox(
                            "Seleziona un turno per visualizzare il dettaglio delle corse",
                            options=df_optimal["Codice turno"].unique(),
                        )
                        
                        if selected_code:
                            detail_text = df_optimal.loc[
                                df_optimal["Codice turno"] == selected_code, "Dettaglio corse"
                            ].iloc[0]
                            
                            st.markdown(
                                f"**Turno {selected_code} – elenco corse (una corsa = andata + ritorno):**"
                            )
                            st.text(detail_text)
                    else:
                        st.info("Nessun turno ottimizzato generato (verificare vincoli o disponibilità corse)")
                    
                    # --------- CONFRONTO CON GREEDY (opzionale) ---------
                    with st.expander("📈 Confronto con algoritmo Greedy"):
                        if not df_greedy.empty:
                            st.markdown("**Turni generati con algoritmo greedy (euristico):**")
                            df_greedy_view = df_greedy[display_cols]
                            st.dataframe(df_greedy_view, use_container_width=True)
                        else:
                            st.info("Algoritmo greedy non ha generato turni")

    # --------- GRAFICO ARRIVI/PARTENZE PER GIORNO ---------
    st.markdown("---")
    st.markdown("### 📈 Andamento giornaliero Arrivi / Partenze")

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
