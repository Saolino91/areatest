# streamlit_app.py

"""
App Streamlit per:
1. Caricare un PDF con orari voli (qualsiasi mese, tipo Feb/Mar 2026).
2. Parsare i voli PAX, anche quando un giorno è spezzato su più tabelle / pagine.
3. Raggruppare per giorno della settimana.
4. Visualizzare una matrice voli × date con interfaccia curata e filtri.
5. Esportare la matrice in CSV.
6. Visualizzare un grafico a linee Arrivi/Partenze per giorno.
7. Generare turni guida (prima versione) sulla base dei voli filtrati.
"""

import io
import re
from datetime import date, datetime, timedelta
from typing import List, Optional

import pandas as pd
import pdfplumber
import streamlit as st


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

DAY_PREFIX = {
    "Mon": "Mo",
    "Tue": "Tu",
    "Wed": "We",
    "Thu": "Th",
    "Fri": "Fr",
    "Sat": "Sa",
    "Sun": "Su",
}


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
        # larghezza pagina e ampiezza colonne
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

        # stato corrente per ogni colonna: data e weekday
        current_date_by_col = {i: None for i in range(7)}
        current_weekday_by_col = {i: None for i in range(7)}

        # scorri tutte le pagine
        for page in pdf.pages:
            tables = page.find_tables()
            tables = sorted(tables, key=lambda t: t.bbox[1])  # dall'alto in basso

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

                # caso 1: tabella con intestazione del giorno (es. "Sun 22 Mar 2026")
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
                    start_idx = 2  # riga 1 = header "Flight RouteA/D Type ETA ETD"

                # caso 2: continuazione del giorno corrente di quella colonna
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

                # estrai righe voli
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

    # normalizzazione
    df["Type"] = df["Type"].str.upper().str.strip()
    df["AD"] = df["AD"].str.upper().str.strip()
    df["ETA"] = df["ETA"].str.strip()
    df["ETD"] = df["ETD"].str.strip()

    # solo PAX (CARGO esclusi automaticamente)
    df = df[df["Type"] == "PAX"].copy()
    df.replace({"": None}, inplace=True)

    return df


# =========================
# COSTRUZIONE MATRICE
# =========================

def compute_time_value(row: pd.Series) -> Optional[str]:
    """
    Valore da mettere nella matrice:
    - ETA se AD = A (arrivo)
    - ETD se AD in {P, D, DEP, DEPT} (partenza)
    """
    ad = str(row.get("AD", "")).upper()

    if ad in ("A", "ARR", "ARRIVAL"):
        return row.get("ETA") or None

    if ad in ("P", "D", "DEP", "DEPT", "DEPARTURE"):
        return row.get("ETD") or None

    return None


def build_matrix_for_weekday(flights: pd.DataFrame, weekday: str) -> pd.DataFrame:
    """
    Matrice per un dato weekday:

    - Righe = 3 campi:
        Flight, Route, A/D
    - Colonne = date del periodo (dd-mm)
    - Celle = ETA (se arrivo) o ETD (se partenza)
    """
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
    """
    Colore per la colonna AD:
    - P → rosso
    - A → verde
    """
    if val == "P":
        return "color: #f97373;"  # rosso soft
    if val == "A":
        return "color: #4ade80;"  # verde soft
    return ""


def style_time(row: pd.Series):
    """
    Colora gli orari (colonne data) in base al valore di AD nella riga:
    - se AD = P → orari rossi
    - se AD = A → orari verdi
    """
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
    """Combina una data e una stringa HH:MM in un datetime; None se invalida/vuota."""
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
    """Applica agli eventi originari gli stessi filtri della matrice."""
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
    Costruisce le 'corse bus' a partire dai voli filtrati.

    - Per le PARTENZE (AD = P):
        * si raggruppano voli nello stesso giorno con delta ETD <= 20' consecutivi
        * il bus arriva in aeroporto 1h prima del volo più precoce del gruppo
        * parte da Piazza Cavour 40' prima dell'arrivo in aeroporto.
    - Per gli ARRIVI (AD = A):
        * ogni volo genera una corsa separata
        * partenza bus dall'aeroporto: ETA + 25'
        * arrivo a Piazza Cavour: + 35'.
    """
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
                    if delta_min <= 20:
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

        # ARRIVI
        arr_rows = day_rows[day_rows["AD"] == "A"].copy()
        if not arr_rows.empty:
            arr_rows["flight_dt"] = arr_rows["ETA"].apply(lambda s: combine_date_time(d, s))
            arr_rows = arr_rows.dropna(subset=["flight_dt"])
            arr_rows = arr_rows.sort_values("flight_dt")

            for _, r in arr_rows.iterrows():
                dt_arr = r["flight_dt"]
                bus_dep_apt = dt_arr + timedelta(minutes=25)
                bus_arr_pcv = bus_dep_apt + timedelta(minutes=35)

                trips.append(
                    {
                        "Date": d,
                        "Direction": "APT-PCV",
                        "service_start": bus_dep_apt,
                        "service_end": bus_arr_pcv,
                        "flights": [r["Flight"]],
                        "ad_type": "A",
                        "routes": [r["Route"]],
                    }
                )

    trips = sorted(trips, key=lambda t: t["service_start"])
    return trips


def build_shifts_from_trips(trips: List[dict], weekday: str) -> List[dict]:
    """
    Costruisce turni guida a partire dalle corse bus.

    Strategia (prima versione semplificata):
    - ordina tutte le corse per orario di inizio;
    - accumula sequenze di massimo 3 corse, rispettando un nastro massimo di 8 ore;
    - ogni sequenza = 1 turno;
    - nastro = (fine ultima corsa + post/trasferimento) - (inizio prima corsa - pre/trasferimento)
      con pre+trasferimento = 20', post+ritorno+post = 12'.

    Classificazione:
    - nastro <= 180'  → 'Supplemento'
    - 180' < nastro < 360' → 'Part-time'
    - nastro >= 360' → 'Intero'
    """
    if not trips:
        return []

    trips_sorted = sorted(trips, key=lambda t: t["service_start"])
    shifts_trips: List[List[dict]] = []
    current: List[dict] = []

    def nastro_minutes_for(trip_list: List[dict]) -> float:
        if not trip_list:
            return 0.0
        first = trip_list[0]
        last = trip_list[-1]
        shift_start = first["service_start"] - timedelta(minutes=20)
        shift_end = last["service_end"] + timedelta(minutes=12)
        return (shift_end - shift_start).total_seconds() / 60.0

    for trip in trips_sorted:
        if not current:
            current = [trip]
            continue

        tentative = current + [trip]
        nastro = nastro_minutes_for(tentative)

        # vincoli: max 3 corse, nastro <= 8h
        if len(tentative) <= 3 and nastro <= 8 * 60:
            current.append(trip)
        else:
            shifts_trips.append(current)
            current = [trip]

    if current:
        shifts_trips.append(current)

    shifts: List[dict] = []
    for st_trips in shifts_trips:
        first = st_trips[0]
        last = st_trips[-1]
        shift_start = first["service_start"] - timedelta(minutes=20)
        shift_end = last["service_end"] + timedelta(minutes=12)
        nastro_min = (shift_end - shift_start).total_seconds() / 60.0
        work_min = sum(
            (t["service_end"] - t["service_start"]).total_seconds() / 60.0
            for t in st_trips
        )
        trip_count = len(st_trips)

        # classificazione semplice
        if nastro_min <= 180:
            tipo_turno = "Supplemento"
        elif nastro_min < 360:
            tipo_turno = "Part-time"
        else:
            tipo_turno = "Intero"

        # dettaglio corse
        trip_summaries = []
        for t in st_trips:
            dir_label = "PCV→APT" if t["Direction"] == "PCV-APT" else "APT→PCV"
            time_str = f"{t['service_start'].strftime('%H:%M')}-{t['service_end'].strftime('%H:%M')}"
            fl_str = ",".join(t["flights"])
            trip_summaries.append(f"{dir_label} {time_str} ({fl_str})")
        detail = " | ".join(trip_summaries)

        shifts.append(
            {
                "weekday": weekday,
                "shift_start_dt": shift_start,
                "shift_end_dt": shift_end,
                "nastro_min": nastro_min,
                "work_min": work_min,
                "trip_count": trip_count,
                "detail": detail,
                "dates_covered": sorted({t["Date"] for t in st_trips}),
            }
        )

    return shifts


def assign_shift_codes(shifts: List[dict], weekday: str) -> List[dict]:
    """Assegna i codici turno secondo la codifica richiesta."""
    if not shifts:
        return shifts

    prefix = DAY_PREFIX.get(weekday, weekday[:2])

    am_shifts = [s for s in shifts if s["shift_start_dt"].hour < 12]
    pm_shifts = [s for s in shifts if s["shift_start_dt"].hour >= 12]

    am_shifts = sorted(am_shifts, key=lambda s: s["shift_start_dt"])
    pm_shifts = sorted(pm_shifts, key=lambda s: s["shift_start_dt"])

    for idx, s in enumerate(am_shifts, start=1):
        num = min(idx, 49)
        trips = s["trip_count"]
        if trips >= 3:
            suffix = "I"
        elif trips == 2:
            suffix = "P"
        else:
            suffix = "S"
        s["code"] = f"{prefix}{num:02d}{suffix}"

    for j, s in enumerate(pm_shifts, start=50):
        num = min(j, 99)
        trips = s["trip_count"]
        if trips >= 3:
            suffix = "I"
        elif trips == 2:
            suffix = "P"
        else:
            suffix = "S"
        s["code"] = f"{prefix}{num:02d}{suffix}"

    return am_shifts + pm_shifts


def generate_driver_shifts(filtered_flights: pd.DataFrame, weekday: str) -> pd.DataFrame:
    """
    Pipeline completa:
    - da voli filtrati → corse bus → turni (lista dict) → DataFrame pronto per UI.
    """
    trips = build_bus_trips_from_flights(filtered_flights)
    if not trips:
        return pd.DataFrame()

    shifts_list = build_shifts_from_trips(trips, weekday)
    if not shifts_list:
        return pd.DataFrame()

    shifts_list = assign_shift_codes(shifts_list, weekday)

    rows = []
    for s in shifts_list:
        start_dt = s["shift_start_dt"]
        end_dt = s["shift_end_dt"]
        nastro_min = int(round(s["nastro_min"]))
        work_min = int(round(s["work_min"]))
        trip_count = s["trip_count"]
        code = s.get("code", "")
        tipo_turno = "Intero"
        if nastro_min <= 180:
            tipo_turno = "Supplemento"
        elif nastro_min < 360:
            tipo_turno = "Part-time"

        rows.append(
            {
                "Codice turno": code,
                "Tipo giorno": WEEKDAY_LABELS_IT.get(weekday, weekday),
                "Tipo turno": tipo_turno,
                "Data (esempio)": s["dates_covered"][0].strftime("%d/%m/%Y"),
                "Inizio turno": start_dt.strftime("%H:%M"),
                "Fine turno": end_dt.strftime("%H:%M"),
                "Durata nastro (min)": nastro_min,
                "Durata lavoro (min)": work_min,
                "Corse servite": trip_count,
                "Dettaglio corse": s["detail"],
            }
        )

    df_shifts = pd.DataFrame(rows)
    df_shifts = df_shifts.sort_values(["Inizio turno", "Codice turno"]).reset_index(drop=True)
    return df_shifts


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
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Titolo
    st.title("✈️ Flight Matrix")

    # Intro card
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

    # Parsing
    with st.spinner("Parsing del PDF in corso..."):
        flights_df = parse_pdf_to_flights_df(uploaded_file)

    if flights_df.empty:
        st.error("Non sono stati trovati voli PAX o la struttura del PDF non è riconosciuta.")
        return

    # Metriche globali
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

    # Giorni effettivamente presenti
    weekdays_present = sorted(
        flights_df["Weekday"].unique(),
        key=lambda x: WEEKDAY_ORDER.index(x),
    )

    # Sidebar
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

    # Dati per la tipologia di giorno selezionata
    label_it = WEEKDAY_LABELS_IT.get(selected_weekday, selected_weekday)
    weekday_ops = flights_df[flights_df["Weekday"] == selected_weekday]
    weekday_flights_count = len(weekday_ops)
    weekday_dates_count = weekday_ops["Date"].nunique()

    # Badge giorno
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

    # Numero voli per tipologia di giorno
    st.markdown(
        f"**Voli PAX per questo tipo di giorno:** {weekday_flights_count} "
        f"(su {weekday_dates_count} {label_it.lower()} nel periodo caricato)"
    )

    # Legend arrivi/partenze
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

    # Applica i filtri alla matrice
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
        # Rinomina colonne per la visualizzazione
        display_df = matrix_filtered.rename(
            columns={"Flight": "Codice Volo", "Route": "Aeroporto"}
        )

        # Stile AD + orari
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

        # Export CSV matrice
        csv_buffer = display_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Scarica matrice in CSV",
            data=csv_buffer,
            file_name=f"flight_matrix_{label_it.lower()}.csv",
            mime="text/csv",
        )

        # --------- GENERAZIONE TURNI GUIDA ---------
        st.markdown("### Turni guida generati")

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
                shifts_df = generate_driver_shifts(flights_for_turns, selected_weekday)
                if shifts_df.empty:
                    st.warning("Non è stato possibile generare turni compatibili con i vincoli.")
                else:
                    st.dataframe(shifts_df, use_container_width=True)
                    csv_shifts = shifts_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇️ Scarica turni guida in CSV",
                        data=csv_shifts,
                        file_name=f"turni_{label_it.lower()}.csv",
                        mime="text/csv",
                    )

    # --------- GRAFICO ARRIVI/PARTENZE PER GIORNO ---------
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
