import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import re
from sklearn.neighbors import BallTree
import logging

# Konfigurasi logging agar tidak terlalu verbose saat startup
logging.basicConfig(level=logging.WARNING)

# === Load data & Global Variables (Minimal) ===
try:
    df = pd.read_csv("data/combined/combined.csv", parse_dates=['time'])
except FileNotFoundError:
    print("Warning: 'data/combined/combined.csv' not found. Creating dummy data.")
    # Dummy data for demonstration if file is missing
    data = {
        'time': pd.to_datetime(['2025-10-15T12:00:00Z', '2025-10-16T08:30:00Z', '2024-05-20T10:00:00Z', '2023-01-01T00:00:00Z', '2025-10-14T11:00:00Z']),
        'latitude': [-6.2088, -7.7956, -8.4095, 0.7893, -6.9034],
        'longitude': [106.8456, 110.3695, 115.1889, 113.9213, 107.6191],
        'depth': [10.0, 50.5, 12.3, 150.0, 20.0],
        'magnitude': [5.5, 4.2, 6.1, 7.0, 3.5],
        'place': ['8km S of Jakarta', 'Yogyakarta Region', 'Bali', 'Kalimantan Tengah', 'Bandung'],
    }
    df = pd.DataFrame(data)

# Load clustering data
has_clusters = False
try:
    df_clustered = pd.read_excel("data/best_df_indonesia_cluster.xlsx")
    # Samakan tipe datetime sebelum merge
    df['time'] = pd.to_datetime(df['time'], utc=True, errors='coerce')
    df_clustered['time'] = pd.to_datetime(df_clustered['time'], utc=True, errors='coerce')

    # Add cluster column to main df if not exists
    if 'cluster' not in df.columns:
        # Merge by matching coordinates and time (adjust based on your data structure)
        df = df.merge(
            df_clustered[['time', 'latitude', 'longitude', 'cluster']],
            on=['time', 'latitude', 'longitude'],
            how='left'
        )
    has_clusters = True
    print(f"✓ Clustering data loaded: {len(df_clustered)} records, {df_clustered['cluster'].nunique()} clusters")
except FileNotFoundError:
    print("Warning: 'data/df_indonesia_cluster.xlsx' not found. Clustering features disabled.")
    df['cluster'] = -1  # Default cluster jika tidak ada data clustering
    has_clusters = False

# --- Deteksi provinsi Indonesia ---
try:
    worldcities = pd.read_csv("data/worldcities.csv")
    indo = worldcities[worldcities["country"] == "Indonesia"].copy()
    indo_coords = np.radians(indo[["lat", "lng"]].values)
    tree = BallTree(indo_coords, metric="haversine")
    
    def detect_province_fast(lat, lon):
        if pd.isna(lat) or pd.isna(lon):
            return "Lainnya"
        dist, idx = tree.query(np.radians([[lat, lon]]), k=1)
        nearest = indo.iloc[idx[0][0]]
        if dist[0][0] * 6371 < 150: 
            return str(nearest["admin_name"]).replace("Province", "").strip()
        return "Lainnya"
    
    df["province"] = df.apply(lambda r: detect_province_fast(r["latitude"], r["longitude"]), axis=1)

except FileNotFoundError:
    print("Warning: 'data/worldcities.csv' not found. Using simple place matching.")
    # Fallback province detection
    def detect_province_fast_fallback(lat, lon):
        if lat < -5 and lon < 110: return "Sumatera/Jawa Barat"
        if lat > -1 and lon > 120: return "Sulawesi/Maluku"
        return "Lainnya"
    df["province"] = df.apply(lambda r: detect_province_fast_fallback(r["latitude"], r["longitude"]), axis=1)

# Pastikan kolom time bertipe datetime dengan timezone aware
df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")

# ======================================================================
#                FEATURE ENGINEERING: KATEGORI EXPLORASI
# ======================================================================

# 1. Kategori waktu kejadian gempa (time of day)
df["hour"] = df["time"].dt.hour

time_labels = [
    "Dini Hari (00.00–05.59)",
    "Pagi (06.00–11.59)",
    "Siang (12.00–14.59)",
    "Sore (15.00–17.59)",
    "Malam (18.00–23.59)"
]

time_conditions = [
    (df["hour"] >= 0) & (df["hour"] <= 5),
    (df["hour"] >= 6) & (df["hour"] <= 11),
    (df["hour"] >= 12) & (df["hour"] <= 14),
    (df["hour"] >= 15) & (df["hour"] <= 17),
    (df["hour"] >= 18) & (df["hour"] <= 23),
]

df["time_of_day"] = np.select(time_conditions, time_labels, default="Unknown")

# 2. Kategori magnitudo
mag_labels = [
    "Minor (3.0–3.9)",
    "Light (4.0–4.9)",
    "Moderate (5.0–5.9)",
    "Strong (6.0–6.9)",
    "Major (7.0–7.9)",
    "Great (≥8.0)"
]

mag_conditions = [
    (df["magnitude"] >= 3.0) & (df["magnitude"] < 4.0),
    (df["magnitude"] >= 4.0) & (df["magnitude"] < 5.0),
    (df["magnitude"] >= 5.0) & (df["magnitude"] < 6.0),
    (df["magnitude"] >= 6.0) & (df["magnitude"] < 7.0),
    (df["magnitude"] >= 7.0) & (df["magnitude"] < 8.0),
    (df["magnitude"] >= 8.0),
]

df["mag_category"] = np.select(mag_conditions, mag_labels, default="<3.0 (Micro)")

# 3. Kategori kedalaman
depth_labels = [
    "Shallow-focus (<60 km)",
    "Intermediate-depth (60–300 km)",
    "Deep-focus (>300 km)"
]

depth_conditions = [
    df["depth"] < 60,
    (df["depth"] >= 60) & (df["depth"] <= 300),
    df["depth"] > 300,
]

df["depth_category"] = np.select(depth_conditions, depth_labels, default="Unknown")

# 4. Kategori musim Indonesia
# Musim hujan: Oktober–Maret, Musim kemarau: April–September
df["month"] = df["time"].dt.month

season_conditions = [
    df["month"].isin([10, 11, 12, 1, 2, 3])
]
season_labels = [
    "Musim Hujan (Okt–Mar)"
]

df["season"] = np.select(season_conditions, season_labels, default="Musim Kemarau (Apr–Sep)")

# ======================================================================
#                     PRE-CALCULATION & CONSTANTS
# ======================================================================
valid_provinces = df[df["province"] != "Lainnya"]['province'].unique()
top_province = (
    df[df["province"] != "Lainnya"]["province"].value_counts().idxmax()
    if len(valid_provinces) > 0 else 'Lainnya'
)

min_mag_data, max_mag_data = df['magnitude'].min(), df['magnitude'].max()
min_year_data, max_year_data = df['time'].dt.year.min(), df['time'].dt.year.max()
default_years_selection = sorted(df['time'].dt.year.unique(), reverse=True)[:5] # 5 tahun terakhir
default_start_year = default_years_selection[-1] if default_years_selection else min_year_data
default_end_year = default_years_selection[0] if default_years_selection else max_year_data
center_lat, center_lon = -2.5489, 118.0149 # Pusat Indonesia

# === Setup aplikasi ===
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)
server = app.server

app.title = "SeismoTrack - Earthquake Dashboard"

# ----------------------------------------------------------------------
#                         HELPER FUNCTION: Data Filtering
# ----------------------------------------------------------------------
def filter_data(provinces_input, mag_range, years, start_year, end_year):
    """Fungsi pembantu untuk memfilter DataFrame berdasarkan semua input."""
    
    # 1. Handle Province Default
    if not provinces_input:
        provinces = [top_province] if top_province != 'Lainnya' else df['province'].unique().tolist()
    else:
        provinces = provinces_input
        
    # 2. Handle Year Filter dengan validasi yang lebih baik
    all_years = df["time"].dt.year
    year_filter = None
    
    # Cek apakah ada input year range yang valid
    has_valid_range = (
        start_year is not None and 
        end_year is not None and 
        start_year >= min_year_data and 
        end_year <= max_year_data and 
        start_year <= end_year
    )
    
    if years and len(years) > 0:
        # Prioritas 1: Multiple years selection (hanya jika ada isinya)
        year_filter = all_years.isin(years)
    elif has_valid_range:
        # Prioritas 2: Year Range input (jika valid)
        year_filter = all_years.between(start_year, end_year)
    else:
        # Default: 5 tahun terakhir
        year_filter = all_years.between(max_year_data - 4, max_year_data)

    # 3. Main Filter
    dff = df[
        (df["magnitude"].between(mag_range[0], mag_range[1])) &
        (year_filter) &
        (df["province"].isin(provinces))
    ].sort_values("time", ascending=False)
    
    return dff, provinces


# ======================================================================
#                            CUSTOM CSS
# ======================================================================
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            }
            .sidebar {
                background: white;
                border-radius: 20px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.07);
                min-height: 95vh;
            }
            .sidebar h3 {
                color: #ff6b35;
                font-weight: 700;
                font-size: 1.5rem;
            }
            .nav-link {
                border-radius: 12px;
                margin-bottom: 8px;
                color: #64748b;
                font-weight: 500;
                transition: all 0.3s ease;
            }
            .nav-link:hover {
                background: #fff4f0;
                color: #ff6b35;
                transform: translateX(5px);
            }
            .nav-link.active {
                background: linear-gradient(135deg, #ff6b35 0%, #ff8c42 100%);
                color: white !important;
                box-shadow: 0 4px 10px rgba(255,107,53,0.3);
            }
            .main-content {
                background: transparent;
            }
            .welcome-header {
                background: white;
                border-radius: 20px;
                padding: 30px;
                margin-bottom: 30px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.05);
                border-left: 5px solid #ff6b35;
            }
            .welcome-header h2 {
                color: #1e293b;
                font-weight: 700;
                margin-bottom: 8px;
            }
            .welcome-header p {
                color: #64748b;
                margin: 0;
            }
            .stat-card-modern {
                background: white;
                border-radius: 20px;
                padding: 15px 12px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.05);
                transition: transform 0.3s ease, box-shadow 0.3s ease;
                height: 100%;
                border-left: 4px solid #ff6b35;
                text-align: center;
            }
            .stat-card-modern:hover {
                transform: translateY(-5px);
                box-shadow: 0 8px 15px rgba(0,0,0,0.1);
            }
            .stat-value {
                font-size: 1.8rem;
                font-weight: 700;
                color: #ff6b35;
                margin: 0 0 10px 0;
            }
            .stat-label {
                color: #64748b;
                font-size: 0.9rem;
                font-weight: 500;
                margin: 0;
            }
            .filter-section {
                background: white;
                border-radius: 20px;
                padding: 30px;
                margin-bottom: 30px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.05);
                border-top: 3px solid #ff6b35;
            }
            .filter-section h5 {
                color: #1e293b;
                font-weight: 700;
                margin-bottom: 25px;
            }
            .chart-container {
                background: white;
                border-radius: 20px;
                padding: 30px;
                margin-bottom: 30px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.05);
                border-top: 3px solid #ff6b35;
            }
            .chart-container h5 {
                color: #1e293b;
                font-weight: 700;
                margin-bottom: 20px;
            }
            .btn-reset {
                background: linear-gradient(135deg, #ff6b35 0%, #ff8c42 100%);
                border: none;
                border-radius: 12px;
                color: white;
                padding: 10px 20px;
                font-weight: 600;
                transition: all 0.3s ease;
            }
            .btn-reset:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 10px rgba(255,107,53,0.3);
            }
            .Select-control, .Select-menu-outer {
                border-radius: 12px !important;
            }
            .rc-slider-track {
                background: linear-gradient(to right, #ff6b35, #ff8c42) !important;
            }
            .rc-slider-handle {
                border-color: #ff6b35 !important;
            }
            .table-modern {
                border-radius: 15px;
                overflow: hidden;
            }
            .table-modern thead {
                background: linear-gradient(135deg, #ff6b35 0%, #ff8c42 100%);
                color: white;
            }
            .table-modern tbody tr:hover {
                background: #fff4f0;
            }
            .text-orange {
                color: #ff6b35;
            }
            .article-card {
                background: white;
                border-radius: 15px;
                padding: 0;
                margin-bottom: 20px;
                overflow: hidden;
                border: 1px solid #e2e8f0;
                border-top: 3px solid #ff6b35;
                transition: transform 0.3s ease, box-shadow 0.3s ease;
            }
            .article-card:hover {
                transform: translateY(-3px);
                box-shadow: 0 8px 20px rgba(255,107,53,0.15);
            }
            .article-image {
                width: 100%;
                height: 180px;
                object-fit: cover;
            }
            .article-content {
                padding: 20px;
            }
            .article-title {
                font-size: 1.1rem;
                font-weight: 700;
                color: #1e293b;
                margin-bottom: 10px;
                line-height: 1.4;
            }
            .article-desc {
                color: #64748b;
                font-size: 0.9rem;
                line-height: 1.6;
                margin-bottom: 15px;
            }
            .article-link {
                color: #ff6b35;
                font-weight: 600;
                text-decoration: none;
                font-size: 0.9rem;
            }
            .article-link:hover {
                color: #ff8c42;
                text-decoration: underline;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# ======================================================================
#                            SIDEBAR
# ======================================================================
sidebar = dbc.Col([
    html.Div([
        html.H3("🌍 SeismoTrack", className="mb-4"),
        dbc.Nav([
            dbc.NavLink("📊 Earthquake Overview", href="/overview", active="exact"),
            dbc.NavLink("🌐 Frequency & Depth Analysis", href="/analysis", active="exact"),
            dbc.NavLink("📍 Regional Summary", href="/regional", active="exact"),
            html.Hr(className="my-3"),
            dbc.NavLink("⚙️ Safety & Emergency", href="/settings", active="exact"),
            dbc.NavLink("❓ Help & Support", href="/help", active="exact"),
        ], vertical=True, pills=True),
    ])
], md=2, className="sidebar p-4")

# ======================================================================
#                            PAGE 1: Overview
# ======================================================================
overview_page = html.Div([
    # Welcome Header
    html.Div([
        html.H2("Welcome Back, ! Seismie", className="mb-2"),
        html.P("Explore today's earthquake updates and see what the Earth's been up to.", className="mb-0")
    ], className="welcome-header"),

    # --- Statistik Cards ---
    dbc.Row([
        dbc.Col([
            html.Div([
                html.Div(id="total-quakes", className="stat-value"),
                html.Div("Total Earthquakes", className="stat-label")
            ], className="stat-card-modern")
        ], md=3, className="mb-3"),

        dbc.Col([
            html.Div([
                html.Div(id="avg-mag", className="stat-value"),
                html.Div("Avg. Magnitude", className="stat-label")
            ], className="stat-card-modern")
        ], md=3, className="mb-3"),

        dbc.Col([
            html.Div([
                html.Div(id="deepest", className="stat-value"),
                html.Div("Deepest Earthquake", className="stat-label")
            ], className="stat-card-modern")
        ], md=3, className="mb-3"),

        dbc.Col([
            html.Div([
                html.Div(id="shallowest", className="stat-value"),
                html.Div("Shallowest Earthquake", className="stat-label")
            ], className="stat-card-modern")
        ], md=3, className="mb-3"),
    ]),

    # --- Filter Section ---
    html.Div([
        html.Div([
            html.H5("🔍 Filter Options", className="mb-0"),
            html.Button("🔄 Reset View", id="reset-view", n_clicks=0, className="btn-reset")
        ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "25px"}),

        dbc.Row([
            dbc.Col([
                html.Label("Regional (Province)", className="fw-semibold mb-2", style={"color": "#64748b"}),
                dcc.Dropdown(
                    id='province-filter',
                    options=[{'label': p, 'value': p} for p in sorted(df['province'].unique())],
                    value=[top_province] if top_province != 'Lainnya' else [], 
                    multi=True,
                    placeholder="Select provinces...",
                    style={"borderRadius": "12px"}
                ),
            ], md=6),

            dbc.Col([
                html.Label("Magnitude Range", className="fw-semibold mb-2", style={"color": "#64748b"}),
                dcc.RangeSlider(
                    id='mag-filter',
                    min=float(min_mag_data), max=float(max_mag_data), step=0.1,
                    marks={i: str(i) for i in range(int(min_mag_data), int(max_mag_data) + 1)},
                    value=[float(min_mag_data), float(max_mag_data)]
                ),
            ], md=6),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col([
                html.Label("Select Years (Multi-select)", className="fw-semibold mb-2", style={"color": "#64748b"}),
                dcc.Dropdown(
                    id='year-filter',
                    options=[
                        {'label': str(int(y)), 'value': int(y)}
                        for y in sorted(df['time'].dt.year.dropna().unique(), reverse=True)
                    ],
                    value=[], 
                    multi=True,
                    placeholder="Select years (or use range below)...",
                    style={"borderRadius": "12px"}
                ),
            ], md=6),
            
            dbc.Col([
                html.Label("Or Year Range", className="fw-semibold mb-2", style={"color": "#64748b"}),
                html.Div([
                    dcc.Input(
                        id='start-year',
                        type='number',
                        placeholder=f'Start ({int(min_year_data)})',
                        min=int(min_year_data), max=int(max_year_data), step=1,
                        value=int(default_start_year),
                        style={'width': '48%', 'marginRight': '4%', 'borderRadius': '12px', 'border': '1px solid #e2e8f0', 'padding': '8px'}
                    ),
                    dcc.Input(
                        id='end-year',
                        type='number',
                        placeholder=f'End ({int(max_year_data)})',
                        min=int(min_year_data), max=int(max_year_data), step=1,
                        value=int(default_end_year),
                        style={'width': '48%', 'borderRadius': '12px', 'border': '1px solid #e2e8f0', 'padding': '8px'}
                    )
                ], style={'display': 'flex'})
            ], md=6),
        ]),
    ], className="filter-section"),

    # --- Map Section ---
    html.Div([
        html.H5("🗺️ Earthquake Distribution Map"),
        dcc.Graph(
            id="map-graph", 
            style={"height": "500px"},
            config={
                'doubleClick': False,
                'scrollZoom': True,
                'displayModeBar': True,
                'modeBarButtonsToRemove': ['lasso2d', 'select2d']
            }
        ),
    ], className="chart-container"),

    # --- Recent Earthquakes ---
    html.Div([
        html.Div([
            html.H5("📋 Filtered Earthquake Data", className="mb-0"),
            html.Button("⬇️ Download Data", id="download-btn", className="btn-reset")
        ], style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "20px"}),
        html.Div(id="recent-table"),
        dcc.Download(id="download-data")
    ], className="chart-container")
])

# ======================================================================
#                            OTHER PAGES
# ======================================================================
analysis_page = html.Div([
    html.Div([
        html.H2("Frequency & Depth Analysis", className="mb-2"),
        html.P("Analisis distribusi magnitudo, kedalaman, waktu kejadian, musim, dan pola clustering gempa di Indonesia.", className="mb-0")
    ], className="welcome-header"),
    
    # Row 1: Histogram dan Scatter
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H5("📊 Magnitude Distribution"),
                dcc.Graph(id="magnitude-histogram")
            ], className="chart-container")
        ], md=6),
        dbc.Col([
            html.Div([
                html.H5("📈 Magnitude vs Depth Correlation"),
                dcc.Graph(id="mag-depth-scatter")
            ], className="chart-container")
        ], md=6),
    ], className="mb-4"),
    
    # Row 2: DBSCAN Clustering Results
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H5("🔍 DBSCAN Clustering - Geographic Distribution"),
                
                # === FILTER CLUSTER ===
                html.Label("Filter Cluster:"),
                dcc.Dropdown(
                    id="cluster-filter",
                    placeholder="Pilih cluster...",
                    clearable=True
                ),
                html.Br(),

                dcc.Graph(id="cluster-map")
            ], className="chart-container")
        ], md=8),

        dbc.Col([
            html.Div([
                html.H5("📊 Cluster Statistics"),
                html.Div(id="cluster-stats")
            ], className="chart-container")
        ], md=4),
    ], className="mb-4"),
    
    # Row 3: Exploratory Dashboard (Time, Magnitude, Depth, Season)
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H5("⏰ Distribusi Gempa per Waktu Kejadian"),
                dcc.Graph(id="time-of-day-bar")
            ], className="chart-container")
        ], md=6),
        dbc.Col([
            html.Div([
                html.H5("💥 Distribusi Kategori Kekuatan Gempa"),
                dcc.Graph(id="mag-category-bar")
            ], className="chart-container")
        ], md=6),
    ], className="mb-4"),
    
    dbc.Row([
        dbc.Col([
            html.Div([
                html.H5("🌋 Distribusi Kategori Kedalaman Gempa"),
                dcc.Graph(id="depth-category-bar")
            ], className="chart-container")
        ], md=6),
        dbc.Col([
            html.Div([
                html.H5("🌧️ Distribusi Gempa per Musim di Indonesia"),
                dcc.Graph(id="season-bar")
            ], className="chart-container")
        ], md=6),
    ])
])

regional_page = html.Div([
    html.Div([
        html.H2("Regional Summary", className="mb-2"),
        html.P("Lihat ringkasan aktivitas gempa per provinsi dan pola klasternya.", className="mb-0")
    ], className="welcome-header"),
    
    html.Div([
        html.H5("📍 Average Magnitude by Province"),
        dcc.Graph(
            figure=px.bar(
                df.groupby("province")["magnitude"].mean().reset_index().sort_values("magnitude", ascending=False),
                x="province", y="magnitude", color="magnitude", color_continuous_scale="OrRd",
                title=""
            )
        )
    ], className="chart-container")
])

settings_page = html.Div([
    html.Div([
        html.H2("⚙️ Earthquake Safety & Emergency Info", className="mb-2"),
        html.P("Panduan keselamatan saat gempa dan lokasi posko pengungsian terdekat.", className="mb-0")
    ], className="welcome-header"),
    
    dbc.Row([
        # Tips Section with Article Links
        dbc.Col([
            html.Div([
                html.H5("🚨 Tips Keselamatan Saat Gempa", className="mb-3"),
                
                # Input untuk menambah artikel
                html.Div([
                    html.H6("📎 Tambah Artikel Referensi:", className="fw-bold mb-2"),
                    dbc.Row([
                        dbc.Col([
                            dcc.Input(
                                id='article-title-input',
                                type='text',
                                placeholder='Judul Artikel',
                                style={'width': '100%', 'borderRadius': '12px', 'border': '1px solid #e2e8f0', 'padding': '10px', 'marginBottom': '10px'}
                            ),
                        ], md=12),
                        dbc.Col([
                            dcc.Input(
                                id='article-url-input',
                                type='url',
                                placeholder='https://example.com/artikel-gempa',
                                style={'width': '100%', 'borderRadius': '12px', 'border': '1px solid #e2e8f0', 'padding': '10px', 'marginBottom': '10px'}
                            ),
                        ], md=12),
                        dbc.Col([
                            dcc.Input(
                                id='article-image-input',
                                type='url',
                                placeholder='URL Gambar (opsional)',
                                style={'width': '100%', 'borderRadius': '12px', 'border': '1px solid #e2e8f0', 'padding': '10px', 'marginBottom': '10px'}
                            ),
                        ], md=12),
                        dbc.Col([
                            dcc.Textarea(
                                id='article-desc-input',
                                placeholder='Deskripsi singkat artikel (opsional, max 150 karakter)',
                                style={'width': '100%', 'borderRadius': '12px', 'border': '1px solid #e2e8f0', 'padding': '10px', 'marginBottom': '10px', 'minHeight': '60px', 'resize': 'vertical'}
                            ),
                        ], md=12),
                        dbc.Col([
                            html.Button("➕ Tambah Artikel", id="add-article-btn", className="btn-reset", style={'width': '100%'}),
                        ], md=12),
                    ]),
                    html.Div(id="article-feedback", className="small text-success mt-2")
                ], className="mb-4 p-3", style={'background': '#f8f9fa', 'borderRadius': '12px'}),
                
                # Daftar artikel yang tersimpan
                html.Div([
                    html.H6("📚 Artikel Referensi:", className="fw-bold mb-3"),
                    html.Div(id="articles-list")
                ], className="mb-4", style={"lineHeight": "1.8", "maxHeight": "400px", "overflowY": "auto", "paddingRight": "10px"}),
                
                html.Hr(),
                
                # Tips standar
                html.Div([
                    html.Div([
                        html.H6("1️⃣ Saat Di Dalam Ruangan:", className="text-orange fw-bold mb-2"),
                        html.Ul([
                            html.Li("DROP - Jatuhkan diri ke lantai"),
                            html.Li("COVER - Berlindung di bawah meja yang kuat"),
                            html.Li("HOLD ON - Pegang kaki meja sampai guncangan berhenti"),
                            html.Li("Jauhi jendela, kaca, dan benda yang bisa jatuh"),
                            html.Li("Jangan menggunakan lift saat evakuasi"),
                        ], className="mb-3"),
                        
                        html.H6("2️⃣ Saat Di Luar Ruangan:", className="text-orange fw-bold mb-2"),
                        html.Ul([
                            html.Li("Jauhi bangunan, tiang listrik, dan pohon"),
                            html.Li("Cari tempat terbuka dan aman"),
                            html.Li("Jika di kendaraan, berhenti di tempat aman"),
                            html.Li("Tetap di dalam kendaraan sampai guncangan berhenti"),
                        ], className="mb-3"),
                        
                        html.H6("3️⃣ Setelah Gempa:", className="text-orange fw-bold mb-2"),
                        html.Ul([
                            html.Li("Periksa kondisi diri dan orang sekitar"),
                            html.Li("Waspada terhadap gempa susulan"),
                            html.Li("Keluar dari bangunan jika ada kerusakan struktural"),
                            html.Li("Dengarkan informasi dari radio atau TV"),
                            html.Li("Hubungi keluarga melalui SMS (jangan telepon)"),
                        ]),
                    ], style={"lineHeight": "1.8", "maxHeight": "200px", "overflowY": "auto", "paddingRight": "10px"})
                ])
            ], className="chart-container")
        ], md=6),
        
        # Map Section with Input
        dbc.Col([
            html.Div([
                html.H5("🏕️ Posko Pengungsian Terdekat", className="mb-3"),
                
                # Input untuk menambah posko
                html.Div([
                    html.H6("📍 Tambah Posko Baru:", className="fw-bold mb-2"),
                    dbc.Row([
                        dbc.Col([
                            dcc.Input(
                                id='posko-name-input',
                                type='text',
                                placeholder='Nama Posko (contoh: SDN Jakarta 1)',
                                style={'width': '100%', 'borderRadius': '12px', 'border': '1px solid #e2e8f0', 'padding': '10px', 'marginBottom': '10px'}
                            ),
                        ], md=12),
                        dbc.Col([
                            dcc.Input(
                                id='posko-gmaps-input',
                                type='text',
                                placeholder='Link Google Maps (contoh: https://maps.app.goo.gl/xxx)',
                                style={'width': '100%', 'borderRadius': '12px', 'border': '1px solid #e2e8f0', 'padding': '10px', 'marginBottom': '10px'}
                            ),
                        ], md=9),
                        dbc.Col([
                            html.Button("➕ Tambah", id="add-posko-btn", className="btn-reset", style={'width': '100%'}),
                        ], md=3),
                    ]),
                    html.Div(id="posko-feedback", className="small mt-2")
                ], className="mb-3 p-3", style={'background': '#f8f9fa', 'borderRadius': '12px'}),
                
                # Peta
                dcc.Graph(
                    id="evacuation-map",
                    config={'displayModeBar': False},
                    style={"height": "400px"}
                ),
                
                # Daftar posko
                html.Div([
                    html.H6("📍 Daftar Posko:", className="fw-bold mt-3 mb-2"),
                    html.Div(id="posko-list")
                ])
            ], className="chart-container")
        ], md=6),
    ])
])

help_page = html.Div(
    style={"backgroundColor": "#e9eff6", "padding": "30px 0"},
    children=[
        dbc.Container([

            # HEADER
            html.Div([
                html.H2("Help & Support", 
                        className="mb-1", 
                        style={"fontWeight": "700", "color": "#0d1b2a"}),

                html.P(
                    "Panduan penggunaan dashboard, pusat bantuan, dan kontak layanan teknis.",
                    className="text-muted",
                    style={"marginBottom": "0"}
                ),
            ],
            style={
                "backgroundColor": "white",
                "padding": "25px 35px",
                "borderRadius": "18px",
                "borderLeft": "8px solid #ff7a35",
                "boxShadow": "0 2px 6px rgba(0,0,0,0.05)",
                "marginBottom": "30px"
            }),

            # PANDUAN
            dbc.Card([
                dbc.CardHeader("Panduan Penggunaan Dashboard", className="fw-bold"),
                dbc.CardBody([
                    html.P("""
                        Dashboard ini dirancang untuk analisis gempa Indonesia, termasuk visualisasi lokasi,
                        magnitudo, kedalaman, dan clustering DBSCAN. Gunakan sidebar untuk navigasi antar halaman
                        dan dropdown filter untuk menyesuaikan tampilan analisis.
                    """)
                ])
            ], className="shadow-sm mb-4"),

            # CUSTOMER SERVICE
            dbc.Card([
                dbc.CardHeader("Pusat Bantuan & Customer Service", className="fw-bold"),
                dbc.CardBody([
                    html.P("Hubungi layanan berikut jika Anda membutuhkan bantuan:", className="mb-2"),
                    html.Ul([
                        html.Li("Email Dukungan Teknis: support.dashboard@example.com"),
                        html.Li("Hotline CS (08.00 – 17.00 WIB): +62 812-3456-7890"),
                        html.Li("Live Chat Bantuan: https://help-dummy.example.com/chat"),
                        html.Li("Dokumentasi Sistem: https://help-dummy.example.com/docs"),
                    ])
                ])
            ], className="shadow-sm mb-4"),

            # FAQ
            dbc.Card([
                dbc.CardHeader("FAQ – Pertanyaan yang Sering Diajukan", className="fw-bold"),
                dbc.CardBody([
                    html.Ul([
                        html.Li("Cluster tidak muncul → Pastikan file dataset mengandung kolom 'cluster'."),
                        html.Li("Peta tidak tampil → Pastikan koneksi internet stabil karena Mapbox butuh akses online."),
                        html.Li("Ingin mengganti dataset → Upload file baru pada direktori /data."),
                    ])
                ])
            ], className="shadow-sm mb-5"),

        ])
    ]
)

# ======================================================================
#                            ROUTING
# ======================================================================
app.layout = dbc.Container([
    dcc.Location(id='url'),
    dbc.Row([
        sidebar,
        dbc.Col(html.Div(id='page-content'), md=10, className="main-content p-4")
    ])
], fluid=True, style={"padding": "20px"})


@app.callback(Output('page-content', 'children'), Input('url', 'pathname'))
def display_page(pathname):
    if pathname in ['/', '/overview']: 
        return overview_page
    elif pathname == '/analysis': 
        return analysis_page
    elif pathname == '/regional': 
        return regional_page
    elif pathname == '/settings': 
        return settings_page
    elif pathname == '/help': 
        return help_page
    else: 
        return html.Div([
            html.Div([
                html.H2("404 - Page Not Found", className="text-danger mb-2"),
                html.P("Halaman yang Anda cari tidak ditemukan.", className="mb-0")
            ], className="welcome-header")
        ])

# ======================================================================
#                            CALLBACK UTAMA
# ======================================================================
@app.callback(
    Output("total-quakes", "children"),
    Output("avg-mag", "children"),
    Output("deepest", "children"),
    Output("shallowest", "children"),
    Output("map-graph", "figure"),
    Output("recent-table", "children"),

    Input("province-filter", "value"),
    Input("mag-filter", "value"),
    Input("year-filter", "value"),
    Input("start-year", "value"),
    Input("end-year", "value"),
    Input("map-graph", "clickData"),
    Input("reset-view", "n_clicks"),
)
def update_dashboard(provinces_input, mag_range, years, start_year, end_year, clickData, n_clicks):
    
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None
    
    # 1. FILTER DATA
    dff, current_provinces = filter_data(provinces_input, mag_range, years, start_year, end_year)

    # 2. CALCULATE STATISTICS
    total_quakes = len(dff)
    avg_mag = f"{dff['magnitude'].mean():.2f}" if total_quakes else "0.00"
    deepest = f"{dff['depth'].max():.1f} km" if total_quakes else "0.0 km"
    shallowest = f"{dff['depth'].min():.1f} km" if total_quakes else "0.0 km"

    # 3. MAP VIEW LOGIC (SELALU TITIK DETAIL)
    lat_center_view, lon_center_view, zoom_level = center_lat, center_lon, 3.5 

    if not dff.empty:
        data_lat_center = dff["latitude"].mean()
        data_lon_center = dff["longitude"].mean()
        
        if total_quakes > 500:
            zoom_level_data = 4.0
        elif total_quakes > 100:
            zoom_level_data = 5.0
        elif total_quakes > 20:
            zoom_level_data = 6.0
        else:
            zoom_level_data = 7.0
        
        if triggered_id == "map-graph" and isinstance(clickData, dict) and 'points' in clickData:
            point = clickData["points"][0]
            lat_center_view, lon_center_view = point["lat"], point["lon"]
            zoom_level = 7.5  # zoom detail ke titik
        elif triggered_id == "reset-view":
            lat_center_view, lon_center_view = data_lat_center, data_lon_center
            zoom_level = zoom_level_data
        elif triggered_id in ["province-filter", "mag-filter", "year-filter", "start-year", "end-year"] or triggered_id is None:
            lat_center_view, lon_center_view = data_lat_center, data_lon_center
            zoom_level = zoom_level_data

    # Create Map - SELALU TITIK DETAIL
    fig_map = px.scatter_mapbox(
        dff,
        lat="latitude",
        lon="longitude",
        color="magnitude",
        size="magnitude",
        hover_name="place",
        hover_data={
            "depth": True,
            "time": True,
            "province": True,
            "latitude": ':.2f',
            "longitude": ':.2f',
            "magnitude": True
        },
        color_continuous_scale="OrRd",
        zoom=zoom_level,
        center={"lat": lat_center_view, "lon": lon_center_view},
        height=500,
    )

    fig_map.update_layout(
        mapbox_style="open-street-map",
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        dragmode='pan'
    )

    # 4. CREATE TABLE - Show ALL filtered data with scroll
    if dff.empty:
        table = html.P(
            "No earthquake data available for the selected filters.", 
            className="text-muted text-center p-4"
        )
    else:
        display_df = dff[["time", "place", "magnitude", "depth", "province"]].copy()
        display_df['time'] = display_df['time'].dt.strftime('%Y-%m-%d %H:%M') 
        display_df['depth'] = display_df['depth'].apply(lambda x: f"{x:.1f} km")
        display_df.columns = ["Time", "Location", "Magnitude", "Depth", "Province"]
        
        table = html.Div([
            html.P(
                f"Showing all {len(dff)} filtered earthquakes", 
                className="text-muted small mb-2"
            ),
            html.Div([
                dbc.Table.from_dataframe(
                    display_df,
                    striped=True,
                    bordered=False,
                    hover=True,
                    className="table-modern mb-0"
                )
            ], style={
                'maxHeight': '500px', 
                'overflowY': 'auto',
                'overflowX': 'auto',
                'border': '1px solid #e2e8f0',
                'borderRadius': '12px'
            })
        ])

    return total_quakes, avg_mag, deepest, shallowest, fig_map, table


# Download Callback
@app.callback(
    Output("download-data", "data"),
    Input("download-btn", "n_clicks"),
    State("province-filter", "value"),
    State("mag-filter", "value"),
    State("year-filter", "value"),
    State("start-year", "value"),
    State("end-year", "value"),
    prevent_initial_call=True
)
def download_filtered_data(n_clicks, provinces_input, mag_range, years, start_year, end_year):
    if n_clicks:
        dff, _ = filter_data(provinces_input, mag_range, years, start_year, end_year)
        return dcc.send_data_frame(dff.to_csv, "filtered_earthquake_data.csv", index=False)


# Evacuation Map Callback with Dynamic Data
@app.callback(
    Output("evacuation-map", "figure"),
    Output("posko-list", "children"),
    Output("posko-feedback", "children"),
    Output("posko-feedback", "className"),
    Input("add-posko-btn", "n_clicks"),
    State("posko-name-input", "value"),
    State("posko-gmaps-input", "value"),
    State("evacuation-map", "figure"),
    prevent_initial_call=False
)
def update_evacuation_map(n_clicks, name, gmaps_link, current_fig):
    # Initialize dengan data dummy
    if 'evacuation_data' not in globals():
        global evacuation_data
        evacuation_data = pd.DataFrame({
            'name': [
                'SDN Surabaya 1',
                'GOR Kertajaya', 
                'Masjid Al-Akbar',
                'Lapangan Manahan',
                'Balai Kota Surabaya'
            ],
            'lat': [-7.2575, -7.2875, -7.3305, -7.2658, -7.2697],
            'lon': [112.7521, 112.7417, 112.7277, 112.7378, 112.7508],
            'address': [
                'Jl. Diponegoro No. 123',
                'Jl. Kertajaya No. 45',
                'Jl. Raya Masjid No. 1',
                'Jl. Ahmad Yani No. 88',
                'Jl. Taman Surya No. 1'
            ]
        })
    
    feedback = ""
    feedback_class = "small mt-2"
    
    # Tambah posko baru jika tombol diklik
    ctx = dash.callback_context
    if ctx.triggered and ctx.triggered[0]["prop_id"] == "add-posko-btn.n_clicks":
        if name and gmaps_link:
            # Extract koordinat dari Google Maps link
            lat, lon = extract_coordinates_from_gmaps(gmaps_link)
            
            if lat and lon:
                new_posko = pd.DataFrame({
                    'name': [name],
                    'lat': [lat],
                    'lon': [lon],
                    'address': ['Dari Google Maps']
                })
                evacuation_data = pd.concat([evacuation_data, new_posko], ignore_index=True)
                feedback = f"✓ Posko '{name}' berhasil ditambahkan!"
                feedback_class = "small mt-2 text-success"
            else:
                feedback = "⚠️ Link Google Maps tidak valid atau koordinat tidak ditemukan"
                feedback_class = "small mt-2 text-warning"
        else:
            feedback = "⚠️ Mohon isi nama dan link Google Maps"
            feedback_class = "small mt-2 text-danger"
    
    # Buat peta
    fig = px.scatter_mapbox(
        evacuation_data,
        lat="lat",
        lon="lon",
        hover_name="name",
        hover_data={"address": True, "lat": False, "lon": False},
        zoom=12,
        center={"lat": evacuation_data['lat'].mean(), "lon": evacuation_data['lon'].mean()},
        height=400,
    )
    
    fig.update_traces(
        marker=dict(size=20, color='#ff6b35', symbol='marker'),
    )
    
    fig.update_layout(
        mapbox_style="open-street-map",
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    # Buat daftar posko
    posko_list = []
    for idx, row in evacuation_data.iterrows():
        posko_list.append(
            html.P(f"🏫 {row['name']} - {row['address']}", className="mb-2", style={"color": "#64748b"})
        )
    
    return fig, posko_list, feedback, feedback_class


def extract_coordinates_from_gmaps(url):
    """Extract koordinat dari berbagai format Google Maps URL"""
    try:
        # Format 1: @lat,lon atau /@lat,lon
        if '@' in url:
            coords = url.split('@')[1].split(',')[:2]
            lat, lon = float(coords[0]), float(coords[1])
            return lat, lon
        
        # Format 2: ?q=lat,lon
        if '?q=' in url:
            coords = url.split('?q=')[1].split(',')[:2]
            lat, lon = float(coords[0]), float(coords[1])
            return lat, lon
        
        # Format 3: /place/.../@lat,lon
        if '/place/' in url and '@' in url:
            coords = url.split('@')[1].split(',')[:2]
            lat, lon = float(coords[0]), float(coords[1])
            return lat, lon
            
    except Exception:
        pass
    
    return None, None


# Articles Management Callback
@app.callback(
    Output("articles-list", "children"),
    Output("article-feedback", "children"),
    Input("add-article-btn", "n_clicks"),
    State("article-title-input", "value"),
    State("article-url-input", "value"),
    State("article-image-input", "value"),
    State("article-desc-input", "value"),
    prevent_initial_call=False
)
def manage_articles(n_clicks, title, url, image_url, description):
    # Initialize artikel dummy dengan gambar dan deskripsi
    if 'articles' not in globals():
        global articles
        articles = [
            {
                'title': '10 Cara Menyelamatkan Diri dari Gempa Bumi yang Wajib Diketahui',
                'url': 'https://www.bmkg.go.id/gempabumi/panduan-gempa.bmkg',
                'image': 'https://images.unsplash.com/photo-1590859808308-3d2d9c515b1a?w=600&h=400&fit=crop',
                'description': 'Gempa bumi adalah bencana alam yang tidak dapat diprediksi. Kenali 10 langkah penting untuk menyelamatkan diri dan keluarga saat terjadi gempa bumi.'
            },
            {
                'title': 'Panduan Evakuasi Darurat untuk Keluarga',
                'url': 'https://www.bnpb.go.id/artikel/evakuasi-gempa',
                'image': 'https://images.unsplash.com/photo-1551836022-d5d88e9218df?w=600&h=400&fit=crop',
                'description': 'Persiapkan rencana evakuasi keluarga Anda. Artikel ini membahas langkah-langkah praktis untuk menghadapi situasi darurat gempa bumi dengan aman.'
            },
            {
                'title': 'Pertolongan Pertama untuk Korban Gempa',
                'url': 'https://www.pmi.or.id/p3k-gempa',
                'image': 'https://images.unsplash.com/photo-1584820927498-cfe5211fd8bf?w=600&h=400&fit=crop',
                'description': 'Pelajari teknik pertolongan pertama yang tepat untuk membantu korban gempa. Termasuk cara menangani luka, patah tulang, dan kondisi darurat lainnya.'
            },
            {
                'title': 'Membangun Rumah Tahan Gempa',
                'url': 'https://www.pu.go.id/rumah-tahan-gempa',
                'image': 'https://images.unsplash.com/photo-1503387762-592deb58ef4e?w=600&h=400&fit=crop',
                'description': 'Konstruksi bangunan yang tepat dapat menyelamatkan nyawa. Simak panduan membangun dan merenovasi rumah agar lebih tahan terhadap guncangan gempa.'
            }
        ]
    
    feedback = ""
    
    # Tambah artikel baru
    ctx = dash.callback_context
    if ctx.triggered and ctx.triggered[0]["prop_id"] == "add-article-btn.n_clicks":
        if title and url:
            new_article = {
                'title': title,
                'url': url,
                'image': image_url if image_url else 'https://images.unsplash.com/photo-1451187580459-43490279c0fa?w=600&h=400&fit=crop',
                'description': description if description else 'Baca artikel lengkap untuk informasi lebih detail tentang keselamatan gempa bumi.'
            }
            articles.append(new_article)
            feedback = f"✓ Artikel '{title}' berhasil ditambahkan!"
        else:
            feedback = "⚠️ Mohon isi minimal judul dan URL artikel"
    
    # Render daftar artikel dengan card style
    article_items = []
    for idx, article in enumerate(articles):
        article_card = html.Div([
            # Image
            html.Img(
                src=article['image'],
                className="article-image"
            ),
            # Content
            html.Div([
                html.Div(article['title'], className="article-title"),
                html.Div(article['description'], className="article-desc"),
                html.A(
                    "Baca Selengkapnya →",
                    href=article['url'],
                    target="_blank",
                    className="article-link"
                ),
            ], className="article-content")
        ], className="article-card")
        
        article_items.append(article_card)
    
    return article_items, feedback


# ======================================================================
#                    ANALYSIS PAGE CALLBACK (NEW CHARTS)
# ======================================================================
@app.callback(
    Output("magnitude-histogram", "figure"),
    Output("mag-depth-scatter", "figure"),
    Output("cluster-map", "figure"),
    Output("cluster-stats", "children"),
    Output("cluster-filter", "options"),
    Output("time-of-day-bar", "figure"),
    Output("mag-category-bar", "figure"),
    Output("depth-category-bar", "figure"),
    Output("season-bar", "figure"),
    Input("cluster-filter", "value"),
    Input("url", "pathname")
)
def update_analysis_page(selected_cluster, pathname):

    # Hanya aktif di /analysis
    if pathname != "/analysis":
        empty_fig = {}
        return empty_fig, empty_fig, empty_fig, "", [], empty_fig, empty_fig, empty_fig, empty_fig

    # ==============================
    # 1. Histogram Magnitude
    # ==============================
    fig_hist = px.histogram(
        df,
        x="magnitude",
        nbins=20,
        color_discrete_sequence=["#ff6b35"]
    )
    fig_hist.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        xaxis_title="Magnitude",
        yaxis_title="Frequency"
    )

    # ==============================
    # 2. Scatter Magnitude vs Depth
    # ==============================
    if has_clusters and "cluster" in df.columns:
        color_arg = "cluster"
    else:
        color_arg = "province"

    fig_scatter = px.scatter(
        df,
        x="magnitude",
        y="depth",
        color=color_arg,
        hover_data=["place", "time"]
    )
    fig_scatter.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Magnitude",
        yaxis_title="Depth (km)",
        yaxis_autorange="reversed"
    )

    # ==============================
    # 3. FILTER DATA CLUSTER
    # ==============================
    if selected_cluster is not None:
        df_map = df[df["cluster"] == selected_cluster].copy()
    else:
        df_map = df[df["cluster"] >= 0].copy()

    # ==============================
    # 4. MAPBOX CLUSTER MAP
    # ==============================
    if df_map.empty:
        fig_map = go.Figure()
        fig_map.update_layout(
            mapbox_style="open-street-map",
            mapbox=dict(center=dict(lat=center_lat, lon=center_lon), zoom=3.5),
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            paper_bgcolor='rgba(0,0,0,0)',
            annotations=[dict(
                text="Tidak ada data untuk cluster ini",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=0.5
            )]
        )
    else:
        fig_map = px.scatter_mapbox(
            df_map,
            lat="latitude",
            lon="longitude",
            color="cluster",
            size="magnitude",
            hover_name="place",
            zoom=4,
            center={
                "lat": df_map["latitude"].mean(),
                "lon": df_map["longitude"].mean()
            },
            height=500,
            color_continuous_scale="Viridis"
        )
        fig_map.update_layout(
            mapbox_style="open-street-map",
            margin={"r": 0, "t": 0, "l": 0, "b": 0},
            paper_bgcolor='rgba(0,0,0,0)'
        )

    # ==============================
    # 5. STATISTIK CLUSTER
    # ==============================
    if selected_cluster is not None:
        stats = html.Div([
            html.H5(f"Cluster {selected_cluster}"),
            html.P(f"Total earthquakes: {len(df_map)}"),
            html.P(f"Avg. Magnitude: {df_map['magnitude'].mean():.2f}" if not df_map.empty else "Avg. Magnitude: -"),
            html.P(f"Avg. Depth: {df_map['depth'].mean():.1f} km" if not df_map.empty else "Avg. Depth: -")
        ])
    else:
        stats = html.Div([
            html.H5("Semua Cluster"),
            html.P(f"Total earthquakes: {len(df_map)}")
        ])

    # ==============================
    # 6. FILTER DROPDOWN OPTIONS
    # ==============================
    clusters = sorted(df.loc[df["cluster"] >= 0, "cluster"].dropna().unique())
    options = [{"label": f"Cluster {int(c)}", "value": int(c)} for c in clusters]

    # ==============================
    # 7. EXPLORATORY CHARTS (WARNA DISESUAIKAN)
    # ==============================

    # a. Distribusi per Waktu Kejadian
    order_time = [
        "Dini Hari (00.00–05.59)",
        "Pagi (06.00–11.59)",
        "Siang (12.00–14.59)",
        "Sore (15.00–17.59)",
        "Malam (18.00–23.59)"
    ]
    time_counts = (
        df["time_of_day"]
        .value_counts()
        .reindex(order_time)
        .dropna()
        .reset_index()
    )
    time_counts.columns = ["Time of Day", "Count"]
    fig_time = px.bar(
        time_counts,
        x="Time of Day",
        y="Count",
        text="Count"
    )
    fig_time.update_traces(
        textposition="outside",
        marker_color="#ff6b35"
    )
    fig_time.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Waktu Kejadian",
        yaxis_title="Jumlah Gempa",
        xaxis_tickangle=-20,
        margin=dict(t=20, b=60, l=40, r=20)
    )

    # b. Distribusi Kategori Magnitudo
    order_mag = [
        "Minor (3.0–3.9)",
        "Light (4.0–4.9)",
        "Moderate (5.0–5.9)",
        "Strong (6.0–6.9)",
        "Major (7.0–7.9)",
        "Great (≥8.0)"
    ]
    mag_counts = (
        df[df["mag_category"].isin(order_mag)]["mag_category"]
        .value_counts()
        .reindex(order_mag)
        .dropna()
        .reset_index()
    )
    mag_counts.columns = ["Magnitude Category", "Count"]
    fig_mag_cat = px.bar(
        mag_counts,
        x="Magnitude Category",
        y="Count",
        text="Count"
    )
    fig_mag_cat.update_traces(
        textposition="outside",
        marker_color="#ff6b35"
    )
    fig_mag_cat.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Kategori Magnitudo",
        yaxis_title="Jumlah Gempa",
        xaxis_tickangle=-20,
        margin=dict(t=20, b=60, l=40, r=20)
    )

    # c. Distribusi Kategori Kedalaman
    order_depth = [
        "Shallow-focus (<60 km)",
        "Intermediate-depth (60–300 km)",
        "Deep-focus (>300 km)"
    ]
    depth_counts = (
        df[df["depth_category"].isin(order_depth)]["depth_category"]
        .value_counts()
        .reindex(order_depth)
        .dropna()
        .reset_index()
    )
    depth_counts.columns = ["Depth Category", "Count"]
    fig_depth_cat = px.bar(
        depth_counts,
        x="Depth Category",
        y="Count",
        text="Count"
    )
    fig_depth_cat.update_traces(
        textposition="outside",
        marker_color="#ff6b35"
    )
    fig_depth_cat.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Kategori Kedalaman",
        yaxis_title="Jumlah Gempa",
        xaxis_tickangle=-10,
        margin=dict(t=20, b=60, l=40, r=20)
    )

    # d. Distribusi Musim
    order_season = [
        "Musim Hujan (Okt–Mar)",
        "Musim Kemarau (Apr–Sep)"
    ]
    season_counts = (
        df["season"]
        .value_counts()
        .reindex(order_season)
        .dropna()
        .reset_index()
    )
    season_counts.columns = ["Season", "Count"]
    fig_season = px.bar(
        season_counts,
        x="Season",
        y="Count",
        text="Count"
    )
    fig_season.update_traces(
        textposition="outside",
        marker_color="#ff6b35"
    )
    fig_season.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Musim",
        yaxis_title="Jumlah Gempa",
        xaxis_tickangle=-5,
        margin=dict(t=20, b=60, l=40, r=20)
    )

    return (
        fig_hist,
        fig_scatter,
        fig_map,
        stats,
        options,
        fig_time,
        fig_mag_cat,
        fig_depth_cat,
        fig_season
    )

if __name__ == "__main__": 
    app.run(debug=True)

