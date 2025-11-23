import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import numpy as np
import re
from sklearn.neighbors import BallTree
import logging
from dash.exceptions import PreventUpdate
import time

# Konfigurasi logging agar tidak terlalu verbose saat startup
logging.basicConfig(level=logging.WARNING)

# Global variable untuk mencegah double-click
last_update_time = 0
UPDATE_COOLDOWN = 0.5  # 500ms cooldown

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

# --- Deteksi provinsi Indonesia ---
try:
    worldcities = pd.read_csv("data/worldcities.csv")
    indo = worldcities[worldcities["country"] == "Indonesia"].copy()
    indo_coords = np.radians(indo[["lat", "lng"]].values)
    tree = BallTree(indo_coords, metric="haversine")
    
    def detect_province_fast(lat, lon):
        if pd.isna(lat) or pd.isna(lon): return "Lainnya"
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

# === Pre-calculation and Constants ===
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
    external_stylesheets=[dbc.themes.FLATLY, "/assets/custom.css"],
    suppress_callback_exceptions=True
)
app.title = "Realtime Earthquake Dashboard"

# ----------------------------------------------------------------------
#                         HELPER FUNCTION: Data Filtering (IMPROVED)
# ----------------------------------------------------------------------
def filter_data(provinces_input, mag_range, years, start_year, end_year):
    """
    Fungsi pembantu untuk memfilter DataFrame berdasarkan semua input.
    IMPROVED: Validasi input lebih ketat dan prioritas filter yang jelas.
    """
    
    # 1. Validasi dan Handle Province
    if not provinces_input or (isinstance(provinces_input, list) and len(provinces_input) == 0):
        # Default: gunakan top province atau semua jika tidak ada
        provinces = [top_province] if top_province != 'Lainnya' else df['province'].unique().tolist()
    else:
        # Pastikan provinces adalah list
        provinces = provinces_input if isinstance(provinces_input, list) else [provinces_input]
        # Filter hanya province yang valid
        provinces = [p for p in provinces if p in df['province'].unique()]
        if len(provinces) == 0:
            provinces = df['province'].unique().tolist()
    
    # 2. Validasi Magnitude Range
    if not mag_range or len(mag_range) != 2:
        mag_range = [min_mag_data, max_mag_data]
    else:
        mag_range = [
            max(min_mag_data, min(mag_range[0], mag_range[1])),
            min(max_mag_data, max(mag_range[0], mag_range[1]))
        ]
    
    # 3. Handle Year Filter dengan PRIORITAS JELAS
    all_years = df["time"].dt.year
    
    # PRIORITAS 1: Multi-select years (jika ada pilihan)
    if years and isinstance(years, list) and len(years) > 0:
        # Validasi years dalam range data
        valid_years = [y for y in years if min_year_data <= y <= max_year_data]
        if len(valid_years) > 0:
            year_filter = all_years.isin(valid_years)
        else:
            # Jika tidak ada tahun valid, gunakan default
            year_filter = all_years.between(max_year_data - 4, max_year_data)
    
    # PRIORITAS 2: Year Range (start_year dan end_year)
    elif start_year is not None and end_year is not None:
        # Validasi range
        start_year = max(min_year_data, min(start_year, max_year_data))
        end_year = min(max_year_data, max(end_year, min_year_data))
        
        # Pastikan start <= end
        if start_year > end_year:
            start_year, end_year = end_year, start_year
        
        year_filter = all_years.between(start_year, end_year)
    
    # DEFAULT: 5 tahun terakhir
    else:
        year_filter = all_years.between(max_year_data - 4, max_year_data)

    # 4. Apply ALL FILTERS dengan error handling
    try:
        dff = df[
            (df["magnitude"] >= mag_range[0]) &
            (df["magnitude"] <= mag_range[1]) &
            (year_filter) &
            (df["province"].isin(provinces))
        ].copy()  # Gunakan copy untuk menghindari SettingWithCopyWarning
        
        # Sort by time descending
        dff = dff.sort_values("time", ascending=False)
        
    except Exception as e:
        print(f"Error in filtering: {e}")
        dff = df.copy()
        provinces = df['province'].unique().tolist()
    
    return dff, provinces


# ======================================================================
#                            SIDEBAR
# ======================================================================
sidebar = dbc.Col([
    html.H3("🌍 SeismoTrack", className="fw-bold text-orange mb-4"),
    dbc.Nav([
        dbc.NavLink("📊 Earthquake Overview", href="/overview", active="exact"),
        dbc.NavLink("🌐 Frequency & Depth Analysis", href="/analysis", active="exact"),
        dbc.NavLink("📍 Regional Summary & Cluster", href="/regional", active="exact"),
        html.Hr(),
        dbc.NavLink("⚙️ Profile", href="/profile", active="exact"),
        dbc.NavLink("❓ Help & Support", href="/help", active="exact"),
    ], vertical=True, pills=True, className="sidebar-nav"),
], md=2, className="sidebar p-4 rounded-4 shadow-sm bg-white")

# ======================================================================
#                            PAGE 1: Overview
# ======================================================================
overview_page = html.Div([
    html.H2("Welcome Back, Seismie!", className="fw-bold mb-1"),
    html.P("Explore today's earthquake updates and see what the Earth's been up to.",
            className="text-muted mb-4"),

    # --- Statistik Cards ---
    dbc.Row([
        dbc.Col(html.Div(className="stat-card", children=[
            html.H4(id="total-quakes", className="fw-bold mb-1 text-orange"),
            html.P("Total Earthquakes", className="mb-0 text-muted small")
        ]), md=3),

        dbc.Col(html.Div(className="stat-card", children=[
            html.H4(id="avg-mag", className="fw-bold mb-1 text-orange"),
            html.P("Average Magnitude", className="mb-0 text-muted small")
        ]), md=3),

        dbc.Col(html.Div(className="stat-card", children=[
            html.H4(id="deepest", className="fw-bold mb-1 text-orange"),
            html.P("Deepest Earthquake (km)", className="mb-0 text-muted small")
        ]), md=3),

        dbc.Col(html.Div(className="stat-card", children=[
            html.H4(id="shallowest", className="fw-bold mb-1 text-orange"),
            html.P("Shallowest Earthquake (km)", className="mb-0 text-muted small")
        ]), md=3),
    ], className="mb-4 g-3"),

    # --- Filter Card ---
    html.Div([
        html.H5("🔍 Filter Data", className="fw-bold text-secondary mb-3"),

        dbc.Row([
            dbc.Col([
                html.Label("Regional (Provinsi)", className="fw-semibold small"),
                html.Div([
                    html.Button("🔄 Reset View", id="reset-view", n_clicks=0,
                                 className="btn btn-outline-secondary btn-sm mb-2")
                ]),
                dcc.Dropdown(
                    id='province-filter',
                    options=[{'label': p, 'value': p} for p in sorted(df['province'].unique())],
                    value=[top_province] if top_province != 'Lainnya' else [], 
                    multi=True,
                    placeholder="Pilih satu atau lebih provinsi...",
                    clearable=True
                ),
            ], md=4),

            dbc.Col([
                html.Label("Rentang Magnitudo", className="fw-semibold small"),
                dcc.RangeSlider(
                    id='mag-filter',
                    min=min_mag_data, 
                    max=max_mag_data, 
                    step=0.1,
                    marks={i: str(i) for i in range(int(min_mag_data), int(max_mag_data) + 1)},
                    value=[min_mag_data, max_mag_data],
                    tooltip={"placement": "bottom", "always_visible": True}
                ),
            ], md=4),

            dbc.Col([
                html.Label("Pilih Tahun (Multi-select)", className="fw-semibold small"),
                dcc.Dropdown(
                    id='year-filter',
                    options=[
                        {'label': str(y), 'value': y}
                        for y in sorted(df['time'].dt.year.unique(), reverse=True)
                    ],
                    value=default_years_selection, 
                    multi=True,
                    placeholder="Pilih tahun...",
                    clearable=True
                ),
            ], md=4),
        ], className="g-3"),
        
        dbc.Row([
            dbc.Col([
                html.Label("Atau Rentang Tahun (Range)", className="fw-semibold small mt-3"),
                html.Div([
                    dcc.Input(
                        id='start-year',
                        type='number',
                        placeholder=f'Tahun awal ({min_year_data})',
                        min=min_year_data, 
                        max=max_year_data, 
                        step=1,
                        value=None,  # Set None agar tidak konflik dengan multi-select
                        style={'width': '45%', 'marginRight': '10px'}
                    ),
                    dcc.Input(
                        id='end-year',
                        type='number',
                        placeholder=f'Tahun akhir ({max_year_data})',
                        min=min_year_data, 
                        max=max_year_data, 
                        step=1,
                        value=None,  # Set None agar tidak konflik dengan multi-select
                        style={'width': '45%'}
                    )
                ], style={'display': 'flex', 'justifyContent': 'space-between'})
            ], md=8),
        ], className="g-3 mt-2"),

        # Info text untuk user
        html.Small("💡 Gunakan Multi-select ATAU Range, tidak keduanya. Multi-select diprioritaskan.", 
                   className="text-muted d-block mt-2"),
    ], className="filter-card p-4 bg-white rounded-4 shadow-sm mb-4"),

    # Loading indicator
    dcc.Loading(
        id="loading",
        type="circle",
        children=[
            # --- Map Graph ---
            html.Div([
                html.H5("🗺️ Earthquake Map", className="fw-bold text-secondary mb-3"),
                dcc.Graph(
                    id="map-graph", 
                    style={"height": "500px"},
                    config={
                        'doubleClick': False,  # Nonaktifkan double-click reset
                        'scrollZoom': True,
                        'displayModeBar': True,
                        'displaylogo': False
                    }
                ),
            ], className="bg-white rounded-4 shadow-sm p-3 mb-4"),

            # --- Recent Table ---
            html.Div([
                html.H5("📋 Recent Earthquakes", className="fw-bold text-secondary mb-3"),
                html.Div(id="recent-table")
            ], className="bg-white rounded-4 shadow-sm p-3")
        ]
    )
])

# ======================================================================
#                            PAGES 2-5 (Sama)
# ======================================================================
analysis_page = html.Div([
    html.H2("Frequency & Depth Analysis", className="fw-bold mb-3"),
    html.P("Analisis distribusi magnitudo dan kedalaman gempa di Indonesia.", className="text-muted"),
    dcc.Graph(figure=px.histogram(df, x="magnitude", nbins=20, color_discrete_sequence=["#f97316"], title="Distribusi Magnitudo Gempa")),
    dcc.Graph(figure=px.scatter(df, x="magnitude", y="depth", color="province", color_discrete_sequence=px.colors.qualitative.Set2, title="Korelasi Magnitudo vs Kedalaman"))
])

regional_page = html.Div([
    html.H2("Regional Summary & Cluster", className="fw-bold mb-3"),
    html.P("Lihat ringkasan aktivitas gempa per provinsi dan pola klasternya.", className="text-muted"),
    dcc.Graph(
        figure=px.bar(
            df.groupby("province")["magnitude"].mean().reset_index().sort_values("magnitude", ascending=False),
            x="province", y="magnitude", color="magnitude", color_continuous_scale="OrRd",
            title="Rata-rata Magnitudo per Provinsi"
        )
    )
])

profile_page = html.Div([html.H2("Profile", className="fw-bold mb-3"), html.P("Halaman ini bisa berisi informasi pengguna, pengaturan, dan preferensi.")])
help_page = html.Div([html.H2("Help & Support", className="fw-bold mb-3"), html.P("Panduan penggunaan dashboard dan kontak bantuan.")])


# ======================================================================
#                            ROUTING
# ======================================================================
app.layout = dbc.Container([
    dcc.Location(id='url'),
    dbc.Row([
        sidebar,
        dbc.Col(html.Div(id='page-content', className="main-content p-4"), md=10)
    ]),
    # Store untuk menyimpan state terakhir (mencegah update berlebihan)
    dcc.Store(id='last-filter-state', data={})
], fluid=True)


@app.callback(Output('page-content', 'children'), Input('url', 'pathname'))
def display_page(pathname):
    if pathname in ['/', '/overview']: return overview_page
    elif pathname == '/analysis': return analysis_page
    elif pathname == '/regional': return regional_page
    elif pathname == '/profile': return profile_page
    elif pathname == '/help': return help_page
    else: return html.H3("404 - Page not found", className="text-danger")

# ======================================================================
#                   CALLBACK UTAMA (Overview) - IMPROVED
# ======================================================================
@app.callback(
    Output("total-quakes", "children"),
    Output("avg-mag", "children"),
    Output("deepest", "children"),
    Output("shallowest", "children"),
    Output("map-graph", "figure"),
    Output("recent-table", "children"),
    Output("last-filter-state", "data"),

    Input("province-filter", "value"),
    Input("mag-filter", "value"),
    Input("year-filter", "value"),
    Input("start-year", "value"),
    Input("end-year", "value"),
    Input("map-graph", "clickData"),
    Input("reset-view", "n_clicks"),
    
    State("last-filter-state", "data"),
    prevent_initial_call=False
)
def update_dashboard(provinces_input, mag_range, years, start_year, end_year, 
                     clickData, reset_clicks, last_state):
    
    global last_update_time
    
    # PREVENT DOUBLE-CLICK: Cek cooldown
    current_time = time.time()
    if current_time - last_update_time < UPDATE_COOLDOWN:
        raise PreventUpdate
    last_update_time = current_time
    
    # Deteksi trigger
    ctx = dash.callback_context
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None
    
    # Cek apakah filter benar-benar berubah (untuk mencegah update tidak perlu)
    current_filter = {
        'provinces': provinces_input,
        'mag_range': mag_range,
        'years': years,
        'start_year': start_year,
        'end_year': end_year
    }
    
    # Jika filter tidak berubah DAN bukan click/reset, skip update
    if (last_state == current_filter and 
        triggered_id not in ["map-graph", "reset-view"]):
        raise PreventUpdate
    
    # 1. FILTER DATA (dengan fungsi yang sudah diperbaiki)
    dff, current_provinces = filter_data(provinces_input, mag_range, years, start_year, end_year)

    # 2. CALCULATE STATISTICS
    total_quakes = len(dff)
    avg_mag = f"{dff['magnitude'].mean():.2f}" if total_quakes > 0 else "0.00"
    deepest = f"{dff['depth'].max():.2f}" if total_quakes > 0 else "0.00"
    shallowest = f"{dff['depth'].min():.2f}" if total_quakes > 0 else "0.00"

    # 3. MAP VIEW LOGIC (dengan handling yang lebih baik)
    lat_center_view, lon_center_view, zoom_level = center_lat, center_lon, 3.5

    if not dff.empty:
        data_lat_center = dff["latitude"].mean()
        data_lon_center = dff["longitude"].mean()
        
        # Auto zoom berdasarkan jumlah data
        if total_quakes > 500: 
            zoom_level_data = 4.0
        elif total_quakes > 100: 
            zoom_level_data = 5.0
        elif total_quakes > 20: 
            zoom_level_data = 6.0
        else: 
            zoom_level_data = 7.0
        
        # Tentukan view berdasarkan trigger
        if triggered_id == "map-graph" and clickData and 'points' in clickData:
            # Zoom ke titik yang diklik
            try:
                point = clickData["points"][0]
                lat_center_view = point.get("lat", data_lat_center)
                lon_center_view = point.get("lon", data_lon_center)
                zoom_level = 8.0  # Zoom lebih dekat saat klik
            except (KeyError, IndexError):
                lat_center_view, lon_center_view = data_lat_center, data_lon_center
                zoom_level = zoom_level_data
            
        elif triggered_id == "reset-view":
            # Reset ke view semua data
            lat_center_view, lon_center_view = data_lat_center, data_lon_center
            zoom_level = zoom_level_data
        else:
            # Filter change: auto-fit ke data baru
            lat_center_view, lon_center_view = data_lat_center, data_lon_center
            zoom_level = zoom_level_data
    
    # 4. CREATE MAP
    province_display = ', '.join(current_provinces[:3])
    if len(current_provinces) > 3:
        province_display += f' +{len(current_provinces)-3} lainnya'
    
    fig = px.scatter_mapbox(
        dff,
        lat="latitude",
        lon="longitude",
        color="magnitude",
        size="magnitude",
        hover_name="place",
        hover_data={
            "depth": ":.2f", 
            "time": True, 
            "province": True, 
            "latitude": ':.4f', 
            "longitude": ':.4f', 
            "magnitude": ':.2f'
        },
        color_continuous_scale="OrRd",
        zoom=zoom_level,
        center={"lat": lat_center_view, "lon": lon_center_view},
        height=500,
        title=f"Earthquake Distribution (Total: {total_quakes} | Provinces: {province_display})",
    )

    fig.update_layout(
        mapbox_style="open-street-map",
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        hovermode='closest'
    )

    # 5. CREATE TABLE
    if dff.empty:
        table = html.Div([
            html.P("⚠️ No earthquake data available for the selected filters.", 
                   className="text-muted text-center p-4"),
            html.P("Try adjusting your filters.", className="text-muted text-center small")
        ])
    else:
        recent = dff.head(10)[["time", "place", "magnitude", "depth", "province"]].copy()
        recent['time'] = recent['time'].dt.strftime('%Y-%m-%d %H:%M:%S') 
        recent.columns = ["Time", "Place", "Magnitude", "Depth (km)", "Province"]
        table = dbc.Table.from_dataframe(
            recent,
            striped=True,
            bordered=True,
            hover=True,
            className="table table-striped table-hover mb-0"
        )

    return total_quakes, avg_mag, deepest, shallowest, fig, table, current_filter

if __name__ == "__main__": 
    app.run(debug=True, dev_tools_hot_reload=False)  # Matikan hot reload untuk stabilitas
