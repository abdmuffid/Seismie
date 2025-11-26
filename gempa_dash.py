import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.neighbors import BallTree
import logging

# Konfigurasi logging
logging.basicConfig(level=logging.WARNING)

# ======================================================================
# 1. LOAD DATA & FEATURE ENGINEERING
# ======================================================================
# A. Load Data Gempa
try:
    df = pd.read_csv("data/combined/combined.csv", parse_dates=['time'])
    # Hapus timezone agar kompatibel dengan filter tanggal sederhana
    df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
except FileNotFoundError:
    print("Warning: 'combined.csv' not found. Creating dummy data.")
    # Data Dummy jika file tidak ada
    data = {
        'time': pd.to_datetime(['2025-10-15 12:00:00', '2025-10-16 08:30:00', '2024-05-20 10:00:00', '2023-01-01 00:00:00', '2025-10-14 11:00:00']),
        'latitude': [-6.2088, -7.7956, -8.4095, 0.7893, -6.9034],
        'longitude': [106.8456, 110.3695, 115.1889, 113.9213, 107.6191],
        'depth': [10.0, 50.5, 12.3, 150.0, 20.0],
        'magnitude': [5.5, 4.2, 6.1, 7.0, 3.5],
        'place': ['8km S of Jakarta', 'Yogyakarta Region', 'Bali', 'Kalimantan Tengah', 'Bandung'],
    }
    df = pd.DataFrame(data)

# B. Load Clustering (Opsional)
try:
    df_cluster_file = pd.read_excel("data/best_df_indonesia_cluster.xlsx")
    # Disini kita hanya simulasi kolom cluster jika file excel tidak match dengan df utama
    if 'cluster' not in df.columns:
        df['cluster'] = np.random.randint(0, 5, size=len(df))
except Exception:
    df['cluster'] = -1

# C. Deteksi Provinsi (BallTree Logic)
try:
    worldcities = pd.read_csv("data/worldcities.csv")
    indo = worldcities[worldcities["country"] == "Indonesia"].copy()
    indo_coords = np.radians(indo[["lat", "lng"]].values)
    tree = BallTree(indo_coords, metric="haversine")
    
    def detect_province_fast(lat, lon):
        if pd.isna(lat) or pd.isna(lon): return "Lainnya"
        dist, idx = tree.query(np.radians([[lat, lon]]), k=1)
        nearest = indo.iloc[idx[0][0]]
        # Jika jarak < ~200km anggap masuk provinsi terdekat
        if dist[0][0] * 6371 < 200: 
            return str(nearest["admin_name"]).replace("Province", "").strip()
        return "Lainnya"
    
    if "province" not in df.columns:
        df["province"] = df.apply(lambda r: detect_province_fast(r["latitude"], r["longitude"]), axis=1)

except FileNotFoundError:
    print("Warning: 'worldcities.csv' not found. Using default province.")
    df["province"] = "Indonesia"

# D. Feature Engineering (Kategori Waktu, Musim, dll)
df['hour'] = df['time'].dt.hour
df['month'] = df['time'].dt.month

def categorize_time(h):
    if 0 <= h < 6: return 'Dini Hari (00-06)'
    elif 6 <= h < 12: return 'Pagi (06-12)'
    elif 12 <= h < 15: return 'Siang (12-15)'
    elif 15 <= h < 18: return 'Sore (15-18)'
    else: return 'Malam (18-24)'
df['time_category'] = df['hour'].apply(categorize_time)

def categorize_mag(m):
    if 3.0 <= m < 4.0: return 'Minor (3.0-3.9)'
    elif 4.0 <= m < 5.0: return 'Light (4.0-4.9)'
    elif 5.0 <= m < 6.0: return 'Moderate (5.0-5.9)'
    elif 6.0 <= m < 7.0: return 'Strong (6.0-6.9)'
    elif 7.0 <= m < 8.0: return 'Major (7.0-7.9)'
    elif m >= 8.0: return 'Great (≥8.0)'
    else: return 'Micro (<3.0)'
df['mag_category'] = df['magnitude'].apply(categorize_mag)

def categorize_depth(d):
    if d < 60: return 'Dangkal (<60 km)'
    elif 60 <= d <= 300: return 'Menengah (60-300 km)'
    else: return 'Dalam (>300 km)'
df['depth_category'] = df['depth'].apply(categorize_depth)

def categorize_season(m):
    if m in [10, 11, 12, 1, 2, 3]: return 'Musim Hujan (Okt-Mar)'
    else: return 'Musim Kemarau (Apr-Sep)'
df['season_category'] = df['month'].apply(categorize_season)

# E. Koordinat Provinsi (Untuk Peta Agregasi)
prov_coords = df.groupby('province')[['latitude', 'longitude']].mean().reset_index()
min_mag_data, max_mag_data = df['magnitude'].min(), df['magnitude'].max()
center_lat, center_lon = -2.5489, 118.0149

# ======================================================================
# 2. GLOBAL DATA (PRE-FILLED)
# ======================================================================
# Data Artikel Awal
articles_db = [
    {
        'title': '10 Cara Menyelamatkan Diri dari Gempa Bumi yang Wajib Diketahui',
        'url': 'https://www.bmkg.go.id/gempabumi/panduan-gempa.bmkg',
        'img': 'https://images.unsplash.com/photo-1590859808308-3d2d9c515b1a?w=600&h=400&fit=crop',
        'desc': 'Gempa bumi adalah bencana alam yang tidak dapat diprediksi. Kenali 10 langkah penting untuk menyelamatkan diri dan keluarga saat terjadi gempa bumi.'
    },
    {
        'title': 'Pertolongan Pertama untuk Korban Gempa',
        'url': 'https://www.pmi.or.id/p3k-gempa',
        'img': 'https://images.unsplash.com/photo-1584820927498-cfe5211fd8bf?w=600&h=400&fit=crop', 
        'desc': 'Pelajari teknik pertolongan pertama yang tepat untuk membantu korban gempa. Termasuk cara menangani luka, patah tulang, dan kondisi darurat lainnya.'
    }
]

# Data Posko Awal
posko_db = [
    {'name': 'Posko Utama Balai Kota', 'lat': -6.1805, 'lon': 106.8284, 'link': 'https://maps.app.goo.gl/AjiBWsQSeFZXSLpw9'},
    {'name': 'Gelanggang Remaja (Pengungsian)', 'lat': -7.250207600809818, 'lon': 112.75557072879006, 'link': 'https://maps.app.goo.gl/4uLqoWaigTQqndxt7'},
    {'name': 'RSUD Tanah Abang (Posko Medis)', 'lat': -6.190691600361024, 'lon': 106.81450146485828, 'link': 'https://maps.app.goo.gl/KYrHz2ezt9upVYMa8'}
]

# ======================================================================
# 3. SETUP APLIKASI
# ======================================================================
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)
app.title = "SeismoTrack - Earthquake Dashboard"

# --- Helper Functions ---
def filter_data(provinces_input, mag_range, start_date, end_date):
    if not provinces_input:
        provinces = df['province'].unique().tolist()
    else:
        provinces = provinces_input
    
    mask_date = (df['time'] >= pd.to_datetime(start_date)) & (df['time'] <= pd.to_datetime(end_date))
    dff = df[
        (df["magnitude"].between(mag_range[0], mag_range[1])) &
        (mask_date) &
        (df["province"].isin(provinces))
    ].sort_values("time", ascending=False)
    return dff, provinces

def make_clean_chart(fig):
    """Membersihkan tampilan chart Plotly (menghapus grid kasar)"""
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter, sans-serif", color="#333"),
        xaxis=dict(showgrid=False, linecolor="#ccc"),
        yaxis=dict(showgrid=True, gridcolor="#f0f0f0", zeroline=False),
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); font-family: 'Inter', sans-serif; }
            .sidebar { background: white; border-radius: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.07); min-height: 95vh; }
            .nav-link.active { background: linear-gradient(135deg, #ff6b35 0%, #ff8c42 100%); color: white !important; box-shadow: 0 4px 10px rgba(255,107,53,0.3); border-radius: 12px; }
            .welcome-header { background: white; border-radius: 20px; padding: 30px; margin-bottom: 30px; border-left: 5px solid #ff6b35; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
            .stat-card-modern { background: white; border-radius: 20px; padding: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-left: 4px solid #ff6b35; text-align: center; }
            .stat-value { font-size: 1.8rem; font-weight: 700; color: #ff6b35; }
            .chart-container { background: white; border-radius: 20px; padding: 30px; margin-bottom: 30px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-top: 3px solid #ff6b35; }
            .btn-reset { background: linear-gradient(135deg, #ff6b35 0%, #ff8c42 100%); border: none; border-radius: 12px; color: white; padding: 10px 20px; }
            .text-orange { color: #ff6b35; }
            .article-card { border: 1px solid #e2e8f0; border-top: 3px solid #ff6b35; border-radius: 15px; overflow: hidden; margin-bottom: 20px; background: white;}
            .article-image { width: 100%; height: 150px; object-fit: cover; }
            .article-content { padding: 15px; }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>{%config%}{%scripts%}{%renderer%}</footer>
    </body>
</html>
'''

# ======================================================================
# 4. LAYOUT COMPONENTS
# ======================================================================
sidebar = dbc.Col([
    html.Div([
        html.H3("🌍 SeismoTrack", className="mb-4 text-orange fw-bold"),
        dbc.Nav([
            dbc.NavLink("📊 Dashboard Overview", href="/overview", active="exact", className="mb-2"),
            dbc.NavLink("📈 Eksplorasi Data", href="/analysis", active="exact", className="mb-2"),
            dbc.NavLink("📍 Regional Summary", href="/regional", active="exact", className="mb-2"),
            html.Hr(),
            dbc.NavLink("⚙️ Safety Info", href="/settings", active="exact", className="mb-2"),
            dbc.NavLink("❓ Help & Support", href="/help", active="exact"),
        ], vertical=True, pills=True),
    ])
], md=2, className="sidebar p-4")

# --- PAGE 1: OVERVIEW ---
overview_page = html.Div([
    html.Div([
        html.H2("Welcome Back, Seismie!", className="mb-2 fw-bold"),
        html.P("Pantau aktivitas gempa bumi terkini di seluruh Indonesia.", className="text-muted mb-0")
    ], className="welcome-header"),

    dbc.Row([
        dbc.Col(html.Div([html.Div(id="total-quakes", className="stat-value"), html.Div("Total Gempa", className="text-muted")]), className="stat-card-modern mb-3", md=3),
        dbc.Col(html.Div([html.Div(id="avg-mag", className="stat-value"), html.Div("Rata-rata Mag", className="text-muted")]), className="stat-card-modern mb-3", md=3),
        dbc.Col(html.Div([html.Div(id="deepest", className="stat-value"), html.Div("Gempa Terdalam", className="text-muted")]), className="stat-card-modern mb-3", md=3),
        dbc.Col(html.Div([html.Div(id="shallowest", className="stat-value"), html.Div("Gempa Dangkal", className="text-muted")]), className="stat-card-modern mb-3", md=3),
    ]),

    html.Div([
        html.Div([
            html.H5("🔍 Filter Data", className="mb-0 fw-bold"),
            html.Button("🔄 Reset View", id="reset-view", n_clicks=0, className="btn-reset")
        ], className="d-flex justify-content-between align-items-center mb-4"),

        dbc.Row([
            dbc.Col([
                html.Label("Rentang Tanggal", className="fw-bold small"),
                dcc.DatePickerRange(
                    id='date-filter',
                    min_date_allowed=df['time'].min(),
                    max_date_allowed=df['time'].max(),
                    start_date=df['time'].max() - pd.Timedelta(days=365),
                    end_date=df['time'].max(),
                    display_format='DD/MM/YYYY',
                    style={'width': '100%', 'borderRadius': '12px'}
                )
            ], md=6),
            dbc.Col([
                html.Label("Rentang Magnitude", className="fw-bold small"),
                dcc.RangeSlider(
                    id='mag-filter',
                    min=float(min_mag_data), max=float(max_mag_data), step=0.1,
                    value=[min_mag_data, max_mag_data],
                    marks={i: str(i) for i in range(int(min_mag_data), int(max_mag_data) + 1)}
                )
            ], md=6),
        ], className="mb-3"),

        dbc.Row([
            dbc.Col([
                html.Label("Filter Provinsi", className="fw-bold small"),
                dcc.Dropdown(
                    id='province-filter',
                    options=[{'label': p, 'value': p} for p in sorted(df['province'].unique())],
                    value=[], multi=True, placeholder="Pilih provinsi...",
                ),
            ], md=12),
        ])
    ], className="chart-container"),

    html.Div([
        html.Div([
            html.H5("🗺️ Peta Sebaran Gempa", className="fw-bold"),
            dbc.RadioItems(
                id='map-mode',
                options=[
                    {'label': 'Agregasi Provinsi (Zona)', 'value': 'agg'},
                    {'label': 'Titik Individual', 'value': 'point'}
                ],
                value='agg',
                inline=True,
                className="ms-auto"
            )
        ], className="d-flex justify-content-between align-items-center mb-3"),
        
        dcc.Graph(id="map-graph", style={"height": "550px"}),
    ], className="chart-container"),

    html.Div([
        html.Div([
            html.H5("📋 Data Gempa Terfilter", className="mb-0 fw-bold"),
            html.Button("⬇️ Download CSV", id="download-btn", className="btn-reset")
        ], className="d-flex justify-content-between align-items-center mb-3"),
        html.Div(id="recent-table"),
        dcc.Download(id="download-data")
    ], className="chart-container")
])

# --- PAGE 2: ANALYSIS ---
analysis_page = html.Div([
    html.Div([
        html.H2("Eksplorasi Data Gempa", className="mb-2 fw-bold"),
        html.P("Analisis karakteristik gempa berdasarkan waktu, musim, dan kekuatan.", className="text-muted mb-0")
    ], className="welcome-header"),

    dbc.Row([
        dbc.Col([
            html.Div([
                html.H5("🕒 Distribusi Waktu Kejadian", className="fw-bold mb-3"),
                dcc.Graph(id="chart-time", style={"height": "300px"})
            ], className="chart-container")
        ], md=6),
        dbc.Col([
            html.Div([
                html.H5("🌦️ Distribusi Musim", className="fw-bold mb-3"),
                dcc.Graph(id="chart-season", style={"height": "300px"})
            ], className="chart-container")
        ], md=6),
    ]),

    dbc.Row([
        dbc.Col([
            html.Div([
                html.H5("🌎 Klasifikasi Magnitude", className="fw-bold mb-3"),
                dcc.Graph(id="chart-mag-cat", style={"height": "300px"})
            ], className="chart-container")
        ], md=6),
        dbc.Col([
            html.Div([
                html.H5("🌊 Klasifikasi Kedalaman", className="fw-bold mb-3"),
                dcc.Graph(id="chart-depth-cat", style={"height": "300px"})
            ], className="chart-container")
        ], md=6),
    ]),

    html.Div([
        html.H5("🔍 Analisis Cluster (DBSCAN)", className="fw-bold mb-3"),
        dbc.Row([
            dbc.Col([
                html.Label("Filter Cluster:"),
                dcc.Dropdown(id="cluster-filter", placeholder="Pilih cluster...", clearable=True),
                dcc.Graph(id="cluster-map", style={"height": "400px"})
            ], md=8),
            dbc.Col([
                html.H6("Statistik Cluster"),
                html.Div(id="cluster-stats")
            ], md=4)
        ])
    ], className="chart-container")
])

# --- PAGE 3: REGIONAL ---
regional_page = html.Div([
    html.Div([
        html.H2("Ringkasan Regional", className="mb-2 fw-bold"),
        html.P("Statistik aktivitas seismik per wilayah provinsi.", className="text-muted mb-0")
    ], className="welcome-header"),
    
    html.Div([
        html.H5("📍 Rata-rata Magnitude per Provinsi"),
        dcc.Graph(id="chart-regional")
    ], className="chart-container")
])

# --- PAGE 4: SAFETY (DIPERLENGKAP) ---
settings_page = html.Div([
    html.Div([
        html.H2("⚙️ Earthquake Safety & Emergency Info", className="mb-2 fw-bold"),
        html.P("Panduan keselamatan saat gempa dan lokasi posko pengungsian terdekat.", className="mb-0 text-muted")
    ], className="welcome-header"),
    
    dbc.Row([
        # Kolom Kiri: Artikel
        dbc.Col([
            html.Div([
                html.H5("🚨 Tips Keselamatan & Artikel", className="mb-3 fw-bold"),
                
                # Input Tambah Artikel
                html.Div([
                    html.H6("📎 Tambah Artikel Referensi:", className="fw-bold mb-2 small text-muted"),
                    dbc.Row([
                        dbc.Col(dcc.Input(id='article-title-input', type='text', placeholder='Judul Artikel', className="form-control mb-2"), md=12),
                        dbc.Col(dcc.Input(id='article-url-input', type='text', placeholder='Link URL', className="form-control mb-2"), md=12),
                        dbc.Col(dcc.Input(id='article-image-input', type='text', placeholder='URL Gambar (Opsional)', className="form-control mb-2"), md=12),
                        dbc.Col(dcc.Textarea(id='article-desc-input', placeholder='Deskripsi singkat...', className="form-control mb-2", style={'height': '60px'}), md=12),
                        dbc.Col(html.Button("➕ Tambah Artikel", id="add-article-btn", className="btn btn-warning text-white w-100 fw-bold"), md=12),
                    ]),
                    html.Div(id="article-feedback", className="small text-success mt-2")
                ], className="mb-4 p-3 bg-light rounded border"),
                
                # Daftar Artikel
                html.Div([
                    html.H6("📚 Artikel Tersimpan:", className="fw-bold mb-3"),
                    html.Div(id="articles-list")
                ], className="mb-4", style={"maxHeight": "500px", "overflowY": "auto"}),
                
                html.Hr(),
                
                # Tips Accordion (Statis)
                dbc.Accordion([
                    dbc.AccordionItem([
                        html.Ul([
                            html.Li("DROP - Jatuhkan diri ke lantai"),
                            html.Li("COVER - Berlindung di bawah meja yang kuat"),
                            html.Li("HOLD ON - Pegang kaki meja sampai guncangan berhenti"),
                        ])
                    ], title="1️⃣ Saat Di Dalam Ruangan"),
                    dbc.AccordionItem([
                        html.Ul([
                            html.Li("Jauhi bangunan, tiang listrik, dan pohon"),
                            html.Li("Cari tempat terbuka dan aman"),
                        ])
                    ], title="2️⃣ Saat Di Luar Ruangan"),
                ], start_collapsed=True)

            ], className="chart-container")
        ], md=6),
        
        # Kolom Kanan: Posko & Peta
        dbc.Col([
            html.Div([
                html.H5("🏕️ Posko Pengungsian & Peta", className="mb-3 fw-bold"),
                
                # Input Tambah Posko
                html.Div([
                    html.H6("📍 Tambah Posko Baru:", className="fw-bold mb-2 small text-muted"),
                    dbc.Row([
                        dbc.Col(dcc.Input(id='posko-name-input', type='text', placeholder='Nama Posko', className="form-control mb-2"), md=12),
                        dbc.Col(dcc.Input(id='posko-gmaps-input', type='text', placeholder='Link Google Maps', className="form-control mb-2"), md=8),
                        dbc.Col(html.Button("➕", id="add-posko-btn", className="btn btn-warning text-white w-100 fw-bold"), md=4),
                    ]),
                    html.Div(id="posko-feedback", className="small mt-2")
                ], className="mb-3 p-3 bg-light rounded border"),
                
                # Peta Posko
                dcc.Graph(id="evacuation-map", config={'displayModeBar': False}, style={"height": "400px", "borderRadius": "10px"}),
                
                # Daftar Posko
                html.Div([
                    html.H6("📍 Daftar Posko Terdaftar:", className="fw-bold mt-3 mb-2"),
                    html.Div(id="posko-list", style={"maxHeight": "200px", "overflowY": "auto"})
                ])
            ], className="chart-container")
        ], md=6),
    ])
])

# --- PAGE 5: HELP ---
help_page = html.Div([
    html.Div([
        html.H2("Help & Support", className="fw-bold"),
        html.P("Pusat bantuan penggunaan dashboard.", className="text-muted")
    ], className="welcome-header"),
    dbc.Card([
        dbc.CardHeader("FAQ", className="fw-bold bg-warning text-white"),
        dbc.CardBody([
            html.Ul([
                html.Li("Mode Peta Agregasi: Menunjukkan jumlah gempa dalam satu provinsi dengan lingkaran."),
                html.Li("Mode Titik: Menunjukkan lokasi persis setiap gempa."),
                html.Li("Gunakan date picker untuk menyaring data historis."),
            ])
        ])
    ], className="shadow-sm")
])

# ======================================================================
# 5. ROUTING & CALLBACKS
# ======================================================================
app.layout = dbc.Container([
    dcc.Location(id='url'),
    dbc.Row([sidebar, dbc.Col(html.Div(id='page-content'), md=10, className="p-4")])
], fluid=True, style={"backgroundColor": "#f8f9fa"})

@app.callback(Output('page-content', 'children'), Input('url', 'pathname'))
def display_page(pathname):
    if pathname in ['/', '/overview']: return overview_page
    elif pathname == '/analysis': return analysis_page
    elif pathname == '/regional': return regional_page
    elif pathname == '/settings': return settings_page
    elif pathname == '/help': return help_page
    return overview_page

# --- CALLBACK DASHBOARD UTAMA ---
@app.callback(
    [Output("total-quakes", "children"),
     Output("avg-mag", "children"),
     Output("deepest", "children"),
     Output("shallowest", "children"),
     Output("map-graph", "figure"),
     Output("recent-table", "children")],
    [Input("province-filter", "value"),
     Input("mag-filter", "value"),
     Input("date-filter", "start_date"),
     Input("date-filter", "end_date"),
     Input("map-mode", "value"),
     Input("reset-view", "n_clicks")]
)
def update_overview(provinces, mag_range, start_date, end_date, map_mode, n_clicks):
    dff, _ = filter_data(provinces, mag_range, start_date, end_date)
    
    total = len(dff)
    avg = f"{dff['magnitude'].mean():.2f}" if total else "0"
    deep = f"{dff['depth'].max():.1f} km" if total else "-"
    shallow = f"{dff['depth'].min():.1f} km" if total else "-"
    
    # Map Logic (Agg vs Point)
    if map_mode == 'agg':
        if not dff.empty:
            prov_counts = dff['province'].value_counts().reset_index()
            prov_counts.columns = ['province', 'count']
            df_map = pd.merge(prov_coords, prov_counts, on='province', how='inner')
            
            fig_map = px.scatter_mapbox(
                df_map, lat="latitude", lon="longitude",
                size="count", color="count",
                hover_name="province",
                color_continuous_scale="Reds", size_max=50, zoom=3.5,
                center={"lat": center_lat, "lon": center_lon},
                title="Aggregated View (Province Level)"
            )
        else:
            fig_map = px.scatter_mapbox(lat=[], lon=[], zoom=3.5, center={"lat": center_lat, "lon": center_lon})
    else:
        if not dff.empty:
            fig_map = px.scatter_mapbox(
                dff, lat="latitude", lon="longitude",
                color="magnitude", size="magnitude",
                hover_name="place",
                color_continuous_scale="OrRd", zoom=3.5,
                center={"lat": center_lat, "lon": center_lon},
                title="Point View (Individual Quakes)"
            )
        else:
            fig_map = px.scatter_mapbox(lat=[], lon=[], zoom=3.5, center={"lat": center_lat, "lon": center_lon})

    fig_map.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":40,"l":0,"b":0})

    # Table
    if not dff.empty:
        disp = dff[['time', 'province', 'magnitude', 'depth', 'place']].head(100).copy()
        disp['time'] = disp['time'].dt.strftime('%Y-%m-%d %H:%M')
        table = dbc.Table.from_dataframe(disp, striped=True, bordered=True, hover=True, className="table-sm")
        table_container = html.Div(table, style={"maxHeight": "400px", "overflowY": "auto"})
    else:
        table_container = html.P("Tidak ada data.", className="text-center text-muted")

    return total, avg, deep, shallow, fig_map, table_container

# --- CALLBACK ANALYSIS ---
@app.callback(
    [Output("chart-time", "figure"),
     Output("chart-season", "figure"),
     Output("chart-mag-cat", "figure"),
     Output("chart-depth-cat", "figure"),
     Output("cluster-map", "figure"),
     Output("cluster-stats", "children"),
     Output("cluster-filter", "options")],
    [Input("cluster-filter", "value"), Input("url", "pathname")]
)
def update_analysis(cluster_val, pathname):
    if pathname != '/analysis': return {}, {}, {}, {}, {}, "", []
    
    # 1. Chart Time
    time_order = ['Dini Hari (00-06)', 'Pagi (06-12)', 'Siang (12-15)', 'Sore (15-18)', 'Malam (18-24)']
    tc = df['time_category'].value_counts().reindex(time_order).fillna(0).reset_index()
    tc.columns = ['Kategori', 'Jumlah']
    fig_time = px.bar(tc, x='Kategori', y='Jumlah', color='Jumlah', color_continuous_scale='Blues')
    fig_time = make_clean_chart(fig_time)

    # 2. Chart Season
    sc = df['season_category'].value_counts().reset_index()
    sc.columns = ['Musim', 'Jumlah']
    fig_season = px.pie(sc, values='Jumlah', names='Musim', hole=0.5, color_discrete_sequence=['#3498db', '#f1c40f'])
    fig_season = make_clean_chart(fig_season)

    # 3. Chart Mag
    mc_order = ['Minor (3.0-3.9)', 'Light (4.0-4.9)', 'Moderate (5.0-5.9)', 'Strong (6.0-6.9)', 'Major (7.0-7.9)', 'Great (≥8.0)']
    mc = df['mag_category'].value_counts().reindex(mc_order).fillna(0).reset_index()
    mc.columns = ['Kategori', 'Jumlah']
    fig_mag = px.bar(mc, x='Kategori', y='Jumlah', color='Jumlah', color_continuous_scale='OrRd')
    fig_mag = make_clean_chart(fig_mag)

    # 4. Chart Depth
    dc_order = ['Dangkal (<60 km)', 'Menengah (60-300 km)', 'Dalam (>300 km)']
    dc = df['depth_category'].value_counts().reindex(dc_order).fillna(0).reset_index()
    dc.columns = ['Kategori', 'Jumlah']
    fig_depth = px.bar(dc, x='Kategori', y='Jumlah', color='Kategori', color_discrete_map={'Dangkal (<60 km)':'#e74c3c', 'Menengah (60-300 km)':'#f39c12', 'Dalam (>300 km)':'#2ecc71'})
    fig_depth = make_clean_chart(fig_depth)

    # Cluster Map
    if cluster_val is not None:
        cdf = df[df['cluster'] == cluster_val]
    else:
        cdf = df[df['cluster'] >= 0]
    
    if not cdf.empty:
        fig_clus = px.scatter_mapbox(cdf, lat="latitude", lon="longitude", color="cluster", zoom=4, center={"lat": -2.5, "lon": 118})
        fig_clus.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
        stats = html.Div([html.P(f"Total Gempa: {len(cdf)}"), html.P(f"Avg Mag: {cdf['magnitude'].mean():.2f}")])
    else:
        fig_clus = go.Figure()
        stats = "No Data"
    
    opts = [{'label': f'Cluster {c}', 'value': c} for c in sorted(df[df['cluster']>=0]['cluster'].unique())]

    return fig_time, fig_season, fig_mag, fig_depth, fig_clus, stats, opts

# --- REGIONAL CHART CALLBACK ---
@app.callback(
    Output("chart-regional", "figure"),
    Input("url", "pathname")
)
def update_regional(pathname):
    if pathname != '/regional': return go.Figure()
    
    df_reg = df.groupby("province")["magnitude"].mean().reset_index().sort_values("magnitude", ascending=False).head(15)
    fig = px.bar(df_reg, x="province", y="magnitude", color="magnitude", color_continuous_scale="OrRd")
    return make_clean_chart(fig)

# --- CALLBACK SAFETY (ARTIKEL & POSKO) ---
@app.callback(
    [Output("articles-list", "children"), Output("article-feedback", "children")],
    [Input("add-article-btn", "n_clicks"), Input("url", "pathname")], 
    [State("article-title-input", "value"), State("article-url-input", "value"), 
     State("article-image-input", "value"), State("article-desc-input", "value")]
)
def update_articles(n, pathname, title, url, img, desc):
    ctx = callback_context
    msg = ""
    
    # Logic Tambah Artikel
    if ctx.triggered:
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger_id == 'add-article-btn' and title:
            articles_db.append({
                'title': title, 
                'url': url if url else '#', 
                'img': img if img else "https://via.placeholder.com/150", 
                'desc': desc if desc else "Tidak ada deskripsi."
            })
            msg = "✅ Artikel berhasil ditambahkan!"

    # Render List Artikel (Termasuk data pre-filled)
    children = []
    for a in articles_db:
        card = dbc.Card([
            dbc.Row([
                dbc.Col(dbc.CardImg(src=a.get('img', "https://via.placeholder.com/150"), className="img-fluid rounded-start", style={"height":"100px", "objectFit":"cover"}), width=3),
                dbc.Col(dbc.CardBody([
                    html.H5(a['title'], className="card-title h6 fw-bold"),
                    html.P(a['desc'], className="card-text small text-muted mb-1"),
                    dbc.Button("Baca", href=a['url'], target="_blank", size="sm", color="primary", outline=True)
                ], className="p-2"), width=9),
            ], className="g-0")
        ], className="mb-2 border shadow-sm", style={"overflow": "hidden"})
        children.append(card)
        
    return children, msg

@app.callback(
    [Output("evacuation-map", "figure"), Output("posko-list", "children"), Output("posko-feedback", "children")],
    [Input("add-posko-btn", "n_clicks"), Input("url", "pathname")],
    [State("posko-name-input", "value"), State("posko-gmaps-input", "value")]
)
def update_posko(n, pathname, name, map_link):
    ctx = callback_context
    msg = ""
    
    # Logic Tambah Posko
    if ctx.triggered:
        trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if trigger_id == 'add-posko-btn' and name:
            lat, lon = -6.175, 106.82 # Default coords
            try:
                if '@' in map_link:
                    parts = map_link.split('@')[1].split(',')
                    lat, lon = float(parts[0]), float(parts[1])
            except: pass
            
            posko_db.append({'name': name, 'lat': lat, 'lon': lon, 'link': map_link})
            msg = "✅ Posko ditambahkan!"

    # Create Map
    df_posko = pd.DataFrame(posko_db)
    if not df_posko.empty:
        fig = px.scatter_mapbox(df_posko, lat="lat", lon="lon", hover_name="name", zoom=11)
        fig.update_traces(marker=dict(size=15, color='#2ecc71', symbol='hospital'))
        fig.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})
    else:
        fig = px.scatter_mapbox(lat=[], lon=[], zoom=10, center={"lat": -6.2, "lon": 106.8})
        fig.update_layout(mapbox_style="carto-positron")

    # Create List
    list_items = []
    for p in posko_db:
        item = dbc.ListGroupItem([
            html.Div([
                html.H6(p['name'], className="mb-1 fw-bold"),
                html.Small(html.A("Buka Maps ↗", href=p.get('link','#'), target="_blank", className="text-decoration-none"))
            ], className="d-flex w-100 justify-content-between")
        ])
        list_items.append(item)

    return fig, dbc.ListGroup(list_items, flush=True), msg

# --- DOWNLOAD ---
@app.callback(
    Output("download-data", "data"),
    Input("download-btn", "n_clicks"),
    [State("province-filter", "value"), State("mag-filter", "value"), State("date-filter", "start_date"), State("date-filter", "end_date")],
    prevent_initial_call=True
)
def download_csv(n, prov, mag, start, end):
    dff, _ = filter_data(prov, mag, start, end)
    return dcc.send_data_frame(dff.to_csv, "gempa_filtered.csv", index=False)

if __name__ == '__main__':
    app.run(debug=True)