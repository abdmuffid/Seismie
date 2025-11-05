import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import re

# === Load data ===
df = pd.read_csv("data/combined/combined.csv", parse_dates=['time'])

# --- Daftar provinsi Indonesia ---
provinsi_list = [
    "Aceh", "Sumatera Utara", "Sumatera Barat", "Riau", "Kepulauan Riau",
    "Jambi", "Bengkulu", "Sumatera Selatan", "Lampung", "Banten", "Jawa Barat",
    "Jakarta", "Jawa Tengah", "DI Yogyakarta", "Jawa Timur", "Bali",
    "Nusa Tenggara Barat", "Nusa Tenggara Timur", "Kalimantan Barat",
    "Kalimantan Tengah", "Kalimantan Selatan", "Kalimantan Timur", "Kalimantan Utara",
    "Sulawesi Utara", "Sulawesi Tengah", "Sulawesi Selatan", "Sulawesi Tenggara",
    "Gorontalo", "Maluku", "Maluku Utara", "Papua", "Papua Barat"
]

# --- Deteksi provinsi di kolom 'place' ---
import pandas as pd
import re

# Load dataset kota dunia
world = pd.read_csv("data/worldcities.csv")

# Ambil hanya yang dari Indonesia
indonesia = world[world['country'] == 'Indonesia'][['city_ascii', 'admin_name']].dropna()

# Ubah ke dict mapping kota → provinsi
city_to_province = dict(zip(indonesia['city_ascii'].str.lower(), indonesia['admin_name']))

# Daftar provinsi valid untuk dropdown
provinsi_list = sorted(indonesia['admin_name'].unique().tolist())

def detect_province(place):
    """Deteksi provinsi berdasarkan nama kota yang muncul di kolom place."""
    if not isinstance(place, str):
        return "Lainnya"

    place_clean = place.lower()
    
    # 1️⃣ Cek nama kota di mapping
    for city, prov in city_to_province.items():
        if re.search(r"\b" + re.escape(city) + r"\b", place_clean):
            return prov

    # 2️⃣ Jika tidak ditemukan
    return "Lainnya"

df["province"] = df["place"].apply(detect_province)

print(df[["place", "province"]].head())

# === Setup aplikasi ===
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY, "/assets/custom.css"],
    suppress_callback_exceptions=True
)
app.title = "Realtime Earthquake Dashboard"

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

    html.Div([
        html.H5("🔍 Filter Data", className="fw-bold text-secondary mb-3"),

        dbc.Row([
            dbc.Col([
                html.Label("Regional (Provinsi)", className="fw-semibold small"),
                dcc.Dropdown(
                    id='province-filter',
                    options=[{'label': p, 'value': p} for p in sorted(df['province'].unique())],
                    value=[], multi=True,
                    placeholder="Pilih satu atau lebih provinsi..."
                ),
            ], md=4),

            dbc.Col([
                html.Label("Rentang Magnitudo", className="fw-semibold small"),
                dcc.RangeSlider(
                    id='mag-filter',
                    min=df['magnitude'].min(),
                    max=df['magnitude'].max(),
                    step=0.1,
                    marks={i: str(i) for i in range(
                        int(df['magnitude'].min()), int(df['magnitude'].max()) + 1)},
                    value=[df['magnitude'].min(), df['magnitude'].max()]
                ),
            ], md=4),

            dbc.Col([
                html.Label("Rentang Waktu", className="fw-semibold small"),
                dcc.DatePickerRange(
                    id='date-filter',
                    start_date=df['time'].min().date(),
                    end_date=df['time'].max().date(),
                    className="w-100"
                ),
            ], md=4),
        ], className="g-3"),

        html.Label("Cluster (opsional)", className="fw-semibold small mt-3"),
        dcc.Dropdown(
            id='cluster-filter',
            options=[{'label': f'Cluster {i}', 'value': i} for i in range(1, 4)],
            value=[], multi=True,
            placeholder="(Belum aktif, simulasi saja)"
        ),
    ], className="filter-card p-4 bg-white rounded-4 shadow-sm mb-4"),

    html.Div([
        html.H5("🗺️ Earthquake Map", className="fw-bold text-secondary mb-3"),
        dcc.Graph(id="map-graph", style={"height": "500px"}),
    ], className="bg-white rounded-4 shadow-sm p-3 mb-4"),

    html.Div([
        html.H5("📋 Recent Earthquakes", className="fw-bold text-secondary mb-3"),
        html.Div(id="recent-table")
    ], className="bg-white rounded-4 shadow-sm p-3")
])

# ======================================================================
#                            PAGE 2: Analysis
# ======================================================================
analysis_page = html.Div([
    html.H2("Frequency & Depth Analysis", className="fw-bold mb-3"),
    html.P("Analisis distribusi magnitudo dan kedalaman gempa di Indonesia.",
           className="text-muted"),
    dcc.Graph(
        figure=px.histogram(df, x="magnitude", nbins=20, color_discrete_sequence=["#f97316"],
                            title="Distribusi Magnitudo Gempa")
    ),
    dcc.Graph(
        figure=px.scatter(df, x="magnitude", y="depth", color="province",
                          color_discrete_sequence=px.colors.qualitative.Set2,
                          title="Korelasi Magnitudo vs Kedalaman")
    )
])

# ======================================================================
#                            PAGE 3: Regional
# ======================================================================
regional_page = html.Div([
    html.H2("Regional Summary & Cluster", className="fw-bold mb-3"),
    html.P("Lihat ringkasan aktivitas gempa per provinsi dan pola klasternya.",
           className="text-muted"),
    dcc.Graph(
        figure=px.bar(
            df.groupby("province")["magnitude"].mean().reset_index().sort_values("magnitude", ascending=False),
            x="province", y="magnitude", color="magnitude", color_continuous_scale="OrRd",
            title="Rata-rata Magnitudo per Provinsi"
        )
    )
])

# ======================================================================
#                            PAGE 4: Profile
# ======================================================================
profile_page = html.Div([
    html.H2("Profile", className="fw-bold mb-3"),
    html.P("Halaman ini bisa berisi informasi pengguna, pengaturan, dan preferensi."),
])

# ======================================================================
#                            PAGE 5: Help
# ======================================================================
help_page = html.Div([
    html.H2("Help & Support", className="fw-bold mb-3"),
    html.P("Panduan penggunaan dashboard dan kontak bantuan."),
])

# ======================================================================
#                            ROUTING
# ======================================================================
app.layout = dbc.Container([
    dcc.Location(id='url'),
    dbc.Row([
        sidebar,
        dbc.Col(html.Div(id='page-content', className="main-content p-4"), md=10)
    ])
], fluid=True)


@app.callback(Output('page-content', 'children'),
              Input('url', 'pathname'))
def display_page(pathname):
    if pathname in ['/', '/overview']:
        return overview_page
    elif pathname == '/analysis':
        return analysis_page
    elif pathname == '/regional':
        return regional_page
    elif pathname == '/profile':
        return profile_page
    elif pathname == '/help':
        return help_page
    else:
        return html.H3("404 - Page not found", className="text-danger")

# ======================================================================
#                            CALLBACK UTAMA (Overview)
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
    Input("date-filter", "start_date"),
    Input("date-filter", "end_date"),
    Input("cluster-filter", "value")
)
def update_dashboard(provinces, mag_range, start_date, end_date, clusters):
    dff = df[
        (df["magnitude"].between(mag_range[0], mag_range[1])) &
        (df["time"].dt.date.between(pd.to_datetime(start_date).date(), pd.to_datetime(end_date).date()))
    ]
    if provinces:
        dff = dff[dff["province"].isin(provinces)]

    total_quakes = len(dff)
    avg_mag = f"{dff['magnitude'].mean():.2f}" if total_quakes > 0 else "0.00"
    deepest = f"{dff['depth'].max():.2f}" if total_quakes > 0 else "0.00"
    shallowest = f"{dff['depth'].min():.2f}" if total_quakes > 0 else "0.00"

    fig = px.scatter_mapbox(
        dff, lat="latitude", lon="longitude", color="magnitude", size="magnitude",
        hover_name="place", hover_data=["depth", "time"],
        color_continuous_scale="OrRd", zoom=4, height=500
    )
    fig.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})

    recent = dff.sort_values("time", ascending=False).head(5)
    table = dbc.Table.from_dataframe(
        recent[["time", "place", "magnitude", "depth"]],
        striped=True, bordered=True, hover=True, className="table table-striped table-hover mb-0"
    )
    return total_quakes, avg_mag, deepest, shallowest, fig, table


if __name__ == "__main__":
    app.run(debug=True)