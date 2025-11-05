import pandas as pd
import glob
import os

def load_and_combine(usgs_folder, emsc_folder, output_csv="data/combined/combined.csv"):
    all_dfs = []

    # === USGS ===
    usgs_files = glob.glob(os.path.join(usgs_folder, "*.csv"))
    for file in usgs_files:
        try:
            usgs = pd.read_csv(file)
            usgs.columns = [c.strip().lower() for c in usgs.columns]

            rename_map = {
                'mag': 'magnitude',
                'magnitude': 'magnitude',
                'place': 'place',
                'depth': 'depth',
                'latitude': 'latitude',
                'longitude': 'longitude',
                'time': 'time'
            }
            usgs = usgs.rename(columns=rename_map)
            usgs['source'] = 'USGS'

            expected_cols = ['time', 'latitude', 'longitude', 'depth', 'magnitude', 'place', 'source']
            for col in expected_cols:
                if col not in usgs.columns:
                    usgs[col] = None

            usgs = usgs[expected_cols]
            all_dfs.append(usgs)
        except Exception as e:
            print(f"⚠️ Gagal baca {file}: {e}")

    # === EMSC ===
    emsc_files = glob.glob(os.path.join(emsc_folder, "*.csv"))
    for file in emsc_files:
        try:
            emsc = pd.read_csv(file)
            emsc.columns = [c.strip().lower() for c in emsc.columns]

            # gabungkan date dan time kalau ada
            if 'date' in emsc.columns and 'time' in emsc.columns:
                emsc['time'] = pd.to_datetime(emsc['date'] + ' ' + emsc['time'], errors='coerce')
            elif 'datetime' in emsc.columns:
                emsc['time'] = pd.to_datetime(emsc['datetime'], errors='coerce')

            rename_map = {
                'lat': 'latitude',
                'lon': 'longitude',
                'latitude': 'latitude',
                'longitude': 'longitude',
                'depth': 'depth',
                'mag': 'magnitude',
                'magnitude': 'magnitude',
                'region': 'place',
                'location': 'place'
            }
            emsc = emsc.rename(columns=rename_map)
            emsc['source'] = 'EMSC'

            expected_cols = ['time', 'latitude', 'longitude', 'depth', 'magnitude', 'place', 'source']
            for col in expected_cols:
                if col not in emsc.columns:
                    emsc[col] = None

            emsc = emsc[expected_cols]
            all_dfs.append(emsc)
        except Exception as e:
            print(f"⚠️ Gagal baca {file}: {e}")

    # === Gabungkan semua ===
    df = pd.concat(all_dfs, ignore_index=True)
    df.drop_duplicates(subset=['time', 'latitude', 'longitude'], inplace=True)
    df.dropna(subset=['time', 'latitude', 'longitude', 'magnitude'], inplace=True)
    df = df.sort_values('time', ascending=False).reset_index(drop=True)

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"✅ Combined dataset saved to {output_csv} with {len(df)} records from {len(usgs_files)} USGS and {len(emsc_files)} EMSC files.")
    return df


if __name__ == "__main__":
    load_and_combine("data/usgs", "data/emsc")
