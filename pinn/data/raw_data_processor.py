
"""
Preprocessing module to create a spatiotemporal dataset for COVID-19 modeling.

It merges:
- NYT COVID-19 US county-level dataset
- US Census county shapefiles (centroids as lat/lon)
- US Census 2022 county population estimates

Outputs:
- A CSV with columns [lon, lat, t, u], where u = (cases / population) * 100,000

Usage:
    python raw_data_processor.py
    or
    from raw_data_processor import generate_processed_dataset
"""

import os
import requests
import zipfile
import pandas as pd
import geopandas as gpd
import urllib3
from config import config

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)



def load_nyt_data():
    url = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv"
    df = pd.read_csv(url, parse_dates=["date"])
    df = df[df["fips"].notna()]
    df["fips"] = df["fips"].astype(float).astype(int).astype(str).str.zfill(5)
    return df

def load_geometry():
    shapefile_url = "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_county_20m.zip"
    zip_path = config["zip_path"]
    extract_path = config["extract_path"]

    if not os.path.exists(zip_path):
        r = requests.get(shapefile_url, verify=False)
        with open(zip_path, 'wb') as f:
            f.write(r.content)

    if not os.path.exists(extract_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)

    shp_files = [f for f in os.listdir(extract_path) if f.endswith(".shp")]
    assert len(shp_files) == 1
    gdf = gpd.read_file(os.path.join(extract_path, shp_files[0]))
    gdf["fips"] = gdf["STATEFP"].astype(str).str.zfill(2) + gdf["COUNTYFP"].astype(str).str.zfill(3)
    gdf_proj = gdf.to_crs(epsg=3857)
    gdf["lon"] = gdf_proj.centroid.to_crs(epsg=4326).x
    gdf["lat"] = gdf_proj.centroid.to_crs(epsg=4326).y
    return gdf[["fips", "lon", "lat"]]


def load_population_data():
    pop_url = "https://www2.census.gov/programs-surveys/popest/datasets/2020-2022/counties/totals/co-est2022-alldata.csv"
    pop_df = pd.read_csv(pop_url, encoding="latin1")
    pop_df['fips'] = pop_df['STATE'].astype(str).str.zfill(2) + pop_df['COUNTY'].astype(str).str.zfill(3)
    return pop_df[['fips', 'POPESTIMATE2022']].rename(columns={'POPESTIMATE2022': 'population'})


def generate_processed_dataset(state_filter=None, top_n=None, output_path=config["data_path"], date_cutoff=None):
    df = load_nyt_data()
    gdf = load_geometry()
    pop_df = load_population_data()

    df = df.merge(gdf, on="fips", how="inner")
    df = df.merge(pop_df, on="fips", how="inner")

    if state_filter:
        df = df[df['state'].isin(state_filter)]

    if top_n:
        top_fips = df.groupby('fips')['cases'].max().nlargest(top_n).index
        df = df[df['fips'].isin(top_fips)]

    if date_cutoff:
        df = df[df['date'] <= pd.to_datetime(date_cutoff)]

    t0 = df['date'].min()
    df['t'] = (df['date'] - t0).dt.days
    df['u'] = df['cases'] / df['population'] * 1e5
    df_out = df[['lon', 'lat', 't', 'u']].sort_values(by=['lon', 'lat', 't'])
    # os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_out.to_csv(output_path, index=False)
    print(f"Saved normalized dataset to: {output_path}")

if __name__ == "__main__":
    generate_processed_dataset()
