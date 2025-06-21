import os
import requests
import zipfile
import pandas as pd
import geopandas as gpd
import urllib3

# Disable SSL warnings for trusted source (census.gov)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def load_nyt_data():
    print("Loading NYT COVID-19 county-level data...")
    url = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv"
    df = pd.read_csv(url, parse_dates=["date"])

    print("Raw NYT shape:", df.shape)

    df = df[df["fips"].notna()]
    df["fips"] = df["fips"].astype(float).astype(int).astype(str).str.zfill(5)

    print("After FIPS normalization:", df.shape)
    print("Sample NYT FIPS:", df["fips"].drop_duplicates().sort_values().head().tolist())
    return df


def load_geometry():
    print("Downloading and extracting county shapefiles...")
    shapefile_url = "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_us_county_20m.zip"
    zip_path = "D:/Programming/Summer-2025-DL-Project-PINN/data/county_shapefiles.zip"
    extract_path = "D:/Programming/Summer-2025-DL-Project-PINN/data/county_shapefiles"

    # Download
    if not os.path.exists(zip_path):
        r = requests.get(shapefile_url, verify=False)
        with open(zip_path, 'wb') as f:
            f.write(r.content)

    if not os.path.exists(extract_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    # Find .shp file
    shp_files = [f for f in os.listdir(extract_path) if f.endswith(".shp")]
    assert len(shp_files) == 1, "Could not find shapefile in extracted ZIP"
    shp_file_path = os.path.join(extract_path, shp_files[0])

    # Drop unnecessary columns and compute centroids
    print("Reading shapefile and computing centroids...")
    gdf = gpd.read_file(shp_file_path)

    # Standardize FIPS code (make 5-digit strings)
    gdf["fips"] = gdf["STATEFP"].astype(str).str.zfill(2) + gdf["COUNTYFP"].astype(str).str.zfill(3)
    gdf = gdf[["fips", "geometry"]].copy()

    # Project to compute accurate centroids, then back to lat/lon
    gdf_proj = gdf.to_crs(epsg=3857)
    gdf["lon"] = gdf_proj.centroid.to_crs(epsg=4326).x
    gdf["lat"] = gdf_proj.centroid.to_crs(epsg=4326).y

    print("Sample SHAPEFILE FIPS:", gdf["fips"].drop_duplicates().sort_values().head().tolist())
    return gdf


def load_population_data():
    print("Loading county population estimates...")
    pop_url = "https://www2.census.gov/programs-surveys/popest/datasets/2020-2022/counties/totals/co-est2022-alldata.csv"
    pop_df = pd.read_csv(pop_url, encoding="latin1")
    pop_df['fips'] = pop_df['STATE'].astype(str).str.zfill(2) + pop_df['COUNTY'].astype(str).str.zfill(3)
    pop_df = pop_df[['fips', 'POPESTIMATE2022']].rename(columns={'POPESTIMATE2022': 'population'})
    return pop_df


def main(state_filter=None, top_n=None, output_path="D:/Programming/Summer-2025-DL-Project-PINN/data/covid_county_cases.csv"):
    df = load_nyt_data()
    gdf = load_geometry()
    pop_df = load_population_data()

    print("Merging datasets...")
    print("NYT shape before merge:", df.shape)

    df = df.merge(gdf, on="fips", how="inner")
    print("After geo merge:", df.shape)

    df = df.merge(pop_df, on="fips", how="inner")
    print("After population merge:", df.shape)

    if state_filter:
        print(f"Filtering to state: {state_filter}")
        df = df[df['state'].isin(state_filter)]

    if top_n:
        print(f"Selecting top {top_n} counties by total cases...")
        top_fips = df.groupby('fips')['cases'].max().nlargest(top_n).index
        df = df[df['fips'].isin(top_fips)]

    # Normalize time
    t0 = df['date'].min()
    df['t'] = (df['date'] - t0).dt.days

    # Normalize cases per 100k population
    df['u'] = df['cases'] / df['population'] * 1e5

    df_out = df[['lon', 'lat', 't', 'u']].sort_values(by=['lon', 'lat', 't'])

    # Save
    # Normalize time and cases
    t0 = df["date"].min()
    df["t"] = (df["date"] - t0).dt.days
    df["u"] = df["cases"] / df["population"] * 1e5

    df_out = df[["lon", "lat", "t", "u"]].sort_values(by=["lon", "lat", "t"])
    df_out.to_csv(output_path, index=False)
    print(f"Saved normalized dataset to: {output_path}")



if __name__ == "__main__":
    main()

