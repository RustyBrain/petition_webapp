import pandas as pd
import geopandas as gpd
import requests
import ssl
import io
import zipfile
import streamlit as st


def defined_qcut(df, value_series, number_of_bins, bins_for_extras, labels=False):
    """
    Allows users to define how values are split into bins when clustering.
    :param df: Dataframe of values
    :param value_series: Name of value column to rank upon
    :param number_of_bins: Integer of number of bins to create
    :param bins_for_extras: Ordered list of bin numbers to assign uneven splits
    :param labels: Optional. Labels for bins if required
    :return: A dataframe with a new column 'bins' which contains the cluster numbers
    """
    if max(bins_for_extras) > number_of_bins or any(x < 0 for x in bins_for_extras):
        raise ValueError('Attempted to allocate to a bin that doesnt exist')
    base_number, number_of_values_to_allocate = divmod(df[value_series].count(), number_of_bins)
    bins_for_extras = bins_for_extras[:number_of_values_to_allocate]
    if number_of_values_to_allocate == 0:
        df['bins'] = pd.qcut(df[value_series], number_of_bins, labels=labels)
        return df
    elif number_of_values_to_allocate > len(bins_for_extras):
        raise ValueError('There are more values to allocate than the list provided, please select more bins')
    bins = {}
    for i in range(number_of_bins):
        number_of_values_in_bin = base_number
        if i in bins_for_extras:
            number_of_values_in_bin += 1
        bins[i] = number_of_values_in_bin
    df['rank'] = df[value_series].rank()
    df = df.sort_values(by=['rank'])
    df['bins'] = 0
    row_to_start_allocate = 0
    row_to_end_allocate = 0
    for bin_number, number_in_bin in bins.items():
        row_to_end_allocate += number_in_bin
        bins.update({bin_number: [number_in_bin, row_to_start_allocate, row_to_end_allocate]})
        row_to_start_allocate = row_to_end_allocate
    conditions = [df['rank'].iloc[v[1]: v[2]] for k, v in bins.items()]
    series_to_add = pd.Series()
    for idx, series in enumerate(conditions):
        series[series > -1] = idx
        series_to_add = series_to_add.append(series)
    df['bins'] = series_to_add
    df['bins'] = df['bins'] + 1
    return df


ssl._create_default_https_context = ssl._create_unverified_context

r = requests.get("https://petition.parliament.uk/petitions/586700.json")
constituency_data = r.json()['data']['attributes']['signatures_by_constituency']
petition_data = pd.DataFrame(constituency_data)
deprivation_data = pd.read_excel("https://researchbriefings.files.parliament.uk/documents/CBP-7327/CBP7327.xlsx",
                                 sheet_name='2019 Scores', skiprows=6, skipfooter=6)

pop_est_zip = requests.get("https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/populationandmigration/"
                           "populationestimates/datasets/parliamentaryconstituencymidyearpopulationestimates/"
                           "mid2019sape22dt7/sape22dt7mid2019parliconsyoaestimatesunformatted.zip")

z = zipfile.ZipFile(io.BytesIO(pop_est_zip.content))
z.extractall()

merged = pd.merge(petition_data, deprivation_data, left_on='name', right_on='Constituency')

pop_est = pd.read_excel("SAPE22DT7-mid-2019-parlicon-syoa-estimates-unformatted.xlsx", sheet_name='Mid-2019 Persons',
                        skiprows=4)

pop_est = pop_est[['PCON11CD', 0, 1, 2, 3, 4]]
pop_est['total_under_5'] = pop_est.sum(axis=1)

merged = pd.merge(merged, pop_est, left_on='ons_code', right_on='PCON11CD')

df = defined_qcut(merged, 'Index of Multiple Deprivation', 10, [0, 9, 1, 2, 3, 4, 5, 6, 7, 8])
df['sig_per_child'] = df['signature_count'] / df['total_under_5']

geog = gpd.read_file('http://geoportal1-ons.opendata.arcgis.com/datasets/4c191bee309d4b2d8b2c5b94d2512af9_0.geojson')

geog = geog.merge(df, left_on='pcon11cd', right_on='PCON11CD', how='left')

st.map(geog)

