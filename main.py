import pandas as pd
import geopandas as gpd
import requests
import ssl
import io
import zipfile
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pydeck as pdk



def defined_qcut(df, value_series, number_of_bins, bins_for_extras, labels=False, col_name=None):
    """
    Allows users to define how values are split into bins when clustering.
    :param df: Dataframe of values
    :param value_series: Name of value column to rank upon
    :param number_of_bins: Integer of number of bins to create
    :param bins_for_extras: Ordered list of bin numbers to assign uneven splits
    :param labels: Optional. Labels for bins if required
    :param col_name: Optional. Name of column to return, 'bins' if False
    :return: A dataframe with a new column 'bins' which contains the cluster numbers
    """
    if not col_name:
        col_name = 'bins'
    if max(bins_for_extras) > number_of_bins or any(x < 0 for x in bins_for_extras):
        raise ValueError('Attempted to allocate to a bin that doesnt exist')
    base_number, number_of_values_to_allocate = divmod(df[value_series].count(), number_of_bins)
    bins_for_extras = bins_for_extras[:number_of_values_to_allocate]
    if number_of_values_to_allocate == 0:
        df[col_name] = pd.qcut(df[value_series], number_of_bins, labels=labels)
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
    df[col_name] = 0
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
    df[col_name] = series_to_add
    df[col_name] = df[col_name] + 1
    return df


ssl._create_default_https_context = ssl._create_unverified_context

@st.cache
def get_geojson():
    return gpd.read_file('http://geoportal1-ons.opendata.arcgis.com/datasets/4c191bee309d4b2d8b2c5b94d2512af9_0.geojson')


geog = get_geojson()


@st.cache
def get_deprivation_and_population_data():
    deprivation_data = pd.read_excel("https://researchbriefings.files.parliament.uk/documents/CBP-7327/CBP7327.xlsx",
                                     sheet_name='2019 Scores', skiprows=6, skipfooter=6)

    pop_est_zip = requests.get("https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/populationandmigration/"
                               "populationestimates/datasets/parliamentaryconstituencymidyearpopulationestimates/"
                               "mid2019sape22dt7/sape22dt7mid2019parliconsyoaestimatesunformatted.zip")
    return deprivation_data, pop_est_zip


deprivation_data, pop_est_zip = get_deprivation_and_population_data()

z = zipfile.ZipFile(io.BytesIO(pop_est_zip.content))
z.extractall()

# "https://petition.parliament.uk/petitions/586700.json" <- initally designed for
url = st.text_input('Add the URL for the petition from https://petition.parliament.uk/petitions')

if url != "":
    r = requests.get(url + ".json")
    constituency_data = r.json()['data']['attributes']['signatures_by_constituency']
    petition_data = pd.DataFrame(constituency_data)

    merged = pd.merge(petition_data, deprivation_data, left_on='name', right_on='Constituency')

    pop_est = pd.read_excel("SAPE22DT7-mid-2019-parlicon-syoa-estimates-unformatted.xlsx", sheet_name='Mid-2019 Persons',
                            skiprows=4)

    pop_est = pop_est[['PCON11CD', 0, 1, 2, 3, 4]]
    pop_est['total_under_5'] = pop_est.sum(axis=1)

    merged = pd.merge(merged, pop_est, left_on='ons_code', right_on='PCON11CD')

    df = defined_qcut(merged, 'Index of Multiple Deprivation', 10, [0, 9, 1, 2, 3, 4, 5, 6, 7, 8], col_name='IMD')
    df = defined_qcut(df, 'Income deprivation affecting children index', 10, [0, 9, 1, 2, 3, 4, 5, 6, 7, 8],
                      col_name='Income deprivation affecting children')
    df = defined_qcut(df, 'Employment deprivation', 10, [0, 9, 1, 2, 3, 4, 5, 6, 7, 8],
                      col_name='Employment deprivation decile')

    df['sig_per_child'] = df['signature_count'] / df['total_under_5']

    geog_with_data = geog.merge(df, left_on='pcon19nm', right_on='name', how='left')

    geog_with_data = geog_with_data[geog_with_data['pcon19cd'].str.startswith('E', na=False)]

    geog_with_data_only = geog_with_data[['geometry', 'IMD']]
    metadata = r.json()['data']['attributes']
    st.success('Successfully got data!')

    st.title(metadata['action'])
    """
    This page runs analysis on the a petition that has been submitted to the UK Gov petition site. 
    
    The analysis focuses on where the people who have signed the petition live, and the make up of that area.
    """
    st.subheader('Petition information')
    st.write(metadata['background'])

    st.subheader('Number of signatures:' )
    st.write(metadata['signature_count'])


    INITIAL_VIEW_STATE = pdk.ViewState(latitude=52, longitude=0, zoom=4, max_zoom=16, pitch=25, bearing=0)

    """
    # Geographical distribution of signatories. 
    The bars represent the number of signatures per child under 5 in those areas. The shading of the constituencies 
    represent the deprivation in that area (more grey is less deprived). 
    """

    geog_lat_long = geog_with_data[['long', 'lat', 'sig_per_child']]
    geog_lat_long.dropna(inplace=True)
    long = geog_lat_long['long'].repeat(round(geog_lat_long['sig_per_child'] * 1000, 0))
    lat = geog_lat_long['lat'].repeat(round(geog_lat_long['sig_per_child'] * 1000, 0))
    hex_data = pd.concat([long, lat], axis=1)


    st.pydeck_chart(pdk.Deck(
        map_style='mapbox://styles/mapbox/light-v9',
        initial_view_state=INITIAL_VIEW_STATE,
        layers=[
            pdk.Layer(
                "GeoJsonLayer",
                data=geog_with_data_only,
                get_fill_color="[255, 255, 255 / IMD]",
                opacity=0.8,
                stroked=False,
                filled=True,
                extruded=True,
                wireframe=True,
            ),
            pdk.Layer(
               'HexagonLayer',
               data=hex_data,
               get_position='[long, lat]',
               radius=3000,
               elevation_scale=4,
               elevation_range=[0, 70000],
               extruded=True,
            ),
        ],
        height=1200,
        width=800,
    ))


    """
    # Deprivation
    The signatures per child under 5 split by deprivation. The less deprived areas have a significantly higher response rate
     than the more deprived areas. 
    """
    deprivation = df.groupby(['IMD']).mean()
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.barplot(deprivation.index, deprivation['sig_per_child'].values, ax=ax)
    ax.set(xlabel='Deprivation Decile (1 is least deprived)', ylabel='Signatures per child')
    st.pyplot(fig)
    plt.clf()

    """
    # Income deprivation affecting children
    The signatures per child under 5 split by income deprivation affecting children. The less deprived areas have a 
    significantly higher response rate than the more deprived areas. 
    """
    income_kids = df.groupby(['Income deprivation affecting children']).mean()
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.barplot(income_kids.index, income_kids['sig_per_child'].values, ax=ax)
    ax.set(xlabel='Income deprivation affecting children decile (1 is least deprived)', ylabel='Signatures per child')
    st.pyplot(fig)
    plt.clf()

    """
    # Employment deprivation 
    The signatures per child under 5 split by employment deprivation. The less deprived areas have a 
    significantly higher response rate than the more deprived areas. 
    """
    employment_deprivation = df.groupby(['Employment deprivation decile']).mean()
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.barplot(employment_deprivation.index, employment_deprivation['sig_per_child'].values, ax=ax)
    ax.set(xlabel='Employment deprivation decile (1 is least deprived)', ylabel='Signatures per child')
    st.pyplot(fig)
    plt.clf()
