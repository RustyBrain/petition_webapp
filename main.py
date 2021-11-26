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

# Set page header to display in browser
st.set_page_config(page_title='Petition Analyser')

# Set the title
st.title("Petition analyser - who's been signing what petitions")


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


# Need this to be able to get the geojson file
ssl._create_default_https_context = ssl._create_unverified_context

@st.cache
def get_geojson():
    """
    Gets the geojson file for map
    :return:
    """
    return gpd.read_file('http://geoportal1-ons.opendata.arcgis.com/datasets/4c191bee309d4b2d8b2c5b94d2512af9_0.geojson')


# Get the geojson file for UK
geog = get_geojson()


@st.cache
def get_deprivation_and_population_data():
    """
    Gets deprivation information and population estimates
    :return:
    """
    deprivation_data = pd.read_excel("https://researchbriefings.files.parliament.uk/documents/CBP-7327/CBP7327.xlsx",
                                     sheet_name='2019 Scores', skiprows=6, skipfooter=6)

    pop_est_zip = requests.get("https://www.ons.gov.uk/file?uri=/peoplepopulationandcommunity/populationandmigration/"
                               "populationestimates/datasets/parliamentaryconstituencymidyearpopulationestimates/"
                               "mid2019sape22dt7/sape22dt7mid2019parliconsyoaestimatesunformatted.zip")
    return deprivation_data, pop_est_zip

# Get the deprivation data and population estimates, extract and store locally
deprivation_data, pop_est_zip = get_deprivation_and_population_data()
z = zipfile.ZipFile(io.BytesIO(pop_est_zip.content))
z.extractall()

# Set up user input for the petition
# "https://petition.parliament.uk/petitions/586700.json" <- initally designed for
url = st.text_input('Add the URL for the petition from https://petition.parliament.uk/petitions')

# Once we have a url
if url != "":
    # Get the data from the petitions website and check to see if it is valid
    try:
        r = requests.get(url + ".json")
        if r.status_code == 404:
            st.error("This doesn't seem to be a valid url from the petitions website")
    except:
        st.error("This doesn't seem to be a valid url from the petitions website")


    # get the data for each Constituency as a df and get the metadata as dict
    constituency_data = r.json()['data']['attributes']['signatures_by_constituency']
    petition_data = pd.DataFrame(constituency_data)
    metadata = r.json()['data']['attributes']

    # merge the petition and deprivation data
    merged = pd.merge(petition_data, deprivation_data, left_on='name', right_on='Constituency')

    # Get the population estimates. If children are mentioned in the description of the petition then set the age to
    # under 5s, else use all ages
    pop_est = pd.read_excel("SAPE22DT7-mid-2019-parlicon-syoa-estimates-unformatted.xlsx",
                            sheet_name='Mid-2019 Persons', skiprows=4)
    if 'child' in metadata['background']:
        pop_est = pop_est[['PCON11CD', 0, 1, 2, 3, 4]]
        pop_est['total'] = pop_est.sum(axis=1)
        sig_per = 'Signatures per child'
    else:
        pop_est['total'] = pop_est['All Ages']
        sig_per = 'Signatures per person'

    # Merge in the population estimates
    merged = pd.merge(merged, pop_est, left_on='ons_code', right_on='PCON11CD')

    # Allocate the constituency data, populations and deprivation into deprivation deciles using standard qcut method
    df = defined_qcut(merged, 'Index of Multiple Deprivation', 10, [0, 9, 1, 2, 3, 4, 5, 6, 7, 8], col_name='IMD')
    df = defined_qcut(df, 'Income deprivation affecting children index', 10, [0, 9, 1, 2, 3, 4, 5, 6, 7, 8],
                      col_name='Income deprivation affecting children')
    df = defined_qcut(df, 'Income deprivation', 10, [0, 9, 1, 2, 3, 4, 5, 6, 7, 8],
                      col_name='Income deprivation')
    df = defined_qcut(df, 'Employment deprivation', 10, [0, 9, 1, 2, 3, 4, 5, 6, 7, 8],
                      col_name='Employment deprivation decile')

    # Get the signatures per population
    df['sig_per'] = df['signature_count'] / df['total']

    # Merge the data onto the geojson
    geog_with_data = geog.merge(df, left_on='pcon19nm', right_on='name', how='left')

    # Filter out Scotland, Wales and NI (don't have deprivation data in same format)
    geog_with_data = geog_with_data[geog_with_data['pcon19cd'].str.startswith('E', na=False)]

    # Keep only the relevant fields
    geog_with_data_only = geog_with_data[['geometry', 'IMD']]

    # Inform the user
    st.success('Successfully got data!')

    st.header(metadata['action'])
    """
    This page runs analysis on the a petition that has been submitted to the UK Gov petition site. 
    
    The analysis focuses on where the people who have signed the petition live, and the make up of that area.
    """
    # Inform the user about the petition and how many sigs it got
    st.subheader('Petition information')
    st.write(metadata['background'])

    st.subheader('Number of signatures:' )
    st.write(metadata['signature_count'])

    # Set the inital state of the map
    INITIAL_VIEW_STATE = pdk.ViewState(latitude=52, longitude=0, zoom=4, max_zoom=16, pitch=25, bearing=0)

    """
    ## Geographical distribution of signatories. 
    
    """
    st.write(f"The bars represent the number of {sig_per} in those areas. The shading of the constituencies \
              represent the deprivation in that area (more grey is less deprived).")

    # Display map
    geog_lat_long = geog_with_data[['long', 'lat', 'sig_per']]
    geog_lat_long.dropna(inplace=True)
    long = geog_lat_long['long'].repeat(round(geog_lat_long['sig_per'] * 1000, 0))
    lat = geog_lat_long['lat'].repeat(round(geog_lat_long['sig_per'] * 1000, 0))
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

    # Plot the signatures by deprivation decile for overall deprivation, income deprivation, and employment deprivation

    """
    ## Deprivation
     
    """
    st.write(f"The {sig_per} split by deprivation.")
    deprivation = df.groupby(['IMD']).mean()
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.barplot(deprivation.index, deprivation['sig_per'].values, ax=ax)
    ax.set(xlabel='Deprivation Decile (1 is least deprived)', ylabel=sig_per)
    st.pyplot(fig)
    plt.clf()

    """
    ## Income deprivation
    """
    st.write(f"The {sig_per} split by income deprivation.")
    if sig_per == 'Signatures per child':
        income = df.groupby(['Income deprivation affecting children']).mean()
        xlab = 'Income deprivation affecting children decile (1 is least deprived)'
    else:
        income = df.groupby(['Income deprivation']).mean()
        xlab = 'Income deprivation decile (1 is least deprived)'
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.barplot(income.index, income['sig_per'].values, ax=ax)
    ax.set(xlabel=xlab, ylabel=sig_per)
    st.pyplot(fig)
    plt.clf()

    """
    ## Employment deprivation 
    
    """
    st.write(f"The {sig_per} split by employment deprivation.")
    employment_deprivation = df.groupby(['Employment deprivation decile']).mean()
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.barplot(employment_deprivation.index, employment_deprivation['sig_per'].values, ax=ax)
    ax.set(xlabel='Employment deprivation decile (1 is least deprived)', ylabel=sig_per)
    st.pyplot(fig)
    plt.clf()
