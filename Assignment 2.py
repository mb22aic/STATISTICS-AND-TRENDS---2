# -*- coding: utf-8 -*-
"""
@author: Manoj Mathappan
"""

# importing requiredpackages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


def read_file(file_name):
    """
    This function will read data from all required data sets
    Parameters
    ----------
    file_name : string
        Name of the file tobe read into the datarame.
    Returns
    -------
    A dataframe loaded from the file and it's transpose.
    """

    address = file_name
    df = pd.read_csv(address)
    df_transpose = pd.DataFrame.transpose(df)
    # Header setting
    header = df_transpose.iloc[0].values.tolist()
    df_transpose.columns = header
    return (df, df_transpose)


def clean_df(df):
    """

    Parameters
    ----------
    df : dataframe
        Dataframe that needs to be cleaned and index converted.
    Returns
    -------
    df : dataframe
        dataframe with required columns and index as int.
    """

    # Cleaning the dataframe
    df = df.iloc[1:]
    df = df.iloc[11:55]

    # Converting index ot int
    df.index = df.index.astype(int)
    df = df[df.index > 1989]

    # cleaning empty cells
    df = df.dropna(axis="columns")
    return df


def addlabels(x, y):
    """
    This function is to add value lables to bar chart
    """

    for i in range(len(x)):
        plt.text(
            i,
            y[i] / 0.969,
            y[i],
            ha="center",
        )


def country_df(country_name):
    """
    Creates a dataframe for the country with Agricultural Methane Emission, Non-Agricultural Methane Emission, Poverty Ratio, Access to electricity
    Parameters
    ----------
    country_name : string
        Name of the country to create the dataframe.
    Returns
    -------
        Newly created dataframe.
    """

    # creates dataframe name
    df_name = "df_" + country_name
    # creates dataframe
    df_name = pd.concat(
        [
            df_non_agri_methane_countries[country_name].astype(float),
            df_agri_methane_countries[country_name].astype(float),
            df_pov_countries[country_name].astype(float),
            df_electric_countries[country_name].astype(float),
        ],
        axis=1,
    )
    # Gives column names
    df_name.columns.values[0] = "Non Agricultural Methane"
    df_name.columns.values[1] = "Agricultural Methane"
    df_name.columns.values[2] = "Poverty headcount ratio"
    df_name.columns.values[3] = "Access to electricity"
    df_name["Agricultural Methane"] = (
        df_name["Agricultural Methane"]
        .interpolate(method="linear", axis=0)
        .ffill()
        .bfill()
    )
    return df_name


def heatmap(country_name):
    """
    Creates a correlation heatmap for the country given as argument.
    Parameters
    ----------
    country_name : string
        Name of the country to create the heatmap for.

    """

    # creates dataframe name
    df_name = "df_" + country_name
    # cals function to create dataframe
    df_name = country_df(country_name)
    # plots heatmap
    sb.heatmap(df_name.corr(), cmap="magma", annot=True)
    plt.title(country_name)
    # saves figure
    filename = country_name + "_heatmap.png"
    plt.savefig(filename, dpi=500, bbox_inches="tight")
    plt.show()


# Reads the files
df_non_agri_methane_total, df_non_agri_methane_countries = read_file(
    "Non Agricultural Methane Emission.csv"
)
df_agri_methane_total, df_agri_methane_countries = read_file(
    "Agricultural Methane Emission.csv"
)
df_pov_total, df_pov_countries = read_file("Poverty headcount ratio.csv")
df_electric_total, df_electric_countries = read_file("Access to electricity.csv")
df_emtvspov_total, df_emtvspov_countries = read_file(
    "Agricultural Methane Emission vs Poverty Ratio.csv"
)


# Creates a list of countries and years to use in the plots
countries = [
    "Bangladesh",
    "China",
    "Germany",
    "France",
    "United Kingdom",
    "India",
    "Kenya",
    "Saudi Arabia",
    "United States",
]
years = [1990, 1994, 1998, 2002, 2006, 2010, 2014]


"""
Non Agricultural methane emissions bar graph
Creating bar graph of Non Agricultural methane emissions (% of total) of multiple countries from 1990-2014
"""

# Cleaning the dataframe
df_non_agri_methane_countries = clean_df(df_non_agri_methane_countries)

# selecting only required data
df_non_agri_methane_time = pd.DataFrame.transpose(df_non_agri_methane_countries)
df_non_agri_methane_subset_time = df_non_agri_methane_time[years].copy()
df_non_agri_methane_subset_time = df_non_agri_methane_subset_time.loc[
    df_non_agri_methane_subset_time.index.isin(countries)
]

# plotting the data
n = len(countries)
r = np.arange(n)
width = 0.1

plt.bar(
    r - 0.3,
    df_non_agri_methane_subset_time[1990],
    color="burlywood",
    width=width,
    edgecolor="black",
    label="1990",
)
plt.bar(
    r - 0.2,
    df_non_agri_methane_subset_time[1994],
    color="darkseagreen",
    width=width,
    edgecolor="black",
    label="1995",
)
plt.bar(
    r - 0.1,
    df_non_agri_methane_subset_time[1998],
    color="aquamarine",
    width=width,
    edgecolor="black",
    label="2000",
)
plt.bar(
    r,
    df_non_agri_methane_subset_time[2002],
    color="crimson",
    width=width,
    edgecolor="black",
    label="2005",
)
plt.bar(
    r + 0.1,
    df_non_agri_methane_subset_time[2006],
    color="gold",
    width=width,
    edgecolor="black",
    label="2010",
)
plt.bar(
    r + 0.2,
    df_non_agri_methane_subset_time[2010],
    color="steelblue",
    width=width,
    edgecolor="black",
    label="2014",
)
plt.bar(
    r + 0.3,
    df_non_agri_methane_subset_time[2014],
    color="darkgrey",
    width=width,
    edgecolor="black",
    label="2014",
)
plt.xlabel("Countries")
plt.ylabel("Non Agricultural methane emissions")
plt.xticks(width + r, countries, rotation=90)
plt.legend()
plt.title("Non Agricultural methane emissions (% of total)")
plt.savefig("Non Agricultural methane emissions.png", bbox_inches="tight", dpi=500)
plt.show()


"""
Agricultural methane emissions bar graph
Creating bar graph of Agricultural methane emissions (% of total) by mutiple countries from 1990-2014
"""

# Cleaning the dataframe
df_agri_methane_countries = clean_df(df_agri_methane_countries)

# selecting only required data
df_agri_methane_time = pd.DataFrame.transpose(df_agri_methane_countries)
df_agri_methane_subset_time = df_agri_methane_time[years].copy()
df_agri_methane_subset_time = df_agri_methane_subset_time.loc[
    df_agri_methane_subset_time.index.isin(countries)
]

# plotting the data
n = len(countries)
r = np.arange(n)
width = 0.1
plt.bar(
    r - 0.3,
    df_agri_methane_subset_time[1990],
    color="aqua",
    width=width,
    edgecolor="black",
    label="1990",
)
plt.bar(
    r - 0.2,
    df_agri_methane_subset_time[1994],
    color="deepskyblue",
    width=width,
    edgecolor="black",
    label="1994",
)
plt.bar(
    r - 0.1,
    df_agri_methane_subset_time[1998],
    color="navy",
    width=width,
    edgecolor="black",
    label="1998",
)
plt.bar(
    r,
    df_agri_methane_subset_time[2002],
    color="turquoise",
    width=width,
    edgecolor="black",
    label="2002",
)
plt.bar(
    r + 0.1,
    df_agri_methane_subset_time[2006],
    color="blue",
    width=width,
    edgecolor="black",
    label="2006",
)
plt.bar(
    r + 0.2,
    df_agri_methane_subset_time[2010],
    color="darkgrey",
    width=width,
    edgecolor="black",
    label="2010",
)
plt.bar(
    r + 0.3,
    df_agri_methane_subset_time[2014],
    color="steelblue",
    width=width,
    edgecolor="black",
    label="2014",
)
plt.xlabel("Countries")
plt.ylabel("Agricultural methane emissions")
plt.xticks(width + r, countries, rotation=90)
plt.legend()
plt.title("Agricultural methane emissions (% of total)")
plt.savefig("Agricultural methane emissions.png", dpi=500, bbox_inches="tight")
plt.show()


"""
Scatter plot Poverty headcount ratio at national poverty lines (% of population)
Creates a scatter plot of Poverty headcount ratio at national poverty lines (% of population) of multiple countries during 1990-2014.
"""

# Clean function is not used here because it will remove entirety Column
df_pov_countries = df_pov_countries.iloc[1:]
df_pov_countries = df_pov_countries.iloc[11:55]
df_pov_countries.index = df_pov_countries.index.astype(int)
df_pov_countries = df_pov_countries[df_pov_countries.index > 1990]

df_emtvspov_countries = df_emtvspov_countries.iloc[1:]
df_emtvspov_countries = df_emtvspov_countries.iloc[11:55]
df_emtvspov_countries.index = df_emtvspov_countries.index.astype(int)
df_emtvspov_countries = df_emtvspov_countries[df_emtvspov_countries.index > 1990]


# plotting the data
plt.figure()
plt.scatter(
    df_emtvspov_countries["Bangladesh"], df_pov_countries["Bangladesh"], alpha=0.5
)
plt.scatter(df_emtvspov_countries["China"], df_pov_countries["China"], alpha=0.5)
plt.scatter(df_emtvspov_countries["Germany"], df_pov_countries["Germany"], alpha=0.5)
plt.scatter(df_emtvspov_countries["France"], df_pov_countries["France"], alpha=0.5)
plt.scatter(
    df_emtvspov_countries["United Kingdom"],
    df_pov_countries["United Kingdom"],
    alpha=0.5,
)
plt.scatter(df_emtvspov_countries["India"], df_pov_countries["India"], alpha=0.5)
plt.scatter(df_emtvspov_countries["Kenya"], df_pov_countries["Kenya"], alpha=0.5)
plt.scatter(
    df_emtvspov_countries["Saudi Arabia"], df_pov_countries["Saudi Arabia"], alpha=0.5
)
plt.scatter(
    df_emtvspov_countries["United States"], df_pov_countries["United States"], alpha=0.5
)
plt.xlabel("Methane emissions (kt of CO2 equivalent)")
plt.ylabel("Poverty headcount ratio (%)")
plt.legend(
    [
        "Bangladesh",
        "China",
        "Germany",
        "France",
        "United Kingdom",
        "India",
        "Kenya",
        "Saudi Arabia",
        "United States",
        "Israel",
    ],
    prop={"size": 7},
)
plt.title("Poverty headcount ratio vs Methane Emission")
plt.savefig("Poverty headcount ratio.png", dpi=500, bbox_inches="tight")
plt.show()


"""
Access to electricity (% of population) plot graph
Creates a plot chart of Access to electricity (% of population)
of multiple countries during 1990-2014
"""
# Cleaning the dataframe
df_electric_countries = df_electric_countries.iloc[1:]
df_electric_countries = df_electric_countries.iloc[11:55]
df_electric_countries.index = df_electric_countries.index.astype(int)
df_electric_countries = df_electric_countries[df_electric_countries.index > 1990]

# plotting the data
plt.plot(df_electric_countries.index, df_electric_countries["Bangladesh"])
plt.plot(df_electric_countries.index, df_electric_countries["China"])
plt.plot(df_electric_countries.index, df_electric_countries["Germany"])
plt.plot(df_electric_countries.index, df_electric_countries["France"])
plt.plot(df_electric_countries.index, df_electric_countries["United Kingdom"])
plt.plot(df_electric_countries.index, df_electric_countries["India"])
plt.plot(df_electric_countries.index, df_electric_countries["Kenya"])
plt.plot(df_electric_countries.index, df_electric_countries["Saudi Arabia"])
plt.plot(df_electric_countries.index, df_electric_countries["United States"])
plt.xlabel("Year")
plt.ylabel("Access to electricity")
plt.legend(
    [
        "Bangladesh",
        "China",
        "Germany",
        "France",
        "United Kingdom",
        "India",
        "Kenya",
        "Saudi Arabia",
        "United States",
        "Israel",
    ],
    prop={"size": 7},
    loc="upper left",
)
plt.title("Access to electricity (% of population)")
plt.savefig("Access to electricity.png", dpi=500, bbox_inches="tight")
plt.show()


# using describe() function of pandas to get the stats of 9 Countries access to electricity for last 5 years

df_electric_countries_t = df_electric_total[
    df_electric_total["Country Name"].isin(countries)
]
print(df_electric_countries_t.describe().iloc[:, -5:])

"""
Calls the functions to create the heatmaps
"""

heatmap("India")
