import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import matplotlib as mpl
import numba
import squarify
import numpy as np
from math import pi
from matplotlib import rcParams
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture as GMM
from umap import UMAP
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler as st
import pandas as pd
import community
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from datetime import date
from warnings import filterwarnings
filterwarnings('ignore')


# +
def get_date_today():
    
    """
    Get today's date in yymmdd format. 
    """
    
    today = date.today()
    today = str(today)

    year = today.split('-')[0][2:4]
    month = today.split('-')[1]
    day = today.split('-')[2]

    date_today = year + month + day
    
    return date_today

def read(fname):
    """
    Simple helper function to load data. 
    """
    path_to_data = '../data/'
    df = pd.read_csv(path_to_data +fname, delimiter = ';', error_bad_lines = False, 
                    encoding = 'UTF-8')
    
    return df 

def stringify_id(df, id_col):
    """
    Helper function to stringify ids. 
    """
    
    df[id_col] = [str(x) for x in df['id'].values]
    
    return df[id_col]


# -------   PLOTTING FUNCTIONS -------------------------

rcParams['axes.titlepad'] = 20 

def set_plotting_style():
      
    """
    Plotting style parameters, based on the RP group. 
    """    
        
    tw = 1.5

    rc = {'lines.linewidth': 2,
        'axes.labelsize': 18,
        'axes.titlesize': 21,
        'xtick.major' : 16,
        'ytick.major' : 16,
        'xtick.major.width': tw,
        'xtick.minor.width': tw,
        'ytick.major.width': tw,
        'ytick.minor.width': tw,
        'xtick.labelsize': 'large',
        'ytick.labelsize': 'large',
        'font.family': 'sans',
        'weight':'bold',
        'grid.linestyle': ':',
        'grid.linewidth': 1.5,
        'grid.color': '#ffffff',
        'mathtext.fontset': 'stixsans',
        'mathtext.sf': 'fantasy',
        'legend.frameon': True,
        'legend.fontsize': 12, 
       "xtick.direction": "in","ytick.direction": "in"}



    plt.rc('text.latex', preamble=r'\usepackage{sfmath}')
    plt.rc('mathtext', fontset='stixsans', sf='sans')
    sns.set_style('ticks', rc=rc)

    #sns.set_palette("colorblind", color_codes=True)
    sns.set_context('notebook', rc=rc)



def ecdf(x, plot = None, label = None):
    '''
	Compute and plot ECDF. 

	----------------------
	Inputs

	x: array or list, distribution of a random variable
    
    plot: bool, if True return the plot of the ECDF

    label: string, label for the plot
	
	Outputs 

	x_sorted : sorted x array
	ecdf : array containing the ECDF of x


    '''
    x_sorted = np.sort(x)
    
    n = len (x)
    

    ecdf = np.linspace(0, 1, len(x_sorted))

    if label is None and plot is True: 
        
        plt.scatter(x_sorted, ecdf, alpha = 0.7)

    
    elif label is not None and plot is True: 
        
        plt.scatter(x_sorted, ecdf, alpha = 0.7, label = label)
        
    return x_sorted, ecdf


def make_treemap(x_keys, x_counts):
    
    '''
    
    Wrapper function to plot treemap using the squarify module. 
    
    -------------------------------------------
    x_keys = names of the different categories 
    x_counts = counts of the given categories
    
    '''
    
    norm = mpl.colors.Normalize(vmin=min(x_counts), vmax=max(x_counts))
    colors = [mpl.cm.Greens(norm(value)) for value in x_counts]
    
    plt.figure(figsize=(14,8))
    squarify.plot(label= x_keys, sizes= x_counts, color = colors, alpha=.6)
    plt.axis('off');

    


def make_radar_chart(x_keys, x_counts):
    
    '''
    Wrapper function to make radar chart.
    
    ------------------------------------------
    
    x_keys = names of the different categories 
    x_counts = counts of the given categories    
    
    '''
    
    categories = list(x_keys)
    N = len(categories)
    
    if N > 30: 
        
        print('The categories are too big to visualize in a treemap.')
        
    else:    

        values = list(x_counts)
        values.append(values[0])
        values_sum = np.sum(values[:-1])

        percentages= [(val/values_sum)*100 for val in values]

        #angles
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]

        sns.set_style('whitegrid')

        # Initialize figure
        plt.figure(1, figsize=(7, 7))

        # Initialise the polar plot
        ax = plt.subplot(111, polar=True)

        # Draw one ax per variable + add labels labels yet
        plt.xticks(angles[:-1], categories, color='grey', size=12)

        #Set first variable to the vertical axis 
        ax.set_theta_offset(pi / 2)

        #Set clockwise rotation
        ax.set_theta_direction(-1)

        #Set yticks to gray color 

        ytick_1, ytick_2, ytick_3 = np.round(max(percentages)/3),np.round((max(percentages)/3)*2),np.round(max(percentages)/3)*3

        plt.yticks([ytick_1, ytick_2, ytick_3], [ytick_1, ytick_2, ytick_3],
                   color="grey", size=10)

        plt.ylim(0, int(max(percentages)) + 4)


        # Plot data
        ax.plot(angles, percentages, linewidth=1,color = 'lightgreen')

        # Fill area
        ax.fill(angles, percentages, 'lightgreen', alpha=0.3);    




def plot_distplot_feature(data, col_name):
    
    """
    Get a histogram with the y axis in log-scale
    """
    
    plt.hist(data[col_name].values, bins = int(data.shape[0]/10000),
             color = 'dodgerblue')
    
    plt.yscale('log')
    plt.xlabel(col_name)
    plt.ylabel('frequency')


def plot_boxplot_feature(data, col_name, hue_col_name):
    
    
    """ 
    
    Get a boxplot with the variable in the x axis, in log scale. 
    
    You also need to provide a hue column name. 
    
    """
    sns.boxplot(data = data, x = col_name, y = hue_col_name, palette = 'RdBu')
    
    plt.xscale('log')



def palette(cmap = None):

	palette = sns.cubehelix_palette(start = 0, rot=0, hue = 1, light = 0.9, dark = 0.15)
	

	if cmap == True:
		palette = sns.cubehelix_palette(start = 0, rot=0, hue = 1, light = 0.9, dark = 0.15, as_cmap = True)

	return palette 


# ---------PANDAS FUNCTIONS FOR DATA EXPLORATION -------------------------

def count_feature_types(data):
    
    """
    
    Get the dtype counts for a dataframe's columns. 
    
    """
    
    df_feature_type = data.dtypes.sort_values().to_frame('feature_type')\
    .groupby(by='feature_type').size().to_frame('count').reset_index()
    
    return df_feature_type


def get_df_missing_columns(data):
    
    '''
    
    Get a dataframe of the missing values in each column with its corresponding dtype.
    
    '''
    
    # Generate a DataFrame with the % of missing values for each column
    df_missing_values = (data.isnull().sum(axis = 0) / len(data) * 100)\
                        .sort_values(ascending = False)\
                        .to_frame('% missing_values').reset_index()
    
    # Generate a DataFrame that indicated the data type for each column
    df_feature_type = data.dtypes.to_frame('feature_type').reset_index()
    
    # Merge frames
    missing_cols_df = pd.merge(df_feature_type, df_missing_values, on = 'index',
                         how = 'inner')

    missing_cols_df.sort_values(['% missing_values', 'feature_type'], inplace = True)
    
    
    return missing_cols_df


def find_constant_features(data):
    
    """
    
    Get a list of the constant features in a dataframe. 
    
    """
    const_features = []
    for column in list(data.columns):
        if data[column].unique().size < 2:
            const_features.append(column)
    return const_features


def duplicate_columns(frame):
    '''
    Get a list of the duplicate columns in a pandas dataframe.
    '''
    groups = frame.columns.to_series().groupby(frame.dtypes).groups
    dups = []

    for t, v in groups.items():

        cs = frame[v].columns
        vs = frame[v]
        lcs = len(cs)

        for i in range(lcs):
            ia = vs.iloc[:,i].values
            for j in range(i+1, lcs):
                ja = vs.iloc[:,j].values
                if np.array_equal(ia, ja):
                    dups.append(cs[i])
                    break
    return dups


def get_duplicate_columns(df):
        
    """
    Returns a list of duplicate columns 
    """
    
    groups = df.columns.to_series().groupby(df.dtypes).groups
    dups = []

    for t, v in groups.items():

        cs = df[v].columns
        vs = df[v]
        lcs = len(cs)

        for i in range(lcs):
            ia = vs.iloc[:,i].values
            for j in range(i+1, lcs):
                ja = vs.iloc[:,j].values
                if np.array_equal(ia, ja):
                    dups.append(cs[i])
                    break
    return dups


def get_df_stats(df):
    
    """
    Wrapper for dataframe stats. 
    
    Output: missing_cols_df, const_feats, dup_cols_list
    """
    missing_cols_df = get_df_missing_columns(df)
    const_features_list = find_constant_features(df)
    dup_cols_list = duplicate_columns(df)

    return missing_cols_df, const_features_list, dup_cols_list



def convert_to_datetime(df, col):
	    
    """
    Convert a column to datetime format
    """    
    col = pd.to_datetime(col)
    
    return col



def test_missing_data(df, fname):
    
    """Look for missing entries in a DataFrame."""
    
    assert np.all(df.notnull()), fname + ' contains missing data'



def col_encoding(df, column):
    
    """
    Returns a one hot encoding of a categorical colunmn of a DataFrame.
    
    ------------------------------------------------------------------
    
    inputs~~

    -df:
    -column: name of the column to be one-hot-encoded in string format.
    
    outputs~~
    
    - hot_encoded: one-hot-encoding in matrix format. 
    
    """
    
    le = LabelEncoder()
    
    label_encoded = le.fit_transform(df[column].values)
    
    hot = OneHotEncoder(sparse = False)
    
    hot_encoded = hot.fit_transform(label_encoded.reshape(len(label_encoded), 1))
    
    return hot_encoded


def one_hot_df(df, cat_col_list):
    
    """
    Make one hot encoding on categoric columns.
    
    Returns a dataframe for the categoric columns provided.
    -------------------------
    inputs
    
    - df: original input DataFrame
    - cat_col_list: list of categorical columns to encode.
    
    outputs
    - df_hot: one hot encoded subset of the original DataFrame.
    """

    df_hot = pd.DataFrame()

    for col in cat_col_list:     

        encoded_matrix = col_encoding(df, col)

        df_ = pd.DataFrame(encoded_matrix,
                           columns = [col+ ' ' + str(int(i))\
                                      for i in range(encoded_matrix.shape[1])])

        df_hot = pd.concat([df_hot, df_], axis = 1)
        
    return df_hot


# OTHER FUNCTIONS

def plot_kmeans(kmeans, X, n_clusters=4, rseed=0, ax=None):
    
    """
    Wrapper from JakeVDP data analysis handbook
    """
    labels = kmeans.fit_predict(X)

    # plot the input data
    ax = ax or plt.gca()
    ax.axis('equal')
    ax.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis', zorder=2)

    # plot the representation of the KMeans model
    centers = kmeans.cluster_centers_
    radii = [cdist(X[labels == i], [center]).max()
             for i, center in enumerate(centers)]
    for c, r in zip(centers, radii):
        ax.add_patch(plt.Circle(c, r, fc='#CCCCCC', lw=3, alpha=0.5, zorder=1))




@numba.jit(nopython=True)
def draw_bs_sample(data):
    """
    Draw a bootstrap sample from a 1D data set.
    """
    return np.random.choice(data, size=len(data))


def net_stats(G):
    
    '''Get basic network stats and plots. Specifically degree and clustering coefficient distributions.'''
    
    net_degree_distribution= []

    for i in list(G.degree()):
        net_degree_distribution.append(i[1])
        
    print("Number of nodes in the network: %d" %G.number_of_nodes())
    print("Number of edges in the network: %d" %G.number_of_edges())
    print("Avg node degree: %.2f" %np.mean(list(net_degree_distribution)))
    print('Avg clustering coefficient: %.2f'%nx.cluster.average_clustering(G))
    print('Network density: %.2f'%nx.density(G))

    
    fig, axes = plt.subplots(1,2, figsize = (16,4))

    axes[0].hist(list(net_degree_distribution), bins=20, color = 'lightblue')
    axes[0].set_xlabel("Degree $k$")
    
    #axes[0].set_ylabel("$P(k)$")
    
    axes[1].hist(list(nx.clustering(G).values()), bins= 20, color = 'lightgrey')
    axes[1].set_xlabel("Clustering Coefficient $C$")
    #axes[1].set_ylabel("$P(k)$")
    axes[1].set_xlim([0,1])


def get_network_hubs(ntw):
    
    """
    input: NetworkX ntw
    output:Prints a list of global regulator name and eigenvector centrality score pairs
    """
    
    eigen_cen = nx.eigenvector_centrality(ntw)
    
    hubs = sorted(eigen_cen.items(), key = lambda cc:cc[1], reverse = True)[:10]
    
    return hubs


def get_network_clusters(network_lcc, n_clusters):
    
    """
    input = an empyty list
    
    output = a list with the netoworks clusters
    
    """
    cluster_list = []
    
    for i in range(n_clusters):

        cluster_lcc = [n for n in network_lcc.nodes()\
                       if network_lcc.node[n]['modularity'] == i]

        cluster_list.append(cluster_lcc)

    return cluster_list


