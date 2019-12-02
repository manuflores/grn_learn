# -*- coding: utf-8 -*-
# + {}
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import matplotlib as mpl
import numba
import squarify
import numpy as np
from math import pi
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture as GMM
from umap import UMAP
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from datetime import date
from warnings import filterwarnings
import os
import community

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import regularizers
from keras.utils import np_utils
from keras.metrics import categorical_accuracy
from keras.layers import Dropout
import keras.backend as K

filterwarnings('ignore')


# +
def get_gene_data(data, gene_name_column, test_gene_list):
    
    """Extract data from specific genes given a larger dataframe.
    
    Inputs
    
    * data: large dataframe from where to filter
    * gene_name_column: column to filter from
    * test_gene_list : a list of genes you want to get
    
    Output
    * dataframe with the genes you want
    """
    
    gene_profiles = pd.DataFrame()

    for gene in data[gene_name_column].values:

        if gene in test_gene_list: 

            df_ = data[(data[gene_name_column] == gene)]

            gene_profiles = pd.concat([gene_profiles, df_])
    
    gene_profiles.drop_duplicates(inplace = True)
    
    return gene_profiles

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
    
    Wrapper from J. Bois' BeBi103 course. 
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

def download_and_preprocess_data(org, data_dir = None, variance_ratio = 0.8, 
                                output_path = '~/Downloads/'):
    
    """
    General function to download and preprocess dataset from Colombos. 
    Might have some issues for using with Windows. If you're using windows
    I recommend using the urllib for downloading the dataset. 
    
    Params
    -------
    
    
    data_path (str): path to directory + filename. If none it will download the data
                     from the internet. 
                     
    org (str) : Organism to work with. Available datasets are E. coli (ecoli), 
                B.subtilis (bsubt), P. aeruginosa (paeru), M. tb (mtube), etc. 
                Source: http://colombos.net/cws_data/compendium_data/
                
    variance (float): Fraction of the variance explained to make the PCA denoising. 
    
    Returns
    --------
    
    denoised (pd.DataFrame)
    
    """
    #Check if dataset is in directory
    if data_dir is None:
        
        download_cmd = 'wget http://colombos.net/cws_data/compendium_data/'\
                      + org + '_compendium_data.zip'
        
        unzip_cmd = 'unzip '+org +'_compendium_data.zip'
        
        os.system(download_cmd)
        os.system(unzip_cmd)
        
        df = pd.read_csv('colombos_'+ org + '_exprdata_20151029.txt',
                         sep = '\t', skiprows= np.arange(6))
        
        df.rename(columns = {'Gene name': 'gene name'}, inplace = True)
        
        df['gene name'] = df['gene name'].apply(lambda x: x.lower())
        
    else: 
        
        df = pd.read_csv(data_dir, sep = '\t', skiprows= np.arange(6))
        try : 
            df.rename(columns = {'Gene name': 'gene name'}, inplace = True)
        except:
            pass
    annot = df.iloc[:, :3]
    data = df.iloc[:, 3:]

    preprocess = make_pipeline(SimpleImputer( strategy = 'median'),
                               StandardScaler(), )

    scaled_data = preprocess.fit_transform(data)
    
    # Initialize PCA object
    pca = PCA(variance_ratio, random_state = 42).fit(scaled_data)
    
    # Project to PCA space
    projected = pca.fit_transform(scaled_data)
    
    # Reconstruct the dataset using 80% of the variance of the data 
    reconstructed = pca.inverse_transform(projected)

    # Save into a dataframe
    reconstructed_df = pd.DataFrame(reconstructed, columns = data.columns.to_list())

    # Concatenate with annotation data
    denoised_df = pd.concat([annot, reconstructed_df], axis = 1)
    
    denoised_df['gene name'] = denoised_df['gene name'].apply(lambda x: x.lower())

    # Export dataset 
    denoised_df.to_csv(output_path + 'denoised_' + org + '.csv', index = False)

def annot_data_trn(tf_tf_net_path = None,
                   trn_path = None,
                   denoised_data_path = None,
                   org = 'ecoli',
                   output_path = '~/Downloads/'):

    # Load TF-TF net and TRN
    
    if tf_tf_net_path is None: 
        os.system('wget http://regulondb.ccg.unam.mx/menu/download/datasets/files/network_tf_tf.txt')


        tf_trn = pd.read_csv('network_tf_tf.txt',
                         delimiter= '\t',
                         comment= '#', 
                         names = ['TF', 'TG', 'regType', 'ev', 'confidence', 'unnamed'], 
                         usecols = np.arange(5))

    else: 
        tf_trn = pd.read_csv(tf_tf_net_path,
                         delimiter= '\t',
                         comment= '#', 
                         names = ['TF', 'TG', 'regType', 'ev', 'confidence', 'unnamed'], 
                         usecols = np.arange(5))

    if trn_path is None: 
        # by default download the E. coli trn
        os.system('wget http://regulondb.ccg.unam.mx/menu/download/datasets/files/network_tf_gene.txt')

        trn = pd.read_csv('network_tf_gene.txt',
                         delimiter= '\t',
                         comment= '#', 
                         names = ['TF', 'TG', 'regType', 'ev', 'confidence', 'unnamed'], 
                         usecols = np.arange(5))

    else: 
        try:
            trn =  pd.read_csv(trn_path)
        except:
            trn = pd.read_csv(trn_path,
                             delimiter= '\t',
                             comment= '#', 
                             names = ['TF', 'TG', 'regType', 'ev', 'confidence', 'unnamed'], 
                             usecols = np.arange(5))
            
            #print('TRN probably has a different column annotation.')

    # Lowercase gene names for both datasets
    tf_trn.TF = tf_trn.TF.apply(lambda x: x.lower())
    tf_trn.TG = tf_trn.TG.apply(lambda x: x.lower())

    trn.TF = trn.TF.apply(lambda x: x.lower())
    trn.TG = trn.TG.apply(lambda x: x.lower())

    

    # Turn the TF TRN dataframe into a graph object
    net = nx.from_pandas_edgelist(df= tf_trn, source= 'TF', target='TG')

    # Compute the LCC
    net= max(nx.connected_component_subgraphs(net), key=len)

    #Cluster TF net 

    communities = community.best_partition(net)

    # Get number of clusters
    n_clusters_tf = max(communities.values())

    # Embed cluster annotation in net 
    nx.set_node_attributes(net, values= communities, name='modularity')

    # Get np.array of TF clusters
    cluster_list = np.array(get_network_clusters(net, n_clusters_tf))

    # Get cluster sizes 

    cluster_sizes = np.array([len(clus) for clus in cluster_list])

    # Select only the clusters with more than 5 TFs

    clus_list = cluster_list[cluster_sizes > 5]


    # Get a DataFrame of the TGs in each cluster

    tgs_ = pd.DataFrame()

    for ix, clus in enumerate(clus_list):
        
        clus_trn = get_gene_data(trn, 'TF', clus)
        clus_tgs = list(set(clus_trn['TG'].values))
        
        tgs_df = pd.DataFrame({'TGs': clus_tgs})
        
        tgs_df['cluster'] = ix + 1
        
        tgs_ = pd.concat([tgs_, tgs_df])


    # -----Start constructing the annotated dataset ------

    if denoised_data_path is None: 
        try:
            denoised = pd.read_csv('denoised_coli.csv')
        except: 
            import download_and_preprocess_data as d

            d.download_and_preprocess_data(org)

    else: 
        denoised = pd.read_csv(denoised_data_path)


    # Get nrows of denoised data
    nrows_data = denoised.shape[0]


    # Initialize one-hot-matrix

    one_hot_mat = np.zeros((nrows_data, n_clusters_tf))

    # Populate one-hot-matrix


    for ix, gene in enumerate(denoised['gene name']):
        
        gene_clus = tgs_[tgs_['TGs'] == gene]
        
        if gene_clus.shape[0] > 0:
            
            clusters = gene_clus.cluster.values
            clus_ix = [clus - 1 for clus in clusters]
            
            one_hot_mat[ix, clus_ix] = 1
            
        else: 
            pass

    # Make one-hot-matrix into a dataframe

    one_hot_df = pd.DataFrame(one_hot_mat, 
                    columns = ['cluster ' + str(i) for i in np.arange(1, n_clusters_tf + 1 )])


    # Get the n_samples of smallest cluster
    clus_samples = one_hot_mat.sum(axis = 0)

    min_clus_samples = min(clus_samples)

    # Separate denoised and annotated data 
    #annot = denoised.iloc[:, :3].values
    #denoised_data = denoised.iloc[:, 3:].values

    # Apply UMAP to denoised data 

    #denoised_reduced = umap.UMAP(n_components = int(min_clus_samples), 
    #                         random_state = seed).fit_transform(denoised_data)

    # Turn UMAP data into a dataframe
    #denoised_umap = pd.DataFrame(denoised_reduced,
    #    columns = ['UMAP ' + str(int(x)) for x in np.arange(1, min_clus_samples+ 1)]
    #)

    # Denoised UMAP data plus annotation and one hot matrix 
    denoised_hot = pd.concat([denoised, one_hot_df], axis = 1)

    # add a column corresponding to genes that are TGs 
    one_hot_sum = one_hot_mat.sum(axis = 1)# helper indicator array
    denoised_hot['TG'] = [1 if val > 0 else 0 for i, val in enumerate(one_hot_sum)]

    denoised_hot.to_csv('~/Downloads/denoised_umap_hot.csv', index = False)

def annot_data_trn(
    tf_tf_net_path=None,
    trn_path=None,
    denoised_data_path=None,
    org="ecoli",
    output_path= "~/Downloads/"):

    """
    Annotate the preprocessed dataset with network clusters as a one-hot-matrix. 
    Performs the operation on E. coli by default. 

    Params 
    -------


    Returns 
    --------

    """

    # Load TF-TF net and TRN

    if tf_tf_net_path is None and org is None:
        os.system(
            "wget http://regulondb.ccg.unam.mx/menu/download/datasets/files/network_tf_tf.txt"
        )

        tf_trn = pd.read_csv(
            "network_tf_tf.txt",
            delimiter="\t",
            comment="#",
            names=["TF", "TG", "regType", "ev", "confidence", "unnamed"],
            usecols=np.arange(5),
        )

    else:
        try: 
            tf_trn = pd.read_csv(tf_tf_net_path)
            
        except: 
            tf_trn = pd.read_csv(
            tf_tf_net_path,
            delimiter="\t",
            comment="#",
            names=["TF", "TG", "regType", "ev", "confidence", "unnamed"],
            usecols=np.arange(5),
        )

    if trn_path is None:
        os.system(
            "wget http://regulondb.ccg.unam.mx/menu/download/datasets/files/network_tf_gene.txt"
        )

        trn = pd.read_csv(
            "network_tf_gene.txt",
            delimiter="\t",
            comment="#",
            names=["TF", "TG", "regType", "ev", "confidence", "unnamed"],
            usecols=np.arange(5),
        )

    else:
        try: 
            trn = pd.read_csv(trn_path)
        except:
            trn = pd.read_csv(
                trn_path,
                delimiter="\t",
                comment="#",
                names=["TF", "TG", "regType", "ev", "confidence", "unnamed"],
                usecols=np.arange(5),
            )

    # Lowercase gene names for both datasets
    tf_trn.TF = tf_trn.TF.apply(lambda x: x.lower())
    tf_trn.TG = tf_trn.TG.apply(lambda x: x.lower())

    trn.TF = trn.TF.apply(lambda x: x.lower())
    trn.TG = trn.TG.apply(lambda x: x.lower())

    # Turn the TF TRN dataframe into a graph object
    net = nx.from_pandas_edgelist(
        df=tf_trn, source="TF", target="TG"
    )

    # Compute the LCC
    net = max(nx.connected_component_subgraphs(net), key=len)

    # Cluster TF net

    communities = community.best_partition(net)

    # Get number of clusters
    n_clusters_tf = max(communities.values())

    #  Embed cluster annotation in net
    nx.set_node_attributes(net, values=communities, name="modularity")

    # Get np.array of TF clusters
    cluster_list = np.array(get_network_clusters(net, n_clusters_tf))

    # Get cluster sizes

    cluster_sizes = np.array([len(clus) for clus in cluster_list])

    # Select only the clusters with more than 5 TFs

    clus_list = cluster_list[cluster_sizes > 5]

    # Get a DataFrame of the TGs in each cluster

    tgs_ = pd.DataFrame()

    for ix, clus in enumerate(clus_list):

        clus_trn = get_gene_data(trn, "TF", clus)
        clus_tgs = list(set(clus_trn["TG"].values))

        tgs_df = pd.DataFrame({"TGs": clus_tgs})

        tgs_df["cluster"] = ix + 1

        tgs_ = pd.concat([tgs_, tgs_df])

    # -----Start constructing the annotated dataset ------

    if denoised_data_path is None:
        try:
            denoised = pd.read_csv("denoised_coli.csv")
        except:
            download_and_preprocess_data(org)

    else:
        try:
            denoised = pd.read_csv(denoised_data_path + 'denoised_'+ org + '.csv') 
        except:
            'Could not load denoised dataset. Check file name input.'

    # Get nrows of denoised data
    nrows_data = denoised.shape[0]

    # Initialize one-hot-matrix
    one_hot_mat = np.zeros((nrows_data, n_clusters_tf))

    # Populate one-hot-matrix
    for ix, gene in enumerate(denoised["gene name"]):

        gene_clus = tgs_[tgs_["TGs"] == gene]

        if gene_clus.shape[0] > 0:

            clusters = gene_clus.cluster.values
            clus_ix = [clus - 1 for clus in clusters]

            one_hot_mat[ix, clus_ix] = 1

        else:
            pass

    # Make one-hot-matrix into a dataframe
    one_hot_df = pd.DataFrame(
        one_hot_mat,
        columns=["cluster " + str(i) for i in np.arange(1, n_clusters_tf + 1)],
    )

    # Get the n_samples of smallest cluster
    clus_samples = one_hot_mat.sum(axis=0)
    min_clus_samples = min(clus_samples)

    # Join the denoised dataset and one hot matrix
    denoised_hot = pd.concat([denoised, one_hot_df], axis=1)

    # add a column corresponding to genes that are TGs
    one_hot_sum = one_hot_mat.sum(axis=1)  # helper indicator array
    denoised_hot["TG"] = [1 if val > 0 else 0 for i, val in enumerate(one_hot_sum)]


    if output_path is "~/Downloads/":
        denoised_hot.to_csv("~/Downloads/denoised_hot_" + org + ".csv", index=False)

    else: 
        denoised_hot.to_csv( output_path + "denoised_hot_" + org + ".csv", index=False)
            

def train_keras_multilabel_nn(X_train,
                              y_train,
                              partial_x_train,
                              partial_y_train,
                              x_val= None, 
                              y_val= None,
                              n_units=64,
                              epochs=20,
                              n_deep_layers=1,
                              batch_size=128): 
    
    '''
    Trains a Keras model. 
    
    Assumes there are a X_train, y_train, x_val, y_val datasets.
    
    Params
    -------
    n_units: number of neurons per deep layer. 
    epochs:  number of epochs to train the net. 
    n_deep_layers: number of layers in the deep neural net. 
    batch_size : batch size to train the net with.
    
    Returns
    --------
    nn (Keras model): neural net model
    history (pd.DataFrame): history of the training procedure, includes 
                         training and validation accuracy and loss.
    
    '''
    nn = Sequential()
    
    #initial layer
    nn.add(Dense(n_units, activation='relu', input_shape=(X_train.shape[1],)))
    
    #extra deep layers
    for i in range(n_deep_layers):
        nn.add(Dense(n_units, activation='relu',
                     kernel_regularizer=regularizers.l2(0.001))
               )
        nn.add(Dropout(0.25))
        
    #add final output layer
    nn.add(Dense(y_train.shape[1], activation='softmax'))
    nn.compile(optimizer='rmsprop',
              loss='binary_crossentropy', 
              metrics=['accuracy'])
    
    #print neural net architecture
    nn.summary()
    
    #fit and load history
    if x_val and y_val == None:
        history = nn.fit(X_train, y_train, epochs=epochs,
                    batch_size= batch_size,
                    verbose = 0)
        
    else:
        history = nn.fit(partial_x_train, partial_y_train, epochs=epochs,
                    batch_size= batch_size, validation_data=(x_val, y_val),
                    verbose = 0)
    
    history_df = pd.DataFrame(history.history)
    
    return nn, history_df
# -





