import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np 
import pandas as pd 
import squarify
from matplotlib import rcParams


def bokeh_style():

    '''
    Formats bokeh plotting enviroment. Based on the RPgroup PBoC style.
    '''
    theme_json = {'attrs':{'Axis': {
            'axis_label_text_font': 'Helvetica',
            'axis_label_text_font_style': 'normal'
            },
            'Legend': {
                'border_line_width': 1.5,
                'background_fill_alpha': 0.5
            },
            'Text': {
                'text_font_style': 'normal',
               'text_font': 'Helvetica'
            },
            'Title': {
                #'background_fill_color': '#FFEDC0',
                'text_font_style': 'normal',
                'align': 'center',
                'text_font': 'Helvetica',
                'offset': 2,
            }}}

    return theme_json

# +
# -------   PLOTTING FUNCTIONS -------------------------


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

    rcParams['axes.titlepad'] = 20 


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


