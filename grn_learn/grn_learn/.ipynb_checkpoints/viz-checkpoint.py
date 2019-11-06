# +
import matplotlib.pyplot as plt 
import matplotlib as mpl
import seaborn as sns

def set_plotting_style():
      
    tw = 1.5
    rc = {'lines.linewidth': 2,
            'axes.labelsize': 24,
            'axes.titlesize': 26,
            'xtick.major' : 16,
            'ytick.major' : 16,
            'xtick.major.width': tw,
            'xtick.minor.width': tw,
            'ytick.major.width': tw,
            'ytick.minor.width': tw,
            'xtick.labelsize': 'large',
            'ytick.labelsize': 'large',
            'font.family': 'Helvetica',
            'weight':'bold',
            'grid.linestyle': ':',
            'grid.linewidth': 1.5,
            'grid.color': '#ffffff',
            'mathtext.fontset': 'stixsans',
            'mathtext.sf': 'fantasy',
            'legend.frameon': True,
            'legend.fontsize': 14, 
           "xtick.direction": "in","ytick.direction": "in"}
    
    #plt.rc('text.latex', preamble=r'\usepackage{sfmath}')
    #plt.rc('mathtext', fontset='stixsans', sf='sans')
    sns.set_style('ticks', rc=rc)
    sns.set_palette("colorblind", color_codes=True)
    sns.set_context('notebook', rc=rc)
    

def bokeh_style():

    '''
    Formats bokeh plotting enviroment.
    Based on the RPgroup PBoC style created by G. Chure and M. Razo.
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
                'text_font': 'Helvetica',
                'text_font_size': 18
            },
            'Title': {
                #'background_fill_color': '#FFEDC0',
                'text_font_style': 'normal',
                'align': 'center',
                'text_font': 'Helvetica',
                'offset': 2,
            }}}

    return theme_json
# -


