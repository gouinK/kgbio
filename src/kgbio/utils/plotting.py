import re
import os
import argparse
import matplotlib
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import rcParams
from matplotlib import font_manager
from matplotlib.legend import Legend


## TODO: make legend formatting better
def fix_plot(axs, plot_dict=None, fontdir=None):
    
    '''
    Convenience function to change matplotlib formatting.
    default_plot_dict = {
                            'weight': 'regular',
                            'family': 'Helvetica',
                            'fontsize': 8,
                            'pad': 2,
                            'grid': False,
                            'x_rotation': 0,
                            'y_rotation': 0,
                            'legend': True,
                            'fixlegend': False,
                            'snsfig': True,
                            'rm_legend_title': False,
                            'legend_loc': 'upper left',
                            'markerscale': 0.2,
                            'frame': True,
                            'subplot_behavior': True
                        }
    '''

    if fontdir is not None:
        try:
            font_files = font_manager.findSystemFonts(fontpaths=fontdir)

            for font_file in font_files:
                font_manager.fontManager.addfont(font_file)
        except:
            pass

    ## default settings
    default_plot_dict = {
                            'weight': 'regular',
                            'family': 'Helvetica',
                            'fontsize': 8,
                            'pad': 2,
                            'grid': False,
                            'x_rotation': 0,
                            'y_rotation': 0,
                            'legend': True,
                            'fixlegend': False,
                            'snsfig': True,
                            'rm_legend_title': False,
                            'legend_loc': 'upper left',
                            'markerscale': 0.2,
                            'frame': True,
                            'subplot_behavior': True
                        }

    ## Update settings based on input dict
    for key in plot_dict.keys():
        default_plot_dict[key] = plot_dict[key]

    # set font
    plt.rcParams['font.family'] = default_plot_dict['family']
    plt.rcParams['font.weight'] = default_plot_dict['weight']

    _= axs.tick_params(axis= 'both', labelsize= default_plot_dict['fontsize'], pad= default_plot_dict['pad'])
    _= axs.grid(visible= default_plot_dict['grid'])
    _= axs.set_frame_on(default_plot_dict['frame'])

    if 'xlim' in default_plot_dict.keys():
        _= axs.set_xlim(default_plot_dict['xlim'])
    if 'ylim' in default_plot_dict.keys():
        _= axs.set_ylim(default_plot_dict['ylim'])
    if 'xticks' in default_plot_dict.keys():
        _= axs.set_xticks(default_plot_dict['xticks'])
    if 'xticklabels' in default_plot_dict.keys():
        _= axs.set_xticklabels(default_plot_dict['xticklabels'], fontsize= default_plot_dict['fontsize'], rotation= default_plot_dict['x_rotation'], weight= default_plot_dict['weight'], fontfamily=default_plot_dict['family'])
    if 'yticks' in default_plot_dict.keys():
        _= axs.set_yticks(default_plot_dict['yticks'])
    if 'yticklabels' in default_plot_dict.keys():
        _= axs.set_yticklabels(default_plot_dict['yticklabels'], fontsize= default_plot_dict['fontsize'], rotation= default_plot_dict['y_rotation'], weight= default_plot_dict['weight'], fontfamily=default_plot_dict['family'])
    if 'xlabel' in default_plot_dict.keys():
        _= axs.set_xlabel(default_plot_dict['xlabel'], fontsize= default_plot_dict['fontsize'], labelpad= default_plot_dict['pad'], weight= default_plot_dict['weight'], fontfamily= default_plot_dict['family'])
    if 'ylabel' in default_plot_dict.keys():
        _= axs.set_ylabel(default_plot_dict['ylabel'], fontsize= default_plot_dict['fontsize'], labelpad= default_plot_dict['pad'], weight= default_plot_dict['weight'], fontfamily= default_plot_dict['family'])
    if 'title' in default_plot_dict.keys():
        _= axs.set_title(default_plot_dict['title'], fontsize= default_plot_dict['fontsize'], pad= default_plot_dict['pad']*4, weight= 'bold', fontfamily= default_plot_dict['family'])
    if 'logy' in default_plot_dict.keys():
        _= axs.set_yscale('log', base=10)
    if 'logx' in default_plot_dict.keys():
        _= axs.set_xscale('log', base=10)

    if default_plot_dict['subplot_behavior']:
        try:
            _= axs.label_outer()
        except Exception as e:
            pass

    if default_plot_dict['legend'] and default_plot_dict['fixlegend']:
            
        axs.legend(handletextpad= 0.5, loc= default_plot_dict['legend_loc'])
        
        if default_plot_dict['rm_legend_title']:
            axs.legend().set_title('')
            
        if default_plot_dict['snsfig']:
            plt.setp(axs.get_legend().get_texts(),fontsize= default_plot_dict['fontsize']/2)
            plt.setp(axs.get_legend().get_title(),fontsize= default_plot_dict['fontsize']/2)
            plt.setp(axs.get_legend().get_frame(),visible=False)
            [plt.setp(x, linewidth=1) for x in axs.get_legend().findobj() if isinstance(x,matplotlib.lines.Line2D)]
            [plt.setp(x, width=3, height=3) for x in axs.get_legend().findobj() if isinstance(x,matplotlib.patches.Rectangle)]
            [plt.setp(x, width=0.1, height=0.1) for x in axs.get_legend().findobj() if isinstance(x,matplotlib.offsetbox.DrawingArea)]
            [plt.setp(x, sizes=[1]) for x in axs.get_legend().findobj() if isinstance(x,matplotlib.collections.PathCollection)]
        else:
            axs.legend(fontsize= default_plot_dict['fontsize']/2, markerscale= default_plot_dict['markerscale'], frameon=False)
                
    elif not default_plot_dict['legend']:
        axs.get_legend().remove()

    return axs