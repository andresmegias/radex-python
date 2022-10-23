#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RADEX Online Python Interface
-----------------------------
Version 1.4

Copyright (C) 2022  Andrés Megías Toledano

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.
"""

config_file = 'L1517B-CH3OH-online.yaml'

import os
import re
import io
import sys
import copy
import time
import pickle
import platform
import requests
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.optimize import minimize
from bs4 import BeautifulSoup
import richvalues as rv

def combine_rms(v):
    """
    Return the final rms of the sum of signals with the given rms noises.

    Parameters
    ----------
    v : list / array (float)
        Individual rms noises of each signal.

    Returns
    -------
    y : float
        Resulting rms noise.
    """
    y = np.sqrt(1 / np.sum(1/np.array(v)**2))
    return y

def remove_extra_spaces(input_text):
    """
    Remove extra spaces from a text string.

    Parameters
    ----------
    input_text : str
        Input text string.

    Returns
    -------
    text : str
        Resulting text.
    """
    text = input_text
    for i in range(15):
        if '  ' in text:
            text = text.replace('  ', ' ')
    if text.startswith(' '):
        text = text[1:]
    return text

def format_species_name(input_name, simplify_numbers=True):
    """
    Format text as a molecule name, with subscripts and upperscripts.

    Parameters
    ----------
    input_name : str
        Input text.
    simplify_numbers : bool, optional
        Remove the brackets between single numbers.

    Returns
    -------
    output_name : str
        Formatted molecule name.
    """
    original_name = copy.copy(input_name)
    # upperscripts
    possible_upperscript, in_upperscript = False, False
    output_name = ''
    upperscript = ''
    inds = []
    for i, char in enumerate(original_name):
        if (char.isupper() and not possible_upperscript
                and '-' in original_name[i:]):
            inds += [i]
            possible_upperscript = True
        elif char.isupper():
            inds += [i]
        if char == '-' and not in_upperscript:
            inds += [i]
            in_upperscript = True
        if not possible_upperscript:
            output_name += char
        if in_upperscript and i != inds[-1]:
            if char.isdigit():
                upperscript += char
            if char == '-' or i+1 == len(original_name):
                if len(inds) > 2:
                    output_name += original_name[inds[0]:inds[-2]]
                output_name += ('$^{' + upperscript + '}$'
                                + original_name[inds[-2]:inds[-1]])
                upperscript = ''
                in_upperscript, possible_upperscript = False, False
                inds = []
    output_name = output_name.replace('^$_', '$^').replace('$$', '')
    if output_name.endswith('+') or output_name.endswith('-'):
        symbol = output_name[-1]
        output_name = output_name.replace(symbol, '$^{'+symbol+'}$')
    original_name = copy.copy(output_name)
    # subscripts
    output_name, subscript, prev_char = '', '', ''
    in_bracket = False
    for i,char in enumerate(original_name):
        if char == '{':
            in_bracket = True
        elif char == '}':
            in_bracket = False
        if (char.isdigit() and not in_bracket
                and prev_char not in ['=', '-', '{', ',']):
            subscript += char
        else:
            if len(subscript) > 0:
                output_name += '$_{' + subscript + '}$'
                subscript = ''
            output_name += char
        if i+1 == len(original_name) and len(subscript) > 0:
            output_name += '$_{' + subscript + '}$'
        prev_char = char
    # vibrational numbers
    output_name = output_name.replace(',vt', ', vt')
    output_name = output_name.replace(', vt=', '$, v_t=$')
    output_name = output_name.replace(',v=', '$, v=$')
    for i,char in enumerate(output_name):
        if output_name[i:].startswith('$v_t=$'):
            output_name = output_name[:i+5] + \
                output_name[i+5:].replace('$_','').replace('_$','')
    # some formatting
    output_name = output_name.replace('$$', '').replace('__', '')
    # remove brackets from single numbers
    if simplify_numbers:
        single_numbers = re.findall('{(.?)}', output_name)
        for number in set(single_numbers):
            output_name = output_name.replace('{'+number+'}', number)
    return output_name

url = 'https://var.sron.nl/radex/radex.php'
session = requests.Session()
s = session.get(url)
soup = BeautifulSoup(s.content, 'html.parser')
options = soup.findChildren('select')
options = str(options).split('\n')
species_abreviations = {}
for line in options:
    if '<option ' in line:
        line = line.replace('selected=""', '')
        name = line.split('">')[1].replace(' ','')
        abreviation = line.split('=')[1].split('>')[0].replace('"','')
        species_abreviations[name] = abreviation

def radex_online(molecule, min_freq, max_freq, kin_temp, backgr_temp,
                 h2_num_dens, col_dens, line_width):
    """
    Perform the RADEX on-line calculation with the given parameters.

    Parameters
    ----------
    molecule : str
        RADEX name of the molecule.
    min_freq : TYPE
        Minimum frequency (GHz).
    max_freq : str
        Maximum frequency (GHz).
    kin_temp : str
        Kinetic temperature (K).
    backgr_temp : str
        Background temperature (K).
    h2_num_dens : str
        H2 number density (/cm3).
    col_dens : str
        Column density of the molecule (/cm2).
    line_width : str
        Line width of the lines of the molecule.

    Returns
    -------
    transitions_df : dataframe
        List of calculated transitions by RADEX on-line
    """
    
    def create_url(molecule, min_freq, max_freq, kin_temp, backgr_temp,
                   h2_num_dens, col_dens, line_width):
        """
        Create the corresponding URL for RADEX on-line.
        """
        molecule = species_abreviations[molecule]
        url = ('https://var.sron.nl/radex/radex.php?action=derive&' +
               'molfile={}&fmin={}&fmax={}&tbg={}&tkin={}&nh2={}&cold={}&dv={}'
               .format(molecule, min_freq, max_freq, backgr_temp, kin_temp,
                       h2_num_dens, col_dens, line_width))
        return url
    
    url = create_url(molecule, min_freq, max_freq, kin_temp, backgr_temp,
                     h2_num_dens, col_dens, line_width)
    session = requests.Session()
    s = session.get(url)
    soup = BeautifulSoup(s.content, 'html.parser')
    tables = soup.findChildren('table')
    table = tables[1]
    rows = table.findChildren(['tr'])
    rows = rows[2:-1]
    columns = ['transition', 'frequency (GHz)', 'temperature (K)', 'depth',
               'intensity (K)']
    transitions_df = pd.DataFrame(columns=columns)
    for i,row in enumerate(rows):
        row = row.findChildren('td')
        transition = remove_extra_spaces(row[0].get_text().replace('\xa0',''))
        transition = transition.replace(' ','')
        frequency = float(row[1].get_text())
        temperature = float(row[2].get_text())
        depth = float(row[3].get_text())
        intensity = float(row[4].get_text())
        transitions_df.loc[i] = [transition, frequency, temperature, depth,
                                 intensity]
    return transitions_df

def get_madcuba_transitions(madcuba_csv, radex_table, molecule,
                            lines_margin=0.0002):
    """
    Extract the information from the MADCUBA table for comparing it with RADEX.

    Parameters
    ----------
    madcuba_csv : dataframe
        MADCUBA table.
    radex_table : dataframe
        RADEX table with the desired transitions.
    molecule : str
        Molecule name in RADEX.
    lines_margin : float, optional
        Margin for identifying the lines in MADCUBA, in MHz. The default is 0.2.

    Returns
    -------
    madcuba_table : dataframe
        Desired dataframe, analogous to the given RADEX table.
    """
    madcuba_table = copy.copy(radex_table)
    madcuba_transitions = madcuba_csv[madcuba_csv['Formula']==molecule]
    for i,line in radex_table.iterrows():
        frequency = float(line['frequency (GHz)']) * 1e3
        difference = abs(madcuba_transitions['Frequency'] - frequency)
        cond = difference < lines_margin
        rows = madcuba_transitions[cond]
        intensity = 0
        if len(rows) == 0:
            madcuba_table.at[i,'intensity (K)'] = 0
            madcuba_table.at[i,'transition'] = np.nan
            madcuba_table.at[i,'frequency (GHz)'] = frequency
            madcuba_table.at[i,'temperature (K)'] = np.nan
            madcuba_table.at[i,'depth'] = np.nan
        else:
            for j,row in rows.iterrows():
                intensity += float(row['Intensity'])
            madcuba_table.at[i,'intensity (K)'] = intensity
            madcuba_table.at[i,'transition'] = row['Transition']
            madcuba_table.at[i,'frequency (GHz)'] = row['Frequency'] / 1e3
            madcuba_table.at[i,'temperature (K)'] = row['Tex/Te']
            madcuba_table.at[i,'depth'] = row['tau/taul']
    return madcuba_table

def radex_uncertainty(molecule, observed_transitions,
                      min_freq, max_freq, kin_temp, backgr_temp, h2_num_dens,
                      col_dens, line_width, lim, num_values=21, inds=None):
    """
    Calculate the uncertainty for a given column density with RADEX online.

    Parameters
    ----------
    molecule : str
        RADEX name of the molecule.
    min_freq : TYPE
        Minimum frequency (GHz).
    max_freq : str
        Maximum frequency (GHz).
    kin_temp : str
        Kinetic temperature (K).
    backgr_temp : str
        Background temperature (K).
    h2_num_dens : str
        H2 number density (/cm3).
    col_dens : str
        Column density of the molecule (/cm2).
    line_width : str
        Line width of the lines of the molecule.
    lim : float
        Reference loss value for calculating the uncertainty.
    num_values : int, optional
        Number of values of column density for evaluating the loss.
        The default is 21.
    inds : list (int), optional
        List of the indices of the desired transitions from RADEX output.
        The default is None (all transitions).

    Returns
    -------
    col_dens : float
        Final value of the column density.
    col_dens_uncs : list
        Uncertainties (lower and upper) for the columns density.
    (col_dens_values, losses) : tuple(array)
        Column density values and corresponding losses.
    """
    if num_values%2 == 0:
        print('The number of points (N) must be odd, so we add 1 to N.')
        num_values += 1
    observed_lines = observed_transitions['intensity (K)'].values
    observed_uncs = observed_transitions['intensity unc. (K)'].values
    frac = 1
    min_frac = 1e-5
    ind_min = 0
    ref_loss = 2*lim
    x = copy.copy(col_dens)
    while ref_loss > lim and frac > min_frac:
        if ind_min == 1:
            frac /= 2
            print('Reduced relative range to {}.'.format(frac))
        x1 = x - frac*x/2
        x2 = x + frac*x/2
        losses = []
        col_dens_values = [x1, x, x2]
        for col_dens_i in col_dens_values:
            radex_transitions = \
                radex_online(molecule, min_freq, max_freq, kin_temp,
                             backgr_temp, h2_num_dens, col_dens_i, line_width)
            radex_lines = radex_transitions.iloc[inds]['intensity (K)'].values
            radex_freqs = radex_transitions['frequency (GHz)'].values
            inds = np.argsort(radex_freqs)
            radex_lines = radex_lines[inds]
            losses += [loss_function(observed_lines, observed_uncs,
                                     radex_lines)]
        ref_loss = np.mean(losses) / 3
        ind_min = np.argmin(losses)
        x = col_dens_values[ind_min]
        if ind_min != 1:
            print('Updated minimum value to {:e}.'.format(x))
    col_dens_values = np.linspace(x1, x2, num_values)
    losses = []
    for i,col_dens_i in enumerate(col_dens_values):
        radex_transitions = \
            radex_online(molecule, min_freq, max_freq, kin_temp, backgr_temp,
                         h2_num_dens, col_dens_i, line_width)
        radex_lines = radex_transitions.iloc[inds]['intensity (K)'].values    
        loss = loss_function(observed_lines, observed_uncs,
                             radex_lines)
        losses += [loss]
        print('Iteration {}/{}.'.format(i+1, num_values).replace('e+', 'e'))
    losses = np.array(losses)
    ic = np.argmin(losses)
    if ic != (num_values-1) // 2:
        print('Updated minimum value to {:e}.'.format(x))
    x1_old = copy.copy(x1)
    x2_old = copy.copy(x2)
    ind_min = np.argmin(losses)
    x = col_dens_values[ind_min]
    col_dens = x
    x1 = x - frac*x/2
    x2 = x + frac*x/2
    X = np.linspace(x1, x2, int(10*num_values))
    Y = np.interp(X, col_dens_values, losses)
    cond1 = (X >= x1_old) & (X < x) & (Y < lim)
    cond2 = (X <= x2_old) & (X > x) & (Y < lim)
    if cond1.sum() > 0:
        unc_1 = x - min(X[cond1])
    else:
        unc_1 = col_dens_values[max(0, ic-1)]
    if cond2.sum() > 0:
        unc_2 = max(X[cond2]) - x
    else:
        unc_2 = col_dens_values[min(ic+1, i)]
    col_dens_uncs = (unc_1, unc_2)
    return col_dens, col_dens_uncs, (col_dens_values, losses)

def loss_function(observed_lines, observed_uncs, reference_lines):
    """
    Compute the difference between the observed lines and the reference ones.

    Parameters
    ----------
    observed_lines : array
        Observed intensities of the lines.
    observed_uncs : array
        Observed uncertainties for the line intensities.
    reference_lines : array
        Reference intensities of the lines.

    Returns
    -------
    difference : float
        Value of the difference between both input tables.

    """
    loss = 0
    for obs_intensity, obs_intensity_unc, ref_intensity in \
            zip(observed_lines, observed_uncs, reference_lines):
        obs_intensity = float(obs_intensity)
        obs_intensity_unc = float(obs_intensity_unc)
        ref_intensity = float(ref_intensity)
        loss += ((obs_intensity - ref_intensity) / obs_intensity_unc)**2
    return loss

def loss_distribution(observed_lines, observed_uncs, reference_lines,
                        num_variants=10000):
    """
    Compute the loss distribution between the observed and reference liens.

    Parameters
    ----------
    observed_lines : array
        Observed intensities of the lines.
    observed_lines_uncs : array
        Uncertainties of the intensities of the observed lines.
    reference_lines : array
        Reference intensities of the lines.
    num_variants : int
        Number of points to create the distribution of intensities from the
        uncertainties. The default is 10000.

    Returns
    -------
    losses : array (float)
        Values of the resulting losses between the observed and reference lines.

    """
    observed_lines = np.array(observed_lines)
    observed_uncs = np.array(observed_uncs)
    reference_lines = np.array(reference_lines)
    observed_lines_variants = \
        np.repeat(observed_lines.reshape(-1,1).transpose(),
                  num_variants, axis=0)
    for i, unc in enumerate(observed_uncs):
        observed_lines_variants[:,i] += \
            np.maximum(0, np.random.normal(0, unc, num_variants))
    losses = []
    for observed_lines_i in observed_lines_variants:
        losses += [loss_function(observed_lines_i, observed_uncs,
                                 reference_lines)]
    losses = np.array(losses)
    return losses

def ticks_format(value, index):
    """
    Format the input value.
    
    Francesco Montesano
    
    Get the value and returns the value formatted.
    To have all the number of the same size they are all returned as LaTeX
    strings
    """
    if value == 0:
        return '0'
    else:
        exp = np.floor(np.log10(value))
        base = value/10**exp
    if 0 <= exp <= 2:   
        return '${0:d}$'.format(int(value))
    elif exp == -1:
        return '${0:.1f}$'.format(value)
    elif exp == -2:
        return '${0:.2f}$'.format(value)
    else:
        return '${0:d}\\times10^{{{1:d}}}$'.format(int(base), int(exp))

def save_subplots(fig, size='auto', name='plot', form='pdf'):
    """
    Save each of the subplots of the input figure as a individual plots.

    Parameters
    ----------
    fig : matplotlib figure
        Figure object from Matplotlib.
    size : tuple (float), optional.
        Size of the individual figures for the subplots, in inches.
        If 'auto', it automatically calculate the appropiate size, although
        it will not work with subplots which have a colorbar.
    name : str, optional
        Main name of the files to be created. A number will be added to
        indicate the subplot of the original figure. The default is 'plot'.
    form : str, optional
        Format of the output files.

    Returns
    -------
    None.
    """
    c = 0
    suptitle = fig._suptitle.get_text()
    for i, oax in enumerate(fig.axes):
        if oax.get_label() == '<colorbar>':
            c += 1
            continue
        buf = io.BytesIO()
        pickle.dump(fig, buf)
        buf.seek(0)
        new_fig = pickle.load(buf)
        new_fig.suptitle('')
        new_ax = []
        colorbar = False
        for j, ax in enumerate(new_fig.axes):
            colorbar = ((j == i+1)
                        and (j < len(fig.axes)
                             and fig.axes[j].get_label() == '<colorbar>'))
            if j == i or colorbar:
                new_ax.extend([ax])
            else:
                new_fig.delaxes(ax)
        new_ax[0].change_geometry(1,1,1)
        title = new_ax[0].get_title()
        new_ax[0].set_title(suptitle + '\n' + title, fontweight='bold')
        new_size = copy.copy(size)
        if type(size) is str and size == 'auto':
            old_geometry = np.array(oax.get_geometry()[:2])[::-1]
            new_size = fig.get_size_inches() / old_geometry
        if len(new_ax) == 2:
            new_ax[0].change_geometry(1,2,1)
            new_ax[1].change_geometry(1,2,2)
            new_size[0] = new_size[0] * 1.5
        new_fig.set_size_inches(new_size, forward=True)
        new_fig.set_dpi(fig.dpi)
        plt.tight_layout()
        plt.savefig('{}-{}.{}'.format(name, i+1-c, form), bbox_inches='tight')
        plt.close()

#%%

t1 = time.time()

print()
print('RADEX Online Python Interpace')
print('-----------------------------')

# Default options.
default_config = {
    'input files': [],
    'output files': {'RADEX results': 'radex-output.yaml'},
    'lines margin (MHz)': 0.2,
    'RADEX parameters': [],
    'maximum iterations for optimization': 200,
    'grid points': 21,
    'show RADEX plots': True,
    'save RADEX plots': False,
    'save individual plots': False,
    'load previous RADEX results': False
    }

# Folder sepatator.
separator = '/'
operating_system = platform.system()
if operating_system == 'Windows':
    separator = '\\'

# Configuration file.
original_folder = os.getcwd() + separator
if len(sys.argv) != 1:
    config_file = sys.argv[1]
config_path = original_folder + config_file 
config_folder = separator.join(config_path.split(separator)[:-1]) + separator
os.chdir(config_folder)
if os.path.isfile(config_path):
    with open(config_path) as file:
        config = yaml.safe_load(file)
else:
    raise FileNotFoundError('Configuration file not found.')
config = {**default_config, **config} 

if 'input files' in config and 'MADCUBA table' in config['input files']:      
    config_input_madcuba = config['input files']['MADCUBA table']
if 'input files' in config and 'processing table' in config['input files']:      
    config_input_processing = config['input files']['processing table']
if not 'RADEX results' in config['output files']:
    config['output files']['RADEX results'] = \
        default_config['output files']['RADEX results']
output_file = config['output files']['RADEX results']

lines_margin = config['lines margin (MHz)'] / 1e3
radex_list = config['RADEX parameters']
show_plots = config['show RADEX plots']
save_plots = config['save RADEX plots']
save_ind_plots = config['save individual plots']
load_previous_results = config['load previous RADEX results']
max_iters = config['maximum iterations for optimization']
grid_points = config['grid points']

    
#%%    

all_results = []
results_file = config_file.replace('.yaml', '.pkl')
if load_previous_results:
    with open(results_file, 'rb') as file:
        previous_results = pickle.load(file)

radex_dict = {}
k = 0

for molecule in radex_list:
    
    if molecule.startswith('--'):
        continue
    
    lines_dict = {}
    
    h2_num_dens = radex_list[molecule]['hydrogen number density (/cm3)']
    if type(h2_num_dens) == list:
        num_sources = len(h2_num_dens)
    else:
        num_sources = 1
    
    for s in range(num_sources):
        
        k += 1
         
        if 'input files' in config and 'MADCUBA table' in config['input files']:   
            if len(config_input_madcuba) > 0: 
                input_madcuba = config_input_madcuba[s]
            else:
                input_madcuba = copy.copy(config_input_madcuba)
            table_madcuba = pd.read_csv(input_madcuba, sep=',')    
            source = input_madcuba.split('-')[0]
            print('\nSource {}: {}'.format(s+1, source))
        else:
            source = 'source ' + str(s+1)
            print('\nSource {}.'.format(s+1))
        
        if 'input files' in config and 'processing table' in config['input files']:
            if len(config_input_processing) > 0:
                input_processing = config_input_processing[s]
            else:
                input_processing = copy.copy(input_processing)
            table_proc = pd.read_csv(input_processing, sep=',')    
    
        radex_params = radex_list[molecule]
        if s == 0:
            radex_dict[molecule] = {}
        ref_lines = radex_params['observed lines']
        ref_frequencies = np.array(ref_lines['frequency (MHz)'])
        ref_intensities = np.array(ref_lines['intensity (K)'])
        observed_uncs = ref_lines['rms noise (mK)']
        kin_temp = radex_params['kinetic temperature (K)']
        backgr_temp = radex_params['background temperature (K)']
        h2_num_dens = radex_params['hydrogen number density (/cm3)']
        line_width = radex_params['line width (km/s)']
        col_dens = radex_params['column density (/cm2)']
        if len(ref_intensities) > 0 and hasattr(ref_intensities[0], '__iter__'):
            ref_intensities = ref_intensities[s]
        if type(observed_uncs) is not str and len(observed_uncs) > 0:
            observed_uncs = observed_uncs[s]
        if type(kin_temp) is not str and  hasattr(kin_temp, '__iter__'):
            kin_temp = kin_temp[s]
        if type(backgr_temp) is not str and hasattr(backgr_temp, '__iter__'):
            backgr_temp = backgr_temp[s]
        if type(h2_num_dens) is not str and hasattr(h2_num_dens, '__iter__'):
            h2_num_dens = h2_num_dens[s]
        if type(line_width) is not str and hasattr(line_width, '__iter__'):
            line_width = line_width[s]
        if type(col_dens) is not str and hasattr(col_dens, '__iter__'):
            col_dens = col_dens[s]
        if 'MADCUBA name' in radex_params:
            madcuba_name = radex_params['MADCUBA name']
            transitions = table_madcuba[table_madcuba['Formula']==madcuba_name]
            lines = ref_frequencies
            min_freq = str(min(lines) / 1e3 - lines_margin/3)
            max_freq = str(max(lines) / 1e3 + lines_margin/3)
            difference = abs(transitions['Frequency'] - min(lines)) / 1e3
            cond = difference < lines_margin
            transitions = transitions[cond]
            if kin_temp == 'auto':
                kin_temp = str(transitions['Tex/Te'].values[0])
            if line_width == 'auto':
                line_width = str(transitions['Width'].values[0])
            if col_dens == 'auto':
                col_dens = str(10**transitions['N/EM'].values[0])

        radex_transitions = \
            radex_online(molecule, min_freq, max_freq, kin_temp,
                         backgr_temp, h2_num_dens, col_dens,
                         line_width)

        observed_transitions = \
            pd.DataFrame({'frequency (GHz)': ref_frequencies/1e3,
                          'intensity (K)': ref_intensities})
        if observed_uncs == 'auto':
            observed_uncs = np.zeros(len(ref_frequencies))
            for frequency,intensity in zip(ref_frequencies,
                                           ref_intensities):
                cond = \
                    observed_transitions['frequency (GHz)'] == frequency/1e3
                i = observed_transitions[cond].index.values[0]
                cond = ((frequency > table_proc['min. frequency (MHz)'])
                        & (frequency < table_proc['max. frequency (MHz)']))
                spectra = table_proc[cond]
                rms = combine_rms(spectra['rms noise (mK)'])
                observed_uncs[i] = rms / 1e3
                observed_transitions.at[i,'intensity (K)'] = intensity
        observed_transitions['intensity unc. (K)'] = observed_uncs
        inds = np.argsort(ref_frequencies)
        observed_transitions['frequency (GHz)'] = \
            observed_transitions['frequency (GHz)'][inds].values
        observed_transitions['intensity (K)'] = \
            observed_transitions['intensity (K)'][inds].values
        observed_transitions['intensity unc. (K)'] = \
            observed_transitions['intensity unc. (K)'][inds].values
        observed_lines = observed_transitions['intensity (K)'].values
        
        inds = []
        for i, row_i in radex_transitions.iterrows():
            for j, row_j in observed_transitions.iterrows():
                freq_i = row_i['frequency (GHz)']
                freq_j = row_j['frequency (GHz)']
                if abs(freq_i - freq_j) < lines_margin:
                    inds += [i]
                    break
        
        if not load_previous_results:
            
            print('Minimizing column density for {} with RADEX.'
                  .format(molecule))
            def radex_online_function(params):
                log_x = params
                x = 10**log_x
                if hasattr(x, '__iter__'):
                    x = x[0]
                col_dens = '{:.6e}'.format(x).replace('e+', 'e')
                radex_transitions = \
                    radex_online(molecule, min_freq, max_freq, kin_temp,
                                 backgr_temp, h2_num_dens, col_dens, line_width)
                radex_lines = radex_transitions.iloc[inds]['intensity (K)'].values
                loss = loss_function(observed_lines, observed_uncs,
                                     radex_lines)
                print('col. dens (/cm2): {:.2e}, loss: {:.2f}'
                      .format(float(col_dens), loss))
                return loss
            guess = np.log10(float(col_dens))
            results = minimize(radex_online_function, guess, method='COBYLA',
                               options={'maxiter': max_iters,'disp': False})
            col_dens = float(10**results.x)
        
            radex_transitions = \
                radex_online(molecule, min_freq, max_freq, kin_temp,
                             backgr_temp, h2_num_dens, col_dens, line_width)
            radex_lines = radex_transitions.iloc[inds]['intensity (K)'].values    
            
            print('Calculating column density uncertainty for {} with RADEX.'
                  .format(molecule))
            losses_lim = loss_distribution(observed_lines, observed_uncs,
                                           radex_lines)
            loss_ref = loss_function(observed_lines, observed_uncs, radex_lines)
            idx = np.argmin(abs(np.sort(losses_lim) - loss_ref))              
            perc = max(68.27, 100*idx/len(losses_lim) + 68.27/2)
            perc = min(100, perc)
            lim = np.percentile(losses_lim, perc)
            col_dens, col_dens_uncs, (col_dens_values, losses) = \
                radex_uncertainty(molecule, observed_transitions,
                                  min_freq, max_freq, kin_temp,
                                  backgr_temp, h2_num_dens, col_dens,
                                  line_width, lim=lim, num_values=grid_points,
                                  inds=inds)
            
        else:
            
            col_dens, col_dens_uncs, col_dens_values, losses = \
                previous_results[k-1]
            print('Loaded previous results for {}.'.format(molecule)) 
        
        all_results += [[col_dens, col_dens_uncs, col_dens_values, losses]]  
            
        radex_transitions = \
            radex_online(molecule, min_freq, max_freq, kin_temp,
                         backgr_temp, h2_num_dens, col_dens, line_width)
        radex_lines = radex_transitions.iloc[inds]['intensity (K)'].values 

        losses_lim = loss_distribution(observed_lines, observed_uncs,
                                       radex_lines)
        loss_ref = loss_function(observed_lines, observed_uncs, radex_lines)
        idx = np.argmin(abs(np.sort(losses_lim) - loss_ref))           
        perc = max(68.27, 100*idx/len(losses_lim) + 68.27/2)
        perc = min(100, perc)
        lim = np.percentile(losses_lim, perc)
            
        positions = np.arange(len(observed_lines))
        labels = radex_transitions['frequency (GHz)'].values
        radex_freqs = observed_transitions['frequency (GHz)'].values
        radex_freqs = ['{:.6f}'.format(freq) for freq in radex_freqs]
            
        if show_plots or save_plots:
            
            plt.close(k)
            
            fig = plt.figure(k, figsize=(14, 4))
            plt.clf()
            
            ax = plt.subplot(1,3,1)
            plt.plot(col_dens_values, losses, '.', color='darkgreen',
                     label='{} - {}'.format(source, molecule))
            plt.axhline(y=lim, color='darkblue', linestyle='--')
            plt.axvline(x=col_dens, color='gray', linestyle='-')
            plt.axvline(col_dens - col_dens_uncs[0], color='gray',
                        linestyle='--')
            plt.axvline(col_dens + col_dens_uncs[1], color='gray',
                        linestyle='--')
            plt.xlabel('column density (cm$^{-2}$)')
            plt.ylabel('loss')
            plt.title('Loss values', fontweight='bold')
            plt.ylim(bottom=0)
            ax.xaxis.major.formatter._useMathText = True
            num_variants = int(1e5)
            loss_distr = \
                loss_distribution(observed_lines, observed_uncs, radex_lines,
                                  num_variants=num_variants)
            loss_ref = loss_function(observed_lines, observed_uncs,
                                     radex_lines)
    
            ax = plt.subplot(1,3,2)
            plt.hist(losses_lim, color='gray', histtype='stepfilled',
                     edgecolor='black', bins='auto', density=True)
            if loss_ref == 0:
                lw = 3
                alpha = 1
            else:
                lw = 1
                alpha = 0.6
            plt.axvline(loss_ref, color='tab:blue', linewidth=lw, alpha=alpha,
                        zorder=3)
            plt.plot([], [], color='tab:blue', linewidth=1, label='no sampling')
            plt.axvline(lim, color='darkblue', label='threshold quantile',
                        linestyle='--', linewidth=1)
            plt.xlabel('loss')
            plt.ylabel('frequency density')
            plt.yscale('log')
            if losses_lim.min() == 0:
                lim_value = lim / 10
            else:
                lim_value = losses_lim.min()
            _, bins = np.histogram(losses_lim)
            lim_value = max(bins[1], lim_value)
            linthresh = 10**(round(np.log10(lim_value)))
            plt.xscale('symlog', linthresh=linthresh)
            plt.xlim(left=0)
            ylim1, ylim2 = plt.ylim()
            plt.ylim(bottom=100*ylim1)
            plt.title('Loss distribution at optimized parameters',
                      fontweight='bold')
            plt.legend()
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(ticks_format))
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(ticks_format))
            
            plt.subplot(1,3,3)
            plt.bar(positions, observed_lines, color='gray', edgecolor='black',
                    width=0.5, label='observed', linewidth=2, zorder=2)
            plt.bar(positions, radex_lines, color='white', alpha=0.5,
                    edgecolor='tab:green', width=0.5, hatch='/',
                    label='modelled', linewidth=1, zorder=2)
            plt.errorbar(positions, observed_lines, observed_uncs, fmt='.',
                         color='black', capsize=2, capthick=1, zorder=3, ms=0.5)
            plt.xticks(positions, labels)
            plt.ylim(bottom=0)
            plt.xlabel('frequency (GHz)')
            plt.ylabel('intensity (K)')
            plt.title('Lines height')
            molecule_title = format_species_name(molecule.replace('-',' '))
            plt.suptitle('{} - {}'.format(source, molecule_title),
                         fontweight='bold')
            plt.legend()
            plt.tight_layout()

            if save_plots:
                file_name = '{}-{}.png'.format(source, molecule)
                plt.savefig(file_name, dpi=200)
                print('Saved plots in {}.'.format(file_name))
                if save_ind_plots:
                    size = fig.get_size_inches() / np.array([1, 3])
                    file_name = file_name.replace('.png', '')
                    save_subplots(fig, size, name=file_name)
                    print('Saved also individual plots.')
            if show_plots:
                if not any('SPYDER' in name for name in os.environ):
                    print('Click the close window button to continue.')
                plt.show()
                plt.pause(0.1)
        
        col_dens = rv.RichValue(col_dens, col_dens_uncs, num_sf=2)
        col_dens = str(col_dens)
        
        if num_sources == 1:
            radex_dict[molecule]['column density (/cm2)'] = col_dens
            radex_dict[molecule]['kinetic temperature (K)'] = kin_temp
        else:
            if s == 0:
                radex_dict[molecule]['column density (/cm2)'] = []
                radex_dict[molecule]['kinetic temperature (K)'] = []
            radex_dict[molecule]['column density (/cm2)'] += [col_dens]
            radex_dict[molecule]['kinetic temperature (K)'] += [kin_temp]
            
        radex_lines = [str(line) for line in radex_lines]
        line_width = '{:.3f}'.format(float(line_width))
        for frequency, intensity in zip(radex_freqs, radex_lines):
            if frequency not in lines_dict:
                lines_dict[frequency] = []
            lines_dict[frequency] += [{}]
            lines_dict[frequency][s]['intensity (K)'] = intensity
            lines_dict[frequency][s]['width (km/s)'] = line_width
        radex_dict[molecule]['lines (GHz)'] = lines_dict
    
#%%
            
nonlte_dict = {'non-LTE molecules': radex_dict}

with open(output_file, 'w') as file:
    yaml.dump(nonlte_dict, file, default_flow_style=False)
print('\nSaved RADEX output in {}.'.format(output_file))

with open(results_file, 'wb') as file:
    pickle.dump(all_results, file)
print('\nSaved results of the calculations in {}.\n'.format(results_file))

t2 = time.time()
t = t2 - t1
print('Elapsed time: {:.0f} min + {:.0f} s.'.format(t//60, t%60))

print()