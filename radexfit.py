#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RADEX Line Fitter
-----------------
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

radex_path = '/Users/andresmegias/Documents/RADEX/bin/radex'
config_file = 'examples/L1517B-HC3N.yaml'

import os
import re
import io
import sys
import copy
import time
import pickle
import platform
import subprocess
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib as mpl
from io import StringIO
from scipy.optimize import minimize
import richvalues as rv

if len(sys.argv) != 1:
    radex_path = 'radex'

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
    output_name = output_name.replace('^$_', '$^').replace('$$', '')
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

def radex(molecule, min_freq, max_freq, kin_temp, backgr_temp, h2_num_dens,
          col_dens, line_width):
    """
    Perform the RADEX calculation with the given parameters.

    Parameters
    ----------
    molecule : str
        RADEX name of the molecule.
    min_freq : TYPE
        Minimum frequency (GHz).
    max_freq : str
        Maximum frequency (GHz).
    kin_temp : str / array
        Kinetic temperature (K).
    backgr_temp : str
        Background temperature (K).
    h2_num_dens : str / array
        H2 number density (/cm3).
    col_dens : str / array
        Column density of the molecule (/cm2).
    line_width : str
        Line width of the lines of the molecule.

    Returns
    -------
    transitions_df : dataframe
        List of calculated transitions by RADEX on-line
    """
    radex_name = species_abreviations[molecule]
    def create_radex_input(molecule, min_freq, max_freq, kin_temp, backgr_temp,
                           h2_num_dens, col_dens, line_width):
        """
        Create the RADEX input file with the given parameters.
        """
        text = []
        radex_name = species_abreviations[molecule]
        if type(col_dens) is str or not hasattr(col_dens, '__iter__'):
            col_dens = [col_dens]
        if type(h2_num_dens) is str or not hasattr(h2_num_dens, '__iter__'):
            h2_num_dens = [h2_num_dens]
        if type(kin_temp) is str or not hasattr(kin_temp, '__iter__'):
            kin_temp = [kin_temp]
        num_calcs = 0
        for col_dens_i in col_dens:
            for h2_num_dens_i in h2_num_dens:
                for kin_temp_i in kin_temp:
                    text += ['{}.dat'.format(radex_name)]
                    text += ['{}.out'.format(radex_name)]
                    text += ['{} {}'.format(min_freq, max_freq)]
                    text += ['{}'.format(kin_temp_i)]
                    text += ['1', 'H2', '{:e}'.format(float(h2_num_dens_i))]
                    text += ['{}'.format(backgr_temp)]
                    text += ['{:e}'.format(float(col_dens_i))]
                    text += ['{}'.format(line_width)]
                    text += ['1']
                    num_calcs += 1
        text[-1] = text[-1].replace('1', '0')
        for i in range(len(text) - 1):
            text[i] += '\n'
        with open('{}.inp'.format(radex_name), 'w') as file:
            file.writelines(text)
        return num_calcs
    num_calcs = create_radex_input(molecule, min_freq, max_freq, kin_temp,
                                   backgr_temp, h2_num_dens, col_dens,
                                   line_width)
    subprocess.run('{} < {}.inp > radex_log.txt'
                   .format(radex_path, radex_name), shell=True)
    with open('{}.out'.format(radex_name), 'r') as file:
        output_text = file.readlines()
    i = -1
    write_line = False
    tables = [[] for j in range(num_calcs)]
    for line in output_text:
        if line.startswith('* Radex version'):
            write_line = False
            if i >= 0:
                del tables[i][1]
            i += 1
        if write_line:
            tables[i] += [line]
        if line.startswith('Calculation finished'):
            write_line = True
    del tables[num_calcs-1][1]
    transitions_df = [[] for j in range(num_calcs)]
    for i in range(num_calcs):
        tables[i] = ''.join(tables[i])
        result_df = pd.read_csv(StringIO(tables[i]), delimiter='\s+')
        transitions_dict = {'transition': result_df['LINE'].values,
                            'frequency (GHz)': result_df['FREQ'].values,
                            'temperature (K)': result_df['T_EX'].values,
                            'depth': result_df['TAU'].values,
                            'intensity (K)': result_df['T_R'].values}
        transitions_df[i] = pd.DataFrame(transitions_dict)
    return transitions_df

def get_madcuba_transitions(madcuba_csv, radex_table, molecule,
                            lines_margin=0.2):
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
                      len_sample=int(1e5)):
    """
    Compute the loss distribution between the observed and reference lines.

    Parameters
    ----------
    observed_lines : list (float)
        Intensities of the observed transitions.
    observed_uncs : list (float)
        Intensity uncertainties of the observed transitions.
    reference_lines : list (float)
        Intensities of the theoretical transitions.
    len_sample : int
        Number of points to create the distribution of intensities from the
        uncertainties. The default is int(1e5).

    Returns
    -------
    losses : array (float)
        Values of the resulting losses between the observed and reference lines.

    """
    observed_lines = np.array(observed_lines)
    observed_uncs = np.array(observed_uncs)
    reference_lines = np.array(reference_lines)
    observed_lines_sample = np.repeat(observed_lines.reshape(-1,1).transpose(),
                                      len_sample, axis=0)
    for i, unc in enumerate(observed_uncs):
        observed_lines_sample[:,i] += np.random.normal(0., unc, len_sample)
        # observed_lines_sample = np.maximum(0., observed_lines_sample)
    losses = [loss_function(observed_lines_i, observed_uncs, reference_lines)
              for observed_lines_i in observed_lines_sample]
    losses = np.array(losses)
    return losses

def get_loss_from_radex(name, min_freq, max_freq, kin_temp, backgr_temp,
                        h2_num_dens, col_dens, line_width,
                        observed_lines, observed_uncs, inds):
    """
    Compute the loss from a RADEX calculation with the given parameters.

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
    observed_lines : list (float)
        Intensities of the observed transitions.
    observed_uncs : list (float)
        Intensity uncertainties of the observed transitions.
    inds : list (int)
        List of the indices of the desired transitions from RADEX output.

    Returns
    -------
    losses : array
        Array of the losses of the input parameters.
    """
    radex_results = radex(name, min_freq, max_freq, kin_temp, backgr_temp,
                          h2_num_dens, col_dens, line_width)
    losses = []
    for radex_transitions in radex_results:
        radex_lines = radex_transitions.iloc[inds]['intensity (K)'].values
        loss = loss_function(observed_lines, observed_uncs, radex_lines)
        losses += [loss]
    if len(losses) == 1:
        losses = loss
    return losses

def radex_uncertainty_grid(observed_transitions, molecule, min_freq, max_freq,
                           kin_temp, backgr_temp, h2_num_dens, col_dens,
                           line_width, lim, fit_params, grid_points=15,
                           max_enlargings=10, inds=None):
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
    fit_params: str
        Parameters whose uncertainties will be calculated. Possible values are:
        - column density
        - column density, H2 number density
        - column density, temperature
    grid_points : int, optional
        Number of values of column density for evaluating the loss.
        The default is 15.
    max_enlargings : int, optional
        Number of times that the possible uncertainty range will be enlarged
        from the initial ranges obtained fixing the other variable.
        The default is 10.
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
    params = (molecule, min_freq, max_freq, kin_temp, backgr_temp, h2_num_dens,
              col_dens, line_width)
    observed_lines = observed_transitions['intensity (K)'].values
    observed_uncs = observed_transitions['intensity unc. (K)'].values
    if type(inds) == type(None):
        inds = np.arange(len(observed_lines))
                          
    def calculate_range(params, param_name):
        """
        Obtain the right range to calculate the losses for the uncertainties.

        Parameters
        ----------
        params : float
            Values of the RADEX parameters.
        param_name : str
            Name of the variable. Possibles values are: 'column density',
            'H2 number density', and 'temperature'.

        Returns
        -------
        x1 : float
            Inferior limit.
        x2 : float
            Superior limit.
        """
        (name, min_freq, max_freq, kin_temp, backgr_temp, h2_num_dens,
         col_dens, line_width) = params
        names = np.array(['column density', 'H2 number density', 'temperature'])
        idx = np.array([6, 5, 3])[names == param_name][0]
        x = params[idx]
        min_x = 0.001*x
        frac = 1
        min_frac = 1e-5
        good_range = False
        all_fracs = []
        while not good_range and frac > min_frac:
            x1 = x - frac*x/2
            x2 = x + frac*x/2
            x1 = max(min_x, x1)
            losses = []
            x_values = [x1, x, x2]
            for x_i in x_values:
                if param_name == 'column density':
                    col_dens = x_i
                elif param_name == 'H2 number density':
                    h2_num_dens = x_i
                elif param_name == 'temperature':
                    kin_temp = x_i
                loss = get_loss_from_radex(name, min_freq, max_freq, kin_temp,
                                           backgr_temp, h2_num_dens, col_dens,
                                           line_width, observed_lines,
                                           observed_uncs, inds)
                losses += [loss]
            ind_min = np.argmin(losses)
            x = x_values[ind_min]
            if ind_min != 1:
                raise Exception('Variable was not really minimized.')
            small_range, large_range = False, False
            if min(losses[0], losses[-1]) < lim:
                small_range = True
            threshold = losses[1] + 2 * (lim - losses[1])
            if ((np.mean([losses[0], losses[1]]) > threshold)
                and (np.mean([losses[1], losses[-1]]) > threshold)):
                large_range = True
            if large_range == small_range or x1 == min_x:
                good_range = True
            if not good_range:
                if large_range and frac/2 not in all_fracs:
                    frac /= 2
                    print('Reduced relative range to {}.'.format(frac))
                elif small_range and frac*2 not in all_fracs:
                    frac *= 2
                    print('Increased relative range to {}.'.format(frac))
                else:
                    good_range = True
            all_fracs += [frac]
                
        return x1, x2
    
    def get_uncertainty(x_values, params, param_name):
        """
        Obtain the uncertainties given an input array of values of a variable.

        Parameters
        ----------
        x_values : array (float)
            Input values in which range the uncertainties will be calculated.
        params : tuple
            Values of the RADEX parameters.
        param_name : str
            Name of the variable. Possibles values are: 'column density',
            'H2 number density', and 'temperature'.

        Returns
        -------
        x : float
            Central value.
        unc1 : float
            Inferior uncertainty.
        unc2 : float
            Superior uncertainty.
        losses : array (float)
            Array of the loss values for the input values.
        """
        (name, min_freq, max_freq, kin_temp, backgr_temp, h2_num_dens,
         col_dens, line_width) = params
        num_points = len(x_values)
        names = np.array(['column density', 'H2 number density', 'temperature'])
        idx = np.array([6, 5, 3])[names == param_name][0]
        x = params[idx]
        if param_name == 'column density':
            col_dens = x_values
        elif param_name == 'H2 number density':
            h2_num_dens = x_values
        elif param_name == 'temperature':
            kin_temp = x_values
        print('Calculating...')
        losses = get_loss_from_radex(molecule, min_freq, max_freq, kin_temp,
                                     backgr_temp, h2_num_dens, col_dens,
                                     line_width, observed_lines,
                                     observed_uncs, inds)
        ic = np.argmin(losses)
        if ic != (num_points-1) // 2:
            x = x_values[ic]
            print('Updated minimum value to {:4e}.'.format(x))
        x1_old, x2_old = x_values[0], x_values[-1]
        frac = 2 * (x2_old - x1_old) / x
        x1 = x - frac*x/2
        x2 = x + frac*x/2
        X = np.linspace(x1, x2, int(10*num_points))
        Y = np.interp(X, x_values, losses)
        cond1 = (X >= x1_old) & (X < x) & (Y < lim)
        cond2 = (X <= x2_old) & (X > x) & (Y < lim)
        if cond1.sum() > 0:
            unc_1 = x - min(X[cond1])
        else:
            unc_1 = x_values[max(0, ic-1)]
        if cond2.sum() > 0:
            unc_2 = max(X[cond2]) - x
        else:
            unc_2 = x_values[min(ic+1, len(x_values)-1)]
        return x, unc_1, unc_2, losses
    
    def enlarge_uncs_2d(xa0, xb0, xa_values, xb_values, xa_losses, xb_losses,
                        name_a, name_b, params, lim=lim, grid_points=grid_points,
                        max_enlargings=max_enlargings):
        """
        Explore the bidimensional space parameter to calculate uncertainties.

        Parameters
        ----------
        xa0 : float
            Optimized value for the first variable.
        xb0 : float
            Optimized value for the second variable.
        xa_values : array (float)
            Input values for the first variable.
        xb_values : array (float)
            Input values for the second variable.
        xa_losses : array (float)
            Losses for the values of the first variable when the second one
            has its optimized value.
        xb_losses : array (float)
            Losses for the values of the second variable when the first one
            has its optimzed value.
        name_a : str
            Name of the first variable. Possibles values are: 'column density',
            'H2 number density', and 'temperature'.
        name_b : str
            Name of the second variable. Possibles values are: 'column density',
            'H2 number density', and 'temperature'.
        params : tuple
            Values of the RADEX parameters.

        Returns
        -------
        xa_uncs : list (float)
            New uncertainties for the first variable.
        xb_uncs : list (float)
            New uncertainties for the second variabl.
        xa_new : array (float)
            All values of the first variable where the loss is calculated.
        xb_new : array (float)
            All values of the second variables where the loss is calculated.
        losses : array (float)
            Loss value for each of the values of the input variables.
        """
        print('Exploring the parameter space to improve the values of the '
              + 'uncertainties.')
        (molecule, min_freq, max_freq, kin_temp, backgr_temp,
         h2_num_dens, col_dens, line_width) = params
        num_points_a = copy.copy(grid_points)
        num_points_b = copy.copy(grid_points)
        xa_new = copy.copy(xa_values)
        xb_new = xb0 * np.ones(len(xa_values))
        losses = copy.copy(xa_losses)
        xa_new = np.append(xa_new, xa0 * np.ones(len(xb_values)))
        xb_new = np.append(xb_new, xb_values)
        losses = np.append(losses, xb_losses)
        xa1, xa2 = min(xa_values), max(xa_values)
        xb1, xb2 = min(xb_values), max(xb_values)
        xa_range = xa2 - xa1
        xb_range = xb2 - xb1
        xa_m = (xa1 + xa2) / 2
        xb_m = (xb1 + xb2) / 2
        factor = 2
        xa1n, xa2n = xa_m - factor*xa_range/2, xa_m + factor*xa_range/2
        xb1n, xb2n = xb_m - factor*xb_range/2, xb_m + factor*xb_range/2
        xa1n = max(0.01*xa_m, xa1n)
        xb1n = max(0.01*xb_m, xb1n)
        xa_range = xa2n - xa1n
        xb_range = xb2n - xb1n
        resolution_a = xa_range / num_points_a
        resolution_b = xb_range / num_points_b
        xa_arr = np.linspace(xa1n, xa2n, grid_points)
        xb_arrs = [np.linspace(xb1n, xb2n, grid_points) for xai in xa_arr]
        explore = True
        num_enlargings = 0
        while explore and num_enlargings <= max_enlargings:
            xa1o, xa2o = copy.copy(xa1n), copy.copy(xa2n)
            xb1o, xb2o = copy.copy(xb1n), copy.copy(xb2n)
            for i, xai in enumerate(xa_arr):
                print('Iteration {}/{}...'.format(i+1, len(xa_arr)))
                xb_arr = xb_arrs[i]
                if name_a == 'column density':
                    col_dens = xai
                elif name_a == 'H2 number density':
                    h2_num_dens = xai
                elif name_a == 'temperature':
                    kin_temp = xai
                if name_b == 'column density':
                    col_dens = xb_arr
                elif name_b == 'H2 number density':
                    h2_num_dens = xb_arr
                elif name_b == 'temperature':
                    kin_temp = xb_arr
                losses_i = \
                    get_loss_from_radex(molecule, min_freq, max_freq,
                                        kin_temp, backgr_temp, h2_num_dens,
                                        col_dens, line_width,
                                        observed_lines, observed_uncs, inds)
                xa_new = np.append(xa_new, xai*np.ones(len(xb_arr)))
                xb_new = np.append(xb_new, xb_arr)
                losses = np.append(losses, losses_i)
            cond = losses < lim
            xa_cond = xa_new[cond]
            xb_cond = xb_new[cond]
            xa_interv = [xa_cond.min(), xa_cond.max()]
            xb_interv = [xb_cond.min(), xb_cond.max()]
            xa_range = xa2o - xa1o
            xb_range = xb2o - xb1o
            margin_a = 6 * resolution_a
            margin_b = 6 * resolution_b
            enlarge_xa_p = ((xa2n - xa_interv[1]) < margin_a)
            enlarge_xa_m = ((xa_interv[0] - xa1n) < margin_a)
            enlarge_xb_p = ((xb2n - xb_interv[1]) < margin_b)
            enlarge_xb_m = ((xb_interv[0] - xb1n) < margin_b)
            if enlarge_xa_m and (xa1n - (xa2 - xa1)*factor/2) < 0.1*xa_m:
                enlarge_xa_m = False
            if enlarge_xb_m and (xb1n - (xb2 - xb1)*factor/2) < 0.1*xb_m:
                enlarge_xb_m = False
            if enlarge_xa_p or enlarge_xa_m or enlarge_xb_p or enlarge_xb_m:
                print('Enlarging region.')
                num_enlargings += 1
                explore = True
                if enlarge_xa_p:
                    xa2n += (xa2 - xa1) * factor/2
                if enlarge_xa_m:
                    xa1n -= (xa2 - xa1) * factor/2
                if enlarge_xb_p:
                    xb2n += (xb2 - xb1) * factor/2
                if enlarge_xb_m:
                    xb1n -= (xb2 - xb1) * factor/2
                num_points_a = int(round((xa2n - xa1n) / resolution_a))
                num_points_b = int(round((xb2n - xb1n) / resolution_b))
                xa_arr, xb_arrs = np.array([]), []
                xa_all = np.linspace(xa1n, xa2n, num_points_a)
                xb_all = np.linspace(xb1n, xb2n, num_points_b)
                for i, xai in enumerate(xa_all):
                    xb_arr = np.array([])
                    for xbi in xb_all:
                        inside = ((xa1o <= xai <= xa2o) & (xb1o <= xbi <= xb2o))
                        if not inside:
                            xa_arr = np.append(xa_arr, xai)
                            xb_arr = np.append(xb_arr, xbi)
                    if len(xb_arr) > 0:
                        xb_arrs += [xb_arr]
                xa_arr = np.unique(xa_arr)
            else:
                explore = False
        return xa_interv, xb_interv, xa_new, xb_new, losses
    
    if grid_points%2 == 0:
        print('The number of points (N) must be odd, so we add 1 to N.')
        grid_points += 1
    if fit_params == 'column density':
        print('Variable: column density')
        x1, x2 = calculate_range(params, 'column density')
        x_values = np.linspace(x1, x2, grid_points)
        x, unc1, unc2, losses = get_uncertainty(x_values, params,
                                                'column density')
        results = (rv.RichValue(x, [unc1, unc2]), (x_values, losses))
    elif fit_params == 'column density, H2 number density':
        num_points_ind = max(15, grid_points//3)
        if num_points_ind % 2 == 0:
            num_points_ind += 1
        print('Variable: column density')
        x1, x2 = calculate_range(params, 'column density')
        x_values = np.linspace(x1, x2, num_points_ind)
        x, unc1, unc2, losses = get_uncertainty(x_values, params,
                                                'column density')
        xa0 = copy.copy(x)
        xa_values = copy.copy(x_values)
        xa_losses = copy.copy(losses)
        print('Variable: H2 number density')
        x1, x2 = calculate_range(params, 'H2 number density')
        x_values = np.linspace(x1, x2, num_points_ind)
        x, unc1, unc2, losses = get_uncertainty(x_values, params,
                                                'H2 number density')
        xb0 = copy.copy(x)
        xb_values = copy.copy(x_values)
        xb_losses = copy.copy(losses)
        xa_interv, xb_interv, xa_new, xb_new, losses = \
            enlarge_uncs_2d(xa0, xb0, xa_values, xb_values, xa_losses,
                            xb_losses, 'column density', 'H2 number density',
                            params, lim, grid_points=grid_points)
        xa_uncs = [xa0 - xa_interv[0], xa_interv[1] - xa0]
        xb_uncs = [xb0 - xb_interv[0], xb_interv[1] - xb0]
        results = (rv.RichValue(xa0, xa_uncs), rv.RichValue(xb0, xb_uncs),
                   (xa_new, xb_new, losses))
    elif fit_params == 'column density, temperature':
        num_points_ind = max(15, grid_points//3)
        if num_points_ind % 2 == 0:
            num_points_ind += 1
        print('Variable: column density (/cm2)')
        x1, x2 = calculate_range(params, 'column density')
        x_values = np.linspace(x1, x2, num_points_ind)
        x, unc1, unc2, losses = get_uncertainty(x_values, params,
                                                'column density')
        xa0 = copy.copy(x)
        xa_values = copy.copy(x_values)
        xa_losses = copy.copy(losses)
        print('Variable: temperature (K)')
        x1, x2 = calculate_range(params, 'temperature')
        x_values = np.linspace(x1, x2, num_points_ind)
        x, unc1, unc2, losses = get_uncertainty(x_values, params,
                                                'temperature')
        xb0 = copy.copy(x)
        xb_values = copy.copy(x_values)
        xb_losses = copy.copy(losses)
        xa_interv, xb_interv, xa_new, xb_new, losses = \
            enlarge_uncs_2d(xa0, xb0, xa_values, xb_values, xa_losses,
                            xb_losses, 'column density', 'temperature', params,
                            lim, grid_points=grid_points)
        xa_uncs = [xa0 - xa_interv[0], xa_interv[1] - xa0]
        xb_uncs = [xb0 - xb_interv[0], xb_interv[1] - xb0]
        results = (rv.RichValue(xa0, xa_uncs), rv.RichValue(xb0, xb_uncs),
                   (xa_new, xb_new, losses))
    else:
        raise Exception('Error: incorrect fit parameters.')
        
    return results

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
        new_ax[0].set_subplotspec(plt.GridSpec(1,1)[0])
        title = new_ax[0].get_title()
        new_ax[0].set_title(suptitle + '\n' + title, fontweight='bold')
        new_size = size
        if type(size) is str and size == 'auto':
            old_geometry = np.array(oax.get_subplotspec().get_geometry()[:2])[::-1]
            new_size = fig.get_size_inches() / old_geometry
        if len(new_ax) == 2:
            new_ax[0].set_subplotspec(plt.GridSpec(1,2)[0])
            new_ax[1].set_subplotspec(plt.GridSpec(1,2)[0])
            new_size[0] = new_size[0] * 1.5
        new_fig.set_size_inches(new_size, forward=True)
        new_fig.set_dpi(fig.dpi)
        plt.tight_layout()
        plt.savefig('{}-{}.{}'.format(name, i+1-c, form), bbox_inches='tight')
        plt.close()

species_abreviations = {
    'CO': 'co',
    '13CO': '13co',
    'C18O': 'c18o',
    'C17O': 'c17o',
    'CS': 'cs',
    'p-H2S': 'ph2s',
    'o-H2S': 'oh2s',
    'HCO+': 'hco+',
    'DCO+': 'dco+',
    'H13CO+': 'h13co+',
    'HC18O+': 'hc18o+',
    'HC17O+': 'hc17o+',
    'Oatom': 'oatom',
    'Catom': 'catom',
    'C+ion': 'c+',
    'N2H+': 'n2h+',
    'HCN': 'hcn@hfs',
    'H13CN': 'h13cn',
    'HC15N': 'hc15n',
    'HC3N': 'hc3n',
    'HNC': 'hnc',
    'SiO': 'sio',
    '29SiO': '29sio',
    'SiS': 'sis',
    'O2': 'o2',
    'CN': 'cn',
    'SO': 'so',
    'SO2': 'so2',
    'o-SiC2': 'o-sic2',
    'OCS': 'ocs',
    'HCS+': 'hcs+',
    'o-H2CO': 'o-h2co',
    'p-H2CO': 'p-h2co',
    'o-H2CS': 'oh2cs',
    'p-H2CS': 'ph2cs',
    'CH3OH-E': 'e-ch3oh',
    'CH3OH-A': 'a-ch3oh',
    'CH3CN': 'ch3cn',
    'o-C3H2': 'o-c3h2',
    'p-C3H2': 'p-c3h2',
    'OH': 'oh',
    'o-H2O': 'o-h2o',
    'p-H2O': 'p-h2o',
    'HDO': 'hdo',
    'HCl': 'hcl@hfs',
    'o-NH3': 'o-nh3',
    'p-NH3': 'p-nh3',
    'o-H3O+': 'o-h3o+',
    'p-H3O+': 'p-h3o+',
    'HNCO': 'hnco',
    'NO': 'no',
    'HF': 'hf'
     }

#%%

plt.close('all')
t1 = time.time()

print()
print('RADEX Line Fitter')
print('-----------------')

# Default options.
default_config = {
    'input files': [],
    'output files': {'RADEX results': 'radex-output.yaml'},
    'RADEX path': '',
    'lines margin (MHz)': 0.2,
    'RADEX parameters': [],
    'RADEX fit parameters': 'column density',
    'load previous RADEX results': False,
    'maximum iterations for optimization': 1000,
    'grid points': 15,
    'show RADEX plots': True,
    'save RADEX plots': False,
    'save individual plots': False
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
if 'input files' in config and 'spectra table' in config['input files']:      
    config_input_spectra = config['input files']['spectra table']
if not 'RADEX results' in config['output files']:
    config['output files']['RADEX results'] = \
        default_config['output files']['RADEX results']
output_file = config['output files']['RADEX results']
if config['RADEX path'] != '':
    radex_path = config['RADEX path']
lines_margin = config['lines margin (MHz)'] / 1e3
radex_list = config['RADEX parameters']
load_previous_results = config['load previous RADEX results']
max_iters = config['maximum iterations for optimization']
grid_points = config['grid points']
show_plots = config['show RADEX plots']
save_plots = config['save RADEX plots']
save_ind_plots = config['save individual plots']
if ('RADEX fit parameters' in config and config['RADEX fit parameters']
    not in ['column density', 'column density, H2 number density',
            'column density, temperature']):
    raise Exception('RADEX fit parameters not properly written.')
    
#%%    


all_results = []
results_file = config_file.replace('.yaml', '.pkl').split('/')[-1]
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
        
        if 'input files' in config and 'spectra table' in config['input files']:
            if len(config_input_spectra) > 0:
                input_spectra = config_input_spectra[s]
            else:
                input_spectra = copy.copy(input_spectra)
            table_proc = pd.read_csv(input_spectra, sep=',')    
    
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
        if 'fit parameters' in radex_params:
            fit_params = radex_params['fit parameters']
        else:
            fit_params = config['RADEX fit parameters']
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
        if type(fit_params) is not str and hasattr(fit_params, '__iter__'):
            fit_params = fit_params[s]
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
        observed_uncs = observed_transitions['intensity unc. (K)'].values
        
        kin_temp = float(kin_temp)
        h2_num_dens = float(h2_num_dens)
        backgr_temp = float(backgr_temp)
        col_dens = float(col_dens)
        line_width = float(line_width)
        
        radex_transitions = \
            radex(molecule, min_freq, max_freq, kin_temp,
                  backgr_temp, h2_num_dens, col_dens, line_width)[0]
        inds = []
        for i, row_i in radex_transitions.iterrows():
            for j, row_j in observed_transitions.iterrows():
                freq_i = row_i['frequency (GHz)']
                freq_j = row_j['frequency (GHz)']
                if abs(freq_i - freq_j) < lines_margin:
                    inds += [i]
                    break
        
        if fit_params == 'column density':
             def radex_function(params):
                 log_x = params
                 x = 10**log_x
                 loss = get_loss_from_radex(molecule, min_freq, max_freq,
                                            kin_temp, backgr_temp, h2_num_dens,
                                            x, line_width, observed_lines,
                                            observed_uncs, inds)
                 print('col. dens. (/cm2): {:.3e}, loss: {:.3e}'
                       .format(float(x), loss))
                 return loss
        elif fit_params == 'column density, H2 number density':
             def radex_function(params):
                 log_x1, log_x2 = params
                 x1 = 10**log_x1
                 x2 = 10**log_x2
                 loss = get_loss_from_radex(molecule, min_freq, max_freq,
                                            kin_temp, backgr_temp, x2, x1,
                                            line_width, observed_lines,
                                            observed_uncs, inds)
                 print(('col. dens. (/cm2): {:.3e}, H2 dens. (/cm3): {:.3e}, '
                       + 'loss: {:.3e}').format(float(x1), float(x2), loss))
                 return loss
        elif fit_params == 'column density, temperature':
             def radex_function(params):
                 log_x1, x2 = params
                 x1 = 10**log_x1
                 loss = get_loss_from_radex(molecule, min_freq, max_freq, x2,
                                            backgr_temp, h2_num_dens, x1,
                                            line_width, observed_lines,
                                            observed_uncs, inds)
                 print(('col. dens. (/cm2): {:.3e}, temp. (K): {:.2f},'
                        + ' loss: {:.3e}').format(float(x1), float(x2), loss))
                 return loss

        if not load_previous_results:

            print('Minimizing column density for {} with RADEX.'
                  .format(molecule))            

            if fit_params == 'column density':
                guess = np.log10(float(col_dens))
                results = minimize(radex_function, guess, method='COBYLA',
                                   options={'maxiter': max_iters, 'disp': False})
                col_dens = float(10**results.x)
            elif fit_params == 'column density, H2 number density':
                guess = (np.log10(float(col_dens)), np.log10(float(h2_num_dens)))
                results = minimize(radex_function, guess, method='Nelder-Mead',
                                   options={'maxiter': max_iters, 'disp': False})
                col_dens = float(10**results.x[0])
                h2_num_dens = float(10**results.x[1])
            elif fit_params == 'column density, temperature':
                guess = (np.log10(float(col_dens)), float(kin_temp))
                results = minimize(radex_function, guess, method='Nelder-Mead',
                                   options={'maxiter': max_iters, 'disp': False})
                col_dens = float(10**results.x[0])
                kin_temp = float(results.x[1])
               
            radex_transitions = \
                radex(molecule, min_freq, max_freq, kin_temp,
                      backgr_temp, h2_num_dens, col_dens, line_width)[0]
            radex_lines = radex_transitions.iloc[inds]['intensity (K)'].values    
            losses_lim = loss_distribution(observed_lines, observed_uncs,
                                           radex_lines)
            loss_ref = loss_function(observed_lines, observed_uncs, radex_lines)
            idx = np.argmin(abs(np.sort(losses_lim) - loss_ref))
            lim = np.percentile(losses_lim, 68.27)
            
            print('Calculating uncertainty for {} with RADEX.'
                  .format(molecule))
            
            unc_results = \
                radex_uncertainty_grid(observed_transitions, molecule,
                                       min_freq, max_freq, kin_temp, backgr_temp,
                                       h2_num_dens, col_dens, line_width, lim,
                                       fit_params, grid_points=grid_points,
                                       inds=inds)
        else:
            
            col_dens, h2_num_dens, kin_temp, unc_results = \
                previous_results[k-1]
            print('Loaded previous results for {}.'.format(molecule))
            
        if fit_params == 'column density':
            col_dens_values, col_dens_losses = unc_results[1]
            col_dens = unc_results[0].main
            col_dens_uncs = unc_results[0].unc
        else:
            col_dens = unc_results[0].main
            col_dens_uncs = unc_results[0].unc        
            if fit_params == 'column density, H2 number density':
                h2_num_dens = unc_results[1].main
                h2_num_dens_uncs = unc_results[1].unc
            elif fit_params == 'column density, temperature':
                kin_temp = unc_results[1].main
                kin_temp_uncs = unc_results[1].unc    
            
        all_results += [[col_dens, h2_num_dens, kin_temp, unc_results]]

        radex_transitions = \
            radex(molecule, min_freq, max_freq, kin_temp,
                  backgr_temp, h2_num_dens, col_dens, line_width)[0]
        radex_lines = radex_transitions.iloc[inds]['intensity (K)'].values
        radex_freqs = observed_transitions['frequency (GHz)'].values
        radex_freqs = ['{:.6f}'.format(freq) for freq in radex_freqs]
        losses_lim = loss_distribution(observed_lines, observed_uncs,
                                       radex_lines)
        loss_ref = loss_function(observed_lines, observed_uncs, radex_lines)
        idx = np.argmin(abs(np.sort(losses_lim) - loss_ref))           
        lim = np.percentile(losses_lim, 68.27)
        
        print(radex_transitions)
        
        positions = np.arange(len(observed_lines))
        labels = radex_transitions['frequency (GHz)'].values
        
        if show_plots or save_plots:
            
            plt.close(k)
            
            def plot_minimum(x, y, x0, x_uncs, lim, label):
                """
                Make a plot of the loss function over the minimum of the input
                variable.
            
                Parameters
                ----------
                x : array (float)
                    Values of the input variable.
                y : array (float)
                    Values of the loss.
                x0 : float
                    Value of the input that minimizes the loss.
                x_uncs : list (float)
                    Lower and upper uncertainties of the input variable.
                lim : float
                    Value of the loss that determines the uncertainties
                label : str
                    Name of the input variable.
            
                Returns
                -------
                None.
            
                """
                plt.plot(x, y, '.', color='darkgreen',
                         label='{} - {}'.format(source, molecule))
                plt.axhline(lim, color='darkblue', linestyle='--')
                plt.axvline(x0, color='gray', linestyle='-')
                plt.axvline(x0 - x_uncs[0], color='gray', linestyle='--')
                plt.axvline(x0 + x_uncs[1], color='gray', linestyle='--')
                plt.xlabel(label)
                plt.ylabel('loss')
                plt.title('Loss values', fontweight='bold')
                ylim1, ylim2 = plt.ylim()
                if ylim1 < ylim2 / 2:
                    ylim1 = 0
                plt.ylim([ylim1, ylim2])
                ax.xaxis.major.formatter._useMathText = True
                ax.yaxis.major.formatter._useMathText = True
                return None
        
            
            n = 1
            if fit_params == 'column density':
                figsize = (14, 4)
            else:
                figsize = (10, 8)
            fig = plt.figure(k, figsize=figsize)
            plt.clf()
            
            if fit_params == 'column density':
                nrows, ncols = 1, 3
                ax = plt.subplot(nrows, ncols, n)
                plot_minimum(col_dens_values, col_dens_losses, col_dens,
                             col_dens_uncs, lim, 'column density (cm$^{-2}$)')
                n += 1
            else:
                nrows, ncols = 2, 2
                x, y, z = unc_results[2]
                x0, y0 = unc_results[:2]
                z0, zmax = z.min(), z.max()
                xp = np.linspace(x.min(), x.max(), 2)
                yp = np.linspace(y.min(), y.max(), 2)
                xx, yy = np.meshgrid(xp, yp)
                zp = 0*xx + 0**yy + lim
                norm = mpl.colors.LogNorm(vmin=0.01*lim, vmax=100*lim)
                xlabel = 'column density (cm$^{-2}$)'
                if fit_params == 'column density, H2 number density':
                    ylabel = 'H$_2$ number density (cm$^{-3})$'
                elif fit_params == 'column density, temperature':
                    ylabel = 'temperature (K)'
                ax = plt.subplot(nrows, ncols, n)
                colors = np.vstack((plt.cm.Oranges(np.linspace(0.2, 0.8, 128)),
                                    plt.cm.Greens(np.linspace(0.5, 1, 128))))
                ccmap = \
                    mpl.colors.LinearSegmentedColormap.from_list('colormap',
                                                                 colors)
                size = 1e4/len(z)
                plt.scatter(x, y, c=z, cmap=ccmap, s=size, alpha=1, norm=norm)
                plt.axhline(y0.main, color='gray', alpha=0.8, zorder=2)
                plt.axvline(x0.main, color='gray', alpha=0.8, zorder=2)
                plt.scatter(x0.main, y0.main, c=0.01*lim, cmap=ccmap,
                            s=size, zorder=3, norm=norm)
                # ax.scatter(x0.main, y0.main, s=2, c='darkorchid')
                cbar = plt.colorbar(label='loss', extend='both')
                cbar.ax.axhline(y=lim, color='darkblue', lw=2)
                # cbar.ax.axhline(y=z.min(), color='darkorchid')
                plt.margins(x=0, y=0)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.locator_params(nbins=4)
                ax.set_title('Loss values', fontweight='bold')
                ax.xaxis.major.formatter._useMathText = True
                ax.yaxis.major.formatter._useMathText = True
                ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())
                ax.xaxis.set_major_formatter(mticker.LogFormatterSciNotation())
                n += 1
                ax = plt.subplot(nrows, ncols, n, projection='3d')
                cond = z != 0
                ax.scatter(x[cond], y[cond], z[cond], c=z[cond], s=size,
                           cmap=ccmap, alpha=0.9, norm=norm)
                ax.scatter(x0.main, y0.main, z0, c=0.01*lim,
                           s=size, cmap=ccmap, alpha=1, norm=norm)
                ax.plot3D([x0.main, x0.main], [y0.main, y0.main],
                          [0, zmax], color='gray', alpha=0.7)
                ax.plot_surface(xx, yy, zp, alpha=0.7, color='darkblue')
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_zlabel('loss')
                ax.set_title('Loss values', fontweight='bold')
                zlim1, zlim2 = z.min(), z.max()
                if zlim1 < zlim2 / 2:
                    zlim1 = 0
                ax.set_zlim([zlim1, 3*lim])
                ax.locator_params(nbins=4)
                ax.xaxis.major.formatter._useMathText = True
                ax.yaxis.major.formatter._useMathText = True
                ax.zaxis.major.formatter._useMathText = True
                ax.yaxis.set_major_formatter(mticker.LogFormatterSciNotation())
                ax.xaxis.set_major_formatter(mticker.LogFormatterSciNotation())
                n += 1
            ax = plt.subplot(nrows, ncols, n)
            plt.hist(losses_lim, color='gray', histtype='stepfilled',
                     edgecolor='black', bins='auto', density=True)
            lw = 3.0 if loss_ref == 0 else 1.5
            plt.axvline(loss_ref, color='tab:blue', linewidth=lw, zorder=3)
            plt.plot([], [], color='tab:blue', linewidth=1.2,
                     label='no sampling')
            plt.axvline(lim, color='darkblue', label='$1\,$σ quantile',
                        linestyle='--', linewidth=1.2)
            plt.xlabel('loss')
            plt.ylabel('frequency density')
            plt.yscale('log')
            lim_value = lim/10 if losses_lim.min() == 0 else losses_lim.min()
            _, bins = np.histogram(losses_lim)
            lim_value = max(bins[1], lim_value)
            linthresh = 10**(round(np.log10(lim_value)))
            plt.xscale('symlog', linthresh=linthresh)
            plt.xlim(left=0)
            ylim1, ylim2 = plt.ylim()
            plt.ylim(bottom=100*ylim1)
            plt.title('Loss distribution at optimized parameters',
                      fontweight='bold')
            plt.legend(loc='upper right')
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(ticks_format))
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(ticks_format))
            # ax.ticklabel_format(style='sci', useMathText=True, useOffset=True)
            
            positions = np.arange(len(observed_lines))
            n += 1
            plt.subplot(nrows, ncols, n)
            plt.bar(positions, observed_lines, color='gray', edgecolor='black',
                    width=0.75, label='observed', linewidth=2, zorder=2)
            plt.bar(positions, radex_lines, color='white', alpha=0.5,
                    edgecolor='tab:green', width=0.75, hatch='/',
                    label='modeled', linewidth=1, zorder=2)
            plt.errorbar(positions, observed_lines, observed_uncs, fmt='.',
                         color='black', capsize=2, capthick=1, zorder=3,
                         ms=0.5)
            plt.xticks(positions, radex_freqs)
            plt.ylim(bottom=0)
            plt.xlabel('frequency (GHz)')
            plt.ylabel('intensity (K)')
            plt.legend()
            plt.title('Lines height', fontweight='bold')
            molecule_title = format_species_name(molecule.replace('-',' '))
            plt.suptitle('{} - {}'.format(source, molecule_title),
                         fontweight='bold')
            plt.tight_layout()
            
            if save_plots:
                file_name = '{}-{}.png'.format(source, molecule)
                plt.savefig(file_name, dpi=200)
                print('Saved plots in {}.'.format(file_name))
                if save_ind_plots:
                    size = fig.get_size_inches() / np.array([ncols, nrows])
                    file_name = file_name.replace('.png', '')
                    save_subplots(fig, size, name=file_name)
                    print('Saved also individual plots.')
            if show_plots:
                if not any('SPYDER' in name for name in os.environ):
                    print('Click the close window button to continue.')
                plt.show()
                plt.pause(0.1)
           
        
        col_dens = rv.RichValue(col_dens, col_dens_uncs)
        col_dens.num_sf = 2
        
        if fit_params == 'column density, H2 number density':
            h2_num_dens = rv.RichValue(h2_num_dens, h2_num_dens_uncs)
            h2_num_dens.num_sf = 2
        elif fit_params == 'column density, temperature':
            kin_temp = rv.RichValue(kin_temp, kin_temp_uncs)
            kin_temp.num_sf = 2
            
        col_dens = str(col_dens)
        h2_num_dens = str(h2_num_dens)
        kin_temp = str(kin_temp)
            
        if num_sources == 1:
            radex_dict[molecule]['column density (/cm2)'] = col_dens
            radex_dict[molecule]['hydrogen number density (/cm3)'] = \
                h2_num_dens
            radex_dict[molecule]['kinetic temperature (K)'] = kin_temp
        else:
            if s == 0:
                radex_dict[molecule]['column density (/cm2)'] = []
                radex_dict[molecule]['hydrogen number density (/cm3)'] = []
                radex_dict[molecule]['kinetic temperature (K)'] = []
            radex_dict[molecule]['column density (/cm2)'] += [col_dens]
            radex_dict[molecule]['hydrogen number density (/cm3)'] += \
                [h2_num_dens]
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
print('Saved results of the calculations in {}.\n'.format(results_file))


t2 = time.time()
t = t2 - t1
print('Elapsed time: {:.0f} min + {:.0f} s.\n'.format(t//60, t%60))