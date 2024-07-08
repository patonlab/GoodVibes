# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import

import math
import os.path
import sys
import numpy as np
import goodvibes.thermo as thermo
from goodvibes.utils import KCAL_TO_AU, J_TO_AU

        
class create_pes:
    """
    Obtain relative thermochemistry between species and for reactions.

    Routine that computes Boltzmann populations of conformer sets at each step of a reaction, obtaining
    relative energetic and thermodynamic values for each step in a reaction pathway.
    Determines reaction pathway from .yaml formatted file containing definitions for where files fit in pathway.

    Attributes:
    format (dict): format of the PES file.
    thermo_data (dict): dictionary containing thermochemistry data for each species.
    temperature (float): temperature to calculate thermochemistry at.
    gconf (bool): whether to apply conformer correction to thermochemistry data.
    QH (bool): whether to apply quasi-harmonic correction to thermochemistry data.
    cosmo (bool): whether to apply COSMO-RS correction to thermochemistry data.
    cosmo_int (bool): whether to apply COSMO-RS correction to thermochemistry data.
    """
    def __init__(self, format, thermo_data, pesnum=0, spc=False, temperature=298.15, gconf=True, QH=False, cosmo=None, cosmo_int=None):
        
        # the PES information 
        self.rxn = format.rxns[pesnum]
        self.path = format.pes_data[self.rxn]
        
        # check that all species in paths are defined
        self.points = []
        # check that the sum of basis functions is the same at each point along the path
        self.nbasis = []

        for i, point in enumerate(self.path):
            print(point)
            
            nbasis = 0
            self.points.append([])
            for pes_species in point.split('+'):
                pes_species = pes_species.strip()

                if pes_species not in format.pes_species_names:
                    print(f"\n!  Caution: {pes_species} is not defined as a species in {file}!\n")  
                    sys.exit()
                else:
                    self.points[i].append(pes_species)
                    for thermo in thermo_data:
                        if pes_species == thermo.pes_name:
                            nbasis += thermo.nbasis
            self.nbasis.append(nbasis)

        print(self.nbasis)

        if len(set(self.nbasis)) != 1:
            print(f"\n!  Caution: The number of basis functions is not the same for all species in {self.rxn}!\n")  
            sys.exit()

        # get the thermochemistry data
        if spc is not False: self.sp_energy = []
        self.scf_energy = []
        self.enthalpy = []
        self.gibbs_free_energy = []
        self.qh_gibbs_free_energy = []
          
        for i, point in enumerate(self.points):
            
            if set(format.zero) == set(point):
                zero_match = i

            sp_energy, scf_energy, enthalpy, gibbs_free_energy, qh_gibbs_free_energy = 0.0, 0.0, 0.0, 0.0, 0.0

            for species in point:
                for thermo in thermo_data:
                    if species == thermo.pes_name:
                        if spc is not False: 
                            sp_energy += thermo.sp_energy
                        scf_energy += thermo.scf_energy
                        enthalpy += thermo.enthalpy
                        gibbs_free_energy += thermo.gibbs_free_energy
                        qh_gibbs_free_energy += thermo.qh_gibbs_free_energy
            
            self.scf_energy.append(scf_energy)
            self.enthalpy.append(enthalpy)
            self.gibbs_free_energy.append(gibbs_free_energy)
            self.qh_gibbs_free_energy.append(qh_gibbs_free_energy)
            
            if spc is not False: 
                self.sp_energy.append(sp_energy)
                
        # get the relative thermochemistry
        if not zero_match:
            zero_match = 0
        try:
            if spc is not False: sp_e_zero = self.sp_energy[zero_match]
            scf_e_zero = self.scf_energy[zero_match]
            h_zero = self.enthalpy[zero_match]
            g_zero = self.gibbs_free_energy[zero_match]
            qh_g_zero = self.qh_gibbs_free_energy[zero_match]
        except IndexError:
            print(f"\n!  Caution: Unable to obtain relative energies for PES !\n")  
            sys.exit()

        # adjust to relative values based on energy zero and unit conversion
        if format.units == 'kJ/mol': unit_conversion = J_TO_AU / 1000.0
        else: unit_conversion = KCAL_TO_AU
        self.scf_energy = (self.scf_energy - scf_e_zero) * unit_conversion
        self.enthalpy = (self.enthalpy - h_zero) * unit_conversion
        self.gibbs_free_energy = (self.gibbs_free_energy - g_zero) * unit_conversion
        self.qh_gibbs_free_energy = (self.qh_gibbs_free_energy - qh_g_zero) * unit_conversion

        if spc is not False: 
            self.sp_energy = (self.sp_energy - sp_e_zero) * unit_conversion
            self.sp_enthalpy = self.enthalpy + self.sp_energy - self.scf_energy
            self.sp_gibbs_free_energy = self.gibbs_free_energy + self.sp_energy - self.scf_energy
            self.sp_qh_gibbs_free_energy = self.qh_gibbs_free_energy + self.sp_energy - self.scf_energy

def jitter(datasets, color, ax, nx, marker, edgecol='black'):
    """Scatter points that may overlap when graphing by randomly offsetting them."""
    import numpy as np
    for i, p in enumerate(datasets):
        y = [p]
        x = np.random.normal(nx, 0.015, size=len(y))
        ax.plot(x, y, alpha=0.5, markersize=7, color=color, marker=marker, markeredgecolor=edgecol,
                markeredgewidth=1, linestyle='None')

def graph_reaction_profile(graph_data, options, log, plt, value='G'):
    """
    Graph a reaction profile using quasi-harmonic Gibbs free energy values.

    Use matplotlib package to graph a reaction pathway potential energy surface.

    Parameters:
    graph_data (get_pes object): potential energy surface object containing relative thermodynamic data.
    log (Logger object): Logger to write status updates to user on command line.
    options (dict): input options for GV.
    plt (matplotlib): matplotlib library reference.
    """
    import matplotlib.path as mpath
    import matplotlib.patches as mpatches

    # default to using Gibbs energy values - but change to E or H if requested
    #sorted_thermo_data = dict(sorted(thermo_data.items(), key=lambda item: item[1].qh_gibbs_free_energy))
    #if value == "E": sorted_thermo_data = dict(sorted(thermo_data.items(), key=lambda item: item[1].scf_energy))
    #elif value == "H": sorted_thermo_data = dict(sorted(thermo_data.items(), key=lambda item: item[1].enthalpy))

    log.write("\n   Graphing Reaction Profile\n")
    data = {}

    # Get PES data
    for i, path in enumerate(graph_data.path):
        g_data = []
        if value == 'G': zero_val = graph_data.qhg_zero[i][0]
        if value == 'H': zero_val = graph_data.h_zero[i][0]
        if value == 'E': zero_val = graph_data.e_zero[i][0]
        for j, e_abs in enumerate(graph_data.e_abs[i]):
            if value == 'G': species = graph_data.qhg_abs[i][j]
            if value == 'H': species = graph_data.h_abs[i][j]
            if value == 'E': species = graph_data.e_abs[i][j]
            relative = species - zero_val
            if graph_data.units == 'kJ/mol':
                formatted_g = J_TO_AU / 1000.0 * relative
            else:
                formatted_g = KCAL_TO_AU * relative  # Defaults to kcal/mol
            g_data.append(formatted_g)
        data[path] = g_data

    # Grab any additional formatting for graph
    with open(options.graph) as f:
        yaml = f.readlines()

    #defaults
    ylim, color, show_conf, show_gconf, show_title = None, None, True, False, True
    label_point, label_xaxis, dpi, dec, legend = False, True, False, 2, False,
    colors, gridlines, title =  None, False, 'Potential Energy Surface'

    for i, line in enumerate(yaml):
        if line.strip().find('FORMAT') > -1:
            for j, line in enumerate(yaml[i + 1:]):
                if line.strip().find('ylim') > -1:
                    try:
                        ylim = line.strip().replace(':', '=').split("=")[1].replace(' ', '').strip().split(',')
                    except IndexError:
                        pass
                if line.strip().find('color') > -1:
                    try:
                        colors = line.strip().replace(':', '=').split("=")[1].replace(' ', '').strip().split(',')
                    except IndexError:
                        pass
                if line.strip().find('title') > -1:
                    try:
                        title_input = line.strip().replace(':', '=').split("=")[1].strip().split(',')[0]
                        if title_input == 'false' or title_input == 'False':
                            show_title = False
                        else:
                            title = title_input
                    except IndexError:
                        pass
                if line.strip().find('dec') > -1:
                    try:
                        dec = int(line.strip().replace(':', '=').split("=")[1].strip().split(',')[0])
                    except IndexError:
                        pass
                if line.strip().find('pointlabel') > -1:
                    try:
                        label_input = line.strip().replace(':', '=').split("=")[1].strip().split(',')[0].lower()
                        if label_input == 'false':
                            label_point = False
                    except IndexError:
                        pass
                if line.strip().find('show_conformers') > -1:
                    try:
                        conformers = line.strip().replace(':', '=').split("=")[1].strip().split(',')[0].lower()
                        if conformers == 'false':
                            show_conf = False
                    except IndexError:
                        pass
                if line.strip().find('show_gconf') > -1:
                    try:
                        gconf_input = line.strip().replace(':', '=').split("=")[1].strip().split(',')[0].lower()
                        if gconf_input == 'true':
                            show_gconf = True
                    except IndexError:
                        pass
                if line.strip().find('xlabel') > -1:
                    try:
                        label_input = line.strip().replace(':', '=').split("=")[1].strip().split(',')[0].lower()
                        if label_input == 'false':
                            label_xaxis = False
                    except IndexError:
                        pass
                if line.strip().find('dpi') > -1:
                    try:
                        dpi = int(line.strip().replace(':', '=').split("=")[1].strip().split(',')[0])
                    except IndexError:
                        pass
                if line.strip().find('legend') > -1:
                    try:
                        legend_input = line.strip().replace(':', '=').split("=")[1].strip().split(',')[0].lower()
                        if legend_input == 'false':
                            legend = False
                    except IndexError:
                        pass
                if line.strip().find('gridlines') > -1:
                    try:
                        gridline_input = line.strip().replace(':', '=').split("=")[1].strip().split(',')[0].lower()
                        if gridline_input == 'true':
                            gridlines = True
                    except IndexError:
                        pass

    # Do some graphing
    Path = mpath.Path
    fig, ax = plt.subplots()
    for i, path in enumerate(graph_data.path):
        for j in range(len(data[path]) - 1):
            if colors is not None:
                if len(colors) > 1:
                    color = colors[i]
                else:
                    color = colors[0]
            else:
                color = 'k'
                colors = ['k']
            if j == 0:
                path_patch = mpatches.PathPatch(
                    Path([(j, data[path][j]), (j + 0.5, data[path][j]), (j + 0.5, data[path][j + 1]),
                          (j + 1, data[path][j + 1])],
                         [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]),
                    label=path, fc="none", transform=ax.transData, color=color)
            else:
                path_patch = mpatches.PathPatch(
                    Path([(j, data[path][j]), (j + 0.5, data[path][j]), (j + 0.5, data[path][j + 1]),
                          (j + 1, data[path][j + 1])],
                         [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]),
                    fc="none", transform=ax.transData, color=color)
            ax.add_patch(path_patch)
            plt.hlines(data[path][j], j - 0.15, j + 0.15,colors=['k'])
        plt.hlines(data[path][-1], len(data[path]) - 1.15, len(data[path]) - 0.85,colors=['k'])

    if show_conf:
        markers = ['o', 's', 'x', 'P', 'D']
        for i in range(len(graph_data.g_qhgvals)):  # i = reaction pathways
            for j in range(len(graph_data.g_qhgvals[i])):  # j = reaction steps
                for k in range(len(graph_data.g_qhgvals[i][j])):  # k = species
                    zero_val = graph_data.g_species_qhgzero[i][j][k]
                    points = graph_data.g_qhgvals[i][j][k]
                    points[:] = [((x - zero_val) + (graph_data.qhg_abs[i][j] - graph_data.qhg_zero[i][0]) + (
                            graph_data.g_rel_val[i][j] - graph_data.qhg_abs[i][j])) * KCAL_TO_AU for x in points]
                    if len(colors) > 1:
                        jitter(points, colors[i], ax, j, markers[k])
                    else:
                        jitter(points, color, ax, j, markers[k])
                    if show_gconf:
                        plt.hlines((graph_data.g_rel_val[i][j] - graph_data.qhg_zero[i][0]) * KCAL_TO_AU, j - 0.15,
                                   j + 0.15, linestyles='dashed')

    # Annotate points with energy level
    if label_point:
        for i, path in enumerate(graph_data.path):
            for i, point in enumerate(data[path]):
                if dec == 1:
                    ax.annotate("{:.1f}".format(point), (i, point - fig.get_figheight() * fig.dpi * 0.025),
                                horizontalalignment='center')
                else:
                    ax.annotate("{:.2f}".format(point), (i, point - fig.get_figheight() * fig.dpi * 0.025),
                                horizontalalignment='center')
    if ylim is not None:
        ax.set_ylim(float(ylim[0]), float(ylim[1]))
    if show_title:
        if title is not None:
            ax.set_title(title)
        else:
            ax.set_title("Reaction Profile")
    if value == 'G': ax.set_ylabel(r"$G_{rel}$ (kcal / mol)")
    if value == 'H': ax.set_ylabel(r"$H_{rel}$ (kcal / mol)")
    if value == 'E': ax.set_ylabel(r"$E_{rel}$ (kcal / mol)")
    plt.minorticks_on()
    ax.tick_params(axis='x', which='minor', bottom=False)
    ax.tick_params(which='minor', labelright=True, right=True)
    ax.tick_params(labelright=True, right=True)
    if gridlines:
        ax.yaxis.grid(linestyle='--', linewidth=0.5)
        ax.xaxis.grid(linewidth=0)
    ax_label = []
    xaxis_text = []
    newax_text_list = []
    for i, path in enumerate(graph_data.path):
        newax_text = []
        ax_label.append(path)
        for j, e_abs in enumerate(graph_data.e_abs[i]):
            if i == 0:
                xaxis_text.append(graph_data.species[i][j])
            else:
                newax_text.append(graph_data.species[i][j])
        newax_text_list.append(newax_text)
    # Label rxn steps
    if label_xaxis:
        if colors is not None:
            plt.xticks(range(len(xaxis_text)), xaxis_text, color=colors[0])
        else:
            plt.xticks(range(len(xaxis_text)), xaxis_text, color='k')
        locs, labels = plt.xticks()
        newax = []
        for i in range(len(ax_label)):
            if i > 0:
                y = ax.twiny()
                newax.append(y)
        for i in range(len(newax)):
            newax[i].set_xticks(locs)
            newax[i].set_xlim(ax.get_xlim())
            if len(colors) > 1:
                newax[i].tick_params(axis='x', colors=colors[i + 1])
            else:
                newax[i].tick_params(axis='x', colors='k')
            newax[i].set_xticklabels(newax_text_list[i + 1])
            newax[i].xaxis.set_ticks_position('bottom')
            newax[i].xaxis.set_label_position('bottom')
            newax[i].xaxis.set_ticks_position('none')
            newax[i].spines['bottom'].set_position(('outward', 15 * (i + 1)))
            newax[i].spines['bottom'].set_visible(False)
    else:
        plt.xticks(range(len(xaxis_text)))
        ax.xaxis.set_ticklabels([])
    if legend:
        plt.legend()
    if dpi is not False:
        plt.savefig('Rxn_profile_' + options.graph.split('.')[0] + '.png', dpi=dpi)
    plt.show()

def sel_striplot(a_name, b_name, a_files, b_files, thermo_data, plt):
    import seaborn
    names = [a_name] * len(a_files) + [b_name] * len(b_files)

    a_thermo = [thermo_data[a] for a in a_files]
    b_thermo = [thermo_data[b] for b in b_files]

    a_energy = [item.scf_energy for item in a_thermo]
    b_energy = [item.scf_energy for item in b_thermo]
    glob_min = min(a_energy+b_energy)

    a_energy = [(en - glob_min) * KCAL_TO_AU for en in a_energy]
    b_energy = [(en - glob_min) * KCAL_TO_AU for en in b_energy]

    seaborn.set(style = 'whitegrid')
    seaborn.stripplot(x=names, y=a_energy+b_energy, jitter=0.1)
    plt.show()
