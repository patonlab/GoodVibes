# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import

import math, os.path, sys
import numpy as np
import goodvibes.thermo as thermo

# PHYSICAL CONSTANTS                                      UNITS
GAS_CONSTANT = 8.3144621  # J / K / mol
J_TO_AU = 4.184 * 627.509541 * 1000.0  # UNIT CONVERSION
KCAL_TO_AU = 627.509541  # UNIT CONVERSION

class get_pes:
    """
    Obtain relative thermochemistry between species and for reactions.

    Routine that computes Boltzmann populations of conformer sets at each step of a reaction, obtaining
    relative energetic and thermodynamic values for each step in a reaction pathway.
    Determines reaction pathway from .yaml formatted file containing definitions for where files fit in pathway.

    Attributes:
        dec (int): decimal places to display after PES calculations.
        units (str): units do display values in, choice of kcal/mol or kJ/mol.
        boltz (str): allows for selectivity calculation to display to user.
        path (list): list of strings defining each reaction pathway.
        species (list): list of strings defining which files correspond to names given in reaction pathway.
        spc_abs (list): list of relative single-point energy values.
        e_abs (list): list of relative energy values.
        zpe_abs (list): list of relative zero point energy values.
        h_abs (list): list of relative enthalpy values.
        qh_abs (list): list of relative quasi-harmonic enthalpy values.
        s_abs (list): list of relative entropy values.
        qs_abs (list): list of relative quasi-harmonic entropy values.
        g_abs (list): list of relative Gibbs free energy values.
        qhg_abs (list): list of relative quasi-harmonic Gibbs free energy values.
        cosmo_qhg_abs (list): list of relative COSMO-RS solvation-corrected quasi-harmonic Gibbs free energy values.
        spc_zero (list): list of single point energy "zero" species values to compare all other steps in pathway to.
        e_zero (list): list of energy "zero" species values to compare all other steps in pathway to.
        zpe_zero (list): list of zero point energy "zero" species values to compare all other steps in pathway to.
        h_zero (list): list of enthalpy "zero" species values to compare all other steps in pathway to.
        qh_zero (list): list of quasi-harmonic enthalpy "zero" species values to compare all other steps in pathway to.
        ts_zero (list): list of T*entropy "zero" species values to compare all other steps in pathway to.
        qhts_zero (list): list of quasi-harmonic T*entropy "zero" species values to compare all other steps in pathway to.
        g_zero (list): list of Gibbs free energy "zero" species values to compare all other steps in pathway to.
        qhg_zero (list): list of quasi-harmonic Gibbs free energy "zero" species values to compare all other steps in pathway to.
        cosmo_qhg_zero (list): list of COSMO-RS solvation-corrected quasi-harmonic Gibbs free energy "zero" species values to compare all other steps in pathway to.
        g_qhgvals (list): relative quasi-harmonic Gibbs free energy values used for graphing.
        g_species_qhgzero (list):quasi-harmonic Gibbs free energy "zero" values used for graphing.
        g_rel_val (list): relative Gibbs free energy values used for graphing.
    """
    def __init__(self, file, thermo_data, log, temperature, gconf=True, QH=False, cosmo=None, cosmo_int=None):
        # Default values
        self.dec, self.units, self.boltz = 2, 'kcal/mol', False

        with open(file) as f:
            data = f.readlines()
        folder, program, names, files, zeros, pes_list = None, None, [], [], [], []
        for i, dline in enumerate(data):
            if dline.strip().find('PES') > -1:
                for j, line in enumerate(data[i + 1:]):
                    if line.strip().startswith('#'):
                        pass
                    elif len(line) <= 2:
                        pass
                    elif line.strip().startswith('---'):
                        break
                    elif line.strip() != '':
                        pathway, pes = line.strip().replace(':', '=').split("=")
                        # Auto-grab first species as zero unless specified
                        pes_list.append(pes)
                        zeros.append(pes.strip().lstrip('[').rstrip(']').split(',')[0])
                        # Look at SPECIES block to determine filenames
            if dline.strip().find('SPECIES') > -1:
                for j, line in enumerate(data[i + 1:]):
                    if line.strip().startswith('---'):
                        break
                    else:
                        if line.lower().strip().find('folder') > -1:
                            try:
                                folder = line.strip().replace('#', '=').split("=")[1].strip()
                            except IndexError:
                                pass
                        else:
                            try:
                                n, f = (line.strip().replace(':', '=').split("="))
                                # Check the specified filename is also one that GoodVibes has thermochemistry for:
                                if f.find('*') == -1 and f not in pes_list:
                                    match = None
                                    for key in thermo_data:
                                        if os.path.splitext(os.path.basename(key))[0] in f.replace('[', '').replace(']', '').replace('+', ',').replace(' ', '').split(','):
                                            match = key
                                    if match:
                                        names.append(n.strip())
                                        files.append(match)
                                    else:
                                        log.write("   Warning! " + f.strip() + ' is specified in ' + file +
                                                  ' but no thermochemistry data found\n')
                                elif f not in pes_list:
                                    match = []
                                    for key in thermo_data:
                                        if os.path.splitext(os.path.basename(key))[0].find(f.strip().strip('*')) == 0:
                                            match.append(key)
                                    if len(match) > 0:
                                        names.append(n.strip())
                                        files.append(match)
                                    else:
                                        log.write("   Warning! " + f.strip() + ' is specified in ' + file +
                                                  ' but no thermochemistry data found\n')
                            except ValueError:
                                if line.isspace():
                                    pass
                                elif line.strip().find('#') > -1:
                                    pass
                                elif len(line) > 2:
                                    warn = "   Warning! " + file + ' input is incorrectly formatted for line:\n\t' + line
                                    log.write(warn)
            # Look at FORMAT block to see if user has specified any formatting rules
            if dline.strip().find('FORMAT') > -1:
                for j, line in enumerate(data[i + 1:]):
                    if line.strip().find('dec') > -1:
                        try:
                            self.dec = int(line.strip().replace(':', '=').split("=")[1].strip())
                        except IndexError:
                            pass
                    if line.strip().find('zero') > -1:
                        zeros = []
                        try:
                            zeros.append(line.strip().replace(':', '=').split("=")[1].strip())
                        except IndexError:
                            pass
                    if line.strip().find('units') > -1:
                        try:
                            self.units = line.strip().replace(':', '=').split("=")[1].strip()
                        except IndexError:
                            pass
                    if line.strip().find('boltz') > -1:
                        try:
                            self.boltz = line.strip().replace(':', '=').split("=")[1].strip()
                        except IndexError:
                            pass

        for i in range(len(files)):
            if len(files[i]) == 1:
                files[i] = files[i][0]
        species = dict(zip(names, files))
        self.path, self.species = [], []
        self.spc_abs, self.e_abs, self.zpe_abs, self.h_abs, self.qh_abs, self.s_abs, self.qs_abs, self.g_abs, self.qhg_abs, self.cosmo_qhg_abs = [], [], [], [], [], [], [], [], [], []
        self.spc_zero, self.e_zero, self.zpe_zero, self.h_zero, self.qh_zero, self.ts_zero, self.qhts_zero, self.g_zero, self.qhg_zero, self.cosmo_qhg_zero = [], [], [], [], [], [], [], [], [], []
        self.g_qhgvals, self.g_species_qhgzero, self.g_rel_val = [], [], []
        # Loop over .yaml file, grab energies, populate arrays and compute Boltzmann factors
        with open(file) as f:
            data = f.readlines()
        for i, dline in enumerate(data):
            if dline.strip().find('PES') > -1:
                n = 0
                for j, line in enumerate(data[i + 1:]):
                    if line.strip().startswith('#') == True:
                        pass
                    elif len(line) <= 2:
                        pass
                    elif line.strip().startswith('---') == True:
                        break
                    elif line.strip() != '':
                        try:
                            self.e_zero.append([])
                            self.spc_zero.append([])
                            self.zpe_zero.append([])
                            self.h_zero.append([])
                            self.qh_zero.append([])
                            self.ts_zero.append([])
                            self.qhts_zero.append([])
                            self.g_zero.append([])
                            self.qhg_zero.append([])
                            self.cosmo_qhg_zero.append([])
                            min_conf = False
                            spc_zero, e_zero, zpe_zero, h_zero, qh_zero, s_zero, qs_zero, g_zero, qhg_zero = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                            h_conf, h_tot, s_conf, s_tot, qh_conf, qh_tot, qs_conf, qs_tot, cosmo_qhg_zero = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                            zero_structures = zeros[n].replace(' ', '').split('+')
                            # Routine for 'zero' values
                            for structure in zero_structures:
                                try:
                                    if not isinstance(species[structure], list):
                                        if hasattr(thermo_data[species[structure]], "sp_energy"):
                                            spc_zero += thermo_data[species[structure]].sp_energy
                                        e_zero += thermo_data[species[structure]].scf_energy
                                        zpe_zero += thermo_data[species[structure]].zpe
                                        h_zero += thermo_data[species[structure]].enthalpy
                                        qh_zero += thermo_data[species[structure]].qh_enthalpy
                                        s_zero += thermo_data[species[structure]].entropy
                                        qs_zero += thermo_data[species[structure]].qh_entropy
                                        g_zero += thermo_data[species[structure]].gibbs_free_energy
                                        qhg_zero += thermo_data[species[structure]].qh_gibbs_free_energy
                                        cosmo_qhg_zero += thermo_data[species[structure]].cosmo_qhg
                                    else:  # If we have a list of different kinds of structures: loop over conformers
                                        g_min, boltz_sum = sys.float_info.max, 0.0
                                        for conformer in species[
                                            structure]:  # Find minimum G, along with associated enthalpy and entropy
                                            if cosmo:
                                                if thermo_data[conformer].cosmo_qhg <= g_min:
                                                    min_conf = thermo_data[conformer]
                                                    g_min = thermo_data[conformer].cosmo_qhg
                                            else:
                                                if thermo_data[conformer].qh_gibbs_free_energy <= g_min:
                                                    min_conf = thermo_data[conformer]
                                                    g_min = thermo_data[conformer].qh_gibbs_free_energy
                                        for conformer in species[structure]:  # Get a Boltzmann sum for conformers
                                            if cosmo:
                                                g_rel = thermo_data[conformer].cosmo_qhg - g_min
                                            else:
                                                g_rel = thermo_data[conformer].qh_gibbs_free_energy - g_min
                                            boltz_fac = math.exp(-g_rel * J_TO_AU / GAS_CONSTANT / temperature)
                                            boltz_sum += boltz_fac
                                        for conformer in species[
                                            structure]:  # Calculate relative data based on Gmin and the Boltzmann sum
                                            if cosmo:
                                                g_rel = thermo_data[conformer].cosmo_qhg - g_min
                                            else:
                                                g_rel = thermo_data[conformer].qh_gibbs_free_energy - g_min
                                            boltz_fac = math.exp(-g_rel * J_TO_AU / GAS_CONSTANT / temperature)
                                            boltz_prob = boltz_fac / boltz_sum
                                            #if no contribution, skip further calculations
                                            if boltz_prob == 0.0:
                                                continue

                                            if hasattr(thermo_data[conformer], "sp_energy") and thermo_data[
                                                conformer].sp_energy != '!':
                                                spc_zero += thermo_data[conformer].sp_energy * boltz_prob
                                            if hasattr(thermo_data[conformer], "sp_energy") and thermo_data[
                                                conformer].sp_energy == '!':
                                                sys.exit(
                                                    "Not all files contain a SPC value, relative values will not be calculated.")
                                            e_zero += thermo_data[conformer].scf_energy * boltz_prob
                                            zpe_zero += thermo_data[conformer].zpe * boltz_prob
                                            # Default calculate gconf correction for conformers, skip if no contribution
                                            if gconf and boltz_prob > 0.0 and boltz_prob != 1.0:
                                                h_conf += thermo_data[conformer].enthalpy * boltz_prob
                                                s_conf += thermo_data[conformer].entropy * boltz_prob
                                                s_conf += -GAS_CONSTANT / J_TO_AU * boltz_prob * math.log(boltz_prob)

                                                qh_conf += thermo_data[conformer].qh_enthalpy * boltz_prob
                                                qs_conf += thermo_data[conformer].qh_entropy * boltz_prob
                                                qs_conf += -GAS_CONSTANT / J_TO_AU * boltz_prob * math.log(boltz_prob)
                                            elif gconf and boltz_prob == 1.0:
                                                h_conf += thermo_data[conformer].enthalpy
                                                s_conf += thermo_data[conformer].entropy
                                                qh_conf += thermo_data[conformer].qh_enthalpy
                                                qs_conf += thermo_data[conformer].qh_entropy
                                            else:
                                                h_zero += thermo_data[conformer].enthalpy * boltz_prob
                                                s_zero += thermo_data[conformer].entropy * boltz_prob
                                                g_zero += thermo_data[conformer].gibbs_free_energy * boltz_prob

                                                qh_zero += thermo_data[conformer].qh_enthalpy * boltz_prob
                                                qs_zero += thermo_data[conformer].qh_entropy * boltz_prob
                                                qhg_zero += thermo_data[conformer].qh_gibbs_free_energy * boltz_prob
                                                cosmo_qhg_zero += thermo_data[conformer].cosmo_qhg * boltz_prob

                                        if gconf:
                                            h_adj = h_conf - min_conf.enthalpy
                                            h_tot = min_conf.enthalpy + h_adj
                                            s_adj = s_conf - min_conf.entropy
                                            s_tot = min_conf.entropy + s_adj
                                            g_corr = h_tot - temperature * s_tot
                                            qh_adj = qh_conf - min_conf.qh_enthalpy
                                            qh_tot = min_conf.qh_enthalpy + qh_adj
                                            qs_adj = qs_conf - min_conf.qh_entropy
                                            qs_tot = min_conf.qh_entropy + qs_adj
                                            if QH:
                                                qg_corr = qh_tot - temperature * qs_tot
                                            else:
                                                qg_corr = h_tot - temperature * qs_tot
                                except KeyError:
                                    log.write(
                                        "   Warning! Structure " + structure + ' has not been defined correctly as energy-zero in ' + file + '\n')
                                    log.write(
                                        "   Make sure this structure matches one of the SPECIES defined in the same file\n")
                                    sys.exit("   Please edit " + file + " and try again\n")
                            # Set zero vals here
                            conformers, single_structure, mix = False, False, False
                            for structure in zero_structures:
                                if not isinstance(species[structure], list):
                                    single_structure = True
                                else:
                                    conformers = True
                            if conformers and single_structure:
                                mix = True
                            if gconf and min_conf is not False:
                                if mix:
                                    h_mix = h_tot + h_zero
                                    s_mix = s_tot + s_zero
                                    g_mix = g_corr + g_zero
                                    qh_mix = qh_tot + qh_zero
                                    qs_mix = qs_tot + qs_zero
                                    qg_mix = qg_corr + qhg_zero
                                    cosmo_qhg_mix = qg_corr + cosmo_qhg_zero
                                    self.h_zero[n].append(h_mix)
                                    self.ts_zero[n].append(s_mix)
                                    self.g_zero[n].append(g_mix)
                                    self.qh_zero[n].append(qh_mix)
                                    self.qhts_zero[n].append(qs_mix)
                                    self.qhg_zero[n].append(qg_mix)
                                    self.cosmo_qhg_zero[n].append(cosmo_qhg_mix)
                                elif conformers:
                                    self.h_zero[n].append(h_tot)
                                    self.ts_zero[n].append(s_tot)
                                    self.g_zero[n].append(g_corr)
                                    self.qh_zero[n].append(qh_tot)
                                    self.qhts_zero[n].append(qs_tot)
                                    self.qhg_zero[n].append(qg_corr)
                                    self.cosmo_qhg_zero[n].append(qg_corr)
                            else:
                                self.h_zero[n].append(h_zero)
                                self.ts_zero[n].append(s_zero)
                                self.g_zero[n].append(g_zero)

                                self.qh_zero[n].append(qh_zero)
                                self.qhts_zero[n].append(qs_zero)
                                self.qhg_zero[n].append(qhg_zero)
                                self.cosmo_qhg_zero[n].append(cosmo_qhg_zero)

                            self.spc_zero[n].append(spc_zero)
                            self.e_zero[n].append(e_zero)
                            self.zpe_zero[n].append(zpe_zero)

                            self.species.append([])
                            self.e_abs.append([])
                            self.spc_abs.append([])
                            self.zpe_abs.append([])
                            self.h_abs.append([])
                            self.qh_abs.append([])
                            self.s_abs.append([])
                            self.g_abs.append([])
                            self.qs_abs.append([])
                            self.qhg_abs.append([])
                            self.cosmo_qhg_abs.append([])
                            self.g_qhgvals.append([])
                            self.g_species_qhgzero.append([])
                            self.g_rel_val.append([])  # graphing

                            pathway, pes = line.strip().replace(':', '=').split("=")
                            pes = pes.strip()
                            points = [entry.strip() for entry in pes.lstrip('[').rstrip(']').split(',')]
                            self.path.append(pathway.strip())
                            # Obtain relative values for each species
                            for i, point in enumerate(points):
                                if point != '':
                                    # Create values to populate
                                    point_structures = point.replace(' ', '').split('+')
                                    e_abs, spc_abs, zpe_abs, h_abs, qh_abs, s_abs, g_abs, qs_abs, qhg_abs, cosmo_qhg_abs = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                                    qh_conf, qh_tot, qs_conf, qs_tot, h_conf, h_tot, s_conf, s_tot, g_corr, qg_corr = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                                    min_conf = False
                                    rel_val = 0.0
                                    self.g_qhgvals[n].append([])
                                    self.g_species_qhgzero[n].append([])
                                    try:
                                        for j, structure in enumerate(point_structures):  # Loop over structures, structures are species specified
                                            zero_conf = 0.0
                                            self.g_qhgvals[n][i].append([])
                                            if not isinstance(species[structure], list):  # Only one conf in structures
                                                e_abs += thermo_data[species[structure]].scf_energy
                                                if hasattr(thermo_data[species[structure]], "sp_energy"):
                                                    spc_abs += thermo_data[species[structure]].sp_energy
                                                zpe_abs += thermo_data[species[structure]].zpe
                                                h_abs += thermo_data[species[structure]].enthalpy
                                                qh_abs += thermo_data[species[structure]].qh_enthalpy
                                                s_abs += thermo_data[species[structure]].entropy
                                                g_abs += thermo_data[species[structure]].gibbs_free_energy
                                                qs_abs += thermo_data[species[structure]].qh_entropy
                                                qhg_abs += thermo_data[species[structure]].qh_gibbs_free_energy
                                                cosmo_qhg_abs += thermo_data[species[structure]].cosmo_qhg
                                                zero_conf += thermo_data[species[structure]].qh_gibbs_free_energy
                                                self.g_qhgvals[n][i][j].append(
                                                    thermo_data[species[structure]].qh_gibbs_free_energy)
                                                rel_val += thermo_data[species[structure]].qh_gibbs_free_energy
                                            else:  # If we have a list of different kinds of structures: loop over conformers
                                                g_min, boltz_sum = sys.float_info.max, 0.0
                                                # Find minimum G, along with associated enthalpy and entropy
                                                for conformer in species[structure]:
                                                    if cosmo:
                                                        if thermo_data[conformer].cosmo_qhg <= g_min:
                                                            min_conf = thermo_data[conformer]
                                                            g_min = thermo_data[conformer].cosmo_qhg
                                                    else:
                                                        if thermo_data[conformer].qh_gibbs_free_energy <= g_min:
                                                            min_conf = thermo_data[conformer]
                                                            g_min = thermo_data[conformer].qh_gibbs_free_energy
                                                # Get a Boltzmann sum for conformers
                                                for conformer in species[structure]:
                                                    if cosmo:
                                                        g_rel = thermo_data[conformer].cosmo_qhg - g_min
                                                    else:
                                                        g_rel = thermo_data[conformer].qh_gibbs_free_energy - g_min
                                                    boltz_fac = math.exp(-g_rel * J_TO_AU / GAS_CONSTANT / temperature)
                                                    boltz_sum += boltz_fac
                                                # Calculate relative data based on Gmin and the Boltzmann sum
                                                for conformer in species[structure]:
                                                    if cosmo:
                                                        g_rel = thermo_data[conformer].cosmo_qhg - g_min
                                                    else:
                                                        g_rel = thermo_data[conformer].qh_gibbs_free_energy - g_min
                                                    boltz_fac = math.exp(-g_rel * J_TO_AU / GAS_CONSTANT / temperature)
                                                    boltz_prob = boltz_fac / boltz_sum
                                                    if boltz_prob == 0.0:
                                                        continue
                                                    if hasattr(thermo_data[conformer], "sp_energy") and thermo_data[
                                                        conformer].sp_energy != '!':
                                                        spc_abs += thermo_data[conformer].sp_energy * boltz_prob
                                                    if hasattr(thermo_data[conformer], "sp_energy") and thermo_data[conformer].sp_energy == '!':
                                                        sys.exit("\n   Not all files contain a SPC value, relative values will not be calculated.\n")
                                                    e_abs += thermo_data[conformer].scf_energy * boltz_prob
                                                    zpe_abs += thermo_data[conformer].zpe * boltz_prob
                                                    if cosmo:
                                                        zero_conf += thermo_data[conformer].cosmo_qhg * boltz_prob
                                                        rel_val += thermo_data[conformer].cosmo_qhg * boltz_prob
                                                    else:
                                                        zero_conf += thermo_data[
                                                                         conformer].qh_gibbs_free_energy * boltz_prob
                                                        rel_val += thermo_data[
                                                                       conformer].qh_gibbs_free_energy * boltz_prob
                                                    # Default calculate gconf correction for conformers, skip if no contribution
                                                    if gconf and boltz_prob > 0.0 and boltz_prob != 1.0:
                                                        h_conf += thermo_data[conformer].enthalpy * boltz_prob
                                                        s_conf += thermo_data[conformer].entropy * boltz_prob
                                                        s_conf += -GAS_CONSTANT / J_TO_AU * boltz_prob * math.log(boltz_prob)

                                                        qh_conf += thermo_data[conformer].qh_enthalpy * boltz_prob
                                                        qs_conf += thermo_data[conformer].qh_entropy * boltz_prob
                                                        qs_conf += -GAS_CONSTANT / J_TO_AU * boltz_prob * math.log(boltz_prob)
                                                    elif gconf and boltz_prob == 1.0:
                                                        h_conf += thermo_data[conformer].enthalpy
                                                        s_conf += thermo_data[conformer].entropy
                                                        qh_conf += thermo_data[conformer].qh_enthalpy
                                                        qs_conf += thermo_data[conformer].qh_entropy
                                                    else:
                                                        h_abs += thermo_data[conformer].enthalpy * boltz_prob
                                                        s_abs += thermo_data[conformer].entropy * boltz_prob
                                                        g_abs += thermo_data[conformer].gibbs_free_energy * boltz_prob

                                                        qh_abs += thermo_data[conformer].qh_enthalpy * boltz_prob
                                                        qs_abs += thermo_data[conformer].qh_entropy * boltz_prob
                                                        qhg_abs += thermo_data[
                                                                       conformer].qh_gibbs_free_energy * boltz_prob
                                                        cosmo_qhg_abs += thermo_data[conformer].cosmo_qhg * boltz_prob
                                                    if cosmo:
                                                        self.g_qhgvals[n][i][j].append(thermo_data[conformer].cosmo_qhg)
                                                    else:
                                                        self.g_qhgvals[n][i][j].append(thermo_data[conformer].qh_gibbs_free_energy)
                                                if gconf:
                                                    h_adj = h_conf - min_conf.enthalpy
                                                    h_tot = min_conf.enthalpy + h_adj
                                                    s_adj = s_conf - min_conf.entropy
                                                    s_tot = min_conf.entropy + s_adj
                                                    g_corr = h_tot - temperature * s_tot
                                                    qh_adj = qh_conf - min_conf.qh_enthalpy
                                                    qh_tot = min_conf.qh_enthalpy + qh_adj
                                                    qs_adj = qs_conf - min_conf.qh_entropy
                                                    qs_tot = min_conf.qh_entropy + qs_adj
                                                    if QH:
                                                        qg_corr = qh_tot - temperature * qs_tot
                                                    else:
                                                        qg_corr = h_tot - temperature * qs_tot
                                            self.g_species_qhgzero[n][i].append(zero_conf)  # Raw data for graphing
                                    except KeyError:
                                        log.write("   Warning! Structure " + structure + ' has not been defined correctly in ' + file + '\n')
                                        sys.exit("   Please edit " + file + " and try again\n")
                                    self.species[n].append(point)
                                    self.e_abs[n].append(e_abs)
                                    self.spc_abs[n].append(spc_abs)
                                    self.zpe_abs[n].append(zpe_abs)
                                    conformers, single_structure, mix = False, False, False
                                    self.g_rel_val[n].append(rel_val)
                                    for structure in point_structures:
                                        if not isinstance(species[structure], list):
                                            single_structure = True
                                        else:
                                            conformers = True
                                    if conformers and single_structure:
                                        mix = True
                                    if gconf and min_conf is not False:
                                        if mix:
                                            h_mix = h_tot + h_abs
                                            s_mix = s_tot + s_abs
                                            g_mix = g_corr + g_abs
                                            qh_mix = qh_tot + qh_abs
                                            qs_mix = qs_tot + qs_abs
                                            qg_mix = qg_corr + qhg_abs
                                            cosmo_qhg_mix = qg_corr + cosmo_qhg_zero
                                            self.h_abs[n].append(h_mix)
                                            self.s_abs[n].append(s_mix)
                                            self.g_abs[n].append(g_mix)
                                            self.qh_abs[n].append(qh_mix)
                                            self.qs_abs[n].append(qs_mix)
                                            self.qhg_abs[n].append(qg_mix)
                                            self.cosmo_qhg_abs[n].append(cosmo_qhg_mix)
                                        elif conformers:
                                            self.h_abs[n].append(h_tot)
                                            self.s_abs[n].append(s_tot)
                                            self.g_abs[n].append(g_corr)
                                            self.qh_abs[n].append(qh_tot)
                                            self.qs_abs[n].append(qs_tot)
                                            self.qhg_abs[n].append(qg_corr)
                                            self.cosmo_qhg_abs[n].append(qg_corr)
                                    else:
                                        self.h_abs[n].append(h_abs)
                                        self.s_abs[n].append(s_abs)
                                        self.g_abs[n].append(g_abs)

                                        self.qh_abs[n].append(qh_abs)
                                        self.qs_abs[n].append(qs_abs)
                                        self.qhg_abs[n].append(qhg_abs)
                                        self.cosmo_qhg_abs[n].append(cosmo_qhg_abs)
                                else:
                                    self.species[n].append('none')
                                    self.e_abs[n].append(float('nan'))

                            n = n + 1
                        except IndexError:
                            pass

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
            plt.hlines(data[path][j], j - 0.15, j + 0.15)
        plt.hlines(data[path][-1], len(data[path]) - 1.15, len(data[path]) - 0.85)

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

def tabulate(thermo_data, options, log, show=False):
    table = []
    ''' Tabulate relative values'''
    stars = "   " + "*" * 128
    if options.spc: stars = stars + "*" * 23
    if show != False:
        if options.gconf:
            log.write('\n   Gconf correction requested to be applied to below relative values using quasi-harmonic Boltzmann factors\n')
        for key in thermo_data:
            if not hasattr(thermo_data[key], "qh_gibbs_free_energy"):
                pes_error = "\nWarning! Could not find thermodynamic data for " + key + "\n"
                sys.exit(pes_error)
            if not hasattr(thermo_data[key], "sp_energy") and options.spc is not False:
                pes_error = "\nWarning! Could not find thermodynamic data for " + key + "\n"
                sys.exit(pes_error)
    # Interval applied to PES
    if options.temperature_interval:
        stars = stars + '*' * 22
        for i in range(len(interval)):
            bbe_vals = []
            for j in range(len(interval_bbe_data)):
                bbe_vals.append(interval_bbe_data[j][i])
            interval_thermo_data.append(dict(zip(file_list, bbe_vals)))
        j = 0
        for i in interval:
            temp = float(i)
            if options.cosmo_int is False:
                pes = get_pes(options.pes, interval_thermo_data[j], options, log, temp, options.gconf, options.QH)
            else:
                pes = get_pes(options.pes, interval_thermo_data[j], options, log, temp, options.gconf, options.QH,
                              cosmo=True)
            for k, path in enumerate(pes.path):
                if options.QH:
                    zero_vals = [pes.spc_zero[k][0], pes.e_zero[k][0], pes.zpe_zero[k][0], pes.h_zero[k][0],
                                 pes.qh_zero[k][0], temp * pes.ts_zero[k][0], temp * pes.qhts_zero[k][0],
                                 pes.g_zero[k][0], pes.qhg_zero[k][0]]
                else:
                    zero_vals = [pes.spc_zero[k][0], pes.e_zero[k][0], pes.zpe_zero[k][0], pes.h_zero[k][0],
                                 temp * pes.ts_zero[k][0], temp * pes.qhts_zero[k][0], pes.g_zero[k][0],
                                 pes.qhg_zero[k][0]]
                if options.cosmo_int:
                    zero_vals.append(pes.solv_qhg_abs[k][0])
                if pes.boltz:
                    e_sum, h_sum, g_sum, qhg_sum = 0.0, 0.0, 0.0, 0.0
                    sels = []
                    for l, e_abs in enumerate(pes.e_abs[k]):
                        if options.QH:
                            species = [pes.spc_abs[k][l], pes.e_abs[k][l], pes.zpe_abs[k][l], pes.h_abs[k][l],
                                       pes.qh_abs[k][l], temp * pes.s_abs[k][l], temp * pes.qs_abs[k][l],
                                       pes.g_abs[k][l], pes.qhg_abs[k][l]]
                        else:
                            species = [pes.spc_abs[k][l], pes.e_abs[k][l], pes.zpe_abs[k][l], pes.h_abs[k][l],
                                       temp * pes.s_abs[k][l], temp * pes.qs_abs[k][l], pes.g_abs[k][l],
                                       pes.qhg_abs[k][l]]
                        relative = [species[x] - zero_vals[x] for x in range(len(zero_vals))]
                        e_sum += math.exp(-relative[1] * thermo.J_TO_AU / thermo.GAS_CONSTANT / temp)
                        h_sum += math.exp(-relative[3] * thermo.J_TO_AU / thermo.GAS_CONSTANT / temp)
                        g_sum += math.exp(-relative[7] * thermo.J_TO_AU / thermo.GAS_CONSTANT / temp)
                        qhg_sum += math.exp(-relative[8] * thermo.J_TO_AU / thermo.GAS_CONSTANT / temp)
                if options.spc is False:
                    log.write("\n   " + '{:<40}'.format("RXN: " + path + " (" + pes.units + ")  at T: " + str(temp)))
                    if options.QH and options.cosmo_int:
                        log.write('{:>13} {:>10} {:>13} {:>13} {:>10} {:>10} {:>13} {:>13} '
                                  '{:>13}'.format(" DE", "DZPE", "DH", "qh-DH", "T.DS", "T.qh-DS", "DG(T)",
                                                  "qh-DG(T)", 'Solv-qh-G(T)'), thermodata=True)
                    elif options.QH:
                        log.write('{:>13} {:>10} {:>13} {:>13} {:>10} {:>10} {:>13} '
                                  '{:>13}'.format(" DE", "DZPE", "DH", "qh-DH", "T.DS", "T.qh-DS", "DG(T)",
                                                  "qh-DG(T)"), thermodata=True)
                    elif options.cosmo_int:
                        log.write('{:>13} {:>10} {:>13} {:>13} {:>10} {:>10} {:>13} '
                                  '{:>13}'.format(" DE", "DZPE", "DH", "T.DS", "T.qh-DS", "DG(T)", "qh-DG(T)",
                                                  'Solv-qh-G(T)'), thermodata=True)
                    else:
                        log.write('{:>13} {:>10} {:>13} {:>10} {:>10} {:>13} '
                                  '{:>13}'.format(" DE", "DZPE", "DH", "T.DS", "T.qh-DS", "DG(T)", "qh-DG(T)"),
                                  thermodata=True)
                else:
                    log.write("\n   " + '{:<40}'.format("RXN: " + path + " (" + pes.units + ")  at T: " +
                                                        str(temp)))
                    if options.QH and options.cosmo_int:
                        log.write('{:>13} {:>13} {:>10} {:>13} {:>13} {:>10} {:>10} {:>14} {:>14} {:>14}'.format(
                            " DE_SPC", "DE", "DZPE", "DH_SPC", "qh-DH_SPC", "T.DS", "T.qh-DS", "DG(T)_SPC",
                            "qh-DG(T)_SPC", 'Solv-qh-G(T)_SPC'), thermodata=True)
                    elif options.QH:
                        log.write('{:>13} {:>13} {:>10} {:>13} {:>13} {:>10} {:>10} {:>14} '
                                  '{:>14}'.format(" DE_SPC", "DE", "DZPE", "DH_SPC", "qh-DH_SPC", "T.DS",
                                                  "T.qh-DS", "DG(T)_SPC", "qh-DG(T)_SPC"), thermodata=True)
                    elif options.cosmo_int:
                        log.write('{:>13} {:>13} {:>10} {:>13} {:>13} {:>10} {:>10} {:>14} '
                                  '{:>14}'.format(" DE_SPC", "DE", "DZPE", "DH_SPC", "T.DS", "T.qh-DS",
                                                  "DG(T)_SPC", "qh-DG(T)_SPC", 'Solv-qh-G(T)_SPC'),
                                  thermodata=True)
                    else:
                        log.write('{:>13} {:>13} {:>10} {:>13} {:>10} {:>10} {:>14} '
                                  '{:>14}'.format(" DE_SPC", "DE", "DZPE", "DH_SPC", "T.DS", "T.qh-DS",
                                                  "DG(T)_SPC", "qh-DG(T)_SPC"), thermodata=True)
                log.write("\n" + stars)

                for l, e_abs in enumerate(pes.e_abs[k]):
                    if options.QH:
                        species = [pes.spc_abs[k][l], pes.e_abs[k][l], pes.zpe_abs[k][l], pes.h_abs[k][l],
                                   pes.qh_abs[k][l], temp * pes.s_abs[k][l], temp * pes.qs_abs[k][l],
                                   pes.g_abs[k][l], pes.qhg_abs[k][l]]
                    else:
                        species = [pes.spc_abs[k][l], pes.e_abs[k][l], pes.zpe_abs[k][l], pes.h_abs[k][l],
                                   temp * pes.s_abs[k][l], temp * pes.qs_abs[k][l], pes.g_abs[k][l],
                                   pes.qhg_abs[k][l]]
                    if options.cosmo_int:
                        species.append(pes.solv_qhg_abs[k][l])
                    relative = [species[x] - zero_vals[x] for x in range(len(zero_vals))]
                    if pes.units == 'kJ/mol':
                        formatted_list = [thermo.J_TO_AU / 1000.0 * x for x in relative]
                    else:
                        formatted_list = [thermo.KCAL_TO_AU * x for x in relative]  # Defaults to kcal/mol
                    log.write("\no  ")
                    if options.spc is False:
                        formatted_list = formatted_list[1:]
                        format_1 = '{:<39} {:13.1f} {:10.1f} {:13.1f} {:13.1f} {:10.1f} {:10.1f} {:13.1f} ' \
                                   '{:13.1f} {:13.1f}'
                        format_2 = '{:<39} {:13.2f} {:10.2f} {:13.2f} {:13.2f} {:10.2f} {:10.2f} {:13.2f} ' \
                                   '{:13.2f} {:13.2f}'
                        if options.QH and options.cosmo_int:
                            if pes.dec == 1:
                                log.write(format_1.format(pes.species[k][l], *formatted_list), thermodata=True)
                            if pes.dec == 2:
                                log.write(format_2.format(pes.species[k][l], *formatted_list), thermodata=True)
                        elif options.QH or options.cosmo_int:
                            if pes.dec == 1:
                                log.write(format_1.format(pes.species[k][l], *formatted_list), thermodata=True)
                            if pes.dec == 2:
                                log.write(format_2.format(pes.species[k][l], *formatted_list), thermodata=True)
                        else:
                            if pes.dec == 1:
                                log.write(format_1.format(pes.species[k][l], *formatted_list), thermodata=True)
                            if pes.dec == 2:
                                log.write(format_2.format(pes.species[k][l], *formatted_list), thermodata=True)
                    else:
                        if options.QH and options.cosmo_int:
                            if pes.dec == 1:
                                log.write('{:<39} {:13.1f} {:13.1f} {:10.1f} {:13.1f} {:13.1f} {:10.1f} {:10.1f} '
                                          '{:13.1f} {:13.1f} {:13.1f}'.format(pes.species[k][l], *formatted_list),
                                          thermodata=True)
                            if pes.dec == 2:
                                log.write('{:<39} {:13.1f} {:13.2f} {:10.2f} {:13.2f} {:13.2f} {:10.2f} {:10.2f} {:13.2f} {:13.2f} {:13.2f}'.format(
                                        pes.species[k][l], *formatted_list), thermodata=True)
                        elif options.QH or options.cosmo_int:
                            if pes.dec == 1:
                                log.write('{:<39} {:13.1f} {:13.1f} {:10.1f} {:13.1f} {:13.1f} {:10.1f} {:10.1f} {:13.1f} {:13.1f}'.format(
                                        pes.species[k][l], *formatted_list), thermodata=True)
                            if pes.dec == 2:
                                log.write('{:<39} {:13.1f} {:13.2f} {:10.2f} {:13.2f} {:13.2f} {:10.2f} {:10.2f} {:13.2f} {:13.2f}'.format(
                                        pes.species[k][l], *formatted_list), thermodata=True)
                        else:
                            if pes.dec == 1:
                                log.write('{:<39} {:13.1f} {:13.1f} {:10.1f} {:13.1f} {:10.1f} {:10.1f} {:13.1f} {:13.1f}'.format(
                                        pes.species[k][l], *formatted_list), thermodata=True)
                            if pes.dec == 2:
                                log.write('{:<39} {:13.2f} {:13.2f} {:10.2f} {:13.2f} {:10.2f} {:10.2f} {:13.2f} {:13.2f}'.format(
                                        pes.species[k][l], *formatted_list), thermodata=True)
                    if pes.boltz:
                        boltz = [math.exp(-relative[1] * thermo.J_TO_AU / thermo.GAS_CONSTANT / options.temperature) / e_sum,
                                 math.exp(-relative[3] * thermo.J_TO_AU / thermo.GAS_CONSTANT / options.temperature) / h_sum,
                                 math.exp(-relative[6] * thermo.J_TO_AU / thermo.GAS_CONSTANT / options.temperature) / g_sum,
                                 math.exp(-relative[7] * thermo.J_TO_AU / thermo.GAS_CONSTANT / options.temperature) / qhg_sum]
                        selectivity = [boltz[x] * 100.0 for x in range(len(boltz))]
                        log.write("\n  " + '{:<39} {:13.2f}%{:24.2f}%{:35.2f}%{:13.2f}%'.format('', *selectivity))
                        sels.append(selectivity)
                    formatted_list = [round(formatted_list[x], 6) for x in range(len(formatted_list))]
                if pes.boltz == 'ee' and len(sels) == 2:
                    ee = [sels[0][x] - sels[1][x] for x in range(len(sels[0]))]
                    if options.spc is False:
                        log.write("\n" + stars + "\n   " + '{:<39} {:13.1f}%{:24.1f}%{:35.1f}%{:13.1f}%'.format('ee (%)',
                                                                                                          *ee))
                    else:
                        log.write("\n" + stars + "\n   " + '{:<39} {:27.1f} {:24.1f} {:35.1f} {:13.1f} '.format('ee (%)',
                                                                                                          *ee))
                log.write("\n" + stars + "\n")
            j += 1
    else:
        if options.cosmo:
            pes = get_pes(thermo_data, options, log, cosmo=True)
        else:
            pes = get_pes(options.pes, thermo_data, log, options.temperature, options.gconf, options.QH)

        # Output the relative energy data
        for i, path in enumerate(pes.path):
            if options.QH:
                zero_vals = [pes.spc_zero[i][0], pes.e_zero[i][0], pes.zpe_zero[i][0], pes.h_zero[i][0],
                             pes.qh_zero[i][0], options.temperature * pes.ts_zero[i][0],
                             options.temperature * pes.qhts_zero[i][0], pes.g_zero[i][0], pes.qhg_zero[i][0]]
            else:
                zero_vals = [pes.spc_zero[i][0], pes.e_zero[i][0], pes.zpe_zero[i][0], pes.h_zero[i][0],
                             options.temperature * pes.ts_zero[i][0], options.temperature * pes.qhts_zero[i][0],
                             pes.g_zero[i][0], pes.qhg_zero[i][0]]
            if options.cosmo:
                zero_vals.append(pes.solv_qhg_zero[i][0])
            if pes.boltz:
                e_sum, h_sum, g_sum, qhg_sum, solv_qhg_sum = 0.0, 0.0, 0.0, 0.0, 0.0
                sels = []
                for j, e_abs in enumerate(pes.e_abs[i]):
                    if options.QH:
                        species = [pes.spc_abs[i][j], pes.e_abs[i][j], pes.zpe_abs[i][j], pes.h_abs[i][j],
                                   pes.qh_abs[i][j], options.temperature * pes.s_abs[i][j],
                                   options.temperature * pes.qs_abs[i][j], pes.g_abs[i][j], pes.qhg_abs[i][j]]
                    else:
                        species = [pes.spc_abs[i][j], pes.e_abs[i][j], pes.zpe_abs[i][j], pes.h_abs[i][j],
                                   options.temperature * pes.s_abs[i][j], options.temperature * pes.qs_abs[i][j],
                                   pes.g_abs[i][j], pes.qhg_abs[i][j]]
                    if options.cosmo:
                        species.append(pes.solv_qhg_abs[i][j])

                    relative = [species[x] - zero_vals[x] for x in range(len(zero_vals))]
                    e_sum += math.exp(-relative[1] * thermo.J_TO_AU / thermo.GAS_CONSTANT / options.temperature)
                    h_sum += math.exp(-relative[3] * thermo.J_TO_AU / thermo.GAS_CONSTANT / options.temperature)
                    g_sum += math.exp(-relative[7] * thermo.J_TO_AU / thermo.GAS_CONSTANT / options.temperature)
                    qhg_sum += math.exp(-relative[8] * thermo.J_TO_AU / thermo.GAS_CONSTANT / options.temperature)
                    solv_qhg_sum += math.exp(-relative[9] * thermo.J_TO_AU / thermo.GAS_CONSTANT / options.temperature)

            if show != False:
                if options.spc is False:
                    log.write("\n   " + '{:<40}'.format("RXN: " + path + " (" + pes.units + ") ", ))
                    if options.QH and options.cosmo:
                        log.write('{:>13} {:>10} {:>13} {:>13} {:>10} {:>10} {:>13} {:>13} '
                                  '{:>13}'.format(" DE", "DZPE", "DH", "qh-DH", "T.DS", "T.qh-DS", "DG(T)", "qh-DG(T)",
                                                  'Solv-qh-G(T)'), thermodata=True)
                    elif options.QH:
                        log.write('{:>13} {:>10} {:>13} {:>13} {:>10} {:>10} {:>13} '
                                  '{:>13}'.format(" DE", "DZPE", "DH", "qh-DH", "T.DS", "T.qh-DS", "DG(T)", "qh-DG(T)"),
                                  thermodata=True)
                    elif options.cosmo:
                        log.write('{:>13} {:>10} {:>13} {:>13} {:>10} {:>10} {:>13} '
                                  '{:>13}'.format(" DE", "DZPE", "DH", "T.DS", "T.qh-DS", "DG(T)", "qh-DG(T)",
                                                  'Solv-qh-G(T)'), thermodata=True)
                    else:
                        log.write('{:>13} {:>10} {:>13} {:>10} {:>10} {:>13} '
                                  '{:>13}'.format(" DE", "DZPE", "DH", "T.DS", "T.qh-DS", "DG(T)", "qh-DG(T)"),
                                  thermodata=True)
                else:
                    log.write("\n   " + '{:<40}'.format("RXN: " + path + " (" + pes.units + ") ", ))
                    if options.QH and options.cosmo:
                        log.write('{:>13} {:>13} {:>10} {:>13} {:>13} {:>10} {:>10} {:>14} {:>14} '
                                  '{:>14}'.format(" DE_SPC", "DE", "DZPE", "DH_SPC", "qh-DH_SPC", "T.DS", "T.qh-DS",
                                                  "DG(T)_SPC", "qh-DG(T)_SPC", 'Solv-qh-G(T)_SPC'), thermodata=True)
                    elif options.QH:
                        log.write('{:>13} {:>13} {:>10} {:>13} {:>13} {:>10} {:>10} {:>14} '
                                  '{:>14}'.format(" DE_SPC", "DE", "DZPE", "DH_SPC", "qh-DH_SPC", "T.DS", "T.qh-DS",
                                                  "DG(T)_SPC", "qh-DG(T)_SPC"), thermodata=True)
                    elif options.cosmo:
                        log.write('{:>13} {:>13} {:>10} {:>13} {:>13} {:>10} {:>10} {:>14} '
                                  '{:>14}'.format(" DE_SPC", "DE", "DZPE", "DH_SPC", "T.DS", "T.qh-DS",
                                                  "DG(T)_SPC", "qh-DG(T)_SPC", 'Solv-qh-G(T)_SPC'), thermodata=True)
                    else:
                        log.write('{:>13} {:>13} {:>10} {:>13} {:>10} {:>10} {:>14} '
                                  '{:>14}'.format(" DE_SPC", "DE", "DZPE", "DH_SPC", "T.DS", "T.qh-DS", "DG(T)_SPC",
                                                  "qh-DG(T)_SPC"), thermodata=True)

                log.write("\n" + stars)

            for j, e_abs in enumerate(pes.e_abs[i]):
                if options.QH:
                    species = [pes.spc_abs[i][j], pes.e_abs[i][j], pes.zpe_abs[i][j], pes.h_abs[i][j],
                               pes.qh_abs[i][j], options.temperature * pes.s_abs[i][j],
                               options.temperature * pes.qs_abs[i][j], pes.g_abs[i][j], pes.qhg_abs[i][j]]
                else:
                    species = [pes.spc_abs[i][j], pes.e_abs[i][j], pes.zpe_abs[i][j], pes.h_abs[i][j],
                               options.temperature * pes.s_abs[i][j], options.temperature * pes.qs_abs[i][j],
                               pes.g_abs[i][j], pes.qhg_abs[i][j]]
                if options.cosmo:
                    species.append(pes.solv_qhg_abs[i][j])
                relative = [species[x] - zero_vals[x] for x in range(len(zero_vals))]
                if pes.units == 'kJ/mol':
                    formatted_list = [thermo.J_TO_AU / 1000.0 * x for x in relative]
                else:
                    formatted_list = [thermo.KCAL_TO_AU * x for x in relative]  # Defaults to kcal/mol
                if show != False: log.write("\no  ")
                if options.spc is False:
                    formatted_list = formatted_list[1:]
                    if show != False:
                        if options.QH and options.cosmo:
                            if pes.dec == 1:
                                log.write('{:<39} {:13.1f} {:10.1f} {:13.1f} {:13.1f} {:10.1f} {:10.1f} {:13.1f} '
                                          '{:13.1f} {:13.1f}'.format(pes.species[i][j], *formatted_list),
                                          thermodata=True)
                            if pes.dec == 2:
                                log.write('{:<39} {:13.2f} {:10.2f} {:13.2f} {:13.2f} {:10.2f} {:10.2f} {:13.2f} '
                                          '{:13.2f} {:13.2f}'.format(pes.species[i][j], *formatted_list),
                                          thermodata=True)
                        elif options.QH or options.cosmo:
                            if pes.dec == 1:
                                log.write('{:<39} {:13.1f} {:10.1f} {:13.1f} {:13.1f} {:10.1f} {:10.1f} {:13.1f} '
                                          '{:13.1f}'.format(pes.species[i][j], *formatted_list), thermodata=True)
                            if pes.dec == 2:
                                log.write('{:<39} {:13.2f} {:10.2f} {:13.2f} {:13.2f} {:10.2f} {:10.2f} {:13.2f} '
                                          '{:13.2f}'.format(pes.species[i][j], *formatted_list), thermodata=True)
                        else:
                            if pes.dec == 1:
                                log.write('{:<39} {:13.1f} {:10.1f} {:13.1f} {:10.1f} {:10.1f} {:13.1f} '
                                          '{:13.1f}'.format(pes.species[i][j], *formatted_list), thermodata=True)
                            if pes.dec == 2:
                                log.write('{:<39} {:13.2f} {:10.2f} {:13.2f} {:10.2f} {:10.2f} {:13.2f} '
                                          '{:13.2f}'.format(pes.species[i][j], *formatted_list), thermodata=True)
                else:
                    if show != False:
                        if options.QH and options.cosmo:
                            if pes.dec == 1:
                                log.write('{:<39} {:13.1f} {:13.1f} {:10.1f} {:13.1f} {:13.1f} {:10.1f} {:10.1f} '
                                          '{:13.1f} {:13.1f} {:13.1f}'.format(pes.species[i][j], *formatted_list),
                                          thermodata=True)
                            if pes.dec == 2:
                                log.write('{:<39} {:13.1f} {:13.2f} {:10.2f} {:13.2f} {:13.2f} {:10.2f} {:10.2f} '
                                          '{:13.2f} {:13.2f} {:13.2f}'.format(pes.species[i][j], *formatted_list),
                                          thermodata=True)
                        elif options.QH or options.cosmo:
                            if pes.dec == 1:
                                log.write('{:<39} {:13.1f} {:13.1f} {:10.1f} {:13.1f} {:13.1f} {:10.1f} {:10.1f} '
                                          '{:13.1f} {:13.1f}'.format(pes.species[i][j], *formatted_list),
                                          thermodata=True)
                            if pes.dec == 2:
                                log.write('{:<39} {:13.1f} {:13.2f} {:10.2f} {:13.2f} {:13.2f} {:10.2f} {:10.2f} '
                                          '{:13.2f} {:13.2f}'.format(pes.species[i][j], *formatted_list),
                                          thermodata=True)
                        else:
                            if pes.dec == 1:
                                log.write('{:<39} {:13.1f} {:13.1f} {:10.1f} {:13.1f} {:10.1f} {:10.1f} {:13.1f} '
                                          '{:13.1f}'.format(pes.species[i][j], *formatted_list), thermodata=True)
                            if pes.dec == 2:
                                log.write('{:<39} {:13.2f} {:13.2f} {:10.2f} {:13.2f} {:10.2f} {:10.2f} {:13.2f} '
                                          '{:13.2f}'.format(pes.species[i][j], *formatted_list), thermodata=True)
                if pes.boltz:
                    boltz = [math.exp(-relative[1] * thermo.J_TO_AU / thermo.GAS_CONSTANT / options.temperature) / e_sum,
                             math.exp(-relative[3] * thermo.J_TO_AU / thermo.GAS_CONSTANT / options.temperature) / h_sum,
                             math.exp(-relative[6] * thermo.J_TO_AU / thermo.GAS_CONSTANT / options.temperature) / g_sum,
                             math.exp(-relative[7] * thermo.J_TO_AU / thermo.GAS_CONSTANT / options.temperature) / qhg_sum]
                    selectivity = [boltz[x] * 100.0 for x in range(len(boltz))]
                    log.write("\n  " + '{:<39} {:13.2f}%{:24.2f}%{:35.2f}%{:13.2f}%'.format('', *selectivity))
                    sels.append(selectivity)
                formatted_list = [round(formatted_list[x], 6) for x in range(len(formatted_list))]
                table.append(formatted_list)
            if pes.boltz == 'ee' and len(sels) == 2:
                ee = [sels[0][x] - sels[1][x] for x in range(len(sels[0]))]
                if show != False:
                    if options.spc is False:
                        log.write("\n" + stars + "\n   " + '{:<39} {:13.1f}%{:24.1f}%{:35.1f}%{:13.1f}%'.format('ee (%)', *ee))
                    else:
                        log.write("\n" + stars + "\n   " + '{:<39} {:27.1f} {:24.1f} {:35.1f} {:13.1f} '.format('ee (%)', *ee))
            if show != False: log.write("\n" + stars + "\n")

    return pes.species, table

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
