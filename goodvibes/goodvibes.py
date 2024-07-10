''' Evaluation of quasi-harmonic thermochemistry from Gaussian.
    Partion functions are evaluated from vibrational frequencies
    and rotational temperatures from the standard output.'''

#!/usr/bin/python
from __future__ import print_function, absolute_import

import os.path
import sys
import time
from argparse import ArgumentParser
import warnings

from goodvibes.io import Logger, gv_header, write_to_xyz, write_to_sdf, gv_summary, repair_gauss_outputs, pes_tabulate
from goodvibes.io import SUPPORTED_EXTENSIONS, load_filelist, get_cc_packages, get_cc_species, get_levels_of_theory, gv_tabulate, read_pes_yaml, pes_summary
from goodvibes.thermo import QrrhoThermo
from goodvibes.utils import detect_symm, get_vib_scaling, get_cpu_time, check_dup, sort_conformers, get_boltz_facs, get_selectivity
from goodvibes.pes import create_pes
from goodvibes.version import __version__

warnings.filterwarnings("ignore") # this is to suppress warnings from cclib/scipy

LOGDATE = time.strftime("%Y%m%d_%H%M")
xyzfile = 'GoodVibes_'+LOGDATE+'.xyz'
sdffile = 'GoodVibes_'+LOGDATE+'.sdf'
csvfile = 'GoodVibes_'+LOGDATE+'.csv'
logfile = 'GoodVibes_'+LOGDATE

def parse_args():
    '''user defined arguments: use -h to list all possible arguments and default values'''
    parser = ArgumentParser()
    parser.add_argument("-q", dest="Q", action="store_true", default=False,
        help="Quasi-harmonic entropy & enthalpy correction (default S=Grimme, H=Head-Gordon)")
    parser.add_argument("--qs", dest="QS", default="grimme", type=str.lower, metavar="QS",
        choices=('grimme', 'truhlar'),
        help="Quasi-harmonic entropy correction (Grimme or Truhlar) (default Grimme)",)
    parser.add_argument("--qh", dest="QH", action="store_true", default=False,
        help="Type of quasi-harmonic enthalpy correction (Head-Gordon)")
    parser.add_argument("-f", dest="freq_cutoff", default=100, type=float, metavar="FREQ_CUTOFF",
        help="Cut-off frequency for entropy and enthalpy (default = 100 cm-1)",)
    parser.add_argument("--fs", dest="S_freq_cutoff", default=100.0, type=float,
        metavar="S_FREQ_CUTOFF", help="Cut-off frequency for entropy (default = 100 cm-1)")
    parser.add_argument("--fh", dest="H_freq_cutoff", default=100.0, type=float,
        metavar="H_FREQ_CUTOFF", help="Cut-off frequency for enthalpy (default = 100 cm-1)")
    parser.add_argument("-t", dest="temperature", default=298.15, type=float, metavar="TEMP",
        help="Temperature (K) (default 298.15)")
    parser.add_argument("-c", dest="conc", default=False, type=float, metavar="CONC",
        help="Concentration (mol/l) (default 1 atm)")
    parser.add_argument("--ti", dest="temperature_interval", default=False, metavar="TI",
         help="Initial temp, final temp, step size (K)")
    parser.add_argument("-v", dest="freq_scale_factor", default=False, type=float,
        metavar="SCALE_FACTOR",
         help="Frequency scaling factor. If not set, try to find a suitable value"
        " in database. If not found, use 1.0")
    parser.add_argument("--vmm", dest="mm_freq_scale_factor", default=False, type=float,
        metavar="MM_SCALE_FACTOR", help="Additional frequency scaling factor"
        " used in ONIOM calculations")
    parser.add_argument("--spc", dest="spc", type=str, default=False, metavar="SPC",
        help="Indicates single point corrections (default False)")
    parser.add_argument("--boltz", dest="boltz", action="store_true", default=False,
        help="Show Boltzmann factors")
    parser.add_argument("--cpu", dest="cputime", action="store_true", default=False,
        help="Total CPU time")
    parser.add_argument("--xyz", dest="xyz", action="store_true", default=False,
        help="Write Cartesians to a .xyz file (default False)")
    parser.add_argument("--sdf", dest="sdf", action="store_true", default=False,
        help="Write Cartesians to a .sdf file (default False)")
    parser.add_argument("--csv", dest="csv", action="store_true", default=False,
        help="Write .csv output file format")
    parser.add_argument("--imag", dest="imag_freq", action="store_true", default=False,
        help="Print imaginary frequencies (default False)")
    parser.add_argument("--invertifreq", dest="invert", nargs='?', const=True, default=False,
        help="Make low lying imaginary frequencies positive (cutoff > -50.0 cm-1)")
    parser.add_argument("--dup", dest="duplicate", action="store_true", default=False,
        help="Remove possible duplicates from thermochemical analysis")
    parser.add_argument("--sort", dest="sort", action="store_true", default=False,
        help="Sort results by energy")
    parser.add_argument("--cosmo", dest="cosmo", default=False, metavar="COSMO-RS",
        help="Filename of a COSMO-RS .tab output file")
    parser.add_argument("--cosmo_int", dest="cosmo_int", default=False, metavar="COSMO-RS",
        help="COSMO-RS .tab output filename along with a temperature range (K): "
        "file.tab,'Initial_T, Final_T'")
    parser.add_argument("--output", dest="output", default="output", metavar="OUTPUT",
        help="Change the output file name to GoodVibes_\"output\".dat")
    parser.add_argument("--pes", dest="pes", default=False, metavar="PES",
        help="Tabulate relative values")
    parser.add_argument("--nogconf", dest="gconf", action="store_false", default=True,
        help="Turn off ensemble Gibbs energies which include Sconf (default calculate Gconf)")
    parser.add_argument("--ee", dest="ee", default=False, type=str,
        help="Tabulate selectivity values (excess, ratio) from a mixture, "
        "provide pattern for two types such as *_R*,*_S*")
    parser.add_argument("--check", dest="check", action="store_true", default=False,
        help="Automated checking for consistency")
    parser.add_argument("--media", dest="media", default=False, metavar="MEDIA",
        help="Entropy correction for standard concentration of solvents")
    parser.add_argument("--custom_ext", type=str, default='',
        help="List of additional file extensions to support, beyond .log "
        " or .out, use separated by commas (ie, '.qfi, .gaussian').")
    parser.add_argument("--dp", dest='decimalplaces', type=int, default=5,
        help="Decimal places used in printing (default = 5). ")
    parser.add_argument("--graph", dest='graph', default=False, metavar="GRAPH",
        help="Graph a reaction profile based on free energies calculated. ")
    parser.add_argument("--nosymm", dest='nosymm', action="store_true", default=False,
        help="Turn off symmetry detection.")
    parser.add_argument("--bav", dest='inertia', default="global",type=str,
        choices=['global','conf'],
        help="Choice of how the moment of inertia is computed. Options = 'global' or 'conf'."
        "'global' will use the same value for all input molecules of 10*10-44,"
        "'conf' will use parsed rotational constants from each Gaussian output file.")

    (options, args) = parser.parse_known_args()

    # If requested, turn on head-gordon enthalpy correction
    if options.Q:
        options.QH = True

    # Default value for inverting spurious imaginary frequencies
    if options.invert:
        options.invert == -50.0
    elif options.invert > 0:
        options.invert = -1 * options.invert

    if options.freq_cutoff != 100.0:
        options.S_freq_cutoff = options.freq_cutoff
        options.H_freq_cutoff = options.freq_cutoff

    return options

def main():
    '''Main function for GoodVibes. Called when the script is run from the command line.'''
    # Get command line inputs.
    options = parse_args()

    # If user has specified different file extensions
    if options.custom_ext or os.environ.get('GOODVIBES_CUSTOM_EXT', ''):
        custom_extensions = options.custom_ext.split(',') + os.environ.get('GOODVIBES_CUSTOM_EXT', '').split(',')
        for ext in custom_extensions:
            SUPPORTED_EXTENSIONS.add(ext.strip())

    # Start a log for the results
    log = Logger(logfile, options.output)

    # Get the filenames from the command line prompt
    files, sp_files, user_args = load_filelist(sys.argv[1:], options.spc)
    log.write('\n' + user_args + '\n\n')

    # fix issues with Gaussian output files
    repair_gauss_outputs(files)
        
    # Global summary of user defined options and methods to be used
    gv_header(log, files)

    # Do some file parsing
    package_list = get_cc_packages(log, files)
    species_list = get_cc_species(log, files, package_list)

    if len(species_list) == 0:
        log.write('\nx  No species found. Exiting.\n\n')
        sys.exit()

    if not options.nosymm: # auto-detect point group symmetry
        detect_symm(species_list)

    model_chemistry = get_levels_of_theory(log, species_list)  # methods used and vibrational scaling factors
    options.freq_scale_factor = get_vib_scaling(log, model_chemistry, options.freq_scale_factor)

    # Single point corrections if requested
    if options.spc is not False:
        sp_package_list = get_cc_packages(log, sp_files)
        sp_list = get_cc_species(log, sp_files, sp_package_list)
    else:
        sp_list = []

    # check we have identical numbers of species and single point corrections
    if options.spc is not False and len(species_list) != len(sp_list):
        log.write('\n\nx  Number of species and single point corrections do not match!\n\n')
        sys.exit()

    # Generate thermochemical data
    thermo_data = []
    for i, species in enumerate(species_list):
        # attempt to match a single point energy output otherwise return False
        spc = next((x for x in sp_list if x.name == species.name+'_'+options.spc), False)

        try:
            thermo = QrrhoThermo(species, qs=options.QS, qh=options.QH, s_freq_cutoff=options.S_freq_cutoff,
            h_freq_cutoff=options.H_freq_cutoff, temperature=options.temperature, conc=options.conc,
            freq_scale_factor=options.freq_scale_factor, spc=spc, invert=options.invert,
            cosmo=options.cosmo, mm_freq_scale_factor=options.mm_freq_scale_factor,inertia=options.inertia)
            thermo_data.append(thermo)
        except:
            log.write(f'x  Failed to generate information for {species.name}\n')

    if len(thermo_data) == 0:
        log.write('\nx  No thermochemical data generated. Exiting.\n\n')
        sys.exit()

    if options.duplicate is not False: # Filter duplicate species
        check_dup(thermo_data)

    if options.sort is not False: # Sort conformers by energy
        sort_conformers(thermo_data)

    if options.boltz is not False: # Generate Boltzmann factors
        if options.spc: get_boltz_facs(thermo_data, temperature = options.temperature, use_gibbs=True, spc=True)
        else: get_boltz_facs(thermo_data, temperature = options.temperature, use_gibbs=True)
    
    # Standard mode: tabulate thermochemistry for file(s) at a single temperature and concentration
    if options.temperature_interval is False:
        gv_df = gv_tabulate(thermo_data) # convert thermochemical data to a pandas dataframe
        formatted_df = gv_summary(gv_df, spc=options.spc, nosymm=options.nosymm, imag=options.imag_freq, boltz=options.boltz) # nice print
        log.write_df(formatted_df, options.decimalplaces)

    if options.csv is not False: # Write the data to a csv file
        formatted_df.to_csv(csvfile, index=False)

    if options.ee is not False and options.boltz is not False: # Calculate selectivity
        excess, ratio, ddg = get_selectivity(options.ee, thermo_data, options.temperature)
        log.write(f'\n\n!  SELECTIVITY: {options.ee}: {ratio} | an excess of {excess:.2f} | effective DDG: {ddg:.2f} kcal/mol\n')

    if options.cputime is not False: # Total CPU time for all calculations (including those that were filtered)
        total_cpu = get_cpu_time(species_list + sp_list)
        log.write(f'\n   Total CPU time for all calculations: {total_cpu}\n')

    if options.pes: # Compute relative values from a yaml file defining the pathways
        pes_format = read_pes_yaml(options.pes, thermo_data) # gets the user arguments from the yaml file
        pes_list =  [create_pes(pes_format, thermo_data, i, spc=options.spc, temperature=options.temperature, gconf=True, QH=False, cosmo=None, cosmo_int=None) for i, rxn in enumerate(pes_format.rxns)]
        
        pes_dfs = [pes_tabulate(pes) for pes in pes_list] # convert to dataframes and print
        for pes_df in pes_dfs:
            df = pes_summary(pes_df, pes_format, spc = options.spc)
            log.write_df(df, pes_format.decimalplaces)

    if options.xyz: # Create an xyz file for the structures
        write_to_xyz(log, thermo_data, xyzfile)
        log.write(f'\n   Writing structures to XYZ file: {xyzfile}\n')

    if options.sdf: # Create an sdf file for the structures
        write_to_sdf(log, thermo_data, sdffile)
        log.write(f'\n   Writing structures to SDF file: {sdffile}\n')

   # Close the log
    log.finalize()

if __name__ == "__main__":
    main()
