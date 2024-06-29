#!/usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import

"""####################################################################
#                             GoodVibes                               #
#  Evaluation of quasi-harmonic thermochemistry from Gaussian.        #
#  Partion functions are evaluated from vibrational frequencies       #
#  and rotational temperatures from the standard output.              #
###########      Last modified:  June 26 , 2024            ############
####################################################################"""

import os.path, sys
from glob import glob
from argparse import ArgumentParser

from goodvibes import vib_scale_factors
from goodvibes import pes
from goodvibes import io
from goodvibes import thermo
from goodvibes import utils
from goodvibes import media
    
# VERSION NUMBER
__version__ = "4.0"

# most compchem outputs look like this:
SUPPORTED_EXTENSIONS = set(('.out', '.log'))

# user defined arguments on the commandline: use -h to list all possible arguments and default values
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-q", dest="Q", action="store_true", default=False,
                        help="Quasi-harmonic entropy correction and enthalpy correction applied (default S=Grimme, "
                             "H=Head-Gordon)")
    parser.add_argument("--qs", dest="QS", default="grimme", type=str.lower, metavar="QS",
                        choices=('grimme', 'truhlar'),
                        help="Type of quasi-harmonic entropy correction (Grimme or Truhlar) (default Grimme)", )
    parser.add_argument("--qh", dest="QH", action="store_true", default=False,
                        help="Type of quasi-harmonic enthalpy correction (Head-Gordon)")
    parser.add_argument("-f", dest="freq_cutoff", default=100, type=float, metavar="FREQ_CUTOFF",
                        help="Cut-off frequency for both entropy and enthalpy (wavenumbers) (default = 100)", )
    parser.add_argument("--fs", dest="S_freq_cutoff", default=100.0, type=float, metavar="S_FREQ_CUTOFF",
                        help="Cut-off frequency for entropy (wavenumbers) (default = 100)")
    parser.add_argument("--fh", dest="H_freq_cutoff", default=100.0, type=float, metavar="H_FREQ_CUTOFF",
                        help="Cut-off frequency for enthalpy (wavenumbers) (default = 100)")
    parser.add_argument("-t", dest="temperature", default=298.15, type=float, metavar="TEMP",
                        help="Temperature (K) (default 298.15)")
    parser.add_argument("-c", dest="conc", default=False, type=float, metavar="CONC",
                        help="Concentration (mol/l) (default 1 atm)")
    parser.add_argument("--ti", dest="temperature_interval", default=False, metavar="TI",
                        help="Initial temp, final temp, step size (K)")
    parser.add_argument("-v", dest="freq_scale_factor", default=False, type=float, metavar="SCALE_FACTOR",
                        help="Frequency scaling factor. If not set, try to find a suitable value in database. "
                             "If not found, use 1.0")
    parser.add_argument("--vmm", dest="mm_freq_scale_factor", default=False, type=float, metavar="MM_SCALE_FACTOR",
                        help="Additional frequency scaling factor used in ONIOM calculations")
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
                        help="Make low lying imaginary frequencies positive (cutoff > -50.0 wavenumbers)")
    parser.add_argument("--freespace", dest="freespace", default="none", type=str, metavar="FREESPACE",
                        help="Solvent (H2O, toluene, DMF, AcOH, chloroform) (default none)")
    parser.add_argument("--dup", dest="duplicate", action="store_true", default=False,
                        help="Remove possible duplicates from thermochemical analysis")
    parser.add_argument("--cosmo", dest="cosmo", default=False, metavar="COSMO-RS",
                        help="Filename of a COSMO-RS .tab output file")
    parser.add_argument("--cosmo_int", dest="cosmo_int", default=False, metavar="COSMO-RS",
                        help="Filename of a COSMO-RS .tab output file along with a temperature range (K): "
                             "file.tab,'Initial_T, Final_T'")
    parser.add_argument("--output", dest="output", default="output", metavar="OUTPUT",
                        help="Change the default name of the output file to GoodVibes_\"output\".dat")
    parser.add_argument("--pes", dest="pes", default=False, metavar="PES",
                        help="Tabulate relative values")
    parser.add_argument("--nogconf", dest="gconf", action="store_false", default=True,
                        help="Calculate a free-energy correction related to multi-configurational space (default "
                             "calculate Gconf)")
    parser.add_argument("--ee", dest="ee", default=False, type=str,
                        help="Tabulate selectivity values (excess, ratio) from a mixture, provide pattern for two "
                             "types such as *_R*,*_S*")
    parser.add_argument("--check", dest="check", action="store_true", default=False,
                        help="Checks if calculations were done with the same program, level of theory and solvent, "
                             "as well as detects potential duplicates")
    parser.add_argument("--media", dest="media", default=False, metavar="MEDIA",
                        help="Entropy correction for standard concentration of solvents")
    parser.add_argument("--custom_ext", type=str, default='',
                        help="List of additional file extensions to support, beyond .log or .out, use separated by "
                             "commas (ie, '.qfi, .gaussian'). It can also be specified with environment variable "
                             "GOODVIBES_CUSTOM_EXT")
    parser.add_argument("--graph", dest='graph', default=False, metavar="GRAPH",
                        help="Graph a reaction profile based on free energies calculated. ")
    parser.add_argument("--ssymm", dest='ssymm', action="store_true", default=False,
                        help="Turn on the symmetry correction.")
    parser.add_argument("--bav", dest='inertia', default="global",type=str,choices=['global','conf'],
                        help="Choice of how the moment of inertia is computed. Options = 'global' or 'conf'."
                            "'global' will use the same moment of inertia for all input molecules of 10*10-44,"
                            "'conf' will compute moment of inertia from parsed rotational constants from each Gaussian output file.")
    parser.add_argument("--g4", dest="g4", action="store_true", default=False,
                        help="Use this option when using G4 calculations in Gaussian")
    parser.add_argument("--glowfreq", type=str, default='',
                        help="Specify the base name of the two files generated by the GLOWfreq program (NAME.ROVIBprop and NAME.MECPprop)")
    # Parse Arguments
    (options, args) = parser.parse_known_args()

    # If requested, turn on head-gordon enthalpy correction
    if options.Q: options.QH = True

    # Default value for inverting spurious imaginary frequencies
    if options.invert: options.invert == -50.0
    elif options.invert > 0: options.invert = -1 * options.invert
    
    return options, args

# returns file list with acceptable extensions and user-requested arguments 
def load_files(arglist, spc = False):
    files = []
    user_args = '   Requested: '
    for elem in arglist:
        try:
            if os.path.splitext(elem)[1].lower() in SUPPORTED_EXTENSIONS:  # Look for file names
                for file in glob(elem):
                    # if we don't expect single point calculations then grab everything
                    if spc is False or spc == 'link': files.append(file)
                    else:
                        # if we expect single point calculations need to check for both opt and single point separately
                        if file.find('_' + spc + ".") == -1:
                            files.append(file)
                            name, ext = os.path.splitext(file)
                            if not (os.path.exists(name + '_' + spc + '.log') or os.path.exists(
                                    name + '_' + spc + '.out')) and spc != 'link':
                                sys.exit("\nError! SPC calculation file '{}' not found! Make sure files are named with "
                                         "the convention: 'filename_spc' or specify link job.\nFor help, use option '-h'\n"
                                         "".format(name + '_' + spc))
            else:
                user_args += elem + ' '
        except IndexError:
            pass
    return files, user_args

def main():    
    # Get command line inputs. 
    options, args = parse_args()
    
    # If user has specified different file extensions
    if options.custom_ext or os.environ.get('GOODVIBES_CUSTOM_EXT', ''):
        custom_extensions = options.custom_ext.split(',') + os.environ.get('GOODVIBES_CUSTOM_EXT', '').split(',')
        for ext in custom_extensions: SUPPORTED_EXTENSIONS.add(ext.strip())

    # Start a log for the results
    log = io.Logger("Goodvibes", options.output, options.csv)
           
    # Get the filenames from the command line prompt
    files, user_args = load_files(sys.argv[1:], options.spc)
    log.write('\n' + user_args + '\n\n')

    # Global summary of user defined options and methods to be used
    # need to separate this out into updating options vs printing !
    io.gv_header(log, files, options, __version__)

    # Loop over all specified output files and compute quasi-harmonic thermochemistry
    bbe_vals = []
    for file in files:
        
        bbe = thermo.calc_bbe(file, options.QS, options.QH, options.S_freq_cutoff, options.H_freq_cutoff, options.temperature,
                       options.conc, options.freq_scale_factor, options.freespace, options.spc, options.invert,
                       cosmo=options.cosmo, ssymm=options.ssymm, mm_freq_scale_factor=options.mm_freq_scale_factor, 
                       inertia=options.inertia, g4=options.g4, glowfreq=options.glowfreq)

        # Populate bbe_vals with indivual bbe entries for each file
        bbe_vals.append(bbe)

    # Creates a new dictionary object thermo_data, which attaches the thermochemical data to each species name
    thermo_data = dict(zip(files, bbe_vals))  # The collected thermochemical data for all files
 
    # Perform checks for consistent options provided in calculation files (level of theory)
    if options.check:
        utils.check_files(log, files, thermo_data, options)

    # Standard mode: tabulate thermochemistry for file(s) at a single temperature and concentration
    if options.temperature_interval is False:
        io.gv_summary(log, files, thermo_data, options)
 
    # Create an xyz file for the structures
    if options.xyz: 
        io.write_to_xyz(log, files, thermo_data)
    
    # Create an sdf file for the structures
    if options.sdf: 
        io.write_to_sdf(log, files, thermo_data)

    # Close the log
    log.finalize()

if __name__ == "__main__":
    main()