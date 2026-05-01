# -*- coding: utf-8 -*-
"""PES driver: legacy `get_pes` class re-implemented on top of the v4.2 model.

The class signature is unchanged; everything that printed/plotted off
`get_pes` instances in v4.0/v4.1 still does. Internally we delegate
parsing and rollup to `pes_loader` + `pes_model`, then project the
resulting `PESResult` into the legacy parallel-list attribute layout
that `output.print_pes_results` and `pes.graph_reaction_profile`
already consume.

This shim ships through the v4.x line. v5.1 removes `get_pes` entirely
in favor of the structured API (item 12 / Sub-plan B in ROADMAP.md).
"""
import logging

from .pes_loader import load_pes

log = logging.getLogger('goodvibes')

# PHYSICAL CONSTANTS                                      UNITS
GAS_CONSTANT = 8.3144621                                 # J / K / mol
J_TO_AU = 4.184 * 627.509541 * 1000.0                    # UNIT CONVERSION
KCAL_TO_AU = 627.509541                                  # UNIT CONVERSION


class get_pes:
    """Back-compat surface around `pes_loader.load_pes`.

    Reads a PES definition file (legacy line-based or modern YAML),
    computes Boltzmann-weighted thermo at each pathway point with
    optional gconf correction, and exposes the same parallel-list
    attribute layout as the pre-v4.2 implementation: `path`, `species`,
    `*_abs`, `*_zero`, plus `g_qhgvals` / `g_species_qhgzero` /
    `g_rel_val` for the reaction-profile plot.
    """

    def __init__(self, file, thermo_data, temperature, gconf, QH):
        # 1. Parse + resolve via the new pipeline.
        result = load_pes(file, thermo_data, temperatures=[temperature])

        # Surface options.
        self.units = result.options.units
        self.dec = result.options.decimals
        # Legacy `boltz` flag (string like 'ee' or False) lives in format_extras.
        # Re-parsed from the original file because PESOptions discards it.
        self.boltz = _read_boltz(file)

        # 2. Initialise legacy parallel-list attributes (one slot per pathway).
        self.path = []
        self.species = []
        self.spc_zero = []; self.e_zero = []; self.zpe_zero = []
        self.h_zero = [];   self.qh_zero = []
        self.ts_zero = [];  self.qhts_zero = []
        self.g_zero = [];   self.qhg_zero = []
        self.spc_abs = [];  self.e_abs = [];  self.zpe_abs = []
        self.h_abs = [];    self.qh_abs = []
        self.s_abs = [];    self.qs_abs = []
        self.g_abs = [];    self.qhg_abs = []
        self.g_qhgvals = []
        self.g_species_qhgzero = []
        self.g_rel_val = []

        # 3. Project each pathway into the legacy layout.
        for pathway in result.pathways:
            self.path.append(pathway.name)
            zero_th = pathway.zero.thermo(temperature, gconf=gconf, QH=QH)

            # Zero values (single-element lists per legacy convention).
            self.spc_zero.append([zero_th.sp_energy if zero_th.sp_energy is not None else 0.0])
            self.e_zero.append([zero_th.scf_energy])
            self.zpe_zero.append([zero_th.zpe])
            self.h_zero.append([zero_th.enthalpy])
            self.qh_zero.append([zero_th.qh_enthalpy])
            self.ts_zero.append([zero_th.entropy])      # raw S; T applied at print
            self.qhts_zero.append([zero_th.qh_entropy])
            self.g_zero.append([zero_th.gibbs])
            self.qhg_zero.append([zero_th.qh_gibbs])

            # Per-step abs lists.
            self.species.append([])
            self.spc_abs.append([]); self.e_abs.append([]); self.zpe_abs.append([])
            self.h_abs.append([]);  self.qh_abs.append([])
            self.s_abs.append([]);  self.qs_abs.append([])
            self.g_abs.append([]);  self.qhg_abs.append([])
            self.g_qhgvals.append([])
            self.g_species_qhgzero.append([])
            self.g_rel_val.append([])

            for point in pathway.points:
                pt_th = point.thermo(temperature, gconf=gconf, QH=QH)
                self.species[-1].append(point.label)
                self.spc_abs[-1].append(
                    pt_th.sp_energy if pt_th.sp_energy is not None else 0.0
                )
                self.e_abs[-1].append(pt_th.scf_energy)
                self.zpe_abs[-1].append(pt_th.zpe)
                self.h_abs[-1].append(pt_th.enthalpy)
                self.qh_abs[-1].append(pt_th.qh_enthalpy)
                self.s_abs[-1].append(pt_th.entropy)
                self.qs_abs[-1].append(pt_th.qh_entropy)
                self.g_abs[-1].append(pt_th.gibbs)
                self.qhg_abs[-1].append(pt_th.qh_gibbs)

                # Per-conformer arrays for the reaction-profile plot.
                # g_qhgvals[step] is a list of per-species lists of conformer qh-G;
                # g_species_qhgzero[step] is the per-species Boltzmann reference;
                # g_rel_val[step] is the stoichiometric sum across species.
                step_qhgvals = []
                step_species_zero = []
                rel_val = 0.0
                for coeff, cset in point.species:
                    step_qhgvals.append(
                        [b.qh_gibbs_free_energy for b in cset.bbes]
                    )
                    if cset.is_single:
                        species_zero = cset.bbes[0].qh_gibbs_free_energy
                    else:
                        species_zero = cset.boltzmann_weighted(temperature).qh_gibbs
                    step_species_zero.append(species_zero)
                    rel_val += coeff * species_zero
                self.g_qhgvals[-1].append(step_qhgvals)
                self.g_species_qhgzero[-1].append(step_species_zero)
                self.g_rel_val[-1].append(rel_val)


def _read_boltz(path):
    """Recover the legacy `boltz:` FORMAT key (string or False).

    Lives outside `PESOptions` because it gates output behavior, not thermo
    math. Returns False if not set or the file can't be opened.
    """
    try:
        with open(path) as f:
            for line in f:
                stripped = line.strip()
                if not stripped or stripped.startswith('#'):
                    continue
                if 'boltz' not in stripped.lower():
                    continue
                # `boltz: ee` or `boltz = ee`
                for sep in (':', '='):
                    if sep in stripped:
                        key, _, value = stripped.partition(sep)
                        if key.strip().lower() == 'boltz':
                            return value.strip()
                        break
    except OSError:
        pass
    return False


def jitter(datasets, color, ax, nx, marker, edgecol='black'):
    """
    Randomly offsets and plots vertically overlapping points to reduce marker overlap on a matplotlib Axes.
    
    Parameters:
        datasets (iterable): Iterable of numeric y-values (each element plotted as a point).
        color (str or tuple): Marker color.
        ax (matplotlib.axes.Axes): Axes on which to draw the markers.
        nx (float): Center x-coordinate around which points are jittered.
        marker (str): Marker style string.
        edgecol (str, optional): Marker edge color. Defaults to 'black'.
    """
    import numpy as np
    for i, p in enumerate(datasets):
        y = [p]
        x = np.random.normal(nx, 0.015, size=len(y))
        ax.plot(x, y, alpha=0.5, markersize=7, color=color, marker=marker, markeredgecolor=edgecol,
                markeredgewidth=1, linestyle='None')

def graph_reaction_profile(graph_data, options, plt):
    """
    Render a reaction energy profile plot using quasi-harmonic Gibbs free energies.
    
    Parameters:
        graph_data (get_pes): Populated PES object providing pathway labels and thermodynamic arrays used for plotting (e.g., `path`, `qhg_abs`, `qhg_zero`, `e_abs`, `species`, `g_qhgvals`, `g_species_qhgzero`, `g_rel_val`, and `units`).
        options (object): Options container with a `graph` attribute pointing to a formatting file that controls plot appearance (ylim, colors, title, dpi, etc.). If `dpi` is set in that file, an image file named "Rxn_profile_<options.graph stem>.png" will be written.
        plt (module): The matplotlib.pyplot module (used to create and display the figure).
    
    """
    import matplotlib.path as mpath
    import matplotlib.patches as mpatches

    log.info("\n   Graphing Reaction Profile\n")
    data = {}
    # Get PES data
    for i, path in enumerate(graph_data.path):
        g_data = []
        zero_val = graph_data.qhg_zero[i][0]
        for j, e_abs in enumerate(graph_data.e_abs[i]):
            species = graph_data.qhg_abs[i][j]
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
    label_point, label_xaxis, dpi, dec, legend = False, True, None, 2, False,
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
    ax.set_ylabel(r"$G_{rel}$ (kcal / mol)")
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
    if dpi is not None:
        plt.savefig('Rxn_profile_' + options.graph.split('.')[0] + '.png', dpi=dpi)
    plt.show()
