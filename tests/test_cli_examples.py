"""
CLI integration tests derived from readme_cli_examples.

Each test runs GoodVibes via subprocess and checks that every data line
in the expected output matches the actual output (structure name + all
numerical columns).
"""

import os
import subprocess
import sys
import re

import pytest


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def example(filename):
    """Return the path to an example file, relative to ROOT_DIR."""
    return os.path.join('goodvibes', 'examples', filename)


def run_goodvibes(args):
    """Run ``python -m goodvibes <args>`` and return stdout as a string."""
    cmd = [sys.executable, '-m', 'goodvibes'] + args
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=ROOT_DIR,
    )
    assert result.returncode == 0, (
        f"goodvibes exited with code {result.returncode}\n"
        f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
    )
    return result.stdout


def parse_data_lines(text):
    """Extract thermochemistry data rows from GoodVibes output.

    Identifies lines starting with ``o`` that contain a structure name
    followed by numeric columns (at least 5 values).  Skips banner lines
    like ``o  Requested:`` or ``o  Found vibrational ...``.

    Returns a list of (name, [float, ...]) tuples.
    """
    rows = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith('o  '):
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue  # Skip malformed lines
        name = parts[1]
        values = []
        for tok in parts[2:]:
            try:
                values.append(float(tok))
            except ValueError:
                break
        # Real data rows have many numeric columns; skip banner lines
        if len(values) < 5:
            continue
        rows.append((name, values))
    return rows


def assert_data_matches(actual_text, expected_rows, precision=6):
    """Assert that each expected (name, values) row appears in actual output."""
    actual_rows = parse_data_lines(actual_text)
    actual_dict = {}
    for name, values in actual_rows:
        actual_dict.setdefault(name, []).append(values)

    for name, expected_values in expected_rows:
        assert name in actual_dict, (
            f"Structure '{name}' not found in output. "
            f"Found: {list(actual_dict.keys())}"
        )
        # Pop the first matching row for this name (supports repeated names
        # like in temperature interval output)
        actual_values = actual_dict[name].pop(0)
        if not actual_dict[name]:
            del actual_dict[name]
        assert len(actual_values) == len(expected_values), (
            f"{name}: expected {len(expected_values)} values, "
            f"got {len(actual_values)}\n"
            f"  expected: {expected_values}\n  actual:   {actual_values}"
        )
        for i, (exp, act) in enumerate(zip(expected_values, actual_values)):
            assert round(act, precision) == round(exp, precision), (
                f"{name} column {i}: expected {exp}, got {act}"
            )


def assert_line_present(text, pattern):
    """Assert that at least one line matches the regex *pattern*."""
    for line in text.splitlines():
        if re.search(pattern, line):
            return
    raise AssertionError(f"No line matching {pattern!r} found in output")


def assert_separator_lines(text, count=None):
    """Assert separator lines (rows of ``*``) are present."""
    sep_lines = [line for line in text.splitlines() if line.strip().startswith('***')]
    if count is not None:
        assert len(sep_lines) >= count, (
            f"Expected at least {count} separator lines, found {len(sep_lines)}"
        )


def assert_header(text, columns):
    """Assert the header line contains all expected column names."""
    for line in text.splitlines():
        if 'Structure' in line:
            for col in columns:
                assert col in line, f"Header missing column {col!r}: {line}"
            return
    raise AssertionError("No header line containing 'Structure' found")


# ---------------------------------------------------------------------------
# Example 1: Grimme-type quasi-harmonic correction with a cut-off of 150 cm-1
# ---------------------------------------------------------------------------
class TestExample1:
    """python -m goodvibes examples/methylaniline.out -f 150"""

    EXPECTED = [
        ('methylaniline', [-326.664901, 0.142118, -326.514489, 0.039668, 0.039465, -326.554157, -326.553954]),
    ]
    HEADER_COLS = ['E', 'ZPE', 'H', 'T.S', 'T.qh-S', 'G(T)', 'qh-G(T)']

    @pytest.fixture(scope='class')
    def output(self):
        return run_goodvibes([example('methylaniline.out'), '-f', '150'])

    def test_header(self, output):
        assert_header(output, self.HEADER_COLS)

    def test_separator_lines(self, output):
        assert_separator_lines(output, count=2)

    def test_data_lines(self, output):
        assert_data_matches(output, self.EXPECTED)


# ---------------------------------------------------------------------------
# Example 2a: SPC with link job
# ---------------------------------------------------------------------------
class TestExample2a:
    """python -m goodvibes examples/ethane_spc.out --spc link"""

    EXPECTED = [
        ('ethane_spc', [-79.858399, -79.830421, 0.074561, -79.779414, 0.027540, 0.027542, -79.806954, -79.806956]),
    ]
    HEADER_COLS = ['E_SPC', 'E', 'ZPE', 'H_SPC', 'T.S', 'T.qh-S', 'G(T)_SPC', 'qh-G(T)_SPC']

    @pytest.fixture(scope='class')
    def output(self):
        return run_goodvibes([example('ethane_spc.out'), '--spc', 'link'])

    def test_header(self, output):
        assert_header(output, self.HEADER_COLS)

    def test_separator_lines(self, output):
        assert_separator_lines(output, count=2)

    def test_data_lines(self, output):
        assert_data_matches(output, self.EXPECTED)


# ---------------------------------------------------------------------------
# Example 2b: SPC with TZ suffix
# ---------------------------------------------------------------------------
class TestExample2b:
    """python -m goodvibes examples/ethane.out --spc TZ"""

    EXPECTED = [
        ('ethane', [-79.858399, -79.830421, 0.074561, -79.779414, 0.027540, 0.027542, -79.806954, -79.806956]),
    ]
    HEADER_COLS = ['E_SPC', 'E', 'ZPE', 'H_SPC', 'T.S', 'T.qh-S', 'G(T)_SPC', 'qh-G(T)_SPC']

    @pytest.fixture(scope='class')
    def output(self):
        return run_goodvibes([example('ethane.out'), '--spc', 'TZ'])

    def test_header(self, output):
        assert_header(output, self.HEADER_COLS)

    def test_separator_lines(self, output):
        assert_separator_lines(output, count=2)

    def test_data_lines(self, output):
        assert_data_matches(output, self.EXPECTED)


# ---------------------------------------------------------------------------
# Example 3: Changed temperature (1000 K) and concentration (1 mol/l)
# ---------------------------------------------------------------------------
class TestExample3:
    """python -m goodvibes examples/methylaniline.out -t 1000 -c 1.0"""

    EXPECTED = [
        ('methylaniline', [-326.664901, 0.142118, -326.452307, 0.218212, 0.216559, -326.670519, -326.668866]),
    ]
    HEADER_COLS = ['E', 'ZPE', 'H', 'T.S', 'T.qh-S', 'G(T)', 'qh-G(T)']

    @pytest.fixture(scope='class')
    def output(self):
        return run_goodvibes([example('methylaniline.out'), '--temp', '1000', '--conc', '1.0'])

    def test_header(self, output):
        assert_header(output, self.HEADER_COLS)

    def test_separator_lines(self, output):
        assert_separator_lines(output, count=2)

    def test_data_lines(self, output):
        assert_data_matches(output, self.EXPECTED)

    def test_temperature_shown(self, output):
        assert_line_present(output, r'Temperature = 1000')

    def test_concentration_shown(self, output):
        assert_line_present(output, r'Concentration = 1\.0')


# ---------------------------------------------------------------------------
# Example 4: Temperature interval 300-1000 K, Truhlar, cut-off 120 cm-1
# ---------------------------------------------------------------------------
class TestExample4:
    """python -m goodvibes examples/methylaniline.out --ti '300,1000,100' --qs truhlar -f 120"""

    EXPECTED = [
        ('methylaniline', [300.0, -326.514399, 0.040005, 0.039842, -326.554404, -326.554241]),
        ('methylaniline', [400.0, -326.508735, 0.059816, 0.059596, -326.568551, -326.568331]),
        ('methylaniline', [500.0, -326.501670, 0.082625, 0.082349, -326.584296, -326.584020]),
        ('methylaniline', [600.0, -326.493429, 0.108148, 0.107816, -326.601577, -326.601245]),
        ('methylaniline', [700.0, -326.484222, 0.136095, 0.135707, -326.620317, -326.619930]),
        ('methylaniline', [800.0, -326.474218, 0.166216, 0.165772, -326.640434, -326.639990]),
        ('methylaniline', [900.0, -326.463545, 0.198300, 0.197800, -326.661845, -326.661346]),
        ('methylaniline', [1000.0, -326.452307, 0.232169, 0.231614, -326.684476, -326.683921]),
    ]
    HEADER_COLS = ['Temp/K', 'H', 'T.S', 'T.qh-S', 'G(T)', 'qh-G(T)']

    @pytest.fixture(scope='class')
    def output(self):
        return run_goodvibes([
            example('methylaniline.out'),
            '--ti', '300,1000,100', '--qs', 'truhlar', '-f', '120',
        ])

    def test_header(self, output):
        assert_header(output, self.HEADER_COLS)

    def test_data_lines(self, output):
        assert_data_matches(output, self.EXPECTED)

    def test_all_temperatures_present(self, output):
        rows = parse_data_lines(output)
        temps = [v[0] for _, v in rows]
        assert temps == [300.0, 400.0, 500.0, 600.0, 700.0, 800.0, 900.0, 1000.0]

    def test_truhlar_reference(self, output):
        assert_line_present(output, r'low frequencies are adjusted to')


# ---------------------------------------------------------------------------
# Example 5: Vibrational scaling factor
# ---------------------------------------------------------------------------
class TestExample5:
    """python -m goodvibes examples/methylaniline.out -v 0.95"""

    EXPECTED = [
        ('methylaniline', [-326.664901, 0.135012, -326.521265, 0.040238, 0.040091, -326.561503, -326.561356]),
    ]
    HEADER_COLS = ['E', 'ZPE', 'H', 'T.S', 'T.qh-S', 'G(T)', 'qh-G(T)']

    @pytest.fixture(scope='class')
    def output(self):
        return run_goodvibes([example('methylaniline.out'), '-v', '0.95'])

    def test_header(self, output):
        assert_header(output, self.HEADER_COLS)

    def test_separator_lines(self, output):
        assert_separator_lines(output, count=2)

    def test_data_lines(self, output):
        assert_data_matches(output, self.EXPECTED)

    def test_scale_factor_noted(self, output):
        assert_line_present(output, r'vibrational scale factor 0\.95')


# ---------------------------------------------------------------------------
# Example 7: Multiple files with --cpu
# Uses explicit file list rather than glob to match the readme expected output.
# ---------------------------------------------------------------------------
class TestExample7:
    """python -m goodvibes examples/*.out --cpu  (subset of files from readme)"""

    EXPECTED = [
        ('Al_298K',       [-242.328708, 0.000000, -242.326347, 0.017670, 0.017670, -242.344018, -242.344018]),
        ('Al_400K',       [-242.328708, 0.000000, -242.326347, 0.017670, 0.017670, -242.344018, -242.344018]),
        ('H2O',           [ -76.368128, 0.020772,  -76.343577, 0.021458, 0.021458,  -76.365035,  -76.365035]),
        ('HCN_singlet',   [ -93.358851, 0.015978,  -93.339373, 0.022896, 0.022896,  -93.362269,  -93.362269]),
        ('HCN_triplet',   [ -93.153787, 0.012567,  -93.137780, 0.024070, 0.024070,  -93.161850,  -93.161850]),
        ('allene',        [-116.569605, 0.053913, -116.510916, 0.027618, 0.027621, -116.538534, -116.538537]),
        ('benzene',       [-232.227201, 0.101377, -232.120521, 0.032742, 0.032745, -232.153263, -232.153265]),
        ('ethane',        [ -79.830421, 0.075238,  -79.750770, 0.027523, 0.027525,  -79.778293,  -79.778295]),
        ('isobutane',     [-158.458811, 0.132380, -158.319804, 0.034241, 0.034252, -158.354046, -158.354056]),
        ('methylaniline', [-326.664901, 0.142118, -326.514489, 0.039668, 0.039535, -326.554157, -326.554024]),
        ('neopentane',    [-197.772980, 0.160311, -197.604824, 0.036952, 0.036966, -197.641776, -197.641791]),
    ]

    FILES = [
        'Al_298K.out', 'Al_400K.out', 'H2O.out', 'HCN_singlet.out',
        'HCN_triplet.out', 'allene.out', 'benzene.out', 'ethane.out',
        'isobutane.out', 'methylaniline.out', 'neopentane.out',
    ]

    @pytest.fixture(scope='class')
    def output(self):
        return run_goodvibes([example(f) for f in self.FILES] + ['--cpu'])

    def test_header(self, output):
        assert_header(output, ['E', 'ZPE', 'H', 'T.S', 'T.qh-S', 'G(T)', 'qh-G(T)'])

    def test_data_lines(self, output):
        assert_data_matches(output, self.EXPECTED)

    def test_all_structures_present(self, output):
        rows = parse_data_lines(output)
        names = [name for name, _ in rows]
        expected_names = [name for name, _ in self.EXPECTED]
        assert names == expected_names

    def test_cpu_line(self, output):
        assert_line_present(output, r'TOTAL CPU\s+\d+ days\s+\d+ hrs\s+\d+ mins\s+\d+ secs')


# ---------------------------------------------------------------------------
# Example 8: Entropy symmetry correction (--ssymm)
# Requires platform-specific C binaries; skip if unavailable.
# ---------------------------------------------------------------------------
class TestExample8:
    """python -m goodvibes examples/allene.out ... --ssymm"""

    EXPECTED = [
        ('allene',    [-116.569605, 0.053913, -116.510916, 0.026309, 0.026312, -116.537225, -116.537228]),
        ('benzene',   [-232.227201, 0.101377, -232.120521, 0.030396, 0.030399, -232.150917, -232.150919]),
        ('ethane',    [ -79.830421, 0.075238,  -79.750770, 0.025831, 0.025833,  -79.776601,  -79.776603]),
        ('isobutane', [-158.458811, 0.132380, -158.319804, 0.033204, 0.033214, -158.353008, -158.353019]),
        ('neopentane',[-197.772980, 0.160311, -197.604824, 0.034606, 0.034620, -197.639430, -197.639444]),
    ]
    HEADER_COLS = ['E', 'ZPE', 'H', 'T.S', 'T.qh-S', 'G(T)', 'qh-G(T)', 'Point Group']

    FILES = ['allene.out', 'benzene.out', 'ethane.out', 'isobutane.out', 'neopentane.out']

    @pytest.fixture(scope='class')
    def output(self):
        cmd = [example(f) for f in self.FILES] + ['--ssymm']
        result = subprocess.run(
            [sys.executable, '-m', 'goodvibes'] + cmd,
            capture_output=True, text=True, cwd=ROOT_DIR,
        )
        if result.returncode != 0:
            pytest.skip(
                'ssymm failed (likely missing platform-specific binary): '
                + result.stderr.splitlines()[-1] if result.stderr.strip() else 'unknown error'
            )
        return result.stdout

    def test_header(self, output):
        assert_header(output, self.HEADER_COLS)

    def test_data_lines(self, output):
        assert_data_matches(output, self.EXPECTED)

    def test_point_groups_present(self, output):
        """Each data line should end with a point group label."""
        expected_groups = {'D2d', 'D6h', 'D3d', 'C3v', 'Td'}
        found = set()
        for line in output.splitlines():
            stripped = line.strip()
            if stripped.startswith('o  ') and 'Requested' not in stripped:
                last_token = stripped.split()[-1]
                if not last_token.replace('-', '').replace('.', '').lstrip('-').isdigit():
                    found.add(last_token)
        assert found == expected_groups, f"Expected {expected_groups}, found {found}"


# ---------------------------------------------------------------------------
# --check: consistency validation
# ---------------------------------------------------------------------------
class TestCheckFlag:
    """python -m goodvibes examples/ethane.out --check"""

    @pytest.fixture(scope='class')
    def output(self):
        return run_goodvibes([example('ethane.out'), '--check'])

    def test_check_output_present(self, output):
        """--check should produce validation output."""
        assert_line_present(output, r'Checks for thermochemistry')

    def test_data_still_present(self, output):
        """Normal thermochemistry output should still appear."""
        rows = parse_data_lines(output)
        assert len(rows) >= 1


# ---------------------------------------------------------------------------
# --xyz: coordinate export
# ---------------------------------------------------------------------------
class TestXYZFlag:
    """python -m goodvibes examples/ethane.out --xyz"""

    @pytest.fixture(scope='class')
    def output(self):
        out = run_goodvibes([example('ethane.out'), '--xyz'])
        yield out
        # Clean up generated xyz file
        xyz_file = os.path.join(ROOT_DIR, 'GoodVibes_output.xyz')
        if os.path.exists(xyz_file):
            os.remove(xyz_file)

    def test_xyz_file_created(self, output):
        """--xyz should create a GoodVibes_output.xyz file."""
        xyz_file = os.path.join(ROOT_DIR, 'GoodVibes_output.xyz')
        assert os.path.exists(xyz_file), "Expected GoodVibes_output.xyz to be created"

    def test_xyz_file_has_content(self, output):
        """The xyz file should contain atom coordinates."""
        xyz_file = os.path.join(ROOT_DIR, 'GoodVibes_output.xyz')
        with open(xyz_file) as f:
            content = f.read()
        # XYZ format: first line is atom count
        lines = content.strip().splitlines()
        assert len(lines) > 2
        atom_count = int(lines[0].strip())
        assert atom_count > 0


# ---------------------------------------------------------------------------
# --csv: CSV output format
# ---------------------------------------------------------------------------
class TestCSVFlag:
    """python -m goodvibes examples/ethane.out --csv"""

    @pytest.fixture(scope='class')
    def output(self):
        out = run_goodvibes([example('ethane.out'), '--csv'])
        yield out
        # Clean up generated csv file
        csv_file = os.path.join(ROOT_DIR, 'GoodVibes_output.csv')
        if os.path.exists(csv_file):
            os.remove(csv_file)

    def test_csv_file_created(self, output):
        """--csv should create a GoodVibes_output.csv file."""
        csv_file = os.path.join(ROOT_DIR, 'GoodVibes_output.csv')
        assert os.path.exists(csv_file), "Expected GoodVibes_output.csv to be created"

    def test_csv_contains_commas(self, output):
        """The CSV file should contain comma-separated values."""
        csv_file = os.path.join(ROOT_DIR, 'GoodVibes_output.csv')
        with open(csv_file) as f:
            content = f.read()
        assert ',' in content


# ---------------------------------------------------------------------------
# --boltz: Boltzmann weighting with multiple conformers
# ---------------------------------------------------------------------------
class TestBoltzFlag:
    """python -m goodvibes examples/isobutane.out examples/neopentane.out --boltz"""

    @pytest.fixture(scope='class')
    def output(self):
        return run_goodvibes([
            example('isobutane.out'), example('neopentane.out'), '--boltz',
        ])

    def test_boltz_output_present(self, output):
        """--boltz should produce Boltzmann weighting output."""
        assert_line_present(output, r'Boltz')

    def test_data_still_present(self, output):
        """Normal thermochemistry output should still appear."""
        rows = parse_data_lines(output)
        assert len(rows) >= 2
