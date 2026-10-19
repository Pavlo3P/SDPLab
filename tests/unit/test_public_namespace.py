import subprocess
import sys


def test_subpackages_reachable_from_bare_import():
    # A fresh interpreter, so earlier tests' imports cannot mask the bug.
    code = (
        "import sdplab\n"
        "assert sdplab.examples.generate_max_cut\n"
        "assert sdplab.special.pauli.PauliString\n"
        "assert sdplab.special.qot.QOTConstraintOp\n"
        "assert sdplab.linalg.kron_sum\n"
        "assert {'examples', 'special'} <= set(dir(sdplab))\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
