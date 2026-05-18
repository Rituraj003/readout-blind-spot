"""DEPRECATED — see notebooks/ouro_scale_sensitivity.ipynb for the canonical
script.

This file previously contained a Python version of the Ouro 1.4B scale
sensitivity experiment. That implementation scaled the post-RMSNorm hidden
state, which is the wrong intervention: the paper's claim is that scaling
the *pre*-readout hidden state by alpha is absorbed by RMSNorm, so the
correct test scales BEFORE applying the readout norm. The notebook does
this correctly; this script did not.

The figure used in paper_v4 (figures/scale_sensitivity.pdf) was produced
by the notebook. Do not rerun this script.

If you need to regenerate the figure, open
notebooks/ouro_scale_sensitivity.ipynb on a CUDA host (Colab T4 or above
works) and execute all cells.
"""

import sys

if __name__ == "__main__":
    print(__doc__, file=sys.stderr)
    sys.exit(1)
