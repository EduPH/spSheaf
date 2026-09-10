"""SSAM's sctransform normalization without the R `arrow`/`feather` packages.

``ssam._analysis.run_sctransform`` exchanges data with R through feather and
needs the R ``arrow`` package, which is not always installable. This is the
same call (``sctransform::vst`` on genes x local-maximum vectors, returning
the Pearson residuals ``res$y`` and the fitted model parameters) with CSV as
the exchange format. Install it with ``ssam_use_csv_sctransform()`` before
``SSAMAnalysis.normalize_vectors_sctransform()``.
"""
from __future__ import annotations

import os
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


def run_sctransform_csv(data, clip_range=None, verbose=True, debug_path=None,
                        plot_model_pars=False, **kwargs):
    with TemporaryDirectory() as tmp:
        if debug_path:
            tmp = debug_path
        ifn, ofn, pfn, rfn = [os.path.join(tmp, e) for e in ("in.csv", "out.csv", "pars.csv", "script.R")]
        df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(np.asarray(data), columns=[str(e) for e in range(data.shape[1])])
        df.to_csv(ifn, index=False)
        opts = "".join(', %s = "%s"' % (k, v) if isinstance(v, str) else ", %s = %s" % (k, v) for k, v in kwargs.items())
        rcmd = ('library(sctransform); mat <- t(as.matrix(read.csv("{0}", check.names=FALSE))); '
                'colnames(mat) <- 1:ncol(mat); res <- vst(mat{1}, return_gene_attr=TRUE, return_cell_attr=TRUE); '
                'write.csv(as.data.frame(t(res$y)), "{2}", row.names=FALSE); '
                'write.csv(as.data.frame(res$model_pars_fit), "{3}", row.names=TRUE);').format(ifn, opts, ofn, pfn)
        with open(rfn, "w") as f:
            f.write(rcmd)
        proc = subprocess.run(["Rscript", rfn], capture_output=True, text=True)
        if proc.returncode != 0 or not os.path.exists(ofn):
            raise RuntimeError("sctransform failed:\n" + proc.stdout[-2000:] + proc.stderr[-2000:])
        if verbose:
            print(proc.stdout[-500:])
        out = pd.read_csv(ofn)
        pars = pd.read_csv(pfn, index_col=0)
        pars.index = pars.index.astype(str)
        # vst drops genes it cannot fit; restore them as zero residuals and
        # NaN parameters (SSAM's nan_to_num then zeroes them in the field) so
        # the gene axis matches the vector field
        out = out.reindex(columns=df.columns, fill_value=0.0)
        pars = pars.reindex(df.columns)
        return out, pars


def ssam_use_csv_sctransform():
    import ssam._analysis as A
    A.run_sctransform = run_sctransform_csv
