#!/usr/bin/env python
"""
Inspect the Takei IF-binned data (Phase 2b) so the chromatin-mark loader can be
locked to the real schema.

Works on either the downloaded zip (``--zip``) or an already-extracted folder
(``--dir``), plus the target table (``--excel`` = IF_supp_table_hyb_targets.xlsx).
It reports: whether the zip is complete, the entry list + file types, a sniff of
the first data file (CSV columns / head, or image dimensions), and the
mark -> (hyb, channel) mapping from the Excel.

Run this once the Pos0 zip has finished downloading and paste the output back.

Usage
-----
    python inspect_if.py --zip IF-binned1x1x1-E14-replicate1-Pos0.zip \
        --excel IF_supp_table_hyb_targets.xlsx
    # or, after unzipping:
    python inspect_if.py --dir if_binned_pos0 --excel IF_supp_table_hyb_targets.xlsx
"""
import argparse
import io
import os
import zipfile


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--zip", default=None, help="IF-binned ...Pos0.zip")
    p.add_argument("--dir", default=None, help="extracted IF folder")
    p.add_argument("--excel", default=None, help="IF_supp_table_hyb_targets.xlsx")
    p.add_argument("--max-entries", type=int, default=30)
    return p


def sniff_bytes(name, data):
    """Print a short structural summary of one in-memory file."""
    ext = name.lower().rsplit(".", 1)[-1] if "." in name else ""
    print(f"\n--- sniff: {name} ({len(data):,} bytes, .{ext}) ---")
    if ext in ("csv", "tsv", "txt"):
        import pandas as pd
        sep = "\t" if ext == "tsv" else ","
        df = pd.read_csv(io.BytesIO(data), sep=sep, nrows=2000)
        print("columns:", list(df.columns))
        print(df.head(5).to_string())
        for c in df.columns:
            if df[c].dtype.kind in "if":
                print(f"  {c}: range {df[c].min():.2f}..{df[c].max():.2f}")
    elif ext in ("tif", "tiff", "png", "jpg", "jpeg"):
        try:
            from PIL import Image
            im = Image.open(io.BytesIO(data))
            print(f"image: size={im.size} mode={im.mode} n_frames={getattr(im,'n_frames',1)}")
        except Exception as e:
            print("image read failed:", e)
    elif ext in ("npy",):
        import numpy as np
        arr = np.load(io.BytesIO(data), allow_pickle=True)
        print("ndarray:", arr.shape, arr.dtype)
    elif ext in ("mat",):
        print("MATLAB .mat — load with scipy.io.loadmat / h5py")
    else:
        print("first bytes:", data[:80])


def inspect_zip(path, max_entries):
    print(f"zip: {path}  ({os.path.getsize(path):,} bytes)")
    try:
        z = zipfile.ZipFile(path)
    except zipfile.BadZipFile:
        print("  [!] cannot open as zip — the download is INCOMPLETE/corrupt "
              "(the central directory at the end of the file is missing). "
              "Finish the download, then re-run.")
        return
    names = z.namelist()
    from collections import Counter
    print(f"  entries: {len(names)}")
    print("  ext counts:", dict(Counter(
        n.rsplit('.', 1)[-1] if '.' in n and not n.endswith('/') else 'dir'
        for n in names)))
    for n in names[:max_entries]:
        print(f"    {n}  ({z.getinfo(n).file_size:,} B)")
    # sniff the first real (non-dir) data file
    data_files = [n for n in names if not n.endswith("/")
                  and z.getinfo(n).file_size > 0]
    if data_files:
        first = data_files[0]
        sniff_bytes(first, z.read(first))


def inspect_dir(path, max_entries):
    files = []
    for root, _, fs in os.walk(path):
        for f in fs:
            files.append(os.path.join(root, f))
    print(f"dir: {path}  ({len(files)} files)")
    for f in sorted(files)[:max_entries]:
        print(f"    {os.path.relpath(f, path)}  ({os.path.getsize(f):,} B)")
    if files:
        first = sorted(files)[0]
        with open(first, "rb") as fh:
            sniff_bytes(first, fh.read())


def inspect_excel(path):
    import pandas as pd
    print(f"\n=== target table: {path} ===")
    xls = pd.ExcelFile(path)
    print("sheets:", xls.sheet_names)
    for sh in xls.sheet_names[:4]:
        df = xls.parse(sh)
        print(f"\n[{sh}] columns:", list(df.columns))
        print(df.head(20).to_string())


def main():
    a = build_parser().parse_args()
    if a.zip:
        inspect_zip(a.zip, a.max_entries)
    if a.dir:
        inspect_dir(a.dir, a.max_entries)
    if a.excel and os.path.exists(a.excel):
        inspect_excel(a.excel)
    elif a.excel:
        print(f"\n[!] excel not found: {a.excel}")


if __name__ == "__main__":
    main()
