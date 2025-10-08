#!/usr/bin/env python3
from pathlib import Path
import os

root = Path("data/niftiData")

# Patterns we may have created: *.bak.nii.gz or *.bak.N
patterns = ["*.bak.nii.gz", "*.bak.nii", "*.bak"]

restored = 0
skipped = 0
for pat in patterns:
    for bak in root.rglob(pat):
        # Derive intended original name
        name = bak.name
        if name.endswith(".bak.nii.gz"):
            orig = bak.with_name(name.replace(".bak.nii.gz", ".nii.gz"))
        elif name.endswith(".bak.nii"):
            orig = bak.with_name(name.replace(".bak.nii", ".nii"))
        else:
            # last resort: strip ".bak" once
            orig = bak.with_name(name.replace(".bak", "", 1))

        if orig.exists():
            skipped += 1
            continue
        os.replace(str(bak), str(orig))
        restored += 1

print(f"Restored {restored} files, skipped {skipped} (already present).")
