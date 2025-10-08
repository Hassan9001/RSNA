#!/usr/bin/env python3
import argparse, os, csv, tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import numpy as np
import nibabel as nib

def is_dircos_orthonormal(R, tol=1e-5):
    return np.allclose(R.T @ R, np.eye(3), atol=tol) and np.isclose(np.linalg.det(R), 1.0, atol=1e-3)

def orthonormalize_affine(img):
    A = img.affine.copy()
    R = A[:3, :3]
    t = A[:3, 3]
    spac = np.linalg.norm(R, axis=0)
    spac[spac == 0] = 1.0
    Rn = R @ np.diag(1.0 / spac)
    U, _, Vt = np.linalg.svd(Rn)
    R_ortho = U @ Vt
    if np.linalg.det(R_ortho) < 0:
        U[:, -1] *= -1
        R_ortho = U @ Vt
    A_fixed = np.eye(4, dtype=np.float64)
    A_fixed[:3, :3] = R_ortho @ np.diag(spac)
    A_fixed[:3, 3] = t
    fixed = nib.Nifti1Image(img.dataobj, A_fixed, header=img.header)
    fixed.set_qform(A_fixed, code=1)
    fixed.set_sform(A_fixed, code=1)
    return fixed

def check_and_fix(path: Path, dry_run=False):
    try:
        img = nib.load(str(path))
    except Exception as e:
        return (str(path), "read_error", str(e))

    R = img.affine[:3, :3]
    norms = np.linalg.norm(R, axis=0, keepdims=True)
    norms[norms == 0] = 1.0
    dircos = R / norms

    if is_dircos_orthonormal(dircos):
        return (str(path), "ok", "")

    if dry_run:
        return (str(path), "would_fix", "")

    try:
        fixed_img = orthonormalize_affine(img)

        # choose temp path with correct extension in same dir
        tmp_path = path.with_name(path.name + ".tmp.nii.gz")
        # ensure we don't collide
        i = 1
        while tmp_path.exists():
            tmp_path = path.with_name(f"{path.name}.tmp{i}.nii.gz")
            i += 1

        nib.save(fixed_img, str(tmp_path))

        # sanity: load temp and verify orthonormal again
        test_img = nib.load(str(tmp_path))
        R2 = test_img.affine[:3, :3]
        norms2 = np.linalg.norm(R2, axis=0, keepdims=True)
        norms2[norms2 == 0] = 1.0
        if not is_dircos_orthonormal(R2 / norms2):
            tmp_path.unlink(missing_ok=True)
            return (str(path), "fix_error", "temp_not_orthonormal")

        # create unique backup name only if needed
        if path.name.endswith(".nii.gz"):
            bak = path.with_name(path.name[:-7] + ".bak.nii.gz")
        elif path.suffix == ".nii":
            bak = path.with_suffix(".bak.nii")
        else:
            bak = path.with_name(path.name + ".bak")
        j = 1
        while bak.exists():
            stem = path.name[:-7] if path.name.endswith(".nii.gz") else path.stem
            ext = ".nii.gz" if path.name.endswith(".nii.gz") else path.suffix
            bak = path.with_name(f"{stem}.bak.{j}{ext}")
            j += 1

        # swap atomically
        os.replace(str(path), str(bak))      # original -> backup
        os.replace(str(tmp_path), str(path)) # tmp -> original
        return (str(path), "fixed", f"backup={bak.name}")

    except Exception as e:
        # clean up any tmp file left
        try:
            if 'tmp_path' in locals() and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
        return (str(path), "fix_error", str(e))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=str)
    ap.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() // 2))
    ap.add_argument("--log", type=str, default="header_fix_log.csv")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    files = sorted(set(root.rglob("*.nii"))) + sorted(set(root.rglob("*.nii.gz")))
    if not files:
        print(f"No NIfTI files under: {root}")
        return

    print(f"Scanning {len(files)} files with workers={args.workers} (dry_run={args.dry_run})...")
    results = []
    if args.workers <= 1:
        for f in files:
            results.append(check_and_fix(f, args.dry_run))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(check_and_fix, f, args.dry_run): f for f in files}
            for i, fut in enumerate(as_completed(futs), 1):
                results.append(fut.result())
                if i % 50 == 0:
                    print(f"  processed {i}/{len(files)} ...")

    log_path = root / args.log
    with open(log_path, "w", newline="") as fp:
        w = csv.writer(fp); w.writerow(["path","status","info"]); w.writerows(results)

    counts = {}
    for _, s, _ in results: counts[s] = counts.get(s, 0) + 1
    print("Done.", counts)
    print(f"Log: {log_path}")

if __name__ == "__main__":
    main()
