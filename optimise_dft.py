import os
import re
import argparse
import subprocess

def gaussian_out_completed(path):
    """Return True only for a normally terminated Gaussian calculation."""
    try:
        with open(path, "r", errors="ignore") as f:
            return (
                "Normal termination of Gaussian 16"
                in f.read()[-300000:]
            )
    except Exception:
        return False

def parse_gaussian_scf_energy(gauss_out):
    """
    Parse the last 'SCF Done:' energy from a Gaussian .out file.
    Returns energy in Hartree (float) or None if not found.
    """
    last_energy = None
    try:
        with open(gauss_out, "r", errors="ignore") as f:
            for line in f:
                if "SCF Done:" in line:
                    m = re.search(r"SCF Done:\s+E\([A-Za-z0-9]+\)\s*=\s*([-\d\.Ee+]+)", line)
                    if m:
                        last_energy = float(m.group(1))
    except Exception:
        return None
    return last_energy

def list_out_files(sps_dir):
    return sorted(
        [os.path.join(sps_dir, f) for f in os.listdir(sps_dir) if f.endswith(".out")]
    )

def match_gjf_for_out(out_path):
    """Map conf_X.out to conf_X.gjf in the same directory."""
    base = os.path.splitext(os.path.basename(out_path))[0]
    return os.path.join(os.path.dirname(out_path), f"{base}.gjf")

def read_gjf_coordinates(gjf_path):
    """
    Read charge, multiplicity, and Cartesian block from a Gaussian .gjf file.
    Coordinates are returned as a list of lines 'El  x  y  z'.
    """
    with open(gjf_path, "r") as f:
        lines = [l.strip("\n") for l in f]

    i = 0

    # skip initial lines
    while i < len(lines) and lines[i].strip():
        i += 1

    # skip first blank
    while i < len(lines) and not lines[i].strip():
        i += 1
    
    # skip title line
    i += 1

    # skip blank after title
    while i < len(lines) and not lines[i].strip():
        i += 1

    chrg, mult = 0, 1
    if i < len(lines):
        parts = lines[i].split()
        chrg, mult = int(parts[0]), int(parts[1])
        i += 1

    coords = []
    while i < len(lines) and lines[i].strip():
        coords.append(lines[i])
        i += 1

    return chrg, mult, coords

def g16_mem_gb(mem):
    """Gaussian %mem in GB"""
    if mem <= 1000:
        return int(round(mem / 1000, 0))
    else:
        return int(round(mem / 1000, 0)) - 1

def write_gjf_full_opt(fname, qc_method, basis_set, solvent, dispersion, chrg, mult, coords, mem, cpus, outdir="."):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, fname)
    with open(path, "w") as f:
        f.write(f"%mem={g16_mem_gb(mem)}GB\n")
        f.write(f"%nprocshared={cpus}\n")
        route_line = f"# {qc_method} {basis_set} opt freq=noraman scrf=(smd,solvent={solvent})"
        if dispersion and dispersion.lower() != "none":
            route_line += f" EmpiricalDispersion={dispersion}"
        f.write(f"{route_line}\n\n")
        f.write(f"Final optimisation\n\n")
        f.write(f"{chrg} {mult}\n")
        for line in coords:
            f.write(line + "\n")
        f.write("\n")
    return path

def write_gjf_ts_frozen(fname, qc_method, basis_set, solvent, dispersion, chrg, mult, coords, 
                        mem, cpus, frozen_bonds, chk_name="ts_frozen.chk", outdir="."):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, fname)
    with open(path, "w") as f:
        f.write(f"%mem={g16_mem_gb(mem)}GB\n")
        f.write(f"%nprocshared={cpus}\n")
        f.write(f"%chk={chk_name}\n")
        route_line = f"# {qc_method} {basis_set} opt=(modredundant,loose) scrf=(smd,solvent={solvent})"
        if dispersion and dispersion.lower() != "none":
            route_line += f" EmpiricalDispersion={dispersion}"
        f.write(f"{route_line}\n\n")
        f.write(f"Frozen TS optimisation\n\n")
        f.write(f"{chrg} {mult}\n")
        for line in coords:
            f.write(line + "\n")
        f.write("\n")
        for a1, a2 in frozen_bonds:
            f.write(f"B {a1} {a2} F\n")
        f.write("\n")
    return path

def write_gjf_ts_full_from_chk(fname, qc_method, basis_set, solvent, dispersion, mem, cpus, chk_name="ts_frozen.chk", outdir="."):
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, fname)
    with open(path, "w") as f:
        f.write(f"%mem={g16_mem_gb(mem)}GB\n")
        f.write(f"%nprocshared={cpus}\n")
        f.write(f"%chk={chk_name}\n")
        route_line = f"# {qc_method} {basis_set} opt=(ts,calcfc,recalcfc=20,noeigentest) freq=noraman scrf=(smd,solvent={solvent})"
        if dispersion and dispersion.lower() != "none":
            route_line += f" EmpiricalDispersion={dispersion}"
        route_line += " Geom=AllCheck Guess=Read"
        f.write(f"{route_line}\n\n")
        f.write(f"Full TS optimisation\n\n")
        f.write(f"\n")
    return path

def write_slurm_script(job_name, gjf_file, time, mem, cpus, partition="nodes", outdir="."):
    """Write a Slurm submission script for the optimisation."""
    os.makedirs(outdir, exist_ok=True)
    slm_path = os.path.join(outdir, f"{job_name}.slm")
    out_file = f"{job_name}.out"
    err_file = f"{job_name}.err"

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --partition={partition}
#SBATCH --output={out_file}
#SBATCH --error={err_file}

set -e

module purge
module load gaussian/16

echo "$GAUSS_SCRDIR"
mkdir -p "$GAUSS_SCRDIR"
chmod 700 "$GAUSS_SCRDIR"

echo "Running Gaussian single-point for {job_name}"
g16 < {gjf_file}

rm -rf $GAUSS_SCRDIR
"""
    with open(slm_path, "w") as f:
        f.write(script)
    return slm_path

def submit_sbatch(slm_path, workdir="."):
    """Submit an sbatch job and wait for it to finish."""

    cmd = ["sbatch", "--wait", os.path.basename(slm_path)]

    try:
        res = subprocess.run(cmd, cwd=workdir, check=True, capture_output=True, text=True)
        if res.stdout:
            print(res.stdout.strip())

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] sbatch failed for {slm_path}: {e}")
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        raise
    
def pick_lowest_energy_conformer(sps_dir, conformers=None):
    """
    Return (out_path, gjf_path, energy) of the lowest-energy
    successfully completed conformer.
    """
    if conformers:
        outs = [
            os.path.join(sps_dir, f"{conf}.out")
            for conf in conformers
        ]
    else:
        outs = list_out_files(sps_dir)

    if not outs:
        raise FileNotFoundError(
            f"No conformer outputs found in {sps_dir}"
        )
    best = None
    for outp in outs:
        if not gaussian_out_completed(outp):
            print(
                f"[WARN] Gaussian calculation did not terminate normally: "
                f"{outp}; skipping."
            )
            continue

        e = parse_gaussian_scf_energy(outp)
        if e is None:
            print(f"[WARN] No SCF energy found in {outp}; skipping.")
            continue
        if (best is None) or (e < best[2]):
            best = (outp, match_gjf_for_out(outp), e)
    if best is None:
        raise RuntimeError(
            "Could not find any valid SCF energies in SPS outputs."
        )
    print(
        f"[INFO] Lowest energy: {best[2]:.10f} Ha "
        f"from {os.path.basename(best[0])}"
    )
    return best

def main():
    parser = argparse.ArgumentParser(
        description="Choose lowest-energy conformer from SPS results and run DFT optimisation (normal or TS)"
    )
    parser.add_argument("--jobname", default="opt", help="Job name")
    parser.add_argument("--sps_dir", default="sps", help="Directory containing conf_*.out and conf_*.gjf")
    parser.add_argument("--qc_method", default="M062X", help="Quantum chemistry method (e.g. M062X)")
    parser.add_argument("--basis_set", default="def2tzvp", help="Basis set")
    parser.add_argument("--solvent", default="none", help="Solvent for SMD model")
    parser.add_argument("--dispersion", choices=["GD2", "GD3", "GD3BJ", "none"], default="none", help="Empirical dispersion correction")
    parser.add_argument("--chrg", type=int, default=None, help="Molecular charge")
    parser.add_argument("--mult", type=int, default=None, help="Spin multiplicity")
    parser.add_argument("--time", default="2:00:00", help="Required walltime")
    parser.add_argument("--mem", type=int, default=16000, help="Memory requirement (MB)")
    parser.add_argument("--cpus", type=int, default=16, help="Number of processors required")
    parser.add_argument("--partition", default="nodes", help="SLURM partition")
    parser.add_argument("--opt_dir", default=".", help="Directory to place optimisation inputs/scripts")
    parser.add_argument("--conformers", nargs="+", help="Conformer names eligible for final DFT selection")

    # TS options   
    parser.add_argument("--ts", action="store_true", help="Run 2 step TS optimisation")
    parser.add_argument("--constraints", nargs="+", help="Frozen TS bonds as pairs: 1-2 3-4 ...")

    args = parser.parse_args()

    outp, gjf, energy = pick_lowest_energy_conformer(args.sps_dir, args.conformers)
    chrg_in, mult_in, coords = read_gjf_coordinates(gjf)

    chrg = args.chrg if args.chrg is not None else chrg_in
    mult = args.mult if args.mult is not None else mult_in

    os.makedirs(args.opt_dir, exist_ok=True)

    if not args.ts:
        gjf_name = args.jobname + "_opt.gjf"
        job_name = args.jobname + "_opt"
        gjf_path = write_gjf_full_opt(
            outdir=args.opt_dir,
            fname=gjf_name,
            qc_method=args.qc_method,
            basis_set=args.basis_set,
            solvent=args.solvent,
            dispersion=args.dispersion,
            chrg=chrg,
            mult=mult,
            coords=coords,
            mem=args.mem,
            cpus=args.cpus
        )
        slm = write_slurm_script(
            outdir=args.opt_dir,
            job_name=job_name,
            gjf_file=os.path.basename(gjf_path),
            time=args.time,
            mem=args.mem,
            cpus=args.cpus,
            partition=args.partition
        )
        submit_sbatch(slm, workdir=args.opt_dir)

        out_path = os.path.join(
            args.opt_dir,
            f"{job_name}.out",
        )

        if not gaussian_out_completed(out_path):
            raise RuntimeError(
                f"Gaussian optimisation did not terminate normally: {out_path}"
            )

    else:
        if not args.constraints:
            raise ValueError("TS optimisation requires --constraints like: 1-2 3-4...")
        
        frozen_bonds = []
        for pair in args.constraints:
            try:
                a, b = map(int, pair.split("-"))
                frozen_bonds.append((a, b))
            except Exception:
                raise ValueError(f"Bad constraint '{pair}'. Use format i-j")

        frozen_gjf = args.jobname + "_frozen.gjf"
        frozen_job = args.jobname + "_frozen"
        chk_name = args.jobname + "_frozen.chk"
        frozen_path = write_gjf_ts_frozen(
            outdir=args.opt_dir,
            fname=frozen_gjf,
            qc_method=args.qc_method,
            basis_set=args.basis_set,
            solvent=args.solvent,
            dispersion=args.dispersion,
            chrg=chrg,
            mult=mult,
            coords=coords,
            mem=args.mem,
            cpus=args.cpus,
            frozen_bonds=frozen_bonds,
            chk_name=chk_name
        )
        frozen_slm = write_slurm_script(
            outdir=args.opt_dir,
            job_name=frozen_job,
            gjf_file=os.path.basename(frozen_path),
            time=args.time,
            mem=args.mem,
            cpus=args.cpus,
            partition=args.partition
        )
        submit_sbatch(frozen_slm, workdir=args.opt_dir)

        frozen_out = os.path.join(
            args.opt_dir,
            f"{frozen_job}.out",
        )

        if not gaussian_out_completed(frozen_out):
            raise RuntimeError(
                f"Frozen TS optimisation did not terminate normally: {frozen_out}"
            )

        full_gjf = args.jobname + "_full.gjf"
        full_job = args.jobname + "_full"
        full_path = write_gjf_ts_full_from_chk(
            outdir=args.opt_dir,
            fname=full_gjf,
            qc_method=args.qc_method,
            basis_set=args.basis_set,
            solvent=args.solvent,
            dispersion=args.dispersion,
            mem=args.mem,
            cpus=args.cpus,
            chk_name=chk_name
        )
        full_slm = write_slurm_script(
            outdir=args.opt_dir,
            job_name=full_job,
            gjf_file=os.path.basename(full_path),
            time=args.time,
            mem=args.mem,
            cpus=args.cpus,
            partition=args.partition
        )

        submit_sbatch(full_slm, workdir=args.opt_dir)

        full_out = os.path.join(
            args.opt_dir,
            f"{full_job}.out",
        )

        if not gaussian_out_completed(full_out):
            raise RuntimeError(
                f"Full TS optimisation did not terminate normally: {full_out}"
            )

    print("[INFO] DFT optimisation completed successfully.")

if __name__ == "__main__":
    main()