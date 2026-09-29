import os
import math
import argparse
import subprocess

def read_xyz(filename):
    """Read atomic symbols and coordinates from an XYZ file"""
    atoms = []
    with open(filename, 'r') as f:
        lines = f.readlines()[2:]
        for line in lines:
            if line.strip():
                parts = line.split()
                symbol = parts[0]
                x, y, z = map(float, parts[1:4])
                atoms.append((symbol, x, y, z))
    return atoms

def calc_distance(atom1, atom2):
    """Return Euclidean distance between two atoms"""
    _, x1, y1, z1 = atom1
    _, x2, y2, z2 = atom2
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)

def write_constraints_file(atoms, constraints, filename="constraints.inp"):
    """Write constraints file for xTB/CREST TS optimisation"""
    with open(filename, 'w') as f:
        f.write('$constrain\n')
        f.write('force constant=1.0\n')
        for a1, a2 in constraints:
            dist = calc_distance(atoms[a1-1], atoms[a2-1])
            f.write(f"distance: {a1}, {a2}, {dist:.4f}\n")
        f.write('$end\n')
    print(f"[INFO] Constraints file written: {filename}")

def write_slurm_script(
    job_name, xyz_file, chrg, uhf, solvent, thresh=1.0, ts=False, time="12:00:00",
    mem=16000, cpus=16, partition="nodes"
):
    """Write a Slurm submission script for CREST conformer search and CREGEN redundant
    conformer elimination."""
    out_file = f"{job_name}.out"
    err_file = f"{job_name}.err"
    
    if ts:
        crest_cmd = f"crest {xyz_file} --cinp constraints.inp --gfn2 --chrg {chrg} --uhf {uhf} --gbsa {solvent} -T {cpus}"
    else: 
        crest_cmd = f"crest {xyz_file} --gfn2 --chrg {chrg} --uhf {uhf} --gbsa {solvent} -T {cpus}"

    cregen_cmd = f"crest crest_best.xyz --cregen crest_conformers.xyz --rthr {thresh}"

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
conda activate auto-opt

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "Using CREST installation:"
which crest

echo "Starting CREST job: {job_name}"
echo "Running on $(hostname)"
echo "Using {cpus} CPUs"

{crest_cmd}

echo "CREST job completed."

echo "CREGEN redundant conformer elimination with threshold {thresh} Å."

{cregen_cmd}

test -s crest_ensemble.xyz

echo "CREGEN completed."
"""
    with open(f"{job_name}.slm", "w") as f:
        f.write(script)
    print(f"[INFO] Slurm script written: {job_name}.slm")

def submit_job(job_name):
    """Submit CREST job via sbatch and wait for it to finish."""
    try:
        subprocess.run(
            ["sbatch", "--wait", f"{job_name}.slm"],
            check=True,
        )
        print("[INFO] CREST job completed successfully.")
    except subprocess.CalledProcessError as e:
        print("[ERROR] CREST job failed:", e)
        raise

def main():
    parser = argparse.ArgumentParser(description="Automate xTB geometry optimisation via Slurm.")
    parser.add_argument("xyz", help="Input XYZ file")
    parser.add_argument("--chrg", type=int, default=0, help="Molecular charge")
    parser.add_argument("--uhf", type=int, default=0, help="Number of unpaired electrons")
    parser.add_argument("--solvent", default="none", help="Solvent for GBSA model")
    parser.add_argument("--thresh", type=float, default=1.0, help="Threshold for CREGEN redundant conformer elimination")
    parser.add_argument("--jobname", default="xtb_job", help="Job name")
    parser.add_argument("--time", default="12:00:00", help="Required walltime")
    parser.add_argument("--mem", type=int, default=16000, help="Memory requirement (MB)")
    parser.add_argument("--cpus", type=int, default=16, help="Number of processors required")
    parser.add_argument("--ts", action="store_true", help="Indicate if transition state (TS)")
    parser.add_argument("--constraints", nargs="+", help="List of fixed bonds as pairs, e.g. 1-2 3-4")

    args = parser.parse_args()

    atoms = read_xyz(args.xyz)

    if args.ts:
        if not args.constraints:
            raise ValueError("Transition state job requires --constraints argument.")
        bonds = [tuple(map(int, pair.split('-'))) for pair in args.constraints]
        write_constraints_file(atoms, bonds)

    write_slurm_script(
        job_name=args.jobname,
        xyz_file=args.xyz,
        chrg=args.chrg,
        uhf=args.uhf,
        solvent=args.solvent,
        thresh=args.thresh,
        time=args.time,
        mem=args.mem,
        cpus=args.cpus,
        ts=args.ts
    )

    submit_job(args.jobname)

if __name__ == "__main__":
    main()