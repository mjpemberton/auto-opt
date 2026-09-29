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
    """Write constraints file for xTB TS optimisation"""
    with open(filename, 'w') as f:
        f.write('$constrain\n')
        f.write('force constant=1.0\n')
        for a1, a2 in constraints:
            dist = calc_distance(atoms[a1-1], atoms[a2-1])
            f.write(f"distance: {a1}, {a2}, {dist:.4f}\n")
        f.write('$end\n')
    print(f"[INFO] Constraints file written: {filename}")

def write_slurm_script(
    job_name, xyz_file, chrg, uhf, solvent, ts=False, time="00:30:00",
    mem=1000, cpus=1, partition="nodes"
):
    """Write a Slurm submission script for xTB optimisation."""
    out_file = f"{job_name}.out"
    err_file = f"{job_name}.err"
    
    if ts:
        xtb_cmd = f"xtb --input constraints.inp {xyz_file} --opt --chrg {chrg} --uhf {uhf} --gbsa {solvent}"
    else: 
        xtb_cmd = f"xtb  {xyz_file} --opt --chrg {chrg} --uhf {uhf} --gbsa {solvent}"

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

echo "Using xTB installation:"
which xtb

echo "Starting xTB job: {job_name}"
echo "Running on $(hostname)"
echo "Using {cpus} CPUs"

{xtb_cmd}

test -s xtbopt.xyz

echo "xTB job completed."
"""
    with open(f"{job_name}.slm", "w") as f:
        f.write(script)
    print(f"[INFO] Slurm script written: {job_name}.slm")

def submit_job(job_name):
    """Submit job via sbatch and wait for it to finish."""
    try:
        subprocess.run(
            ["sbatch", "--wait", f"{job_name}.slm"],
            check=True,
        )
        print("[INFO] Job completed successfully.")
    except subprocess.CalledProcessError as e:
        print("[ERROR] xTB job failed:", e)
        raise

def main():
    parser = argparse.ArgumentParser(description="Automate xTB geometry optimisation via Slurm.")
    parser.add_argument("xyz", help="Input XYZ file")
    parser.add_argument("--chrg", type=int, default=0, help="Molecular charge")
    parser.add_argument("--uhf", type=int, default=0, help="Number of unpaired electrons")
    parser.add_argument("--solvent", default="none", help="Solvent for GBSA model")
    parser.add_argument("--jobname", default="xtb_job", help="Job name")
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
        ts=args.ts
    )

    submit_job(args.jobname)

if __name__ == "__main__":
    main()