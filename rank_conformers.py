import os
import argparse
import subprocess

def read_crest_xyz(filename):
    """Parse crest_ensemble.xyz into list of conformers."""
    with open(filename, "r") as f:
        lines = [l.strip() for l in f.readlines()]
    
    conformers = []
    i = 0
    while i < len(lines):
        if lines[i].isdigit():
            natoms = int(lines[i])
            comment = lines[i+1]
            atoms = lines[i+2:i+2+natoms]
            conformers.append((comment, atoms))
            i += natoms + 2
        else:
            i += 1
    return conformers

def write_gjf(conf_num, comment, atoms, chrg, mult, qc_method, basis_set, solvent, dispersion, mem, cpus, outdir):
    """Write Gaussian .gjf input for a conformer."""
    g16_mem = int(round(mem / 1000, 0))
    if g16_mem == 1:
        pass
    else:
        g16_mem -= 1
    filename = os.path.join(outdir, f"conf_{conf_num}.gjf")
    with open(filename, "w") as f:
        f.write(f"%mem={g16_mem}GB\n")
        f.write(f"%nprocshared={cpus}\n")
        route_line = f"# {qc_method} {basis_set} scrf=(smd,solvent={solvent})"
        if dispersion and dispersion.lower() != "none":
            route_line += f" EmpiricalDispersion={dispersion}"
        f.write(f"{route_line}\n\n")
        f.write(f"conf_{conf_num}\n\n")
        f.write(f"{chrg} {mult}\n")
        for line in atoms:
            f.write(line + "\n")
        f.write("\n")
    return filename

def write_slurm_script(conf_num, gjf_file, qc_method, time, mem, cpus, partition="nodes", outdir="."):
    """Write a Slurm submission script Gaussian single-point."""
    job_name = f"conf_{conf_num}"
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

module purge
module load gaussian/16

echo "$GAUSS_SCRDIR"
mkdir -p "$GAUSS_SCRDIR"
chmod 700 "$GAUSS_SCRDIR"

echo "Running Gaussian single-point for {job_name}"
g16 < {gjf_file}

rm -rf $GAUSS_SCRDIR
"""
    filename = f"conf_{conf_num}.slm"
    with open(os.path.join(outdir, filename), "w") as f:
        f.write(script)
    return filename

def submit_job(slurm_script, outdir):
    """Submit job via sbatch and return its Slurm job ID."""
    try:
        result = subprocess.run(
            ["sbatch", "--parsable", slurm_script],
            cwd=outdir,
            check=True,
            capture_output=True,
            text=True,
        )

        job_id = result.stdout.strip().split(";")[0]

        print(f"[INFO] Submitted {slurm_script} as job {job_id}.")
        return job_id

    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to submit {slurm_script}: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Generate Gaussian inputs for CREST conformers and submit via Slurm.")
    parser.add_argument("ensemble", help="Path to CREST ensemble (usually crest_ensemble.xyz)")
    parser.add_argument("--chrg", type=int, default=0, help="Molecular charge")
    parser.add_argument("--mult", type=int, default=1, help="Spin multiplicity")
    parser.add_argument("--qc_method", default="M062X", help="Quantum chemistry method (e.g. M062X)")
    parser.add_argument("--basis_set", default="def2tzvp", help="Basis set")
    parser.add_argument("--solvent", default="none", help="Solvent for SMD model")
    parser.add_argument("--dispersion", choices=["GD2", "GD3", "GD3BJ", "none"], default="none", help="Empirical dispersion correction")
    parser.add_argument("--time", default="2:00:00", help="Required walltime")
    parser.add_argument("--mem", type=int, default=4000, help="Memory requirement (MB)")
    parser.add_argument("--cpus", type=int, default=4, help="Number of processors required")
    parser.add_argument("--max_confs", type=int, default=50, help="Maximum number of conformers to process")
    parser.add_argument("--partition", default="nodes", help="SLURM partition")

    args = parser.parse_args()

    conformers = read_crest_xyz(args.ensemble)
    total = len(conformers)
    print(f"[INFO] Found {total} conformers in {args.ensemble}")

    n_to_take = min(total, args.max_confs)
    selected = conformers[:n_to_take]
    print(f"[INFO] Taking top {n_to_take} conformers for single point calculations")

    outdir = "sps"
    os.makedirs(outdir, exist_ok=True)
    jobids_path = os.path.join(outdir, "jobids.txt")

    with open(jobids_path, "w") as job_file:
        for i, (comment, atoms) in enumerate(selected, start=1):
            gjf = write_gjf(
                conf_num=i,
                comment=comment,
                atoms=atoms,
                chrg=args.chrg,
                mult=args.mult,
                qc_method=args.qc_method,
                basis_set=args.basis_set,
                solvent=args.solvent,
                dispersion=args.dispersion,
                mem=args.mem,
                cpus=args.cpus,
                outdir=outdir
            )
            slurm = write_slurm_script(
                conf_num=i,
                gjf_file=os.path.basename(gjf),
                qc_method=args.qc_method,
                time=args.time,
                mem=args.mem,
                cpus=args.cpus,
                partition=args.partition,
                outdir=outdir
            )
            
            job_id = submit_job(slurm, outdir)
            job_file.write(f"conf_{i} {job_id}\n")
            job_file.flush()

    print("[INFO] All conformer jobs have been submitted.")

if __name__ == "__main__":
    main()