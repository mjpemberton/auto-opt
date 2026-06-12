import os
import sys
import json
import time
import argparse
import subprocess
import traceback
from contextlib import redirect_stdout, redirect_stderr

def run(cmd, cwd=None, capture=False):
    printable_cmd = " ".join(str(part) for part in cmd)
    print(f"[CMD] {printable_cmd}", flush=True)

    try:
        if capture:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            if result.stdout:
                print(result.stdout, end="", flush=True)

            if result.stderr:
                print(result.stderr, end="", file=sys.stderr, flush=True)

            return result

        return subprocess.run(
            cmd,
            cwd=cwd,
            check=True,
            text=True,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )

    except subprocess.CalledProcessError as exc:
        print(
            f"[ERROR] Command exited with status {exc.returncode}: "
            f"{printable_cmd}",
            file=sys.stderr,
            flush=True,
        )

        if exc.stdout:
            print(
                "[COMMAND STDOUT]",
                exc.stdout,
                sep="\n",
                file=sys.stderr,
                flush=True,
            )

        if exc.stderr:
            print(
                "[COMMAND STDERR]",
                exc.stderr,
                sep="\n",
                file=sys.stderr,
                flush=True,
            )

        raise

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path

def file_exists(path):
    return os.path.isfile(path)

def wait_for_file(path, poll=10, must_contain=None):
    print(f"[WAIT] for {os.path.basename(path)} ...")
    while True:
        if file_exists(path):
            if not must_contain:
                print(f"[OK] Found {path}")
                return
            try:
                with open(path, "r", errors="ignore") as f:
                    tail = f.read()[-300000:]
                if must_contain in tail:
                    print(f"[OK] {path} contains required text.")
                    return
            except Exception:
                pass
        time.sleep(poll)

def gaussian_out_completed(path):
    if not file_exists(path):
        return False
    try:
        with open(path, "r", errors="ignore") as f:
            return "Normal termination of Gaussian 16" in f.read()[-300000:]
    except Exception:
        return False

def wait_for_gaussian_dir(sps_dir, poll=10):
    print(f"[WAIT] Gaussian SPS jobs in {sps_dir}")
    while True:
        outs = [f for f in os.listdir(sps_dir) if f.endswith(".out")]
        if outs and all(gaussian_out_completed(os.path.join(sps_dir, o)) for o in outs):
            print("[OK] All SPS jobs completed.")
            return
        time.sleep(poll)

def stage_xtb(xyz_file, cfg):
    xtb_dir = ensure_dir("xtb")
    xtbopt = os.path.abspath(os.path.join(xtb_dir, "xtbopt.xyz"))

    xtb = cfg["xtb"]
    cmd = [
        sys.executable, "/home/i/mjp218/auto_opt/optimise_xtb.py",
        xyz_file,
        "--chrg", str(xtb["chrg"]),
        "--uhf", str(xtb["uhf"]),
        "--solvent", str(xtb["solvent"]),
        "--jobname", str(xtb["jobname"]),
    ]

    if xtb.get("ts"):
        cmd.append("--ts")
        if xtb.get("constraints"):
            cmd += ["--constraints"] + [str(c) for c in xtb["constraints"]]

    run(cmd, cwd=xtb_dir)
    wait_for_file(xtbopt)
    return(xtbopt)

def stage_crest(cfg, xtbopt_path):
    crest_dir = ensure_dir("crest")
    ensemble = os.path.abspath(os.path.join(crest_dir, "crest_ensemble.xyz"))
    crest = cfg["crest"]

    cmd = [
        sys.executable, "/home/i/mjp218/auto_opt/crest_search.py",
        xtbopt_path,
        "--chrg", str(crest["chrg"]),
        "--uhf", str(crest["uhf"]),
        "--solvent", str(crest["solvent"]),
        "--thresh", str(crest.get("thresh", 1.0)),
        "--time", str(crest["time"]),
        "--mem", str(crest["mem"]),
        "--cpus", str(crest["cpus"]),
        "--jobname", crest.get("jobname", "crest_job"),
    ]

    if crest.get("ts"):
        cmd.append("--ts")
        if crest.get("constraints"):
            cmd += ["--constraints"] + [str(c) for c in crest["constraints"]]

    run(cmd, cwd=crest_dir)
    wait_for_file(ensemble)
    return(ensemble)

def stage_rank(cfg, ensemble_path):
    crest_dir = "crest"
    sps_dir = ensure_dir(os.path.join(crest_dir, "sps"))
    rank = cfg["rank"]

    cmd = [
        sys.executable, "/home/i/mjp218/auto_opt/rank_conformers.py",
        ensemble_path,
        "--chrg", str(rank["chrg"]),
        "--mult", str(rank["mult"]),
        "--qc_method", str(rank["qc_method"]),
        "--basis_set", str(rank["basis_set"]),
        "--solvent", str(rank["solvent"]),
        "--dispersion", rank.get("dispersion", "none"),
        "--time", str(rank["time"]),
        "--mem", str(rank["mem"]),
        "--cpus", str(rank["cpus"]),
        "--max_confs", str(rank["max_confs"]),
    ]

    run(cmd, cwd=crest_dir)
    wait_for_gaussian_dir(sps_dir)
    return(sps_dir)

def stage_dft(cfg, sps_dir):
    opt_dir = ensure_dir("opt")
    dft = cfg["dft"]

    cmd = [
        sys.executable, "/home/i/mjp218/auto_opt/optimise_dft.py",
        "--sps_dir", os.path.join("..", sps_dir),
        "--qc_method", str(dft["qc_method"]),
        "--basis_set", str(dft["basis_set"]),
        "--solvent", str(dft["solvent"]),
        "--dispersion", dft.get("dispersion", "none"),
        "--time", str(dft["time"]),
        "--mem", str(dft["mem"]),
        "--cpus", str(dft["cpus"]),
        "--opt_dir", ".",
        "--jobname", dft.get("jobname", "opt"),
    ]

    if dft.get("chrg") is not None:
        cmd += ["--chrg", str(dft["chrg"])]
    if dft.get("mult") is not None:
        cmd += ["--mult", str(dft["mult"])]
    
    if dft.get("ts"):
        cmd.append("--ts")
        if dft.get("constraints"):
            cmd += ["--constraints"] + [str(c) for c in dft["constraints"]]

    run(cmd, cwd=opt_dir)

    jobname = dft.get("jobname", "opt")
    if dft.get("ts"):
        frozen_out = os.path.abspath(os.path.join(opt_dir, f"{jobname}_frozen.out"))
        full_out = os.path.abspath(os.path.join(opt_dir, f"{jobname}_full.out"))
        wait_for_file(frozen_out)
        wait_for_file(full_out)
        while not gaussian_out_completed(full_out):
            print("[INFO] Waiting for TS full optimisation completion...")
            time.sleep(30)
        print(f"[OK] TS full optimisation completed: {full_out}")
    else:
        out = os.path.abspath(os.path.join(opt_dir, f"{jobname}_opt.out"))
        wait_for_file(out)
        while not gaussian_out_completed(out):
            print("[INFO] Waiting for final optimisation completion...")
            time.sleep(30)
        print(f"[OK] Final DFT optimisation completed: {out}")

    
def main():
    parser = argparse.ArgumentParser(description="Run full xTB, CREST, conformer search and DFT workflow")
    parser.add_argument("xyz", help="Input XYZ file in current directory")
    parser.add_argument("--config", required=True, help="JSON config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = json.load(f)

    log_path = os.path.join(os.getcwd(), "auto_opt.log")
    completed_successfully = False

    with open(log_path, "w", buffering=1) as log:
        with redirect_stdout(log), redirect_stderr(log):
            print("--- Running auto-opt pipeline ---")
            print(f"Working directory: {os.getcwd()}")
            print(f"Input structure: {args.xyz}")
            print(f"Configuration: {args.config}")
            print()

            try:
                xyz_path = os.path.abspath(args.xyz)

                print("Beginning xTB optimisation")
                xtbopt = stage_xtb(xyz_path, cfg)
                print("xTB optimisation complete\n")

                print("Beginning CREST conformer search")
                ensemble = stage_crest(cfg, xtbopt)
                print("CREST conformer search complete\n")

                print("Beginning conformer ranking")
                sps_dir = stage_rank(cfg, ensemble)
                print("Conformer ranking complete\n")

                print("Beginning final DFT optimisation")
                stage_dft(cfg, sps_dir)
                print("Final DFT optimisation complete\n")

                completed_successfully = True
                print("[SUCCESS] Pipeline completed successfully.")

            except BaseException as exc:
                print(
                    f"\n[ERROR] Pipeline terminated with "
                    f"{type(exc).__name__}: {exc}",
                    file=sys.stderr,
                )
                print("[TRACEBACK]", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                raise

            finally:
                if not completed_successfully:
                    print(
                        "[STATUS] Pipeline did not complete successfully.",
                        file=sys.stderr,
                    )

                print("--- End of auto-opt run ---")
                log.flush()
                try:
                    os.fsync(log.fileno())
                except OSError:
                    pass

if __name__ == "__main__":
    main()