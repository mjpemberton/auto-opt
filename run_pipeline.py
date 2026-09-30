import os
import sys
import json
import time
import hashlib
import shutil
import argparse
import subprocess
import traceback
from contextlib import redirect_stdout, redirect_stderr

STATE_FILE = ".auto_opt_state.json"
GENERATED_DIRS = ("xtb", "crest", "opt")

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

def sha256_file(path):
    """Return the SHA256 hash of a file."""
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()

def sha256_json(data):
    """Hash JSON content independent of whitespace/key ordering."""
    canonical = json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(canonical).hexdigest()

def make_new_state(args, cfg):
    return {
        "state_version": 1,
        "inputs": {
            "xyz": os.path.basename(args.xyz),
            "xyz_sha256": sha256_file(args.xyz),
            "config": os.path.basename(args.config),
            "config_sha256": sha256_json(cfg),
        },
        "stages": {
            "xtb": {"complete": False},
            "crest": {"complete": False},
            "rank": {"complete": False},
            "dft": {"complete": False},
        },
    }

def save_state(state):
    tmp_path = STATE_FILE + ".tmp"

    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=2)
        f.write("\n")

    os.replace(tmp_path, STATE_FILE)

def load_state():
    with open(STATE_FILE, "r") as f:
        return json.load(f)

def prepare_state(args, cfg):
    current_state = make_new_state(args, cfg)

    if args.fresh:
        print("[STATE] Starting fresh calculation.")

        for directory in GENERATED_DIRS:
            if os.path.isdir(directory):
                print(f"[STATE] Removing existing {directory}/")
                shutil.rmtree(directory)

        if os.path.exists(STATE_FILE):
            os.remove(STATE_FILE)

        save_state(current_state)
        return current_state

    if not os.path.exists(STATE_FILE):
        existing = [
            directory
            for directory in GENERATED_DIRS
            if os.path.exists(directory)
        ]

        if existing:
            raise RuntimeError(
                "Existing calculation found without state. "
                "Run with --fresh to discard it."
            )

        print("[STATE] No previous calculation found; starting new calculation.")
        save_state(current_state)
        return current_state

    state = load_state()

    if state.get("state_version") != 1:
        raise RuntimeError(
            "Unrecognised auto-opt state file. "
            "Run with --fresh to restart."
        )

    old_inputs = state.get("inputs", {})
    new_inputs = current_state["inputs"]

    if (
        old_inputs.get("xyz_sha256") != new_inputs["xyz_sha256"]
        or old_inputs.get("config_sha256") != new_inputs["config_sha256"]
    ):
        raise RuntimeError(
            "Inputs have changed since the previous run. "
            "Run with --fresh to restart."
        )

    print("[STATE] Existing state matches current inputs; resuming calculation.")
    return state

def gaussian_out_completed(path):
    if not os.path.isfile(path):
        return False
    try:
        with open(path, "r", errors="ignore") as f:
            return "Normal termination of Gaussian 16" in f.read()[-300000:]
    except Exception:
        return False

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

    if not os.path.isfile(xtbopt):
        raise RuntimeError(
            "xTB completed but xtbopt.xyz was not produced."
        )

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

    if not os.path.isfile(ensemble):
        raise RuntimeError(
            "CREST completed but crest_ensemble.xyz was not produced."
        )

    return ensemble

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

    jobids_path = os.path.join(sps_dir, "jobids.txt")
    jobs = read_conformer_jobs(jobids_path)

    if not jobs:
        raise RuntimeError("No conformer jobs were submitted.")

    conformers = [conformer for conformer, job_id in jobs]
    job_ids = [job_id for conformer, job_id in jobs]

    wait_for_slurm_jobs(job_ids)

    successful, failed = classify_conformers(
        sps_dir,
        conformers,
    )

    print(
        f"[INFO] Conformer ranking finished: "
        f"{len(successful)} successful, {len(failed)} failed."
    )

    if failed:
        print(
            "[WARN] Failed conformers: "
            + ", ".join(failed)
        )

    if not successful:
        raise RuntimeError(
            "All conformer calculations failed; "
            "cannot continue to DFT optimisation."
        )

    return sps_dir, conformers, successful, failed

def read_conformer_jobs(jobids_path):
    """Return [(conformer_name, job_id), ...] from jobids.txt."""
    jobs = []

    with open(jobids_path, "r") as f:
        for line in f:
            if not line.strip():
                continue

            conformer, job_id = line.split()
            jobs.append((conformer, job_id))

    return jobs

def wait_for_slurm_jobs(job_ids, poll=10):
    """Wait until all supplied Slurm job IDs have left the queue."""
    job_ids = set(job_ids)

    print(f"[WAIT] Waiting for {len(job_ids)} conformer jobs...")

    while True:
        result = subprocess.run(
            ["squeue", "-h", "-u", os.environ["USER"], "-o", "%i"],
            check=True,
            capture_output=True,
            text=True,
        )

        active_jobs = set(result.stdout.split())
        remaining = job_ids & active_jobs

        if not remaining:
            print("[OK] All conformer jobs have finished.")
            return

        time.sleep(poll)

def classify_conformers(sps_dir, conformers):
    successful = []
    failed = []

    for conformer in conformers:
        out_path = os.path.join(sps_dir, f"{conformer}.out")
        gjf_path = os.path.join(sps_dir, f"{conformer}.gjf")

        if (
            os.path.isfile(gjf_path)
            and gaussian_out_completed(out_path)
        ):
            successful.append(conformer)
        else:
            failed.append(conformer)

    return successful, failed

def ranking_output_hash(sps_dir, conformers):
    """Hash the complete set of expected conformer inputs and outputs."""
    h = hashlib.sha256()

    for conformer in sorted(conformers):
        for extension in (".gjf", ".out"):
            filename = f"{conformer}{extension}"
            path = os.path.join(sps_dir, filename)

            h.update(filename.encode("utf-8"))
            h.update(b"\0")

            if not os.path.isfile(path):
                h.update(b"MISSING")
                continue

            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)

    return h.hexdigest()

def stage_dft(cfg, sps_dir, successful):
    opt_dir = ensure_dir("opt")
    dft = cfg["dft"]

    if not successful:
        raise RuntimeError("No successful conformers available for final DFT optimisation.")

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

    cmd += ["--conformers"] + successful

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
        final_out = os.path.abspath(
            os.path.join(opt_dir, f"{jobname}_full.out")
        )
    else:
        final_out = os.path.abspath(
            os.path.join(opt_dir, f"{jobname}_opt.out")
        )

    if not gaussian_out_completed(final_out):
        raise RuntimeError(
            f"Final DFT output did not terminate normally: {final_out}"
        )

    return final_out

    
def main():
    parser = argparse.ArgumentParser(description="Run full xTB, CREST, conformer search and DFT workflow")
    parser.add_argument("xyz", help="Input XYZ file in current directory")
    parser.add_argument("--config", required=True, help="JSON config file")
    parser.add_argument("--fresh", action="store_true", help="Discard previous auto-opt outputs and start from scratch")
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
                state = prepare_state(args, cfg)

                xyz_path = os.path.abspath(args.xyz)

                ######################
                #         xTB        #
                ######################

                xtbopt = os.path.abspath(os.path.join("xtb", "xtbopt.xyz"))
                xtb_state = state["stages"]["xtb"]

                if xtb_state["complete"]:
                    if not os.path.isfile(xtbopt):
                        raise RuntimeError(
                            "State records xTB as complete, but xtb/xtbopt.xyz is missing. "
                            "Run with --fresh to restart."
                        )

                    if sha256_file(xtbopt) != xtb_state.get("output_sha256"):
                        raise RuntimeError(
                            "xtb/xtbopt.xyz has changed since the previous run. "
                            "Run with --fresh to restart."
                        )

                    print("[STATE] xTB already completed successfully; reusing xtbopt.xyz.")

                else:
                    # An incomplete previous xTB attempt should not leave files that can
                    # contaminate the new attempt.
                    if os.path.isdir("xtb"):
                        shutil.rmtree("xtb")

                    print("Beginning xTB optimisation")
                    xtbopt = stage_xtb(xyz_path, cfg)

                    state["stages"]["xtb"] = {
                        "complete": True,
                        "output_sha256": sha256_file(xtbopt),
                    }
                    save_state(state)

                    print("xTB optimisation complete\n")

                ######################
                #        CREST       #
                ######################

                ensemble = os.path.abspath(
                    os.path.join("crest", "crest_ensemble.xyz")
                )
                crest_state = state["stages"]["crest"]

                if crest_state["complete"]:
                    if not os.path.isfile(ensemble):
                        raise RuntimeError(
                            "State records CREST as complete, but "
                            "crest/crest_ensemble.xyz is missing. "
                            "Run with --fresh to restart."
                        )

                    if sha256_file(ensemble) != crest_state.get("output_sha256"):
                        raise RuntimeError(
                            "crest/crest_ensemble.xyz has changed since the previous run. "
                            "Run with --fresh to restart."
                        )

                    print(
                        "[STATE] CREST already completed successfully; "
                        "reusing crest_ensemble.xyz."
                    )

                else:
                    if os.path.isdir("crest"):
                        shutil.rmtree("crest")

                    print("Beginning CREST conformer search")
                    ensemble = stage_crest(cfg, xtbopt)

                    state["stages"]["crest"] = {
                        "complete": True,
                        "output_sha256": sha256_file(ensemble),
                    }
                    save_state(state)

                    print("CREST conformer search complete\n")

                #########################
                #   Conformer Ranking   #
                #########################

                sps_dir = os.path.join("crest", "sps")
                rank_state = state["stages"]["rank"]

                if rank_state["complete"]:
                    successful = rank_state.get("successful", [])
                    failed = rank_state.get("failed", [])

                    conformers = successful + failed

                    if not successful:
                        raise RuntimeError(
                            "State records no successful conformers. "
                            "Run with --fresh to restart."
                        )

                    if len(conformers) != rank_state.get("expected"):
                        raise RuntimeError(
                            "Conformer ranking state is inconsistent. "
                            "Run with --fresh to restart."
                        )

                    current_hash = ranking_output_hash(
                        sps_dir,
                        conformers,
                    )

                    if current_hash != rank_state.get("output_sha256"):
                        raise RuntimeError(
                            "Conformer ranking files have changed since the previous run. "
                            "Run with --fresh to restart."
                        )

                    print(
                        f"[STATE] Conformer ranking already complete: "
                        f"{len(successful)} successful, {len(failed)} failed."
                    )

                else:
                    if os.path.isdir(sps_dir):
                        shutil.rmtree(sps_dir)

                    if os.path.isdir("opt"):
                        shutil.rmtree("opt")

                    print("Beginning conformer ranking")

                    sps_dir, conformers, successful, failed = stage_rank(
                        cfg,
                        ensemble,
                    )

                    state["stages"]["rank"] = {
                        "complete": True,
                        "expected": len(conformers),
                        "successful": successful,
                        "failed": failed,
                        "output_sha256": ranking_output_hash(
                            sps_dir,
                            conformers,
                        ),
                    }

                    save_state(state)

                    print("Conformer ranking complete\n")

                ######################
                #         DFT        #
                ######################

                dft = cfg["dft"]
                jobname = dft.get("jobname", "opt")

                if dft.get("ts"):
                    final_out = os.path.abspath(
                        os.path.join("opt", f"{jobname}_full.out")
                    )
                else:
                    final_out = os.path.abspath(
                        os.path.join("opt", f"{jobname}_opt.out")
                    )

                dft_state = state["stages"]["dft"]

                if dft_state["complete"]:
                    if not gaussian_out_completed(final_out):
                        raise RuntimeError(
                            "State records DFT as complete, but the final Gaussian "
                            "output is missing or did not terminate normally. "
                            "Run with --fresh to restart."
                        )

                    if sha256_file(final_out) != dft_state.get("output_sha256"):
                        raise RuntimeError(
                            "Final DFT output has changed since the previous run. "
                            "Run with --fresh to restart."
                        )

                    print(
                        "[STATE] Final DFT optimisation already completed successfully."
                    )

                else:
                    if os.path.isdir("opt"):
                        shutil.rmtree("opt")

                    print("Beginning final DFT optimisation")

                    final_out = stage_dft(
                        cfg,
                        sps_dir,
                        successful,
                    )

                    state["stages"]["dft"] = {
                        "complete": True,
                        "output_sha256": sha256_file(final_out),
                    }

                    save_state(state)

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