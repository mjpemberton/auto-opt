# auto-opt

Automation tools for computational chemistry geometry optimisation workflows with xTB and Gaussian16.

## Job restarts

If you manually stop `run_pipeline.py`, check `squeue -u "$USER"` and cancel any Slurm jobs it submitted before restarting.


## Error with duplicate pipeline launches

Each `run_auto_opt.sh` should use `flock`:

```bash
nohup flock -n .auto_opt.lock \
    python /path/to/auto_opt/run_pipeline.py \
    molecule.xyz --config molecule.json \
    </dev/null >/dev/null 2>&1 &
```

This prevents more than one pipeline from running in the same calculation directory. Without the lock, accidentally running `run_auto_opt.sh` twice can lead to duplicate Slurm jobs.

To check for running pipelines:

```bash
pgrep -af '/path/to/auto_opt/run_pipeline.py'
```

To see which directory a launcher belongs to:

```bash
readlink /proc/<PID>/cwd
```

To stop a launcher:

```bash
kill <PID>
```

The lock is released automatically when the launcher exits; `.auto_opt.lock` does not need to be deleted.

Stopping the launcher does **not** cancel Slurm jobs that it has already submitted. Check these with:

```bash
squeue -u "$USER"
```

and cancel an unwanted job with:

```bash
scancel <JOBID>
```

When launching several calculations, use:

```bash
for script in */run_auto_opt.sh; do
    dir="${script%/run_auto_opt.sh}"
    (cd "$dir" && ./run_auto_opt.sh)
done
```

Running this loop more than once is safe provided each `run_auto_opt.sh` uses `flock`.