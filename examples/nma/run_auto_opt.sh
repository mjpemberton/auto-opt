nohup flock -n .auto_opt.lock \
    python /home/i/mjp218/auto_opt/run_pipeline.py \
    nma_ts.xyz --config nma_ts.json \
    </dev/null >/dev/null 2>&1 &
