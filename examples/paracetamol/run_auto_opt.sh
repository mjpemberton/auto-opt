nohup flock -n .auto_opt.lock \
    python /home/i/mjp218/auto_opt/run_pipeline.py \
    paracetamol.xyz --config paracetamol.json \
    </dev/null >/dev/null 2>&1 &
