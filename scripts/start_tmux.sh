#!/usr/bin/env bash
cd /home/ubuntu/demsni
tmux new-session -d -s trit_train "bash -c './launch_4x4090.sh 2>&1 | tee /home/ubuntu/demsni/train_run.log'"
echo "TMUX_SESSION_LAUNCHED"
