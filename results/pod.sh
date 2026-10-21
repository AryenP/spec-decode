#!/bin/bash
# helper: results/pod.sh ssh '<cmd>'  |  results/pod.sh push  |  results/pod.sh pull <tag>
HOST=${POD_HOST:-213.173.111.9}; PORT=${POD_PORT:-39080}
OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR)
case "$1" in
  ssh)  shift; ssh "${OPTS[@]}" -p "$PORT" root@"$HOST" "$@" ;;
  push) rsync -az -e "ssh ${OPTS[*]} -p $PORT" --exclude .venv --exclude .git --exclude __pycache__ --exclude .pytest_cache \
          --exclude 'results/*.jsonl' --exclude 'results/*.log' --exclude 'results/*.md' --exclude 'results/*.done' \
          "$(dirname "$0")/../" root@"$HOST":/workspace/spec-decode/ ;;
  pull) rsync -az -e "ssh ${OPTS[*]} -p $PORT" root@"$HOST":/workspace/spec-decode/results/ "$(dirname "$0")/" ;;
esac
