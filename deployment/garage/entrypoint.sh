#!/bin/sh
set -e

trap 'echo "Received termination signal, stopping garage..."; kill -TERM "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID"; exit 0' TERM INT

echo "Starting garage server..."
/garage -c /etc/garage.toml server --single-node --default-bucket &
SERVER_PID=$!

sleep 2
if ! kill -0 "$SERVER_PID" 2>/dev/null; then
  echo "ERROR: garage server process exited immediately."
  wait "$SERVER_PID"
  exit 1
fi

COUNT=0
until /garage -c /etc/garage.toml status; do
  COUNT=$((COUNT + 1))
  if [ "$COUNT" -ge 20 ]; then
    echo "ERROR: garage status still failing after 20 tries. Giving up."
    kill -TERM "$SERVER_PID" 2>/dev/null
    exit 1
  fi
  echo "Waiting for garage to be ready... (attempt $COUNT)"
  sleep 2
done

/garage -c /etc/garage.toml bucket create "$S3_DVC_BUCKET" || true
/garage -c /etc/garage.toml bucket allow --read --write --owner --key "$GARAGE_DEFAULT_ACCESS_KEY" "$S3_DVC_BUCKET" || true

echo "Garage fully initialized."
wait "$SERVER_PID"