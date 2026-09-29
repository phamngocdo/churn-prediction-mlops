#!/usr/bin/env bash
set -euo pipefail

ENV_FILE=".env"
ENV_EXAMPLE=".env.example"
GARAGE_TOML="deployment/garage/garage.toml"
GARAGE_TOML_EXAMPLE="deployment/garage/garage.toml.example"

if [ ! -f "$ENV_FILE" ]; then
  cp "$ENV_EXAMPLE" "$ENV_FILE"
  echo "Created $ENV_FILE from $ENV_EXAMPLE"
fi

set_if_empty () {
  local key="$1"
  local value="$2"
  local current
  current=$(grep -E "^${key}=" "$ENV_FILE" | cut -d '=' -f2- || true)
  if [ -z "$current" ]; then
    if [[ "$OSTYPE" == "darwin"* ]]; then
      sed -i '' "s|^${key}=.*|${key}=${value}|" "$ENV_FILE"
    else
      sed -i "s|^${key}=.*|${key}=${value}|" "$ENV_FILE"
    fi
    echo "Generated ${key}"
  fi
}

set_if_empty "GARAGE_RPC_SECRET" "$(openssl rand -hex 32)"
set_if_empty "GARAGE_ADMIN_TOKEN" "$(openssl rand -base64 32)"
set_if_empty "GARAGE_METRICS_TOKEN" "$(openssl rand -base64 32)"
set_if_empty "GARAGE_DEFAULT_ACCESS_KEY" "GK$(openssl rand -hex 16)"
set_if_empty "GARAGE_DEFAULT_SECRET_KEY" "$(openssl rand -hex 32)"
set_if_empty "POSTGRES_PASSWORD" "$(openssl rand -hex 16)"

if [ ! -f "$GARAGE_TOML" ]; then
  set -a
  # shellcheck source=/dev/null
  source "$ENV_FILE"
  set +a
  envsubst < "$GARAGE_TOML_EXAMPLE" > "$GARAGE_TOML"
  echo "Created $GARAGE_TOML from template"
fi

echo "Setup complete. Review $ENV_FILE and $GARAGE_TOML if needed."