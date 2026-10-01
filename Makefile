.PHONY: \
	install lock setup \
	up down logs ps clean \
	garage-status garage-layout garage-buckets garage-bucket-info \
	garage-keys garage-key-info garage-health \
	train

# ============================================================
# Environment
# ============================================================

install:
	uv venv --python 3.12
	uv pip install --no-cache -e ".[dev]"

lock:
	uv pip freeze > requirements-lock.txt

setup:
	@chmod +x scripts/setup_env.sh
	@./scripts/setup_env.sh


# ============================================================
# Docker lifecycle
# ============================================================

up: setup
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f --tail=100

ps:
	docker compose ps

clean:
	docker compose down -v --remove-orphans


# ============================================================
# Garage
# ============================================================

GARAGE_EXEC = docker compose exec -T garage /garage -c /etc/garage.toml

garage-status:
	@$(GARAGE_EXEC) status

garage-layout:
	@$(GARAGE_EXEC) layout show

garage-buckets:
	@$(GARAGE_EXEC) bucket list

garage-bucket-info:
	@test -n "$(BUCKET)" || \
		(echo "Usage: make garage-bucket-info BUCKET=dvc-store" && exit 1)
	@$(GARAGE_EXEC) bucket info $(BUCKET)

garage-keys:
	@$(GARAGE_EXEC) key list

garage-key-info:
	@test -n "$(KEY)" || \
		(echo "Usage: make garage-key-info KEY=dev-key" && exit 1)
	@$(GARAGE_EXEC) key info $(KEY)

garage-health:
	@echo "== Cluster status =="
	@$(GARAGE_EXEC) status
	@echo ""
	@echo "== Buckets =="
	@$(GARAGE_EXEC) bucket list


# ============================================================
# Training
# ============================================================

train:
	docker compose --profile train run --rm train