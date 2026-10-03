.PHONY: \
	install lock setup \
	up down logs ps clean \
	minio-status minio-buckets minio-bucket-info minio-health \
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
# MinIO (AIStor)
# ============================================================
# AIStor's server image does not bundle the `mc` client, so instead of
# `docker compose exec minio ...` (like Garage's binary), we reuse the
# `init-minio-bucket` service's image/credentials/network via
# `docker compose run` and call the AWS CLI against MinIO's S3-compatible
# endpoint. --no-deps avoids re-triggering the service's depends_on chain
# since minio is normally already running when you call these targets.

MINIO_EXEC = docker compose run --rm --no-deps --entrypoint aws init-minio-bucket --endpoint-url http://minio:9000

minio-status:
	@echo "== MinIO health =="
	@curl -sf http://localhost:9000/minio/health/live > /dev/null \
		&& echo "MinIO is reachable on :9000" \
		|| echo "MinIO is NOT reachable on :9000"

minio-buckets:
	@$(MINIO_EXEC) s3 ls

minio-bucket-info:
	@test -n "$(BUCKET)" || \
		(echo "Usage: make minio-bucket-info BUCKET=mlflow-artifacts" && exit 1)
	@$(MINIO_EXEC) s3api head-bucket --bucket $(BUCKET) \
		&& echo "Bucket '$(BUCKET)' exists and is reachable."

minio-health: minio-status
	@echo ""
	@echo "== Buckets =="
	@$(MINIO_EXEC) s3 ls


# ============================================================
# Training
# ============================================================

train:
	docker compose --profile train run --rm train