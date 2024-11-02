help: ## Display help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'


build-dev-env: ## Build development environment
	docker compose up --build -d dev

run-dev-env: ## Run development environment
	docker compose up -d dev;\
	docker exec -it dev bash
	# docker run -it --gpus all \
 #    --name dev \
 #    -v .:/home/tts \
 #    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
 #    nvcr.io/nvidia/pytorch:24.08-py3 bash

build-triton-env: ## Build triton environment
	docker compose up --build -d triton

run-triton-env: ## Run triton environment
	docker compose up -d triton;\
	docker exec -it triton bash


.PHONY: clean
