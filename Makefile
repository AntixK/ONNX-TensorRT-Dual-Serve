help: ## Display help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'


build-dev-env: ## Build development environment
	docker compose up --build -d

run-dev-env: ## Run development environment
	docker compose up -d;\
	docker exec -it dev bash

.PHONY: clean