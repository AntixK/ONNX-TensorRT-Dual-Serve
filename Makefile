help: ## Display help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'


build-dev-env: ## Build development environment
	docker compose up --build -d dev

run-dev-env: ## Run development environment
	# docker compose up -d dev;\
	# docker exec -it dev bash
	docker run -it --gpus all \
    --name dev \
    -v .:/home/tts \
    --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    nvcr.io/nvidia/pytorch:24.08-py3 bash

build-triton-env: ## Build triton environment
	docker compose up --build -d triton

run-triton-env: ## Run triton environment
	docker compose up -d triton;\
	docker exec -it triton bash

docker run -it --gpus all -v .:/trt_optimize nvcr.io/nvidia/tensorrt:24.08-py3

docker run --gpus all --rm --runtime=nvidia --env CUDA_VISIBLE_DEVICES=0 -p 8000:8000 -p 8001:8001 -p 8002:8002 -v /home/antixk/Anand/Projects/ONNX-TensorRT-Dual-Serve/tts_triton:/models nvcr.io/nvidia/tritonserver:24.09-py3 tritonserver --model-repository=/models --strict-model-config=false


docker run --gpus all --rm --runtime=nvidia --env CUDA_VISIBLE_DEVICES=0 -p 8000:8000 -p 8001:8001 -p 8002:8002 -v /home/antixk/Anand/Projects/ONNX-TensorRT-Dual-Serve/tts_triton:/models nvcr.io/nvidia/tritonserver:24.09-py3 tritonserver --model-repository=/models --disable-auto-complete-config

.PHONY: clean
