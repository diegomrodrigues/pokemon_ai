.PHONY: build run stop clean shell local-run help

help:
	@echo "Available commands:"
	@echo "  make build      - Build the Docker image"
	@echo "  make run        - Run the application in Docker"
	@echo "  make stop       - Stop the running Docker container"
	@echo "  make clean      - Remove Docker containers and images"
	@echo "  make shell      - Open a shell in the running container"
	@echo "  make local-run  - Run the application locally"

build:
	docker-compose build

run:
	docker-compose up

stop:
	docker-compose down

clean:
	docker-compose down --rmi all
	docker system prune -f

shell:
	docker-compose exec pokemon-api bash

local-run:
	python run.py 