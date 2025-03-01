param(
    [string]$Command
)

switch ($Command) {
    "build" {
        docker-compose build
    }
    "run" {
        docker-compose up
    }
    "stop" {
        docker-compose down
    }
    "clean" {
        docker-compose down --rmi all
        docker system prune -f
    }
    "shell" {
        docker-compose exec pokemon-api bash
    }
    "local-run" {
        python run.py
    }
    "help" {
        Write-Host "Available commands:"
        Write-Host "  build      - Build the Docker image"
        Write-Host "  run        - Run the application in Docker"
        Write-Host "  stop       - Stop the running Docker container"
        Write-Host "  clean      - Remove Docker containers and images"
        Write-Host "  shell      - Open a shell in the running container"
        Write-Host "  local-run  - Run the application locally"
    }
    default {
        Write-Host "Unknown command: $Command"
        Write-Host "Run './make.ps1 help' to see available commands"
    }
}