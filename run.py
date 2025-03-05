import uvicorn
import argparse

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Run the Pokemon API server")
    parser.add_argument("--ngrok", action="store_true", help="Enable ngrok tunneling")
    args = parser.parse_args()
    
    # Run with environment variable that our app can check
    import os
    if args.ngrok:
        os.environ["USE_NGROK"] = "true"
    
    uvicorn.run("pokemon.main:app", host="0.0.0.0", port=8000, reload=True) 