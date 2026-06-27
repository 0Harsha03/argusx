import sys
from pathlib import Path
import logging

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO)

from app.services.model_registry import ModelRegistry

def main():
    print("Testing ModelRegistry initialization...")
    registry = ModelRegistry()
    registry.load_all()
    
    status = registry.status()
    print("Registry Status:")
    for key, value in status.items():
        print(f"  {key}: {value}")
        
    if registry.is_ready:
        print("ModelRegistry loaded successfully!")
    else:
        print("ModelRegistry failed to load properly.")

if __name__ == "__main__":
    main()
