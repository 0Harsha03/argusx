import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.model_registry import ModelRegistry
from app.services.adaptive_detection_pipeline import AdaptiveDetectionPipeline

def main():
    print("Loading ModelRegistry...")
    registry = ModelRegistry()
    registry.load_all()
    
    print("Instantiating AdaptiveDetectionPipeline...")
    pipeline = AdaptiveDetectionPipeline(registry)
    
    pi_route = pipeline._router.injection_route
    print("PromptInjectionRoute instantiated!")
    
    print("Checking member variables...")
    print(f"platt_db exists: {hasattr(pi_route, 'platt_db') and pi_route.platt_db is not None}")
    print(f"platt_rf exists: {hasattr(pi_route, 'platt_rf') and pi_route.platt_rf is not None}")
    
    print("Testing existing runtime execution...")
    res = pipeline.analyze("Hello world!")
    print(f"Decision: {res['decision']}")
    print(f"Final Score: {res['final_score']}")

if __name__ == "__main__":
    main()
