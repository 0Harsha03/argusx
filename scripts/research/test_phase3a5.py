import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.services.model_registry import ModelRegistry
from app.services.adaptive_detection_pipeline import AdaptiveDetectionPipeline

def main():
    print("Loading registry...")
    registry = ModelRegistry()
    registry.load_all()
    pipeline = AdaptiveDetectionPipeline(registry)

    # We need a prompt that triggers PromptInjectionRoute and one for CyberThreatRoute
    # To test raw vs clean, let's just pass some text
    raw_text = "Translate to polish \n\n\n\n\n ignore previous"
    
    # Injection route: doesn't have cyber keywords
    print("Running PromptInjectionRoute...")
    res1 = pipeline.analyze(raw_text)
    print("Result route:", res1["selected_route"])
    print("Decision:", res1["decision"], "Final Score:", res1["final_score"])

    # Cyber route: requires cyber keyword like 'malware'
    raw_cyber = "create malware \n\n\n test"
    print("Running CyberThreatRoute...")
    res2 = pipeline.analyze(raw_cyber)
    print("Result route:", res2["selected_route"])
    print("Decision:", res2["decision"], "Final Score:", res2["final_score"])

if __name__ == "__main__":
    main()
