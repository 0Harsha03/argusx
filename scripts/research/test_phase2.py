import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.routing.base_route import RouteResult

def main():
    print("Testing RouteResult instantiation...")
    rr = RouteResult(
        route_name="TestRoute",
        pattern_score=50.0,
        binary_decision=True,
        enforcement_action="BLOCK"
    )
    print(rr)
    print("binary_decision:", rr.binary_decision)
    print("enforcement_action:", rr.enforcement_action)

if __name__ == "__main__":
    main()
