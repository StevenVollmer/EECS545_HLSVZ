from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import render_budget_snapshot


def main() -> None:
    preview = render_budget_snapshot("Dana", 999_950.0, -3.2, 3)
    print(preview)


if __name__ == "__main__":
    main()
