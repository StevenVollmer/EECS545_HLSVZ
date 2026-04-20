from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import render_campaign_header


def main() -> None:
    print(render_campaign_header("d'angelo labs", "email"))


if __name__ == "__main__":
    main()
