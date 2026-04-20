from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.presenters.contact_card import render_contact_card


def main() -> None:
    print(render_contact_card("o'brien-smith", "Core Platform"))


if __name__ == "__main__":
    main()
