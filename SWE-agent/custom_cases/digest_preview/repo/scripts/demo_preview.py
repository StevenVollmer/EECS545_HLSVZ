from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import render_digest_preview


def main() -> None:
    preview = render_digest_preview("o'connor-smith", 12840.5, 3)
    print(preview)


if __name__ == "__main__":
    main()
