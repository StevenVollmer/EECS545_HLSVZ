from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main import roster_signature


def main() -> None:
    print(roster_signature('Infra', "d'arcy-lee"))


if __name__ == '__main__':
    main()
