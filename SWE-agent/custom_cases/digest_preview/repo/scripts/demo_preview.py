from app.main import render_digest_preview


def main() -> None:
    preview = render_digest_preview("o'connor-smith", 12840.5, 3)
    print(preview)


if __name__ == "__main__":
    main()
