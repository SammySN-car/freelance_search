from .config import get_settings
from .discord_bot import create_bot


def main() -> None:
    """Run the Discord bot."""
    settings = get_settings()
    create_bot(settings).run(settings.discord_token)


if __name__ == "__main__":
    main()
