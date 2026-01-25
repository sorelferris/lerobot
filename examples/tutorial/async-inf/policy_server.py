from lerobot.async_inference.configs import PolicyServerConfig
from lerobot.async_inference.policy_server import serve


def main():
    host = "0.0.0.0"  # something like "127.0.0.1" if you're exposing to localhost
    port = 8000  # something like 8080

    config = PolicyServerConfig(
        host=host,
        port=port,
    )
    serve(config)


if __name__ == "__main__":
    main()
