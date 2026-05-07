import uvicorn

from mlxserve.api.app import app
from mlxserve.config import settings


def main() -> None:
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
