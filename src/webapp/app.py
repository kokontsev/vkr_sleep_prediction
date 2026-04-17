from __future__ import annotations

from pathlib import Path

from flask import Flask

from .routes import register_routes
from .services.db import init_db


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )

    app.config["SECRET_KEY"] = "sleep-prediction-secret-key"
    app.config["UPLOAD_FOLDER"] = str(Path("uploads").resolve())
    app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 10 MB

    Path(app.config["UPLOAD_FOLDER"]).mkdir(parents=True, exist_ok=True)

    init_db()
    register_routes(app)

    return app


app = create_app()


if __name__ == "__main__":
    app.run(debug=True)