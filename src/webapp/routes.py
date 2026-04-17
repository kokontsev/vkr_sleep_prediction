from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from flask import (
    Flask,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)
from werkzeug.utils import secure_filename

from .services.db import get_prediction_history, save_prediction
from .services.explainer import get_global_top_features
from .services.feature_list import MODEL_FEATURES
from .services.predictor import SleepPredictor
from .services.preprocess import preprocess_uploaded_dataframe
from .services.validator import validate_uploaded_csv


def register_routes(app: Flask) -> None:
    predictor = SleepPredictor()

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/upload", methods=["GET", "POST"])
    def upload():
        if request.method == "GET":
            return render_template("upload.html", required_columns=MODEL_FEATURES)

        if "file" not in request.files:
            flash("Файл не был передан.", "error")
            return redirect(url_for("upload"))

        file = request.files["file"]
        filename = secure_filename(file.filename or "")

        if not filename:
            flash("Не выбрано имя файла.", "error")
            return redirect(url_for("upload"))

        file_bytes = file.read()

        validation = validate_uploaded_csv(
            filename=filename,
            file_bytes=file_bytes,
            required_columns=[
                "SEQN",
                "calendar_date",
                "sleep_efficiency",
            ],
            min_rows=1,
        )

        if not validation.is_valid:
            for err in validation.errors:
                flash(err, "error")
            for warn in validation.warnings:
                flash(warn, "warning")
            return redirect(url_for("upload"))

        df = validation.dataframe
        if df is None:
            flash("Не удалось прочитать данные.", "error")
            return redirect(url_for("upload"))

        try:
            prep = preprocess_uploaded_dataframe(df)
            prediction = predictor.predict_last_row(prep.inference_df)
            top_features = get_global_top_features(
                predictor.model,
                feature_names=prediction.used_features,
                top_n=5,
            )

            prediction_id = save_prediction(
                filename=filename,
                rows_total=len(prep.raw_df),
                rows_after_preprocessing=len(prep.inference_df),
                probability=prediction.probability,
                predicted_class=prediction.predicted_class,
                predicted_label=prediction.predicted_label,
                threshold=prediction.used_threshold,
                top_features=top_features,
            )

            return render_template(
                "result.html",
                prediction_id=prediction_id,
                filename=filename,
                rows_total=len(prep.raw_df),
                rows_after_preprocessing=len(prep.inference_df),
                probability=prediction.probability,
                predicted_class=prediction.predicted_class,
                predicted_label=prediction.predicted_label,
                threshold=prediction.used_threshold,
                top_features=top_features,
            )

        except Exception as exc:
            flash(f"Ошибка при обработке файла: {exc}", "error")
            return redirect(url_for("upload"))

    @app.route("/history")
    def history():
        rows = get_prediction_history(limit=100)

        parsed_rows = []
        for row in rows:
            row_dict = dict(row)
            raw_top = row_dict.get("top_features_json") or "[]"
            try:
                row_dict["top_features"] = json.loads(raw_top)
            except Exception:
                row_dict["top_features"] = []
            parsed_rows.append(row_dict)

        return render_template("history.html", history_rows=parsed_rows)

    @app.route("/about")
    def about():
        return render_template("about.html")