from contextlib import closing, contextmanager

import hashlib
import hmac
import math
import os
import pickle
import secrets
import sqlite3
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import pandas as pd
from flask import Flask, abort, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from job_pipeline_steps import INPUT_COLUMNS, JobInputError

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "job_runtime_pipeline.pkl"
payload = MODEL.read_bytes()
if hashlib.sha256(payload).hexdigest() != "eb688419f8af9fabe85800b341ccf828ed83f2f6df623b80f75bd6a7fd13f351":
    raise RuntimeError("The model differs from the verified project file.")
pipeline = pickle.loads(payload)
users = sorted(pipeline.named_steps["feature_engineer"].history_)
encoder = pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["encoder"]
choices = dict(zip(("queue", "job_mode", "gpu_enabled", "project_prefix"), encoder.categories_))
defaults = dict(zip(INPUT_COLUMNS, ("user0001", "2026-01-01T00:00", "4", "60", "", "batch-normal", "batch", "False", "AIU")))

app = Flask(__name__)
import json
settings = json.loads((ROOT / 'instance' / 'hosted.json').read_text(encoding='utf8'))
app.config.update(SECRET_KEY=settings['secret'], SESSION_COOKIE_SECURE=True,
                  SESSION_COOKIE_SAMESITE='Lax', MAX_CONTENT_LENGTH=32768)

@app.before_request
def protect_form():
    if request.method == "POST" and not hmac.compare_digest(request.form.get("token", "").encode(), session.get("token", "!").encode()):
        abort(400)
    session.setdefault("token", secrets.token_urlsafe(24))

@app.route("/", methods=["GET", "POST"])
def login():
    if session.get("user"):
        return redirect(url_for("predict"))
    error = None
    if request.method == "POST":
        name = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if check_password_hash(settings["password_hash"], password) and name == "demo":
            session.clear()
            session["user"] = name
            return redirect(url_for("predict"))
        error = "Check the classroom password and try again."
    return render_template("login.html", error=error)

@app.post("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

def duration(seconds):
    total = round(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return " ".join(part for part in (f"{hours} hr" if hours else "", f"{minutes} min" if minutes else "", f"{secs} sec" if secs or not total else "") if part)

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if not session.get("user"):
        return redirect(url_for("login"))
    fields = defaults.copy()
    result = error = None
    if request.method == "POST":
        fields = {key: request.form.get(key, "").strip() for key in INPUT_COLUMNS}
        try:
            if fields["user_id"] not in users:
                raise JobInputError("Choose a User ID from the saved history.")
            for key, values in choices.items():
                if fields[key] not in values:
                    raise JobInputError(f"Choose a listed value for {key.replace('_', ' ')}.")
            row = fields.copy()
            row["init_priority"] = row["init_priority"] or float("nan")
            if row["gpu_enabled"] == "Unknown":
                row["gpu_enabled"] = float("nan")
            estimate = float(pipeline.predict(pd.DataFrame([row], columns=INPUT_COLUMNS))[0])
            if not math.isfinite(estimate) or estimate < 0:
                raise JobInputError("The model could not produce a finite runtime for this job.")
            limit = float(row["wall_time"]) * 60
            scale = max(limit, estimate)
            diff = round(abs(limit - estimate))
            comparison = (f"{duration(diff)} {'before' if estimate < limit else 'beyond'} the requested limit." if diff else "At the requested limit (to the nearest second).")
            result = dict(estimate=duration(estimate), seconds=f"{estimate:,.2f}", limit=duration(limit),
                          estimate_width=100*estimate/scale, limit_width=100*limit/scale, comparison=comparison)
        except JobInputError as exc:
            error = str(exc)
    return render_template("predict.html", users=users, choices=choices, fields=fields, result=result, error=error)

