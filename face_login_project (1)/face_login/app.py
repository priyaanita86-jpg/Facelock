"""Face-first login: the password box unlocks only if the live face
matches the account owner's registered face (with blink liveness check)."""
import base64
import io
import json
import os
import re
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from functools import wraps

import face_recognition
import numpy as np
from flask import (Flask, g, jsonify, redirect, render_template,
                   request, session, url_for)
from PIL import Image
from werkzeug.security import check_password_hash, generate_password_hash

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "change-this-in-production")

DB_PATH = os.path.join(os.path.dirname(__file__), "users.db")

# ---- security settings ------------------------------------------------
MATCH_THRESHOLD = 0.50     # lower = stricter (library default is 0.60)
MAX_FACE_FAILS = 3         # failed face attempts before lockout
LOCK_MINUTES = 10
FACE_VALID_SECONDS = 120   # time allowed to type the password after face OK
MAX_PASSWORD_TRIES = 3
EAR_CLOSED = 0.21          # eye aspect ratio when eyes are closed
EAR_OPEN = 0.25            # eye aspect ratio when eyes are open
MIN_GOOD_FRAMES = 6


# ---- database ---------------------------------------------------------
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    with sqlite3.connect(DB_PATH) as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            face_encodings TEXT NOT NULL,
            failed_face INTEGER NOT NULL DEFAULT 0,
            locked_until TEXT
        );
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            event TEXT NOT NULL,
            ts TEXT NOT NULL
        );
        """)


def now():
    return datetime.now(timezone.utc)


def log_event(username, event):
    db = get_db()
    db.execute("INSERT INTO logs (username, event, ts) VALUES (?, ?, ?)",
               (username, event, now().isoformat(timespec="seconds")))
    db.commit()


def get_user(username):
    return get_db().execute("SELECT * FROM users WHERE username = ?",
                            (username,)).fetchone()


def lock_until(user):
    if user["locked_until"]:
        until = datetime.fromisoformat(user["locked_until"])
        if until > now():
            return until
    return None


def record_face_failure(user):
    db = get_db()
    fails = user["failed_face"] + 1
    locked = None
    if fails >= MAX_FACE_FAILS:
        locked = (now() + timedelta(minutes=LOCK_MINUTES)).isoformat()
        fails = 0
        log_event(user["username"], "account_locked")
    db.execute("UPDATE users SET failed_face = ?, locked_until = ? WHERE id = ?",
               (fails, locked, user["id"]))
    db.commit()


def reset_face_failures(user):
    db = get_db()
    db.execute("UPDATE users SET failed_face = 0, locked_until = NULL "
               "WHERE id = ?", (user["id"],))
    db.commit()


# ---- face helpers -----------------------------------------------------
def decode_frame(data_url):
    raw = base64.b64decode(data_url.split(",", 1)[1])
    return np.array(Image.open(io.BytesIO(raw)).convert("RGB"))


def eye_ratio(eye):
    """Eye aspect ratio from the 6 landmark points of one eye."""
    p = [np.array(pt, dtype=float) for pt in eye]
    vertical = np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])
    horizontal = 2.0 * np.linalg.norm(p[0] - p[3])
    return vertical / horizontal


def analyse_frame(img):
    """Return (face_location, eye_ratio) or None unless exactly 1 face."""
    locations = face_recognition.face_locations(img)
    if len(locations) != 1:
        return None
    marks = face_recognition.face_landmarks(img, locations)[0]
    ear = (eye_ratio(marks["left_eye"]) + eye_ratio(marks["right_eye"])) / 2
    return locations, ear


def encode(img, locations):
    return face_recognition.face_encodings(img, known_face_locations=locations)[0]


def check_face(user, frames):
    """Return (ok, reason). Checks identity match AND a blink (liveness)."""
    good = []
    for data_url in frames:
        try:
            img = decode_frame(data_url)
        except Exception:
            continue
        result = analyse_frame(img)
        if result:
            good.append((img, result[0], result[1]))
    if len(good) < MIN_GOOD_FRAMES:
        return False, "no_clear_face"

    # identity: compare up to 5 open-eye frames with the stored encodings
    stored = [np.array(e) for e in json.loads(user["face_encodings"])]
    open_frames = [g_ for g_ in good if g_[2] >= EAR_OPEN][:5]
    if not open_frames:
        return False, "no_clear_face"
    distances = []
    for img, locs, _ in open_frames:
        enc = encode(img, locs)
        distances.append(float(np.min(face_recognition.face_distance(stored, enc))))
    if float(np.median(distances)) > MATCH_THRESHOLD:
        return False, "mismatch"

    # liveness: the eyes must close and re-open during the capture
    ears = [g_[2] for g_ in good]
    if not (min(ears) < EAR_CLOSED and max(ears) >= EAR_OPEN):
        return False, "no_blink"
    return True, "ok"


# ---- pages ------------------------------------------------------------
def login_required(view):
    @wraps(view)
    def wrapper(*a, **kw):
        if "user" not in session:
            return redirect(url_for("login"))
        return view(*a, **kw)
    return wrapper


@app.route("/")
def index():
    return redirect(url_for("dashboard" if "user" in session else "login"))


@app.route("/register")
def register():
    return render_template("register.html")


@app.post("/api/register")
def api_register():
    data = request.get_json(silent=True) or {}
    username = str(data.get("username", "")).strip().lower()
    password = str(data.get("password", ""))
    frames = data.get("frames", [])
    if not re.fullmatch(r"[a-z0-9_.]{3,30}", username):
        return jsonify(ok=False, message="Username: 3-30 letters, digits, _ or .")
    if len(password) < 8:
        return jsonify(ok=False, message="Password must be at least 8 characters.")
    if not 3 <= len(frames) <= 12:
        return jsonify(ok=False, message="Capture the face again.")
    if get_user(username):
        return jsonify(ok=False, message="That username is taken.")

    encodings = []
    for data_url in frames:
        try:
            img = decode_frame(data_url)
        except Exception:
            continue
        result = analyse_frame(img)
        if result and result[1] >= EAR_OPEN:      # one face, eyes open
            encodings.append(encode(img, result[0]).tolist())
    if len(encodings) < 3:
        return jsonify(ok=False, message="Need 3+ clear frames showing only "
                       "your face with eyes open. Improve the lighting and retry.")

    db = get_db()
    db.execute("INSERT INTO users (username, password_hash, face_encodings) "
               "VALUES (?, ?, ?)",
               (username, generate_password_hash(password),
                json.dumps(encodings[:5])))
    db.commit()
    log_event(username, "registered")
    return jsonify(ok=True, message="Account created. You can log in now.")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip().lower()
        session.clear()
        session["pending_user"] = username
        return redirect(url_for("face_step"))
    return render_template("login.html")


@app.route("/face")
def face_step():
    if "pending_user" not in session:
        return redirect(url_for("login"))
    return render_template("face.html", username=session["pending_user"])


@app.post("/api/verify-face")
def api_verify_face():
    username = session.get("pending_user")
    if not username:
        return jsonify(ok=False, message="Start from the login page."), 400
    frames = (request.get_json(silent=True) or {}).get("frames", [])
    if not 6 <= len(frames) <= 30:
        return jsonify(ok=False, message="Bad capture. Try again."), 400

    user = get_user(username)
    if user is None:                       # same message as a wrong face
        log_event(username, "face_fail_unknown_user")
        return jsonify(ok=False, message="Face verification failed.")

    until = lock_until(user)
    if until:
        return jsonify(ok=False, locked=True,
                       message=f"Too many failed attempts. Locked until "
                               f"{until:%H:%M} UTC.")

    ok, reason = check_face(user, frames)
    if ok:
        reset_face_failures(user)
        session["face_ok_user"] = username
        session["face_ok_until"] = time.time() + FACE_VALID_SECONDS
        session["pw_tries"] = 0
        log_event(username, "face_ok")
        return jsonify(ok=True, redirect=url_for("password_step"))

    log_event(username, f"face_fail_{reason}")
    record_face_failure(user)
    messages = {
        "no_clear_face": "Could not see exactly one clear face. Check lighting.",
        "mismatch": "Face verification failed.",
        "no_blink": "Liveness check failed. Blink once while the camera runs.",
    }
    return jsonify(ok=False, message=messages[reason])


def face_is_verified():
    return (session.get("face_ok_user") == session.get("pending_user")
            and session.get("face_ok_until", 0) > time.time())


@app.route("/password", methods=["GET", "POST"])
def password_step():
    if not face_is_verified():
        return redirect(url_for("login"))      # no face check, no password box
    error = None
    if request.method == "POST":
        username = session["pending_user"]
        user = get_user(username)
        if user and check_password_hash(user["password_hash"],
                                        request.form.get("password", "")):
            session.clear()
            session["user"] = username
            log_event(username, "login_success")
            return redirect(url_for("dashboard"))
        session["pw_tries"] = session.get("pw_tries", 0) + 1
        log_event(username, "password_fail")
        if session["pw_tries"] >= MAX_PASSWORD_TRIES:
            session.clear()                   # must pass the face check again
            return redirect(url_for("login"))
        error = "Wrong password."
    return render_template("password.html", error=error)


@app.route("/dashboard")
@login_required
def dashboard():
    rows = get_db().execute(
        "SELECT event, ts FROM logs WHERE username = ? ORDER BY id DESC LIMIT 10",
        (session["user"],)).fetchall()
    return render_template("dashboard.html", user=session["user"], logs=rows)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


if __name__ == "__main__":
    init_db()
    # The webcam only works on localhost or HTTPS.
    app.run(debug=False)
