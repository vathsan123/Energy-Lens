import os
import io
import json
import calendar
from datetime import datetime
from uuid import UUID

import joblib
import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, render_template, request, redirect, url_for
from flask_login import LoginManager, login_user, logout_user, current_user
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from database import db, get_database_url
from models import User, Household, MeterReading, ApplianceReading, LoginEvent


# =====================================================
# CONFIG
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATA_PATH = os.path.join(ROOT_DIR, "TN_Electricity_Appliance_Unified.csv")
MODEL_PATH = os.path.join(ROOT_DIR, "ExtraTrees_TNEB_Model.joblib")
METADATA_PATH = os.path.join(ROOT_DIR, "ExtraTrees_TNEB_Model_Metadata.json")

RANDOM_STATE = 42
DAILY_MAE_FOR_RANGE = 1.217
MIN_READINGS_FOR_FORECAST = 60

APPLIANCE_COLS = [
    "Refrigerator",
    "Lights_Fans",
    "TV_Monitor",
    "AC",
    "Water_Heater",
    "Washing_Machine",
    "Motor_Pump",
]


# =====================================================
# APP SETUP
# =====================================================

app = Flask(__name__)

app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY",
    "change-this-secret-key-for-production",
)
app.config["SQLALCHEMY_DATABASE_URI"] = get_database_url()
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

login_manager = LoginManager()
login_manager.login_view = "login"
login_manager.login_message = "Please login to access your dashboard."
login_manager.init_app(app)

DATA_DF = None
MODEL = None
FEATURE_COLS = None
MODEL_STATUS = None
TARGET_TRANSFORM = "log1p"


# =====================================================
# AUTH
# =====================================================

PUBLIC_ENDPOINTS = {
    "login",
    "signup",
    "forgot_password",
    "static",
    "favicon",
    "health",
}


@login_manager.user_loader
def load_user(user_id):
    try:
        return db.session.get(User, UUID(str(user_id)))
    except Exception:
        return None


@app.before_request
def require_login():
    endpoint = request.endpoint

    if endpoint in PUBLIC_ENDPOINTS:
        return None

    if request.path.startswith("/static/"):
        return None

    if current_user.is_authenticated:
        return None

    if request.path.startswith("/api/"):
        return jsonify({"error": "Authentication required", "redirect": "/login"}), 401

    return redirect(url_for("login", next=request.path))


def record_login_event(email, status, user=None):
    try:
        event = LoginEvent(
            user_id=user.id if user else None,
            email=email,
            login_status=status,
            ip_address=request.remote_addr,
            user_agent=request.headers.get("User-Agent"),
        )
        db.session.add(event)
    except Exception:
        # Login event storage must never block login itself.
        pass


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    error = None

    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        email = request.form.get("email", "").strip().lower()
        phone = request.form.get("phone", "").strip()
        password = request.form.get("password", "").strip()

        household_name = request.form.get("household_name", "My Home").strip()
        city = request.form.get("city", "").strip()
        district = request.form.get("district", "").strip()
        eb_service_number = request.form.get("eb_service_number", "").strip()
        billing_cycle_start_date = request.form.get("billing_cycle_start_date", "").strip()

        if not full_name or not email or not password:
            error = "Name, email and password are required."
        elif User.query.filter_by(email=email).first():
            error = "An account with this email already exists."
        else:
            try:
                user = User(full_name=full_name, email=email, phone=phone)
                user.set_password(password)
                db.session.add(user)
                db.session.flush()

                cycle_date = None
                if billing_cycle_start_date:
                    cycle_date = datetime.strptime(billing_cycle_start_date, "%Y-%m-%d").date()

                household = Household(
                    user_id=user.id,
                    household_name=household_name or "My Home",
                    city=city,
                    district=district,
                    state="Tamil Nadu",
                    eb_service_number=eb_service_number,
                    billing_cycle_start_date=cycle_date,
                )
                db.session.add(household)

                login_user(user)
                user.last_login_at = datetime.utcnow()
                record_login_event(email, "success", user)
                db.session.commit()

                return redirect(url_for("index"))
            except Exception as exc:
                db.session.rollback()
                error = f"Could not create account: {exc}"

    return render_template("signup.html", error=error)


@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    error = None
    email_value = ""

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "").strip()
        email_value = email

        user = User.query.filter_by(email=email).first()
        is_active = bool(getattr(user, "is_active", True)) if user else False

        if user and is_active and user.check_password(password):
            login_user(user)
            user.last_login_at = datetime.utcnow()
            record_login_event(email, "success", user)
            db.session.commit()

            next_url = request.args.get("next") or url_for("index")
            if not next_url.startswith("/"):
                next_url = url_for("index")
            return redirect(next_url)

        record_login_event(email, "failed", user)
        db.session.commit()
        error = "Invalid email or password. Please try again."

    return render_template(
        "login.html",
        error=error,
        email_value=email_value,
        demo_email="",
        demo_password="",
    )


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    message = None
    error = None

    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        user = User.query.filter_by(email=email).first()

        if user:
            message = (
                "Password reset request received. For production, this should send "
                "a secure reset link to your email."
            )
        else:
            error = "No account found with this email."

    return render_template("forgot_password.html", message=message, error=error)


@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for("login"))


@app.route("/favicon.ico")
def favicon():
    return "", 204


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


# =====================================================
# HOUSEHOLD DATA
# =====================================================

def get_current_household():
    household = (
        Household.query
        .filter_by(user_id=current_user.id)
        .order_by(Household.created_at.asc())
        .first()
    )

    if household:
        return household

    household = Household(
        user_id=current_user.id,
        household_name="My Home",
        state="Tamil Nadu",
    )
    db.session.add(household)
    db.session.commit()
    return household

# =====================================================
# PROFILE / SETTINGS ROUTES
# =====================================================

def parse_profile_date(value):
    value = (value or "").strip()

    if not value:
        return None

    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except Exception:
        return None


def set_if_model_has_attr(obj, attr_name, value):
    """
    Safely set an attribute only if the SQLAlchemy model has that column.
    This prevents errors if a column exists in PostgreSQL but is not defined
    in models.py, or vice versa.
    """
    if hasattr(obj, attr_name):
        setattr(obj, attr_name, value)


@app.route("/profile", methods=["GET", "POST"])
def profile():
    household = get_current_household()

    error = None
    message = None

    if request.method == "POST":
        try:
            # -----------------------------
            # User details
            # -----------------------------
            full_name = request.form.get("full_name", "").strip()
            phone = request.form.get("phone", "").strip()

            if full_name:
                set_if_model_has_attr(current_user, "full_name", full_name)

            set_if_model_has_attr(current_user, "phone", phone)
            set_if_model_has_attr(current_user, "updated_at", datetime.utcnow())

            # -----------------------------
            # Household details
            # -----------------------------
            household_name = request.form.get("household_name", "My Home").strip() or "My Home"
            address = request.form.get("address", "").strip()
            city = request.form.get("city", "").strip()
            district = request.form.get("district", "").strip()
            state = request.form.get("state", "Tamil Nadu").strip() or "Tamil Nadu"
            eb_service_number = request.form.get("eb_service_number", "").strip()
            billing_cycle_start_date = request.form.get("billing_cycle_start_date", "").strip()

            set_if_model_has_attr(household, "household_name", household_name)
            set_if_model_has_attr(household, "address", address)
            set_if_model_has_attr(household, "city", city)
            set_if_model_has_attr(household, "district", district)
            set_if_model_has_attr(household, "state", state)
            set_if_model_has_attr(household, "eb_service_number", eb_service_number)
            set_if_model_has_attr(
                household,
                "billing_cycle_start_date",
                parse_profile_date(billing_cycle_start_date),
            )
            set_if_model_has_attr(household, "updated_at", datetime.utcnow())

            db.session.commit()
            message = "Profile and household settings updated successfully."

        except Exception as exc:
            db.session.rollback()
            error = f"Could not update profile: {exc}"

    return render_template(
        "profile.html",
        household=household,
        error=error,
        message=message,
    )


@app.route("/api/profile", methods=["GET"])
def api_get_profile():
    household = get_current_household()

    billing_date = getattr(household, "billing_cycle_start_date", None)

    return jsonify({
        "user": {
            "id": str(current_user.id),
            "fullName": getattr(current_user, "full_name", ""),
            "email": getattr(current_user, "email", ""),
            "phone": getattr(current_user, "phone", ""),
            "createdAt": current_user.created_at.isoformat()
                if getattr(current_user, "created_at", None)
                else None,
            "lastLoginAt": current_user.last_login_at.isoformat()
                if getattr(current_user, "last_login_at", None)
                else None,
        },
        "household": {
            "id": str(household.id),
            "householdName": getattr(household, "household_name", ""),
            "address": getattr(household, "address", ""),
            "city": getattr(household, "city", ""),
            "district": getattr(household, "district", ""),
            "state": getattr(household, "state", ""),
            "ebServiceNumber": getattr(household, "eb_service_number", ""),
            "billingCycleStartDate": billing_date.isoformat() if billing_date else None,
        },
    })


@app.route("/api/profile", methods=["POST"])
def api_update_profile():
    household = get_current_household()
    payload = request.get_json(force=True) or {}

    try:
        user_payload = payload.get("user", {}) or {}
        household_payload = payload.get("household", {}) or {}

        full_name = str(user_payload.get("fullName", "")).strip()
        phone = str(user_payload.get("phone", "")).strip()

        if full_name:
            set_if_model_has_attr(current_user, "full_name", full_name)

        set_if_model_has_attr(current_user, "phone", phone)
        set_if_model_has_attr(current_user, "updated_at", datetime.utcnow())

        household_name = str(
            household_payload.get(
                "householdName",
                getattr(household, "household_name", "My Home") or "My Home",
            )
        ).strip() or "My Home"

        address = str(household_payload.get("address", "")).strip()
        city = str(household_payload.get("city", "")).strip()
        district = str(household_payload.get("district", "")).strip()
        state = str(household_payload.get("state", "Tamil Nadu")).strip() or "Tamil Nadu"
        eb_service_number = str(household_payload.get("ebServiceNumber", "")).strip()
        cycle_date = household_payload.get("billingCycleStartDate")

        set_if_model_has_attr(household, "household_name", household_name)
        set_if_model_has_attr(household, "address", address)
        set_if_model_has_attr(household, "city", city)
        set_if_model_has_attr(household, "district", district)
        set_if_model_has_attr(household, "state", state)
        set_if_model_has_attr(household, "eb_service_number", eb_service_number)
        set_if_model_has_attr(
            household,
            "billing_cycle_start_date",
            parse_profile_date(cycle_date),
        )
        set_if_model_has_attr(household, "updated_at", datetime.utcnow())

        db.session.commit()

        return jsonify({
            "success": True,
            "message": "Profile updated successfully.",
        })

    except Exception as exc:
        db.session.rollback()
        return jsonify({
            "success": False,
            "error": str(exc),
        }), 400

def actual_reading_count(df):
    if df is None or df.empty:
        return 0
    return int(df.attrs.get("actual_readings_count", len(df)))


def load_current_household_dataframe():
    household = get_current_household()

    meter_rows = (
        MeterReading.query
        .filter_by(household_id=household.id)
        .order_by(MeterReading.reading_date.asc(), MeterReading.created_at.asc())
        .all()
    )

    if not meter_rows:
        empty = pd.DataFrame()
        empty.attrs["actual_readings_count"] = 0
        return empty

    appliance_rows = (
        ApplianceReading.query
        .filter_by(household_id=household.id)
        .order_by(ApplianceReading.reading_date.asc())
        .all()
    )

    appliance_map = {row.reading_date: row for row in appliance_rows}
    records = []

    for meter in meter_rows:
        app_row = appliance_map.get(meter.reading_date)
        records.append({
            "Date": pd.to_datetime(meter.reading_date),
            "Total_Units": float(meter.total_units or 0),
            "Refrigerator": float(app_row.refrigerator or 0) if app_row else 0.0,
            "Lights_Fans": float(app_row.lights_fans or 0) if app_row else 0.0,
            "TV_Monitor": float(app_row.tv_monitor or 0) if app_row else 0.0,
            "AC": float(app_row.ac or 0) if app_row else 0.0,
            "Water_Heater": float(app_row.water_heater or 0) if app_row else 0.0,
            "Washing_Machine": float(app_row.washing_machine or 0) if app_row else 0.0,
            "Motor_Pump": float(app_row.motor_pump or 0) if app_row else 0.0,
        })

    df = pd.DataFrame(records)
    df = df.sort_values("Date")

    # If duplicates exist for the same day, keep the latest value for that date.
    agg_map = {"Total_Units": "last"}
    for col in APPLIANCE_COLS:
        agg_map[col] = "last"
    df = df.groupby("Date", as_index=False).agg(agg_map)

    real_count = int(len(df))
    df = df.set_index("Date")
    df.index = pd.to_datetime(df.index)

    full_index = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full_index)
    df.index.name = "Date"

    df["Total_Units"] = pd.to_numeric(df["Total_Units"], errors="coerce")
    df["Total_Units"] = df["Total_Units"].interpolate(limit_direction="both").fillna(0)

    for col in APPLIANCE_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["Year"] = df.index.year
    df["Month"] = df.index.month
    df["Month_Name"] = df.index.strftime("%b")
    df["Day"] = df.index.day
    df["DayOfWeek"] = df.index.dayofweek
    df["Day_Name"] = df.index.strftime("%a")
    df["Week"] = df.index.isocalendar().week.astype(int)
    df["IsWeekend"] = (df.index.dayofweek >= 5).astype(int)
    df["Is_Summer"] = df["Month"].isin([3, 4, 5, 6]).astype(int)
    df.attrs["actual_readings_count"] = real_count

    return df


def set_model_attr_if_present(obj, attr_name, value):
    if hasattr(obj, attr_name):
        setattr(obj, attr_name, value)


def save_live_reading_to_db(reading_date, total_units, appliances):
    household = get_current_household()
    reading_date = pd.to_datetime(reading_date).date()
    total_units = max(0.0, float(total_units or 0))

    meter = MeterReading.query.filter_by(
        household_id=household.id,
        reading_date=reading_date,
    ).first()

    if meter is None:
        meter = MeterReading(
            household_id=household.id,
            reading_date=reading_date,
            total_units=total_units,
            source="manual",
        )
        db.session.add(meter)
    else:
        meter.total_units = total_units
        meter.source = "manual"
        set_model_attr_if_present(meter, "updated_at", datetime.utcnow())

    values = {
        "refrigerator": float(appliances.get("Refrigerator", 0) or 0),
        "lights_fans": float(appliances.get("Lights_Fans", 0) or 0),
        "tv_monitor": float(appliances.get("TV_Monitor", 0) or 0),
        "ac": float(appliances.get("AC", 0) or 0),
        "water_heater": float(appliances.get("Water_Heater", 0) or 0),
        "washing_machine": float(appliances.get("Washing_Machine", 0) or 0),
        "motor_pump": float(appliances.get("Motor_Pump", 0) or 0),
    }
    total_appliance_units = sum(values.values())

    appliance = ApplianceReading.query.filter_by(
        household_id=household.id,
        reading_date=reading_date,
    ).first()

    if appliance is None:
        appliance = ApplianceReading(
            household_id=household.id,
            reading_date=reading_date,
            source="manual",
            **values,
        )
        db.session.add(appliance)
    else:
        for key, value in values.items():
            setattr(appliance, key, value)
        appliance.source = "manual"
        set_model_attr_if_present(appliance, "updated_at", datetime.utcnow())

    set_model_attr_if_present(appliance, "total_appliance_units", total_appliance_units)
    db.session.commit()
    return household


# =====================================================
# BILLING HELPERS
# =====================================================

def tneb_domestic_bill(units):
    units = max(0.0, float(units or 0))

    if units <= 500:
        slabs = [
            (100, 0.00),
            (100, 2.35),
            (200, 4.70),
            (100, 6.30),
        ]
    else:
        slabs = [
            (100, 0.00),
            (300, 4.70),
            (100, 6.30),
            (100, 8.40),
            (200, 9.45),
            (200, 10.50),
            (float("inf"), 11.55),
        ]

    remaining = units
    bill = 0.0

    for slab_units, rate in slabs:
        if remaining <= 0:
            break
        used = min(remaining, slab_units)
        bill += used * rate
        remaining -= used

    return round(bill, 2)


def bill_breakdown(units):
    units = max(0.0, float(units or 0))

    if units <= 500:
        slabs = [
            ("0-100", 100, 0.00),
            ("101-200", 100, 2.35),
            ("201-400", 200, 4.70),
            ("401-500", 100, 6.30),
        ]
    else:
        slabs = [
            ("0-100", 100, 0.00),
            ("101-400", 300, 4.70),
            ("401-500", 100, 6.30),
            ("501-600", 100, 8.40),
            ("601-800", 200, 9.45),
            ("801-1000", 200, 10.50),
            ("Above 1000", float("inf"), 11.55),
        ]

    rows = []
    remaining = units

    for slab_name, slab_units, rate in slabs:
        if remaining <= 0:
            break
        used = min(remaining, slab_units)
        amount = used * rate
        rows.append({
            "slab": slab_name,
            "units": round(float(used), 2),
            "rate": round(float(rate), 2),
            "amount": round(float(amount), 2),
        })
        remaining -= used

    return rows


def marginal_rate(units):
    units = max(0.0, float(units or 0))

    if units <= 500:
        if units < 100:
            return 0.0
        if units < 200:
            return 2.35
        if units < 400:
            return 4.70
        return 6.30

    if units < 400:
        return 4.70
    if units < 500:
        return 6.30
    if units < 600:
        return 8.40
    if units < 800:
        return 9.45
    if units < 1000:
        return 10.50
    return 11.55


def slab_info(units):
    units = max(0.0, float(units or 0))

    if units <= 500:
        current = "Lower domestic band up to 500 units"
    elif units <= 600:
        current = "501-600 unit slab"
    elif units <= 800:
        current = "601-800 unit slab"
    elif units <= 1000:
        current = "801-1000 unit slab"
    else:
        current = "Above 1000 unit slab"

    thresholds = [500, 600, 800, 1000]
    next_threshold = None

    for threshold in thresholds:
        if units < threshold:
            next_threshold = threshold
            break

    if next_threshold is None:
        remaining = 0.0
        message = "You are already above the highest tracked slab threshold."
    else:
        remaining = round(next_threshold - units, 2)
        message = f"You have {remaining:.2f} units left before crossing {next_threshold} units."

    return {
        "currentSlab": current,
        "nextThreshold": next_threshold,
        "unitsRemaining": remaining,
        "message": message,
    }


def forecast_range(expected_units, days=60, daily_mae=DAILY_MAE_FOR_RANGE):
    expected_units = max(0.0, float(expected_units or 0))
    margin = float(daily_mae) * np.sqrt(max(1, int(days)))

    lower_units = max(0.0, expected_units - margin)
    upper_units = expected_units + margin

    return {
        "lowerUnits": round(lower_units, 2),
        "expectedUnits": round(expected_units, 2),
        "upperUnits": round(upper_units, 2),
        "lowerBill": tneb_domestic_bill(lower_units),
        "expectedBill": tneb_domestic_bill(expected_units),
        "upperBill": tneb_domestic_bill(upper_units),
        "method": "Range estimated from final-test daily MAE and forecast horizon.",
    }


def make_alerts(projected_units):
    projected_units = max(0.0, float(projected_units or 0))

    if projected_units >= 1000:
        return {
            "level": "danger",
            "title": "Very high bill risk",
            "message": "Projected usage is above 1000 units. Immediate reduction is recommended.",
        }

    if projected_units >= 800:
        return {
            "level": "danger",
            "title": "High bill risk",
            "message": "Projected usage is in the 801-1000 unit slab. Reduce heavy appliance usage.",
        }

    if projected_units >= 600:
        return {
            "level": "caution",
            "title": "Slab warning",
            "message": "Projected usage is above 600 units. Monitor AC, water heater and motor pump usage.",
        }

    if projected_units >= 500:
        return {
            "level": "caution",
            "title": "Near major slab boundary",
            "message": "Projected usage is crossing 500 units. Small daily savings can reduce your bill.",
        }

    return {
        "level": "good",
        "title": "Usage under control",
        "message": "Projected usage is within the lower domestic band.",
    }


# =====================================================
# ML HELPERS
# =====================================================

def find_existing_file(*paths):
    for path in paths:
        if path and os.path.exists(path):
            return path
    return None


def load_metadata():
    metadata_file = find_existing_file(
        METADATA_PATH,
        os.path.join(BASE_DIR, "ExtraTrees_TNEB_Model_Metadata.json"),
    )
    if not metadata_file:
        return {}

    try:
        with open(metadata_file, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return {}


def normalize_columns(df):
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]

    lower_map = {col.lower().replace(" ", "_"): col for col in df.columns}

    if "date" not in df.columns:
        for key, original in lower_map.items():
            if key in {"reading_date", "day", "timestamp"}:
                df = df.rename(columns={original: "Date"})
                break

    if "Total_Units" not in df.columns:
        for candidate in ["total_units", "units", "total_unit", "kwh", "total_kwh", "energy"]:
            if candidate in lower_map:
                df = df.rename(columns={lower_map[candidate]: "Total_Units"})
                break

    appliance_aliases = {
        "refrigerator": "Refrigerator",
        "fridge": "Refrigerator",
        "lights_fans": "Lights_Fans",
        "lights_and_fans": "Lights_Fans",
        "lights_fan": "Lights_Fans",
        "tv_monitor": "TV_Monitor",
        "tv": "TV_Monitor",
        "ac": "AC",
        "air_conditioner": "AC",
        "water_heater": "Water_Heater",
        "heater": "Water_Heater",
        "washing_machine": "Washing_Machine",
        "motor_pump": "Motor_Pump",
        "pump": "Motor_Pump",
    }

    lower_map = {col.lower().replace(" ", "_"): col for col in df.columns}
    for alias, standard in appliance_aliases.items():
        if standard not in df.columns and alias in lower_map:
            df = df.rename(columns={lower_map[alias]: standard})

    return df


def load_training_dataset():
    data_file = find_existing_file(
        DATA_PATH,
        os.path.join(BASE_DIR, "TN_Electricity_Appliance_Unified.csv"),
    )

    if not data_file:
        return pd.DataFrame()

    df = pd.read_csv(data_file)
    df = normalize_columns(df)

    if "Date" not in df.columns:
        df["Date"] = pd.date_range("2020-01-01", periods=len(df), freq="D")

    if "Total_Units" not in df.columns:
        raise ValueError("Training CSV must contain a Total_Units column.")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date")
    df = df.drop_duplicates(subset=["Date"], keep="last")
    df = df.set_index("Date")
    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    df["Total_Units"] = pd.to_numeric(df["Total_Units"], errors="coerce")
    df["Total_Units"] = df["Total_Units"].interpolate(limit_direction="both").fillna(0)

    for col in APPLIANCE_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    if len(df) > 1:
        full_index = pd.date_range(df.index.min(), df.index.max(), freq="D")
        df = df.reindex(full_index)
        df.index.name = "Date"
        df["Total_Units"] = df["Total_Units"].interpolate(limit_direction="both").fillna(0)
        for col in APPLIANCE_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


def create_features(df):
    if df is None or df.empty:
        return pd.DataFrame()

    feature_df = df.copy()

    if "Date" in feature_df.columns:
        feature_df["Date"] = pd.to_datetime(feature_df["Date"], errors="coerce")
        feature_df = feature_df.dropna(subset=["Date"])
        feature_df = feature_df.set_index("Date")

    feature_df.index = pd.to_datetime(feature_df.index)
    feature_df = feature_df.sort_index()
    feature_df.index.name = "Date"

    if "Total_Units" not in feature_df.columns:
        feature_df["Total_Units"] = 0.0

    feature_df["Total_Units"] = pd.to_numeric(feature_df["Total_Units"], errors="coerce")

    for col in APPLIANCE_COLS:
        if col not in feature_df.columns:
            feature_df[col] = 0.0
        feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce").fillna(0.0)

    feature_df["Year"] = feature_df.index.year
    feature_df["Month"] = feature_df.index.month
    feature_df["Day"] = feature_df.index.day
    feature_df["DayOfWeek"] = feature_df.index.dayofweek
    feature_df["Week"] = feature_df.index.isocalendar().week.astype(int)
    feature_df["IsWeekend"] = (feature_df.index.dayofweek >= 5).astype(int)
    feature_df["Is_Summer"] = feature_df["Month"].isin([3, 4, 5, 6]).astype(int)

    feature_df["Month_sin"] = np.sin(2 * np.pi * feature_df["Month"] / 12)
    feature_df["Month_cos"] = np.cos(2 * np.pi * feature_df["Month"] / 12)
    feature_df["DayOfWeek_sin"] = np.sin(2 * np.pi * feature_df["DayOfWeek"] / 7)
    feature_df["DayOfWeek_cos"] = np.cos(2 * np.pi * feature_df["DayOfWeek"] / 7)

    for lag in [1, 2, 3, 7, 14, 30, 60]:
        feature_df[f"lag_{lag}"] = feature_df["Total_Units"].shift(lag)
        feature_df[f"Lag_{lag}"] = feature_df[f"lag_{lag}"]

    shifted = feature_df["Total_Units"].shift(1)
    for window in [3, 7, 14, 30, 60]:
        feature_df[f"rolling_mean_{window}"] = shifted.rolling(window).mean()
        feature_df[f"Rolling_Mean_{window}"] = feature_df[f"rolling_mean_{window}"]
        feature_df[f"rolling_std_{window}"] = shifted.rolling(window).std()
        feature_df[f"Rolling_Std_{window}"] = feature_df[f"rolling_std_{window}"]
        feature_df[f"rolling_min_{window}"] = shifted.rolling(window).min()
        feature_df[f"rolling_max_{window}"] = shifted.rolling(window).max()

    feature_df["rolling_mean_7_30_ratio"] = (
        feature_df["rolling_mean_7"] / feature_df["rolling_mean_30"].replace(0, np.nan)
    )
    feature_df["rolling_mean_30_60_ratio"] = (
        feature_df["rolling_mean_30"] / feature_df["rolling_mean_60"].replace(0, np.nan)
    )

    feature_df["Total_Appliance_Units"] = feature_df[APPLIANCE_COLS].sum(axis=1)

    for col in APPLIANCE_COLS:
        feature_df[f"{col}_rolling_mean_7"] = feature_df[col].shift(1).rolling(7).mean()
        feature_df[f"{col}_rolling_mean_30"] = feature_df[col].shift(1).rolling(30).mean()

    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    return feature_df


def get_feature_columns(feature_df, model_obj=None, metadata=None):
    metadata = metadata or {}

    for key in ["feature_columns", "features", "feature_cols", "model_features"]:
        cols = metadata.get(key)
        if isinstance(cols, list) and cols:
            return [str(col) for col in cols]

    if model_obj is not None and hasattr(model_obj, "feature_names_in_"):
        return [str(col) for col in model_obj.feature_names_in_]

    if feature_df is None or feature_df.empty:
        return []

    excluded = {"Total_Units", "Month_Name", "Day_Name"}
    numeric_cols = []

    for col in feature_df.columns:
        if col in excluded:
            continue
        if pd.api.types.is_numeric_dtype(feature_df[col]):
            numeric_cols.append(col)

    return numeric_cols


def train_or_load_model(base_df):
    global TARGET_TRANSFORM

    metadata = load_metadata()
    TARGET_TRANSFORM = metadata.get("target_transform", metadata.get("target", "log1p"))

    model_file = find_existing_file(
        MODEL_PATH,
        os.path.join(BASE_DIR, "ExtraTrees_TNEB_Model.joblib"),
    )

    loaded_model = None
    loaded_features = None

    if model_file:
        try:
            loaded = joblib.load(model_file)
            if isinstance(loaded, dict):
                loaded_model = loaded.get("model") or loaded.get("estimator") or loaded.get("pipeline")
                loaded_features = loaded.get("feature_columns") or loaded.get("features") or loaded.get("feature_cols")
                TARGET_TRANSFORM = loaded.get("target_transform", TARGET_TRANSFORM)
            else:
                loaded_model = loaded

            if loaded_model is not None:
                feature_df_for_loaded = create_features(base_df) if base_df is not None and not base_df.empty else pd.DataFrame()
                feature_cols = loaded_features or get_feature_columns(feature_df_for_loaded, loaded_model, metadata)
                if feature_cols:
                    return loaded_model, feature_cols, "loaded"
        except Exception as exc:
            print(f"Model load warning: {exc}")

    if base_df is None or base_df.empty:
        return None, [], "fallback_no_training_data"

    feature_df = create_features(base_df)
    feature_cols = get_feature_columns(feature_df, None, metadata)
    feature_df = feature_df.dropna(subset=feature_cols + ["Total_Units"])

    if len(feature_df) < 120 or not feature_cols:
        return None, feature_cols, "fallback_not_enough_training_data"

    model = ExtraTreesRegressor(
        n_estimators=500,
        random_state=RANDOM_STATE,
        min_samples_leaf=2,
        n_jobs=-1,
    )

    y = feature_df["Total_Units"].astype(float)
    if TARGET_TRANSFORM == "log1p":
        y_train = np.log1p(y)
    else:
        y_train = y

    model.fit(feature_df[feature_cols], y_train)

    try:
        joblib.dump(model, MODEL_PATH)
    except Exception:
        pass

    return model, feature_cols, "trained"


def initialize():
    global DATA_DF, MODEL, FEATURE_COLS, MODEL_STATUS

    if DATA_DF is not None and MODEL_STATUS is not None:
        return DATA_DF, MODEL, FEATURE_COLS, MODEL_STATUS

    try:
        DATA_DF = load_training_dataset()
    except Exception as exc:
        print(f"Training data load warning: {exc}")
        DATA_DF = pd.DataFrame()

    try:
        MODEL, FEATURE_COLS, MODEL_STATUS = train_or_load_model(DATA_DF)
    except Exception as exc:
        print(f"Model initialize warning: {exc}")
        MODEL, FEATURE_COLS, MODEL_STATUS = None, [], "fallback_model_error"

    return DATA_DF, MODEL, FEATURE_COLS, MODEL_STATUS


def safe_recent_average(series, days=7):
    if series is None or len(series) == 0:
        return 0.0
    values = pd.Series(series).dropna().astype(float)
    if values.empty:
        return 0.0
    return float(values.tail(days).mean())


def make_future_row(history, target_date, feature_cols=None, appliance_df=None):
    history = pd.Series(history).dropna().astype(float).sort_index()
    history.index = pd.to_datetime(history.index)
    target_date = pd.to_datetime(target_date)

    row = {
        "Year": int(target_date.year),
        "Month": int(target_date.month),
        "Day": int(target_date.day),
        "DayOfWeek": int(target_date.dayofweek),
        "Week": int(target_date.isocalendar().week),
        "IsWeekend": int(target_date.dayofweek >= 5),
        "Is_Summer": int(target_date.month in [3, 4, 5, 6]),
        "Month_sin": float(np.sin(2 * np.pi * target_date.month / 12)),
        "Month_cos": float(np.cos(2 * np.pi * target_date.month / 12)),
        "DayOfWeek_sin": float(np.sin(2 * np.pi * target_date.dayofweek / 7)),
        "DayOfWeek_cos": float(np.cos(2 * np.pi * target_date.dayofweek / 7)),
    }

    fallback = safe_recent_average(history, 7)

    for lag in [1, 2, 3, 7, 14, 30, 60]:
        value = float(history.iloc[-lag]) if len(history) >= lag else fallback
        row[f"lag_{lag}"] = value
        row[f"Lag_{lag}"] = value

    for window in [3, 7, 14, 30, 60]:
        recent = history.tail(window)
        mean_value = float(recent.mean()) if len(recent) else fallback
        std_value = float(recent.std()) if len(recent) > 1 else 0.0
        min_value = float(recent.min()) if len(recent) else fallback
        max_value = float(recent.max()) if len(recent) else fallback

        row[f"rolling_mean_{window}"] = mean_value
        row[f"Rolling_Mean_{window}"] = mean_value
        row[f"rolling_std_{window}"] = std_value
        row[f"Rolling_Std_{window}"] = std_value
        row[f"rolling_min_{window}"] = min_value
        row[f"rolling_max_{window}"] = max_value

    row["rolling_mean_7_30_ratio"] = row["rolling_mean_7"] / row["rolling_mean_30"] if row["rolling_mean_30"] else 1.0
    row["rolling_mean_30_60_ratio"] = row["rolling_mean_30"] / row["rolling_mean_60"] if row["rolling_mean_60"] else 1.0

    total_appliance = 0.0
    for col in APPLIANCE_COLS:
        if appliance_df is not None and col in appliance_df.columns and len(appliance_df):
            value = safe_recent_average(appliance_df[col], 7)
            value_30 = safe_recent_average(appliance_df[col], 30)
        else:
            value = 0.0
            value_30 = 0.0

        row[col] = value
        row[f"{col}_rolling_mean_7"] = value
        row[f"{col}_rolling_mean_30"] = value_30
        total_appliance += value

    row["Total_Appliance_Units"] = total_appliance

    if feature_cols:
        for col in feature_cols:
            if col not in row:
                row[col] = 0.0

    return row


def predict_units(model, feature_cols, row, fallback_value):
    fallback_value = max(0.0, float(fallback_value or 0))

    if model is None or not feature_cols:
        return fallback_value

    try:
        x = pd.DataFrame([row])
        x = x.reindex(columns=feature_cols, fill_value=0.0)
        x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        raw_pred = float(model.predict(x)[0])

        if TARGET_TRANSFORM == "log1p":
            pred = float(np.expm1(raw_pred))
        else:
            pred = raw_pred

        if not np.isfinite(pred):
            return fallback_value

        # Safety clamp: stop one bad ML output from breaking the product UI.
        recent_cap = max(50.0, fallback_value * 4.0)
        pred = min(max(0.0, pred), recent_cap)
        return pred
    except Exception as exc:
        print(f"Prediction warning: {exc}")
        return fallback_value


def forecast_from_series(model, history_input, feature_cols, days):
    days = int(days or 0)
    if days <= 0:
        return pd.DataFrame(columns=["date", "units"])

    if isinstance(history_input, pd.DataFrame):
        df = history_input.copy()
        history = df["Total_Units"].dropna().astype(float).sort_index()
        appliance_df = df[[col for col in APPLIANCE_COLS if col in df.columns]].copy()
    else:
        history = pd.Series(history_input).dropna().astype(float).sort_index()
        appliance_df = None

    if history.empty:
        return pd.DataFrame(columns=["date", "units"])

    history.index = pd.to_datetime(history.index)
    records = []

    for _ in range(days):
        next_date = history.index.max() + pd.Timedelta(days=1)
        fallback = safe_recent_average(history, 7)
        future_row = make_future_row(history, next_date, feature_cols, appliance_df)
        predicted_units = predict_units(model, feature_cols, future_row, fallback)

        records.append({
            "date": next_date.strftime("%Y-%m-%d"),
            "units": round(float(predicted_units), 3),
        })

        history.loc[next_date] = predicted_units

    return pd.DataFrame(records)


# =====================================================
# SMART ANALYTICS HELPERS
# =====================================================

def available_appliances(df):
    if df is None or df.empty:
        return []
    return [col for col in APPLIANCE_COLS if col in df.columns]


def anomaly_summary(df):
    if df is None or df.empty:
        return {"status": "No readings yet", "latestDate": None, "latestUnits": 0}

    latest_date = df.index.max()
    latest_units = float(df.loc[latest_date, "Total_Units"])
    prior = df.iloc[:-1]["Total_Units"].dropna().tail(30)

    if len(prior) < 14:
        return {
            "status": "Not enough history",
            "latestDate": latest_date.strftime("%Y-%m-%d"),
            "latestUnits": round(latest_units, 2),
        }

    mean30 = float(prior.mean())
    std30 = float(prior.std()) if len(prior) > 1 else 0.0
    threshold = mean30 + 2 * std30
    is_anomaly = bool(latest_units > threshold)

    return {
        "latestDate": latest_date.strftime("%Y-%m-%d"),
        "latestUnits": round(latest_units, 2),
        "mean30": round(mean30, 2),
        "threshold": round(threshold, 2),
        "isAnomaly": is_anomaly,
        "status": "High usage anomaly" if is_anomaly else "Normal",
    }


def appliance_insights(df, projected_cycle_units):
    if df is None or df.empty:
        return {"totals": [], "opportunities": []}

    appliances = available_appliances(df)
    if not appliances:
        return {"totals": [], "opportunities": []}

    recent = df.tail(30)
    previous = df.iloc[-60:-30] if len(df) >= 60 else df.iloc[0:0]
    total_recent = float(recent["Total_Units"].sum())
    rate = marginal_rate(projected_cycle_units)

    totals = []
    opportunities = []

    for col in appliances:
        recent_units = float(recent[col].sum())
        prev_units = float(previous[col].sum()) if len(previous) else 0.0
        share = (recent_units / total_recent * 100) if total_recent else 0.0
        trend = recent_units - prev_units

        totals.append({
            "appliance": col.replace("_", " "),
            "units": round(recent_units, 2),
            "share": round(share, 2),
            "trend": round(trend, 2),
        })

        saving_pct = 0.10
        if col == "AC":
            saving_pct = 0.12
        elif col in {"Water_Heater", "Motor_Pump"}:
            saving_pct = 0.15

        save_units = recent_units * saving_pct
        if recent_units > 0 and share >= 5:
            opportunities.append({
                "appliance": col.replace("_", " "),
                "currentUnits30d": round(recent_units, 2),
                "share": round(share, 2),
                "suggestedReductionPercent": int(saving_pct * 100),
                "estimatedUnitsSaved": round(save_units, 2),
                "estimatedRupeesSaved": round(save_units * rate, 2),
            })

    totals = sorted(totals, key=lambda item: item["units"], reverse=True)
    opportunities = sorted(opportunities, key=lambda item: item["estimatedUnitsSaved"], reverse=True)
    return {"totals": totals, "opportunities": opportunities[:6]}


def dynamic_recommendations(df, projected_units, used_so_far, remaining_days):
    insights = appliance_insights(df, projected_units)
    rate = marginal_rate(projected_units)
    slab = slab_info(projected_units)
    recs = []

    if projected_units > 500:
        reduce_to_500 = projected_units - 500
        per_day = reduce_to_500 / max(1, remaining_days)
        recs.append({
            "priority": "High",
            "title": "Control the 500-unit slab trigger",
            "message": (
                f"Your cycle is projected at {projected_units:.1f} units. "
                f"To finish below 500 units, reduce about {reduce_to_500:.1f} units total, "
                f"or {per_day:.2f} units/day for the remaining {remaining_days} days."
            ),
            "impactUnits": round(reduce_to_500, 2),
            "impactRs": round(reduce_to_500 * rate, 2),
        })
    elif slab.get("nextThreshold"):
        recs.append({
            "priority": "Good",
            "title": "You are still below a major slab",
            "message": slab["message"] + " Keep daily usage steady to avoid a higher slab.",
            "impactUnits": 0,
            "impactRs": 0,
        })

    for opp in insights["opportunities"][:4]:
        app_name = opp["appliance"]
        recs.append({
            "priority": "Medium",
            "title": f"Reduce {app_name} usage by {opp['suggestedReductionPercent']}%",
            "message": (
                f"{app_name} used {opp['currentUnits30d']} units in the last 30 days "
                f"({opp['share']}% of usage). A {opp['suggestedReductionPercent']}% reduction "
                f"can save about {opp['estimatedUnitsSaved']} units and roughly ₹{opp['estimatedRupeesSaved']}."
            ),
            "impactUnits": opp["estimatedUnitsSaved"],
            "impactRs": opp["estimatedRupeesSaved"],
        })

    anom = anomaly_summary(df)
    if anom.get("isAnomaly"):
        excess = max(0.0, anom["latestUnits"] - anom["mean30"])
        recs.insert(0, {
            "priority": "High",
            "title": "Investigate today's abnormal usage",
            "message": (
                f"Latest usage was {anom['latestUnits']} units, above the normal threshold of "
                f"{anom['threshold']} units. Check AC, heater, or motor pump usage today."
            ),
            "impactUnits": round(excess, 2),
            "impactRs": round(excess * rate, 2),
        })

    return recs[:6]


def monthly_report(df, start=None, end=None):
    if df is None or df.empty:
        return {
            "start": None,
            "end": None,
            "days": 0,
            "totalUnits": 0,
            "avgDaily": 0,
            "peakDay": None,
            "peakUnits": 0,
            "appliances": [],
        }

    if start is None:
        latest = df.index.max()
        start = pd.Timestamp(latest.year, latest.month, 1)
    else:
        start = pd.to_datetime(start)

    if end is None:
        end = df.index.max()
    else:
        end = pd.to_datetime(end)

    dff = df.loc[start:end].copy()
    appliances = available_appliances(dff)

    report = {
        "start": start.strftime("%Y-%m-%d"),
        "end": end.strftime("%Y-%m-%d"),
        "days": int(len(dff)),
        "totalUnits": round(float(dff["Total_Units"].sum()), 2) if len(dff) else 0,
        "avgDaily": round(float(dff["Total_Units"].mean()), 2) if len(dff) else 0,
        "peakDay": dff["Total_Units"].idxmax().strftime("%Y-%m-%d") if len(dff) else None,
        "peakUnits": round(float(dff["Total_Units"].max()), 2) if len(dff) else 0,
        "appliances": [],
    }

    if appliances and len(dff):
        app_total = dff[appliances].sum().sort_values(ascending=False)
        total_app = float(app_total.sum())
        report["appliances"] = [
            {
                "appliance": col.replace("_", " "),
                "units": round(float(value), 2),
                "share": round(float(value / total_app * 100), 2) if total_app else 0,
            }
            for col, value in app_total.items()
        ]

    return report


# =====================================================
# EMPTY RESPONSE HELPERS
# =====================================================

def empty_overview_response(model_status="loaded"):
    return jsonify({
        "emptyState": True,
        "notEnoughHistory": True,
        "latestDate": None,
        "latestUnits": 0,
        "usedThisMonth": 0,
        "projectedMonthUnits": 0,
        "monthEnd": None,
        "cycleStart": None,
        "cycleEnd": None,
        "cycleDay": 0,
        "cycleUsedSoFar": 0,
        "cycleForecastRemaining": 0,
        "projectedCycleUnits": 0,
        "projectedBill": 0,
        "historical60": 0,
        "differencePct": 0,
        "alert": {
            "level": "good",
            "title": "No readings yet",
            "message": "Add your first appliance reading from Live Check to start building your household profile.",
        },
        "billBreakdown": [],
        "monthChart": [],
        "modelStatus": model_status,
        "slabInfo": {
            "currentSlab": "No data",
            "nextThreshold": 500,
            "unitsRemaining": 500,
            "message": "Add readings to begin slab tracking.",
        },
        "forecastRange": forecast_range(0),
    })


def empty_analytics_response():
    return jsonify({
        "summary": {
            "totalUnits": 0,
            "avgDaily": 0,
            "peakDate": None,
            "peakUnits": 0,
            "summerAvg": None,
            "nonSummerAvg": None,
            "topAppliance": "No data",
        },
        "daily": [],
        "weekly": [],
        "monthly": [],
        "yearly": [],
        "dayOfWeek": [],
        "applianceTotals": [],
        "applianceMonthly": [],
        "heatmap": [],
        "billCycles": [],
    })


def empty_features_response():
    return jsonify({
        "cycle": {
            "start": None,
            "end": None,
            "day": 0,
            "usedSoFar": 0,
            "remainingDays": 0,
            "forecastRemaining": 0,
            "projectedUnits": 0,
            "projectedBill": 0,
        },
        "slabInfo": {
            "currentSlab": "No data",
            "nextThreshold": 500,
            "unitsRemaining": 500,
            "message": "Add readings to begin slab tracking.",
        },
        "billBreakdown": [],
        "forecastRange": forecast_range(0),
        "recommendations": [
            {
                "priority": "Start",
                "title": "Add your first reading",
                "message": "Use Live Check to enter appliance-wise units. The app will start building your household profile.",
                "impactUnits": 0,
                "impactRs": 0,
            }
        ],
        "applianceInsights": {"totals": [], "opportunities": []},
        "anomaly": {"status": "No readings yet"},
        "monthlyReport": monthly_report(pd.DataFrame()),
        "recentUsage": [],
    })

    # =====================================================
# SMART ANALYTICS HELPERS
# =====================================================

def available_appliances(df):
    if df is None or df.empty:
        return []

    return [col for col in APPLIANCE_COLS if col in df.columns]


def anomaly_summary(df):
    if df is None or df.empty:
        return {
            "status": "No readings yet",
            "latestDate": None,
            "latestUnits": 0,
        }

    latest_date = df.index.max()
    latest_units = float(df.loc[latest_date, "Total_Units"])

    prior = df.iloc[:-1]["Total_Units"].dropna().tail(30)

    if len(prior) < 14:
        return {
            "status": "Not enough history",
            "latestDate": latest_date.strftime("%Y-%m-%d"),
            "latestUnits": round(latest_units, 2),
        }

    mean30 = float(prior.mean())
    std30 = float(prior.std()) if len(prior) > 1 else 0.0
    threshold = mean30 + 2 * std30
    is_anomaly = bool(latest_units > threshold)

    return {
        "latestDate": latest_date.strftime("%Y-%m-%d"),
        "latestUnits": round(latest_units, 2),
        "mean30": round(mean30, 2),
        "threshold": round(threshold, 2),
        "isAnomaly": is_anomaly,
        "status": "High usage anomaly" if is_anomaly else "Normal",
    }


def appliance_insights(df, projected_cycle_units):
    if df is None or df.empty:
        return {
            "totals": [],
            "opportunities": [],
        }

    appliances = available_appliances(df)

    if not appliances:
        return {
            "totals": [],
            "opportunities": [],
        }

    recent = df.tail(30)
    previous = df.iloc[-60:-30] if len(df) >= 60 else df.iloc[0:0]

    total_recent = float(recent["Total_Units"].sum())
    rate = marginal_rate(projected_cycle_units)

    totals = []
    opportunities = []

    for col in appliances:
        recent_units = float(recent[col].sum())
        prev_units = float(previous[col].sum()) if len(previous) else 0.0
        share = (recent_units / total_recent * 100) if total_recent else 0.0
        trend = recent_units - prev_units

        totals.append({
            "appliance": col.replace("_", " "),
            "units": round(recent_units, 2),
            "share": round(share, 2),
            "trend": round(trend, 2),
        })

        saving_pct = 0.10

        if col == "AC":
            saving_pct = 0.12
        elif col == "Water_Heater":
            saving_pct = 0.15
        elif col == "Motor_Pump":
            saving_pct = 0.15

        save_units = recent_units * saving_pct

        if recent_units > 0 and share >= 5:
            opportunities.append({
                "appliance": col.replace("_", " "),
                "currentUnits30d": round(recent_units, 2),
                "share": round(share, 2),
                "suggestedReductionPercent": int(saving_pct * 100),
                "estimatedUnitsSaved": round(save_units, 2),
                "estimatedRupeesSaved": round(save_units * rate, 2),
            })

    totals = sorted(totals, key=lambda x: x["units"], reverse=True)
    opportunities = sorted(
        opportunities,
        key=lambda x: x["estimatedUnitsSaved"],
        reverse=True,
    )

    return {
        "totals": totals,
        "opportunities": opportunities[:6],
    }


def dynamic_recommendations(df, projected_units, used_so_far, remaining_days):
    insights = appliance_insights(df, projected_units)
    rate = marginal_rate(projected_units)
    slab = slab_info(projected_units)

    recs = []

    if projected_units > 500:
        reduce_to_500 = projected_units - 500
        per_day = reduce_to_500 / max(1, remaining_days)

        recs.append({
            "priority": "High",
            "title": "Control the 500-unit slab trigger",
            "message": (
                f"Your cycle is projected at {projected_units:.1f} units. "
                f"To finish below 500 units, reduce about {reduce_to_500:.1f} units total, "
                f"or {per_day:.2f} units/day for the remaining {remaining_days} days."
            ),
            "impactUnits": round(reduce_to_500, 2),
            "impactRs": round(reduce_to_500 * rate, 2),
        })

    elif slab.get("nextThreshold"):
        recs.append({
            "priority": "Good",
            "title": "You are still below a major slab",
            "message": slab["message"] + " Keep daily usage steady to avoid a higher slab.",
            "impactUnits": 0,
            "impactRs": 0,
        })

    for opp in insights["opportunities"][:4]:
        app_name = opp["appliance"]

        recs.append({
            "priority": "Medium",
            "title": f"Reduce {app_name} usage by {opp['suggestedReductionPercent']}%",
            "message": (
                f"{app_name} used {opp['currentUnits30d']} units in the last 30 days "
                f"({opp['share']}% of usage). A {opp['suggestedReductionPercent']}% reduction "
                f"can save about {opp['estimatedUnitsSaved']} units and roughly ₹{opp['estimatedRupeesSaved']}."
            ),
            "impactUnits": opp["estimatedUnitsSaved"],
            "impactRs": opp["estimatedRupeesSaved"],
        })

    anom = anomaly_summary(df)

    if anom.get("isAnomaly"):
        excess_units = max(0, anom["latestUnits"] - anom["mean30"])

        recs.insert(0, {
            "priority": "High",
            "title": "Investigate today's abnormal usage",
            "message": (
                f"Latest usage was {anom['latestUnits']} units, above the normal threshold of "
                f"{anom['threshold']} units. Check AC, heater, or motor pump usage today."
            ),
            "impactUnits": round(excess_units, 2),
            "impactRs": round(excess_units * rate, 2),
        })

    return recs[:6]


def monthly_report(df, start=None, end=None):
    if df is None or df.empty:
        return {
            "start": None,
            "end": None,
            "days": 0,
            "totalUnits": 0,
            "avgDaily": 0,
            "peakDay": None,
            "peakUnits": 0,
            "appliances": [],
        }

    if start is None:
        latest = df.index.max()
        start = pd.Timestamp(latest.year, latest.month, 1)
    else:
        start = pd.to_datetime(start)

    if end is None:
        end = df.index.max()
    else:
        end = pd.to_datetime(end)

    dff = df.loc[start:end].copy()
    appliances = available_appliances(dff)

    report = {
        "start": start.strftime("%Y-%m-%d"),
        "end": end.strftime("%Y-%m-%d"),
        "days": int(len(dff)),
        "totalUnits": round(float(dff["Total_Units"].sum()), 2) if len(dff) else 0,
        "avgDaily": round(float(dff["Total_Units"].mean()), 2) if len(dff) else 0,
        "peakDay": dff["Total_Units"].idxmax().strftime("%Y-%m-%d") if len(dff) else None,
        "peakUnits": round(float(dff["Total_Units"].max()), 2) if len(dff) else 0,
        "appliances": [],
    }

    if appliances and len(dff):
        app_total = dff[appliances].sum().sort_values(ascending=False)
        total_app = float(app_total.sum())

        report["appliances"] = [
            {
                "appliance": col.replace("_", " "),
                "units": round(float(value), 2),
                "share": round(float(value / total_app * 100), 2) if total_app else 0,
            }
            for col, value in app_total.items()
        ]

    return report


    # =====================================================
# PERSISTENT ABNORMAL CONSUMPTION ALERTS
# =====================================================

def save_abnormal_consumption_alert_if_needed(household, df):
    """
    Saves an abnormal consumption alert into PostgreSQL if latest usage is high.
    It creates only one alert per household per reading_date.
    Uses the existing alerts table.
    """
    if df is None or df.empty:
        return None

    try:
        anomaly = anomaly_summary(df)

        if not anomaly.get("isAnomaly"):
            return anomaly

        latest_date_value = anomaly.get("latestDate")
        latest_units = float(anomaly.get("latestUnits", 0) or 0)
        mean30 = float(anomaly.get("mean30", 0) or 0)
        threshold = float(anomaly.get("threshold", 0) or 0)

        if not latest_date_value:
            return anomaly

        reading_date = datetime.strptime(latest_date_value, "%Y-%m-%d").date()

        from sqlalchemy import text

        existing = db.session.execute(text("""
            SELECT id
            FROM alerts
            WHERE household_id = :household_id
              AND alert_type = 'abnormal_consumption'
              AND reading_date = :reading_date
            LIMIT 1
        """), {
            "household_id": str(household.id),
            "reading_date": reading_date,
        }).first()

        if existing:
            return anomaly

        excess_units = max(0.0, latest_units - mean30)

        severity = "warning"
        if threshold > 0 and latest_units >= threshold * 1.30:
            severity = "critical"

        title = "Abnormal consumption detected"

        message = (
            f"Usage on {latest_date_value} was {latest_units:.2f} units, "
            f"above your normal threshold of {threshold:.2f} units. "
            f"This is about {excess_units:.2f} units higher than your recent average. "
            f"Please check AC, water heater, motor pump, or any appliance left running."
        )

        db.session.execute(text("""
            INSERT INTO alerts (
                id,
                household_id,
                alert_date,
                alert_type,
                severity,
                title,
                message,
                reading_date,
                is_read,
                created_at
            ) VALUES (
                gen_random_uuid(),
                :household_id,
                :alert_date,
                'abnormal_consumption',
                :severity,
                :title,
                :message,
                :reading_date,
                FALSE,
                NOW()
            )
        """), {
            "household_id": str(household.id),
            "alert_date": reading_date,
            "severity": severity,
            "title": title,
            "message": message,
            "reading_date": reading_date,
        })

        try:
            send_push_for_alert(
             household.id,
            "abnormal_consumption",
            severity,
            title,
            message,
    )
        except Exception as exc:
             print(f"Push abnormal alert warning: {exc}")

        db.session.commit()

        return anomaly

    except Exception as exc:
        db.session.rollback()
        print(f"Abnormal alert save warning: {exc}")
        return None


# =====================================================
# ROUTES
# =====================================================

@app.route("/")
def index():
    initialize()
    return render_template("index.html")


@app.route("/api/meta")
def api_meta():
    _, _, _, model_status = initialize()
    household = get_current_household()
    household_df = load_current_household_dataframe()

    return jsonify({
        "user": {
            "id": str(current_user.id),
            "name": getattr(current_user, "full_name", ""),
            "email": getattr(current_user, "email", ""),
        },
        "household": {
            "id": str(household.id),
            "name": getattr(household, "household_name", "My Home"),
            "city": getattr(household, "city", ""),
            "district": getattr(household, "district", ""),
            "state": getattr(household, "state", "Tamil Nadu"),
            "ebServiceNumber": getattr(household, "eb_service_number", ""),
        },
        "dataStart": household_df.index.min().strftime("%Y-%m-%d") if not household_df.empty else None,
        "dataEnd": household_df.index.max().strftime("%Y-%m-%d") if not household_df.empty else None,
        "rows": actual_reading_count(household_df),
        "modelStatus": model_status,
        "appliances": APPLIANCE_COLS,
        "minReadingsForForecast": MIN_READINGS_FOR_FORECAST,
    })


@app.route("/api/overview")
def api_overview():
    _, model, feature_cols, model_status = initialize()
    df = load_current_household_dataframe()

    if df.empty:
        return empty_overview_response(model_status)

    latest_date = df.index.max()
    latest_units = float(df.loc[latest_date, "Total_Units"])
    current_month_start = pd.Timestamp(latest_date.year, latest_date.month, 1)
    used_this_month = float(df.loc[current_month_start:latest_date, "Total_Units"].sum())
    total_so_far = float(df["Total_Units"].sum())
    count = actual_reading_count(df)

    if count < MIN_READINGS_FOR_FORECAST:
        return jsonify({
            "emptyState": False,
            "notEnoughHistory": True,
            "latestDate": latest_date.strftime("%Y-%m-%d"),
            "latestUnits": round(latest_units, 2),
            "usedThisMonth": round(used_this_month, 2),
            "projectedMonthUnits": round(used_this_month, 2),
            "monthEnd": latest_date.strftime("%Y-%m-%d"),
            "cycleStart": None,
            "cycleEnd": None,
            "cycleDay": count,
            "cycleUsedSoFar": round(total_so_far, 2),
            "cycleForecastRemaining": 0,
            "projectedCycleUnits": round(total_so_far, 2),
            "projectedBill": tneb_domestic_bill(total_so_far),
            "historical60": round(total_so_far, 2),
            "differencePct": 0,
            "alert": {
                "level": "caution",
                "title": "More readings needed",
                "message": f"Forecasting activates after {MIN_READINGS_FOR_FORECAST} daily readings. Keep entering daily appliance units.",
            },
            "billBreakdown": bill_breakdown(total_so_far),
            "monthChart": [
                {
                    "date": idx.strftime("%Y-%m-%d"),
                    "units": round(float(row["Total_Units"]), 3),
                    "type": "Actual",
                }
                for idx, row in df.iterrows()
            ],
            "modelStatus": model_status,
            "slabInfo": slab_info(total_so_far),
            "forecastRange": forecast_range(total_so_far, days=max(1, count)),
        })

    cycle_start_param = request.args.get("cycle_start")
    if cycle_start_param:
        cycle_start = pd.to_datetime(cycle_start_param)
    else:
        cycle_start = latest_date - pd.Timedelta(days=59)

    cycle_end = cycle_start + pd.Timedelta(days=59)

    month_start = pd.Timestamp(latest_date.year, latest_date.month, 1)
    month_end = pd.Timestamp(
        latest_date.year,
        latest_date.month,
        calendar.monthrange(latest_date.year, latest_date.month)[1],
    )

    used_this_month = float(df.loc[month_start:latest_date, "Total_Units"].sum())
    remaining_month_days = max(0, (month_end - latest_date).days)
    month_forecast = forecast_from_series(model, df, feature_cols, remaining_month_days)
    forecast_month_units = float(month_forecast["units"].sum()) if len(month_forecast) else 0.0
    projected_month_units = used_this_month + forecast_month_units

    if latest_date < cycle_start:
        cycle_day = 0
        used_so_far = 0.0
        remaining_cycle_days = 60
    else:
        cycle_day = min(60, (latest_date - cycle_start).days + 1)
        used_so_far = float(df.loc[cycle_start:min(latest_date, cycle_end), "Total_Units"].sum())
        remaining_cycle_days = max(0, 60 - cycle_day)

    cycle_forecast = forecast_from_series(model, df, feature_cols, remaining_cycle_days)
    cycle_forecast_units = float(cycle_forecast["units"].sum()) if len(cycle_forecast) else 0.0
    projected_cycle_units = used_so_far + cycle_forecast_units
    projected_bill = tneb_domestic_bill(projected_cycle_units)

    month_chart = []
    for idx, row in df.loc[month_start:latest_date].iterrows():
        month_chart.append({
            "date": idx.strftime("%Y-%m-%d"),
            "units": round(float(row["Total_Units"]), 3),
            "type": "Actual",
        })

    for _, row in month_forecast.iterrows():
        month_chart.append({
            "date": row["date"],
            "units": round(float(row["units"]), 3),
            "type": "Forecast",
        })

    historical_60 = float(df["Total_Units"].tail(60).sum())
    diff_pct = ((projected_cycle_units - historical_60) / historical_60) * 100 if historical_60 else 0

    return jsonify({
        "emptyState": False,
        "notEnoughHistory": False,
        "latestDate": latest_date.strftime("%Y-%m-%d"),
        "latestUnits": round(latest_units, 2),
        "usedThisMonth": round(used_this_month, 2),
        "projectedMonthUnits": round(projected_month_units, 2),
        "monthEnd": month_end.strftime("%Y-%m-%d"),
        "cycleStart": cycle_start.strftime("%Y-%m-%d"),
        "cycleEnd": cycle_end.strftime("%Y-%m-%d"),
        "cycleDay": int(cycle_day),
        "cycleUsedSoFar": round(used_so_far, 2),
        "cycleForecastRemaining": round(cycle_forecast_units, 2),
        "projectedCycleUnits": round(projected_cycle_units, 2),
        "projectedBill": projected_bill,
        "historical60": round(historical_60, 2),
        "differencePct": round(float(diff_pct), 2),
        "alert": make_alerts(projected_cycle_units),
        "billBreakdown": bill_breakdown(projected_cycle_units),
        "monthChart": month_chart,
        "modelStatus": model_status,
        "slabInfo": slab_info(projected_cycle_units),
        "forecastRange": forecast_range(projected_cycle_units),
    })


@app.route("/api/analytics")
def api_analytics():
    df = load_current_household_dataframe()

    if df.empty:
        return empty_analytics_response()

    start = pd.to_datetime(request.args.get("start")) if request.args.get("start") else df.index.min()
    end = pd.to_datetime(request.args.get("end")) if request.args.get("end") else df.index.max()
    dff = df.loc[start:end].copy()

    if dff.empty:
        return empty_analytics_response()

    appliances = available_appliances(dff)

    daily = dff[["Total_Units"]].copy()
    daily["Rolling7"] = daily["Total_Units"].rolling(7).mean()
    daily_records = [
        {
            "date": idx.strftime("%Y-%m-%d"),
            "units": round(float(row["Total_Units"]), 3),
            "rolling7": None if pd.isna(row["Rolling7"]) else round(float(row["Rolling7"]), 3),
        }
        for idx, row in daily.iterrows()
    ]

    weekly = dff.resample("W")["Total_Units"].sum().reset_index()
    weekly_records = [
        {"date": row["Date"].strftime("%Y-%m-%d"), "units": round(float(row["Total_Units"]), 2)}
        for _, row in weekly.iterrows()
    ]

    monthly = dff.resample("M")["Total_Units"].sum().reset_index()
    monthly_records = [
        {"month": row["Date"].strftime("%b %Y"), "units": round(float(row["Total_Units"]), 2)}
        for _, row in monthly.iterrows()
    ]

    yearly = dff.groupby(dff.index.year)["Total_Units"].sum().reset_index()
    yearly.columns = ["year", "units"]
    yearly_records = [
        {"year": int(row["year"]), "units": round(float(row["units"]), 2)}
        for _, row in yearly.iterrows()
    ]

    dow = (
        dff.groupby("Day_Name")["Total_Units"]
        .mean()
        .reindex(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"])
        .reset_index()
    )
    dow_records = [
        {
            "day": row["Day_Name"],
            "units": 0 if pd.isna(row["Total_Units"]) else round(float(row["Total_Units"]), 2),
        }
        for _, row in dow.iterrows()
    ]

    appliance_totals = []
    appliance_monthly = []

    if appliances:
        app_total = dff[appliances].sum().sort_values(ascending=False)
        total_app_units = float(app_total.sum())
        appliance_totals = [
            {
                "appliance": col.replace("_", " "),
                "units": round(float(val), 2),
                "share": round(float(val / total_app_units * 100), 2) if total_app_units else 0,
            }
            for col, val in app_total.items()
        ]

        app_month = dff[appliances].resample("M").sum().reset_index()
        for _, row in app_month.iterrows():
            item = {"month": row["Date"].strftime("%b %Y")}
            for col in appliances:
                item[col.replace("_", " ")] = round(float(row[col]), 2)
            appliance_monthly.append(item)

    heatmap = [
        {
            "month": idx.strftime("%b %Y"),
            "day": int(idx.day),
            "units": round(float(row["Total_Units"]), 2),
        }
        for idx, row in dff.iterrows()
    ]

    bill_cycles = []
    cur_start = start
    cycle_no = 1

    while cur_start <= end:
        cur_end = cur_start + pd.Timedelta(days=59)
        cycle_data = dff.loc[cur_start:min(cur_end, end)]

        if len(cycle_data):
            units = float(cycle_data["Total_Units"].sum())
            bill_cycles.append({
                "cycle": cycle_no,
                "start": cur_start.strftime("%Y-%m-%d"),
                "end": min(cur_end, end).strftime("%Y-%m-%d"),
                "days": int(len(cycle_data)),
                "units": round(units, 2),
                "bill": tneb_domestic_bill(units),
            })
            cycle_no += 1

        cur_start = cur_end + pd.Timedelta(days=1)

    peak_idx = dff["Total_Units"].idxmax()
    summer_avg = dff.loc[dff["Is_Summer"] == 1, "Total_Units"].mean()
    normal_avg = dff.loc[dff["Is_Summer"] == 0, "Total_Units"].mean()

    return jsonify({
        "summary": {
            "totalUnits": round(float(dff["Total_Units"].sum()), 2),
            "avgDaily": round(float(dff["Total_Units"].mean()), 2),
            "peakDate": peak_idx.strftime("%Y-%m-%d"),
            "peakUnits": round(float(dff.loc[peak_idx, "Total_Units"]), 2),
            "summerAvg": None if pd.isna(summer_avg) else round(float(summer_avg), 2),
            "nonSummerAvg": None if pd.isna(normal_avg) else round(float(normal_avg), 2),
            "topAppliance": appliance_totals[0]["appliance"] if appliance_totals else "N/A",
        },
        "daily": daily_records,
        "weekly": weekly_records,
        "monthly": monthly_records,
        "yearly": yearly_records,
        "dayOfWeek": dow_records,
        "applianceTotals": appliance_totals,
        "applianceMonthly": appliance_monthly,
        "heatmap": heatmap,
        "billCycles": bill_cycles,
    })


@app.route("/api/features")
def api_features():
    _, model, feature_cols, _ = initialize()
    df = load_current_household_dataframe()

    if df.empty:
        return empty_features_response()

    count = actual_reading_count(df)

    if count < MIN_READINGS_FOR_FORECAST:
        total_units = float(df["Total_Units"].sum())
        report = monthly_report(df)
        recent = df.reset_index()

        return jsonify({
            "notEnoughHistory": True,
            "cycle": {
                "start": df.index.min().strftime("%Y-%m-%d"),
                "end": df.index.max().strftime("%Y-%m-%d"),
                "day": count,
                "usedSoFar": round(total_units, 2),
                "remainingDays": max(0, MIN_READINGS_FOR_FORECAST - count),
                "forecastRemaining": 0,
                "projectedUnits": round(total_units, 2),
                "projectedBill": tneb_domestic_bill(total_units),
            },
            "slabInfo": slab_info(total_units),
            "billBreakdown": bill_breakdown(total_units),
            "forecastRange": forecast_range(total_units, days=max(1, count)),
            "recommendations": [
                {
                    "priority": "Start",
                    "title": "Keep entering daily readings",
                    "message": f"Forecasting activates after {MIN_READINGS_FOR_FORECAST} daily readings. Your current entries are saved and visible in analytics.",
                    "impactUnits": 0,
                    "impactRs": 0,
                }
            ],
            "applianceInsights": appliance_insights(df, total_units),
            "anomaly": anomaly_summary(df),
            "monthlyReport": report,
            "recentUsage": [
                {
                    "date": row["Date"].strftime("%Y-%m-%d"),
                    "units": round(float(row["Total_Units"]), 2),
                }
                for _, row in recent.iterrows()
            ],
        })

    latest_date = df.index.max()
    household = get_current_household()
    save_abnormal_consumption_alert_if_needed(household, df)
    cycle_start_param = request.args.get("cycle_start")

    if cycle_start_param:
        cycle_start = pd.to_datetime(cycle_start_param)
    else:
        cycle_start = latest_date - pd.Timedelta(days=59)

    cycle_end = cycle_start + pd.Timedelta(days=59)
    cycle_day = min(60, max(0, (latest_date - cycle_start).days + 1))

    if latest_date >= cycle_start:
        used_so_far = float(df.loc[cycle_start:min(latest_date, cycle_end), "Total_Units"].sum())
    else:
        used_so_far = 0.0

    remaining_days = max(0, 60 - cycle_day)
    rem_forecast = forecast_from_series(model, df, feature_cols, remaining_days)
    rem_units = float(rem_forecast["units"].sum()) if len(rem_forecast) else 0.0
    projected_units = used_so_far + rem_units
    projected_bill = tneb_domestic_bill(projected_units)

    insights = appliance_insights(df, projected_units)
    report = monthly_report(df)
    recent = df.tail(60).reset_index()
    recent_usage = [
        {"date": row["Date"].strftime("%Y-%m-%d"), "units": round(float(row["Total_Units"]), 2)}
        for _, row in recent.iterrows()
    ]

    return jsonify({
        "notEnoughHistory": False,
        "cycle": {
            "start": cycle_start.strftime("%Y-%m-%d"),
            "end": cycle_end.strftime("%Y-%m-%d"),
            "day": int(cycle_day),
            "usedSoFar": round(used_so_far, 2),
            "remainingDays": int(remaining_days),
            "forecastRemaining": round(rem_units, 2),
            "projectedUnits": round(projected_units, 2),
            "projectedBill": projected_bill,
        },
        "slabInfo": slab_info(projected_units),
        "billBreakdown": bill_breakdown(projected_units),
        "forecastRange": forecast_range(projected_units),
        "recommendations": dynamic_recommendations(df, projected_units, used_so_far, remaining_days),
        "applianceInsights": insights,
        "anomaly": anomaly_summary(df),
        "monthlyReport": report,
        "recentUsage": recent_usage,
    })


@app.route("/api/realtime", methods=["POST"])
def api_realtime():
    _, model, feature_cols, _ = initialize()
    payload = request.get_json(force=True) or {}

    date_value = payload.get("date") or datetime.now().strftime("%Y-%m-%d")
    today_date = pd.to_datetime(date_value)
    appliance_payload = payload.get("appliances", {}) or {}

    clean_appliances = {}
    for col in APPLIANCE_COLS:
        try:
            clean_appliances[col] = max(0.0, float(appliance_payload.get(col, 0) or 0))
        except Exception:
            clean_appliances[col] = 0.0

    appliance_total = sum(clean_appliances.values())

    if appliance_total > 0:
        today_units = appliance_total
    else:
        today_units = max(0.0, float(payload.get("units", 0) or 0))

    messages = []

    try:
        save_live_reading_to_db(today_date, today_units, clean_appliances)
        messages.append("Reading saved securely to your household database.")
    except Exception as exc:
        db.session.rollback()
        messages.append(f"Warning: reading could not be saved to database: {exc}")

    df = load_current_household_dataframe()
    household = get_current_household()
    save_abnormal_consumption_alert_if_needed(household, df)
    count = actual_reading_count(df)

    if count < MIN_READINGS_FOR_FORECAST:
        return jsonify({
            "messages": messages + [
                f"Forecasting activates after {MIN_READINGS_FOR_FORECAST} daily readings. Today's reading has been saved."
            ],
            "date": today_date.strftime("%Y-%m-%d"),
            "todayUnits": round(today_units, 2),
            "manualAppliances": clean_appliances,
            "manualApplianceTotal": round(appliance_total, 2),
            "readingsCount": count,
            "readingsNeeded": max(0, MIN_READINGS_FOR_FORECAST - count),
            "tomorrowPrediction": 0,
            "next60Units": 0,
            "next60Bill": 0,
            "next60Forecast": [],
            "forecastRange": forecast_range(0),
            "anomaly": anomaly_summary(df) if not df.empty else {"status": "Not enough history"},
            "cycle": None,
            "alert": {
                "level": "caution",
                "title": "More readings needed",
                "message": "Keep entering daily readings. Forecasting starts after 60 days.",
            },
        })

    cycle_start_value = payload.get("cycleStart")
    if cycle_start_value:
        cycle_start = pd.to_datetime(cycle_start_value)
    else:
        cycle_start = today_date - pd.Timedelta(days=59)

    tomorrow_forecast = forecast_from_series(model, df, feature_cols, 1)
    next60 = forecast_from_series(model, df, feature_cols, 60)

    next60_units = float(next60["units"].sum()) if len(next60) else 0.0
    next60_bill = tneb_domestic_bill(next60_units)
    cycle_end = cycle_start + pd.Timedelta(days=59)
    cycle_info = None

    if today_date > cycle_end:
        messages.append(f"Invalid cycle: selected cycle ended on {cycle_end.strftime('%Y-%m-%d')}.")
    elif today_date < cycle_start:
        messages.append("Invalid cycle: start date is after selected reading date.")
    else:
        cycle_day = (today_date - cycle_start).days + 1
        used_so_far = float(df.loc[cycle_start:today_date, "Total_Units"].sum())
        remaining = max(0, 60 - cycle_day)
        rem_forecast = forecast_from_series(model, df, feature_cols, remaining)
        rem_units = float(rem_forecast["units"].sum()) if len(rem_forecast) else 0.0
        projected_units = used_so_far + rem_units

        cycle_info = {
            "cycleStart": cycle_start.strftime("%Y-%m-%d"),
            "cycleEnd": cycle_end.strftime("%Y-%m-%d"),
            "cycleDay": int(cycle_day),
            "usedSoFar": round(used_so_far, 2),
            "remainingDays": int(remaining),
            "forecastRemaining": round(rem_units, 2),
            "projectedUnits": round(projected_units, 2),
            "projectedBill": tneb_domestic_bill(projected_units),
            "billBreakdown": bill_breakdown(projected_units),
            "alert": make_alerts(projected_units),
            "forecastRange": forecast_range(projected_units),
        }

    tomorrow_prediction = float(tomorrow_forecast.iloc[0]["units"]) if len(tomorrow_forecast) else 0.0

    return jsonify({
        "messages": messages,
        "date": today_date.strftime("%Y-%m-%d"),
        "todayUnits": round(today_units, 2),
        "manualAppliances": clean_appliances,
        "manualApplianceTotal": round(appliance_total, 2),
        "readingsCount": count,
        "readingsNeeded": 0,
        "tomorrowPrediction": round(tomorrow_prediction, 2),
        "next60Units": round(next60_units, 2),
        "next60Bill": next60_bill,
        "next60Forecast": next60.to_dict(orient="records"),
        "forecastRange": forecast_range(next60_units),
        "anomaly": anomaly_summary(df),
        "cycle": cycle_info,
        "alert": make_alerts(next60_units),
    })


@app.route("/api/download/monthly-report")
def download_monthly_report():
    df = load_current_household_dataframe()

    if df.empty:
        dff = pd.DataFrame()
    else:
        start = pd.to_datetime(request.args.get("start")) if request.args.get("start") else pd.Timestamp(df.index.max().year, df.index.max().month, 1)
        end = pd.to_datetime(request.args.get("end")) if request.args.get("end") else df.index.max()
        dff = df.loc[start:end].copy()

    appliances = available_appliances(dff) if not dff.empty else []
    rows = []

    if dff.empty:
        rows.append({"Metric": "Status", "Value": "No readings yet"})
    else:
        rows.extend([
            {"Metric": "Start", "Value": dff.index.min().strftime("%Y-%m-%d")},
            {"Metric": "End", "Value": dff.index.max().strftime("%Y-%m-%d")},
            {"Metric": "Total Units / kWh", "Value": round(float(dff["Total_Units"].sum()), 2)},
            {"Metric": "Average Daily Units / kWh", "Value": round(float(dff["Total_Units"].mean()), 2)},
        ])

        for col in appliances:
            rows.append({"Metric": col.replace("_", " "), "Value": round(float(dff[col].sum()), 2)})

    output = io.StringIO()
    pd.DataFrame(rows).to_csv(output, index=False)

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=monthly_energy_report.csv"},
    )


@app.route("/api/download/cycle-report")
def download_cycle_report():
    _, model, feature_cols, _ = initialize()
    df = load_current_household_dataframe()
    rows = []

    if df.empty:
        rows.append({"Metric": "Status", "Value": "No readings yet"})
    else:
        latest_date = df.index.max()
        cycle_start = pd.to_datetime(request.args.get("cycle_start")) if request.args.get("cycle_start") else latest_date - pd.Timedelta(days=59)
        cycle_end = cycle_start + pd.Timedelta(days=59)
        cycle_day = min(60, max(0, (latest_date - cycle_start).days + 1))

        if latest_date >= cycle_start:
            used_so_far = float(df.loc[cycle_start:min(latest_date, cycle_end), "Total_Units"].sum())
        else:
            used_so_far = 0.0

        remaining_days = max(0, 60 - cycle_day)
        rem_forecast = forecast_from_series(model, df, feature_cols, remaining_days) if actual_reading_count(df) >= MIN_READINGS_FOR_FORECAST else pd.DataFrame()
        rem_units = float(rem_forecast["units"].sum()) if len(rem_forecast) else 0.0
        projected_units = used_so_far + rem_units

        rows.extend([
            {"Metric": "Cycle Start", "Value": cycle_start.strftime("%Y-%m-%d")},
            {"Metric": "Cycle End", "Value": cycle_end.strftime("%Y-%m-%d")},
            {"Metric": "Cycle Day", "Value": cycle_day},
            {"Metric": "Units Used So Far / kWh", "Value": round(used_so_far, 2)},
            {"Metric": "Forecast Remaining Units / kWh", "Value": round(rem_units, 2)},
            {"Metric": "Projected Cycle Units / kWh", "Value": round(projected_units, 2)},
            {"Metric": "Projected Bill", "Value": tneb_domestic_bill(projected_units)},
        ])

    output = io.StringIO()
    pd.DataFrame(rows).to_csv(output, index=False)

    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=cycle_energy_report.csv"},
    )


@app.route("/api/performance")
def api_performance():
    base_df, _, feature_cols, _ = initialize()

    if base_df is None or base_df.empty or not feature_cols:
        return jsonify({
            "mae": None,
            "rmse": None,
            "r2": None,
            "wape": None,
            "message": "Training dataset or feature columns are not available.",
        })

    feature_df = create_features(base_df)
    feature_df = feature_df.dropna(subset=feature_cols + ["Total_Units"])

    n = len(feature_df)
    if n < 120:
        return jsonify({
            "mae": None,
            "rmse": None,
            "r2": None,
            "wape": None,
            "message": "Not enough rows for performance evaluation.",
        })

    test_size = 120 if n >= 360 else max(30, int(n * 0.2))
    train = feature_df.iloc[:-test_size]
    test = feature_df.iloc[-test_size:]

    model_eval = ExtraTreesRegressor(
        n_estimators=500,
        random_state=RANDOM_STATE,
        min_samples_leaf=2,
        n_jobs=-1,
    )

    y_train = np.log1p(train["Total_Units"]) if TARGET_TRANSFORM == "log1p" else train["Total_Units"]
    model_eval.fit(train[feature_cols], y_train)

    raw_pred = model_eval.predict(test[feature_cols])
    pred = np.expm1(raw_pred) if TARGET_TRANSFORM == "log1p" else raw_pred
    pred = np.maximum(0, pred)

    y = test["Total_Units"].values
    mae = mean_absolute_error(y, pred)
    rmse = mean_squared_error(y, pred) ** 0.5
    r2 = r2_score(y, pred)
    wape = np.abs(y - pred).sum() / y.sum() * 100 if y.sum() else 0

    return jsonify({
        "mae": round(float(mae), 3),
        "rmse": round(float(rmse), 3),
        "r2": round(float(r2), 3),
        "wape": round(float(wape), 2),
    })


# =====================================================
# CONSUMPTION LIMIT / TARGET + SMART ALERTS
# Add this block in app.py after forecast_from_series() is defined,
# or anywhere before if __name__ == "__main__".
# =====================================================

def get_db_text():
    from sqlalchemy import text
    return text


def parse_limit_date(value, fallback=None):
    value = (value or "").strip()
    if not value:
        return fallback
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except Exception:
        return fallback


def ensure_consumption_limit_table():
    text = get_db_text()
    db.session.execute(text("""
        CREATE EXTENSION IF NOT EXISTS pgcrypto;

        CREATE TABLE IF NOT EXISTS consumption_limits (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID NOT NULL,
            household_id UUID NOT NULL,
            limit_units NUMERIC(12,3) NOT NULL DEFAULT 500,
            period_start_date DATE NOT NULL,
            period_end_date DATE NOT NULL,
            threshold_1_percent INTEGER NOT NULL DEFAULT 70,
            threshold_2_percent INTEGER NOT NULL DEFAULT 85,
            threshold_3_percent INTEGER NOT NULL DEFAULT 95,
            notify_in_app BOOLEAN NOT NULL DEFAULT TRUE,
            notify_email BOOLEAN NOT NULL DEFAULT FALSE,
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """))
    db.session.commit()


def get_active_consumption_limit(household_id):
    text = get_db_text()
    ensure_consumption_limit_table()
    row = db.session.execute(text("""
        SELECT *
        FROM consumption_limits
        WHERE household_id = :household_id
          AND is_active = TRUE
        ORDER BY created_at DESC
        LIMIT 1
    """), {"household_id": str(household_id)}).mappings().first()
    return row


def insert_limit_alert_once(household_id, alert_type, severity, title, message, reading_date=None):
    """Create one alert per household/type/day to avoid duplicates."""
    text = get_db_text()
    reading_date = reading_date or datetime.utcnow().date()

    existing = db.session.execute(text("""
        SELECT id
        FROM alerts
        WHERE household_id = :household_id
          AND alert_type = :alert_type
          AND alert_date = :alert_date
        LIMIT 1
    """), {
        "household_id": str(household_id),
        "alert_type": alert_type,
        "alert_date": reading_date,
    }).first()

    if existing:
        return

    db.session.execute(text("""
        INSERT INTO alerts (
            id,
            household_id,
            alert_date,
            alert_type,
            severity,
            title,
            message,
            reading_date,
            is_read,
            created_at
        ) VALUES (
            gen_random_uuid(),
            :household_id,
            :alert_date,
            :alert_type,
            :severity,
            :title,
            :message,
            :reading_date,
            FALSE,
            NOW()
        )
    """), {
        "household_id": str(household_id),
        "alert_date": reading_date,
        "alert_type": alert_type,
        "severity": severity,
        "title": title,
        "message": message,
        "reading_date": reading_date,
    })
    try:
        send_push_for_alert(
        household_id,
        alert_type,
        severity,
        title,
        message,
    )
    except Exception as exc:
        print(f"Push alert warning: {exc}")



def calculate_consumption_limit_status(limit_row, df, projected_units=None):
    if not limit_row:
        return {
            "configured": False,
            "message": "No consumption limit configured."
        }

    limit_units = float(limit_row["limit_units"] or 0)
    start_date = pd.to_datetime(limit_row["period_start_date"])
    end_date = pd.to_datetime(limit_row["period_end_date"])
    today = pd.Timestamp(datetime.now().date())

    if df is None or df.empty:
        used_units = 0.0
    else:
        dff = df.loc[start_date:min(today, end_date)].copy()
        used_units = float(dff["Total_Units"].sum()) if len(dff) else 0.0

    period_days = max(1, int((end_date - start_date).days) + 1)
    elapsed_days = max(1, min(period_days, int((min(today, end_date) - start_date).days) + 1))
    remaining_days = max(0, int((end_date - today).days))

    avg_daily = used_units / elapsed_days if elapsed_days else 0.0
    remaining_units = max(0.0, limit_units - used_units)
    safe_daily_remaining = remaining_units / max(1, remaining_days) if remaining_days > 0 else 0.0

    if projected_units is None:
        projected_units = used_units + (avg_daily * remaining_days)

    projected_units = float(projected_units or 0)
    usage_percent = (used_units / limit_units * 100) if limit_units else 0.0
    projected_percent = (projected_units / limit_units * 100) if limit_units else 0.0
    predicted_excess = max(0.0, projected_units - limit_units)

    threshold_1 = int(limit_row["threshold_1_percent"] or 70)
    threshold_2 = int(limit_row["threshold_2_percent"] or 85)
    threshold_3 = int(limit_row["threshold_3_percent"] or 95)

    risk = "good"
    title = "Within limit"
    message = "Your usage is currently within your configured consumption limit."

    if predicted_excess > 0:
        risk = "danger"
        title = "Limit likely to exceed"
        reduce_per_day = predicted_excess / max(1, remaining_days)
        message = (
            f"You may exceed your {limit_units:.0f} unit limit by about "
            f"{predicted_excess:.1f} units. Reduce around {reduce_per_day:.2f} units/day."
        )
    elif usage_percent >= threshold_3:
        risk = "danger"
        title = f"{threshold_3}% limit reached"
        message = f"You have used {used_units:.1f} of {limit_units:.0f} units. Reduce usage immediately."
    elif usage_percent >= threshold_2:
        risk = "warning"
        title = f"{threshold_2}% limit reached"
        message = f"You are nearing your consumption limit. Remaining units: {remaining_units:.1f}."
    elif usage_percent >= threshold_1:
        risk = "caution"
        title = f"{threshold_1}% limit reached"
        message = f"You have crossed {threshold_1}% of your limit. Monitor usage carefully."

    return {
        "configured": True,
        "id": str(limit_row["id"]),
        "limitUnits": round(limit_units, 2),
        "periodStart": start_date.strftime("%Y-%m-%d"),
        "periodEnd": end_date.strftime("%Y-%m-%d"),
        "usedUnits": round(used_units, 2),
        "remainingUnits": round(remaining_units, 2),
        "usagePercent": round(usage_percent, 2),
        "projectedUnits": round(projected_units, 2),
        "projectedPercent": round(projected_percent, 2),
        "predictedExcess": round(predicted_excess, 2),
        "periodDays": period_days,
        "elapsedDays": elapsed_days,
        "remainingDays": remaining_days,
        "avgDaily": round(avg_daily, 2),
        "safeDailyRemaining": round(safe_daily_remaining, 2),
        "thresholds": [threshold_1, threshold_2, threshold_3],
        "risk": risk,
        "title": title,
        "message": message,
    }


def evaluate_and_store_consumption_limit_alerts(household, df, projected_units=None):
    limit_row = get_active_consumption_limit(household.id)
    status = calculate_consumption_limit_status(limit_row, df, projected_units)

    if not status.get("configured"):
        return status

    today = datetime.utcnow().date()

    if status["predictedExcess"] > 0:
        insert_limit_alert_once(
            household.id,
            "limit_predicted_exceed",
            "critical",
            "Consumption limit likely to exceed",
            status["message"],
            today,
        )
    elif status["usagePercent"] >= status["thresholds"][2]:
        insert_limit_alert_once(
            household.id,
            "limit_95_percent",
            "critical",
            "95% consumption limit reached",
            status["message"],
            today,
        )
    elif status["usagePercent"] >= status["thresholds"][1]:
        insert_limit_alert_once(
            household.id,
            "limit_85_percent",
            "warning",
            "85% consumption limit reached",
            status["message"],
            today,
        )
    elif status["usagePercent"] >= status["thresholds"][0]:
        insert_limit_alert_once(
            household.id,
            "limit_70_percent",
            "info",
            "70% consumption limit reached",
            status["message"],
            today,
        )

    try:
        db.session.commit()
    except Exception:
        db.session.rollback()

    return status


@app.route("/api/consumption-limit", methods=["GET"])
def api_get_consumption_limit():
    _, model, feature_cols, _ = initialize()
    household = get_current_household()
    df = load_current_household_dataframe()
    limit_row = get_active_consumption_limit(household.id)

    projected_units = None
    if limit_row and df is not None and not df.empty:
        end_date = pd.to_datetime(limit_row["period_end_date"])
        latest_date = df.index.max()
        remaining_days = max(0, int((end_date - latest_date).days))
        if actual_reading_count(df) >= MIN_READINGS_FOR_FORECAST and remaining_days > 0:
            forecast_df = forecast_from_series(model, df, feature_cols, remaining_days)
            used = float(df.loc[pd.to_datetime(limit_row["period_start_date"]):latest_date, "Total_Units"].sum())
            projected_units = used + (float(forecast_df["units"].sum()) if len(forecast_df) else 0.0)

    status = evaluate_and_store_consumption_limit_alerts(household, df, projected_units)
    return jsonify(status)


@app.route("/api/consumption-limit", methods=["POST"])
def api_save_consumption_limit():
    text = get_db_text()
    ensure_consumption_limit_table()

    household = get_current_household()
    payload = request.get_json(force=True) or {}

    limit_units = max(1.0, float(payload.get("limitUnits", 500) or 500))
    period_start = parse_limit_date(payload.get("periodStart"), datetime.utcnow().date())
    period_end = parse_limit_date(payload.get("periodEnd"), period_start + pd.Timedelta(days=59))

    if hasattr(period_end, "date"):
        period_end = period_end.date()
    if hasattr(period_start, "date"):
        period_start = period_start.date()

    if period_end < period_start:
        return jsonify({"success": False, "error": "Period end date must be after start date."}), 400

    threshold_1 = int(payload.get("threshold1", 70) or 70)
    threshold_2 = int(payload.get("threshold2", 85) or 85)
    threshold_3 = int(payload.get("threshold3", 95) or 95)

    threshold_1 = min(max(threshold_1, 1), 100)
    threshold_2 = min(max(threshold_2, threshold_1), 100)
    threshold_3 = min(max(threshold_3, threshold_2), 100)

    try:
        db.session.execute(text("""
            UPDATE consumption_limits
            SET is_active = FALSE,
                updated_at = NOW()
            WHERE household_id = :household_id
              AND is_active = TRUE
        """), {"household_id": str(household.id)})

        db.session.execute(text("""
            INSERT INTO consumption_limits (
                id,
                user_id,
                household_id,
                limit_units,
                period_start_date,
                period_end_date,
                threshold_1_percent,
                threshold_2_percent,
                threshold_3_percent,
                notify_in_app,
                notify_email,
                is_active,
                created_at,
                updated_at
            ) VALUES (
                gen_random_uuid(),
                :user_id,
                :household_id,
                :limit_units,
                :period_start,
                :period_end,
                :threshold_1,
                :threshold_2,
                :threshold_3,
                TRUE,
                FALSE,
                TRUE,
                NOW(),
                NOW()
            )
        """), {
            "user_id": str(current_user.id),
            "household_id": str(household.id),
            "limit_units": limit_units,
            "period_start": period_start,
            "period_end": period_end,
            "threshold_1": threshold_1,
            "threshold_2": threshold_2,
            "threshold_3": threshold_3,
        })

        db.session.commit()

        df = load_current_household_dataframe()
        limit_row = get_active_consumption_limit(household.id)
        status = calculate_consumption_limit_status(limit_row, df)

        return jsonify({
            "success": True,
            "message": "Consumption limit saved successfully.",
            "status": status,
        })

    except Exception as exc:
        db.session.rollback()
        return jsonify({"success": False, "error": str(exc)}), 400


# =====================================================
# ALERT READ / UNREAD MANAGEMENT
# =====================================================

@app.route("/api/alerts", methods=["GET"])
def api_get_alerts():
    from sqlalchemy import text

    household = get_current_household()

    rows = db.session.execute(text("""
        SELECT
            id,
            alert_date,
            alert_type,
            severity,
            title,
            message,
            reading_date,
            is_read,
            created_at
        FROM alerts
        WHERE household_id = :household_id
        ORDER BY is_read ASC, created_at DESC
        LIMIT 100
    """), {"household_id": str(household.id)}).mappings().all()

    unread_count = db.session.execute(text("""
        SELECT COUNT(*)
        FROM alerts
        WHERE household_id = :household_id
          AND COALESCE(is_read, FALSE) = FALSE
    """), {"household_id": str(household.id)}).scalar() or 0

    return jsonify({
        "unreadCount": int(unread_count),
        "alerts": [
            {
                "id": str(row["id"]),
                "alertDate": row["alert_date"].isoformat() if row["alert_date"] else None,
                "alertType": row["alert_type"],
                "severity": row["severity"],
                "title": row["title"],
                "message": row["message"],
                "readingDate": row["reading_date"].isoformat() if row["reading_date"] else None,
                "isRead": bool(row["is_read"]),
                "createdAt": row["created_at"].isoformat() if row["created_at"] else None,
            }
            for row in rows
        ]
    })


@app.route("/api/alerts/<alert_id>/read", methods=["POST"])
def api_mark_alert_read(alert_id):
    from sqlalchemy import text

    household = get_current_household()

    try:
        result = db.session.execute(text("""
            UPDATE alerts
            SET is_read = TRUE
            WHERE id = CAST(:alert_id AS UUID)
              AND household_id = :household_id
        """), {
            "alert_id": alert_id,
            "household_id": str(household.id),
        })

        db.session.commit()

        return jsonify({
            "success": True,
            "updated": int(result.rowcount or 0),
            "message": "Alert marked as read."
        })

    except Exception as exc:
        db.session.rollback()
        return jsonify({
            "success": False,
            "error": str(exc)
        }), 400


@app.route("/api/alerts/read-all", methods=["POST"])
def api_mark_all_alerts_read():
    from sqlalchemy import text

    household = get_current_household()

    try:
        result = db.session.execute(text("""
            UPDATE alerts
            SET is_read = TRUE
            WHERE household_id = :household_id
              AND COALESCE(is_read, FALSE) = FALSE
        """), {
            "household_id": str(household.id),
        })

        db.session.commit()

        return jsonify({
            "success": True,
            "updated": int(result.rowcount or 0),
            "message": "All alerts marked as read."
        })

    except Exception as exc:
        db.session.rollback()
        return jsonify({
            "success": False,
            "error": str(exc)
        }), 400


@app.route("/api/alerts/unread-count", methods=["GET"])
def api_alert_unread_count():
    from sqlalchemy import text

    household = get_current_household()

    unread_count = db.session.execute(text("""
        SELECT COUNT(*)
        FROM alerts
        WHERE household_id = :household_id
          AND COALESCE(is_read, FALSE) = FALSE
    """), {
        "household_id": str(household.id),
    }).scalar() or 0

    return jsonify({
        "unreadCount": int(unread_count)
    })

# =====================================================
# BROWSER / PWA PUSH NOTIFICATIONS
# Requires: pip install pywebpush cryptography
# =====================================================

def ensure_push_subscriptions_table():
    from sqlalchemy import text

    db.session.execute(text("""
        CREATE EXTENSION IF NOT EXISTS pgcrypto;

        CREATE TABLE IF NOT EXISTS push_subscriptions (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID NOT NULL,
            household_id UUID NOT NULL,
            endpoint TEXT NOT NULL,
            p256dh TEXT NOT NULL,
            auth TEXT NOT NULL,
            user_agent TEXT,
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        CREATE UNIQUE INDEX IF NOT EXISTS ux_push_subscriptions_endpoint
        ON push_subscriptions(endpoint);
    """))

    db.session.commit()


def get_vapid_public_key():
    return os.environ.get("VAPID_PUBLIC_KEY", "").strip()


def get_vapid_private_key_value():
    private_key_file = os.environ.get("VAPID_PRIVATE_KEY_FILE", "").strip()
    private_key_text = os.environ.get("VAPID_PRIVATE_KEY", "").strip()

    if private_key_file:
        path = private_key_file

        if not os.path.isabs(path):
            path = os.path.join(BASE_DIR, path)

        if os.path.exists(path):
            return path

    return private_key_text


def send_push_to_household(household_id, title, message, url="/"):
    """Send browser push notification to all active subscriptions for a household."""
    from sqlalchemy import text

    public_key = get_vapid_public_key()
    private_key = get_vapid_private_key_value()
    claim_email = os.environ.get("VAPID_CLAIM_EMAIL", "mailto:admin@tnebsmart.local")

    if not public_key or not private_key:
        print("Push skipped: VAPID keys are not configured.")
        return {
            "sent": 0,
            "skipped": True,
            "reason": "VAPID keys missing",
        }

    try:
        from pywebpush import webpush, WebPushException
    except Exception as exc:
        print(f"Push skipped: pywebpush not installed: {exc}")
        return {
            "sent": 0,
            "skipped": True,
            "reason": "pywebpush missing",
        }

    rows = db.session.execute(text("""
        SELECT id, endpoint, p256dh, auth
        FROM push_subscriptions
        WHERE household_id = :household_id
          AND is_active = TRUE
    """), {
        "household_id": str(household_id),
    }).mappings().all()

    sent = 0

    for row in rows:
        subscription_info = {
            "endpoint": row["endpoint"],
            "keys": {
                "p256dh": row["p256dh"],
                "auth": row["auth"],
            },
        }

        payload = json.dumps({
            "title": title,
            "body": message,
            "url": url,
            "tag": "tneb-smart-alert",
        })

        try:
            webpush(
                subscription_info=subscription_info,
                data=payload,
                vapid_private_key=private_key,
                vapid_claims={
                    "sub": claim_email,
                },
            )

            sent += 1

        except WebPushException as exc:
            status_code = getattr(getattr(exc, "response", None), "status_code", None)

            print(f"Push send warning: {exc}")

            if status_code in (404, 410):
                db.session.execute(text("""
                    UPDATE push_subscriptions
                    SET is_active = FALSE,
                        updated_at = NOW()
                    WHERE id = :id
                """), {
                    "id": str(row["id"]),
                })

                db.session.commit()

        except Exception as exc:
            print(f"Push send warning: {exc}")

    return {
        "sent": sent,
        "skipped": False,
    }

def send_push_for_alert(household_id, alert_type, severity, title, message):
    """
    Sends notifications for important alerts and respects notification preferences.
    Channels:
    - browser push
    - WhatsApp
    - SMS
    """
    important_alert_types = {
        "limit_predicted_exceed",
        "limit_95_percent",
        "abnormal_consumption",
        "high_bill_prediction",
    }

    important_severities = {
        "critical",
        "warning",
    }

    if alert_type not in important_alert_types and severity not in important_severities:
        return {
            "push": {"sent": 0, "skipped": True, "reason": "Not important enough"},
            "sms": {"sent": 0, "skipped": True, "reason": "Not important enough"},
            "whatsapp": {"sent": 0, "skipped": True, "reason": "Not important enough"},
        }

    prefs = get_notification_preferences(household_id)

    if not alert_type_allowed_by_preferences(prefs, alert_type):
        return {
            "push": {"sent": 0, "skipped": True, "reason": "Disabled by notification preferences"},
            "sms": {"sent": 0, "skipped": True, "reason": "Disabled by notification preferences"},
            "whatsapp": {"sent": 0, "skipped": True, "reason": "Disabled by notification preferences"},
        }

    push_title = f"Energy Lens: {title}"
    push_message = message or "You have a new electricity usage alert."

    results = {}

    # -----------------------------
    # Browser push
    # -----------------------------
    if bool(prefs["push_enabled"]):
        if bool(prefs["max_one_push_per_day"]) and push_already_sent_today(household_id, alert_type):
            results["push"] = {
                "sent": 0,
                "skipped": True,
                "reason": "Max one push per day enabled",
            }
        else:
            results["push"] = send_push_to_household(
                household_id,
                push_title,
                push_message,
                "/",
            )

            try:
                if results["push"].get("sent", 0) > 0:
                    log_push_notification(
                        household_id,
                        alert_type,
                        push_title,
                        push_message,
                        "sent",
                        None,
                    )
                elif results["push"].get("skipped"):
                    log_push_notification(
                        household_id,
                        alert_type,
                        push_title,
                        push_message,
                        "skipped",
                        results["push"].get("reason"),
                    )
            except Exception as exc:
                print(f"Push log warning: {exc}")
    else:
        results["push"] = {
            "sent": 0,
            "skipped": True,
            "reason": "Push disabled",
        }

    # -----------------------------
    # WhatsApp
    # -----------------------------
    if bool(prefs["whatsapp_enabled"]):
        results["whatsapp"] = send_whatsapp_for_alert(
            household_id,
            alert_type,
            push_title,
            push_message,
        )
    else:
        results["whatsapp"] = {
            "sent": 0,
            "skipped": True,
            "reason": "WhatsApp disabled",
        }

    # -----------------------------
    # SMS
    # -----------------------------
    if bool(prefs["sms_enabled"]):
        results["sms"] = send_sms_for_alert(
            household_id,
            alert_type,
            push_title,
            push_message,
        )
    else:
        results["sms"] = {
            "sent": 0,
            "skipped": True,
            "reason": "SMS disabled",
        }

    return results

@app.route("/api/push/vapid-public-key", methods=["GET"])
def api_push_vapid_public_key():
    public_key = get_vapid_public_key()

    if not public_key:
        return jsonify({
            "configured": False,
            "error": "VAPID_PUBLIC_KEY is not configured in .env",
        }), 400

    return jsonify({
        "configured": True,
        "publicKey": public_key,
    })


@app.route("/api/push/subscribe", methods=["POST"])
def api_push_subscribe():
    from sqlalchemy import text

    ensure_push_subscriptions_table()

    household = get_current_household()
    payload = request.get_json(force=True) or {}

    endpoint = payload.get("endpoint")
    keys = payload.get("keys", {}) or {}
    p256dh = keys.get("p256dh")
    auth = keys.get("auth")

    if not endpoint or not p256dh or not auth:
        return jsonify({
            "success": False,
            "error": "Invalid push subscription payload.",
        }), 400

    try:
        db.session.execute(text("""
            INSERT INTO push_subscriptions (
                id,
                user_id,
                household_id,
                endpoint,
                p256dh,
                auth,
                user_agent,
                is_active,
                created_at,
                updated_at
            ) VALUES (
                gen_random_uuid(),
                :user_id,
                :household_id,
                :endpoint,
                :p256dh,
                :auth,
                :user_agent,
                TRUE,
                NOW(),
                NOW()
            )
            ON CONFLICT (endpoint)
            DO UPDATE SET
                user_id = EXCLUDED.user_id,
                household_id = EXCLUDED.household_id,
                p256dh = EXCLUDED.p256dh,
                auth = EXCLUDED.auth,
                user_agent = EXCLUDED.user_agent,
                is_active = TRUE,
                updated_at = NOW()
        """), {
            "user_id": str(current_user.id),
            "household_id": str(household.id),
            "endpoint": endpoint,
            "p256dh": p256dh,
            "auth": auth,
            "user_agent": request.headers.get("User-Agent"),
        })

        db.session.commit()

        return jsonify({
            "success": True,
            "message": "Push notifications enabled for this browser.",
        })

    except Exception as exc:
        db.session.rollback()

        return jsonify({
            "success": False,
            "error": str(exc),
        }), 400


@app.route("/api/push/test", methods=["POST"])
def api_push_test():
    """
    Sends the latest real alert as a push notification.
    If no alert exists, sends a fallback test notification.
    """
    from sqlalchemy import text

    household = get_current_household()

    latest_alert = db.session.execute(text("""
        SELECT
            alert_type,
            severity,
            title,
            message,
            alert_date,
            created_at
        FROM alerts
        WHERE household_id = :household_id
        ORDER BY created_at DESC
        LIMIT 1
    """), {
        "household_id": str(household.id),
    }).mappings().first()

    if latest_alert:
        push_title = f"TNEB Smart: {latest_alert['title']}"
        push_message = latest_alert["message"] or "You have a new electricity alert."
    else:
        push_title = "TNEB Smart Test Notification"
        push_message = "Push notifications are working for your account."

    result = send_push_to_household(
        household.id,
        push_title,
        push_message,
        "/",
    )

    return jsonify({
        "success": True,
        "message": "Latest alert push sent.",
        "alertUsed": {
            "title": latest_alert["title"] if latest_alert else None,
            "message": latest_alert["message"] if latest_alert else None,
            "type": latest_alert["alert_type"] if latest_alert else None,
            "severity": latest_alert["severity"] if latest_alert else None,
        } if latest_alert else None,
        "result": result,
    })


# =====================================================
# NOTIFICATION PREFERENCES
# =====================================================

def ensure_notification_preferences_table():
    from sqlalchemy import text

    db.session.execute(text("""
        CREATE EXTENSION IF NOT EXISTS pgcrypto;

        CREATE TABLE IF NOT EXISTS notification_preferences (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID NOT NULL,
            household_id UUID NOT NULL,
            push_enabled BOOLEAN NOT NULL DEFAULT TRUE,
            limit_alerts_enabled BOOLEAN NOT NULL DEFAULT TRUE,
            abnormal_alerts_enabled BOOLEAN NOT NULL DEFAULT TRUE,
            bill_alerts_enabled BOOLEAN NOT NULL DEFAULT TRUE,
            max_one_push_per_day BOOLEAN NOT NULL DEFAULT FALSE,
            whatsapp_enabled BOOLEAN NOT NULL DEFAULT FALSE,
            sms_enabled BOOLEAN NOT NULL DEFAULT FALSE,
            email_enabled BOOLEAN NOT NULL DEFAULT FALSE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );

        CREATE UNIQUE INDEX IF NOT EXISTS ux_notification_preferences_household
        ON notification_preferences(household_id);

        CREATE TABLE IF NOT EXISTS notification_logs (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id UUID,
            household_id UUID NOT NULL,
            alert_type TEXT,
            channel TEXT NOT NULL DEFAULT 'push',
            title TEXT,
            message TEXT,
            status TEXT NOT NULL DEFAULT 'sent',
            error_message TEXT,
            sent_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
    """))

    db.session.commit()


def get_notification_preferences(household_id, user_id=None):
    from sqlalchemy import text

    ensure_notification_preferences_table()

    row = db.session.execute(text("""
        SELECT *
        FROM notification_preferences
        WHERE household_id = :household_id
        LIMIT 1
    """), {
        "household_id": str(household_id),
    }).mappings().first()

    if row:
        return row

    if user_id is None:
        user_id = current_user.id

    db.session.execute(text("""
        INSERT INTO notification_preferences (
            id,
            user_id,
            household_id,
            push_enabled,
            limit_alerts_enabled,
            abnormal_alerts_enabled,
            bill_alerts_enabled,
            max_one_push_per_day,
            whatsapp_enabled,
            sms_enabled,
            email_enabled,
            created_at,
            updated_at
        ) VALUES (
            gen_random_uuid(),
            :user_id,
            :household_id,
            TRUE,
            TRUE,
            TRUE,
            TRUE,
            FALSE,
            FALSE,
            FALSE,
            FALSE,
            NOW(),
            NOW()
        )
    """), {
        "user_id": str(user_id),
        "household_id": str(household_id),
    })

    db.session.commit()

    row = db.session.execute(text("""
        SELECT *
        FROM notification_preferences
        WHERE household_id = :household_id
        LIMIT 1
    """), {
        "household_id": str(household_id),
    }).mappings().first()

    return row


def alert_type_allowed_by_preferences(prefs, alert_type):
    if not prefs:
        return True

    if not bool(prefs["push_enabled"]):
        return False

    limit_types = {
        "limit_predicted_exceed",
        "limit_70_percent",
        "limit_85_percent",
        "limit_95_percent",
    }

    abnormal_types = {
        "abnormal_consumption",
    }

    bill_types = {
        "high_bill_prediction",
    }

    if alert_type in limit_types and not bool(prefs["limit_alerts_enabled"]):
        return False

    if alert_type in abnormal_types and not bool(prefs["abnormal_alerts_enabled"]):
        return False

    if alert_type in bill_types and not bool(prefs["bill_alerts_enabled"]):
        return False

    return True


def push_already_sent_today(household_id, alert_type):
    from sqlalchemy import text

    count = db.session.execute(text("""
        SELECT COUNT(*)
        FROM notification_logs
        WHERE household_id = :household_id
          AND alert_type = :alert_type
          AND channel = 'push'
          AND sent_at::date = CURRENT_DATE
          AND status = 'sent'
    """), {
        "household_id": str(household_id),
        "alert_type": alert_type,
    }).scalar() or 0

    return int(count) > 0


def log_push_notification(household_id, alert_type, title, message, status="sent", error_message=None):
    from sqlalchemy import text

    user_id = getattr(current_user, "id", None)

    db.session.execute(text("""
        INSERT INTO notification_logs (
            id,
            user_id,
            household_id,
            alert_type,
            channel,
            title,
            message,
            status,
            error_message,
            sent_at
        ) VALUES (
            gen_random_uuid(),
            :user_id,
            :household_id,
            :alert_type,
            'push',
            :title,
            :message,
            :status,
            :error_message,
            NOW()
        )
    """), {
        "user_id": str(user_id) if user_id else None,
        "household_id": str(household_id),
        "alert_type": alert_type,
        "title": title,
        "message": message,
        "status": status,
        "error_message": error_message,
    })

    db.session.commit()


@app.route("/api/notification-preferences", methods=["GET"])
def api_get_notification_preferences():
    household = get_current_household()
    prefs = get_notification_preferences(household.id, current_user.id)

    return jsonify({
        "pushEnabled": bool(prefs["push_enabled"]),
        "limitAlertsEnabled": bool(prefs["limit_alerts_enabled"]),
        "abnormalAlertsEnabled": bool(prefs["abnormal_alerts_enabled"]),
        "billAlertsEnabled": bool(prefs["bill_alerts_enabled"]),
        "maxOnePushPerDay": bool(prefs["max_one_push_per_day"]),
        "whatsappEnabled": bool(prefs["whatsapp_enabled"]),
        "smsEnabled": bool(prefs["sms_enabled"]),
        "emailEnabled": bool(prefs["email_enabled"]),
    })


@app.route("/api/notification-preferences", methods=["POST"])
def api_save_notification_preferences():
    from sqlalchemy import text

    ensure_notification_preferences_table()

    household = get_current_household()
    payload = request.get_json(force=True) or {}

    push_enabled = bool(payload.get("pushEnabled", True))
    limit_alerts_enabled = bool(payload.get("limitAlertsEnabled", True))
    abnormal_alerts_enabled = bool(payload.get("abnormalAlertsEnabled", True))
    bill_alerts_enabled = bool(payload.get("billAlertsEnabled", True))
    max_one_push_per_day = bool(payload.get("maxOnePushPerDay", False))

    # Placeholders for later channels.
    whatsapp_enabled = bool(payload.get("whatsappEnabled", False))
    sms_enabled = bool(payload.get("smsEnabled", False))
    email_enabled = bool(payload.get("emailEnabled", False))

    try:
        db.session.execute(text("""
            INSERT INTO notification_preferences (
                id,
                user_id,
                household_id,
                push_enabled,
                limit_alerts_enabled,
                abnormal_alerts_enabled,
                bill_alerts_enabled,
                max_one_push_per_day,
                whatsapp_enabled,
                sms_enabled,
                email_enabled,
                created_at,
                updated_at
            ) VALUES (
                gen_random_uuid(),
                :user_id,
                :household_id,
                :push_enabled,
                :limit_alerts_enabled,
                :abnormal_alerts_enabled,
                :bill_alerts_enabled,
                :max_one_push_per_day,
                :whatsapp_enabled,
                :sms_enabled,
                :email_enabled,
                NOW(),
                NOW()
            )
            ON CONFLICT (household_id)
            DO UPDATE SET
                push_enabled = EXCLUDED.push_enabled,
                limit_alerts_enabled = EXCLUDED.limit_alerts_enabled,
                abnormal_alerts_enabled = EXCLUDED.abnormal_alerts_enabled,
                bill_alerts_enabled = EXCLUDED.bill_alerts_enabled,
                max_one_push_per_day = EXCLUDED.max_one_push_per_day,
                whatsapp_enabled = EXCLUDED.whatsapp_enabled,
                sms_enabled = EXCLUDED.sms_enabled,
                email_enabled = EXCLUDED.email_enabled,
                updated_at = NOW()
        """), {
            "user_id": str(current_user.id),
            "household_id": str(household.id),
            "push_enabled": push_enabled,
            "limit_alerts_enabled": limit_alerts_enabled,
            "abnormal_alerts_enabled": abnormal_alerts_enabled,
            "bill_alerts_enabled": bill_alerts_enabled,
            "max_one_push_per_day": max_one_push_per_day,
            "whatsapp_enabled": whatsapp_enabled,
            "sms_enabled": sms_enabled,
            "email_enabled": email_enabled,
        })

        db.session.commit()

        return jsonify({
            "success": True,
            "message": "Notification preferences saved successfully."
        })

    except Exception as exc:
        db.session.rollback()

        return jsonify({
            "success": False,
            "error": str(exc)
        }), 400

# =====================================================
# SMS / WHATSAPP NOTIFICATION HELPERS
# Provider: Twilio
# =====================================================

def normalize_phone_e164(phone):
    """
    Converts Indian 10-digit phone numbers to E.164 format.
    Example: 7904147421 -> +917904147421
    """
    phone = str(phone or "").strip()

    if not phone:
        return None

    phone = phone.replace(" ", "").replace("-", "").replace("(", "").replace(")", "")

    if phone.startswith("+"):
        return phone

    digits = "".join(ch for ch in phone if ch.isdigit())

    if len(digits) == 10:
        return "+91" + digits

    if len(digits) == 12 and digits.startswith("91"):
        return "+" + digits

    return None


def get_household_user_phone(household_id):
    from sqlalchemy import text

    row = db.session.execute(text("""
        SELECT u.phone
        FROM households h
        JOIN users u ON u.id = h.user_id
        WHERE h.id = :household_id
        LIMIT 1
    """), {
        "household_id": str(household_id),
    }).mappings().first()

    if not row:
        return None

    return normalize_phone_e164(row["phone"])


def log_external_notification(household_id, alert_type, channel, title, message, status, error_message=None):
    from sqlalchemy import text

    try:
        user_id = getattr(current_user, "id", None)

        db.session.execute(text("""
            INSERT INTO notification_logs (
                id,
                user_id,
                household_id,
                alert_type,
                channel,
                title,
                message,
                status,
                error_message,
                sent_at
            ) VALUES (
                gen_random_uuid(),
                :user_id,
                :household_id,
                :alert_type,
                :channel,
                :title,
                :message,
                :status,
                :error_message,
                NOW()
            )
        """), {
            "user_id": str(user_id) if user_id else None,
            "household_id": str(household_id),
            "alert_type": alert_type,
            "channel": channel,
            "title": title,
            "message": message,
            "status": status,
            "error_message": error_message,
        })

        db.session.commit()

    except Exception as exc:
        db.session.rollback()
        print(f"Notification log warning: {exc}")


def get_twilio_client():
    sid = os.environ.get("TWILIO_ACCOUNT_SID", "").strip()
    token = os.environ.get("TWILIO_AUTH_TOKEN", "").strip()

    if not sid or not token:
        return None, "Twilio credentials missing"

    try:
        from twilio.rest import Client
        return Client(sid, token), None
    except Exception as exc:
        return None, str(exc)


def send_sms_for_alert(household_id, alert_type, title, message):
    """
    Sends SMS using Fast2SMS for Indian phone numbers.
    Requires FAST2SMS_API_KEY in .env.
    """
    api_key = os.environ.get("FAST2SMS_API_KEY", "").strip()
    route = os.environ.get("FAST2SMS_ROUTE", "q").strip() or "q"

    if not api_key:
        log_external_notification(
            household_id,
            alert_type,
            "sms",
            title,
            message,
            "skipped",
            "FAST2SMS_API_KEY missing",
        )
        return {
            "sent": 0,
            "skipped": True,
            "reason": "FAST2SMS_API_KEY missing",
        }

    phone = get_household_user_phone(household_id)

    if not phone:
        log_external_notification(
            household_id,
            alert_type,
            "sms",
            title,
            message,
            "skipped",
            "User phone missing or invalid",
        )
        return {
            "sent": 0,
            "skipped": True,
            "reason": "User phone missing or invalid",
        }

    mobile = phone.replace("+91", "").replace("+", "").strip()

    if len(mobile) != 10 or not mobile.isdigit():
        log_external_notification(
            household_id,
            alert_type,
            "sms",
            title,
            message,
            "skipped",
            "Fast2SMS requires a valid 10-digit Indian mobile number",
        )
        return {
            "sent": 0,
            "skipped": True,
            "reason": "Invalid Indian mobile number",
        }

    sms_text = f"{title}\n{message}"[:900]

    try:
        import requests

        url = "https://www.fast2sms.com/dev/bulkV2"

        headers = {
            "authorization": api_key,
            "Content-Type": "application/json",
        }

        payload = {
            "route": route,
            "message": sms_text,
            "language": "english",
            "flash": 0,
            "numbers": mobile,
        }

        response = requests.post(url, json=payload, headers=headers, timeout=15)

        try:
            data = response.json()
        except Exception:
            data = {
                "raw": response.text,
            }

        if response.status_code >= 400 or data.get("return") is False:
            error_message = str(data)

            log_external_notification(
                household_id,
                alert_type,
                "sms",
                title,
                message,
                "failed",
                error_message,
            )

            print(f"Fast2SMS send warning: {error_message}")

            return {
                "sent": 0,
                "skipped": False,
                "error": error_message,
            }

        log_external_notification(
            household_id,
            alert_type,
            "sms",
            title,
            message,
            "sent",
            None,
        )

        return {
            "sent": 1,
            "skipped": False,
            "provider": "fast2sms",
            "response": data,
        }

    except Exception as exc:
        log_external_notification(
            household_id,
            alert_type,
            "sms",
            title,
            message,
            "failed",
            str(exc),
        )

        print(f"Fast2SMS send warning: {exc}")

        return {
            "sent": 0,
            "skipped": False,
            "error": str(exc),
        }


def send_whatsapp_for_alert(household_id, alert_type, title, message):
    """
    Sends WhatsApp message using Twilio WhatsApp.
    For Twilio sandbox, the user must join the sandbox first.
    """
    whatsapp_from = os.environ.get("TWILIO_WHATSAPP_FROM", "").strip()

    if not whatsapp_from:
        log_external_notification(
            household_id,
            alert_type,
            "whatsapp",
            title,
            message,
            "skipped",
            "TWILIO_WHATSAPP_FROM missing",
        )
        return {
            "sent": 0,
            "skipped": True,
            "reason": "TWILIO_WHATSAPP_FROM missing",
        }

    phone = get_household_user_phone(household_id)

    if not phone:
        log_external_notification(
            household_id,
            alert_type,
            "whatsapp",
            title,
            message,
            "skipped",
            "User phone missing or invalid",
        )
        return {
            "sent": 0,
            "skipped": True,
            "reason": "User phone missing or invalid",
        }

    client, error = get_twilio_client()

    if not client:
        log_external_notification(
            household_id,
            alert_type,
            "whatsapp",
            title,
            message,
            "skipped",
            error,
        )
        return {
            "sent": 0,
            "skipped": True,
            "reason": error,
        }

    body = f"⚡ {title}\n\n{message}"

    try:
        client.messages.create(
            body=body[:1500],
            from_=whatsapp_from,
            to="whatsapp:" + phone,
        )

        log_external_notification(
            household_id,
            alert_type,
            "whatsapp",
            title,
            message,
            "sent",
            None,
        )

        return {
            "sent": 1,
            "skipped": False,
        }

    except Exception as exc:
        log_external_notification(
            household_id,
            alert_type,
            "whatsapp",
            title,
            message,
            "failed",
            str(exc),
        )

        print(f"WhatsApp send warning: {exc}")

        return {
            "sent": 0,
            "skipped": False,
            "error": str(exc),
        }

# =====================================================
# AUTOMATED ALERT SCHEDULER
# Requires: pip install apscheduler
# Add this block before if __name__ == "__main__".
# Then call start_alert_scheduler() inside the app.app_context() startup block.
# =====================================================

ALERT_SCHEDULER = None


def load_household_dataframe_by_id(household_id):
    """Load daily meter/appliance readings for a specific household without current_user."""
    meter_rows = (
        MeterReading.query
        .filter_by(household_id=household_id)
        .order_by(MeterReading.reading_date.asc(), MeterReading.created_at.asc())
        .all()
    )

    if not meter_rows:
        empty = pd.DataFrame()
        empty.attrs["actual_readings_count"] = 0
        return empty

    appliance_rows = (
        ApplianceReading.query
        .filter_by(household_id=household_id)
        .order_by(ApplianceReading.reading_date.asc())
        .all()
    )

    appliance_map = {row.reading_date: row for row in appliance_rows}
    records = []

    for meter in meter_rows:
        app_row = appliance_map.get(meter.reading_date)
        records.append({
            "Date": pd.to_datetime(meter.reading_date),
            "Total_Units": float(meter.total_units or 0),
            "Refrigerator": float(app_row.refrigerator or 0) if app_row else 0.0,
            "Lights_Fans": float(app_row.lights_fans or 0) if app_row else 0.0,
            "TV_Monitor": float(app_row.tv_monitor or 0) if app_row else 0.0,
            "AC": float(app_row.ac or 0) if app_row else 0.0,
            "Water_Heater": float(app_row.water_heater or 0) if app_row else 0.0,
            "Washing_Machine": float(app_row.washing_machine or 0) if app_row else 0.0,
            "Motor_Pump": float(app_row.motor_pump or 0) if app_row else 0.0,
        })

    df = pd.DataFrame(records).sort_values("Date")

    agg_map = {"Total_Units": "last"}
    for col in APPLIANCE_COLS:
        agg_map[col] = "last"

    df = df.groupby("Date", as_index=False).agg(agg_map)
    real_count = int(len(df))
    df = df.set_index("Date")
    df.index = pd.to_datetime(df.index)

    full_index = pd.date_range(df.index.min(), df.index.max(), freq="D")
    df = df.reindex(full_index)
    df.index.name = "Date"

    df["Total_Units"] = pd.to_numeric(df["Total_Units"], errors="coerce")
    df["Total_Units"] = df["Total_Units"].interpolate(limit_direction="both").fillna(0)

    for col in APPLIANCE_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["Year"] = df.index.year
    df["Month"] = df.index.month
    df["Month_Name"] = df.index.strftime("%b")
    df["Day"] = df.index.day
    df["DayOfWeek"] = df.index.dayofweek
    df["Day_Name"] = df.index.strftime("%a")
    df["Week"] = df.index.isocalendar().week.astype(int)
    df["IsWeekend"] = (df.index.dayofweek >= 5).astype(int)
    df["Is_Summer"] = df["Month"].isin([3, 4, 5, 6]).astype(int)
    df.attrs["actual_readings_count"] = real_count

    return df


def automated_limit_status_from_sql(limit_row):
    """Calculate limit risk using SQL and simple projection. Works in scheduler context."""
    from sqlalchemy import text

    household_id = limit_row["household_id"]
    limit_units = float(limit_row["limit_units"] or 0)
    start_date = limit_row["period_start_date"]
    end_date = limit_row["period_end_date"]
    today = datetime.utcnow().date()

    used_units = db.session.execute(text("""
        SELECT COALESCE(SUM(total_units), 0)
        FROM meter_readings
        WHERE household_id = :household_id
          AND reading_date BETWEEN :start_date AND LEAST(:today, :end_date)
    """), {
        "household_id": str(household_id),
        "start_date": start_date,
        "today": today,
        "end_date": end_date,
    }).scalar() or 0

    used_units = float(used_units or 0)

    period_days = max(1, (end_date - start_date).days + 1)
    elapsed_days = max(1, min(period_days, (min(today, end_date) - start_date).days + 1))
    remaining_days = max(0, (end_date - today).days)
    avg_daily = used_units / elapsed_days
    projected_units = used_units + (avg_daily * remaining_days)

    # If ML data exists and enough readings are available, use simple recent average guard.
    # This avoids importing current_user-dependent logic in the scheduler.
    recent_avg = db.session.execute(text("""
        SELECT COALESCE(AVG(total_units), 0)
        FROM (
            SELECT total_units
            FROM meter_readings
            WHERE household_id = :household_id
            ORDER BY reading_date DESC
            LIMIT 30
        ) x
    """), {"household_id": str(household_id)}).scalar() or 0

    recent_avg = float(recent_avg or 0)
    if recent_avg > 0 and remaining_days > 0:
        projected_units = max(projected_units, used_units + (recent_avg * remaining_days))

    usage_percent = (used_units / limit_units * 100) if limit_units else 0
    predicted_excess = max(0.0, projected_units - limit_units)

    return {
        "household_id": household_id,
        "limit_units": limit_units,
        "used_units": used_units,
        "projected_units": projected_units,
        "usage_percent": usage_percent,
        "predicted_excess": predicted_excess,
        "remaining_days": remaining_days,
        "threshold_1": int(limit_row["threshold_1_percent"] or 70),
        "threshold_2": int(limit_row["threshold_2_percent"] or 85),
        "threshold_3": int(limit_row["threshold_3_percent"] or 95),
    }


def run_automated_alert_checks():
    """Checks all active households and creates/sends alerts automatically."""
    from sqlalchemy import text

    with app.app_context():
        print("Running automated TNEB alert checks...")

        # 1) Consumption limit alerts for all active limits.
        limits = db.session.execute(text("""
            SELECT *
            FROM consumption_limits
            WHERE is_active = TRUE
        """)).mappings().all()

        for limit_row in limits:
            try:
                status = automated_limit_status_from_sql(limit_row)
                household_id = status["household_id"]
                today = datetime.utcnow().date()

                if status["predicted_excess"] > 0:
                    message = (
                        f"You may exceed your {status['limit_units']:.0f} unit limit by about "
                        f"{status['predicted_excess']:.1f} units. Reduce around "
                        f"{(status['predicted_excess'] / max(1, status['remaining_days'])):.2f} units/day."
                    )
                    insert_limit_alert_once(
                        household_id,
                        "limit_predicted_exceed",
                        "critical",
                        "Consumption limit likely to exceed",
                        message,
                        today,
                    )
                elif status["usage_percent"] >= status["threshold_3"]:
                    insert_limit_alert_once(
                        household_id,
                        "limit_95_percent",
                        "critical",
                        "95% consumption limit reached",
                        f"You have used {status['usage_percent']:.1f}% of your configured limit.",
                        today,
                    )
                elif status["usage_percent"] >= status["threshold_2"]:
                    insert_limit_alert_once(
                        household_id,
                        "limit_85_percent",
                        "warning",
                        "85% consumption limit reached",
                        f"You have used {status['usage_percent']:.1f}% of your configured limit.",
                        today,
                    )
                elif status["usage_percent"] >= status["threshold_1"]:
                    insert_limit_alert_once(
                        household_id,
                        "limit_70_percent",
                        "info",
                        "70% consumption limit reached",
                        f"You have used {status['usage_percent']:.1f}% of your configured limit.",
                        today,
                    )
            except Exception as exc:
                db.session.rollback()
                print(f"Automated limit alert warning: {exc}")

        # 2) Abnormal consumption alerts for all households with readings.
        household_rows = db.session.execute(text("""
            SELECT DISTINCT household_id
            FROM meter_readings
        """)).mappings().all()

        for row in household_rows:
            try:
                household = db.session.get(Household, row["household_id"])
                if not household:
                    continue
                df = load_household_dataframe_by_id(household.id)
                save_abnormal_consumption_alert_if_needed(household, df)
            except Exception as exc:
                db.session.rollback()
                print(f"Automated abnormal alert warning: {exc}")

        print("Automated TNEB alert checks completed.")


@app.route("/api/admin/run-alert-checks", methods=["POST"])
def api_run_alert_checks_now():
    """Manual test endpoint for automated alert engine."""
    run_automated_alert_checks()
    return jsonify({"success": True, "message": "Automated alert checks completed."})
@app.route("/api/admin/run-alert-checks-now", methods=["GET"])
def api_run_alert_checks_now_get():
    run_automated_alert_checks()
    return jsonify({
        "success": True,
        "message": "Automated alert checks completed."
    })


def start_alert_scheduler():
    """Starts background scheduler. For local testing interval is configurable."""
    global ALERT_SCHEDULER

    if os.environ.get("ENABLE_ALERT_SCHEDULER", "true").lower() not in {"1", "true", "yes", "on"}:
        print("Alert scheduler disabled by ENABLE_ALERT_SCHEDULER.")
        return

    if ALERT_SCHEDULER is not None:
        return

    try:
        from apscheduler.schedulers.background import BackgroundScheduler
    except Exception as exc:
        print(f"Alert scheduler not started. Install apscheduler. Error: {exc}")
        return

    interval_minutes = int(os.environ.get("ALERT_CHECK_INTERVAL_MINUTES", "60") or 60)

    scheduler = BackgroundScheduler(daemon=True)
    scheduler.add_job(
        run_automated_alert_checks,
        "interval",
        minutes=interval_minutes,
        id="tneb_automated_alert_checks",
        replace_existing=True,
        max_instances=1,
    )
    scheduler.start()
    ALERT_SCHEDULER = scheduler
    print(f"Alert scheduler started. Interval: {interval_minutes} minutes.")

# =====================================================
# NOTIFICATION CENTER PAGE
# =====================================================

@app.route("/alerts")
def alerts_page():
    return render_template("alerts.html")

# =====================================================
# BILL COMPARISON FEATURE
# =====================================================

def get_cycle_usage_for_household(household_id, start_date, end_date):
    from sqlalchemy import text

    units = db.session.execute(text("""
        SELECT COALESCE(SUM(total_units), 0)
        FROM meter_readings
        WHERE household_id = :household_id
          AND reading_date BETWEEN :start_date AND :end_date
    """), {
        "household_id": str(household_id),
        "start_date": start_date,
        "end_date": end_date,
    }).scalar() or 0

    return float(units or 0)


def save_high_bill_prediction_alert_once(household_id, alert_date, title, message):
    from sqlalchemy import text

    existing = db.session.execute(text("""
        SELECT id
        FROM alerts
        WHERE household_id = :household_id
          AND alert_type = 'high_bill_prediction'
          AND alert_date = :alert_date
        LIMIT 1
    """), {
        "household_id": str(household_id),
        "alert_date": alert_date,
    }).first()

    if existing:
        return

    db.session.execute(text("""
        INSERT INTO alerts (
            id,
            household_id,
            alert_date,
            alert_type,
            severity,
            title,
            message,
            reading_date,
            is_read,
            created_at
        ) VALUES (
            gen_random_uuid(),
            :household_id,
            :alert_date,
            'high_bill_prediction',
            'warning',
            :title,
            :message,
            :reading_date,
            FALSE,
            NOW()
        )
    """), {
        "household_id": str(household_id),
        "alert_date": alert_date,
        "title": title,
        "message": message,
        "reading_date": alert_date,
    })

    try:
        send_push_for_alert(
            household_id,
            "high_bill_prediction",
            "warning",
            title,
            message,
        )
    except Exception as exc:
        print(f"High bill push warning: {exc}")

    db.session.commit()


@app.route("/api/bill-comparison")
def api_bill_comparison():
    _, model, feature_cols, _ = initialize()

    household = get_current_household()
    df = load_current_household_dataframe()

    if df is None or df.empty:
        return jsonify({
            "hasData": False,
            "message": "No readings available yet."
        })

    latest_date = df.index.max().date()

    cycle_start_param = request.args.get("cycle_start")

    if cycle_start_param:
        current_start = pd.to_datetime(cycle_start_param).date()
    else:
        # Prefer household billing cycle start date if available and sensible.
        billing_start = getattr(household, "billing_cycle_start_date", None)

        if billing_start:
            current_start = billing_start

            while current_start + pd.Timedelta(days=59) < latest_date:
                current_start = (pd.to_datetime(current_start) + pd.Timedelta(days=60)).date()
        else:
            current_start = (pd.to_datetime(latest_date) - pd.Timedelta(days=59)).date()

    current_end = (pd.to_datetime(current_start) + pd.Timedelta(days=59)).date()
    previous_start = (pd.to_datetime(current_start) - pd.Timedelta(days=60)).date()
    previous_end = (pd.to_datetime(current_start) - pd.Timedelta(days=1)).date()

    previous_units = get_cycle_usage_for_household(
        household.id,
        previous_start,
        previous_end,
    )

    previous_bill = tneb_domestic_bill(previous_units)

    if latest_date >= current_start:
        current_used_units = get_cycle_usage_for_household(
            household.id,
            current_start,
            min(latest_date, current_end),
        )
    else:
        current_used_units = 0.0

    current_cycle_day = max(0, min(60, (latest_date - current_start).days + 1))
    remaining_days = max(0, 60 - current_cycle_day)

    current_forecast_units = 0.0

    if actual_reading_count(df) >= MIN_READINGS_FOR_FORECAST and remaining_days > 0:
        forecast_df = forecast_from_series(model, df, feature_cols, remaining_days)
        current_forecast_units = float(forecast_df["units"].sum()) if len(forecast_df) else 0.0
    elif current_cycle_day > 0:
        avg_daily = current_used_units / max(1, current_cycle_day)
        current_forecast_units = avg_daily * remaining_days

    current_projected_units = current_used_units + current_forecast_units
    current_projected_bill = tneb_domestic_bill(current_projected_units)

    diff_units = current_projected_units - previous_units
    diff_bill = current_projected_bill - previous_bill
    diff_percent = (diff_bill / previous_bill * 100) if previous_bill else 0

    risk = "good"
    title = "Bill under control"
    message = "Your predicted bill is close to or lower than your previous cycle."

    if previous_bill > 0 and diff_percent >= 50:
        risk = "danger"
        title = "High bill increase expected"
        message = f"Your predicted bill is {diff_percent:.1f}% higher than your previous cycle."
    elif previous_bill > 0 and diff_percent >= 25:
        risk = "warning"
        title = "Bill increase warning"
        message = f"Your predicted bill is {diff_percent:.1f}% higher than your previous cycle."
    elif previous_bill > 0 and diff_percent >= 10:
        risk = "caution"
        title = "Slight bill increase expected"
        message = f"Your predicted bill is {diff_percent:.1f}% higher than your previous cycle."

    if risk in {"warning", "danger"}:
        try:
            save_high_bill_prediction_alert_once(
                household.id,
                latest_date,
                title,
                message,
            )
        except Exception as exc:
            db.session.rollback()
            print(f"High bill alert warning: {exc}")

    return jsonify({
        "hasData": True,
        "previousCycle": {
            "start": previous_start.strftime("%Y-%m-%d"),
            "end": previous_end.strftime("%Y-%m-%d"),
            "units": round(previous_units, 2),
            "bill": previous_bill,
        },
        "currentCycle": {
            "start": current_start.strftime("%Y-%m-%d"),
            "end": current_end.strftime("%Y-%m-%d"),
            "cycleDay": int(current_cycle_day),
            "remainingDays": int(remaining_days),
            "usedUnits": round(current_used_units, 2),
            "forecastUnits": round(current_forecast_units, 2),
            "projectedUnits": round(current_projected_units, 2),
            "projectedBill": current_projected_bill,
        },
        "difference": {
            "units": round(diff_units, 2),
            "bill": round(diff_bill, 2),
            "percent": round(diff_percent, 2),
        },
        "risk": risk,
        "title": title,
        "message": message,
    })

# =====================================================
# RUN
# =====================================================

if __name__ == "__main__":
    print("Starting TNEB Flask app...")
    print("Open this URL in your browser: http://127.0.0.1:5000")

    with app.app_context():
        try:
            db.create_all()
        except Exception as exc:
            print(f"Database create_all warning: {exc}")
            initialize()
            start_alert_scheduler()

    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)