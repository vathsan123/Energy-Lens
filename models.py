from datetime import datetime
from uuid import uuid4

from flask_login import UserMixin
from sqlalchemy.dialects.postgresql import UUID, JSONB
from werkzeug.security import generate_password_hash, check_password_hash

from database import db


class User(UserMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    full_name = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False, index=True)
    phone = db.Column(db.String(30))

    password_hash = db.Column(db.Text, nullable=False)

    is_active = db.Column(db.Boolean, default=True)

    created_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow)
    updated_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow)
    last_login_at = db.Column(db.DateTime(timezone=True))

    households = db.relationship(
        "Household",
        backref="user",
        cascade="all, delete-orphan"
    )

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Household(db.Model):
    __tablename__ = "households"

    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    user_id = db.Column(
        UUID(as_uuid=True),
        db.ForeignKey("users.id"),
        nullable=False
    )

    household_name = db.Column(db.String(150), nullable=False, default="My Home")

    address = db.Column(db.Text)
    city = db.Column(db.String(100))
    district = db.Column(db.String(100))
    state = db.Column(db.String(100), default="Tamil Nadu")
    pincode = db.Column(db.String(20))

    eb_service_number = db.Column(db.String(100))
    tariff_plan = db.Column(db.String(100), default="TN_DOMESTIC_LT_IA")

    billing_cycle_start_date = db.Column(db.Date)
    timezone = db.Column(db.String(50), default="Asia/Kolkata")

    created_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow)
    updated_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow)

    meter_readings = db.relationship(
        "MeterReading",
        backref="household",
        cascade="all, delete-orphan"
    )

    appliance_readings = db.relationship(
        "ApplianceReading",
        backref="household",
        cascade="all, delete-orphan"
    )


class MeterReading(db.Model):
    __tablename__ = "meter_readings"

    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    household_id = db.Column(
        UUID(as_uuid=True),
        db.ForeignKey("households.id"),
        nullable=False
    )

    reading_date = db.Column(db.Date, nullable=False)
    total_units = db.Column(db.Numeric(12, 3), nullable=False)

    source = db.Column(db.String(50), default="manual")
    notes = db.Column(db.Text)

    created_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow)
    updated_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint("household_id", "reading_date"),
    )


class ApplianceReading(db.Model):
    __tablename__ = "appliance_readings"

    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    household_id = db.Column(
        UUID(as_uuid=True),
        db.ForeignKey("households.id"),
        nullable=False
    )

    reading_date = db.Column(db.Date, nullable=False)

    refrigerator = db.Column(db.Numeric(12, 3), default=0)
    lights_fans = db.Column(db.Numeric(12, 3), default=0)
    tv_monitor = db.Column(db.Numeric(12, 3), default=0)
    ac = db.Column(db.Numeric(12, 3), default=0)
    water_heater = db.Column(db.Numeric(12, 3), default=0)
    washing_machine = db.Column(db.Numeric(12, 3), default=0)
    motor_pump = db.Column(db.Numeric(12, 3), default=0)

    source = db.Column(db.String(50), default="manual")

    created_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow)
    updated_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint("household_id", "reading_date"),
    )

    @property
    def total_appliance_units(self):
        return float(
            (self.refrigerator or 0) +
            (self.lights_fans or 0) +
            (self.tv_monitor or 0) +
            (self.ac or 0) +
            (self.water_heater or 0) +
            (self.washing_machine or 0) +
            (self.motor_pump or 0)
        )


class LoginEvent(db.Model):
    __tablename__ = "login_events"

    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    user_id = db.Column(UUID(as_uuid=True), db.ForeignKey("users.id"))
    email = db.Column(db.String(255))

    login_status = db.Column(db.String(50), nullable=False)

    ip_address = db.Column(db.String(100))
    user_agent = db.Column(db.Text)

    created_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow)


class ForecastRun(db.Model):
    __tablename__ = "forecast_runs"

    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    household_id = db.Column(
        UUID(as_uuid=True),
        db.ForeignKey("households.id"),
        nullable=False
    )

    forecast_start_date = db.Column(db.Date, nullable=False)
    forecast_horizon_days = db.Column(db.Integer, default=60)

    model_name = db.Column(db.String(150))
    model_version = db.Column(db.String(100))

    input_history_start = db.Column(db.Date)
    input_history_end = db.Column(db.Date)

    forecast_total_units = db.Column(db.Numeric(12, 3))
    estimated_bill = db.Column(db.Numeric(12, 2))

    lower_units = db.Column(db.Numeric(12, 3))
    expected_units = db.Column(db.Numeric(12, 3))
    upper_units = db.Column(db.Numeric(12, 3))

    lower_bill = db.Column(db.Numeric(12, 2))
    expected_bill = db.Column(db.Numeric(12, 2))
    upper_bill = db.Column(db.Numeric(12, 2))

    forecast_metadata = db.Column(JSONB, default={})

    created_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow)


class ForecastDailyValue(db.Model):
    __tablename__ = "forecast_daily_values"

    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    forecast_run_id = db.Column(
        UUID(as_uuid=True),
        db.ForeignKey("forecast_runs.id"),
        nullable=False
    )

    forecast_date = db.Column(db.Date, nullable=False)
    predicted_units = db.Column(db.Numeric(12, 3), nullable=False)

    created_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow)


class Alert(db.Model):
    __tablename__ = "alerts"

    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    household_id = db.Column(
        UUID(as_uuid=True),
        db.ForeignKey("households.id"),
        nullable=False
    )

    alert_date = db.Column(db.Date, nullable=False)

    alert_type = db.Column(db.String(100), nullable=False)
    severity = db.Column(db.String(50), nullable=False)

    title = db.Column(db.String(200), nullable=False)
    message = db.Column(db.Text, nullable=False)

    is_read = db.Column(db.Boolean, default=False)

    created_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow)


class Recommendation(db.Model):
    __tablename__ = "recommendations"

    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    household_id = db.Column(
        UUID(as_uuid=True),
        db.ForeignKey("households.id"),
        nullable=False
    )

    recommendation_date = db.Column(db.Date, nullable=False)

    appliance_name = db.Column(db.String(100))
    priority = db.Column(db.String(50))

    title = db.Column(db.String(200), nullable=False)
    message = db.Column(db.Text, nullable=False)

    estimated_units_saved = db.Column(db.Numeric(12, 3))
    estimated_rupees_saved = db.Column(db.Numeric(12, 2))

    is_dismissed = db.Column(db.Boolean, default=False)

    created_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow)


class Report(db.Model):
    __tablename__ = "reports"

    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid4)

    household_id = db.Column(
        UUID(as_uuid=True),
        db.ForeignKey("households.id"),
        nullable=False
    )

    report_type = db.Column(db.String(50), nullable=False)
    start_date = db.Column(db.Date)
    end_date = db.Column(db.Date)

    total_units = db.Column(db.Numeric(12, 3))
    estimated_bill = db.Column(db.Numeric(12, 2))

    report_json = db.Column(JSONB, default={})

    created_at = db.Column(db.DateTime(timezone=True), default=datetime.utcnow)