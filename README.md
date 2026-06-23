# Energy Lens

Energy Lens is a Tamil Nadu-focused electricity usage monitoring, bill forecasting, and smart alert web application.

It helps households track appliance-wise electricity usage, estimate future bills, detect abnormal consumption, set consumption limits, and receive alerts through in-app notifications, browser push notifications, and WhatsApp/SMS integrations.

> This is an independent educational/demo project. It is not affiliated with TNEB, TANGEDCO, or any government electricity board.

---

## Features

- User signup/login
- PostgreSQL database storage
- Household profile management
- Live daily appliance-wise energy entry
- Usage analytics dashboard
- Monthly and seasonal trend analysis
- Forecasting after sufficient readings
- Tamil Nadu domestic bill estimation logic
- Consumption limit tracking
- Smart alerts for predicted overuse
- Abnormal consumption detection
- Notification center with read/unread management
- Browser push notifications
- WhatsApp alert support via Twilio
- SMS alert support via Fast2SMS
- Automated background alert scheduler
- Current vs previous bill comparison

---

## Tech Stack

- Python
- Flask
- Flask-Login
- Flask-SQLAlchemy
- PostgreSQL
- Pandas
- NumPy
- scikit-learn
- ExtraTreesRegressor
- Chart.js
- HTML/CSS/JavaScript
- Browser Push API
- Twilio WhatsApp API
- Fast2SMS API
- APScheduler

---

## Project Structure

```text
tneb_product_app/
├── app.py
├── database.py
├── models.py
├── requirements.txt
├── .env.example
├── templates/
│   ├── index.html
│   ├── login.html
│   ├── signup.html
│   ├── profile.html
│   └── alerts.html
├── static/
│   ├── css/
│   ├── js/
│   └── sw.js
└── README.md