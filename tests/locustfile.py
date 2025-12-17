"""
Locust Load Test for Sentio ML Platform
Tests both anonymous and registered users.

Setup:
    pip install locust

Usage:
    cd tests



    # Headless (command line only)
    locust -f locustfile.py --host=http://34.51.186.204 \
        --users 50 --spawn-rate 10 --run-time 2m \
        --headless --html=report.html

    # Alternative with web UI (interactive)
    locust -f locustfile.py --host=http://34.51.186.204

    # Local testing
    locust -f locustfile.py --host=http://localhost:8000

    Options:
    --users         Number of concurrent users (default: 50)
    --spawn-rate    Users spawned per second (default: 10)
    --run-time      Test duration, e.g., 30s, 2m, 1h
    --html          Generate HTML report
"""

import random
import re

import string

from locust import HttpUser, task, between

# Sample texts for predictions
TEXTS = [
    'I had a normal day today. Went to work and came back home.',
    "I have so much work and the deadline is tomorrow. Can't handle this.",
    "I don't feel like doing anything. Everything seems pointless.",
    "I can't stop worrying. My heart races for no reason.",
    'Just finished my tasks. Feeling okay and relaxed.',
    'The pressure at work is unbearable. Completely overwhelmed.',
    'I feel empty inside. Nothing brings me joy anymore.',
    'Watched a movie with friends. It was fun!',
    "I'm tired all the time but can't sleep properly.",
    'Today was great! Got promoted and celebrated with family.',
]


def get_csrf(response):
    """Get CSRF token from response"""
    token = response.cookies.get('csrftoken', '')
    if not token:
        match = re.search(
            r'csrfmiddlewaretoken.*?value=["\']([^"\']+)["\']', response.text
        )
        if match:
            token = match.group(1)
    return token


# ==============================================
# ANONYMOUS USER - Does NOT login
# ==============================================
class AnonymousUser(HttpUser):
    """
    Anonymous users who make predictions WITHOUT logging in.
    Weight: 2 (40% of traffic)
    """

    weight = 2
    wait_time = between(2, 5)

    csrf = None

    def on_start(self):
        """Just get CSRF token, don't login"""
        r = self.client.get('/')
        self.csrf = get_csrf(r)

    @task(1)
    def view_home(self):
        """Visit home page"""
        r = self.client.get('/')
        self.csrf = get_csrf(r)

    @task(5)
    def make_prediction_anonymous(self):
        """Make prediction as anonymous user"""
        # GET input page
        r = self.client.get('/predictions/input')
        self.csrf = get_csrf(r)

        if r.status_code != 200:
            return

        # POST prediction (no login required)
        r = self.client.post(
            '/predictions/input',
            data={
                'csrfmiddlewaretoken': self.csrf,
                'text': random.choice(TEXTS),
            },
            headers={'X-CSRFToken': self.csrf},
            allow_redirects=True,
        )
        self.csrf = get_csrf(r)

    @task(1)
    def health_check(self):
        """Check health endpoints"""
        self.client.get('/health/live/')
        self.client.get('/health/ready/')


# ==============================================
# REGISTERED USER - Registers, logs in, uses app
# ==============================================
class RegisteredUser(HttpUser):
    """
    Users who register, login, make predictions, view history.
    Weight: 3 (60% of traffic)
    """

    weight = 3
    wait_time = between(2, 5)

    csrf = None
    username = None
    password = 'TestPass123!'

    def on_start(self):
        """Register and login"""
        r = self.client.get('/')
        self.csrf = get_csrf(r)

        # Generate random username
        self.username = 'test_' + ''.join(random.choices(string.ascii_lowercase, k=6))

        # Register
        self.do_register()

        # Login
        self.do_login()

    def do_register(self):
        """Register new account"""
        r = self.client.get('/accounts/register/')
        self.csrf = get_csrf(r)

        r = self.client.post(
            '/accounts/register/',
            data={
                'csrfmiddlewaretoken': self.csrf,
                'username': self.username,
                'email': f'{self.username}@test.com',
                'password1': self.password,
                'password2': self.password,
                'consent': 'on',
            },
            headers={'X-CSRFToken': self.csrf},
            allow_redirects=True,
        )
        self.csrf = get_csrf(r)

    def do_login(self):
        """Login to account"""
        r = self.client.get('/accounts/login/')
        self.csrf = get_csrf(r)

        r = self.client.post(
            '/accounts/login/',
            data={
                'csrfmiddlewaretoken': self.csrf,
                'username': self.username,
                'password': self.password,
            },
            headers={'X-CSRFToken': self.csrf},
            allow_redirects=True,
        )
        self.csrf = get_csrf(r)

    @task(1)
    def view_home(self):
        """Visit home page"""
        r = self.client.get('/')
        self.csrf = get_csrf(r)

    @task(5)
    def make_prediction(self):
        """Make prediction as logged-in user"""
        r = self.client.get('/predictions/input')
        self.csrf = get_csrf(r)

        if r.status_code != 200:
            return

        r = self.client.post(
            '/predictions/input',
            data={
                'csrfmiddlewaretoken': self.csrf,
                'text': random.choice(TEXTS),
            },
            headers={'X-CSRFToken': self.csrf},
            allow_redirects=True,
        )
        self.csrf = get_csrf(r)

    @task(3)
    def view_history(self):
        """View prediction history (profile page)"""
        r = self.client.get('/accounts/profile/')
        self.csrf = get_csrf(r)

    @task(1)
    def health_check(self):
        """Check health endpoints"""
        self.client.get('/health/live/')
        self.client.get('/health/ready/')
