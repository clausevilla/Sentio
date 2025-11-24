# Create your tests here.
from django.contrib.auth.models import User
from django.contrib.messages import get_messages
from django.test import TestCase
from django.urls import reverse

from apps.accounts.models import UserProfile


class RegistrationViewTests(TestCase):
    # Test setup for registration view tests
    def setUp(self):
        self.register_url = reverse('accounts:register')

    # Test registration view GET request
    def test_register_view_get(self):
        response = self.client.get(self.register_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'accounts/register.html')

    # Test successful registration
    def test_register_view_post_success(self):
        form_data = {
            'username': 'testuser',
            'email': 'testuser@example.com',
            'password1': 'strongpassword123',
            'password2': 'strongpassword123',
        }
        response = self.client.post(self.register_url, data=form_data)
        self.assertEqual(
            response.status_code, 302
        )  # Redirect after successful registration
        self.assertTrue(User.objects.filter(username='testuser').exists())
        user = User.objects.get(username='testuser')
        self.assertTrue(user.check_password('strongpassword123'))
        messages = list(get_messages(response.wsgi_request))
        self.assertEqual(len(messages), 1)
        self.assertEqual(
            str(messages[0]), 'Registration successful. You can now log in.'
        )

    # Test registration with password mismatch
    def test_register_view_post_password_mismatch(self):
        form_data = {
            'username': 'testuser',
            'email': 'testuser@example.com',
            'password1': 'strongpassword123',
            'password2': 'differentpassword',
        }
        response = self.client.post(self.register_url, data=form_data)
        self.assertEqual(response.status_code, 200)  # Form re-rendered with errors
        self.assertFalse(User.objects.filter(username='testuser').exists())
        self.assertFormError(
            response, 'form', 'password2', "The two password fields didn't match."
        )

    # Test registration with existing username
    def test_register_view_post_existing_username(self):
        User.objects.create_user(username='existinguser', password='somepassword')
        form_data = {
            'username': 'existinguser',
            'email': 'existinguser@example.com',
            'password1': 'strongpassword123',
            'password2': 'strongpassword123',
        }
        response = self.client.post(self.register_url, data=form_data)
        self.assertEqual(response.status_code, 200)  # Form re-rendered with errors
        self.assertFormError(
            response, 'form', 'username', 'A user with that username already exists.'
        )


class ProfileViewTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser', password='strongpassword123'
        )
        self.profile_url = reverse('accounts:profile')

    def test_profile_view_requires_login(self):
        response = self.client.get(self.profile_url)
        self.assertEqual(response.status_code, 302)  # Redirect to login

    def test_profile_view_logged_in(self):
        self.client.login(username='testuser', password='strongpassword123')
        response = self.client.get(self.profile_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'accounts/profile.html')
        self.assertContains(response, 'testuser')

    def test_profile_view_displays_user_info(self):
        UserProfile.objects.create(user=self.user, bio='This is a test bio.')
        self.client.login(username='testuser', password='strongpassword123')
        response = self.client.get(self.profile_url)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'This is a test bio.')


class LoginViewTests(TestCase):
    # Tests for login view to ensure authentication works correctly
    def setUp(self):
        self.login_url = reverse('accounts:login')
        self.user = User.objects.create_user(
            username='testuser', password='strongpassword123'
        )

    def test_login_view_get(self):
        response = self.client.get(self.login_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'accounts/login.html')

    # Test successful login attempts
    def test_login_view_post_success(self):
        form_data = {
            'username': 'testuser',
            'password': 'strongpassword123',
        }
        response = self.client.post(self.login_url, data=form_data)
        self.assertEqual(response.status_code, 302)  # Redirect after successful login
        self.assertTrue(response.wsgi_request.user.is_authenticated)

    # Test login with invalid credentials
    def test_login_view_post_invalid_credentials(self):
        form_data = {
            'username': 'testuser',
            'password': 'wrongpassword',
        }
        response = self.client.post(self.login_url, data=form_data)
        self.assertEqual(response.status_code, 200)  # Form re-rendered with errors
        self.assertFalse(response.wsgi_request.user.is_authenticated)
        self.assertContains(response, 'Please enter a correct username and password.')


# Tests for logout view to ensure users can log out properly
class LogoutViewTests(TestCase):
    def setUp(self):
        self.logout_url = reverse('accounts:logout')
        self.user = User.objects.create_user(
            username='testuser', password='strongpassword123'
        )

    # Test logout view requires login
    def test_logout_view_requires_login(self):
        response = self.client.get(self.logout_url)
        self.assertEqual(response.status_code, 302)  # Redirect to login

    # Test successful logout
    def test_logout_view_logged_in(self):
        self.client.login(username='testuser', password='strongpassword123')
        response = self.client.get(self.logout_url)
        self.assertEqual(response.status_code, 302)  # Redirect after logout
        self.assertFalse(response.wsgi_request.user.is_authenticated)
