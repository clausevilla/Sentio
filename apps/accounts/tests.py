# Author: Lian Shi
# Disclaimer: LLM has been used to help generate tests for change password and delete account API endpoints.

import json

from django.contrib.auth.models import User
from django.contrib.messages import get_messages
from django.test import TestCase
from django.urls import reverse


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
        self.assertTrue(len(messages) >= 1)
        self.assertIn('Your account has been created successfully', str(messages[0]))

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
        form = response.context['form']
        self.assertFormError(form, 'password2', 'Passwords do not match.')

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
        form = response.context['form']
        self.assertFormError(
            form,
            'username',
            'This username is already taken. Please choose another.',
        )

    # Test registration with exisiting email
    def test_register_view_post_existing_email(self):
        User.objects.create_user(
            username='user1', email='existingemail@example.com', password='somepassword'
        )
        form_data = {
            'username': 'newuser',
            'email': 'existingemail@example.com',
            'password1': 'strongpassword123',
            'password2': 'strongpassword123',
        }
        response = self.client.post(self.register_url, data=form_data)
        self.assertEqual(response.status_code, 200)  # Form re-rendered with errors
        form = response.context['form']
        self.assertFormError(
            form,
            'email',
            'An account with this email already exists.',
        )

    # Test registration with short username
    def test_register_view_post_short_username(self):
        form_data = {
            'username': 'ab',
            'email': 'shortusername@example.com',
            'password1': 'strongpassword123',
            'password2': 'strongpassword123',
        }
        response = self.client.post(self.register_url, data=form_data)
        self.assertEqual(response.status_code, 200)  # Form re-rendered with errors
        form = response.context['form']
        self.assertFormError(
            form,
            'username',
            'Username must be at least 3 characters long.',
        )

    # Test registration with empty username
    def test_register_view_post_empty_username(self):
        form_data = {
            'username': '',
            'email': 'emptyusername@example.com',
            'password1': 'strongpassword123',
            'password2': 'strongpassword123',
        }
        response = self.client.post(self.register_url, data=form_data)
        self.assertEqual(response.status_code, 200)  # Form re-rendered with errors
        form = response.context['form']
        self.assertFormError(
            form,
            'username',
            'This field is required.',
        )


class ProfileViewTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser', password='strongpassword123'
        )
        self.profile_url = reverse('accounts:profile')

    # Test profile view requires login
    def test_profile_view_requires_login(self):
        response = self.client.get(self.profile_url)
        self.assertEqual(response.status_code, 302)

    # Test profile view for logged-in user
    def test_profile_view_logged_in(self):
        self.client.login(username='testuser', password='strongpassword123')
        response = self.client.get(self.profile_url)
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'accounts/profile.html')
        self.assertContains(response, 'testuser')


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

    # Test login with invalid password
    def test_login_view_post_invalid_password(self):
        form_data = {
            'username': 'testuser',
            'password': 'wrongpassword',
        }
        response = self.client.post(self.login_url, data=form_data)
        self.assertEqual(response.status_code, 200)  # Form re-rendered with errors
        self.assertFalse(response.wsgi_request.user.is_authenticated)
        self.assertContains(response, 'Invalid username or password. Please try again.')

    # Test login with non-existing username
    def test_login_view_post_non_existing_username(self):
        form_data = {
            'username': 'nonexistinguser',
            'password': 'somepassword',
        }
        response = self.client.post(self.login_url, data=form_data)
        self.assertEqual(response.status_code, 200)  # Form re-rendered with errors
        self.assertFalse(response.wsgi_request.user.is_authenticated)
        self.assertContains(response, 'Invalid username or password. Please try again.')


# Tests for logout view to ensure users can log out properly
class LogoutViewTests(TestCase):
    def setUp(self):
        self.logout_url = reverse('accounts:logout')
        self.user = User.objects.create_user(
            username='testuser', password='strongpassword123'
        )

    # Test logout view redirects to login view when not logged in
    def test_logout_view_requires_login(self):
        response = self.client.get(self.logout_url)
        self.assertEqual(response.status_code, 302)

    # Test successful logout for authenticated user
    def test_logout_view_logged_in(self):
        self.client.login(username='testuser', password='strongpassword123')
        response = self.client.get(self.logout_url)
        self.assertEqual(response.status_code, 302)  # Redirect after logout
        self.assertFalse(response.wsgi_request.user.is_authenticated)


class ProfileAPITests(TestCase):
    # Test suite for profile-related API endpoints

    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            email='testuser@example.com',
            password='strongpassword123',
        )
        self.change_password_url = reverse('accounts:change_password_api')
        self.delete_data_url = reverse('accounts:delete_all_data_api')
        self.delete_account_url = reverse('accounts:delete_account_api')

    # Change Password API Tests

    def test_change_password_success(self):
        # Test successful password change with valid credentials
        self.client.login(username='testuser', password='strongpassword123')
        data = {
            'currentPassword': 'strongpassword123',
            'newPassword': 'NewStrongPass456',
        }
        response = self.client.post(
            self.change_password_url,
            data=json.dumps(data),
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.content)
        self.assertTrue(response_data['success'])

        # Verify password was actually changed
        self.user.refresh_from_db()
        self.assertTrue(self.user.check_password('NewStrongPass456'))

        # Verify user is still logged in after password change
        self.assertTrue(response.wsgi_request.user.is_authenticated)

    def test_change_password_incorrect_current_password(self):
        # Test that password change fails with incorrect current password
        self.client.login(username='testuser', password='strongpassword123')
        data = {
            'currentPassword': 'wrongpassword',
            'newPassword': 'NewStrongPass456',
        }
        response = self.client.post(
            self.change_password_url,
            data=json.dumps(data),
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 400)
        response_data = json.loads(response.content)
        self.assertFalse(response_data['success'])
        self.assertEqual(response_data['error'], 'incorrect_password')

    def test_change_password_same_as_current(self):
        # Test that password change fails when new password matches current password
        self.client.login(username='testuser', password='strongpassword123')
        data = {
            'currentPassword': 'strongpassword123',
            'newPassword': 'strongpassword123',
        }
        response = self.client.post(
            self.change_password_url,
            data=json.dumps(data),
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 400)
        response_data = json.loads(response.content)
        self.assertFalse(response_data['success'])
        self.assertEqual(response_data['error'], 'same_password')

    def test_change_password_too_short(self):
        # Test that password change fails when new password is too short
        self.client.login(username='testuser', password='strongpassword123')
        data = {
            'currentPassword': 'strongpassword123',
            'newPassword': 'short',
        }
        response = self.client.post(
            self.change_password_url,
            data=json.dumps(data),
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 400)
        response_data = json.loads(response.content)
        self.assertFalse(response_data['success'])
        self.assertEqual(response_data['error'], 'weak_password')

    def test_change_password_missing_fields(self):
        # Test that password change fails when required fields are missing
        self.client.login(username='testuser', password='strongpassword123')
        data = {'currentPassword': 'strongpassword123'}
        response = self.client.post(
            self.change_password_url,
            data=json.dumps(data),
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 400)
        response_data = json.loads(response.content)
        self.assertFalse(response_data['success'])
        self.assertEqual(response_data['error'], 'missing_fields')

    def test_change_password_requires_authentication(self):
        # Test that password change endpoint requires authentication
        data = {
            'currentPassword': 'strongpassword123',
            'newPassword': 'NewStrongPass456',
        }
        response = self.client.post(
            self.change_password_url,
            data=json.dumps(data),
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 302)

    # Delete All Data API Tests

    def test_delete_all_data_success(self):
        # Test successful deletion of all user data with correct password
        self.client.login(username='testuser', password='strongpassword123')
        data = {'password': 'strongpassword123'}
        response = self.client.post(
            self.delete_data_url,
            data=json.dumps(data),
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.content)
        self.assertTrue(response_data['success'])

        # Verify user account still exists
        self.assertTrue(User.objects.filter(username='testuser').exists())

        # Verify user is still logged in
        self.assertTrue(response.wsgi_request.user.is_authenticated)

    def test_delete_all_data_incorrect_password(self):
        # Test that data deletion fails with incorrect password
        self.client.login(username='testuser', password='strongpassword123')
        data = {'password': 'wrongpassword'}
        response = self.client.post(
            self.delete_data_url,
            data=json.dumps(data),
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 400)
        response_data = json.loads(response.content)
        self.assertFalse(response_data['success'])
        self.assertEqual(response_data['error'], 'incorrect_password')

    def test_delete_all_data_missing_password(self):
        # Test that data deletion fails when password is not provided
        self.client.login(username='testuser', password='strongpassword123')
        data = {}
        response = self.client.post(
            self.delete_data_url,
            data=json.dumps(data),
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 400)
        response_data = json.loads(response.content)
        self.assertFalse(response_data['success'])
        self.assertEqual(response_data['error'], 'missing_password')

    def test_delete_all_data_requires_authentication(self):
        # Test that data deletion endpoint requires authentication
        data = {'password': 'strongpassword123'}
        response = self.client.post(
            self.delete_data_url,
            data=json.dumps(data),
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 302)

    # Delete Account API Tests

    def test_delete_account_success(self):
        # Test successful account deletion with correct password
        self.client.login(username='testuser', password='strongpassword123')
        data = {'password': 'strongpassword123'}
        response = self.client.post(
            self.delete_account_url,
            data=json.dumps(data),
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 200)
        response_data = json.loads(response.content)
        self.assertTrue(response_data['success'])

        # Verify user account no longer exists
        self.assertFalse(User.objects.filter(username='testuser').exists())

        # Verify user is logged out
        self.assertFalse(response.wsgi_request.user.is_authenticated)

    def test_delete_account_incorrect_password(self):
        # Test that account deletion fails with incorrect password
        self.client.login(username='testuser', password='strongpassword123')
        data = {'password': 'wrongpassword'}
        response = self.client.post(
            self.delete_account_url,
            data=json.dumps(data),
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 400)
        response_data = json.loads(response.content)
        self.assertFalse(response_data['success'])
        self.assertEqual(response_data['error'], 'incorrect_password')

        # Verify user account still exists
        self.assertTrue(User.objects.filter(username='testuser').exists())

    def test_delete_account_missing_password(self):
        # Test that account deletion fails when password is not provided
        self.client.login(username='testuser', password='strongpassword123')
        data = {}
        response = self.client.post(
            self.delete_account_url,
            data=json.dumps(data),
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 400)
        response_data = json.loads(response.content)
        self.assertFalse(response_data['success'])
        self.assertEqual(response_data['error'], 'missing_password')

    def test_delete_account_requires_authentication(self):
        # Test that account deletion endpoint requires authentication
        data = {'password': 'strongpassword123'}
        response = self.client.post(
            self.delete_account_url,
            data=json.dumps(data),
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 302)  # Redirect to login page
