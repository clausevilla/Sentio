"""
Forms for user registration and authentication
"""

from django import forms
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError


class RegisterForm(UserCreationForm):
    """
    Custom registration form with email field and enhanced validation
    """

    email = forms.EmailField(
        required=True,
        widget=forms.EmailInput(
            attrs={
                'class': 'form-control',
                'placeholder': 'Enter your email',
                'id': 'email',
            }
        ),
    )
    username = forms.CharField(
        max_length=150,
        required=True,
        widget=forms.TextInput(
            attrs={
                'class': 'form-control',
                'placeholder': 'Enter your username',
                'id': 'username',
            }
        ),
    )
    password1 = forms.CharField(
        label='Password',
        widget=forms.PasswordInput(
            attrs={
                'class': 'form-control',
                'placeholder': 'Enter your password',
                'id': 'password1',
            }
        ),
    )
    password2 = forms.CharField(
        label='Confirm Password',
        widget=forms.PasswordInput(
            attrs={
                'class': 'form-control',
                'placeholder': 'Confirm your password',
                'id': 'password2',
            }
        ),
    )

    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']

    def clean_username(self):
        """
        Validate username - check if it already exists and meets requirements
        """
        username = self.cleaned_data.get('username')

        # Check if username already exists
        if User.objects.filter(username=username).exists():
            raise ValidationError(
                'This username is already taken. Please choose another.'
            )

        # Check username length
        if len(username) < 3:
            raise ValidationError('Username must be at least 3 characters long.')

        # Check if username contains only valid characters
        if not username.replace('_', '').isalnum():
            raise ValidationError(
                'Username can only contain letters, numbers, and underscores.'
            )

        return username

    def clean_email(self):
        """
        Validate email - check if it already exists
        """
        email = self.cleaned_data.get('email')

        if User.objects.filter(email=email).exists():
            raise ValidationError('An account with this email already exists.')

        return email

    def clean_password2(self):
        """
        Validate that the two password fields match
        """
        password1 = self.cleaned_data.get('password1')
        password2 = self.cleaned_data.get('password2')

        if password1 and password2 and password1 != password2:
            raise ValidationError('Passwords do not match.')

        return password2

    def save(self, commit=True):
        """
        Save the user with email
        """
        user = super().save(commit=False)
        user.email = self.cleaned_data['email']

        if commit:
            user.save()

        return user


class LoginForm(AuthenticationForm):
    """
    Custom login form with styled fields
    """

    username = forms.CharField(
        widget=forms.TextInput(
            attrs={
                'class': 'form-control',
                'placeholder': 'Enter your username',
                'id': 'username',
            }
        )
    )
    password = forms.CharField(
        widget=forms.PasswordInput(
            attrs={
                'class': 'form-control',
                'placeholder': 'Enter your password',
                'id': 'password',
            }
        )
    )
