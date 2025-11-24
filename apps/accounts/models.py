"""
Models for user accounts

Note: We're using Django's built-in User model which already includes:
- username (CharField, max 150 chars, unique)
- email (EmailField)
- password (automatically hashed by Django)
- date_joined (DateTimeField, auto-set on creation)
- is_active (BooleanField, default True)
- is_staff (BooleanField, default False)
- is_superuser (BooleanField, default False)
- last_login (DateTimeField, nullable)

The built-in User model provides everything we need for basic authentication.

"""




