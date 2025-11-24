"""
Models for user accounts
Owner: Authentication Team

Note: We're using Django's built-in User model which already includes:
- username (CharField, max 150 chars, unique)
- email (EmailField)
- password (automatically hashed by Django)
- date_joined (DateTimeField, auto-set on creation)
- first_name, last_name (optional CharField)
- is_active (BooleanField, default True)
- is_staff (BooleanField, default False)
- is_superuser (BooleanField, default False)
- last_login (DateTimeField, nullable)

The built-in User model provides everything we need for basic authentication.
If you need to add custom fields to users in the future, you can create
a Profile model with a OneToOne relationship to User.
"""


# Currently using Django's built-in User model
# No custom models needed for basic authentication

# Example of how to extend User model if needed in the future:
"""
class UserProfile(models.Model):
    '''
    Extended user profile with additional fields
    '''
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    bio = models.TextField(max_length=500, blank=True)
    phone_number = models.CharField(max_length=20, blank=True)
    avatar = models.ImageField(upload_to='avatars/', blank=True, null=True)

    # Add any other custom fields here

    def __str__(self):
        return f'{self.user.username} Profile'

    class Meta:
        verbose_name = 'User Profile'
        verbose_name_plural = 'User Profiles'
"""
