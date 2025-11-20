// accounts/register.js - Registration page with validation

document.addEventListener('DOMContentLoaded', function() {
    const registerForm = document.querySelector('form[action*="register"]');
    const usernameInput = document.getElementById('username');
    const emailInput = document.getElementById('email');
    const password1Input = document.getElementById('password1');
    const password2Input = document.getElementById('password2');

    if (registerForm) {
        registerForm.addEventListener('submit', function(e) {
            let isValid = true;

            // Clear previous errors
            clearErrors();

            // Validate username
            if (usernameInput) {
                const username = usernameInput.value.trim();
                if (username === '') {
                    showError(usernameInput, 'Username is required');
                    isValid = false;
                } else if (username.length < 3) {
                    showError(usernameInput, 'Username must be at least 3 characters');
                    isValid = false;
                } else if (!/^[a-zA-Z0-9_]+$/.test(username)) {
                    showError(usernameInput, 'Username can only contain letters, numbers, and underscores');
                    isValid = false;
                }
            }

            // Validate email
            if (emailInput) {
                const email = emailInput.value.trim();
                if (email === '') {
                    showError(emailInput, 'Email is required');
                    isValid = false;
                } else if (!isValidEmail(email)) {
                    showError(emailInput, 'Please enter a valid email address');
                    isValid = false;
                }
            }

            // Validate password
            if (password1Input) {
                const password = password1Input.value;
                if (password === '') {
                    showError(password1Input, 'Password is required');
                    isValid = false;
                } else if (password.length < 8) {
                    showError(password1Input, 'Password must be at least 8 characters');
                    isValid = false;
                }
            }

            // Validate password confirmation
            if (password2Input && password1Input) {
                if (password2Input.value === '') {
                    showError(password2Input, 'Please confirm your password');
                    isValid = false;
                } else if (password2Input.value !== password1Input.value) {
                    showError(password2Input, 'Passwords do not match');
                    isValid = false;
                }
            }

            if (!isValid) {
                e.preventDefault();
                return false;
            }

            // Show loading state
            const submitBtn = registerForm.querySelector('button[type="submit"]');
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.textContent = 'Creating Account...';
            }
        });
    }

    // Real-time password strength indicator
    if (password1Input) {
        // Update strength on input change by calling updatePasswordStrength
        password1Input.addEventListener('input', function() {
            updatePasswordStrength(this.value);
        });

        // Add password strength indicator
        const formGroup = password1Input.closest('.form-group');
        if (formGroup) {
            const strengthIndicator = document.createElement('div');
            strengthIndicator.className = 'password-strength';
            strengthIndicator.innerHTML = `
                <div class="strength-bar">
                    <div class="strength-bar-fill"></div>
                </div>
                <div class="strength-text"></div>
            `;
            formGroup.appendChild(strengthIndicator);
        }
    }

    // Real-time password match indicator
    if (password2Input && password1Input) {
        password2Input.addEventListener('input', function() {

            clearError(this);
            clearSuccess(this);

            if (this.value && password1Input.value) {
                if (this.value === password1Input.value) {
                    showSuccess(this, '✓ Passwords match');
                } else {
                    showError(this, 'x Passwords do not match');
                }
            }
        });
    }

    // Clear errors on input
    [usernameInput, emailInput, password1Input, password2Input].forEach(input => {
        if (input) {
            input.addEventListener('input', function() {
                clearError(this);
                clearSuccess(this);
            });
        }
    });

    // Add show/hide password toggles
    addPasswordToggles();
});

// Simple email format validation to check if email is valid
function isValidEmail(email) {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
}

// Simple password strength check
function isStrongPassword(password) {
    // At least one uppercase, one lowercase, one number
    return /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).+$/.test(password);
}

// Update password strength indicator according to the password entered
function updatePasswordStrength(password) {
    const strengthBar = document.querySelector('.strength-bar-fill');
    const strengthText = document.querySelector('.strength-text');

    if (!strengthBar || !strengthText) return;

    let strength = 0;
    let text = '';
    let color = '';

    if (password.length === 0) {
        strength = 0;
        text = '';
    } else if (password.length < 6) {
        strength = 25;
        text = 'Weak';
        color = '#ff4444';
    } else if (password.length < 8) {
        strength = 50;
        text = 'Fair';
        color = '#ffa500';
    } else {
        strength = 75;
        text = 'Good';
        color = '#7FAF93';

        // Bonus points for complexity
        if (/[A-Z]/.test(password)) strength += 5;
        if (/[a-z]/.test(password)) strength += 5;
        if (/\d/.test(password)) strength += 5;
        if (/[^A-Za-z0-9]/.test(password)) strength += 10;

        if (strength >= 90) {
            text = 'Strong';
            color = '#4CAF50';
        }
    }

    strengthBar.style.width = strength + '%';
    strengthBar.style.backgroundColor = color;
    strengthText.textContent = text;
    strengthText.style.color = color;
}

// Show error message for input such as username, email, password
function showError(input, message) {
    const formGroup = input.closest('.form-group');
    if (!formGroup) return;

    input.classList.add('error');
    input.classList.remove('success');

    let errorDiv = formGroup.querySelector('.error-message');
    if (!errorDiv) {
        errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        formGroup.appendChild(errorDiv);
    }
    errorDiv.textContent = message;
    errorDiv.style.color = '#ff4444'

    // Remove success message if exists
    const successDiv = formGroup.querySelector('.success-message');
    if (successDiv) successDiv.remove();
}

// Show success message for input such as password confirmation
function showSuccess(input, message) {
    const formGroup = input.closest('.form-group');
    if (!formGroup) return;

    input.classList.add('success');
    input.classList.remove('error');

    let successDiv = formGroup.querySelector('.success-message');
    if (!successDiv) {
        successDiv = document.createElement('div');
        successDiv.className = 'success-message';
        formGroup.appendChild(successDiv);
    }
    successDiv.textContent = message;
    successDiv.style.color = '#4CAF50';

    // Remove error message if exists
    const errorDiv = formGroup.querySelector('.error-message');
    if (errorDiv) errorDiv.remove();
}

// Clear error message and styling for a specific input
function clearError(input) {
    const formGroup = input.closest('.form-group');
    if (!formGroup) return;

    input.classList.remove('error');
    const errorDiv = formGroup.querySelector('.error-message');
    if (errorDiv) errorDiv.remove();
}

// Clear success message
function clearSuccess(input) {
    const formGroup = input.closest('.form-group');
    if (!formGroup) return;

    input.classList.remove('success');
    const successDiv = formGroup.querySelector('.success-message');
    if (successDiv) successDiv.remove();
}

// Clear all errors and success messages from the form
function clearErrors() {
    document.querySelectorAll('.error-message').forEach(el => el.remove());
    document.querySelectorAll('.success-message').forEach(el => el.remove());
    document.querySelectorAll('.error, .success').forEach(el => {
        el.classList.remove('error', 'success');
    });
}

// Add show/hide password toggles to password fields
function addPasswordToggles() {
    const passwordInputs = [
        document.getElementById('password1'),
        document.getElementById('password2')
    ];

    passwordInputs.forEach(input => {
        if (!input) return;

        const toggleButton = document.createElement('button');
        toggleButton.type = 'button';
        toggleButton.className = 'password-toggle';
        toggleButton.innerHTML = '<i class="fas fa-eye"></i>';

        const formGroup = input.closest('.form-group');
        if (formGroup) {
            formGroup.style.position = 'relative';
            formGroup.appendChild(toggleButton);

            toggleButton.addEventListener('click', function() {
                const type = input.type === 'password' ? 'text' : 'password';
                input.type = type;
                this.innerHTML = type === 'password'
                    ? '<i class="fas fa-eye"></i>'
                    : '<i class="fas fa-eye-slash"></i>';
            });
        }
    });
}

