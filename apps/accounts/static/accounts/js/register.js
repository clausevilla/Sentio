// Author: Lian Shi */
// Disclaimer: LLM has been used to help with password strength indication and consent group/password toggle button bug fix

// accounts/register.js - Registration page with validation

document.addEventListener('DOMContentLoaded', function () {
    const registerForm = document.getElementById('registerForm');
    const usernameInput = document.getElementById('username');
    const emailInput = document.getElementById('email');
    const password1Input = document.getElementById('password1');
    const password2Input = document.getElementById('password2');
    const consentCheckbox = document.getElementById('consent');
    const consentGroup = document.getElementById('consentGroup');
    const consentError = document.getElementById('consentError');

    if (registerForm) {
        registerForm.addEventListener('submit', function (e) {
            let isValid = true;

            // Clear previous errors
            clearErrors();
            clearConsentError();

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

            // Validate consent checkbox
            if (consentCheckbox && !consentCheckbox.checked) {
                showConsentError();
                isValid = false;
            }

            // If validation failed, prevent form submission
            if (!isValid) {
                e.preventDefault();
                // Scroll to first error
                const firstError = document.querySelector('.error, .consent-section.error');
                if (firstError) {
                    firstError.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }
                return false;
            }

            // Show loading state on submit button
            const submitBtn = registerForm.querySelector('button[type="submit"]');
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.textContent = 'Creating Account...';
            }
        });
    }

    // Clear consent error when checkbox is checked
    if (consentCheckbox) {
        consentCheckbox.addEventListener('change', function () {
            if (this.checked) {
                clearConsentError();
            }
        });
    }

    // Real-time password strength indicator
    if (password1Input) {
        // Update strength on input change by calling updatePasswordStrength
        password1Input.addEventListener('input', function () {
            updatePasswordStrength(this.value);
        });

        // Add password strength indicator
        const formGroup = password1Input.closest('.form-group');
        if (formGroup) {
            // Check if strength indicator already exists
            if (!formGroup.querySelector('.password-strength')) {
                const strengthIndicator = document.createElement('div');
                strengthIndicator.className = 'password-strength';
                strengthIndicator.innerHTML = `
                    <div class="strength-bar">
                        <div class="strength-bar-fill"></div>
                    </div>
                    <div class="strength-text"></div>
                `;
                const inputWrapper = password1Input.closest('.input-wrapper');
                if (inputWrapper) {
                    inputWrapper.insertAdjacentElement('afterend', strengthIndicator);
                } else {
                    formGroup.appendChild(strengthIndicator);
                }
            }
        }
    }

    // Real-time password match indicator
    if (password2Input && password1Input) {
        password2Input.addEventListener('input', function () {

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
            input.addEventListener('input', function () {
                clearError(this);
                clearSuccess(this);
            });
        }
    });

    // handle password toggles to password fields
    document.querySelectorAll('.password-toggle').forEach(function (toggle) {
        toggle.addEventListener('click', function () {
            const inputWrapper = this.closest('.input-wrapper');
            const input = inputWrapper ? inputWrapper.querySelector('input') : null;
            const icon = this.querySelector('i');

            if (input) {
                if (input.type === 'password') {
                    input.type = 'text';
                    icon.classList.remove('fa-eye-slash');
                    icon.classList.add('fa-eye');
                } else {
                    input.type = 'password';
                    icon.classList.remove('fa-eye');
                    icon.classList.add('fa-eye-slash');
                }
            }
        });
    });
});


// Simple email format validation to check if email is valid
function isValidEmail(email) {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
}



// Update password strength indicator according to the password entered
function updatePasswordStrength(password) {
    const strengthBar = document.querySelector('.strength-bar-fill');
    const strengthText = document.querySelector('.strength-text');
    const strengthIndicator = document.querySelector('.password-strength');

    if (!strengthBar || !strengthText || !strengthIndicator) return;

    // hide indicator if password is empty
    if (password.length === 0) {
        strengthIndicator.style.display = 'none';
        strengthBar.style.width = '0%';
        strengthText.textContent = '';
        return;
    }
    strengthIndicator.style.display = 'block';

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

// Show consent error - adds error class to consent section
function showConsentError() {
    const consentGroup = document.getElementById('consentGroup');
    if (consentGroup) {
        consentGroup.classList.add('error');
    }
}

// Clear consent error - removes error class from consent section
function clearConsentError() {
    const consentGroup = document.getElementById('consentGroup');
    if (consentGroup) {
        consentGroup.classList.remove('error');
    }
}