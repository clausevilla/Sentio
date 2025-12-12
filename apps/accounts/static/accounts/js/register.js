// Author: Lian Shi
// Disclaimer: LLM has been used to help with password strength indication, real-time validation, and AJAX form submission

/**
 * Registration Page JavaScript
 * Features:
 * - Real-time validation for all fields
 * - AJAX username/email availability checking
 * - Password strength indicator
 * - AJAX form submission (no page refresh)
 */

document.addEventListener('DOMContentLoaded', function () {
    const registerForm = document.getElementById('registerForm');
    const usernameInput = document.getElementById('username');
    const emailInput = document.getElementById('email');
    const password1Input = document.getElementById('password1');
    const password2Input = document.getElementById('password2');
    const consentCheckbox = document.getElementById('consent');
    const submitBtn = registerForm ? registerForm.querySelector('button[type="submit"]') : null;

    // Debounce timers for API calls
    let usernameTimer = null;
    let emailTimer = null;

    // Validation state tracking
    const validationState = {
        username: false,
        email: false,
        password1: false,
        password2: false,
        consent: false
    };

    // Get CSRF token
    const csrfToken = document.querySelector('[name=csrfmiddlewaretoken]')?.value || '';

    // API endpoints
    const API = {
        checkUsername: '/accounts/api/check-username/',
        checkEmail: '/accounts/api/check-email/',
        register: '/accounts/api/register/'
    };

    // ========================================
    // PASSWORD STRENGTH INDICATOR SETUP
    // ========================================
    if (password1Input) {
        const formGroup = password1Input.closest('.form-group');
        if (formGroup && !formGroup.querySelector('.password-strength')) {
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

    // ========================================
    // REAL-TIME USERNAME VALIDATION
    // ========================================
    if (usernameInput) {
        usernameInput.addEventListener('input', function () {
            const username = this.value.trim();

            // Clear previous timer
            if (usernameTimer) clearTimeout(usernameTimer);

            // Clear previous state
            clearError(this);
            clearSuccess(this);
            validationState.username = false;

            // Empty check
            if (!username) {
                return;
            }

            // Client-side validation first
            if (username.length < 3) {
                showError(this, 'Username must be at least 3 characters');
                return;
            }

            if (!/^[a-zA-Z0-9_]+$/.test(username)) {
                showError(this, 'Only letters, numbers, and underscores allowed');
                return;
            }

            // Show checking state
            showChecking(this, 'Checking availability...');

            // Debounced API call to check availability
            usernameTimer = setTimeout(() => {
                checkUsernameAvailability(username);
            }, 500);
        });

        // Also validate on blur
        usernameInput.addEventListener('blur', function () {
            const username = this.value.trim();
            if (username && !validationState.username) {
                // Clear checking state and revalidate
                if (usernameTimer) clearTimeout(usernameTimer);
                checkUsernameAvailability(username);
            }
        });
    }

    // ========================================
    // REAL-TIME EMAIL VALIDATION
    // ========================================
    if (emailInput) {
        emailInput.addEventListener('input', function () {
            const email = this.value.trim();

            // Clear previous timer
            if (emailTimer) clearTimeout(emailTimer);

            // Clear previous state
            clearError(this);
            clearSuccess(this);
            validationState.email = false;

            // Empty check
            if (!email) {
                return;
            }

            // Client-side format validation
            if (!isValidEmail(email)) {
                showError(this, 'Please enter a valid email address');
                return;
            }

            // Show checking state
            showChecking(this, 'Checking availability...');

            // Debounced API call to check availability
            emailTimer = setTimeout(() => {
                checkEmailAvailability(email);
            }, 500);
        });

        // Also validate on blur
        emailInput.addEventListener('blur', function () {
            const email = this.value.trim();
            if (email && !validationState.email) {
                if (emailTimer) clearTimeout(emailTimer);
                if (isValidEmail(email)) {
                    checkEmailAvailability(email);
                }
            }
        });
    }

    // ========================================
    // REAL-TIME PASSWORD VALIDATION
    // ========================================
    if (password1Input) {
        password1Input.addEventListener('input', function () {
            const password = this.value;

            clearError(this);
            clearSuccess(this);
            validationState.password1 = false;

            // Update strength indicator
            updatePasswordStrength(password);

            if (!password) {
                return;
            }

            // Check requirements
            const requirements = checkPasswordRequirements(password);

            if (!requirements.valid) {
                showError(this, requirements.message);
                return;
            }

            validationState.password1 = true;
            showSuccess(this, '✓ Password meets requirements');

            // Re-validate password2 if it has a value
            if (password2Input && password2Input.value) {
                validatePasswordMatch();
            }
        });
    }

    // ========================================
    // REAL-TIME PASSWORD CONFIRMATION
    // ========================================
    if (password2Input) {
        password2Input.addEventListener('input', function () {
            validatePasswordMatch();
        });
    }

    // ========================================
    // CONSENT CHECKBOX VALIDATION
    // ========================================
    if (consentCheckbox) {
        consentCheckbox.addEventListener('change', function () {
            validationState.consent = this.checked;
            if (this.checked) {
                clearConsentError();
            } else {
                showConsentError();
            }
        });
    }

    // ========================================
    // FORM SUBMISSION (AJAX)
    // ========================================
    if (registerForm) {
        registerForm.addEventListener('submit', async function (e) {
            e.preventDefault();

            // Clear all previous errors
            clearAllErrors();

            // Run all validations
            let isValid = true;

            // Validate username
            if (usernameInput) {
                const username = usernameInput.value.trim();
                if (!username) {
                    showError(usernameInput, 'Username is required');
                    isValid = false;
                } else if (!validationState.username) {
                    // Need to check availability
                    const available = await checkUsernameAvailabilitySync(username);
                    if (!available) {
                        isValid = false;
                    }
                }
            }

            // Validate email
            if (emailInput) {
                const email = emailInput.value.trim();
                if (!email) {
                    showError(emailInput, 'Email is required');
                    isValid = false;
                } else if (!validationState.email) {
                    const available = await checkEmailAvailabilitySync(email);
                    if (!available) {
                        isValid = false;
                    }
                }
            }

            // Validate password
            if (password1Input) {
                const password = password1Input.value;
                if (!password) {
                    showError(password1Input, 'Password is required');
                    isValid = false;
                } else if (!validationState.password1) {
                    const requirements = checkPasswordRequirements(password);
                    if (!requirements.valid) {
                        showError(password1Input, requirements.message);
                        isValid = false;
                    }
                }
            }

            // Validate password confirmation
            if (password2Input) {
                if (!password2Input.value) {
                    showError(password2Input, 'Please confirm your password');
                    isValid = false;
                } else if (!validationState.password2) {
                    if (password2Input.value !== password1Input.value) {
                        showError(password2Input, 'Passwords do not match');
                        isValid = false;
                    }
                }
            }

            // Validate consent
            if (consentCheckbox && !consentCheckbox.checked) {
                showConsentError();
                isValid = false;
            }

            // If validation failed, scroll to first error
            if (!isValid) {
                const firstError = document.querySelector('.form-group .error, .consent-section.error');
                if (firstError) {
                    firstError.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }
                return;
            }

            // Show loading state
            setSubmitLoading(true);

            // Prepare form data
            const formData = {
                username: usernameInput.value.trim(),
                email: emailInput.value.trim(),
                password1: password1Input.value,
                password2: password2Input.value,
                consent: consentCheckbox.checked
            };

            try {
                const response = await fetch(API.register, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'X-CSRFToken': csrfToken,
                        'X-Requested-With': 'XMLHttpRequest'
                    },
                    credentials: 'same-origin',
                    body: JSON.stringify(formData)
                });

                const data = await response.json();

                if (data.success) {
                    // Show success message
                    showFormSuccess(data.message);

                    // Redirect after short delay
                    setTimeout(() => {
                        window.location.href = data.redirect || '/predictions/input';
                    }, 1000);
                } else {
                    // Show errors from server
                    setSubmitLoading(false);

                    if (data.errors) {
                        displayServerErrors(data.errors);
                    }
                }
            } catch (error) {
                console.error('Registration error:', error);
                setSubmitLoading(false);
                showFormError('An error occurred. Please try again.');
            }
        });
    }

    // ========================================
    // PASSWORD TOGGLE
    // ========================================
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

    // ========================================
    // HELPER FUNCTIONS
    // ========================================

    /**
     * Check username availability via API
     */
    async function checkUsernameAvailability(username) {
        try {
            const response = await fetch(API.checkUsername, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrfToken,
                    'X-Requested-With': 'XMLHttpRequest'
                },
                credentials: 'same-origin',
                body: JSON.stringify({ username })
            });

            const data = await response.json();

            clearChecking(usernameInput);

            if (data.available) {
                validationState.username = true;
                showSuccess(usernameInput, '✓ Username is available');
            } else {
                validationState.username = false;
                showError(usernameInput, data.error || 'Username not available');
            }
        } catch (error) {
            console.error('Username check error:', error);
            clearChecking(usernameInput);
            showError(usernameInput, 'Could not verify username');
        }
    }

    /**
     * Check username availability (synchronous version for form submit)
     */
    async function checkUsernameAvailabilitySync(username) {
        try {
            const response = await fetch(API.checkUsername, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrfToken,
                    'X-Requested-With': 'XMLHttpRequest'
                },
                credentials: 'same-origin',
                body: JSON.stringify({ username })
            });

            const data = await response.json();

            if (data.available) {
                validationState.username = true;
                showSuccess(usernameInput, '✓ Username is available');
                return true;
            } else {
                validationState.username = false;
                showError(usernameInput, data.error || 'Username not available');
                return false;
            }
        } catch (error) {
            showError(usernameInput, 'Could not verify username');
            return false;
        }
    }

    /**
     * Check email availability via API
     */
    async function checkEmailAvailability(email) {
        try {
            const response = await fetch(API.checkEmail, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrfToken,
                    'X-Requested-With': 'XMLHttpRequest'
                },
                credentials: 'same-origin',
                body: JSON.stringify({ email })
            });

            const data = await response.json();

            clearChecking(emailInput);

            if (data.available) {
                validationState.email = true;
                showSuccess(emailInput, '✓ Email is available');
            } else {
                validationState.email = false;
                showError(emailInput, data.error || 'Email not available');
            }
        } catch (error) {
            console.error('Email check error:', error);
            clearChecking(emailInput);
            showError(emailInput, 'Could not verify email');
        }
    }

    /**
     * Check email availability (synchronous version for form submit)
     */
    async function checkEmailAvailabilitySync(email) {
        try {
            const response = await fetch(API.checkEmail, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': csrfToken,
                    'X-Requested-With': 'XMLHttpRequest'
                },
                credentials: 'same-origin',
                body: JSON.stringify({ email })
            });

            const data = await response.json();

            if (data.available) {
                validationState.email = true;
                showSuccess(emailInput, '✓ Email is available');
                return true;
            } else {
                validationState.email = false;
                showError(emailInput, data.error || 'Email not available');
                return false;
            }
        } catch (error) {
            showError(emailInput, 'Could not verify email');
            return false;
        }
    }

    /**
     * Validate password match
     */
    function validatePasswordMatch() {
        clearError(password2Input);
        clearSuccess(password2Input);
        validationState.password2 = false;

        const password1 = password1Input.value;
        const password2 = password2Input.value;

        if (!password2) {
            return;
        }

        if (password1 === password2) {
            validationState.password2 = true;
            showSuccess(password2Input, '✓ Passwords match');
        } else {
            showError(password2Input, 'Passwords do not match');
        }
    }

    /**
     * Check password requirements
     */
    function checkPasswordRequirements(password) {
        if (password.length < 8) {
            return { valid: false, message: 'Password must be at least 8 characters' };
        }

        // Django's default password validators typically check for:
        // - Not entirely numeric
        // - Not too similar to username
        // - Not a common password

        if (/^\d+$/.test(password)) {
            return { valid: false, message: 'Password cannot be entirely numeric' };
        }

        // Check if username field exists and password is too similar
        if (usernameInput) {
            const username = usernameInput.value.trim().toLowerCase();
            if (username && password.toLowerCase().includes(username)) {
                return { valid: false, message: 'Password is too similar to username' };
            }
        }

        return { valid: true, message: '' };
    }

    /**
     * Simple email format validation
     */
    function isValidEmail(email) {
        return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email);
    }

    /**
     * Update password strength indicator
     */
    function updatePasswordStrength(password) {
        const strengthBar = document.querySelector('.strength-bar-fill');
        const strengthText = document.querySelector('.strength-text');
        const strengthIndicator = document.querySelector('.password-strength');

        if (!strengthBar || !strengthText || !strengthIndicator) return;

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

        if (password.length < 6) {
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

        strengthBar.style.width = Math.min(strength, 100) + '%';
        strengthBar.style.backgroundColor = color;
        strengthText.textContent = text;
        strengthText.style.color = color;
    }

    /**
     * Show error message for an input
     */
    function showError(input, message) {
        const formGroup = input.closest('.form-group');
        if (!formGroup) return;

        input.classList.add('error');
        input.classList.remove('success');

        // Remove any existing messages
        removeMessages(formGroup);

        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.innerHTML = '<i class="fas fa-exclamation-circle"></i> ' + message;
        formGroup.appendChild(errorDiv);
    }

    /**
     * Show success message for an input
     */
    function showSuccess(input, message) {
        const formGroup = input.closest('.form-group');
        if (!formGroup) return;

        input.classList.add('success');
        input.classList.remove('error');

        // Remove any existing messages
        removeMessages(formGroup);

        const successDiv = document.createElement('div');
        successDiv.className = 'success-message';
        successDiv.innerHTML = message;
        formGroup.appendChild(successDiv);
    }

    /**
     * Show checking/loading state for an input
     */
    function showChecking(input, message) {
        const formGroup = input.closest('.form-group');
        if (!formGroup) return;

        input.classList.remove('error', 'success');

        // Remove any existing messages
        removeMessages(formGroup);

        const checkingDiv = document.createElement('div');
        checkingDiv.className = 'checking-message';
        checkingDiv.innerHTML = '<i class="fas fa-spinner fa-spin"></i> ' + message;
        formGroup.appendChild(checkingDiv);
    }

    /**
     * Clear checking state
     */
    function clearChecking(input) {
        const formGroup = input.closest('.form-group');
        if (!formGroup) return;

        const checkingDiv = formGroup.querySelector('.checking-message');
        if (checkingDiv) checkingDiv.remove();
    }

    /**
     * Remove all messages from a form group
     */
    function removeMessages(formGroup) {
        formGroup.querySelectorAll('.error-message, .success-message, .checking-message').forEach(el => el.remove());
    }

    /**
     * Clear error for an input
     */
    function clearError(input) {
        const formGroup = input.closest('.form-group');
        if (!formGroup) return;

        input.classList.remove('error');
        const errorDiv = formGroup.querySelector('.error-message');
        if (errorDiv) errorDiv.remove();
    }

    /**
     * Clear success for an input
     */
    function clearSuccess(input) {
        const formGroup = input.closest('.form-group');
        if (!formGroup) return;

        input.classList.remove('success');
        const successDiv = formGroup.querySelector('.success-message');
        if (successDiv) successDiv.remove();
    }

    /**
     * Clear all errors and messages
     */
    function clearAllErrors() {
        document.querySelectorAll('.error-message, .success-message, .checking-message').forEach(el => el.remove());
        document.querySelectorAll('.error, .success').forEach(el => {
            el.classList.remove('error', 'success');
        });
        clearConsentError();

        // Remove any form-level alerts
        document.querySelectorAll('.form-alert').forEach(el => el.remove());
    }

    /**
     * Show consent error
     */
    function showConsentError() {
        const consentGroup = document.getElementById('consentGroup');
        if (consentGroup) {
            consentGroup.classList.add('error');
        }
    }

    /**
     * Clear consent error
     */
    function clearConsentError() {
        const consentGroup = document.getElementById('consentGroup');
        if (consentGroup) {
            consentGroup.classList.remove('error');
        }
    }

    /**
     * Set submit button loading state
     */
    function setSubmitLoading(loading) {
        if (!submitBtn) return;

        if (loading) {
            submitBtn.disabled = true;
            submitBtn.dataset.originalText = submitBtn.textContent;
            submitBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Creating Account...';
        } else {
            submitBtn.disabled = false;
            submitBtn.textContent = submitBtn.dataset.originalText || 'Register';
        }
    }

    /**
     * Show form-level success message
     */
    function showFormSuccess(message) {
        // Remove any existing alerts
        document.querySelectorAll('.form-alert').forEach(el => el.remove());

        const alert = document.createElement('div');
        alert.className = 'form-alert form-alert-success';
        alert.innerHTML = '<i class="fas fa-check-circle"></i> ' + message;

        registerForm.insertBefore(alert, registerForm.firstChild);
    }

    /**
     * Show form-level error message
     */
    function showFormError(message) {
        // Remove any existing alerts
        document.querySelectorAll('.form-alert').forEach(el => el.remove());

        const alert = document.createElement('div');
        alert.className = 'form-alert form-alert-error';
        alert.innerHTML = '<i class="fas fa-exclamation-circle"></i> ' + message;

        registerForm.insertBefore(alert, registerForm.firstChild);
    }

    /**
     * Display server-side validation errors
     */
    function displayServerErrors(errors) {
        const fieldMap = {
            'username': usernameInput,
            'email': emailInput,
            'password1': password1Input,
            'password2': password2Input,
            'consent': null
        };

        for (const [field, messages] of Object.entries(errors)) {
            if (field === '__all__') {
                // Form-level error
                showFormError(messages[0]);
            } else if (field === 'consent') {
                showConsentError();
            } else if (fieldMap[field]) {
                showError(fieldMap[field], messages[0]);
            }
        }

        // Scroll to first error
        const firstError = document.querySelector('.form-alert, .form-group .error-message, .consent-section.error');
        if (firstError) {
            firstError.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    }
});