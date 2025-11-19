// accounts/login.js - Login page functionality

document.addEventListener('DOMContentLoaded', function() {
    const loginForm = document.querySelector('form[action*="login"]');
    const usernameInput = document.getElementById('username');
    const passwordInput = document.getElementById('password');

    if (loginForm) {
        loginForm.addEventListener('submit', function(e) {
            let isValid = true;

            // Clear previous errors
            clearErrors();

            // Validate username
            if (usernameInput && usernameInput.value.trim() === '') {
                showError(usernameInput, 'Username is required');
                isValid = false;
            }

            // Validate password
            if (passwordInput && passwordInput.value.trim() === '') {
                showError(passwordInput, 'Password is required');
                isValid = false;
            }

            if (!isValid) {
                e.preventDefault();
                return false;
            }

            // Show loading state
            const submitBtn = loginForm.querySelector('button[type="submit"]');
            if (submitBtn) {
                submitBtn.disabled = true;
                submitBtn.textContent = 'Logging in...';
            }
        });
    }

    // Add input event listeners to clear errors on input
    [usernameInput, passwordInput].forEach(input => {
        if (input) {
            input.addEventListener('input', function() {
                clearError(this);
            });
        }
    });

    // Show/hide password toggle
    addPasswordToggle();
});

function showError(input, message) {
    const formGroup = input.closest('.form-group');
    if (!formGroup) return;

    // Add error class
    input.classList.add('error');

    // Create or update error message
    let errorDiv = formGroup.querySelector('.error-message');
    if (!errorDiv) {
        errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        formGroup.appendChild(errorDiv);
    }
    errorDiv.textContent = message;
}

function clearError(input) {
    const formGroup = input.closest('.form-group');
    if (!formGroup) return;

    input.classList.remove('error');
    const errorDiv = formGroup.querySelector('.error-message');
    if (errorDiv) {
        errorDiv.remove();
    }
}

function clearErrors() {
    document.querySelectorAll('.error-message').forEach(el => el.remove());
    document.querySelectorAll('.error').forEach(el => el.classList.remove('error'));
}

function addPasswordToggle() {
    const passwordInput = document.getElementById('password');
    if (!passwordInput) return;

    const toggleButton = document.createElement('button');
    toggleButton.type = 'button';
    toggleButton.className = 'password-toggle';
    toggleButton.innerHTML = '<i class="fas fa-eye"></i>';
    toggleButton.setAttribute('aria-label', 'Toggle password visibility');

    const formGroup = passwordInput.closest('.form-group');
    if (formGroup) {
        formGroup.style.position = 'relative';
        formGroup.appendChild(toggleButton);

        toggleButton.addEventListener('click', function() {
            const type = passwordInput.type === 'password' ? 'text' : 'password';
            passwordInput.type = type;
            this.innerHTML = type === 'password'
                ? '<i class="fas fa-eye"></i>'
                : '<i class="fas fa-eye-slash"></i>';
        });
    }
}

