// ===================================
// Simple Profile Page
// ===================================
console.log('Toast element:', document.getElementById('dynamicToast'));
console.log('Toast icon:', document.getElementById('toastIcon'));
console.log('Toast title:', document.getElementById('toastTitle'));
console.log('Toast message:', document.getElementById('toastMessage'));

// Get CSRF token for Django
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

const csrftoken = getCookie('csrftoken');

// Show toast message - creates toast dynamically
function showMessage(message, type = 'success') {
    // Remove any existing dynamic toast
    const existingToast = document.getElementById('dynamicToast');
    if (existingToast) {
        existingToast.remove();
    }

    // Set icon and title based on type
    let icon = 'fa-check-circle';
    let title = 'Success';
    if (type === 'error') {
        icon = 'fa-exclamation-circle';
        title = 'Error';
    } else if (type === 'warning') {
        icon = 'fa-exclamation-triangle';
        title = 'Warning';
    }

    // Create toast HTML
    const toast = document.createElement('div');
    toast.id = 'dynamicToast';
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <div class="toast-icon">
            <i class="fas ${icon}"></i>
        </div>
        <div class="toast-content">
            <div class="toast-title">${title}</div>
            <div class="toast-message">${message}</div>
        </div>
        <button class="toast-close" onclick="hideToast()">
            <i class="fas fa-times"></i>
        </button>
        <div class="toast-progress"></div>
    `;

    // Add to toast container
    const container = document.getElementById('toastContainer');
    if (container) {
        container.appendChild(toast);
    } else {
        document.body.appendChild(toast);
    }

    // Auto-hide after 3 seconds
    setTimeout(hideToast, 3000);
}

// Hide toast
function hideToast() {
    const toast = document.getElementById('dynamicToast');
    if (!toast) return;

    toast.classList.add('hiding');
    setTimeout(() => {
        toast.remove();
    }, 300);
}

// Password validation
function validatePassword(password) {
    const errors = [];

    if (password.length < 8) {
        errors.push('Password must be at least 8 characters');
    }


    return {
        isValid: errors.length === 0,
        errors: errors
    };
}

// Change Password Form
document.getElementById('changePasswordForm')?.addEventListener('submit', async function (e) {
    e.preventDefault();

    // Clear errors
    document.querySelectorAll('.form-error').forEach(el => el.textContent = '');

    const currentPassword = document.getElementById('currentPassword').value;
    const newPassword = document.getElementById('newPassword').value;
    const confirmPassword = document.getElementById('confirmPassword').value;

    let hasError = false;

    // Validate new password
    const validation = validatePassword(newPassword);
    if (!validation.isValid) {
        document.getElementById('newPasswordError').textContent = validation.errors.join('. ');
        hasError = true;
    }

    // Check passwords match
    if (newPassword !== confirmPassword) {
        document.getElementById('confirmPasswordError').textContent = 'Passwords do not match';
        hasError = true;
    }

    // Check new password is different
    if (currentPassword === newPassword) {
        document.getElementById('newPasswordError').textContent = 'New password must be different';
        hasError = true;
    }

    if (hasError) return;

    try {
        const response = await fetch('/accounts/api/change-password/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrftoken
            },
            body: JSON.stringify({
                currentPassword,
                newPassword
            })
        });

        const data = await response.json();

        if (response.ok) {
            showMessage('✅ Password changed successfully!', 'success');
            document.getElementById('changePasswordForm').reset();
        } else {
            if (data.error === 'incorrect_password') {
                document.getElementById('currentPasswordError').textContent = 'Current password is incorrect';
            } else {
                showMessage('❌ ' + (data.message || 'Failed to change password'), 'error');
            }
        }
    } catch (error) {
        console.error('Error:', error);
        showMessage('❌ An error occurred. Please try again.', 'error');
    }
});

// Delete Data Modal
function openDeleteDataModal() {
    document.getElementById('deleteDataModal').classList.add('active');
    document.body.style.overflow = 'hidden';
}

function closeDeleteDataModal() {
    document.getElementById('deleteDataModal').classList.remove('active');
    document.body.style.overflow = 'auto';
    document.getElementById('deleteDataForm').reset();
    document.getElementById('confirmDeleteDataBtn').disabled = true;
}

// Enable delete button when text matches
document.getElementById('deleteDataConfirm')?.addEventListener('input', function (e) {
    const btn = document.getElementById('confirmDeleteDataBtn');
    btn.disabled = e.target.value !== 'DELETE ALL DATA';
});

// Delete Data Form
document.getElementById('deleteDataForm')?.addEventListener('submit', async function (e) {
    e.preventDefault();

    const password = document.getElementById('deleteDataPassword').value;
    const confirmation = document.getElementById('deleteDataConfirm').value;

    if (confirmation !== 'DELETE ALL DATA') {
        showMessage('Please type the confirmation text exactly.', 'error');
        closeDeleteDataModal();
        return;
    }

    try {
        const response = await fetch('/accounts/api/delete-all-data/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrftoken
            },
            body: JSON.stringify({ password })
        });

        const data = await response.json();

        if (response.ok) {
            closeDeleteDataModal();
            showMessage('All your data has been deleted', 'success');
            setTimeout(() => {
                window.location.reload();
            }, 3000);
        } else {
            if (data.error === 'incorrect_password') {
                showMessage('Incorrect password. Please try again.', 'error');
                closeDeleteDataModal();
            } else {
                showMessage(data.message || 'Failed to delete data', 'error');
                closeDeleteDataModal();
            }
        }
    } catch (error) {
        console.error('Error:', error);
        showMessage('An error occurred. Please try again.', 'error');
        closeDeleteDataModal();
    }
});

// Delete Account Modal
function openDeleteAccountModal() {
    document.getElementById('deleteAccountModal').classList.add('active');
    document.body.style.overflow = 'hidden';
}

function closeDeleteAccountModal() {
    document.getElementById('deleteAccountModal').classList.remove('active');
    document.body.style.overflow = 'auto';
    document.getElementById('deleteAccountForm').reset();
    document.getElementById('confirmDeleteAccountBtn').disabled = true;
}

// Enable delete button when text matches
document.getElementById('deleteAccountConfirm')?.addEventListener('input', function (e) {
    const btn = document.getElementById('confirmDeleteAccountBtn');
    btn.disabled = e.target.value !== 'DELETE MY ACCOUNT';
});

// Delete Account Form
document.getElementById('deleteAccountForm')?.addEventListener('submit', async function (e) {
    e.preventDefault();

    const password = document.getElementById('deleteAccountPassword').value;
    const confirmation = document.getElementById('deleteAccountConfirm').value;

    if (confirmation !== 'DELETE MY ACCOUNT') {
        showMessage('Please type the confirmation text exactly.', 'warning');
        closeDeleteAccountModal();
        return;
    }

    try {
        const response = await fetch('/accounts/api/delete-account/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrftoken
            },
            body: JSON.stringify({ password })
        });

        const data = await response.json();

        if (response.ok) {
            closeDeleteAccountModal();
            showMessage('Your account has been deleted. Redirecting...', 'success');
            setTimeout(() => {
                window.location.href = '/';
            }, 3000);

        } else {
            if (data.error === 'incorrect_password') {
                showMessage('Incorrect password. Please try again.', 'error');
                closeDeleteAccountModal();
            } else {
                showMessage(data.message || 'Failed to delete account', 'error');
                closeDeleteAccountModal();
            }
        }
    } catch (error) {
        console.error('Error:', error);
        showMessage('An error occurred. Please try again.', 'error');
        closeDeleteAccountModal();
    }
});

// Close modal when clicking outside
document.getElementById('deleteDataModal')?.addEventListener('click', function (e) {
    if (e.target === this) closeDeleteDataModal();
});

document.getElementById('deleteAccountModal')?.addEventListener('click', function (e) {
    if (e.target === this) closeDeleteAccountModal();
});

// Close with Escape key
document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') {
        closeDeleteDataModal();
        closeDeleteAccountModal();
    }
});