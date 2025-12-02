// Author: Lian Shi
// accounts/consent.js - Consent page functionality

document.addEventListener('DOMContentLoaded', function() {
    const consentCheckbox = document.getElementById('consentCheckbox');
    const consentButton = document.getElementById('consentButton');
    const checkboxContainer = document.getElementById('consentCheckboxContainer');

    if (consentCheckbox && consentButton) {
        consentCheckbox.addEventListener('change', function() {
            consentButton.disabled = !this.checked;

            if (this.checked) {
                checkboxContainer.classList.add('checked');
            } else {
                checkboxContainer.classList.remove('checked');
            }
        });
    }

    const consentForm = document.getElementById('consentForm');
    if (consentForm) {
        consentForm.addEventListener('submit', function(e) {
            if (!consentCheckbox.checked) {
                e.preventDefault();
                alert('Please check the consent box to continue.');
                return false;
            }

            consentButton.disabled = true;
            consentButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
        });
    }
});

