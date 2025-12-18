/* Author: Lian Shi, Claudia Sevilla, Karl Byland */

// predictions/input.js - Text analysis input page functionality



function showAlert(message) {
    const alert = document.getElementById('customAlert');
    const alertMsg = document.getElementById('alertMessage');

    if (alert && alertMsg) {
        alertMsg.textContent = message;
        alert.style.display = 'block';
        alert.classList.add('show');

        setTimeout(() => {
            alert.classList.remove('show');
            setTimeout(() => alert.style.display = 'none', 300);
        }, 5000);
    }
}

document.getElementById('alertClose')?.addEventListener('click', () => {
    const alert = document.getElementById('customAlert');
    alert.classList.remove('show');
    setTimeout(() => alert.style.display = 'none', 300);
});


document.addEventListener('DOMContentLoaded', async function () {
    const textInput = document.getElementById('textInput');
    const charCount = document.getElementById('charCount');
    const analyzeBtn = document.getElementById('analyzeBtn');
    const clearBtn = document.getElementById('clearBtn');
    const analysisForm = document.getElementById('analysisForm');
    const loadingOverlay = document.getElementById('loadingOverlay');

    const response = await fetch("/predictions/api/strings/");
    const strings = await response.json();

    // Example texts for different mental states
    const examples = {
        depression: strings["example_texts"][0],

        stress: strings["example_texts"][1],

        normal: strings["example_texts"][2],
    };

    // Character counter
    if (textInput && charCount) {
        textInput.addEventListener('input', function () {
            const length = this.value.length;
            charCount.textContent = `${length} / 5000`;

            // Change color based on length
            if (length > 4500) {
                charCount.style.color = '#ff4444';
            } else if (length > 4000) {
                charCount.style.color = '#ffa500';
            } else {
                charCount.style.color = '#666';
            }
        });
    }

    // Example buttons
    const exampleButtons = document.querySelectorAll('.btn-example');
    exampleButtons.forEach(button => {
        button.addEventListener('click', function () {
            const example = this.getAttribute('data-example');
            if (examples[example] && textInput) {
                textInput.value = examples[example];
                // Trigger input event to update character count
                textInput.dispatchEvent(new Event('input'));
            }
        });
    });

    // Clear button
    if (clearBtn && textInput) {
        clearBtn.addEventListener('click', function (e) {
            e.preventDefault();
            textInput.value = '';
            textInput.dispatchEvent(new Event('input'));
            textInput.focus();
        });
    }

    // Form submission
    if (analysisForm) {
        analysisForm.addEventListener('submit', function (e) {
            // Show loading overlay
            if (loadingOverlay) {
                loadingOverlay.style.display = 'flex';
            }

            // Validate text length
            if (textInput && textInput.value.trim().length < 10) {
                e.preventDefault();
                showAlert('Please enter at least 10 characters of text to analyze.');
                if (loadingOverlay) {
                    loadingOverlay.style.display = 'none';
                }
                return false;
            }

            if(textInput.value.trim().split(" ").length < 3) {
                e.preventDefault();
                showAlert('Please enter at least 3 words of text to analyze.');
                if (loadingOverlay) {
                    loadingOverlay.style.display = 'none';
                }
                return false;
            }

            // Disable submit button to prevent double submission
            if (analyzeBtn) {
                analyzeBtn.disabled = true;
            }
        });
    }

    // Auto-resize textarea
    if (textInput) {
        textInput.addEventListener('input', function () {
            this.style.height = 'auto';
            this.style.height = (this.scrollHeight) + 'px';
        });
    }
});