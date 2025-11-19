// predictions/result.js - Analysis results page with animations

document.addEventListener('DOMContentLoaded', function() {
    // Animate circular progress bar
    animateProgressBar();

    // Animate indicator bars
    animateIndicators();

    // Animate probability bars
    animateProbabilities();

    // Add fade-in animation to cards
    animateCards();
});

function animateProgressBar() {
    const progressBar = document.getElementById('progressBar');
    const scoreText = document.getElementById('scoreText');
    const mentalStateBadge = document.getElementById('mentalStateBadge');

    if (!progressBar || !scoreText) return;

    const confidence = parseInt(scoreText.textContent);
    const circumference = 2 * Math.PI * 95; // radius = 95

    // Set initial state
    progressBar.style.strokeDasharray = circumference;
    progressBar.style.strokeDashoffset = circumference;

    // Animate
    setTimeout(() => {
        const offset = circumference - (confidence / 100) * circumference;
        progressBar.style.strokeDashoffset = offset;

        // Animate number
        animateNumber(scoreText, 0, confidence, 1500);
    }, 100);

    // Color based on confidence level
    if (confidence >= 70) {
        progressBar.style.stroke = '#ff4444';
    } else if (confidence >= 50) {
        progressBar.style.stroke = '#ffa500';
    } else {
        progressBar.style.stroke = '#7FAF93';
    }

    // Animate badge
    if (mentalStateBadge) {
        setTimeout(() => {
            mentalStateBadge.style.opacity = '1';
            mentalStateBadge.style.transform = 'scale(1)';
        }, 800);
    }
}

function animateIndicators() {
    const indicators = document.querySelectorAll('.indicator-bar');

    indicators.forEach((bar, index) => {
        const width = bar.style.width;
        bar.style.width = '0%';

        setTimeout(() => {
            bar.style.width = width;
        }, 200 + index * 100);
    });
}

function animateProbabilities() {
    const probBars = document.querySelectorAll('.prob-bar');

    probBars.forEach((bar, index) => {
        const width = bar.style.width;
        bar.style.width = '0%';

        setTimeout(() => {
            bar.style.width = width;
        }, 300 + index * 50);
    });
}

function animateCards() {
    const cards = document.querySelectorAll('.indicator-card, .probabilities-card, .analyzed-text-card, .recommendations-card');

    cards.forEach((card, index) => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';

        setTimeout(() => {
            card.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, 400 + index * 100);
    });
}

function animateNumber(element, start, end, duration) {
    const range = end - start;
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Easing function (ease-out)
        const easeOut = 1 - Math.pow(1 - progress, 3);
        const current = Math.floor(start + range * easeOut);

        element.textContent = current + '%';

        if (progress < 1) {
            requestAnimationFrame(update);
        } else {
            element.textContent = end + '%';
        }
    }

    requestAnimationFrame(update);
}

// Add print-specific styling
window.addEventListener('beforeprint', function() {
    // Reset animations for printing
    const progressBar = document.getElementById('progressBar');
    if (progressBar) {
        progressBar.style.transition = 'none';
    }
});