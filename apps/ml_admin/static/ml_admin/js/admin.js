/**
 * ML Admin - Common Utilities
 */

// CSRF Token
function getCSRF() {
    const cookie = document.cookie.split(';').find(c => c.trim().startsWith('csrftoken='));
    return cookie ? cookie.split('=')[1] : null;
}

// Toast Notifications
function toast(message, type = 'success') {
    const container = document.getElementById('toasts');
    if (!container) return;
    
    const el = document.createElement('div');
    el.className = `toast ${type}`;
    el.innerHTML = `
        <i class="fas fa-${type === 'success' ? 'check' : 'exclamation'}-circle"></i>
        <span>${message}</span>
    `;
    container.appendChild(el);
    
    setTimeout(() => {
        el.classList.add('fade');
        setTimeout(() => el.remove(), 300);
    }, 3000);
}

// Modal Functions
function openModal(id) {
    const modal = document.getElementById(id);
    if (modal) {
        modal.classList.add('open');
        document.body.style.overflow = 'hidden';
    }
}

function closeModal(id) {
    const modal = document.getElementById(id);
    if (modal) {
        modal.classList.remove('open');
        document.body.style.overflow = '';
    }
}

// Close modal on overlay click
document.addEventListener('click', function(e) {
    if (e.target.classList.contains('modal') && e.target.classList.contains('open')) {
        e.target.classList.remove('open');
        document.body.style.overflow = '';
    }
});

// Close modal on Escape key
document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
        const openModals = document.querySelectorAll('.modal.open');
        openModals.forEach(modal => {
            modal.classList.remove('open');
        });
        document.body.style.overflow = '';
    }
});

// Sidebar Toggle (Mobile)
function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    const overlay = document.getElementById('sidebarOverlay');
    if (sidebar) sidebar.classList.toggle('open');
    if (overlay) overlay.classList.toggle('open');
}

function closeSidebar() {
    const sidebar = document.getElementById('sidebar');
    const overlay = document.getElementById('sidebarOverlay');
    if (sidebar) sidebar.classList.remove('open');
    if (overlay) overlay.classList.remove('open');
}

// API Helper
async function apiCall(url, options = {}) {
    const defaults = {
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': getCSRF(),
        },
    };
    
    const config = { ...defaults, ...options };
    if (options.headers) {
        config.headers = { ...defaults.headers, ...options.headers };
    }
    
    try {
        const response = await fetch(url, config);
        const data = await response.json();
        return { ok: response.ok, data };
    } catch (error) {
        return { ok: false, data: { error: error.message } };
    }
}

// Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Format number with commas
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

// Confirm action helper
function confirmAction(message, callback) {
    if (confirm(message)) {
        callback();
    }
}

// Chart colors
const CHART_COLORS = [
    '#4A7C59', // Primary green
    '#E07A5F', // Coral
    '#7B68A6', // Purple
    '#5B7C99', // Blue
    '#F4A261', // Orange
    '#A8C9B8', // Light green
];

// Chart default options
const CHART_DEFAULTS = {
    doughnut: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {
                position: 'bottom',
                labels: {
                    padding: 15,
                    usePointStyle: true,
                }
            }
        },
        cutout: '60%',
    },
    line: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { display: false }
        },
        scales: {
            y: { beginAtZero: true }
        }
    },
    bar: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: { display: false }
        },
        scales: {
            y: { beginAtZero: true }
        }
    }
};

// Create doughnut chart helper
function createDoughnutChart(canvasId, labels, data, options = {}) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    
    if (!data || data.length === 0) {
        ctx.parentElement.innerHTML = '<div class="empty-state small"><p>No data</p></div>';
        return null;
    }
    
    return new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: labels,
            datasets: [{
                data: data,
                backgroundColor: CHART_COLORS.slice(0, data.length),
                borderWidth: 0,
            }]
        },
        options: { ...CHART_DEFAULTS.doughnut, ...options }
    });
}

// Create line chart helper
function createLineChart(canvasId, labels, data, options = {}) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    
    if (!data || data.length === 0) {
        ctx.parentElement.innerHTML = '<div class="empty-state small"><p>No data</p></div>';
        return null;
    }
    
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: options.label || 'Value',
                data: data,
                borderColor: CHART_COLORS[0],
                backgroundColor: 'rgba(74, 124, 89, 0.1)',
                fill: true,
                tension: 0.3,
            }]
        },
        options: { ...CHART_DEFAULTS.line, ...options }
    });
}

// Create bar chart helper
function createBarChart(canvasId, labels, data, options = {}) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    
    if (!data || data.length === 0) {
        ctx.parentElement.innerHTML = '<div class="empty-state small"><p>No data</p></div>';
        return null;
    }
    
    return new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: options.label || 'Count',
                data: data,
                backgroundColor: CHART_COLORS[0],
            }]
        },
        options: { ...CHART_DEFAULTS.bar, ...options }
    });
}
