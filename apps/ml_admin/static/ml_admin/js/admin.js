/* Author: Lian Shi*/
/* Disclaimer: LLM has used to help with implement chart display functions */

/**
 * ML Admin - Common Utilities JS
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
document.addEventListener('click', function (e) {
    if (e.target.classList.contains('modal') && e.target.classList.contains('open')) {
        e.target.classList.remove('open');
        document.body.style.overflow = '';
    }
});

// Close modal on Escape key
document.addEventListener('keydown', function (e) {
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


// ================================
// Custom Confirm Modal
// ================================

function showConfirm({ title, message, type = 'warning', confirmText = 'Confirm', cancelText = 'Cancel', danger = false }) {
    return new Promise((resolve) => {
        // Remove existing
        const existing = document.getElementById('customConfirmModal');
        if (existing) existing.remove();

        const icons = {
            warning: 'fa-exclamation-triangle',
            danger: 'fa-trash',
            info: 'fa-info-circle',
            success: 'fa-check-circle'
        };

        const modal = document.createElement('div');
        modal.id = 'customConfirmModal';
        modal.className = 'confirm-modal';
        modal.innerHTML = `
            <div class="confirm-box">
                <div class="confirm-header ${type}">
                    <i class="fas ${icons[type] || icons.warning}"></i>
                    <h4>${title}</h4>
                </div>
                <div class="confirm-body">${message}</div>
                <div class="confirm-footer">
                    <button class="btn btn-cancel">${cancelText}</button>
                    <button class="btn btn-confirm ${danger ? 'danger' : ''}">${confirmText}</button>
                </div>
            </div>
        `;

        document.body.appendChild(modal);
        requestAnimationFrame(() => modal.classList.add('open'));

        const closeIt = (result) => {
            modal.classList.remove('open');
            setTimeout(() => modal.remove(), 200);
            resolve(result);
        };

        modal.querySelector('.btn-cancel').onclick = () => closeIt(false);
        modal.querySelector('.btn-confirm').onclick = () => closeIt(true);
        modal.onclick = (e) => { if (e.target === modal) closeIt(false); };
    });
}


// ================================
// NOTIFICATION SYSTEM
// ================================

const NOTIFICATION_KEY = 'ml_admin_notifications';
const NOTIFICATION_POLL_INTERVAL = 3000; // 3 seconds

let notifications = [];
let notificationPollTimer = null;
let lastKnownStates = { jobs: {}, uploads: {} };

// Initialize notifications on page load
document.addEventListener('DOMContentLoaded', function () {
    initNotifications();
});

function initNotifications() {
    loadNotifications();
    renderNotifications();
    updateNotificationBadge();
    initLastKnownStates();
    startNotificationPolling();

    // Close dropdown when clicking outside
    document.addEventListener('click', function (e) {
        const dropdown = document.getElementById('notificationDropdown');
        const bell = document.querySelector('.notification-bell');
        if (dropdown && bell && !dropdown.contains(e.target) && !bell.contains(e.target)) {
            dropdown.classList.remove('open');
        }
    });
}

// LocalStorage
function loadNotifications() {
    try {
        const stored = localStorage.getItem(NOTIFICATION_KEY);
        if (stored) {
            notifications = JSON.parse(stored);
            const dayAgo = Date.now() - (24 * 60 * 60 * 1000);
            notifications = notifications.filter(n => n.timestamp > dayAgo);
            saveNotifications();
        }
    } catch (e) {
        notifications = [];
    }
}

function saveNotifications() {
    try {
        localStorage.setItem(NOTIFICATION_KEY, JSON.stringify(notifications));
    } catch (e) { }
}

function initLastKnownStates() {
    try {
        const jobsDataEl = document.getElementById('jobsData');
        if (jobsDataEl) {
            JSON.parse(jobsDataEl.textContent).forEach(job => {
                lastKnownStates.jobs[job.id] = job.status;
            });
        }
    } catch (e) { }
}

// Add notification
function addNotification(notification) {
    const id = `${notification.type}_${notification.entityId}_${notification.status}_${Date.now()}`;

    // Prevent duplicates within 5 seconds
    const recentDupe = notifications.find(n =>
        n.type === notification.type &&
        n.entityId === notification.entityId &&
        n.status === notification.status &&
        (Date.now() - n.timestamp) < 5000
    );
    if (recentDupe) return;

    notifications.unshift({
        id, ...notification,
        timestamp: Date.now(),
        read: false
    });

    if (notifications.length > 50) notifications = notifications.slice(0, 50);

    saveNotifications();
    renderNotifications();
    updateNotificationBadge();

    // Use existing toast
    toast(`${notification.title}: ${notification.message}`,
        notification.status === 'completed' ? 'success' : 'error');
}

// Dropdown UI
function toggleNotificationDropdown() {
    const dropdown = document.getElementById('notificationDropdown');
    if (dropdown) {
        dropdown.classList.toggle('open');
        if (dropdown.classList.contains('open')) markNotificationsAsRead();
    }
}

function renderNotifications() {
    const list = document.getElementById('notificationList');
    if (!list) return;

    if (notifications.length === 0) {
        list.innerHTML = `
            <div class="notification-empty">
                <i class="fas fa-bell-slash"></i>
                <p>No notifications</p>
            </div>
        `;
        return;
    }

    list.innerHTML = notifications.slice(0, 20).map(n => {
        const icon = n.type === 'training' ? 'fa-flask' : 'fa-database';
        const statusIcon = n.status === 'completed' ? 'fa-check-circle' :
            n.status === 'failed' ? 'fa-times-circle' : 'fa-clock';
        return `
            <div class="notification-item ${n.read ? 'read' : 'unread'}"
                 onclick="handleNotificationClick('${n.type}')">
                <div class="notification-icon ${n.status}">
                    <i class="fas ${icon}"></i>
                </div>
                <div class="notification-content">
                    <div class="notification-title">${n.title}</div>
                    <div class="notification-message">${n.message}</div>
                    <div class="notification-time">
                        <i class="fas ${statusIcon} ${n.status}"></i>
                        ${formatNotificationTime(n.timestamp)}
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

function updateNotificationBadge() {
    const badge = document.getElementById('notificationBadge');
    if (!badge) return;

    const unreadCount = notifications.filter(n => !n.read).length;
    badge.textContent = unreadCount > 9 ? '9+' : unreadCount;
    badge.style.display = unreadCount > 0 ? 'flex' : 'none';
}

function markNotificationsAsRead() {
    notifications.forEach(n => n.read = true);
    saveNotifications();
    updateNotificationBadge();
    renderNotifications();
}

function clearAllNotifications() {
    notifications = [];
    saveNotifications();
    renderNotifications();
    updateNotificationBadge();
}

function handleNotificationClick(type) {
    window.location.href = type === 'training' ? '/management/training/' : '/management/data/';
}

function formatNotificationTime(timestamp) {
    const seconds = Math.floor((Date.now() - timestamp) / 1000);
    if (seconds < 60) return 'Just now';
    if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
    if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
    return `${Math.floor(seconds / 86400)}d ago`;
}

// Polling
function startNotificationPolling() {
    setTimeout(checkNotificationUpdates, 2000);
    notificationPollTimer = setInterval(checkNotificationUpdates, NOTIFICATION_POLL_INTERVAL);
}

async function checkNotificationUpdates() {
    try {
        // Check training jobs
        const jobsRes = await fetch('/management/api/notifications/jobs/');
        if (jobsRes.ok) {
            const jobsData = await jobsRes.json();
            if (jobsData.success && jobsData.jobs) {
                jobsData.jobs.forEach(job => {
                    const prev = lastKnownStates.jobs[job.id];
                    if (prev && prev !== job.status) {
                        if (job.status === 'COMPLETED') {
                            addNotification({
                                type: 'training', entityId: job.id,
                                title: 'Training Completed',
                                message: `${job.model_type || 'Model'} training finished`,
                                status: 'completed'
                            });
                        } else if (job.status === 'FAILED') {
                            addNotification({
                                type: 'training', entityId: job.id,
                                title: 'Training Failed',
                                message: `Job #${job.id} encountered an error`,
                                status: 'failed'
                            });
                        }
                    }
                    lastKnownStates.jobs[job.id] = job.status;
                });
                updateTrainingTable(jobsData.jobs);
            }
        }

        // Check data uploads
        const uploadsRes = await fetch('/management/api/notifications/uploads/');
        if (uploadsRes.ok) {
            const uploadsData = await uploadsRes.json();
            if (uploadsData.success && uploadsData.uploads) {
                uploadsData.uploads.forEach(upload => {
                    const prev = lastKnownStates.uploads[upload.id];
                    if (prev && prev !== upload.status) {
                        if (upload.status === 'completed') {
                            addNotification({
                                type: 'upload', entityId: upload.id,
                                title: 'Data Processing Complete',
                                message: `${upload.file_name} - ${formatNumber(upload.row_count || 0)} records`,
                                status: 'completed'
                            });
                        } else if (upload.status === 'failed') {
                            addNotification({
                                type: 'upload', entityId: upload.id,
                                title: 'Data Processing Failed',
                                message: `${upload.file_name} encountered an error`,
                                status: 'failed'
                            });
                        }
                    }
                    lastKnownStates.uploads[upload.id] = upload.status;
                });
                updateDatasetRows(uploadsData.uploads);

            }
        }
    } catch (e) {
        // Silently fail
    }
}

// Auto-update data page dataset rows
function updateDatasetRows(uploads) {
    uploads.forEach(upload => {
        const row = document.querySelector(`.dataset-row[data-upload-id="${upload.id}"]`);
        if (!row) return;

        const statusDiv = row.querySelector('.dataset-status');
        if (!statusDiv) return;

        const currentBadge = statusDiv.querySelector('.badge');
        if (!currentBadge) return;

        // Determine new status
        let newClass = '';
        let newContent = '';

        if (upload.status === 'completed') {
            newClass = 'badge success';
            newContent = '<i class="fas fa-check"></i> Validated';
        } else if (upload.status === 'processing') {
            newClass = 'badge running';
            newContent = '<i class="fas fa-spinner fa-spin"></i> Processing';
        } else if (upload.status === 'failed') {
            newClass = 'badge failed';
            newContent = '<i class="fas fa-times"></i> Failed';
        } else {
            newClass = 'badge pending';
            newContent = '<i class="fas fa-clock"></i> Pending';
        }

        // Skip if no change
        if (currentBadge.className === newClass) return;

        // Update badge
        currentBadge.className = newClass;
        currentBadge.innerHTML = newContent;

        // Update row count if available
        if (upload.row_count) {
            const metaSpans = row.querySelectorAll('.dataset-meta span');
            metaSpans.forEach(span => {
                if (span.innerHTML.includes('fa-table')) {
                    span.innerHTML = `<i class="fas fa-table"></i> ${formatNumber(upload.row_count)} rows`;
                }
            });
        }

        // Flash animation
        row.classList.add('status-updated');
        setTimeout(() => row.classList.remove('status-updated'), 2000);

        // Remove pending class if completed
        if (upload.status === 'completed') {
            row.classList.remove('pending');
        }
    });
}

// Auto-update training page table
function updateTrainingTable(jobs) {
    const tbody = document.querySelector('.jobs-table tbody');
    if (!tbody) return;

    jobs.forEach(job => {
        const row = tbody.querySelector(`tr[data-job-id="${job.id}"]`);
        if (!row) return;

        const badge = row.querySelector('.badge');
        if (!badge || badge.textContent.trim() === job.status) return;

        badge.className = `badge ${job.status.toLowerCase()}`;
        const icon = job.status === 'COMPLETED' ? 'fa-check-circle' :
            job.status === 'FAILED' ? 'fa-times-circle' :
                job.status === 'RUNNING' ? 'fa-spinner fa-spin' : 'fa-clock';
        badge.innerHTML = `<i class="fas ${icon}"></i> ${job.status}`;

        row.classList.add('status-updated');
        setTimeout(() => row.classList.remove('status-updated'), 2000);
    });
}