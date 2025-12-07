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

    const config = {...defaults, ...options};
    if (options.headers) {
        config.headers = {...defaults.headers, ...options.headers};
    }

    try {
        const response = await fetch(url, config);
        const data = await response.json();
        return {ok: response.ok, data};
    } catch (error) {
        return {ok: false, data: {error: error.message}};
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
            legend: {display: false}
        },
        scales: {
            y: {beginAtZero: true}
        }
    },
    bar: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
            legend: {display: false}
        },
        scales: {
            y: {beginAtZero: true}
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
        options: {...CHART_DEFAULTS.doughnut, ...options}
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
        options: {...CHART_DEFAULTS.line, ...options}
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
        options: {...CHART_DEFAULTS.bar, ...options}
    });
}


// ================================
// Custom Confirm Modal
// ================================

function showConfirm({
                         title,
                         message,
                         type = 'warning',
                         confirmText = 'Confirm',
                         cancelText = 'Cancel',
                         danger = false
                     }) {
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
        modal.onclick = (e) => {
            if (e.target === modal) closeIt(false);
        };
    });
}


// ================================
// NOTIFICATION SYSTEM
// ================================

const NOTIFICATION_KEY = 'ml_admin_notifications';
const NOTIFICATION_POLL_INTERVAL = 10000; // Every 10s

let notifications = [];
let notificationPollTimer = null;
let lastKnownStates = {jobs: {}, uploads: {}};

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
    } catch (e) {
    }
}

function initLastKnownStates() {
    try {
        const jobsDataEl = document.getElementById('jobsData');
        if (jobsDataEl) {
            JSON.parse(jobsDataEl.textContent).forEach(job => {
                lastKnownStates.jobs[job.id] = job.status;
            });
        }
    } catch (e) {
    }
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
                // Pass full data including running/pending counts
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

// Auto-update training page table and banner
function updateTrainingTable(jobs) {
    // Update stored jobsData for modal
    const jobsDataScript = document.getElementById('jobsData');
    if (jobsDataScript) {
        try {
            let storedJobs = JSON.parse(jobsDataScript.textContent);
            jobs.forEach(apiJob => {
                const stored = storedJobs.find(j => j.id === apiJob.id);
                if (stored) {
                    stored.status = apiJob.status;
                    stored.progress_log = apiJob.progress_log || stored.progress_log;
                    stored.completed_at = apiJob.completed_at;
                }
            });
            jobsDataScript.textContent = JSON.stringify(storedJobs);
        } catch (e) {
        }
    }

    // Update job rows in the training jobs table
    const tbody = document.querySelector('#trainingJobsTable tbody');
    if (!tbody) return;

    jobs.forEach(job => {
        const row = tbody.querySelector(`tr[data-job-id="${job.id}"]`);
        if (!row) return;

        // Update duration for running jobs (always, regardless of status change)
        if (job.status === 'RUNNING') {
            const cells = row.querySelectorAll('td');
            if (cells.length >= 5) {
                const durationCell = cells[4];
                const elapsed = formatDuration(job.started_at, new Date().toISOString());
                durationCell.innerHTML = `<span class="duration running"><i class="fas fa-clock"></i> ${elapsed}</span>`;
            }

            // Update progress bar for any job with epoch data (cells[5])
            if (cells.length >= 6 && job.total_epochs) {
                const progressCell = cells[5];
                const current = job.current_epoch || 0;
                const pct = Math.round((current / job.total_epochs) * 100);
                progressCell.innerHTML = `
                    <div class="epoch-progress">
                        <div class="epoch-bar">
                            <div class="epoch-fill" style="width: ${pct}%"></div>
                        </div>
                        <span class="epoch-text">${current}/${job.total_epochs}</span>
                    </div>
                `;
            }
        }

        const badge = row.querySelector('.badge');
        if (!badge) return;

        // Get current status from badge CSS class (more reliable than textContent)
        const statusClasses = ['pending', 'running', 'completed', 'failed', 'cancelled'];
        const currentStatus = statusClasses.find(s => badge.classList.contains(s));
        const newStatus = job.status.toLowerCase();

        // Skip if status hasn't changed
        if (currentStatus === newStatus) return;

        // Remove old status class and add new one
        statusClasses.forEach(s => badge.classList.remove(s));
        badge.classList.add(newStatus);

        // Update badge icon and text
        let icon = 'fa-clock';
        if (job.status === 'COMPLETED') icon = 'fa-check';
        else if (job.status === 'FAILED') icon = 'fa-times';
        else if (job.status === 'CANCELLED') icon = 'fa-ban';
        else if (job.status === 'RUNNING') icon = 'fa-spinner fa-spin';

        badge.innerHTML = `<i class="fas ${icon}"></i> ${job.status}`;

        // Update f1 cscore column if job completed successfully
        if (job.status === 'COMPLETED' && job.f1_score !== null) {
            const cells = row.querySelectorAll('td');
            // Accuracy is typically the 8th column (index 7)
            if (cells.length >= 8) {
                const accuracyCell = cells[7];
                accuracyCell.innerHTML = `<span class="accuracy-value">${(job.f1_score * 100).toFixed(2)}%</span>`;
            }
        }

        // Update accuracy column if job failed
        if (job.status === 'FAILED' || job.status === 'CANCELLED') {
            const cells = row.querySelectorAll('td');
            if (cells.length >= 8) {
                const accuracyCell = cells[7];
                accuracyCell.innerHTML = '<span class="text-danger">—</span>';
            }
        }

        // Remove cancel button if job is no longer running/pending
        if (job.status !== 'RUNNING' && job.status !== 'PENDING') {
            const actionsCell = row.querySelector('.actions-cell');
            if (actionsCell) {
                const cancelBtn = actionsCell.querySelector('.btn-icon.danger');
                if (cancelBtn) {
                    cancelBtn.remove();
                }
            }
        }

        // Update duration column
        if (job.completed_at) {
            const cells = row.querySelectorAll('td');
            // Duration is typically the 5th column (index 5)
            if (cells.length >= 5) {
                const durationCell = cells[4];
                const duration = formatDuration(new Date(job.started_at), new Date(job.completed_at));
                durationCell.innerHTML = `<span class="duration">${duration}</span>`;
            }
        }

        // Add highlight animation
        row.classList.add('status-updated');
        setTimeout(() => row.classList.remove('status-updated'), 2000);
    });
}

/**
 * Updates the "X training job(s) currently running" banner at the top of the training page.
 * Only shows the banner on the training page (where #trainingJobsTable exists).
 */
function updateTrainingBanner(runningCount, pendingCount) {
    // Only show banner on the training page
    const trainingTable = document.querySelector('#trainingJobsTable');
    if (!trainingTable) return;

    const existingBanner = document.querySelector('.alert.info.training-banner');
    const contentArea = document.querySelector('.content');

    // Calculate total active jobs
    const totalActive = (runningCount || 0) + (pendingCount || 0);

    if (totalActive > 0) {
        // Need to show or update the banner
        const bannerText = totalActive === 1
            ? `<strong>1</strong> training job currently ${runningCount > 0 ? 'running' : 'pending'}`
            : `<strong>${totalActive}</strong> training job(s) currently active`;

        if (existingBanner) {
            // Update existing banner
            existingBanner.querySelector('span').innerHTML = bannerText;
        } else if (contentArea) {
            // Create new banner at the top of content area
            const banner = document.createElement('div');
            banner.className = 'alert info training-banner';
            banner.innerHTML = `
                <i class="fas fa-spinner fa-spin"></i>
                <span>${bannerText}</span>
            `;
            contentArea.insertBefore(banner, contentArea.firstChild);
        }
    } else {
        // No active jobs - remove the banner if it exists
        if (existingBanner) {
            existingBanner.style.transition = 'opacity 0.3s';
            existingBanner.style.opacity = '0';
            setTimeout(() => existingBanner.remove(), 300);
        }
    }
}

/**
 * Formats duration between two dates in a human-readable format.
 */
function formatDuration(start, end) {
    const startDate = typeof start === 'string' ? new Date(start) : start;
    const endDate = typeof end === 'string' ? new Date(end) : end;
    const diffMs = endDate - startDate;
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMins / 60);

    if (diffHours > 0) {
        const remainingMins = diffMins % 60;
        if (remainingMins > 0) {
            return `${diffHours} hour${diffHours !== 1 ? 's' : ''}, ${remainingMins} minute${remainingMins !== 1 ? 's' : ''}`;
        }
        return `${diffHours} hour${diffHours !== 1 ? 's' : ''}`;
    }
    return `${diffMins} minute${diffMins !== 1 ? 's' : ''}`;
}