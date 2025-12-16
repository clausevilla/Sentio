/* Author: Lian Shi*/

/**
 * ML Admin - Data Page
 */

let selectedFile = null;
let currentUploadId = null;
let taskIdCounter = 0;
let activeTasks = {};
let statusPollingIntervals = {};
let distModalChart = null;

// Display names for dataset types
const DATASET_TYPE_LABELS = {
    'train': 'Training',
    'test': 'Test',
    'increment': 'Increment'
};

// Format number with commas
function formatNumber(num) {
    return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}
// Display names for pipeline types, to be matched with model later
const PIPELINE_LABELS = {
    'full': 'Full',
    'partial': 'Partial',
    'raw': 'Raw'
};

document.addEventListener('DOMContentLoaded', function () {
    initDragAndDrop();
    initLabelDistChart();
    resumePendingUploads();
});

// CSRF Token
function getCSRF() {
    const cookie = document.cookie.split(';').find(c => c.trim().startsWith('csrftoken='));
    return cookie ? cookie.split('=')[1] : null;
}

// ================================
// Pipeline Selection
// ================================
function selectPipeline(element, value) {
    // Update radio button
    const radio = element.querySelector('input[type="radio"]');
    if (radio) radio.checked = true;

    // Update visual selection
    document.querySelectorAll('.pipeline-option').forEach(opt => {
        opt.classList.remove('selected');
    });
    element.classList.add('selected');
}

// Label Distribution Chart
function initLabelDistChart() {
    if (typeof labelDistData === 'undefined' || !labelDistData || labelDistData.length === 0) return;

    createDoughnutChart(
        'labelDistChart',
        labelDistData.map(d => d.label || 'Unknown'),
        labelDistData.map(d => d.count)
    );
}

// Drag and Drop
function initDragAndDrop() {
    const zone = document.getElementById('uploadZone');
    if (!zone) return;

    zone.addEventListener('dragover', function (e) {
        e.preventDefault();
        zone.classList.add('dragover');
    });

    zone.addEventListener('dragleave', function (e) {
        e.preventDefault();
        zone.classList.remove('dragover');
    });

    zone.addEventListener('drop', function (e) {
        e.preventDefault();
        zone.classList.remove('dragover');

        const files = e.dataTransfer.files;
        if (files.length > 0) {
            const input = document.getElementById('fileInput');
            input.files = files;
            fileSelected(input);
        }
    });
}

// File Selection
function fileSelected(input) {
    if (!input.files || !input.files[0]) return;

    const file = input.files[0];

    if (!file.name.toLowerCase().endsWith('.csv')) {
        toast('Please select a CSV file', 'error');
        clearFile();
        return;
    }

    selectedFile = file;

    document.getElementById('uploadZone').style.display = 'none';
    document.getElementById('fileSelected').style.display = 'flex';
    document.getElementById('fileName').textContent = file.name;
    document.getElementById('uploadBtn').disabled = false;
}

// Clear File
function clearFile() {
    selectedFile = null;

    const input = document.getElementById('fileInput');
    if (input) input.value = '';

    const zone = document.getElementById('uploadZone');
    const selected = document.getElementById('fileSelected');
    const btn = document.getElementById('uploadBtn');

    if (zone) zone.style.display = 'flex';
    if (selected) selected.style.display = 'none';
    if (btn) btn.disabled = true;
}

// ================================
// Task Management
// ================================

function showTasksPanel() {
    document.getElementById('uploadTasks').style.display = 'block';
}

function hideTasksPanel() {
    if (Object.keys(activeTasks).length === 0) {
        document.getElementById('uploadTasks').style.display = 'none';
    }
}

function addTask(fileName, datasetType, uploadId = null, pipelineType = 'full') {
    const taskId = ++taskIdCounter;

    // Get display labels
    const displayLabel = DATASET_TYPE_LABELS[datasetType] || datasetType;
    const pipelineLabel = PIPELINE_LABELS[pipelineType] || 'Full';

    activeTasks[taskId] = {
        id: taskId,
        fileName: fileName,
        datasetType: datasetType,
        pipelineType: pipelineType,
        uploadId: uploadId,
        status: 'processing',
        stage: 'uploading'
    };

    showTasksPanel();

    const taskList = document.getElementById('taskList');
    const taskHtml = `
        <div class="task-item" id="task-${taskId}">
            <div class="task-icon processing">
                <i class="fas fa-spinner fa-spin"></i>
            </div>
            <div class="task-info">
                <div class="task-name">${escapeHtml(fileName)}</div>
                <div class="task-status processing">
                    <span class="task-status-text">Uploading...</span>
                    <span class="task-type-badge">${displayLabel}</span>
                    <span class="task-pipeline-badge ${pipelineType}">${pipelineLabel}</span>
                </div>
                <div class="task-stages">
                    <div class="stage-item active" data-stage="upload">
                        <span class="stage-dot"></span>
                        <span class="stage-name">Upload</span>
                    </div>
                    <div class="stage-item" data-stage="cleaning">
                        <span class="stage-dot"></span>
                        <span class="stage-name">Cleaning and preprocessing</span>
                    </div>
                    <div class="stage-item" data-stage="complete">
                        <span class="stage-dot"></span>
                        <span class="stage-name">Complete</span>
                    </div>
                </div>
                <div class="task-progress">
                    <div class="task-progress-bar" style="width: 10%"></div>
                </div>
            </div>
        </div>
    `;

    taskList.insertAdjacentHTML('afterbegin', taskHtml);

    return taskId;
}

function updateTaskStage(taskId, stage, statusText = null, progress = null) {
    const task = activeTasks[taskId];
    if (!task) return;

    task.stage = stage;

    const taskEl = document.getElementById(`task-${taskId}`);
    if (!taskEl) return;

    const stages = ['upload', 'cleaning', 'complete'];
    const stageIndex = stages.indexOf(stage);

    taskEl.querySelectorAll('.stage-item').forEach((el, i) => {
        el.classList.remove('active', 'done');
        if (i < stageIndex) {
            el.classList.add('done');
        } else if (i === stageIndex) {
            el.classList.add('active');
        }
    });

    if (statusText) {
        const statusTextEl = taskEl.querySelector('.task-status-text');
        if (statusTextEl) statusTextEl.textContent = statusText;
    }

    if (progress !== null) {
        const progressBar = taskEl.querySelector('.task-progress-bar');
        if (progressBar) progressBar.style.width = `${progress}%`;
    }
}

function updateTaskStatus(taskId, status, message, details = null) {
    const task = activeTasks[taskId];
    if (!task) return;

    task.status = status;

    if (statusPollingIntervals[taskId]) {
        clearInterval(statusPollingIntervals[taskId]);
        delete statusPollingIntervals[taskId];
    }

    const taskEl = document.getElementById(`task-${taskId}`);
    if (!taskEl) return;

    const iconEl = taskEl.querySelector('.task-icon');
    const statusEl = taskEl.querySelector('.task-status');
    const statusTextEl = taskEl.querySelector('.task-status-text');
    const progressEl = taskEl.querySelector('.task-progress');
    const stagesEl = taskEl.querySelector('.task-stages');

    iconEl.className = `task-icon ${status}`;
    if (status === 'processing') {
        iconEl.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';
    } else if (status === 'success') {
        iconEl.innerHTML = '<i class="fas fa-check"></i>';
        taskEl.querySelectorAll('.stage-item').forEach(el => {
            el.classList.remove('active');
            el.classList.add('done');
        });
    } else if (status === 'error') {
        iconEl.innerHTML = '<i class="fas fa-times"></i>';
    }

    statusEl.className = `task-status ${status}`;
    statusTextEl.textContent = message;

    if (status === 'success' || status === 'error') {
        if (progressEl) progressEl.style.display = 'none';
        if (stagesEl) stagesEl.style.display = 'none';

        const actionsHtml = `
            <div class="task-actions">
                ${status === 'success' && details ? `<span class="task-detail">${details}</span>` : ''}
                <button class="task-dismiss" onclick="dismissTask(${taskId})">Dismiss</button>
            </div>
        `;
        const existingActions = taskEl.querySelector('.task-actions');
        if (existingActions) {
            existingActions.remove();
        }
        taskEl.querySelector('.task-info').insertAdjacentHTML('beforeend', actionsHtml);
    }
}

function dismissTask(taskId) {
    if (statusPollingIntervals[taskId]) {
        clearInterval(statusPollingIntervals[taskId]);
        delete statusPollingIntervals[taskId];
    }

    const taskEl = document.getElementById(`task-${taskId}`);
    if (taskEl) {
        taskEl.style.opacity = '0';
        taskEl.style.transform = 'translateX(20px)';
        setTimeout(() => {
            taskEl.remove();
            delete activeTasks[taskId];
            hideTasksPanel();
        }, 200);
    }
}

// Status Polling
function startStatusPolling(taskId, uploadId) {
    statusPollingIntervals[taskId] = setInterval(async () => {
        try {
            const response = await fetch(`/management/api/data/${uploadId}/status/`);
            const data = await response.json();

            if (data.success) {
                if (data.status === 'processing') {
                    updateTaskStage(taskId, 'cleaning', 'Processing data...', 40);
                    // Also update the dataset row status if it exists
                    updateDatasetRowStatus(uploadId, 'processing');
                } else if (data.status === 'completed') {
                    updateTaskStatus(taskId, 'success', 'Completed', `${formatNumber(data.row_count)} records`);
                    // Update dataset row in-place (no reload needed!)
                    updateDatasetRowComplete(uploadId, data);
                } else if (data.status === 'failed') {
                    updateTaskStatus(taskId, 'error', 'Processing failed');
                    // Update dataset row to show failed status
                    updateDatasetRowStatus(uploadId, 'failed');
                }
            }
        } catch (error) {
            console.error('Status polling error:', error);
        }
    }, 2000);
}

// ================================
// Dynamic Dataset Row Management
// ================================

/**
 * Creates a new dataset row in the list immediately after upload.
 * This allows the user to see the new dataset without page reload.
 */
function addDatasetRow(upload) {
    const datasetList = document.querySelector('.dataset-list');

    // If there's an empty state message, remove it and create the list container
    const emptyState = document.querySelector('.card-body.no-pad .empty-state');
    if (emptyState) {
        const cardBody = emptyState.parentElement;
        emptyState.remove();
        const newList = document.createElement('div');
        newList.className = 'dataset-list';
        cardBody.appendChild(newList);
    }

    const list = document.querySelector('.dataset-list');
    if (!list) return;

    // Determine pipeline badge HTML
    let pipelineBadgeHtml = '';
    if (upload.pipeline_type) {
        const pipelineIcons = {
            'raw': '<i class="fas fa-feather"></i> Raw',
            'partial': '<i class="fas fa-adjust"></i> Partial',
            'full': '<i class="fas fa-broom"></i> Full'
        };
        pipelineBadgeHtml = `
            <span class="pipeline-badge ${upload.pipeline_type}">
                ${pipelineIcons[upload.pipeline_type] || pipelineIcons['full']}
            </span>
        `;
    }

    // Create the new row HTML
    const rowHtml = `
        <div class="dataset-row pending" data-upload-id="${upload.id}">
            <!-- Main Info -->
            <div class="dataset-main">
                <div class="dataset-icon">
                    <i class="fas fa-file-csv"></i>
                </div>
                <div class="dataset-info">
                    <div class="dataset-name">${escapeHtml(upload.file_name)}</div>
                    <div class="dataset-meta">
                        <span><i class="fas fa-calendar"></i> ${upload.uploaded_at}</span>
                        <span><i class="fas fa-user"></i> ${escapeHtml(upload.uploaded_by || '—')}</span>
                        <span><i class="fas fa-table"></i> ${formatNumber(upload.row_count || 0)} rows</span>
                        ${pipelineBadgeHtml}
                    </div>
                </div>
            </div>

            <!-- Split Info -->
            <div class="dataset-split">
                <div class="split-badge training" title="Training records">
                    <i class="fas fa-graduation-cap"></i>
                    <span>${formatNumber(upload.training_count || 0)}</span>
                </div>
                <div class="split-badge test" title="Test records">
                    <i class="fas fa-flask"></i>
                    <span>${formatNumber(upload.test_count || 0)}</span>
                </div>
            </div>

            <!-- Status -->
            <div class="dataset-status">
                <span class="badge pending"><i class="fas fa-clock"></i> Pending</span>
            </div>

            <!-- Actions -->
            <div class="dataset-actions">
                <button class="btn btn-sm btn-secondary" onclick="viewRecords(${upload.id}, '${escapeHtml(upload.file_name)}')" title="View Records">
                    <i class="fas fa-eye"></i> View
                </button>
                <button class="btn btn-sm btn-secondary" onclick="viewDistribution(${upload.id}, '${escapeHtml(upload.file_name)}')" title="View Distribution">
                    <i class="fas fa-chart-pie"></i> Distribution
                </button>
                <button class="btn btn-sm btn-secondary" onclick="manageSplit(${upload.id}, '${escapeHtml(upload.file_name)}')" title="Manage Split">
                    <i class="fas fa-random"></i> Split
                </button>
                <button class="btn btn-sm btn-danger" onclick="deleteDataset(${upload.id}, '${escapeHtml(upload.file_name)}')" title="Delete">
                    <i class="fas fa-trash"></i>
                </button>
            </div>
        </div>
    `;

    // Insert at the top of the list
    list.insertAdjacentHTML('afterbegin', rowHtml);

    // Add highlight animation
    const newRow = list.querySelector(`[data-upload-id="${upload.id}"]`);
    if (newRow) {
        newRow.classList.add('status-updated');
        setTimeout(() => newRow.classList.remove('status-updated'), 2000);
    }

    // Update the total count in header
    updateDatasetCount(1);
}

/**
 * Updates the status badge of an existing dataset row.
 */
function updateDatasetRowStatus(uploadId, status) {
    const row = document.querySelector(`.dataset-row[data-upload-id="${uploadId}"]`);
    if (!row) return;

    const statusDiv = row.querySelector('.dataset-status');
    if (!statusDiv) return;

    const badge = statusDiv.querySelector('.badge');
    if (!badge) return;

    // Update based on status
    if (status === 'processing') {
        badge.className = 'badge running';
        badge.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing';
    } else if (status === 'failed') {
        badge.className = 'badge failed';
        badge.innerHTML = '<i class="fas fa-times"></i> Failed';
    } else if (status === 'pending') {
        badge.className = 'badge pending';
        badge.innerHTML = '<i class="fas fa-clock"></i> Pending';
    }
}

/**
 * Updates a dataset row when processing completes successfully.
 * Updates status, row count, and removes pending styling.
 */
function updateDatasetRowComplete(uploadId, data) {
    const row = document.querySelector(`.dataset-row[data-upload-id="${uploadId}"]`);
    if (!row) return;

    // Update status badge to validated/success
    const statusDiv = row.querySelector('.dataset-status');
    if (statusDiv) {
        const badge = statusDiv.querySelector('.badge');
        if (badge) {
            badge.className = 'badge success';
            badge.innerHTML = '<i class="fas fa-check"></i> Validated';
        }
    }

    // Update row count in metadata
    if (data.row_count) {
        const metaSpans = row.querySelectorAll('.dataset-meta span');
        metaSpans.forEach(span => {
            if (span.innerHTML.includes('fa-table')) {
                span.innerHTML = `<i class="fas fa-table"></i> ${formatNumber(data.row_count)} rows`;
            }
        });
    }

    // Remove pending class
    row.classList.remove('pending');

    // Add highlight animation
    row.classList.add('status-updated');
    setTimeout(() => row.classList.remove('status-updated'), 2000);

    // Fetch updated split counts and update them
    fetchAndUpdateSplitCounts(uploadId);
}

/**
 * Fetches the train/test split counts for an upload and updates the row.
 */
async function fetchAndUpdateSplitCounts(uploadId) {
    try {
        const response = await fetch(`/management/api/data/${uploadId}/split/`);
        const data = await response.json();

        if (data.success && data.breakdown) {
            const row = document.querySelector(`.dataset-row[data-upload-id="${uploadId}"]`);
            if (!row) return;

            // Update training count
            const trainBadge = row.querySelector('.split-badge.training span');
            if (trainBadge) {
                trainBadge.textContent = formatNumber(data.breakdown.training || 0);
            }

            // Update test count
            const testBadge = row.querySelector('.split-badge.test span');
            if (testBadge) {
                testBadge.textContent = formatNumber(data.breakdown.test || 0);
            }
        }
    } catch (error) {
        console.error('Error fetching split counts:', error);
    }
}

/**
 * Updates the total dataset count in the header.
 */
function updateDatasetCount(delta) {
    const countEl = document.querySelector('.card-header .header-count');
    if (countEl) {
        const match = countEl.textContent.match(/(\d+)/);
        if (match) {
            const newCount = parseInt(match[1]) + delta;
            countEl.textContent = `${newCount} total`;
        }
    }

    // Also update the stat card
    const statValue = document.querySelector('.stat-card .stat-value');
    if (statValue) {
        const currentCount = parseInt(statValue.textContent) || 0;
        statValue.textContent = currentCount + delta;
    }
}

function resumePendingUploads() {
    if (typeof pendingUploads === 'undefined' || pendingUploads.length === 0) {
        return;
    }

    pendingUploads.forEach(upload => {
        // Create task in UI
        const taskId = addTask(upload.file_name, upload.status);
        activeTasks[taskId].uploadId = upload.id;

        // Update stage based on current status
        if (upload.status === 'processing') {
            updateTaskStage(taskId, 'cleaning', 'Processing...', 50);
        } else {
            updateTaskStage(taskId, 'upload', 'Pending...', 20);
        }

        // Start polling for this upload
        startStatusPolling(taskId, upload.id);
    });
}

// Upload File
async function uploadFile() {

    if (!selectedFile) {
        toast('Please select a file first', 'error');
        return;
    }

    // Save file reference BEFORE clearing
    const fileToUpload = selectedFile;
    const fileName = selectedFile.name;
    const datasetType = document.getElementById('datasetType').value;


    // Get selected pipeline
    const pipelineInput = document.querySelector('input[name="pipeline"]:checked');
    const pipelineType = pipelineInput ? pipelineInput.value : 'full';

    // Create FormData with the file BEFORE clearing
    const formData = new FormData();
    formData.append('csv_file', fileToUpload);
    formData.append('dataset_type', datasetType);
    formData.append('pipeline_type', pipelineType);

    // Now we can clear the UI
    const taskId = addTask(fileName, datasetType, null, pipelineType);
    clearFile();

    try {
        updateTaskStage(taskId, 'upload', 'Uploading...', 10);

        const response = await fetch(URLS.uploadCsv, {
            method: 'POST',
            headers: { 'X-CSRFToken': getCSRF() },
            body: formData,
        });

        const data = await response.json();

        if (data.success) {
            activeTasks[taskId].uploadId = data.upload_id;
            updateTaskStage(taskId, 'cleaning', 'Processing...', 30);

            // Immediately add the new row to the dataset list (no reload needed!)
            if (data.upload) {
                addDatasetRow(data.upload);
            }

            startStatusPolling(taskId, data.upload_id);
        } else {
            updateTaskStatus(taskId, 'error', data.error || 'Upload failed');
        }
    } catch (error) {
        console.error('Upload error:', error);
        console.error('Error stack:', error.stack);
        updateTaskStatus(taskId, 'error', 'Upload failed: ' + error.message);
    }
}

// Delete Dataset
async function deleteDataset(id, name) {
    const confirmed = await showConfirm({
        title: 'Delete Dataset',
        message: `Are you sure you want to delete "<strong>${escapeHtml(name)}</strong>" and all its records? This action cannot be undone.`,
        type: 'danger',
        confirmText: 'Delete',
        cancelText: 'Cancel',
        danger: true
    });

    if (!confirmed) return;

    const { ok, data } = await apiCall(`/management/api/data/${id}/delete/`, {
        method: 'POST'
    });

    if (ok && data.success) {
        toast('Deleted successfully');
        removeDatasetRow(id);
    } else {
        toast(data.error || 'Delete failed', 'error');
    }
}

/**
 * Removes a dataset row from the list with a fade-out animation.
 * Also updates the count and shows empty state if needed.
 */
function removeDatasetRow(uploadId) {
    const row = document.querySelector(`.dataset-row[data-upload-id="${uploadId}"]`);
    if (!row) return;

    // Animate out
    row.style.transition = 'opacity 0.3s, transform 0.3s';
    row.style.opacity = '0';
    row.style.transform = 'translateX(-20px)';

    setTimeout(() => {
        row.remove();

        // Update the count
        updateDatasetCount(-1);

        // Check if list is now empty
        const list = document.querySelector('.dataset-list');
        if (list && list.children.length === 0) {
            // Replace with empty state
            const cardBody = list.parentElement;
            list.remove();
            cardBody.innerHTML = `
                <div class="empty-state">
                    <i class="fas fa-cloud-upload-alt"></i>
                    <h4>No Datasets Yet</h4>
                    <p>Upload your first CSV using the panel above</p>
                </div>
            `;
        }
    }, 300);
}

// View Records Modal


let currentSort = { column: 'id', direction: 'asc' };
let currentPerPage = 20;
let currentPage = 1;
let currentFilter = 'all';
let totalPages = 1;
let totalRecords = 0;
let isModalInitialized = false;
let currentRecordsData = [];
let currentViewIndex = 0;

// Available labels for filter
const LABELS = ['Normal', 'Depression', 'Stress', 'Suicidal'];

// View Records Modal
function viewRecords(id, name) {
    currentUploadId = id;
    currentSort = { column: 'id', direction: 'asc' };
    currentPage = 1;
    currentFilter = 'all';
    isModalInitialized = false;

    document.getElementById('recordsTitle').textContent = name;
    openModal('recordsModal');
    loadRecords(1);
}

// Initialize modal structure (only once)
function initRecordsModal() {
    const modalBody = document.querySelector('#recordsModal .modal-body');

    // Build filter options
    let filterOptions = '<option value="all">All Labels</option>';
    LABELS.forEach(label => {
        filterOptions += `<option value="${label}">${label}</option>`;
    });

    modalBody.innerHTML = `
        <!-- Toolbar -->
        <div class="records-toolbar">
            <div class="toolbar-left">
                <div class="records-pagination">
                    <button class="page-btn" id="btnFirst" title="First page">
                        <i class="fas fa-angle-double-left"></i>
                    </button>
                    <button class="page-btn" id="btnPrev" title="Previous page">
                        <i class="fas fa-angle-left"></i>
                    </button>
                    <span class="page-info" id="pageInfo">Page 1 of 1</span>
                    <button class="page-btn" id="btnNext" title="Next page">
                        <i class="fas fa-angle-right"></i>
                    </button>
                    <button class="page-btn" id="btnLast" title="Last page">
                        <i class="fas fa-angle-double-right"></i>
                    </button>
                </div>

                <div class="filter-select">
                    <label><i class="fas fa-filter"></i></label>
                    <select id="labelFilter">
                        ${filterOptions}
                    </select>
                </div>
            </div>

            <div class="toolbar-right">
                <div class="per-page-select">
                    <span>Show</span>
                    <select id="perPageSelect">
                        <option value="10">10</option>
                        <option value="20" selected>20</option>
                        <option value="50">50</option>
                    </select>
                </div>

                <div class="records-info">
                    <strong id="totalRecords">0</strong> records
                </div>
            </div>
        </div>

        <!-- Table -->
        <div class="records-table-wrap">
            <table class="data-table" id="recordsTable">
                <thead>
                    <tr>
                        <th class="col-id sortable" data-column="id">
                            ID <span class="sort-icon"></span>
                        </th>
                        <th class="col-text sortable" data-column="text">
                            Text <span class="sort-icon"></span>
                        </th>
                        <th class="col-label sortable" data-column="label">
                            Label <span class="sort-icon"></span>
                        </th>
                    </tr>
                </thead>
                <tbody id="recordsTableBody">
                    <tr>
                        <td colspan="3" class="records-empty-cell">
                            <i class="fas fa-spinner fa-spin"></i>
                            <p>Loading...</p>
                        </td>
                    </tr>
                </tbody>
            </table>
        </div>
    `;

    // Event listeners
    document.getElementById('btnFirst').addEventListener('click', () => loadRecords(1));
    document.getElementById('btnPrev').addEventListener('click', () => loadRecords(currentPage - 1));
    document.getElementById('btnNext').addEventListener('click', () => loadRecords(currentPage + 1));
    document.getElementById('btnLast').addEventListener('click', () => loadRecords(totalPages));

    document.getElementById('perPageSelect').addEventListener('change', (e) => {
        currentPerPage = parseInt(e.target.value);
        loadRecords(1);
    });

    document.getElementById('labelFilter').addEventListener('change', (e) => {
        currentFilter = e.target.value;
        loadRecords(1);
    });

    // Sortable headers
    document.querySelectorAll('#recordsTable th.sortable').forEach(th => {
        th.addEventListener('click', () => {
            const column = th.dataset.column;
            sortRecords(column);
        });
    });

    isModalInitialized = true;
}

// Load Records
async function loadRecords(page) {
    if (!isModalInitialized) {
        initRecordsModal();
    }

    currentPage = page;

    const tbody = document.getElementById('recordsTableBody');
    tbody.style.opacity = '0.5';
    tbody.style.pointerEvents = 'none';

    try {
        const params = new URLSearchParams({
            page: page,
            per_page: currentPerPage,
            sort_by: currentSort.column,
            sort_dir: currentSort.direction
        });

        // Add filter if not "all"
        if (currentFilter !== 'all') {
            params.append('label', currentFilter);
        }

        const response = await fetch(`/management/api/data/${currentUploadId}/records/?${params}`);
        const data = await response.json();

        if (!data.records || data.records.length === 0) {
            tbody.innerHTML = `
                <tr>
                    <td colspan="3" class="records-empty-cell">
                        <i class="fas fa-inbox"></i>
                        <p>No records found${currentFilter !== 'all' ? ` with label "${currentFilter}"` : ''}</p>
                    </td>
                </tr>
            `;
            tbody.style.opacity = '1';
            tbody.style.pointerEvents = 'auto';
            updatePagination(1, 1, 0);
            return;
        }

        totalPages = data.pages;
        totalRecords = data.total;
        currentRecordsData = data.records;

        // Render rows
        tbody.innerHTML = renderTableRows(data.records);
        tbody.style.opacity = '1';
        tbody.style.pointerEvents = 'auto';

        // Update UI
        updatePagination(page, data.pages, data.total);
        updateSortIndicators();

    } catch (error) {
        console.error('Error loading records:', error);
        tbody.innerHTML = `
            <tr>
                <td colspan="3" class="records-empty-cell">
                    <i class="fas fa-exclamation-triangle"></i>
                    <p>Error loading records</p>
                </td>
            </tr>
        `;
        tbody.style.opacity = '1';
        tbody.style.pointerEvents = 'auto';
    }
}

// Render table rows
function renderTableRows(records) {
    return records.map((record, index) => `
        <tr onclick="viewFullText(${index})" title="Click to view full text">
            <td class="col-id">${record.id}</td>
            <td class="col-text">
                <div class="text-cell">${escapeHtml(record.text)}</div>
            </td>
            <td class="col-label">
                <span class="label-badge ${record.label.toLowerCase()}">${record.label}</span>
            </td>
        </tr>
    `).join('');
}

// Update pagination
function updatePagination(page, pages, total) {
    document.getElementById('pageInfo').textContent = `Page ${page} of ${pages}`;
    document.getElementById('totalRecords').textContent = formatNumber(total);

    document.getElementById('btnFirst').disabled = page <= 1;
    document.getElementById('btnPrev').disabled = page <= 1;
    document.getElementById('btnNext').disabled = page >= pages;
    document.getElementById('btnLast').disabled = page >= pages;
}

// Update sort indicators
function updateSortIndicators() {
    document.querySelectorAll('#recordsTable th.sortable').forEach(th => {
        th.classList.remove('sort-asc', 'sort-desc');
        if (th.dataset.column === currentSort.column) {
            th.classList.add(currentSort.direction === 'asc' ? 'sort-asc' : 'sort-desc');
        }
    });
}

// Sort records
function sortRecords(column) {
    if (currentSort.column === column) {
        currentSort.direction = currentSort.direction === 'asc' ? 'desc' : 'asc';
    } else {
        currentSort.column = column;
        currentSort.direction = 'asc';
    }
    loadRecords(1);
}

// Full Text View Modal


function viewFullText(index) {
    currentViewIndex = index;
    const record = currentRecordsData[index];
    if (!record) return;

    // Create or get the text view modal
    let modal = document.getElementById('textViewModal');
    if (!modal) {
        modal = document.createElement('div');
        modal.id = 'textViewModal';
        modal.className = 'modal text-view-modal';
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h3><i class="fas fa-file-alt"></i> Full Text</h3>
                    <button class="modal-close" onclick="closeModal('textViewModal')">&times;</button>
                </div>
                <div class="modal-body">
                    <div class="text-view-header">
                        <span class="text-view-id" id="textViewId">ID: 0</span>
                        <span class="text-view-label" id="textViewLabel"></span>
                    </div>
                    <div class="text-view-content" id="textViewContent"></div>
                    <div class="text-view-footer">
                        <div class="text-view-nav">
                            <button class="btn btn-secondary" id="btnPrevText" onclick="navigateText(-1)">
                                <i class="fas fa-chevron-left"></i> Previous
                            </button>
                            <button class="btn btn-secondary" id="btnNextText" onclick="navigateText(1)">
                                Next <i class="fas fa-chevron-right"></i>
                            </button>
                        </div>
                        <span class="text-view-count" id="textViewCount"></span>
                    </div>
                </div>
            </div>
        `;
        document.body.appendChild(modal);

        // Close on outside click
        modal.addEventListener('click', (e) => {
            if (e.target === modal) closeModal('textViewModal');
        });
    }

    updateTextViewModal(record, index);
    modal.classList.add('open');
}

function updateTextViewModal(record, index) {
    document.getElementById('textViewId').textContent = `ID: ${record.id}`;
    document.getElementById('textViewLabel').innerHTML = `<span class="label-badge ${record.label.toLowerCase()}">${record.label}</span>`;
    document.getElementById('textViewContent').textContent = record.text;
    document.getElementById('textViewCount').textContent = `${index + 1} of ${currentRecordsData.length} on this page`;

    // Update nav buttons
    document.getElementById('btnPrevText').disabled = index <= 0;
    document.getElementById('btnNextText').disabled = index >= currentRecordsData.length - 1;
}

function navigateText(direction) {
    const newIndex = currentViewIndex + direction;
    if (newIndex >= 0 && newIndex < currentRecordsData.length) {
        currentViewIndex = newIndex;
        updateTextViewModal(currentRecordsData[newIndex], newIndex);
    }
}

// Keyboard navigation for text view
document.addEventListener('keydown', (e) => {
    const textModal = document.getElementById('textViewModal');
    if (textModal && textModal.classList.contains('open')) {
        if (e.key === 'ArrowLeft') {
            navigateText(-1);
        } else if (e.key === 'ArrowRight') {
            navigateText(1);
        } else if (e.key === 'Escape') {
            closeModal('textViewModal');
        }
    }
});


// View Distribution Modal
async function viewDistribution(id, name) {
    document.getElementById('distModalTitle').textContent = `${name} - Label Distribution`;
    openModal('distributionModal');

    const content = document.getElementById('distModalContent');
    content.innerHTML = '<div class="loading"><i class="fas fa-spinner fa-spin"></i> Loading...</div>';

    try {
        const response = await fetch(`/management/api/data/${id}/distribution/`);
        const data = await response.json();

        if (!data.success || !data.distribution || data.distribution.length === 0) {
            content.innerHTML = '<div class="empty-state small"><p>No distribution data</p></div>';
            return;
        }

        const total = data.total || 1;

        let barsHtml = '';
        data.distribution.forEach((item, i) => {
            const percent = ((item.count / total) * 100).toFixed(1);
            barsHtml += `
                <div class="dist-item">
                    <span class="dist-label">${item.label}</span>
                    <div class="dist-bar">
                        <div class="dist-fill" style="width: ${percent}%; background: ${CHART_COLORS[i % CHART_COLORS.length]}"></div>
                    </div>
                    <span class="dist-count">${formatNumber(item.count)} (${percent}%)</span>
                </div>
            `;
        });

        content.innerHTML = `
            <div class="dist-modal-chart"><canvas id="distModalChart"></canvas></div>
            <div class="dist-modal-bars">${barsHtml}</div>
            <p class="hint text-center" style="margin-top: 1rem;"><strong>${formatNumber(total)}</strong> total records</p>
        `;

        // Create chart
        if (distModalChart) {
            distModalChart.destroy();
        }

        const ctx = document.getElementById('distModalChart');
        if (ctx) {
            distModalChart = new Chart(ctx, {
                type: 'doughnut',
                data: {
                    labels: data.distribution.map(d => d.label),
                    datasets: [{
                        data: data.distribution.map(d => d.count),
                        backgroundColor: CHART_COLORS.slice(0, data.distribution.length),
                        borderWidth: 0,
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    cutout: '60%',
                }
            });
        }

    } catch (error) {
        content.innerHTML = '<div class="empty-state small"><p>Error loading distribution</p></div>';
    }
}

// Escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ================================
// Split Management
// ================================

let currentSplitUploadId = null;

function manageSplit(id, name) {
    currentSplitUploadId = id;
    openModal('splitModal');
    loadSplitInfo(id, name);
}

async function loadSplitInfo(id, name) {
    const content = document.getElementById('splitContent');
    content.innerHTML = '<div class="loading"><i class="fas fa-spinner fa-spin"></i> Loading...</div>';

    try {
        const response = await fetch(`/management/api/data/${id}/split/`);
        const data = await response.json();

        if (!data.success) {
            content.innerHTML = `<div class="empty-state small"><p>Error: ${data.error}</p></div>`;
            return;
        }

        const b = data.breakdown;
        const total = b.total || 1;
        const testPercent = Math.round((b.test / total) * 100);

        content.innerHTML = `
            <h4 class="split-title">${escapeHtml(name)}</h4>

            <div class="split-current">
                <div class="split-stat training">
                    <span class="split-count">${formatNumber(b.training)}</span>
                    <span class="split-label">Training</span>
                </div>
                <div class="split-stat test">
                    <span class="split-count">${formatNumber(b.test)}</span>
                    <span class="split-label">Test</span>
                </div>
                <div class="split-stat increment">
                    <span class="split-count">${formatNumber(b.increment)}</span>
                    <span class="split-label">Increment</span>
                </div>
            </div>

            <div class="split-bar">
                <div class="split-bar-train" style="width: ${100 - testPercent}%"></div>
                <div class="split-bar-test" style="width: ${testPercent}%"></div>
            </div>
            <div class="split-bar-labels">
                <span>Training ${100 - testPercent}%</span>
                <span>Test ${testPercent}%</span>
            </div>

            <div class="split-actions">
                <div class="split-slider-group">
                    <label>Random Split:</label>
                    <div class="split-slider-row">
                        <input type="range" id="splitSlider" min="5" max="50" value="20"
                               oninput="document.getElementById('splitValue').textContent = this.value + '%'">
                        <span id="splitValue">20%</span> test
                    </div>
                    <button class="btn btn-primary btn-block" onclick="applySplit('split')">
                        <i class="fas fa-random"></i> Apply Random Split
                    </button>
                </div>

                <div class="split-quick-actions">
                    <label>Quick Actions:</label>
                    <button class="btn btn-secondary btn-block" onclick="applySplit('all_training')">
                        <i class="fas fa-graduation-cap"></i> All -> Training
                    </button>
                    <button class="btn btn-secondary btn-block" onclick="applySplit('all_test')">
                        <i class="fas fa-flask"></i> All -> Test
                    </button>
                </div>
            </div>

            <div class="split-note">
                <i class="fas fa-info-circle"></i>
                Random split is stratified by label to maintain class distribution.
            </div>
        `;
    } catch (error) {
        content.innerHTML = `<div class="empty-state small"><p>Error loading split info</p></div>`;
    }
}

async function applySplit(action) {
    if (!currentSplitUploadId) return;

    const testPercent = document.getElementById('splitSlider')?.value || 20;

    let confirmMsg = '';
    if (action === 'split') {
        confirmMsg = `Re-split records randomly with ${testPercent}% as test data?`;
    } else if (action === 'all_training') {
        confirmMsg = 'Move ALL records to Training set?';
    } else if (action === 'all_test') {
        confirmMsg = 'Move ALL records to Test set?';
    }

    const ok = await showConfirm({
        title: 'Update Split',
        message: confirmMsg,
        type: 'warning',
        confirmText: 'Confirm'
    });
    if (!ok) return;

    const content = document.getElementById('splitContent');
    content.innerHTML = '<div class="loading"><i class="fas fa-spinner fa-spin"></i> Updating...</div>';

    try {
        const { ok, data } = await apiCall(`/management/api/data/${currentSplitUploadId}/split/update/`, {
            method: 'POST',
            body: JSON.stringify({
                action: action,
                test_percent: parseInt(testPercent),
            })
        });

        if (ok && data.success) {
            toast(data.message);
            closeModal('splitModal');
            setTimeout(() => location.reload(), 1000);
        } else {
            toast(data.error || 'Update failed', 'error');
            loadSplitInfo(currentSplitUploadId, '');
        }
    } catch (error) {
        toast('Update failed', 'error');
        loadSplitInfo(currentSplitUploadId, '');
    }
}