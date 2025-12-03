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
    'unlabeled': 'Unlabeled'
};

// TODO : Display names for pipeline types, to be matched with model later
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

// ================================
// Pipeline Selection
// ================================
// TODO : Update pipeline types to match model later
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

//TODO: CHECK PIPELINE NAMES FROM MODEL LATER
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
                } else if (data.status === 'completed') {
                    updateTaskStatus(taskId, 'success', 'Completed', `${formatNumber(data.row_count)} records`);
                    setTimeout(() => location.reload(), 2000);
                } else if (data.status === 'failed') {
                    updateTaskStatus(taskId, 'error', 'Processing failed');
                }
            }
        } catch (error) {
            console.error('Status polling error:', error);
        }
    }, 2000);
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
    console.log('uploadFile() called');
    console.log('selectedFile:', selectedFile);
    console.log('URLS:', URLS);

    if (!selectedFile) {
        toast('Please select a file first', 'error');
        return;
    }

    // Save file reference BEFORE clearing
    const fileToUpload = selectedFile;
    const fileName = selectedFile.name;
    const datasetType = document.getElementById('datasetType').value;

    // TODO UPDATE LATER IF NEEDED
    // Get selected pipeline
    const pipelineInput = document.querySelector('input[name="pipeline"]:checked');
    const pipelineType = pipelineInput ? pipelineInput.value : 'full';

    console.log('Uploading:', fileName, 'as', datasetType, 'with pipeline:', pipelineType);
    console.log('Upload URL:', URLS.uploadCsv);

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

        console.log('Making fetch request to:', URLS.uploadCsv);

        const response = await fetch(URLS.uploadCsv, {
            method: 'POST',
            headers: { 'X-CSRFToken': getCSRF() },
            body: formData,
        });

        console.log('Response status:', response.status);

        const data = await response.json();
        console.log('Response data:', data);

        if (data.success) {
            activeTasks[taskId].uploadId = data.upload_id;
            updateTaskStage(taskId, 'cleaning', 'Processing...', 30);
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
function deleteDataset(id, name) {
    confirmAction(`Delete "${name}" and all its records?`, async function () {
        const { ok, data } = await apiCall(`/management/api/data/${id}/delete/`, {
            method: 'POST'
        });

        if (ok && data.success) {
            toast('Deleted successfully');
            setTimeout(() => location.reload(), 1000);
        } else {
            toast(data.error || 'Delete failed', 'error');
        }
    });
}

// View Records Modal
function viewRecords(id, name) {
    currentUploadId = id;
    document.getElementById('recordsTitle').textContent = name;
    openModal('recordsModal');
    loadRecords(1);
}

// Load Records
async function loadRecords(page) {
    const content = document.getElementById('recordsContent');
    content.innerHTML = '<div class="loading"><i class="fas fa-spinner fa-spin"></i> Loading...</div>';

    try {
        const response = await fetch(`/management/api/data/${currentUploadId}/records/?page=${page}`);
        const data = await response.json();

        if (!data.records || data.records.length === 0) {
            content.innerHTML = '<div class="empty-state small"><p>No records</p></div>';
            document.getElementById('recordsPagination').innerHTML = '';
            return;
        }

        let html = `
            <table class="data-table">
                <thead>
                    <tr>
                        <th>ID</th>
                        <th>Text</th>
                        <th>Label</th>
                    </tr>
                </thead>
                <tbody>
        `;

        data.records.forEach(record => {
            html += `
                <tr>
                    <td>${record.id}</td>
                    <td class="text-cell">${escapeHtml(record.text)}</td>
                    <td><span class="badge">${record.label}</span></td>
                </tr>
            `;
        });

        html += '</tbody></table>';
        content.innerHTML = html;

        const pagination = document.getElementById('recordsPagination');
        if (data.pages > 1) {
            let pagHtml = '';
            if (page > 1) {
                pagHtml += `<button class="page-btn" onclick="loadRecords(${page - 1})"><i class="fas fa-chevron-left"></i></button>`;
            }
            pagHtml += `<span class="page-info">Page ${page} of ${data.pages}</span>`;
            if (page < data.pages) {
                pagHtml += `<button class="page-btn" onclick="loadRecords(${page + 1})"><i class="fas fa-chevron-right"></i></button>`;
            }
            pagination.innerHTML = pagHtml;
        } else {
            pagination.innerHTML = '';
        }

    } catch (error) {
        content.innerHTML = '<div class="empty-state small"><p>Error loading records</p></div>';
    }
}

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
                <div class="split-stat unlabeled">
                    <span class="split-count">${formatNumber(b.unlabeled)}</span>
                    <span class="split-label">Unlabeled</span>
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
                        <i class="fas fa-graduation-cap"></i> All → Training
                    </button>
                    <button class="btn btn-secondary btn-block" onclick="applySplit('all_test')">
                        <i class="fas fa-flask"></i> All → Test
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

    if (!confirm(confirmMsg)) return;

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