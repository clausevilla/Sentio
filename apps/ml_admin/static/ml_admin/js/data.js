/**
 * ML Admin - Data Page
 */

let selectedFile = null;
let currentUploadId = null;

document.addEventListener('DOMContentLoaded', function() {
    initDragAndDrop();
    initLabelDistChart();
});

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
    
    zone.addEventListener('dragover', function(e) {
        e.preventDefault();
        zone.classList.add('dragover');
    });
    
    zone.addEventListener('dragleave', function(e) {
        e.preventDefault();
        zone.classList.remove('dragover');
    });
    
    zone.addEventListener('drop', function(e) {
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
    document.getElementById('filePreview').style.display = 'flex';
    document.getElementById('fileName').textContent = file.name;
    document.getElementById('uploadBtn').disabled = false;
}

// Clear File
function clearFile() {
    selectedFile = null;
    
    const input = document.getElementById('fileInput');
    if (input) input.value = '';
    
    const zone = document.getElementById('uploadZone');
    const preview = document.getElementById('filePreview');
    const btn = document.getElementById('uploadBtn');
    
    if (zone) zone.style.display = 'block';
    if (preview) preview.style.display = 'none';
    if (btn) btn.disabled = true;
}

// Upload File
async function uploadFile() {
    if (!selectedFile) return;
    
    const btn = document.getElementById('uploadBtn');
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
    
    const formData = new FormData();
    formData.append('csv_file', selectedFile);
    
    try {
        const response = await fetch(URLS.uploadCsv, {
            method: 'POST',
            headers: { 'X-CSRFToken': getCSRF() },
            body: formData,
        });
        
        const data = await response.json();
        
        if (data.success) {
            toast(data.message);
            closeModal('uploadModal');
            setTimeout(() => location.reload(), 1000);
        } else {
            toast(data.error || 'Upload failed', 'error');
            btn.disabled = false;
            btn.innerHTML = '<i class="fas fa-upload"></i> Upload';
        }
    } catch (error) {
        toast('Upload failed', 'error');
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-upload"></i> Upload';
    }
}

// Delete Dataset
function deleteDataset(id, name) {
    confirmAction(`Delete "${name}" and all its records?`, async function() {
        const { ok, data } = await apiCall(`/ml-admin/api/data/${id}/delete/`, {
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
        const response = await fetch(`/ml-admin/api/data/${currentUploadId}/records/?page=${page}`);
        const data = await response.json();
        
        if (!data.records || data.records.length === 0) {
            content.innerHTML = '<div class="empty-state small"><p>No records</p></div>';
            document.getElementById('recordsPagination').innerHTML = '';
            return;
        }
        
        // Build table
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
        
        // Pagination
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
        const response = await fetch(`/ml-admin/api/data/${id}/split/`);
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
            
            <hr style="margin: 1.5rem 0; border: none; border-top: 1px solid var(--gray-200);">
            
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
        const { ok, data } = await apiCall(`/ml-admin/api/data/${currentSplitUploadId}/split/update/`, {
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
