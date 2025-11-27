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