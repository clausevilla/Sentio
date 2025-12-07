/* Author: Lian Shi*/
/* Disclaimer: LLM has been used to helo set up modelc comparison modal js */


/*--TODO : MODIFY TO REFELCT UPDATED DATABASE and methods IF NEEDED*/

/**
 * ML Admin - Models Page with Compare Feature
 */

// Track selected models for comparison
let selectedModels = [];
const MAX_COMPARE = 4;

// Color palette for comparison
const COMPARE_COLORS = ['#4A7C59', '#3B82F6', '#F59E0B', '#8B5CF6'];

document.addEventListener('DOMContentLoaded', function () {
    updateCompareButton();
});

// ================================
// Compare Feature
// ================================

function toggleCompare(modelId) {
    const index = selectedModels.indexOf(modelId);
    const row = document.querySelector(`tr[data-model-id="${modelId}"]`);
    const checkbox = row?.querySelector('input[type="checkbox"]');

    if (index > -1) {
        // Remove from selection
        selectedModels.splice(index, 1);
        row?.classList.remove('selected');
    } else {
        // Add to selection (max 4)
        if (selectedModels.length >= MAX_COMPARE) {
            toast(`Maximum ${MAX_COMPARE} models can be compared`, 'error');
            if (checkbox) checkbox.checked = false;
            return;
        }
        selectedModels.push(modelId);
        row?.classList.add('selected');
    }

    updateCompareButton();
}

function updateCompareButton() {
    const btn = document.getElementById('compareBtn');
    const countEl = document.getElementById('compareCount');

    if (countEl) countEl.textContent = selectedModels.length;
    if (btn) btn.disabled = selectedModels.length < 2;
}

function openCompareModal() {
    if (selectedModels.length < 2) {
        toast('Select at least 2 models to compare', 'error');
        return;
    }

    // Get selected model data
    const models = selectedModels.map(id => modelsById[id]).filter(m => m);
    if (models.length < 2) {
        toast('Could not load model data', 'error');
        return;
    }

    // Build comparison content
    const content = buildCompareContent(models);
    document.getElementById('compareContent').innerHTML = content;
    openModal('compareModal');
}

function buildCompareContent(models) {
    // Metrics to compare
    const metrics = [
        { key: 'accuracy', label: 'Accuracy', icon: 'fa-bullseye', format: v => v !== null ? (v * 100).toFixed(2) : '—', isPercent: true },
        { key: 'precision', label: 'Precision', icon: 'fa-crosshairs', format: v => v !== null ? (v * 100).toFixed(2) : '—', isPercent: false },
        { key: 'recall', label: 'Recall', icon: 'fa-redo', format: v => v !== null ? (v * 100).toFixed(2) : '—', isPercent: false },
        { key: 'f1', label: 'F1 Score', icon: 'fa-chart-line', format: v => v !== null ? (v * 100).toFixed(2) : '—', isPercent: false },
    ];

    // Find best for each metric
    const bestValues = {};
    metrics.forEach(metric => {
        const values = models.map(m => m[metric.key]).filter(v => v !== null);
        bestValues[metric.key] = values.length > 0 ? Math.max(...values) : null;
    });

    // Count leading metrics per model
    const leadingCount = {};
    models.forEach(m => leadingCount[m.id] = 0);

    metrics.forEach(metric => {
        models.forEach(m => {
            if (m[metric.key] !== null && m[metric.key] === bestValues[metric.key]) {
                leadingCount[m.id]++;
            }
        });
    });

    // Build header with model cards showing leading indicator
    let headerHtml = '<div class="compare-models-header">';
    models.forEach((model, i) => {
        const leading = leadingCount[model.id];
        const isTopPerformer = leading === Math.max(...Object.values(leadingCount)) && leading > 0;

        headerHtml += `
            <div class="compare-model-card ${model.is_active ? 'active' : ''} ${isTopPerformer ? 'top-performer' : ''}">
                <div class="model-card-color" style="background: ${COMPARE_COLORS[i]}"></div>
                <div class="model-card-content">
                    <div class="model-card-name">${model.name}</div>
                    <div class="model-card-meta">
                        <span class="model-card-type">${model.model_type_display}</span>
                        ${model.is_active ? '<span class="model-card-active">Deployed</span>' : ''}
                    </div>
                </div>
                <div class="model-card-indicator">
                    ${leading > 0 ? `
                        <span class="leading-badge ${isTopPerformer ? 'top' : ''}">
                            ${isTopPerformer ? '<i class="fas fa-crown"></i>' : ''}
                            ${leading}/${metrics.length}
                        </span>
                        <span class="leading-label">leading</span>
                    ` : '<span class="leading-label">—</span>'}
                </div>
            </div>
        `;
    });
    headerHtml += '</div>';

    // Build metrics comparison table
    let tableHtml = `
        <div class="compare-metrics-section">
            <h4><i class="fas fa-bullseye"></i> Performance Metrics</h4>
            <table class="compare-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        ${models.map((m, i) => `
                            <th>
                                <span class="th-color" style="background: ${COMPARE_COLORS[i]}"></span>
                                ${m.name}
                            </th>
                        `).join('')}
                    </tr>
                </thead>
                <tbody>
    `;

    metrics.forEach(metric => {
        tableHtml += `<tr><td class="metric-name"><i class="fas ${metric.icon}"></i> ${metric.label}</td>`;

        models.forEach((model, i) => {
            const value = model[metric.key];
            const isBest = value !== null && value === bestValues[metric.key];
            const barWidth = metric.isPercent ? (value || 0) : (value !== null ? (value * 100) : 0);

            tableHtml += `
                <td>
                    <div class="compare-cell ${isBest ? 'best' : ''}">
                        <span class="compare-value">${metric.format(value)}</span>
                        <div class="compare-bar">
                            <div class="compare-bar-fill" style="width: ${barWidth}%; background: ${COMPARE_COLORS[i]}"></div>
                        </div>
                        ${isBest && value !== null ? '<i class="fas fa-check-circle best-icon"></i>' : ''}
                    </div>
                </td>
            `;
        });

        tableHtml += '</tr>';
    });

    tableHtml += '</tbody></table></div>';

    // Build info comparison
    let infoHtml = `
        <div class="compare-info-section">
            <h4><i class="fas fa-info-circle"></i> Model Details</h4>
            <table class="compare-info-table">
                <thead>
                    <tr>
                        <th></th>
                        ${models.map((m, i) => `
                            <th>
                                <span class="th-color" style="background: ${COMPARE_COLORS[i]}"></span>
                                ${m.name}
                            </th>
                        `).join('')}
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td class="info-label">Type</td>
                        ${models.map(m => `<td><span class="type-badge type-${m.model_type}">${m.model_type_display}</span></td>`).join('')}
                    </tr>
                    <tr>
                        <td class="info-label">Created</td>
                        ${models.map(m => `<td>${m.created_at}</td>`).join('')}
                    </tr>
                    <tr>
                        <td class="info-label">Created By</td>
                        ${models.map(m => `<td>${m.created_by}</td>`).join('')}
                    </tr>
                    <tr>
                        <td class="info-label">Dataset</td>
                        ${models.map(m => `<td>${m.job_dataset}</td>`).join('')}
                    </tr>
                    <tr>
                        <td class="info-label">Status</td>
                        ${models.map(m => `<td>${m.is_active ? '<span class="badge success">Active</span>' : '<span class="badge secondary">Inactive</span>'}</td>`).join('')}
                    </tr>
                    <tr>
                        <td class="info-label">Actions</td>
                        ${models.map(m => `
                            <td>
                                <div class="compare-actions">
                                    ${m.is_active ? `
                                        <span class="deployed-tag"><i class="fas fa-check-circle"></i> Deployed</span>
                                    ` : `
                                        <button class="btn btn-primary btn-sm" onclick="deployModel(${m.id}, '${m.name}'); closeModal('compareModal');">
                                            <i class="fas fa-rocket"></i> Deploy
                                        </button>
                                        <button class="btn btn-danger btn-sm" onclick="deleteModel(${m.id}, '${m.name}'); closeModal('compareModal');">
                                            <i class="fas fa-trash"></i>
                                        </button>
                                    `}
                                </div>
                            </td>
                        `).join('')}
                    </tr>
                </tbody>
            </table>
        </div>
    `;

    return headerHtml + tableHtml + infoHtml;
}

// Clear all selections
function clearCompareSelection() {
    selectedModels = [];
    document.querySelectorAll('.models-table input[type="checkbox"]').forEach(cb => {
        cb.checked = false;
    });
    document.querySelectorAll('.models-table tr.selected').forEach(row => {
        row.classList.remove('selected');
    });
    updateCompareButton();
}

// ================================
// Model Actions
// ================================

// Deploy model
async function deployModel(id, name) {
    const confirmed = await showConfirm({
        title: 'Deploy Model',
        message: `Deploy <strong>${name}</strong> as the active model?`,
        type: 'info',
        confirmText: 'Deploy'
    });
    if (!confirmed) return;

    const { ok, data } = await apiCall(`/management/api/models/${id}/activate/`, {
        method: 'POST'
    });

    if (ok && data.success) {
        toast(data.message || 'Model deployed successfully');
        setTimeout(() => location.reload(), 1000);
    } else {
        toast(data.error || 'Deploy failed', 'error');
    }
}

// Delete model
async function deleteModel(id, name) {
    const confirmed = await showConfirm({
        title: 'Delete Model',
        message: `Delete <strong>${name}</strong>? This cannot be undone.`,
        type: 'danger',
        confirmText: 'Delete',
        danger: true
    });
    if (!confirmed) return;

    const { ok, data } = await apiCall(`/management/api/models/${id}/delete/`, {
        method: 'POST'
    });

    if (ok && data.success) {
        toast('Model deleted');
        setTimeout(() => location.reload(), 1000);
    } else {
        toast(data.error || 'Delete failed', 'error');
    }
}