/* Author: Lian Shi */
/* Disclaimer: LLM has used to help with implement training page functionalities to fit our database */

// State
let currentTab = 'train';
let selectedDatasets = {};
let datasetDistributions = {};
let selectedModelId = null;
let selectedModelName = null;
let distChart = null;
let testSetChart = null;

document.addEventListener('DOMContentLoaded', function() {
    initAlgorithmListeners();
    initTestSetModal();
    updateSummary();
});

// ================================
// Tab Switching
// ================================

function switchTab(tab) {
    currentTab = tab;

    // Update tab buttons
    document.querySelectorAll('.col-tab:not(.static)').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.tab === tab);
    });

    // Toggle content
    document.getElementById('trainContent').style.display = (tab === 'train') ? 'flex' : 'none';
    document.getElementById('retrainContent').style.display = (tab === 'retrain') ? 'flex' : 'none';

    // Toggle summary
    document.getElementById('trainSummary').style.display = (tab === 'train') ? 'flex' : 'none';
    document.getElementById('retrainSummary').style.display = (tab === 'retrain') ? 'flex' : 'none';

    // Toggle buttons
    document.getElementById('trainBtn').style.display = (tab === 'train') ? 'inline-flex' : 'none';
    document.getElementById('retrainBtn').style.display = (tab === 'retrain') ? 'inline-flex' : 'none';

    updateSummary();
}

// ================================
// Algorithm Selection
// ================================

function initAlgorithmListeners() {
    document.querySelectorAll('input[name="algorithm"]').forEach(radio => {
        radio.addEventListener('change', updateSummary);
    });
}

// ================================
// Model Selection (Retrain)
// ================================

function selectModel(element, modelId) {
    // Update UI
    document.querySelectorAll('.model-option').forEach(opt => {
        opt.classList.remove('selected');
    });
    element.classList.add('selected');
    element.querySelector('input').checked = true;

    // Store selection
    selectedModelId = modelId;
    const nameEl = element.querySelector('.model-name');
    selectedModelName = nameEl ? nameEl.textContent.trim().split('\n')[0].trim() : 'Model #' + modelId;

    updateSummary();
}

// ================================
// Dataset Selection (Shared)
// ================================

function toggleDataset(element, id, count) {
    const checkbox = element.querySelector('input[type="checkbox"]');
    checkbox.checked = !checkbox.checked;
    element.classList.toggle('selected', checkbox.checked);

    // Get distribution data
    let distribution = [];
    try {
        distribution = JSON.parse(element.dataset.distribution || '[]');
    } catch (e) {
        distribution = [];
    }

    if (checkbox.checked) {
        selectedDatasets[id] = count;
        datasetDistributions[id] = distribution;
    } else {
        delete selectedDatasets[id];
        delete datasetDistributions[id];
    }

    updateDistribution();
    updateSummary();
}

// ================================
// Summary Update
// ================================

function updateSummary() {
    const datasetCount = Object.keys(selectedDatasets).length;
    let totalRecords = 0;
    Object.values(selectedDatasets).forEach(c => totalRecords += c);

    if (currentTab === 'train') {
        // Train summary
        const algoRadio = document.querySelector('input[name="algorithm"]:checked');
        const algoName = algoRadio ? ALGORITHM_NAMES[algoRadio.value] : '-';

        document.getElementById('sumAlgo').textContent = algoName;
        document.getElementById('sumDatasets').textContent = datasetCount;
        document.getElementById('sumRecords').textContent = formatNumber(totalRecords);

        // Enable/disable button
        document.getElementById('trainBtn').disabled = datasetCount === 0;
    } else {
        // Retrain summary
        document.getElementById('sumModel').textContent = selectedModelName || '-';
        document.getElementById('sumRetrainDatasets').textContent = datasetCount;
        document.getElementById('sumRetrainRecords').textContent = formatNumber(totalRecords);

        // Enable/disable button (need both model and datasets)
        document.getElementById('retrainBtn').disabled = !selectedModelId || datasetCount === 0;
    }
}

// ================================
// Distribution Preview
// ================================

function updateDistribution() {
    const distEmpty = document.getElementById('distEmpty');
    const distContent = document.getElementById('distContent');
    const distBars = document.getElementById('distBars');

    const selectedIds = Object.keys(selectedDatasets);

    if (selectedIds.length === 0) {
        distEmpty.style.display = 'flex';
        distContent.style.display = 'none';
        if (distChart) {
            distChart.destroy();
            distChart = null;
        }
        return;
    }

    // Combine distributions
    const combined = {};
    selectedIds.forEach(id => {
        const dist = datasetDistributions[id] || [];
        dist.forEach(item => {
            const label = item.label || 'Unknown';
            combined[label] = (combined[label] || 0) + item.count;
        });
    });

    const labels = Object.keys(combined).sort();
    const counts = labels.map(l => combined[l]);
    const total = counts.reduce((a, b) => a + b, 0);

    if (labels.length === 0 || total === 0) {
        distEmpty.innerHTML = '<i class="fas fa-exclamation-circle"></i><span>No distribution data</span>';
        distEmpty.style.display = 'flex';
        distContent.style.display = 'none';
        return;
    }

    // Show content
    distEmpty.style.display = 'none';
    distContent.style.display = 'flex';

    // Mini chart
    const ctx = document.getElementById('distChart');
    if (distChart) {
        distChart.data.labels = labels;
        distChart.data.datasets[0].data = counts;
        distChart.update();
    } else {
        distChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: counts,
                    backgroundColor: CHART_COLORS.slice(0, labels.length),
                    borderWidth: 0,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                plugins: {
                    legend: { display: false },
                    tooltip: { enabled: true }
                },
                cutout: '55%',
            }
        });
    }

    // Bars
    let barsHtml = '';
    labels.forEach((label, i) => {
        const pct = ((counts[i] / total) * 100).toFixed(0);
        barsHtml += `
            <div class="dist-bar-row">
                <span class="dist-bar-label">${label}</span>
                <div class="dist-bar-track">
                    <div class="dist-bar-fill" style="width: ${pct}%; background: ${CHART_COLORS[i % CHART_COLORS.length]}"></div>
                </div>
                <span class="dist-bar-pct">${pct}%</span>
            </div>
        `;
    });
    distBars.innerHTML = barsHtml;
}

// ================================
// Test Set Modal
// ================================

function initTestSetModal() {
    if (typeof testSetDistribution === 'undefined' || !testSetDistribution || testSetDistribution.length === 0) return;

    const total = testSetDistribution.reduce((sum, d) => sum + d.count, 0);

    // Render bars
    const barsDiv = document.getElementById('testSetBars');
    if (barsDiv) {
        let html = '';
        testSetDistribution.forEach((item, i) => {
            const pct = ((item.count / total) * 100).toFixed(1);
            html += `
                <div class="dist-bar-row">
                    <span class="dist-bar-label">${item.label}</span>
                    <div class="dist-bar-track">
                        <div class="dist-bar-fill" style="width: ${pct}%; background: ${CHART_COLORS[i % CHART_COLORS.length]}"></div>
                    </div>
                    <span class="dist-bar-pct">${item.count} (${pct}%)</span>
                </div>
            `;
        });
        barsDiv.innerHTML = html;
    }

    // Create chart when modal opens
    const modal = document.getElementById('testSetModal');
    if (modal) {
        const observer = new MutationObserver(function() {
            if (modal.classList.contains('open') && !testSetChart) {
                const ctx = document.getElementById('testSetChart');
                if (ctx) {
                    testSetChart = new Chart(ctx, {
                        type: 'doughnut',
                        data: {
                            labels: testSetDistribution.map(d => d.label),
                            datasets: [{
                                data: testSetDistribution.map(d => d.count),
                                backgroundColor: CHART_COLORS.slice(0, testSetDistribution.length),
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
            }
        });
        observer.observe(modal, { attributes: true, attributeFilter: ['class'] });
    }
}

// ================================
// Start Training
// ================================

async function startTraining(mode) {
    const ids = Object.keys(selectedDatasets).map(Number);

    if (ids.length === 0) {
        toast('Select at least one dataset', 'error');
        return;
    }

    let btn, confirmMsg, payload;

    if (mode === 'retrain') {
        if (!selectedModelId) {
            toast('Select a model to retrain', 'error');
            return;
        }
        btn = document.getElementById('retrainBtn');
        confirmMsg = `Retrain ${selectedModelName} with ${ids.length} dataset(s)?`;
        payload = {
            upload_ids: ids,
            mode: 'retrain',
            base_model_id: selectedModelId
        };
    } else {
        const algoRadio = document.querySelector('input[name="algorithm"]:checked');
        const algorithm = algoRadio ? algoRadio.value : 'logistic_regression';
        const algoName = ALGORITHM_NAMES[algorithm];

        btn = document.getElementById('trainBtn');
        confirmMsg = `Start training with ${algoName}?`;
        payload = {
            upload_ids: ids,
            mode: 'new',
            algorithm: algorithm
        };
    }

    if (!confirm(confirmMsg)) return;

    btn.disabled = true;
    const originalText = btn.innerHTML;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Starting...';

    const { ok, data } = await apiCall(URLS.startTraining, {
        method: 'POST',
        body: JSON.stringify(payload)
    });

    if (ok && data.success) {
        toast(data.message);
        setTimeout(() => location.reload(), 1500);
    } else {
        toast(data.error || 'Failed to start training', 'error');
        btn.disabled = false;
        btn.innerHTML = originalText;
    }
}