/**
 * ML Admin - Training Page
 */

// State
let currentTab = 'train';
let trainSelectedDatasets = {};
let trainDatasetDistributions = {};
let retrainSelectedDatasets = {};
let retrainDatasetDistributions = {};
let selectedModelId = null;
let selectedModelName = null;
let distributionChart = null;
let testSetChart = null;

document.addEventListener('DOMContentLoaded', function() {
    initAlgorithmListeners();
    initTestSetChart();
});

// ================================
// Tab Switching
// ================================

function switchTab(tab) {
    currentTab = tab;

    // Update tab buttons
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('active');
    });
    event.currentTarget.classList.add('active');

    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById(tab + 'Tab').classList.add('active');

    // Update distribution title
    const distTitle = document.getElementById('distTitle');
    if (distTitle) {
        distTitle.textContent = (tab === 'retrain')
            ? 'New Data Distribution'
            : 'Selected Data Distribution';
    }

    // Update distribution chart for current tab
    updateDistributionChart();
}

// ================================
// Test Set Chart (Modal)
// ================================

function initTestSetChart() {
    if (typeof testSetDistribution === 'undefined' || !testSetDistribution || testSetDistribution.length === 0) return;

    const total = testSetDistribution.reduce((sum, d) => sum + d.count, 0);

    const barsDiv = document.getElementById('testSetBars');
    if (barsDiv) {
        let html = '';
        testSetDistribution.forEach((item, i) => {
            const percent = ((item.count / total) * 100).toFixed(1);
            html += `
                <div class="dist-item">
                    <span class="dist-label">${item.label}</span>
                    <div class="dist-bar">
                        <div class="dist-fill" style="width: ${percent}%; background: ${CHART_COLORS[i % CHART_COLORS.length]}"></div>
                    </div>
                    <span class="dist-count">${formatNumber(item.count)} (${percent}%)</span>
                </div>
            `;
        });
        barsDiv.innerHTML = html;
    }

    // Create chart when modal opens
    const modal = document.getElementById('testSetModal');
    if (modal) {
        const observer = new MutationObserver(function(mutations) {
            mutations.forEach(function(mutation) {
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
        });
        observer.observe(modal, { attributes: true, attributeFilter: ['class'] });
    }
}

// Algorithm selection listeners
function initAlgorithmListeners() {
    const radios = document.querySelectorAll('input[name="algorithm"]');
    radios.forEach(radio => {
        radio.addEventListener('change', updateTrainSummary);
    });
}


// Model Selection (Retrain tab)


function selectModelToRetrain(element, modelId) {
    // Update UI
    document.querySelectorAll('.model-option').forEach(opt => {
        opt.classList.remove('selected');
        opt.querySelector('input').checked = false;
    });

    element.classList.add('selected');
    element.querySelector('input').checked = true;

    // Store selection
    selectedModelId = modelId;
    const nameEl = element.querySelector('.model-option-name');
    selectedModelName = nameEl ? nameEl.textContent.trim().split('\n')[0].trim() : 'Model #' + modelId;

    updateRetrainSummary();
}

// ================================
// Dataset Selection
// ================================

function toggleDataset(element, id, count, mode) {
    const checkbox = element.querySelector('input[type="checkbox"]');
    checkbox.checked = !checkbox.checked;
    element.classList.toggle('selected', checkbox.checked);

    // Get distribution data from element
    const distData = element.dataset.distribution;
    let distribution = [];
    try {
        distribution = JSON.parse(distData || '[]');
    } catch (e) {
        distribution = [];
    }

    // Update appropriate state based on mode
    const datasets = (mode === 'retrain') ? retrainSelectedDatasets : trainSelectedDatasets;
    const distributions = (mode === 'retrain') ? retrainDatasetDistributions : trainDatasetDistributions;

    if (checkbox.checked) {
        datasets[id] = count;
        distributions[id] = distribution;
    } else {
        delete datasets[id];
        delete distributions[id];
    }

    // Update UI
    if (mode === 'retrain') {
        updateRetrainSummary();
    } else {
        updateTrainSummary();
    }
    updateDistributionChart();
}


// Update training summary

function updateTrainSummary() {
    const count = Object.keys(trainSelectedDatasets).length;
    const summary = document.getElementById('trainSummary');
    const btn = document.getElementById('trainBtn');

    if (count > 0) {
        summary.style.display = 'block';
        btn.disabled = false;

        const algoRadio = document.querySelector('input[name="algorithm"]:checked');
        const algoName = algoRadio ? ALGORITHM_NAMES[algoRadio.value] : '-';
        document.getElementById('sumAlgo').textContent = algoName;
        document.getElementById('sumDatasets').textContent = count;

        let total = 0;
        Object.values(trainSelectedDatasets).forEach(c => total += c);
        document.getElementById('sumRecords').textContent = formatNumber(total);
    } else {
        summary.style.display = 'none';
        btn.disabled = true;
    }
}

function updateRetrainSummary() {
    const count = Object.keys(retrainSelectedDatasets).length;
    const summary = document.getElementById('retrainSummary');
    const btn = document.getElementById('retrainBtn');

    const hasModel = selectedModelId !== null;
    const hasData = count > 0;

    // Show summary if anything is selected
    if (hasModel || hasData) {
        summary.style.display = 'block';
        document.getElementById('sumModel').textContent = selectedModelName || '(select a model)';
        document.getElementById('sumRetrainDatasets').textContent = count;

        let total = 0;
        Object.values(retrainSelectedDatasets).forEach(c => total += c);
        document.getElementById('sumRetrainRecords').textContent = formatNumber(total);
    } else {
        summary.style.display = 'none';
    }

    // Enable button only when both model AND data selected
    btn.disabled = !(hasModel && hasData);
}


// Distribution Chart


function updateDistributionChart() {
    const display = document.getElementById('distributionDisplay');
    const chartWrap = document.getElementById('distributionChartWrap');
    const barsDiv = document.getElementById('distributionBars');

    // Get datasets based on current tab
    const datasets = (currentTab === 'retrain') ? retrainSelectedDatasets : trainSelectedDatasets;
    const distributions = (currentTab === 'retrain') ? retrainDatasetDistributions : trainDatasetDistributions;

    const selectedIds = Object.keys(datasets);

    if (selectedIds.length === 0) {
        display.style.display = 'block';
        display.innerHTML = `
            <div class="empty-state small">
                <i class="fas fa-hand-pointer"></i>
                <p>Select dataset(s) to see distribution</p>
            </div>
        `;
        chartWrap.style.display = 'none';
        barsDiv.style.display = 'none';

        if (distributionChart) {
            distributionChart.destroy();
            distributionChart = null;
        }
        return;
    }

    // Combine distributions
    const combined = {};
    selectedIds.forEach(id => {
        const dist = distributions[id] || [];
        dist.forEach(item => {
            const label = item.label || 'Unknown';
            combined[label] = (combined[label] || 0) + item.count;
        });
    });

    const labels = Object.keys(combined).sort();
    const counts = labels.map(l => combined[l]);
    const total = counts.reduce((a, b) => a + b, 0);

    if (labels.length === 0) {
        display.style.display = 'block';
        display.innerHTML = `
            <div class="empty-state small">
                <i class="fas fa-exclamation-triangle"></i>
                <p>No distribution data available</p>
            </div>
        `;
        chartWrap.style.display = 'none';
        barsDiv.style.display = 'none';
        return;
    }

    // Show chart and bars
    display.style.display = 'none';
    chartWrap.style.display = 'block';
    barsDiv.style.display = 'block';

    // Update or create chart
    const ctx = document.getElementById('selectedDistChart');

    if (distributionChart) {
        distributionChart.data.labels = labels;
        distributionChart.data.datasets[0].data = counts;
        distributionChart.update();
    } else {
        distributionChart = new Chart(ctx, {
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
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                cutout: '60%',
            }
        });
    }

    // Update bars
    let barsHtml = '';
    labels.forEach((label, i) => {
        const percent = ((counts[i] / total) * 100).toFixed(1);
        barsHtml += `
            <div class="dist-item">
                <span class="dist-label">${label}</span>
                <div class="dist-bar">
                    <div class="dist-fill" style="width: ${percent}%; background: ${CHART_COLORS[i % CHART_COLORS.length]}"></div>
                </div>
                <span class="dist-count">${formatNumber(counts[i])} (${percent}%)</span>
            </div>
        `;
    });
    barsDiv.innerHTML = barsHtml;
}


// Start Training


async function startTraining(mode) {
    let ids, algorithm, modelId, confirmMsg, btn;

    if (mode === 'retrain') {
        // Retrain mode
        ids = Object.keys(retrainSelectedDatasets).map(Number);
        modelId = selectedModelId;

        if (!modelId) {
            toast('Select a model to retrain', 'error');
            return;
        }
        if (ids.length === 0) {
            toast('Select at least one dataset', 'error');
            return;
        }

        confirmMsg = `Retrain ${selectedModelName} with ${ids.length} dataset(s)?`;
        btn = document.getElementById('retrainBtn');
    } else {
        // New training mode
        ids = Object.keys(trainSelectedDatasets).map(Number);

        if (ids.length === 0) {
            toast('Select at least one dataset', 'error');
            return;
        }

        const algoRadio = document.querySelector('input[name="algorithm"]:checked');
        algorithm = algoRadio ? algoRadio.value : 'logistic_regression';
        const algoName = ALGORITHM_NAMES[algorithm];

        confirmMsg = `Start training with ${algoName}?`;
        btn = document.getElementById('trainBtn');
    }

    if (!confirm(confirmMsg)) return;

    btn.disabled = true;
    const originalText = btn.innerHTML;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Starting...';

    const payload = {
        upload_ids: ids,
        mode: mode,
    };

    if (mode === 'retrain') {
        payload.base_model_id = modelId;
    } else {
        payload.algorithm = algorithm;
    }

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