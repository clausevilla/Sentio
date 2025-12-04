/* Author: Lian Shi */
/* Disclaimer: LLM has used to help with implement training page functionalities to fit our database */

// State
let currentTab = 'train';
let selectedDatasets = {};
let datasetDistributions = {};
let selectedModelId = null;
let selectedModelName = null;
let selectedModelType = null;
let distChart = null;
let testSetChart = null;

document.addEventListener('DOMContentLoaded', function () {
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

function selectModel(element, modelId, modelType, modelName) {
    // Update UI
    document.querySelectorAll('.model-option').forEach(opt => {
        opt.classList.remove('selected');
        opt.querySelector('input').checked = false;
    });

    element.classList.add('selected');
    element.querySelector('input').checked = true;

    // Store selection
    selectedModelId = modelId;
    selectedModelType = modelType;
    selectedModelName = modelName;

    // Set algorithm for params
    currentParamsAlgorithm = modelType;

    updateRetrainSummary();
}

// ================================
// Dataset Selection (Shared)
// ================================

function toggleDataset(element, id, count, event) {
    const checkbox = element.querySelector('input[type="checkbox"]');

    if (!event || event.target.type !== 'checkbox') {
        checkbox.checked = !checkbox.checked;
    }
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

function updateRetrainSummary() {
    updateSummary();
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
        const observer = new MutationObserver(function () {
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
            base_model_id: selectedModelId,
            params: getCurrentParams()
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
            algorithm: algorithm,
            params: getCurrentParams()
        };
    }

    const confirmed = await showConfirm({
        title: 'Start Training',
        message: confirmMsg,
        type: 'warning',
        confirmText: 'Confirm'
    });
    if (!confirmed) return;

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

// ================================
// Job Details Modal
// ================================

function showJobDetails(jobId) {
    let jobsData = [];
    try {
        const dataEl = document.getElementById('jobsData');
        if (dataEl) {
            jobsData = JSON.parse(dataEl.textContent);
        }
    } catch (e) {
        console.error('Failed to parse jobs data:', e);
        return;
    }

    const job = jobsData.find(j => j.id === jobId);
    if (!job) {
        toast('Job not found', 'error');
        return;
    }

    // Update modal title
    document.getElementById('jobModalTitle').textContent = `Job #${job.id} Details`;

    // Build modal content
    let html = `
        <div class="job-info-grid">
            <div class="job-detail-section">
                <div class="job-detail-label">Status</div>
                <div class="job-detail-value">
                    <span class="badge ${job.status.toLowerCase()}">${job.status}</span>
                </div>
            </div>
            <div class="job-detail-section">
                <div class="job-detail-label">Initiated By</div>
                <div class="job-detail-value">${job.initiated_by}</div>
            </div>
            <div class="job-detail-section">
                <div class="job-detail-label">Started</div>
                <div class="job-detail-value">${job.started_at}</div>
            </div>
            <div class="job-detail-section">
                <div class="job-detail-label">Completed</div>
                <div class="job-detail-value">${job.completed_at || '—'}</div>
            </div>
            <div class="job-detail-section">
                <div class="job-detail-label">Dataset</div>
                <div class="job-detail-value">${job.dataset}</div>
            </div>
            <div class="job-detail-section">
                <div class="job-detail-label">Records</div>
                <div class="job-detail-value">${job.records ? formatNumber(job.records) : '—'}</div>
            </div>
        </div>
    `;

    // Show model info if completed
    if (job.model) {
        html += `
            <hr style="margin: 1.25rem 0; border: none; border-top: 1px solid var(--gray-200);">
            <div class="job-detail-section">
                <div class="job-detail-label">Resulting Model</div>
                <div class="job-detail-value">
                    <span class="job-model-name"><i class="fas fa-cube"></i> ${job.model.name}</span>
                </div>
            </div>
            <div class="job-detail-section" style="margin-top: 1rem;">
                <div class="job-detail-label">Performance Metrics</div>
                <div class="job-metrics-grid" style="margin-top: 0.5rem;">
                    <div class="job-metric">
                        <span class="job-metric-value">${job.model.accuracy ? job.model.accuracy.toFixed(1) + '%' : '—'}</span>
                        <span class="job-metric-label">Accuracy</span>
                    </div>
                    <div class="job-metric">
                        <span class="job-metric-value">${job.model.precision ? job.model.precision.toFixed(1) + '%' : '—'}</span>
                        <span class="job-metric-label">Precision</span>
                    </div>
                    <div class="job-metric">
                        <span class="job-metric-value">${job.model.recall ? job.model.recall.toFixed(1) + '%' : '—'}</span>
                        <span class="job-metric-label">Recall</span>
                    </div>
                    <div class="job-metric">
                        <span class="job-metric-value">${job.model.f1_score ? job.model.f1_score.toFixed(3) : '—'}</span>
                        <span class="job-metric-label">F1 Score</span>
                    </div>
                </div>
            </div>
        `;
    }

    if (job.status === 'FAILED' && job.error_message) {
        html += `
            <hr style="margin: 1.25rem 0; border: none; border-top: 1px solid var(--gray-200);">
            <div class="job-error-box">
                <div class="job-error-title">
                    <i class="fas fa-exclamation-triangle"></i> Error Message
                </div>
                <div class="job-error-message">${job.error_message}</div>
            </div>
        `;
    }

    document.getElementById('jobModalBody').innerHTML = html;
    openModal('jobDetailsModal');
}

/* ================================
   Algorithm Parameters Modal
================================ */

// Algorithm parameter definitions, use SELECT dropdowns with specific choices
const ALGORITHM_PARAMS = {
    logistic_regression: {
        name: 'Logistic Regression',
        icon: 'fa-chart-line',
        params: [
            // Logistic Regression specific
            { key: 'max_iter', label: 'Max Iterations', default: 1000, options: [100, 500, 1000, 2000, 5000], hint: 'Maximum iterations for solver convergence' },
            { key: 'regularization_strength', label: 'Regularization (C)', default: 1.0, options: [0.01, 0.1, 0.5, 1.0, 2.0, 10.0], hint: 'Inverse regularization strength' },
            { key: 'solver', label: 'Solver', default: 'lbfgs', options: ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'], hint: 'Optimization algorithm' },
            // TF-IDF shared
            { key: 'ngram_range_min', label: 'N-gram Min', default: 1, options: [1, 2], hint: 'Minimum n-gram size' },
            { key: 'ngram_range_max', label: 'N-gram Max', default: 2, options: [1, 2, 3], hint: 'Maximum n-gram size (must be ≥ min)' },
            { key: 'min_df', label: 'Min Doc Frequency', default: 2, options: [1, 2, 5, 10], hint: 'Ignore terms in fewer documents' },
            { key: 'max_df', label: 'Max Doc Frequency', default: 0.95, options: [0.8, 0.9, 0.95, 1.0], hint: 'Ignore terms in more than X% docs' },
            { key: 'tfidf_max_features', label: 'Max Features', default: 'None', options: ['None', 5000, 10000, 20000, 50000], hint: 'Max vocabulary size (None = unlimited)' },
        ]
    },
    random_forest: {
        name: 'Random Forest',
        icon: 'fa-tree',
        params: [
            // Random Forest specific
            { key: 'n_estimators', label: 'Number of Trees', default: 100, options: [50, 100, 200, 500, 1000], hint: 'Number of trees in the forest' },
            { key: 'max_depth', label: 'Max Depth', default: 'None', options: ['None', 5, 10, 20, 50, 100], hint: 'Max depth of trees (None = unlimited)' },
            { key: 'min_samples_split', label: 'Min Samples Split', default: 2, options: [2, 5, 10, 20], hint: 'Min samples to split a node' },
            { key: 'min_samples_leaf', label: 'Min Samples Leaf', default: 1, options: [1, 2, 5, 10], hint: 'Min samples at leaf node' },
            { key: 'rf_max_features', label: 'Max Features (Split)', default: 'sqrt', options: ['sqrt', 'log2', 'None'], hint: 'Features to consider for best split' },
            { key: 'n_jobs', label: 'Parallel Jobs', default: -1, options: [-1, 1, 2, 4], hint: '-1 = use all CPUs' },
            // TF-IDF shared
            { key: 'ngram_range_min', label: 'N-gram Min', default: 1, options: [1, 2], hint: 'Minimum n-gram size' },
            { key: 'ngram_range_max', label: 'N-gram Max', default: 2, options: [1, 2, 3], hint: 'Maximum n-gram size (must be ≥ min)' },
            { key: 'min_df', label: 'Min Doc Frequency', default: 2, options: [1, 2, 5, 10], hint: 'Ignore terms in fewer documents' },
            { key: 'max_df', label: 'Max Doc Frequency', default: 0.95, options: [0.8, 0.9, 0.95, 1.0], hint: 'Ignore terms in more than X% docs' },
            { key: 'tfidf_max_features', label: 'Max Features', default: 'None', options: ['None', 5000, 10000, 20000, 50000], hint: 'Max vocabulary size' },
        ]
    },
    lstm: {
        name: 'LSTM (RNN)',
        icon: 'fa-network-wired',
        params: [
            // LSTM specific
            { key: 'embed_dim', label: 'Embedding Dim', default: 64, options: [32, 64, 128, 256], hint: 'Word embedding dimensions' },
            { key: 'hidden_dim', label: 'Hidden Dim', default: 64, options: [32, 64, 128, 256], hint: 'LSTM hidden state size' },
            // Shared neural networks
            { key: 'num_layers', label: 'Number of Layers', default: 2, options: [1, 2, 3, 4, 6], hint: 'Stacked LSTM layers' },
            { key: 'dropout', label: 'Dropout', default: 0.1, options: [0.0, 0.1, 0.2, 0.3, 0.5], hint: 'Dropout rate for regularization' },
            { key: 'max_seq_length', label: 'Max Sequence Length', default: 512, options: [128, 256, 512, 1024], hint: 'Maximum input text length' },
            { key: 'vocab_size', label: 'Vocabulary Size', default: 30000, options: [10000, 20000, 30000, 50000], hint: 'Maximum vocabulary size' },
            { key: 'learning_rate', label: 'Learning Rate', default: 0.0001, options: [0.00001, 0.0001, 0.001, 0.01], hint: 'Optimizer learning rate' },
            { key: 'batch_size', label: 'Batch Size', default: 32, options: [8, 16, 32, 64, 128], hint: 'Training batch size' },
            { key: 'epochs', label: 'Epochs', default: 10, options: [5, 10, 20, 50], hint: 'Training epochs' },
            { key: 'patience', label: 'Early Stop Patience', default: 5, options: [2, 3, 5, 10], hint: 'Epochs to wait before early stopping' },
            { key: 'expand_vocab', label: 'Expand Vocabulary', default: 'False', options: ['True', 'False'], hint: 'Expand vocab with new data' },
        ]
    },
    transformer: {
        name: 'Transformer',
        icon: 'fa-microchip',
        params: [
            // Transformer specific
            { key: 'd_model', label: 'Model Dimension', default: 128, options: [64, 128, 256, 512], hint: 'Transformer dimension (must be divisible by n_head)' },
            { key: 'n_head', label: 'Attention Heads', default: 4, options: [2, 4, 8], hint: 'Number of attention heads' },
            { key: 'dim_feedforward', label: 'Feedforward Dim', default: 256, options: [128, 256, 512, 1024], hint: 'Feedforward network dimension' },
            // Shared neural networks
            { key: 'num_layers', label: 'Number of Layers', default: 2, options: [1, 2, 3, 4, 6], hint: 'Transformer encoder layers' },
            { key: 'dropout', label: 'Dropout', default: 0.1, options: [0.0, 0.1, 0.2, 0.3, 0.5], hint: 'Dropout rate' },
            { key: 'max_seq_length', label: 'Max Sequence Length', default: 512, options: [128, 256, 512, 1024], hint: 'Maximum input length' },
            { key: 'vocab_size', label: 'Vocabulary Size', default: 30000, options: [10000, 20000, 30000, 50000], hint: 'Maximum vocabulary size' },
            { key: 'learning_rate', label: 'Learning Rate', default: 0.0001, options: [0.00001, 0.0001, 0.001, 0.01], hint: 'Optimizer learning rate' },
            { key: 'batch_size', label: 'Batch Size', default: 32, options: [8, 16, 32, 64, 128], hint: 'Training batch size' },
            { key: 'epochs', label: 'Epochs', default: 10, options: [5, 10, 20, 50], hint: 'Training epochs' },
            { key: 'patience', label: 'Early Stop Patience', default: 5, options: [2, 3, 5, 10], hint: 'Epochs to wait before early stopping' },
            { key: 'expand_vocab', label: 'Expand Vocabulary', default: 'False', options: ['True', 'False'], hint: 'Expand vocab with new data' },
        ]
    }
};

// Incremental/Retrain defaults - override these when in retrain mode (neural networks only)
const INCREMENTAL_DEFAULTS = {
    learning_rate: 0.00001,   // 10x lower than full (0.0001)
    epochs: 5,                // Half of full (10)
    patience: 3,              // Lower than full (5)
    expand_vocab: 'True',     // Expand vocab with new data (vs False for full)
};

// Which algorithms are neural networks (use incremental NN defaults)
const NEURAL_NETWORK_ALGOS = ['lstm', 'transformer', 'rnn'];

// Alias for different naming conventions
ALGORITHM_PARAMS['rnn'] = ALGORITHM_PARAMS['lstm'];
ALGORITHM_PARAMS['LSTM'] = ALGORITHM_PARAMS['lstm'];
ALGORITHM_PARAMS['RNN'] = ALGORITHM_PARAMS['lstm'];
ALGORITHM_PARAMS['Transformer'] = ALGORITHM_PARAMS['transformer'];

// Current parameter values - stored per mode+algorithm
// Key format: 'new_logistic_regression' or 'retrain_lstm'
let currentParams = {};
let currentParamsAlgorithm = null;
let currentParamsMode = null;  // 'new' or 'retrain'

// ================================
// Open Parameters Modal
// ================================
function openParamsModal(algoKey) {
    // Normalize algorithm key
    const normalizedKey = normalizeAlgoKey(algoKey);
    const algo = ALGORITHM_PARAMS[normalizedKey];

    if (!algo) {
        console.error('Unknown algorithm:', algoKey);
        toast('Unknown algorithm type', 'error');
        return;
    }

    currentParamsAlgorithm = normalizedKey;

    // Check if we're in retrain mode (retrain tab is active)
    const isIncremental = currentTab === 'retrain';
    currentParamsMode = isIncremental ? 'retrain' : 'new';

    // Create unique key for this mode+algorithm combination
    const paramsKey = `${currentParamsMode}_${normalizedKey}`;

    // Initialize params with mode-specific defaults if not already set
    if (!currentParams[paramsKey]) {
        currentParams[paramsKey] = {};
        algo.params.forEach(p => {
            currentParams[paramsKey][p.key] = getDefaultValue(normalizedKey, p.key, isIncremental);
        });
    }

    // Store mode info on modal
    const modal = document.getElementById('paramsModal');
    if (modal) {
        modal.dataset.isIncremental = isIncremental;
        modal.dataset.paramsKey = paramsKey;
        modal.dataset.algorithm = normalizedKey;
    }

    // Update modal title with mode indicator
    const modeLabel = isIncremental ? ' <span class="mode-badge retrain"><i class="fas fa-sync-alt"></i> Fine-tuning</span>' : '';
    document.getElementById('paramsModalTitle').innerHTML = `
        <i class="fas ${algo.icon}"></i> ${algo.name} Parameters${modeLabel}
    `;

    // Update info text based on mode
    const infoEl = document.querySelector('.params-info');
    if (infoEl) {
        if (isIncremental) {
            infoEl.innerHTML = `<i class="fas fa-info-circle"></i> <strong>Fine-tuning mode:</strong> Lower learning rate and fewer epochs by default for incremental training.`;
            infoEl.classList.add('retrain-info');
        } else {
            infoEl.innerHTML = `<i class="fas fa-info-circle"></i> Adjust training parameters below or use defaults. Hover over <i class="fas fa-question-circle"></i> for hints.`;
            infoEl.classList.remove('retrain-info');
        }
    }

    // Build params grid
    const grid = document.getElementById('paramsGrid');
    let html = '';

    algo.params.forEach(param => {
        const value = currentParams[paramsKey][param.key];
        const defaultVal = getDefaultValue(normalizedKey, param.key, isIncremental);
        const isModified = String(value) !== String(defaultVal);
        html += renderParamInput(param, value, isModified);
    });

    grid.innerHTML = html;

    // Add event listeners for validation and change tracking
    grid.querySelectorAll('.param-input').forEach(input => {
        input.addEventListener('change', handleParamChange);
    });

    // Show modal
    openModal('paramsModal');
}

// ================================
// Normalize Algorithm Key so 
// ================================
function normalizeAlgoKey(algoKey) {
    if (!algoKey) return 'logistic_regression';
    const key = algoKey.toLowerCase().replace(/[^a-z_]/g, '');
    if (key === 'rnn') return 'lstm';
    return key;
}

// ================================
// Check if Neural Network
// ================================
function isNeuralNetwork(algoKey) {
    const normalized = normalizeAlgoKey(algoKey);
    return NEURAL_NETWORK_ALGOS.includes(normalized);
}

// ================================
// Get Default Value
// ================================
function getDefaultValue(algoKey, paramKey, isIncremental = false) {
    const normalizedKey = normalizeAlgoKey(algoKey);
    const algoConfig = ALGORITHM_PARAMS[normalizedKey];

    if (!algoConfig) return null;

    const param = algoConfig.params.find(p => p.key === paramKey);
    if (!param) return null;

    // Check incremental defaults for neural networks
    if (isIncremental && isNeuralNetwork(normalizedKey) && INCREMENTAL_DEFAULTS[paramKey] !== undefined) {
        return INCREMENTAL_DEFAULTS[paramKey];
    }

    return param.default;
}

// ================================
// Handle Parameter Change
// ================================
function handleParamChange(e) {
    const key = e.target.dataset.key;
    const value = e.target.value;

    const modal = document.getElementById('paramsModal');
    const paramsKey = modal?.dataset.paramsKey;
    const algoKey = modal?.dataset.algorithm;
    const isIncremental = modal?.dataset.isIncremental === 'true';

    if (!paramsKey || !algoKey) return;

    // Store the value
    currentParams[paramsKey][key] = value;

    // Update modified state
    const defaultVal = getDefaultValue(algoKey, key, isIncremental);
    const isModified = String(value) !== String(defaultVal);
    const paramItem = e.target.closest('.param-item');
    if (paramItem) {
        paramItem.classList.toggle('modified', isModified);
    }

    // Cross-field validation
    validateParams(algoKey);
}

// ================================
// Validate Parameters
// ================================
function validateParams(algoKey) {
    const modal = document.getElementById('paramsModal');
    const paramsKey = modal?.dataset.paramsKey;
    if (!paramsKey) return { valid: true, errors: [] };

    const params = currentParams[paramsKey] || {};
    const errors = [];

    // Clear previous errors
    document.querySelectorAll('.param-error').forEach(el => el.remove());
    document.querySelectorAll('.param-item.error').forEach(el => el.classList.remove('error'));

    // Validate ngram_range for traditional ML
    if (params.ngram_range_min !== undefined && params.ngram_range_max !== undefined) {
        const min = parseInt(params.ngram_range_min);
        const max = parseInt(params.ngram_range_max);
        if (max < min) {
            errors.push({
                key: 'ngram_range_max',
                message: 'N-gram Max must be ≥ N-gram Min'
            });
        }
    }

    // Validate d_model % n_head == 0 for transformer
    if (algoKey === 'transformer' && params.d_model !== undefined && params.n_head !== undefined) {
        const dModel = parseInt(params.d_model);
        const nHead = parseInt(params.n_head);
        if (dModel % nHead !== 0) {
            errors.push({
                key: 'd_model',
                message: `Model Dimension (${dModel}) must be divisible by Attention Heads (${nHead})`
            });
        }
    }

    // Show errors
    errors.forEach(err => {
        const input = document.querySelector(`[data-key="${err.key}"]`);
        if (input) {
            const paramItem = input.closest('.param-item');
            if (paramItem) {
                paramItem.classList.add('error');
                const errorDiv = document.createElement('div');
                errorDiv.className = 'param-error';
                errorDiv.innerHTML = `<i class="fas fa-exclamation-circle"></i> ${err.message}`;
                paramItem.appendChild(errorDiv);
            }
        }
    });

    return { valid: errors.length === 0, errors };
}

// ================================
// Render Parameter Input (SELECT only)
// ================================
function renderParamInput(param, value, isModified = false) {
    const modifiedClass = isModified ? 'modified' : '';

    const options = param.options.map(opt => {
        const optStr = String(opt);
        const valStr = String(value);
        const selected = optStr === valStr ? 'selected' : '';
        return `<option value="${opt}" ${selected}>${opt}</option>`;
    }).join('');

    return `
        <div class="param-item ${modifiedClass}">
            <label class="param-label">
                ${param.label}
                <span class="param-hint" title="${param.hint}">
                    <i class="fas fa-question-circle"></i>
                </span>
            </label>
            <select class="param-input" data-key="${param.key}">
                ${options}
            </select>
        </div>
    `;
}

// ================================
// Close Modal
// ================================
function closeParamsModal() {
    // Validate before closing
    const modal = document.getElementById('paramsModal');
    const algoKey = modal?.dataset.algorithm;

    if (algoKey) {
        const validation = validateParams(algoKey);
        if (!validation.valid) {
            toast('Please fix validation errors', 'error');
            return;
        }
    }

    closeModal('paramsModal');

    // Update the config button to show modified indicator
    updateConfigButtonState();
}

// ================================
// Update config button modified state
// ================================
function updateConfigButtonState() {
    if (!currentParamsAlgorithm) return;

    const isModified = hasCustomParams(currentParamsAlgorithm);

    // For train tab - find by radio value
    const trainRadio = document.querySelector(`input[name="algorithm"][value="${currentParamsAlgorithm}"]`);
    if (trainRadio) {
        const btn = trainRadio.closest('.algo-option')?.querySelector('.algo-config-btn');
        if (btn) {
            btn.classList.toggle('modified', isModified);
        }
    }

    // For retrain tab - find by selected model
    const selectedModel = document.querySelector('.model-option.selected .algo-config-btn');
    if (selectedModel) {
        selectedModel.classList.toggle('modified', isModified);
    }
}

// ================================
// Reset to Defaults
// ================================
function resetParamsToDefaults() {
    if (!currentParamsAlgorithm) return;

    const algo = ALGORITHM_PARAMS[currentParamsAlgorithm];
    if (!algo) return;

    // Get mode from modal
    const modal = document.getElementById('paramsModal');
    const isIncremental = modal?.dataset.isIncremental === 'true';
    const paramsKey = modal?.dataset.paramsKey || `${isIncremental ? 'retrain' : 'new'}_${currentParamsAlgorithm}`;

    // Reset values with mode-specific defaults
    currentParams[paramsKey] = {};
    algo.params.forEach(param => {
        const defaultVal = getDefaultValue(currentParamsAlgorithm, param.key, isIncremental);
        currentParams[paramsKey][param.key] = defaultVal;

        // Update input
        const input = document.querySelector(`[data-key="${param.key}"]`);
        if (input) {
            input.value = defaultVal;
        }

        // Remove modified class
        const paramItem = input?.closest('.param-item');
        if (paramItem) {
            paramItem.classList.remove('modified');
        }
    });

    // Clear any validation errors
    document.querySelectorAll('.param-error').forEach(el => el.remove());
    document.querySelectorAll('.param-item.error').forEach(el => el.classList.remove('error'));

    const modeText = isIncremental ? 'fine-tuning' : 'training';
    toast(`Parameters reset to ${modeText} defaults`);
}

// ================================
// Get Current Parameters
// ================================
function getCurrentParams() {
    const algoKey = currentParamsAlgorithm || getSelectedAlgorithm();
    const isIncremental = currentTab === 'retrain';
    const paramsKey = `${isIncremental ? 'retrain' : 'new'}_${algoKey}`;

    let params = currentParams[paramsKey];

    // If no custom params set, return defaults
    if (!params) {
        const algo = ALGORITHM_PARAMS[algoKey];
        if (!algo) {
            // Still add training_mode even if no algo config
            return { training_mode: isIncremental ? 'incremental' : 'full' };
        }

        params = {};
        algo.params.forEach(p => {
            params[p.key] = getDefaultValue(algoKey, p.key, isIncremental);
        });
    }

    // Always add training_mode based on current tab (not user-selectable)
    return {
        ...params,
        training_mode: isIncremental ? 'incremental' : 'full'
    };
}

// ================================
// Get Selected Algorithm
// ================================
function getSelectedAlgorithm() {
    // For retrain - use stored model type
    if (currentTab === 'retrain' && selectedModelType) {
        return normalizeAlgoKey(selectedModelType);
    }

    // For new training - check radio button
    const radio = document.querySelector('input[name="algorithm"]:checked');
    if (radio) return radio.value;

    return 'logistic_regression';
}

// ================================
// Check if params are modified from defaults
// ================================
function hasCustomParams(algoKey) {
    const algo = ALGORITHM_PARAMS[algoKey];
    if (!algo) return false;

    const isIncremental = currentTab === 'retrain';
    const paramsKey = `${isIncremental ? 'retrain' : 'new'}_${algoKey}`;
    const params = currentParams[paramsKey];

    if (!params) return false;

    return algo.params.some(p => {
        const defaultVal = getDefaultValue(algoKey, p.key, isIncremental);
        return String(params[p.key]) !== String(defaultVal);
    });
}