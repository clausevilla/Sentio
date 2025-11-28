/**
 * ML Admin - Training Page
 */

let selectedDatasets = {};
let datasetDistributions = {};
let distributionChart = null;
let testSetChart = null;

document.addEventListener('DOMContentLoaded', function() {
    initAlgorithmListeners();
    initTestSetChart();
});

// Initialize test set chart (in modal)
function initTestSetChart() {
    if (typeof testSetDistribution === 'undefined' || !testSetDistribution || testSetDistribution.length === 0) return;
    
    const total = testSetDistribution.reduce((sum, d) => sum + d.count, 0);
    
    // Render bars
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
        radio.addEventListener('change', updateSummary);
    });
}

// Toggle dataset selection
function toggleDataset(element, id, count) {
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
    
    if (checkbox.checked) {
        selectedDatasets[id] = count;
        datasetDistributions[id] = distribution;
    } else {
        delete selectedDatasets[id];
        delete datasetDistributions[id];
    }
    
    updateSummary();
    updateDistributionChart();
}

// Update training summary
function updateSummary() {
    const count = Object.keys(selectedDatasets).length;
    const summary = document.getElementById('summary');
    const btn = document.getElementById('trainBtn');
    
    if (count > 0) {
        summary.style.display = 'block';
        btn.disabled = false;
        
        // Algorithm name
        const algoRadio = document.querySelector('input[name="algorithm"]:checked');
        const algoName = algoRadio ? ALGORITHM_NAMES[algoRadio.value] : '-';
        document.getElementById('sumAlgo').textContent = algoName;
        
        // Dataset count
        document.getElementById('sumDatasets').textContent = count;
        
        // Total records
        let total = 0;
        Object.values(selectedDatasets).forEach(c => total += c);
        document.getElementById('sumRecords').textContent = formatNumber(total);
    } else {
        summary.style.display = 'none';
        btn.disabled = true;
    }
}

// Update distribution chart based on selected datasets
function updateDistributionChart() {
    const display = document.getElementById('distributionDisplay');
    const chartWrap = document.getElementById('distributionChartWrap');
    const barsDiv = document.getElementById('distributionBars');
    
    const selectedIds = Object.keys(selectedDatasets);
    
    if (selectedIds.length === 0) {
        // No selection
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
    
    // Combine distributions from all selected datasets
    const combined = {};
    
    selectedIds.forEach(id => {
        const dist = datasetDistributions[id] || [];
        dist.forEach(item => {
            const label = item.label || 'Unknown';
            combined[label] = (combined[label] || 0) + item.count;
        });
    });
    
    // Convert to arrays
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
                plugins: {
                    legend: { display: false }
                },
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

// Start training
async function startTraining() {
    const ids = Object.keys(selectedDatasets).map(Number);
    if (ids.length === 0) {
        toast('Select at least one dataset', 'error');
        return;
    }
    
    const algoRadio = document.querySelector('input[name="algorithm"]:checked');
    const algorithm = algoRadio ? algoRadio.value : 'logistic_regression';
    const algoName = ALGORITHM_NAMES[algorithm];
    
    if (!confirm(`Start training with ${algoName}?`)) return;
    
    const btn = document.getElementById('trainBtn');
    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Starting...';
    
    const { ok, data } = await apiCall(URLS.startTraining, {
        method: 'POST',
        body: JSON.stringify({
            upload_ids: ids,
            algorithm: algorithm
        })
    });
    
    if (ok && data.success) {
        toast(data.message);
        setTimeout(() => location.reload(), 1500);
    } else {
        toast(data.error || 'Failed to start training', 'error');
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-play"></i> Start Training';
    }
}
