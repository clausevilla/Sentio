/**
 * ML Admin - Training Page
 */

let selectedDatasets = {};

document.addEventListener('DOMContentLoaded', function() {
    initAlgorithmListeners();
});

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

    if (checkbox.checked) {
        selectedDatasets[id] = count;
    } else {
        delete selectedDatasets[id];
    }

    updateSummary();
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