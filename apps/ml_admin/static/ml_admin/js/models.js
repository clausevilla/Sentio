/* Author: Lian Shi*/


/**
 * ML Admin - Models Page
 */

document.addEventListener('DOMContentLoaded', function () {
    initComparisonChart();
});

// Deploy model
async function deployModel(id, name) {
    if (!confirm(`Deploy "${name}" as the active model?`)) return;

    const { ok, data } = await apiCall(`/management/api/models/${id}/activate/`, {
        method: 'POST'
    });

    if (ok && data.success) {
        toast(data.message);
        setTimeout(() => location.reload(), 1000);
    } else {
        toast(data.error || 'Deploy failed', 'error');
    }
}

// Delete model
async function deleteModel(id, name) {
    if (!confirm(`Delete "${name}"? This cannot be undone.`)) return;

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

// Model comparison chart
function initComparisonChart() {
    if (typeof modelsData === 'undefined' || !modelsData || modelsData.length < 2) return;

    const ctx = document.getElementById('compareChart');
    if (!ctx) return;

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: modelsData.map(m => m.name),
            datasets: [
                {
                    label: 'Accuracy %',
                    data: modelsData.map(m => m.accuracy || 0),
                    backgroundColor: CHART_COLORS[0],
                },
                {
                    label: 'F1 Score × 100',
                    data: modelsData.map(m => (m.f1 || 0) * 100),
                    backgroundColor: CHART_COLORS[3],
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                y: { beginAtZero: true, max: 100 }
            },
            plugins: {
                legend: { position: 'bottom' }
            }
        }
    });
}
