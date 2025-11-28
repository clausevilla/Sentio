/**
 * ML Admin - Dashboard Page
 */

document.addEventListener('DOMContentLoaded', function() {
    initModelComparisonChart();
});

function initModelComparisonChart() {
    if (typeof modelsCompareData === 'undefined' || !modelsCompareData || modelsCompareData.length < 2) return;
    
    const ctx = document.getElementById('modelCompareChart');
    if (!ctx) return;
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: modelsCompareData.map(m => m.name),
            datasets: [
                {
                    label: 'Accuracy %',
                    data: modelsCompareData.map(m => m.accuracy || 0),
                    backgroundColor: CHART_COLORS[0],
                },
                {
                    label: 'F1 × 100',
                    data: modelsCompareData.map(m => (m.f1 || 0) * 100),
                    backgroundColor: CHART_COLORS[3],
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: { y: { beginAtZero: true, max: 100 } },
            plugins: { legend: { position: 'bottom' } }
        }
    });
}
