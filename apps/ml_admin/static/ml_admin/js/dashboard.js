/**
 * ML Admin - Dashboard Page
 */

document.addEventListener('DOMContentLoaded', function() {
    initDistributionChart();
});

function initDistributionChart() {
    if (typeof distributionData === 'undefined' || !distributionData) return;

    createDoughnutChart(
        'distChart',
        distributionData.map(d => d.label || 'Unknown'),
        distributionData.map(d => d.count)
    );
}