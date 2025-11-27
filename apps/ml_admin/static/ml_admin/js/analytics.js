/**
 * ML Admin - Analytics Page
 */

document.addEventListener('DOMContentLoaded', function() {
    initMentalStateChart();
    initDailyChart();
    initLabelChart();
    initTypeChart();
});

// Mental State Distribution Chart
function initMentalStateChart() {
    if (typeof mentalStateData === 'undefined' || !mentalStateData) return;

    createDoughnutChart(
        'mentalStateChart',
        mentalStateData.map(d => d.mental_state || 'Unknown'),
        mentalStateData.map(d => d.count)
    );
}

// Daily Predictions Chart
function initDailyChart() {
    if (typeof dailyData === 'undefined' || !dailyData) return;

    createLineChart(
        'dailyChart',
        dailyData.map(d => d.date),
        dailyData.map(d => d.count),
        { label: 'Predictions' }
    );
}

// Label Distribution Chart
function initLabelChart() {
    if (typeof labelData === 'undefined' || !labelData) return;

    createDoughnutChart(
        'labelChart',
        labelData.map(d => d.label || 'Unknown'),
        labelData.map(d => d.count)
    );
}

// Dataset Type Chart
function initTypeChart() {
    if (typeof typeData === 'undefined' || !typeData) return;

    createBarChart(
        'typeChart',
        typeData.map(d => d.dataset_type || 'Unknown'),
        typeData.map(d => d.count),
        { label: 'Records' }
    );
}