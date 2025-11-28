/**
 * ML Admin - Analytics Page (User/App focused)
 */

document.addEventListener('DOMContentLoaded', function() {
    initMentalStateChart();
    initDailyChart();
    initSignupsChart();
});

// Mental State Distribution Chart
function initMentalStateChart() {
    if (typeof mentalStateData === 'undefined' || !mentalStateData || mentalStateData.length === 0) return;
    
    createDoughnutChart(
        'mentalStateChart',
        mentalStateData.map(d => d.mental_state || 'Unknown'),
        mentalStateData.map(d => d.count)
    );
}

// Daily Predictions Chart
function initDailyChart() {
    if (typeof dailyData === 'undefined' || !dailyData || dailyData.length === 0) return;
    
    createLineChart(
        'dailyChart',
        dailyData.map(d => d.date),
        dailyData.map(d => d.count),
        { label: 'Predictions' }
    );
}

// User Signups Chart
function initSignupsChart() {
    if (typeof signupsData === 'undefined' || !signupsData || signupsData.length === 0) return;
    
    const ctx = document.getElementById('signupsChart');
    if (!ctx) return;
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: signupsData.map(d => d.date),
            datasets: [{
                label: 'New Users',
                data: signupsData.map(d => d.count),
                backgroundColor: CHART_COLORS[0],
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: { y: { beginAtZero: true, ticks: { stepSize: 1 } } }
        }
    });
}
