/* Author: Lian Shi*/
/* Disclaimer: LLM has used to help with implement date-related graph display functions */

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

    const ctx = document.getElementById('mentalStateChart');
    if (!ctx) return;

    new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: mentalStateData.map(d => d.mental_state || 'Unknown'),
            datasets: [{
                data: mentalStateData.map(d => d.count),
                backgroundColor: CHART_COLORS.slice(0, mentalStateData.length),
                borderWidth: 0,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'right',
                    labels: {
                        padding: 15,
                        usePointStyle: true,
                    }
                }
            },
            cutout: '60%',
        }
    });
}

// Daily Predictions Chart - Shows all 14 days with proper scaling
function initDailyChart() {
    if (typeof dailyData === 'undefined' || !dailyData || dailyData.length === 0) return;

    const ctx = document.getElementById('dailyChart');
    if (!ctx) return;

    // Calculate max value for better Y-axis scaling
    const maxValue = Math.max(...dailyData.map(d => d.count), 1);
    const suggestedMax = Math.ceil(maxValue * 1.2); // 20% padding above max

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: dailyData.map(d => d.date),
            datasets: [{
                label: 'Predictions',
                data: dailyData.map(d => d.count),
                borderColor: CHART_COLORS[0],
                backgroundColor: CHART_COLORS[0] + '20',
                tension: 0.3,
                fill: true,
                pointRadius: 4,
                pointHoverRadius: 6,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index',
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    padding: 12,
                    titleFont: { size: 14 },
                    bodyFont: { size: 13 },
                }
            },
            scales: {
                x: {
                    grid: {
                        display: false,
                    },
                    ticks: {
                        maxRotation: 45,
                        minRotation: 0,
                        autoSkip: true,
                        maxTicksLimit: 14,
                    },
                    offset: true, // Add padding from Y-axis
                },
                y: {
                    beginAtZero: true,
                    suggestedMax: suggestedMax,
                    ticks: {
                        // Auto-calculate step size based on max value
                        callback: function(value) {
                            if (Number.isInteger(value)) {
                                return value;
                            }
                            return null;
                        },
                        stepSize: maxValue > 10 ? Math.ceil(maxValue / 5) : 1,
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)',
                    }
                }
            },
            layout: {
                padding: {
                    left: 10,
                    right: 10,
                }
            }
        }
    });
}

// User Signups Chart - Shows all 30 days
function initSignupsChart() {
    if (typeof signupsData === 'undefined' || !signupsData || signupsData.length === 0) return;

    const ctx = document.getElementById('signupsChart');
    if (!ctx) return;

    // Calculate max value for better Y-axis scaling
    const maxValue = Math.max(...signupsData.map(d => d.count), 1);
    const suggestedMax = Math.ceil(maxValue * 1.2); // 20% padding above max

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: signupsData.map(d => d.date),
            datasets: [{
                label: 'New Users',
                data: signupsData.map(d => d.count),
                backgroundColor: CHART_COLORS[0],
                borderRadius: 4,
                maxBarThickness: 20,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    padding: 12,
                    callbacks: {
                        title: function(context) {
                            return context[0].label;
                        },
                        label: function(context) {
                            const value = context.parsed.y;
                            return value === 1 ? '1 new user' : `${value} new users`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { display: false },
                    ticks: {
                        maxRotation: 45,
                        minRotation: 0,
                        autoSkip: true,
                        maxTicksLimit: 15, // Show fewer labels for 30 days
                    },
                    offset: true, // Add padding from Y-axis
                },
                y: {
                    beginAtZero: true,
                    suggestedMax: suggestedMax,
                    ticks: {
                        callback: function(value) {
                            if (Number.isInteger(value)) {
                                return value;
                            }
                            return null;
                        },
                        stepSize: maxValue > 10 ? Math.ceil(maxValue / 5) : 1,
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)',
                    }
                }
            },
            layout: {
                padding: {
                    left: 10,
                    right: 10,
                }
            }
        }
    });
}