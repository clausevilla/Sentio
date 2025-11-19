// accounts/history.js - Analysis history page with charts and filtering

document.addEventListener('DOMContentLoaded', function() {
    // Initialize trend chart
    initializeTrendChart();

    // Setup filters
    setupStateFilter();
    setupTimeFilter();

    // Add animations
    animateStats();
});

function initializeTrendChart() {
    const canvas = document.getElementById('trendChart');
    if (!canvas) return;

    // Get data from Django template (you'll need to pass this from your view)
    // For now, using example data structure
    const chartData = {
        labels: [], // Dates
        datasets: [
            {
                label: 'Normal',
                data: [],
                borderColor: '#7FAF93',
                backgroundColor: 'rgba(127, 175, 147, 0.1)',
                tension: 0.4
            },
            {
                label: 'Depression',
                data: [],
                borderColor: '#4A90E2',
                backgroundColor: 'rgba(74, 144, 226, 0.1)',
                tension: 0.4
            },
            {
                label: 'Anxiety',
                data: [],
                borderColor: '#FFA500',
                backgroundColor: 'rgba(255, 165, 0, 0.1)',
                tension: 0.4
            },
            {
                label: 'Stress',
                data: [],
                borderColor: '#FF6B6B',
                backgroundColor: 'rgba(255, 107, 107, 0.1)',
                tension: 0.4
            }
        ]
    };

    const config = {
        type: 'line',
        data: chartData,
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Number of Occurrences'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Date'
                    }
                }
            }
        }
    };

    const chart = new Chart(canvas, config);

    // Store chart instance for updates
    window.mentalHealthChart = chart;
}

function setupStateFilter() {
    const stateFilter = document.getElementById('stateFilter');
    if (!stateFilter) return;

    stateFilter.addEventListener('change', function() {
        const selectedState = this.value;
        const historyItems = document.querySelectorAll('.history-item');

        historyItems.forEach(item => {
            if (selectedState === 'all' || item.dataset.state === selectedState) {
                item.style.display = 'block';
                // Add fade-in animation
                item.style.animation = 'fadeIn 0.3s ease';
            } else {
                item.style.display = 'none';
            }
        });
    });
}

function setupTimeFilter() {
    const filterButtons = document.querySelectorAll('.filter-btn');

    filterButtons.forEach(button => {
        button.addEventListener('click', function() {
            // Update active state
            filterButtons.forEach(btn => btn.classList.remove('active'));
            this.classList.add('active');

            const period = this.dataset.period;
            updateChartForPeriod(period);
        });
    });
}


//!!!  Remember to implement actual data fetching logic later !!!
function updateChartForPeriod(period) {
    // This would typically fetch new data via AJAX
    // For now, just showing the structure and updating the chart title to indicate loading
    // need to implement actual data fetching logic later

    if (!window.mentalHealthChart) return;

    const chart = window.mentalHealthChart;

    // Show loading state
    chart.options.plugins.title = {
        display: true,
        text: 'Loading...'
    };
    chart.update();

    // Fetch data (example - need to implement the actual AJAX call and endpoint later)
    /*
    fetch(`/api/mental-health-trends/?period=${period}`)
        .then(response => response.json())
        .then(data => {
            chart.data.labels = data.labels;
            chart.data.datasets.forEach((dataset, i) => {
                dataset.data = data.datasets[i].data;
            });
            chart.options.plugins.title.display = false;
            chart.update();
        });
    */
}

function animateStats() {
    const statValues = document.querySelectorAll('.stat-value');

    statValues.forEach((stat, index) => {
        const finalValue = parseInt(stat.textContent);
        stat.textContent = '0';

        setTimeout(() => {
            animateNumber(stat, 0, finalValue, 1000);
        }, index * 100);
    });
}

function animateNumber(element, start, end, duration) {
    const range = end - start;
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        const easeOut = 1 - Math.pow(1 - progress, 3);
        const current = Math.floor(start + range * easeOut);

        element.textContent = current;

        if (progress < 1) {
            requestAnimationFrame(update);
        } else {
            element.textContent = end;
        }
    }

    requestAnimationFrame(update);
}
