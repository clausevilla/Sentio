// hiostory .js - Account history page functionality

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

    // !!! Remember to implement actual data fetching logic later !!!
    // For now, using example data structure
    const chartData = {
        labels: [], // Dates
        datasets: [
            {
                label: 'Normal',
                data: [],
                borderColor: '#27AE60',
                backgroundColor: 'rgba(39, 174, 96, 0.1)',
                tension: 0.4
            },
            {
                label: 'Depression',
                data: [],
                borderColor: '#2874A6',
                backgroundColor: 'rgba(40, 116, 166, 0.1)',
                tension: 0.4
            },
            {
                label: 'Anxiety',
                data: [],
                borderColor: '#F39C12',
                backgroundColor: 'rgba(243, 156, 18, 0.1)',
                tension: 0.4
            },
            {
                label: 'Stress',
                data: [],
                borderColor: '#E74C3C',
                backgroundColor: 'rgba(231, 76, 60, 0.1)',
                tension: 0.4
            },
            {
                label: 'Suicidal',
                data: [],
                borderColor: '#7D3C98',
                backgroundColor: 'rgba(125, 60, 152, 0.1)',
                tension: 0.4
            },
            {
                label: 'Bipolar',
                data: [],
                borderColor: '#D68910',
                backgroundColor: 'rgba(214, 137, 16, 0.1)',
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