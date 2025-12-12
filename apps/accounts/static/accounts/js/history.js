// Author: Lian Shi
// Disclaimer: LLM has been used to generate chart display and fine tuning was done manually

/**
 * History Page JavaScript
 */

document.addEventListener('DOMContentLoaded', function() {
    console.log('History page DOM loaded');

    // Verify configuration is available (set by the Django template)
    if (typeof window.historyConfig === 'undefined') {
        console.error('History config not found - make sure the config script runs before history.js');
        return;
    }

    console.log('History config found:', window.historyConfig);

    // Store chart references globally so we can destroy them before recreating
    window.mentalHealthChart = null;
    window.distributionChart = null;

    // Only initialize visualizations if there's data to display
    if (window.historyConfig.hasData) {
        console.log('Data exists, initializing visualizations...');

        // Small delay ensures DOM is fully ready
        setTimeout(function() {
            initializeDistributionChart();
            animateDistributionBars();
            initializeTrendChart('week');
        }, 100);
    } else {
        console.log('No data available, skipping chart initialization');
    }

    // Setup interactive controls (these work even without data)
    setupStateFilter();
    setupTimeFilter();
    setupDeleteButtons();
    setupExpandableText();

    // Add animations
    animateStats();
});

/**
 * Initialize the distribution pie chart (doughnut style).
 */
function initializeDistributionChart() {
    console.log('Initializing distribution pie chart');

    var canvas = document.getElementById('distributionChart');
    if (!canvas) {
        console.error('Canvas #distributionChart not found');
        return;
    }

    var ctx = canvas.getContext('2d');
    var distribution = window.historyConfig.distribution;
    var colors = window.historyConfig.colors;

    console.log('Distribution data:', distribution);

    // Create the doughnut chart
    window.distributionChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Normal', 'Depression', 'Stress', 'Suicidal'],
            datasets: [{
                data: [
                    distribution.normal,
                    distribution.depression,
                    distribution.stress,
                    distribution.suicidal
                ],
                backgroundColor: [
                    colors.normal,
                    colors.depression,
                    colors.stress,
                    colors.suicidal
                ],
                borderColor: '#ffffff',
                borderWidth: 2,
                hoverOffset: 8
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '60%',
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(255, 255, 255, 0.95)',
                    titleColor: '#1F3D3D',
                    bodyColor: '#2C4A4A',
                    borderColor: '#A8C9B8',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: true,
                    callbacks: {
                        label: function(context) {
                            var total = window.historyConfig.totalAnalyses;
                            var value = context.raw;
                            var percentage = total > 0 ? Math.round((value / total) * 100) : 0;
                            return ' ' + value + ' analyses (' + percentage + '%)';
                        }
                    }
                }
            },
            animation: {
                duration: 1000,
                easing: 'easeOutQuart'
            }
        }
    });

    console.log('Distribution chart created successfully');
}

/**
 * Initialize the trend chart with data from the API.
 *
 * This fetches mental health trend data for the specified time period
 * and renders it as a line chart. The chart shows how mental state
 * predictions have changed over time.
 *
 * @param {string} period - Time period: 'week', 'month', or 'all'
 */
function initializeTrendChart(period) {
    console.log('Initializing trend chart for period:', period);

    var canvas = document.getElementById('trendChart');
    if (!canvas) {
        console.error('Canvas #trendChart not found');
        return;
    }

    var ctx = canvas.getContext('2d');
    var chartUrl = window.historyConfig.chartDataUrl;
    var colors = window.historyConfig.colors;

    console.log('Fetching data from:', chartUrl + '?period=' + period);

    // Destroy existing chart to prevent memory leaks
    if (window.mentalHealthChart instanceof Chart) {
        console.log('Destroying existing trend chart');
        window.mentalHealthChart.destroy();
        window.mentalHealthChart = null;
    }

    // Show loading indicator
    ctx.fillStyle = '#3D5A5A';
    ctx.font = '14px -apple-system, BlinkMacSystemFont, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Loading...', canvas.width / 2, canvas.height / 2);

    // Fetch data from the Django API
    fetch(chartUrl + '?period=' + period, {
        method: 'GET',
        headers: {
            'X-Requested-With': 'XMLHttpRequest',
            'Accept': 'application/json'
        },
        credentials: 'same-origin'
    })
    .then(function(response) {
        console.log('API response status:', response.status);
        if (!response.ok) {
            throw new Error('Network error: ' + response.status);
        }
        return response.json();
    })
    .then(function(data) {
        console.log('Chart data received:', data);

        // Clear the loading message
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (!data.labels || !data.datasets) {
            throw new Error('Invalid data structure');
        }

        // Map datasets with Sentio colors for line chart
        var pointStyles = ['circle', 'rect', 'triangle', 'rectRot'];
        var datasetIndex = 0;

        // Detect mobile for point sizes
        var isMobileForPoints = window.innerWidth <= 768;

        var datasets = data.datasets.map(function(dataset) {
            var color = colors.normal;
            if (dataset.label === 'Depression') color = colors.depression;
            else if (dataset.label === 'Stress') color = colors.stress;
            else if (dataset.label === 'Suicidal') color = colors.suicidal;

            var style = pointStyles[datasetIndex % pointStyles.length];
            datasetIndex++;

            return {
                label: dataset.label,
                data: dataset.data,
                borderColor: color,
                backgroundColor: color + '15',
                tension: 0.3,
                fill: true,
                pointRadius: isMobileForPoints ? 3 : 5,
                pointHoverRadius: isMobileForPoints ? 5 : 8,
                borderWidth: isMobileForPoints ? 2 : 2.5,
                pointBackgroundColor: color,
                pointBorderColor: '#ffffff',
                pointBorderWidth: isMobileForPoints ? 1.5 : 2,
                pointStyle: style
            };
        });

        // Create the line chart
        // Detect mobile for responsive options
        var isMobile = window.innerWidth <= 768;
        var isSmallMobile = window.innerWidth <= 480;

        window.mentalHealthChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.labels,
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                layout: {
                    padding: {
                        top: isMobile ? 5 : 10
                    }
                },
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                plugins: {
                    legend: {
                        position: 'top',
                        labels: {
                            usePointStyle: true,
                            padding: isMobile ? 12 : 25,
                            font: { size: isMobile ? 10 : 13, weight: '600' },
                            boxWidth: isMobile ? 8 : 12
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(255, 255, 255, 0.95)',
                        titleColor: '#1F3D3D',
                        bodyColor: '#2C4A4A',
                        borderColor: '#A8C9B8',
                        borderWidth: 1,
                        padding: 10,
                        displayColors: true,
                        callbacks: {
                            label: function(context) {
                                return context.dataset.label + ': ' + context.raw + ' prediction(s)';
                            }
                        }
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            stepSize: 1,
                            font: { size: isMobile ? 10 : 13 },
                            color: '#3D5A5A',
                            callback: function(value) {
                                if (Math.floor(value) === value) return value;
                            }
                        },
                        title: {
                            display: !isSmallMobile,
                            text: 'Predictions',
                            font: { size: 11, weight: '600' },
                            color: '#3D5A5A'
                        },
                        grid: { color: 'rgba(74, 124, 89, 0.1)' },
                        border: { display: true, color: '#3D5A5A' }
                    },
                    x: {
                        ticks: {
                            font: { size: isMobile ? 9 : 13 },
                            color: '#3D5A5A',
                            maxRotation: isMobile ? 60 : 45,
                            minRotation: isMobile ? 45 : 0,
                            autoSkip: isMobile,
                            maxTicksLimit: isMobile ? 8 : 20
                        },
                        grid: { display: false },
                        border: { display: true, color: '#3D5A5A' }
                    }
                },
                animation: {
                    duration: 800,
                    easing: 'easeOutQuart'
                }
            }
        });

        console.log('Trend chart created successfully');
    })
    .catch(function(error) {
        console.error('Error fetching chart data:', error);
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.fillStyle = '#E07A5F';
        ctx.font = '14px -apple-system, BlinkMacSystemFont, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Unable to load chart', canvas.width / 2, canvas.height / 2);
    });
}

/**
 * Animate the horizontal distribution bars from 0% to their target width.
 *
 * Each bar animates with a slight stagger (150ms between bars) to create
 * a cascading effect that draws attention across all four states.
 */
function animateDistributionBars() {
    console.log('Animating distribution bars');

    // FIXED: Using correct class name '.dist-bar' to match HTML
    var bars = document.querySelectorAll('.dist-bar');
    console.log('Found', bars.length, 'distribution bars');

    if (bars.length === 0) {
        console.warn('No distribution bars found');
        return;
    }

    bars.forEach(function(bar, index) {
        var targetWidth = bar.getAttribute('data-target');
        console.log('Bar', index, 'target width:', targetWidth);

        if (targetWidth !== null && targetWidth !== '') {
            var targetValue = parseFloat(targetWidth);

            // Stagger each bar's animation
            setTimeout(function() {
                bar.style.transition = 'width 1s cubic-bezier(0.4, 0, 0.2, 1)';
                bar.style.width = targetValue + '%';
                console.log('Bar', index, 'animated to', targetValue + '%');
            }, index * 150);
        }
    });
}

/**
 * Setup the state filter dropdown for the history list.
 *
 * When users select a mental state from the dropdown, only history items
 * matching that state will be shown. Includes a smooth fade animation.
 */
function setupStateFilter() {
    var stateFilter = document.getElementById('stateFilter');
    if (!stateFilter) {
        console.log('State filter not found (may be expected if no data)');
        return;
    }

    console.log('Setting up state filter');

    stateFilter.addEventListener('change', function() {
        var selectedState = this.value;
        console.log('Filter changed to:', selectedState);

        var historyItems = document.querySelectorAll('.history-item');
        var noResultsMessage = document.getElementById('noResultsMessage');
        var visibleCount = 0;

        historyItems.forEach(function(item) {
            var itemState = item.getAttribute('data-state');

            if (selectedState === 'all' || itemState === selectedState) {
                item.style.display = 'block';
                item.style.opacity = '0';
                item.style.transform = 'translateY(10px)';

                setTimeout(function() {
                    item.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
                    item.style.opacity = '1';
                    item.style.transform = 'translateY(0)';
                }, 10);

                visibleCount++;
            } else {
                item.style.display = 'none';
            }
        });

        console.log('Visible items:', visibleCount);

        if (noResultsMessage) {
            noResultsMessage.style.display = visibleCount === 0 ? 'block' : 'none';
        }
    });
}

/**
 * Setup the time filter buttons for the trend chart.
 *
 * Clicking a time period button (Week, Month, All Time) updates the
 * visual active state and reloads the trend chart with new data.
 */
function setupTimeFilter() {
    // FIXED: Using correct class name '.filter-btn' to match HTML
    var filterButtons = document.querySelectorAll('.filter-btn');

    if (filterButtons.length === 0) {
        console.log('Time filter buttons not found (may be expected if no data)');
        return;
    }

    console.log('Setting up time filter with', filterButtons.length, 'buttons');

    filterButtons.forEach(function(button) {
        button.addEventListener('click', function() {
            console.log('Filter button clicked:', this.getAttribute('data-period'));

            // Remove active class from all buttons
            filterButtons.forEach(function(btn) {
                btn.classList.remove('active');
            });

            // Add active class to clicked button
            this.classList.add('active');

            // Get period and reload chart
            var period = this.getAttribute('data-period');
            initializeTrendChart(period);
        });
    });
}

/**
 * Setup delete buttons for each analysis item.
 *
 * Each analysis item has a delete button that, when clicked, shows a
 * confirmation dialog and then deletes the analysis via the API.
 */
function setupDeleteButtons() {
    var deleteButtons = document.querySelectorAll('.delete-analysis-btn');

    if (deleteButtons.length === 0) {
        console.log('No delete buttons found (may be expected if no data)');
        return;
    }

    console.log('Setting up', deleteButtons.length, 'delete buttons');

    deleteButtons.forEach(function(button) {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();

            var analysisId = this.getAttribute('data-id');
            var historyItem = this.closest('.history-item');

            console.log('Delete button clicked for analysis:', analysisId);

            // Show custom confirmation modal
            showConfirm({
                title: 'Delete Analysis',
                message: 'Are you sure you want to delete this analysis? This action cannot be undone.',
                type: 'danger',
                confirmText: 'Delete',
                cancelText: 'Cancel',
                danger: true
            }).then(function(confirmed) {
                if (confirmed) {
                    deleteAnalysis(analysisId, historyItem);
                }
            });
        });
    });
}

/**
 * Delete an analysis via the API.
 *
 * This sends a DELETE request to the server to remove the analysis.
 * On success, the item is removed from the DOM with a fade-out animation.
 *
 * @param {string} analysisId - The ID of the analysis to delete
 * @param {HTMLElement} historyItem - The DOM element to remove on success
 */
function deleteAnalysis(analysisId, historyItem) {
    // Build the delete URL by replacing the placeholder with the actual ID
    var deleteUrl = window.historyConfig.deleteAnalysisUrl.replace('{id}', analysisId);

    console.log('Deleting analysis:', analysisId);
    console.log('Delete URL:', deleteUrl);

    // Show loading state on the item
    historyItem.style.opacity = '0.5';
    historyItem.style.pointerEvents = 'none';

    fetch(deleteUrl, {
        method: 'DELETE',
        headers: {
            'X-Requested-With': 'XMLHttpRequest',
            'X-CSRFToken': window.historyConfig.csrfToken,
            'Content-Type': 'application/json'
        },
        credentials: 'same-origin'
    })
    .then(function(response) {
        console.log('Delete response status:', response.status);
        if (!response.ok) {
            throw new Error('Delete failed: ' + response.status);
        }
        return response.json();
    })
    .then(function(data) {
        console.log('Delete successful:', data);

        // Animate the item out
        historyItem.style.transition = 'all 0.3s ease';
        historyItem.style.transform = 'translateX(-20px)';
        historyItem.style.opacity = '0';
        historyItem.style.maxHeight = historyItem.offsetHeight + 'px';

        setTimeout(function() {
            historyItem.style.maxHeight = '0';
            historyItem.style.padding = '0';
            historyItem.style.margin = '0';
            historyItem.style.border = 'none';
        }, 300);

        setTimeout(function() {
            historyItem.remove();

            // Update the total count in the stats
            updateStatsAfterDelete();

            // Check if there are any items left
            var remainingItems = document.querySelectorAll('.history-item');
            if (remainingItems.length === 0) {
                // Reload the page to show empty state
                window.location.reload();
            }
        }, 500);

        // Show success message
        showToast('Analysis deleted successfully', 'success');
    })
    .catch(function(error) {
        console.error('Delete error:', error);

        // Restore the item
        historyItem.style.opacity = '1';
        historyItem.style.pointerEvents = 'auto';

        // Show error message
        showToast('Failed to delete analysis. Please try again.', 'error');
    });
}

/**
 * Update the statistics cards after an analysis is deleted.
 *
 * This decrements the total count and updates the display.
 */
function updateStatsAfterDelete() {
    // Update total analyses count
    var totalStatValue = document.querySelector('.stat-card:first-child .stat-value');
    if (totalStatValue) {
        var currentTotal = parseInt(totalStatValue.textContent) || 0;
        if (currentTotal > 0) {
            totalStatValue.textContent = currentTotal - 1;
        }
    }
}

/**
 * Animate the statistics numbers with a counting-up effect.
 *
 * Creates visual interest when the page loads and draws attention
 * to the key metrics at the top of the page.
 */
function animateStats() {
    // FIXED: Using correct class name '.stat-value' to match HTML
    var statValues = document.querySelectorAll('.stat-value[data-count]');
    console.log('Found', statValues.length, 'stat values to animate');

    statValues.forEach(function(stat, index) {
        var finalValue = parseInt(stat.getAttribute('data-count'));

        if (!isNaN(finalValue)) {
            stat.textContent = '0';

            // Stagger the animations
            setTimeout(function() {
                animateNumber(stat, 0, finalValue, 800);
            }, index * 100);
        }
    });
}

/**
 * Animate a number from start to end over a duration.
 *
 * Uses requestAnimationFrame for smooth 60fps animation with an ease-out
 * cubic function that starts fast and slows down at the end.
 *
 * @param {HTMLElement} element - Element to update
 * @param {number} start - Starting value
 * @param {number} end - Ending value
 * @param {number} duration - Animation duration in milliseconds
 */
function animateNumber(element, start, end, duration) {
    var range = end - start;
    var startTime = performance.now();

    function update(currentTime) {
        var elapsed = currentTime - startTime;
        var progress = Math.min(elapsed / duration, 1);

        // Ease-out cubic: starts fast, slows at end
        var easeOut = 1 - Math.pow(1 - progress, 3);
        var current = Math.floor(start + range * easeOut);

        element.textContent = current;

        if (progress < 1) {
            requestAnimationFrame(update);
        } else {
            element.textContent = end;
        }
    }

    requestAnimationFrame(update);
}

/**
 * Show a custom confirmation modal dialog.
 *
 * This replaces the browser's built-in confirm() with a styled modal
 * that matches the Sentio design system.
 *
 * @param {Object} options - Configuration options
 * @param {string} options.title - Modal title
 * @param {string} options.message - Modal message
 * @param {string} options.type - Type: 'warning', 'danger', 'info', 'success'
 * @param {string} options.confirmText - Text for confirm button
 * @param {string} options.cancelText - Text for cancel button
 * @param {boolean} options.danger - If true, styles confirm button as danger
 * @returns {Promise<boolean>} - Resolves to true if confirmed, false if cancelled
 */
function showConfirm(options) {
    var title = options.title || 'Confirm';
    var message = options.message || 'Are you sure?';
    var type = options.type || 'warning';
    var confirmText = options.confirmText || 'Confirm';
    var cancelText = options.cancelText || 'Cancel';
    var danger = options.danger || false;

    return new Promise(function(resolve) {
        // Remove any existing modal
        var existing = document.getElementById('customConfirmModal');
        if (existing) existing.remove();

        // Icon mapping
        var icons = {
            warning: 'fa-exclamation-triangle',
            danger: 'fa-trash-alt',
            info: 'fa-info-circle',
            success: 'fa-check-circle'
        };

        // Create modal element
        var modal = document.createElement('div');
        modal.id = 'customConfirmModal';
        modal.className = 'confirm-modal';
        modal.innerHTML =
            '<div class="confirm-box">' +
                '<div class="confirm-header ' + type + '">' +
                    '<i class="fas ' + (icons[type] || icons.warning) + '"></i>' +
                    '<h4>' + title + '</h4>' +
                '</div>' +
                '<div class="confirm-body">' + message + '</div>' +
                '<div class="confirm-footer">' +
                    '<button class="btn btn-cancel">' + cancelText + '</button>' +
                    '<button class="btn btn-confirm ' + (danger ? 'danger' : '') + '">' + confirmText + '</button>' +
                '</div>' +
            '</div>';

        document.body.appendChild(modal);

        // Trigger open animation
        requestAnimationFrame(function() {
            modal.classList.add('open');
        });

        // Close function
        var closeModal = function(result) {
            modal.classList.remove('open');
            setTimeout(function() {
                modal.remove();
            }, 200);
            resolve(result);
        };

        // Event listeners
        modal.querySelector('.btn-cancel').onclick = function() { closeModal(false); };
        modal.querySelector('.btn-confirm').onclick = function() { closeModal(true); };
        modal.onclick = function(e) {
            if (e.target === modal) closeModal(false);
        };

        // ESC key to close
        var handleEsc = function(e) {
            if (e.key === 'Escape') {
                closeModal(false);
                document.removeEventListener('keydown', handleEsc);
            }
        };
        document.addEventListener('keydown', handleEsc);
    });
}

/**
 * Show a toast notification message.
 *
 * This displays a brief notification at the top of the screen that
 * automatically fades out after a few seconds.
 *
 * @param {string} message - The message to display
 * @param {string} type - Type: 'success', 'error', 'warning', 'info'
 * @param {number} duration - How long to show the toast (ms), default 3000
 */
function showToast(message, type, duration) {
    type = type || 'info';
    duration = duration || 3000;

    // Find or create toast container
    var container = document.querySelector('.toast-container');
    if (!container) {
        container = document.createElement('div');
        container.className = 'toast-container';
        document.body.appendChild(container);
    }

    // Icon mapping
    var icons = {
        success: 'fa-check-circle',
        error: 'fa-times-circle',
        warning: 'fa-exclamation-triangle',
        info: 'fa-info-circle'
    };

    // Create toast element
    var toast = document.createElement('div');
    toast.className = 'toast ' + type;
    toast.innerHTML =
        '<div class="toast-icon"><i class="fas ' + (icons[type] || icons.info) + '"></i></div>' +
        '<div class="toast-content">' +
            '<div class="toast-message">' + message + '</div>' +
        '</div>' +
        '<button class="toast-close"><i class="fas fa-times"></i></button>';

    container.appendChild(toast);

    // Close button handler
    toast.querySelector('.toast-close').onclick = function() {
        toast.classList.add('hiding');
        setTimeout(function() {
            toast.remove();
        }, 300);
    };

    // Auto-remove after duration
    setTimeout(function() {
        if (toast.parentElement) {
            toast.classList.add('hiding');
            setTimeout(function() {
                toast.remove();
            }, 300);
        }
    }, duration);
}

/**
 * Setup expandable text functionality for history items.
 *
 * Long text entries are truncated to 2 lines by default.
 * A "show more" button appears for texts that overflow.
 * Clicking toggles between truncated and full view.
 */
function setupExpandableText() {
    var textElements = document.querySelectorAll('.item-text');

    textElements.forEach(function(textEl) {
        // Temporarily remove line clamp to measure true height
        textEl.style.display = 'block';
        textEl.style.webkitLineClamp = 'unset';
        textEl.style.overflow = 'visible';

        var fullHeight = textEl.scrollHeight;

        // Remove inline styles - let CSS take over
        textEl.style.display = '';
        textEl.style.webkitLineClamp = '';
        textEl.style.webkitBoxOrient = '';
        textEl.style.overflow = '';

        var clampedHeight = textEl.clientHeight;

        // Check if text is actually truncated
        if (fullHeight > clampedHeight + 5) {
            // Create show more button
            var showMoreBtn = document.createElement('button');
            showMoreBtn.className = 'show-more-btn';
            showMoreBtn.textContent = 'Show more';

            // Insert button after text element
            textEl.parentNode.insertBefore(showMoreBtn, textEl.nextSibling);

            // Toggle functionality
            showMoreBtn.addEventListener('click', function(e) {
                e.stopPropagation();

                if (textEl.classList.contains('expanded')) {
                    textEl.classList.remove('expanded');
                    showMoreBtn.textContent = 'Show more';
                } else {
                    textEl.classList.add('expanded');
                    showMoreBtn.textContent = 'Show less';
                }
            });
        }
    });
}