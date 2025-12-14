// Author: Lian Shi
// Disclaimer: LLM has been used to generate chart display and fine tuning was done manually

/**
 * History Page JavaScript
 */

document.addEventListener('DOMContentLoaded', function () {
    // Verify configuration is available
    if (typeof window.historyConfig === 'undefined') {
        showToast('Configuration error. Please refresh the page.', 'error');
        return;
    }

    // Store chart references globally
    window.mentalHealthChart = null;
    window.distributionChart = null;

    // Only initialize visualizations if there's data
    if (window.historyConfig.hasData) {
        setTimeout(function () {
            initializeDistributionChart();
            animateDistributionBars();
            initializeTrendChart('week');
        }, 100);
    }

    // Setup interactive controls
    setupStateFilter();
    setupTimeFilter();
    setupDeleteButtons();
    setupExpandableText();
    setupPagination();
    animateStats();
});

/**
 * Initialize the distribution pie chart
 */
function initializeDistributionChart() {
    var canvas = document.getElementById('distributionChart');
    if (!canvas) return;

    var ctx = canvas.getContext('2d');
    var distribution = window.historyConfig.distribution;
    var colors = window.historyConfig.colors;

    try {
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
                            label: function (context) {
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
    } catch (error) {
        showToast('Failed to load distribution chart', 'error');
    }
}

/**
 * Initialize the trend chart with data from the API
 */
function initializeTrendChart(period) {
    var canvas = document.getElementById('trendChart');
    if (!canvas) return;

    var ctx = canvas.getContext('2d');
    var chartUrl = window.historyConfig.chartDataUrl;
    var colors = window.historyConfig.colors;

    // Destroy existing chart
    if (window.mentalHealthChart instanceof Chart) {
        window.mentalHealthChart.destroy();
        window.mentalHealthChart = null;
    }

    // Show loading
    ctx.fillStyle = '#3D5A5A';
    ctx.font = '14px -apple-system, BlinkMacSystemFont, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('Loading...', canvas.width / 2, canvas.height / 2);

    fetch(chartUrl + '?period=' + period, {
        method: 'GET',
        headers: {
            'X-Requested-With': 'XMLHttpRequest',
            'Accept': 'application/json'
        },
        credentials: 'same-origin'
    })
        .then(function (response) {
            if (!response.ok) {
                throw new Error('Failed to load chart data');
            }
            return response.json();
        })
        .then(function (data) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);

            if (!data.labels || !data.datasets) {
                throw new Error('Invalid data format');
            }

            var pointStyles = ['circle', 'rect', 'triangle', 'rectRot'];
            var datasetIndex = 0;
            var isMobileForPoints = window.innerWidth <= 768;

            var datasets = data.datasets.map(function (dataset) {
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
                        padding: { top: isMobile ? 5 : 10 }
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
                                label: function (context) {
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
                                callback: function (value) {
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
        })
        .catch(function (error) {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.fillStyle = '#E07A5F';
            ctx.font = '14px -apple-system, BlinkMacSystemFont, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Unable to load chart', canvas.width / 2, canvas.height / 2);
            showToast('Failed to load trend chart. Please try again.', 'error');
        });
}

/**
 * Animate the distribution bars
 */
function animateDistributionBars() {
    var bars = document.querySelectorAll('.dist-bar');
    if (bars.length === 0) return;

    bars.forEach(function (bar, index) {
        var targetWidth = bar.getAttribute('data-target');
        if (targetWidth !== null && targetWidth !== '') {
            var targetValue = parseFloat(targetWidth);
            setTimeout(function () {
                bar.style.transition = 'width 1s cubic-bezier(0.4, 0, 0.2, 1)';
                bar.style.width = targetValue + '%';
            }, index * 150);
        }
    });
}

/**
 * Setup state filter dropdown
 */
function setupStateFilter() {
    var stateFilter = document.getElementById('stateFilter');
    if (!stateFilter) return;

    stateFilter.addEventListener('change', function () {
        var selectedState = this.value;
        var historyItems = document.querySelectorAll('.history-item');
        var noResultsMessage = document.getElementById('noResultsMessage');
        var visibleCount = 0;

        historyItems.forEach(function (item) {
            var itemState = item.getAttribute('data-state');

            if (selectedState === 'all' || itemState === selectedState) {
                item.style.display = 'block';
                item.style.opacity = '0';
                item.style.transform = 'translateY(10px)';

                setTimeout(function () {
                    item.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
                    item.style.opacity = '1';
                    item.style.transform = 'translateY(0)';
                }, 10);

                visibleCount++;
            } else {
                item.style.display = 'none';
            }
        });

        if (noResultsMessage) {
            noResultsMessage.style.display = visibleCount === 0 ? 'block' : 'none';
        }
    });
}

/**
 * Setup time filter buttons
 */
function setupTimeFilter() {
    var filterButtons = document.querySelectorAll('.filter-btn');
    if (filterButtons.length === 0) return;

    filterButtons.forEach(function (button) {
        button.addEventListener('click', function () {
            filterButtons.forEach(function (btn) {
                btn.classList.remove('active');
            });

            this.classList.add('active');
            var period = this.getAttribute('data-period');
            initializeTrendChart(period);
        });
    });
}

/**
 * Setup AJAX pagination
 */
function setupPagination() {
    var paginationContainer = document.querySelector('.pagination');
    if (!paginationContainer) return;

    paginationContainer.addEventListener('click', function (e) {
        var link = e.target.closest('.page-link');
        if (!link) return;

        e.preventDefault();

        var href = link.getAttribute('href');
        if (!href || href === '#') return;

        fetchPageContent(href);
    });
}

/**
 * Fetch page content via AJAX
 */
function fetchPageContent(url) {
    var historyList = document.getElementById('historyList');
    var paginationContainer = document.querySelector('.pagination');
    var listCard = document.querySelector('.history-list-card');

    if (!historyList) return;

    historyList.style.opacity = '0.5';
    historyList.style.pointerEvents = 'none';

    fetch(url, {
        method: 'GET',
        headers: { 'X-Requested-With': 'XMLHttpRequest' },
        credentials: 'same-origin'
    })
        .then(function (response) {
            if (!response.ok) {
                throw new Error('Failed to load page');
            }
            return response.text();
        })
        .then(function (html) {
            var parser = new DOMParser();
            var doc = parser.parseFromString(html, 'text/html');

            var newHistoryList = doc.getElementById('historyList');
            var newPagination = doc.querySelector('.pagination');

            if (newHistoryList) {
                historyList.innerHTML = newHistoryList.innerHTML;

                var items = historyList.querySelectorAll('.history-item');
                items.forEach(function (item, index) {
                    item.style.opacity = '0';
                    item.style.transform = 'translateY(10px)';
                    setTimeout(function () {
                        item.style.transition = 'opacity 0.3s ease, transform 0.3s ease';
                        item.style.opacity = '1';
                        item.style.transform = 'translateY(0)';
                    }, index * 50);
                });
            }

            if (newPagination && paginationContainer) {
                paginationContainer.innerHTML = newPagination.innerHTML;
            }

            setupDeleteButtons();
            setupExpandableText();

            window.history.pushState({}, '', url);

            if (listCard) {
                listCard.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        })
        .catch(function (error) {
            showToast('Failed to load page. Please try again.', 'error');
        })
        .finally(function () {
            historyList.style.opacity = '1';
            historyList.style.pointerEvents = 'auto';
        });
}

/**
 * Setup delete buttons
 */
function setupDeleteButtons() {
    var deleteButtons = document.querySelectorAll('.delete-analysis-btn');
    if (deleteButtons.length === 0) return;

    deleteButtons.forEach(function (button) {
        var newButton = button.cloneNode(true);
        button.parentNode.replaceChild(newButton, button);

        newButton.addEventListener('click', function (e) {
            e.stopPropagation();
            var analysisId = this.getAttribute('data-id');
            handleDeleteAnalysis(analysisId, this);
        });
    });
}

/**
 * Handle delete analysis
 */
function handleDeleteAnalysis(analysisId, buttonElement) {
    showConfirm({
        title: 'Delete Analysis',
        message: 'Are you sure you want to delete this analysis? This action cannot be undone.',
        type: 'danger',
        confirmText: 'Delete',
        cancelText: 'Cancel',
        danger: true
    }).then(function (confirmed) {
        if (!confirmed) return;

        var historyItem = buttonElement.closest('.history-item');
        if (historyItem) {
            historyItem.style.opacity = '0.5';
            historyItem.style.pointerEvents = 'none';
        }

        var deleteUrl = window.historyConfig.deleteAnalysisUrl.replace('{id}', analysisId);

        fetch(deleteUrl, {
            method: 'DELETE',
            headers: {
                'X-CSRFToken': window.historyConfig.csrfToken,
                'X-Requested-With': 'XMLHttpRequest',
                'Accept': 'application/json'
            },
            credentials: 'same-origin'
        })
            .then(function (response) {
                if (!response.ok) {
                    throw new Error('Delete failed');
                }
                return response.json();
            })
            .then(function (data) {
                if (data.success) {
                    showToast('Analysis deleted successfully', 'success');

                    if (historyItem) {
                        historyItem.style.transition = 'all 0.3s ease';
                        historyItem.style.transform = 'translateX(-100%)';
                        historyItem.style.opacity = '0';

                        setTimeout(function () {
                            historyItem.remove();

                            var remainingItems = document.querySelectorAll('.history-item');
                            if (remainingItems.length === 0) {
                                window.location.reload();
                            }
                        }, 300);
                    }
                } else {
                    throw new Error(data.error || 'Delete failed');
                }
            })
            .catch(function (error) {
                showToast('Failed to delete analysis. Please try again.', 'error');

                if (historyItem) {
                    historyItem.style.opacity = '1';
                    historyItem.style.pointerEvents = 'auto';
                }
            });
    });
}

/**
 * Animate stats cards
 */
function animateStats() {
    var statCards = document.querySelectorAll('.stat-card');

    statCards.forEach(function (card, index) {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';

        setTimeout(function () {
            card.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
            card.style.opacity = '1';
            card.style.transform = 'translateY(0)';
        }, index * 100);
    });

    var statValues = document.querySelectorAll('.stat-value[data-count]');

    statValues.forEach(function (element) {
        var targetValue = parseInt(element.getAttribute('data-count'), 10);
        if (!isNaN(targetValue) && targetValue > 0) {
            element.textContent = '0';
            setTimeout(function () {
                animateNumber(element, 0, targetValue, 1000);
            }, 500);
        }
    });
}

/**
 * Animate number from start to end
 */
function animateNumber(element, start, end, duration) {
    var range = end - start;
    var startTime = performance.now();

    function update(currentTime) {
        var elapsed = currentTime - startTime;
        var progress = Math.min(elapsed / duration, 1);
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
 * Show confirm modal
 */
function showConfirm(options) {
    var title = options.title || 'Confirm';
    var message = options.message || 'Are you sure?';
    var type = options.type || 'warning';
    var confirmText = options.confirmText || 'Confirm';
    var cancelText = options.cancelText || 'Cancel';
    var danger = options.danger || false;

    return new Promise(function (resolve) {
        var existing = document.getElementById('customConfirmModal');
        if (existing) existing.remove();

        var icons = {
            warning: 'fa-exclamation-triangle',
            danger: 'fa-trash-alt',
            info: 'fa-info-circle',
            success: 'fa-check-circle'
        };

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

        requestAnimationFrame(function () {
            modal.classList.add('open');
        });

        var closeModal = function (result) {
            modal.classList.remove('open');
            setTimeout(function () {
                modal.remove();
            }, 200);
            resolve(result);
        };

        modal.querySelector('.btn-cancel').onclick = function () { closeModal(false); };
        modal.querySelector('.btn-confirm').onclick = function () { closeModal(true); };
        modal.onclick = function (e) {
            if (e.target === modal) closeModal(false);
        };

        var handleEsc = function (e) {
            if (e.key === 'Escape') {
                closeModal(false);
                document.removeEventListener('keydown', handleEsc);
            }
        };
        document.addEventListener('keydown', handleEsc);
    });
}

/**
 * Show toast notification
 */
function showToast(message, type, duration) {
    type = type || 'info';
    duration = duration || 3000;

    var container = document.querySelector('.toast-container');
    if (!container) {
        container = document.createElement('div');
        container.className = 'toast-container';
        document.body.appendChild(container);
    }

    var icons = {
        success: 'fa-check-circle',
        error: 'fa-times-circle',
        warning: 'fa-exclamation-triangle',
        info: 'fa-info-circle'
    };

    var toast = document.createElement('div');
    toast.className = 'toast ' + type;
    toast.innerHTML =
        '<div class="toast-icon"><i class="fas ' + (icons[type] || icons.info) + '"></i></div>' +
        '<div class="toast-content">' +
        '<div class="toast-message">' + message + '</div>' +
        '</div>' +
        '<button class="toast-close"><i class="fas fa-times"></i></button>';

    container.appendChild(toast);

    toast.querySelector('.toast-close').onclick = function () {
        toast.classList.add('hiding');
        setTimeout(function () {
            toast.remove();
        }, 300);
    };

    setTimeout(function () {
        if (toast.parentElement) {
            toast.classList.add('hiding');
            setTimeout(function () {
                toast.remove();
            }, 300);
        }
    }, duration);
}

/**
 * Setup expandable text
 */
function setupExpandableText() {
    var textElements = document.querySelectorAll('.item-text');

    textElements.forEach(function (textEl) {
        if (textEl.nextElementSibling && textEl.nextElementSibling.classList.contains('show-more-btn')) {
            return;
        }

        textEl.style.display = 'block';
        textEl.style.webkitLineClamp = 'unset';
        textEl.style.overflow = 'visible';

        var fullHeight = textEl.scrollHeight;

        textEl.style.display = '';
        textEl.style.webkitLineClamp = '';
        textEl.style.webkitBoxOrient = '';
        textEl.style.overflow = '';

        var clampedHeight = textEl.clientHeight;

        if (fullHeight > clampedHeight + 5) {
            var showMoreBtn = document.createElement('button');
            showMoreBtn.className = 'show-more-btn';
            showMoreBtn.textContent = 'Show more';

            textEl.parentNode.insertBefore(showMoreBtn, textEl.nextSibling);

            showMoreBtn.addEventListener('click', function (e) {
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

// Handle browser back/forward
window.addEventListener('popstate', function (e) {
    fetchPageContent(window.location.href);
});