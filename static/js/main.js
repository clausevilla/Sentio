// ===================================
// Sentio - Main JavaScript
// Handles: Mobile Menu, User Dropdown, Toasts, Smooth Scroll
// ===================================

document.addEventListener('DOMContentLoaded', function() {

    // ========================================
    // MOBILE MENU TOGGLE
    // ========================================
    const mobileToggle = document.getElementById('mobileMenuToggle');
    const navLinks = document.getElementById('navLinks');
    const menuIconOpen = document.getElementById('menuIconOpen');
    const menuIconClose = document.getElementById('menuIconClose');

    function openMobileMenu() {
        if (navLinks) {
            navLinks.classList.add('active');
            if (mobileToggle) mobileToggle.setAttribute('aria-expanded', 'true');
            if (menuIconOpen) menuIconOpen.style.display = 'none';
            if (menuIconClose) menuIconClose.style.display = 'block';
            document.body.style.overflow = 'hidden'; // Prevent background scroll
        }
    }

    function closeMobileMenu() {
        if (navLinks) {
            navLinks.classList.remove('active');
            if (mobileToggle) mobileToggle.setAttribute('aria-expanded', 'false');
            if (menuIconOpen) menuIconOpen.style.display = 'block';
            if (menuIconClose) menuIconClose.style.display = 'none';
            document.body.style.overflow = ''; // Restore scroll
        }
    }

    function toggleMobileMenu() {
        if (navLinks && navLinks.classList.contains('active')) {
            closeMobileMenu();
        } else {
            openMobileMenu();
        }
    }

    if (mobileToggle) {
        mobileToggle.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            toggleMobileMenu();
        });
    }

    // ========================================
    // USER DROPDOWN TOGGLE
    // ========================================
    const dropdownToggle = document.getElementById('userDropdownToggle');
    const dropdownMenu = document.getElementById('userDropdownMenu');

    function openDropdown() {
        if (dropdownMenu) {
            dropdownMenu.classList.add('show');
            if (dropdownToggle) dropdownToggle.classList.add('active');
        }
    }

    function closeDropdown() {
        if (dropdownMenu) {
            dropdownMenu.classList.remove('show');
            if (dropdownToggle) dropdownToggle.classList.remove('active');
        }
    }

    function toggleDropdown() {
        if (dropdownMenu && dropdownMenu.classList.contains('show')) {
            closeDropdown();
        } else {
            openDropdown();
        }
    }

    if (dropdownToggle) {
        dropdownToggle.addEventListener('click', function(e) {
            e.preventDefault();
            e.stopPropagation();
            toggleDropdown();
        });
    }

    // Close dropdown when clicking menu items
    if (dropdownMenu) {
        dropdownMenu.querySelectorAll('.dropdown-item').forEach(function(item) {
            item.addEventListener('click', function() {
                closeDropdown();
                // Also close mobile menu if on mobile
                if (window.innerWidth <= 768) {
                    closeMobileMenu();
                }
            });
        });
    }

    // ========================================
    // CLOSE MENUS ON OUTSIDE CLICK
    // ========================================
    document.addEventListener('click', function(e) {
        // Close dropdown if clicking outside
        if (dropdownToggle && dropdownMenu) {
            if (!dropdownToggle.contains(e.target) && !dropdownMenu.contains(e.target)) {
                closeDropdown();
            }
        }

        // Close mobile menu if clicking outside (on mobile only)
        if (navLinks && mobileToggle && window.innerWidth <= 768) {
            if (!navLinks.contains(e.target) && !mobileToggle.contains(e.target)) {
                closeMobileMenu();
            }
        }
    });

    // ========================================
    // CLOSE MOBILE MENU ON LINK CLICK
    // (but not on dropdown toggle)
    // ========================================
    if (navLinks) {
        // Only regular links, not the dropdown button
        navLinks.querySelectorAll('a:not(.dropdown-item)').forEach(function(link) {
            link.addEventListener('click', function() {
                if (window.innerWidth <= 768) {
                    closeMobileMenu();
                }
            });
        });
    }

    // ========================================
    // CLOSE MOBILE MENU ON WINDOW RESIZE
    // ========================================
    window.addEventListener('resize', function() {
        if (window.innerWidth > 768) {
            closeMobileMenu();
            closeDropdown();
        }
    });

    // ========================================
    // TOAST AUTO-DISMISS
    // ========================================
    const toasts = document.querySelectorAll('.toast');
    toasts.forEach(function(toast, index) {
        setTimeout(function() {
            if (toast && !toast.classList.contains('hiding')) {
                toast.classList.add('hiding');
                setTimeout(function() {
                    if (toast.parentElement) {
                        toast.remove();
                    }
                }, 300);
            }
        }, 3000 + (index * 200)); // Stagger multiple toasts
    });

    // ========================================
    // SMOOTH SCROLLING FOR ANCHOR LINKS
    // ========================================
    document.querySelectorAll('a[href^="#"]').forEach(function(anchor) {
        anchor.addEventListener('click', function(e) {
            const href = this.getAttribute('href');
            if (href !== '#' && document.querySelector(href)) {
                e.preventDefault();

                // Close mobile menu first
                closeMobileMenu();

                // Then scroll
                document.querySelector(href).scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });

    // ========================================
    // AUTO-HIDE ALERTS (non-toast alerts)
    // Excludes alerts with .alert-persistent class
    // ========================================
    const alerts = document.querySelectorAll('.alert:not(.toast):not(.alert-persistent)');
    alerts.forEach(function(alert) {
        setTimeout(function() {
            alert.style.transition = 'opacity 0.3s ease';
            alert.style.opacity = '0';
            setTimeout(function() {
                alert.remove();
            }, 300);
        }, 5000);
    });
});

// ========================================
// CSRF TOKEN HELPER (for Django forms)
// ========================================
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

function getCSRFToken() {
    return getCookie('csrftoken');
}