/**
 * Fake News Detector - Client Side JavaScript Logic
 */

document.addEventListener('DOMContentLoaded', () => {
    // --- Elements ---
    const newsForm = document.getElementById('news-form');
    const newsInput = document.getElementById('news-text-input');
    const charCounter = document.getElementById('char-counter');
    const wordCounter = document.getElementById('word-counter');
    const btnSubmit = document.getElementById('btn-analyze');
    
    const resultPlaceholder = document.getElementById('result-placeholder');
    const loadingSpinner = document.getElementById('loading-spinner');
    const resultBox = document.getElementById('result-box');
    
    const statusBadge = document.getElementById('status-badge');
    const confidenceVal = document.getElementById('confidence-val');
    const progressBarFill = document.getElementById('progress-bar-fill');
    const probabilityVal = document.getElementById('probability-val');
    const wordCountVal = document.getElementById('word-count-val');
    const predictionTime = document.getElementById('prediction-time');

    // Preset buttons
    const presetBtns = document.querySelectorAll('.preset-btn');

    // Sample Texts
    const SAMPLES = {
        sample1: "NASA's James Webb Space Telescope discovers ancient galaxy forming stars at record speed in the early universe.",
        sample2: "Federal Reserve maintains benchmark interest rates amidst steady inflation indicators and low unemployment figures.",
        sample3: "Secret government lab creates weather control machine that causes devastating hurricanes on demand!",
        sample4: "Drinking boiled ocean water turns grey hair pitch black in three days according to underground scientists!"
    };

    // --- Word & Character Counter ---
    if (newsInput) {
        newsInput.addEventListener('input', () => {
            const val = newsInput.value;
            if (charCounter) charCounter.textContent = `${val.length} chars`;
            if (wordCounter) {
                const words = val.trim() ? val.trim().split(/\s+/).length : 0;
                wordCounter.textContent = `${words} words`;
            }
        });
    }

    // --- Preset Click Handler ---
    presetBtns.forEach(btn => {
        btn.addEventListener('click', (e) => {
            e.preventDefault();
            const key = btn.getAttribute('data-sample');
            if (SAMPLES[key] && newsInput) {
                newsInput.value = SAMPLES[key];
                newsInput.dispatchEvent(new Event('input'));
                newsInput.focus();
            }
        });
    });

    // --- AJAX Prediction Submit Handler ---
    if (newsForm) {
        newsForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const text = newsInput.value.strip ? newsInput.value.strip() : newsInput.value.trim();
            if (!text) {
                alert('Please enter or paste an article or headline to analyze.');
                return;
            }

            // Show loading state
            if (resultPlaceholder) resultPlaceholder.style.display = 'none';
            if (resultBox) resultBox.style.display = 'none';
            if (loadingSpinner) loadingSpinner.style.display = 'block';
            if (btnSubmit) btnSubmit.disabled = true;

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ text })
                });

                const data = await response.json();

                if (!response.ok || !data.success) {
                    throw new Error(data.error || 'Prediction failed.');
                }

                // Hide loading spinner
                if (loadingSpinner) loadingSpinner.style.display = 'none';
                if (resultBox) resultBox.style.display = 'block';

                // Update Result UI
                const isFake = data.label === 'Fake';
                
                if (statusBadge) {
                    statusBadge.textContent = isFake ? '⚠️ FAKE NEWS' : '✅ REAL NEWS';
                    statusBadge.className = `status-badge ${isFake ? 'fake' : 'real'}`;
                }

                if (confidenceVal) {
                    confidenceVal.textContent = `${data.confidence}%`;
                    confidenceVal.style.color = isFake ? 'var(--fake-color)' : 'var(--real-color)';
                }

                if (progressBarFill) {
                    progressBarFill.className = `progress-bar-fill ${isFake ? 'fake' : 'real'}`;
                    setTimeout(() => {
                        progressBarFill.style.width = `${data.confidence}%`;
                    }, 50);
                }

                if (probabilityVal) probabilityVal.textContent = data.probability;
                if (wordCountVal) wordCountVal.textContent = data.token_count;
                if (predictionTime) predictionTime.textContent = data.timestamp;

                // Dynamically append to history list if visible on dashboard
                addRecentCheckToDOM(data);

            } catch (err) {
                if (loadingSpinner) loadingSpinner.style.display = 'none';
                if (resultPlaceholder) {
                    resultPlaceholder.style.display = 'block';
                    resultPlaceholder.innerHTML = `<div style="color: var(--fake-color); font-weight:600;">Error: ${err.message}</div>`;
                }
            } finally {
                if (btnSubmit) btnSubmit.disabled = false;
            }
        });
    }

    // --- Helper to append recent check dynamically ---
    function addRecentCheckToDOM(data) {
        const recentContainer = document.getElementById('recent-checks-list');
        if (!recentContainer) return;

        const isFake = data.label === 'Fake';
        const itemHtml = `
            <div class="history-item">
                <div class="history-item-body">
                    <div class="history-item-text">"${data.snippet}"</div>
                    <div class="history-item-meta">
                        <span>${data.timestamp}</span>
                        <span>Confidence: <strong>${data.confidence}%</strong></span>
                    </div>
                </div>
                <span class="badge ${isFake ? 'fake' : 'real'}">${data.label}</span>
            </div>
        `;
        recentContainer.insertAdjacentHTML('afterbegin', itemHtml);
    }
});

// --- Delete item function for history page ---
async function deleteHistoryItem(id) {
    if (!confirm('Are you sure you want to delete this prediction entry?')) return;

    try {
        const response = await fetch(`/history/delete/${id}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        const data = await response.json();
        if (data.success) {
            const el = document.getElementById(`history-row-${id}`);
            if (el) {
                el.style.opacity = '0';
                setTimeout(() => el.remove(), 300);
            }
        } else {
            alert('Failed to delete item.');
        }
    } catch (e) {
        console.error(e);
        alert('Error deleting history item.');
    }
}
