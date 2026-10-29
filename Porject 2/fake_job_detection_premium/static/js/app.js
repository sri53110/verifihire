const form = document.getElementById("analyze-form");
const titleInput = document.getElementById("job-title");
const descriptionInput = document.getElementById("job-description");
const charCount = document.getElementById("char-count");
const errorBox = document.getElementById("form-error");
const submitButton = document.getElementById("analyze-button");
const emptyState = document.getElementById("empty-state");
const resultContent = document.getElementById("result-content");

descriptionInput.addEventListener("input", () => {
  charCount.textContent = `${descriptionInput.value.length.toLocaleString()} / 20,000`;
});

async function checkHealth() {
  const label = document.getElementById("model-status");
  const dot = document.querySelector(".status-dot");
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    if (data.model_ready) {
      label.textContent = "ML model ready";
      dot.classList.add("ready");
    } else {
      label.textContent = "Model setup needed";
    }
  } catch (_) {
    label.textContent = "API unavailable";
  }
}
checkHealth();

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  hideError();
  const payload = Object.fromEntries(new FormData(form).entries());
  if (!payload.title.trim() && !payload.description.trim()) {
    showError("Enter a job title or description first.");
    return;
  }
  submitButton.disabled = true;
  submitButton.querySelector(".button-label").textContent = "Analyzing text…";
  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(payload)
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Analysis failed. Please try again.");
    renderResult(data);
    document.getElementById("result-panel").scrollIntoView({behavior:"smooth", block:"nearest"});
  } catch (error) {
    showError(error.message || "Could not reach the analysis service.");
  } finally {
    submitButton.disabled = false;
    submitButton.querySelector(".button-label").textContent = "Analyze job posting";
  }
});

function renderResult(data) {
  emptyState.hidden = true;
  resultContent.hidden = false;
  const isFake = data.prediction === 1;
  const banner = document.getElementById("result-banner");
  banner.className = `result-banner ${isFake ? "fake" : "legit"}`;
  document.getElementById("result-symbol").textContent = isFake ? "!" : "✓";
  document.getElementById("result-label").textContent = data.label;
  document.getElementById("result-explanation").textContent = data.explanation;

  const score = data.risk_score;
  document.getElementById("risk-score").textContent = score === null || score === undefined ? "N/A" : score;
  document.getElementById("risk-tag").textContent = score === null || score === undefined ? "SCORE UNAVAILABLE" :
    score >= 70 ? "HIGH MODEL SCORE" : score >= 40 ? "MODERATE MODEL SCORE" : "LOW MODEL SCORE";
  document.getElementById("risk-fill").style.width = `${Math.max(0, Math.min(100, score ?? 0))}%`;

  const signals = Array.isArray(data.warning_signals) ? data.warning_signals : [];
  document.getElementById("signal-count").textContent = `${signals.length} found`;
  const list = document.getElementById("signal-list");
  list.replaceChildren();
  if (!signals.length) {
    const item = document.createElement("div");
    item.className = "no-signals";
    item.textContent = "No listed warning phrases were detected. This does not establish that the posting is safe.";
    list.appendChild(item);
  } else {
    signals.forEach((signal) => {
      const item = document.createElement("div");
      item.className = "signal-item";
      const bullet = document.createElement("span");
      bullet.className = "signal-bullet";
      bullet.textContent = "!";
      const text = document.createElement("span");
      text.textContent = signal;
      item.append(bullet, text);
      list.appendChild(item);
    });
  }
  document.getElementById("recommendation-text").textContent = isFake
    ? "Do not pay fees or share sensitive details. Verify the company using its independently found official website and contact details."
    : "Still verify the company domain, recruiter identity, and interview process independently. A low model score is not a safety guarantee.";
}

document.getElementById("reset-button").addEventListener("click", () => {
  form.reset();
  charCount.textContent = "0 / 20,000";
  resultContent.hidden = true;
  emptyState.hidden = false;
  hideError();
  document.getElementById("analyzer").scrollIntoView({behavior:"smooth", block:"start"});
});

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
}
function hideError() {
  errorBox.textContent = "";
  errorBox.hidden = true;
}
