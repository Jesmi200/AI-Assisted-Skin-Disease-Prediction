// ---------------------------------------------------------------------
// Point this at your deployed backend (Hugging Face Space / Cloud Run /
// Render / Railway URL). No trailing slash.
// ---------------------------------------------------------------------
const BACKEND_URL =  "https://jesmi200-skin-disease-api.hf.space";
const form = document.getElementById("upload-form");
const imageInput = document.getElementById("image");
const filenameLabel = document.getElementById("filename");
const uploadCard = document.getElementById("upload-card");
const loading = document.getElementById("loading");
const resultSection = document.getElementById("result-section");
const flash = document.getElementById("flash");
const resetBtn = document.getElementById("reset-btn");

imageInput.addEventListener("change", () => {
  if (imageInput.files.length > 0) {
    filenameLabel.textContent = imageInput.files[0].name;
  }
});

function showFlash(message) {
  flash.textContent = message;
  flash.style.display = "block";
}

function hideFlash() {
  flash.style.display = "none";
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  hideFlash();

  const file = imageInput.files[0];
  if (!file) {
    showFlash("Please choose an image first.");
    return;
  }

  const previewUrl = URL.createObjectURL(file);

  const formData = new FormData();
  formData.append("image", file);

  uploadCard.style.display = "none";
  loading.style.display = "block";

  try {
    const response = await fetch(`${BACKEND_URL}/predict`, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Prediction failed.");
    }

    renderResult(data, previewUrl);
  } catch (err) {
    showFlash(`Error: ${err.message}`);
    uploadCard.style.display = "block";
  } finally {
    loading.style.display = "none";
  }
});

function renderResult(data, previewUrl) {
  document.getElementById("preview-img").src = previewUrl;

  // Top predictions list with confidence bars
  const list = document.getElementById("prediction-list");
  list.innerHTML = "";
  data.top_results.forEach((r) => {
    const li = document.createElement("li");
    li.innerHTML = `
      <div class="pred-row">
        <span class="pred-label">${r.label}</span>
        <span class="pred-conf">${r.confidence.toFixed(2)}%</span>
      </div>
      <div class="bar-bg">
        <div class="bar-fill" style="width: ${r.confidence}%;"></div>
      </div>
    `;
    list.appendChild(li);
  });

  document.getElementById("top-prediction-heading").textContent =
    `Predicted Condition: ${data.top_prediction.label}`;
  document.getElementById("confidence-tag").textContent =
    `Confidence: ${data.top_prediction.confidence.toFixed(2)}%`;

  const clinicalDiv = document.getElementById("clinical-info");
  if (data.clinical_info) {
    const c = data.clinical_info;
    clinicalDiv.innerHTML = `
      <h3>Description</h3><p>${c.description}</p>
      <h3>Common Symptoms</h3><p>${c.symptoms}</p>
      <h3>Possible Causes</h3><p>${c.causes}</p>
      <h3>Treatment</h3><p>${c.treatment}</p>
      <h3>Advice</h3><p>${c.advice}</p>
    `;
  } else {
    clinicalDiv.innerHTML = "<p>Clinical information unavailable for this class.</p>";
  }

  document.getElementById("gradcam-before").src = data.gradcam_before;
  document.getElementById("gradcam-after").src = data.gradcam_after;
  document.getElementById("disclaimer-box").textContent = data.disclaimer;

  resultSection.style.display = "block";
}

resetBtn.addEventListener("click", () => {
  form.reset();
  filenameLabel.textContent = "";
  resultSection.style.display = "none";
  uploadCard.style.display = "block";
  hideFlash();
});
