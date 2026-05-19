let contentFile = null,
  styleFile = null;

let contentURL = null,
  styleURL = null;

let startTime = null;

/* ── Slider ── */

document.getElementById("alphaSlider").addEventListener("input", function () {
  document.getElementById("alphaVal").textContent = parseFloat(
    this.value,
  ).toFixed(2);
});

/* ── Zone setup ── */

function setupZone(inputId, zoneId, previewId, isContent) {
  const input = document.getElementById(inputId);
  const zone = document.getElementById(zoneId);
  const preview = document.getElementById(previewId);

  zone.addEventListener("click", (e) => {
    if (e.target !== input) {
      input.click();
    }
  });

  zone.addEventListener("dragover", (e) => {
    e.preventDefault();
    zone.classList.add("drag-active");
  });

  zone.addEventListener("dragleave", () => {
    zone.classList.remove("drag-active");
  });

  zone.addEventListener("drop", (e) => {
    e.preventDefault();

    zone.classList.remove("drag-active");

    const file = e.dataTransfer.files[0];

    if (file && file.type.startsWith("image/")) {
      handleFile(file, zone, preview, isContent);
    }
  });

  input.addEventListener("change", (e) => {
    const file = e.target.files[0];

    if (file) {
      handleFile(file, zone, preview, isContent);
    }
  });
}

function handleFile(file, zone, preview, isContent) {
  const reader = new FileReader();

  reader.onload = (ev) => {
    const url = ev.target.result;

    preview.style.backgroundImage = `url(${url})`;

    zone.classList.add("filled");

    if (isContent) {
      contentFile = file;
      contentURL = url;
    } else {
      styleFile = file;
      styleURL = url;
    }

    checkReady();
  };

  reader.readAsDataURL(file);
}

function checkReady() {
  document.getElementById("generateBtn").disabled = !(contentFile && styleFile);
}

/* ── Steps ── */

function setStep(n) {
  for (let i = 1; i <= 4; i++) {
    const el = document.getElementById("step" + i);

    el.className = "step" + (i < n ? " done" : i === n ? " active" : "");
  }
}

/* ── Generate ── */

async function generate() {
  if (!contentFile || !styleFile) return;

  const btn = document.getElementById("generateBtn");

  const progress = document.getElementById("progressWrap");

  const empty = document.getElementById("emptyState");

  const result = document.getElementById("resultArea");

  btn.disabled = true;

  progress.classList.add("active");

  result.classList.remove("visible");

  empty.style.display = "none";

  startTime = Date.now();

  const alpha = document.getElementById("alphaSlider").value;

  const size = document.getElementById("sizeSelect").value;

  setStep(1);

  const stepTimer = setTimeout(() => setStep(2), 800);

  const stepTimer2 = setTimeout(() => setStep(3), 1800);

  const formData = new FormData();

  formData.append("content", contentFile);
  formData.append("style", styleFile);
  formData.append("alpha", alpha);
  formData.append("size", size);

  try {
    const res = await fetch("/stylize", {
      method: "POST",
      body: formData,
    });

    const data = await res.json();

    clearTimeout(stepTimer);
    clearTimeout(stepTimer2);

    if (data.error) {
      throw new Error(data.error);
    }

    setStep(4);

    const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);

    const ts = Date.now();

    document.getElementById("resultImg").src = data.result_url + "?t=" + ts;

    document.getElementById("downloadBtn").href = data.result_url;

    document.getElementById("cmpContent").src = contentURL;

    document.getElementById("cmpStyle").src = styleURL;

    document.getElementById("cmpResult").src = data.result_url + "?t=" + ts;

    document.getElementById("resultAlpha").textContent =
      parseFloat(alpha).toFixed(2);

    document.getElementById("resultSize").textContent = size + " × " + size;

    document.getElementById("resultTime").textContent = elapsed + "s";

    setTimeout(() => {
      progress.classList.remove("active");

      result.classList.add("visible");

      result.scrollIntoView({
        behavior: "smooth",
        block: "nearest",
      });

      showToast("Style transfer complete — " + elapsed + "s", true);
    }, 400);
  } catch (err) {
    clearTimeout(stepTimer);
    clearTimeout(stepTimer2);

    progress.classList.remove("active");

    empty.style.display = "";

    showToast("Error: " + err.message, false);
  }

  btn.disabled = false;
}

/* ── Reset ── */

function resetAll() {
  contentFile = styleFile = contentURL = styleURL = null;

  ["contentPreview", "stylePreview"].forEach(
    (id) => (document.getElementById(id).style.backgroundImage = ""),
  );

  ["contentZone", "styleZone"].forEach((id) =>
    document.getElementById(id).classList.remove("filled", "drag-active"),
  );

  ["contentInput", "styleInput"].forEach(
    (id) => (document.getElementById(id).value = ""),
  );

  document.getElementById("resultArea").classList.remove("visible");

  document.getElementById("emptyState").style.display = "";

  document.getElementById("generateBtn").disabled = true;

  document.getElementById("progressWrap").classList.remove("active");

  window.scrollTo({
    top: 0,
    behavior: "smooth",
  });
}

/* ── Toast ── */

function showToast(msg, success = false) {
  const t = document.getElementById("toast");

  t.textContent = msg;

  t.className = "toast" + (success ? " toast-success" : "") + " visible";

  setTimeout(() => {
    t.className = "toast" + (success ? " toast-success" : "");
  }, 3500);
}

/* ── Init ── */

setupZone("contentInput", "contentZone", "contentPreview", true);

setupZone("styleInput", "styleZone", "stylePreview", false);
