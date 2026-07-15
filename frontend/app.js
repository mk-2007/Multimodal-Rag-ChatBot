/* app.js — vanilla JS, no build step. Talks to the FastAPI backend on the
   same origin (backend/app.py serves this file, so no CORS setup needed). */

const els = {
  textInput: document.getElementById("query-text"),
  askBtn: document.getElementById("ask-btn"),
  askLabel: document.querySelector(".ask-label"),
  askSpinner: document.querySelector(".ask-spinner"),
  attachBtn: document.getElementById("attach-btn"),
  imageInput: document.getElementById("image-input"),
  imageChip: document.getElementById("image-chip"),
  imagePreview: document.getElementById("image-preview"),
  imageRemove: document.getElementById("image-remove"),
  answerZone: document.getElementById("answer-zone"),
  answerEmpty: document.getElementById("answer-empty"),
  inputCapsule: document.getElementById("input-capsule"),
  drawer: document.getElementById("drawer"),
  drawerTab: document.getElementById("drawer-tab"),
  drawerTabCount: document.getElementById("drawer-tab-count"),
  drawerClose: document.getElementById("drawer-close"),
  drawerList: document.getElementById("drawer-list"),
  beamLayer: document.getElementById("beam-layer"),
};

// Resolved hex values for SVG stroke (CSS vars don't resolve inside dynamically
// injected SVG the same way, so keep a plain lookup too) -- must match the
// --mod-text / --mod-table / --mod-figure values in style.css
const MODALITY_HEX = { text: "#9fc8e8", table: "#8fd6ae", figure: "#ff8552" };

let attachedFile = null;

/* ---------- textarea auto-grow ---------- */
els.textInput.addEventListener("input", () => {
  els.textInput.style.height = "auto";
  els.textInput.style.height = Math.min(els.textInput.scrollHeight, 200) + "px";
});

els.textInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    handleAsk();
  }
});

/* ---------- image attach ---------- */
els.attachBtn.addEventListener("click", () => els.imageInput.click());

els.imageInput.addEventListener("change", () => {
  const file = els.imageInput.files[0];
  if (!file) return;
  attachedFile = file;
  els.imagePreview.src = URL.createObjectURL(file);
  els.imageChip.hidden = false;
});

els.imageRemove.addEventListener("click", () => {
  attachedFile = null;
  els.imageInput.value = "";
  els.imageChip.hidden = true;
});

/* ---------- suggestion chips ---------- */
document.querySelectorAll(".suggestion").forEach((btn) => {
  btn.addEventListener("click", () => {
    els.textInput.value = btn.dataset.q;
    handleAsk();
  });
});

/* ---------- text formatting: math notation + light markdown ---------- */
// Converts the underscore/caret convention we ask Gemini for (d_model, x^2, W_i^Q)
// into real <sub>/<sup> HTML, plus minimal **bold**/`code`/paragraph handling.
// Input is escaped first, so this is safe to drop in via innerHTML.
function formatAnswerText(raw) {
  let html = escapeHtml(raw);

  // Superscript first isn't safe if a token has both _ and ^ (e.g. W_i^Q) --
  // process subscript, then superscript, so nested tokens resolve left-to-right.
  html = html.replace(
    /([A-Za-z][A-Za-z0-9]*)_(\{([^{}]+)\}|([A-Za-z0-9]+))/g,
    (_, base, __, braced, plain) => `${base}<sub>${braced || plain}</sub>`
  );
  html = html.replace(
    /([A-Za-z0-9][A-Za-z0-9<>\/]*)\^(\{([^{}]+)\}|(-?[A-Za-z0-9]+))/g,
    (_, base, __, braced, plain) => `${base}<sup>${braced || plain}</sup>`
  );

  // Minimal markdown: **bold**, `code`
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");

  // Paragraphs on blank lines, single newlines become <br>
  html = html
    .split(/\n\s*\n/)
    .map((p) => `<p>${p.replace(/\n/g, "<br>")}</p>`)
    .join("");

  return html;
}

/* ---------- ask flow ---------- */
els.askBtn.addEventListener("click", handleAsk);

async function handleAsk() {
  const text = els.textInput.value.trim();
  if (!text && !attachedFile) return;

  setLoading(true);
  clearBeams();

  try {
    let data;
    if (attachedFile) {
      const form = new FormData();
      form.append("file", attachedFile);
      form.append("question", text);
      const res = await fetch("/api/image-query", { method: "POST", body: form });
      data = await parseResponse(res);
      renderAnswer(text || "What does this image show?", data.answer, {
        imagePreviewUrl: els.imagePreview.src,
      });
    } else {
      const res = await fetch("/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: text }),
      });
      data = await parseResponse(res);
      renderAnswer(text, data.answer);
    }
    renderSources(data.sources || []);
  } catch (err) {
    renderError(err.message || String(err));
  } finally {
    setLoading(false);
  }
}

async function parseResponse(res) {
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.detail || `Request failed (${res.status})`);
  }
  return data;
}

function setLoading(isLoading) {
  els.askBtn.disabled = isLoading;
  els.askLabel.hidden = isLoading;
  els.askSpinner.hidden = !isLoading;
}

/* ---------- rendering ---------- */
function renderAnswer(askedText, answerText, opts = {}) {
  els.answerEmpty.hidden = true;
  const existing = els.answerZone.querySelector(".answer-card, .error-card");
  if (existing) existing.remove();

  const card = document.createElement("div");
  card.className = "answer-card";
  card.id = "answer-card";

  const asked = document.createElement("p");
  asked.className = "asked";
  if (opts.imagePreviewUrl) {
    asked.innerHTML = `<img class="attached-thumb" src="${opts.imagePreviewUrl}" alt=""> ${escapeHtml(askedText)}`;
  } else {
    asked.textContent = askedText;
  }

  const body = document.createElement("div");
  body.className = "body";
  body.innerHTML = formatAnswerText(answerText);

  card.appendChild(asked);
  card.appendChild(body);
  els.answerZone.appendChild(card);
}

function renderError(message) {
  els.answerEmpty.hidden = true;
  const existing = els.answerZone.querySelector(".answer-card, .error-card");
  if (existing) existing.remove();

  const card = document.createElement("div");
  card.className = "error-card";
  card.innerHTML = `<strong>Couldn't get an answer.</strong><br>${escapeHtml(message)}`;
  els.answerZone.appendChild(card);
}

function renderSources(sources) {
  els.drawerList.innerHTML = "";
  els.drawerTabCount.textContent = sources.length;

  sources.forEach((s, i) => {
    const card = document.createElement("div");
    card.className = "source-card";
    card.style.setProperty("--card-color", MODALITY_HEX[s.modality] || MODALITY_HEX.text);
    card.dataset.expanded = "false";

    const head = document.createElement("div");
    head.className = "source-card-head";
    head.innerHTML = `
      <span class="modality-dot"></span>
      <span class="source-ref">${escapeHtml(s.source_ref)}</span>
      <span class="source-score">${s.score.toFixed(3)}</span>
    `;

    const heading = document.createElement("p");
    heading.className = "source-heading";
    heading.textContent = truncate(s.heading_context, 90);

    const content = document.createElement("div");
    content.className = "source-content";
    if (s.image_url) {
      const img = document.createElement("img");
      img.className = "source-thumb";
      img.src = s.image_url;
      img.alt = s.heading_context || s.source_ref;
      content.appendChild(img);
    }
    const contentText = document.createElement("div");
    contentText.innerHTML = formatAnswerText(s.content);
    content.appendChild(contentText);

    card.appendChild(head);
    card.appendChild(heading);
    card.appendChild(content);
    card.addEventListener("click", () => {
      card.dataset.expanded = card.dataset.expanded === "true" ? "false" : "true";
    });

    els.drawerList.appendChild(card);
  });

  if (sources.length > 0) {
    openDrawer();
    requestAnimationFrame(() => requestAnimationFrame(drawBeams));
  }
}

/* ---------- drawer control ---------- */
function openDrawer() {
  els.drawer.dataset.open = "true";
  els.drawer.setAttribute("aria-hidden", "false");
  els.drawerTab.dataset.open = "true";
  els.drawerTab.setAttribute("aria-expanded", "true");
}
function closeDrawer() {
  els.drawer.dataset.open = "false";
  els.drawer.setAttribute("aria-hidden", "true");
  els.drawerTab.dataset.open = "false";
  els.drawerTab.setAttribute("aria-expanded", "false");
}
els.drawerTab.addEventListener("click", () => {
  const isOpen = els.drawer.dataset.open === "true";
  isOpen ? closeDrawer() : openDrawer();
});
els.drawerClose.addEventListener("click", closeDrawer);

/* ---------- attention beams (signature animation) ---------- */
function clearBeams() {
  els.beamLayer.innerHTML = "";
}

function drawBeams() {
  const answerCard = document.getElementById("answer-card");
  if (!answerCard) return;
  const sourceCards = Array.from(els.drawerList.querySelectorAll(".source-card"));
  if (sourceCards.length === 0) return;

  const originRect = answerCard.getBoundingClientRect();
  const origin = { x: originRect.right - 24, y: originRect.top + 24 };

  clearBeams();
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");

  sourceCards.forEach((card, i) => {
    const rect = card.getBoundingClientRect();
    // Only draw to cards actually visible in the viewport-sized drawer list
    if (rect.top < 0 || rect.top > window.innerHeight) return;
    const target = { x: rect.left, y: rect.top + rect.height / 2 };
    const color = card.style.getPropertyValue("--card-color");
    const scoreEl = card.querySelector(".source-score");
    const score = parseFloat(scoreEl ? scoreEl.textContent : "0.5");
    const strokeWidth = 0.8 + score * 2;

    const midX = (origin.x + target.x) / 2;
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    const d = `M ${origin.x} ${origin.y} C ${midX} ${origin.y}, ${midX} ${target.y}, ${target.x} ${target.y}`;
    path.setAttribute("d", d);
    path.setAttribute("class", "beam-path");
    path.style.stroke = color;
    path.style.strokeWidth = strokeWidth;
    svg.appendChild(path);

    setTimeout(() => path.classList.add("animate"), i * 90);
  });

  els.beamLayer.appendChild(svg);
  setTimeout(clearBeams, 4000);
}

window.addEventListener("resize", clearBeams);

/* ---------- utils ---------- */
function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}
function truncate(str, n) {
  if (!str) return "";
  return str.length > n ? str.slice(0, n - 1) + "…" : str;
}

/* ---------- startup status check ---------- */
(async function checkStatus() {
  try {
    const res = await fetch("/api/status");
    const data = await res.json();
    if (!data.ready) {
      renderError(
        data.error ||
          "Vector store not built yet. Run `python3 main.py` once from the project root, then reload this page."
      );
    }
  } catch {
    // backend not reachable at all -- leave the default empty state, ask flow will surface the real error
  }
})();
