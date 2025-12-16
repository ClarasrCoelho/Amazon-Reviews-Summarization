const API = "http://127.0.0.1:8000";

function qs(name) {
  return new URLSearchParams(window.location.search).get(name);
}

function fmt(n) {
  if (n === null || n === undefined) return "";
  if (typeof n === "number") return n.toFixed(2);
  return String(n);
}

async function apiGet(path) {
  const r = await fetch(`${API}${path}`);
  return await r.json();
}

async function apiPost(path, body) {
  const r = await fetch(`${API}${path}`, {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify(body)
  });
  return await r.json();
}

/* ---------- INDEX (lista/pesquisa) ---------- */
async function loadBooks() {
  const q = qs("q") || "";
  const box = document.getElementById("q");
  if (box) box.value = q;

  const url = q ? `/books?query=${encodeURIComponent(q)}&limit=60` : `/books?limit=60`;
  const books = await apiGet(url);

  const grid = document.getElementById("grid");
  grid.innerHTML = "";

  if (!books || books.length === 0) {
    grid.innerHTML = `<div class="card" style="grid-column:1/-1">Sem resultados. Experimenta outro termo.</div>`;
    return;
  }

  for (const b of books) {
    const asin = b.asin;
    const title = b.title || asin;

    const rating = b.avg_rating ?? null;
    const reviews = b.review_count ?? 0;

    const el = document.createElement("div");
    el.className = "bookcard";
    el.onclick = () => window.location.href = `book.html?asin=${encodeURIComponent(asin)}`;

    const img = b.imUrl ? `<img src="${b.imUrl}" alt="">` : `<span class="small">sem capa</span>`;
    el.innerHTML = `
      <div class="thumb">${img}</div>
      <div class="meta">
        <div class="title">${title}</div>
        <div class="small">
          <span class="badge">ASIN: ${asin}</span>
          <span class="badge">${reviews} reviews</span>
          ${rating ? `<span class="badge">⭐ ${fmt(rating)}</span>` : ""}
        </div>
        <div class="small" style="margin-top:6px">
          Sentimento médio: <b>${b.avg_sentiment !== null && b.avg_sentiment !== undefined ? fmt(b.avg_sentiment) : "-"}</b>
        </div>
      </div>
    `;
    grid.appendChild(el);
  }
}

function doSearch() {
  const q = (document.getElementById("q").value || "").trim();
  window.location.href = q ? `index.html?q=${encodeURIComponent(q)}` : "index.html";
}

/* ---------- BOOK PAGE ---------- */
let pollTimer = null;

async function loadBookPage() {
  const asin = qs("asin");
  if (!asin) {
    document.getElementById("main").innerHTML = `<div class="card">Falta o asin na URL.</div>`;
    return;
  }

  const data = await apiGet(`/books/${encodeURIComponent(asin)}`);
  if (data.error) {
    document.getElementById("main").innerHTML = `<div class="card">${data.error}</div>`;
    return;
  }

  const book = data.book;
  const title = book.title || asin;

  document.getElementById("title").textContent = title;
  document.getElementById("asin").textContent = asin;
  document.getElementById("sent").textContent = (book.avg_sentiment ?? "-");
  document.getElementById("rating").textContent = (book.avg_rating ?? "-");
  document.getElementById("count").textContent = (book.review_count ?? 0);

  const cover = document.getElementById("cover");
  cover.innerHTML = book.imUrl ? `<img src="${book.imUrl}" alt="">` : "";

  // summary no topo das reviews
  const sb = document.getElementById("summarybox");
  sb.textContent = data.summary_for_reviews ? data.summary_for_reviews : "(Sem resumo disponível neste ASIN.)";

  renderReviews(data.user_reviews || []);
}

function renderReviews(reviews) {
  const box = document.getElementById("reviews");
  box.innerHTML = "";

  if (!reviews || reviews.length === 0) {
    box.innerHTML = `<div class="small">Ainda não há reviews adicionadas “em tempo real”. Sê a primeira 🙂</div>`;
    return;
  }

  for (const r of reviews) {
    const el = document.createElement("div");
    el.className = "review";
    const who = r.user || "user";
    const ts = r.ts ? new Date(r.ts * 1000).toLocaleString() : "";
    el.innerHTML = `
      <div class="who">${who} ${r.rating ? `— ⭐ ${r.rating}` : ""}</div>
      <div class="txt">${escapeHtml(r.text || "")}</div>
      <div class="ts">${ts}</div>
    `;
    box.appendChild(el);
  }
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&","&amp;")
    .replaceAll("<","&lt;")
    .replaceAll(">","&gt;")
    .replaceAll('"',"&quot;")
    .replaceAll("'","&#039;");
}

async function submitReview() {
  const asin = qs("asin");
  const user = (document.getElementById("user").value || "local-user").trim();
  const text = (document.getElementById("text").value || "").trim();
  const ratingStr = (document.getElementById("rating_in").value || "").trim();
  const rating = ratingStr ? Number(ratingStr) : null;

  if (!text) {
    alert("Escreve uma review primeiro.");
    return;
  }

  await apiPost("/reviews", {asin, text, rating, user});
  document.getElementById("text").value = "";

  // refresh imediato
  await loadBookPage();
}

function toggleLive() {
  const enabled = document.getElementById("live").checked;
  if (pollTimer) {
    clearInterval(pollTimer);
    pollTimer = null;
  }
  if (enabled) {
    pollTimer = setInterval(loadBookPage, 2500);
  }
}

/* ---------- MODELS ---------- */
async function loadModels() {
  const data = await apiGet("/models");
  document.getElementById("modelsjson").textContent = JSON.stringify(data, null, 2);
}

/* ---------- init ---------- */
window.LABGDD = {
  loadBooks,
  doSearch,
  loadBookPage,
  submitReview,
  toggleLive,
  loadModels,
};
