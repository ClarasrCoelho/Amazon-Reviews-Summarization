const API = "http://127.0.0.1:8000";

/* -------- utils -------- */
function qs(name) {
  return new URLSearchParams(window.location.search).get(name);
}

function escapeHtml(s) {
  return String(s)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function fmtMaybe(n, digits = 3) {
  if (n === null || n === undefined || n === "") return "—";
  const x = Number(n);
  return Number.isFinite(x) ? x.toFixed(digits) : String(n);
}

async function apiGet(path, timeoutMs = 8000) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);

  try {
    const r = await fetch(`${API}${path}`, {
      signal: ctrl.signal,
      cache: "no-store"
    });

    const txt = await r.text();

    if (!r.ok) return { error: `API ${r.status}: ${txt.slice(0, 300)}` };
    if (!txt) return {};

    try {
      return JSON.parse(txt);
    } catch (e) {
      return { error: `Resposta não é JSON válido em ${path}: ${txt.slice(0, 200)}` };
    }
  } catch (e) {
    if (e && e.name === "AbortError") {
      return { error: `Timeout a chamar ${path} (>${timeoutMs}ms). Backend/S3 lento.` };
    }
    return { error: `Falha a ligar ao backend (${API}). O uvicorn está a correr? (${e?.message || e})` };
  } finally {
    clearTimeout(timer);
  }
}

async function apiPost(path, body, timeoutMs = 15000) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);

  try {
    const r = await fetch(`${API}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: ctrl.signal,
      cache: "no-store"
    });

    const txt = await r.text();
    if (!r.ok) return { error: `API ${r.status}: ${txt.slice(0, 400)}` };
    if (!txt) return {};

    try {
      return JSON.parse(txt);
    } catch {
      return { error: `Resposta não é JSON válido no POST ${path}: ${txt.slice(0, 200)}` };
    }
  } catch (e) {
    if (e && e.name === "AbortError") return { error: `Timeout no POST ${path} (>${timeoutMs}ms)` };
    return { error: `POST falhou (${e?.name || "erro"}): ${e?.message || String(e)}` };
  } finally {
    clearTimeout(timer);
  }
}

/* -------- capas -------- */
function coverUrlFromBook(b) {
  const asin = String(b?.asin || "").trim();
  const im = String(b?.imUrl || b?.imageURLHighRes || b?.imageURL || "").trim();
  if (im) return im;
  if (asin) return `https://covers.openlibrary.org/b/isbn/${encodeURIComponent(asin)}-M.jpg?default=false`;
  return "";
}

/* =========================================================
   OVERRIDES LOCAIS (rating + sentimento) — impede o poll de “voltar atrás”
========================================================= */
function _ratingKey(asin) {
  return `rating_override_${String(asin || "").trim()}`;
}
function setRatingOverride(asin, avg, count) {
  try {
    const payload = { avg: Number(avg), count: Number(count), ts: Date.now() };
    localStorage.setItem(_ratingKey(asin), JSON.stringify(payload));
  } catch { /* ignore */ }
}
function getRatingOverride(asin) {
  try {
    const raw = localStorage.getItem(_ratingKey(asin));
    if (!raw) return null;
    const o = JSON.parse(raw);
    if (!o) return null;
    if (!Number.isFinite(Number(o.avg))) return null;
    if (!Number.isFinite(Number(o.count))) return null;
    return o;
  } catch {
    return null;
  }
}

// ✅ novo: sentimento
function _sentKey(asin) {
  return `sent_override_${String(asin || "").trim()}`;
}
function setSentOverride(asin, avg, count) {
  try {
    const payload = { avg: Number(avg), count: Number(count), ts: Date.now() };
    localStorage.setItem(_sentKey(asin), JSON.stringify(payload));
  } catch { /* ignore */ }
}
function getSentOverride(asin) {
  try {
    const raw = localStorage.getItem(_sentKey(asin));
    if (!raw) return null;
    const o = JSON.parse(raw);
    if (!o) return null;
    if (!Number.isFinite(Number(o.avg))) return null;
    if (!Number.isFinite(Number(o.count))) return null;
    return o;
  } catch {
    return null;
  }
}

/* =========================================================
   INDEX (HOME)
========================================================= */
function renderBookCard(b) {
  const asin = String(b.asin || "").trim();
  const title = b.title || ("Livro " + asin);
  const rating = (b.avg_rating ?? null);
  const reviews = (b.review_count ?? 0);
  const cover = coverUrlFromBook(b);

  const thumbHtml = cover
    ? `<div class="thumb">
         <img src="${cover}" alt="${escapeHtml(title)}" referrerpolicy="no-referrer"
           onerror="this.onerror=null; const p=this.parentElement; this.remove(); p.classList.add('coverfallback'); p.textContent='sem capa';" />
       </div>`
    : `<div class="thumb coverfallback">sem capa</div>`;

  return `
    <a class="bookcard" href="book.html?asin=${encodeURIComponent(asin)}">
      ${thumbHtml}
      <div class="meta">
        <div class="title">${escapeHtml(title)}</div>
        <div class="small" style="margin-top:6px">
          <span class="badge">${escapeHtml(reviews)} reviews</span>
          ${rating !== null ? `<span class="badge">⭐ ${fmtMaybe(rating,2)}</span>` : ""}
        </div>
      </div>
    </a>
  `;
}

async function loadBooks() {
  const grid = document.getElementById("grid");
  if (!grid) return;

  const q = (qs("q") || "").trim();
  const box = document.getElementById("q");
  if (box) box.value = q;

  const url = q ? `/books?q=${encodeURIComponent(q)}&limit=60` : `/books?limit=60`;
  const data = await apiGet(url);

  grid.innerHTML = "";

  if (data?.error) {
    grid.innerHTML = `<div class="card" style="grid-column:1/-1">${escapeHtml(data.error)}</div>`;
    return;
  }

  const items = Array.isArray(data) ? data : (data.items || []);
  if (!items.length) {
    grid.innerHTML = `<div class="card" style="grid-column:1/-1">Sem resultados.</div>`;
    return;
  }

  for (const b of items) {
    const wrapper = document.createElement("div");
    wrapper.innerHTML = renderBookCard(b);
    grid.appendChild(wrapper.firstElementChild);
  }
}

function doSearch() {
  const q = (document.getElementById("q")?.value || "").trim();
  window.location.href = q ? `index.html?q=${encodeURIComponent(q)}` : "index.html";
}

/* =========================================================
   BOOK PAGE
========================================================= */
let pollTimer = null;

function renderReviews(reviews) {
  const box = document.getElementById("reviews");
  if (!box) return;
  box.innerHTML = "";

  if (!reviews || !reviews.length) {
    box.innerHTML = `<div class="small">Ainda não há reviews live. Sê a primeira 🙂</div>`;
    return;
  }

  for (const r of reviews) {
    const el = document.createElement("div");
    el.className = "review";
    const who = r.user || "user";
    const ts = r.ts ? new Date(r.ts * 1000).toLocaleString() : "";
    const hasRating = r.rating !== null && r.rating !== undefined && r.rating !== "";
    const ratingTxt = hasRating ? `— ⭐ ${escapeHtml(String(r.rating))}` : "";

    el.innerHTML = `
      <div class="who">${escapeHtml(who)} ${ratingTxt}</div>
      <div class="txt">${escapeHtml(r.text || "")}</div>
      <div class="ts">${escapeHtml(ts)}</div>
    `;
    box.appendChild(el);
  }
}

async function loadBookPage() {
  const asin = qs("asin");
  if (!asin) return;

  localStorage.setItem("last_asin", asin);

  const ml = document.getElementById("modelsLink");
  if (ml) ml.href = `models.html?asin=${encodeURIComponent(asin)}`;

  const data = await apiGet(`/books/${encodeURIComponent(asin)}`);
  if (data?.error) {
    const main = document.getElementById("main");
    if (main) main.innerHTML = `<div class="card">${escapeHtml(data.error)}</div>`;
    return;
  }

  const book = data.book || {};
  const title = book.title || asin;

  const t = document.getElementById("title");
  if (t) t.textContent = title;

  const a = document.getElementById("asin");
  if (a) a.textContent = asin;

  // ✅ sentimento: mostrar global (batch + live), com fallback
  const sentEl = document.getElementById("sent");
  const sentStats = data.sentiment_stats || {};
  const backendSent =
    (sentStats.global_sentiment !== null && sentStats.global_sentiment !== undefined)
      ? sentStats.global_sentiment
      : (book.avg_sentiment ?? "-");

  const ovS = getSentOverride(asin);
  const shownSent = ovS ? ovS.avg : backendSent;

  if (sentEl) sentEl.textContent = fmtMaybe(shownSent, 6);

  const stats = data.rating_stats || {};
  const ratingEl = document.getElementById("rating");
  const countEl = document.getElementById("count");

  const backendAvg =
    (stats.global_avg !== null && stats.global_avg !== undefined)
      ? stats.global_avg
      : (book.avg_rating ?? null);

  const backendCount =
    (stats.global_count !== null && stats.global_count !== undefined)
      ? stats.global_count
      : (book.review_count ?? 0);

  const ov = getRatingOverride(asin);
  const shownAvg = ov ? ov.avg : backendAvg;
  const shownCount = ov ? ov.count : backendCount;

  if (ratingEl) ratingEl.textContent = fmtMaybe(shownAvg, 6);
  if (countEl) countEl.textContent = String(shownCount);

  const cover = document.getElementById("cover");
  if (cover) {
    const coverUrl = coverUrlFromBook({ asin, ...book });
    cover.innerHTML = coverUrl
      ? `<img src="${coverUrl}" alt="${escapeHtml(title)}" referrerpolicy="no-referrer"
          onerror="this.onerror=null; this.style.display='none';" />`
      : "";
  }

  const sb = document.getElementById("summarybox");
  if (sb) sb.textContent = data.summary || "(Sem resumo.)";

  renderReviews(data.user_reviews || []);
  if (!pollTimer) pollTimer = setInterval(loadBookPage, 2500);
}

// ✅ proxy sentimento (igual ao backend): rating 1..5 -> [-1, 1]
function sentimentProxyFromRating(r) {
  const x = Number(r);
  if (!Number.isFinite(x) || x < 1 || x > 5) return null;
  return (x - 3) / 2;
}

async function submitReview() {
  if (document.activeElement) document.activeElement.blur();

  const asin = qs("asin");
  const user = (document.getElementById("user")?.value || "local-user").trim();
  const text = (document.getElementById("text")?.value || "").trim();
  const ratingStr = (document.getElementById("rating_in")?.value || "").trim();
  const btn = document.getElementById("submitBtn");

  const rating = Number(String(ratingStr).replace(",", "."));
  if (!text) return alert("Escreve uma review primeiro.");
  if (!Number.isFinite(rating) || rating < 1 || rating > 5)
    return alert("Rating obrigatório (1..5).");

  btn.disabled = true;
  btn.textContent = "A submeter...";

  const resp = await apiPost(
    `/books/${encodeURIComponent(asin)}/reviews`,
    { user, rating, text }
  );

  btn.disabled = false;
  btn.textContent = "Submeter";

  if (resp?.error) {
    alert(resp.error);
    return;
  }

  // ✅ update imediato do rating + override
  try {
    const ratingEl = document.getElementById("rating");
    const countEl = document.getElementById("count");

    const curAvg = parseFloat((ratingEl?.textContent || "").replace(",", "."));
    const curCount = parseInt((countEl?.textContent || "").trim(), 10);

    if (Number.isFinite(curAvg) && Number.isFinite(curCount) && curCount >= 0) {
      const newCount = curCount + 1;
      const newAvg = ((curAvg * curCount) + rating) / newCount;

      if (ratingEl) ratingEl.textContent = newAvg.toFixed(6);
      if (countEl) countEl.textContent = String(newCount);

      setRatingOverride(asin, newAvg, newCount);
    }
  } catch { /* ignore */ }

  // ✅ update imediato do sentimento + override
  try {
    const sentEl = document.getElementById("sent");
    const sentVal = sentimentProxyFromRating(rating); // [-1,1]
    if (sentVal !== null) {
      const curSent = parseFloat((sentEl?.textContent || "").replace(",", "."));

      // Para contar, usamos o mesmo N que o rating no topo (global_count),
      // porque o backend também usa review_count como base_count.
      const countEl = document.getElementById("count");
      const curCount = parseInt((countEl?.textContent || "").trim(), 10);

      if (Number.isFinite(curSent) && Number.isFinite(curCount) && curCount >= 0) {
        // ⚠️ atenção: o count aqui já foi incrementado no bloco do rating.
        // Vamos inferir o "count antigo" a partir do +1.
        const newCount = curCount;
        const oldCount = Math.max(0, newCount - 1);

        // newAvg = (oldAvg*oldCount + sentVal)/newCount
        const newSent = (oldCount === 0)
          ? sentVal
          : ((curSent * oldCount) + sentVal) / newCount;

        if (sentEl) sentEl.textContent = Number(newSent).toFixed(6);
        setSentOverride(asin, newSent, newCount);
      }
    }
  } catch { /* ignore */ }

  document.getElementById("text").value = "";

  await loadBookPage();
  if (!pollTimer) pollTimer = setInterval(loadBookPage, 2500);
}

/* =========================================================
   MODELS PAGE
========================================================= */
function row(label, value) {
  const v = (value === null || value === undefined || value === "") ? "—" : value;
  return `
    <div class="krow">
      <div class="k">${escapeHtml(label)}</div>
      <div class="v">${escapeHtml(String(v))}</div>
    </div>
  `;
}

function card(title, innerHtml) {
  return `
    <div class="card">
      <h3 class="section-title" style="margin-top:0">${escapeHtml(title)}</h3>
      ${innerHtml}
    </div>
  `;
}

function renderModelsUI(data) {
  const box = document.getElementById("metricsBox");
  if (!box) return;

  if (!data || data.error) {
    box.innerHTML = `<div class="card">${escapeHtml(data?.error || "Sem dados.")}</div>`;
    return;
  }

  const asin = data.asin || "";
  const servingOk = !!data.serving_available;

  const head = `
    <div class="card" style="margin-bottom:12px">
      <div style="font-weight:900;font-size:22px">Modelos & Resultados</div>
      <div class="small" style="margin-top:6px">
        ASIN: <b>${escapeHtml(asin)}</b> · serving: ${servingOk ? "✅" : "❌"}
      </div>
      ${data.warning ? `<div class="small" style="color:#b42318;margin-top:6px">${escapeHtml(data.warning)}</div>` : ""}
    </div>
  `;

  const m = data.metrics || {};
  const rating = m.rating || {};
  const streaming = m.streaming || {};
  const sent = m.sentiment || {};
  const sentEval = m.sentiment_eval || {};
  const sumQ = m.summary_quality || {};
  const sumEval = m.summary_eval || {};
  const sums = data.summaries || {};

  const cRating = card("Rating", `
    <div class="kv">
      ${row("Média (dataset)", fmtMaybe(rating.base_avg, 3))}
      ${row("N (dataset)", rating.base_count ?? "—")}
      ${row("N live válidas", rating.live_count_valid ?? "—")}
      ${row("Média global", fmtMaybe(rating.global_avg, 3))}
      ${row("Δ rating", fmtMaybe(rating.delta_rating, 6))}
    </div>
  `);

  const cStream = card("Streaming (live)", `
    <div class="kv">
      ${row("Reviews live (total)", streaming.user_reviews_total ?? "—")}
      ${row("Reviews com texto", streaming.user_reviews_with_text ?? "—")}
    </div>
  `);

  // ✅ Sentimento agora com global
  let sentEvalHtml = `<div class="small" style="margin-top:8px;color:#6b7280">${escapeHtml(sentEval.note || sentEval.status || "—")}</div>`;
  if (sentEval.status === "ok" && sentEval.proxy_from_stars) {
    const p = sentEval.proxy_from_stars;
    sentEvalHtml = `
      <div class="kv" style="margin-top:10px">
        ${row("Precision", fmtMaybe(p.precision, 3))}
        ${row("Recall", fmtMaybe(p.recall, 3))}
        ${row("F1", fmtMaybe(p.f1, 3))}
        ${row("Accuracy", fmtMaybe(p.accuracy, 3))}
        ${row("Support", p.support ?? "—")}
      </div>
      <div class="small" style="margin-top:8px;color:#6b7280">${escapeHtml(sentEval.note || "")}</div>
    `;
  }

  const cSent = card("Sentimento", `
    <div class="kv">
      ${row("Sentimento (dataset)", fmtMaybe(sent.base_sentiment, 6))}
      ${row("N (dataset)", sent.base_count ?? "—")}
      ${row("N live válidas", sent.live_count_valid ?? "—")}
      ${row("Sentimento global", fmtMaybe(sent.global_sentiment, 6))}
      ${row("Δ sentimento", fmtMaybe(sent.delta_sentiment, 6))}
    </div>
    ${sentEvalHtml}
  `);

  let sumQHtml = `
    <div class="kv">
      ${row("Palavras (resumo)", sumQ.summary_words ?? "—")}
      ${row("Palavras (fonte)", sumQ.source_words ?? "—")}
      ${row("Compression", fmtMaybe(sumQ.compression_ratio, 3))}
      ${row("Redundância", fmtMaybe(sumQ.redundancy_score, 3))}
      ${row("Novidade", (sumQ.novelty_score === null || sumQ.novelty_score === undefined) ? "—" : fmtMaybe(sumQ.novelty_score, 3))}
      ${row("Freshness (s)", (sumQ.freshness_seconds === null || sumQ.freshness_seconds === undefined) ? "—" : fmtMaybe(sumQ.freshness_seconds, 0))}
    </div>
  `;
  if (sumQ.status && sumQ.status !== "ok") {
    sumQHtml += `<div class="small" style="margin-top:8px;color:#6b7280">${escapeHtml(sumQ.note || sumQ.status)}</div>`;
  }
  const cSumQ = card("Qualidade do resumo", sumQHtml);

  let rougeHtml = `<div class="small" style="color:#6b7280">${escapeHtml(sumEval.note || "—")}</div>`;
  if (sumEval.rouge_proxy) {
    const r = sumEval.rouge_proxy;
    rougeHtml = `
      <div class="kv">
        ${row("ROUGE-1 (F1)", fmtMaybe(r.rouge1_f1, 3))}
        ${row("ROUGE-2 (F1)", fmtMaybe(r.rouge2_f1, 3))}
        ${row("ROUGE-L (F1)", fmtMaybe(r.rougeL_f1, 3))}
      </div>
      <div class="small" style="margin-top:8px;color:#6b7280">${escapeHtml(sumEval.note || "")}</div>
    `;
  }
  const cRouge = card("Avaliação do resumo (ROUGE)", rougeHtml);

  const cSummaries = card("Resumos", `
    <div class="small" style="margin-bottom:6px">Resumo (batch / dataset)</div>
    <div class="summarybox" style="margin-bottom:12px">${escapeHtml(sums.batch_summary || "(sem resumo batch)")}</div>

    <div class="small" style="margin-bottom:6px">Resumo (serving / final)</div>
    <div class="summarybox">${escapeHtml(sums.serving_summary || "(ainda não há resumo final)")}</div>
  `);

  box.innerHTML = head + `
    <div class="metrics-grid">
      ${cRating}
      ${cStream}
      ${cSent}
      ${cSumQ}
      ${cRouge}
      <div style="grid-column:1/-1">${cSummaries}</div>
    </div>
  `;
}

/* ---------- eval cache (FRONT) ---------- */
let evalCache = null;
let evalCacheTs = 0;

async function getEvalCached(maxAgeMs = 60000) {
  const now = Date.now();
  if (evalCache && (now - evalCacheTs) < maxAgeMs) return evalCache;
  evalCache = await apiGet(`/eval`, 12000);
  evalCacheTs = now;
  return evalCache;
}

function renderModelsUI_WithBatch(data, evalBatch) {
  const box = document.getElementById("metricsBox");
  if (!box) return;

  renderModelsUI(data);

  let m = null;

  if (evalBatch && typeof evalBatch === "object" && !evalBatch.error) {
    if (evalBatch.sentiment_eval_full_batch && typeof evalBatch.sentiment_eval_full_batch === "object") {
      m = evalBatch.sentiment_eval_full_batch;
    } else if ("f1" in evalBatch || "accuracy" in evalBatch || "precision" in evalBatch || "recall" in evalBatch) {
      m = evalBatch;
    }
  }

  const html = (!m) ? `
    <div class="card" style="margin-bottom:12px">
      <h3 class="section-title" style="margin-top:0">Avaliação global (batch)</h3>
      <div class="small" style="opacity:.85">Sem métricas batch.</div>
      ${evalBatch?.error ? `<div class="small" style="opacity:.75;margin-top:6px">${escapeHtml(String(evalBatch.error))}</div>` : ``}
      ${evalBatch ? `<div class="small" style="opacity:.65;margin-top:6px">(/eval: ${escapeHtml(JSON.stringify(evalBatch).slice(0, 180))}${JSON.stringify(evalBatch).length>180 ? "…" : ""})</div>` : ``}
    </div>
  ` : `
    <div class="card" style="margin-bottom:12px">
      <h3 class="section-title" style="margin-top:0">Avaliação global (batch)</h3>
      <div class="kv">
        ${row("F1", fmtMaybe(m.f1, 3))}
        ${row("Accuracy", fmtMaybe(m.accuracy, 3))}
        ${row("Precision", fmtMaybe(m.precision, 3))}
        ${row("Recall", fmtMaybe(m.recall, 3))}
        ${row("Support", m.support ?? "—")}
      </div>
    </div>
  `;

  box.innerHTML = html + box.innerHTML;
}

async function loadModelsLiveUI() {
  const asin = qs("asin") || localStorage.getItem("last_asin");
  const box = document.getElementById("metricsBox");
  if (!box) return;

  if (!asin) {
    box.innerHTML = `<div class="card">Falta ?asin=...</div>`;
    return;
  }

  box.innerHTML = `<div class="card">A carregar métricas…</div>`;

  const data = await apiGet(`/models/${encodeURIComponent(asin)}`, 12000);
  const evalBatch = await getEvalCached(60000);

  renderModelsUI_WithBatch(data, evalBatch);

  if (!data?.serving_available) {
    let tries = 0;
    const t = setInterval(async () => {
      tries++;
      const st = await apiGet(`/serving_status/${encodeURIComponent(asin)}`, 8000);
      if (st?.ready) {
        const latest = await apiGet(`/models/${encodeURIComponent(asin)}`, 15000);
        renderModelsUI_WithBatch(latest, evalBatch);
        clearInterval(t);
      }
      if (tries >= 12) clearInterval(t);
    }, 5000);
  }
}

/* ---------- init ---------- */
window.LABGDD = {
  loadBooks,
  doSearch,
  loadBookPage,
  submitReview,
  loadModelsLiveUI
};

document.addEventListener("DOMContentLoaded", () => {
  const path = (location.pathname || "").toLowerCase();

  if (path.endsWith("/index.html") || path.endsWith("/") || document.getElementById("grid")) {
    loadBooks();
  }

  if (path.endsWith("/book.html") || document.getElementById("summarybox")) {
    loadBookPage();
  }

  if (path.endsWith("/models.html") || document.getElementById("metricsBox")) {
    loadModelsLiveUI();
  }
});
