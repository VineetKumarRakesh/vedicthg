// ── VedAI script.js — Fixed: null-guards, per-conv docs, action buttons ────
"use strict";

const STORAGE_KEY = "vedai_sessions_v1";

// ── Safe element getter — prevents null crash if ID missing ─────────────────
function el(id) { return document.getElementById(id); }

const messagesEl    = el("messages");
const welcomeCard   = el("welcomeCard");
const inputEl       = el("messageInput");
const sendBtn       = el("sendBtn");
const newChatBtn    = el("newChatBtn");
const statusPill    = el("statusPill");
const chatScroll    = el("chatScroll");
const historyListEl = el("historyList");
const tempToggleBtn = el("tempChatBtn");
const tempBanner    = el("tempBanner");
const docChipsBar   = el("docChipsBar");
const docFileInput  = el("docFileInput");
const uploadStatus  = el("uploadStatus");
const shareBtn      = el("shareBtn");
const shareToast    = el("shareToast");
const shareLink     = el("shareLink");
const shareToastClose = el("shareToastClose");

// Modals
const renameBackdrop   = el("renameBackdrop");
const renameInput      = el("renameInput");
const renameCancelBtn  = el("renameCancelBtn");
const renameConfirmBtn = el("renameConfirmBtn");
const deleteBackdrop   = el("deleteBackdrop");
const deleteCancelBtn  = el("deleteCancelBtn");
const deleteConfirmBtn = el("deleteConfirmBtn");

// ── State ──────────────────────────────────────────────────────────────────
let sessions         = [];
let tempSessions     = [];
let currentSessionId = null;
let busy             = false;
let tempMode         = false;
let pendingRenameId  = null;
let pendingDeleteId  = null;

function getSessionList()    { return tempMode ? tempSessions : sessions; }
function setSessionList(arr) { if (tempMode) tempSessions = arr; else sessions = arr; }

// ── Persistence ─────────────────────────────────────────────────────────────
function loadSessions() {
  try { return JSON.parse(localStorage.getItem(STORAGE_KEY) || "[]"); }
  catch { return []; }
}
function saveSessions() {
  if (tempMode) return;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
}

// ── Temp chat ────────────────────────────────────────────────────────────────
function applyTempUI(enabled) {
  if (tempToggleBtn) {
    tempToggleBtn.classList.toggle("active", enabled);
    tempToggleBtn.setAttribute("aria-pressed", String(enabled));
  }
  if (tempBanner) tempBanner.style.display = enabled ? "flex" : "none";
}

function enableTempMode() {
  tempMode     = true;
  tempSessions = [];
  const s = makeFreshSession();
  tempSessions.unshift(s);
  currentSessionId = s.id;
  applyTempUI(true);
  renderHistory();
  renderMessages();
  setStatus("Temporary chat", "busy");
}

function disableTempMode() {
  tempMode     = false;
  tempSessions = [];
  sessions = loadSessions();
  if (sessions.length === 0) {
    const s = makeFreshSession();
    sessions.unshift(s);
    saveSessions();
  }
  currentSessionId = sessions[0].id;
  applyTempUI(false);
  renderHistory();
  renderMessages();
  setStatus("Ready", "ready");
  loadConfig();
  loadDocs();
}

if (tempToggleBtn) {
  tempToggleBtn.addEventListener("click", () => {
    if (tempMode) disableTempMode();
    else          enableTempMode();
  });
}

// ── Core helpers ─────────────────────────────────────────────────────────────
function makeId() {
  return (typeof crypto !== "undefined" && crypto.randomUUID)
    ? crypto.randomUUID()
    : Date.now().toString(36) + Math.random().toString(36).slice(2);
}

function formatTime(iso) {
  try {
    const d = new Date(iso);
    const diffMin = Math.floor((Date.now() - d) / 60000);
    if (diffMin < 1)  return "just now";
    if (diffMin < 60) return `${diffMin}m ago`;
    const diffH = Math.floor(diffMin / 60);
    if (diffH < 24)   return `${diffH}h ago`;
    return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  } catch { return ""; }
}

function shortTitle(text) {
  const clean = text.replace(/\s+/g, " ").trim();
  if (!clean) return "New conversation";
  return clean.length > 40 ? clean.slice(0, 40) + "…" : clean;
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function renderContent(text) {
  // Protect code + math blocks from mangling
  const stash = [];
  const PH = "\uFFF9PH";
  function protect(s) { stash.push(s); return `${PH}${stash.length - 1}${PH}`; }

  text = text.replace(/```[\w]*\n?([\s\S]*?)```/g,
    (_, c) => protect(`<pre><code>${escapeHtml(c.trim())}</code></pre>`));
  text = text.replace(/`([^`\n]+)`/g,
    (_, c) => protect(`<code>${escapeHtml(c)}</code>`));
  text = text.replace(/\\\[[\s\S]*?\\\]/g, m => protect(m));
  text = text.replace(/\$\$[\s\S]*?\$\$/g, m => protect(m));
  text = text.replace(/\\\([\s\S]*?\\\)/g, m => protect(m));

  // Bold/italic emphasis — skip HTML escape so we can inject tags
  let out = text
    .replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");

  out = out.replace(/\*\*([^*]+)\*\*/g,
    '<strong style="color:var(--text-primary);font-weight:600;">$1</strong>');
  out = out.replace(/\*([^*]+)\*/g, '<em>$1</em>');
  out = out.replace(/### (.+)/g, '<strong>$1</strong>');
  out = out.replace(/#### (.+)/g, '<em>$1</em>');
  out = out.replace(/\n/g, "<br>");

  // Restore stash
  const PHre = new RegExp(PH.replace(/[.*+?^${}()|[\]\\]/g,'\\$&') + "(\\d+)" + PH.replace(/[.*+?^${}()|[\]\\]/g,'\\$&'), 'g');
  out = out.replace(PHre, (_, i) => stash[parseInt(i)]);

  return out;
}

const _KATEX_OPTS = {
  delimiters: [
    { left: '$$', right: '$$', display: true  },
    { left: '$',  right: '$',  display: false },
    { left: '\\[', right: '\\]', display: true  },
    { left: '\\(', right: '\\)', display: false },
  ],
  throwOnError: false,
};

function renderMath(el) {
  if (typeof renderMathInElement === 'function') {
    try { renderMathInElement(el, _KATEX_OPTS); } catch(e) {}
  } else {
    (window._mathQueue = window._mathQueue || []).push(el);
  }
}

window.onKatexReady = function() {
  (window._mathQueue || []).forEach(e => {
    try { renderMathInElement(e, _KATEX_OPTS); } catch(e) {}
  });
  window._mathQueue = [];
  document.querySelectorAll('.message-bubble').forEach(e => {
    try { renderMathInElement(e, _KATEX_OPTS); } catch(e) {}
  });
};

function autoResize() {
  if (!inputEl) return;
  inputEl.style.height = "auto";
  inputEl.style.height = Math.min(inputEl.scrollHeight, 200) + "px";
}

function setStatus(text, kind = "ready") {
  if (statusPill) { statusPill.textContent = text; statusPill.dataset.state = kind; }
}

function scrollToBottom() {
  if (chatScroll) requestAnimationFrame(() => { chatScroll.scrollTop = chatScroll.scrollHeight; });
}

function getCurrentSession() {
  const list = getSessionList();
  return list.find(s => s.id === currentSessionId) || null;
}

function makeFreshSession() {
  return { id: makeId(), title: "New conversation", updatedAt: new Date().toISOString(), messages: [] };
}

function createSession() {
  const session = makeFreshSession();
  const list = getSessionList();
  list.unshift(session);
  currentSessionId = session.id;
  saveSessions();
  return session;
}

function ensureSession() {
  return getCurrentSession() || createSession();
}

// ── Typing indicator ──────────────────────────────────────────────────────────
const THINKING_PHRASES = [
  "Thinking about your question…",
  "Searching the web for answers…",
  "Reading search results…",
  "Analysing information…",
  "Cross-checking facts…",
  "Crafting a detailed response…",
  "Almost ready, please wait…",
  "Processing — large models take a moment…",
  "Still working, thank you for your patience…",
  "Finalising your answer…",
];
let thinkingInterval = null;
let thinkingPhaseIdx = 0;

function showTypingIndicator() {
  removeTypingIndicator();
  const row = document.createElement("div");
  row.className = "message-row assistant typing-row";
  row.id = "typingIndicator";

  const dot = document.createElement("div");
  dot.className = "avatar-dot";
  const dotImg = document.createElement("img");
  dotImg.src = "/static/assets/Signature.png";
  dotImg.alt = "VedAI";
  dotImg.className = "avatar-img";
  dot.appendChild(dotImg);
  row.appendChild(dot);

  const bubble = document.createElement("div");
  bubble.className = "message-bubble assistant typing-bubble";
  bubble.innerHTML = `
    <span class="typing-phrase" id="typingPhrase">${THINKING_PHRASES[0]}</span>
    <span class="typing-dots"><span></span><span></span><span></span></span>`;
  row.appendChild(bubble);
  if (messagesEl) messagesEl.appendChild(row);
  scrollToBottom();

  thinkingPhaseIdx = 0;
  thinkingInterval = setInterval(() => {
    thinkingPhaseIdx = (thinkingPhaseIdx + 1) % THINKING_PHRASES.length;
    const el = document.getElementById("typingPhrase");
    if (el) el.textContent = THINKING_PHRASES[thinkingPhaseIdx];
  }, 7000);
}

function removeTypingIndicator() {
  if (thinkingInterval) { clearInterval(thinkingInterval); thinkingInterval = null; }
  const el = document.getElementById("typingIndicator");
  if (el) el.remove();
}

// ── Rename modal ──────────────────────────────────────────────────────────────
function openRenameModal(sessionId) {
  const session = getSessionList().find(s => s.id === sessionId);
  if (!session || !renameBackdrop || !renameInput) return;
  pendingRenameId = sessionId;
  renameInput.value = session.title === "New conversation" ? "" : session.title;
  renameBackdrop.classList.add("open");
  renameInput.focus();
  renameInput.select();
}

function closeRenameModal() {
  if (renameBackdrop) renameBackdrop.classList.remove("open");
  pendingRenameId = null;
  if (renameInput) renameInput.value = "";
}

function confirmRename() {
  const newTitle = renameInput ? renameInput.value.trim() : "";
  if (!newTitle || !pendingRenameId) { closeRenameModal(); return; }
  const session = getSessionList().find(s => s.id === pendingRenameId);
  if (session) {
    session.title = newTitle;
    session.updatedAt = new Date().toISOString();
    saveSessions();
    renderHistory();
  }
  closeRenameModal();
}

if (renameCancelBtn)  renameCancelBtn.addEventListener("click", closeRenameModal);
if (renameConfirmBtn) renameConfirmBtn.addEventListener("click", confirmRename);
if (renameInput) {
  renameInput.addEventListener("keydown", e => {
    if (e.key === "Enter") { e.preventDefault(); confirmRename(); }
    if (e.key === "Escape") closeRenameModal();
  });
}
if (renameBackdrop) {
  renameBackdrop.addEventListener("click", e => { if (e.target === renameBackdrop) closeRenameModal(); });
}

// ── Delete modal ──────────────────────────────────────────────────────────────
function openDeleteModal(sessionId) {
  pendingDeleteId = sessionId;
  if (deleteBackdrop) deleteBackdrop.classList.add("open");
}
function closeDeleteModal() {
  if (deleteBackdrop) deleteBackdrop.classList.remove("open");
  pendingDeleteId = null;
}
function confirmDelete() {
  if (!pendingDeleteId) { closeDeleteModal(); return; }
  const list = getSessionList().filter(s => s.id !== pendingDeleteId);
  setSessionList(list);
  if (currentSessionId === pendingDeleteId) {
    const l = getSessionList();
    if (l.length === 0) createSession();
    else currentSessionId = l[0].id;
  }
  saveSessions();
  renderHistory();
  renderMessages();
  closeDeleteModal();
}
if (deleteCancelBtn)  deleteCancelBtn.addEventListener("click", closeDeleteModal);
if (deleteConfirmBtn) deleteConfirmBtn.addEventListener("click", confirmDelete);
if (deleteBackdrop) {
  deleteBackdrop.addEventListener("click", e => { if (e.target === deleteBackdrop) closeDeleteModal(); });
}
document.addEventListener("keydown", e => {
  if (e.key === "Escape") { closeRenameModal(); closeDeleteModal(); }
});

// ── History render ────────────────────────────────────────────────────────────
const iconPencil = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"/><path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"/></svg>`;
const iconTrash  = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6M14 11v6"/><path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/></svg>`;

function renderHistory() {
  if (!historyListEl) return;
  historyListEl.innerHTML = "";

  if (tempMode) {
    const notice = document.createElement("div");
    notice.className = "temp-history-notice";
    notice.textContent = "🔒 Temp chat — not saved";
    historyListEl.appendChild(notice);
    return;
  }

  const sorted = [...sessions].sort((a, b) => new Date(b.updatedAt) - new Date(a.updatedAt));

  if (sorted.length === 0) {
    const empty = document.createElement("div");
    empty.style.cssText = "font-size:12px;color:var(--text-tertiary);padding:8px 4px;";
    empty.textContent = "No conversations yet";
    historyListEl.appendChild(empty);
    return;
  }

  sorted.forEach(session => {
    const row = document.createElement("div");
    row.className = "history-item" + (session.id === currentSessionId ? " active" : "");

    const body = document.createElement("button");
    body.className = "history-item-body";
    body.innerHTML = `
      <span class="history-title">${escapeHtml(session.title)}</span>
      <span class="history-meta">${escapeHtml(formatTime(session.updatedAt))}</span>`;
    body.addEventListener("click", () => {
      currentSessionId = session.id;
      renderHistory();
      renderMessages();
      loadDocs(); // load docs for this specific conversation
    });

    const actions = document.createElement("div");
    actions.className = "history-actions";

    const renameBtn = document.createElement("button");
    renameBtn.className = "history-action-btn rename";
    renameBtn.title = "Rename";
    renameBtn.innerHTML = iconPencil;
    renameBtn.addEventListener("click", e => { e.stopPropagation(); openRenameModal(session.id); });

    const deleteBtn = document.createElement("button");
    deleteBtn.className = "history-action-btn delete";
    deleteBtn.title = "Delete";
    deleteBtn.innerHTML = iconTrash;
    deleteBtn.addEventListener("click", e => { e.stopPropagation(); openDeleteModal(session.id); });

    actions.appendChild(renameBtn);
    actions.appendChild(deleteBtn);
    row.appendChild(body);
    row.appendChild(actions);
    historyListEl.appendChild(row);
  });
}

// ── Message action buttons (copy, thumbs up/down, rewrite) ───────────────────
function buildActionBar(content) {
  const actions = document.createElement("div");
  actions.className = "msg-actions";

  // ── Copy ──────────────────────────────────────────────────────────────────
  const copyBtn = document.createElement("button");
  copyBtn.className = "msg-icon-btn";
  copyBtn.title = "Copy response";
  copyBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg><span class="btn-label">Copy</span>`;
  copyBtn.addEventListener("click", () => {
    if (!navigator.clipboard) { alert("Clipboard not available"); return; }
    navigator.clipboard.writeText(content).then(() => {
      copyBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg><span class="btn-label">Copied!</span>`;
      copyBtn.classList.add("active");
      setTimeout(() => {
        copyBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg><span class="btn-label">Copy</span>`;
        copyBtn.classList.remove("active");
      }, 2000);
    }).catch(() => { alert("Could not copy to clipboard"); });
  });

  // ── Thumbs Up ─────────────────────────────────────────────────────────────
  const thumbUpBtn = document.createElement("button");
  thumbUpBtn.className = "msg-icon-btn";
  thumbUpBtn.title = "Good response";
  thumbUpBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 9V5a3 3 0 0 0-3-3l-4 9v11h11.28a2 2 0 0 0 2-1.7l1.38-9a2 2 0 0 0-2-2.3H14z"/><path d="M7 22H4a2 2 0 0 1-2-2v-7a2 2 0 0 1 2-2h3"/></svg><span class="btn-label">Good</span>`;
  thumbUpBtn.addEventListener("click", () => {
    thumbUpBtn.classList.toggle("active");
    thumbDownBtn.classList.remove("active");
  });

  // ── Thumbs Down ───────────────────────────────────────────────────────────
  const thumbDownBtn = document.createElement("button");
  thumbDownBtn.className = "msg-icon-btn";
  thumbDownBtn.title = "Bad response";
  thumbDownBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 15v4a3 3 0 0 0 3 3l4-9V2H5.72a2 2 0 0 0-2 1.7l-1.38 9a2 2 0 0 0 2 2.3H10z"/><path d="M17 2h2.67A2.31 2.31 0 0 1 22 4v7a2.31 2.31 0 0 1-2.33 2H17"/></svg><span class="btn-label">Poor</span>`;
  thumbDownBtn.addEventListener("click", () => {
    thumbDownBtn.classList.toggle("active");
    thumbUpBtn.classList.remove("active");
  });

  // ── Rewrite ───────────────────────────────────────────────────────────────
  const rewriteBtn = document.createElement("button");
  rewriteBtn.className = "msg-icon-btn";
  rewriteBtn.title = "Rewrite response";
  rewriteBtn.innerHTML = `<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="1 4 1 10 7 10"/><path d="M3.51 15a9 9 0 1 0 .49-3.7"/></svg><span class="btn-label">Rewrite</span>`;
  rewriteBtn.addEventListener("click", () => {
    const session = getCurrentSession();
    if (!session || busy) return;
    const msgs = session.messages;
    // Find the assistant message that matches this content
    const idx = msgs.findLastIndex(m => m.role === "assistant" && m.content === content);
    if (idx > 0 && msgs[idx - 1]?.role === "user") {
      const userMsg = msgs[idx - 1].content;
      session.messages.splice(idx - 1, 2);
      saveSessions();
      renderMessages();
      sendMessage(userMsg);
    }
  });

  actions.appendChild(copyBtn);
  actions.appendChild(thumbUpBtn);
  actions.appendChild(thumbDownBtn);
  actions.appendChild(rewriteBtn);
  return actions;
}

// ── Messages render ───────────────────────────────────────────────────────────
function addMessageDom(role, content) {
  const row = document.createElement("div");
  row.className = `message-row ${role}`;

  if (role === "assistant") {
    const dot = document.createElement("div");
    dot.className = "avatar-dot";
    const dotImg = document.createElement("img");
    dotImg.src = "/static/assets/Signature.png";
    dotImg.alt = "VedAI";
    dotImg.className = "avatar-img";
    dot.appendChild(dotImg);
    row.appendChild(dot);
  }

  const bubble = document.createElement("div");
  bubble.className = `message-bubble ${role}`;
  bubble.innerHTML = renderContent(content);
  row.appendChild(bubble);

  // Action bar: left-aligned below the response
  if (role === "assistant") {
    const actionBar = buildActionBar(content);
    row.appendChild(actionBar);
  }

  if (messagesEl) messagesEl.appendChild(row);
  renderMath(bubble);
}

function renderMessages() {
  if (!messagesEl) return;
  messagesEl.innerHTML = "";
  const session = getCurrentSession();
  if (!session || session.messages.length === 0) {
    if (welcomeCard) welcomeCard.style.display = "block";
    loadDocs();
    return;
  }
  if (welcomeCard) welcomeCard.style.display = "none";
  session.messages.forEach(msg => addMessageDom(msg.role, msg.content));
  scrollToBottom();
  loadDocs(); // always refresh doc chips for the current conversation
}

// ── Send ──────────────────────────────────────────────────────────────────────
async function sendMessage(overrideText) {
  const message = (overrideText || (inputEl ? inputEl.value : "")).trim();
  if (!message || busy) return;

  const session = ensureSession();
  const priorHistory = session.messages.map(m => ({ role: m.role, content: m.content }));

  session.messages.push({ role: "user", content: message });
  if (session.title === "New conversation") session.title = shortTitle(message);
  session.updatedAt = new Date().toISOString();
  saveSessions();

  renderHistory();
  renderMessages();
  showTypingIndicator();
  scrollToBottom();

  if (inputEl) inputEl.value = "";
  autoResize();
  busy = true;
  if (sendBtn) sendBtn.disabled = true;
  setStatus(tempMode ? "Temporary chat" : "Thinking…", "busy");

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message, mode: "qa", history: priorHistory, conv_id: currentSessionId || "" }),
    });

    if (!res.ok || !res.body) {
      const ct = res.headers.get("content-type") || "";
      if (ct.includes("application/json")) {
        const data = await res.json();
        throw new Error(data.error || `Request failed (HTTP ${res.status})`);
      }
      throw new Error(`Server error (HTTP ${res.status}). Please refresh and try again.`);
    }

    const reader  = res.body.getReader();
    const decoder = new TextDecoder();
    let   buffer  = "";
    let   reply   = null;

    outer: while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const parts = buffer.split("\n\n");
      buffer = parts.pop();

      for (const part of parts) {
        for (const line of part.split("\n")) {
          if (!line.startsWith("data: ")) continue;
          let evt;
          try { evt = JSON.parse(line.slice(6)); } catch { continue; }
          if (evt.type === "ping") continue;
          if (evt.type === "reply") { reply = evt.reply; break outer; }
          if (evt.type === "error") throw new Error(evt.error || "Generation failed.");
        }
      }
    }

    removeTypingIndicator();
    const finalReply = reply || "Sorry, I could not generate a reply.";
    session.messages.push({ role: "assistant", content: finalReply });
    session.updatedAt = new Date().toISOString();
    saveSessions();
    renderHistory();
    renderMessages();
    setStatus(tempMode ? "Temporary chat" : "Ready", tempMode ? "busy" : "ready");

  } catch (err) {
    removeTypingIndicator();
    session.messages.push({ role: "assistant", content: `Error: ${err.message}` });
    session.updatedAt = new Date().toISOString();
    saveSessions();
    renderHistory();
    renderMessages();
    setStatus("Error", "error");
  } finally {
    busy = false;
    if (sendBtn) sendBtn.disabled = false;
    if (inputEl) inputEl.focus();
  }
}

// ── Events ────────────────────────────────────────────────────────────────────
if (sendBtn)   sendBtn.addEventListener("click", () => sendMessage());
if (inputEl) {
  inputEl.addEventListener("input", autoResize);
  inputEl.addEventListener("keydown", e => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
  });
}

if (newChatBtn) {
  newChatBtn.addEventListener("click", () => {
    createSession();
    saveSessions();
    renderHistory();
    renderMessages();
    setStatus(tempMode ? "Temporary chat" : "Ready", tempMode ? "busy" : "ready");
    if (inputEl) { inputEl.value = ""; autoResize(); inputEl.focus(); }
    loadDocs(); // clear doc chips for fresh conversation
  });
}

document.querySelectorAll(".chip").forEach(chip => {
  chip.addEventListener("click", () => {
    const prompt = chip.dataset.prompt;
    if (inputEl) {
      inputEl.value = prompt;
      autoResize();
      inputEl.focus();
      inputEl.setSelectionRange(prompt.length, prompt.length);
    }
  });
});

// ── Document management ───────────────────────────────────────────────────────
async function loadConfig() { /* placeholder */ }

async function loadDocs() {
  try {
    const cid = encodeURIComponent(currentSessionId || "");
    const r = await fetch(`/api/docs?conv_id=${cid}`);
    if (!r.ok) return;
    const d = await r.json();
    renderDocList(d.docs || []);
  } catch(e) {}
}

function renderDocList(docs) {
  if (!docChipsBar) return;
  if (!docs.length) {
    docChipsBar.innerHTML = "";
    return;
  }
  docChipsBar.innerHTML = docs.map(d => `
    <span class="doc-chip">
      📄 ${d.filename.slice(0, 18)}${d.filename.length > 18 ? "…" : ""}
      <span class="doc-chip-x" data-id="${d.id}" title="Remove">✕</span>
    </span>`).join('');
  docChipsBar.querySelectorAll(".doc-chip-x").forEach(el => {
    el.addEventListener("click", () => removeDoc(el.dataset.id));
  });
}

async function removeDoc(docId) {
  const cid = encodeURIComponent(currentSessionId || "");
  await fetch(`/api/docs/${docId}?conv_id=${cid}`, { method: "DELETE" });
  loadDocs();
}

// File upload handler
if (docFileInput) {
  docFileInput.addEventListener("change", async () => {
    const file = docFileInput.files[0];
    if (!file) return;
    const showStatus = (msg, cls) => {
      if (!uploadStatus) return;
      uploadStatus.textContent = msg;
      uploadStatus.className = "upload-status-inline " + cls;
      uploadStatus.style.display = "inline";
      if (cls === "ok") setTimeout(() => { uploadStatus.style.display = "none"; }, 4000);
    };
    showStatus(`Uploading ${file.name}…`, "busy");
    const fd = new FormData();
    fd.append("file", file);
    fd.append("ttl_minutes", 30);
    fd.append("conv_id", currentSessionId || "");
    try {
      const r = await fetch("/api/docs/upload", { method: "POST", body: fd });
      const d = await r.json();
      if (!r.ok) {
        showStatus(d.error || "Upload failed", "err");
      } else {
        showStatus(`✓ ${file.name} added`, "ok");
        loadDocs();
      }
    } catch(e) {
      showStatus(`Error: ${e.message}`, "err");
    }
    docFileInput.value = "";
  });
}

setInterval(loadDocs, 30000); // refresh TTL countdown

// ── Share ─────────────────────────────────────────────────────────────────────
if (shareBtn) {
  shareBtn.addEventListener("click", async () => {
    const session = getCurrentSession();
    if (!session || !session.messages.length) {
      alert("No messages to share yet.");
      return;
    }
    try {
      const r = await fetch("/api/share", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: session.messages, title: session.title }),
      });
      const d = await r.json();
      const url = window.location.origin + d.url;
      if (shareLink) { shareLink.textContent = url; shareLink.href = url; }
      if (shareToast) shareToast.classList.add("show");
      if (navigator.clipboard) await navigator.clipboard.writeText(url).catch(() => {});
    } catch(e) {
      alert("Share failed: " + e.message);
    }
  });
}

if (shareToastClose) {
  shareToastClose.addEventListener("click", () => {
    if (shareToast) shareToast.classList.remove("show");
  });
}

// ── Init ──────────────────────────────────────────────────────────────────────
sessions = loadSessions();
if (sessions.length === 0) {
  const s = makeFreshSession();
  sessions.unshift(s);
  saveSessions();
}
currentSessionId = sessions[0].id;

applyTempUI(false);
renderHistory();
renderMessages();
autoResize();
setStatus("Ready", "ready");
loadConfig();
loadDocs();
