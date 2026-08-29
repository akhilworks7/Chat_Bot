// =========================================================
// STATE MANAGEMENT & DOM ELEMENTS
// =========================================================
const state = {
  activeTab: 'documents',
  documents: [],
  totalVectors: 0,
  isProcessing: false,
  deleteTargetDoc: null
};

// DOM Elements
const elements = {
  // Navigation Tabs
  tabBtnDocuments: document.getElementById('tabBtnDocuments'),
  tabBtnChatbot: document.getElementById('tabBtnChatbot'),
  paneDocuments: document.getElementById('paneDocuments'),
  paneChatbot: document.getElementById('paneChatbot'),
  docCountBadge: document.getElementById('docCountBadge'),
  systemStatus: document.getElementById('systemStatus'),

  // Metrics
  statTotalDocs: document.getElementById('statTotalDocs'),
  statTotalVectors: document.getElementById('statTotalVectors'),
  statLLMModel: document.getElementById('statLLMModel'),

  // Document Upload & Table
  dropzone: document.getElementById('dropzone'),
  fileInput: document.getElementById('fileInput'),
  btnBrowse: document.getElementById('btnBrowse'),
  btnRefreshDocs: document.getElementById('btnRefreshDocs'),
  uploadProgressBox: document.getElementById('uploadProgressBox'),
  progressFileName: document.getElementById('progressFileName'),
  progressStageBadge: document.getElementById('progressStageBadge'),
  progressBarFill: document.getElementById('progressBarFill'),
  progressSteps: document.getElementById('progressSteps'),
  documentsTableBody: document.getElementById('documentsTableBody'),
  emptyDocsState: document.getElementById('emptyDocsState'),

  // Delete Document Modal
  deleteModal: document.getElementById('deleteModal'),
  deleteDocName: document.getElementById('deleteDocName'),
  btnCancelDelete: document.getElementById('btnCancelDelete'),
  btnConfirmDelete: document.getElementById('btnConfirmDelete'),

  // Clear Chat Modal
  clearChatModal: document.getElementById('clearChatModal'),
  btnCancelClearChat: document.getElementById('btnCancelClearChat'),
  btnConfirmClearChat: document.getElementById('btnConfirmClearChat'),

  // Chatbot
  chatMessages: document.getElementById('chatMessages'),
  chatWelcome: document.getElementById('chatWelcome'),
  chatForm: document.getElementById('chatForm'),
  chatInput: document.getElementById('chatInput'),
  btnSend: document.getElementById('btnSend'),
  btnClearChat: document.getElementById('btnClearChat'),

  // Toast Container
  toastContainer: document.getElementById('toastContainer')
};

// =========================================================
// INITIALIZATION
// =========================================================
document.addEventListener('DOMContentLoaded', () => {
  setupTabNavigation();
  setupUploadHandlers();
  setupChatbotHandlers();
  setupModalHandlers();
  loadInitialData();
});

// =========================================================
// TAB NAVIGATION
// =========================================================
function setupTabNavigation() {
  elements.tabBtnDocuments.addEventListener('click', () => switchTab('documents'));
  elements.tabBtnChatbot.addEventListener('click', () => switchTab('chatbot'));
  
  // Set initial button visibility
  elements.btnClearChat.style.display = 'none';
}

function switchTab(tabName) {
  state.activeTab = tabName;
  
  if (tabName === 'documents') {
    elements.tabBtnDocuments.classList.add('active');
    elements.tabBtnDocuments.setAttribute('aria-selected', 'true');
    elements.tabBtnChatbot.classList.remove('active');
    elements.tabBtnChatbot.setAttribute('aria-selected', 'false');

    elements.paneDocuments.classList.add('active');
    elements.paneChatbot.classList.remove('active');

    // Hide clear chat button on documents tab
    if (elements.btnClearChat) elements.btnClearChat.style.display = 'none';
  } else {
    elements.tabBtnChatbot.classList.add('active');
    elements.tabBtnChatbot.setAttribute('aria-selected', 'true');
    elements.tabBtnDocuments.classList.remove('active');
    elements.tabBtnDocuments.setAttribute('aria-selected', 'false');

    elements.paneChatbot.classList.add('active');
    elements.paneDocuments.classList.remove('active');

    // Show clear chat button on chatbot tab
    if (elements.btnClearChat) elements.btnClearChat.style.display = 'inline-flex';

    // Auto-focus chat input & scroll to latest message
    setTimeout(() => {
      elements.chatInput.focus();
      scrollToBottom();
    }, 100);
  }
}

// =========================================================
// DATA FETCHING & METRICS
// =========================================================
async function loadInitialData() {
  await Promise.all([fetchStats(), fetchDocuments()]);
}

async function fetchStats() {
  try {
    const res = await fetch('/api/v1/documents/stats');
    if (!res.ok) throw new Error('Failed to fetch stats');
    const data = await res.json();
    
    state.totalVectors = data.total_vector_count || 0;
    elements.statTotalVectors.textContent = state.totalVectors.toLocaleString();

    // Fetch system health/info
    const healthRes = await fetch('/');
    if (healthRes.ok) {
      const healthData = await healthRes.json();
      if (healthData.model) {
        elements.statLLMModel.textContent = healthData.model;
        elements.statLLMModel.title = healthData.model;
      }
    }
  } catch (err) {
    console.error('Stats fetch error:', err);
  }
}

async function fetchDocuments() {
  try {
    const res = await fetch('/api/v1/documents/');
    if (!res.ok) throw new Error('Failed to fetch documents list');
    const data = await res.json();
    
    state.documents = data.documents || [];
    renderDocumentsTable(state.documents);
    
    elements.statTotalDocs.textContent = state.documents.length;
    elements.docCountBadge.textContent = state.documents.length;
  } catch (err) {
    console.error('Document list fetch error:', err);
    showToast('Failed to load documents list', 'error');
  }
}

// =========================================================
// DOCUMENT TABLE RENDERING
// =========================================================
function renderDocumentsTable(docs) {
  elements.documentsTableBody.innerHTML = '';

  if (!docs || docs.length === 0) {
    elements.emptyDocsState.classList.remove('hidden');
    return;
  }

  elements.emptyDocsState.classList.add('hidden');

  docs.forEach(doc => {
    const tr = document.createElement('tr');
    
    const formattedSize = formatFileSize(doc.file_size_bytes);
    const formattedDate = formatDate(doc.upload_time);
    
    tr.innerHTML = `
      <td>
        <div class="doc-name-cell">
          <div class="doc-icon-small">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline></svg>
          </div>
          <span title="${escapeHtml(doc.filename)}">${escapeHtml(doc.filename)}</span>
        </div>
      </td>
      <td>${formattedSize}</td>
      <td><strong>${doc.total_chunks || (doc.vector_ids ? doc.vector_ids.length : '-')}</strong></td>
      <td><code>${escapeHtml(doc.namespace || 'documents')}</code></td>
      <td>${formattedDate}</td>
      <td><span class="status-tag">● Indexed</span></td>
      <td class="text-right">
        <button class="btn-delete-doc" data-filename="${escapeHtml(doc.filename)}" title="Delete document and purge vectors">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path></svg>
          Delete
        </button>
      </td>
    `;

    // Attach delete event
    const deleteBtn = tr.querySelector('.btn-delete-doc');
    deleteBtn.addEventListener('click', () => {
      openDeleteModal(doc.filename);
    });

    elements.documentsTableBody.appendChild(tr);
  });
}

// =========================================================
// UPLOAD HANDLERS & MULTI-STAGE PROGRESS
// =========================================================
function setupUploadHandlers() {
  elements.btnBrowse.addEventListener('click', (e) => {
    e.stopPropagation();
    elements.fileInput.click();
  });

  elements.dropzone.addEventListener('click', () => elements.fileInput.click());

  elements.fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleFileUpload(e.target.files[0]);
    }
  });

  // Drag and drop events
  ['dragenter', 'dragover'].forEach(eventName => {
    elements.dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      elements.dropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(eventName => {
    elements.dropzone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      elements.dropzone.classList.remove('dragover');
    });
  });

  elements.dropzone.addEventListener('drop', (e) => {
    if (e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  });

  elements.btnRefreshDocs.addEventListener('click', () => {
    fetchDocuments();
    fetchStats();
    showToast('Document list updated', 'success');
  });
}

async function handleFileUpload(file) {
  if (!file.name.toLowerCase().endsWith('.pdf')) {
    showToast('Please upload a PDF file.', 'error');
    return;
  }

  if (state.isProcessing) {
    showToast('Another document is currently processing. Please wait.', 'error');
    return;
  }

  state.isProcessing = true;
  showUploadProgress(file.name);

  // Progressive simulated pipeline steps while API executes
  const stepInterval = runProgressAnimation();

  const formData = new FormData();
  formData.append('file', file);
  formData.append('namespace', 'documents');

  try {
    const response = await fetch('/api/v1/documents/upload', {
      method: 'POST',
      body: formData
    });

    clearInterval(stepInterval);

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.detail || 'Upload failed');
    }

    const result = await response.json();
    setStage(5, 'Completed', 100);

    showToast(`Successfully indexed ${result.source} (${result.vectors_upserted} vectors)`, 'success');

    // Refresh UI after short delay
    setTimeout(async () => {
      elements.uploadProgressBox.classList.add('hidden');
      elements.fileInput.value = '';
      state.isProcessing = false;
      await loadInitialData();
    }, 1200);

  } catch (err) {
    clearInterval(stepInterval);
    console.error('Upload Error:', err);
    setStageError(err.message);
    showToast(err.message || 'Failed to upload document', 'error');
    state.isProcessing = false;
  }
}

function showUploadProgress(fileName) {
  elements.uploadProgressBox.classList.remove('hidden');
  elements.progressFileName.textContent = fileName;
  setStage(1, 'Uploading...', 15);
}

function runProgressAnimation() {
  const stages = [
    { stage: 2, label: 'Checking Text / Running OCR...', pct: 35 },
    { stage: 3, label: 'Extracting & Chunking...', pct: 60 },
    { stage: 4, label: 'Generating all-MiniLM Embeddings...', pct: 80 },
    { stage: 5, label: 'Upserting to Pinecone...', pct: 92 }
  ];

  let idx = 0;
  return setInterval(() => {
    if (idx < stages.length) {
      setStage(stages[idx].stage, stages[idx].label, stages[idx].pct);
      idx++;
    }
  }, 1200);
}

function setStage(stageNum, label, percent) {
  elements.progressStageBadge.textContent = label;
  elements.progressBarFill.style.width = `${percent}%`;

  const steps = ['stepUpload', 'stepExtract', 'stepChunk', 'stepEmbed', 'stepIndex'];
  steps.forEach((stepId, i) => {
    const el = document.getElementById(stepId);
    if (!el) return;
    el.classList.remove('active', 'completed');
    if (i + 1 < stageNum) {
      el.classList.add('completed');
    } else if (i + 1 === stageNum) {
      el.classList.add('active');
    }
  });
}

function setStageError(errMsg) {
  elements.progressStageBadge.textContent = 'Failed: ' + errMsg;
  elements.progressStageBadge.style.background = 'rgba(239, 68, 68, 0.2)';
  elements.progressStageBadge.style.color = '#EF4444';
  elements.progressBarFill.style.background = '#EF4444';
}

// =========================================================
// MODALS (Document Deletion & Clear Chat)
// =========================================================
function setupModalHandlers() {
  // Document Delete Modal
  elements.btnCancelDelete.addEventListener('click', closeDeleteModal);
  elements.deleteModal.addEventListener('click', (e) => {
    if (e.target === elements.deleteModal) closeDeleteModal();
  });
  elements.btnConfirmDelete.addEventListener('click', confirmDeleteDocument);

  // Clear Chat Modal
  elements.btnCancelClearChat.addEventListener('click', closeClearChatModal);
  elements.clearChatModal.addEventListener('click', (e) => {
    if (e.target === elements.clearChatModal) closeClearChatModal();
  });
  elements.btnConfirmClearChat.addEventListener('click', executeClearChat);
}

function openDeleteModal(docName) {
  state.deleteTargetDoc = docName;
  elements.deleteDocName.textContent = docName;
  elements.deleteModal.classList.remove('hidden');
}

function closeDeleteModal() {
  state.deleteTargetDoc = null;
  elements.deleteModal.classList.add('hidden');
}

async function confirmDeleteDocument() {
  if (!state.deleteTargetDoc) return;
  const docName = state.deleteTargetDoc;

  const btnText = elements.btnConfirmDelete.querySelector('.btn-text');
  elements.btnConfirmDelete.disabled = true;
  btnText.textContent = 'Deleting & Purging Pinecone...';

  try {
    const res = await fetch(`/api/v1/documents/${encodeURIComponent(docName)}`, {
      method: 'DELETE'
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.detail || 'Delete failed');
    }

    const data = await res.json();
    closeDeleteModal();
    showToast(`Deleted ${docName} and purged ${data.vectors_deleted || 0} vectors from Pinecone`, 'success');

    await loadInitialData();
  } catch (err) {
    console.error('Delete error:', err);
    showToast(`Error deleting document: ${err.message}`, 'error');
  } finally {
    elements.btnConfirmDelete.disabled = false;
    btnText.textContent = 'Yes, Delete Everything';
  }
}

function openClearChatModal() {
  const messageRows = elements.chatMessages.querySelectorAll('.message-row');
  if (messageRows.length === 0) {
    return; // Already empty
  }
  elements.clearChatModal.classList.remove('hidden');
}

function closeClearChatModal() {
  elements.clearChatModal.classList.add('hidden');
}

function executeClearChat() {
  elements.chatMessages.innerHTML = '';
  elements.chatMessages.appendChild(elements.chatWelcome);
  elements.chatWelcome.classList.remove('hidden');
  closeClearChatModal();
  showToast('Conversation cleared', 'success');
}

// =========================================================
// CHATBOT IMPLEMENTATION & UNCONSTRAINED RESPONSES
// =========================================================
function setupChatbotHandlers() {
  // Auto-resize textarea
  elements.chatInput.addEventListener('input', () => {
    elements.chatInput.style.height = 'auto';
    elements.chatInput.style.height = `${Math.min(elements.chatInput.scrollHeight, 130)}px`;
  });

  // Submit on Enter (Shift+Enter for newline)
  elements.chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (elements.chatInput.value.trim() && !state.isProcessing) {
        elements.chatForm.dispatchEvent(new Event('submit'));
      }
    }
  });

  // Form Submit
  elements.chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const query = elements.chatInput.value.trim();
    if (!query || state.isProcessing) return;

    elements.chatInput.value = '';
    elements.chatInput.style.height = 'auto';
    await sendUserMessage(query);
  });

  // Suggestion Chips
  document.querySelectorAll('.chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const question = chip.getAttribute('data-question');
      if (question) {
        sendUserMessage(question);
      }
    });
  });

  // Clear Chat Button (Trigger Modal)
  elements.btnClearChat.addEventListener('click', openClearChatModal);
}

async function sendUserMessage(question) {
  state.isProcessing = true;
  elements.btnSend.disabled = true;

  // Hide welcome screen on first message
  if (!elements.chatWelcome.classList.contains('hidden')) {
    elements.chatWelcome.classList.add('hidden');
  }

  // 1. Render User Message
  appendUserMessage(question);

  // 2. Render Temporary Assistant Loading Bubble
  const loadingBubble = appendLoadingBubble();
  scrollToBottom(true);

  try {
    const response = await fetch('/api/v1/chat/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        question: question,
        top_k: 8,
        namespace: 'documents'
      })
    });

    if (!response.ok) {
      const errData = await response.json().catch(() => ({}));
      throw new Error(errData.detail || 'Chat query failed');
    }

    const data = await response.json();

    // Replace loading bubble with full unconstrained response
    replaceLoadingBubbleWithAnswer(loadingBubble, data);
  } catch (err) {
    console.error('Chat error:', err);
    replaceLoadingBubbleWithError(loadingBubble, err.message);
  } finally {
    state.isProcessing = false;
    elements.btnSend.disabled = false;
    scrollToBottom(true);
  }
}

function appendUserMessage(text) {
  const row = document.createElement('div');
  row.className = 'message-row user';
  row.innerHTML = `
    <div class="msg-avatar">U</div>
    <div class="msg-bubble-wrapper">
      <div class="msg-bubble">${escapeHtml(text)}</div>
    </div>
  `;
  elements.chatMessages.appendChild(row);
  scrollToBottom(true);
}

function appendLoadingBubble() {
  const row = document.createElement('div');
  row.className = 'message-row assistant';
  row.innerHTML = `
    <div class="msg-avatar">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a10 10 0 0 1 10 10c0 5.523-4.477 10-10 10S2 17.523 2 12 6.477 2 12 2z"></path></svg>
    </div>
    <div class="msg-bubble-wrapper">
      <div class="msg-bubble" style="display:flex; align-items:center; gap:0.5rem; color:var(--text-muted);">
        <span class="loading-dots">Searching vector knowledge base & generating answer...</span>
      </div>
    </div>
  `;
  elements.chatMessages.appendChild(row);
  return row;
}

function replaceLoadingBubbleWithAnswer(loadingRow, data) {
  const answerHtml = marked.parse(data.answer || '');
  const sources = data.sources || [];
  const latencySeconds = (data.response_time_ms / 1000).toFixed(2);
  const msgId = 'msg_' + Date.now();

  let sourcesHtml = '';
  if (sources.length > 0) {
    sourcesHtml = `
      <div class="sources-accordion">
        <button class="sources-toggle" onclick="toggleSources('${msgId}')">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="6 9 12 15 18 9"></polyline></svg>
          <span>Retrieved Sources (${sources.length} chunks)</span>
        </button>
        <div class="sources-content" id="${msgId}">
          ${sources.map((src, i) => `
            <div class="source-card">
              <div class="source-card-header">
                <span class="source-filename">📄 ${escapeHtml(src.source || 'Document')} (Chunk ${src.chunk_id !== null ? src.chunk_id : i + 1})</span>
                <span class="source-score">Similarity: ${Math.round((src.score || 0) * 100)}%</span>
              </div>
              <div class="source-text">${escapeHtml(src.text)}</div>
            </div>
          `).join('')}
        </div>
      </div>
    `;
  }

  loadingRow.innerHTML = `
    <div class="msg-avatar">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2a10 10 0 0 1 10 10c0 5.523-4.477 10-10 10S2 17.523 2 12 6.477 2 12 2z"></path></svg>
    </div>
    <div class="msg-bubble-wrapper">
      <div class="msg-bubble">
        ${answerHtml}
        ${sourcesHtml}
      </div>
      <div class="msg-meta">
        <span class="latency-pill">⚡ ${latencySeconds}s</span>
        <button class="btn-copy" onclick="copyText(this)" data-copy="${encodeURIComponent(data.answer || '')}" title="Copy to clipboard">
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg>
          <span>Copy</span>
        </button>
      </div>
    </div>
  `;
}

function replaceLoadingBubbleWithError(loadingRow, errorMsg) {
  loadingRow.innerHTML = `
    <div class="msg-avatar" style="background:var(--danger);">!</div>
    <div class="msg-bubble-wrapper">
      <div class="msg-bubble" style="border-color:rgba(239, 68, 68, 0.4); color:#FCA5A5;">
        <strong>Error:</strong> ${escapeHtml(errorMsg)}
      </div>
    </div>
  `;
}

window.toggleSources = function(id) {
  const content = document.getElementById(id);
  if (!content) return;
  const toggle = content.previousElementSibling;
  const isOpen = content.classList.toggle('open');
  if (toggle) toggle.classList.toggle('open', isOpen);
  
  if (isOpen) {
    setTimeout(() => scrollToBottom(true), 100);
  }
};

window.copyText = function(btn) {
  const raw = decodeURIComponent(btn.getAttribute('data-copy') || '');
  navigator.clipboard.writeText(raw).then(() => {
    const span = btn.querySelector('span');
    const orig = span.textContent;
    span.textContent = 'Copied!';
    setTimeout(() => { span.textContent = orig; }, 1800);
  });
};

function scrollToBottom(smooth = false) {
  if (smooth) {
    elements.chatMessages.scrollTo({
      top: elements.chatMessages.scrollHeight,
      behavior: 'smooth'
    });
  } else {
    elements.chatMessages.scrollTop = elements.chatMessages.scrollHeight;
  }
}

// =========================================================
// UTILITIES (Formatting & Toast Notifications)
// =========================================================
function formatFileSize(bytes) {
  if (!bytes || bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

function formatDate(isoString) {
  if (!isoString) return '-';
  try {
    const date = new Date(isoString);
    return date.toLocaleDateString(undefined, {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  } catch (e) {
    return isoString;
  }
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#039;');
}

function showToast(message, type = 'info') {
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `
    <span>${type === 'success' ? '✅' : type === 'error' ? '❌' : 'ℹ️'}</span>
    <span>${escapeHtml(message)}</span>
  `;
  elements.toastContainer.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = '0';
    toast.style.transform = 'translateY(12px)';
    toast.style.transition = 'all 0.3s ease';
    setTimeout(() => toast.remove(), 300);
  }, 3500);
}
