// Archmage Intrinsics Browser — Search + Virtual Scroll
// Vanilla JS, no dependencies

(function () {
  'use strict';

  const ROW_HEIGHT = 36;
  const OVERSCAN = 10;

  let allData = null;       // Full dataset
  let filtered = [];        // Current filtered results
  let selectedIdx = -1;     // Selected row in filtered[]
  let tokenMap = {};        // tokenName → token object
  let safeVariantSet = null; // Set of intrinsic names that have safe variants

  // Current filter state
  let activeArch = 'x86_64';
  let activeToken = '';     // '' = all tokens for this arch

  // DOM refs
  const searchInput = document.getElementById('searchInput');
  const resultCount = document.getElementById('resultCount');
  const virtualScroll = document.getElementById('virtualScroll');
  const scrollContent = document.getElementById('scrollContent');
  const detailPanel = document.getElementById('detailPanel');
  const detailContent = document.getElementById('detailContent');
  const detailClose = document.getElementById('detailClose');
  const tokenTree = document.getElementById('tokenTree');

  // ========== Data Loading ==========

  async function init() {
    resultCount.textContent = 'Loading...';
    try {
      const resp = await fetch('data/intrinsics.json');
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      allData = await resp.json();
    } catch (e) {
      resultCount.textContent = 'Failed to load data';
      console.error('Failed to load intrinsics data:', e);
      return;
    }

    // Build token lookup
    for (const t of allData.tokens) {
      tokenMap[t.name] = t;
      for (const alias of (t.aliases || [])) {
        tokenMap[alias] = t;
      }
    }

    // Build safe variant lookup set
    safeVariantSet = new Set(Object.keys(allData.safeVariants || {}));

    restoreState();
    renderTokenTree();
    applyFilters();
    setupEvents();

    searchInput.focus();
  }

  // ========== Token Tree ==========

  function getTokensForArch(arch) {
    return allData.tokens.filter(t => t.arch === arch);
  }

  function buildTree(tokens) {
    // Build parent→children map
    const childMap = {};  // parentName → [token, ...]
    const roots = [];
    for (const t of tokens) {
      if (!t.parent) {
        roots.push(t);
      } else {
        if (!childMap[t.parent]) childMap[t.parent] = [];
        childMap[t.parent].push(t);
      }
    }

    // Flatten tree in DFS order with depth info
    const flat = [];
    function walk(node, depth) {
      flat.push({ token: node, depth });
      const children = childMap[node.name] || [];
      for (const child of children) {
        walk(child, depth + 1);
      }
    }
    for (const root of roots) {
      walk(root, 0);
    }
    return flat;
  }

  function renderTokenTree() {
    tokenTree.innerHTML = '';
    const tokens = getTokensForArch(activeArch);
    if (tokens.length === 0) return;

    const label = document.createElement('span');
    label.className = 'token-tree-label';
    label.textContent = 'Tokens:';
    tokenTree.appendChild(label);

    // "All" button
    const allBtn = document.createElement('button');
    allBtn.className = 'token-btn token-all' + (activeToken === '' ? ' active' : '');
    allBtn.textContent = 'All';
    allBtn.addEventListener('click', () => {
      activeToken = '';
      updateTokenTreeActive();
      applyFilters();
    });
    tokenTree.appendChild(allBtn);

    // Build and render tree
    const tree = buildTree(tokens);
    for (let i = 0; i < tree.length; i++) {
      const { token: t, depth } = tree[i];

      // Arrow separator showing hierarchy
      if (depth > 0) {
        const sep = document.createElement('span');
        sep.className = 'token-indent';
        sep.textContent = '\u2192'; // →
        tokenTree.appendChild(sep);
      } else if (i > 0) {
        // Space between root-level siblings
        const sep = document.createElement('span');
        sep.className = 'token-sep';
        sep.textContent = '|';
        tokenTree.appendChild(sep);
      }

      const btn = document.createElement('button');
      btn.className = 'token-btn' + (activeToken === t.name ? ' active' : '');
      btn.dataset.token = t.name;
      btn.textContent = t.name;
      btn.title = t.doc || t.display;
      btn.addEventListener('click', () => {
        activeToken = t.name;
        updateTokenTreeActive();
        applyFilters();
      });
      tokenTree.appendChild(btn);
    }
  }

  function updateTokenTreeActive() {
    tokenTree.querySelectorAll('.token-btn').forEach(btn => {
      if (btn.classList.contains('token-all')) {
        btn.classList.toggle('active', activeToken === '');
      } else {
        btn.classList.toggle('active', btn.dataset.token === activeToken);
      }
    });
  }

  // ========== Filtering ==========

  function getActiveValues(filterName) {
    const btns = document.querySelectorAll(`.filter-btn[data-filter="${filterName}"]`);
    const active = [];
    btns.forEach(b => { if (b.classList.contains('active')) active.push(b.dataset.value); });
    return active;
  }

  function applyFilters() {
    const query = searchInput.value.toLowerCase().trim();
    const stabilities = new Set(getActiveValues('stability'));
    const safeties = new Set(getActiveValues('safety'));

    // Collect valid tokens: if a specific token is selected, include it
    // and all descendants (children inherit parent's intrinsics)
    let validTokens = null;
    if (activeToken) {
      validTokens = new Set();
      validTokens.add(activeToken);
      // Add all descendants
      function addDescendants(parentName) {
        for (const t of allData.tokens) {
          if (t.parent === parentName && !validTokens.has(t.name)) {
            validTokens.add(t.name);
            addDescendants(t.name);
          }
        }
      }
      addDescendants(activeToken);
    }

    filtered = allData.intrinsics.filter(i => {
      // Architecture filter (single arch)
      if (i.a !== activeArch) return false;

      // Token filter
      if (validTokens) {
        if (!i.t || !validTokens.has(i.t)) return false;
      }

      // Stability toggle
      const isStable = i.s;
      const showStable = stabilities.has('stable');
      const showUnstable = stabilities.has('unstable');
      if (!showStable && !showUnstable) return false;
      if (!showStable && isStable) return false;
      if (!showUnstable && !isStable) return false;

      // Safety toggle
      const isUnsafe = i.u;
      const showSafe = safeties.has('safe');
      const showUnsafe = safeties.has('unsafe');
      if (!showSafe && !showUnsafe) return false;
      if (!showSafe && !isUnsafe) return false;
      if (!showUnsafe && isUnsafe) return false;

      // Search
      if (query) {
        return i.n.toLowerCase().includes(query) ||
               (i.d && i.d.toLowerCase().includes(query)) ||
               (i.ins && i.ins.toLowerCase().includes(query)) ||
               (i.f && i.f.toLowerCase().includes(query));
      }

      return true;
    });

    resultCount.textContent = `${filtered.length.toLocaleString()} results`;
    selectedIdx = -1;
    closeDetail();
    renderVirtualScroll();
    saveState();
  }

  // ========== Virtual Scroll ==========

  function renderVirtualScroll() {
    const totalHeight = filtered.length * ROW_HEIGHT;
    scrollContent.style.height = totalHeight + 'px';

    // Clear existing rows
    while (scrollContent.firstChild) {
      scrollContent.removeChild(scrollContent.firstChild);
    }

    renderVisibleRows();
  }

  function renderVisibleRows() {
    const scrollTop = virtualScroll.scrollTop;
    const viewHeight = virtualScroll.clientHeight;

    const startIdx = Math.max(0, Math.floor(scrollTop / ROW_HEIGHT) - OVERSCAN);
    const endIdx = Math.min(filtered.length, Math.ceil((scrollTop + viewHeight) / ROW_HEIGHT) + OVERSCAN);

    // Remove out-of-range rows
    const existing = scrollContent.querySelectorAll('.vrow');
    existing.forEach(row => {
      const idx = parseInt(row.dataset.idx);
      if (idx < startIdx || idx >= endIdx) {
        row.remove();
      }
    });

    // Add missing rows
    const existingIdxs = new Set();
    scrollContent.querySelectorAll('.vrow').forEach(row => {
      existingIdxs.add(parseInt(row.dataset.idx));
    });

    const frag = document.createDocumentFragment();
    for (let idx = startIdx; idx < endIdx; idx++) {
      if (existingIdxs.has(idx)) continue;
      frag.appendChild(createRow(idx));
    }
    scrollContent.appendChild(frag);
  }

  function createRow(idx) {
    const i = filtered[idx];
    const row = document.createElement('div');
    row.className = 'vrow' + (idx === selectedIdx ? ' selected' : '');
    row.dataset.idx = idx;
    row.style.top = (idx * ROW_HEIGHT) + 'px';

    const tokenName = i.t || '—';

    const doc = truncateDoc(i.d || '', 80);
    const stableBadge = i.s
      ? '<span class="badge badge-stable">stable</span>'
      : '<span class="badge badge-unstable">nightly</span>';

    let safeBadge;
    if (i.u) {
      if (safeVariantSet.has(i.n)) {
        safeBadge = '<span class="badge badge-has-safe" title="safe_unaligned_simd wrapper available">unsafe*</span>';
      } else {
        safeBadge = '<span class="badge badge-unsafe">unsafe</span>';
      }
    } else {
      safeBadge = '<span class="badge badge-safe">safe</span>';
    }

    row.innerHTML = `
      <div class="col-name">${escHtml(i.n)}</div>
      <div class="col-token">${escHtml(tokenName)}</div>
      <div class="col-desc">${escHtml(doc)}</div>
      <div class="col-badges">${stableBadge}${safeBadge}</div>
    `;

    row.addEventListener('click', () => selectRow(idx));
    return row;
  }

  function selectRow(idx) {
    if (selectedIdx === idx) {
      selectedIdx = -1;
      closeDetail();
      updateSelectedClass();
      return;
    }
    selectedIdx = idx;
    updateSelectedClass();
    showDetail(filtered[idx]);
  }

  function updateSelectedClass() {
    scrollContent.querySelectorAll('.vrow').forEach(row => {
      const rowIdx = parseInt(row.dataset.idx);
      row.classList.toggle('selected', rowIdx === selectedIdx);
    });
  }

  // ========== Detail Panel ==========

  function closeDetail() {
    detailPanel.style.display = 'none';
    virtualScroll.style.paddingBottom = '0';
  }

  function showDetail(i) {
    const token = i.t ? tokenMap[i.t] : null;
    const tokenDisplay = token ? token.name : 'Not covered by archmage';

    // Extract doc links
    const docLinks = extractDocLinks(i.d || '');
    const docText = (i.d || '').split('[')[0].trim();

    // Timing
    let timingHtml = '';
    if (i.tc && allData.timing[i.tc]) {
      const t = allData.timing[i.tc];
      const names = { h: 'Haswell', sk: 'Skylake-X', z4: 'Zen 4', sp: 'Sapphire Rapids' };
      timingHtml = '<div class="detail-timing">';
      for (const [key, label] of Object.entries(names)) {
        const vals = t[key];
        const display = vals ? `${vals[0]}/${vals[1]}` : '—';
        timingHtml += `<div class="timing-col">
          <div class="timing-header">${label}</div>
          <div class="timing-value">${display}</div>
        </div>`;
      }
      timingHtml += '</div>';
    }

    // Safe variant
    let safeHtml = '';
    if (i.u && allData.safeVariants[i.n]) {
      const sig = allData.safeVariants[i.n];
      const archMod = i.a === 'aarch64' ? 'aarch64' : i.a === 'wasm32' ? 'wasm32' : 'x86_64';
      safeHtml = `<div class="safe-variant-note">
        <strong>Safe alternative:</strong>
        <code>safe_unaligned_simd::${archMod}::${escHtml(i.n)}</code><br>
        <span class="safe-variant-sig">${escHtml(sig)}</span>
      </div>`;
    } else if (i.u) {
      safeHtml = `<div class="unsafe-note">
        No safe wrapper available. Requires <code>unsafe</code> block.
      </div>`;
    }

    // Usage example
    const usageExample = buildUsageExample(i, token);

    // Doc links
    let linksHtml = '';
    if (docLinks.length > 0) {
      linksHtml = '<div style="margin-top: 8px;">' +
        docLinks.map(l => `<a class="doc-link" href="${escAttr(l.url)}" target="_blank" rel="noopener">${escHtml(l.text)} ↗</a>`).join(' &nbsp; ') +
        '</div>';
    }

    // Per-token reference link
    let tokenRefHtml = '';
    if (token) {
      tokenRefHtml = `<div style="margin-top: 8px;">
        <a class="doc-link" href="tokens/${escAttr(token.name)}.md" target="_blank" rel="noopener">${escHtml(token.name)} reference ↗</a>
      </div>`;
    }

    detailContent.innerHTML = `
      <h2>${escHtml(i.n)}</h2>
      <div class="detail-grid">
        <div class="detail-field">
          <span class="detail-label">Token</span>
          <span class="detail-value">${escHtml(tokenDisplay)}</span>
        </div>
        <div class="detail-field">
          <span class="detail-label">Features</span>
          <span class="detail-value">${escHtml(i.f || '—')}</span>
        </div>
        <div class="detail-field">
          <span class="detail-label">Instruction</span>
          <span class="detail-value">${escHtml(i.ins || '—')}</span>
        </div>
        <div class="detail-field">
          <span class="detail-label">Stability</span>
          <span class="detail-value">${i.s ? '<span class="badge badge-stable">stable</span>' : '<span class="badge badge-unstable">nightly</span>'}</span>
        </div>
        <div class="detail-field">
          <span class="detail-label">Safety</span>
          <span class="detail-value">${i.u ? '<span class="badge badge-unsafe">unsafe</span>' : '<span class="badge badge-safe">safe</span>'}</span>
        </div>
        <div class="detail-field">
          <span class="detail-label">Architecture</span>
          <span class="detail-value">${escHtml(i.a)}</span>
        </div>
      </div>
      <div style="margin-bottom: 8px; color: var(--text);">${escHtml(docText)}</div>
      ${linksHtml}
      ${tokenRefHtml}
      <div class="detail-field" style="margin-top: 8px;">
        <span class="detail-label">Signature</span>
        <span class="detail-value" style="word-break: break-all;">${escHtml(i.sig || '—')}</span>
      </div>
      ${timingHtml}
      ${safeHtml}
      ${usageExample}
    `;

    detailPanel.style.display = 'block';
    // Add padding so table rows aren't hidden behind the fixed overlay
    requestAnimationFrame(() => {
      virtualScroll.style.paddingBottom = detailPanel.offsetHeight + 'px';
    });
  }

  function buildUsageExample(i, token) {
    if (!token) return '';

    const tokenName = token.name;
    const archMod = i.a === 'aarch64' ? 'aarch64' : i.a === 'wasm32' ? 'wasm32' : 'x86_64';
    let code;

    if (i.u) {
      const safeVar = allData.safeVariants[i.n];
      if (safeVar) {
        code = `#[rite]\nfn example(_: ${tokenName}, /* params */) {\n    // Use safe_unaligned_simd instead of unsafe:\n    let result = safe_unaligned_simd::${archMod}::${i.n}(/* args */);\n}`;
      } else {
        code = `#[rite]\nfn example(_: ${tokenName}, /* params */) {\n    // No safe wrapper — requires unsafe block.\n    let result = unsafe { ${i.n}(/* args */) };\n}`;
      }
    } else {
      code = `#[rite]\nfn example(_: ${tokenName}, /* params */) {\n    // Safe inside #[rite] — no unsafe needed\n    let result = ${i.n}(/* args */);\n}`;
    }

    return `<div class="detail-code">
      <div class="code-label">Usage with archmage</div>
      <pre>${escHtml(code)}</pre>
    </div>`;
  }

  // ========== Helpers ==========

  function truncateDoc(doc, max) {
    const cleaned = doc.split('[')[0].trim().replace(/\.$/, '');
    return cleaned.length > max ? cleaned.substring(0, max) + '...' : cleaned;
  }

  function extractDocLinks(doc) {
    const links = [];
    const re = /\[([^\]]+)\]\(([^)]+)\)/g;
    let m;
    while ((m = re.exec(doc)) !== null) {
      links.push({ text: m[1], url: m[2] });
    }
    return links;
  }

  function escHtml(s) {
    if (!s) return '';
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  function escAttr(s) {
    return s.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  // ========== URL State ==========

  function saveState() {
    const params = new URLSearchParams();
    const query = searchInput.value.trim();
    if (query) params.set('q', query);

    params.set('arch', activeArch);

    if (activeToken) params.set('token', activeToken);

    const stabilities = getActiveValues('stability');
    if (stabilities.length < 2) params.set('stability', stabilities.join(','));

    const safeties = getActiveValues('safety');
    if (safeties.length < 2) params.set('safety', safeties.join(','));

    const hash = params.toString();
    history.replaceState(null, '', hash ? '#' + hash : location.pathname);
  }

  function restoreState() {
    const hash = location.hash.substring(1);
    if (!hash) return;

    const params = new URLSearchParams(hash);

    if (params.has('q')) searchInput.value = params.get('q');

    if (params.has('arch')) {
      activeArch = params.get('arch');
    }

    if (params.has('token')) {
      activeToken = params.get('token');
    }

    if (params.has('stability')) {
      const vals = new Set(params.get('stability').split(','));
      document.querySelectorAll('.filter-btn[data-filter="stability"]').forEach(b => {
        b.classList.toggle('active', vals.has(b.dataset.value));
      });
    }

    if (params.has('safety')) {
      const vals = new Set(params.get('safety').split(','));
      document.querySelectorAll('.filter-btn[data-filter="safety"]').forEach(b => {
        b.classList.toggle('active', vals.has(b.dataset.value));
      });
    }

    // Update arch button state
    document.querySelectorAll('.arch-btn').forEach(b => {
      b.classList.toggle('active', b.dataset.arch === activeArch);
    });
  }

  // ========== Events ==========

  function setupEvents() {
    // Search with debounce
    let debounceTimer;
    searchInput.addEventListener('input', () => {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(applyFilters, 150);
    });

    // Arch buttons — radio behavior (one at a time)
    document.querySelectorAll('.arch-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        if (btn.dataset.arch === activeArch) return; // already selected
        activeArch = btn.dataset.arch;
        activeToken = ''; // reset token filter on arch change
        document.querySelectorAll('.arch-btn').forEach(b => {
          b.classList.toggle('active', b.dataset.arch === activeArch);
        });
        renderTokenTree();
        applyFilters();
      });
    });

    // Stability/safety toggle buttons
    document.querySelectorAll('.filter-btn[data-filter]').forEach(btn => {
      btn.addEventListener('click', () => {
        btn.classList.toggle('active');
        applyFilters();
      });
    });

    // Virtual scroll
    virtualScroll.addEventListener('scroll', renderVisibleRows);

    // Resize
    let resizeTimer;
    window.addEventListener('resize', () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(renderVisibleRows, 100);
    });

    // Detail close
    detailClose.addEventListener('click', () => {
      closeDetail();
      selectedIdx = -1;
      updateSelectedClass();
    });

    // Keyboard navigation
    document.addEventListener('keydown', (e) => {
      if (e.target === searchInput && (e.key === 'ArrowDown' || e.key === 'ArrowUp')) {
        e.preventDefault();
        if (e.key === 'ArrowDown') {
          selectRow(Math.min(selectedIdx + 1, filtered.length - 1));
        } else {
          selectRow(Math.max(selectedIdx - 1, 0));
        }
        if (selectedIdx >= 0) {
          const rowTop = selectedIdx * ROW_HEIGHT;
          const scrollTop = virtualScroll.scrollTop;
          const viewHeight = virtualScroll.clientHeight;
          if (rowTop < scrollTop) {
            virtualScroll.scrollTop = rowTop;
          } else if (rowTop + ROW_HEIGHT > scrollTop + viewHeight) {
            virtualScroll.scrollTop = rowTop + ROW_HEIGHT - viewHeight;
          }
        }
        renderVisibleRows();
        return;
      }

      if (e.key === 'Escape') {
        if (detailPanel.style.display !== 'none') {
          closeDetail();
          selectedIdx = -1;
          updateSelectedClass();
        } else {
          searchInput.value = '';
          searchInput.focus();
          applyFilters();
        }
      }

      if ((e.key === '/' || (e.key === 'k' && (e.ctrlKey || e.metaKey))) && e.target !== searchInput) {
        e.preventDefault();
        searchInput.focus();
        searchInput.select();
      }
    });
  }

  // ========== Start ==========
  init();
})();
