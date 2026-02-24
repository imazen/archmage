// Archmage Intrinsics Browser — Search + Virtual Scroll
// Vanilla JS, no dependencies

(function () {
  'use strict';

  const ROW_HEIGHT = 36;
  const OVERSCAN = 10;

  let allData = null;
  let filtered = [];
  let selectedIdx = -1;
  let tokenMap = {};        // tokenName → token object
  let safeVariantSet = null;

  // Filter state
  let activeArch = 'x86_64';
  // Token selection: null = "All" mode (show everything for arch).
  // Otherwise, a Set of token names that are lit up.
  let litTokens = null;

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

    for (const t of allData.tokens) {
      tokenMap[t.name] = t;
      for (const alias of (t.aliases || [])) {
        tokenMap[alias] = t;
      }
    }
    safeVariantSet = new Set(Object.keys(allData.safeVariants || {}));

    restoreState();
    renderTokenTree();
    applyFilters();
    setupEvents();
    searchInput.focus();
  }

  // ========== Token Hierarchy Helpers ==========

  function getAncestors(tokenName) {
    // Returns array from root to tokenName (inclusive)
    const chain = [];
    let cur = tokenName;
    while (cur) {
      chain.unshift(cur);
      const t = tokenMap[cur];
      cur = t && t.parent ? t.parent : null;
    }
    return chain;
  }

  function getTokensForArch(arch) {
    return allData.tokens.filter(t => t.arch === arch);
  }

  function buildTreeFlat(tokens) {
    // DFS order with depth
    const childMap = {};
    const roots = [];
    for (const t of tokens) {
      if (!t.parent) {
        roots.push(t);
      } else {
        if (!childMap[t.parent]) childMap[t.parent] = [];
        childMap[t.parent].push(t);
      }
    }
    const flat = [];
    function walk(node, depth) {
      flat.push({ token: node, depth });
      for (const child of (childMap[node.name] || [])) {
        walk(child, depth + 1);
      }
    }
    for (const root of roots) walk(root, 0);
    return flat;
  }

  // ========== Token Tree Rendering ==========

  function renderTokenTree() {
    tokenTree.innerHTML = '';
    const tokens = getTokensForArch(activeArch);
    if (tokens.length === 0) return;

    // "All" button
    const allBtn = document.createElement('button');
    allBtn.className = 'token-btn token-all' + (litTokens === null ? ' active' : '');
    allBtn.textContent = 'All';
    allBtn.addEventListener('click', () => {
      litTokens = null;
      syncTokenButtons();
      applyFilters();
    });
    tokenTree.appendChild(allBtn);

    // Tree buttons
    const tree = buildTreeFlat(tokens);
    for (let i = 0; i < tree.length; i++) {
      const { token: t, depth } = tree[i];

      if (depth > 0) {
        const sep = document.createElement('span');
        sep.className = 'token-sep';
        sep.textContent = '\u203a'; // ›
        tokenTree.appendChild(sep);
      } else if (i > 0) {
        const sep = document.createElement('span');
        sep.className = 'token-sep';
        sep.textContent = '|';
        tokenTree.appendChild(sep);
      }

      const btn = document.createElement('button');
      btn.className = 'token-btn';
      btn.dataset.token = t.name;
      btn.textContent = t.name;
      btn.title = t.doc || t.display;
      if (litTokens && litTokens.has(t.name)) btn.classList.add('active');

      btn.addEventListener('click', () => onTokenClick(t.name));
      tokenTree.appendChild(btn);
    }
  }

  function onTokenClick(name) {
    if (litTokens === null) {
      // Was in "All" mode — activate this token + ancestors
      litTokens = new Set(getAncestors(name));
    } else if (litTokens.has(name)) {
      // Already lit — deselect just this one
      litTokens.delete(name);
      if (litTokens.size === 0) litTokens = null; // back to All
    } else {
      // Not lit — activate it + all ancestors
      for (const a of getAncestors(name)) {
        litTokens.add(a);
      }
    }
    syncTokenButtons();
    applyFilters();
  }

  function syncTokenButtons() {
    tokenTree.querySelectorAll('.token-btn').forEach(btn => {
      if (btn.classList.contains('token-all')) {
        btn.classList.toggle('active', litTokens === null);
      } else {
        btn.classList.toggle('active', litTokens !== null && litTokens.has(btn.dataset.token));
      }
    });
  }

  // ========== Safety Model ==========
  // An intrinsic is "effectively safe" if:
  //   - it's natively safe (i.u === false), OR
  //   - it has a safe_unaligned_simd wrapper (name match)
  // The "Unsafe" filter shows only truly unsafe intrinsics (no safe path).

  function isEffectivelySafe(i) {
    return !i.u || safeVariantSet.has(i.n);
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

    filtered = allData.intrinsics.filter(i => {
      if (i.a !== activeArch) return false;

      // Token filter: if specific tokens are lit, only show those
      if (litTokens !== null) {
        if (!i.t || !litTokens.has(i.t)) return false;
      }

      // Stability
      const isStable = i.s;
      const showStable = stabilities.has('stable');
      const showUnstable = stabilities.has('unstable');
      if (!showStable && !showUnstable) return false;
      if (!showStable && isStable) return false;
      if (!showUnstable && !isStable) return false;

      // Safety — wrapped intrinsics (unsafe + safe_unaligned_simd) are BOTH,
      // so they pass whenever either Safe or Unsafe is active.
      const hasWrapper = i.u && safeVariantSet.has(i.n);
      const showSafe = safeties.has('safe');
      const showUnsafe = safeties.has('unsafe');
      if (!showSafe && !showUnsafe) return false;
      if (hasWrapper) {
        // Both safe and unsafe — passes if either toggle is on (already checked above)
      } else if (!i.u) {
        // Natively safe — need Safe toggle
        if (!showSafe) return false;
      } else {
        // Truly unsafe (no wrapper) — need Unsafe toggle
        if (!showUnsafe) return false;
      }

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
    scrollContent.style.height = (filtered.length * ROW_HEIGHT) + 'px';
    while (scrollContent.firstChild) scrollContent.removeChild(scrollContent.firstChild);
    renderVisibleRows();
  }

  function renderVisibleRows() {
    const scrollTop = virtualScroll.scrollTop;
    const viewHeight = virtualScroll.clientHeight;
    const startIdx = Math.max(0, Math.floor(scrollTop / ROW_HEIGHT) - OVERSCAN);
    const endIdx = Math.min(filtered.length, Math.ceil((scrollTop + viewHeight) / ROW_HEIGHT) + OVERSCAN);

    scrollContent.querySelectorAll('.vrow').forEach(row => {
      const idx = parseInt(row.dataset.idx);
      if (idx < startIdx || idx >= endIdx) row.remove();
    });

    const existingIdxs = new Set();
    scrollContent.querySelectorAll('.vrow').forEach(row => existingIdxs.add(parseInt(row.dataset.idx)));

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

    const doc = truncateDoc(i.d || '', 80);
    const stableBadge = i.s
      ? '<span class="badge badge-stable">stable</span>'
      : '<span class="badge badge-unstable">nightly</span>';

    let safeBadge;
    if (!i.u) {
      safeBadge = '<span class="badge badge-safe">safe</span>';
    } else if (safeVariantSet.has(i.n)) {
      safeBadge = '<span class="badge badge-safe-wrapped" title="safe via safe_unaligned_simd">safe*</span>';
    } else {
      safeBadge = '<span class="badge badge-unsafe">unsafe</span>';
    }

    row.innerHTML = `
      <div class="col-name">${escHtml(i.n)}</div>
      <div class="col-token">${escHtml(i.t || '—')}</div>
      <div class="col-desc">${escHtml(doc)}</div>
      <div class="col-badges">${stableBadge}${safeBadge}</div>`;

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
      row.classList.toggle('selected', parseInt(row.dataset.idx) === selectedIdx);
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
    const docLinks = extractDocLinks(i.d || '');
    const docText = (i.d || '').split('[')[0].trim();

    let timingHtml = '';
    if (i.tc && allData.timing[i.tc]) {
      const t = allData.timing[i.tc];
      const names = { h: 'Haswell', sk: 'Skylake-X', z4: 'Zen 4', sp: 'Sapphire Rapids' };
      timingHtml = '<div class="detail-timing">';
      for (const [key, label] of Object.entries(names)) {
        const vals = t[key];
        timingHtml += `<div class="timing-col">
          <div class="timing-header">${label}</div>
          <div class="timing-value">${vals ? vals[0] + '/' + vals[1] : '—'}</div>
        </div>`;
      }
      timingHtml += '</div>';
    }

    let safeHtml = '';
    if (i.u && allData.safeVariants[i.n]) {
      const archMod = i.a === 'aarch64' ? 'aarch64' : i.a === 'wasm32' ? 'wasm32' : 'x86_64';
      safeHtml = `<div class="unsafe-note">
        <strong>Unsafe (raw intrinsic):</strong>
        <span class="safe-variant-sig">${escHtml(i.sig || i.n + '(...)')}</span>
      </div>
      <div class="safe-variant-note">
        <strong>Safe (safe_unaligned_simd):</strong>
        <code>safe_unaligned_simd::${archMod}::${escHtml(i.n)}</code><br>
        <span class="safe-variant-sig">${escHtml(allData.safeVariants[i.n])}</span>
      </div>`;
    } else if (i.u) {
      safeHtml = '<div class="unsafe-note">No safe wrapper available. Requires <code>unsafe</code> block.</div>';
    }

    const usageExample = buildUsageExample(i, token);

    let linksHtml = '';
    if (docLinks.length > 0) {
      linksHtml = '<div style="margin-top: 8px;">' +
        docLinks.map(l => `<a class="doc-link" href="${escAttr(l.url)}" target="_blank" rel="noopener">${escHtml(l.text)} ↗</a>`).join(' &nbsp; ') +
        '</div>';
    }

    let tokenRefHtml = '';
    if (token) {
      tokenRefHtml = `<div style="margin-top: 8px;">
        <a class="doc-link" href="tokens/${escAttr(token.name)}.md" target="_blank" rel="noopener">${escHtml(token.name)} reference ↗</a>
      </div>`;
    }

    detailContent.innerHTML = `
      <h2>${escHtml(i.n)}</h2>
      <div class="detail-grid">
        <div class="detail-field"><span class="detail-label">Token</span><span class="detail-value">${escHtml(tokenDisplay)}</span></div>
        <div class="detail-field"><span class="detail-label">Features</span><span class="detail-value">${escHtml(i.f || '—')}</span></div>
        <div class="detail-field"><span class="detail-label">Instruction</span><span class="detail-value">${escHtml(i.ins || '—')}</span></div>
        <div class="detail-field"><span class="detail-label">Stability</span><span class="detail-value">${i.s ? '<span class="badge badge-stable">stable</span>' : '<span class="badge badge-unstable">nightly</span>'}</span></div>
        <div class="detail-field"><span class="detail-label">Safety</span><span class="detail-value">${!i.u ? '<span class="badge badge-safe">safe</span>' : safeVariantSet.has(i.n) ? '<span class="badge badge-unsafe">unsafe</span> <span class="badge badge-safe-wrapped">safe via safe_unaligned_simd</span>' : '<span class="badge badge-unsafe">unsafe</span>'}</span></div>
        <div class="detail-field"><span class="detail-label">Architecture</span><span class="detail-value">${escHtml(i.a)}</span></div>
      </div>
      <div style="margin-bottom: 8px; color: var(--text);">${escHtml(docText)}</div>
      ${linksHtml}${tokenRefHtml}
      <div class="detail-field" style="margin-top: 8px;"><span class="detail-label">Signature</span><span class="detail-value" style="word-break: break-all;">${escHtml(i.sig || '—')}</span></div>
      ${timingHtml}${safeHtml}${usageExample}`;

    detailPanel.style.display = 'block';
    requestAnimationFrame(() => {
      virtualScroll.style.paddingBottom = detailPanel.offsetHeight + 'px';
    });
  }

  function buildUsageExample(i, token) {
    if (!token) return '';
    const tn = token.name;
    const archMod = i.a === 'aarch64' ? 'aarch64' : i.a === 'wasm32' ? 'wasm32' : 'x86_64';
    let code;
    if (i.u) {
      const sv = allData.safeVariants[i.n];
      if (sv) {
        code = `// Preferred: safe wrapper (no unsafe needed)\n#[rite]\nfn example(_: ${tn}, /* params */) {\n    let result = safe_unaligned_simd::${archMod}::${i.n}(/* args */);\n}\n\n// Raw intrinsic (requires unsafe)\n#[rite]\nfn example_raw(_: ${tn}, /* params */) {\n    let result = unsafe { ${i.n}(/* args */) };\n}`;
      } else {
        code = `#[rite]\nfn example(_: ${tn}, /* params */) {\n    let result = unsafe { ${i.n}(/* args */) };\n}`;
      }
    } else {
      code = `#[rite]\nfn example(_: ${tn}, /* params */) {\n    let result = ${i.n}(/* args */);\n}`;
    }
    return `<div class="detail-code"><div class="code-label">Usage with archmage</div><pre>${escHtml(code)}</pre></div>`;
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
    while ((m = re.exec(doc)) !== null) links.push({ text: m[1], url: m[2] });
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
    if (litTokens) params.set('tokens', [...litTokens].join(','));
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
    if (params.has('arch')) activeArch = params.get('arch');
    if (params.has('tokens')) {
      litTokens = new Set(params.get('tokens').split(',').filter(Boolean));
      if (litTokens.size === 0) litTokens = null;
    }
    if (params.has('stability')) {
      const vals = new Set(params.get('stability').split(','));
      document.querySelectorAll('.filter-btn[data-filter="stability"]').forEach(b =>
        b.classList.toggle('active', vals.has(b.dataset.value)));
    }
    if (params.has('safety')) {
      const vals = new Set(params.get('safety').split(','));
      document.querySelectorAll('.filter-btn[data-filter="safety"]').forEach(b =>
        b.classList.toggle('active', vals.has(b.dataset.value)));
    }
    document.querySelectorAll('.arch-btn').forEach(b =>
      b.classList.toggle('active', b.dataset.arch === activeArch));
  }

  // ========== Events ==========

  function setupEvents() {
    let debounceTimer;
    searchInput.addEventListener('input', () => {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(applyFilters, 150);
    });

    // Arch radio
    document.querySelectorAll('.arch-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        if (btn.dataset.arch === activeArch) return;
        activeArch = btn.dataset.arch;
        litTokens = null;
        document.querySelectorAll('.arch-btn').forEach(b =>
          b.classList.toggle('active', b.dataset.arch === activeArch));
        renderTokenTree();
        applyFilters();
      });
    });

    // Stability/safety toggles — at least one must stay active per group
    document.querySelectorAll('.filter-btn[data-filter]').forEach(btn => {
      btn.addEventListener('click', () => {
        const group = btn.dataset.filter;
        const siblings = document.querySelectorAll(`.filter-btn[data-filter="${group}"]`);
        // If this is the last active one in the group, don't toggle off
        if (btn.classList.contains('active')) {
          let activeCount = 0;
          siblings.forEach(b => { if (b.classList.contains('active')) activeCount++; });
          if (activeCount <= 1) return; // can't deactivate the last one
        }
        btn.classList.toggle('active');
        applyFilters();
      });
    });

    virtualScroll.addEventListener('scroll', renderVisibleRows);

    let resizeTimer;
    window.addEventListener('resize', () => {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(renderVisibleRows, 100);
    });

    detailClose.addEventListener('click', () => {
      closeDetail();
      selectedIdx = -1;
      updateSelectedClass();
    });

    document.addEventListener('keydown', (e) => {
      if (e.target === searchInput && (e.key === 'ArrowDown' || e.key === 'ArrowUp')) {
        e.preventDefault();
        selectRow(e.key === 'ArrowDown'
          ? Math.min(selectedIdx + 1, filtered.length - 1)
          : Math.max(selectedIdx - 1, 0));
        if (selectedIdx >= 0) {
          const rowTop = selectedIdx * ROW_HEIGHT;
          const st = virtualScroll.scrollTop;
          const vh = virtualScroll.clientHeight;
          if (rowTop < st) virtualScroll.scrollTop = rowTop;
          else if (rowTop + ROW_HEIGHT > st + vh) virtualScroll.scrollTop = rowTop + ROW_HEIGHT - vh;
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

  init();
})();
