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

  // DOM refs
  const searchInput = document.getElementById('searchInput');
  const resultCount = document.getElementById('resultCount');
  const tokenFilter = document.getElementById('tokenFilter');
  const virtualScroll = document.getElementById('virtualScroll');
  const scrollContent = document.getElementById('scrollContent');
  const detailPanel = document.getElementById('detailPanel');
  const detailContent = document.getElementById('detailContent');
  const detailClose = document.getElementById('detailClose');
  const tokenLinks = document.getElementById('tokenLinks');

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

    populateTokenFilter();
    populateTokenLinks();
    restoreState();
    applyFilters();
    setupEvents();

    searchInput.focus();
  }

  function populateTokenFilter() {
    // Group by arch
    const groups = { x86_64: [], aarch64: [], wasm32: [] };
    for (const t of allData.tokens) {
      const arch = t.arch || 'x86_64';
      if (!groups[arch]) groups[arch] = [];
      const label = t.aliases && t.aliases.length > 0
        ? `${t.display} (${t.aliases[0]})`
        : t.display;
      groups[arch].push({ value: t.name, label, arch });
    }

    for (const [arch, tokens] of Object.entries(groups)) {
      if (tokens.length === 0) continue;
      const group = document.createElement('optgroup');
      group.label = arch === 'x86_64' ? 'x86' : arch === 'aarch64' ? 'ARM' : 'WASM';
      for (const t of tokens) {
        const opt = document.createElement('option');
        opt.value = t.value;
        opt.textContent = t.label;
        opt.dataset.arch = t.arch;
        group.appendChild(opt);
      }
      tokenFilter.appendChild(group);
    }
  }

  function populateTokenLinks() {
    const label = tokenLinks.querySelector('.token-links-label');
    while (tokenLinks.lastChild !== label) {
      tokenLinks.removeChild(tokenLinks.lastChild);
    }

    for (const t of allData.tokens) {
      const a = document.createElement('a');
      const display = t.aliases && t.aliases.length > 0 ? t.aliases[0] : t.display;
      a.textContent = display;
      a.href = `tokens/${t.name}.md`;
      a.target = '_blank';
      a.rel = 'noopener';
      const archClass = t.arch === 'aarch64' ? 'arch-arm' : t.arch === 'wasm32' ? 'arch-wasm' : 'arch-x86';
      a.className = archClass;
      tokenLinks.appendChild(a);
    }
  }

  // ========== Filtering ==========

  function getActiveValues(filterName) {
    const btns = document.querySelectorAll(`.filter-btn[data-filter="${filterName}"]`);
    const active = [];
    btns.forEach(b => { if (b.classList.contains('active')) active.push(b.dataset.value); });
    return active;
  }

  function ensureArchEnabled(arch) {
    const btn = document.querySelector(`.filter-btn[data-filter="arch"][data-value="${arch}"]`);
    if (btn && !btn.classList.contains('active')) {
      btn.classList.add('active');
    }
  }

  function applyFilters() {
    const query = searchInput.value.toLowerCase().trim();
    const archs = new Set(getActiveValues('arch'));
    const tokenVal = tokenFilter.value;
    const stabilities = new Set(getActiveValues('stability'));
    const safeties = new Set(getActiveValues('safety'));

    filtered = allData.intrinsics.filter(i => {
      // Architecture filter
      if (!archs.has(i.a)) return false;

      // Token filter
      if (tokenVal && i.t !== tokenVal) return false;

      // Stability toggle: both on = all, both off = none, one on = that one
      const isStable = i.s;
      const showStable = stabilities.has('stable');
      const showUnstable = stabilities.has('unstable');
      if (!showStable && !showUnstable) return false;
      if (!showStable && isStable) return false;
      if (!showUnstable && !isStable) return false;

      // Safety toggle: same logic
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
    detailPanel.style.display = 'none';
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

    const token = i.t ? tokenMap[i.t] : null;
    const tokenDisplay = token
      ? (token.aliases && token.aliases.length > 0 ? token.aliases[0] : token.display)
      : '—';

    const doc = truncateDoc(i.d || '', 80);
    const stableBadge = i.s
      ? '<span class="badge badge-stable">stable</span>'
      : '<span class="badge badge-unstable">nightly</span>';

    let safeBadge;
    if (i.u) {
      if (safeVariantSet.has(i.n)) {
        // Unsafe but has a safe_unaligned_simd wrapper
        safeBadge = '<span class="badge badge-has-safe" title="safe_unaligned_simd wrapper available">unsafe*</span>';
      } else {
        safeBadge = '<span class="badge badge-unsafe">unsafe</span>';
      }
    } else {
      safeBadge = '<span class="badge badge-safe">safe</span>';
    }

    row.innerHTML = `
      <div class="col-name">${escHtml(i.n)}</div>
      <div class="col-token">${escHtml(tokenDisplay)}</div>
      <div class="col-desc">${escHtml(doc)}</div>
      <div class="col-badges">${stableBadge}${safeBadge}</div>
    `;

    row.addEventListener('click', () => selectRow(idx));
    return row;
  }

  function selectRow(idx) {
    if (selectedIdx === idx) {
      selectedIdx = -1;
      detailPanel.style.display = 'none';
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

  function showDetail(i) {
    const token = i.t ? tokenMap[i.t] : null;
    const tokenDisplay = token
      ? `${token.name}${token.aliases && token.aliases.length > 0 ? ' (' + token.aliases.join(', ') + ')' : ''}`
      : 'Not covered by archmage';

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

    // Safe variant — prominent display for unsafe intrinsics with safe wrappers
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
      <div class="detail-field" style="margin-top: 8px;">
        <span class="detail-label">Signature</span>
        <span class="detail-value" style="word-break: break-all;">${escHtml(i.sig || '—')}</span>
      </div>
      ${timingHtml}
      ${safeHtml}
      ${usageExample}
    `;

    detailPanel.style.display = 'block';
  }

  function buildUsageExample(i, token) {
    if (!token) return '';

    const tokenName = token.aliases && token.aliases.length > 0 ? token.aliases[0] : token.name;
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

    const archs = getActiveValues('arch');
    if (archs.length < 3) params.set('arch', archs.join(','));

    if (tokenFilter.value) params.set('token', tokenFilter.value);

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
      const archs = new Set(params.get('arch').split(','));
      document.querySelectorAll('.filter-btn[data-filter="arch"]').forEach(b => {
        b.classList.toggle('active', archs.has(b.dataset.value));
      });
    }

    if (params.has('token')) {
      tokenFilter.value = params.get('token');
      // Auto-enable arch for the selected token
      const selectedToken = tokenMap[params.get('token')];
      if (selectedToken) ensureArchEnabled(selectedToken.arch);
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
  }

  // ========== Events ==========

  function setupEvents() {
    // Search with debounce
    let debounceTimer;
    searchInput.addEventListener('input', () => {
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(applyFilters, 150);
    });

    // All toggle buttons (arch, stability, safety)
    document.querySelectorAll('.filter-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        btn.classList.toggle('active');
        applyFilters();
      });
    });

    // Token dropdown — auto-enable the matching arch
    tokenFilter.addEventListener('change', () => {
      if (tokenFilter.value) {
        const selectedToken = tokenMap[tokenFilter.value];
        if (selectedToken) {
          ensureArchEnabled(selectedToken.arch);
        }
      }
      applyFilters();
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
      detailPanel.style.display = 'none';
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
          detailPanel.style.display = 'none';
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
