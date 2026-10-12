(() => {
    const browser = document.querySelector('.expansion-browser');
    if (!browser) return;
    browser.classList.add('js-enabled');
    const target = browser.querySelector('.expansion-target');
    const jump = browser.querySelector('.expansion-jump');
    function preservePosition(update) {
        const cutoff = browser.querySelector('.expansion-toolbar').getBoundingClientRect().bottom + 64;
        const examples = [...browser.querySelectorAll('.expansion-example')];
        const active = examples.find(example => {
            const top = example.getBoundingClientRect().top;
            return top >= cutoff - 64 && top < innerHeight;
        }) || examples.findLast(example => example.getBoundingClientRect().top <= cutoff);
        const offset = active?.getBoundingClientRect().top;
        update();
        if (active) window.scrollBy({ top: active.getBoundingClientRect().top - offset, behavior: 'instant' });
    }
    target.addEventListener('change', () => preservePosition(() => {
        for (const panel of browser.querySelectorAll('.expansion-preset')) {
            panel.hidden = panel.dataset.preset !== target.value;
        }
    }));
    jump.addEventListener('change', () => {
        document.getElementById(jump.value).scrollIntoView({ block: 'start', behavior: 'instant' });
        history.replaceState(null, '', `#${jump.value}`);
    });
    for (const button of browser.querySelectorAll('button[data-view]')) {
        button.addEventListener('click', () => {
            preservePosition(() => { browser.dataset.view = button.dataset.view; });
            for (const other of browser.querySelectorAll('button[data-view]')) {
                other.setAttribute('aria-pressed', String(other === button));
            }
        });
    }
})();
