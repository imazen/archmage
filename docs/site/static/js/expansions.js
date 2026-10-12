(() => {
    const browser = document.querySelector('.expansion-browser');
    if (!browser) return;
    browser.classList.add('js-enabled');
    const target = browser.querySelector('.expansion-target');
    const jump = browser.querySelector('.expansion-jump');
    target.addEventListener('change', () => {
        for (const panel of browser.querySelectorAll('.expansion-preset')) {
            panel.hidden = panel.dataset.preset !== target.value;
        }
    });
    jump.addEventListener('change', () => {
        document.getElementById(jump.value).scrollIntoView({ block: 'start', behavior: 'instant' });
        history.replaceState(null, '', `#${jump.value}`);
    });
    for (const button of browser.querySelectorAll('button[data-view]')) {
        button.addEventListener('click', () => {
            const active = [...browser.querySelectorAll('.expansion-example')]
                .find(example => example.getBoundingClientRect().bottom > 260);
            const offset = active?.getBoundingClientRect().top;
            browser.dataset.view = button.dataset.view;
            if (active) window.scrollBy({ top: active.getBoundingClientRect().top - offset, behavior: 'instant' });
            for (const other of browser.querySelectorAll('button[data-view]')) {
                other.setAttribute('aria-pressed', String(other === button));
            }
        });
    }
})();
