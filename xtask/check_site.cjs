// Browser audit of the attunement path. Requires Playwright and its Chromium.
// Use NODE_PATH when Playwright is installed outside this workspace.
// node xtask/check_site.cjs OUTPUT_DIR [LOCAL_SITE_DIR]
// Without LOCAL_SITE_DIR this checks the deployed GitHub Pages site.
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const path = require('node:path');
const { chromium } = require('playwright');

const [output, local] = process.argv.slice(2);
assert(output, 'provide a screenshot/report output directory');
const base = 'https://imazen.github.io/archmage/';
const pages = ['', 'attunement/', 'attunement/first-kernel/',
    'attunement/helpers/', 'attunement/vectors/', 'attunement/reference/'];

(async () => {
    await fs.mkdir(output, { recursive: true });
    const browser = await chromium.launch();
    const report = [];
    try {
        for (const viewport of [{ width: 1440, height: 1000 }, { width: 390, height: 844 }]) {
            const context = await browser.newContext({ viewport });
            if (local) {
                await context.route(`${base}**`, async route => {
                    const url = new URL(route.request().url());
                    let relative = decodeURIComponent(url.pathname.slice('/archmage/'.length));
                    if (!path.extname(relative)) relative += `${relative.endsWith('/') ? '' : '/'}index.html`;
                    if (relative.startsWith('/')) relative = relative.slice(1);
                    const file = path.resolve(local, relative);
                    assert(file.startsWith(`${path.resolve(local)}/`), 'path escaped site root');
                    try {
                        await route.fulfill({ path: file });
                    } catch (error) {
                        await route.fulfill({ status: 404, body: String(error) });
                    }
                });
            }
            const page = await context.newPage();
            const errors = [];
            const links = new Set();
            page.on('pageerror', error => errors.push(String(error)));
            page.on('response', response => {
                if (response.url().startsWith(base) && response.status() >= 400)
                    errors.push(`${response.status()} ${response.url()}`);
            });
            for (const entry of pages) {
                const response = await page.goto(`${base}${entry}`, { waitUntil: 'networkidle' });
                assert.equal(response.status(), 200, entry);
                assert(await page.locator('h1').first().isVisible(), `missing visible heading: ${entry}`);
                const overflow = await page.evaluate(() => document.documentElement.scrollWidth - innerWidth);
                await page.screenshot({ path: path.join(output, `${viewport.width}-${entry.replaceAll('/', '-') || 'home'}.png`) });
                assert(overflow <= 2, `horizontal page overflow ${overflow}px: ${entry}`);
                for (const href of await page.locator('main a[href]').evaluateAll(nodes => nodes.map(n => n.href)))
                    if (href.startsWith(base)) links.add(href);
                report.push({ page: entry || '/', width: viewport.width, overflow });
            }
            // Check every in-site link in the path's content, including fragments.
            for (const href of links) {
                // A fragment-only navigation otherwise has no HTTP response.
                await page.goto('about:blank');
                const response = await page.goto(href, { waitUntil: 'domcontentloaded' });
                assert.equal(response.status(), 200, href);
                const id = decodeURIComponent(new URL(href).hash.slice(1));
                if (id) assert(await page.evaluate(id => !!document.getElementById(id), id), `missing anchor: ${href}`);
            }
            await page.goto(`${base}attunement/`, { waitUntil: 'networkidle' });
            if (viewport.width < 1024) {
                await page.locator('.navbar label[for="my-drawer-2"]').click();
                assert(await page.locator('#my-drawer-2').isChecked(), 'mobile navigation did not open');
                await page.locator('#sidebar a').filter({ hasText: '1. Attune a Kernel' }).click();
            } else {
                await page.locator('main').getByRole('link', { name: 'Attune a kernel', exact: true }).click();
            }
            await page.waitForURL(/\/attunement\/first-kernel\/?$/);
            assert.deepEqual(errors, [], 'browser errors or missing site resources');
            await context.close();
        }
        await fs.writeFile(path.join(output, 'report.json'), `${JSON.stringify(report, null, 2)}\n`);
        console.log(`PASS: desktop/mobile rendering, content links, anchors, and navigation (${local ? 'local' : 'deployed'}).`);
    } finally {
        await browser.close();
    }
})().catch(error => { console.error(error); process.exitCode = 1; });
