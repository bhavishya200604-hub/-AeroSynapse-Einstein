// Records a walkthrough of the Streamlit workbench using every workspace and option.
// Usage: node record_ui.js <outdir>   (server must run on http://localhost:8517)
const { chromium } = require('playwright');
const OUT = process.argv[2];
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

const CURSOR = `
(() => {
  if (window.__cursor) return; window.__cursor = 1;
  const c = document.createElement('div');
  c.style.cssText = 'position:fixed;z-index:999999;width:22px;height:22px;border-radius:50%;pointer-events:none;' +
    'border:2px solid #38f2ff;box-shadow:0 0 14px #38f2ff,0 0 4px #fff inset;transform:translate(-50%,-50%);' +
    'transition:left .25s ease, top .25s ease;left:-50px;top:-50px';
  document.documentElement.appendChild(c);
  document.addEventListener('mousemove', e => { c.style.left = e.clientX + 'px'; c.style.top = e.clientY + 'px'; }, true);
  document.addEventListener('mousedown', e => {
    const r = document.createElement('div');
    r.style.cssText = 'position:fixed;z-index:999998;pointer-events:none;border-radius:50%;border:3px solid #ff3ec9;' +
      'left:' + e.clientX + 'px;top:' + e.clientY + 'px;width:10px;height:10px;transform:translate(-50%,-50%);' +
      'transition:all .6s ease-out;opacity:1';
    document.documentElement.appendChild(r);
    requestAnimationFrame(() => { r.style.width = '70px'; r.style.height = '70px'; r.style.opacity = '0'; });
    setTimeout(() => r.remove(), 700);
  }, true);
})();`;

(async () => {
  const b = await chromium.launch();
  const ctx = await b.newContext({ viewport: { width: 1280, height: 720 }, recordVideo: { dir: OUT, size: { width: 1280, height: 720 } } });
  const pg = await ctx.newPage();
  await pg.addInitScript(CURSOR);
  const move = async (loc) => { const bb = await loc.boundingBox(); if (bb) { await pg.mouse.move(bb.x + bb.width / 2, bb.y + bb.height / 2, { steps: 18 }); await sleep(250); } };
  const click = async (loc) => { await loc.scrollIntoViewIfNeeded(); await move(loc); await loc.click(); await sleep(600); };
  const idle = async (extra = 1500) => {   // wait until Streamlit finished rerunning
    await sleep(500);
    for (let i = 0; i < 240; i++) { const n = await pg.locator('[data-testid="stStatusWidget"]').count(); if (n === 0) break; await sleep(500); }
    await sleep(extra);
  };
  const ws = async (name) => { await click(pg.locator('[data-testid="stSidebar"] label').filter({ hasText: new RegExp('^' + name + '$') })); await idle(); await pg.mouse.wheel(0, -3000); };
  const pick = async (label, option) => {
    const box = pg.locator('[data-testid="stSelectbox"]').filter({ hasText: label }).first();
    await click(box.locator('div[data-baseweb="select"]'));
    await sleep(500);
    await click(pg.locator('li[role="option"]').filter({ hasText: new RegExp('^' + option.replace(/[()]/g, '\\$&') + '$') }).first());
    await idle(800);
  };
  const slide = async (label, steps) => {
    const s = pg.locator('[data-testid="stSlider"]').filter({ hasText: label }).first().locator('[role="slider"]');
    await click(s);
    for (let i = 0; i < Math.abs(steps); i++) { await pg.keyboard.press(steps > 0 ? 'ArrowRight' : 'ArrowLeft'); await sleep(40); }
    await idle(600);
  };
  const slideTo = async (label, frac) => {
    const tr = pg.locator('[data-testid="stSlider"]').filter({ hasText: label }).first().locator('div[data-baseweb="slider"]').first();
    await tr.scrollIntoViewIfNeeded();
    const bb = await tr.boundingBox();
    if (bb) { const x = bb.x + frac * bb.width, y = bb.y + bb.height / 2; await pg.mouse.move(x, y, { steps: 20 }); await pg.mouse.click(x, y); }
    await idle(700);
  };
  const button = async (name) => { await click(pg.getByRole('button', { name }).first()); };
  const scroll = async (dy, times = 6) => { for (let i = 0; i < times; i++) { await pg.mouse.wheel(0, dy); await sleep(350); } };

  await pg.goto('http://localhost:8517', { waitUntil: 'networkidle' });
  await idle(2500);

  // 1. Physics workspace: every predefined problem
  for (const p of ['burgers1d', 'convdiff1d', 'convdiff_steady', 'heat1d', 'heat2d', 'poisson2d', 'wave1d']) {
    await pick('Problem', p); await scroll(250, 3); await sleep(700); await scroll(-250, 3);
  }

  // 2. Simulation workspace: every solver and option
  await ws('Simulation');
  await slideTo('Péclet number', 0.4);
  for (const sch of ['quick', 'central', 'upwind', 'vanleer']) {
    await pick('Scheme', sch); await button('Run simulation'); await idle(1200); await scroll(300, 3); await sleep(900); await scroll(-300, 3);
  }
  await pick('Solver', 'heat 1D FD');
  await button('Run simulation'); await idle(1000);
  await slide('theta', -1);  // explicit Euler -> stability guard
  await slideTo('Nodes', 0.6); await button('Run simulation'); await idle(1500);
  await pick('Solver', 'Poisson 2D FD'); await button('Run simulation'); await idle(1500); await scroll(300, 4); await sleep(1200); await scroll(-300, 4);
  await pick('Solver', 'lid-driven cavity (NS)'); await button('Run simulation'); await idle(2000); await scroll(300, 4); await sleep(1500); await scroll(-300, 4);
  await pick('Solver', 'airfoil panel method');
  const ti = pg.locator('[data-testid="stTextInput"] input').first();
  await click(ti); await ti.fill(''); await ti.type('4415', { delay: 120 }); await pg.keyboard.press('Enter'); await idle();
  await slideTo('alpha', 0.7); await button('Run simulation'); await idle(1500); await scroll(300, 3); await sleep(1500); await scroll(-300, 3);

  // 3. ML workspace: options + background training with progress
  await ws('ML');
  await slideTo('Péclet number (PINN)', 0.1);
  for (const s of ['rad', 'rar', 'r3', 'lhs', 'sobol', 'uniform', 'rad']) await pick('Collocation sampler', s);
  await slideTo('Iterations', 0.05);
  await button('Train PINN in background');
  for (let i = 0; i < 40; i++) { await sleep(1500); await pg.mouse.wheel(0, 0); const done = await pg.getByText('relative L2 error').count(); if (done) break; await pg.locator('body').press('r').catch(() => {}); }
  await idle(1500); await scroll(350, 8); await sleep(1500); await scroll(-350, 8);

  // 4. Optimisation workspace: optimisers + protocol runs
  await ws('Optimization');
  for (const m of ['ga', 'sqp', 'bo', 'cbo', 'surrogate_tr', 'de']) await pick('Optimiser', m);
  await slideTo('Evaluation budget', 0.08);
  await button('Run optimisation (12-step protocol)'); await idle(2500);
  await scroll(350, 6); await sleep(2000); await scroll(350, 6); await sleep(1500); await scroll(-350, 12);
  await pick('Optimiser', 'surrogate_tr'); await button('Run optimisation (12-step protocol)'); await idle(2500);
  await scroll(350, 6); await sleep(1500); await scroll(-350, 6);

  // 5. Research workspace: records, exports, end-to-end workflow
  await ws('Research');
  for (const e of ['phase1_verification', 'phase3_adaptive_collocation', 'phase6_fluids_aero', 'phase7_validation_engine']) {
    await pick('Experiment', e); await scroll(250, 3); await sleep(800); await scroll(-250, 3);
  }
  await button('Export metrics (JSON)'); await sleep(800);
  await button('Export scalar metrics (CSV)'); await sleep(800);
  await scroll(400, 4);
  await button('Run quick workflow in background');
  for (let i = 0; i < 60; i++) { await sleep(1500); const d = await pg.getByText('Download HTML report').count(); if (d) break; await pg.mouse.move(640, 400 + (i % 2) * 10); await pg.locator('body').press('r').catch(() => {}); }
  await idle(1500); await scroll(300, 4); await sleep(2500);
  await ctx.close(); await b.close();
  console.log('done');
})();
