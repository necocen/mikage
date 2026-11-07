import init, {
  start, pause, resume, pause_in_hook, publish_snapshot, snapshot,
} from './mikage_simulation_control.js';

const policy = new URLSearchParams(location.search).get('policy') || 'fixed';
const result = { policy, errors: [], samples: {} };
const queue = { real: 0, probes: 0 };
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
const state = () => JSON.parse(snapshot());
function check(condition, message, detail = state()) {
  if (!condition) throw new Error(`${message}: ${JSON.stringify(detail)}`);
}
async function waitFor(predicate, message) {
  const deadline = performance.now() + 10000;
  let current;
  while (performance.now() < deadline) {
    current = state();
    if (predicate(current)) return current;
    await sleep(10);
  }
  throw new Error(`${message}: ${JSON.stringify(current)}`);
}
async function pausedIdle() {
  pause();
  await waitFor(s => s.completed === s.submitted && s.frames > 0, 'paused work did not complete');
  // GUI setup may schedule one follow-up repaint. Wait for a stable idle before
  // asserting no per-tick encoders or their notifications are generated.
  let previous = state();
  let previousRaw = queue.real + queue.probes;
  let stableSince = performance.now();
  const deadline = performance.now() + 5000;
  while (performance.now() < deadline) {
    await sleep(40);
    const current = state();
    const currentRaw = queue.real + queue.probes;
    if (current.submitted !== previous.submitted || currentRaw !== previousRaw) {
      stableSince = performance.now();
    }
    if (current.completed === current.submitted && performance.now() - stableSince >= 200) {
      check(current.paused, 'pause request was lost', current);
      return current;
    }
    previous = current;
    previousRaw = currentRaw;
  }
  throw new Error(`paused runner never idled: ${JSON.stringify(previous)}`);
}

addEventListener('error', event => result.errors.push(String(event.message)));
addEventListener('unhandledrejection', event => result.errors.push(String(event.reason)));

try {
  const submit = GPUQueue.prototype.submit;
  GPUQueue.prototype.submit = function(commands) {
    const buffers = Array.from(commands);
    queue[buffers.length ? 'real' : 'probes']++;
    return submit.call(this, buffers);
  };
  await init();
  start(policy);
  await waitFor(s => s.ready && s.completedFrames > 0 && (policy !== 'fixed' || s.ticks >= 20),
                'window runner did not start');

  const idle = await pausedIdle();
  const submitsAtIdle = queue.real + queue.probes;
  await sleep(250);
  const stillIdle = state();
  check(stillIdle.ticks === idle.ticks && stillIdle.tickSubmits === idle.tickSubmits,
        'automatic ticks or their submits continued while paused', stillIdle);
  check(stillIdle.submitted === idle.submitted && stillIdle.completed === idle.completed,
        'idle pause kept submitting GPU work', stillIdle);
  check(queue.real + queue.probes === submitsAtIdle,
        'idle pause kept submitting raw WebGPU work', queue);
  result.samples.paused = stillIdle;

  const worker = new Worker('./snapshot-worker.js');
  try {
    await new Promise((resolve, reject) => {
      const timeout = setTimeout(() => reject(new Error('CPU Worker did not respond')), 3000);
      worker.onmessage = ({ data }) => {
        publish_snapshot(data.value);
        clearTimeout(timeout);
        resolve();
      };
      worker.onerror = event => reject(new Error(event.message));
      worker.postMessage({ value: 173 });
    });
  } finally {
    worker.terminate();
  }
  const updated = await waitFor(s => s.renderedSnapshot === 173 && s.completedFrames > idle.completedFrames,
                                'Worker snapshot redraw did not complete');
  check(updated.ticks === idle.ticks && updated.tickSubmits === idle.tickSubmits,
        'Worker snapshot redraw generated a tick', updated);
  result.samples.workerRedraw = updated;

  const resumedAt = performance.now();
  resume();
  publish_snapshot(174);
  if (policy === 'manual') {
    await waitFor(s => s.renderedSnapshot === 174, 'Manual redraw did not happen');
    check(state().ticks === 0, 'Manual resumed into automatic ticks');
  } else {
    await waitFor(s => s.ticks > idle.ticks, 'resume did not restart automatic ticks');
    await sleep(policy === 'fixed' ? 60 : 20);
    const resumed = await pausedIdle();
    const wallSeconds = (performance.now() - resumedAt) / 1000;
    check(resumed.elapsed - idle.elapsed <= wallSeconds + 0.003,
          'resume accumulated paused wall time', resumed);
    if (policy === 'fixed') {
      check(Math.abs(resumed.elapsed - idle.elapsed - (resumed.ticks - idle.ticks) * 0.0004) < 1e-6,
            'resume changed fixed dt', resumed);
    } else {
      check(resumed.lastDt < 0.2, 'PerRedraw dt included the long pause', resumed);
    }
    result.samples.resumed = resumed;

    for (const hook of ['tick', 'submit', 'complete', 'gui']) {
      const before = await pausedIdle();
      pause_in_hook(hook);
      resume();
      publish_snapshot(before.snapshot + 1);
      const stopped = await waitFor(s => s.hookPauses > before.hookPauses && s.paused,
                                    `${hook} pause was not delivered`);
      await sleep(60);
      const after = state();
      check(after.ticks === stopped.pauseBoundary,
            `${hook} pause allowed a later automatic tick`, after);
      result.samples[`${hook}Pause`] = after;
    }

    // The final requested state is running, but the pause/resume transition must
    // still clear stale scheduling time before another automatic tick starts.
    const beforeRapid = await pausedIdle();
    resume();
    pause();
    resume();
    publish_snapshot(beforeRapid.snapshot + 1);
    await waitFor(s => s.ticks > beforeRapid.ticks, 'rapid pause/resume lost its wake');
    result.samples.rapidTransition = await pausedIdle();
  }
  check(result.errors.length === 0, 'unexpected browser error', result.errors);
  result.ok = true;
} catch (error) {
  result.ok = false;
  result.errors.push(`${String(error)}\n${error.stack || ''}`);
  try { pause(); } catch (_) {}
}
globalThis.__simulationControlResult = result;
result.queue = queue;
document.querySelector('#status').textContent = `${result.ok ? 'PASS' : 'FAIL'}: ${policy}`;
document.title = `${result.ok ? 'PASS' : 'FAIL'}: ${policy} simulation control`;
document.querySelector('#result').textContent = JSON.stringify(result, null, 2);
