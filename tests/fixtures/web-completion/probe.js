// Delay delivery only after a real GPU promise has resolved. Never fabricate
// GPU completion, even in the fault-injection tests.
import init, { run } from './mikage_web_completion.js';

const testCase = new URLSearchParams(location.search).get('case') || 'delayed';
const result = { case: testCase, realSubmits: 0, probes: 0, maxPending: 0, errors: [] };
const submit = GPUQueue.prototype.submit;
const done = GPUQueue.prototype.onSubmittedWorkDone;
const waiting = [];
let pending = 0;
let notifications = 0;
const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
function check(condition, message) {
  if (!condition) throw new Error(message);
}

if (testCase === 'timer-error') {
  const setTimeout = globalThis.setTimeout;
  globalThis.setTimeout = (callback, delay, ...args) => {
    if (delay > 0 && delay <= 8) throw new Error('injected completion timer failure');
    return setTimeout(callback, delay, ...args);
  };
}
addEventListener('error', event => result.errors.push(String(event.message)));
addEventListener('unhandledrejection', event => result.errors.push(String(event.reason)));

GPUQueue.prototype.submit = function(commands) {
  const buffers = Array.from(commands);
  if (!buffers.length) {
    result.probes++;
    // Reverse delivery also exercises a later endpoint completing earlier ones.
    if (testCase !== 'slow') {
      for (const release of waiting.splice(0).reverse()) release();
    }
  } else {
    result.realSubmits++;
  }
  return submit.call(this, buffers);
};
GPUQueue.prototype.onSubmittedWorkDone = function() {
  notifications++;
  result.maxPending = Math.max(result.maxPending, ++pending);
  const realCompletion = done.call(this);
  // Attach rejection handling now, including when the test destroys the device.
  const delivery = realCompletion.then(() => new Promise(resolve => {
    if (testCase === 'normal') resolve();
    else if (testCase === 'slow') setTimeout(resolve, 100);
    else waiting.push(resolve);
  }));
  return delivery.finally(() => pending--);
};

try {
  await init();
  const started = performance.now();
  const running = run(testCase);
  if (testCase === 'slow') {
    // Timer probes must not free capacity while the real notification is held.
    const startupDeadline = performance.now() + 5000;
    while (result.realSubmits < 2 && performance.now() < startupDeadline) await sleep(1);
    await sleep(60);
    check(result.realSubmits === 2, 'submission capacity was bypassed');
    check(result.probes > 0, 'no progress retry while blocked');
  }
  await Promise.race([
    running,
    sleep(5000).then(() => { throw new Error('async completion wait stalled'); }),
  ]);
  result.elapsedMs = performance.now() - started;
  const probesAtEnd = result.probes;
  await sleep(40);
  check(result.probes === probesAtEnd, 'completion timer survived completion/shutdown');
  check(result.errors.length === 0, 'unexpected asynchronous error');
  check(result.maxPending <= 2, 'more than two real submissions in flight');
  check(notifications === result.realSubmits, 'a probe registered a new notification');
  if (['shutdown', 'drop', 'device-loss', 'timer-error'].includes(testCase)) {
    check(result.realSubmits === 1, 'unexpected work after shutdown');
    // Browsers may delay the device-lost notification too. Retries must stop
    // once run() observes it; the stable probesAtEnd check above covers that.
    if (testCase !== 'device-loss') check(result.probes === 0, 'probe after shutdown');
  } else {
    check(result.realSubmits === 6 && pending === 0, 'real work did not complete');
    if (testCase !== 'normal') check(result.probes > 0, 'delayed notifications were not retried');
  }
  result.ok = true;
} catch (error) {
  result.ok = false;
  result.errors.push(String(error.stack || error));
}
globalThis.__webCompletionResult = result;
if (globalThis.document) {
  document.querySelector('#result').textContent = JSON.stringify(result, null, 2);
} else {
  postMessage(result);
}
