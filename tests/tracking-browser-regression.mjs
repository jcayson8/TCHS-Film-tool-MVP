import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';

const baseUrl = process.env.TCHS_TEST_BASE_URL || 'http://127.0.0.1:8081';
const datasetId = String(process.env.TCHS_TEST_DATASET_ID || '4');
const clipId = String(process.env.TCHS_TEST_CLIP_ID || '5');
const originFrame = Number(process.env.TCHS_TEST_ORIGIN_FRAME || 206);
const range = Number(process.env.TCHS_TEST_RANGE || 15);
const expectUnavailable = process.env.TCHS_TEST_EXPECT_UNAVAILABLE === '1';
const chromePath = process.env.CHROME_PATH || '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';

class DevToolsPipe {
  constructor(child) {
    this.child = child;
    this.nextId = 1;
    this.pending = new Map();
    this.buffer = '';
    child.stdio[4].setEncoding('utf8');
    child.stdio[4].on('data', (chunk) => {
      this.buffer += chunk;
      let boundary;
      while ((boundary = this.buffer.indexOf('\0')) >= 0) {
        const raw = this.buffer.slice(0, boundary);
        this.buffer = this.buffer.slice(boundary + 1);
        if (!raw) continue;
        const message = JSON.parse(raw);
        const pending = this.pending.get(message.id);
        if (!pending) continue;
        this.pending.delete(message.id);
        if (message.error) pending.reject(new Error(message.error.message));
        else pending.resolve(message.result);
      }
    });
  }

  send(method, params = {}, sessionId) {
    const id = this.nextId++;
    const message = { id, method, params };
    if (sessionId) message.sessionId = sessionId;
    this.child.stdio[3].write(`${JSON.stringify(message)}\0`);
    return new Promise((resolve, reject) => this.pending.set(id, { resolve, reject }));
  }
}

const sleep = (milliseconds) => new Promise((resolve) => setTimeout(resolve, milliseconds));

async function waitFor(evaluate, predicate, label, timeout = 30000) {
  const started = Date.now();
  while (Date.now() - started < timeout) {
    const value = await evaluate();
    if (predicate(value)) return value;
    await sleep(100);
  }
  throw new Error(`Timed out waiting for ${label}`);
}

const child = spawn(chromePath, [
  '--headless',
  '--no-sandbox',
  '--disable-gpu',
  '--disable-background-timer-throttling',
  '--disable-renderer-backgrounding',
  '--remote-debugging-pipe',
  '--no-first-run',
  '--no-default-browser-check'
], { stdio: ['ignore', 'ignore', 'inherit', 'pipe', 'pipe'] });

const cdp = new DevToolsPipe(child);
let sessionId;

try {
  const { targetId } = await cdp.send('Target.createTarget', { url: `${baseUrl}/` });
  ({ sessionId } = await cdp.send('Target.attachToTarget', { targetId, flatten: true }));
  await cdp.send('Runtime.enable', {}, sessionId);

  const evaluate = async (expression, userGesture = false) => {
    const result = await cdp.send('Runtime.evaluate', {
      expression,
      awaitPromise: true,
      returnByValue: true,
      userGesture
    }, sessionId);
    if (result.exceptionDetails) {
      throw new Error(result.exceptionDetails.exception?.description || result.exceptionDetails.text);
    }
    return result.result.value;
  };

  await waitFor(
    () => evaluate('document.readyState'),
    (value) => value === 'complete',
    'page load'
  );
  await evaluate(`document.querySelector('[data-destination="film-room"]').click()`, true);
  await waitFor(
    () => evaluate(`document.querySelector('#annotationDataset option[value="${datasetId}"]') !== null`),
    Boolean,
    'dataset options'
  );
  await evaluate(`(() => {
    const select=document.querySelector('#annotationDataset');
    select.value=${JSON.stringify(datasetId)};
    select.dispatchEvent(new Event('change',{bubbles:true}));
  })()`);
  await waitFor(
    () => evaluate(`document.querySelector('#annotationClip option[value="${clipId}"]') !== null`),
    Boolean,
    'assigned clip options'
  );
  await evaluate(`(() => {
    const select=document.querySelector('#annotationClip');
    select.value=${JSON.stringify(clipId)};
    select.dispatchEvent(new Event('change',{bubbles:true}));
  })()`);
  await waitFor(
    () => evaluate(`document.querySelector('#annotationVideo').videoWidth > 0`),
    Boolean,
    'video metadata'
  );
  await evaluate(`new Promise((resolve) => {
    const video=document.querySelector('#annotationVideo');
    video.addEventListener('seeked',resolve,{once:true});
    video.currentTime=${originFrame}/30;
  })`);
  await evaluate(`document.querySelector('#loadNearestFrame').click()`, true);
  await waitFor(
    () => evaluate(`document.querySelector('#annotationSaveState').textContent`),
    (value) => value.includes('Saved'),
    'saved origin frame'
  );

  await evaluate(`(() => {
    window.__trackingSaves=[];
    window.__trackingRequest=null;
    window.__trackingResponse=null;
    const originalFetch=window.fetch.bind(window);
    window.fetch=async (input,options={}) => {
      const url=typeof input==='string'?input:input.url;
      if(url==='/api/training/track-frames'){
        const frameBlobs=options.body.getAll('frames');
        const hashes=await Promise.all(frameBlobs.map(async (blob) => {
          const digest=await crypto.subtle.digest('SHA-256',await blob.arrayBuffer());
          return [...new Uint8Array(digest)].map((value)=>value.toString(16).padStart(2,'0')).join('');
        }));
        window.__trackingRequest={
          frameNumbers:JSON.parse(options.body.get('frame_numbers')),
          frameTimes:JSON.parse(options.body.get('frame_times')),
          hashes
        };
        const response=await originalFetch(input,options);
        window.__trackingResponse=await response.clone().json().catch(()=>null);
        return response;
      }
      if(url==='/api/training/frames'&&options.method==='POST'){
        const body=JSON.parse(options.body);
        window.__trackingSaves.push(body);
        return new Response(JSON.stringify({id:999999,version:window.__trackingSaves.length}),{
          status:200,
          headers:{'Content-Type':'application/json'}
        });
      }
      return originalFetch(input,options);
    };
  })()`);

  const savedFrame = await fetch(
    `${baseUrl}/api/training/frames?datasetId=${datasetId}&clipId=${clipId}&timeMs=${Math.round(originFrame / 30 * 1000)}`
  ).then((response) => response.json());
  assert(savedFrame[0]?.annotations?.length, 'Known origin frame must contain annotations');
  const sourceBox = savedFrame[0].annotations[0];
  await evaluate(`document.querySelector('#selectMode').click()`, true);
  await evaluate(`document.querySelector('#annotationCanvas').scrollIntoView({block:'center'})`);
  await sleep(200);
  const canvasRect = await evaluate(`(() => {
    const rect=document.querySelector('#annotationCanvas').getBoundingClientRect();
    return {left:rect.left,top:rect.top,width:rect.width,height:rect.height};
  })()`);
  const clickX = canvasRect.left + (Number(sourceBox.x) + Number(sourceBox.width) / 2) * canvasRect.width;
  const clickY = canvasRect.top + (Number(sourceBox.y) + Number(sourceBox.height) / 2) * canvasRect.height;
  await cdp.send('Input.dispatchMouseEvent', { type: 'mousePressed', x: clickX, y: clickY, button: 'left', clickCount: 1 }, sessionId);
  await cdp.send('Input.dispatchMouseEvent', { type: 'mouseReleased', x: clickX, y: clickY, button: 'left', clickCount: 1 }, sessionId);
  assert.equal(await evaluate(`document.querySelector('#trackForward').disabled`), false, 'Selected accepted box should be trackable');
  await evaluate(`document.querySelector('#trackingRange').value=${range}`);
  await evaluate(`document.querySelector('#trackForward').click()`, true);
  const trackingStatus = await waitFor(
    () => evaluate(`document.querySelector('#trackingProgress').textContent`),
    (value) => /CSRT|KCF|MOSSE|stopped at frame|unavailable|failed/i.test(value),
    'tracking completion',
    60000
  );
  const trackingResponse = await evaluate('window.__trackingResponse');
  const trackingMessage = await evaluate(`document.querySelector('#toast').textContent`);
  if (expectUnavailable) {
    assert.equal(trackingStatus, 'Tracking unavailable');
    assert.match(trackingMessage, /Tracking unavailable — restart the Node server and try again\./);
    console.log(JSON.stringify({ status: 'passed', unavailableGuidance: trackingMessage }, null, 2));
  } else {
  assert.match(
    trackingStatus,
    new RegExp(`(?:CSRT|KCF|MOSSE) · ${range} frames`,'i'),
    `Expected ${range} successful frames, got "${trackingStatus}" (${trackingMessage}; ${JSON.stringify(trackingResponse)})`
  );

  const trackingRequest = await evaluate('window.__trackingRequest');
  const expectedFrames = Array.from({ length: range }, (_, index) => originFrame + index + 1);
  assert.deepEqual(trackingRequest.frameNumbers, expectedFrames, 'Browser must capture frames 207–221 in order');
  assert.equal(new Set(trackingRequest.frameTimes).size, range, 'Captured frame times must be distinct');
  assert.equal(new Set(trackingRequest.hashes).size, range, 'Captured frame images must be distinct');

  assert.deepEqual(
    trackingResponse.frames.map((frame) => frame.video_frame_number),
    expectedFrames,
    'Proxy response must preserve absolute frame numbers'
  );
  assert.equal(new Set(trackingResponse.frames.map((frame) => frame.frame_time_ms)).size, range);

  const trackedCoordinates = [];
  for (const frameNumber of expectedFrames) {
    await evaluate(`document.querySelector('#nextVideoFrame').click()`, true);
    await waitFor(
      () => evaluate(`document.querySelector('#annotationTime').textContent`),
      (value) => value.startsWith(`Frame: ${frameNumber} `),
      `frame ${frameNumber}`
    );
    const before = await evaluate('window.__trackingSaves.length');
    await evaluate(`document.querySelector('#saveAnnotationFrame').click()`, true);
    await waitFor(
      () => evaluate('window.__trackingSaves.length'),
      (value) => value === before + 1,
      `captured save body for frame ${frameNumber}`
    );
    const body = await evaluate('window.__trackingSaves.at(-1)');
    const tracked = body.annotations.find((box) => box.attributes?.source === 'opencv_tracking');
    assert(tracked, `Frame ${frameNumber} must load its tracked annotation`);
    trackedCoordinates.push([tracked.x, tracked.y, tracked.width, tracked.height]);
  }
  assert.ok(
    new Set(trackedCoordinates.map((coordinates) => coordinates.join(','))).size > 1,
    'Tracked coordinates must change across moving real-footage frames'
  );

  await evaluate(`document.querySelector('#nextVideoFrame').click()`, true);
  await waitFor(
    () => evaluate(`document.querySelector('#annotationTime').textContent`),
    (value) => value.startsWith(`Frame: ${originFrame + range + 1} `),
    'first frame outside tracking range'
  );
  const savesBeforeBoundary = await evaluate('window.__trackingSaves.length');
  await evaluate(`document.querySelector('#saveAnnotationFrame').click()`, true);
  await waitFor(
    () => evaluate('window.__trackingSaves.length'),
    (value) => value === savesBeforeBoundary + 1,
    'captured boundary save body'
  );
  const boundaryBody = await evaluate('window.__trackingSaves.at(-1)');
  assert.equal(
    boundaryBody.annotations.some((box) => box.attributes?.source === 'opencv_tracking'),
    false,
    'The first frame beyond the range must not reuse frame 221 tracking coordinates'
  );

  console.log(JSON.stringify({
    status: 'passed',
    originFrame,
    destinationFrames: expectedFrames,
    distinctCapturedFrames: new Set(trackingRequest.hashes).size,
    distinctTrackedCoordinates: new Set(trackedCoordinates.map((coordinates) => coordinates.join(','))).size,
    boundaryFrame: originFrame + range + 1
  }, null, 2));
  }
} finally {
  if (sessionId) await cdp.send('Browser.close').catch(() => {});
  child.kill('SIGTERM');
}
