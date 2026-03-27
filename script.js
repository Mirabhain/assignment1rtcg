/* ============================================================
   Feature extraction: FAST + ORB (own JS implementation)
   No external CV library used
   ============================================================ */

// ─── Logging ────────────────────────────────────────────────
const logBody = document.getElementById('logBody');
function log(msg, type = '') {
  const el = document.createElement('div');
  el.className = 'log-line ' + type;
  el.textContent = msg;
  logBody.appendChild(el);
  logBody.scrollTop = logBody.scrollHeight;
}

// ─── Progress ───────────────────────────────────────────────
const progressWrap  = document.getElementById('progressWrap');
const progressFill  = document.getElementById('progressFill');
const progressLabel = document.getElementById('progressLabel');
function setProgress(pct, label) {
  progressWrap.classList.add('visible');
  progressFill.style.width = pct + '%';
  progressLabel.textContent = label;
  if (pct >= 100) setTimeout(() => progressWrap.classList.remove('visible'), 800);
}

// ─── Status pill ────────────────────────────────────────────
const pill = document.getElementById('statusPill');
function setStatus(text, cls = '') {
  pill.textContent = text;
  pill.className = 'status-pill ' + cls;
}

// ─── Globals ────────────────────────────────────────────────
let sourceImg    = null;
let featureCanvas = null;

// ─── UI Elements ────────────────────────────────────────────
const fileInput   = document.getElementById('fileInput');
const uploadBtn   = document.getElementById('uploadBtn');
const generateBtn = document.getElementById('generateBtn');
const saveBtn     = document.getElementById('saveBtn');
const saveGrayBtn = document.getElementById('saveGrayBtn');

uploadBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleUpload);
generateBtn.addEventListener('click', runPipeline);
saveBtn.addEventListener('click', saveFeatureImage);       // FIX: saveBtn only saves feature image
saveGrayBtn.addEventListener('click', saveGrayscaleImage); // FIX: saveGrayBtn saves grayscale

// ─── Upload handler ─────────────────────────────────────────
function handleUpload(e) {
  const file = e.target.files[0];
  if (!file) return;
  log('Uploading: ' + file.name, 'info');
  const reader = new FileReader();
  reader.onload = ev => {
    const img = new Image();
    img.onload = () => {
      sourceImg = img;
      drawOriginal(img);
      generateBtn.disabled = false;
      saveBtn.disabled = true;
      setStatus('IMAGE LOADED', 'ready');
      log('Image loaded — ' + img.width + 'x' + img.height + 'px', 'ok');
      ['canvas1','canvas2'].forEach(id => {
        document.getElementById(id).style.display = 'none';
        document.getElementById(id.replace('canvas','ph')).style.display = 'flex';
      });
      document.getElementById('badge1').textContent = '— pending';
      document.getElementById('badge2').textContent = '— pending';
      document.getElementById('statsBar').style.display = 'none';
    };
    img.src = ev.target.result;
  };
  reader.readAsDataURL(file);
  fileInput.value = '';
}

function drawOriginal(img) {
  const c = document.getElementById('canvas0');
  const ctx = c.getContext('2d');
  c.width = img.width; c.height = img.height;
  ctx.drawImage(img, 0, 0);
  c.style.display = 'block';
  document.getElementById('ph0').style.display = 'none';
  document.getElementById('badge0').textContent = img.width + ' × ' + img.height;
}

// ─── Main Pipeline ──────────────────────────────────────────
async function runPipeline() {
  if (!sourceImg) return;
  generateBtn.disabled = true;
  saveBtn.disabled = true;
  setStatus('PROCESSING', 'ready');

  const method = document.getElementById('methodSelect').value;
  const thresh = parseFloat(document.getElementById('threshSelect').value);
  const t0     = performance.now();

  // Step 1: Grayscale
  setProgress(10, 'Step 1/3 — Converting to grayscale...');
  log('Converting RGB → Grayscale...', 'info');
  await tick();

  const { gray, width, height } = rgbToGrayscale(sourceImg);
  renderGray(gray, width, height);
  log('Grayscale done.', 'ok');
  await tick();

  // Step 2: Gaussian Blur
  setProgress(35, 'Step 2/3 — Applying Gaussian blur...');
  const blurred = gaussianBlur(gray, width, height);
  log('Gaussian blur applied.', 'ok');
  await tick();

  // Step 3: Feature Detection
  setProgress(55, 'Step 3/3 — Detecting features (' + method.toUpperCase() + ')...');
  let keypoints;

  if (method === 'orb') {
    log('Running ORB Detector (FAST keypoints + orientation)...', 'info');
    keypoints = orbDetect(blurred, width, height, thresh);
  } else {
    log('Running FAST Corner Detector...', 'info');
    keypoints = fastCorners(blurred, width, height, thresh);
  }

  const t1 = performance.now();
  setProgress(90, 'Rendering keypoints...');
  await tick();

  renderFeatures(sourceImg, keypoints, width, height, method);
  setProgress(100, 'Done.');
  featureCanvas = document.getElementById('canvas2');

  // Stats
  document.getElementById('statKP').textContent     = keypoints.length;
  document.getElementById('statSize').textContent   = width + '×' + height;
  document.getElementById('statMethod').textContent = method.toUpperCase();
  document.getElementById('statTime').textContent   = Math.round(t1 - t0);
  document.getElementById('statsBar').style.display = 'flex';

  generateBtn.disabled = false;
  saveBtn.disabled = false;
  setStatus('COMPLETE', 'ready');
  log('Feature detection complete — ' + keypoints.length + ' keypoints found.', 'ok');
}

function tick() { return new Promise(r => setTimeout(r, 0)); }

// ─── 1. RGB → Grayscale ─────────────────────────────────────
// Formula: Y = 0.299R + 0.587G + 0.114B  (ITU-R BT.601)
function rgbToGrayscale(img) {
  const offscreen = document.createElement('canvas');
  offscreen.width  = img.width;
  offscreen.height = img.height;
  const ctx = offscreen.getContext('2d');
  ctx.drawImage(img, 0, 0);
  const imgData = ctx.getImageData(0, 0, img.width, img.height);
  const src  = imgData.data;
  const gray = new Float32Array(img.width * img.height);
  for (let i = 0; i < gray.length; i++) {
    const p = i * 4;
    gray[i] = 0.299 * src[p] + 0.587 * src[p+1] + 0.114 * src[p+2];
  }
  return { gray, width: img.width, height: img.height };
}

function renderGray(gray, w, h) {
  const c   = document.getElementById('canvas1');
  const ctx = c.getContext('2d');
  c.width = w; c.height = h;
  const id = ctx.createImageData(w, h);
  for (let i = 0; i < gray.length; i++) {
    const v = Math.round(gray[i]);
    id.data[i*4]   = v;
    id.data[i*4+1] = v;
    id.data[i*4+2] = v;
    id.data[i*4+3] = 255;
  }
  ctx.putImageData(id, 0, 0);
  c.style.display = 'block';
  document.getElementById('ph1').style.display = 'none';
  document.getElementById('badge1').textContent = w + ' × ' + h + ' | grayscale';
}

// ─── 2. Gaussian Blur (5×5 kernel) ──────────────────────────
function gaussianBlur(gray, w, h) {
  const kernel = [
    2,  4,  5,  4, 2,
    4,  9, 12,  9, 4,
    5, 12, 15, 12, 5,
    4,  9, 12,  9, 4,
    2,  4,  5,  4, 2
  ];
  const kSum = kernel.reduce((a,b) => a+b, 0); // 159
  const out  = new Float32Array(w * h);
  const r    = 2;

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let acc = 0;
      for (let ky = -r; ky <= r; ky++) {
        for (let kx = -r; kx <= r; kx++) {
          const ny = Math.min(Math.max(y + ky, 0), h-1);
          const nx = Math.min(Math.max(x + kx, 0), w-1);
          const ki = (ky+r)*5 + (kx+r);
          acc += gray[ny*w + nx] * kernel[ki];
        }
      }
      out[y*w + x] = acc / kSum;
    }
  }
  return out;
}

// ─── 3. FAST Corner Detector ────────────────────────────────
// Tests 16 pixels on a Bresenham circle radius-3.
// A point is a corner if N=9 consecutive pixels are all
// brighter OR all darker than the centre by threshold t.
function fastCorners(gray, w, h, threshRatio = 0.05) {
  const circle = [
    [-3,0],[-3,1],[-2,2],[-1,3],[0,3],[1,3],[2,2],[3,1],
    [3,0],[3,-1],[2,-2],[1,-3],[0,-3],[-1,-3],[-2,-2],[-3,-1]
  ];
  const N      = 9;
  const t      = 255 * threshRatio * 2;
  const scores = new Float32Array(w * h);
  const margin = 4;

  for (let y = margin; y < h - margin; y++) {
    for (let x = margin; x < w - margin; x++) {
      const Ip    = gray[y*w+x];
      const upper = Ip + t;
      const lower = Ip - t;

      // Quick reject: 4 compass points
      const quick = [
        gray[(y+circle[0][1])*w+(x+circle[0][0])],
        gray[(y+circle[4][1])*w+(x+circle[4][0])],
        gray[(y+circle[8][1])*w+(x+circle[8][0])],
        gray[(y+circle[12][1])*w+(x+circle[12][0])]
      ];
      if (quick.filter(v => v > upper).length < 3 &&
          quick.filter(v => v < lower).length < 3) continue;

      const bright = circle.map(([dx,dy]) => gray[(y+dy)*w+(x+dx)] > upper ? 1 : 0);
      const dark   = circle.map(([dx,dy]) => gray[(y+dy)*w+(x+dx)] < lower ? 1 : 0);

      const isCorner = arr => {
        const doubled = [...arr, ...arr];
        let count = 0;
        for (let i = 0; i < doubled.length; i++) {
          count = doubled[i] ? count + 1 : 0;
          if (count >= N) return true;
        }
        return false;
      };

      if (isCorner(bright) || isCorner(dark)) {
        let score = 0;
        circle.forEach(([dx,dy]) => { score += Math.abs(gray[(y+dy)*w+(x+dx)] - Ip); });
        scores[y*w+x] = score;
      }
    }
  }

  let maxScore = 0;
  scores.forEach(v => { if (v > maxScore) maxScore = v; });
  return nonMaxSuppression(scores, w, h, maxScore * 0.05, 8);
}

// ─── 4. ORB Detector (JS implementation) ────────────────────
// ORB = FAST keypoints + patch orientation (intensity centroid)
// Step 1 : detect corners with FAST
// Step 2 : for each corner compute the patch orientation using
//          the intensity centroid method (same as original ORB paper)
// Step 3 : return keypoints with { x, y, score, angle }
function orbDetect(gray, w, h, threshRatio = 0.05) {
  // --- Step 1: FAST keypoints ---
  log('ORB step 1/2 — FAST keypoint detection...', 'info');
  const keypoints = fastCorners(gray, w, h, threshRatio);

  // --- Step 2: Orientation via intensity centroid ---
  // m10 = Σ x·I(x,y),  m01 = Σ y·I(x,y),  m00 = Σ I(x,y)
  // θ = atan2(m01, m10)
  log('ORB step 2/2 — Computing patch orientation...', 'info');
  const patchR = 8; // patch radius for moment calculation

  keypoints.forEach(kp => {
    let m00 = 0, m10 = 0, m01 = 0;
    for (let dy = -patchR; dy <= patchR; dy++) {
      for (let dx = -patchR; dx <= patchR; dx++) {
        // Stay inside circle of radius patchR
        if (dx*dx + dy*dy > patchR*patchR) continue;
        const nx = Math.min(Math.max(kp.x + dx, 0), w-1);
        const ny = Math.min(Math.max(kp.y + dy, 0), h-1);
        const val = gray[ny*w + nx];
        m00 += val;
        m10 += dx * val;
        m01 += dy * val;
      }
    }
    // angle in degrees (0–360)
    kp.angle = m00 !== 0 ? (Math.atan2(m01, m10) * 180 / Math.PI + 360) % 360 : 0;
  });

  return keypoints;
}

// ─── Non-Maximum Suppression ─────────────────────────────────
function nonMaxSuppression(R, w, h, threshold, radius = 8) {
  const keypoints = [];
  for (let y = radius; y < h - radius; y++) {
    for (let x = radius; x < w - radius; x++) {
      const v = R[y*w+x];
      if (v < threshold) continue;
      let isMax = true;
      outer: for (let dy = -radius; dy <= radius; dy++) {
        for (let dx = -radius; dx <= radius; dx++) {
          if (dx === 0 && dy === 0) continue;
          if (R[(y+dy)*w+(x+dx)] > v) { isMax = false; break outer; }
        }
      }
      if (isMax) keypoints.push({ x, y, score: v, angle: 0 });
    }
  }
  keypoints.sort((a,b) => b.score - a.score);
  return keypoints.slice(0, 800);
}

// ─── Render Features Canvas ──────────────────────────────────
// For ORB: draws a circle + orientation line showing the angle.
// For FAST: draws circle + crosshair.
function renderFeatures(img, keypoints, w, h, method) {
  const c   = document.getElementById('canvas2');
  const ctx = c.getContext('2d');
  c.width = w; c.height = h;

  ctx.drawImage(img, 0, 0, w, h);
  ctx.fillStyle = 'rgba(0,0,0,0.35)';
  ctx.fillRect(0, 0, w, h);

  const isOrb = method === 'orb';

  keypoints.forEach(({ x, y, score, angle }) => {
    const r    = 4;
    const norm = score / (keypoints[0].score || 1);

    // Outer glow
    ctx.beginPath();
    ctx.arc(x, y, r + 3, 0, Math.PI*2);
    ctx.strokeStyle = `rgba(0,229,255,${norm * 0.3})`;
    ctx.lineWidth = 1;
    ctx.stroke();

    // Main circle
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI*2);
    ctx.fillStyle = isOrb
      ? `rgba(0, ${Math.round(180 + norm*75)}, 255, 0.9)`   // blue tone for ORB
      : `rgba(255, ${Math.round(61 + norm*194)}, ${Math.round(113 - norm*80)}, 0.9)`; // orange for FAST
    ctx.fill();

    if (isOrb) {
      // Orientation line — shows the patch angle computed by ORB
      const rad = angle * Math.PI / 180;
      const len = r + 5;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x + Math.cos(rad) * len, y + Math.sin(rad) * len);
      ctx.strokeStyle = 'rgba(255,255,0,0.9)';
      ctx.lineWidth = 1.2;
      ctx.stroke();
    } else {
      // Crosshair for FAST
      ctx.strokeStyle = 'rgba(255,255,255,0.8)';
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.moveTo(x-6, y); ctx.lineTo(x+6, y);
      ctx.moveTo(x, y-6); ctx.lineTo(x, y+6);
      ctx.stroke();
    }
  });

  c.style.display = 'block';
  document.getElementById('ph2').style.display = 'none';
  document.getElementById('badge2').textContent = keypoints.length + ' keypoints';
}

// ─── Save Feature Image ──────────────────────────────────────
function saveFeatureImage() {
  const c = document.getElementById('canvas2');
  if (!c || c.style.display === 'none') return;
  const link = document.createElement('a');
  link.download = 'feature_marker_' + Date.now() + '.png';
  link.href = c.toDataURL('image/png');
  link.click();
  log('Feature image saved as PNG.', 'ok');
}

// ─── Save Grayscale Image ────────────────────────────────────
function saveGrayscaleImage() {
  const c = document.getElementById('canvas1');
  if (!c || c.style.display === 'none') return;
  const link = document.createElement('a');
  link.download = 'grayscale_' + Date.now() + '.png';
  link.href = c.toDataURL('image/png');
  link.click();
  log('Grayscale image saved as PNG.', 'ok');
}