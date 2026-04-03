let detector;
let video;
/** ML always sees this size (must match landmark scale width/640, height/480) */
const ML_INPUT_W = 640;
const ML_INPUT_H = 480;
let mlInputCanvas;
let mlInputCtx;
let hands = [];
let isDetecting = false; // Flag to prevent overlapping detections
let lastDetectedHands = []; // Store last frame with actual detection
let skippedFramesCount = 0; // Count consecutive skipped frames
let handsBuffer;
let lerpedPositions = {}; // Store lerped positions for each hand and finger
let lerpedBasePositions = {}; // Store lerped base positions for rectangle mode
let lerpedPipPositions = {}; // Store lerped PIP positions for bezier curve mode
let positionVelocities = {}; // Store velocity vectors for each hand and finger position (for anti-jitter)
let basePositionVelocities = {}; // Store velocity vectors for base positions
let pipPositionVelocities = {}; // Store velocity vectors for PIP positions
let fingerColors = {}; // Store assigned colors for each hand's fingers
let debugMode = false; // Global debug mode (toggled with 'd' key)
let debugHandDetection = false; // Debug mode for hand detection (checkbox)
let showRawHandData = false; // Show raw hand data points
let showPalmCenter = false; // Show palm center debug dot
let showIntersections = true; // Show intersection/overlap zones with brand overlap colors
let enableDrawHands = true; // Enable drawing hands (default true to maintain current behavior)
let normalizeHands = true; // Normalize hand positions to center of canvas
let multicolor = true; // Finger fills use palette; off = #000000
let bigHands = false; // Scale full hand skeleton + shapes from hand centroid when on
const BIG_HANDS_SCALE = 1.75;

// Hand fill: per-hand selection, shared texture pool
// Each selection is 'brand', 'standardized', '#hexcolor', '__custom__', or 'texture:N'
let handFillSelection = ['brand', 'brand']; // [left hand, right hand]
let handFillCustomColor = '#FF0000';
let handFillTextures = []; // [{img, video, type, objectUrl}, ...] shared pool
let handFillSplitHands = false; // when true, each hand picks independently
let handFillScaleToHand = false; // scale texture to combined hand bounding box
let handBoundingBoxes = []; // [{x, y, w, h}, ...] computed per frame in drawHands
// Per-hand drawing buffers — used when hands need different textures
let perHandBuffers = [null, null];
// When non-null, drawHands routes each hand's shapes to its own buffer
let handDrawTargets = null; // null = all to handsBuffer, or [buf0, buf1]

function scaleCanvasPointFromPivot(cx, cy, pivotCx, pivotCy, factor) {
  if (factor === 1) return { x: cx, y: cy };
  return {
    x: pivotCx + (cx - pivotCx) * factor,
    y: pivotCy + (cy - pivotCy) * factor
  };
}

/** Average of all landmarks in canvas space (same mapping as drawHands). */
function handLandmarksCanvasCentroid(landmarks, layout, normalizeOffsetX, normalizeOffsetY) {
  const nox = normalizeOffsetX || 0;
  const noy = normalizeOffsetY || 0;
  let mx = 0;
  let my = 0;
  let n = 0;
  for (let j = 0; j < landmarks.length; j++) {
    const lm = landmarks[j];
    if (!lm || lm.length < 2) continue;
    mx += lm[0];
    my += lm[1];
    n++;
  }
  if (n === 0) {
    return { x: layout.ox + nox, y: layout.oy + noy };
  }
  mx /= n;
  my /= n;
  return {
    x: layout.ox + mx * layout.sx + nox,
    y: layout.oy + my * layout.sy + noy
  };
}

const handBufferFrames = 0; // Number of frames to delay the visual (0 = no delay, use most recent hand data)

// Save/export state
let saveCountdown = -1; // -1 means not counting down, >= 0 means counting down
let saveCountdownStartTime = 0;
let saveCountdownDelay = 3; // Default delay in seconds
let savePendingAction = null; // 'png' | 'svg' | 'record' — what to do when countdown finishes

// Video recording state
let mediaRecorder = null;
let recordedChunks = [];
let isRecording = false;
let recordDuration = 5; // seconds
let recordStartTime = 0;

// Animation keyframe state
let animKeyframes = []; // [{id, t (0-1), fingers: {handIdx: {fingerName: {pathData, color}}}}]
let animTotalDuration = 2; // seconds
let animPreview = false;
let animPreviewStart = 0;
let animEasingType = 'smooth-step'; // 'linear' | 'ease-in' | 'ease-out' | 'smooth-step' | 'ease-in-out-cubic' | 'custom'
let animCustomBezier = [0.42, 0, 0.58, 1]; // cubic-bezier control points [x1, y1, x2, y2]
const ANIM_PATH_POINTS = 48;

/** Apply the selected easing curve to a 0-1 value. */
function applyAnimEasing(t) {
  t = Math.max(0, Math.min(1, t));
  switch (animEasingType) {
    case 'linear': return t;
    case 'ease-in': return t * t;
    case 'ease-out': return t * (2 - t);
    case 'smooth-step': return t * t * (3 - 2 * t);
    case 'ease-in-out-cubic': return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
    case 'custom': return cubicBezierEase(t, animCustomBezier[0], animCustomBezier[1], animCustomBezier[2], animCustomBezier[3]);
    default: return t * t * (3 - 2 * t);
  }
}

/** Evaluate a cubic-bezier easing curve at time t. */
function cubicBezierEase(t, x1, y1, x2, y2) {
  // Newton-Raphson to find the bezier parameter for a given x (time)
  let guess = t;
  for (let i = 0; i < 8; i++) {
    const u = 1 - guess;
    const bx = 3 * u * u * guess * x1 + 3 * u * guess * guess * x2 + guess * guess * guess;
    const dbx = 3 * u * u * x1 + 6 * u * guess * (x2 - x1) + 3 * guess * guess * (1 - x2);
    if (Math.abs(dbx) < 1e-6) break;
    guess -= (bx - t) / dbx;
    guess = Math.max(0, Math.min(1, guess));
  }
  const u = 1 - guess;
  return 3 * u * u * guess * y1 + 3 * u * guess * guess * y2 + guess * guess * guess;
}

// Fake hand mode — substitutes ML detection with draggable fingertip landmarks.
// Feeds into the SAME pipeline as a real hand. No special overlay, no bypass.
let fakeHandActive = false;
let fakeHandDragging = -1;
let fakeHandMirrored = false;
const FAKE_FINGER_NAMES = ['thumb', 'index', 'middle', 'ring', 'pinky'];
// Draggable tip positions in ML input space (640x480)
let fakeFingerTips = null;

function initFakeHand() {
  // Positions in ML space (640x480). Mirrored on display.
  fakeFingerTips = [
    { x: 414, y: 270 },  // thumb
    { x: 387, y: 206 },  // index
    { x: 355, y: 195 },  // middle
    { x: 327, y: 212 },  // ring
    { x: 312, y: 246 }   // pinky
  ];
}

// Background collage — draggable/resizable image layers
let bgImages = []; // [{img, x, y, w, h, id}, ...]
let bgDragging = -1; // index of image being dragged
let bgResizing = -1; // index of image being resized
let bgDragOffX = 0, bgDragOffY = 0;
let bgResizeStartW = 0, bgResizeStartH = 0, bgResizeStartMX = 0;

function addBgImage(file) {
  const url = URL.createObjectURL(file);
  loadImage(url, (img) => {
    // Scale to fit ~40% of canvas width
    const targetW = width * 0.4;
    const scale = targetW / img.width;
    bgImages.push({
      img, x: width / 2, y: height / 2,
      w: img.width * scale, h: img.height * scale,
      id: Date.now()
    });
    URL.revokeObjectURL(url);
  });
}

function drawBgImages() {
  for (const bi of bgImages) {
    imageMode(CENTER);
    image(bi.img, bi.x, bi.y, bi.w, bi.h);
    // Draw resize handle (bottom-right corner)
    const hx = bi.x + bi.w / 2, hy = bi.y + bi.h / 2;
    noFill(); stroke(0, 0, 0, 60); strokeWeight(1.5);
    rect(bi.x - bi.w / 2, bi.y - bi.h / 2, bi.w, bi.h);
    fill(255); stroke(0, 0, 0, 80); strokeWeight(1);
    ellipse(hx, hy, 12, 12);
    // Draw delete button (top-right corner)
    const dx = bi.x + bi.w / 2 - 8, dy = bi.y - bi.h / 2 + 8;
    fill(255, 60, 60); noStroke();
    ellipse(dx, dy, 16, 16);
    fill(255); textAlign(CENTER, CENTER); textSize(10);
    text('×', dx, dy - 1);
  }
  imageMode(CORNER);
}

function bgImageHitTest(mx, my) {
  // Check from top (last added) to bottom
  for (let i = bgImages.length - 1; i >= 0; i--) {
    const bi = bgImages[i];
    // Delete button
    const dx = bi.x + bi.w / 2 - 8, dy = bi.y - bi.h / 2 + 8;
    if (dist(mx, my, dx, dy) < 10) return { index: i, action: 'delete' };
    // Resize handle (bottom-right)
    const hx = bi.x + bi.w / 2, hy = bi.y + bi.h / 2;
    if (dist(mx, my, hx, hy) < 15) return { index: i, action: 'resize' };
    // Drag (anywhere on image)
    if (mx > bi.x - bi.w / 2 && mx < bi.x + bi.w / 2 &&
        my > bi.y - bi.h / 2 && my < bi.y + bi.h / 2) {
      return { index: i, action: 'drag' };
    }
  }
  return null;
}

function toggleFakeHand() {
  fakeHandActive = !fakeHandActive;
  const btn = document.getElementById('fake-hand-btn');
  if (btn) btn.textContent = fakeHandActive ? 'Stop Fake Hand' : 'Toggle Fake Hand';
  // Clear shape/position state for a clean start in either direction.
  // closeness state is also cleared to prevent inflated smoothedRawScale
  // from fake hand contaminating real hand (or vice versa).
  for (let hi in paperShapes) {
    for (let fn in paperShapes[hi]) {
      if (paperShapes[hi][fn].shape) paperShapes[hi][fn].shape.remove();
      if (paperShapes[hi][fn].lastBezierPath) paperShapes[hi][fn].lastBezierPath.remove();
    }
  }
  paperShapes = {};
  lerpedPositions = {};
  lerpedBasePositions = {};
  lerpedPipPositions = {};
  handClosenessState = {};
  hands = [];
  handFrameBuffer = [];
  isDetecting = false;
  if (fakeHandActive) initFakeHand();
}

/** Build 21 ML-format landmarks from draggable fingertip positions.
 *  Palm joints form a compact cluster so rawScale is reasonable (~80-100).
 *  Finger joints are very close to tips → circles. */
function buildFakeLandmarks() {
  if (!fakeFingerTips) return null;
  // To get circles, fingerRelativeLength = dist(tip, MCP) / rawScale must be < ~0.5.
  // With tips spread ~150px from MCPs, we need rawScale > 300.
  // rawScale = avg(palmWidth, palmLength, wristToIndex, wristToPinky).
  // So we spread palm landmarks far apart to inflate rawScale.
  //
  // Strategy: place MCPs near the tips themselves (each MCP near its own tip).
  // This makes tip-to-MCP distances short (~10px) → fingerRelativeLength near 0.
  // rawScale is large because the MCPs mirror the tip spread.
  const landmarks = [];
  let pcx = 0, pcy = 0;
  for (const ft of fakeFingerTips) { pcx += ft.x; pcy += ft.y; }
  pcx /= 5; pcy /= 5;
  // Desired palm center = below the finger centroid
  const palmCX = pcx, palmCY = pcy + 30;
  const startIndices = [1, 5, 9, 13, 17];
  // Build finger joints — MCPs point toward palm center (not tip centroid)
  // so circles/rectangles face outward from the palm
  for (let f = 0; f < 5; f++) {
    const tx = fakeFingerTips[f].x, ty = fakeFingerTips[f].y;
    const si = startIndices[f];
    // Direction from tip toward palm center
    const dx = palmCX - tx, dy = palmCY - ty;
    const len = Math.sqrt(dx * dx + dy * dy) || 1;
    landmarks[si]     = [tx + dx / len * 10, ty + dy / len * 10, 0]; // MCP
    landmarks[si + 1] = [tx + dx / len * 6,  ty + dy / len * 6,  0]; // PIP
    landmarks[si + 2] = [tx + dx / len * 3,  ty + dy / len * 3,  0]; // DIP
    landmarks[si + 3] = [tx, ty, 0]; // TIP
  }
  // Place wrist so midpoint(wrist, middleMCP) = palmCenter
  // middleMCP = landmarks[9]
  const midMCP = landmarks[9];
  landmarks[0] = [2 * palmCX - midMCP[0], 2 * palmCY - midMCP[1], 0];
  return [{ landmarks, handInViewConfidence: 1.0 }];
}

/** Convert screen mouse position to ML input space for dragging.
 *  The display mirrors horizontally and maps ML→canvas via layout. */
function screenToML(mx, my) {
  const layout = getHandTrackingLayout();
  return {
    x: (width - mx - layout.ox) / layout.sx,
    y: (my - layout.oy) / layout.sy
  };
}

/** Convert ML position to screen position (inverse of pipeline transform). */
function mlToScreen(mlx, mly) {
  const layout = getHandTrackingLayout();
  return {
    x: width - (layout.ox + mlx * layout.sx),
    y: layout.oy + mly * layout.sy
  };
}

function fakeHandHitTest(mx, my) {
  if (!fakeFingerTips) return -1;
  // Use actual rendered positions from lerpedPositions (post-normalization, mirrored)
  // These are in buffer space — display mirrors with translate(width,0) + scale(-1,1)
  // So screen x = width - bufferX
  for (let i = 0; i < FAKE_FINGER_NAMES.length; i++) {
    const fn = FAKE_FINGER_NAMES[i];
    const lp = lerpedPositions[0] && lerpedPositions[0][fn];
    if (!lp) {
      // Fallback to ML→screen conversion if not yet in pipeline
      const screen = mlToScreen(fakeFingerTips[i].x, fakeFingerTips[i].y);
      const dx = mx - screen.x, dy = my - screen.y;
      if (dx * dx + dy * dy < 2500) return i;
      continue;
    }
    // lp is in buffer space, display mirrors it
    const sx = width - lp.x, sy = lp.y;
    const dx = mx - sx, dy = my - sy;
    if (dx * dx + dy * dy < 2500) return i;
  }
  return -1;
}

// Widget pose library
let widgetPoses = {}; // { poseName: { fingers: {handIdx: {fingerName: params}}, refSize: number } }
let widgetPreviewInstance = null; // live EraHand instance for preview
let widgetSize = 150;

// Paper.js shape storage - stores Paper.js shapes per hand and finger
let paperShapes = {}; // Structure: paperShapes[handIndex][fingerName] = { shape: paperObject, type: 'rect'|'circle'|'bezier', color: colorString }

// FPS history for line graph
let fpsHistory = [];
const fpsHistoryMaxLength = 60; // Store last 60 frames (~1 second at 60fps)


// Optional user background (image or video), drawn with object-fit: cover
let backgroundMedia = null;
let backgroundMediaType = null; // 'image' | 'video'
let backgroundMediaObjectUrl = null;
let backgroundMediaLoadToken = 0;
let nomediaImg;
/** @type {'plain'|'webcam'|'greenscreen'|'uploaded'} */
let backgroundMode = 'webcam';


// ============================================================================
// Slider Configuration - Set defaults and ranges here
// ============================================================================
const SLIDER_CONFIG = {
  strokeWeight: {
    min: 20,
    max: 60,
    default: 38,
    label: 'Stroke Weight'
  },
  easing: {
    min: 3,
    max: 100,
    default: 25,
    label: 'Easing',
    suffix: '%'
  },
  skipFrames: {
    min: 0,
    max: 90,
    default: 0,
    label: 'Skip Handpose Analysis',
    suffix: '%'
  }
};

// Dead zone threshold for anti-jitter (pixels)
// Movements smaller than this are considered noise and will be ignored when hand is still
const DEAD_ZONE_THRESHOLD = 7; // pixels
const MIN_VELOCITY_THRESHOLD = 15.5; // pixels per frame - below this, hand is considered still
const VELOCITY_SMOOTHING_ALPHA = 0.1; // EMA smoothing factor for velocity (0-1, lower = smoother)
/** EMA for finger width / stroke scale (getRawHandScale stays instantaneous for debug) */
const RAW_SCALE_DISPLAY_SMOOTHING_ALPHA = 0.22;

/** CottonControlPanel — DOM controls from cotton-controlpanel */
let cp;

// ============================================================================
// Hand Frame Buffer System
// ============================================================================
let handFrameBuffer = []; // Buffer to store hand data for each frame
let frameSkipCounter = 0; // Counter to track frame skipping
let frameBufferDebugInfo = {
  bufferSize: 0,
  targetIndex: -1,
  targetFrameHasData: false,
  interpolationUsed: false,
  gapBefore: -1,
  gapAfter: -1,
  beforeIndex: -1,
  afterIndex: -1,
  interpolationFactor: 0,
  handsReturned: 0
};

// v3c - overall more saturated
const COLOR_PALETTE = [
  "#5AF2AF", // green
  "#2FB3FF", // blue
  "#BFFF10", // lime
  "#E084F4", // purple
  "#F5F500", // yellow
  "#F945A6", // pink
	"#FF4D01"  // red/orange
];

// Standardized brand colors: thumb→pinky = blue, green, red, purple, yellow
const STANDARDIZED_FINGER_COLORS = {
  thumb:  "#2FB3FF", // blue
  index:  "#BFFF10", // lime
  middle: "#FF4D01", // red/orange
  ring:   "#E084F4", // purple
  pinky:  "#F5F500"  // yellow
};

// Ensure every hand has exactly one finger colored hot pink or bright orange/red
const ENSURE_ONE_ACCENT_FINGER = true;

// Accent finger color definitions
const HOT_PINK = "#F945A6";
const BRIGHT_ORANGE_RED = "#FF4D01";

// Color mapping table for intersections
// Maps pairs of colors (sorted alphabetically for consistency) to the resulting intersection color
// Format: 'color1,color2' -> 'resultColor'
// Colors are sorted alphabetically so order doesn't matter (e.g., '#006258,#1216F6' is the same as '#1216F6,#006258')
// Overlap color lookup from brand guidelines (Overlap Color Palette.pdf)
// Keys are sorted lowercase hex pairs. Order: green, blue, lime, red, purple, yellow, pink
// Grid: rows = columns = [#5AF2AF, #2FB3FF, #BFFF10, #FF4D01, #E084F4, #F5F500, #F945A6]
const INTERSECTION_COLOR_MAP = {
  // green (#5AF2AF) overlaps
  '#2fb3ff,#5af2af': '#3B99B8', // green + blue
  '#5af2af,#bfff10': '#28D287', // green + lime
  '#5af2af,#ff4d01': '#B8633F', // green + red
  '#5af2af,#e084f4': '#5A5FF2', // green + purple
  '#5af2af,#f5f500': '#4CDC79', // green + yellow
  '#5af2af,#f945a6': '#E3047C', // green + pink
  // blue (#2FB3FF) overlaps
  '#2fb3ff,#bfff10': '#23B310', // blue + lime
  '#2fb3ff,#ff4d01': '#D32A00', // blue + red
  '#2fb3ff,#e084f4': '#5A5FF2', // blue + purple
  '#2fb3ff,#f5f500': '#BCE100', // blue + yellow
  '#2fb3ff,#f945a6': '#3B99B8', // blue + pink
  // lime (#BFFF10) overlaps
  '#bfff10,#ff4d01': '#D78D47', // lime + red
  '#bfff10,#e084f4': '#6C7DEE', // lime + purple
  '#bfff10,#f5f500': '#DDE415', // lime + yellow
  '#bfff10,#f945a6': '#FD7CDD', // lime + pink
  // red (#FF4D01) overlaps
  '#e084f4,#ff4d01': '#C5010E', // red + purple
  '#f5f500,#ff4d01': '#FF8001', // red + yellow
  '#f945a6,#ff4d01': '#C5010E', // red + pink
  // purple (#E084F4) overlaps
  '#e084f4,#f5f500': '#ED67C9', // purple + yellow
  '#e084f4,#f945a6': '#C739A8', // purple + pink
  // yellow (#F5F500) overlaps
  '#f5f500,#f945a6': '#F9CF45', // yellow + pink
  // Self-overlaps (same color on same color)
  '#5af2af,#5af2af': '#00D580', // green + green
  '#2fb3ff,#2fb3ff': '#2D8AD7', // blue + blue
  '#bfff10,#bfff10': '#B3EE10', // lime + lime
  '#ff4d01,#ff4d01': '#D32A00', // red + red
  '#e084f4,#e084f4': '#C75ADF', // purple + purple
  '#f5f500,#f5f500': '#E1E100', // yellow + yellow
  '#f945a6,#f945a6': '#E3047C', // pink + pink
};

// Precomputed multiply blend results → brand overlap color replacement table.
// Built once at startup from COLOR_PALETTE + INTERSECTION_COLOR_MAP.
// Maps quantized multiply-result RGB → {r, g, b} brand overlap color.
let MULTIPLY_TO_BRAND = null; // built lazily
const OVERLAP_TOLERANCE = 8; // per-channel tolerance for matching multiply results

function hexToRgb(hex) {
  const n = parseInt(hex.replace('#', ''), 16);
  return { r: (n >> 16) & 255, g: (n >> 8) & 255, b: n & 255 };
}

function buildMultiplyLookup() {
  MULTIPLY_TO_BRAND = [];
  // Also store primary colors so we can skip single-color pixels fast
  const primaries = new Set();
  for (const c of COLOR_PALETTE) {
    const rgb = hexToRgb(c);
    primaries.add((rgb.r << 16) | (rgb.g << 8) | rgb.b);
  }
  // For each pair in the intersection map, compute what MULTIPLY would produce
  for (const key in INTERSECTION_COLOR_MAP) {
    const parts = key.split(',');
    if (parts.length !== 2) continue;
    const c1 = hexToRgb(parts[0]), c2 = hexToRgb(parts[1]);
    // Canvas multiply: result = src * dst / 255 (per channel)
    const mr = Math.round(c1.r * c2.r / 255);
    const mg = Math.round(c1.g * c2.g / 255);
    const mb = Math.round(c1.b * c2.b / 255);
    const brand = hexToRgb(INTERSECTION_COLOR_MAP[key]);
    // Skip if the multiply result is the same as a primary (single-finger pixel would be misidentified)
    const mKey = (mr << 16) | (mg << 8) | mb;
    if (primaries.has(mKey)) continue;
    MULTIPLY_TO_BRAND.push({ mr, mg, mb, br: brand.r, bg: brand.g, bb: brand.b });
  }
}

/** Replace multiply-blended overlap pixels with brand overlap colors on a p5.Graphics buffer. */
function fixOverlapColors(buf) {
  if (!MULTIPLY_TO_BRAND) buildMultiplyLookup();
  if (MULTIPLY_TO_BRAND.length === 0) return;
  // Only scan within the combined hand bounding box (not full canvas)
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const bb of handBoundingBoxes) {
    if (!bb) continue;
    minX = Math.min(minX, bb.x); minY = Math.min(minY, bb.y);
    maxX = Math.max(maxX, bb.x + bb.w); maxY = Math.max(maxY, bb.y + bb.h);
  }
  if (minX >= Infinity) return;
  const ctx = buf.drawingContext;
  const pd = pixelDensity();
  // Clamp to canvas and scale for pixel density
  const rx = Math.max(0, Math.floor(minX * pd));
  const ry = Math.max(0, Math.floor(minY * pd));
  const rw = Math.min(Math.ceil((maxX - minX) * pd), ctx.canvas.width - rx);
  const rh = Math.min(Math.ceil((maxY - minY) * pd), ctx.canvas.height - ry);
  if (rw <= 0 || rh <= 0) return;
  const imgData = ctx.getImageData(rx, ry, rw, rh);
  const d = imgData.data;
  const tol = OVERLAP_TOLERANCE;
  const lookups = MULTIPLY_TO_BRAND;
  const n = lookups.length;
  let changed = false;
  for (let px = 0; px < d.length; px += 4) {
    if (d[px + 3] < 128) continue;
    const r = d[px], g = d[px + 1], b = d[px + 2];
    for (let li = 0; li < n; li++) {
      const l = lookups[li];
      if (Math.abs(r - l.mr) <= tol && Math.abs(g - l.mg) <= tol && Math.abs(b - l.mb) <= tol) {
        d[px] = l.br; d[px + 1] = l.bg; d[px + 2] = l.bb;
        changed = true;
        break;
      }
    }
  }
  if (changed) ctx.putImageData(imgData, rx, ry);
}

// Helper function to get intersection color from two source colors
function getIntersectionColor(color1, color2) {
  // Normalize colors (ensure they're in the same format)
  const c1 = color1.toLowerCase();
  const c2 = color2.toLowerCase();
  
  // Create a consistent key (sort colors alphabetically so order doesn't matter)
  const key = [c1, c2].sort().join(',');
  
  // Check if we have a mapping
  if (INTERSECTION_COLOR_MAP[key]) {
    return INTERSECTION_COLOR_MAP[key];
  }
  
  // Default to black if no mapping is specified
  return '#000000';
}

// Finger tip landmark indices
const FINGER_TIPS = {
  thumb: 4,
  index: 8,
  middle: 12,
  ring: 16,
  pinky: 20
};

// Finger base landmark indices
const FINGER_BASES = {
  thumb: 2,  // IP joint for thumb (was PIP, now base)
  index: 5,
  middle: 9,
  ring: 13,
  pinky: 17
};

// Lower knuckle (PIP) landmark indices
const FINGER_PIPS = {
  thumb: 3,  // DIP joint for thumb (next knuckle after IP)
  index: 6,
  middle: 10,
  ring: 14,
  pinky: 18
};

// Finger length multipliers for relative length comparison
// Accounts for different finger types having different typical lengths
const FINGER_LENGTH_MULTIPLIERS = {
  thumb: 0.75,   // Thumb is typically shorter
  index: 1.0,   // Index finger as baseline
  middle: 1.2,  // Middle finger is typically longest
  ring: 1.0,    // Ring finger similar to index
  pinky: 0.8    // Pinky is typically shorter
};

// ============================================================================
// Hand Closeness System
// ============================================================================

// Hand landmark indices for closeness calculation
// 0: wrist, 5: index MCP, 9: middle MCP, 17: pinky MCP
const HAND_CLOSENESS = {
  WRIST: 0,
  INDEX_MCP: 5,
  MIDDLE_MCP: 9,
  PINKY_MCP: 17
};

// Palm landmark indices (all nodes on the palm)
// 0: wrist, 2: thumb MCP, 5: index MCP, 9: middle MCP, 13: ring MCP, 17: pinky MCP
const PALM_LANDMARKS = [0, 2, 5, 9, 13, 17];

// Hand closeness state - per hand index
let handClosenessState = {}; // Structure: handClosenessState[handIndex] = { ... }

// Calibration state - per hand index
let calibrationState = {}; // Structure: calibrationState[handIndex] = { ... }

// Helper function to get or create hand closeness state for a hand
function getHandClosenessState(handIndex) {
  if (!handClosenessState[handIndex]) {
    handClosenessState[handIndex] = {
      rawScaleBuffer: [], // Rolling buffer of rawScale values (last 120 frames)
      bufferSize: 120,
      lastRawScale: 0,
      smoothedRawScale: 0, // EMA-smoothed scale for drawing finger width
      smoothedCloseness: 0, // EMA-smoothed closeness value [0,1]
      emaAlpha: 0.15 // EMA smoothing factor (lower = smoother)
    };
  }
  const s = handClosenessState[handIndex];
  if (typeof s.smoothedRawScale !== 'number') s.smoothedRawScale = 0;
  return s;
}

// Helper function to get or create calibration state for a hand
function getCalibrationState(handIndex) {
  if (!calibrationState[handIndex]) {
    calibrationState[handIndex] = {
      isCalibrated: false,
      isCalibrating: false,
      cleanDistances: {}, // Store clean baseline distances between palm nodes
      cleanRawScale: 0, // Store clean baseline raw scale value
      stabilityBuffer: [], // Buffer to check if hand is still
      stabilityBufferSize: 30, // Number of frames to check for stability
      stabilityThreshold: 0.08 // Maximum allowed change in distances for stability (8%)
    };
  }
  return calibrationState[handIndex];
}

// ============================================================================
// Hand Frame Buffer System Functions
// ============================================================================

/**
 * Interpolate between two hand landmark arrays
 * @param {Array} landmarks1 - First set of landmarks (21 points, each [x, y, z])
 * @param {Array} landmarks2 - Second set of landmarks
 * @param {number} t - Interpolation factor (0 = landmarks1, 1 = landmarks2)
 * @returns {Array} Interpolated landmarks
 */
function interpolateLandmarks(landmarks1, landmarks2, t) {
  if (!landmarks1 || !landmarks2 || landmarks1.length !== landmarks2.length) {
    return landmarks1 || landmarks2 || [];
  }
  
  const interpolated = [];
  for (let i = 0; i < landmarks1.length; i++) {
    const p1 = landmarks1[i];
    const p2 = landmarks2[i];
    interpolated.push([
      p1[0] + (p2[0] - p1[0]) * t,
      p1[1] + (p2[1] - p1[1]) * t,
      p1[2] + (p2[2] - p1[2]) * t
    ]);
  }
  return interpolated;
}

/**
 * Interpolate between two hand objects
 * @param {Object} hand1 - First hand object with landmarks and handInViewConfidence
 * @param {Object} hand2 - Second hand object
 * @param {number} t - Interpolation factor (0 = hand1, 1 = hand2)
 * @returns {Object} Interpolated hand object
 */
function interpolateHand(hand1, hand2, t) {
  if (!hand1 || !hand2) {
    return hand1 || hand2 || null;
  }
  
  const interpolatedLandmarks = interpolateLandmarks(hand1.landmarks, hand2.landmarks, t);
  const interpolatedConfidence = hand1.handInViewConfidence + 
                                 (hand2.handInViewConfidence - hand1.handInViewConfidence) * t;
  
  return {
    landmarks: interpolatedLandmarks,
    handInViewConfidence: interpolatedConfidence
  };
}

/**
 * Interpolate between two arrays of hands
 * @param {Array} hands1 - First array of hand objects
 * @param {Array} hands2 - Second array of hand objects
 * @param {number} t - Interpolation factor (0 = hands1, 1 = hands2)
 * @returns {Array} Interpolated array of hand objects
 */
function interpolateHands(hands1, hands2, t) {
  // For simplicity, interpolate the first hand from each array
  // (could be extended to handle multiple hands and matching)
  if (hands1.length === 0 && hands2.length === 0) {
    return [];
  }
  if (hands1.length === 0) {
    return hands2;
  }
  if (hands2.length === 0) {
    return hands1;
  }
  
  // Interpolate first hand from each array
  const interpolatedHand = interpolateHand(hands1[0], hands2[0], t);
  return interpolatedHand ? [interpolatedHand] : [];
}

/**
 * Get hand data from the buffer, delayed by handBufferFrames
 * If the target frame has no hand data, interpolate between surrounding frames if gap is small enough
 * @returns {Array} Array of hand objects with landmarks, or empty array if no valid data
 */
function getBufferedHands() {
  const bufferSize = handFrameBuffer.length;
  
  // Reset debug info
  frameBufferDebugInfo = {
    bufferSize: bufferSize,
    targetIndex: -1,
    targetFrameHasData: false,
    interpolationUsed: false,
    gapBefore: -1,
    gapAfter: -1,
    beforeIndex: -1,
    afterIndex: -1,
    interpolationFactor: 0,
    handsReturned: 0
  };
  
  // Need at least handBufferFrames frames in buffer (or 1 frame if handBufferFrames is 0)
  const minBufferSize = handBufferFrames > 0 ? handBufferFrames : 1;
  if (bufferSize < minBufferSize) {
    return [];
  }
  
  // Get the frame that's handBufferFrames frames ago
  // If handBufferFrames is 0, use the most recent frame (last index)
  const targetIndex = handBufferFrames > 0 ? bufferSize - handBufferFrames : bufferSize - 1;
  const targetFrame = handFrameBuffer[targetIndex];
  
  frameBufferDebugInfo.targetIndex = targetIndex;
  
  // If target frame has hand data, return it
  if (targetFrame && targetFrame.hands && targetFrame.hands.length > 0) {
    frameBufferDebugInfo.targetFrameHasData = true;
    frameBufferDebugInfo.handsReturned = targetFrame.hands.length;
    return targetFrame.hands;
  }
  
  frameBufferDebugInfo.targetFrameHasData = false;
  
  // Target frame has no hands - look for nearest frames with hands
  let beforeIndex = -1;
  let afterIndex = -1;
  
  // Look backwards for nearest frame with hands
  for (let i = targetIndex - 1; i >= 0; i--) {
    if (handFrameBuffer[i] && handFrameBuffer[i].hands && handFrameBuffer[i].hands.length > 0) {
      beforeIndex = i;
      break;
    }
  }
  
  // Look forwards for nearest frame with hands
  for (let i = targetIndex + 1; i < bufferSize; i++) {
    if (handFrameBuffer[i] && handFrameBuffer[i].hands && handFrameBuffer[i].hands.length > 0) {
      afterIndex = i;
      break;
    }
  }
  
  // Calculate gaps
  const gapBefore = beforeIndex >= 0 ? targetIndex - beforeIndex : Infinity;
  const gapAfter = afterIndex >= 0 ? afterIndex - targetIndex : Infinity;
  const maxGap = Math.max(gapBefore, gapAfter);
  
  frameBufferDebugInfo.beforeIndex = beforeIndex;
  frameBufferDebugInfo.afterIndex = afterIndex;
  frameBufferDebugInfo.gapBefore = gapBefore !== Infinity ? gapBefore : -1;
  frameBufferDebugInfo.gapAfter = gapAfter !== Infinity ? gapAfter : -1;
  
  // Only interpolate if gap is within handBufferFrames and we have data on both sides
  if (beforeIndex >= 0 && afterIndex >= 0 && maxGap <= handBufferFrames) {
    const beforeFrame = handFrameBuffer[beforeIndex];
    const afterFrame = handFrameBuffer[afterIndex];
    const beforeHand = beforeFrame.hands[0];
    const afterHand = afterFrame.hands[0];
    
    if (beforeHand && afterHand) {
      const totalGap = afterIndex - beforeIndex;
      const t = (targetIndex - beforeIndex) / totalGap;
      
      frameBufferDebugInfo.interpolationUsed = true;
      frameBufferDebugInfo.interpolationFactor = t;
      
      const interpolatedLandmarks = interpolateLandmarks(
        beforeHand.landmarks,
        afterHand.landmarks,
        t
      );
      
      frameBufferDebugInfo.handsReturned = 1;
      return [{
        landmarks: interpolatedLandmarks,
        handInViewConfidence: beforeHand.handInViewConfidence + 
                             (afterHand.handInViewConfidence - beforeHand.handInViewConfidence) * t
      }];
    }
  }
  
  // Can't interpolate - return empty (target frame has no hands)
  frameBufferDebugInfo.handsReturned = 0;
  return [];
}

/**
 * Add current hand data to the frame buffer
 * Store reference (not deep copy) for performance
 */
function addToHandBuffer(handsData) {
  handFrameBuffer.push({
    hands: handsData || [] // Store reference, not deep copy
  });
  
  // Keep buffer size to exactly handBufferFrames (or 1 if handBufferFrames is 0)
  const maxBufferSize = handBufferFrames > 0 ? handBufferFrames : 1;
  if (handFrameBuffer.length > maxBufferSize) {
    handFrameBuffer.shift();
  }
}

/**
 * Replace the last N frames in the buffer with new data
 * Used to replace skipped frames with interpolated values
 */
function replaceLastFramesInBuffer(count, newFramesData) {
  // Remove last N frames
  for (let i = 0; i < count && handFrameBuffer.length > 0; i++) {
    handFrameBuffer.pop();
  }
  
  // Add new frames (in reverse order since we're adding to end)
  for (let i = 0; i < newFramesData.length; i++) {
    handFrameBuffer.push({
      hands: newFramesData[i] || []
    });
  }
  
  // Ensure buffer doesn't exceed max size (or 1 if handBufferFrames is 0)
  const maxBufferSize = handBufferFrames > 0 ? handBufferFrames : 1;
  while (handFrameBuffer.length > maxBufferSize) {
    handFrameBuffer.shift();
  }
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Compute 2D Euclidean distance between two points [x, y]
 */
function dist2D(p1, p2) {
  const dx = p1[0] - p2[0];
  const dy = p1[1] - p2[1];
  return Math.sqrt(dx * dx + dy * dy);
}

/**
 * Calculate the angle at point p2 between points p1, p2, p3
 * Returns angle in degrees (0-180)
 * @param {Array} p1 - First point [x, y]
 * @param {Array} p2 - Middle point (vertex) [x, y]
 * @param {Array} p3 - Third point [x, y]
 * @returns {number} Angle in degrees
 */
function calculateAngle(p1, p2, p3) {
  // Vector from p2 to p1
  const v1x = p1[0] - p2[0];
  const v1y = p1[1] - p2[1];
  
  // Vector from p2 to p3
  const v2x = p3[0] - p2[0];
  const v2y = p3[1] - p2[1];
  
  // Calculate dot product
  const dot = v1x * v2x + v1y * v2y;
  
  // Calculate magnitudes
  const mag1 = Math.sqrt(v1x * v1x + v1y * v1y);
  const mag2 = Math.sqrt(v2x * v2x + v2y * v2y);
  
  // Avoid division by zero
  if (mag1 === 0 || mag2 === 0) {
    return 0;
  }
  
  // Calculate angle in radians, then convert to degrees
  const cosAngle = dot / (mag1 * mag2);
  // Clamp to [-1, 1] to avoid numerical errors
  const clampedCos = Math.max(-1, Math.min(1, cosAngle));
  const angleRad = Math.acos(clampedCos);
  const angleDeg = degrees(angleRad);
  
  return angleDeg;
}

/**
 * Clamp value to [0, 1] range
 */
function clamp01(value) {
  return Math.max(0, Math.min(1, value));
}

/**
 * Compute percentile of a sorted array
 * @param {Array} arr - Sorted array of numbers
 * @param {number} p - Percentile (0-100)
 * @returns {number} Percentile value
 */
function percentile(arr, p) {
  if (arr.length === 0) return 0;
  if (arr.length === 1) return arr[0];
  
  const index = (p / 100) * (arr.length - 1);
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  const weight = index - lower;
  
  return arr[lower] * (1 - weight) + arr[upper] * weight;
}

/**
 * EMA (Exponential Moving Average) helper
 * @param {number} prev - Previous EMA value
 * @param {number} next - New value to incorporate
 * @param {number} alpha - Smoothing factor (0-1, lower = smoother)
 * @returns {number} Updated EMA value
 */
function emaNext(prev, next, alpha) {
  return prev + alpha * (next - prev);
}

/**
 * Update velocity and determine if position update should be skipped (anti-jitter)
 * @param {Object} currentPos - Current lerped position { x, y }
 * @param {Object} targetPos - Target position { x, y }
 * @param {Object} velocity - Current velocity vector { x, y } (will be updated)
 * @param {number} lerpAmount - Base lerp amount (0-1)
 * @returns {Object} { shouldSkip: boolean, adaptiveLerpAmount: number, updatedVelocity: { x, y } }
 */
function updateVelocityAndCheckDeadZone(currentPos, targetPos, velocity, lerpAmount) {
  // Calculate distance to target
  const dx = targetPos.x - currentPos.x;
  const dy = targetPos.y - currentPos.y;
  const distance = Math.sqrt(dx * dx + dy * dy);
  
  // Calculate instantaneous velocity (change in position)
  const instantVelX = dx;
  const instantVelY = dy;
  const instantVelMagnitude = distance;
  
  // Update velocity with EMA smoothing
  const updatedVelocity = {
    x: emaNext(velocity.x || 0, instantVelX, VELOCITY_SMOOTHING_ALPHA),
    y: emaNext(velocity.y || 0, instantVelY, VELOCITY_SMOOTHING_ALPHA)
  };
  const velocityMagnitude = Math.sqrt(updatedVelocity.x * updatedVelocity.x + updatedVelocity.y * updatedVelocity.y);
  
  // Check if we should skip update (dead zone + low velocity)
  const shouldSkip = distance < DEAD_ZONE_THRESHOLD && velocityMagnitude < MIN_VELOCITY_THRESHOLD;
  
  // Calculate adaptive lerp amount based on velocity
  // Low velocity = reduce lerp amount (smoother, less responsive)
  // High velocity = use full lerp amount (more responsive)
  let adaptiveLerpAmount = lerpAmount;
  if (!shouldSkip && velocityMagnitude < MIN_VELOCITY_THRESHOLD * 2) {
    // Reduce lerp amount when velocity is low (but not zero)
    const velocityFactor = Math.max(0.3, velocityMagnitude / (MIN_VELOCITY_THRESHOLD * 2));
    adaptiveLerpAmount = lerpAmount * velocityFactor;
  }
  
  return {
    shouldSkip: shouldSkip,
    adaptiveLerpAmount: adaptiveLerpAmount,
    updatedVelocity: updatedVelocity
  };
}

/**
 * Collect all distances between palm landmarks
 * @param {Array} landmarks - Array of 21 landmarks, each [x, y, z]
 * @returns {Object} Object with keys like "0-5" (wrist to index MCP) and distance values
 */
function collectPalmDistances(landmarks) {
  if (!landmarks || landmarks.length < 21) return {};
  
  const distances = {};
  
  // Calculate all pairwise distances between palm landmarks
  for (let i = 0; i < PALM_LANDMARKS.length; i++) {
    for (let j = i + 1; j < PALM_LANDMARKS.length; j++) {
      const idx1 = PALM_LANDMARKS[i];
      const idx2 = PALM_LANDMARKS[j];
      const key = `${idx1}-${idx2}`;
      const dist = dist2D(landmarks[idx1], landmarks[idx2]);
      distances[key] = dist;
    }
  }
  
  return distances;
}

/**
 * Check if hand is relatively still by comparing current distances to recent history
 * Resets buffer if hand is not stable
 * @param {Array} landmarks - Array of 21 landmarks
 * @param {number} handIndex - Index of the hand (0, 1, etc.)
 * @returns {boolean} True if hand appears stable
 */
function isHandStable(landmarks, handIndex) {
  const currentDistances = collectPalmDistances(landmarks);
  const state = getCalibrationState(handIndex);
  
  // Need at least one previous frame to compare
  if (state.stabilityBuffer.length === 0) {
    // First frame - add it to buffer
    state.stabilityBuffer.push(currentDistances);
    return false; // Not stable yet, need more frames
  }
  
  // Check if current frame is stable relative to existing buffer
  const threshold = state.stabilityThreshold;
  const keys = Object.keys(currentDistances);
  let isStable = true;
  
  // Check if current distances are stable relative to buffer
  for (const key of keys) {
    const currentValue = currentDistances[key];
    if (currentValue === 0) {
      isStable = false;
      break;
    }
    
    // Check variation between current frame and buffer
    let minVal = currentValue;
    let maxVal = currentValue;
    
    for (const bufferDistances of state.stabilityBuffer) {
      const val = bufferDistances[key];
      if (val === undefined) {
        isStable = false;
        break;
      }
      minVal = Math.min(minVal, val);
      maxVal = Math.max(maxVal, val);
    }
    
    if (!isStable) break;
    
    // Check if variation is within threshold
    const variation = (maxVal - minVal) / currentValue;
    if (variation > threshold) {
      isStable = false;
      break;
    }
  }
  
  if (isStable) {
    // Hand is stable - add current frame to buffer
    state.stabilityBuffer.push(currentDistances);
    
    // Maintain buffer size
    if (state.stabilityBuffer.length > state.stabilityBufferSize) {
      state.stabilityBuffer.shift();
    }
    
    // Check if we have enough stable frames
    return state.stabilityBuffer.length >= state.stabilityBufferSize;
  } else {
    // Hand is not stable - reset buffer
    state.stabilityBuffer = [];
    // Add current frame as starting point
    state.stabilityBuffer.push(currentDistances);
    return false;
  }
}

/**
 * Calibrate hand scale by collecting clean baseline measurements
 * @param {Array} landmarks - Array of 21 landmarks
 * @param {number} handIndex - Index of the hand (0, 1, etc.)
 * @returns {boolean} True if calibration was successful
 */
function calibrateHandScale(landmarks, handIndex) {
  if (!landmarks || landmarks.length < 21) return false;
  
  const state = getCalibrationState(handIndex);
  
  // Check if hand is stable
  if (!isHandStable(landmarks, handIndex)) {
    return false;
  }
  
  // Collect clean baseline distances
  state.cleanDistances = collectPalmDistances(landmarks);
  
  // Calculate clean raw scale using the original method
  const wrist = landmarks[HAND_CLOSENESS.WRIST];
  const indexMCP = landmarks[HAND_CLOSENESS.INDEX_MCP];
  const middleMCP = landmarks[HAND_CLOSENESS.MIDDLE_MCP];
  const pinkyMCP = landmarks[HAND_CLOSENESS.PINKY_MCP];
  
  const palmWidth = dist2D(indexMCP, pinkyMCP);
  const palmLength = dist2D(wrist, middleMCP);
  const wristToIndex = dist2D(wrist, indexMCP);
  const wristToPinky = dist2D(wrist, pinkyMCP);
  
  state.cleanRawScale = (palmWidth + palmLength + wristToIndex + wristToPinky) / 4;
  
  state.isCalibrated = true;
  state.isCalibrating = false;
  
  return true;
}

/**
 * Compute raw hand scale from 2D landmark distances
 * Uses calibration baseline to compute relative scale values
 * Note: Always uses hand 0's calibration data for all hands
 * @param {Array} landmarks - Array of 21 landmarks, each [x, y, z]
 * @param {number} handIndex - Index of the hand (0, 1, etc.) - used for per-hand state, but calibration comes from hand 0
 * @returns {number} Raw scale value (larger when hand is closer)
 */
function computeHandScale(landmarks, handIndex) {
  if (!landmarks || landmarks.length < 21) return 0;
  // Simple direct measurement — no calibration baseline needed.
  // Average of palm distances gives a stable hand size estimate.
  const wrist = landmarks[HAND_CLOSENESS.WRIST];
  const indexMCP = landmarks[HAND_CLOSENESS.INDEX_MCP];
  const middleMCP = landmarks[HAND_CLOSENESS.MIDDLE_MCP];
  const pinkyMCP = landmarks[HAND_CLOSENESS.PINKY_MCP];
  const palmWidth = dist2D(indexMCP, pinkyMCP);
  const palmLength = dist2D(wrist, middleMCP);
  const wristToIndex = dist2D(wrist, indexMCP);
  const wristToPinky = dist2D(wrist, pinkyMCP);
  return (palmWidth + palmLength + wristToIndex + wristToPinky) / 4;
}

/**
 * Update hand closeness system
 * Should be called each frame when a hand is detected
 * @param {Array} landmarks - Array of 21 landmarks
 * @param {number} handIndex - Index of the hand (0, 1, etc.)
 */
function updateHandCloseness(landmarks, handIndex) {
  const state = getHandClosenessState(handIndex);
  
  // Compute raw scale from current hand pose
  const rawScale = computeHandScale(landmarks, handIndex);
  state.lastRawScale = rawScale;

  // Smoothed scale for finger width (runs every frame; not gated on closeness buffer warmup)
  if (rawScale > 0) {
    if (!state.smoothedRawScale || state.smoothedRawScale <= 0) {
      state.smoothedRawScale = rawScale;
    } else {
      state.smoothedRawScale = emaNext(
        state.smoothedRawScale,
        rawScale,
        RAW_SCALE_DISPLAY_SMOOTHING_ALPHA
      );
    }
  }
  
  // Add to rolling buffer
  state.rawScaleBuffer.push(rawScale);
  
  // Maintain buffer size
  if (state.rawScaleBuffer.length > state.bufferSize) {
    state.rawScaleBuffer.shift(); // Remove oldest value
  }
  
  // Need at least a few samples before computing percentiles
  if (state.rawScaleBuffer.length < 10) {
    // Not enough data yet, use simple normalization
    state.smoothedCloseness = emaNext(state.smoothedCloseness, 0.5, state.emaAlpha);
    return;
  }
  
  // Compute robust min/max using percentiles (p10 and p90)
  const sorted = [...state.rawScaleBuffer].sort((a, b) => a - b);
  const p10 = percentile(sorted, 10);
  const p90 = percentile(sorted, 90);
  
  // Compute normalized closeness [0, 1]
  let closeness = 0;
  const range = p90 - p10;
  
  if (range > 0.001) { // Avoid division by zero
    closeness = (rawScale - p10) / range;
    closeness = clamp01(closeness);
  } else {
    // Degenerate case: all values are very similar
    // Use middle of range as default
    closeness = 0.5;
  }
  
  // Apply EMA smoothing
  state.smoothedCloseness = emaNext(state.smoothedCloseness, closeness, state.emaAlpha);
}

/**
 * Handle case when no hand is detected for a specific hand
 * Slowly decay smoothedCloseness toward 0
 * @param {number} handIndex - Index of the hand
 */
function updateHandClosenessNoHand(handIndex) {
  const state = getHandClosenessState(handIndex);
  // EMA toward 0 (no hand = not close)
  state.smoothedCloseness = emaNext(state.smoothedCloseness, 0, state.emaAlpha);
  state.smoothedRawScale = emaNext(
    state.smoothedRawScale,
    0,
    RAW_SCALE_DISPLAY_SMOOTHING_ALPHA
  );
  // Don't add to buffer, but keep last rawScale
}

/**
 * Get current hand closeness value [0, 1]
 * @param {number} handIndex - Index of the hand (0, 1, etc.)
 * @returns {number} Smoothed closeness value, 0 = far, 1 = close
 */
function getHandCloseness(handIndex) {
  const state = getHandClosenessState(handIndex);
  return state.smoothedCloseness;
}

/**
 * Get raw scale value (for debugging)
 * @param {number} handIndex - Index of the hand (0, 1, etc.)
 * @returns {number} Last computed raw scale
 */
function getRawHandScale(handIndex) {
  const state = getHandClosenessState(handIndex);
  return state.lastRawScale;
}

/** Smoothed hand scale for rendering (finger width, thresholds); debug uses getRawHandScale */
function getSmoothedRawHandScale(handIndex) {
  const state = getHandClosenessState(handIndex);
  const s = state.smoothedRawScale;
  if (s > 0) return s;
  return state.lastRawScale;
}

// Function to shuffle array (Fisher-Yates shuffle)
function shuffleArray(array) {
  let shuffled = [...array];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
}

// Function to assign colors to fingers, ensuring all palette colors are used before reusing
function assignFingerColors(handIndex) {
  if (!fingerColors[handIndex]) {
    let fingerNames = Object.keys(FINGER_TIPS);
    const numFingers = fingerNames.length;
    
    // Collect all colors currently used across all hands
    const usedColors = new Set();
    for (let handIdx in fingerColors) {
      const handFingerColors = fingerColors[handIdx];
      for (let fingerName in handFingerColors) {
        usedColors.add(handFingerColors[fingerName]);
      }
    }
    
    // Find unused colors from the palette
    const unusedColors = COLOR_PALETTE.filter(color => !usedColors.has(color));
    
    // Determine which colors to use for this hand
    let colorsToUse = [];
    
    if (unusedColors.length >= numFingers) {
      // We have enough unused colors, use those (shuffled)
      colorsToUse = shuffleArray(unusedColors).slice(0, numFingers);
    } else {
      // Use all unused colors first, then fill the rest from the entire palette (shuffled)
      colorsToUse = [...unusedColors];
      const remainingNeeded = numFingers - unusedColors.length;
      const shuffledPalette = shuffleArray(COLOR_PALETTE);
      // Add colors from shuffled palette that we still need
      for (let i = 0; i < shuffledPalette.length && colorsToUse.length < numFingers; i++) {
        if (!colorsToUse.includes(shuffledPalette[i])) {
          colorsToUse.push(shuffledPalette[i]);
        }
      }
    }
    
    // Assign colors to fingers for this hand
    fingerColors[handIndex] = {};
    for (let i = 0; i < fingerNames.length; i++) {
      fingerColors[handIndex][fingerNames[i]] = colorsToUse[i];
    }
    
    // Ensure exactly one finger is colored hot pink or bright orange/red
    if (ENSURE_ONE_ACCENT_FINGER) {
      const accentColors = [HOT_PINK, BRIGHT_ORANGE_RED];
      let accentFingerCount = 0;
      let accentFingerName = null;
      
      // Count how many fingers currently have accent colors
      for (let fingerName of fingerNames) {
        const color = fingerColors[handIndex][fingerName];
        if (accentColors.includes(color)) {
          accentFingerCount++;
          accentFingerName = fingerName;
        }
      }
      
      // Helper function to get colors already used in this hand (excluding a specific finger)
      const getUsedColorsInHand = (excludeFinger = null) => {
        const used = new Set();
        for (let fingerName of fingerNames) {
          if (fingerName !== excludeFinger) {
            used.add(fingerColors[handIndex][fingerName]);
          }
        }
        return used;
      };
      
      // Helper function to get an available non-accent color for a finger
      const getAvailableNonAccentColor = (forFinger) => {
        const usedInHand = getUsedColorsInHand(forFinger);
        const nonAccentColors = COLOR_PALETTE.filter(c => !accentColors.includes(c) && !usedInHand.has(c));
        
        // If no unused non-accent colors, fall back to any non-accent color
        if (nonAccentColors.length === 0) {
          return COLOR_PALETTE.find(c => !accentColors.includes(c)) || COLOR_PALETTE[0];
        }
        
        return nonAccentColors[Math.floor(Math.random() * nonAccentColors.length)];
      };
      
      // If no accent finger, randomly assign one
      if (accentFingerCount === 0) {
        const randomFinger = fingerNames[Math.floor(Math.random() * fingerNames.length)];
        const randomAccentColor = accentColors[Math.floor(Math.random() * accentColors.length)];
        // Store the original color to potentially reuse it
        const originalColor = fingerColors[handIndex][randomFinger];
        fingerColors[handIndex][randomFinger] = randomAccentColor;
        // If we replaced a color, we need to ensure no duplicates
        // The original color assignment should have handled uniqueness, so this should be fine
      }
      // If more than one accent finger, keep only one and change others
      else if (accentFingerCount > 1) {
        // Keep the first accent finger found, change all others
        let keptOne = false;
        for (let fingerName of fingerNames) {
          const color = fingerColors[handIndex][fingerName];
          if (accentColors.includes(color)) {
            if (!keptOne) {
              keptOne = true;
            } else {
              // Replace with a non-accent color that's not already used in this hand
              fingerColors[handIndex][fingerName] = getAvailableNonAccentColor(fingerName);
            }
          }
        }
      }
    }
  }
}

function calculateCanvasSize() {
  return { width: windowWidth, height: windowHeight };
}

/**
 * Webcam → canvas layout for hand tracking and drawing.
 * ML landmarks live in ML_INPUT_W×H (stretched from full video). Map: ML → intrinsic → cover (uniform s).
 */
function getHandTrackingLayout() {
  const cw = width;
  const ch = height;
  let iw = 640;
  let ih = 480;
  if (video && video.elt && video.elt.videoWidth > 0 && video.elt.videoHeight > 0) {
    iw = video.elt.videoWidth;
    ih = video.elt.videoHeight;
  }
  const coverS = Math.max(cw / iw, ch / ih);
  const dw = iw * coverS;
  const dh = ih * coverS;
  const ox = (cw - dw) * 0.5;
  const oy = (ch - dh) * 0.5;
  const sx = (coverS * iw) / ML_INPUT_W;
  const sy = (coverS * ih) / ML_INPUT_H;
  return { iw, ih, s: coverS, ox, oy, dw, dh, sx, sy };
}

function mlToCanvasXY(mlx, mly, L) {
  return { x: L.ox + mlx * L.sx, y: L.oy + mly * L.sy };
}

function disposeBackgroundMedia() {
  backgroundMediaLoadToken++;
  if (backgroundMediaType === 'video' && backgroundMedia) {
    try {
      backgroundMedia.remove();
    } catch (e) {
      /* ignore */
    }
  }
  backgroundMedia = null;
  backgroundMediaType = null;
  if (backgroundMediaObjectUrl) {
    URL.revokeObjectURL(backgroundMediaObjectUrl);
    backgroundMediaObjectUrl = null;
  }
}

function handleBackgroundMediaFile(file) {
  if (!file) return;
  disposeBackgroundMedia();
  const loadId = backgroundMediaLoadToken;
  const url = URL.createObjectURL(file);
  backgroundMediaObjectUrl = url;

  if (file.type.startsWith('video/')) {
    backgroundMedia = createVideo([url]);
    backgroundMedia.hide();
    const el = backgroundMedia.elt;
    el.muted = true;
    el.setAttribute('playsinline', '');
    el.setAttribute('webkit-playsinline', '');
    backgroundMedia.volume(0);
    // Native loop + play: p5's play() is not a Promise; p5 loop() can race with play().
    el.loop = true;
    const tryPlay = () => {
      const p = el.play();
      if (p && typeof p.catch === 'function') p.catch(() => {});
    };
    if (el.readyState >= 2) tryPlay();
    else el.addEventListener('loadeddata', tryPlay, { once: true });
    backgroundMediaType = 'video';
    backgroundMode = 'uploaded';
    if (cp) cp.set('bgMode', 'uploaded');
  } else if (file.type.startsWith('image/')) {
    loadImage(
      url,
      (img) => {
        if (loadId !== backgroundMediaLoadToken) return;
        if (backgroundMediaObjectUrl === url) {
          URL.revokeObjectURL(url);
          backgroundMediaObjectUrl = null;
        }
        backgroundMedia = img;
        backgroundMediaType = 'image';
        backgroundMode = 'uploaded';
        if (cp) cp.set('bgMode', 'uploaded');
      },
      () => {
        if (loadId !== backgroundMediaLoadToken) return;
        console.warn('Background image failed to load');
        if (backgroundMediaObjectUrl === url) {
          URL.revokeObjectURL(url);
          backgroundMediaObjectUrl = null;
        }
      }
    );
  } else {
    URL.revokeObjectURL(url);
    backgroundMediaObjectUrl = null;
  }
}

function getIntrinsicMediaSize(media) {
  if (media.elt && media.elt.videoWidth > 0) {
    return { w: media.elt.videoWidth, h: media.elt.videoHeight };
  }
  return { w: media.width || 0, h: media.height || 0 };
}

/** object-fit: cover; mirrorHorizontal matches selfie webcam overlay */
function drawCoverMedia(media, mirrorHorizontal) {
  const cw = width;
  const ch = height;
  const { w: mw, h: mh } = getIntrinsicMediaSize(media);
  if (!mw || !mh) return false;
  const s = max(cw / mw, ch / mh);
  const dw = mw * s;
  const dh = mh * s;
  const ox = (cw - dw) * 0.5;
  const oy = (ch - dh) * 0.5;
  if (mirrorHorizontal) {
    push();
    translate(cw, 0);
    scale(-1, 1);
    image(media, cw - ox - dw, oy, dw, dh);
    pop();
  } else {
    image(media, ox, oy, dw, dh);
  }
  return true;
}

function drawUploadedModeBackground() {
  background(255);
  if (!backgroundMedia || !backgroundMediaType) {
    if (nomediaImg && nomediaImg.width > 0) drawCoverMedia(nomediaImg, false);
    return;
  }
  if (backgroundMediaType === 'video') {
    const v = backgroundMedia.elt;
    if (v.videoWidth > 0 && v.videoHeight > 0) {
      drawCoverMedia(backgroundMedia, false);
    }
    /* else: decoding — keep white until dimensions exist */
  } else {
    drawCoverMedia(backgroundMedia, false);
  }
}

function drawBackgroundLayer() {
  const mode = backgroundMode;
  if (mode === 'plain') {
    background(255);
    return;
  }
  if (mode === 'greenscreen') {
    background(0, 255, 0);
    return;
  }
  if (mode === 'webcam') {
    background(255);
    if (video && video.loadedmetadata) {
      drawCoverMedia(video, true);
    }
    return;
  }
  if (mode === 'uploaded') {
    drawUploadedModeBackground();
    return;
  }
  background(255);
}

/** Load an image or video file as a texture and call back with {img, video, type, objectUrl}. */
function loadTextureFile(file, callback) {
  const url = URL.createObjectURL(file);
  if (file.type.startsWith('video/')) {
    const vid = document.createElement('video');
    vid.src = url;
    vid.loop = true;
    vid.muted = true;
    vid.playsInline = true;
    vid.play();
    callback({ img: null, video: vid, type: 'video', objectUrl: url });
  } else {
    loadImage(url, (img) => {
      callback({ img: img, video: null, type: 'image', objectUrl: url });
    });
  }
}

/** Shared swatch definitions used by both hand rows. */
const HAND_FILL_SWATCHES = [
  { value: 'brand', label: 'Brand colors (random)', multicolor: true, colors: COLOR_PALETTE },
  { value: 'standardized', label: 'Brand colors (standard)', multicolor: true, colors: Object.values(STANDARDIZED_FINGER_COLORS) },
  ...COLOR_PALETTE.map(c => ({ value: c, color: c, label: c })),
  { value: '#FFFFFF', color: '#FFFFFF', label: 'White' },
  { value: '#121212', color: '#121212', label: 'Black' }
];

/** Upload a texture file, add it to the pool, and create swatches in both rows. */
function handleTextureUpload(file, forHand) {
  loadTextureFile(file, (result) => {
    const idx = handFillTextures.length;
    handFillTextures.push(result);
    const value = 'texture:' + idx;
    const label = file.name || 'Texture ' + (idx + 1);

    function addSwatches(thumbUrl) {
      if (cp) {
        cp.addSwatch('handFill0', { value, label, thumbnail: thumbUrl });
        cp.addSwatch('handFill1', { value, label, thumbnail: thumbUrl });
      }
      handFillSelection[forHand] = value;
      if (!handFillSplitHands) handFillSelection[1 - forHand] = value;
      if (cp) {
        cp.set('handFill' + forHand, value);
        if (!handFillSplitHands) cp.set('handFill' + (1 - forHand), value);
      }
    }

    if (result.type === 'video' && result.video) {
      // Capture a frame from the video as a thumbnail
      const vid = result.video;
      function captureFrame() {
        const c = document.createElement('canvas');
        c.width = vid.videoWidth || 64;
        c.height = vid.videoHeight || 64;
        c.getContext('2d').drawImage(vid, 0, 0, c.width, c.height);
        addSwatches(c.toDataURL('image/jpeg', 0.6));
      }
      if (vid.readyState >= 2) captureFrame();
      else vid.addEventListener('loadeddata', captureFrame, { once: true });
    } else {
      // Image — blob URL works directly as CSS background-image
      addSwatches(result.objectUrl);
    }
  });
}

/** Get the combined bounding box of all hands (in handsBuffer space, before mirroring). */
function getAllHandsBounds() {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (let i = 0; i < handBoundingBoxes.length; i++) {
    const bb = handBoundingBoxes[i];
    if (!bb) continue;
    minX = Math.min(minX, bb.x); minY = Math.min(minY, bb.y);
    maxX = Math.max(maxX, bb.x + bb.w); maxY = Math.max(maxY, bb.y + bb.h);
  }
  if (minX >= Infinity) return null;
  // Clamp to canvas
  minX = Math.max(0, Math.floor(minX));
  minY = Math.max(0, Math.floor(minY));
  maxX = Math.min(width, Math.ceil(maxX));
  maxY = Math.min(height, Math.ceil(maxY));
  return { x: minX, y: minY, w: maxX - minX, h: maxY - minY };
}

/** Save PNG of just the hand shapes on transparent background, cropped to bounding box. */
function saveCroppedPNG() {
  const bb = getAllHandsBounds();
  if (!bb) { saveCroppedPNG(); return; }
  // The handsBuffer has shapes (already with textures/overlaps applied).
  // Mirror it (same as display) then crop to the mirrored bounding box.
  const mirroredBB = { x: width - bb.x - bb.w, y: bb.y, w: bb.w, h: bb.h };
  const tempCanvas = document.createElement('canvas');
  const pd = pixelDensity();
  tempCanvas.width = mirroredBB.w * pd;
  tempCanvas.height = mirroredBB.h * pd;
  const tempCtx = tempCanvas.getContext('2d');
  // Draw the mirrored handsBuffer, offset to crop
  tempCtx.scale(pd, pd);
  tempCtx.translate(-mirroredBB.x, -mirroredBB.y);
  tempCtx.translate(width, 0);
  tempCtx.scale(-1, 1);
  tempCtx.drawImage(handsBuffer.drawingContext.canvas, 0, 0, width, height);
  // Download
  const a = document.createElement('a');
  a.href = tempCanvas.toDataURL('image/png');
  a.download = 'hand-tracking.png';
  a.click();
}

/** Export SVG of just the hand shapes, cropped to bounding box.
 *  For texture/video fills, embeds a rasterized image.
 *  For solid colors, exports vector paths. */
function exportSVG() {
  const bb = getAllHandsBounds();
  const mirroredBB = bb ? { x: width - bb.x - bb.w, y: bb.y, w: bb.w, h: bb.h } : { x: 0, y: 0, w: width, h: height };
  const anyTexture = handNeedsTexture(0) || handNeedsTexture(1);

  if (anyTexture) {
    // Texture/video fills can't be represented as vector — rasterize the handsBuffer
    const pd = pixelDensity();
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = mirroredBB.w * pd;
    tempCanvas.height = mirroredBB.h * pd;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.scale(pd, pd);
    tempCtx.translate(-mirroredBB.x, -mirroredBB.y);
    tempCtx.translate(width, 0);
    tempCtx.scale(-1, 1);
    tempCtx.drawImage(handsBuffer.drawingContext.canvas, 0, 0, width, height);
    const dataUrl = tempCanvas.toDataURL('image/png');
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 ${mirroredBB.w} ${mirroredBB.h}" width="${mirroredBB.w}" height="${mirroredBB.h}">
<image width="${mirroredBB.w}" height="${mirroredBB.h}" xlink:href="${dataUrl}"/>
</svg>`;
    const blob = new Blob([svg], { type: 'image/svg+xml' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'hand-tracking.svg'; a.click();
    URL.revokeObjectURL(url);
    return;
  }

  // Vector export for solid color fills
  const group = new paper.Group();
  const allShapes = [];
  for (let handIdx in paperShapes) {
    for (let fingerName in paperShapes[handIdx]) {
      const sd = paperShapes[handIdx][fingerName];
      if (!sd || !sd.shape) continue;
      const path = shapeToPath(sd.shape, sd.strokeWidth);
      if (!path) continue;
      const clone = path.clone();
      clone.fillColor = sd.color || '#000000';
      clone.strokeColor = null;
      group.addChild(clone);
      allShapes.push({ path: clone, color: sd.color || '#000000' });
    }
  }
  // Pairwise intersections with brand overlap colors
  for (let j = 0; j < allShapes.length; j++) {
    for (let k = j + 1; k < allShapes.length; k++) {
      try {
        const intColor = getIntersectionColor(allShapes[j].color, allShapes[k].color);
        if (!intColor || intColor === '#000000') continue;
        const p1 = allShapes[j].path.clone();
        const p2 = allShapes[k].path.clone();
        const intersection = p1.intersect(p2);
        if (intersection && intersection.segments && intersection.segments.length > 0) {
          intersection.fillColor = intColor;
          intersection.strokeColor = null;
          group.addChild(intersection);
        } else if (intersection) { intersection.remove(); }
        p1.remove(); p2.remove();
      } catch (e) { /* skip */ }
    }
  }
  const svgStr = group.exportSVG({ asString: true });
  // Crop viewBox to mirrored bounding box
  const svgRoot = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svgRoot.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
  svgRoot.setAttribute('viewBox', mirroredBB.x + ' ' + mirroredBB.y + ' ' + mirroredBB.w + ' ' + mirroredBB.h);
  svgRoot.setAttribute('width', mirroredBB.w);
  svgRoot.setAttribute('height', mirroredBB.h);
  const mirrorG = document.createElementNS('http://www.w3.org/2000/svg', 'g');
  mirrorG.setAttribute('transform', 'scale(-1,1) translate(-' + width + ',0)');
  mirrorG.innerHTML = svgStr;
  svgRoot.appendChild(mirrorG);
  const svgOutput = new XMLSerializer().serializeToString(svgRoot);
  const blob = new Blob([svgOutput], { type: 'image/svg+xml' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'hand-tracking.svg'; a.click();
  URL.revokeObjectURL(url);
  group.remove();
}

/** Start recording the canvas as a WebM video. */
function startRecording() {
  const canvas = document.querySelector('canvas');
  const stream = canvas.captureStream(30); // 30 fps
  recordedChunks = [];
  const mimeTypes = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'];
  let mime = '';
  for (const m of mimeTypes) {
    if (MediaRecorder.isTypeSupported(m)) { mime = m; break; }
  }
  mediaRecorder = new MediaRecorder(stream, mime ? { mimeType: mime } : {});
  mediaRecorder.ondataavailable = (e) => {
    if (e.data.size > 0) recordedChunks.push(e.data);
  };
  mediaRecorder.onstop = () => {
    const blob = new Blob(recordedChunks, { type: mediaRecorder.mimeType || 'video/webm' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'hand-tracking.webm';
    a.click();
    URL.revokeObjectURL(url);
    isRecording = false;
    recordedChunks = [];
    const btn = document.getElementById('record-button');
    if (btn) btn.textContent = 'Record Video';
  };
  mediaRecorder.start();
  isRecording = true;
  recordStartTime = millis();
  const btn = document.getElementById('record-button');
  if (btn) btn.textContent = 'Stop Recording';
}

/** Stop recording and trigger download. */
function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== 'inactive') {
    mediaRecorder.stop();
  }
}

/** Compute drawBase from tip so that dist(drawBase, drawTip) = 2 * rectWidth.
 *  Matches the live pipeline adjustment at lines 4036-4059. */
function computeDrawBase(tipX, tipY, baseX, baseY, rectWidth) {
  const dx = tipX - baseX, dy = tipY - baseY;
  const len = Math.sqrt(dx * dx + dy * dy);
  if (len < 0.001) return { x: baseX, y: baseY };
  const dirX = dx / len, dirY = dy / len;
  const targetLen = 2 * rectWidth;
  return { x: tipX - dirX * targetLen, y: tipY - dirY * targetLen };
}

/** Compute bezier control points with 30-degree rotation (matches calculateBezierControlPoints). */
function computeRotatedControlPoints(drawBaseX, drawBaseY, drawTipX, drawTipY, pipX, pipY) {
  const baseToPipX = pipX - drawBaseX, baseToPipY = pipY - drawBaseY;
  const tipToPipX = pipX - drawTipX, tipToPipY = pipY - drawTipY;
  let cp1X = drawBaseX + baseToPipX * 0.7;
  let cp1Y = drawBaseY + baseToPipY * 0.7;
  let cp2X = drawTipX + tipToPipX * 0.6;
  let cp2Y = drawTipY + tipToPipY * 0.6;

  const baseToTipX = drawTipX - drawBaseX, baseToTipY = drawTipY - drawBaseY;
  const btLen = Math.sqrt(baseToTipX * baseToTipX + baseToTipY * baseToTipY);
  if (btLen > 0.001) {
    const btNX = baseToTipX / btLen, btNY = baseToTipY / btLen;
    const perpX = -btNY, perpY = btNX;
    const rot = 30 * Math.PI / 180;
    // Rotate cp1 around base
    let v1x = cp1X - drawBaseX, v1y = cp1Y - drawBaseY;
    let v1Len = Math.sqrt(v1x * v1x + v1y * v1y);
    if (v1Len > 0.001) {
      let a1 = Math.atan2(v1y, v1x);
      let dir1 = (v1x * perpX + v1y * perpY) >= 0 ? 1 : -1;
      a1 += dir1 * rot;
      cp1X = drawBaseX + Math.cos(a1) * v1Len;
      cp1Y = drawBaseY + Math.sin(a1) * v1Len;
    }
    // Rotate cp2 around tip
    let v2x = cp2X - drawTipX, v2y = cp2Y - drawTipY;
    let v2Len = Math.sqrt(v2x * v2x + v2y * v2y);
    if (v2Len > 0.001) {
      let a2 = Math.atan2(v2y, v2x);
      let dir2 = (v2x * perpX + v2y * perpY) >= 0 ? -1 : 1;
      a2 += dir2 * rot;
      cp2X = drawTipX + Math.cos(a2) * v2Len;
      cp2Y = drawTipY + Math.sin(a2) * v2Len;
    }
  }
  return { cp1X, cp1Y, cp2X, cp2Y };
}

/** Capture the current hand shapes as an animation keyframe.
 *  Stores the adjusted draw positions + shape params. */
function captureKeyframe() {
  // In fake hand mode, build keyframe directly from fake data
  const kf = { id: Date.now(), t: 0, hands: {} };
  let hasData = false;
  for (let i in paperShapes) {
    kf.hands[i] = {};
    for (let finger in paperShapes[i]) {
      const sd = paperShapes[i][finger];
      if (!sd || !sd.shape) continue;
      const tip = lerpedPositions[i] && lerpedPositions[i][finger];
      const base = lerpedBasePositions[i] && lerpedBasePositions[i][finger];
      const pip = lerpedPipPositions[i] && lerpedPipPositions[i][finger];
      if (!tip || !base) continue;
      const rw = sd.strokeWidth || 30;
      // Compute adjusted draw positions (length = 2 * rectWidth, tip pinned)
      const db = computeDrawBase(tip.x, tip.y, base.x, base.y, rw);
      kf.hands[i][finger] = {
        tipX: tip.x, tipY: tip.y,
        baseX: db.x, baseY: db.y,
        pipX: pip ? pip.x : (db.x + tip.x) / 2,
        pipY: pip ? pip.y : (db.y + tip.y) / 2,
        rectWidth: rw,
        rectBezOrCircle: sd.rectBezOrCircle || 0,
        rectOrBez: sd.rectOrBez || 0,
        color: sd.color || '#000000'
      };
      hasData = true;
    }
  }
  if (!hasData) return null;
  animKeyframes.push(kf);
  redistributeKeyframeTimes();
  updateKeyframeCountLabel();
  renderAnimTimeline();
  return kf;
}

function redistributeKeyframeTimes() {
  const n = animKeyframes.length;
  if (n < 2) { if (n === 1) animKeyframes[0].t = 0; return; }
  // Space evenly with room for the return-to-first segment
  // e.g., 3 keyframes → t = 0, 0.333, 0.667 (last 0.333 is return to first)
  for (let i = 0; i < n; i++) animKeyframes[i].t = i / n;
}

function updateKeyframeCountLabel() {
  const el = document.getElementById('anim-keyframe-count');
  if (el) el.textContent = animKeyframes.length + ' keyframe' + (animKeyframes.length !== 1 ? 's' : '') + ' captured';
}

/** Lerp between two finger param objects. Color comes from kfA (no blending). */
function lerpFingerParams(a, b, t) {
  return {
    tipX: a.tipX + (b.tipX - a.tipX) * t,
    tipY: a.tipY + (b.tipY - a.tipY) * t,
    baseX: a.baseX + (b.baseX - a.baseX) * t,
    baseY: a.baseY + (b.baseY - a.baseY) * t,
    pipX: a.pipX + (b.pipX - a.pipX) * t,
    pipY: a.pipY + (b.pipY - a.pipY) * t,
    rectWidth: a.rectWidth + (b.rectWidth - a.rectWidth) * t,
    rectBezOrCircle: a.rectBezOrCircle + (b.rectBezOrCircle - a.rectBezOrCircle) * t,
    rectOrBez: a.rectOrBez + (b.rectOrBez - a.rectOrBez) * t,
    color: a.color // no color blending
  };
}

/** Compute blended + straightened control points for animation (matches live pipeline). */
/** Compute control points matching the live pipeline's double-straightening.
 *  Pass 1: blend rect/bez CPs + straighten toward 1/3 (builds Paper.js path).
 *  Pass 2: exportPaperBezierToP5 reads those handles and straightens again. */
function computeAnimControlPoints(p) {
  const dx = p.tipX - p.baseX, dy = p.tipY - p.baseY;
  const rcp1X = p.baseX + dx / 3, rcp1Y = p.baseY + dy / 3;
  const rcp2X = p.baseX + (2 * dx) / 3, rcp2Y = p.baseY + (2 * dy) / 3;
  const { cp1X: bcp1X, cp1Y: bcp1Y, cp2X: bcp2X, cp2Y: bcp2Y } =
    computeRotatedControlPoints(p.baseX, p.baseY, p.tipX, p.tipY, p.pipX, p.pipY);
  // Blend based on rectOrBez
  let cp1X = rcp1X + (bcp1X - rcp1X) * p.rectOrBez;
  let cp1Y = rcp1Y + (bcp1Y - rcp1Y) * p.rectOrBez;
  let cp2X = rcp2X + (bcp2X - rcp2X) * p.rectOrBez;
  let cp2Y = rcp2Y + (bcp2Y - rcp2Y) * p.rectOrBez;
  // Pass 1: straighten (as when building the Paper.js path)
  const rbc = p.rectBezOrCircle;
  const sf1 = rbc >= 0.3 ? 1 : (rbc > 0 ? rbc / 0.3 : 0);
  if (sf1 > 0) {
    cp1X += (rcp1X - cp1X) * sf1; cp1Y += (rcp1Y - cp1Y) * sf1;
    cp2X += (rcp2X - cp2X) * sf1; cp2Y += (rcp2Y - cp2Y) * sf1;
  }
  // Pass 2: exportPaperBezierToP5 straightens again with the same factor
  const sf2 = Math.min(1, rbc / 0.3);
  if (sf2 > 0) {
    cp1X += (rcp1X - cp1X) * sf2; cp1Y += (rcp1Y - cp1Y) * sf2;
    cp2X += (rcp2X - cp2X) * sf2; cp2Y += (rcp2Y - cp2Y) * sf2;
  }
  return { cp1X, cp1Y, cp2X, cp2Y };
}

/** Draw a single finger shape from params — matches the live drawing pipeline. */
function drawFingerFromParams(p) {
  const c = color(p.color);
  const rbc = p.rectBezOrCircle;
  const dx = p.tipX - p.baseX, dy = p.tipY - p.baseY;
  const len = Math.sqrt(dx * dx + dy * dy);
  const ang = len > 0.001 ? Math.atan2(dy, dx) : 0;

  if (rbc >= 0.5) {
    // Circle / rounded-square (matches exportPaperCircleToP5)
    const fullDiam = p.rectWidth * 1.28;
    const fullR = fullDiam / 2;
    const tp = (rbc - 0.5) * 2;
    const br = (fullR / 2) + (fullR / 2) * tp;
    const edge = p.rectWidth + (fullDiam - p.rectWidth) * tp;
    // Center interpolates from line-segment midpoint to tip
    const segLen = Math.min(p.rectWidth, len);
    const dirX = len > 0.001 ? dx / len : 0, dirY = len > 0.001 ? dy / len : 0;
    const segMidX = p.tipX - (segLen / 2) * dirX;
    const segMidY = p.tipY - (segLen / 2) * dirY;
    const cx = segMidX + (p.tipX - segMidX) * tp;
    const cy = segMidY + (p.tipY - segMidY) * tp;
    push();
    fill(red(c), green(c), blue(c));
    noStroke();
    translate(cx, cy);
    rotate(ang); // constant angle (matches live: lineSegmentAngle)
    rectMode(CENTER);
    rect(0, 0, edge, edge, br);
    rectMode(CORNER);
    pop();
  } else if (rbc >= 0.3) {
    // Transition rectangle (matches exportTransitionRectangleToP5)
    if (len < 0.001) return;
    const tp = (rbc - 0.3) / 0.2;
    const fullLen = len;
    const startLen = fullLen * 0.7;
    const visLen = startLen - (startLen - p.rectWidth) * tp;
    const fullR = (p.rectWidth * 1.28) / 2;
    const br = Math.max(0, (fullR / 2) * tp);
    const dirX = dx / len, dirY = dy / len;
    const sx = p.tipX - visLen * dirX, sy = p.tipY - visLen * dirY;
    fill(red(c), green(c), blue(c));
    noStroke();
    push();
    translate((sx + p.tipX) / 2, (sy + p.tipY) / 2);
    rotate(ang);
    rectMode(CENTER);
    rect(0, 0, visLen, p.rectWidth, br);
    rectMode(CORNER);
    pop();
  } else {
    // Bezier with 30-degree rotated control points + partial undrawing
    const { cp1X, cp1Y, cp2X, cp2Y } = computeAnimControlPoints(p);
    // Visible portion: 1.0 at rbc=0, 0.7 at rbc=0.3
    let visPortion = rbc > 0 ? 1.0 - (rbc / 0.3) * 0.3 : 1.0;
    noFill();
    strokeCap(SQUARE);
    stroke(red(c), green(c), blue(c));
    strokeWeight(p.rectWidth);
    if (visPortion >= 0.999) {
      bezier(p.baseX, p.baseY, cp1X, cp1Y, cp2X, cp2Y, p.tipX, p.tipY);
    } else {
      // Partial draw from tip backward — sample bezier, use curveVertex for smooth interpolation
      // Matches live pipeline's path sampling + curveVertex approach
      const targetLen = Math.max(p.rectWidth, len * visPortion);
      const steps = Math.max(20, Math.ceil(len * 2));
      const startT = 1.0 - visPortion;
      beginShape();
      noFill();
      for (let s = 0; s <= steps; s++) {
        const t = startT + (1.0 - startT) * (s / steps);
        const u = 1 - t;
        const bx = u*u*u*p.baseX + 3*u*u*t*cp1X + 3*u*t*t*cp2X + t*t*t*p.tipX;
        const by = u*u*u*p.baseY + 3*u*u*t*cp1Y + 3*u*t*t*cp2Y + t*t*t*p.tipY;
        curveVertex(bx, by);
      }
      endShape();
    }
  }
}

/** Find the two keyframes surrounding a loopT value and compute segment t. */
function resolveKeyframePair(loopT) {
  const n = animKeyframes.length;
  if (n === 0) return null;
  if (n === 1) return { kfA: animKeyframes[0], kfB: animKeyframes[0], segT: 0 };
  // Find which segment we're in (including the wrap-around segment from last→first)
  let idx = n - 1; // default to last segment (wrap-around)
  for (let i = 0; i < n - 1; i++) {
    if (loopT >= animKeyframes[i].t && loopT < animKeyframes[i + 1].t) { idx = i; break; }
  }
  const kfA = animKeyframes[idx];
  // Next keyframe wraps around to the first
  const kfB = idx < n - 1 ? animKeyframes[idx + 1] : animKeyframes[0];
  // Segment length (wraps: last keyframe t to 1.0, then 0.0 to first keyframe t)
  const segStart = kfA.t;
  const segEnd = idx < n - 1 ? kfB.t : 1.0;
  const segLen = segEnd - segStart;
  let segT = segLen > 0.001 ? (loopT - segStart) / segLen : 0;
  segT = Math.max(0, Math.min(1, segT));
  segT = applyAnimEasing(segT);
  return { kfA, kfB, segT };
}

/** Lazily-created buffer for animation preview (MULTIPLY blend). */
let animPreviewBuffer = null;

/** Draw the animation preview frame. Returns the current loopT (0-1) for playhead. */
function drawAnimPreview() {
  if (animKeyframes.length < 1) return 0;
  const elapsed = (millis() - animPreviewStart) / 1000;
  const loopT = (elapsed % animTotalDuration) / animTotalDuration;
  const pair = resolveKeyframePair(loopT);
  if (!pair) return 0;
  const { kfA, kfB, segT } = pair;

  // Draw to an offscreen buffer with MULTIPLY blend (matches live view overlap behavior)
  if (!animPreviewBuffer || animPreviewBuffer.width !== width || animPreviewBuffer.height !== height) {
    if (animPreviewBuffer) animPreviewBuffer.remove();
    animPreviewBuffer = createGraphics(width, height);
  }
  animPreviewBuffer.clear();
  animPreviewBuffer.blendMode(MULTIPLY);

  // Draw all finger shapes to the buffer
  const allHands = new Set([...Object.keys(kfA.hands), ...Object.keys(kfB.hands)]);
  for (const hi of allHands) {
    const allFingers = new Set([
      ...Object.keys(kfA.hands[hi] || {}),
      ...Object.keys(kfB.hands[hi] || {})
    ]);
    for (const fn of allFingers) {
      const fA = kfA.hands[hi] && kfA.hands[hi][fn];
      const fB = kfB.hands[hi] && kfB.hands[hi][fn];
      if (!fA && !fB) continue;
      const p = fA && fB ? lerpFingerParams(fA, fB, segT) : (fA || fB);
      drawFingerFromParamsToBuffer(animPreviewBuffer, p);
    }
  }

  // Composite buffer onto main canvas, mirrored
  push();
  translate(width, 0);
  scale(-1, 1);
  image(animPreviewBuffer, 0, 0);
  pop();
  return loopT;
}

/** Draw a finger shape to a p5.Graphics buffer (same as drawFingerFromParams but on a buffer). */
function drawFingerFromParamsToBuffer(buf, p) {
  const c = color(p.color);
  const rbc = p.rectBezOrCircle;
  const dx = p.tipX - p.baseX, dy = p.tipY - p.baseY;
  const len = Math.sqrt(dx * dx + dy * dy);
  const ang = len > 0.001 ? Math.atan2(dy, dx) : 0;

  if (rbc >= 0.5) {
    const fullDiam = p.rectWidth * 1.28;
    const fullR = fullDiam / 2;
    const tp = (rbc - 0.5) * 2;
    const br = (fullR / 2) + (fullR / 2) * tp;
    const edge = p.rectWidth + (fullDiam - p.rectWidth) * tp;
    const segLen = Math.min(p.rectWidth, len);
    const dirX = len > 0.001 ? dx / len : 0, dirY = len > 0.001 ? dy / len : 0;
    const segMidX = p.tipX - (segLen / 2) * dirX, segMidY = p.tipY - (segLen / 2) * dirY;
    const cx = segMidX + (p.tipX - segMidX) * tp;
    const cy = segMidY + (p.tipY - segMidY) * tp;
    buf.push();
    buf.fill(red(c), green(c), blue(c));
    buf.noStroke();
    buf.translate(cx, cy);
    buf.rotate(ang);
    buf.rectMode(CENTER);
    buf.rect(0, 0, edge, edge, br);
    buf.rectMode(CORNER);
    buf.pop();
  } else if (rbc >= 0.3) {
    if (len < 0.001) return;
    const tp = (rbc - 0.3) / 0.2;
    const startLen = len * 0.7;
    const visLen = startLen - (startLen - p.rectWidth) * tp;
    const fullR = (p.rectWidth * 1.28) / 2;
    const br = Math.max(0, (fullR / 2) * tp);
    const dirX = dx / len, dirY = dy / len;
    const sx = p.tipX - visLen * dirX, sy = p.tipY - visLen * dirY;
    buf.fill(red(c), green(c), blue(c));
    buf.noStroke();
    buf.push();
    buf.translate((sx + p.tipX) / 2, (sy + p.tipY) / 2);
    buf.rotate(ang);
    buf.rectMode(CENTER);
    buf.rect(0, 0, visLen, p.rectWidth, br);
    buf.rectMode(CORNER);
    buf.pop();
  } else {
    const { cp1X, cp1Y, cp2X, cp2Y } = computeAnimControlPoints(p);
    let visPortion = rbc > 0 ? 1.0 - (rbc / 0.3) * 0.3 : 1.0;
    buf.noFill();
    buf.strokeCap(SQUARE);
    buf.stroke(red(c), green(c), blue(c));
    buf.strokeWeight(p.rectWidth);
    if (visPortion >= 0.999) {
      buf.bezier(p.baseX, p.baseY, cp1X, cp1Y, cp2X, cp2Y, p.tipX, p.tipY);
    } else {
      const steps = Math.max(20, Math.ceil(len * 2));
      const startT = 1.0 - visPortion;
      buf.beginShape();
      buf.noFill();
      for (let s = 0; s <= steps; s++) {
        const t = startT + (1.0 - startT) * (s / steps);
        const u = 1 - t;
        buf.curveVertex(
          u*u*u*p.baseX + 3*u*u*t*cp1X + 3*u*t*t*cp2X + t*t*t*p.tipX,
          u*u*u*p.baseY + 3*u*u*t*cp1Y + 3*u*t*t*cp2Y + t*t*t*p.tipY
        );
      }
      buf.endShape();
    }
  }
}

/** Normalize a Paper.js path to a fixed-point polygon (for SVG export). */
function normalizePathForAnimation(paperPath, numPoints) {
  numPoints = numPoints || ANIM_PATH_POINTS;
  const len = paperPath.length;
  if (len < 1) return null;
  const pts = [];
  for (let i = 0; i <= numPoints; i++) {
    const pt = paperPath.getPointAt(Math.min((i / numPoints) * len, len));
    if (pt) pts.push({ x: Math.round(pt.x * 100) / 100, y: Math.round(pt.y * 100) / 100 });
  }
  if (pts.length < 2) return null;
  let d = 'M' + pts[0].x + ' ' + pts[0].y;
  for (let i = 1; i < pts.length; i++) d += ' L' + pts[i].x + ' ' + pts[i].y;
  d += 'Z';
  return d;
}

/** Build the timeline DOM element inside the Animation fieldset. */
function buildAnimTimeline() {
  const placeholder = document.getElementById('anim-timeline-wrap');
  if (!placeholder) return;
  placeholder.innerHTML = '';
  const track = document.createElement('div');
  track.className = 'anim-timeline-track';
  track.id = 'anim-timeline-track';
  placeholder.appendChild(track);
  // Playhead
  const playhead = document.createElement('div');
  playhead.className = 'anim-timeline-playhead';
  playhead.id = 'anim-timeline-playhead';
  track.appendChild(playhead);
  renderAnimTimeline();
}

/** Render keyframe dots on the timeline. Called after capture/delete/drag. */
function renderAnimTimeline() {
  const track = document.getElementById('anim-timeline-track');
  if (!track) return;
  // Remove old dots (keep playhead)
  track.querySelectorAll('.anim-timeline-dot').forEach(d => d.remove());
  animKeyframes.forEach((kf, i) => {
    const dot = document.createElement('div');
    dot.className = 'anim-timeline-dot';
    dot.style.left = (kf.t * 100) + '%';
    dot.title = 'Keyframe ' + (i + 1);
    dot.textContent = i + 1;
    dot.dataset.index = i;
    // Drag to reposition
    dot.addEventListener('mousedown', (e) => startDotDrag(e, i));
    dot.addEventListener('touchstart', (e) => startDotDrag(e, i), { passive: false });
    // Right-click or double-click to delete
    dot.addEventListener('dblclick', () => {
      animKeyframes.splice(i, 1);
      redistributeKeyframeTimes();
      updateKeyframeCountLabel();
      renderAnimTimeline();
    });
    track.appendChild(dot);
  });
}

function startDotDrag(e, index) {
  e.preventDefault();
  const track = document.getElementById('anim-timeline-track');
  if (!track) return;
  const rect = track.getBoundingClientRect();
  const n = animKeyframes.length;
  // Bounds: can't go before previous keyframe or after next
  const minT = index > 0 ? animKeyframes[index - 1].t + 0.01 : 0;
  const maxT = index < n - 1 ? animKeyframes[index + 1].t - 0.01 : 1;

  function onMove(ev) {
    const clientX = ev.touches ? ev.touches[0].clientX : ev.clientX;
    let t = (clientX - rect.left) / rect.width;
    t = Math.max(minT, Math.min(maxT, t));
    animKeyframes[index].t = Math.round(t * 1000) / 1000;
    renderAnimTimeline();
  }
  function onUp() {
    window.removeEventListener('mousemove', onMove);
    window.removeEventListener('mouseup', onUp);
    window.removeEventListener('touchmove', onMove);
    window.removeEventListener('touchend', onUp);
  }
  window.addEventListener('mousemove', onMove);
  window.addEventListener('mouseup', onUp);
  window.addEventListener('touchmove', onMove, { passive: false });
  window.addEventListener('touchend', onUp);
}

/** Update the playhead position on the timeline (called from draw). */
function updateAnimPlayhead(loopT) {
  const ph = document.getElementById('anim-timeline-playhead');
  if (ph) ph.style.left = (loopT * 100) + '%';
}

/** Build a Paper.js shape from finger params and return normalized SVG path data.
 *  ALWAYS uses a rounded rectangle to ensure consistent polygon topology
 *  across all shape states — prevents flash when CSS interpolates between them. */
function fingerParamsToSVGPath(p) {
  const rbc = p.rectBezOrCircle;
  const dx = p.tipX - p.baseX, dy = p.tipY - p.baseY;
  const len = Math.sqrt(dx * dx + dy * dy);
  const ang = len > 0.001 ? Math.atan2(dy, dx) : 0;
  const fullDiam = p.rectWidth * 1.28;
  const fullR = fullDiam / 2;

  // All states produce a rounded rectangle with varying dimensions.
  // This ensures consistent 48-point polygon topology for CSS interpolation.
  let rectW, rectH, br, cx, cy;

  if (rbc >= 0.5) {
    // Circle/square
    const tp = (rbc - 0.5) * 2;
    const edge = p.rectWidth + (fullDiam - p.rectWidth) * tp;
    br = (fullR / 2) + (fullR / 2) * tp;
    rectW = edge; rectH = edge;
    const segLen = Math.min(p.rectWidth, len);
    const dirX = len > 0.001 ? dx / len : 0, dirY = len > 0.001 ? dy / len : 0;
    const segMidX = p.tipX - (segLen / 2) * dirX, segMidY = p.tipY - (segLen / 2) * dirY;
    cx = segMidX + (p.tipX - segMidX) * tp;
    cy = segMidY + (p.tipY - segMidY) * tp;
  } else if (rbc >= 0.3) {
    // Transition rectangle
    const tp = (rbc - 0.3) / 0.2;
    const startLen = len * 0.7;
    const visLen = startLen - (startLen - p.rectWidth) * tp;
    br = Math.max(0, (fullR / 2) * tp);
    rectW = visLen; rectH = p.rectWidth;
    const dirX = len > 0.001 ? dx / len : 0, dirY = len > 0.001 ? dy / len : 0;
    const sx = p.tipX - visLen * dirX, sy = p.tipY - visLen * dirY;
    cx = (sx + p.tipX) / 2; cy = (sy + p.tipY) / 2;
  } else {
    // Bezier state — approximate as a rounded rectangle
    // Length = finger length (or partial), width = rectWidth
    let visLen = len;
    if (rbc > 0) visLen = Math.max(p.rectWidth, len * (1.0 - (rbc / 0.3) * 0.3));
    br = 0;
    rectW = visLen; rectH = p.rectWidth;
    const dirX = len > 0.001 ? dx / len : 0, dirY = len > 0.001 ? dy / len : 0;
    const sx = p.tipX - visLen * dirX, sy = p.tipY - visLen * dirY;
    cx = (sx + p.tipX) / 2; cy = (sy + p.tipY) / 2;
  }

  if (rectW < 1 || rectH < 1) return null;
  const paperPath = new paper.Path.Rectangle(
    new paper.Rectangle(cx - rectW / 2, cy - rectH / 2, rectW, rectH),
    new paper.Size(Math.min(br, rectW / 2, rectH / 2), Math.min(br, rectW / 2, rectH / 2))
  );
  paperPath.rotate(ang * 180 / Math.PI, new paper.Point(cx, cy));
  const d = normalizePathForAnimation(paperPath);
  paperPath.remove();
  return d;
}

/** Export keyframes as an animated SVG with CSS @keyframes.
 *  Samples the animation at many time points, reconstructs Paper.js shapes
 *  from interpolated finger params, and generates compatible SVG paths. */
function exportAnimatedSVG() {
  if (animKeyframes.length < 2) {
    console.warn('Need at least 2 keyframes to export animation');
    return;
  }

  // Find common finger IDs across all keyframes
  const fingerSets = animKeyframes.map(kf => {
    const s = new Set();
    for (const hi in kf.hands) for (const fn in kf.hands[hi]) s.add(hi + ':' + fn);
    return s;
  });
  const commonFingers = [...fingerSets[0]].filter(fid => fingerSets.every(s => s.has(fid)));
  if (commonFingers.length === 0) { console.warn('No fingers common to all keyframes'); return; }

  // Sample at many time points for smooth CSS animation
  const numSamples = 30;
  const dur = animTotalDuration;
  let style = '';
  let paths = '';

  for (const fid of commonFingers) {
    const [hi, fn] = fid.split(':');
    const safeFid = fid.replace(/:/g, '-');
    const firstColor = animKeyframes[0].hands[hi][fn].color;

    style += `@keyframes a-${safeFid}{`;
    let firstD = null;
    for (let s = 0; s < numSamples; s++) {
      const loopT = s / numSamples;
      const pct = Math.round(loopT * 100);
      const pair = resolveKeyframePair(loopT);
      if (!pair) continue;
      const fA = pair.kfA.hands[hi] && pair.kfA.hands[hi][fn];
      const fB = pair.kfB.hands[hi] && pair.kfB.hands[hi][fn];
      if (!fA) continue;
      const p = fB ? lerpFingerParams(fA, fB, pair.segT) : fA;
      const d = fingerParamsToSVGPath(p);
      if (!d) continue;
      if (!firstD) firstD = d;
      style += `${pct}%{d:path("${d}")}`;
    }
    // 100% = same as 0% for seamless loop
    if (firstD) style += `100%{d:path("${firstD}")}`;
    style += '}';
    if (firstD) {
      // Each finger gets an id so it can be referenced by clipPath
      paths += `<path id="f-${safeFid}" fill="${firstColor}" d="${firstD}" style="animation:a-${safeFid} ${dur}s linear infinite"/>`;
    }
  }

  // Generate overlap layers using SVG clipPath for each finger pair
  let defs = '';
  let overlapPaths = '';
  for (let j = 0; j < commonFingers.length; j++) {
    for (let k = j + 1; k < commonFingers.length; k++) {
      const [hiA, fnA] = commonFingers[j].split(':');
      const [hiB, fnB] = commonFingers[k].split(':');
      const colorA = animKeyframes[0].hands[hiA][fnA].color;
      const colorB = animKeyframes[0].hands[hiB][fnB].color;
      const overlapColor = getIntersectionColor(colorA, colorB);
      if (!overlapColor || overlapColor === '#000000') continue;

      const safeA = commonFingers[j].replace(/:/g, '-');
      const safeB = commonFingers[k].replace(/:/g, '-');
      // clipPath uses finger A's animated shape as the clipping region
      defs += `<clipPath id="clip-${safeA}"><use href="#f-${safeA}"/></clipPath>`;
      // Draw finger B's shape clipped to finger A, filled with brand overlap color
      // This produces the exact intersection region with the correct color
      overlapPaths += `<path fill="${overlapColor}" clip-path="url(#clip-${safeA})" d="${animKeyframes[0].hands[hiB][fnB] ? (function(){ const p = animKeyframes[0].hands[hiB][fnB]; return fingerParamsToSVGPath(p) || ''; })() : ''}" style="animation:a-${safeB} ${dur}s linear infinite"/>`;
    }
  }

  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}">
<defs>${defs}</defs>
<style>${style}</style>
<rect width="100%" height="100%" fill="#fff"/>
<g transform="scale(-1,1) translate(-${width},0)">${paths}${overlapPaths}</g>
</svg>`;

  const blob = new Blob([svg], { type: 'image/svg+xml' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'hand-animation.svg'; a.click();
  URL.revokeObjectURL(url);
}

/** Capture current hand shapes as a named pose for the widget.
 *  Positions are stored relative to the hand centroid for portability. */
function captureWidgetPose(name) {
  if (!name) return;
  const pose = { fingers: {} };
  let cx = 0, cy = 0, count = 0;
  // First pass: collect raw data and compute centroid
  const raw = {};
  for (let i in paperShapes) {
    raw[i] = {};
    for (let finger in paperShapes[i]) {
      const sd = paperShapes[i][finger];
      if (!sd || !sd.shape) continue;
      const tip = lerpedPositions[i] && lerpedPositions[i][finger];
      const base = lerpedBasePositions[i] && lerpedBasePositions[i][finger];
      const pip = lerpedPipPositions[i] && lerpedPipPositions[i][finger];
      if (!tip || !base) continue;
      const rw = sd.strokeWidth || 30;
      const db = computeDrawBase(tip.x, tip.y, base.x, base.y, rw);
      raw[i][finger] = {
        tipX: tip.x, tipY: tip.y,
        baseX: db.x, baseY: db.y,
        pipX: pip ? pip.x : (db.x + tip.x) / 2,
        pipY: pip ? pip.y : (db.y + tip.y) / 2,
        rectWidth: rw,
        rectBezOrCircle: sd.rectBezOrCircle || 0,
        rectOrBez: sd.rectOrBez || 0,
        color: sd.color || '#000000'
      };
      // Include both tip and base in centroid for better centering
      cx += tip.x + db.x; cy += tip.y + db.y; count += 2;
    }
  }
  if (count === 0) return;
  cx /= count; cy /= count;
  // Second pass: offset by centroid + compute refSize with dynamic padding
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  let maxRW = 0;
  for (let i in raw) {
    pose.fingers[i] = {};
    for (let finger in raw[i]) {
      const r = raw[i][finger];
      const f = {
        tipX: r.tipX - cx, tipY: r.tipY - cy,
        baseX: r.baseX - cx, baseY: r.baseY - cy,
        pipX: r.pipX - cx, pipY: r.pipY - cy,
        rectWidth: r.rectWidth,
        rectBezOrCircle: r.rectBezOrCircle,
        rectOrBez: r.rectOrBez,
        color: r.color
      };
      pose.fingers[i][finger] = f;
      minX = Math.min(minX, f.tipX, f.baseX);
      minY = Math.min(minY, f.tipY, f.baseY);
      maxX = Math.max(maxX, f.tipX, f.baseX);
      maxY = Math.max(maxY, f.tipY, f.baseY);
      if (r.rectWidth > maxRW) maxRW = r.rectWidth;
    }
  }
  pose.refSize = Math.max(maxX - minX, maxY - minY) + maxRW * 3;
  widgetPoses[name] = pose;
  updatePoseList();
}

function updatePoseList() {
  const el = document.getElementById('widget-pose-list');
  if (!el) return;
  const names = Object.keys(widgetPoses);
  el.textContent = names.length === 0 ? 'No poses saved' : names.join(', ');
}

function toggleWidgetPreview(on) {
  if (on) {
    // Destroy existing instance if any
    if (widgetPreviewInstance) { widgetPreviewInstance.destroy(); widgetPreviewInstance = null; }
    if (Object.keys(widgetPoses).length === 0) {
      console.warn('No poses captured — capture at least one pose first');
      return;
    }
    widgetPreviewInstance = EraHand.create({
      poses: widgetPoses,
      size: widgetSize,
      defaultPose: Object.keys(widgetPoses)[0],
      pressPose: widgetPoses.grab ? 'grab' : null,
      cursor: true,
      lerpSpeed: 0.12,
      cursorSmooth: 0.3
    });
    // Set up hover rules if poses exist
    if (widgetPoses.pointer) widgetPreviewInstance.on('a, button', 'pointer');
    if (widgetPoses.grab) widgetPreviewInstance.on('.draggable, input[type="range"]', 'grab');
  } else {
    if (widgetPreviewInstance) { widgetPreviewInstance.destroy(); widgetPreviewInstance = null; }
  }
}

function exportWidgetBundle() {
  if (Object.keys(widgetPoses).length === 0) {
    console.warn('No poses captured');
    return;
  }
  // Fetch the engine source, append poses, download as one file
  fetch('era-hand.js').then(r => r.text()).then(engineSrc => {
    const poses = JSON.stringify({ poses: widgetPoses });
    const bundle = engineSrc + '\n\n// --- Embedded poses ---\nwindow.EraHandPoses = ' + poses + ';\n';
    const blob = new Blob([bundle], { type: 'application/javascript' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'era-hand-bundle.js';
    a.click();
    URL.revokeObjectURL(url);
  });
}

function exportWidgetPoses() {
  const json = JSON.stringify({ poses: widgetPoses }, null, 2);
  const blob = new Blob([json], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = 'era-hand-poses.json'; a.click();
  URL.revokeObjectURL(url);
}

async function setup() {
  const { width, height } = calculateCanvasSize();
  createCanvas(width, height);
	frameRate(60)
  
  // Create graphics buffer for dots
  handsBuffer = createGraphics(width, height);
  
  // Initialize Paper.js on a hidden canvas for shape management (not rendering)
  const hiddenCanvas = document.createElement('canvas');
  hiddenCanvas.width = width;
  hiddenCanvas.height = height;
  hiddenCanvas.style.display = 'none'; // Hide the canvas
  document.body.appendChild(hiddenCanvas);
  
  // Setup Paper.js with the hidden canvas
  paper.setup(hiddenCanvas);
  paper.view.viewSize = new paper.Size(width, height);
  
  // Full-res webcam for display; ML uses a 640×480 downscale (same geometry as old video.size(640,480))
  video = createCapture({
    video: {
      facingMode: 'user',
      width: { ideal: 1920 },
      height: { ideal: 1080 }
    },
    audio: false
  });
  video.hide();
  mlInputCanvas = document.createElement('canvas');
  mlInputCanvas.width = ML_INPUT_W;
  mlInputCanvas.height = ML_INPUT_H;
  mlInputCtx = mlInputCanvas.getContext('2d');
  
  nomediaImg = loadImage('nomedia.png');
  
  // Initialize TensorFlow.js Hand Pose detector
  try {
    const model = handPoseDetection.SupportedModels.MediaPipeHands;
    const detectorConfig = {
      runtime: 'tfjs', // Use tfjs runtime instead of mediapipe (self-contained)
      modelType: 'full',
      maxHands: 2
    };
    detector = await handPoseDetection.createDetector(model, detectorConfig);
    console.log("Handpose detector ready!");
  } catch (error) {
    console.error("Error loading handpose detector:", error);
  }
  
  const controlPanelRoot = document.getElementById('control-panel');
  if (!controlPanelRoot) {
    console.error('Control panel not found: #control-panel');
  } else if (typeof CottonControlPanel === 'undefined') {
    console.error('CottonControlPanel not loaded - add cotton-controlpanel/controlpanel.js before sketch.js');
  } else {
    const config = SLIDER_CONFIG;
    cp = CottonControlPanel.create({
      parent: controlPanelRoot,
      controls: [
        {
          type: 'fieldset',
          id: 'tab-logo',
          legend: 'Logo',
          plain: true,
          controls: [
            {
              type: 'checkbox',
              id: 'normalizeHands',
              label: 'Pin logo to center',
              value: normalizeHands,
              toggle: true,
              onChange: (v) => {
                normalizeHands = v;
              }
            },
            {
              type: 'checkbox',
              id: 'multicolor',
              label: 'Multicolor',
              value: multicolor,
              toggle: true,
              onChange: (v) => {
                multicolor = v;
              }
            },
            {
              type: 'checkbox',
              id: 'bigHands',
              label: 'Big hands',
              value: bigHands,
              toggle: true,
              onChange: (v) => {
                bigHands = v;
              }
            }
          ]
        },
        {
          type: 'fieldset',
          id: 'tab-fill',
          legend: 'Hand Background',
          plain: true,
          controls: [
            {
              type: 'swatches',
              id: 'handFill0',
              value: handFillSelection[0],
              swatches: HAND_FILL_SWATCHES,
              customButton: true,
              customDefault: handFillCustomColor,
              uploadButton: true,
              uploadAccept: 'image/*,video/*',
              onChange: (v) => {
                handFillSelection[0] = v === '__custom__' ? '__custom__' : v;
                if (!handFillSplitHands) handFillSelection[1] = handFillSelection[0];
              },
              onCustomColor: (v) => {
                handFillCustomColor = v;
                handFillSelection[0] = '__custom__';
                if (!handFillSplitHands) handFillSelection[1] = '__custom__';
              },
              onUpload: (file) => { handleTextureUpload(file, 0); }
            },
            {
              type: 'checkbox',
              id: 'handFillSplitHands',
              label: 'Different for each hand',
              value: handFillSplitHands,
              toggle: true,
              onChange: (v) => {
                handFillSplitHands = v;
                const fs = document.getElementById('handFill1-fieldset');
                if (fs) fs.style.display = v ? '' : 'none';
                if (!v) handFillSelection[1] = handFillSelection[0];
              }
            },
            {
              type: 'fieldset',
              id: 'handFill1-fieldset',
              legend: 'Right Hand',
              hidden: true,
              controls: [
                {
                  type: 'swatches',
                  id: 'handFill1',
                  value: handFillSelection[1],
                  swatches: HAND_FILL_SWATCHES,
                  customButton: true,
                  customDefault: handFillCustomColor,
                  uploadButton: true,
                  uploadAccept: 'image/*,video/*',
                  onChange: (v) => {
                    handFillSelection[1] = v === '__custom__' ? '__custom__' : v;
                  },
                  onCustomColor: (v) => {
                    handFillCustomColor = v;
                    handFillSelection[1] = '__custom__';
                  },
                  onUpload: (file) => { handleTextureUpload(file, 1); }
                }
              ]
            },
            {
              type: 'checkbox',
              id: 'handFillScaleToHand',
              label: 'Scale to hand size',
              value: handFillScaleToHand,
              toggle: true,
              onChange: (v) => {
                handFillScaleToHand = v;
              }
            }
          ]
        },
        {
          type: 'fieldset',
          id: 'tab-bg',
          legend: 'Background',
          plain: true,
          controls: [
            {
              type: 'radio',
              id: 'bgMode',
              name: 'bg-mode',
              value: backgroundMode,
              ariaLabel: 'Background mode',
              options: [
                { value: 'plain', label: 'plain' },
                { value: 'webcam', label: 'webcam' },
                { value: 'greenscreen', label: 'greenscreen' },
                { value: 'uploaded', label: 'uploaded media' }
              ],
              onChange: (v) => {
                backgroundMode = v;
              }
            },
            {
              type: 'file',
              id: 'bgUpload',
              accept: 'image/*,video/*',
              buttonLabel: 'Upload background media',
              onChange: (file) => {
                if (file) handleBackgroundMediaFile(file);
              }
            },
            {
              type: 'section',
              label: 'Collage'
            },
            {
              type: 'file',
              id: 'bgCollageUpload',
              accept: 'image/*',
              buttonLabel: 'Add image layer',
              onChange: (file) => {
                if (file) addBgImage(file);
              }
            },
            {
              type: 'button',
              id: 'bgCollageClear',
              label: 'Clear all layers',
              variant: 'secondary',
              block: true,
              onClick: () => { bgImages = []; }
            }
          ]
        },
        {
          type: 'fieldset',
          id: 'tab-export',
          legend: 'Export',
          plain: true,
          controls: [
            {
              type: 'slider',
              id: 'saveDelay',
              min: 0,
              max: 10,
              step: 1,
              value: saveCountdownDelay,
              label: 'Self-timer',
              suffix: 's',
              onChange: (v) => { saveCountdownDelay = v; }
            },
            {
              type: 'section',
              label: 'Capture'
            },
            {
              type: 'group',
              className: 'cotton-cp-export-stack',
              controls: [
                {
                  type: 'button',
                  id: 'save-png-button',
                  label: 'Save PNG',
                  variant: 'primary',
                  block: true,
                  medium: true,
                  onClick: (panel) => {
                    const delay = parseInt(panel.values.saveDelay, 10) || 0;
                    saveCountdownDelay = delay;
                    savePendingAction = 'png';
                    if (delay === 0) { saveCroppedPNG(); savePendingAction = null; }
                    else { saveCountdown = delay; saveCountdownStartTime = millis(); }
                  }
                },
                {
                  type: 'button',
                  id: 'save-svg-button',
                  label: 'Save SVG',
                  variant: 'secondary',
                  block: true,
                  medium: true,
                  onClick: (panel) => {
                    const delay = parseInt(panel.values.saveDelay, 10) || 0;
                    saveCountdownDelay = delay;
                    savePendingAction = 'svg';
                    if (delay === 0) { exportSVG(); savePendingAction = null; }
                    else { saveCountdown = delay; saveCountdownStartTime = millis(); }
                  }
                }
              ]
            },
            {
              type: 'section',
              label: 'Record'
            },
            {
              type: 'slider',
              id: 'recordDuration',
              min: 1,
              max: 30,
              step: 1,
              value: recordDuration,
              label: 'Duration',
              suffix: 's',
              onChange: (v) => { recordDuration = v; }
            },
            {
              type: 'button',
              id: 'record-button',
              label: 'Record Video',
              variant: 'primary',
              block: true,
              medium: true,
              onClick: (panel) => {
                if (isRecording) {
                  stopRecording();
                  return;
                }
                const delay = parseInt(panel.values.saveDelay, 10) || 0;
                saveCountdownDelay = delay;
                savePendingAction = 'record';
                if (delay === 0) { startRecording(); savePendingAction = null; }
                else { saveCountdown = delay; saveCountdownStartTime = millis(); }
              }
            }
          ]
        },
        {
          type: 'fieldset',
          id: 'tab-anim',
          legend: 'Animation',
          plain: true,
          controls: [
            {
              type: 'section',
              label: 'Keyframes'
            },
            {
              type: 'button',
              id: 'anim-capture-btn',
              label: 'Capture Pose',
              variant: 'primary',
              block: true,
              medium: true,
              onClick: () => {
                const kf = captureKeyframe();
                if (!kf) console.warn('No hand shapes visible to capture');
              }
            },
            {
              type: 'button',
              id: 'anim-clear-btn',
              label: 'Clear All',
              variant: 'secondary',
              block: true,
              onClick: () => {
                animKeyframes = [];
                animPreview = false;
                updateKeyframeCountLabel();
                renderAnimTimeline();
              }
            },
            {
              type: 'section',
              id: 'anim-keyframe-count',
              label: '0 keyframes captured'
            },
            {
              type: 'group',
              id: 'anim-timeline-wrap',
              className: 'anim-timeline-wrap'
            },
            {
              type: 'slider',
              id: 'animDuration',
              min: 0.5,
              max: 10,
              step: 0.5,
              value: animTotalDuration,
              label: 'Duration',
              suffix: 's',
              onChange: (v) => { animTotalDuration = v; }
            },
            {
              type: 'select',
              id: 'animEasing',
              label: 'Easing',
              value: animEasingType,
              options: [
                { value: 'linear', label: 'Linear' },
                { value: 'ease-in', label: 'Ease In' },
                { value: 'ease-out', label: 'Ease Out' },
                { value: 'smooth-step', label: 'Smooth Step' },
                { value: 'ease-in-out-cubic', label: 'Ease In-Out Cubic' }
              ],
              onChange: (v) => { animEasingType = v; }
            },
            {
              type: 'section',
              label: 'Preview & Export'
            },
            {
              type: 'checkbox',
              id: 'animPreview',
              label: 'Preview animation',
              value: false,
              toggle: true,
              onChange: (v) => {
                animPreview = v;
                if (v) animPreviewStart = millis();
              }
            },
            {
              type: 'button',
              id: 'anim-export-svg-btn',
              label: 'Export Animated SVG',
              variant: 'primary',
              block: true,
              medium: true,
              onClick: () => { exportAnimatedSVG(); }
            }
          ]
        },
        {
          type: 'fieldset',
          id: 'tab-widget',
          legend: 'Widget',
          plain: true,
          controls: [
            {
              type: 'section',
              label: 'Capture Poses'
            },
            {
              type: 'group',
              className: 'cotton-cp-export-stack',
              controls: [
                {
                  type: 'button',
                  id: 'widget-capture-open',
                  label: 'Save as "open"',
                  variant: 'primary',
                  block: true,
                  medium: true,
                  onClick: () => { captureWidgetPose('open'); }
                },
                {
                  type: 'button',
                  id: 'widget-capture-pointer',
                  label: 'Save as "pointer"',
                  variant: 'secondary',
                  block: true,
                  onClick: () => { captureWidgetPose('pointer'); }
                },
                {
                  type: 'button',
                  id: 'widget-capture-grab',
                  label: 'Save as "grab"',
                  variant: 'secondary',
                  block: true,
                  onClick: () => { captureWidgetPose('grab'); }
                },
                {
                  type: 'button',
                  id: 'widget-capture-custom',
                  label: 'Save as custom name...',
                  variant: 'secondary',
                  block: true,
                  onClick: () => {
                    const name = prompt('Pose name:');
                    if (name && name.trim()) captureWidgetPose(name.trim());
                  }
                }
              ]
            },
            {
              type: 'section',
              id: 'widget-pose-list',
              label: 'No poses saved'
            },
            {
              type: 'section',
              label: 'Preview'
            },
            {
              type: 'slider',
              id: 'widgetSize',
              min: 50,
              max: 400,
              step: 10,
              value: widgetSize,
              label: 'Widget size',
              suffix: 'px',
              onChange: (v) => {
                widgetSize = v;
                if (widgetPreviewInstance) widgetPreviewInstance.setSize(v);
              }
            },
            {
              type: 'slider',
              id: 'widgetOpacity',
              min: 0.1,
              max: 1.0,
              step: 0.05,
              value: 1.0,
              label: 'Opacity',
              onChange: (v) => {
                if (widgetPreviewInstance) widgetPreviewInstance.setOpacity(v);
              }
            },
            {
              type: 'checkbox',
              id: 'widgetPreview',
              label: 'Preview widget cursor',
              value: false,
              toggle: true,
              onChange: (v) => { toggleWidgetPreview(v); }
            },
            {
              type: 'section',
              label: 'Export'
            },
            {
              type: 'button',
              id: 'widget-export-json',
              label: 'Export Poses (JSON)',
              variant: 'primary',
              block: true,
              medium: true,
              onClick: () => { exportWidgetPoses(); }
            },
            {
              type: 'button',
              id: 'widget-export-js',
              label: 'Download era-hand.js',
              variant: 'secondary',
              block: true,
              onClick: () => {
                const a = document.createElement('a');
                a.href = 'era-hand.js';
                a.download = 'era-hand.js';
                a.click();
              }
            },
            {
              type: 'button',
              id: 'widget-export-bundle',
              label: 'Export Bundle (JS + Poses)',
              variant: 'secondary',
              block: true,
              onClick: () => { exportWidgetBundle(); }
            }
          ]
        },
        {
          type: 'fieldset',
          id: 'debug-controls',
          legend: 'Debugging',
          hidden: true,
          controls: [
            { type: 'section', label: 'Press D to show or hide' },
            {
              type: 'slider',
              id: 'strokeWeight',
              min: config.strokeWeight.min,
              max: config.strokeWeight.max,
              value: config.strokeWeight.default,
              label: config.strokeWeight.label
            },
            {
              type: 'slider',
              id: 'easing',
              min: config.easing.min,
              max: config.easing.max,
              value: config.easing.default,
              label: config.easing.label,
              suffix: config.easing.suffix || ''
            },
            {
              type: 'slider',
              id: 'skipFrames',
              min: config.skipFrames.min,
              max: config.skipFrames.max,
              value: config.skipFrames.default,
              label: config.skipFrames.label,
              suffix: config.skipFrames.suffix || ''
            },
            {
              type: 'checkbox',
              id: 'debugHandDetection',
              label: 'Debug hand detection',
              value: debugHandDetection,
              onChange: (v) => {
                debugHandDetection = v;
              }
            },
            {
              type: 'checkbox',
              id: 'showRawHandData',
              label: 'Show raw hand data',
              value: showRawHandData,
              onChange: (v) => {
                showRawHandData = v;
              }
            },
            {
              type: 'checkbox',
              id: 'showIntersections',
              label: 'Show intersections',
              value: showIntersections,
              onChange: (v) => {
                showIntersections = v;
              }
            },
            {
              type: 'checkbox',
              id: 'enableDrawHands',
              label: 'Enable draw hands',
              value: enableDrawHands,
              onChange: (v) => {
                enableDrawHands = v;
              }
            }
          ]
        }
      ]
    });
    backgroundMode = cp.values.bgMode;
    handFillSelection[0] = cp.values.handFill0;
    handFillSelection[1] = cp.values.handFill1 || handFillSelection[0];

    // Build tab bar — "Look" combines Logo + Fill + Background
    const tabs = [
      { label: 'Look', fieldsets: ['tab-logo', 'tab-fill', 'tab-bg'] },
      { label: 'Export', fieldsets: ['tab-export'] },
      { label: 'Animate', fieldsets: ['tab-anim'] },
      { label: 'Widget', fieldsets: ['tab-widget'] }
    ];
    const allFieldsetIds = tabs.flatMap(t => t.fieldsets);
    const tabBar = document.createElement('div');
    tabBar.className = 'cotton-cp-tab-bar';
    const stackBody = cp.root.querySelector('.cotton-cp-stack-body');
    stackBody.insertBefore(tabBar, stackBody.firstChild);

    function setTab(idx) {
      allFieldsetIds.forEach(id => {
        const fs = document.getElementById(id);
        if (fs) fs.style.display = 'none';
      });
      tabs[idx].fieldsets.forEach(id => {
        const fs = document.getElementById(id);
        if (fs) fs.style.display = '';
      });
      tabBar.querySelectorAll('.cotton-cp-tab').forEach((btn, i) => {
        btn.classList.toggle('cotton-cp-tab--active', i === idx);
      });
    }
    tabs.forEach((t, i) => {
      const btn = document.createElement('button');
      btn.className = 'cotton-cp-tab';
      btn.type = 'button';
      btn.textContent = t.label;
      btn.addEventListener('click', () => setTab(i));
      tabBar.appendChild(btn);
    });
    setTab(0);

    // Persistent fake hand controls (visible on all tabs, above tab content)
    const fakeHandWrap = document.createElement('div');
    fakeHandWrap.className = 'cotton-cp-fake-hand-controls';
    fakeHandWrap.innerHTML = `
      <button id="fake-hand-btn" type="button" class="cotton-cp-button cotton-cp-button--secondary cotton-cp-button--block" style="margin-bottom:6px">Toggle Fake Hand</button>
      <div style="display:flex;gap:6px;align-items:center">
        <label style="display:flex;align-items:center;gap:6px;font-size:12px;color:var(--cp-fg);flex:1">
          <input type="checkbox" id="fake-hand-mirror" style="width:14px;height:14px;cursor:pointer"> Left hand
        </label>
        <button id="fake-hand-copy" type="button" class="cotton-cp-button cotton-cp-button--secondary" style="font-size:11px;padding:4px 8px">Copy Positions</button>
      </div>
      <label style="display:flex;align-items:center;gap:6px;font-size:12px;color:var(--cp-fg);margin-top:4px">
        <input type="checkbox" id="show-palm-center" style="width:14px;height:14px;cursor:pointer"> Show palm center
      </label>
    `;
    stackBody.insertBefore(fakeHandWrap, tabBar.nextSibling);
    document.getElementById('fake-hand-btn').addEventListener('click', toggleFakeHand);
    document.getElementById('fake-hand-copy').addEventListener('click', () => {
      if (!fakeFingerTips) return;
      const names = ['thumb', 'index', 'middle', 'ring', 'pinky'];
      const json = JSON.stringify(fakeFingerTips.map((ft, i) => ({ finger: names[i], x: Math.round(ft.x), y: Math.round(ft.y) })), null, 2);
      navigator.clipboard.writeText(json).then(() => {
        const btn = document.getElementById('fake-hand-copy');
        btn.textContent = 'Copied!';
        setTimeout(() => btn.textContent = 'Copy Positions', 1500);
      });
      console.log('Fake hand positions (ML space):', json);
    });
    document.getElementById('fake-hand-mirror').addEventListener('change', (e) => {
      fakeHandMirrored = e.target.checked;
      // Reset to defaults, flip if left hand, then re-center
      initFakeHand();
      if (fakeHandMirrored) {
        for (const ft of fakeFingerTips) ft.x = 640 - ft.x;
        // Re-center around ML midpoint (320)
        let cx = 0;
        for (const ft of fakeFingerTips) cx += ft.x;
        cx /= fakeFingerTips.length;
        const shift = 320 - cx;
        for (const ft of fakeFingerTips) ft.x += shift;
      }
    });
    document.getElementById('show-palm-center').addEventListener('change', (e) => {
      showPalmCenter = e.target.checked;
    });

    // Build the draggable timeline inside the Animation tab
    buildAnimTimeline();
  }
}

function mousePressed() {
  if (fakeHandActive) {
    fakeHandDragging = fakeHandHitTest(mouseX, mouseY);
    if (fakeHandDragging >= 0) return; // fake hand takes priority
  }
  // Background image interaction
  const hit = bgImageHitTest(mouseX, mouseY);
  if (hit) {
    if (hit.action === 'delete') {
      bgImages.splice(hit.index, 1);
    } else if (hit.action === 'drag') {
      bgDragging = hit.index;
      bgDragOffX = mouseX - bgImages[hit.index].x;
      bgDragOffY = mouseY - bgImages[hit.index].y;
    } else if (hit.action === 'resize') {
      bgResizing = hit.index;
      bgResizeStartW = bgImages[hit.index].w;
      bgResizeStartH = bgImages[hit.index].h;
      bgResizeStartMX = mouseX;
    }
  }
}

function mouseReleased() {
  fakeHandDragging = -1;
  bgDragging = -1;
  bgResizing = -1;
}

function mouseDragged() {
  if (bgDragging >= 0 && bgImages[bgDragging]) {
    bgImages[bgDragging].x = mouseX - bgDragOffX;
    bgImages[bgDragging].y = mouseY - bgDragOffY;
  }
  if (bgResizing >= 0 && bgImages[bgResizing]) {
    const bi = bgImages[bgResizing];
    const scaleDelta = (mouseX - bgResizeStartMX) / bgResizeStartW;
    const newW = Math.max(30, bgResizeStartW * (1 + scaleDelta));
    const aspect = bgResizeStartH / bgResizeStartW;
    bi.w = newW;
    bi.h = newW * aspect;
  }
}

function windowResized() {
  const { width: newWidth, height: newHeight } = calculateCanvasSize();
  
  // Preserve the current dots buffer content
  let oldBuffer = handsBuffer.get();
  
  resizeCanvas(newWidth, newHeight);
  
  // Create new buffer and copy old content, scaling it to fit
  handsBuffer = createGraphics(newWidth, newHeight);
  handsBuffer.image(oldBuffer, 0, 0, newWidth, newHeight);
  
  // Invalidate per-hand buffers so they get recreated at the new size
  for (let hi = 0; hi < 2; hi++) {
    if (perHandBuffers[hi]) { perHandBuffers[hi].remove(); perHandBuffers[hi] = null; }
  }

  // Update Paper.js hidden canvas and view size
  if (paper.view && paper.view.element) {
    paper.view.element.width = newWidth;
    paper.view.element.height = newHeight;
    paper.view.viewSize = new paper.Size(newWidth, newHeight);
  }
}

function draw() {
  drawBackgroundLayer();
  if (bgImages.length > 0) drawBgImages();
	const _m0 = getHandFillMode(0), _m1 = getHandFillMode(1);
	const _isPaletteMode = (c) => c === 'brand' || c === 'standardized' || (typeof c === 'string' && c.startsWith('#') && !c.startsWith('texture:'));
	// Use MULTIPLY when fills are palette colors (overlap colors are meaningful)
	// Skip for texture fills, white, or non-color modes
	const _useMultiply = showIntersections && _isPaletteMode(_m0) && _isPaletteMode(_m1);
	handsBuffer.blendMode(_useMultiply ? MULTIPLY : BLEND);
  
  // Hand detection — fake hand substitutes ML with draggable landmarks
  if (fakeHandActive) {
    // Handle drag
    if (fakeHandDragging >= 0 && mouseIsPressed && fakeFingerTips) {
      const ml = screenToML(mouseX, mouseY);
      fakeFingerTips[fakeHandDragging].x = ml.x;
      fakeFingerTips[fakeHandDragging].y = ml.y;
    }
    // Inject fake landmarks into the normal pipeline (same as ML detection)
    const fakeHands = buildFakeLandmarks();
    if (fakeHands) {
      hands = fakeHands;
      addToHandBuffer(hands);
    }
  } else if (detector && video && video.loadedmetadata && !isDetecting) {
      isDetecting = true;
      const vel = video.elt;
      mlInputCtx.drawImage(vel, 0, 0, ML_INPUT_W, ML_INPUT_H);
      // Run detection asynchronously without blocking draw loop
      detector.estimateHands(mlInputCanvas, {
        flipHorizontal: false
      }).then(predictions => {
        // Convert TensorFlow.js MediaPipe Hands format to ml5-like format
        const newHands = predictions.map(prediction => {
          // MediaPipe Hands returns keypoints as an array of {x, y, z, name}
          // We need to convert to array of [x, y, z] for 21 landmarks
          const landmarks = [];
          const keypoints = prediction.keypoints;
          
          // MediaPipe Hands has 21 keypoints in standard order
          // The keypoints array should already be in the correct order (0-20)
          if (keypoints && keypoints.length >= 21) {
            for (let i = 0; i < 21; i++) {
              const kp = keypoints[i];
              landmarks.push([
                kp.x,
                kp.y,
                kp.z || 0
              ]);
            }
          } else {
            // Fallback: create empty landmarks if format is unexpected
            for (let i = 0; i < 21; i++) {
              landmarks.push([0, 0, 0]);
            }
          }
          
          return {
            landmarks: landmarks,
            handInViewConfidence: prediction.score || 1.0
          };
        });
        
        // If we had skipped frames, replace them with interpolated values
        if (skippedFramesCount > 0) {
          const interpolatedFrames = [];
          
          if (lastDetectedHands.length > 0 && newHands.length > 0) {
            // Interpolate between last detection and new detection
            for (let i = 1; i <= skippedFramesCount; i++) {
              const t = i / (skippedFramesCount + 1); // Interpolation factor
              const interpolatedHands = interpolateHands(lastDetectedHands, newHands, t);
              interpolatedFrames.push(interpolatedHands);
            }
          } else if (lastDetectedHands.length > 0) {
            // New detection has no hands, fill with last detected hands
            for (let i = 1; i <= skippedFramesCount; i++) {
              interpolatedFrames.push(JSON.parse(JSON.stringify(lastDetectedHands)));
            }
          } else if (newHands.length > 0) {
            // No previous detection, fill with new hands
            for (let i = 1; i <= skippedFramesCount; i++) {
              interpolatedFrames.push(JSON.parse(JSON.stringify(newHands)));
            }
          } else {
            // Both empty, fill with empty
            for (let i = 1; i <= skippedFramesCount; i++) {
              interpolatedFrames.push([]);
            }
          }
          
          // Replace the last skippedFramesCount frames with interpolated ones
          replaceLastFramesInBuffer(skippedFramesCount, interpolatedFrames);
        }
        
        // Add the new detection to buffer
        addToHandBuffer(newHands);
        
        hands = newHands;
        lastDetectedHands = JSON.parse(JSON.stringify(newHands));
        skippedFramesCount = 0;
        isDetecting = false;
      }).catch(error => {
        console.error("Error detecting hands:", error);
        // On error, reuse previous hands data
        skippedFramesCount++;
        addToHandBuffer(hands);
        isDetecting = false;
      });
  }
  
  // Determine if hands need separate buffers (different textures)
  const tex0 = getTextureForHand(0), tex1 = getTextureForHand(1);
  const needsSplitBuffers = handFillSplitHands && (tex0 || tex1);

  // Clear the buffer each frame
  handsBuffer.clear();

  if (needsSplitBuffers) {
    // Ensure per-hand buffers exist and are the right size
    for (let hi = 0; hi < 2; hi++) {
      if (!perHandBuffers[hi] || perHandBuffers[hi].width !== width || perHandBuffers[hi].height !== height) {
        if (perHandBuffers[hi]) perHandBuffers[hi].remove();
        perHandBuffers[hi] = createGraphics(width, height);
      }
      perHandBuffers[hi].clear();
      perHandBuffers[hi].blendMode(_useMultiply ? MULTIPLY : BLEND);
    }
    handDrawTargets = perHandBuffers;
  } else {
    handDrawTargets = null;
  }

  // Draw lines at each fingertip on the graphics buffer (only if enabled)
  if (enableDrawHands) {
    drawHands();
  }
  handDrawTargets = null; // reset after drawing

  // Replace multiply-blended overlap pixels with exact brand overlap colors
  // Only run when hands are visible and multiply blend is active
  if (showIntersections && _useMultiply && hands.length > 0) {
    if (needsSplitBuffers) {
      for (let hi = 0; hi < 2; hi++) {
        if (perHandBuffers[hi]) fixOverlapColors(perHandBuffers[hi]);
      }
    } else {
      fixOverlapColors(handsBuffer);
    }
  }

  // Helper: resolve drawable source + dimensions from a texture entry
  function texSrc(t) {
    if (!t) return null;
    if (t.type === 'video' && t.video && t.video.videoWidth > 0) {
      return { src: t.video, w: t.video.videoWidth, h: t.video.videoHeight };
    }
    if (t.img && t.img.width > 0) {
      return { src: t.img.canvas || t.img.elt || t.img, w: t.img.width, h: t.img.height };
    }
    return null;
  }

  // Helper: draw texture cover-fitted into a region on a canvas context
  function drawTexCover(ctx, ts, rx, ry, rw, rh) {
    const s = Math.max(rw / ts.w, rh / ts.h);
    ctx.drawImage(ts.src, rx + (rw - ts.w * s) / 2, ry + (rh - ts.h * s) / 2, ts.w * s, ts.h * s);
  }

  if (needsSplitBuffers) {
    // Each hand was drawn to its own buffer — composite texture on each, then merge
    for (let hi = 0; hi < 2; hi++) {
      const buf = perHandBuffers[hi];
      const tex = getTextureForHand(hi);
      if (tex) {
        const ts = texSrc(tex);
        if (ts) {
          const ctx = buf.drawingContext;
          ctx.save();
          ctx.globalCompositeOperation = 'source-atop';
          if (handFillScaleToHand && handBoundingBoxes[hi]) {
            const bb = handBoundingBoxes[hi];
            drawTexCover(ctx, ts, bb.x, bb.y, bb.w, bb.h);
          } else {
            drawTexCover(ctx, ts, 0, 0, buf.width, buf.height);
          }
          ctx.restore();
        }
      }
      // Merge this hand's buffer onto the main handsBuffer
      handsBuffer.image(buf, 0, 0);
    }
  } else {
    // Single buffer path — composite shared texture if needed
    const anyTexture = handNeedsTexture(0) || handNeedsTexture(1);
    if (anyTexture) {
      const ctx = handsBuffer.drawingContext;
      ctx.save();
      ctx.globalCompositeOperation = 'source-atop';
      const tex = tex0 || tex1;
      const ts = texSrc(tex);
      if (ts) {
        if (handFillScaleToHand && handBoundingBoxes.length > 0) {
          // Fit to combined bounding box of all hands
          let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
          for (let hi = 0; hi < handBoundingBoxes.length; hi++) {
            const bb = handBoundingBoxes[hi];
            if (!bb) continue;
            minX = Math.min(minX, bb.x); minY = Math.min(minY, bb.y);
            maxX = Math.max(maxX, bb.x + bb.w); maxY = Math.max(maxY, bb.y + bb.h);
          }
          if (minX < Infinity) drawTexCover(ctx, ts, minX, minY, maxX - minX, maxY - minY);
        } else {
          drawTexCover(ctx, ts, 0, 0, handsBuffer.width, handsBuffer.height);
        }
      }
      ctx.restore();
    }
  }

  // Display the dots buffer on the main canvas, mirrored to match the video
  push();
  translate(width, 0);
  scale(-1, 1);
  image(handsBuffer, 0, 0);
  pop();

  // Draw palm center debug dots
  if (showPalmCenter) {
    const layout = getHandTrackingLayout();
    const rawHands = getBufferedHands();
    // Also get the normalized hands for comparison
    const normResult = calculateNormalizeOffset(rawHands, layout);
    const normHands = normResult.hands;

    for (let i = 0; i < rawHands.length; i++) {
      const lm = rawHands[i].landmarks;
      if (!lm || lm.length < 21) continue;

      // Raw palm center (red "R")
      const rwx = lm[0][0], rwy = lm[0][1];
      const rmx = lm[9][0], rmy = lm[9][1];
      const rpcX = width - (layout.ox + ((rwx + rmx) / 2) * layout.sx);
      const rpcY = layout.oy + ((rwy + rmy) / 2) * layout.sy;
      push();
      fill(255, 0, 0); noStroke();
      ellipse(rpcX, rpcY, 12, 12);
      fill(255); textAlign(CENTER, CENTER); textSize(9);
      text('R', rpcX, rpcY);
      pop();

      // Normalized palm center (blue "N")
      if (normHands[i]) {
        const nlm = normHands[i].landmarks;
        const nwx = nlm[0][0], nwy = nlm[0][1];
        const nmx = nlm[9][0], nmy = nlm[9][1];
        const npcX = width - (layout.ox + ((nwx + nmx) / 2) * layout.sx);
        const npcY = layout.oy + ((nwy + nmy) / 2) * layout.sy;
        push();
        fill(0, 100, 255); noStroke();
        ellipse(npcX, npcY, 12, 12);
        fill(255); textAlign(CENTER, CENTER); textSize(9);
        text('N', npcX, npcY);
        pop();
      }
    }
  }

  // Animation preview overlay
  if (animPreview && animKeyframes.length > 0) {
    const loopT = drawAnimPreview();
    updateAnimPlayhead(loopT);
  }

  // Draw debug information if enabled
  if (debugHandDetection) {
    drawDebugInfo();
    drawRawScaleDebug();
    drawFrameBufferDebug();
  }
  
  // Draw raw hand data if enabled
  if (showRawHandData) {
    drawRawHandData();
  }
  
  // Draw framerate graph only in debug mode
  if (debugMode) {
    drawFramerate();
  }
  
  updateCalibrationOverlay();

  // Handle save countdown (for PNG, SVG, or Record)
  if (saveCountdown >= 0) {
    const elapsed = (millis() - saveCountdownStartTime) / 1000;
    const remaining = Math.ceil(saveCountdownDelay - elapsed);

    if (remaining > 0) {
      push();
      fill(0);
      noStroke();
      textAlign(LEFT, BOTTOM);
      textSize(200);
      text(remaining.toString(), 20, height - 20);
      pop();
      saveCountdown = remaining;
    } else {
      saveCountdown = -1;
      if (savePendingAction === 'png') saveCroppedPNG();
      else if (savePendingAction === 'svg') exportSVG();
      else if (savePendingAction === 'record') startRecording();
      savePendingAction = null;
    }
  }

  // Handle active recording — show indicator and auto-stop after duration
  if (isRecording) {
    const recElapsed = (millis() - recordStartTime) / 1000;
    const recRemaining = Math.max(0, recordDuration - recElapsed);
    // Red recording dot + time remaining
    push();
    fill(255, 0, 0);
    noStroke();
    ellipse(30, 30, 18, 18);
    fill(255);
    textAlign(LEFT, CENTER);
    textSize(16);
    text(recRemaining.toFixed(1) + 's', 46, 30);
    pop();
    if (recRemaining <= 0) stopRecording();
  }
}

function drawRawScaleDebug() {
  // Get buffered hands to find hand position
  let bufferedHands = getBufferedHands();
  
  if (bufferedHands.length === 0) {
    return; // No hand to position next to
  }
  
  const layout = getHandTrackingLayout();
  bufferedHands = calculateNormalizeOffset(bufferedHands, layout).hands;
  
  // Get wrist position (landmark 0) from first hand
  const landmarks = bufferedHands[0].landmarks;
  const wrist = landmarks[0];
  const rawScale = getRawHandScale(0); // Use first hand's raw scale
  
  const wristCanvas = mlToCanvasXY(wrist[0], wrist[1], layout);
  const wristX = width - wristCanvas.x;
  const wristY = wristCanvas.y;
  
  // Position text to the right of the wrist
  const offsetX = 30; // Offset to the right
  const x = wristX + offsetX;
  const y = wristY;
  
  // Draw background rectangle
  push();
  fill(0, 0, 0, 180);
  noStroke();
  textAlign(LEFT, CENTER);
  textSize(14);
  
  const label = "Raw Scale:";
  const value = rawScale.toFixed(1);
  
  // Calculate text width for background
  const textW = textWidth(label + " " + value);
  const textH = textAscent() + textDescent();
  const bgHeight = textH + 8;
  const bgWidth = textW + 16;
  
  rect(x - 8, y - bgHeight / 2, bgWidth, bgHeight, 4);
  
  // Draw text
  fill(255);
  text(label + " " + value, x, y);
  
  pop();
}

function drawFrameBufferDebug() {
  const info = frameBufferDebugInfo;
  
  // Position in top left with padding
  const padding = 20;
  const x = padding;
  let y = padding;
  
  // Draw background rectangle
  push();
  fill(0, 0, 0, 200);
  noStroke();
  textAlign(LEFT, TOP);
  textSize(12);
  
  // Build debug text
  const lines = [
    "Frame Buffer Debug:",
    `Buffer Size: ${info.bufferSize}`,
    `Target Index: ${info.targetIndex >= 0 ? info.targetIndex : 'N/A'}`,
    `Target Has Data: ${info.targetFrameHasData ? 'Yes' : 'No'}`,
    `Hands Returned: ${info.handsReturned}`
  ];
  
  // Add interpolation info if used
  if (info.interpolationUsed) {
    lines.push(`Interpolation: Yes (t=${info.interpolationFactor.toFixed(2)})`);
    if (info.beforeIndex >= 0) {
      lines.push(`Before Index: ${info.beforeIndex} (gap: ${info.gapBefore})`);
    }
    if (info.afterIndex >= 0) {
      lines.push(`After Index: ${info.afterIndex} (gap: ${info.gapAfter})`);
    }
  } else if (!info.targetFrameHasData && info.bufferSize >= handBufferFrames + 1) {
    // Missing frame but no interpolation
    if (info.beforeIndex >= 0) {
      lines.push(`Before Index: ${info.beforeIndex} (gap: ${info.gapBefore})`);
    } else {
      lines.push(`Before Index: None`);
    }
    if (info.afterIndex >= 0) {
      lines.push(`After Index: ${info.afterIndex} (gap: ${info.gapAfter})`);
    } else {
      lines.push(`After Index: None`);
    }
    if (info.beforeIndex < 0 && info.afterIndex < 0) {
      lines.push(`Status: No data found`);
    } else {
      lines.push(`Status: Gap too large or invalid`);
    }
  }
  
  // Calculate dimensions for background
  const textH = textAscent() + textDescent();
  const lineHeight = textH * 1.2;
  const bgHeight = lines.length * lineHeight + 16;
  let maxWidth = 0;
  for (let line of lines) {
    const w = textWidth(line);
    if (w > maxWidth) maxWidth = w;
  }
  const bgWidth = maxWidth + 20;
  
  // Draw background
  rect(x, y, bgWidth, bgHeight, 4);
  
  // Draw text
  fill(255);
  y += 12;
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    // Highlight important lines
    if (line.includes("Interpolation:") || line.includes("Status:")) {
      fill(255, 200, 0); // Yellow for important info
    } else if (line.includes("Target Has Data: No")) {
      fill(255, 100, 100); // Red for missing data
    } else {
      fill(255);
    }
    text(line, x + 10, y);
    y += lineHeight;
  }
  
  pop();
}

function drawRawHandData() {
  // Use the latest hands array (raw detection data, not buffered)
  const rawHands = hands;
  
  if (rawHands.length === 0) {
    return; // No hands to draw
  }
  
  const layout = getHandTrackingLayout();

  // Draw on main canvas, mirrored to match video
  push();
  translate(width, 0);
  scale(-1, 1);
  
  // Draw each hand's landmarks as points
  for (let i = 0; i < rawHands.length; i++) {
    const hand = rawHands[i];
    
    if (!hand.landmarks || hand.landmarks.length === 0) {
      continue;
    }
    
    // Use different colors for different hands
    const handColors = [
      [100, 255, 100], // Light green for hand 1
      [100, 200, 255]  // Light blue for hand 2
    ];
    const handColor = handColors[i % handColors.length];
    
    const pivot = bigHands ? handLandmarksCanvasCentroid(hand.landmarks, layout, 0, 0) : null;
    const rawDbgScale = bigHands ? BIG_HANDS_SCALE : 1;

    // Draw all landmarks as points
    for (let j = 0; j < hand.landmarks.length; j++) {
      const landmark = hand.landmarks[j];
      
      if (!landmark || landmark.length < 2) {
        continue;
      }
      
      let x = layout.ox + landmark[0] * layout.sx;
      let y = layout.oy + landmark[1] * layout.sy;
      if (rawDbgScale !== 1 && pivot) {
        const sp = scaleCanvasPointFromPivot(x, y, pivot.x, pivot.y, rawDbgScale);
        x = sp.x;
        y = sp.y;
      }
      
      // Draw point
      fill(handColor[0], handColor[1], handColor[2], 200);
      stroke(255, 255, 255, 150);
      strokeWeight(1);
      circle(x, y, 8);
      
      // Draw landmark index number
      push();
      fill(255, 255, 255, 255);
      noStroke();
      textAlign(CENTER, CENTER);
      textSize(8);
      text(j, x, y);
      pop();
    }
  }
  
  pop();
}

function drawFramerate() {
  const fps = frameRate();
  
  // Add current FPS to history
  fpsHistory.push(fps);
  if (fpsHistory.length > fpsHistoryMaxLength) {
    fpsHistory.shift(); // Remove oldest value
  }
  
  // Position in bottom right with padding
  const padding = 20;
  const graphWidth = 200;
  const graphHeight = 60;
  const graphX = width - graphWidth - padding;
  const graphY = height - graphHeight - padding;
  
  // Draw background rectangle
  push();
  fill(0, 0, 0, 200);
  noStroke();
  rect(graphX, graphY, graphWidth, graphHeight, 4);
  
  // Use fixed Y-axis range for consistent scaling
  const minFps = 0;
  const maxFps = 60; // Fixed max at 60 FPS
  const fpsRange = maxFps - minFps;
  
  // Draw grid lines
  stroke(255, 255, 255, 30);
  strokeWeight(1);
  const gridLines = 3;
  for (let i = 0; i <= gridLines; i++) {
    const y = graphY + (graphHeight / gridLines) * i;
    line(graphX, y, graphX + graphWidth, y);
  }
  
  // Draw FPS line graph
  if (fpsHistory.length > 1) {
    noFill();
    stroke(100, 255, 100); // Green color for FPS line
    strokeWeight(2);
    beginShape();
    
    for (let i = 0; i < fpsHistory.length; i++) {
      const fpsValue = Math.min(maxFps, Math.max(minFps, fpsHistory[i])); // Clamp to range
      const x = graphX + (graphWidth / (fpsHistory.length - 1)) * i;
      // Map FPS value to graph height (invert Y so higher FPS is at top)
      const normalizedFps = (fpsValue - minFps) / fpsRange;
      const y = graphY + graphHeight - (normalizedFps * graphHeight);
      vertex(x, y);
    }
    
    endShape();
    
    // Draw current FPS point
    fill(100, 255, 100);
    noStroke();
    const lastX = graphX + graphWidth;
    const clampedFps = Math.min(maxFps, Math.max(minFps, fps)); // Clamp to range
    const lastNormalizedFps = (clampedFps - minFps) / fpsRange;
    const lastY = graphY + graphHeight - (lastNormalizedFps * graphHeight);
    circle(lastX, lastY, 4);
  }
  
  // Draw FPS text label
  textAlign(RIGHT, BOTTOM);
  textSize(12);
  fill(255);
  noStroke();
  const label = "FPS:";
  const value = fps.toFixed(1);
  text(label + " " + value, graphX + graphWidth - 5, graphY + graphHeight - 5);
  
  // Draw min/max FPS labels (always show fixed range)
  textAlign(LEFT, TOP);
  textSize(9);
  fill(200, 200, 200, 150);
  text(minFps.toFixed(0), graphX + 5, graphY + graphHeight - 12);
  textAlign(LEFT, BOTTOM);
  text(maxFps.toFixed(0), graphX + 5, graphY + 12);
  
  pop();
}

function updateCalibrationOverlay() {
  const root = document.getElementById('calibration-overlay');
  if (!root) return;
  // No calibration step — always hide
  root.hidden = true;
  root.setAttribute('aria-hidden', 'true');
}

// Helper function to export Paper.js circle to p5
// Now draws a square that morphs into a circle during transition
function exportPaperCircleToP5(circleShape, colorObj, buffer, rectBezOrCircle, fullRadius, strokeWeight, pathShape, drawBaseX, drawBaseY, drawTipX, drawTipY) {
  const finalCenter = circleShape.position; // Circle center (at tip)
  const fullCircleDiameter = fullRadius * 2;
  
  // Calculate transition progress: 0 at 0.5 (square starts), 1 at 1.0 (circle complete)
  let transitionProgress = 0.0;
  if (rectBezOrCircle < 0.5) {
    // Shouldn't happen in circle state, but handle gracefully
    return;
  } else {
    transitionProgress = (rectBezOrCircle - 0.5) * 2; // Maps 0.5->0 to 1.0->1.0
  }
  
  // Calculate the line segment position and angle when rectBezOrCircle = 0.5
  // The line segment is the last strokeWeight length of the bezier path, from tip going backward
  let lineSegmentCenterX = drawTipX;
  let lineSegmentCenterY = drawTipY;
  let lineSegmentAngle = 0;
  
  if (pathShape && pathShape.length > 0) {
    try {
      const pathLength = pathShape.length;
      // At 0.5, the visible portion is strokeWeight length from the tip
      const offsetFromTip = Math.min(strokeWeight, pathLength);
      const startOffset = pathLength - offsetFromTip;
      
      // Get the point at the start of the line segment (going backward from tip)
      const startLocation = pathShape.getLocationAt(startOffset);
      const endLocation = pathShape.getLocationAt(pathLength); // Tip
      
      if (startLocation && startLocation.point && endLocation && endLocation.point) {
        // Calculate center of line segment
        lineSegmentCenterX = (startLocation.point.x + endLocation.point.x) / 2;
        lineSegmentCenterY = (startLocation.point.y + endLocation.point.y) / 2;

        // Calculate angle of line segment
        const dx = endLocation.point.x - startLocation.point.x;
        const dy = endLocation.point.y - startLocation.point.y;
        lineSegmentAngle = Math.atan2(dy, dx);
      } else {
        // getLocationAt returned null (common for exact endpoint) — fall back to base-to-tip angle
        const dx = drawTipX - drawBaseX;
        const dy = drawTipY - drawBaseY;
        const baseToTipLength = Math.sqrt(dx * dx + dy * dy);
        if (baseToTipLength > 0.001) {
          lineSegmentAngle = Math.atan2(dy, dx);
          lineSegmentCenterX = drawTipX;
          lineSegmentCenterY = drawTipY;
        }
      }
    } catch (e) {
      // Fallback: use tip position and calculate angle from base to tip
      const dx = drawTipX - drawBaseX;
      const dy = drawTipY - drawBaseY;
      const baseToTipLength = Math.sqrt(dx * dx + dy * dy);
      if (baseToTipLength > 0.001) {
        lineSegmentAngle = Math.atan2(dy, dx);
        // Line segment center is at tip (since it's just a small segment)
        lineSegmentCenterX = drawTipX;
        lineSegmentCenterY = drawTipY;
      }
    }
  } else {
    // Fallback: use tip position and calculate angle from base to tip
    const dx = drawTipX - drawBaseX;
    const dy = drawTipY - drawBaseY;
    const baseToTipLength = Math.sqrt(dx * dx + dy * dy);
    if (baseToTipLength > 0.001) {
      lineSegmentAngle = Math.atan2(dy, dx);
      lineSegmentCenterX = drawTipX;
      lineSegmentCenterY = drawTipY;
    }
  }
  
  // Interpolate properties based on transition progress
  // Border radius: starts at fullRadius/2 at 0.5 (from rectangle transition), goes to fullRadius at 1.0
  const startBorderRadius = fullRadius / 2; // Already at this value from rectangle transition
  const targetBorderRadius = fullRadius;
  const borderRadius = startBorderRadius + (targetBorderRadius - startBorderRadius) * transitionProgress;
  
  // Edge length: strokeWeight → fullCircleDiameter
  const edgeLength = strokeWeight + (fullCircleDiameter - strokeWeight) * transitionProgress;
  
  // Center: line segment center → circle center
  const centerX = lineSegmentCenterX + (finalCenter.x - lineSegmentCenterX) * transitionProgress;
  const centerY = lineSegmentCenterY + (finalCenter.y - lineSegmentCenterY) * transitionProgress;
  
  // Angle: line segment angle → 0 (or keep line segment angle, depending on preference)
  // For now, keep the line segment angle throughout the transition
  const angle = lineSegmentAngle;
  
  // Draw the square/rounded square
  buffer.fill(red(colorObj), green(colorObj), blue(colorObj));
  buffer.noStroke();
  
  buffer.push();
  buffer.translate(centerX, centerY);
  buffer.rotate(angle);
  
  // Draw rounded rectangle (square with border radius)
  // p5 rectMode defaults to CORNER, but we want CENTER
  buffer.rectMode(CENTER);
  buffer.rect(0, 0, edgeLength, edgeLength, borderRadius);
  buffer.rectMode(CORNER); // Reset to default
  buffer.pop();
}

// Helper function to calculate bezier control points for a finger
// Returns { cp1X, cp1Y, cp2X, cp2Y } - the control points for the bezier curve
function calculateBezierControlPoints(drawBaseX, drawBaseY, drawTipX, drawTipY, pipX, pipY) {
  let baseToPipX = pipX - drawBaseX;
  let baseToPipY = pipY - drawBaseY;
  let tipToPipX = pipX - drawTipX;
  let tipToPipY = pipY - drawTipY;
  
  // Control point 1: extends from base toward PIP, halfway, then moved 20% closer to tip
  let cp1X = drawBaseX + baseToPipX * .7;
  let cp1Y = drawBaseY + baseToPipY * .7;
  // Move 20% closer to tip
  // cp1X = cp1X * 0.8 + drawTipX * 0.2;
  // cp1Y = cp1Y * 0.8 + drawTipY * 0.2;
  
  // Control point 2: extends from tip toward PIP, halfway
  let cp2X = drawTipX + tipToPipX * .6;
  let cp2Y = drawTipY + tipToPipY * .6;
  
  // Rotate control points 10 degrees away from the opposite end of the finger
  // Calculate base-to-tip vector for reference
  let baseToTipX = drawTipX - drawBaseX;
  let baseToTipY = drawTipY - drawBaseY;
  let baseToTipLength = Math.sqrt(baseToTipX * baseToTipX + baseToTipY * baseToTipY);
  
  if (baseToTipLength > 0.001) {
    // Normalize base-to-tip vector
    let baseToTipNormX = baseToTipX / baseToTipLength;
    let baseToTipNormY = baseToTipY / baseToTipLength;
    
    // Perpendicular vector (rotated 90 degrees counterclockwise)
    let perpX = -baseToTipNormY;
    let perpY = baseToTipNormX;
    
    // For cp1: rotate around base anchor point, away from tip
    // Calculate vector from base to cp1
    let cp1VecX = cp1X - drawBaseX;
    let cp1VecY = cp1Y - drawBaseY;
    let cp1Length = Math.sqrt(cp1VecX * cp1VecX + cp1VecY * cp1VecY);
    
    if (cp1Length > 0.001) {
      // Get current angle of cp1 relative to base
      let cp1Angle = atan2(cp1VecY, cp1VecX);
      
      // Get angle of base-to-tip line
      let baseToTipAngle = atan2(baseToTipY, baseToTipX);
      
      // Determine which side of the base-tip line cp1 is on
      let cp1Perp = cp1VecX * perpX + cp1VecY * perpY;
      let rotationDir1 = cp1Perp >= 0 ? 1 : -1;
      
      // Rotate cp1 around base by the rotation angle in the direction away from tip
      let rotationAngle1 = 30 * Math.PI / 180;
      let cp1NewAngle = cp1Angle + rotationDir1 * rotationAngle1;
      
      // Apply rotated angle with same length
      cp1X = drawBaseX + Math.cos(cp1NewAngle) * cp1Length;
      cp1Y = drawBaseY + Math.sin(cp1NewAngle) * cp1Length;
    }
    
    // For cp2: rotate around tip anchor point, away from base
    // Calculate vector from tip to cp2
    let cp2VecX = cp2X - drawTipX;
    let cp2VecY = cp2Y - drawTipY;
    let cp2Length = Math.sqrt(cp2VecX * cp2VecX + cp2VecY * cp2VecY);
    
    if (cp2Length > 0.001) {
      // Get current angle of cp2 relative to tip
      let cp2Angle = atan2(cp2VecY, cp2VecX);
      
      // Determine which side of the base-tip line cp2 is on
      let cp2Perp = cp2VecX * perpX + cp2VecY * perpY;
      // Flip direction for cp2 so it rotates away from base (opposite of cp1)
      let rotationDir2 = cp2Perp >= 0 ? -1 : 1;
      
      // Rotate cp2 around tip by the rotation angle in the direction away from base
      let rotationAngle2 = 30 * Math.PI / 180;
      let cp2NewAngle = cp2Angle + rotationDir2 * rotationAngle2;
      
      // Apply rotated angle with same length
      cp2X = drawTipX + Math.cos(cp2NewAngle) * cp2Length;
      cp2Y = drawTipY + Math.sin(cp2NewAngle) * cp2Length;
    }
  }
  
  return { cp1X, cp1Y, cp2X, cp2Y };
}

// Helper function to export transition rectangle (0.3 to 0.5)
// Draws a rectangle that undraws and increases border radius
function exportTransitionRectangleToP5(colorObj, buffer, rectBezOrCircle, strokeWeight, drawBaseX, drawBaseY, drawTipX, drawTipY, fullRadius) {
  // Calculate transition progress: 0 at 0.3 (rectangle starts), 1 at 0.5 (square ready)
  const transitionProgress = (rectBezOrCircle - 0.3) / (0.5 - 0.3); // Maps 0.3->0 to 0.5->1
  
  // Calculate rectangle properties
  const dx = drawTipX - drawBaseX;
  const dy = drawTipY - drawBaseY;
  const baseToTipLength = Math.sqrt(dx * dx + dy * dy);
  
  if (baseToTipLength < 0.001) return;
  
  // Calculate angle of the line
  const angle = Math.atan2(dy, dx);
  
  // Calculate visible length: starts at 0.7 of full length at 0.3 (matching bezier's final state), undraws to stroke width at 0.5
  // The rectangle should match the bezier's final visual state (straight line at 0.7 of full length)
  const fullLength = baseToTipLength;
  const startLength = fullLength * 0.7; // Start at 70% of full length
  const visibleLength = startLength - (startLength - strokeWeight) * transitionProgress;
  
  // Border radius: starts at 0 at 0.3, increases toward fullRadius/2 at 0.5
  // But we want it to continue smoothly into the square transition
  const borderRadiusAt05 = fullRadius / 2; // Half radius at 0.5 (will continue in square)
  // Ensure border radius starts at exactly 0 when transitionProgress = 0
  const borderRadius = Math.max(0, borderRadiusAt05 * transitionProgress);
  
  // Calculate center position (midpoint of the stroke-width segment at the tip)
  // The rectangle should match the bezier's final state: a straight line segment at the tip
  const tipToBaseDirX = dx / baseToTipLength;
  const tipToBaseDirY = dy / baseToTipLength;
  // Segment goes from tip backward by strokeWeight
  const segmentStartX = drawTipX - visibleLength * tipToBaseDirX;
  const segmentStartY = drawTipY - visibleLength * tipToBaseDirY;
  // Center is midpoint of this segment
  const centerX = (segmentStartX + drawTipX) / 2;
  const centerY = (segmentStartY + drawTipY) / 2;
  
  // Draw the rounded rectangle
  buffer.fill(red(colorObj), green(colorObj), blue(colorObj));
  buffer.noStroke();
  
  buffer.push();
  buffer.translate(centerX, centerY);
  buffer.rotate(angle);
  buffer.rectMode(CENTER);
  buffer.rect(0, 0, visibleLength, strokeWeight, borderRadius);
  buffer.rectMode(CORNER);
  buffer.pop();
}

// Helper function to export Paper.js bezier path to p5
function exportPaperBezierToP5(pathShape, colorObj, strokeWeight, buffer, rectBezOrCircle, drawBaseX, drawBaseY, drawTipX, drawTipY) {
  const segments = pathShape.segments;
  if (segments.length < 2) return;
  
  // If rectBezOrCircle >= 0.3, we should be drawing a rectangle instead
  // This function should only handle bezier drawing when < 0.3
  if (rectBezOrCircle >= 0.3) {
    return; // Rectangle will be drawn instead
  }
  
  // Bezier undraws to 0.7 (70%) of its length as it approaches 0.3
  // At 0.3, it should match the rectangle's starting state (0.7 of full length, straight)
  let visiblePortion = 1.0;
  if (rectBezOrCircle !== undefined && rectBezOrCircle > 0) {
    // At 0: fully visible (1.0), at 0.3: 0.7 of length
    // Interpolate from 1.0 to 0.7 as rectBezOrCircle goes from 0 to 0.3
    visiblePortion = 1.0 - (rectBezOrCircle / 0.3) * (1.0 - 0.7);
  }
  
  // Calculate straightening factor: 0 = curved, 1 = perfectly straight (at 0.3)
  const straighteningFactor = Math.min(1.0, rectBezOrCircle / 0.3);
  
  buffer.noFill();
  buffer.strokeCap(SQUARE);
  buffer.stroke(red(colorObj), green(colorObj), blue(colorObj));
  buffer.strokeWeight(strokeWeight);
  
  // If fully visible, draw normally
  if (visiblePortion >= 0.999) {
    // Convert Paper.js path segments to p5 bezier curves
    for (let i = 0; i < segments.length - 1; i++) {
      const seg1 = segments[i];
      const seg2 = segments[i + 1];
      
      const x1 = seg1.point.x;
      const y1 = seg1.point.y;
      const x4 = seg2.point.x;
      const y4 = seg2.point.y;
      
      // Original control points
      const origX2 = seg1.point.x + seg1.handleOut.x;
      const origY2 = seg1.point.y + seg1.handleOut.y;
      const origX3 = seg2.point.x + seg2.handleIn.x;
      const origY3 = seg2.point.y + seg2.handleIn.y;
      
      // Straightened control points: move to 1/3 along the line from their anchor point
      // Control point 1 (attached to base): 1/3 of the way from base toward tip
      const straightX2 = x1 + (x4 - x1) / 3;
      const straightY2 = y1 + (y4 - y1) / 3;
      // Control point 2 (attached to tip): 1/3 of the way from tip toward base (2/3 from base)
      const straightX3 = x1 + (x4 - x1) * (2 / 3);
      const straightY3 = y1 + (y4 - y1) * (2 / 3);
      
      // Interpolate between original and straight control points
      const x2 = lerp(origX2, straightX2, straighteningFactor);
      const y2 = lerp(origY2, straightY2, straighteningFactor);
      const x3 = lerp(origX3, straightX3, straighteningFactor);
      const y3 = lerp(origY3, straightY3, straighteningFactor);
      
      buffer.bezier(x1, y1, x2, y2, x3, y3, x4, y4);
    }
  } else {
    // Partial reveal: sample points along path and draw only visible portion
    // Reveal/unreveal happens from tip (end) toward base (start)
    // So we show the portion from tip backward toward base
    try {
      const pathLength = pathShape.length;
      // Ensure targetLength is at least stroke width (the bezier should undraw to its width, not 0)
      const targetLength = Math.max(strokeWeight, pathLength * visiblePortion);
      
      // Sample points along the path from tip (pathLength) backward to the reveal point
      // The visible portion goes from tip toward base
      const numSamples = Math.max(20, Math.ceil(pathLength * 2)); // At least 20 samples, more for longer paths
      const points = [];
      
      for (let i = 0; i <= numSamples; i++) {
        // Sample from tip backward: start at pathLength, go back by targetLength
        const offset = pathLength - (i / numSamples) * targetLength;
        if (offset < pathLength - targetLength) break;
        
        try {
          // Sample from tip (pathLength) backward toward base
          // This shows the portion from tip, hiding the rest from base
          const location = pathShape.getLocationAt(offset);
          if (location && location.point) {
            points.push({ x: location.point.x, y: location.point.y });
          }
        } catch (e) {
          // Skip invalid locations
          continue;
        }
      }
      
      // Reverse points so we draw from base to tip (natural drawing order)
      points.reverse();
      
      // Draw the visible portion as a smooth curve
      if (points.length >= 2) {
        buffer.beginShape();
        buffer.vertex(points[0].x, points[0].y);
        // Use curveVertex for smoother curves (needs at least 4 points)
        if (points.length >= 4) {
          for (let i = 1; i < points.length - 2; i++) {
            buffer.curveVertex(points[i].x, points[i].y);
          }
          // Add last point as regular vertex
          buffer.vertex(points[points.length - 1].x, points[points.length - 1].y);
        } else {
          // For fewer points, use regular vertices
          for (let i = 1; i < points.length; i++) {
            buffer.vertex(points[i].x, points[i].y);
          }
        }
        buffer.endShape();
      }
    } catch (e) {
      // Fallback: draw full path if sampling fails
      for (let i = 0; i < segments.length - 1; i++) {
        const seg1 = segments[i];
        const seg2 = segments[i + 1];
        
        const x1 = seg1.point.x;
        const y1 = seg1.point.y;
        const x2 = seg1.point.x + seg1.handleOut.x;
        const y2 = seg1.point.y + seg1.handleOut.y;
        const x3 = seg2.point.x + seg2.handleIn.x;
        const y3 = seg2.point.y + seg2.handleIn.y;
        const x4 = seg2.point.x;
        const y4 = seg2.point.y;
        
        buffer.bezier(x1, y1, x2, y2, x3, y3, x4, y4);
      }
    }
  }
}

// Helper function to convert a bezier path with stroke to a filled outline shape
function bezierPathToOutline(path, strokeWidth, lowRes) {
  if (!path || !path.segments || path.segments.length < 2) return null;

  // Create an outline by offsetting the path on both sides
  // We'll create a closed shape that represents the stroke as a filled area

  const halfWidth = strokeWidth / 2;
  const pathLength = path.length;
  // Low-res mode for intersection computation (much faster boolean ops)
  const samplesPerUnit = lowRes ? 0.15 : 1;
  const numSamples = Math.max(lowRes ? 12 : 50, Math.ceil(pathLength * samplesPerUnit));
  
  // Sample points along the path with their normals
  const points = [];
  const normals = [];
  
  for (let i = 0; i <= numSamples; i++) {
    const offset = i / numSamples;
    try {
      const location = path.getLocationAt(offset * path.length);
      if (location && location.point) {
        const point = location.point;
        let tangent = location.tangent;
        
        // Normalize tangent
        if (tangent && tangent.length > 0) {
          tangent = tangent.normalize();
          // Normal is perpendicular to tangent (rotate 90 degrees counterclockwise)
          const normal = new paper.Point(-tangent.y, tangent.x);
          points.push(point);
          normals.push(normal);
        }
      }
    } catch (e) {
      // Skip invalid locations
      continue;
    }
  }
  
  // If we didn't get enough points, return null
  if (points.length < 2) return null;
  
  // Create outline path
  const outlinePath = new paper.Path();
  
  // Top side (offset in normal direction) - forward along path
  for (let i = 0; i < points.length; i++) {
    const offsetPoint = points[i].add(normals[i].multiply(halfWidth));
    outlinePath.add(new paper.Segment(offsetPoint));
  }
  
  // Bottom side (offset in opposite normal direction) - backward along path
  for (let i = points.length - 1; i >= 0; i--) {
    const offsetPoint = points[i].add(normals[i].multiply(-halfWidth));
    outlinePath.add(new paper.Segment(offsetPoint));
  }
  
  outlinePath.closed = true;
  return outlinePath;
}

// Helper function to convert Paper.js shape to path for boolean operations
function shapeToPath(shape, strokeWidth, lowRes) {
  if (!shape) return null;

  // If it's a bezier path, convert it to a filled outline first
  if (shape instanceof paper.Path && strokeWidth) {
    const outline = bezierPathToOutline(shape, strokeWidth, lowRes);
    if (outline) return outline;
    // If outline creation failed, fall through to return the path as-is
  }
  
  // If already a Path (and not a bezier that needs conversion), return it
  if (shape instanceof paper.Path) {
    return shape;
  }
  
  // If it's a Shape (Circle, Rectangle), convert to Path
  if (shape instanceof paper.Shape && typeof shape.toPath === 'function') {
    return shape.toPath();
  }
  
  // Try to get path property if it exists
  if (shape.path && shape.path instanceof paper.Path) {
    return shape.path;
  }
  
  return null;
}

// Helper function to calculate normalization offset (layout from getHandTrackingLayout)
function calculateNormalizeOffset(bufferedHands, layout) {
  if (!normalizeHands || bufferedHands.length === 0) {
    return { offsetX: 0, offsetY: 0, hands: bufferedHands };
  }
  
  // Create a copy of hands with modified landmarks for normalization
  const normalizedHands = [];
  
  // For 2 hands, determine left vs right based on X position
  // Since video is mirrored when displayed: higher X in video space = right hand (appears on right after mirroring)
  let handPositions = [];
  if (bufferedHands.length === 2) {
    for (let i = 0; i < bufferedHands.length; i++) {
      const hand = bufferedHands[i];
      const landmarks = hand.landmarks;
      
      // Calculate average X position
      let sumX = 0;
      for (let j = 0; j < landmarks.length; j++) {
        sumX += landmarks[j][0];
      }
      const avgX = sumX / landmarks.length;
      
      handPositions.push({ index: i, avgX: avgX });
    }
    
    // Sort by X position: higher X = right hand, lower X = left hand
    handPositions.sort((a, b) => b.avgX - a.avgX);
  }
  
  for (let i = 0; i < bufferedHands.length; i++) {
    const hand = bufferedHands[i];
    const landmarks = hand.landmarks;
    
    // Use palm center (midpoint of wrist and middle MCP) as the hand's center point.
    // This is more stable than averaging all 21 landmarks (which shifts when fingers move).
    const wrist = landmarks[0];
    const middleMCP = landmarks[9];
    const avgX = (wrist[0] + middleMCP[0]) / 2;
    const avgY = (wrist[1] + middleMCP[1]) / 2;
    
    // Use EMA-smoothed scale to prevent tip jitter from ML noise amplification.
    // Falls back to instantaneous scale if smoothed isn't available yet.
    const smoothed = getSmoothedRawHandScale(i);
    const currentRawScale = smoothed > 0 ? smoothed : computeHandScale(landmarks, i);

    const targetRawScale = 100;
    let scaleFactor = 1.0;
    if (currentRawScale > 0) {
      scaleFactor = targetRawScale / currentRawScale;
    }
    
    // First, scale landmarks around the center point to normalize size
    const scaledLandmarks = landmarks.map(landmark => {
      // Translate to center, scale, then translate back
      const relativeX = landmark[0] - avgX;
      const relativeY = landmark[1] - avgY;
      return [
        avgX + relativeX * scaleFactor,
        avgY + relativeY * scaleFactor,
        landmark[2] || 0
      ];
    });
    
    // Palm center after scaling (same point since we scaled around it)
    const scaledAvgX = avgX;
    const scaledAvgY = avgY;
    
    // Convert to canvas coordinates (same mapping as video cover + ML stretch)
    const avgCanvasX = layout.ox + scaledAvgX * layout.sx;
    const avgCanvasY = layout.oy + scaledAvgY * layout.sy;
    
    // Determine target position based on number of hands
    let targetX, targetY;
    
    if (bufferedHands.length === 1) {
      // Single hand: center it
      targetX = width / 2;
      targetY = height / 2;
    } else {
      // Two hands: vertically centered, horizontally at 1/3 and 2/3
      targetY = height / 2; // Both vertically centered
      
      // Find this hand's position in the sorted array
      const positionIndex = handPositions.findIndex(p => p.index === i);
      if (positionIndex === 0) {
        // Right hand (higher X in video space) goes to right side (2/3 width)
        targetX = (width * 2) / 3;
      } else {
        // Left hand (lower X in video space) goes to left side (1/3 width)
        targetX = width / 3;
      }
    }
    
    // Calculate offset for this hand
    const offsetX = targetX - avgCanvasX;
    const offsetY = targetY - avgCanvasY;
    
    // Offset in ML space: canvas delta ÷ (pixels per ML unit)
    const dMlx = (offsetX * ML_INPUT_W) / (layout.s * layout.iw);
    const dMly = (offsetY * ML_INPUT_H) / (layout.s * layout.ih);
    const normalizedLandmarks = scaledLandmarks.map(landmark => [
      landmark[0] + dMlx,
      landmark[1] + dMly,
      landmark[2] || 0
    ]);
    
    // Create normalized hand object
    normalizedHands.push({
      landmarks: normalizedLandmarks,
      handInViewConfidence: hand.handInViewConfidence
    });
  }
  
  // Return zero offsets since we've already applied them to the landmarks
  return { offsetX: 0, offsetY: 0, hands: normalizedHands };
}

/** Get the active fill mode for a given hand index. */
function getHandFillMode(handIndex) {
  if (!handFillSplitHands) return handFillSelection[0];
  return handFillSelection[handIndex] || handFillSelection[0];
}

/** Resolve the effective finger fill color based on per-hand selection. */
function resolveFingerFillColor(paletteColor, fingerName, handIndex) {
  const mode = getHandFillMode(handIndex);
  if (mode === 'brand') return paletteColor;
  if (mode === 'standardized') return STANDARDIZED_FINGER_COLORS[fingerName] || paletteColor;
  if (mode === '__custom__') return handFillCustomColor;
  if (typeof mode === 'string' && mode.startsWith('texture:')) return '#FFFFFF';
  return mode; // hex color
}

/** Get the texture object for a given hand index, or null. */
function getTextureForHand(handIndex) {
  const mode = getHandFillMode(handIndex);
  if (typeof mode !== 'string' || !mode.startsWith('texture:')) return null;
  const idx = parseInt(mode.split(':')[1]);
  return handFillTextures[idx] || null;
}

/** Check if a hand's fill mode needs texture compositing. */
function handNeedsTexture(handIndex) {
  return getTextureForHand(handIndex) !== null;
}

function drawHands() {
  handBoundingBoxes = [];
  const layout = getHandTrackingLayout();
  const mlCanvasScale = (layout.sx + layout.sy) * 0.5;

  // Get buffered hands (delayed by handBufferFrames)
  let bufferedHands = getBufferedHands();

  // Normalize hand positions
  const normalizeResult = calculateNormalizeOffset(bufferedHands, layout);
  const normalizeOffsetX = normalizeResult.offsetX;
  const normalizeOffsetY = normalizeResult.offsetY;
  bufferedHands = normalizeResult.hands;

  // Update hand closeness system for each detected hand
  // In fake hand mode, only run once to establish initial scale, then freeze
  if (!fakeHandActive || !handClosenessState[0] || !handClosenessState[0].smoothedRawScale) {
    for (let i = 0; i < bufferedHands.length; i++) {
      updateHandCloseness(bufferedHands[i].landmarks, i);
    }
  }
  
  // Decay closeness for hands that are no longer detected
  for (let handIndex in handClosenessState) {
    const handIdx = parseInt(handIndex);
    if (handIdx >= bufferedHands.length) {
      updateHandClosenessNoHand(handIdx);
    }
  }
  
  for (let i = 0; i < bufferedHands.length; i++) {
    let hand = bufferedHands[i];
    let landmarks = hand.landmarks;
    // Resolve draw target for this hand (per-hand buffer or shared handsBuffer)
    const hBuf = (handDrawTargets && handDrawTargets[i]) ? handDrawTargets[i] : handsBuffer;
    const bigHandsPivot = bigHands
      ? handLandmarksCanvasCentroid(landmarks, layout, normalizeOffsetX, normalizeOffsetY)
      : null;

    // Assign colors to fingers for this hand if not already assigned
    assignFingerColors(i);
    
    // Initialize positions for this hand if not exists
    if (!lerpedPositions[i]) {
      lerpedPositions[i] = {};
    }
    if (!lerpedBasePositions[i]) {
      lerpedBasePositions[i] = {};
    }
    if (!lerpedPipPositions[i]) {
      lerpedPipPositions[i] = {};
    }
    if (!positionVelocities[i]) {
      positionVelocities[i] = {};
    }
    if (!basePositionVelocities[i]) {
      basePositionVelocities[i] = {};
    }
    if (!pipPositionVelocities[i]) {
      pipPositionVelocities[i] = {};
    }
    
    // Cache per-hand values outside the finger loop
    // When normalizeHands is on, landmarks are scaled to targetRawScale=100,
    // so use 100 for consistent sizing (not the raw scale that varies with distance).
    const cachedRawScale = normalizeHands ? 100 : getSmoothedRawHandScale(i);
    const cachedScaleMultiplier = cachedRawScale / 100;
    const cachedShapeScale = bigHands ? BIG_HANDS_SCALE : 1;

    // Draw each finger
    for (let finger in FINGER_TIPS) {
      let tipIndex = FINGER_TIPS[finger];
      let baseIndex = FINGER_BASES[finger];
      let pipIndex = FINGER_PIPS[finger];
      let tip = landmarks[tipIndex];
      let base = landmarks[baseIndex];
      let pip = landmarks[pipIndex];
      
      // ML → canvas (cover + intrinsic); buffer is mirrored when composited
      let targetTipX = layout.ox + tip[0] * layout.sx + normalizeOffsetX;
      let targetTipY = layout.oy + tip[1] * layout.sy + normalizeOffsetY;
      let targetBaseX = layout.ox + base[0] * layout.sx + normalizeOffsetX;
      let targetBaseY = layout.oy + base[1] * layout.sy + normalizeOffsetY;
      let targetPipX = layout.ox + pip[0] * layout.sx + normalizeOffsetX;
      let targetPipY = layout.oy + pip[1] * layout.sy + normalizeOffsetY;

      if (bigHands && bigHandsPivot) {
        const px = bigHandsPivot.x;
        const py = bigHandsPivot.y;
        let p = scaleCanvasPointFromPivot(targetTipX, targetTipY, px, py, BIG_HANDS_SCALE);
        targetTipX = p.x;
        targetTipY = p.y;
        p = scaleCanvasPointFromPivot(targetBaseX, targetBaseY, px, py, BIG_HANDS_SCALE);
        targetBaseX = p.x;
        targetBaseY = p.y;
        p = scaleCanvasPointFromPivot(targetPipX, targetPipY, px, py, BIG_HANDS_SCALE);
        targetPipX = p.x;
        targetPipY = p.y;
      }
      
      // Get or initialize lerped position for this finger tip
      let key = finger;
      if (!lerpedPositions[i][key]) {
        lerpedPositions[i][key] = { x: targetTipX, y: targetTipY };
      }
      if (!lerpedBasePositions[i][key]) {
        lerpedBasePositions[i][key] = { x: targetBaseX, y: targetBaseY };
      }
      if (!lerpedPipPositions[i][key]) {
        lerpedPipPositions[i][key] = { x: targetPipX, y: targetPipY };
      }
      if (!positionVelocities[i][key]) {
        positionVelocities[i][key] = { x: 0, y: 0 };
      }
      if (!basePositionVelocities[i][key]) {
        basePositionVelocities[i][key] = { x: 0, y: 0 };
      }
      if (!pipPositionVelocities[i][key]) {
        pipPositionVelocities[i][key] = { x: 0, y: 0 };
      }
      
      // Get easing amount from slider (0-100, convert to 0-1)
      let baseLerpAmount = cp ? cp.values.easing / 100 : 0.02;
      
      // Apply anti-jitter: check dead zone and velocity for tip position
      const tipResult = updateVelocityAndCheckDeadZone(
        lerpedPositions[i][key],
        { x: targetTipX, y: targetTipY },
        positionVelocities[i][key],
        baseLerpAmount
      );
      positionVelocities[i][key] = tipResult.updatedVelocity;
      
      // Apply anti-jitter: check dead zone and velocity for base position
      const baseResult = updateVelocityAndCheckDeadZone(
        lerpedBasePositions[i][key],
        { x: targetBaseX, y: targetBaseY },
        basePositionVelocities[i][key],
        baseLerpAmount
      );
      basePositionVelocities[i][key] = baseResult.updatedVelocity;
      
      // Apply anti-jitter: check dead zone and velocity for PIP position
      const pipResult = updateVelocityAndCheckDeadZone(
        lerpedPipPositions[i][key],
        { x: targetPipX, y: targetPipY },
        pipPositionVelocities[i][key],
        baseLerpAmount
      );
      pipPositionVelocities[i][key] = pipResult.updatedVelocity;
      
      // Lerp towards the target positions (skip if in dead zone with low velocity)
      if (!tipResult.shouldSkip) {
        lerpedPositions[i][key].x = lerp(lerpedPositions[i][key].x, targetTipX, tipResult.adaptiveLerpAmount);
        lerpedPositions[i][key].y = lerp(lerpedPositions[i][key].y, targetTipY, tipResult.adaptiveLerpAmount);
      }
      if (!baseResult.shouldSkip) {
        lerpedBasePositions[i][key].x = lerp(lerpedBasePositions[i][key].x, targetBaseX, baseResult.adaptiveLerpAmount);
        lerpedBasePositions[i][key].y = lerp(lerpedBasePositions[i][key].y, targetBaseY, baseResult.adaptiveLerpAmount);
      }
      if (!pipResult.shouldSkip) {
        lerpedPipPositions[i][key].x = lerp(lerpedPipPositions[i][key].x, targetPipX, pipResult.adaptiveLerpAmount);
        lerpedPipPositions[i][key].y = lerp(lerpedPipPositions[i][key].y, targetPipY, pipResult.adaptiveLerpAmount);
      }
      
      // Get stroke weight from slider
      let strokeWeight = cp ? cp.values.strokeWeight : 50;
      
      // Get the assigned color for this finger (monochrome = black)
      let basePaletteColor = multicolor ? fingerColors[i][finger] : '#000000';
      let fingerColor = resolveFingerFillColor(basePaletteColor, finger, i);
      let c = color(fingerColor);
      
      // Draw rectangle from base to tip (or circle if finger is too short, or bezier if bent)
      // The centers of each short side should lie on the base and tip
      let baseX = lerpedBasePositions[i][key].x;
      let baseY = lerpedBasePositions[i][key].y;
      let tipX = lerpedPositions[i][key].x;
      let tipY = lerpedPositions[i][key].y;
      let pipX = lerpedPipPositions[i][key].x;
      let pipY = lerpedPipPositions[i][key].y;
      
      // Calculate rectangle dimensions
      let rectWidth = strokeWeight * cachedScaleMultiplier * mlCanvasScale * cachedShapeScale;
      let rectHeight = dist(baseX, baseY, tipX, tipY);
      
      // Calculate finger's relative length (finger length / raw scale)
      let fingerRelativeLength = 1.0; // Default value
      if (cachedRawScale > 0) {
        // Get actual finger length from landmarks (in video space, then scale to canvas)
        let tipIndex = FINGER_TIPS[finger];
        let baseIndex = FINGER_BASES[finger];
        let tipLandmark = landmarks[tipIndex];
        let baseLandmark = landmarks[baseIndex];
        let fingerLengthPixels = dist2D(tipLandmark, baseLandmark);
        fingerRelativeLength = fingerLengthPixels / cachedRawScale;
      }
      
      // Calculate angle at PIP joint between base, PIP, and tip
      // Returns 0-180 degrees: 180 = straight, <180 = bent
      let angle = calculateAngle(
        [baseX, baseY],
        [pipX, pipY],
        [tipX, tipY]
      );
      
      // Calculate deviation from straight (180 degrees)
      // If deviation > 30 degrees, finger is bent enough for bezier
      let deviationFromStraight = 180 - angle;
      
      // Check if distances from base/tip to PIP are sufficient for bezier
      // Calculate distances in video space (from original landmarks) for accurate comparison
      let pipLandmark = landmarks[pipIndex];
      let baseLandmark = landmarks[baseIndex];
      let tipLandmark = landmarks[tipIndex];
      
      let baseToPipDistance = dist2D(baseLandmark, pipLandmark);
      let tipToPipDistance = dist2D(tipLandmark, pipLandmark);
      let minDistanceThreshold = cachedRawScale * 0.2; // 20% of raw scale
      
      let distancesSufficient = baseToPipDistance >= minDistanceThreshold && 
                                tipToPipDistance >= minDistanceThreshold;
      
      // Set finger length to always be 2x its width, with tip always at fingertip
      // Calculate direction vector from base to tip
      let dx = tipX - baseX;
      let dy = tipY - baseY;
      let baseToTipLength = Math.sqrt(dx * dx + dy * dy);
      
      // Create drawing positions where length is exactly 2x width
      let drawTipX = tipX; // Tip always stays at fingertip
      let drawTipY = tipY;
      
      let drawBaseX, drawBaseY;
      if (baseToTipLength > 0.001) {
        // Normalize direction vector
        let dirX = dx / baseToTipLength;
        let dirY = dy / baseToTipLength;
        
        // Set base position so distance from base to tip equals 2 * rectWidth
        // Base is in the opposite direction from tip
        let targetLength = 2 * rectWidth;
        drawBaseX = tipX - dirX * targetLength;
        drawBaseY = tipY - dirY * targetLength;
      } else {
        // Fallback if length is too small
        drawBaseX = baseX;
        drawBaseY = baseY;
      }
      
      // Adjust PIP proportionally along the new base-to-tip line
      let drawPipX, drawPipY;
      if (baseToTipLength > 0.001) {
        // Calculate PIP's relative position along the original base-to-tip line
        let pipDx = pipX - baseX;
        let pipDy = pipY - baseY;
        // Project PIP onto the base-to-tip line to find its relative position
        let pipProjection = (pipDx * dx + pipDy * dy) / (baseToTipLength * baseToTipLength);
        // Clamp to [0, 1] to ensure PIP is between base and tip
        pipProjection = Math.max(0, Math.min(1, pipProjection));
        
        // Calculate new PIP position along the adjusted base-to-tip line
        let newDx = drawTipX - drawBaseX;
        let newDy = drawTipY - drawBaseY;
        drawPipX = drawBaseX + newDx * pipProjection;
        drawPipY = drawBaseY + newDy * pipProjection;
      } else {
        drawPipX = pipX;
        drawPipY = pipY;
      }
      
      // Initialize Paper.js shape storage for this hand/finger if needed
      if (!paperShapes[i]) {
        paperShapes[i] = {};
      }
      if (!paperShapes[i][key]) {
        paperShapes[i][key] = { shape: null, type: null, color: fingerColor, rectWidth: null, rectHeight: null, strokeWidth: null, rectOrBez: 0.0, rectBezOrCircle: 0.0 };
      }
      // Initialize rectOrBez if it doesn't exist (for existing shapes)
      if (paperShapes[i][key].rectOrBez === undefined) {
        paperShapes[i][key].rectOrBez = 0.0;
      }
      // Initialize rectBezOrCircle if it doesn't exist (for existing shapes)
      if (paperShapes[i][key].rectBezOrCircle === undefined) {
        paperShapes[i][key].rectBezOrCircle = 0.0;
      }
      
      // Determine shape type and create/update Paper.js shapes
      let fingerMultiplier = FINGER_LENGTH_MULTIPLIERS[finger] || 1.0;
      
      // Hysteresis thresholds for circle/rectangle transition
      // When in circle state, use higher threshold to transition out (prevents flickering)
      // When in rectangle/bezier state, use lower threshold to transition to circle
      const currentState = paperShapes[i][key].type;
      const baseThreshold = 0.55 * fingerMultiplier;
      const hysteresisRange = 0.1 * fingerMultiplier; // 10% hysteresis range
      let circleThreshold;
      if (currentState === 'circle') {
        // Currently circle: use higher threshold to transition to rectangle/bezier
        circleThreshold = baseThreshold + hysteresisRange;
      } else {
        // Currently rectangle or bezier: use lower threshold to transition to circle
        circleThreshold = baseThreshold - hysteresisRange;
      }
      
      // Hysteresis thresholds for bezier/rectangle transition
      // When in bezier state, use lower threshold to transition to rectangle
      // When in rectangle state, use higher threshold to transition to bezier
      let bezierDeviationThreshold;
      if (currentState === 'bezier') {
        // Currently bezier: use lower threshold to transition to rectangle
        bezierDeviationThreshold = 25;
      } else {
        // Currently rectangle or circle: use higher threshold to transition to bezier
        bezierDeviationThreshold = 35;
      }
      
      let currentShapeType = null;
      let paperShape = null;
      
      // Determine target state for rectBezOrCircle: 0 = rect/bezier, 1 = circle
      let targetRectBezOrCircle = (fingerRelativeLength < circleThreshold) ? 1.0 : 0.0;
      
      // Update rectBezOrCircle with easing toward target
      let currentRectBezOrCircle = paperShapes[i][key].rectBezOrCircle;
      let distanceToTarget = Math.abs(targetRectBezOrCircle - currentRectBezOrCircle);
      
      if (distanceToTarget > 0.001) {
        // Ease-out: step size is proportional to distance, creating smooth deceleration
        let normalizedDistance = Math.min(distanceToTarget, 1.0);
        let easeOutFactor = normalizedDistance * (2 - normalizedDistance);
        let stepSize = 0.15 * easeOutFactor;
        
        // Move toward target with eased step size
        if (currentRectBezOrCircle < targetRectBezOrCircle) {
          currentRectBezOrCircle = Math.min(currentRectBezOrCircle + stepSize, targetRectBezOrCircle);
        } else {
          currentRectBezOrCircle = Math.max(currentRectBezOrCircle - stepSize, targetRectBezOrCircle);
        }
      }
      paperShapes[i][key].rectBezOrCircle = currentRectBezOrCircle;
      
      // Store full circle radius for scaling during transition
      let fullCircleDiameter = rectWidth * 1.28;
      let fullCircleRadius = fullCircleDiameter / 2;
      paperShapes[i][key].fullCircleRadius = fullCircleRadius;
      
      if (currentRectBezOrCircle >= 0.5) {
        // Circle state (or transitioning to/from circle)
        currentShapeType = 'circle';
        // Always create circle with full radius, scaling happens in export function
        
        // Store the last bezier path for transition calculation (if we just transitioned)
        if (paperShapes[i][key].type === 'bezier' && paperShapes[i][key].shape) {
          // Clone the bezier path for use in transition
          const bezierPath = paperShapes[i][key].shape;
          paperShapes[i][key].lastBezierPath = bezierPath.clone();
        }
        
        if (paperShapes[i][key].type !== 'circle' || !paperShapes[i][key].shape) {
          // Create new circle
          if (paperShapes[i][key].shape) {
            paperShapes[i][key].shape.remove();
          }
          paperShape = new paper.Shape.Circle(new paper.Point(drawTipX, drawTipY), fullCircleRadius);
          paperShapes[i][key].shape = paperShape;
          paperShapes[i][key].type = 'circle';
        } else {
          // Update existing circle - check if position changed
          paperShape = paperShapes[i][key].shape;
          if (Math.abs(paperShape.position.x - drawTipX) > 0.1 || Math.abs(paperShape.position.y - drawTipY) > 0.1) {
            // Position changed, update
            paperShape.position = new paper.Point(drawTipX, drawTipY);
          }
          // Radius is always full, scaling happens in export
        }
      } else {
        // Both rect and bezier states - always calculate both and lerp between them
        currentShapeType = 'bezier';
        
        // Calculate rect control points (straight line - no curvature)
        let dx = drawTipX - drawBaseX;
        let dy = drawTipY - drawBaseY;
        let rectCp1X = drawBaseX + dx / 3;
        let rectCp1Y = drawBaseY + dy / 3;
        let rectCp2X = drawBaseX + (2 * dx) / 3;
        let rectCp2Y = drawBaseY + (2 * dy) / 3;
        
        // Calculate bezier control points (curved based on PIP)
        let { cp1X: bezCp1X, cp1Y: bezCp1Y, cp2X: bezCp2X, cp2Y: bezCp2Y } = calculateBezierControlPoints(drawBaseX, drawBaseY, drawTipX, drawTipY, pipX, pipY);
        
        // Determine target state: 0 = rect, 1 = bezier
        let targetRectOrBez = (deviationFromStraight > bezierDeviationThreshold && distancesSufficient) ? 1.0 : 0.0;
        
        // Update rectOrBez with easing toward target
        let currentRectOrBez = paperShapes[i][key].rectOrBez;
        let distanceToTarget = Math.abs(targetRectOrBez - currentRectOrBez);
        
        if (distanceToTarget > 0.001) {
          // Ease-out: step size is proportional to distance, creating smooth deceleration
          // Base step size of 0.1, scaled by distance (max 1.0) with ease-out curve
          // Using ease-out quad: t * (2 - t) where t is the normalized distance
          let normalizedDistance = Math.min(distanceToTarget, 1.0);
          let easeOutFactor = normalizedDistance * (2 - normalizedDistance);
          let stepSize = 0.15 * easeOutFactor;
          
          // Move toward target with eased step size
          if (currentRectOrBez < targetRectOrBez) {
            currentRectOrBez = Math.min(currentRectOrBez + stepSize, targetRectOrBez);
          } else {
            currentRectOrBez = Math.max(currentRectOrBez - stepSize, targetRectOrBez);
          }
        }
        paperShapes[i][key].rectOrBez = currentRectOrBez;
        
        // Lerp between rect and bezier control points
        let cp1X = lerp(rectCp1X, bezCp1X, currentRectOrBez);
        let cp1Y = lerp(rectCp1Y, bezCp1Y, currentRectOrBez);
        let cp2X = lerp(rectCp2X, bezCp2X, currentRectOrBez);
        let cp2Y = lerp(rectCp2Y, bezCp2Y, currentRectOrBez);
        
        // Straighten control points as we approach rectangle transition (0.3)
        // When rectBezOrCircle approaches 0.3, move control points to 1/3 along the line from their anchor
        if (currentRectBezOrCircle > 0 && currentRectBezOrCircle < 0.3) {
          const straighteningFactor = Math.min(1.0, currentRectBezOrCircle / 0.3);
          // Control point 1 (attached to base): move to 1/3 of the way from base toward tip
          const targetCp1X = drawBaseX + (drawTipX - drawBaseX) / 3;
          const targetCp1Y = drawBaseY + (drawTipY - drawBaseY) / 3;
          // Control point 2 (attached to tip): move to 1/3 of the way from tip toward base (2/3 from base)
          const targetCp2X = drawBaseX + (drawTipX - drawBaseX) * (2 / 3);
          const targetCp2Y = drawBaseY + (drawTipY - drawBaseY) * (2 / 3);
          // Interpolate control points toward their target positions
          cp1X = lerp(cp1X, targetCp1X, straighteningFactor);
          cp1Y = lerp(cp1Y, targetCp1Y, straighteningFactor);
          cp2X = lerp(cp2X, targetCp2X, straighteningFactor);
          cp2Y = lerp(cp2Y, targetCp2Y, straighteningFactor);
        } else if (currentRectBezOrCircle >= 0.3) {
          // At 0.3 and beyond, control points should be at 1/3 positions (perfectly straight)
          cp1X = drawBaseX + (drawTipX - drawBaseX) / 3;
          cp1Y = drawBaseY + (drawTipY - drawBaseY) / 3;
          cp2X = drawBaseX + (drawTipX - drawBaseX) * (2 / 3);
          cp2Y = drawBaseY + (drawTipY - drawBaseY) * (2 / 3);
        }
        
        if (paperShapes[i][key].type !== 'bezier' || !paperShapes[i][key].shape) {
          // Create new bezier path
          if (paperShapes[i][key].shape) {
            paperShapes[i][key].shape.remove();
          }
          paperShape = new paper.Path();
          paperShape.add(new paper.Segment(new paper.Point(drawBaseX, drawBaseY), null, new paper.Point(cp1X - drawBaseX, cp1Y - drawBaseY)));
          paperShape.add(new paper.Segment(new paper.Point(drawTipX, drawTipY), new paper.Point(cp2X - drawTipX, cp2Y - drawTipY), null));
          paperShapes[i][key].shape = paperShape;
          paperShapes[i][key].type = 'bezier';
          paperShapes[i][key].strokeWidth = rectWidth; // Store stroke width for outline conversion
          // Store bezier path for potential transition
          if (paperShapes[i][key].lastBezierPath) {
            paperShapes[i][key].lastBezierPath.remove();
          }
          paperShapes[i][key].lastBezierPath = paperShape.clone();
        } else {
          // Update existing bezier path
          paperShape = paperShapes[i][key].shape;
          const segments = paperShape.segments;
          if (segments.length >= 2) {
            // Update first segment (base)
            segments[0].point = new paper.Point(drawBaseX, drawBaseY);
            segments[0].handleOut = new paper.Point(cp1X - drawBaseX, cp1Y - drawBaseY);
            // Update second segment (tip)
            segments[1].point = new paper.Point(drawTipX, drawTipY);
            segments[1].handleIn = new paper.Point(cp2X - drawTipX, cp2Y - drawTipY);
          }
          // Update stroke width
          paperShapes[i][key].strokeWidth = rectWidth;
          // Only clone for transition when rectBezOrCircle is approaching circle state
          if (paperShapes[i][key].rectBezOrCircle > 0.1) {
            if (paperShapes[i][key].lastBezierPath) paperShapes[i][key].lastBezierPath.remove();
            paperShapes[i][key].lastBezierPath = paperShape.clone();
          }
        }
      }
      
      // Update stored color
      paperShapes[i][key].color = fingerColor;
      
      // Export Paper.js shape to p5 drawing commands
      if (paperShape) {
        let rectBezOrCircle = paperShapes[i][key].rectBezOrCircle;
        if (currentShapeType === 'circle') {
          let fullRadius = paperShapes[i][key].fullCircleRadius || (rectWidth * 1.28 / 2);
          // Get the last bezier path for transition calculation
          let lastBezierPath = paperShapes[i][key].lastBezierPath || null;
          exportPaperCircleToP5(paperShape, c, hBuf, rectBezOrCircle, fullRadius, rectWidth, lastBezierPath, drawBaseX, drawBaseY, drawTipX, drawTipY);
        } else if (currentShapeType === 'bezier') {
          // Check if we should draw transition rectangle instead (0.3 to 0.5)
          if (rectBezOrCircle >= 0.3 && rectBezOrCircle < 0.5) {
            let fullRadius = paperShapes[i][key].fullCircleRadius || (rectWidth * 1.28 / 2);
            exportTransitionRectangleToP5(c, hBuf, rectBezOrCircle, rectWidth, drawBaseX, drawBaseY, drawTipX, drawTipY, fullRadius);
          } else {
            exportPaperBezierToP5(paperShape, c, rectWidth, hBuf, rectBezOrCircle, drawBaseX, drawBaseY, drawTipX, drawTipY);
          }
        }
      }
    }

    // Compute bounding box for this hand from actual Paper.js shape bounds (for texture scaling)
    if (paperShapes[i]) {
      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      for (let fk in paperShapes[i]) {
        const sd = paperShapes[i][fk];
        if (!sd || !sd.shape) continue;
        // For bezier paths, the visual extent includes the stroke width
        const sw = sd.strokeWidth || 0;
        const half = sw / 2;
        const b = sd.shape.strokeBounds || sd.shape.bounds;
        if (b) {
          minX = Math.min(minX, b.left - half);
          minY = Math.min(minY, b.top - half);
          maxX = Math.max(maxX, b.right + half);
          maxY = Math.max(maxY, b.bottom + half);
        }
      }
      if (minX < Infinity) {
        handBoundingBoxes[i] = { x: minX, y: minY, w: maxX - minX, h: maxY - minY };
      }
    }
  }

  // Clean up positions, colors, and Paper.js shapes for hands that are no longer detected
  for (let handIndex in lerpedPositions) {
    if (parseInt(handIndex) >= bufferedHands.length) {
      delete lerpedPositions[handIndex];
      delete lerpedBasePositions[handIndex];
      delete lerpedPipPositions[handIndex];
      delete positionVelocities[handIndex];
      delete basePositionVelocities[handIndex];
      delete pipPositionVelocities[handIndex];
      delete fingerColors[handIndex];
      
      // Clean up Paper.js shapes for this hand
      if (paperShapes[handIndex]) {
        for (let fingerName in paperShapes[handIndex]) {
          if (paperShapes[handIndex][fingerName].shape) {
            paperShapes[handIndex][fingerName].shape.remove();
          }
        }
        delete paperShapes[handIndex];
      }
    }
  }
}


function drawDebugInfo() {
  const layout = getHandTrackingLayout();
  const mlCanvasScale = (layout.sx + layout.sy) * 0.5;

  // Get buffered hands for debug display
  let bufferedHands = getBufferedHands();
  
  const normalizeResult = calculateNormalizeOffset(bufferedHands, layout);
  const normalizeOffsetX = normalizeResult.offsetX;
  const normalizeOffsetY = normalizeResult.offsetY;
  bufferedHands = normalizeResult.hands;
  
  // Highlight base, tip, and PIP indices
  let highlightedIndices = new Set();
  for (let finger in FINGER_TIPS) {
    highlightedIndices.add(FINGER_TIPS[finger]);
    highlightedIndices.add(FINGER_BASES[finger]);
    highlightedIndices.add(FINGER_PIPS[finger]);
  }
  
  // Store landmark positions for text drawing (outside mirrored space)
  let landmarkPositions = [];
  
  // Draw on main canvas, mirrored to match video
  push();
  translate(width, 0);
  scale(-1, 1);
  
  for (let i = 0; i < bufferedHands.length; i++) {
    let hand = bufferedHands[i];
    let landmarks = hand.landmarks;
    const pivot = bigHands ? handLandmarksCanvasCentroid(landmarks, layout, normalizeOffsetX, normalizeOffsetY) : null;
    const dbgScale = bigHands ? BIG_HANDS_SCALE : 1;

    // Draw all landmarks (circles only, text will be drawn outside mirrored space)
    for (let j = 0; j < landmarks.length; j++) {
      let landmark = landmarks[j];
      let x = layout.ox + landmark[0] * layout.sx + normalizeOffsetX;
      let y = layout.oy + landmark[1] * layout.sy + normalizeOffsetY;
      if (dbgScale !== 1 && pivot) {
        const sp = scaleCanvasPointFromPivot(x, y, pivot.x, pivot.y, dbgScale);
        x = sp.x;
        y = sp.y;
      }
      
      // Store position for text drawing (with mirrored x coordinate)
      landmarkPositions.push({
        index: j,
        x: width - x, // Mirror x coordinate
        y: y,
        isHighlighted: highlightedIndices.has(j)
      });
      
      // Check if this index should be highlighted
      let isHighlighted = highlightedIndices.has(j);
      
      if (isHighlighted) {
        // Draw highlighted landmark with larger circle and bright color
        fill(255, 255, 0, 200); // Yellow highlight
        stroke(0, 0, 0, 255);
        strokeWeight(2);
        circle(x, y, 12);
      } else {
        // Draw regular landmark with smaller circle
        fill(200, 200, 200, 150);
        stroke(100, 100, 100, 200);
        strokeWeight(1);
        circle(x, y, 6);
      }
    }
    
  }
  
  pop();
  
  // Draw text/numbers outside mirrored space so they're readable
  for (let pos of landmarkPositions) {
    if (pos.isHighlighted) {
      fill(0, 0, 0, 255);
      noStroke();
      textAlign(CENTER, CENTER);
      textSize(10);
      text(pos.index, pos.x, pos.y);
    } else {
      fill(100, 100, 100, 200);
      noStroke();
      textAlign(CENTER, CENTER);
      textSize(8);
      text(pos.index, pos.x, pos.y);
    }
  }
  
  // Draw finger length relative to raw scale (only for first hand, outside mirrored space)
  if (bufferedHands.length > 0) {
    const landmarks = bufferedHands[0].landmarks;
    const rawScale = getRawHandScale(0); // Use first hand's raw scale
    
    if (rawScale > 0 && landmarks.length >= 21) {
      for (let finger in FINGER_TIPS) {
        const tipIndex = FINGER_TIPS[finger];
        const baseIndex = FINGER_BASES[finger];
        
        const tip = landmarks[tipIndex];
        const base = landmarks[baseIndex];
        
        // Calculate finger length in pixel space
        const fingerLength = dist2D(tip, base);
        
        // Calculate relative length (finger length / raw scale)
        const relativeLength = fingerLength / rawScale;
        
        // Tip in canvas space (match drawHands big-hands scaling), then mirror x for text
        let tipCx = layout.ox + tip[0] * layout.sx + normalizeOffsetX;
        let tipCy = layout.oy + tip[1] * layout.sy + normalizeOffsetY;
        if (bigHands) {
          const pv = handLandmarksCanvasCentroid(landmarks, layout, normalizeOffsetX, normalizeOffsetY);
          const tp = scaleCanvasPointFromPivot(tipCx, tipCy, pv.x, pv.y, BIG_HANDS_SCALE);
          tipCx = tp.x;
          tipCy = tp.y;
        }
        const tipX = width - tipCx;
        const tipY = tipCy;
        
        // Draw relative length text at finger tip
        push();
        fill(0, 0, 0, 255);
        noStroke();
        textAlign(CENTER, BOTTOM);
        textSize(10);
        // Offset text slightly above the tip
        text(relativeLength.toFixed(2), tipX, tipY - 8);
        pop();
      }
    }
  }
  
  // Draw bezier control points for fingers rendered as bezier curves
  // Draw on main canvas, mirrored to match video
  push();
  translate(width, 0);
  scale(-1, 1);
  
  for (let i = 0; i < bufferedHands.length; i++) {
    if (!lerpedPositions[i] || !lerpedBasePositions[i] || !lerpedPipPositions[i]) {
      continue;
    }
    
    const landmarks = bufferedHands[i].landmarks;
    const rawScale = getRawHandScale(i); // Use this hand's raw scale
    
    for (let finger in FINGER_TIPS) {
      const key = finger;
      if (!lerpedPositions[i][key] || !lerpedBasePositions[i][key] || !lerpedPipPositions[i][key]) {
        continue;
      }
      
      // Get lerped positions (these are in canvas space, not mirrored)
      const baseX = lerpedBasePositions[i][key].x;
      const baseY = lerpedBasePositions[i][key].y;
      const tipX = lerpedPositions[i][key].x;
      const tipY = lerpedPositions[i][key].y;
      const pipX = lerpedPipPositions[i][key].x;
      const pipY = lerpedPipPositions[i][key].y;
      
      // Calculate angle to determine if bezier is being used
      // Returns 0-180 degrees: 180 = straight, <180 = bent
      const angle = calculateAngle(
        [baseX, baseY],
        [pipX, pipY],
        [tipX, tipY]
      );
      
      // Calculate deviation from straight (180 degrees)
      const deviationFromStraight = 180 - angle;
      
      // Check if distances from base/tip to PIP are sufficient for bezier
      const pipIndex = FINGER_PIPS[finger];
      const baseIndex = FINGER_BASES[finger];
      const tipIndex = FINGER_TIPS[finger];
      const pipLandmark = landmarks[pipIndex];
      const baseLandmark = landmarks[baseIndex];
      const tipLandmark = landmarks[tipIndex];
      
      const baseToPipDistance = dist2D(baseLandmark, pipLandmark);
      const tipToPipDistance = dist2D(tipLandmark, pipLandmark);
      const minDistanceThreshold = rawScale * 0.2; // 20% of raw scale
      
      const distancesSufficient = baseToPipDistance >= minDistanceThreshold && 
                                  tipToPipDistance >= minDistanceThreshold;
      
      // Use same hysteresis logic as in drawHands for consistency
      const currentState = paperShapes[i] && paperShapes[i][key] ? paperShapes[i][key].type : null;
      let bezierDeviationThreshold;
      if (currentState === 'bezier') {
        // Currently bezier: use lower threshold to transition to rectangle
        bezierDeviationThreshold = 25;
      } else {
        // Currently rectangle or circle: use higher threshold to transition to bezier
        bezierDeviationThreshold = 35;
      }
      
      // Only draw control points if deviation exceeds threshold and distances are sufficient (bezier is being used)
      if (deviationFromStraight > bezierDeviationThreshold && distancesSufficient) {
        // Calculate the same adjusted positions used in actual drawing
        // Get stroke weight from slider (same as in drawHands)
        const strokeWeightValue = cp ? cp.values.strokeWeight : 50;
        
        // Calculate rectWidth (same as in drawHands)
        const scaleMultiplier = rawScale / 100;
        const shapeScale = bigHands ? BIG_HANDS_SCALE : 1;
        const rectWidth = strokeWeightValue * scaleMultiplier * mlCanvasScale * shapeScale;
        
        // Calculate adjusted drawing positions (same logic as in drawHands)
        // Set finger length to always be 2x its width, with tip always at fingertip
        let dx = tipX - baseX;
        let dy = tipY - baseY;
        let baseToTipLength = Math.sqrt(dx * dx + dy * dy);
        
        // Create drawing positions where length is exactly 2x width
        let drawTipX = tipX; // Tip always stays at fingertip
        let drawTipY = tipY;
        
        let drawBaseX, drawBaseY;
        if (baseToTipLength > 0.001) {
          // Normalize direction vector
          let dirX = dx / baseToTipLength;
          let dirY = dy / baseToTipLength;
          
          // Set base position so distance from base to tip equals 2 * rectWidth
          // Base is in the opposite direction from tip
          let targetLength = 2 * rectWidth;
          drawBaseX = tipX - dirX * targetLength;
          drawBaseY = tipY - dirY * targetLength;
        } else {
          // Fallback if length is too small
          drawBaseX = baseX;
          drawBaseY = baseY;
        }
        
        // Calculate control points using the same shared function as in drawHands
        let { cp1X, cp1Y, cp2X, cp2Y } = calculateBezierControlPoints(drawBaseX, drawBaseY, drawTipX, drawTipY, pipX, pipY);
        
        // Draw control point 1 (from base)
        fill(255, 0, 0, 100); // Red
        noStroke();
        // strokeWeight(1);
        circle(cp1X, cp1Y, 8);
        
        // Draw control point 2 (from tip)
        circle(cp2X, cp2Y, 8);
        
        // Draw control arms (lines from adjusted base/tip to control points)
        stroke(150, 150, 150, 150);
        strokeWeight(1);
        // Dashed line style
        drawingContext.setLineDash([5, 5]);
        line(drawBaseX, drawBaseY, cp1X, cp1Y);
        line(drawTipX, drawTipY, cp2X, cp2Y);
        drawingContext.setLineDash([]); // Reset to solid
      }
    }
  }
  
  pop();
  
  // Draw hand orientation angle visualization (only for first hand, outside mirrored space)
  if (bufferedHands.length > 0 && bufferedHands[0].landmarks.length >= 10) {
    const landmarks = bufferedHands[0].landmarks;
    const wrist = landmarks[0]; // Node 0: wrist
    const middleMCP = landmarks[9]; // Node 9: middle MCP

    let wristCx = layout.ox + wrist[0] * layout.sx + normalizeOffsetX;
    let wristCy = layout.oy + wrist[1] * layout.sy + normalizeOffsetY;
    let mcpCx = layout.ox + middleMCP[0] * layout.sx + normalizeOffsetX;
    let mcpCy = layout.oy + middleMCP[1] * layout.sy + normalizeOffsetY;
    if (bigHands) {
      const pv = handLandmarksCanvasCentroid(landmarks, layout, normalizeOffsetX, normalizeOffsetY);
      let wp = scaleCanvasPointFromPivot(wristCx, wristCy, pv.x, pv.y, BIG_HANDS_SCALE);
      wristCx = wp.x;
      wristCy = wp.y;
      wp = scaleCanvasPointFromPivot(mcpCx, mcpCy, pv.x, pv.y, BIG_HANDS_SCALE);
      mcpCx = wp.x;
      mcpCy = wp.y;
    }

    // Mirror x coordinates for display
    const wristX = width - wristCx;
    const wristY = wristCy;
    const mcpX = width - mcpCx;
    const mcpY = mcpCy;
    
    // Calculate angle in radians (normal coordinate space, not mirrored)
    const dx = mcpX - wristX;
    const dy = mcpY - wristY;
    const angle = atan2(dy, dx);
    
    // Length of lines for visualization
    const lineLength = 80;
    
    // Draw filled angle region
    push();
    translate(wristX, wristY);
    
    // Draw filled arc/sector
    fill(173, 216, 230, 150); // Light blue, semi-transparent
    noStroke();
    beginShape();
    vertex(0, 0); // Origin at wrist
    
    // Draw arc from 0 to angle
    const steps = Math.max(10, Math.abs(angle) * 20); // Number of steps based on angle
    const stepSize = angle / steps;
    for (let s = 0; s <= steps; s++) {
      const a = stepSize * s;
      const x = cos(a) * lineLength;
      const y = sin(a) * lineLength;
      vertex(x, y);
    }
    endShape(CLOSE);
    
    // Draw reference line (horizontal to the right)
    stroke(0, 0, 0, 200);
    strokeWeight(2);
    line(0, 0, lineLength, 0);
    
    // Draw line from wrist to middle MCP
    stroke(0, 0, 0, 200);
    strokeWeight(2);
    line(0, 0, dx, dy);
    
    pop();
    
    // Draw angle text next to the angle
    push();
    fill(0, 0, 0, 255);
    noStroke();
    textAlign(LEFT, CENTER);
    textSize(12);
    // Position text along the angle line, offset slightly
    const textOffsetX = cos(angle) * (lineLength + 20);
    const textOffsetY = sin(angle) * (lineLength + 20);
    text(degrees(angle).toFixed(1) + "°", wristX + textOffsetX, wristY + textOffsetY);
    pop();
  }
}
