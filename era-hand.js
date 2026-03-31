(function () {
  'use strict';

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  function lerp(a, b, t) {
    return a + (b - a) * t;
  }

  function dist(x1, y1, x2, y2) {
    var dx = x2 - x1;
    var dy = y2 - y1;
    return Math.sqrt(dx * dx + dy * dy);
  }

  function rotatePoint(px, py, cx, cy, angle) {
    var cos = Math.cos(angle);
    var sin = Math.sin(angle);
    var dx = px - cx;
    var dy = py - cy;
    return {
      x: cx + dx * cos - dy * sin,
      y: cy + dx * sin + dy * cos
    };
  }

  // De Casteljau evaluation of cubic bezier at parameter t
  function bezierPoint(p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y, t) {
    var u = 1 - t;
    var uu = u * u;
    var uuu = uu * u;
    var tt = t * t;
    var ttt = tt * t;
    return {
      x: uuu * p0x + 3 * uu * t * p1x + 3 * u * tt * p2x + ttt * p3x,
      y: uuu * p0y + 3 * uu * t * p1y + 3 * u * tt * p2y + ttt * p3y
    };
  }

  // De Casteljau split: returns control points for the sub-curve [t0, t1]
  function bezierSubcurve(p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y, t0, t1) {
    // Split at t0 to get right half, then split that at adjusted t1
    function splitAt(ax, ay, bx, by, cx, cy, dx, dy, t) {
      var ab = { x: lerp(ax, bx, t), y: lerp(ay, by, t) };
      var bc = { x: lerp(bx, cx, t), y: lerp(by, cy, t) };
      var cd = { x: lerp(cx, dx, t), y: lerp(cy, dy, t) };
      var abc = { x: lerp(ab.x, bc.x, t), y: lerp(ab.y, bc.y, t) };
      var bcd = { x: lerp(bc.x, cd.x, t), y: lerp(bc.y, cd.y, t) };
      var abcd = { x: lerp(abc.x, bcd.x, t), y: lerp(abc.y, bcd.y, t) };
      return {
        left: [ax, ay, ab.x, ab.y, abc.x, abc.y, abcd.x, abcd.y],
        right: [abcd.x, abcd.y, bcd.x, bcd.y, cd.x, cd.y, dx, dy]
      };
    }

    if (t0 === 0) {
      if (t1 === 1) return [p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y];
      return splitAt(p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y, t1).left;
    }

    var rightOfT0 = splitAt(p0x, p0y, p1x, p1y, p2x, p2y, p3x, p3y, t0).right;
    if (t1 === 1) return rightOfT0;
    var newT1 = (t1 - t0) / (1 - t0);
    return splitAt(
      rightOfT0[0], rightOfT0[1], rightOfT0[2], rightOfT0[3],
      rightOfT0[4], rightOfT0[5], rightOfT0[6], rightOfT0[7],
      newT1
    ).left;
  }

  // Draw a rounded rectangle (polyfill for ctx.roundRect)
  function drawRoundedRect(ctx, x, y, w, h, r) {
    r = Math.min(r, w / 2, h / 2);
    if (r < 0) r = 0;
    if (ctx.roundRect) {
      ctx.beginPath();
      ctx.roundRect(x, y, w, h, r);
    } else {
      ctx.beginPath();
      ctx.moveTo(x + r, y);
      ctx.lineTo(x + w - r, y);
      ctx.arcTo(x + w, y, x + w, y + r, r);
      ctx.lineTo(x + w, y + h - r);
      ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
      ctx.lineTo(x + r, y + h);
      ctx.arcTo(x, y + h, x, y + h - r, r);
      ctx.lineTo(x, y + r);
      ctx.arcTo(x, y, x + r, y, r);
      ctx.closePath();
    }
  }

  // ---------------------------------------------------------------------------
  // Finger rendering
  // ---------------------------------------------------------------------------

  function renderFinger(ctx, params, scale) {
    var tipX = params.tipX * scale;
    var tipY = params.tipY * scale;
    var baseX = params.baseX * scale;
    var baseY = params.baseY * scale;
    var pipX = params.pipX * scale;
    var pipY = params.pipY * scale;
    var rectWidth = params.rectWidth * scale;
    var rbc = params.rectBezOrCircle;
    var rob = params.rectOrBez;
    var color = params.color;

    var len = dist(baseX, baseY, tipX, tipY);
    var angle = Math.atan2(tipY - baseY, tipX - baseX);

    ctx.fillStyle = color;
    ctx.strokeStyle = color;

    if (rbc >= 0.5) {
      // Circle / rounded-square mode
      var fullDiam = rectWidth * 1.28;
      var fullR = fullDiam / 2;
      var tp = (rbc - 0.5) * 2; // 0..1

      var borderRadius = fullR / 2 + (fullR / 2) * tp;
      var edgeLength = rectWidth + (fullDiam - rectWidth) * tp;

      // Center interpolates from segment midpoint to tip
      var midX = (baseX + tipX) / 2;
      var midY = (baseY + tipY) / 2;
      var cx = lerp(midX, tipX, tp);
      var cy = lerp(midY, tipY, tp);

      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate(angle);
      drawRoundedRect(ctx, -edgeLength / 2, -edgeLength / 2, edgeLength, edgeLength, borderRadius);
      ctx.fill();
      ctx.restore();

    } else if (rbc >= 0.3) {
      // Transition rectangle
      var fullDiam2 = rectWidth * 1.28;
      var fullR2 = fullDiam2 / 2;
      var tp2 = (rbc - 0.3) / 0.2; // 0..1

      var visibleLength = len * 0.7 - (len * 0.7 - rectWidth) * tp2;
      var br = Math.max(0, (fullR2 / 2) * tp2);

      // Center between visible start and tip
      var visStartDist = len - visibleLength;
      var visStartX = baseX + (tipX - baseX) * (visStartDist / len);
      var visStartY = baseY + (tipY - baseY) * (visStartDist / len);
      var rcx = (visStartX + tipX) / 2;
      var rcy = (visStartY + tipY) / 2;

      ctx.save();
      ctx.translate(rcx, rcy);
      ctx.rotate(angle);
      drawRoundedRect(ctx, -visibleLength / 2, -rectWidth / 2, visibleLength, rectWidth, br);
      ctx.fill();
      ctx.restore();

    } else {
      // Bezier curve mode with SQUARE lineCap
      var DEG30 = Math.PI / 6;

      // Raw control points
      var raw1x = baseX + (pipX - baseX) * 0.7;
      var raw1y = baseY + (pipY - baseY) * 0.7;
      var raw2x = tipX + (pipX - tipX) * 0.6;
      var raw2y = tipY + (pipY - tipY) * 0.6;

      // 30-degree rotation with direction based on perpendicular test (matches live pipeline)
      var btLen = dist(baseX, baseY, tipX, tipY);
      var cp1 = { x: raw1x, y: raw1y };
      var cp2 = { x: raw2x, y: raw2y };
      if (btLen > 0.001) {
        var btNX = (tipX - baseX) / btLen, btNY = (tipY - baseY) / btLen;
        var perpX = -btNY, perpY = btNX;
        // Rotate cp1 around base
        var v1x = raw1x - baseX, v1y = raw1y - baseY;
        var v1L = Math.sqrt(v1x * v1x + v1y * v1y);
        if (v1L > 0.001) {
          var a1 = Math.atan2(v1y, v1x);
          var dir1 = (v1x * perpX + v1y * perpY) >= 0 ? 1 : -1;
          a1 += dir1 * DEG30;
          cp1.x = baseX + Math.cos(a1) * v1L;
          cp1.y = baseY + Math.sin(a1) * v1L;
        }
        // Rotate cp2 around tip (flipped direction)
        var v2x = raw2x - tipX, v2y = raw2y - tipY;
        var v2L = Math.sqrt(v2x * v2x + v2y * v2y);
        if (v2L > 0.001) {
          var a2 = Math.atan2(v2y, v2x);
          var dir2 = (v2x * perpX + v2y * perpY) >= 0 ? -1 : 1;
          a2 += dir2 * DEG30;
          cp2.x = tipX + Math.cos(a2) * v2L;
          cp2.y = tipY + Math.sin(a2) * v2L;
        }
      }

      // Straight control points at 1/3 positions
      var s1x = baseX + (tipX - baseX) / 3;
      var s1y = baseY + (tipY - baseY) / 3;
      var s2x = baseX + (tipX - baseX) * 2 / 3;
      var s2y = baseY + (tipY - baseY) * 2 / 3;

      // Blend: rectOrBez=0 means straight, =1 means curved
      var c1x = lerp(s1x, cp1.x, rob);
      var c1y = lerp(s1y, cp1.y, rob);
      var c2x = lerp(s2x, cp2.x, rob);
      var c2y = lerp(s2y, cp2.y, rob);

      // Double-straighten toward 1/3 positions based on rbc
      var sf = Math.min(1, rbc / 0.3);
      // Pass 1
      c1x += (s1x - c1x) * sf; c1y += (s1y - c1y) * sf;
      c2x += (s2x - c2x) * sf; c2y += (s2y - c2y) * sf;
      // Pass 2
      c1x += (s1x - c1x) * sf; c1y += (s1y - c1y) * sf;
      c2x += (s2x - c2x) * sf; c2y += (s2y - c2y) * sf;

      // Partial undrawing
      var visPortion = 1.0 - (rbc / 0.3) * 0.3;

      ctx.lineCap = 'square';
      ctx.lineWidth = rectWidth;

      if (visPortion < 1.0) {
        var startT = 1.0 - visPortion;
        var sub = bezierSubcurve(baseX, baseY, c1x, c1y, c2x, c2y, tipX, tipY, startT, 1.0);
        ctx.beginPath();
        ctx.moveTo(sub[0], sub[1]);
        ctx.bezierCurveTo(sub[2], sub[3], sub[4], sub[5], sub[6], sub[7]);
        ctx.stroke();
      } else {
        ctx.beginPath();
        ctx.moveTo(baseX, baseY);
        ctx.bezierCurveTo(c1x, c1y, c2x, c2y, tipX, tipY);
        ctx.stroke();
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Pose interpolation
  // ---------------------------------------------------------------------------

  var NUMERIC_KEYS = ['tipX', 'tipY', 'baseX', 'baseY', 'pipX', 'pipY', 'rectWidth', 'rectBezOrCircle', 'rectOrBez'];

  function lerpFinger(a, b, t) {
    var result = {};
    for (var i = 0; i < NUMERIC_KEYS.length; i++) {
      var k = NUMERIC_KEYS[i];
      result[k] = lerp(a[k], b[k], t);
    }
    // Color always takes from target
    result.color = b.color;
    return result;
  }

  /** Convert a pose to a keyed map: { "handIdx:fingerName": params, ... }.
   *  Preserves finger identity for correct lerping between poses. */
  function normalizePose(pose) {
    if (!pose) return {};
    var map = {};
    var fingers = pose.fingers || pose;
    // Handle nested {handIdx: {fingerName: params}}
    for (var hi in fingers) {
      if (typeof fingers[hi] !== 'object') continue;
      for (var fn in fingers[hi]) {
        var p = fingers[hi][fn];
        if (p && typeof p.tipX === 'number') {
          map[hi + ':' + fn] = p;
        }
      }
    }
    return map;
  }

  /** Lerp between two keyed pose maps. Fingers present in both are interpolated;
   *  fingers only in one are included as-is. */
  function lerpPose(poseA, poseB, t) {
    if (!poseA) return poseB;
    if (!poseB) return poseA;
    var result = {};
    // Union of all keys
    var keys = {};
    var k;
    for (k in poseA) keys[k] = true;
    for (k in poseB) keys[k] = true;
    for (k in keys) {
      var a = poseA[k], b = poseB[k];
      if (a && b) result[k] = lerpFinger(a, b, t);
      else result[k] = a || b;
    }
    return result;
  }

  // ---------------------------------------------------------------------------
  // EraHand engine
  // ---------------------------------------------------------------------------

  function create(opts) {
    opts = opts || {};
    var rawPoses = opts.poses || {};
    var size = opts.size || 150;
    var defaultPose = opts.defaultPose || 'open';
    var hideCursor = opts.cursor !== undefined ? opts.cursor : true;
    var poseSpeed = opts.lerpSpeed || 0.12;
    var cursorSmooth = opts.cursorSmooth || 0.3;
    var pressPose = opts.pressPose || null; // pose to show on mousedown/touchstart

    // Normalize all poses to keyed maps and compute per-pose scale
    var poses = {};
    var poseScales = {};
    for (var pn in rawPoses) {
      var rp = rawPoses[pn];
      poses[pn] = normalizePose(rp);
      poseScales[pn] = (rp && rp.refSize) ? rp.refSize : 200;
    }

    // State
    var mouseX = -9999;
    var mouseY = -9999;
    var canvasX = -9999;
    var canvasY = -9999;
    var visible = false;
    var destroyed = false;

    var currentPoseData = poses[defaultPose] || null;
    var targetPoseName = defaultPose;
    var targetPoseData = currentPoseData;
    var poseT = 1.0;
    var renderPose = currentPoseData;
    var isPressed = false;

    // Hover rules: array of { selector, poseName }
    var hoverRules = [];
    var hoverActivePose = null;

    // Create canvas
    var canvas = document.createElement('canvas');
    canvas.width = size;
    canvas.height = size;
    canvas.style.cssText =
      'position:fixed;top:0;left:0;width:' + size + 'px;height:' + size + 'px;' +
      'pointer-events:none;z-index:999999;will-change:transform;display:none;';
    document.body.appendChild(canvas);
    var ctx = canvas.getContext('2d');

    // Hide default cursor
    var cursorStyle = null;
    if (hideCursor) {
      cursorStyle = document.createElement('style');
      cursorStyle.textContent = '* { cursor: none !important; }';
      document.head.appendChild(cursorStyle);
    }

    // -----------------------------------------------------------------------
    // Event handlers
    // -----------------------------------------------------------------------

    function onMouseMove(e) {
      mouseX = e.clientX;
      mouseY = e.clientY;
      if (!visible) {
        visible = true;
        canvas.style.display = 'block';
      }
      checkHoverRules(e.clientX, e.clientY);
    }

    function onTouchMove(e) {
      if (e.touches.length > 0) {
        var t = e.touches[0];
        mouseX = t.clientX;
        mouseY = t.clientY;
        if (!visible) {
          visible = true;
          canvas.style.display = 'block';
        }
        checkHoverRules(t.clientX, t.clientY);
      }
    }

    function onTouchStart(e) {
      if (e.touches.length > 0) {
        var t = e.touches[0];
        mouseX = t.clientX;
        mouseY = t.clientY;
        canvasX = mouseX;
        canvasY = mouseY;
        if (!visible) {
          visible = true;
          canvas.style.display = 'block';
        }
        checkHoverRules(t.clientX, t.clientY);
      }
    }

    function onTouchEnd() {
      visible = false;
      canvas.style.display = 'none';
    }

    function onMouseLeave() {
      visible = false;
      canvas.style.display = 'none';
    }

    function onMouseEnter(e) {
      mouseX = e.clientX;
      mouseY = e.clientY;
      canvasX = mouseX;
      canvasY = mouseY;
      visible = true;
      canvas.style.display = 'block';
    }

    document.addEventListener('mousemove', onMouseMove, { passive: true });
    document.addEventListener('touchmove', onTouchMove, { passive: true });
    document.addEventListener('touchstart', onTouchStart, { passive: true });
    document.addEventListener('touchend', onTouchEnd, { passive: true });
    document.documentElement.addEventListener('mouseleave', onMouseLeave);
    document.documentElement.addEventListener('mouseenter', onMouseEnter);

    function onPressStart() {
      isPressed = true;
      if (pressPose && poses[pressPose] && !hoverActivePose) {
        transitionTo(pressPose);
      }
    }
    function onPressEnd() {
      isPressed = false;
      if (pressPose && !hoverActivePose) {
        transitionTo(targetPoseName);
      }
    }
    document.addEventListener('mousedown', onPressStart);
    document.addEventListener('mouseup', onPressEnd);

    // -----------------------------------------------------------------------
    // Hover detection
    // -----------------------------------------------------------------------

    function checkHoverRules(cx, cy) {
      if (hoverRules.length === 0) {
        if (hoverActivePose !== null) {
          hoverActivePose = null;
          transitionTo(targetPoseName);
        }
        return;
      }
      var el = document.elementFromPoint(cx, cy);
      var matched = null;
      for (var i = 0; i < hoverRules.length; i++) {
        var rule = hoverRules[i];
        if (el && el.closest && el.closest(rule.selector)) {
          matched = rule.poseName;
          break;
        }
      }
      if (matched !== hoverActivePose) {
        hoverActivePose = matched;
        if (matched) {
          transitionTo(matched);
        } else if (isPressed && pressPose && poses[pressPose]) {
          transitionTo(pressPose);
        } else {
          transitionTo(targetPoseName);
        }
      }
    }

    // -----------------------------------------------------------------------
    // Pose transition
    // -----------------------------------------------------------------------

    var activePoseName = defaultPose;

    function transitionTo(name) {
      if (name === activePoseName && poseT >= 1.0) return;
      var newTarget = poses[name];
      if (!newTarget) return;
      // Snapshot the current rendered pose as the source
      currentPoseData = renderPose ? JSON.parse(JSON.stringify(renderPose)) : targetPoseData;
      targetPoseData = newTarget;
      activePoseName = name;
      poseT = 0;
    }

    // -----------------------------------------------------------------------
    // Render loop
    // -----------------------------------------------------------------------

    var rafId;

    function frame() {
      if (destroyed) return;
      rafId = requestAnimationFrame(frame);

      // Smooth cursor follow
      canvasX = lerp(canvasX, mouseX, cursorSmooth);
      canvasY = lerp(canvasY, mouseY, cursorSmooth);

      // Pose interpolation
      if (poseT < 1.0) {
        poseT = Math.min(1.0, poseT + poseSpeed);
        renderPose = lerpPose(currentPoseData, targetPoseData, poseT);
      } else {
        renderPose = targetPoseData;
      }

      if (!visible || !renderPose) return;

      // Position canvas centered on cursor
      var tx = canvasX - size / 2;
      var ty = canvasY - size / 2;
      canvas.style.transform = 'translate(' + tx + 'px,' + ty + 'px)';

      // Clear & draw
      ctx.clearRect(0, 0, size, size);
      ctx.save();
      // Mirror horizontally (hand was captured mirrored)
      ctx.translate(size, 0);
      ctx.scale(-1, 1);
      // Scale from pose coordinate space into canvas, centered
      var refSize = poseScales[activePoseName] || 200;
      var scale = size / refSize;
      ctx.translate(size / 2, size / 2);
      for (var fk in renderPose) {
        renderFinger(ctx, renderPose[fk], scale);
      }
      ctx.restore();
    }

    rafId = requestAnimationFrame(frame);

    // -----------------------------------------------------------------------
    // Public API
    // -----------------------------------------------------------------------

    return {
      setPose: function (name) {
        targetPoseName = name;
        if (!hoverActivePose) {
          transitionTo(name);
        }
      },

      on: function (selector, poseName) {
        hoverRules.push({ selector: selector, poseName: poseName });
      },

      off: function (selector) {
        hoverRules = hoverRules.filter(function (r) {
          return r.selector !== selector;
        });
      },

      destroy: function () {
        destroyed = true;
        cancelAnimationFrame(rafId);
        document.removeEventListener('mousemove', onMouseMove);
        document.removeEventListener('touchmove', onTouchMove);
        document.removeEventListener('touchstart', onTouchStart);
        document.removeEventListener('touchend', onTouchEnd);
        document.documentElement.removeEventListener('mouseleave', onMouseLeave);
        document.documentElement.removeEventListener('mouseenter', onMouseEnter);
        document.removeEventListener('mousedown', onPressStart);
        document.removeEventListener('mouseup', onPressEnd);
        if (canvas.parentNode) canvas.parentNode.removeChild(canvas);
        if (cursorStyle && cursorStyle.parentNode) cursorStyle.parentNode.removeChild(cursorStyle);
      },

      canvas: canvas,
      ctx: ctx
    };
  }

  // ---------------------------------------------------------------------------
  // Export
  // ---------------------------------------------------------------------------

  window.EraHand = {
    create: create
  };

})();
