(function () {
  var canvas = document.createElement("canvas");
  canvas.id = "brownian-canvas";
  document.body.prepend(canvas);
  var ctx = canvas.getContext("2d");

  var NUM_VISIBLE = 18;
  var NUM_HIDDEN = 110;
  var MAX_POINTS = 300;
  var HIST_BINS = 120;
  var HIST_HEIGHT = 50;
  var frameCount = 0;
  var xHistAccum = new Array(HIST_BINS).fill(0);
  var yHistAccum = new Array(HIST_BINS).fill(0);

  // Physics params
  var DT = 5e-4;
  var GAMMA_DT = 0.178;  // log10 slider midpoint: -0.75 → 10^(-0.75) ≈ 0.178
  var T = 5.62;          // log10 slider midpoint: 0.75 → 10^0.75 ≈ 5.62
  var MASS = 1.0;
  var SKIP = 20;

  // Initial condition presets (MB coordinates)
  var IC_PRESETS = {
    saddle:  { x: -0.82, y: 0.62, label: "Saddle" },
    left:    { x: -0.56, y: 1.44, label: "Left Min" },
    right:   { x: 0.62,  y: 0.03, label: "Right Min" },
    center:  { x: -0.05, y: 0.47, label: "Center Min" }
  };
  var currentIC = "saddle";

  // Thermostat constants
  var c1, c2;
  function updateThermostat() {
    c1 = Math.exp(-GAMMA_DT / 2.0);
    c2 = Math.sqrt(MASS * T * (1.0 - c1 * c1));
  }
  updateThermostat();

  // Müller-Brown domain
  var MBX_MIN = -2.0, MBX_MAX = 1.7;
  var MBY_MIN = -0.7, MBY_MAX = 2.5;

  var A  = [-200, -100, -170, 15];
  var aa = [-1, -1, -6.5, 0.7];
  var bb = [0, 0, 11, 0.6];
  var cc = [-10, -10, -6.5, 0.7];
  var x0 = [1, 0, -0.5, -1];
  var y0 = [0, 0.5, 1.5, 1];

  var DARK2 = [
    [27, 158, 119], [217, 95, 2], [117, 112, 179], [231, 41, 138],
    [102, 166, 30], [230, 171, 2], [166, 118, 29], [102, 102, 102]
  ];

  var walkers = [];
  var contourImage = null;

  function mbPotential(x, y) {
    var v = 0;
    for (var i = 0; i < 4; i++) {
      var dx = x - x0[i], dy = y - y0[i];
      v += A[i] * Math.exp(aa[i] * dx * dx + bb[i] * dx * dy + cc[i] * dy * dy);
    }
    return v;
  }

  function mbGradient(x, y) {
    var gx = 0, gy = 0;
    for (var i = 0; i < 4; i++) {
      var dx = x - x0[i], dy = y - y0[i];
      var e = A[i] * Math.exp(aa[i] * dx * dx + bb[i] * dx * dy + cc[i] * dy * dy);
      gx += e * (2 * aa[i] * dx + bb[i] * dy);
      gy += e * (bb[i] * dx + 2 * cc[i] * dy);
    }
    return { x: gx, y: gy };
  }

  function randn() {
    var u1 = Math.random(), u2 = Math.random();
    return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  }

  var scaleX, scaleY;
  function computeScale() {
    scaleX = canvas.width / (MBX_MAX - MBX_MIN);
    scaleY = canvas.height / (MBY_MAX - MBY_MIN);
  }
  function toCanvas(mx, my) {
    return { x: (mx - MBX_MIN) * scaleX, y: (MBY_MAX - my) * scaleY };
  }
  function toMB(cx, cy) {
    return { x: MBX_MIN + cx / scaleX, y: MBY_MAX - cy / scaleY };
  }

  function resize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
    computeScale();
    contourImage = null;
  }

  function drawContours() {
    if (contourImage && contourImage.width === canvas.width) {
      ctx.putImageData(contourImage, 0, 0);
      return;
    }
    var gridW = Math.floor(canvas.width / 4);
    var gridH = Math.floor(canvas.height / 4);
    var vals = [];
    var vmin = Infinity, vmax = -Infinity;
    for (var iy = 0; iy < gridH; iy++) {
      vals[iy] = [];
      for (var ix = 0; ix < gridW; ix++) {
        var mx = MBX_MIN + (ix / gridW) * (MBX_MAX - MBX_MIN);
        var my = MBY_MIN + (1 - iy / gridH) * (MBY_MAX - MBY_MIN);
        var v = mbPotential(mx, my);
        v = Math.max(-200, Math.min(v, 50));
        vals[iy][ix] = v;
        if (v < vmin) vmin = v;
        if (v > vmax) vmax = v;
      }
    }
    var imgData = ctx.createImageData(canvas.width, canvas.height);
    for (var i = 0; i < imgData.data.length; i += 4) {
      imgData.data[i] = 15; imgData.data[i+1] = 15; imgData.data[i+2] = 18; imgData.data[i+3] = 255;
    }
    for (var py = 0; py < canvas.height; py++) {
      for (var px = 0; px < canvas.width; px++) {
        var mb = toMB(px, py);
        if (mb.x < MBX_MIN || mb.x > MBX_MAX || mb.y < MBY_MIN || mb.y > MBY_MAX) continue;
        var ix = Math.floor(((mb.x - MBX_MIN) / (MBX_MAX - MBX_MIN)) * gridW);
        var iy = Math.floor(((MBY_MAX - mb.y) / (MBY_MAX - MBY_MIN)) * gridH);
        ix = Math.max(0, Math.min(ix, gridW - 1));
        iy = Math.max(0, Math.min(iy, gridH - 1));
        var t = (vals[iy][ix] - vmin) / (vmax - vmin);
        var idx = (py * canvas.width + px) * 4;
        imgData.data[idx]     = Math.floor(10 + t * 30);
        imgData.data[idx + 1] = Math.floor(10 + t * 25);
        imgData.data[idx + 2] = Math.floor(18 + t * 45);
      }
    }
    var levels = 15;
    for (var l = 0; l < levels; l++) {
      var threshold = vmin + (l / levels) * (vmax - vmin);
      for (var iy = 0; iy < gridH - 1; iy++) {
        for (var ix = 0; ix < gridW - 1; ix++) {
          var sum = (vals[iy][ix] > threshold ? 1 : 0) + (vals[iy][ix+1] > threshold ? 1 : 0)
                  + (vals[iy+1][ix] > threshold ? 1 : 0) + (vals[iy+1][ix+1] > threshold ? 1 : 0);
          if (sum > 0 && sum < 4) {
            var cmx = MBX_MIN + ((ix + 0.5) / gridW) * (MBX_MAX - MBX_MIN);
            var cmy = MBY_MAX - ((iy + 0.5) / gridH) * (MBY_MAX - MBY_MIN);
            var cp = toCanvas(cmx, cmy);
            var cpx = Math.floor(cp.x), cpy = Math.floor(cp.y);
            var idx = (cpy * canvas.width + cpx) * 4;
            if (idx >= 0 && idx < imgData.data.length - 3) {
              imgData.data[idx]     = Math.min(255, imgData.data[idx] + 40);
              imgData.data[idx + 1] = Math.min(255, imgData.data[idx + 1] + 35);
              imgData.data[idx + 2] = Math.min(255, imgData.data[idx + 2] + 55);
            }
          }
        }
      }
    }
    ctx.putImageData(imgData, 0, 0);
    contourImage = ctx.getImageData(0, 0, canvas.width, canvas.height);
  }

  function initWalker(visible) {
    var color = DARK2[Math.floor(Math.random() * DARK2.length)];
    var ic = IC_PRESETS[currentIC];
    return {
      mx: ic.x + (Math.random() - 0.5) * 0.05,
      my: ic.y + (Math.random() - 0.5) * 0.05,
      px: 0, py: 0,
      points: [], color: color,
      opacity: 0.06 + Math.random() * 0.05,
      visible: visible
    };
  }

  function init() {
    resize();
    updateThermostat();
    walkers = [];
    for (var i = 0; i < NUM_VISIBLE; i++) walkers.push(initWalker(true));
    for (var i = 0; i < NUM_HIDDEN; i++) walkers.push(initWalker(false));
    frameCount = 0;
    xHistAccum = new Array(HIST_BINS).fill(0);
    yHistAccum = new Array(HIST_BINS).fill(0);
  }

  function stepWalker(w) {
    w.px = c1 * w.px + c2 * randn();
    w.py = c1 * w.py + c2 * randn();
    w.mx += 0.5 * DT * w.px / MASS;
    w.my += 0.5 * DT * w.py / MASS;
    var g = mbGradient(w.mx, w.my);
    w.px += DT * (-g.x);
    w.py += DT * (-g.y);
    w.mx += 0.5 * DT * w.px / MASS;
    w.my += 0.5 * DT * w.py / MASS;
    w.px = c1 * w.px + c2 * randn();
    w.py = c1 * w.py + c2 * randn();
    if (w.mx < MBX_MIN) { w.mx = 2 * MBX_MIN - w.mx; w.px = Math.abs(w.px); }
    if (w.mx > MBX_MAX) { w.mx = 2 * MBX_MAX - w.mx; w.px = -Math.abs(w.px); }
    if (w.my < MBY_MIN) { w.my = 2 * MBY_MIN - w.my; w.py = Math.abs(w.py); }
    if (w.my > MBY_MAX) { w.my = 2 * MBY_MAX - w.my; w.py = -Math.abs(w.py); }
  }

  function rgba(c, a) {
    return "rgba(" + c[0] + "," + c[1] + "," + c[2] + "," + a + ")";
  }

  var HIST_DECAY = 0.998;
  function accumulateHistogram() {
    for (var i = 0; i < HIST_BINS; i++) { xHistAccum[i] *= HIST_DECAY; yHistAccum[i] *= HIST_DECAY; }
    for (var i = 0; i < walkers.length; i++) {
      var w = walkers[i];
      var cp = toCanvas(w.mx, w.my);
      var xBin = Math.floor((cp.x / canvas.width) * HIST_BINS);
      var yBin = Math.floor((cp.y / canvas.height) * HIST_BINS);
      if (xBin >= 0 && xBin < HIST_BINS) xHistAccum[xBin]++;
      if (yBin >= 0 && yBin < HIST_BINS) yHistAccum[yBin]++;
    }
  }

  function drawHistograms() {
    var xMax = 0, yMax = 0;
    for (var i = 0; i < HIST_BINS; i++) {
      if (xHistAccum[i] > xMax) xMax = xHistAccum[i];
      if (yHistAccum[i] > yMax) yMax = yHistAccum[i];
    }
    if (xMax === 0) xMax = 1;
    if (yMax === 0) yMax = 1;
    var binW = canvas.width / HIST_BINS;
    var binH = canvas.height / HIST_BINS;
    for (var i = 0; i < HIST_BINS; i++) {
      var h = (xHistAccum[i] / xMax) * HIST_HEIGHT;
      if (h < 1) continue;
      ctx.fillStyle = "rgba(255, 255, 255, 0.1)";
      ctx.fillRect(i * binW, canvas.height - h, binW - 1, h);
    }
    for (var i = 0; i < HIST_BINS; i++) {
      var w = (yHistAccum[i] / yMax) * HIST_HEIGHT;
      if (w < 1) continue;
      ctx.fillStyle = "rgba(255, 255, 255, 0.1)";
      ctx.fillRect(0, i * binH, w, binH - 1);
    }
  }

  // ── Control panel ──
  var panel = document.createElement("div");
  panel.id = "sim-controls";
  panel.innerHTML = [
    '<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">',
    '  <label>kBT</label><input type="range" id="ctrl-T" min="-1" max="2.5" step="0.01" value="0.75" style="width:100px">',
    '</div>',
    '<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">',
    '  <label>γΔt</label><input type="range" id="ctrl-G" min="-3" max="1.5" step="0.01" value="-0.75" style="width:100px">',
    '</div>',
    '<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">',
    '  <label>IC</label>',
    '  <select id="ctrl-IC" style="background:#222;color:#ccc;border:1px solid #444;border-radius:4px;padding:2px 4px">',
    '    <option value="saddle">Saddle</option>',
    '    <option value="left">Left Min</option>',
    '    <option value="right">Right Min</option>',
    '    <option value="center">Center Min</option>',
    '  </select>',
    '</div>',
    '<button id="ctrl-restart" style="background:#333;color:#ccc;border:1px solid #555;border-radius:4px;padding:4px 12px;cursor:pointer">Restart</button>'
  ].join("");
  panel.style.cssText = "position:fixed;bottom:16px;right:16px;z-index:1000;background:rgba(0,0,0,0.7);backdrop-filter:blur(8px);padding:12px 16px;border-radius:8px;border:1px solid rgba(255,255,255,0.1);font-family:monospace;font-size:13px;color:#aaa;pointer-events:auto;";
  document.body.appendChild(panel);

  var sliderT = document.getElementById("ctrl-T");
  var sliderG = document.getElementById("ctrl-G");
  var selectIC = document.getElementById("ctrl-IC");
  var btnRestart = document.getElementById("ctrl-restart");

  sliderT.addEventListener("input", function () {
    T = Math.pow(10, parseFloat(this.value));
    updateThermostat();
  });
  sliderG.addEventListener("input", function () {
    GAMMA_DT = Math.pow(10, parseFloat(this.value));
    updateThermostat();
  });
  selectIC.addEventListener("change", function () {
    currentIC = this.value;
  });
  btnRestart.addEventListener("click", function () {
    init();
  });

  // ── Draw loop ──
  var lastFrameTime = 0;
  var FRAME_INTERVAL = 1000 / 20;

  function draw(timestamp) {
    requestAnimationFrame(draw);
    if (timestamp - lastFrameTime < FRAME_INTERVAL) return;
    lastFrameTime = timestamp;
    frameCount++;

    drawContours();

    for (var s = 0; s < SKIP; s++) {
      for (var i = 0; i < walkers.length; i++) stepWalker(walkers[i]);
    }

    for (var i = 0; i < walkers.length; i++) {
      var w = walkers[i];
      if (!w.visible) continue;
      var cp = toCanvas(w.mx, w.my);
      w.points.push({ x: cp.x, y: cp.y });
      if (w.points.length > MAX_POINTS) w.points.shift();
    }

    for (var i = 0; i < walkers.length; i++) {
      var w = walkers[i];
      if (!w.visible || w.points.length < 2) continue;
      for (var j = 1; j < w.points.length; j++) {
        var alpha = (j / w.points.length) * w.opacity;
        ctx.strokeStyle = rgba(w.color, alpha);
        ctx.lineWidth = 2 + (j / w.points.length) * 4;
        ctx.beginPath();
        ctx.moveTo(w.points[j - 1].x, w.points[j - 1].y);
        ctx.lineTo(w.points[j].x, w.points[j].y);
        ctx.stroke();
      }
      var head = w.points[w.points.length - 1];
      ctx.beginPath();
      ctx.arc(head.x, head.y, 6, 0, Math.PI * 2);
      ctx.fillStyle = rgba(w.color, w.opacity * 1.5);
      ctx.fill();
      ctx.beginPath();
      ctx.arc(head.x, head.y, 10, 0, Math.PI * 2);
      ctx.fillStyle = rgba(w.color, w.opacity * 0.4);
      ctx.fill();
    }

    accumulateHistogram();
    drawHistograms();
  }

  window.addEventListener("resize", function () { resize(); contourImage = null; });
  init();
  requestAnimationFrame(draw);
})();
