(function () {
  var canvas = document.createElement("canvas");
  canvas.id = "brownian-canvas";
  document.body.prepend(canvas);
  var ctx = canvas.getContext("2d");

  var paths = [];
  var NUM_PATHS = 12;
  var STEP_SIZE = 1.8;
  var MAX_POINTS = 600;
  var FADE_RATE = 0.012;

  function resize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  }

  function initPath() {
    return {
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      points: [],
      hue: 270 + Math.random() * 60,
      opacity: 0.3 + Math.random() * 0.3,
      drift: { x: (Math.random() - 0.5) * 0.3, y: (Math.random() - 0.5) * 0.3 }
    };
  }

  function init() {
    resize();
    paths = [];
    for (var i = 0; i < NUM_PATHS; i++) {
      paths.push(initPath());
    }
  }

  function step(p) {
    var angle = Math.random() * Math.PI * 2;
    p.x += Math.cos(angle) * STEP_SIZE + p.drift.x;
    p.y += Math.sin(angle) * STEP_SIZE + p.drift.y;

    if (p.x < -50 || p.x > canvas.width + 50 || p.y < -50 || p.y > canvas.height + 50) {
      p.x = Math.random() * canvas.width;
      p.y = Math.random() * canvas.height;
      p.points = [];
    }

    p.points.push({ x: p.x, y: p.y });
    if (p.points.length > MAX_POINTS) {
      p.points.shift();
    }
  }

  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    for (var i = 0; i < paths.length; i++) {
      var p = paths[i];
      step(p);

      if (p.points.length < 2) continue;

      ctx.beginPath();
      ctx.moveTo(p.points[0].x, p.points[0].y);

      for (var j = 1; j < p.points.length; j++) {
        var alpha = (j / p.points.length) * p.opacity;
        ctx.strokeStyle = "hsla(" + p.hue + ", 70%, 70%, " + alpha + ")";
        ctx.lineWidth = 1 + (j / p.points.length) * 1.2;
        ctx.beginPath();
        ctx.moveTo(p.points[j - 1].x, p.points[j - 1].y);
        ctx.lineTo(p.points[j].x, p.points[j].y);
        ctx.stroke();
      }
    }

    requestAnimationFrame(draw);
  }

  window.addEventListener("resize", resize);
  init();
  draw();
})();
