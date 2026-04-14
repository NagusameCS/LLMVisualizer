// ══════════════════════════════════════════════════════════════
// LLM Network Visualizer – 3D WebGL Renderer (Three.js)
// Interactive transformer architecture with weight textures,
// orbit camera, raycasting hover, and entrance animations.
// ══════════════════════════════════════════════════════════════
(function () {
  "use strict";

  /* ── tiny helpers ──────────────────────────────────────────── */
  function spec(v, h, l, hd, kv, inter, seq, hdim, gate) {
    return { vocab: v, hidden: h, layers: l, heads: hd, kvHeads: kv, inter: inter, seq: seq, headDim: hdim, gate: gate !== false };
  }
  function fmtNum(n) { return n >= 1e9 ? (n/1e9).toFixed(1)+"B" : n >= 1e6 ? (n/1e6).toFixed(1)+"M" : n >= 1e3 ? (n/1e3).toFixed(1)+"K" : String(n); }

  function mulberry32(a) {
    return function () {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      var t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  function hashStr(s) {
    var h = 0;
    for (var i = 0; i < s.length; i++) h = Math.imul(31, h) + s.charCodeAt(i) | 0;
    return h >>> 0;
  }

  /* ── architecture database ─────────────────────────────────── */
  var DB = {
    "llama2-7b":   spec(32000,4096,32,32,32,11008,4096,128),
    "llama2-13b":  spec(32000,5120,40,40,40,13824,4096,128),
    "llama2-70b":  spec(32000,8192,80,64,8,28672,4096,128),
    "llama3-8b":   spec(128256,4096,32,32,8,14336,8192,128),
    "llama3-70b":  spec(128256,8192,80,64,8,28672,8192,128),
    "llama3.1-8b": spec(128256,4096,32,32,8,14336,131072,128),
    "llama3.1-70b":spec(128256,8192,80,64,8,28672,131072,128),
    "llama3.1-405b":spec(128256,16384,126,128,8,53248,131072,128),
    "llama3.2-1b": spec(128256,2048,16,32,8,8192,131072,64),
    "llama3.2-3b": spec(128256,3072,28,24,8,8192,131072,128),
    "llama3.3-70b":spec(128256,8192,80,64,8,28672,131072,128),
    "mistral-7b":  spec(32000,4096,32,32,8,14336,32768,128),
    "mixtral-8x7b":spec(32000,4096,32,32,8,14336,32768,128),
    "mixtral-8x22b":spec(32000,6144,56,48,8,16384,65536,128),
    "phi2-2.7b":   spec(51200,2560,32,32,32,10240,2048,80,false),
    "phi3-3.8b":   spec(32064,3072,32,32,32,8192,4096,96),
    "phi3-14b":    spec(32064,5120,40,40,10,17920,4096,128),
    "gemma-2b":    spec(256128,2048,18,8,1,16384,8192,256),
    "gemma-7b":    spec(256128,3072,28,16,16,24576,8192,256),
    "gemma2-2b":   spec(256128,2304,26,8,4,9216,8192,256),
    "gemma2-9b":   spec(256128,3584,42,16,8,14336,8192,256),
    "gemma2-27b":  spec(256128,4608,46,32,16,36864,8192,128),
    "qwen2-0.5b":  spec(151936,896,24,14,2,4864,32768,64),
    "qwen2-1.5b":  spec(151936,1536,28,12,2,8960,32768,128),
    "qwen2-7b":    spec(152064,3584,28,28,4,18944,131072,128),
    "qwen2-72b":   spec(152064,8192,80,64,8,29568,131072,128),
    "qwen2.5-0.5b":spec(151936,896,24,14,2,4864,32768,64),
    "qwen2.5-1.5b":spec(151936,1536,28,12,2,8960,32768,128),
    "qwen2.5-3b":  spec(151936,2048,36,16,2,11008,32768,128),
    "qwen2.5-7b":  spec(152064,3584,28,28,4,18944,131072,128),
    "qwen2.5-14b": spec(152064,5120,48,40,8,13824,131072,128),
    "qwen2.5-32b": spec(152064,5120,64,40,8,27648,131072,128),
    "qwen2.5-72b": spec(152064,8192,80,64,8,29568,131072,128),
    "codellama-7b": spec(32016,4096,32,32,32,11008,16384,128),
    "codellama-13b":spec(32016,5120,40,40,40,13824,16384,128),
    "codellama-34b":spec(32016,8192,48,64,8,22016,16384,128),
    "codellama-70b":spec(32016,8192,80,64,8,28672,16384,128),
    "deepseek-coder-1.3b":spec(32256,2048,24,16,16,5504,16384,128),
    "deepseek-coder-6.7b":spec(32256,4096,32,32,32,11008,16384,128),
    "deepseek-coder-33b":spec(32256,7168,62,56,56,19456,16384,128),
    "starcoder2-3b":spec(49152,3072,30,24,2,12288,16384,128),
    "starcoder2-7b":spec(49152,4608,32,36,4,12288,16384,128),
    "starcoder2-15b":spec(49152,6144,40,48,4,24576,16384,128),
    "vicuna-7b":   spec(32000,4096,32,32,32,11008,4096,128),
    "vicuna-13b":  spec(32000,5120,40,40,40,13824,4096,128),
    "falcon-7b":   spec(65024,4544,32,71,1,4544,2048,64,false),
    "falcon-40b":  spec(65024,8192,60,128,8,32768,2048,64,false),
    "command-r-35b":spec(256000,8192,40,64,8,22528,131072,128),
    "yi-6b":       spec(64000,4096,32,32,4,11008,4096,128),
    "yi-34b":      spec(64000,7168,60,56,8,20480,4096,128),
  };

  /* ── resolve architecture from model name + param hints ──── */
  function resolveArchitecture(modelName, paramHints) {
    if (!modelName) return null;
    var name = modelName.toLowerCase().replace(/[_\s]/g, "-");
    var hints = (paramHints || []).map(function (h) { return h.toLowerCase().replace(/\s/g, ""); });
    for (var hi = 0; hi < hints.length; hi++) {
      var key = name + "-" + hints[hi];
      if (DB[key]) return Object.assign({ family: name, size: hints[hi] }, DB[key]);
    }
    for (var hi2 = 0; hi2 < hints.length; hi2++) {
      for (var k in DB) {
        if (k.startsWith(name) && k.endsWith(hints[hi2])) return Object.assign({ family: name, size: hints[hi2] }, DB[k]);
      }
    }
    for (var k2 in DB) {
      if (k2.startsWith(name)) return Object.assign({ family: name, size: k2.split("-").pop() }, DB[k2]);
    }
    var base = name.replace(/[\d.]/g, "").replace(/-+$/, "");
    for (var k3 in DB) {
      if (k3.startsWith(base)) return Object.assign({ family: base, size: k3.split("-").pop() }, DB[k3]);
    }
    var paramB = 0;
    for (var hi3 = 0; hi3 < hints.length; hi3++) {
      var m = hints[hi3].match(/([\d.]+)\s*b/);
      if (m) { paramB = parseFloat(m[1]); break; }
    }
    if (paramB > 0) return Object.assign({ family: name, size: paramB + "b" }, estimateArch(paramB));
    return Object.assign({ family: name, size: "?b" }, estimateArch(7));
  }

  function estimateArch(paramB) {
    var P = paramB * 1e9;
    var H = Math.round(Math.pow(P / 0.09, 1 / 3) / 128) * 128;
    H = Math.max(512, Math.min(H, 16384));
    var L = Math.max(4, Math.round(H / 128));
    var heads = Math.max(4, Math.round(H / 128));
    var kvHeads = Math.max(1, Math.round(heads / 4));
    var inter = Math.round(H * 8 / 3 / 256) * 256;
    return spec(32000, H, L, heads, kvHeads, inter, 4096, Math.round(H / heads));
  }

  /* ── weight texture generation ─────────────────────────────── */
  function weightRGB(v) {
    var r, g, b;
    if (v < 0.5) {
      var t = v / 0.5;
      r = 8 + 72 * (1 - t) | 0;
      g = 12 + 95 * (1 - t) | 0;
      b = 22 + 215 * (1 - t) | 0;
    } else {
      var t2 = (v - 0.5) / 0.5;
      r = 8 + 230 * t2 | 0;
      g = 12 + 105 * t2 | 0;
      b = 22 + 15 * t2 | 0;
    }
    return [r, g, b];
  }

  function makeWeightCanvas(res, seed) {
    var c = document.createElement("canvas");
    c.width = res; c.height = res;
    var ctx = c.getContext("2d");
    var img = ctx.createImageData(res, res);
    var rng = mulberry32(seed);
    for (var i = 0; i < res * res; i++) {
      var u1 = rng() + 1e-6, u2 = rng();
      var z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(6.2831853 * u2);
      var v = Math.max(0, Math.min(1, 0.5 + z * 0.22));
      var rgb = weightRGB(v);
      img.data[i * 4]     = rgb[0];
      img.data[i * 4 + 1] = rgb[1];
      img.data[i * 4 + 2] = rgb[2];
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    return c;
  }

  /* ── 3D dimension compression ──────────────────────────────── */
  function vd(dim) {
    return Math.max(0.25, Math.min(Math.pow(dim, 0.33) * 0.14, 5));
  }

  var COLORS = {
    embed:  0x7c3aed,
    norm:   0x4b5563,
    q:      0x3b82f6,
    k:      0x06b6d4,
    v:      0x10b981,
    attnOut:0x6366f1,
    gate:   0xf59e0b,
    up:     0xf97316,
    down:   0xef4444,
    output: 0xa855f7
  };

  var MAX_DISPLAY_LAYERS = 16;

  /* ── compute 3D layout ─────────────────────────────────────── */
  function computeLayout3D(arch) {
    var blocks = [];
    var y = 0;
    var BH = 0.2;
    var NH = 0.07;
    var GAP = 0.4;
    var LGAP = 0.8;
    var idx = 0;

    function add(id, label, type, dims, color, px, py, pz, sx, sy, sz, desc, layer) {
      var params = dims.length === 2 ? dims[0] * dims[1] : dims[0];
      blocks.push({
        id: id, label: label, type: type, dims: dims, color: color,
        px: px, py: py, pz: pz,
        sx: sx, sy: sy, sz: sz,
        desc: desc, layer: layer,
        params: params,
        seed: hashStr(arch.family + id),
        index: idx++
      });
    }

    // Token Embedding
    add("emb", "Token Embed", "embedding",
        [arch.vocab, arch.hidden], COLORS.embed,
        0, y, 0,
        vd(arch.hidden), BH * 2, vd(arch.vocab),
        "Token embedding  " + arch.vocab.toLocaleString() + " \u00d7 " + arch.hidden,
        null);
    y -= BH * 2 + GAP;

    // Determine visible layers
    var layerIndices = [];
    if (arch.layers <= MAX_DISPLAY_LAYERS) {
      for (var li = 0; li < arch.layers; li++) layerIndices.push(li);
    } else {
      for (var a = 0; a < 4; a++) layerIndices.push(a);
      layerIndices.push(-1);
      for (var b = arch.layers - 4; b < arch.layers; b++) layerIndices.push(b);
    }

    for (var lIdx = 0; lIdx < layerIndices.length; lIdx++) {
      var i = layerIndices[lIdx];

      if (i === -1) {
        add("gap", "\u00b7\u00b7\u00b7 " + (arch.layers - 8) + " more layers \u00b7\u00b7\u00b7", "gap",
            [1], 0x333333,
            0, y, 0,
            3, 0.04, 0.3,
            (arch.layers - 8) + " layers omitted for clarity",
            null);
        y -= 0.04 + LGAP;
        continue;
      }

      var lid = "l" + i;
      var kvDim = arch.kvHeads * arch.headDim;

      // LN1
      add(lid + "_ln1", "LN", "norm",
          [arch.hidden], COLORS.norm,
          0, y, 0,
          vd(arch.hidden) * 2.2, NH, 0.3,
          "Layer " + i + " pre-attention norm  dim=" + arch.hidden, i);
      y -= NH + GAP * 0.5;

      // Q K V side by side
      var qdim = arch.heads * arch.headDim;
      var qw = vd(qdim), kw = vd(kvDim), vw = vd(kvDim);
      var qd = vd(arch.hidden);
      var totalW = qw + kw + vw + 0.3;
      var startX = -totalW / 2;

      add(lid + "_q", "Q", "linear",
          [arch.hidden, qdim], COLORS.q,
          startX + qw / 2, y, 0,
          qw, BH, qd,
          "Query projection  " + arch.hidden + " \u2192 " + qdim, i);

      add(lid + "_k", "K", "linear",
          [arch.hidden, kvDim], COLORS.k,
          startX + qw + 0.15 + kw / 2, y, 0,
          kw, BH, qd,
          "Key projection  " + arch.hidden + " \u2192 " + kvDim, i);

      add(lid + "_v", "V", "linear",
          [arch.hidden, kvDim], COLORS.v,
          startX + qw + 0.3 + kw + vw / 2, y, 0,
          vw, BH, qd,
          "Value projection  " + arch.hidden + " \u2192 " + kvDim, i);

      y -= BH + GAP;

      // Attn Out
      add(lid + "_o", "Attn Out", "linear",
          [qdim, arch.hidden], COLORS.attnOut,
          0, y, 0,
          vd(arch.hidden), BH, vd(qdim),
          "Attention output  " + qdim + " \u2192 " + arch.hidden, i);
      y -= BH + GAP;

      // LN2
      add(lid + "_ln2", "LN", "norm",
          [arch.hidden], COLORS.norm,
          0, y, 0,
          vd(arch.hidden) * 2.2, NH, 0.3,
          "Layer " + i + " pre-FFN norm  dim=" + arch.hidden, i);
      y -= NH + GAP * 0.5;

      // FFN
      if (arch.gate) {
        var gw = vd(arch.inter), uw = vd(arch.inter);
        var ffnTotalW = gw + uw + 0.15;
        var ffnStartX = -ffnTotalW / 2;

        add(lid + "_gate", "Gate", "linear",
            [arch.hidden, arch.inter], COLORS.gate,
            ffnStartX + gw / 2, y, 0,
            gw, BH, vd(arch.hidden),
            "SwiGLU gate  " + arch.hidden + " \u2192 " + arch.inter, i);

        add(lid + "_up", "Up", "linear",
            [arch.hidden, arch.inter], COLORS.up,
            ffnStartX + gw + 0.15 + uw / 2, y, 0,
            uw, BH, vd(arch.hidden),
            "FFN up  " + arch.hidden + " \u2192 " + arch.inter, i);
        y -= BH + GAP;
      } else {
        add(lid + "_up", "Up", "linear",
            [arch.hidden, arch.inter], COLORS.up,
            0, y, 0,
            vd(arch.inter), BH, vd(arch.hidden),
            "FFN up  " + arch.hidden + " \u2192 " + arch.inter, i);
        y -= BH + GAP;
      }

      // Down
      add(lid + "_down", "Down", "linear",
          [arch.inter, arch.hidden], COLORS.down,
          0, y, 0,
          vd(arch.hidden), BH, vd(arch.inter),
          "FFN down  " + arch.inter + " \u2192 " + arch.hidden, i);
      y -= BH + LGAP;
    }

    // Final LN
    add("final_ln", "Final LN", "norm",
        [arch.hidden], COLORS.norm,
        0, y, 0,
        vd(arch.hidden) * 2.2, NH, 0.3,
        "Final layer norm  dim=" + arch.hidden, null);
    y -= NH + GAP;

    // LM Head
    add("lm_head", "LM Head", "output",
        [arch.hidden, arch.vocab], COLORS.output,
        0, y, 0,
        vd(arch.vocab), BH * 2, vd(arch.hidden),
        "LM head  " + arch.hidden + " \u2192 " + arch.vocab.toLocaleString(), null);
    y -= BH * 2;

    return {
      blocks: blocks,
      center: { x: 0, y: y / 2, z: 0 },
      totalHeight: Math.abs(y)
    };
  }

  /* ══════════════════════════════════════════════════════════ */
  /*  NetworkVisualizer – Three.js 3D renderer                  */
  /* ══════════════════════════════════════════════════════════ */
  function NetworkVisualizer(container, arch, modelName) {
    if (typeof THREE === "undefined") {
      container.innerHTML = '<p style="color:#888;text-align:center;padding:60px 20px;font-family:monospace">3D visualization requires Three.js. Please refresh the page.</p>';
      this.destroy = function () { container.innerHTML = ""; };
      return;
    }

    this.container = container;
    this.arch = arch;
    this.modelName = modelName || arch.family || "model";
    this.layout = computeLayout3D(arch);
    this.meshes = [];
    this.hoveredMesh = null;
    this.prevHovered = null;
    this.destroyed = false;
    this.animStart = performance.now();
    this.needsRender = true;

    // Camera orbit state
    this.theta = Math.PI / 4;
    this.phi = Math.PI / 3.2;
    this.radius = Math.max(6, this.layout.totalHeight * 0.5);
    this.targetPos = null; // set after THREE is confirmed

    this.isDragging = false;
    this.isPanning = false;
    this.lastMouseX = 0;
    this.lastMouseY = 0;
    this.mouseScreenX = 0;
    this.mouseScreenY = 0;

    this.init();
  }

  var NV = NetworkVisualizer.prototype;

  NV.init = function () {
    var w = this.container.clientWidth;
    var h = this.container.clientHeight || 600;

    // Scene
    this.scene = new THREE.Scene();

    // Gradient background
    var bgCanvas = document.createElement("canvas");
    bgCanvas.width = 2; bgCanvas.height = 512;
    var bgCtx = bgCanvas.getContext("2d");
    var grad = bgCtx.createLinearGradient(0, 0, 0, 512);
    grad.addColorStop(0, "#111827");
    grad.addColorStop(0.5, "#08090d");
    grad.addColorStop(1, "#0c1220");
    bgCtx.fillStyle = grad;
    bgCtx.fillRect(0, 0, 2, 512);
    this.scene.background = new THREE.CanvasTexture(bgCanvas);

    // Fog
    this.scene.fog = new THREE.FogExp2(0x08090d, 0.018);

    // Camera
    this.camera = new THREE.PerspectiveCamera(50, w / h, 0.1, 500);
    this.targetPos = new THREE.Vector3(
      this.layout.center.x,
      this.layout.center.y,
      this.layout.center.z
    );
    this.updateCameraPosition();

    // Renderer
    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      powerPreference: "high-performance"
    });
    this.renderer.setSize(w, h);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.3;

    this.container.innerHTML = "";
    this.container.appendChild(this.renderer.domElement);
    this.renderer.domElement.style.cursor = "grab";

    // Lighting
    this.setupLighting();

    // Create blocks
    this.createBlocks();

    // Ambient particles
    this.createParticles();

    // Raycaster
    this.raycaster = new THREE.Raycaster();
    this.mouseNDC = new THREE.Vector2(-999, -999);

    // Tooltip
    this.tooltip = document.createElement("div");
    this.tooltip.className = "network-tooltip";
    this.tooltip.style.display = "none";
    this.container.style.position = "relative";
    this.container.appendChild(this.tooltip);

    // HUD overlay
    this.hud = document.createElement("div");
    this.hud.style.cssText = "position:absolute;top:12px;left:14px;pointer-events:none;z-index:5;";
    this.hud.innerHTML =
      '<div style="font:700 14px \'Space Grotesk\',sans-serif;color:rgba(230,237,243,0.88)">' +
      this.escapeHtml(this.modelName) + '</div>' +
      '<div style="font:400 11px \'IBM Plex Mono\',monospace;color:rgba(139,148,158,0.7);margin-top:2px">' +
      this.arch.layers + ' layers \u00b7 ' + this.arch.heads + ' heads \u00b7 hidden ' +
      this.arch.hidden + ' \u00b7 FFN ' + this.arch.inter + '</div>';
    this.container.appendChild(this.hud);

    // Controls hint
    this.hint = document.createElement("div");
    this.hint.style.cssText = "position:absolute;bottom:10px;right:14px;pointer-events:none;z-index:5;" +
      "font:400 11px 'IBM Plex Mono',monospace;color:rgba(139,148,158,0.35);text-align:right;";
    this.hint.textContent = "Left-drag rotate \u00b7 Right-drag pan \u00b7 Scroll zoom";
    this.container.appendChild(this.hint);

    // Events
    this._onWheel = this.onWheel.bind(this);
    this._onPointerDown = this.onPointerDown.bind(this);
    this._onPointerMove = this.onPointerMove.bind(this);
    this._onPointerUp = this.onPointerUp.bind(this);
    this._onResize = this.onResize.bind(this);
    this._onContextMenu = function (e) { e.preventDefault(); };

    var el = this.renderer.domElement;
    el.addEventListener("wheel", this._onWheel, { passive: false });
    el.addEventListener("pointerdown", this._onPointerDown);
    el.addEventListener("pointermove", this._onPointerMove);
    el.addEventListener("pointerup", this._onPointerUp);
    el.addEventListener("pointerleave", this._onPointerUp);
    el.addEventListener("contextmenu", this._onContextMenu);
    window.addEventListener("resize", this._onResize);

    // Start render loop
    this.animate();
  };

  NV.setupLighting = function () {
    this.scene.add(new THREE.AmbientLight(0xb0c4de, 0.5));

    var dir = new THREE.DirectionalLight(0xffffff, 0.9);
    dir.position.set(8, 15, 10);
    this.scene.add(dir);

    var fill = new THREE.DirectionalLight(0x4488ff, 0.25);
    fill.position.set(-5, -10, 5);
    this.scene.add(fill);

    var rim = new THREE.DirectionalLight(0xff8844, 0.2);
    rim.position.set(-8, 5, -8);
    this.scene.add(rim);
  };

  NV.createBlocks = function () {
    var blocks = this.layout.blocks;
    var texCache = {};

    for (var i = 0; i < blocks.length; i++) {
      var b = blocks[i];
      var geo = new THREE.BoxGeometry(b.sx, b.sy, b.sz);

      // Weight texture
      var texRes = b.type === "norm" || b.type === "gap" ? 16 : 48;
      var cacheKey = b.seed + "_" + texRes;
      if (!texCache[cacheKey]) {
        texCache[cacheKey] = makeWeightCanvas(texRes, b.seed);
      }
      var tex = new THREE.CanvasTexture(texCache[cacheKey]);
      tex.minFilter = THREE.NearestFilter;
      tex.magFilter = THREE.NearestFilter;

      var color = new THREE.Color(b.color);
      var metalness = b.type === "embedding" || b.type === "output" ? 0.3 : b.type === "norm" ? 0.0 : 0.15;
      var roughness = b.type === "norm" ? 0.9 : 0.5;
      var emissiveBase = b.type === "gap" ? 0.05 : 0.12;

      var mat = new THREE.MeshStandardMaterial({
        map: tex,
        color: color,
        metalness: metalness,
        roughness: roughness,
        emissive: color,
        emissiveIntensity: emissiveBase
      });

      var mesh = new THREE.Mesh(geo, mat);
      mesh.position.set(b.px, b.py, b.pz);
      mesh.scale.set(0, 0, 0);
      mesh.userData = b;
      mesh.userData._emissiveBase = emissiveBase;

      this.scene.add(mesh);
      this.meshes.push(mesh);

      // Edge wireframe
      if (b.type !== "gap") {
        var edges = new THREE.EdgesGeometry(geo);
        var edgeMat = new THREE.LineBasicMaterial({
          color: 0xffffff,
          transparent: true,
          opacity: 0.1
        });
        mesh.add(new THREE.LineSegments(edges, edgeMat));
      }
    }
  };

  NV.createParticles = function () {
    var count = 300;
    var positions = new Float32Array(count * 3);
    var rng = mulberry32(42);
    var hh = this.layout.totalHeight;
    for (var i = 0; i < count; i++) {
      positions[i * 3]     = (rng() - 0.5) * 30;
      positions[i * 3 + 1] = -rng() * hh * 1.2;
      positions[i * 3 + 2] = (rng() - 0.5) * 30;
    }
    var geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    var mat = new THREE.PointsMaterial({
      color: 0x4488ff,
      size: 0.04,
      transparent: true,
      opacity: 0.2,
      sizeAttenuation: true
    });
    this.particles = new THREE.Points(geo, mat);
    this.scene.add(this.particles);
  };

  NV.updateCameraPosition = function () {
    var x = this.targetPos.x + this.radius * Math.sin(this.phi) * Math.cos(this.theta);
    var y = this.targetPos.y + this.radius * Math.cos(this.phi);
    var z = this.targetPos.z + this.radius * Math.sin(this.phi) * Math.sin(this.theta);
    this.camera.position.set(x, y, z);
    this.camera.lookAt(this.targetPos);
    this.needsRender = true;
  };

  /* ── pointer events ────────────────────────────────────────── */
  NV.onWheel = function (e) {
    e.preventDefault();
    var factor = e.deltaY > 0 ? 1.08 : 1 / 1.08;
    this.radius = Math.max(2, Math.min(this.radius * factor, 200));
    this.updateCameraPosition();
  };

  NV.onPointerDown = function (e) {
    if (e.button === 0) this.isDragging = true;
    else if (e.button === 2) this.isPanning = true;
    this.lastMouseX = e.clientX;
    this.lastMouseY = e.clientY;
    this.renderer.domElement.setPointerCapture(e.pointerId);
    this.renderer.domElement.style.cursor = "grabbing";
  };

  NV.onPointerMove = function (e) {
    var rect = this.renderer.domElement.getBoundingClientRect();
    this.mouseScreenX = e.clientX - rect.left;
    this.mouseScreenY = e.clientY - rect.top;
    this.mouseNDC.x = (this.mouseScreenX / rect.width) * 2 - 1;
    this.mouseNDC.y = -(this.mouseScreenY / rect.height) * 2 + 1;

    var dx = e.clientX - this.lastMouseX;
    var dy = e.clientY - this.lastMouseY;
    this.lastMouseX = e.clientX;
    this.lastMouseY = e.clientY;

    if (this.isDragging) {
      this.theta -= dx * 0.005;
      this.phi = Math.max(0.15, Math.min(Math.PI - 0.15, this.phi - dy * 0.005));
      this.updateCameraPosition();
    } else if (this.isPanning) {
      var right = new THREE.Vector3();
      var up = new THREE.Vector3();
      var forward = new THREE.Vector3();
      this.camera.getWorldDirection(forward);
      right.crossVectors(forward, this.camera.up).normalize();
      up.crossVectors(right, forward).normalize();
      var speed = this.radius * 0.002;
      this.targetPos.addScaledVector(right, -dx * speed);
      this.targetPos.addScaledVector(up, dy * speed);
      this.updateCameraPosition();
    }

    this.needsRender = true;
  };

  NV.onPointerUp = function () {
    this.isDragging = false;
    this.isPanning = false;
    this.renderer.domElement.style.cursor = "grab";
  };

  NV.onResize = function () {
    var w = this.container.clientWidth;
    var h = this.container.clientHeight || 600;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
    this.needsRender = true;
  };

  /* ── animation loop ────────────────────────────────────────── */
  NV.animate = function () {
    if (this.destroyed) return;
    requestAnimationFrame(this.animate.bind(this));

    var now = performance.now();
    var elapsed = now - this.animStart;
    var animating = false;

    // Entrance animation
    for (var i = 0; i < this.meshes.length; i++) {
      var m = this.meshes[i];
      var delay = m.userData.index * 18;
      var t = Math.max(0, Math.min(1, (elapsed - delay) / 450));
      if (t < 1) animating = true;
      var s = t * t * (3 - 2 * t); // smoothstep
      m.scale.set(s, s, s);
    }

    if (animating) this.needsRender = true;

    // Particles rotation
    if (this.particles) {
      this.particles.rotation.y += 0.0003;
      this.needsRender = true;
    }

    // Hover detection
    this.doHoverDetection();

    // Render only when needed
    if (this.needsRender) {
      this.renderer.render(this.scene, this.camera);
      this.needsRender = animating || !!this.particles;
    }
  };

  /* ── hover detection ───────────────────────────────────────── */
  NV.doHoverDetection = function () {
    this.raycaster.setFromCamera(this.mouseNDC, this.camera);
    var intersects = this.raycaster.intersectObjects(this.meshes);
    var newHovered = intersects.length > 0 ? intersects[0].object : null;

    if (newHovered !== this.prevHovered) {
      // Restore previous
      if (this.prevHovered) {
        this.prevHovered.material.emissiveIntensity = this.prevHovered.userData._emissiveBase;
      }
      // Highlight new
      if (newHovered && newHovered.scale.x >= 0.95) {
        newHovered.material.emissiveIntensity = 0.55;
        this.hoveredMesh = newHovered;
      } else {
        this.hoveredMesh = null;
      }
      this.prevHovered = newHovered;
      this.needsRender = true;
    }

    this.updateTooltip();
  };

  /* ── tooltip ───────────────────────────────────────────────── */
  NV.updateTooltip = function () {
    if (!this.hoveredMesh || this.hoveredMesh.scale.x < 0.95) {
      this.tooltip.style.display = "none";
      return;
    }

    var b = this.hoveredMesh.userData;
    var dimStr = b.dims.length === 2
      ? b.dims[0].toLocaleString() + " \u00d7 " + b.dims[1].toLocaleString()
      : b.dims[0].toLocaleString();
    var layerStr = b.layer !== null && b.layer !== undefined ? "Layer " + b.layer + " \u00b7 " : "";

    this.tooltip.innerHTML =
      "<strong>" + layerStr + this.escapeHtml(b.label) + "</strong><br>" +
      '<span style="opacity:0.7">' + this.escapeHtml(b.desc) + "</span><br>" +
      '<span style="color:#58a6ff">' + dimStr + "</span>  \u00b7  " + fmtNum(b.params) + " params";

    var tx = this.mouseScreenX + 16;
    var ty = this.mouseScreenY - 10;
    var cw = this.container.clientWidth;
    if (tx + 280 > cw) tx = this.mouseScreenX - 280;
    if (ty < 0) ty = 10;
    this.tooltip.style.display = "block";
    this.tooltip.style.left = tx + "px";
    this.tooltip.style.top = ty + "px";
  };

  NV.escapeHtml = function (s) {
    var d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  };

  /* ── destroy ───────────────────────────────────────────────── */
  NV.destroy = function () {
    this.destroyed = true;
    var el = this.renderer.domElement;
    el.removeEventListener("wheel", this._onWheel);
    el.removeEventListener("pointerdown", this._onPointerDown);
    el.removeEventListener("pointermove", this._onPointerMove);
    el.removeEventListener("pointerup", this._onPointerUp);
    el.removeEventListener("pointerleave", this._onPointerUp);
    el.removeEventListener("contextmenu", this._onContextMenu);
    window.removeEventListener("resize", this._onResize);

    for (var i = 0; i < this.meshes.length; i++) {
      var m = this.meshes[i];
      m.geometry.dispose();
      if (m.material.map) m.material.map.dispose();
      m.material.dispose();
    }
    if (this.particles) {
      this.particles.geometry.dispose();
      this.particles.material.dispose();
    }
    this.renderer.dispose();
    this.container.innerHTML = "";
  };

  /* ── exports ───────────────────────────────────────────────── */
  window.NetworkVisualizer = NetworkVisualizer;
  window.resolveArchitecture = resolveArchitecture;
})();
// ══════════════════════════════════════════════════════════════
// LLM Network Visualizer – bbycroft.net-inspired renderer
// Interactive transformer architecture on a pannable/zoomable canvas.
// Weight textures are seeded per-model so each model looks unique.
// ══════════════════════════════════════════════════════════════
(function () {
  "use strict";

  /* ── tiny helpers ──────────────────────────────────────────── */
  function spec(v, h, l, hd, kv, inter, seq, hdim, gate) {
    return { vocab: v, hidden: h, layers: l, heads: hd, kvHeads: kv, inter: inter, seq: seq, headDim: hdim, gate: gate !== false };
  }
  function fmtNum(n) { return n >= 1e9 ? (n/1e9).toFixed(1)+"B" : n >= 1e6 ? (n/1e6).toFixed(1)+"M" : n >= 1e3 ? (n/1e3).toFixed(1)+"K" : String(n); }

  /* ── architecture database ─────────────────────────────────── */
  const DB = {
    "llama2-7b":   spec(32000,4096,32,32,32,11008,4096,128),
    "llama2-13b":  spec(32000,5120,40,40,40,13824,4096,128),
    "llama2-70b":  spec(32000,8192,80,64,8,28672,4096,128),
    "llama3-8b":   spec(128256,4096,32,32,8,14336,8192,128),
    "llama3-70b":  spec(128256,8192,80,64,8,28672,8192,128),
    "llama3.1-8b": spec(128256,4096,32,32,8,14336,131072,128),
    "llama3.1-70b":spec(128256,8192,80,64,8,28672,131072,128),
    "llama3.1-405b":spec(128256,16384,126,128,8,53248,131072,128),
    "llama3.2-1b": spec(128256,2048,16,32,8,8192,131072,64),
    "llama3.2-3b": spec(128256,3072,28,24,8,8192,131072,128),
    "llama3.3-70b":spec(128256,8192,80,64,8,28672,131072,128),
    "mistral-7b":  spec(32000,4096,32,32,8,14336,32768,128),
    "mixtral-8x7b":spec(32000,4096,32,32,8,14336,32768,128),
    "mixtral-8x22b":spec(32000,6144,56,48,8,16384,65536,128),
    "phi2-2.7b":   spec(51200,2560,32,32,32,10240,2048,80,false),
    "phi3-3.8b":   spec(32064,3072,32,32,32,8192,4096,96),
    "phi3-14b":    spec(32064,5120,40,40,10,17920,4096,128),
    "gemma-2b":    spec(256128,2048,18,8,1,16384,8192,256),
    "gemma-7b":    spec(256128,3072,28,16,16,24576,8192,256),
    "gemma2-2b":   spec(256128,2304,26,8,4,9216,8192,256),
    "gemma2-9b":   spec(256128,3584,42,16,8,14336,8192,256),
    "gemma2-27b":  spec(256128,4608,46,32,16,36864,8192,128),
    "qwen2-0.5b":  spec(151936,896,24,14,2,4864,32768,64),
    "qwen2-1.5b":  spec(151936,1536,28,12,2,8960,32768,128),
    "qwen2-7b":    spec(152064,3584,28,28,4,18944,131072,128),
    "qwen2-72b":   spec(152064,8192,80,64,8,29568,131072,128),
    "qwen2.5-0.5b":spec(151936,896,24,14,2,4864,32768,64),
    "qwen2.5-1.5b":spec(151936,1536,28,12,2,8960,32768,128),
    "qwen2.5-3b":  spec(151936,2048,36,16,2,11008,32768,128),
    "qwen2.5-7b":  spec(152064,3584,28,28,4,18944,131072,128),
    "qwen2.5-14b": spec(152064,5120,48,40,8,13824,131072,128),
    "qwen2.5-32b": spec(152064,5120,64,40,8,27648,131072,128),
    "qwen2.5-72b": spec(152064,8192,80,64,8,29568,131072,128),
    "codellama-7b": spec(32016,4096,32,32,32,11008,16384,128),
    "codellama-13b":spec(32016,5120,40,40,40,13824,16384,128),
    "codellama-34b":spec(32016,8192,48,64,8,22016,16384,128),
    "codellama-70b":spec(32016,8192,80,64,8,28672,16384,128),
    "deepseek-coder-1.3b":spec(32256,2048,24,16,16,5504,16384,128),
    "deepseek-coder-6.7b":spec(32256,4096,32,32,32,11008,16384,128),
    "deepseek-coder-33b":spec(32256,7168,62,56,56,19456,16384,128),
    "starcoder2-3b":spec(49152,3072,30,24,2,12288,16384,128),
    "starcoder2-7b":spec(49152,4608,32,36,4,12288,16384,128),
    "starcoder2-15b":spec(49152,6144,40,48,4,24576,16384,128),
    "vicuna-7b":   spec(32000,4096,32,32,32,11008,4096,128),
    "vicuna-13b":  spec(32000,5120,40,40,40,13824,4096,128),
    "falcon-7b":   spec(65024,4544,32,71,1,4544,2048,64,false),
    "falcon-40b":  spec(65024,8192,60,128,8,32768,2048,64,false),
    "command-r-35b":spec(256000,8192,40,64,8,22528,131072,128),
    "yi-6b":       spec(64000,4096,32,32,4,11008,4096,128),
    "yi-34b":      spec(64000,7168,60,56,8,20480,4096,128),
  };

  /* ── resolve architecture from model name + param hints ──── */
  function resolveArchitecture(modelName, paramHints) {
    if (!modelName) return null;
    const name = modelName.toLowerCase().replace(/[_\s]/g, "-");
    const hints = (paramHints || []).map(function (h) { return h.toLowerCase().replace(/\s/g, ""); });

    // try direct keys
    for (const hint of hints) {
      var key = name + "-" + hint;
      if (DB[key]) return Object.assign({ family: name, size: hint }, DB[key]);
    }
    // prefix match
    for (const hint of hints) {
      for (const k in DB) {
        if (k.startsWith(name) && k.endsWith(hint)) return Object.assign({ family: name, size: hint }, DB[k]);
      }
    }
    // family-only match with first hint
    for (const k in DB) {
      if (k.startsWith(name)) return Object.assign({ family: name, size: k.split("-").pop() }, DB[k]);
    }
    // fuzzy: strip digits/dots from name, try common families
    var base = name.replace(/[\d.]/g, "").replace(/-+$/, "");
    for (const k in DB) {
      if (k.startsWith(base)) return Object.assign({ family: base, size: k.split("-").pop() }, DB[k]);
    }
    // estimate from param hints
    var paramB = 0;
    for (const h of hints) {
      var m = h.match(/([\d.]+)\s*b/);
      if (m) { paramB = parseFloat(m[1]); break; }
    }
    if (paramB > 0) return Object.assign({ family: name, size: paramB + "b" }, estimateArch(paramB));
    return Object.assign({ family: name, size: "?b" }, estimateArch(7));
  }

  function estimateArch(paramB) {
    var P = paramB * 1e9;
    var H = Math.round(Math.pow(P / 0.09, 1 / 3) / 128) * 128;
    H = Math.max(512, Math.min(H, 16384));
    var L = Math.max(4, Math.round(H / 128));
    var heads = Math.max(4, Math.round(H / 128));
    var kvHeads = Math.max(1, Math.round(heads / 4));
    var inter = Math.round(H * 8 / 3 / 256) * 256;
    return spec(32000, H, L, heads, kvHeads, inter, 4096, Math.round(H / heads));
  }

  /* ── Mulberry32 PRNG ───────────────────────────────────────── */
  function mulberry32(a) {
    return function () {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      var t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }
  function hashStr(s) {
    var h = 0;
    for (var i = 0; i < s.length; i++) h = Math.imul(31, h) + s.charCodeAt(i) | 0;
    return h >>> 0;
  }

  /* ── weight value → RGB (diverging blue–dark–orange) ───────── */
  function weightRGB(v) {
    // v in [0,1];  0 = negative, 0.5 = near zero, 1 = positive
    var r, g, b;
    if (v < 0.5) {
      var t = v / 0.5;                         // 0→1 as v goes 0→0.5
      r = 20 + (80 - 20) * (1 - t) | 0;       // blue-ish
      g = 30 + (140 - 30) * (1 - t) | 0;
      b = 40 + (240 - 40) * (1 - t) | 0;
    } else {
      var t2 = (v - 0.5) / 0.5;               // 0→1 as v goes 0.5→1
      r = 20 + (240 - 20) * t2 | 0;           // orange-ish
      g = 30 + (140 - 30) * t2 | 0;
      b = 40 + (50  - 40) * t2 | 0;
    }
    return [r, g, b];
  }

  /* ── generate weight texture as offscreen canvas ───────────── */
  function makeTexture(res, seed) {
    var c = document.createElement("canvas");
    c.width = res; c.height = res;
    var ctx = c.getContext("2d");
    var img = ctx.createImageData(res, res);
    var rng = mulberry32(seed);
    for (var i = 0; i < res * res; i++) {
      // Box-Muller for normal-ish distribution centred at 0.5
      var u1 = rng() + 1e-6, u2 = rng();
      var z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(6.2831853 * u2);
      var v = Math.max(0, Math.min(1, 0.5 + z * 0.18));
      var rgb = weightRGB(v);
      img.data[i * 4]     = rgb[0];
      img.data[i * 4 + 1] = rgb[1];
      img.data[i * 4 + 2] = rgb[2];
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    return c;
  }

  /* ── visual dimension: sqrt-compressed so huge dims stay sane  */
  function vd(dim) { return Math.round(6 + Math.min(Math.sqrt(dim) * 0.7, 180)); }

  /* ── component colour palette ──────────────────────────────── */
  var COLOURS = {
    embed:  "#7c3aed",
    norm:   "#4b5563",
    q:      "#3b82f6",
    k:      "#06b6d4",
    v:      "#10b981",
    attnOut:"#6366f1",
    gate:   "#f59e0b",
    up:     "#f97316",
    down:   "#ef4444",
    output: "#a855f7",
  };

  /* ── compute layout → elements[] + connections[] ───────────── */
  function computeLayout(arch) {
    var GAP = 10, SGAP = 5, LGAP = 28, SECGAP = 40;
    var elems = [], conns = [];
    var s = function (d) { return vd(d); };
    var x = 0, prevId = null;

    // helper: push an element
    function add(id, type, label, dims, ex, ey, ew, eh, col, layer, desc) {
      var params = dims.length === 2 ? dims[0] * dims[1] : dims[0];
      elems.push({ id: id, type: type, label: label, dims: dims, x: ex, y: ey, w: ew, h: eh,
                   color: col, layer: layer, desc: desc, params: params,
                   seed: hashStr(arch.family + id) });
    }
    function connect(a, b, t) { conns.push([a, b, t || "inner"]); }

    // ─── Token Embedding ────────────────────────────────────
    var embW = s(arch.hidden), embH = s(arch.vocab);
    add("emb", "embedding", "Token\nEmbed", [arch.vocab, arch.hidden], x, 0, embW, embH, COLOURS.embed, null,
        "Token embedding  " + arch.vocab.toLocaleString() + " × " + arch.hidden);
    prevId = "emb";
    x += embW + SECGAP;

    // helper for total layer height (computed once to align layers)
    var kvDim = arch.kvHeads * arch.headDim;
    var normH = 6;
    var qH = s(arch.heads * arch.headDim), kH = s(kvDim), vH2 = s(kvDim);
    var qkvH = Math.max(qH, kH, vH2);
    var oH = s(arch.hidden);
    var gateH = s(arch.inter), upH = s(arch.inter), downH = s(arch.hidden);
    var ffnTopH = arch.gate ? Math.max(gateH, upH) : upH;
    var layerH = normH + GAP + qkvH + GAP + oH + GAP + normH + GAP + ffnTopH + GAP + downH;

    // centre embedding vertically to match layer height
    elems[0].y = Math.max(0, (layerH - embH) / 2);

    // ─── Transformer Layers ─────────────────────────────────
    for (var i = 0; i < arch.layers; i++) {
      var lx = x, ly = 0;
      var lid = "l" + i;

      // Layer background bounds (used for drawing)
      var qW = s(arch.hidden), kW = s(arch.hidden), vW = s(arch.hidden);
      var layerW = Math.max(qW + SGAP + kW + SGAP + vW,
                            s(arch.hidden),
                            arch.gate ? s(arch.hidden) + SGAP + s(arch.hidden) : s(arch.hidden),
                            s(arch.inter));

      // Attention Norm
      add(lid+"_ln1", "norm", "LN", [arch.hidden], lx, ly, layerW, normH, COLOURS.norm, i, "Pre-attention layer norm");
      connect(prevId, lid+"_ln1", i === 0 ? "major" : "inter");
      ly += normH + GAP;

      // Q K V
      var qx = lx, kx = lx + qW + SGAP, vx = kx + kW + SGAP;
      add(lid+"_q", "linear", "Q", [arch.hidden, arch.heads*arch.headDim], qx, ly, qW, qH, COLOURS.q, i,
          "Query projection  " + arch.hidden + " → " + arch.heads*arch.headDim);
      add(lid+"_k", "linear", "K", [arch.hidden, kvDim], kx, ly, kW, kH, COLOURS.k, i,
          "Key projection  " + arch.hidden + " → " + kvDim);
      add(lid+"_v", "linear", "V", [arch.hidden, kvDim], vx, ly, vW, vH2, COLOURS.v, i,
          "Value projection  " + arch.hidden + " → " + kvDim);
      connect(lid+"_ln1", lid+"_q", "inner");
      connect(lid+"_ln1", lid+"_k", "inner");
      connect(lid+"_ln1", lid+"_v", "inner");
      ly += qkvH + GAP;

      // Attention output
      add(lid+"_o", "linear", "Attn\nOut", [arch.heads*arch.headDim, arch.hidden], lx, ly, s(arch.hidden), oH, COLOURS.attnOut, i,
          "Attention output  " + arch.heads*arch.headDim + " → " + arch.hidden);
      connect(lid+"_q", lid+"_o", "inner");
      connect(lid+"_k", lid+"_o", "inner");
      connect(lid+"_v", lid+"_o", "inner");
      ly += oH + GAP;

      // FFN Norm
      add(lid+"_ln2", "norm", "LN", [arch.hidden], lx, ly, layerW, normH, COLOURS.norm, i, "Pre-FFN layer norm");
      connect(lid+"_o", lid+"_ln2", "inner");
      ly += normH + GAP;

      // FFN
      if (arch.gate) {
        var gw = s(arch.hidden), uw = s(arch.hidden);
        add(lid+"_gate", "linear", "Gate", [arch.hidden, arch.inter], lx, ly, gw, gateH, COLOURS.gate, i,
            "SwiGLU gate  " + arch.hidden + " → " + arch.inter);
        add(lid+"_up", "linear", "Up", [arch.hidden, arch.inter], lx + gw + SGAP, ly, uw, upH, COLOURS.up, i,
            "FFN up projection  " + arch.hidden + " → " + arch.inter);
        connect(lid+"_ln2", lid+"_gate", "inner");
        connect(lid+"_ln2", lid+"_up", "inner");
        ly += ffnTopH + GAP;

        add(lid+"_down", "linear", "Down", [arch.inter, arch.hidden], lx, ly, s(arch.inter), downH, COLOURS.down, i,
            "FFN down projection  " + arch.inter + " → " + arch.hidden);
        connect(lid+"_gate", lid+"_down", "inner");
        connect(lid+"_up", lid+"_down", "inner");
      } else {
        add(lid+"_up", "linear", "Up", [arch.hidden, arch.inter], lx, ly, s(arch.hidden), upH, COLOURS.up, i,
            "FFN up  " + arch.hidden + " → " + arch.inter);
        connect(lid+"_ln2", lid+"_up", "inner");
        ly += ffnTopH + GAP;

        add(lid+"_down", "linear", "Down", [arch.inter, arch.hidden], lx, ly, s(arch.inter), downH, COLOURS.down, i,
            "FFN down  " + arch.inter + " → " + arch.hidden);
        connect(lid+"_up", lid+"_down", "inner");
      }

      prevId = lid + "_down";
      x += layerW + LGAP;
    }

    // ─── Final Norm ──────────────────────────────────────────
    add("final_ln", "norm", "Final\nLN", [arch.hidden], x, layerH / 2 - 3, s(arch.hidden), normH, COLOURS.norm, null,
        "Final layer norm  " + arch.hidden);
    connect(prevId, "final_ln", "major");
    x += s(arch.hidden) + SECGAP;

    // ─── LM Head ────────────────────────────────────────────
    var lmW = s(arch.hidden), lmH2 = s(arch.vocab);
    add("lm_head", "output", "LM\nHead", [arch.hidden, arch.vocab], x, Math.max(0, (layerH - lmH2) / 2), lmW, lmH2, COLOURS.output, null,
        "Language model head  " + arch.hidden + " → " + arch.vocab.toLocaleString());
    connect("final_ln", "lm_head", "major");
    x += lmW;

    // Build element map
    var elemMap = {};
    for (var j = 0; j < elems.length; j++) elemMap[elems[j].id] = elems[j];

    // Compute bounding box
    var minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
    for (var j2 = 0; j2 < elems.length; j2++) {
      var e = elems[j2];
      if (e.x < minX) minX = e.x;
      if (e.y < minY) minY = e.y;
      if (e.x + e.w > maxX) maxX = e.x + e.w;
      if (e.y + e.h > maxY) maxY = e.y + e.h;
    }

    return {
      elements: elems,
      connections: conns,
      elemMap: elemMap,
      bounds: { x: minX - 20, y: minY - 30, w: maxX - minX + 40, h: maxY - minY + 60 },
      arch: arch
    };
  }

  /* ══════════════════════════════════════════════════════════ */
  /*  NetworkVisualizer class                                   */
  /* ══════════════════════════════════════════════════════════ */
  function NetworkVisualizer(container, arch, modelName) {
    this.container = container;
    this.arch = arch;
    this.modelName = modelName || arch.family || "model";
    this.layout = computeLayout(arch);

    // Camera
    this.zoom = 1;
    this.panX = 0;
    this.panY = 0;

    // Interaction state
    this.dragging = false;
    this.dragStartX = 0;
    this.dragStartY = 0;
    this.dragPanX = 0;
    this.dragPanY = 0;
    this.hoveredElement = null;
    this.mouseScreenX = 0;
    this.mouseScreenY = 0;

    // Texture cache
    this.texCache = {};

    // Canvas
    this.canvas = null;
    this.ctx = null;
    this.width = 0;
    this.height = 0;
    this.dpr = 1;
    this._raf = null;

    // Bound handlers
    this._onWheel = this.onWheel.bind(this);
    this._onPointerDown = this.onPointerDown.bind(this);
    this._onPointerMove = this.onPointerMove.bind(this);
    this._onPointerUp = this.onPointerUp.bind(this);
    this._onResize = this.onResize.bind(this);

    this.init();
  }

  var NV = NetworkVisualizer.prototype;

  NV.init = function () {
    // Create canvas
    this.canvas = document.createElement("canvas");
    this.canvas.style.cssText = "width:100%;height:100%;display:block;";
    this.container.innerHTML = "";
    this.container.appendChild(this.canvas);

    // Tooltip div
    this.tooltip = document.createElement("div");
    this.tooltip.className = "network-tooltip";
    this.tooltip.style.display = "none";
    this.container.appendChild(this.tooltip);

    this.ctx = this.canvas.getContext("2d");
    this.onResize();
    this.zoomToFit();

    // Events
    this.canvas.addEventListener("wheel", this._onWheel, { passive: false });
    this.canvas.addEventListener("pointerdown", this._onPointerDown);
    this.canvas.addEventListener("pointermove", this._onPointerMove);
    this.canvas.addEventListener("pointerup", this._onPointerUp);
    this.canvas.addEventListener("pointerleave", this._onPointerUp);
    window.addEventListener("resize", this._onResize);

    this.markDirty();

    // Re-render once fonts load
    var self = this;
    if (document.fonts && document.fonts.ready) {
      document.fonts.ready.then(function () { self.markDirty(); });
    }
  };

  NV.onResize = function () {
    var rect = this.container.getBoundingClientRect();
    this.dpr = window.devicePixelRatio || 1;
    this.width = rect.width;
    this.height = rect.height;
    this.canvas.width = rect.width * this.dpr;
    this.canvas.height = rect.height * this.dpr;
    this.markDirty();
  };

  NV.markDirty = function () {
    if (this._raf) return;
    var self = this;
    this._raf = requestAnimationFrame(function () {
      self._raf = null;
      self.render();
    });
  };

  NV.zoomToFit = function () {
    var b = this.layout.bounds;
    var zx = this.width / b.w;
    var zy = this.height / b.h;
    this.zoom = Math.min(zx, zy) * 0.92;
    this.panX = (this.width - b.w * this.zoom) / 2 - b.x * this.zoom;
    this.panY = (this.height - b.h * this.zoom) / 2 - b.y * this.zoom;
  };

  NV.worldToScreen = function (wx, wy) {
    return [wx * this.zoom + this.panX, wy * this.zoom + this.panY];
  };

  NV.screenToWorld = function (sx, sy) {
    return [(sx - this.panX) / this.zoom, (sy - this.panY) / this.zoom];
  };

  /* ── render ────────────────────────────────────────────────── */
  NV.render = function () {
    var ctx = this.ctx;
    var W = this.width, H = this.height;
    var dpr = this.dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    // Background
    ctx.fillStyle = "#0d1117";
    ctx.fillRect(0, 0, W, H);

    // Apply camera
    ctx.save();
    ctx.translate(this.panX, this.panY);
    ctx.scale(this.zoom, this.zoom);

    // Layer backgrounds
    this.drawLayerBackgrounds(ctx);

    // Connections
    this.drawConnections(ctx);

    // Elements
    var elems = this.layout.elements;
    for (var i = 0; i < elems.length; i++) {
      this.drawElement(ctx, elems[i]);
    }

    ctx.restore();

    // HUD (screen space)
    this.drawHUD(ctx);

    // Tooltip
    this.updateTooltip();
  };

  /* ── layer backgrounds ─────────────────────────────────────── */
  NV.drawLayerBackgrounds = function (ctx) {
    if (this.zoom < 0.06) return;
    var arch = this.arch;
    var elems = this.layout.elements;

    for (var li = 0; li < arch.layers; li++) {
      var minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
      for (var j = 0; j < elems.length; j++) {
        var e = elems[j];
        if (e.layer !== li) continue;
        if (e.x < minX) minX = e.x;
        if (e.y < minY) minY = e.y;
        if (e.x + e.w > maxX) maxX = e.x + e.w;
        if (e.y + e.h > maxY) maxY = e.y + e.h;
      }
      if (minX === Infinity) continue;

      var pad = 6;
      ctx.fillStyle = "rgba(22,27,34,0.55)";
      ctx.beginPath();
      this.roundRect(ctx, minX - pad, minY - pad - 14, maxX - minX + pad * 2, maxY - minY + pad * 2 + 14, 4);
      ctx.fill();
      ctx.strokeStyle = "rgba(48,54,61,0.5)";
      ctx.lineWidth = 1 / this.zoom;
      ctx.stroke();

      // Layer label
      if (this.zoom > 0.1) {
        var fs = Math.min(10, 10 / this.zoom);
        ctx.font = "600 " + fs + "px 'IBM Plex Mono',monospace";
        ctx.fillStyle = "rgba(139,148,158,0.8)";
        ctx.textAlign = "center";
        ctx.textBaseline = "bottom";
        ctx.fillText("Layer " + li, (minX + maxX) / 2, minY - pad - 2);
      }
    }
  };

  /* ── connections ───────────────────────────────────────────── */
  NV.drawConnections = function (ctx) {
    var conns = this.layout.connections;
    var map = this.layout.elemMap;
    var zoom = this.zoom;
    ctx.lineWidth = 1.2 / zoom;

    for (var i = 0; i < conns.length; i++) {
      var c = conns[i];
      var f = map[c[0]], t = map[c[1]];
      if (!f || !t) continue;

      // LOD: skip inner connections when very zoomed out
      if (c[2] === "inner" && zoom < 0.18) continue;
      if (c[2] === "inter" && zoom < 0.10) continue;

      var fCx = f.x + f.w / 2, fCy = f.y + f.h / 2;
      var tCx = t.x + t.w / 2, tCy = t.y + t.h / 2;
      var dx = Math.abs(tCx - fCx), dy = Math.abs(tCy - fCy);
      var x1, y1, x2, y2;

      if (dx > dy * 0.8) {
        // Horizontal
        x1 = f.x + f.w; y1 = fCy;
        x2 = t.x; y2 = tCy;
      } else {
        // Vertical
        x1 = fCx; y1 = f.y + f.h;
        x2 = tCx; y2 = t.y;
      }

      var alpha = c[2] === "major" ? 0.25 : c[2] === "inter" ? 0.15 : 0.10;
      ctx.strokeStyle = "rgba(136,198,255," + alpha + ")";
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      // Bezier for smooth curves
      if (dx > dy * 0.8) {
        var cpx = (x1 + x2) / 2;
        ctx.bezierCurveTo(cpx, y1, cpx, y2, x2, y2);
      } else {
        var cpy = (y1 + y2) / 2;
        ctx.bezierCurveTo(x1, cpy, x2, cpy, x2, y2);
      }
      ctx.stroke();
    }
  };

  /* ── draw single element ───────────────────────────────────── */
  NV.drawElement = function (ctx, el) {
    var zoom = this.zoom;
    var sw = el.w * zoom, sh = el.h * zoom;

    // Off-screen culling
    var sp = this.worldToScreen(el.x, el.y);
    if (sp[0] + sw < 0 || sp[0] > this.width || sp[1] + sh < 0 || sp[1] > this.height) return;

    // Base fill
    ctx.globalAlpha = 0.82;
    ctx.fillStyle = el.color;
    ctx.beginPath();
    this.roundRect(ctx, el.x, el.y, el.w, el.h, 2);
    ctx.fill();

    // Weight texture overlay (only when visible enough)
    if (sw > 20 && sh > 20 && el.type !== "norm") {
      var tex = this.getTexture(el);
      ctx.globalAlpha = 0.55;
      ctx.drawImage(tex, el.x, el.y, el.w, el.h);
    }

    ctx.globalAlpha = 1;

    // Border
    ctx.strokeStyle = "rgba(255,255,255,0.12)";
    ctx.lineWidth = 1 / zoom;
    ctx.beginPath();
    this.roundRect(ctx, el.x, el.y, el.w, el.h, 2);
    ctx.stroke();

    // Hover highlight
    if (this.hoveredElement === el) {
      ctx.strokeStyle = "#58a6ff";
      ctx.lineWidth = 2.2 / zoom;
      ctx.beginPath();
      this.roundRect(ctx, el.x - 1 / zoom, el.y - 1 / zoom, el.w + 2 / zoom, el.h + 2 / zoom, 3);
      ctx.stroke();
    }

    // Labels (adaptive to screen size)
    if (sw > 18 && sh > 12) {
      var fs = Math.max(5, Math.min(12, el.w * 0.25, el.h * 0.35));
      ctx.font = "700 " + fs + "px 'IBM Plex Mono',monospace";
      ctx.fillStyle = "#e6edf3";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      var lines = el.label.split("\n");
      var lh = fs * 1.2;
      var startY = el.y + el.h / 2 - (lines.length - 1) * lh / 2;
      for (var li = 0; li < lines.length; li++) {
        ctx.fillText(lines[li], el.x + el.w / 2, startY + li * lh);
      }

      // Dimension sublabel
      if (sw > 50 && sh > 35 && el.dims.length === 2) {
        var subFs = fs * 0.6;
        ctx.font = "400 " + subFs + "px 'IBM Plex Mono',monospace";
        ctx.fillStyle = "rgba(200,210,220,0.55)";
        ctx.fillText(el.dims[0].toLocaleString() + "×" + el.dims[1].toLocaleString(),
                     el.x + el.w / 2, startY + lines.length * lh);
      }
    }
  };

  /* ── texture cache ─────────────────────────────────────────── */
  NV.getTexture = function (el) {
    if (this.texCache[el.id]) return this.texCache[el.id];
    var res = 48;
    this.texCache[el.id] = makeTexture(res, el.seed);
    return this.texCache[el.id];
  };

  /* ── HUD (model info overlay) ──────────────────────────────── */
  NV.drawHUD = function (ctx) {
    // Model name badge top-left
    ctx.font = "700 13px 'Space Grotesk',sans-serif";
    ctx.fillStyle = "rgba(230,237,243,0.85)";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText(this.modelName, 14, 12);

    ctx.font = "400 11px 'IBM Plex Mono',monospace";
    ctx.fillStyle = "rgba(139,148,158,0.7)";
    var info = this.arch.layers + " layers  ·  " + this.arch.heads + " heads  ·  " +
               "hidden " + this.arch.hidden + "  ·  FFN " + this.arch.inter;
    ctx.fillText(info, 14, 30);

    // Controls hint bottom-right
    ctx.textAlign = "right";
    ctx.textBaseline = "bottom";
    ctx.fillStyle = "rgba(139,148,158,0.4)";
    ctx.fillText("Scroll to zoom  ·  Drag to pan  ·  Hover for details", this.width - 14, this.height - 10);

    // Zoom indicator
    ctx.fillText("zoom " + this.zoom.toFixed(2) + "×", this.width - 14, this.height - 26);
  };

  /* ── tooltip ───────────────────────────────────────────────── */
  NV.updateTooltip = function () {
    var el = this.hoveredElement;
    if (!el) {
      this.tooltip.style.display = "none";
      return;
    }

    var paramStr = fmtNum(el.params) + " params";
    var dimStr = el.dims.length === 2
      ? el.dims[0].toLocaleString() + " × " + el.dims[1].toLocaleString()
      : el.dims[0].toLocaleString();

    var layerLabel = el.layer !== null && el.layer !== undefined ? "Layer " + el.layer + " · " : "";

    this.tooltip.innerHTML =
      "<strong>" + layerLabel + el.label.replace(/\n/g, " ") + "</strong><br>" +
      '<span style="opacity:0.7">' + this.escapeHtml(el.desc) + "</span><br>" +
      '<span style="color:#58a6ff">' + dimStr + "</span>  ·  " + paramStr;

    // Position
    var tx = this.mouseScreenX + 16;
    var ty = this.mouseScreenY - 10;
    if (tx + 260 > this.width) tx = this.mouseScreenX - 260;
    if (ty < 0) ty = 10;
    this.tooltip.style.display = "block";
    this.tooltip.style.left = tx + "px";
    this.tooltip.style.top = ty + "px";
  };

  NV.escapeHtml = function (s) {
    var d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  };

  /* ── hit test ──────────────────────────────────────────────── */
  NV.findElementAt = function (sx, sy) {
    var wp = this.screenToWorld(sx, sy);
    var wx = wp[0], wy = wp[1];
    var elems = this.layout.elements;
    for (var i = elems.length - 1; i >= 0; i--) {
      var e = elems[i];
      if (wx >= e.x && wx <= e.x + e.w && wy >= e.y && wy <= e.y + e.h) return e;
    }
    return null;
  };

  /* ── pointer events ────────────────────────────────────────── */
  NV.onWheel = function (e) {
    e.preventDefault();
    var rect = this.canvas.getBoundingClientRect();
    var mx = e.clientX - rect.left;
    var my = e.clientY - rect.top;

    var factor = e.deltaY < 0 ? 1.12 : 1 / 1.12;
    var newZoom = Math.max(0.02, Math.min(this.zoom * factor, 12));
    var ratio = newZoom / this.zoom;

    this.panX = mx - ratio * (mx - this.panX);
    this.panY = my - ratio * (my - this.panY);
    this.zoom = newZoom;
    this.markDirty();
  };

  NV.onPointerDown = function (e) {
    if (e.button !== 0) return;
    this.dragging = true;
    this.dragStartX = e.clientX;
    this.dragStartY = e.clientY;
    this.dragPanX = this.panX;
    this.dragPanY = this.panY;
    this.canvas.setPointerCapture(e.pointerId);
    this.canvas.style.cursor = "grabbing";
  };

  NV.onPointerMove = function (e) {
    var rect = this.canvas.getBoundingClientRect();
    this.mouseScreenX = e.clientX - rect.left;
    this.mouseScreenY = e.clientY - rect.top;

    if (this.dragging) {
      this.panX = this.dragPanX + (e.clientX - this.dragStartX);
      this.panY = this.dragPanY + (e.clientY - this.dragStartY);
      this.markDirty();
    } else {
      var el = this.findElementAt(this.mouseScreenX, this.mouseScreenY);
      if (el !== this.hoveredElement) {
        this.hoveredElement = el;
        this.canvas.style.cursor = el ? "pointer" : "grab";
        this.markDirty();
      }
    }
  };

  NV.onPointerUp = function () {
    this.dragging = false;
    this.canvas.style.cursor = this.hoveredElement ? "pointer" : "grab";
  };

  /* ── roundRect helper ──────────────────────────────────────── */
  NV.roundRect = function (ctx, x, y, w, h, r) {
    if (ctx.roundRect) { ctx.roundRect(x, y, w, h, r); return; }
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.arcTo(x + w, y, x + w, y + r, r);
    ctx.lineTo(x + w, y + h - r);
    ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
    ctx.lineTo(x + r, y + h);
    ctx.arcTo(x, y + h, x, y + h - r, r);
    ctx.lineTo(x, y + r);
    ctx.arcTo(x, y, x + r, y, r);
  };

  /* ── destroy ───────────────────────────────────────────────── */
  NV.destroy = function () {
    this.canvas.removeEventListener("wheel", this._onWheel);
    this.canvas.removeEventListener("pointerdown", this._onPointerDown);
    this.canvas.removeEventListener("pointermove", this._onPointerMove);
    this.canvas.removeEventListener("pointerup", this._onPointerUp);
    this.canvas.removeEventListener("pointerleave", this._onPointerUp);
    window.removeEventListener("resize", this._onResize);
    if (this._raf) cancelAnimationFrame(this._raf);
    this.container.innerHTML = "";
    this.texCache = {};
  };

  /* ── exports ───────────────────────────────────────────────── */
  window.NetworkVisualizer = NetworkVisualizer;
  window.resolveArchitecture = resolveArchitecture;
})();
