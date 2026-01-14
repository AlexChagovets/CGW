// main.js (PA#3) - Triangles + Gouraud + MIP-mapped texture + SCALE + UV TILING (MIP demo)
import { Model } from './model.js';
'use strict';

let gl;
let canvas;
let surface;
let program;
let spaceball;

// params
let params = {
  a: 4, n: 0.5, m: 6, b: 6, phi: 0,
  Nu: 120,
  Nr: 120,
  scale: 0.35,
  uvScale: 16.0
};

// shader locations
let loc = null;

// texture
let mipTex = null;

// animation time
let startTime = 0;

// ---------- helpers ----------
function mat3FromMat4(m) {
  // OK for rotation + translation + UNIFORM scale
  return new Float32Array([
    m[0], m[1], m[2],
    m[4], m[5], m[6],
    m[8], m[9], m[10],
  ]);
}

function mulMat4Vec4(m, v) {
  // column-major
  const x = m[0] * v[0] + m[4] * v[1] + m[8]  * v[2] + m[12] * v[3];
  const y = m[1] * v[0] + m[5] * v[1] + m[9]  * v[2] + m[13] * v[3];
  const z = m[2] * v[0] + m[6] * v[1] + m[10] * v[2] + m[14] * v[3];
  const w = m[3] * v[0] + m[7] * v[1] + m[11] * v[2] + m[15] * v[3];
  return [x, y, z, w];
}

function rebuildSurface() {
  if (!surface || !gl) return;
  surface.build(params);
  surface.upload(gl);
}

// ---------- MIP texture generation ----------
function makeMipCanvas(size, level) {
  const c = document.createElement('canvas');
  c.width = size;
  c.height = size;
  const ctx = c.getContext('2d');

  // strong different color per mip level
  ctx.fillStyle = `hsl(${(level * 60) % 360}, 85%, 55%)`;
  ctx.fillRect(0, 0, size, size);

  // grid
  ctx.strokeStyle = 'rgba(0,0,0,0.45)';
  ctx.lineWidth = Math.max(1, size / 128);
  const step = Math.max(4, Math.floor(size / 8));
  for (let i = 0; i <= size; i += step) {
    ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, size); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(size, i); ctx.stroke();
  }

  // label
  ctx.fillStyle = 'rgba(255,255,255,0.95)';
  ctx.font = `bold ${Math.max(10, Math.floor(size / 3))}px Arial`;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(`MIP ${level}`, size / 2, size / 2);

  return c;
}

function createMipTexture(gl) {
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);

  const baseSize = 256; // power of 2
  const maxLevel = Math.floor(Math.log2(baseSize));

  for (let level = 0; level <= maxLevel; level++) {
    const size = baseSize >> level;
    const img = makeMipCanvas(size, level);
    gl.texImage2D(gl.TEXTURE_2D, level, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
  }

  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);

  // sharp jumps between MIP levels (best for demo)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST_MIPMAP_NEAREST);

  // magnification (no mip)
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

  gl.bindTexture(gl.TEXTURE_2D, null);
  return tex;
}

// ---------- render ----------
function drawFrame(timeMs) {
  if (!gl || !program || !surface || !spaceball || !loc) return;

  if (!startTime) startTime = timeMs;
  const t = (timeMs - startTime) * 0.001;

  gl.viewport(0, 0, gl.canvas.width, gl.canvas.height);
  gl.clearColor(0, 0, 0, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

  const aspect = gl.canvas.width / gl.canvas.height;

  // narrower FOV helps show minification (more MIP switching)
  const projection = m4.perspective(Math.PI / 10, aspect, 0.1, 300);

  const view = spaceball.getViewMatrix();

  const rotateToPointZero = m4.axisRotation([0.707, 0.707, 0], 0.7);

  // push camera a bit further => easier to reach higher mips
  const translateToPointZero = m4.translation(0, 0, -18);

  // uniform scale
  const s = Math.max(0.01, params.scale);
  const scaleM = m4.scaling(s, s, s);

  // ModelView = T * S * R * View
  const modelView = m4.multiply(
    translateToPointZero,
    m4.multiply(scaleM, m4.multiply(rotateToPointZero, view))
  );
  const mvp = m4.multiply(projection, modelView);

  // rotating point light (world) -> view
  const lightR = 12.0;
  const lightH = 6.0;
  const lightWorld = [lightR * Math.cos(t), lightH, lightR * Math.sin(t), 1.0];
  const lightView4 = mulMat4Vec4(modelView, lightWorld);

  gl.useProgram(program);

  gl.uniformMatrix4fv(loc.uMVP, false, mvp);
  gl.uniformMatrix4fv(loc.uMV, false, modelView);

  // IMPORTANT: because we do uniform scale, N = mat3(MV) is ok,
  // but still normalize in shader (you already do).
  gl.uniformMatrix3fv(loc.uN, false, mat3FromMat4(modelView));

  gl.uniform3f(loc.uLight, lightView4[0], lightView4[1], lightView4[2]);

  // pass UV tiling to shader
  gl.uniform1f(loc.uUVScale, params.uvScale);

  // bind texture
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, mipTex);
  gl.uniform1i(loc.uTex, 0);

  surface.draw(gl, program, loc);

  requestAnimationFrame(drawFrame);
}

// ---------- GL init ----------
function initGL() {
  if (typeof vertexShaderSource === 'undefined' || typeof fragmentShaderSource === 'undefined') {
    throw new Error('shader.gpu did not define vertexShaderSource/fragmentShaderSource');
  }

  program = createProgram(gl, vertexShaderSource, fragmentShaderSource);

  loc = {
    aVertex: gl.getAttribLocation(program, 'vertex'),
    aNormal: gl.getAttribLocation(program, 'normal'),
    aTexcoord: gl.getAttribLocation(program, 'texcoord'),

    uMVP: gl.getUniformLocation(program, 'ModelViewProjectionMatrix'),
    uMV: gl.getUniformLocation(program, 'ModelViewMatrix'),
    uN: gl.getUniformLocation(program, 'NormalMatrix'),
    uLight: gl.getUniformLocation(program, 'LightPosView'),
    uTex: gl.getUniformLocation(program, 'uTex'),
    uUVScale: gl.getUniformLocation(program, 'uUVScale'),
  };

  // quick diagnostics (helps a lot)
  if (loc.aVertex < 0) console.error('Attribute vertex not found');
  if (loc.aNormal < 0) console.error('Attribute normal not found');
  if (loc.aTexcoord < 0) console.error('Attribute texcoord not found');
  if (!loc.uMVP) console.error('Uniform ModelViewProjectionMatrix not found');
  if (!loc.uMV) console.error('Uniform ModelViewMatrix not found');
  if (!loc.uN) console.error('Uniform NormalMatrix not found');
  if (!loc.uLight) console.error('Uniform LightPosView not found');
  if (!loc.uTex) console.error('Uniform uTex not found');
  if (!loc.uUVScale) console.error('Uniform uUVScale not found');

  surface = new Model();
  rebuildSurface();

  mipTex = createMipTexture(gl);

  gl.enable(gl.DEPTH_TEST);
}

// ---------- shader compilation ----------
function createProgram(gl, vShader, fShader) {
  const vsh = gl.createShader(gl.VERTEX_SHADER);
  gl.shaderSource(vsh, vShader);
  gl.compileShader(vsh);
  if (!gl.getShaderParameter(vsh, gl.COMPILE_STATUS)) {
    throw new Error('Vertex shader error: ' + gl.getShaderInfoLog(vsh));
  }

  const fsh = gl.createShader(gl.FRAGMENT_SHADER);
  gl.shaderSource(fsh, fShader);
  gl.compileShader(fsh);
  if (!gl.getShaderParameter(fsh, gl.COMPILE_STATUS)) {
    throw new Error('Fragment shader error: ' + gl.getShaderInfoLog(fsh));
  }

  const prog = gl.createProgram();
  gl.attachShader(prog, vsh);
  gl.attachShader(prog, fsh);
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    throw new Error('Program link error: ' + gl.getProgramInfoLog(prog));
  }
  return prog;
}

// ---------- UI ----------
function setupUI() {
  function bindSlider(id, key, format = (v) => v) {
    const el = document.getElementById(id);
    const out = document.getElementById(id + 'Val');
    if (!el || !out) return;

    const update = () => {
      const isFloat = String(el.step).includes('.');
      params[key] = isFloat ? parseFloat(el.value) : Number(el.value);
      out.textContent = format(params[key]);

      // rebuild only if geometry depends on it
      if (key !== 'scale' && key !== 'uvScale') rebuildSurface();
    };

    el.addEventListener('input', update);
    update();
  }

  bindSlider('a', 'a');
  bindSlider('n', 'n', v => v.toFixed(2));
  bindSlider('m', 'm');
  bindSlider('b', 'b', v => v.toFixed(2));
  bindSlider('phi', 'phi', v => v.toFixed(2));
  bindSlider('Nu', 'Nu');
  bindSlider('Nr', 'Nr');

  bindSlider('scale', 'scale', v => v.toFixed(2));
  bindSlider('uvScale', 'uvScale', v => v.toFixed(0));

  const shotBtn = document.getElementById('shot');
  if (shotBtn) {
    shotBtn.addEventListener('click', () => {
      const url = canvas.toDataURL('image/png');
      const a = document.createElement('a');
      a.href = url;
      a.download = 'surface_pa3_mip.png';
      a.click();
    });
  }
}

// ---------- entry ----------
function init() {
  canvas = document.getElementById('webglcanvas');
  gl = canvas.getContext('webgl');

  if (!gl) {
    document.getElementById('canvas-holder').innerHTML = '<p>WebGL not supported</p>';
    return;
  }

  try {
    initGL();
  } catch (e) {
    document.getElementById('canvas-holder').innerHTML = '<p>Could not init WebGL: ' + e + '</p>';
    console.error(e);
    return;
  }

  // trackball used only for view matrix (we run our own render loop)
  spaceball = new TrackballRotator(canvas, () => {}, 0);

  setupUI();
  requestAnimationFrame(drawFrame);
}

// ✅ IMPORTANT: only ONE init method
window.addEventListener('load', init);
