// model.js (GWC mesh + UV for MIP textures)
// Surface of revolution with damping circular waves
// x = r cos u, y = r sin u, z = a e^{-n r} sin((m*pi/b) r + phi)

export class Model {
  constructor() {
    this.positions = new Float32Array();
    this.normals = new Float32Array();
    this.texcoords = new Float32Array();   // <-- NEW (s,t)
    this.indices = null;                   // Uint16Array or Uint32Array

    this.posVBO = null;
    this.nrmVBO = null;
    this.uvVBO = null;                     // <-- NEW
    this.ibo = null;

    this.indexCount = 0;
    this.indexType = null;                 // gl.UNSIGNED_SHORT or gl.UNSIGNED_INT
  }

  static surfacePoint(r, u, p) {
    const { a, n, m, b, phi } = p;
    const w = (m * Math.PI) / b;
    const z = a * Math.exp(-n * r) * Math.sin(w * r + phi);
    const x = r * Math.cos(u);
    const y = r * Math.sin(u);
    return [x, y, z];
  }

  build(params) {
    const p = { ...params };

    const rMin = 0.0;
    const rMax = p.b;

    const uMin = 0.0;
    const uMax = Math.PI * 2.0;

    const Nr = Math.max(2, p.Nr | 0);
    const Nu = Math.max(3, p.Nu | 0);

    const vertCount = Nr * Nu;

    const pos = new Float32Array(vertCount * 3);
    const nrm = new Float32Array(vertCount * 3); // accum -> normalize
    const uv  = new Float32Array(vertCount * 2); // <-- NEW

    // --- generate positions + UV ---
    let pk = 0;
    let uk = 0;

    for (let i = 0; i < Nr; i++) {
      const t = i / (Nr - 1);                // r normalized (0..1)
      const r = rMin + t * (rMax - rMin);

      for (let j = 0; j < Nu; j++) {
        const s = j / (Nu - 1);              // u normalized (0..1)
        const u = uMin + s * (uMax - uMin);

        const [x, y, z] = Model.surfacePoint(r, u, p);
        pos[pk++] = x;
        pos[pk++] = y;
        pos[pk++] = z;

        uv[uk++] = s; // s = u normalized
        uv[uk++] = t; // t = r normalized
      }
    }

    // --- generate indices (2 triangles per quad) ---
    const triCount = (Nr - 1) * (Nu - 1) * 2;
    const indexCount = triCount * 3;

    const useUint32 = vertCount > 65535;
    const idx = useUint32 ? new Uint32Array(indexCount) : new Uint16Array(indexCount);

    let t = 0;
    const vid = (i, j) => i * Nu + j;

    for (let i = 0; i < Nr - 1; i++) {
      for (let j = 0; j < Nu - 1; j++) {
        const a = vid(i, j);
        const b = vid(i + 1, j);
        const c = vid(i, j + 1);
        const d = vid(i + 1, j + 1);

        // a-b-c
        idx[t++] = a; idx[t++] = b; idx[t++] = c;
        // b-d-c
        idx[t++] = b; idx[t++] = d; idx[t++] = c;
      }
    }

    // --- normals: Average face normal (sum adjacent face normals) ---
    for (let ii = 0; ii < idx.length; ii += 3) {
      const i0 = idx[ii];
      const i1 = idx[ii + 1];
      const i2 = idx[ii + 2];

      const p0x = pos[i0 * 3], p0y = pos[i0 * 3 + 1], p0z = pos[i0 * 3 + 2];
      const p1x = pos[i1 * 3], p1y = pos[i1 * 3 + 1], p1z = pos[i1 * 3 + 2];
      const p2x = pos[i2 * 3], p2y = pos[i2 * 3 + 1], p2z = pos[i2 * 3 + 2];

      // e1 = p1 - p0
      const e1x = p1x - p0x, e1y = p1y - p0y, e1z = p1z - p0z;
      // e2 = p2 - p0
      const e2x = p2x - p0x, e2y = p2y - p0y, e2z = p2z - p0z;

      // n = e1 x e2
      const nx = e1y * e2z - e1z * e2y;
      const ny = e1z * e2x - e1x * e2z;
      const nz = e1x * e2y - e1y * e2x;

      // accumulate
      nrm[i0 * 3]     += nx; nrm[i0 * 3 + 1] += ny; nrm[i0 * 3 + 2] += nz;
      nrm[i1 * 3]     += nx; nrm[i1 * 3 + 1] += ny; nrm[i1 * 3 + 2] += nz;
      nrm[i2 * 3]     += nx; nrm[i2 * 3 + 1] += ny; nrm[i2 * 3 + 2] += nz;
    }

    // normalize
    for (let v = 0; v < vertCount; v++) {
      const x = nrm[v * 3];
      const y = nrm[v * 3 + 1];
      const z = nrm[v * 3 + 2];
      const len = Math.hypot(x, y, z);

      if (len > 1e-12) {
        nrm[v * 3]     = x / len;
        nrm[v * 3 + 1] = y / len;
        nrm[v * 3 + 2] = z / len;
      } else {
        nrm[v * 3] = 0;
        nrm[v * 3 + 1] = 0;
        nrm[v * 3 + 2] = 1;
      }
    }

    this.positions = pos;
    this.normals = nrm;
    this.texcoords = uv;       // <-- NEW
    this.indices = idx;
    this.indexCount = idx.length;
  }

  upload(gl) {
    if (!this.posVBO) this.posVBO = gl.createBuffer();
    if (!this.nrmVBO) this.nrmVBO = gl.createBuffer();
    if (!this.uvVBO)  this.uvVBO  = gl.createBuffer(); // <-- NEW
    if (!this.ibo)    this.ibo    = gl.createBuffer();

    gl.bindBuffer(gl.ARRAY_BUFFER, this.posVBO);
    gl.bufferData(gl.ARRAY_BUFFER, this.positions, gl.STATIC_DRAW);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.nrmVBO);
    gl.bufferData(gl.ARRAY_BUFFER, this.normals, gl.STATIC_DRAW);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.uvVBO);         // <-- NEW
    gl.bufferData(gl.ARRAY_BUFFER, this.texcoords, gl.STATIC_DRAW);

    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ibo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, this.indices, gl.STATIC_DRAW);

    // resolve index type
    if (this.indices instanceof Uint32Array) {
      const ext = gl.getExtension('OES_element_index_uint');
      if (!ext) {
        console.warn('OES_element_index_uint not supported. Reduce Nr/Nu so vertices <= 65535.');
      }
      this.indexType = gl.UNSIGNED_INT;
    } else {
      this.indexType = gl.UNSIGNED_SHORT;
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
  }

  // loc: { aVertex, aNormal, aTexcoord }
  draw(gl, program, loc) {
    gl.useProgram(program);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.posVBO);
    gl.vertexAttribPointer(loc.aVertex, 3, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(loc.aVertex);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.nrmVBO);
    gl.vertexAttribPointer(loc.aNormal, 3, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(loc.aNormal);

    gl.bindBuffer(gl.ARRAY_BUFFER, this.uvVBO);
    gl.vertexAttribPointer(loc.aTexcoord, 2, gl.FLOAT, false, 0, 0);
    gl.enableVertexAttribArray(loc.aTexcoord);

    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.ibo);
    gl.drawElements(gl.TRIANGLES, this.indexCount, this.indexType, 0);

    gl.bindBuffer(gl.ARRAY_BUFFER, null);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, null);
  }
}
