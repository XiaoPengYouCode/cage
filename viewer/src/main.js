import './style.css';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

// ---------------------------------------------------------------------------
// Model registry
// ---------------------------------------------------------------------------

const MODELS = {
  voronoi: {
    url: '/data/hybrid_exact_shell_2000.glb?v=3',
    label: 'Voronoi Block',
    group: 'Legacy',
    mode: 'voronoi',
  },
  lumbar_vertebra: {
    url: '/data/lumbar_vertebra_hd.glb',
    label: 'Lumbar Vertebra',
    group: 'Anatomy',
    mode: 'mesh',
  },
  cage_raw: {
    url: '/data/cage_raw.glb',
    label: 'Cage — FEA Voxels (400 µm)',
    group: 'Cage',
    mode: 'mesh',
    color: new THREE.Color(0.55, 0.75, 0.95),
  },
  cage_in_vertebra: {
    url: '/data/cage_in_vertebra.glb',
    label: 'Cage in Vertebra',
    group: 'Cage',
    mode: 'mesh',
  },
  '681_raw': {
    url: '/data/681_raw.glb',
    label: '681 — Raw Import (400 µm)',
    group: '681',
    mode: 'mesh',
    color: new THREE.Color(0.76, 0.60, 0.42),
  },
  '681_skeleton_density': {
    url: '/data/681_skeleton_density.glb',
    label: '681 — Scaffold mesh (density seeds)',
    group: '681',
    mode: 'mesh',
    color: new THREE.Color(0.55, 0.75, 0.45),
  },
  '681_skeleton_cvt500': {
    url: '/data/681_skeleton_cvt500.glb',
    label: '681 — Scaffold mesh (CVT 500 iters)',
    group: '681',
    mode: 'mesh',
    color: new THREE.Color(0.45, 0.72, 0.90),
  },
  '681_voronoi_cells_seeds6_cvt1': {
    url: '/data/681_voronoi_cells_seeds6_cvt1.glb',
    label: '681 — Voronoi cells (seeds=6, cvt=1)',
    group: 'Debug',
    mode: 'mesh',
  },
};

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------

const params = new URLSearchParams(window.location.search);
let modelKey = params.get('model') ?? 'voronoi';
if (!MODELS[modelKey]) modelKey = 'voronoi';

// Render the full app shell (panel + canvas)
const app = document.querySelector('#app');
app.innerHTML = `
  <div class="panel">
    <div class="panel-section">
      <label class="select-label" for="model-select">Model</label>
      <select id="model-select">${_buildOptions()}</select>
    </div>
    <div class="panel-section">
      <div class="meta" id="meta">Loading…</div>
    </div>
    <div class="panel-section controls" id="controls-section">
      <label><span>Show faces</span><input id="showFaces" type="checkbox" checked /></label>
      <label><span>Background</span><input id="bg" type="color" value="#1a1a2e" /></label>
    </div>
    <div class="badge-row" id="badges"></div>
    <div class="note" id="note"></div>
  </div>
  <div class="canvas-wrap" id="canvas-wrap"></div>
`;

function _buildOptions() {
  // Group options by .group field
  const groups = {};
  for (const [key, cfg] of Object.entries(MODELS)) {
    if (!groups[cfg.group]) groups[cfg.group] = [];
    groups[cfg.group].push({ key, label: cfg.label });
  }
  return Object.entries(groups).map(([grp, items]) => {
    const opts = items.map(({ key, label }) =>
      `<option value="${key}"${key === modelKey ? ' selected' : ''}>${label}</option>`
    ).join('');
    return `<optgroup label="${grp}">${opts}</optgroup>`;
  }).join('');
}

// ---------------------------------------------------------------------------
// Three.js setup (shared across model switches)
// ---------------------------------------------------------------------------

const wrap = document.querySelector('#canvas-wrap');
const metaEl = document.querySelector('#meta');
const badgesEl = document.querySelector('#badges');
const noteEl = document.querySelector('#note');
const controlsSection = document.querySelector('#controls-section');

const renderer = new THREE.WebGLRenderer({ antialias: true, logarithmicDepthBuffer: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(wrap.clientWidth, wrap.clientHeight);
wrap.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, wrap.clientWidth / wrap.clientHeight, 1, 2000);
const orbitControls = new OrbitControls(camera, renderer.domElement);
orbitControls.enableDamping = true;

const ambientLight = scene.add(new THREE.AmbientLight(0xffffff, 1.5));
const keyLight = new THREE.DirectionalLight(0xffffff, 2.0);
keyLight.position.set(220, 260, 280);
scene.add(keyLight);
const fillLight = new THREE.DirectionalLight(0xffffff, 0.7);
fillLight.position.set(-180, -100, 160);
scene.add(fillLight);
const rimLight = new THREE.DirectionalLight(0x8888ff, 0.4);
rimLight.position.set(0, -200, -150);
scene.add(rimLight);

const world = new THREE.Group();
scene.add(world);

// Axis gizmo
const gizmoScene = new THREE.Scene();
const gizmoCamera = new THREE.OrthographicCamera(-1.6, 1.6, 1.6, -1.6, 0, 10);
gizmoCamera.position.set(0, 0, 5);
const GIZMO_SIZE = 110;

(function buildGizmo() {
  const arrow = (dir, color, label) => {
    gizmoScene.add(new THREE.ArrowHelper(dir.normalize(), new THREE.Vector3(), 1.0, color, 0.28, 0.16));
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = 64;
    const ctx = canvas.getContext('2d');
    ctx.font = 'bold 40px sans-serif';
    ctx.textAlign = ctx.textBaseline = 'middle';
    ctx.fillStyle = `#${color.toString(16).padStart(6, '0')}`;
    ctx.fillText(label, 32, 32);
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(canvas), depthTest: false }));
    sprite.scale.set(0.38, 0.38, 1);
    sprite.position.copy(dir.clone().normalize().multiplyScalar(1.35));
    gizmoScene.add(sprite);
  };
  arrow(new THREE.Vector3(1, 0, 0), 0xff4444, 'X');
  arrow(new THREE.Vector3(0, 1, 0), 0x44cc44, 'Y');
  arrow(new THREE.Vector3(0, 0, 1), 0x4488ff, 'Z');
})();

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const loader = new GLTFLoader();
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
let cellGroups = [];
let selectedCell = null;
let currentBg = '#1a1a2e';
let currentMode = null;
let dblClickListener = null;

renderer.setClearColor(currentBg);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const REFERENCE_RADIUS = 80; // mm

function frameFromBounds(box) {
  const center = box.getCenter(new THREE.Vector3());
  const r = REFERENCE_RADIUS;
  camera.position.set(center.x + r * 1.3, center.y + r * 0.6, center.z + r * 1.4);
  camera.near = r * 0.001;
  camera.far = r * 100;
  camera.updateProjectionMatrix();
  orbitControls.target.copy(center);
  orbitControls.update();
}

function clearSelection() {
  if (!selectedCell) return;
  for (const mesh of selectedCell.meshes) {
    mesh.material.color.copy(mesh.userData.baseColor);
    mesh.material.emissive?.setHex?.(0x000000);
    mesh.material.emissiveIntensity = 0.0;
  }
  selectedCell = null;
  noteEl.textContent = 'Double-click a cell to select it.';
}

function selectCell(record) {
  if (selectedCell?.seedId === record.seedId) { clearSelection(); return; }
  clearSelection();
  selectedCell = record;
  for (const mesh of record.meshes) {
    mesh.material.color.copy(mesh.userData.baseColor).offsetHSL(0, 0, 0.10);
    mesh.material.emissive = new THREE.Color(0xffd166);
    mesh.material.emissiveIntensity = 0.28;
  }
  noteEl.textContent = `Selected cell seedId=${record.seedId} (${record.cellLabel})`;
}

// ---------------------------------------------------------------------------
// Scene loaders
// ---------------------------------------------------------------------------

async function loadMeshScene(model) {
  const gltf = await loader.loadAsync(model.url);
  world.add(gltf.scene);

  let triCount = 0, meshCount = 0;
  gltf.scene.traverse((obj) => {
    if (!obj.isMesh) return;
    if (!obj.geometry.getAttribute('normal')) obj.geometry.computeVertexNormals();
    const importedColor = model.color
      ? model.color.clone()
      : (obj.material?.color ? obj.material.color.clone() : new THREE.Color(0.76, 0.60, 0.42));
    obj.material = new THREE.MeshStandardMaterial({
      color: importedColor, roughness: 0.65, metalness: 0.05,
      side: THREE.FrontSide, polygonOffset: true,
      polygonOffsetFactor: 1, polygonOffsetUnits: 1,
    });
    obj.userData.baseColor = importedColor.clone();
    const idx = obj.geometry.index;
    triCount += idx ? idx.count / 3 : obj.geometry.attributes.position.count / 3;
    meshCount += 1;
  });

  const bounds = new THREE.Box3().setFromObject(gltf.scene);
  frameFromBounds(bounds);
  const size = bounds.getSize(new THREE.Vector3());
  metaEl.innerHTML = [
    `Triangles: <strong>${Math.round(triCount).toLocaleString()}</strong>`,
    `Meshes: <strong>${meshCount}</strong>`,
    `Size: <strong>${size.x.toFixed(1)} × ${size.y.toFixed(1)} × ${size.z.toFixed(1)} mm</strong>`,
  ].join('<br>');
  badgesEl.innerHTML = ['Three.js WebGL', 'Voxelized mesh', 'Exposed-face render']
    .map(b => `<div class="badge">${b}</div>`).join('');
  noteEl.textContent = 'Orbit: drag  ·  Zoom: scroll';
}

async function loadVoronoiScene(model) {
  const gltf = await loader.loadAsync(model.url);
  world.add(gltf.scene);

  cellGroups = [];
  const roots = gltf.scene.children.filter(
    c => c.userData?.seedId !== undefined || c.name?.startsWith('cell-'),
  );
  let faceCount = 0, boundaryCount = 0;

  for (const root of roots) {
    const record = {
      seedId: root.userData.seedId ?? -1,
      isShell: Boolean(root.userData.isShell),
      cellLabel: root.userData.cellLabel ?? (root.userData.isShell ? 'shell' : 'non-shell'),
      group: root, meshes: [], lines: [],
      basePosition: root.position.clone(),
      explodeDir: new THREE.Vector3(),
    };
    root.traverse((obj) => {
      if (obj.isMesh) {
        if (!obj.geometry.getAttribute('normal')) obj.geometry.computeVertexNormals();
        const c = obj.material?.color ? obj.material.color.clone() : new THREE.Color(0.7, 0.7, 0.8);
        obj.material = new THREE.MeshStandardMaterial({
          color: c, side: THREE.DoubleSide,
          roughness: obj.name?.includes('cylinder') ? 0.82 : 0.94, metalness: 0.0,
          polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
        });
        obj.userData.baseColor = c.clone();
        record.meshes.push(obj);
        faceCount += 1;
      }
      if (obj.isLineSegments || obj.isLine) {
        obj.material = new THREE.LineBasicMaterial({ color: 0x111111, depthTest: true, depthWrite: false });
        obj.renderOrder = 2;
        obj.userData.baseColor = new THREE.Color(0x111111);
        record.lines.push(obj);
        boundaryCount += 1;
      }
    });
    if (record.meshes.length) cellGroups.push(record);
  }

  const bounds = new THREE.Box3().setFromObject(gltf.scene);
  const worldCenter = bounds.getCenter(new THREE.Vector3());
  frameFromBounds(bounds);
  for (const record of cellGroups) {
    const box = new THREE.Box3().setFromObject(record.group);
    const center = box.getCenter(new THREE.Vector3());
    record.explodeDir.copy(center.clone().sub(worldCenter));
    if (record.explodeDir.lengthSq() < 1e-12) record.explodeDir.set(0, 0, 1);
    else record.explodeDir.normalize();
  }

  const numShell = cellGroups.filter(c => c.isShell).length;
  metaEl.innerHTML = [
    `Cells: <strong>${roots.length}</strong>`,
    `Shell: <strong>${numShell}</strong>`,
    `Non-shell: <strong>${roots.length - numShell}</strong>`,
    `Faces: <strong>${faceCount}</strong>`,
  ].join('<br>');
  badgesEl.innerHTML = ['Three.js WebGL', 'Full Voronoi export', 'Shell-labeled', 'Dblclick select']
    .map(b => `<div class="badge">${b}</div>`).join('');
  noteEl.textContent = 'Double-click a cell to select it.';
}

// ---------------------------------------------------------------------------
// Controls binding (re-bound on each model switch)
// ---------------------------------------------------------------------------

function rebuildVoronoiControls() {
  // Add voronoi-specific controls if not already present
  if (!document.querySelector('#showLines')) {
    controlsSection.insertAdjacentHTML('beforeend', `
      <label><span>Show boundaries</span><input id="showLines" type="checkbox" checked /></label>
      <label><span>Explosion</span><input id="explode" type="range" min="0" max="80" step="1" value="0" /></label>
    `);
  }
}

function removeVoronoiControls() {
  document.querySelector('#showLines')?.closest('label')?.remove();
  document.querySelector('#explode')?.closest('label')?.remove();
}

function bindControls(mode) {
  document.querySelector('#showFaces')?.addEventListener('change', (e) => {
    if (mode === 'voronoi') {
      for (const cell of cellGroups) for (const m of cell.meshes) m.visible = e.target.checked;
    } else {
      world.traverse(obj => { if (obj.isMesh) obj.visible = e.target.checked; });
    }
  });
  document.querySelector('#showLines')?.addEventListener('change', (e) => {
    for (const cell of cellGroups) for (const l of cell.lines) l.visible = e.target.checked;
  });
  document.querySelector('#explode')?.addEventListener('input', (e) => {
    const amt = Number(e.target.value);
    for (const cell of cellGroups) {
      cell.group.position.copy(cell.basePosition).addScaledVector(cell.explodeDir, amt);
    }
  });
  document.querySelector('#bg')?.addEventListener('input', (e) => {
    currentBg = e.target.value;
    renderer.setClearColor(currentBg);
  });

  if (mode === 'voronoi') {
    dblClickListener = (event) => {
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const hits = raycaster.intersectObjects(cellGroups.flatMap(i => i.meshes), false);
      if (!hits.length) { clearSelection(); return; }
      const record = cellGroups.find(i => i.meshes.includes(hits[0].object));
      if (record) selectCell(record);
    };
    renderer.domElement.addEventListener('dblclick', dblClickListener);
  }
}

// ---------------------------------------------------------------------------
// Model loader (switches without full page reload)
// ---------------------------------------------------------------------------

async function loadModel(key) {
  const model = MODELS[key];
  if (!model) return;

  // Teardown previous scene
  if (dblClickListener) {
    renderer.domElement.removeEventListener('dblclick', dblClickListener);
    dblClickListener = null;
  }
  clearSelection();
  cellGroups = [];
  while (world.children.length) world.remove(world.children[0]);
  metaEl.textContent = 'Loading…';
  badgesEl.innerHTML = '';
  noteEl.textContent = '';

  // Update controls
  if (model.mode === 'voronoi') {
    rebuildVoronoiControls();
  } else {
    removeVoronoiControls();
    // Reset explosion if switching away from voronoi
    document.querySelector('#explode') && (document.querySelector('#explode').value = 0);
  }
  document.querySelector('#showFaces').checked = true;

  // Sync URL without reload
  const url = new URL(window.location.href);
  url.searchParams.set('model', key);
  window.history.replaceState({}, '', url.toString());

  currentMode = model.mode;
  if (model.mode === 'mesh') {
    await loadMeshScene(model);
  } else {
    await loadVoronoiScene(model);
  }
  bindControls(model.mode);
}

// ---------------------------------------------------------------------------
// Model selector event
// ---------------------------------------------------------------------------

document.querySelector('#model-select').addEventListener('change', (e) => {
  modelKey = e.target.value;
  loadModel(modelKey);
});

// ---------------------------------------------------------------------------
// Resize
// ---------------------------------------------------------------------------

window.addEventListener('resize', () => {
  camera.aspect = wrap.clientWidth / wrap.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(wrap.clientWidth, wrap.clientHeight);
});

// ---------------------------------------------------------------------------
// Render loop
// ---------------------------------------------------------------------------

function animate() {
  orbitControls.update();

  renderer.setViewport(0, 0, wrap.clientWidth, wrap.clientHeight);
  renderer.setScissor(0, 0, wrap.clientWidth, wrap.clientHeight);
  renderer.setScissorTest(false);
  renderer.render(scene, camera);

  const gx = wrap.clientWidth - GIZMO_SIZE;
  renderer.setViewport(gx, 0, GIZMO_SIZE, GIZMO_SIZE);
  renderer.setScissor(gx, 0, GIZMO_SIZE, GIZMO_SIZE);
  renderer.setScissorTest(true);
  renderer.clearDepth();
  const camDir = new THREE.Vector3();
  camera.getWorldDirection(camDir);
  gizmoCamera.position.copy(camDir).negate().multiplyScalar(5);
  gizmoCamera.quaternion.copy(camera.quaternion);
  renderer.render(gizmoScene, gizmoCamera);
  renderer.setScissorTest(false);
  renderer.setClearColor(currentBg);

  requestAnimationFrame(animate);
}
animate();

// Initial load
loadModel(modelKey);
