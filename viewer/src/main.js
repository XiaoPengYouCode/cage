import './style.css';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

// Allow ?model=lumbar_vertebra or ?model=voronoi (default) via URL param
const params = new URLSearchParams(window.location.search);
const modelKey = params.get('model') ?? 'voronoi';

const MODELS = {
  voronoi: {
    url: '/data/hybrid_exact_shell_2000.glb?v=3',
    title: 'Three.js Voronoi Block Viewer',
    mode: 'voronoi',
  },
  lumbar_vertebra: {
    url: '/data/lumbar_vertebra_hd.glb',
    title: 'Lumbar Vertebra — Voxelized Mesh',
    mode: 'mesh',
  },
  cage_raw: {
    url: '/data/cage_raw.glb',
    title: 'Lumbar Cage — FEA Voxel Model (400 µm)',
    mode: 'mesh',
    color: new THREE.Color(0.55, 0.75, 0.95),
  },
  cage_in_vertebra: {
    url: '/data/cage_in_vertebra.glb',
    title: 'Cage in Vertebra — Merged Voxel Model',
    mode: 'mesh',
  },
};

const model = MODELS[modelKey] ?? MODELS.voronoi;

const app = document.querySelector('#app');
app.innerHTML = `
  <div class="panel">
    <h1>${model.title}</h1>
    <div class="meta" id="meta">Loading scene…</div>
    <div class="controls">
      <label><span>Show faces</span><input id="showFaces" type="checkbox" checked /></label>
      <label><span>Background</span><input id="bg" type="color" value="#1a1a2e" /></label>
      ${model.mode === 'voronoi' ? `
      <label><span>Show boundaries</span><input id="showLines" type="checkbox" checked /></label>
      <label><span>Explosion</span><input id="explode" type="range" min="0" max="80" step="1" value="0" /></label>
      ` : ''}
    </div>
    <div class="badge-row" id="badges"></div>
    <div class="model-switcher">
      <a href="?model=voronoi" class="${modelKey === 'voronoi' ? 'active' : ''}">Voronoi</a>
      <a href="?model=lumbar_vertebra" class="${modelKey === 'lumbar_vertebra' ? 'active' : ''}">Lumbar Vertebra</a>
      <a href="?model=cage_raw" class="${modelKey === 'cage_raw' ? 'active' : ''}">Cage Raw</a>
      <a href="?model=cage_in_vertebra" class="${modelKey === 'cage_in_vertebra' ? 'active' : ''}">Cage in Vertebra</a>
    </div>
    <div class="note" id="selection-note">${model.mode === 'voronoi' ? 'Double-click a cell to select it.' : 'Use mouse to orbit, scroll to zoom.'}</div>
  </div>
  <div class="canvas-wrap" id="canvas-wrap"></div>
`;

const wrap = document.querySelector('#canvas-wrap');
const metaEl = document.querySelector('#meta');
const badgesEl = document.querySelector('#badges');
const selectionNoteEl = document.querySelector('#selection-note');

let currentBg = model.mode === 'mesh' ? '#1a1a2e' : '#f4f6fb';
const bgDefault = currentBg;
const renderer = new THREE.WebGLRenderer({
  antialias: true,
  logarithmicDepthBuffer: true,
});
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(wrap.clientWidth, wrap.clientHeight);
renderer.setClearColor(bgDefault);
wrap.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, wrap.clientWidth / wrap.clientHeight, 1, 2000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

scene.add(new THREE.AmbientLight(0xffffff, model.mode === 'mesh' ? 1.2 : 1.9));
const keyLight = new THREE.DirectionalLight(0xffffff, model.mode === 'mesh' ? 2.2 : 1.65);
keyLight.position.set(220, 260, 280);
scene.add(keyLight);
const fillLight = new THREE.DirectionalLight(0xffffff, model.mode === 'mesh' ? 0.8 : 0.65);
fillLight.position.set(-180, -100, 160);
scene.add(fillLight);
const rimLight = new THREE.DirectionalLight(0x8888ff, 0.4);
rimLight.position.set(0, -200, -150);
scene.add(rimLight);

const world = new THREE.Group();
scene.add(world);

// ── Axis gizmo (inset, bottom-right) ────────────────────────────────────────
const gizmoScene  = new THREE.Scene();
const gizmoCamera = new THREE.OrthographicCamera(-1.6, 1.6, 1.6, -1.6, 0, 10);
gizmoCamera.position.set(0, 0, 5);

const _axisArrow = (dir, color, label) => {
  const origin = new THREE.Vector3(0, 0, 0);
  const arrow  = new THREE.ArrowHelper(dir.normalize(), origin, 1.0, color, 0.28, 0.16);
  gizmoScene.add(arrow);

  // Text label as a sprite
  const canvas  = document.createElement('canvas');
  canvas.width  = 64;
  canvas.height = 64;
  const ctx = canvas.getContext('2d');
  ctx.font          = 'bold 40px sans-serif';
  ctx.textAlign     = 'center';
  ctx.textBaseline  = 'middle';
  ctx.fillStyle     = `#${color.toString(16).padStart(6, '0')}`;
  ctx.fillText(label, 32, 32);
  const tex = new THREE.CanvasTexture(canvas);
  const mat = new THREE.SpriteMaterial({ map: tex, depthTest: false });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(0.38, 0.38, 1);
  sprite.position.copy(dir.clone().normalize().multiplyScalar(1.35));
  gizmoScene.add(sprite);
};

_axisArrow(new THREE.Vector3(1, 0, 0), 0xff4444, 'X');
_axisArrow(new THREE.Vector3(0, 1, 0), 0x44cc44, 'Y');
_axisArrow(new THREE.Vector3(0, 0, 1), 0x4488ff, 'Z');

const GIZMO_SIZE = 110; // px

const loader = new GLTFLoader();
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const cellGroups = [];
let selectedCell = null;

// Fixed camera distance so all models share the same world scale.
// A lumbar vertebra is ~90 mm across; we frame at that reference so
// the cage (12 mm) appears proportionally small next to it.
const REFERENCE_RADIUS = 80; // mm

function frameFromBounds(box) {
  const center = box.getCenter(new THREE.Vector3());
  const r = REFERENCE_RADIUS;
  camera.position.set(
    center.x + r * 1.3,
    center.y + r * 0.6,
    center.z + r * 1.4,
  );
  camera.near = r * 0.001;
  camera.far  = r * 100;
  camera.updateProjectionMatrix();
  controls.target.copy(center);
  controls.update();
}

function clearSelection() {
  if (!selectedCell) return;
  for (const mesh of selectedCell.meshes) {
    mesh.material.color.copy(mesh.userData.baseColor);
    mesh.material.emissive?.setHex?.(0x000000);
    mesh.material.emissiveIntensity = 0.0;
  }
  selectedCell = null;
  selectionNoteEl.textContent = 'Double-click a cell to select it.';
}

function selectCell(cellRecord) {
  if (selectedCell?.seedId === cellRecord.seedId) { clearSelection(); return; }
  clearSelection();
  selectedCell = cellRecord;
  for (const mesh of cellRecord.meshes) {
    mesh.material.color.copy(mesh.userData.baseColor).offsetHSL(0, 0, 0.10);
    mesh.material.emissive = new THREE.Color(0xffd166);
    mesh.material.emissiveIntensity = 0.28;
  }
  selectionNoteEl.textContent = `Selected cell seedId=${cellRecord.seedId} (${cellRecord.cellLabel})`;
}

// ── Mesh-mode loader (lumbar vertebra and similar single-mesh GLBs) ──────────
async function loadMeshScene() {
  const gltf = await loader.loadAsync(model.url);
  world.add(gltf.scene);

  let triCount = 0;
  let meshCount = 0;

  gltf.scene.traverse((obj) => {
    if (!obj.isMesh) return;
    if (!obj.geometry.getAttribute('normal')) obj.geometry.computeVertexNormals();

    const importedColor = model.color
      ? model.color.clone()
      : (obj.material?.color ? obj.material.color.clone() : new THREE.Color(0.76, 0.60, 0.42));

    obj.material = new THREE.MeshStandardMaterial({
      color: importedColor,
      roughness: 0.65,
      metalness: 0.05,
      side: THREE.FrontSide,
      polygonOffset: true,
      polygonOffsetFactor: 1,
      polygonOffsetUnits: 1,
    });
    obj.userData.baseColor = importedColor.clone();
    obj.castShadow = true;
    obj.receiveShadow = true;

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
  ].join('<br />');
  const modelBadges = modelKey === 'cage_raw'
    ? ['Three.js WebGL', 'FEA NPZ voxels', '400 µm resolution', 'Exposed-face mesh']
    : ['Three.js WebGL', 'Voxelized STL', 'Exposed-face mesh'];
  badgesEl.innerHTML = modelBadges.map(b => `<div class="badge">${b}</div>`).join('');
}

// ── Voronoi-mode loader (original cell-labelled GLB) ─────────────────────────
async function loadVoronoiScene() {
  const gltf = await loader.loadAsync(model.url);
  world.add(gltf.scene);

  const roots = gltf.scene.children.filter(
    (child) => child.userData?.seedId !== undefined || child.name?.startsWith('cell-'),
  );
  let faceCount = 0;
  let boundaryCount = 0;

  for (const root of roots) {
    const record = {
      seedId: root.userData.seedId ?? -1,
      isShell: Boolean(root.userData.isShell),
      cellLabel: root.userData.cellLabel ?? (root.userData.isShell ? 'shell' : 'non-shell'),
      group: root,
      meshes: [],
      lines: [],
      basePosition: root.position.clone(),
      explodeDir: new THREE.Vector3(),
    };
    root.traverse((obj) => {
      if (obj.isMesh) {
        if (!obj.geometry.getAttribute('normal')) obj.geometry.computeVertexNormals();
        const importedColor = obj.material?.color ? obj.material.color.clone() : new THREE.Color(0.7, 0.7, 0.8);
        obj.material = new THREE.MeshStandardMaterial({
          color: importedColor,
          side: THREE.DoubleSide,
          roughness: obj.name?.includes('cylinder') ? 0.82 : 0.94,
          metalness: 0.0,
          polygonOffset: true,
          polygonOffsetFactor: 1,
          polygonOffsetUnits: 1,
        });
        obj.userData.baseColor = importedColor.clone();
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

  const numShellCells = cellGroups.filter((c) => c.isShell).length;
  metaEl.innerHTML = [
    `Cells: <strong>${roots.length}</strong>`,
    `Shell cells: <strong>${numShellCells}</strong>`,
    `Non-shell: <strong>${roots.length - numShellCells}</strong>`,
    `Faces: <strong>${faceCount}</strong>`,
    `Boundaries: <strong>${boundaryCount}</strong>`,
  ].join('<br />');
  badgesEl.innerHTML = `
    <div class="badge">Three.js WebGL</div>
    <div class="badge">Full Voronoi export</div>
    <div class="badge">Shell-labeled cells</div>
    <div class="badge">Double-click select</div>
  `;
}

function bindControls() {
  document.querySelector('#showFaces')?.addEventListener('change', (e) => {
    if (model.mode === 'voronoi') {
      for (const cell of cellGroups) for (const m of cell.meshes) m.visible = e.target.checked;
    } else {
      world.traverse((obj) => { if (obj.isMesh) obj.visible = e.target.checked; });
    }
  });
  document.querySelector('#showLines')?.addEventListener('change', (e) => {
    for (const cell of cellGroups) for (const l of cell.lines) l.visible = e.target.checked;
  });
  document.querySelector('#explode')?.addEventListener('input', (e) => {
    const amount = Number(e.target.value);
    for (const cell of cellGroups) {
      cell.group.position.copy(cell.basePosition).addScaledVector(cell.explodeDir, amount);
    }
  });
  document.querySelector('#bg')?.addEventListener('input', (e) => {
    currentBg = e.target.value;
    renderer.setClearColor(currentBg);
  });

  if (model.mode === 'voronoi') {
    renderer.domElement.addEventListener('dblclick', (event) => {
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const intersects = raycaster.intersectObjects(cellGroups.flatMap((i) => i.meshes), false);
      if (!intersects.length) { clearSelection(); return; }
      const record = cellGroups.find((i) => i.meshes.includes(intersects[0].object));
      if (record) selectCell(record);
    });
  }
}

window.addEventListener('resize', () => {
  camera.aspect = wrap.clientWidth / wrap.clientHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(wrap.clientWidth, wrap.clientHeight);
});

bindControls();
if (model.mode === 'mesh') loadMeshScene();
else loadVoronoiScene();

function animate() {
  controls.update();

  // Main scene
  renderer.setViewport(0, 0, wrap.clientWidth, wrap.clientHeight);
  renderer.setScissor(0, 0, wrap.clientWidth, wrap.clientHeight);
  renderer.setScissorTest(false);
  renderer.render(scene, camera);

  // Gizmo inset — bottom-right corner
  // The gizmo camera sits at a fixed position in its own scene (0,0,5) looking
  // at the origin.  We want the axes to reflect the same orientation as the main
  // camera, so we extract the pure rotation from the main camera's world matrix
  // (ignoring its world-space position / orbit distance) and apply it to the
  // gizmo camera.  This keeps the 2-D inset origin pinned while the axes rotate.
  const gx = wrap.clientWidth  - GIZMO_SIZE;
  const gy = 0;  // Y=0 is bottom edge in WebGL viewport coords
  renderer.setViewport(gx, gy, GIZMO_SIZE, GIZMO_SIZE);
  renderer.setScissor( gx, gy, GIZMO_SIZE, GIZMO_SIZE);
  renderer.setScissorTest(true);
  renderer.clearDepth();  // only clear depth — gizmo draws over main scene with no background

  // Extract rotation-only from main camera, apply to gizmo camera position
  // so it orbits around the gizmo origin exactly as the main camera orbits its target.
  const camDir = new THREE.Vector3();
  camera.getWorldDirection(camDir);          // unit vector pointing into scene
  gizmoCamera.position.copy(camDir).negate().multiplyScalar(5);
  gizmoCamera.quaternion.copy(camera.quaternion);

  renderer.render(gizmoScene, gizmoCamera);
  renderer.setScissorTest(false);
  renderer.setClearColor(currentBg);

  requestAnimationFrame(animate);
}
animate();
