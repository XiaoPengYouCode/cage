import './style.css';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

const DATA_URL = '/data/hybrid_exact_shell_2000.glb';

const app = document.querySelector('#app');
app.innerHTML = `
  <div class="panel">
    <h1>Three.js Voronoi Block Viewer</h1>
    <div class="meta" id="meta">Loading scene…</div>
    <div class="controls">
      <label><span>Show faces</span><input id="showFaces" type="checkbox" checked /></label>
      <label><span>Show boundaries</span><input id="showLines" type="checkbox" checked /></label>
      <label><span>Explosion</span><input id="explode" type="range" min="0" max="80" step="1" value="0" /></label>
      <label><span>Background</span><input id="bg" type="color" value="#f4f6fb" /></label>
    </div>
    <div class="badge-row" id="badges"></div>
    <div class="note" id="selection-note">Double-click a cell to select it.</div>
  </div>
  <div class="canvas-wrap" id="canvas-wrap"></div>
`;

const wrap = document.querySelector('#canvas-wrap');
const metaEl = document.querySelector('#meta');
const badgesEl = document.querySelector('#badges');
const selectionNoteEl = document.querySelector('#selection-note');

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(wrap.clientWidth, wrap.clientHeight);
renderer.setClearColor('#f4f6fb');
wrap.appendChild(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, wrap.clientWidth / wrap.clientHeight, 0.1, 4000);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.target.set(100, 100, 40);

scene.add(new THREE.AmbientLight(0xffffff, 1.9));
const keyLight = new THREE.DirectionalLight(0xffffff, 1.65);
keyLight.position.set(220, 260, 280);
scene.add(keyLight);
const fillLight = new THREE.DirectionalLight(0xffffff, 0.65);
fillLight.position.set(-180, -100, 160);
scene.add(fillLight);

const world = new THREE.Group();
scene.add(world);

const loader = new GLTFLoader();
const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const cellGroups = [];
let selectedCell = null;

function updateMeta(stats) {
  metaEl.innerHTML = [
    `Shell cells: <strong>${stats.numShellCells}</strong>`,
    `Faces: <strong>${stats.numFaces}</strong>`,
    `Boundaries: <strong>${stats.numBoundaries}</strong>`,
    `Double-click to select block`,
  ].join('<br />');
  badgesEl.innerHTML = `
    <div class="badge">Three.js WebGL</div>
    <div class="badge">Opaque closed cells</div>
    <div class="badge">Double-click select</div>
  `;
}

let worldCenter = new THREE.Vector3(100, 100, 40);

function frameFromBounds(box) {
  const size = box.getSize(new THREE.Vector3());
  const center = box.getCenter(new THREE.Vector3());
  worldCenter.copy(center);
  const radius = Math.max(size.x, size.y, size.z) * 0.8;
  camera.position.set(center.x + radius * 1.3, center.y + radius * 1.1, center.z + radius * 0.9);
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
  for (const line of selectedCell.lines) {
    line.material.color.copy(line.userData.baseColor);
    line.material.opacity = 1.0;
  }
  selectedCell.group.renderOrder = 0;
  selectedCell = null;
  selectionNoteEl.textContent = 'Double-click a cell to select it.';
}

function selectCell(cellRecord) {
  if (selectedCell?.seedId === cellRecord.seedId) {
    clearSelection();
    return;
  }
  clearSelection();
  selectedCell = cellRecord;
  for (const mesh of cellRecord.meshes) {
    mesh.material.color.copy(mesh.userData.baseColor).offsetHSL(0, 0, 0.10);
    mesh.material.emissive = new THREE.Color(0xffd166);
    mesh.material.emissiveIntensity = 0.28;
  }
  for (const line of cellRecord.lines) {
    line.material.color = new THREE.Color(0xff6b00);
    line.material.opacity = 1.0;
  }
  cellRecord.group.renderOrder = 10;
  selectionNoteEl.textContent = `Selected cell seedId=${cellRecord.seedId}`;
}

async function loadScene() {
  const gltf = await loader.loadAsync(DATA_URL);
  world.add(gltf.scene);

  const roots = gltf.scene.children.filter((child) => child.name?.startsWith('cell-'));
  let faceCount = 0;
  let boundaryCount = 0;

  for (const root of roots) {
    const record = { seedId: root.userData.seedId ?? -1, group: root, meshes: [], lines: [], basePosition: root.position.clone(), explodeDir: new THREE.Vector3() };
    root.traverse((obj) => {
      if (obj.isMesh) {
        if (!obj.geometry.getAttribute('normal')) {
          obj.geometry.computeVertexNormals();
        }
        const importedColor = obj.material?.color ? obj.material.color.clone() : new THREE.Color(0.7, 0.7, 0.8);
        const isCylinder = obj.name?.includes('cylinder');
        obj.material = new THREE.MeshStandardMaterial({
          color: importedColor,
          side: THREE.DoubleSide,
          roughness: isCylinder ? 0.82 : 0.94,
          metalness: 0.0,
          transparent: false,
          opacity: 1.0,
          polygonOffset: true,
          polygonOffsetFactor: 1,
          polygonOffsetUnits: 1,
        });
        obj.userData.baseColor = importedColor.clone();
        record.meshes.push(obj);
        faceCount += 1;
      }
      if (obj.isLineSegments || obj.isLine) {
        obj.material = new THREE.LineBasicMaterial({ color: 0x111111, transparent: false, opacity: 1.0, depthTest: true, depthWrite: false });
        obj.renderOrder = 2;
        obj.userData.baseColor = new THREE.Color(0x111111);
        record.lines.push(obj);
        boundaryCount += 1;
      }
    });
    if (record.meshes.length) {
      const box = new THREE.Box3().setFromObject(root);
      const center = box.getCenter(new THREE.Vector3());
      record.explodeDir.copy(center.clone().sub(worldCenter));
      if (record.explodeDir.lengthSq() < 1e-12) {
        record.explodeDir.set(0, 0, 1);
      } else {
        record.explodeDir.normalize();
      }
      cellGroups.push(record);
    }
  }

  const bounds = new THREE.Box3().setFromObject(gltf.scene);
  frameFromBounds(bounds);
  for (const record of cellGroups) {
    const box = new THREE.Box3().setFromObject(record.group);
    const center = box.getCenter(new THREE.Vector3());
    record.explodeDir.copy(center.clone().sub(worldCenter));
    if (record.explodeDir.lengthSq() < 1e-12) {
      record.explodeDir.set(0, 0, 1);
    } else {
      record.explodeDir.normalize();
    }
  }
  updateMeta({ numShellCells: cellGroups.length, numFaces: faceCount, numBoundaries: boundaryCount });
}

function bindControls() {
  const showFaces = document.querySelector('#showFaces');
  const showLines = document.querySelector('#showLines');
  const explode = document.querySelector('#explode');
  const bg = document.querySelector('#bg');

  showFaces.addEventListener('change', () => {
    for (const cell of cellGroups) {
      for (const mesh of cell.meshes) mesh.visible = showFaces.checked;
    }
  });
  showLines.addEventListener('change', () => {
    for (const cell of cellGroups) {
      for (const line of cell.lines) line.visible = showLines.checked;
    }
  });
  explode.addEventListener('input', () => {
    const amount = Number(explode.value);
    for (const cell of cellGroups) {
      cell.group.position.copy(cell.basePosition).addScaledVector(cell.explodeDir, amount);
    }
  });
  bg.addEventListener('input', () => {
    renderer.setClearColor(bg.value);
  });

  renderer.domElement.addEventListener('dblclick', (event) => {
    const rect = renderer.domElement.getBoundingClientRect();
    pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    const intersects = raycaster.intersectObjects(cellGroups.flatMap((item) => item.meshes), false);
    if (!intersects.length) {
      clearSelection();
      return;
    }
    const hit = intersects[0].object;
    const record = cellGroups.find((item) => item.meshes.includes(hit));
    if (record) {
      selectCell(record);
    }
  });
}

function onResize() {
  const width = wrap.clientWidth;
  const height = wrap.clientHeight;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
}
window.addEventListener('resize', onResize);

bindControls();
loadScene();

function animate() {
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}
animate();
