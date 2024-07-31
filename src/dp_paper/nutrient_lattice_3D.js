import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const p = 0.27;
// const p = 0.22;
const L = 150;
// const N_steps = 500;
const N_steps = 350;
const timeskip = 1;

let data = await (await fetch(`./outputs/DP_3D_lattice/p=${p}_L=${L}_N_steps=${N_steps}.json`)).json();

let lattice = data;
console.log(lattice);

const canvas = d3.select("#c");

// Create a scene
var scene = new THREE.Scene();

// Create an orthographic camera
var aspect = window.innerWidth / window.innerHeight;
var camera = new THREE.OrthographicCamera(-aspect, aspect, 1.2, -1.8, 0.1, 1000);
camera.position.x = 1.5;
camera.position.y = -1;
camera.position.z = 0;


// Create a renderer
var renderer = new THREE.WebGLRenderer({ canvas: canvas.node(), alpha: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
d3.select('div#lattice').node().appendChild(renderer.domElement);


// Create a directional light
var light = new THREE.DirectionalLight(0xffffff, 4);
camera.add(light);

// Add the camera to the scene
scene.add(camera);

let group = new THREE.Group();

// Create a geometry
var geometry = new THREE.BoxGeometry(1, 1, 1);
// Create a material
var material = new THREE.MeshStandardMaterial({ color: 0xff0000, transparent: true, opacity: 1 });

// Calculate the number of target sites
var targetCount = 0;
for (let i = 0; i < N_steps / timeskip; i++) {
    for (let j = 0; j < L; j++) {
        for (let k = 0; k < L; k++) {
            if (lattice[i][j][k] == 1) {
                targetCount++;
            }
        }
    }
}

// Create an InstancedMesh with the correct number of instances
var mesh = new THREE.InstancedMesh(geometry, material, targetCount);

// Loop over the lattice and position each instance
var instance = 0;
for (let i = 0; i < N_steps / timeskip; i++) {
    for (let j = 0; j < L; j++) {
        for (let k = 0; k < L; k++) {
            if (lattice[i][j][k] == 1) {
                var matrix = new THREE.Matrix4().makeScale(1 / L, 1 / L, 1 / L).premultiply(new THREE.Matrix4().makeTranslation(i / L - 0.5, j / L - 0.5, k / L - 0.5));
                mesh.setMatrixAt(instance, matrix);
                instance++;
            }
        }
    }
}

group.add(mesh);

// Create a plane geometry
var planeGeometry = new THREE.PlaneGeometry(1, 1);

var heatmapCanvas = document.createElement('canvas');
heatmapCanvas.width = L;
heatmapCanvas.height = L;
var heatmapContext = heatmapCanvas.getContext('2d');
for (let j = 0; j < L; j++) {
    for (let k = 0; k < L; k++) {
        let value = lattice[N_steps / timeskip - 1][j][k];
        heatmapContext.fillStyle = value === 1 ? 'rgba(0, 0, 0, 1)' : 'rgba(255, 255, 255, 0.5)';
        heatmapContext.fillRect(k, j, 1, 1);
    }
}
var heatmapTexture = new THREE.CanvasTexture(heatmapCanvas);
heatmapTexture.minFilter = THREE.NearestFilter;
heatmapTexture.magFilter = THREE.NearestFilter;

var planeMaterial = new THREE.MeshBasicMaterial({ map: heatmapTexture, transparent: true, side: THREE.DoubleSide });
var plane = new THREE.Mesh(planeGeometry, planeMaterial);

plane.position.set((N_steps) / timeskip / (L) - 0.5, -0.5/L, -0.5/L);  // 0.9 is chosen to make it slightly over the top (1 is overlap)
plane.rotation.x = Math.PI / 2;
plane.rotation.y = Math.PI / 2;
plane.rotation.z = Math.PI / 2;
plane.scale.set(1, 1, 1);

group.add(plane);

// Create a bounding box helper for the sphere
var boxHelper = new THREE.BoxHelper(group, 0x888888);
group.add(boxHelper);

scene.add(group);

// group.rotation.x = -0.4;
group.rotation.x = -1.9;


// Create controls
var controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(1, 0, 0);
controls.enableZoom = true;
controls.enableRotate = true;
controls.enablePan = true;


function resizeRendererToDisplaySize(renderer) {
    const canvas = renderer.domElement;
    const pixelRatio = window.devicePixelRatio;
    const width = canvas.clientWidth * pixelRatio | 0;
    const height = canvas.clientHeight * pixelRatio | 0;
    const needResize = canvas.width !== width || canvas.height !== height;
    if (needResize) {
        renderer.setSize(width, height, false);
    }
    return needResize;
}


// Create a function to animate the scene
function animate() {
    requestAnimationFrame(animate);
    // group.rotation.x += 0.001;
    // group.rotation.y += 0.001;
    controls.update();
    if (resizeRendererToDisplaySize(renderer)) {
        const canvas = renderer.domElement;
        camera.aspect = canvas.clientWidth / canvas.clientHeight;
        camera.updateProjectionMatrix();
    }
    renderer.render(scene, camera);
}

// Start the animation
animate();

