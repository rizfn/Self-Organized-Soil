import * as THREE from 'three';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// Create a scene
const scene = new THREE.Scene();

// Create a camera
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.z = 40;

// Create a renderer with a transparent background
const renderer = new THREE.WebGLRenderer({ alpha: true });
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(renderer.domElement);

// Create controls
const controls = new OrbitControls(camera, renderer.domElement);

// Add lights
const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
scene.add(ambientLight);

const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
directionalLight.position.set(1, 1, 1);
scene.add(directionalLight);

const directionalLight2 = new THREE.DirectionalLight(0xffffff, 1);
directionalLight2.position.set(1, 0, 0);
scene.add(directionalLight2);

const pointLight = new THREE.PointLight(0xffffff, 100);
pointLight.position.set(10, 0, 0);
scene.add(pointLight);

// Load the STL file
const loader = new STLLoader();
loader.load('../../../src/visualizations_simple/3D_model/ESPL.stl', function (geometry) {
    // Create a mesh using the loaded geometry
    const material = new THREE.MeshPhongMaterial({ color: 0x901a1e });  // changed material
    const mesh = new THREE.Mesh(geometry, material);
    mesh.scale.set(0.1, 0.1, 0.1);  // scale down
    mesh.position.set(-15, -15, 0);  // center

    // Add the mesh to the scene
    scene.add(mesh);
});

controls.target.set(0, 0, 15);  // center

console.log("Loaded STL file");

// Render the scene
function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}
animate();