import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const N_spec = 4;
const theta = 0.06;
// const N_spec = 5;
// const theta = 0.04;


d3.tsv(`../data/multispec_nutrient/${N_spec}spec/nosoil_theta_${theta}.tsv`).then(data => {
    const steps = [];
    const latticeData = [];

    data.forEach(row => {
        steps.push(parseInt(row.step));
        latticeData.push(JSON.parse(row.lattice));
    });

    console.log(steps);

    const L = latticeData[0].length;

    let lattice = latticeData[0];



    const canvas = d3.select("#c");

    // Create a scene
    var scene = new THREE.Scene();

    // Create a camera
    var camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.z = 1.5;

    // Create a renderer
    var renderer = new THREE.WebGLRenderer({ canvas: canvas.node() });
    renderer.setSize(window.innerWidth, window.innerHeight);
    d3.select('div#lattice').node().appendChild(renderer.domElement);


    // Create a directional light
    var light = new THREE.DirectionalLight(0xffffff, 2);
    camera.add(light);

    // Add the camera to the scene
    scene.add(camera);


    // Create a group
    var group = new THREE.Group();


    // Select the div to hold the checkboxes
    var controlsDiv = d3.select('div#controls');

    // Define the checkbox data
    let checkboxes = ['empty'];
    for (let i = 1; i <= N_spec; i++) {
        checkboxes.push(`nutrient${i}`);
    }
    for (let i = 1; i <= N_spec; i++) {
        checkboxes.push(`worm${i}`);
    }

    // Bind the data to the divs
    var divSelection = controlsDiv.selectAll('div')
        .data(checkboxes)
        .enter()
        .append('div');

    // Append a checkbox to each div
    const checkboxSelection = divSelection.append('input')
        .attr('type', 'checkbox')
        .attr('id', function (d) { return d; });

    // Append a label to each div
    divSelection.append('label')
        .attr('for', function (d) { return d; })
        .text(function (d) { return d; });


    // Define the colors
    const colors = {
        'empty': 0xffffff, // white
    };

    const nutrient_colours = [0x90EE90, 0xADD8E6, 0xEE82EE, 0xFF6347, 0xF5DEB3]; // lightgreen, lightblue, violet, tomato, wheat
    const worm_colours = [0x008000, 0x0000FF, 0x800080, 0xFF0000, 0xB8860B]; // green, blue, purple, red, darkgoldenrod
    // const worm_colours = [0x0079FF, 0x00DFA2, 0xF6FA70, 0xFF0060]  // only for temporary visualization

    for (let i = 1; i <= N_spec; i++) {
        colors[`nutrient${i}`] = nutrient_colours[i - 1];
        colors[`worm${i}`] = worm_colours[i - 1];
    }


    // Listen for change events on the checkboxes
    checkboxSelection.on('change', function (event, d) {
        // Remove all meshes from the group
        while (group.children.length > 0) {
            var object = group.children[0];
            object.geometry.dispose();
            object.material.dispose();
            group.remove(object);
        }

        // Add a mesh for each checked checkbox
        checkboxSelection.each(function (d) {
            var checkbox = d3.select(this);
            if (checkbox.property('checked')) {
                var target_site = checkboxes.indexOf(d);
                var color = colors[d];

                var mesh = createMesh(target_site, color);
                group.add(mesh);
            }
        });
    });

    // After creating the checkboxes
    checkboxSelection.each(function (d) {
        var checkbox = d3.select(this);
        if (d.startsWith('worm')) {
            checkbox.property('checked', true);
        }
    });
    checkboxSelection.dispatch('change');

    // Create a function to create a mesh for a given target site and color
    function createMesh(target_site, color) {
        // Create a geometry
        var geometry = new THREE.BoxGeometry(1, 1, 1);
        // Create a material
        var material = new THREE.MeshStandardMaterial({ color: color, transparent: true, opacity: 1 });

        // Calculate the number of target sites
        var targetCount = 0;
        for (let i = 0; i < L; i++) {
            for (let j = 0; j < L; j++) {
                for (let k = 0; k < L; k++) {
                    if (lattice[i][j][k] === target_site) {
                        targetCount++;
                    }
                }
            }
        }

        // Create an InstancedMesh with the correct number of instances
        var mesh = new THREE.InstancedMesh(geometry, material, targetCount);

        // Loop over the lattice and position each instance
        var instance = 0;
        for (let i = 0; i < L; i++) {
            for (let j = 0; j < L; j++) {
                for (let k = 0; k < L; k++) {
                    if (lattice[i][j][k] === target_site) {
                        var matrix = new THREE.Matrix4().makeScale(1 / L, 1 / L, 1 / L).premultiply(new THREE.Matrix4().makeTranslation(i / L - 0.5, j / L - 0.5, k / L - 0.5));
                        mesh.setMatrixAt(instance, matrix);
                        instance++;
                    }
                }
            }
        }

        return mesh;
    }


    // Add the group to the scene
    scene.add(group);

    // Create controls
    var controls = new OrbitControls(camera, renderer.domElement);
    controls.enableZoom = true;
    controls.enableRotate = true;
    controls.enablePan = true;

    window.addEventListener('resize', function () {
        // Update the camera's aspect ratio
        camera.aspect = window.innerWidth / window.innerHeight;
        camera.updateProjectionMatrix();

        // Update the size of the renderer
        renderer.setSize(window.innerWidth, window.innerHeight);
    }, false);

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
    let currentStep = 0;
    let fps = 30; // Desired frames per second
    let now;
    let then = Date.now();
    let interval = 1000 / fps;
    let delta;

    // In the animate function
    function animate() {
        requestAnimationFrame(animate);

        now = Date.now();
        delta = now - then;

        if (delta > interval) {
            console.log(currentStep);
            // Update the lattice
            lattice = latticeData[currentStep];
            currentStep = (currentStep + 1) % latticeData.length;

            // Remove all meshes from the group
            while (group.children.length > 0) {
                var object = group.children[0];
                object.geometry.dispose();
                object.material.dispose();
                group.remove(object);
            }

            // Add a mesh for each checked checkbox
            checkboxSelection.each(function (d) {
                var checkbox = d3.select(this);
                if (checkbox.property('checked')) {
                    var target_site = checkboxes.indexOf(d);
                    var color = colors[d];

                    var mesh = createMesh(target_site, color);
                    group.add(mesh);
                }
            });

            group.rotation.x += 0.01;
            group.rotation.y += 0.01;
            controls.update();
            if (resizeRendererToDisplaySize(renderer)) {
                const canvas = renderer.domElement;
                camera.aspect = canvas.clientWidth / canvas.clientHeight;
                camera.updateProjectionMatrix();
            }
            renderer.render(scene, camera);

            then = now - (delta % interval);
        }
    }

    // Start the animation
    animate();
})