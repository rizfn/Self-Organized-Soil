import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';


// Fetch the lattice data
d3.json('../data/nutrient/lattice3D_rho=1_delta=0.json').then(function (data) {

    var filteredData = data.filter(function (d) {
        return d.sigma === 0.1578947368 && d.theta === 0.0947368421;
    });

    const lattice = filteredData[0].soil_lattice;
    const L = lattice.length;
    console.log(lattice);

    const canvas = d3.select("#c")
        .attr("display", "block")
        .attr("width", "100%")
        .attr("height", "100%");

    // Create a scene
    var scene = new THREE.Scene();

    // Create a camera
    var camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.z = 1.5;

    // Create a renderer
    var renderer = new THREE.WebGLRenderer({ canvas: canvas.node() });
    // renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);


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
    var checkboxes = ['empty', 'nutrient', 'soil', 'worm'];

    // // Bind the data to the checkboxes
    // var checkboxSelection = controlsDiv.selectAll('input')
    //     .data(checkboxes)
    //     .enter()
    //     .append('input')
    //     .attr('type', 'checkbox')
    //     .attr('id', function(d) { return d; })
    //     .property('checked', function(d) { return d === 'soil'; });

    // // Add a label for each checkbox
    // checkboxSelection.each(function(d) {
    //     controlsDiv.append('label')
    //         .attr('for', d)
    //         .text(d);

    //     // Add a line break after each checkbox
    //     controlsDiv.append('br');
    // });


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


    // Listen for change events on the checkboxes
    checkboxSelection.on('change', function (event, d) {
        // Remove all meshes from the group
        while (group.children.length > 0) {
            group.remove(group.children[0]);
        }

        // Add a mesh for each checked checkbox
        checkboxSelection.each(function (d) {
            var checkbox = d3.select(this);
            if (checkbox.property('checked')) {
                var target_site;
                var color;

                switch (d) {
                    case 'empty':
                        target_site = 0;
                        color = 0xffffff; // white
                        break;
                    case 'nutrient':
                        target_site = 1;
                        color = 0xadd8e6; // light blue
                        break;
                    case 'soil':
                        target_site = 2;
                        color = 0xa54a2a; // brown
                        break;
                    case 'worm':
                        target_site = 3;
                        color = 0x008000; // green
                        break;
                }

                var mesh = createMesh(target_site, color);
                group.add(mesh);
            }
        });
    });

    d3.select("#soil").property("checked", true).dispatch("change");


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


    function resizeRendererToDisplaySize(renderer) {
        const canvas = renderer.domElement;
        const pixelRatio = window.devicePixelRatio;
        const width  = canvas.clientWidth  * pixelRatio | 0;
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
});