const L = 100;

const gpu = new GPU.GPU({ mode: 'gpu' });

const smoothDensityLattice = gpu.createKernel(function (densityLattice, densitySmoothingFactor) {
    const x = this.thread.y;  // note: Opposite. GPU.js is row-major (?), the outputs were getting transposed each step
    const y = this.thread.x;
    const numRows = this.constants.numRows;
    const numCols = this.constants.numCols;
    const rolledLattice = (
        densityLattice[(x + 1) % numRows][y] +
        densityLattice[(x - 1 + numRows) % numRows][y] +
        densityLattice[x][(y + 1) % numCols] +
        densityLattice[x][(y - 1 + numCols) % numCols]
    );
    const smoothedValue = densityLattice[x][y] + densitySmoothingFactor * rolledLattice;
    return smoothedValue / (1 + 4 * densitySmoothingFactor);
}, {
    constants: {
        numRows: L,
        numCols: L,
    },
    output: [L, L],
});

const smoothWormLattice = gpu.createKernel(function (wormLattice, wormSmoothingFactor) {
    const x = this.thread.y;  // opposite to deal with transposed outputs
    const y = this.thread.x;
    const numRows = this.constants.numRows;
    const numCols = this.constants.numCols;
    const rolledLattice = (
        wormLattice[(x + 1) % numRows][y] +
        wormLattice[(x - 1 + numRows) % numRows][y] +
        wormLattice[x][(y + 1) % numCols] +
        wormLattice[x][(y - 1 + numCols) % numCols]
    );
    const smoothedValue = wormLattice[x][y] + wormSmoothingFactor * rolledLattice;
    return smoothedValue / (1 + 4 * wormSmoothingFactor);
}, {
    constants: {
        numRows: L,
        numCols: L,
    },
    output: [L, L],
});
    

const reproduceWorms = gpu.createKernel(function (densityLattice, wormLattice, birthFactor) {
    const x = this.thread.y;  // opposite to deal with transposed outputs
    const y = this.thread.x;
    const targetWormCount = (
        densityLattice[x][y] <= 0.5 ?
            4 * densityLattice[x][y] - 1 :
            -4 * densityLattice[x][y] + 3
    );
    let newWormLattice = wormLattice[x][y] + (birthFactor * (wormLattice[x][y] * wormLattice[x][y] * targetWormCount));
    newWormLattice = Math.max(-1, Math.min(1, newWormLattice));
    return newWormLattice;
}, {
    constants: {
        numRows: L,
        numCols: L,
    },
    output: [L, L],
});

const interactWormSoil = gpu.createKernel(function (densityLattice, wormLattice, interactionFactor) {
    const x = this.thread.y;  // opposite to deal with transposed outputs
    const y = this.thread.x;
    const numRows = this.constants.numRows;
    const numCols = this.constants.numCols;
    const pushedLattice = (
        densityLattice[(x + 1) % numRows][y] * wormLattice[(x + 1) % numRows][y] +
        densityLattice[(x - 1 + numRows) % numRows][y] * wormLattice[(x - 1 + numRows) % numRows][y] +
        densityLattice[x][(y + 1) % numCols] * wormLattice[x][(y + 1) % numCols] +
        densityLattice[x][(y - 1 + numCols) % numCols] * wormLattice[x][(y - 1 + numCols) % numCols]
    );
    const newValue = (
        densityLattice[x][y] + interactionFactor * (pushedLattice - 4 * densityLattice[x][y] * wormLattice[x][y]) / 4
    );
    return newValue;
}, {
    constants: {
        numRows: L,
        numCols: L,
    },
    output: [L, L],
});

let initialDensity = 0.8;
let sF_d = 0.6;
let sF_w = 0.6;
let bF = 0.1;
let iF = 1;

let densityLattice = gpu.createKernel(function (initialDensity) {
    return initialDensity / 0.5 * Math.random();
}).setOutput([L, L])(initialDensity);

let wormLattice = gpu.createKernel(function () {
    return Math.random();
}).setOutput([L, L])();




// todo: use GPU memory instead

let simulationId;

// on spacebar, restart the simulation
document.addEventListener("keydown", function (event) {
    if (event.code === "Space") {
        restartSimulation();
    }
});

function restartSimulation() {
    if (simulationId) {
        cancelAnimationFrame(simulationId);
    }
    densityLattice = gpu.createKernel(function (initialDensity) {
        return initialDensity / 0.5 * Math.random();
    }
    ).setOutput([L, L])(initialDensity);
    wormLattice = gpu.createKernel(function () {
        return Math.random();
    }).setOutput([L, L])();
    updateAndRender(0);
}

// add button to restart the simulation
d3.select("div#input-section")
    .append("button")
    .attr("id", "restart_button")
    .text("Restart")
    .on("click", function () {
        restartSimulation();
    });

// Define the slider properties
const sliders = [
    { id: 'smoothing_d', label: 'Density Smoothing Factor', value: sF_d, callback: function () { sF_d = +this.value; } },
    { id: 'smoothing_w', label: 'Worm Smoothing Factor', value: sF_w, callback: function () { sF_w = +this.value; } },
    { id: 'birth', label: 'Birth Factor', value: bF, callback: function () { bF = +this.value; } },
    { id: 'interaction', label: 'Interaction Factor', value: iF, callback: function () { iF = +this.value; } },
    { id: 'initialDensity', label: 'Initial Density', value: initialDensity, callback: function () { initialDensity = +this.value; } }
];

// Create the sliders
const sliderSection = d3.select("div#input-section").append("div").attr("id", "slider_section");

sliders.forEach(slider => {
    const section = sliderSection.append("div")
        .attr("id", `${slider.id}_slider_section`)
        .attr("class", "slider")
        .style("display", "flex")
        .style("flex-direction", "column")
        .style("align-items", "center")
        .style("justify-content", "center");

    section.append("label")
        .attr("for", `${slider.id}_slider`)
        .text(slider.label);

    section.append("input")
        .attr("type", "range")
        .attr("min", 0)
        .attr("max", 1)
        .attr("step", 0.01)
        .attr("value", slider.value)
        .attr("id", `${slider.id}_slider`)
        .on("input", slider.callback);
});

// set the dimensions and margins of the graph
var margin = { top: 10, right: 10, bottom: 10, left: 10 },
    width = 0.8 * innerWidth / 2 - margin.left - margin.right,
    height = innerHeight - margin.top - margin.bottom;

function getOffset(element) {
    const rect = element.getBoundingClientRect();
    return {
        left: rect.left + window.scrollX,
        top: rect.top + window.scrollY
    };
}

const latticeSize = Math.min(width, height);

// append the svg object to the body of the page
var densityLatticeSvg = d3.select("div#lattices")
    .style("height", "100vh")
    .append("svg")
    .attr("id", "density_lattice")
    .attr("width", latticeSize + margin.left + margin.right)
    .attr("height", latticeSize + margin.top + margin.bottom)
    .append("g")
    .attr("transform",
        "translate(" + margin.left + "," + margin.top + ")");

var colors = d3.scaleSequential()
    .interpolator(d3.interpolateViridis)
    .domain([0, 1]);

var x = d3.scaleBand()
    .range([0, latticeSize])
    .domain(Array.from(Array(L).keys()))
    .padding(0.05);

var y = d3.scaleBand()
    .range([0, latticeSize])
    .domain(Array.from(Array(L).keys()))
    .padding(0.05);

densityLatticeSvg.selectAll("g.row")
    .data(densityLattice)
    .enter()
    .append("g")
    .attr("class", "row")
    .attr("transform", function (d, i) {
        return "translate(" + 0 + "," + y(i) + ")";
    })
    .selectAll("rect")
    .data(function (d) { return d; })
    .enter()
    .append("rect")
    .attr("x", function (d, i) {
        return x(i);
    })
    .attr("height", x.bandwidth())
    .attr("width", y.bandwidth())
    .attr("fill", function (d) { return colors(d); });


var wormLatticeSvg = d3.select("div#lattices")
    .append("svg")
    .attr("id", "worm_lattice")
    .attr("width", latticeSize + margin.left + margin.right)
    .attr("height", latticeSize + margin.top + margin.bottom)
    .append("g")
    .attr("transform",
        "translate(" + margin.left + "," + margin.top + ")");

wormLatticeSvg.selectAll("g.row")
    .data(wormLattice)
    .enter()
    .append("g")
    .attr("class", "row")
    .attr("transform", function (d, i) {
        return "translate(" + 0 + "," + y(i) + ")";
    })
    .selectAll("rect")
    .data(function (d) { return d; })
    .enter()
    .append("rect")
    .attr("x", function (d, i) {
        return x(i);
    })
    .attr("height", x.bandwidth())
    .attr("width", y.bandwidth())
    .attr("fill", function (d) { return colors(d); });



function update_lattices(density_lattice, worm_lattice) {

    densityLatticeSvg.selectAll("g.row")
        .data(density_lattice)
        .selectAll("rect")
        .data(function (d) { return d; })
        .attr("fill", function (d) { return colors(d); });

    wormLatticeSvg.selectAll("g.row")
        .data(worm_lattice)
        .selectAll("rect")
        .data(function (d) { return d; })
        .attr("fill", function (d) { return colors(d); });
}



function updateAndRender(i) {
    console.log(i);
    densityLattice = smoothDensityLattice(densityLattice, sF_d);
    wormLattice = smoothWormLattice(wormLattice, sF_w);
    wormLattice = reproduceWorms(densityLattice, wormLattice, bF);
    densityLattice = interactWormSoil(densityLattice, wormLattice, iF);
    if (i % 5 == 0) {
        update_lattices(densityLattice, wormLattice);
    }
    simulationId = requestAnimationFrame(() => updateAndRender(i + 1));
}