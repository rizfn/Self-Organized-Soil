const L = 100

const gpu = new GPU.GPU({ mode: 'gpu' });

const smoothDensityLattice = gpu.createKernel(function (densityLattice, smoothingFactor) {
    const x = this.thread.x;
    const y = this.thread.y;
    const numRows = this.constants.numRows;
    const numCols = this.constants.numCols;
    const rolledLattice = (
        densityLattice[(x + 1) % numRows][y] +
        densityLattice[(x - 1 + numRows) % numRows][y] +
        densityLattice[x][(y + 1) % numCols] +
        densityLattice[x][(y - 1 + numCols) % numCols]
    );
    const smoothedValue = densityLattice[x][y] + smoothingFactor * rolledLattice;
    return smoothedValue / (1 + 4 * smoothingFactor);
}, {
    constants: {
        numRows: L,
        numCols: L,
    },
    output: [L, L],
});

const reproduceWorms = gpu.createKernel(function (densityLattice, wormLattice, birthFactor) {
    const x = this.thread.x;
    const y = this.thread.y;
    const targetWormCount1 = (
        densityLattice[x][y] <= 0.5 ?
            2 * densityLattice[x][y] :
            -2 * densityLattice[x][y] + 2
    );
    const targetValue = densityLattice[x][y] > 1 ? 0 : targetWormCount1;
    const newValue = (1 - birthFactor) * wormLattice[x][y] + birthFactor * targetValue;
    return newValue;
}, {
    constants: {
        numRows: L,
        numCols: L,
    },
    output: [L, L],
});

const interactWormSoil = gpu.createKernel(function (densityLattice, wormLattice, interactionFactor) {
    const x = this.thread.x;
    const y = this.thread.y;
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
let sF = 0.6;
let bF = 0.1;
let iF = 1;

let densityLattice = gpu.createKernel(function (initialDensity) {
    return initialDensity / 0.5 * Math.random();
}).setOutput([L, L])(initialDensity);

let wormLattice = gpu.createKernel(function () {
    return Math.random();
}).setOutput([L, L])();




// todo: MAKE SLIDERS WORK, use GPU memory instead

let isRunning = false;
let simulationId;

function startSimulation() {
    isRunning = true;
    updateAndRender(0);
}

function stopSimulation() {
    isRunning = false;
    if (simulationId) {
        cancelAnimationFrame(simulationId);
    }
}

// add button to restart the simulation
d3.select("div#input-section")
    .append("button")
    .attr("id", "restart_button")
    .text("Restart")
    .on("click", function () {
        stopSimulation();
        densityLattice = gpu.createKernel(function (initialDensity) {
            return initialDensity / 0.5 * Math.random();
        }
        ).setOutput([L, L])(initialDensity);

        wormLattice = gpu.createKernel(function () {
            return Math.random();
        }).setOutput([L, L])();

        startSimulation();
    });

// Define the slider properties
const sliders = [
    { id: 'smoothing', label: 'Smoothing Factor', value: sF, callback: function () { sF = +this.value; } },
    { id: 'birth', label: 'Birth Factor', value: bF, callback: function () { bF = +this.value; } },
    { id: 'interaction', label: 'Interaction Factor', value: iF, callback: function () { iF = +this.value; } },
    { id: 'initialDensity', label: 'Initial Density', value: initialDensity, callback: function () { initialDensity = +this.value; } }
];

// Create the sliders
const sliderSection = d3.select("div#input-section").append("div").attr("id", "slider_section");

sliders.forEach(slider => {
    const section = sliderSection.append("div").attr("id", `${slider.id}_slider_section`).attr("class", "slider");

    section.append("label").attr("for", `${slider.id}_slider`).text(slider.label + '\n');

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
    width = innerWidth / 2.3 - margin.left - margin.right,
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
    .append("svg")
    .attr("id", "density_lattice")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
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
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
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
    if (!isRunning) {
        return;
    }

    console.log(i);
    densityLattice = smoothDensityLattice(densityLattice, sF);
    wormLattice = reproduceWorms(densityLattice, wormLattice, bF);
    densityLattice = interactWormSoil(densityLattice, wormLattice, iF);
    if (i % 5 == 0) {update_lattices(densityLattice, wormLattice);}
    simulationId = requestAnimationFrame(() => updateAndRender(i + 1));
}

updateAndRender(0);
