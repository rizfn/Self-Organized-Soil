async function loadData(url) {
    try {
        const response = await fetch(url);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        return data;
    } catch (error) {
        console.error('Error fetching the JSON data:', error);
    }
}

let data_10_00_80_10 = await loadData("../data/nutrient/meanfield_attractors/S0_0.1_E0_0.0_N0_0.8_W0_0.1.json");
let data_30_30_30_10 = await loadData("../data/nutrient/meanfield_attractors/S0_0.3_E0_0.3_N0_0.3_W0_0.1.json");
let data_80_00_00_20 = await loadData("../data/nutrient/meanfield_attractors/S0_0.8_E0_0.0_N0_0.0_W0_0.2.json");
let data_25_25_25_25 = await loadData("../data/nutrient/meanfield_attractors/S0_0.25_E0_0.25_N0_0.25_W0_0.25.json");

console.log(data_10_00_80_10);

// add 4 radio buttons to switch between meanfield, stochastic, parallel, 3d, wellmixed data
var form = d3.select("div#select-data")
    .append("form")
    .attr("id", "radio-buttons")
    .attr("class", "radio-buttons")
    .style("margin-bottom", "20px");

function createRadioButton(form, labelText, value, checked = false) {
    var form_label = form.append("label")
        .attr("class", "radio-label")
        .text(labelText)
        .append("input")
        .attr("type", "radio")
        .attr("name", "radio")
        .attr("value", value)
        .on("change", function () {
            change_data(this.value)
        });

    if (checked) {
        form_label.attr("checked", "checked");
    }

    form_label.append("span")
        .attr("class", "checkmark");
}

createRadioButton(form, "[25, 25, 25, 25] (1)", "25_25_25_25", true);
createRadioButton(form, "[30, 30, 30, 10]  (2)", "30_30_30_10");
createRadioButton(form, "[80, 0, 0, 20] (3)", "80_00_00_20");
createRadioButton(form, "[10, 0, 80, 10] (4)", "10_00_80_10");

// on 1,2,3,4 set radio buttons
document.addEventListener('keydown', function (event) {
    if (event.code === 'Digit1') {
        document.getElementById("radio-buttons").elements[0].checked = true;
        change_data('25_25_25_25')
    }
    if (event.code === 'Digit2') {
        document.getElementById("radio-buttons").elements[1].checked = true;
        change_data('30_30_30_10')
    }
    if (event.code === 'Digit3') {
        document.getElementById("radio-buttons").elements[2].checked = true;
        change_data('80_00_00_20')
    }
    if (event.code === 'Digit4') {
        document.getElementById("radio-buttons").elements[3].checked = true;
        change_data('10_00_80_10')
    }
});

var colorMap = d3.scaleOrdinal()
    .domain(["Soil", "Empty", "Oscillating", "Stable"])
    .range(["#996220", "#e8e9f3", "#429ea6", "#d7cf07"]);

let data = data_25_25_25_25;

// set the dimensions and margins of the graph
var margin = { top: 40, right: 40, bottom: 100, left: 100 },
    width = innerWidth * 0.8 - margin.left - margin.right,
    height = innerHeight - margin.top - margin.bottom;

function getOffset(element) {
    const rect = element.getBoundingClientRect();
    return {
        left: rect.left + window.scrollX,
        top: rect.top + window.scrollY
    };
}

var rows = d3.map(data, function (d) { return d.theta; })
var cols = d3.map(data, function (d) { return d.sigma; })

var x = d3.scaleBand()
    .range([0, width])
    .domain(rows);

var y = d3.scaleBand()
    .range([height, 0])
    .domain(cols);

// Create new scales for plotting the function
var xLinear = d3.scaleLinear()
    .range([0, width])
    .domain(d3.extent(data, d => d.theta));

var yLinear = d3.scaleLinear()
    .range([height, 0])
    .domain(d3.extent(data, d => d.sigma));

// Create the canvas
var canvas = d3.select("div#raster")
    .append("canvas")
    .attr("id", "canvas")
    // Subtract the top margin from the height
    .attr("height", height + margin.bottom)
    .attr("width", width + margin.left + margin.right)
    .style("position", "absolute")
    .style("top", margin.top + "px");

var context = canvas.node().getContext("2d");

// Translate the canvas context to the right by margin.left
context.translate(margin.left, 0);

// Create an SVG container for the axes
var svg = d3.select("div#raster")
    .append("svg")
    .attr("id", "svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom);

// Add the X Axis
svg.append("g")
    .attr("transform", "translate(" + margin.left + "," + (height + margin.top) + ")")
    .call(d3.axisBottom(xLinear).tickFormat(d3.format(".2f")));

// Add the Y Axis
svg.append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")")
    .call(d3.axisLeft(yLinear).tickFormat(d3.format(".2f")));

// Add X axis label:
svg.append("text")
    .attr("text-anchor", "end")
    .attr("x", width / 2 + margin.left + 50)
    .attr("y", height + margin.top + 40)
    .text("Death rate (θ)");

// Y axis label:
svg.append("text")
    .attr("text-anchor", "end")
    .attr("transform", "rotate(-90)")
    .attr("y", margin.left - 50)
    .attr("x", -height / 2 + 20)
    .text("Soil filling rate (σ)");

function draw(data) {
    // Clear the canvas
    context.clearRect(0, 0, width, height);

    // Draw the new state
    data.forEach(function (d) {
        context.beginPath();
        context.rect(x(d.theta), y(d.sigma), x.bandwidth(), y.bandwidth());
        context.fillStyle = colorMap(d.state);
        context.fill();
        context.closePath();
    });

    // Draw the existing line
    context.beginPath();
    context.strokeStyle = 'k'; // Change this to the color you want
    context.lineWidth = 0.5; // Change this to the width you want

    // Calculate the start and end points of the line
    let thetaStart = xLinear.domain()[0];
    let thetaEnd = xLinear.domain()[xLinear.domain().length - 1];

    // Draw the line
    for (let theta = thetaStart; theta <= thetaEnd; theta += (thetaEnd - thetaStart) / 1000) {
        let sigma = (1 - 2 * Math.sqrt(theta)) / theta;

        // Only draw the line if sigma is greater than 0
        if (sigma > 0) {
            // Convert the theta and sigma values to pixel coordinates
            let xPos = xLinear(theta);
            let yPos = yLinear(sigma);

            // Draw a line to the current point
            context.lineTo(xPos, yPos);
        } else {
            // If sigma is less than or equal to 0, stop drawing the line
            context.stroke();
            context.beginPath();
        }
    }

    context.stroke();

    // // Draw the new straight line
    // context.beginPath();
    // context.strokeStyle = 'red'; // Change this to the color you want
    // context.lineWidth = 3; // Change this to the width you want

    // // Define the start and end points of the new line
    // let startPoint = { sigma: 0.53, theta: 0.14 };
    // let endPoint = { sigma: 0.1, theta: 0.13 };

    // // Convert the start and end points to pixel coordinates
    // let startX = xLinear(startPoint.theta);
    // let startY = yLinear(startPoint.sigma);
    // let endX = xLinear(endPoint.theta);
    // let endY = yLinear(endPoint.sigma);

    // // Draw the new line
    // context.moveTo(startX, startY);
    // context.lineTo(endX, endY);

    // context.stroke();
}

// Call the draw function instead of creating SVG rectangles
draw(data);

// Define your color map
var colorMap = d3.scaleOrdinal()
    .domain(["Soil", "Empty", "Oscillating", "Stable"])
    .range(["#996220", "#e8e9f3", "#429ea6", "#d7cf07"]);

// Select the div#select-data
var legendDiv = d3.select("div#select-data");

// Create the legend
colorMap.domain().forEach(function (state, i) {
    var legendRow = legendDiv.append('div')
        .attr('class', 'legend')
        .style('display', 'flex')
        .style('align-items', 'center')
        .style('margin-left', '30px')
        .style('margin-top', '10px')
        .style('margin-bottom', '0px');

    // Add the colored square
    legendRow.append('div')
        .attr('class', 'legend-color')
        .style('background-color', colorMap(state))
        .style('border', '1px solid black')
        .style('width', '1em')
        .style('height', '1em')
        .style('margin-right', '8px');

    // Add the label
    legendRow.append('div')
        .attr('class', 'legend-label')
        .text(state)
        .style('font-size', '16px');
});

function change_data(state) {
    console.log('Changing data to ' + state);

    if (state == '25_25_25_25') {
        data = data_25_25_25_25;
    }
    else if (state == '30_30_30_10') {
        data = data_30_30_30_10;
    }
    else if (state == '80_00_00_20') {
        data = data_80_00_00_20;
    }
    else if (state == '10_00_80_10') {
        data = data_10_00_80_10;
    }

    draw(data);
}