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

const data_2x = await loadData("../data/twospec_samenutrient/lattice2D_L_256_rho1_0.125_rho2_0.25.json");
const data_4x = await loadData("../data/twospec_samenutrient/lattice2D_L_256_rho1_0.125_rho2_0.5.json");
const data_8x = await loadData("../data/twospec_samenutrient/lattice2D_L_256_rho1_0.125_rho2_1.json");

const datasets = [data_2x, data_4x, data_8x];
const titles = ["2x", "4x", "8x"];
const ids = ["raster_2x", "raster_4x", "raster_8x"];

datasets.forEach((data, index) => {
    data.sort((a, b) => {
        if (a.sigma === b.sigma) {
            // If sigma is equal, sort by theta
            return a.theta - b.theta;
        }
        return a.sigma - b.sigma; // Otherwise, sort by sigma
    });

    var step_list = data.reduce(function (a, d) {
        if (a.indexOf(d.step) === -1) {
            a.push(d.step);
        }
        return a;
    }, []);

    data.forEach((d) => {
        const lattice = d.soil_lattice;
        const L = lattice.length
        const zeros = lattice.reduce((a, b) => a + b.filter((x) => x === 0).length, 0);
        const ones = lattice.reduce((a, b) => a + b.filter((x) => x === 1).length, 0);
        const twos = lattice.reduce((a, b) => a + b.filter((x) => x === 2).length, 0);
        const threes = lattice.reduce((a, b) => a + b.filter((x) => x === 3).length, 0);
        const fours = lattice.reduce((a, b) => a + b.filter((x) => x === 4).length, 0);
        d.vacancy = zeros / L**2;
        d.nutrient = ones / L**2;
        d.soil = twos / L**2;
        d.green = threes / L**2;
        d.blue = fours / L**2;
        delete d.soil_lattice;
    });

    console.log(data);

    let step = step_list[0]  // start from 1st step

    let filtereddata = data.filter(function(d){ return d.step == step });

    console.log(filtereddata);

    // set the dimensions and margins of the graph
    var margin = {top: 40, right: 10, bottom: 40, left: 40},
        width = innerWidth / 3 - margin.left - margin.right,
        height = innerHeight / 1.2 - margin.top - margin.bottom;

    function getOffset(element) {
        const rect = element.getBoundingClientRect();
        return {
            left: rect.left + window.scrollX,
            top: rect.top + window.scrollY
        };
    }

    var rows = d3.map(data, function(d){return d.theta;})
    var cols = d3.map(data, function(d){return d.sigma;}) 

    var x = d3.scaleBand()
        .range([0, width])
        .domain(rows)
        .padding(0.05);

    var y = d3.scaleBand()
        .range([height, 0])
        .domain(cols)
        .padding(0.05);

    function colour_raster(d) {
        if (d.soil == 0) {
            return "rgba(224, 224, 224, 1)";
        } else if (d.green > 0 && d.blue > 0) {
            // If green and blue are non-zero, return yellow
            return "rgba(200, 200, 0, 1)";
        } else if (d.green > 0) {
            // If only green is non-zero, return green
            return "rgba(0, 200, 0, 1)";
        } else if (d.blue > 0) {
            // If only blue is non-zero, return blue
            return "rgba(0, 0, 255, 1)";
        } else {
            // If both are zero, return brown
            return "rgba(165, 42, 42, 1)";
        }
    }

    // append the svg object to the body of the page
    var svg = d3.select("div#visualization")
        .append("svg")
        .attr("id", ids[index])
        .attr("width", width + margin.left + margin.right)
        .attr("height", height + margin.top + margin.bottom)
        .append("g")
        .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    // label axes
    svg.append("text")
        .attr("class", "axis_label")
        .attr("transform", "translate(" + (width / 2) + " ," + (height + margin.bottom / 2) + ")")
        .style("text-anchor", "middle")
        .text("Worm death rate (θ)");

    svg.append("text")
        .attr("class", "axis_label")
        .attr("transform", "rotate(-90)")
        .attr("y", 0 - margin.left / 1.5)
        .attr("x", 0 - (height / 2))
        .attr("dy", "1em")
        .style("text-anchor", "middle")
        .text("Soil filling rate (σ)");

    svg.append("text")
        .attr("class", "title")
        .attr("x", width / 2)
        .attr("y", -margin.top / 2)
        .attr("text-anchor", "middle")
        .text(titles[index]);

    var Tooltip = d3.select("div#visualization")
        .append("div")
        .style("opacity", 0)
        .attr("class", "tooltip")

    // Three function that change the tooltip when user hover / move / leave a cell
    var mouseover_rgb = function(event, d) {
        Tooltip
            .style("opacity", 1)
            .style("position", "absolute")
        d3.select(this)
            .style("stroke", "black")
    }
    var mousemove_rgb = function(event, d) {
        var heatmap2_location = getOffset(document.getElementById(ids[index]))
        Tooltip
            .html("theta=" + d3.format("0.2f")(d.theta) + ", " + "sigma=" + d3.format("0.2f")(d.sigma) + "<br>" + d3.format("0.2")(d.soil) + ", " + d3.format("0.2")(d.vacancy) + ", " + d3.format("0.2")(d.nutrient) + ", " + d3.format("0.2")(d.green) + ", " + d3.format("0.2")(d.blue))
            .style("left", (d3.pointer(event)[0] + heatmap2_location.left + 30) + "px")
            .style("top", (d3.pointer(event)[1] + heatmap2_location.top - 20) + "px")
    }
    var mouseleave_rgb = function(event, d) {
        Tooltip
            .style("opacity", 0)
        d3.select(this)
            .style("stroke", "none")
    }

    svg.selectAll(".cell")
        .data(filtereddata)
        .enter()
        .append("rect")
        .attr("class", "cell")
        .attr("x", function(d) { return x(d.theta) })
        .attr("y", function(d) { return y(d.sigma) })
        .attr("width", x.bandwidth() )
        .attr("height", y.bandwidth() )
        .style("fill", function(d) { return colour_raster(d) } )
        .on("mouseover", mouseover_rgb)
        .on("mousemove", mousemove_rgb)
        .on("mouseleave", mouseleave_rgb);
});

// Add colorbar
var colorbarMargin = {top: 10, right: 0, bottom: 25, left: 40},
    colorbarWidth = innerWidth - colorbarMargin.left - colorbarMargin.right,
    colorbarHeight = innerHeight / 40;

var svg_colorbar = d3.select("div#colorbar")
    .append("svg")
    .attr("width", colorbarWidth + colorbarMargin.left + colorbarMargin.right)
    .attr("height", colorbarHeight + colorbarMargin.top + colorbarMargin.bottom)
    .append("g")
    .attr("transform", "translate(" + colorbarMargin.left + "," + colorbarMargin.top + ")");

var colorScale = d3.scaleOrdinal()
    .domain(["Empty", "Soil", "Coexistence", "Host survival"])
    .range(["rgba(224, 224, 224, 1)", "rgba(165, 42, 42, 1)", "rgba(200, 200, 0, 1)", "rgba(0, 200, 0, 1)"]);  // Updated colors

var legend = svg_colorbar.selectAll(".legend")
    .data(colorScale.domain())
    .enter().append("g")
    .attr("class", "legend")
    .attr("transform", function(d, i) { return "translate(" + i * (colorbarWidth / 4) + ",0)"; });

legend.append("rect")
    .attr("x", 0)
    .attr("y", 0)
    .attr("width", colorbarWidth / 4 - 10)
    .attr("height", colorbarHeight)
    .style("fill", colorScale);

legend.append("text")
    .attr("x", (colorbarWidth / 4 - 10) / 2)
    .attr("y", colorbarHeight + 15)  // Adjusted position for visibility
    .attr("dy", ".35em")
    .style("text-anchor", "middle")
    .style("fill", "black")  // Ensure text is visible
    .text(function(d) { return d; });