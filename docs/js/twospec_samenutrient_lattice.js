const data = await (await fetch("../data/twospec_samenutrient/lattice_L=100_sigma=0.5_theta=0.025_rho1=0.5_mu1=0.5.json")).json();

// check if mobile
if (/iPhone|iPad|iPod|Android/i.test(navigator.userAgent)) {
	// add a button to run refilterdata
	var button = d3.select("div#lattice")
		.append("button")
		.attr("id", "refilterdata")
		.html("Refilter data")
		.on("click", refilter_data);
}


// on spacebar, call refilterdata
document.addEventListener('keydown', function(event) {
	if (event.code === 'Space') {
		refilter_data()
	}
});

function calculateNeighbours(c, L) {
	return [[(c[0]-1+L)%L, c[1]], [(c[0]+1)%L, c[1]], [c[0], (c[1]-1+L)%L], [c[0], (c[1]+1)%L]];
}

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

	// iterate over each cell in the lattice
	let soil_boundaries = 0
	for (let i = 0; i < L; i++) {
		for (let j = 0; j < L; j++) {
			if (lattice[i][j] === 1) {
				// calculate the neighbours of the cell
				const neighbours = calculateNeighbours([i, j], L);
				// calculate the number of neighbours who's lattice value is 1
				const soil_neighbours = neighbours.reduce((a, b) => a + (lattice[b[0]][b[1]] === 1), 0);
				soil_boundaries += 4 - soil_neighbours;
			}
		}
	}
	// divide by the total number of possible boundaries: checkerboard pattern, so L^2/2 soil cells with 4 neighbours each
	d.soil_boundaries = soil_boundaries / (((L**2)/2) * 4);
});

console.log(data);


let step = step_list[0]  // start from 1st step

let filtereddata = data.filter(function(d){ return d.step == step });

console.log(filtereddata);

// set the dimensions and margins of the graph
var margin = {top: 10, right: 10, bottom: 40, left: 40},
  width = innerWidth/2 - margin.left - margin.right,
  height = innerHeight - margin.top - margin.bottom;


function getOffset(element) {
  const rect = element.getBoundingClientRect();
  return {
    left: rect.left + window.scrollX,
    top: rect.top + window.scrollY
  };
}

var rows = d3.map(data, function(d){return d.mu2;})
var cols = d3.map(data, function(d){return d.rho2;}) 

var x = d3.scaleBand()
	.range([0, width])
	.domain(rows)
	.padding(0.05);
	

var y = d3.scaleBand()
	.range([height, 0])
	.domain(cols)
	.padding(0.05);


let current_soil_lattice_state = {"mu2": 0, "rho2": 0};

// create a heatmap on mouseclick
var mousedown = function(event, d) {
	current_soil_lattice_state = {"mu2": d.mu2, "rho2": d.rho2};
	console.log(current_soil_lattice_state);
	update_soil_lattice(d.soil_lattice)
}
		
function colour_raster(d) {
    const RGBtotal = (d.soil + d.green + d.blue) || 1;
    return "rgba(" + d.soil * 255 / RGBtotal + "," + d.green * 255 / RGBtotal + "," + d.blue * 255 / RGBtotal + "," + (1-d.vacancy**2) + ")"
}

// append the svg object to the body of the page
var svg_soil = d3.select("div#raster")
  .append("svg")
    .attr("id", "soil_amounts")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform",
          "translate(" + margin.left + "," + margin.top + ")");

// label axes
svg_soil.append("text")
    .attr("class", "axis_label")
    .attr("transform", "translate(" + (width / 2) + " ," + (height + margin.bottom / 2) + ")")
    .style("text-anchor", "middle")
    .text("Blue nutrient-creation rate (μ2)");

svg_soil.append("text")
    .attr("class", "axis_label")
    .attr("transform", "rotate(-90)")
    .attr("y", 0 - margin.left / 1.5)
    .attr("x", 0 - (height / 2))
    .attr("dy", "1em")
    .style("text-anchor", "middle")
    .text("Blue reproduction rate (ρ2)");


    var Tooltip = d3.select("div#raster")
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
	var heatmap2_location = getOffset(document.getElementById("soil_amounts"))
	Tooltip
        .html("mu=" + d3.format("0.2f")(d.mu2) + ", " + "rho=" + d3.format("0.2f")(d.rho2) + "<br>" + d3.format("0.2")(d.soil) + ", " + d3.format("0.2")(d.vacancy) + ", " + d3.format("0.2")(d.nutrient) + ", " + d3.format("0.2")(d.green) + ", " + d3.format("0.2")(d.blue))
		.style("left", (d3.pointer(event)[0] + heatmap2_location.left + 30) + "px")
		.style("top", (d3.pointer(event)[1] + heatmap2_location.top - 20) + "px")
	}
	var mouseleave_rgb = function(event, d) {
	Tooltip
		.style("opacity", 0)
	d3.select(this)
		.style("stroke", "none")
	}


svg_soil.selectAll(".cell")
	.data(filtereddata)
	.enter()
	.append("rect")
		.attr("class", "cell")
		.attr("x", function(d) { return x(d.mu2) })
		.attr("y", function(d) { return y(d.rho2) })
		.attr("width", x.bandwidth() )
		.attr("height", y.bandwidth() )
		.style("fill", function(d) { return colour_raster(d) } )
		.on("mouseover", mouseover_rgb)
		.on("mousemove", mousemove_rgb)
		.on("mouseleave", mouseleave_rgb)
		.on("mousedown", mousedown);


const L = filtereddata[0].soil_lattice.length;

const soil_lattice_size = Math.min(innerWidth/2.1, innerHeight)

var svg_lattice = d3.select("div#lattice")
	.append("svg")
	.attr("id", "soil_lattice")
	.attr("width", soil_lattice_size)
	.attr("height", soil_lattice_size)

var x_lattice = d3.scaleBand()
	.range([0, soil_lattice_size])
	.domain(Array.from(Array(L).keys()))
	.padding(0.05);
	

var y_lattice = d3.scaleBand()
	.range([soil_lattice_size, 0])
	.domain(Array.from(Array(L).keys()))
	.padding(0.05);


function color_lattice(d) {
	if (d === 0) {
		return "rgb(224, 224, 224)"
	} else if (d === 1) {
        return "rgb(150, 200, 200)"
    } else if (d === 2) {
		return "rgb(102, 51, 0)"
	} else if (d === 3) {
		return "rgb(0, 150, 0)"
	} else if (d === 4) {
        return "rgb(0, 0, 150)"
    }
}

const init_lattice = Array(L).fill().map(() => Array(L).fill(0));


svg_lattice.selectAll("g.row")
	.data(init_lattice)
	.enter()
	.append("g")
	.attr("class", "row")
	.attr("transform", function(d, i){
		return "translate("  + 0 +  "," + y_lattice(i) + ")";
	})
	.selectAll("rect")
	.data(function (d, i) {return d;})
	.enter()
	.append("rect")
	.attr("x", function(d, i, j){
		return x_lattice(i);
	})
	.attr("height", x_lattice.bandwidth())
	.attr("width", y_lattice.bandwidth())
	.attr("fill", function(d){return color_lattice(d);});
	


function update_soil_lattice(soil_lattice) {

	console.log('Updating soil lattice')

	const t = d3.transition().duration(750)

	svg_lattice.selectAll("g.row")
		.data(soil_lattice)
		.selectAll("rect")
		.data(function(d, i) {return d;})
		.transition(t)
		.attr("fill", function(d){return color_lattice(d);});
}

function refilter_data() {
	
	const t = d3.transition().duration(750)

	console.log('Refiltering data')

	// select the next entry in steplist
	step = step_list[(step_list.indexOf(step) + 1) % step_list.length]

	filtereddata = data.filter(function(d){ return d.step == step });

	// add a 0.5s popup window  to show the current step
	var popup = d3.select("div#visualization")
		.append("div")
		.attr("class", "popup")	
		.html(step)
		.transition(t)
		.style("opacity", 1)
		.transition(t)
		.style("opacity", 0)
		.remove();


	// update the rgb heatmap
	svg_soil.selectAll(".cell")
		.data(filtereddata)
		.transition(t)
		.style("fill", function(d) { return colour_raster(d) } )

	const soil_lattice = filtereddata.filter(function(d) {return d.mu2 == current_soil_lattice_state.mu2 && d.rho2 == current_soil_lattice_state.rho2})[0].soil_lattice
	// update the lattice
	update_soil_lattice(soil_lattice)
}
