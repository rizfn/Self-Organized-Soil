// const { default: data } = await import("../data/two_species/soil_neighbours_r=1.json", { assert: { type: "json" } });
const { default: data } = await import("../data/two_species/double_d_r=1.json", { assert: { type: "json" } });

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
	// calculate the fraction of 1s, 2s and 0s in the matrix
	const zeros = lattice.reduce((a, b) => a + b.filter((x) => x === 0).length, 0);
	const ones = lattice.reduce((a, b) => a + b.filter((x) => x === 1).length, 0);
	const twos = lattice.reduce((a, b) => a + b.filter((x) => x === 2).length, 0);
    const threes = lattice.reduce((a, b) => a + b.filter((x) => x === 3).length, 0);
    const fours = lattice.reduce((a, b) => a + b.filter((x) => x === 4).length, 0);
	d.redvacancy = zeros / L**2;
    d.bluevacancy = ones / L**2;
	d.soil = twos / L**2;
	d.red = threes / L**2;
    d.blue = fours / L**2;

	// iterate over each cell in the lattice
	let soil_boundaries = 0
	for (let i = 0; i < L; i++) {
		for (let j = 0; j < L; j++) {
			if (lattice[i][j] === 2) {
				// calculate the neighbours of the cell
				const neighbours = calculateNeighbours([i, j], L);
				// calculate the number of neighbours who's lattice value is 2
				const soil_neighbours = neighbours.reduce((a, b) => a + (lattice[b[0]][b[1]] === 2), 0);
				soil_boundaries += 4 - soil_neighbours;
			}
		}
	}
	// divide by the total number of possible boundaries: checkerboard pattern, so L^2/2 soil cells with 4 neighbours each
	d.soil_boundaries = soil_boundaries / (((L**2)/2) * 4);
});

console.log(data);


let step = step_list[1]  // start from 2nd step, cause why not

let filtereddata = data.filter(function(d){ return d.step == step });

console.log(filtereddata);

// set the dimensions and margins of the graph
var margin = {top: 10, right: 10, bottom: 10, left: 10},
  width = innerWidth/2 - margin.left - margin.right,
  height = innerHeight/2 - margin.top - margin.bottom;


function getOffset(element) {
  const rect = element.getBoundingClientRect();
  return {
    left: rect.left + window.scrollX,
    top: rect.top + window.scrollY
  };
}
          
// append the svg object to the body of the page
var svg = d3.select("div#raster")
  .append("svg")
    .attr("id", "soil_boundaries")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform",
          "translate(" + margin.left + "," + margin.top + ")");

var rows = d3.map(data, function(d){return d.d;})
var cols = d3.map(data, function(d){return d.s;}) 

var x = d3.scaleBand()
	.range([0, width])
	.domain(rows)
	.padding(0.05);
	

var y = d3.scaleBand()
	.range([height, 0])
	.domain(cols)
	.padding(0.05);


var colors = d3.scaleSequential()
	.interpolator(d3.interpolateViridis)
	.domain([d3.min(data, function(d) {return d.soil_boundaries;}), d3.max(data, function(d) {return d.soil_boundaries;})])  

var Tooltip = d3.select("div#raster")
	.append("div")
	.style("opacity", 0)
	.attr("class", "tooltip")

	// Three function that change the tooltip when user hover / move / leave a cell
	var mouseover = function(event, d) {
	Tooltip
		.style("opacity", 1)
	d3.select(this)
		.style("stroke", "black")
	}
	var mousemove = function(event, d) {
	var heatmap_location = getOffset(document.getElementById("soil_boundaries"))
	Tooltip
		.html("d=" + d3.format("0.2f")(d.d) + ", " + "s=" + d3.format("0.2f")(d.s) + "<br>" + d3.format("0.4f")(d.soil_boundaries))
		.style("left", (d3.pointer(event)[0] + heatmap_location.left + 30) + "px")
		.style("top", (d3.pointer(event)[1] + heatmap_location.top - 20) + "px")
	}
	var mouseleave = function(event, d) {
	Tooltip
		.style("opacity", 0)
	d3.select(this)
		.style("stroke", "none")
	}

let current_soil_lattice_state = {"d": 0, "s": 0};

// create a heatmap on mouseclick
var mousedown = function(event, d) {
	current_soil_lattice_state = {"d": d.d, "s": d.s};
	console.log(current_soil_lattice_state);
	update_soil_lattice(d.soil_lattice)
}

svg.selectAll(".cell")
	.data(filtereddata)
	.enter()
	.append("rect")
		.attr("class", "cell")
		.attr("x", function(d) { return x(d.d) })
		.attr("y", function(d) { return y(d.s) })
		.attr("width", x.bandwidth())
		.attr("height", y.bandwidth())
		.style("fill", function(d) { return colors(d.soil_boundaries)} )
		.on("mouseover", mouseover)
		.on("mousemove", mousemove)
		.on("mouseleave", mouseleave)
		.on("mousedown", mousedown);
		

// append the svg object to the body of the page
var svg_soil = d3.select("div#raster")
  .append("svg")
    .attr("id", "soil_amounts")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform",
          "translate(" + margin.left + "," + margin.top + ")");

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
        .html("d=" + d3.format("0.2f")(d.d) + ", " + "s=" + d3.format("0.2f")(d.s) + "<br>" + d3.format("0.2")(d.soil) + ", " + d3.format("0.2")(d.redvacancy) + ", " + d3.format("0.2")(d.bluevacancy) + ", " + d3.format("0.2")(d.red) + ", " + d3.format("0.2")(d.blue))
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
		.attr("x", function(d) { return x(d.d) })
		.attr("y", function(d) { return y(d.s) })
		.attr("width", x.bandwidth() )
		.attr("height", y.bandwidth() )
		.style("fill", function(d) {return "rgb(" + d.soil*255 + "," + (d.redvacancy+d.bluevacancy)*255 + "," + (d.red+d.blue)*255 + ")" } )
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
		return "rgb(255, 200, 200)"
	} else if (d === 1) {
		return "rgb(200, 200, 255)"
	} else if (d === 2) {
		return "rgb(102, 51, 0)"
	} else if (d === 3) {
        return "rgb(200, 50, 50)"
    } else if (d === 4) {
        return "rgb(50, 50, 200)"
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

	// update the heatmap
	svg.selectAll(".cell")
		.data(filtereddata)
		.transition(t)
		.style("fill", function(d) { return colors(d.soil_boundaries)} )

	// update the rgb heatmap
	svg_soil.selectAll(".cell")
		.data(filtereddata)
		.transition(t)
		.style("fill", function(d) { return "rgb(" + d.soil*255 + "," + (d.redvacancy+d.bluevacancy)*255 + "," + (d.red+d.blue)*255 + ")" } )

	const soil_lattice = filtereddata.filter(function(d) {return d.d == current_soil_lattice_state.d && d.s == current_soil_lattice_state.s})[0].soil_lattice
	// update the lattice
	update_soil_lattice(soil_lattice)
}
