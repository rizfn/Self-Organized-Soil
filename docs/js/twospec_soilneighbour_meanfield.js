let { default: data_meanfield } = await import("../data/two_species/soil_neighbours_meanfield_r=1.json", { assert: { type: "json" } });
let { default: data_stochastic } = await import("../data/two_species/soil_neighbours_r=1.json", { assert: { type: "json" } });
// let { default: data_3D } = await import("../data/two_species/3D_stochastic_dynamics_r=1.json", { assert: { type: "json" } });
let { default: data_double_d } = await import("../data/two_species/double_d_r=1.json", { assert: { type: "json" } });

// add 4 radio buttons to switch between meanfield, stochastic, parallel, 3d, wellmixed data
var form = d3.select("div#select-data")
	.append("form")
	.attr("id", "radio-buttons")
	.attr("class", "radio-buttons");

var form_label = form.append("label")
	.attr("class", "radio-label")
	.text("Meanfield (1)")
	.append("input")
	.attr("type", "radio")
	.attr("name", "radio")
	.attr("value", "meanfield")
	.attr("checked", "checked")
	.on("change", function() {
		change_data(this.value)
	});

form_label.append("span")
	.attr("class", "checkmark");

form_label = form.append("label")
	.attr("class", "radio-label")
	.text("Stochastic (2)")
	.append("input")
	.attr("type", "radio")
	.attr("name", "radio")
	.attr("value", "stochastic")
	.on("change", function() {
		change_data(this.value)
	});

form_label.append("span")
	.attr("class", "checkmark");

form_label = form.append("label")
	.attr("class", "radio-label")
	.text("3D (3)")
	.append("input")
	.attr("type", "radio")
	.attr("name", "radio")
	.attr("value", "3D")
	.on("change", function() {	
		change_data(this.value)
	});

form_label.append("span")
	.attr("class", "checkmark");

form_label = form.append("label")
	.attr("class", "radio-label")
	.text("Double d (4)")
	.append("input")
	.attr("type", "radio")
	.attr("name", "radio")
	.attr("value", "double_d")
	.on("change", function() {	
		change_data(this.value)
	});


// on 1,2,3,4,5 set radio buttons
document.addEventListener('keydown', function(event) {
	if (event.code === 'Digit1') {
		// set radio button to meanfield
		document.getElementById("radio-buttons").elements[0].checked = true;
		change_data('meanfield')
	}
	else if (event.code === 'Digit2') {
		// set radio button to stochastic
		document.getElementById("radio-buttons").elements[1].checked = true;
		change_data('stochastic')
	}
	else if (event.code === 'Digit3') {
		// set radio button to 3D
		document.getElementById("radio-buttons").elements[2].checked = true;
		change_data('3D')
	}
	else if (event.code === 'Digit4') {
		// set radio button to wellmixed
		document.getElementById("radio-buttons").elements[3].checked = true;
		change_data('double_d')
	}
});

function filter_max_step(data) {
	let step_list = data.reduce(function (a, d) {
		if (a.indexOf(d.step) === -1) {
		  a.push(d.step);
		}
		return a;
	  }, []);
	return data.filter(function(d) {return d.step == d3.max(step_list)});
}

data_meanfield = filter_max_step(data_meanfield);
data_stochastic = filter_max_step(data_stochastic);
// data_3D = filter_max_step(data_3D);
data_double_d = filter_max_step(data_double_d);

console.log(data_meanfield);
console.log(data_stochastic);
// console.log(data_3D);
console.log(data_double_d);

let data = data_meanfield;

data_stochastic.forEach((d) => {
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
});

// data_3D.forEach((d) => {
// 	const lattice = d.soil_lattice;
// 	const L = lattice.length
// 	// calculate the fraction of 1s, 2s and 0s in the 3D matrix
// 	const ones = lattice.reduce((a, b) => a + b.reduce((c, d) => c + d.filter((x) => x === 1).length, 0), 0);
// 	const twos = lattice.reduce((a, b) => a + b.reduce((c, d) => c + d.filter((x) => x === 2).length, 0), 0);
// 	const zeros = lattice.reduce((a, b) => a + b.reduce((c, d) => c + d.filter((x) => x === 0).length, 0), 0);
// 	d.vacancy = zeros / L**3;
// 	d.soil = ones / L**3;
// 	d.bacteria = twos / L**3;
// });

data_double_d.forEach((d) => {
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
});


// set the dimensions and margins of the graph
var margin = {top: 40, right: 40, bottom: 100, left: 100},
  width = innerWidth*0.8 - margin.left - margin.right,
  height = innerHeight - margin.top - margin.bottom;


function getOffset(element) {
  const rect = element.getBoundingClientRect();
  return {
    left: rect.left + window.scrollX,
    top: rect.top + window.scrollY
  };
}

var rows = d3.map(data_meanfield, function(d){return d.d;})
var cols = d3.map(data_meanfield, function(d){return d.s;}) 

var x = d3.scaleBand()
	.range([0, width])
	.domain(rows)
	.padding(0.05);

var y = d3.scaleBand()
	.range([height, 0])
	.domain(cols)
	.padding(0.05);
          
// append the svg object to the body of the page
var svg_soil = d3.select("div#raster")
  .append("svg")
    .attr("id", "soil_amounts")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform",
          "translate(" + margin.left + "," + margin.top + ")");


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
        .html("d=" + d3.format("0.2f")(d.d) + ", " + "s=" + d3.format("0.2f")(d.s) + "<br>" + d3.format("0.2f")(d.soil) + ", " + d3.format("0.2f")(d.redvacancy + d.bluevacancy) + ", " + d3.format("0.2f")(d.red + d.blue))
		.style("left", (d3.pointer(event)[0] + heatmap2_location.left + 30) + "px")
		.style("top", (d3.pointer(event)[1] + heatmap2_location.top - 20) + "px")
	}
	var mouseleave_rgb = function(event, d) {
	Tooltip
		.style("opacity", 0)
	d3.select(this)
		.style("stroke", "none")
	}

svg_soil.append("g")
	.attr("class", "axis")
	.attr("transform", "translate(0," + height + ")")
	.call(d3.axisBottom(x).tickFormat(d3.format(".2f")))
	.selectAll("text")
		.attr("transform", "translate(-10,0)rotate(-45)")
		.style("text-anchor", "end");

// label axes
svg_soil.append("text")
	.attr("class", "axis_label")
	.attr("transform", "translate(" + (width/2) + " ," + (height + margin.top) + ")")
	.style("text-anchor", "middle")
	.text("Death rate (θ)");

svg_soil.append("g")
	.attr("class", "axis")
	.call(d3.axisLeft(y).tickFormat(d3.format(".2f")));

// label y axis
svg_soil.append("text")
.attr("class", "axis_label")
.attr("transform", "rotate(-90)")
.attr("y", 0 - margin.left/1.5)
.attr("x",0 - (height / 2))
.attr("dy", "1em")
.style("text-anchor", "middle")
.text("Soil-filling rate (σ)");


svg_soil.selectAll(".cell")
	.data(data)
	.enter()
	.append("rect")
		.attr("class", "cell")
		.attr("x", function(d) { return x(d.d) })
		.attr("y", function(d) { return y(d.s) })
		.attr("width", x.bandwidth() )
		.attr("height", y.bandwidth() )
		.style("fill", function(d) { return "rgb(" + d.soil*255 + "," + (d.redvacancy+d.bluevacancy)*255 + "," + (d.red+d.blue)*255 + ")" } )
		.on("mouseover", mouseover_rgb)
		.on("mousemove", mousemove_rgb)
		.on("mouseleave", mouseleave_rgb);


function change_data(state) {
	
	const t = d3.transition().duration(750)

	console.log('Changing data');

	if (state == 'meanfield') {
		data = data_meanfield;
	} 
	else if (state == 'stochastic') {
		data = data_stochastic;
	} 
	else if (state == '3D') {
		data = data_3D;
	}
	else if (state == 'double_d') {
		data = data_double_d;
	}

	// update the rgb heatmap
	svg_soil.selectAll(".cell")
		.data(data)
		.transition(t)
		.style("fill", function(d) { return "rgb(" + d.soil*255 + "," + (d.redvacancy+d.bluevacancy)*255 + "," + (d.red+d.blue)*255 + ")" } );

}
	