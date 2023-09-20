const { default: data } = await import("../data/single_species/mean_field_data_r=1.json", { assert: { type: "json" } });

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

console.log(data);

let step = 1000

let filtereddata = data.filter(function(d){ return d.step == step });

console.log(filtereddata);

// set the dimensions and margins of the graph
var margin = {top: 40, right: 40, bottom: 100, left: 100},
  width = innerWidth - margin.left - margin.right,
  height = innerHeight - margin.top - margin.bottom;


function getOffset(element) {
  const rect = element.getBoundingClientRect();
  return {
    left: rect.left + window.scrollX,
    top: rect.top + window.scrollY
  };
}

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
          
var Tooltip = d3.select("div#raster")
	.append("div")
	.style("opacity", 0)
	.attr("class", "tooltip")

let current_soil_lattice_state = {"d": 0, "s": 0};


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
        .html("d=" + d3.format("0.2f")(d.d) + ", " + "s=" + d3.format("0.2f")(d.s) + "<br>" + d3.format("0.2f")(d.soil) + ", " + d3.format("0.2f")(d.vacancy) + ", " + d3.format("0.2f")(d.bacteria))
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
	.text("d");


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
.text("s");


svg_soil.selectAll(".cell")
	.data(filtereddata)
	.enter()
	.append("rect")
		.attr("class", "cell")
		.attr("x", function(d) { return x(d.d) })
		.attr("y", function(d) { return y(d.s) })
		.attr("width", x.bandwidth() )
		.attr("height", y.bandwidth() )
		.style("fill", function(d) { return "rgb(" + d.soil*255 + "," + d.vacancy*255 + "," + d.bacteria*255 + ")" } )
		.on("mouseover", mouseover_rgb)
		.on("mousemove", mousemove_rgb)
		.on("mouseleave", mouseleave_rgb);


function refilter_data() {
	
	const t = d3.transition().duration(750)

	console.log('Refiltering data')

	if (step == 100) {
		step = 1000
	}
	else if (step == 1000) {
		step = 10000
	}
	else if (step == 10000) {
		step = 100000
	}
	else if (step == 100000) {
		step = 100
	}

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
		.style("fill", function(d) { return "rgb(" + d.soil*255 + "," + d.vacancy*255 + "," + d.bacteria*255 + ")" } )
}
