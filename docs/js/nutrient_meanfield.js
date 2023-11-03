let { default: data_meanfield } = await import("../data/nutrient/meanfield_rho=1_delta=0.json", { assert: { type: "json" } });
let { default: data_rho_05 } = await import("../data/nutrient/meanfield_rho=0.5_delta=0.json", { assert: { type: "json" } });
let { default: data_delta_05 } = await import("../data/nutrient/meanfield_rho=1_delta=0.5.json", { assert: { type: "json" } });
let { default: data_lattice } = await import("../data/nutrient/lattice_rho=1_delta=0.json", { assert: { type: "json" } });

// add 4 radio buttons to switch between meanfield, stochastic, parallel, 3d, wellmixed data
var form = d3.select("div#select-data")
    .append("form")
    .attr("id", "radio-buttons")
    .attr("class", "radio-buttons");

function createRadioButton(form, labelText, value, checked = false) {
    var form_label = form.append("label")
        .attr("class", "radio-label")
        .text(labelText)
        .append("input")
        .attr("type", "radio")
        .attr("name", "radio")
        .attr("value", value)
        .on("change", function() {
            change_data(this.value)
        });

    if (checked) {
        form_label.attr("checked", "checked");
    }

    form_label.append("span")
        .attr("class", "checkmark");
}

createRadioButton(form, "Meanfield (1)", "meanfield", true);
createRadioButton(form, "Rho = 0.5 (2)", "rho05");
createRadioButton(form, "Delta = 0.5 (3)", "delta05");
createRadioButton(form, "Lattice (4)", "lattice");


// on 1,2,3,4 set radio buttons
document.addEventListener('keydown', function(event) {
	if (event.code === 'Digit1') {
		// set radio button to meanfield
		document.getElementById("radio-buttons").elements[0].checked = true;
		change_data('meanfield')
	}
    if (event.code === 'Digit2') {
        // set radio button to rh0 0.5
        document.getElementById("radio-buttons").elements[1].checked = true;
        change_data('rho05')
    }
    if (event.code === 'Digit3') {
        // set radio button to delta 0.5
        document.getElementById("radio-buttons").elements[2].checked = true;
        change_data('delta05')
    }
    if (event.code === 'Digit4') {
        // set radio button to lattice
        document.getElementById("radio-buttons").elements[3].checked = true;
        change_data('lattice')
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
data_rho_05 = filter_max_step(data_rho_05);
data_delta_05 = filter_max_step(data_delta_05);
data_lattice = filter_max_step(data_lattice);

data_lattice.forEach((d) => {
	const lattice = d.soil_lattice;
	const L = lattice.length
	// calculate the fraction of 1s, 2s and 0s in the matrix
	const zeros = lattice.reduce((a, b) => a + b.filter((x) => x === 0).length, 0);
	const ones = lattice.reduce((a, b) => a + b.filter((x) => x === 1).length, 0);
	const twos = lattice.reduce((a, b) => a + b.filter((x) => x === 2).length, 0);
    const threes = lattice.reduce((a, b) => a + b.filter((x) => x === 3).length, 0);
	d.vacancy = zeros / L**2;
	d.nutrient = ones / L**2;
	d.soil = twos / L**2;
    d.worm = threes / L**2;
});


console.log(data_meanfield);

let data = data_meanfield;

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

var rows = d3.map(data_meanfield, function(d){return d.theta;})
var cols = d3.map(data_meanfield, function(d){return d.sigma;}) 

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
        .html("d=" + d3.format("0.2f")(d.theta) + ", " + "s=" + d3.format("0.2f")(d.sigma) + "<br>" + d3.format("0.2f")(d.soil) + ", " + d3.format("0.2f")(d.vacancy) + ", " + d3.format("0.2f")(d.nutrient) + ", " + d3.format("0.2f")(d.worm))
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
		.attr("x", function(d) { return x(d.theta) })
		.attr("y", function(d) { return y(d.sigma) })
		.attr("width", x.bandwidth() )
		.attr("height", y.bandwidth() )
		.style("fill", function(d) { return "rgb(" + d.soil*255 + "," + d.vacancy*255 + "," + d.worm*255 + ")" } )
		.on("mouseover", mouseover_rgb)
		.on("mousemove", mousemove_rgb)
		.on("mouseleave", mouseleave_rgb);


function change_data(state) {
	
	const t = d3.transition().duration(750)

	console.log('Changing data');

	if (state == 'meanfield') {
		data = data_meanfield;
	} 
    else if (state == 'rho05') {
        data = data_rho_05;
    }
    else if (state == 'delta05') {
        data = data_delta_05;
    }
    else if (state == 'lattice') {
        data = data_lattice;
    }

	// update the rgb heatmap
	svg_soil.selectAll(".cell")
		.data(data)
		.transition(t)
		.style("fill", function(d) { return "rgb(" + d.soil*255 + "," + d.vacancy*255 + "," + d.worm*255 + ")" } );

}
	