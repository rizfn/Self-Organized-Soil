let { default: data_nullclines } = await import("../data/single_species/mean_field_data_dense_r=1.json", { assert: { type: "json" } });


const s_list = data_nullclines.reduce(function (a, d) {
	if (a.indexOf(d.s) === -1) {
	  a.push(d.s);
	}
	return a;
 }, []);

// add a slider with values from s_list
var slider = d3.select("div#select-data")
    .append("div")
    .attr("class", "slider")
    .append("input")
    .attr("type", "range")
    .attr("min", 0)
    .attr("max", s_list.length - 1)
    .attr("step", 1)
    .attr("value", 0)
    .attr("id", "s-slider")
    .on("input", function () {
        refilter_data(s_list[this.value])
    }
    );

// mention current value of slider
d3.select("div#select-data")
    .append("div")
    .attr("id", "slider-value")
    .text("s = " + s_list[slider.property("value")]);


let filtereddata = d3.filter(data_nullclines, function (d) { return d.s === s_list[0]; });
console.log(filtereddata);

// plot data.bacteria against data.soil in 2d
var margin = { top: 20, right: 20, bottom: 40, left: 40 },
    width = innerWidth / 1.5 - margin.left - margin.right,
    height = innerHeight - margin.top - margin.bottom;

var x = d3.scaleLinear()
    .domain([0, 1])
    .range([0, width]);

var y = d3.scaleLinear()    
    .domain([0, 1])
    .range([height, 0]);

var xAxis = d3.axisBottom(x);

var yAxis = d3.axisLeft(y);

var svg = d3.select("div#lineplot")
    .append("svg")
    .attr("id", "scatterplot")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)

var g = svg.append("g")
    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

// add the x axis and x-label
g.append("g")
    .attr("class", "x axis")
    .attr("transform", "translate(0," + height + ")")
    .call(xAxis);

g.append("text")
    .attr("class", "x label")
    .attr("text-anchor", "end")
    .attr("x", width)
    .attr("y", height - 6)
    .text("soil");

// add the y axis and y-label
g.append("g")
    .attr("class", "y axis")
    .call(yAxis);

g.append("text")
    .attr("class", "y label")
    .attr("text-anchor", "end")
    .attr("y", 6)
    .attr("dy", ".75em")
    .attr("transform", "rotate(-90)")
    .text("bacteria");

// add the data
g.selectAll(".dot")
    .data(filtereddata)
    .enter()
    .append("circle")
    .attr("class", "dot")
    .attr("r", 3.5)
    .attr("cx", function (d) { return x(d.soil); })
    .attr("cy", function (d) { return y(d.bacteria); })
    .attr("fill", function (d) { return "rgb(" + d.soil * 255 + "," + d.vacancy * 255 + "," + d.bacteria * 255 + ")" })
    .on("mouseover", mouseover)
    .on("mousemove", mousemove)
    .on("mouseleave", mouseleave);

// add a tooltip which gives the values of d for each point
var Tooltip = d3.select("div#lineplot")
    .append("div")
    .attr("class", "tooltip")
    .style("opacity", 0);

function mouseover(event, d) {
    Tooltip
        .style("opacity", 1)
        .style("position", "absolute")
    d3.select(this)
        .style("stroke", "black")
}

function mousemove(event, d) {
    var scatterplot_location = getOffset(document.getElementById("scatterplot"))
    Tooltip
        .html("d=" + d3.format("0.2f")(d.d) + ", " + "s=" + d3.format("0.2f")(d.s) + "<br>" + d3.format("0.2f")(d.soil) + ", " + d3.format("0.2f")(d.vacancy) + ", " + d3.format("0.2f")(d.bacteria))
        .style("left", (d3.pointer(event)[0] + scatterplot_location.left + 30) + "px")
        .style("top", (d3.pointer(event)[1] + scatterplot_location.top - 20) + "px")
}

function mouseleave(event, d) {
    Tooltip
        .style("opacity", 0)
    d3.select(this)
        .style("stroke", "none")
}

function getOffset(el) {
    const rect = el.getBoundingClientRect();
    return {
        left: rect.left + window.scrollX,
        top: rect.top + window.scrollY
    };
}





function refilter_data(s) {
    filtereddata = d3.filter(data_nullclines, function (d) { return d.s === s; });
    console.log(filtereddata);
    g.selectAll(".dot")
        .data(filtereddata)
        .transition()
        .duration(100)
        .attr("cx", function (d) { return x(d.soil); })
        .attr("cy", function (d) { return y(d.bacteria); })
        .style("fill", function (d) { return "rgb(" + d.soil * 255 + "," + d.vacancy * 255 + "," + d.bacteria * 255 + ")" });

    // update slider value
    d3.select("#slider-value")
        .text("s = " + s);
}