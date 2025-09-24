// set the dimensions and margins of the graph
const margin = {top: 10, right: 30, bottom: 30, left: 60},
    width = 460 - margin.left - margin.right,
    height = 400 - margin.top - margin.bottom;

// append the svg object to the body of the page
const svg = d3.select("#my_dataviz")
  .append("svg")
    .attr("width", width + margin.left + margin.right)
    .attr("height", height + margin.top + margin.bottom)
  .append("g")
    .attr("transform", `translate(${margin.left},${margin.top})`);

// retrieve the data from the `data-graph` attribute in the script tag
var scriptTag = document.querySelector('script[src="/js/line_chart.js"]');
const graph_data = JSON.parse(scriptTag.dataset.graph);

// Use the embedded graph_data instead of loading a CSV
const parseDate = d3.timeParse("%Y-%m-%d");
const data = graph_data.map(d => ({
  date: parseDate(d.date),
  value: +d.value
}));

// Now I can use this dataset:
  
// Add X axis --> it is a date format
const x = d3.scaleTime()
  .domain(d3.extent(data, d => d.date))
  .range([0, width]);

svg.append("g")
  .attr("transform", `translate(0, ${height})`)
  .call(d3.axisBottom(x));

// Add Y axis
const y = d3.scaleLinear()
  .domain([0, d3.max(data, d => d.value)])
  .range([height, 0]);

svg.append("g")
  .call(d3.axisLeft(y));

// Add the line
svg.append("path")
  .datum(data)
  .attr("fill", "none")
  .attr("stroke", "steelblue")
  .attr("stroke-width", 1.5)
  .attr("d", d3.line()
    .x(d => x(d.date))
    .y(d => y(d.value))
  );