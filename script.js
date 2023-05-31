// This function is just a test for the actual search results will be able to change later for anything else
// Please let me know if you need more help with anything else.

function search() {
  var searchInput = document.getElementById("searchInput");
  var query = searchInput.value;


  var searchResults = [
    ["https://www.example.com", "This is an example website"],
    ["https://www.example2.com", "This is another example website"],
  ];

  displayResults(searchResults);
}

function displayResults(results) {
  var resultsContainer = document.getElementById("resultsContainer");
  resultsContainer.innerHTML = "";

  if (results.length === 0) {
    resultsContainer.innerHTML = "<p>No results found.</p>";
    return;
  }

  results.forEach(function(result) {
    var url = result[0];
    var summary = result[1];

    var resultDiv = document.createElement("div");
    resultDiv.className = "result";

    var link = document.createElement("a");
    link.href = url;
    link.textContent = url;
    link.target = "_blank";

    var summaryPara = document.createElement("p");
    summaryPara.textContent = summary || "No summary available";

    resultDiv.appendChild(link);
    resultDiv.appendChild(summaryPara);

    resultsContainer.appendChild(resultDiv);
  });
}
