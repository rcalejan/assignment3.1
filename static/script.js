// This function is just a test for the actual search results will be able to change later for anything else
// Please let me know if you need more help with anything else.
document.getElementById('searchButton').addEventListener('click', function() {
  var inputText = document.getElementById('searchInput').value;
  var url = 'http://localhost:5000/api?query=' + encodeURIComponent(inputText);
  fetch(url)
  .then(function(response) {
    if (!response.ok) {
      throw new Error('Request failed');
    }
    return response.json();
  })
  .then(function(data) {
    displayResults(data);
  })
  .catch(function(error) {
    console.log('Error:', error);
  });
});

function displayResults(results) {
  var resultsContainer = document.getElementById("resultsContainer");
  resultsContainer.innerHTML = "";

  if (results.length === 0) {
    resultsContainer.innerHTML = "<p>No results found.</p>";
    return;
  }
  var num_documents = results[0]
  var time_taken = results[1]
  var timeContainer = document.getElementById("time")
  timeContainer.innerHTML = `found ${num_documents} documents in ${time_taken} seconds`
  urls = results[2]
  urls.forEach(function(url) {

    var resultDiv = document.createElement("div");
    resultDiv.className = "result";

    var link = document.createElement("a");
    link.href = url;
    link.textContent = url;
    link.target = "_blank";


    resultDiv.appendChild(link);

    resultsContainer.appendChild(resultDiv);
  });
}
