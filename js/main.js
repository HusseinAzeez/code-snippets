function runOCR(url) {
    Tesseract.recognize('./data/10.png', {
        lang: 'deu'
    }).then(function (result) {
        document.getElementById("ocr_results")
            .innerText = result.text;
    }).progress(function (result) {
        document.getElementById("ocr_status")
            .innerText = result["status"] + " (" +
            (result["progress"] * 100) + "%)";
    });
}

document.getElementById("go-button")
    .addEventListener("click", function (e) {
        var url = document.getElementById("url").value;
        runOCR(url);
    });