async function loadMarkdown(filename) {
    document.getElementById("home").style.display = "none";

    try {
        const response = await fetch(filename);
        if (!response.ok) throw new Error("Markdown file not found");

        const markdown = await response.text();
        console.log("Fetched markdown content:", markdown);

        // Convert Markdown to HTML
        document.getElementById("article-content").innerHTML = marked.parse(markdown);

        // Ensure MathJax has fully loaded before rendering math
        if (typeof MathJax !== "undefined") {
            console.log("MathJax is loaded, processing math...");
            // Render LaTeX (MathJax will process the math expressions)
            await MathJax.typesetPromise([document.getElementById("article-content")]);
        } else {
            console.error("MathJax is not defined.");
        }

        hljs.highlightAll();
    } catch (err) {
        console.error("Error loading markdown:", err.message);
        document.getElementById("article-content").innerHTML = `<p>Error loading article: ${err.message}</p>`;
    }
}

// Load a specific blarticleog post (change the file name as needed)
//loadMarkdown("articles/MLPs.md");
