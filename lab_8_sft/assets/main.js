document.addEventListener("DOMContentLoaded", () => {
  const button = document.getElementById("infer-btn");
  const input = document.getElementById("question");
  const output = document.getElementById("answer");
  const status = document.getElementById("status");

  button.addEventListener("click", async () => {
    const text = input.value.trim();
    if (!text) return (status.textContent = "Please enter a query.");

    button.disabled = true;
    status.textContent = "Running...";
    output.textContent = "";

    try {
      const response = await fetch("/infer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: text })
      });

      const data = await response.json();
      output.textContent = data.infer;
      status.textContent = "Done.";
    } catch (err) {
      status.textContent = 'Error: ' + err.message;
    } finally {
      button.disabled = false;
    }
  });
});
