document.addEventListener('DOMContentLoaded', () => {
    const btn = document.querySelector('button');
    const result = document.getElementById('result');

    btn.addEventListener('click', async () => {
        const questionText = document.getElementById('question').value;
        const contextText = document.getElementById('context').value;

        result.textContent = "Wait...";

        try {
            const response = await fetch('/infer', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    question: questionText,
                    context: contextText
                })
            });

            if (!response.ok) {
                throw new Error(`Server Error: ${response.status}`);
            }

            const data = await response.json();
            result.textContent = data.infer;
        } catch (error) {
            result.textContent = "Error";
            console.error(error);
        }
    });
});
