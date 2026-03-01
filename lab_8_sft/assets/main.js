document.addEventListener('DOMContentLoaded', () => {
    const btn = document.querySelector('button');
    const result = document.getElementById('result');
    const checkbox = document.getElementById('base-model-checkbox');

    btn.addEventListener('click', async () => {
        const questionText = document.getElementById('question').value;
        const isBaseModel = checkbox.checked;

        result.textContent = "Wait...";
        btn.disabled = true;

        try {
            const response = await fetch('/infer', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    question: questionText,
                    use_base_model: isBaseModel
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
        } finally {
            btn.disabled = false;
        }
    });
});