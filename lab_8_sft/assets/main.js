const btn = document.getElementById('analyze-button');
const input = document.getElementById('text-input');
const output = document.getElementById('text-output');
const checkbox = document.getElementById('use-base-model');

btn.addEventListener('click', async function() {
    const inputText = input.value.trim();

    if (!inputText) {
        alert('Please enter text to analyze');
        return;
    }

    btn.disabled = true;
    btn.textContent = 'Analyzing...';

    try {
        const response = await fetch('/infer', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                question: inputText,
                use_base: checkbox.checked
            })
        });

        const data = await response.json();

        const classNames = {
            '0': 'Bad',
            '1': 'Good',
            '2': 'Neutral'
        };

        output.value = classNames[data.infer];
        alert('Analysis complete!');

    } catch (error) {
        output.value = '';
        alert('Analysis error: ' + error.message);
    } finally {
        btn.disabled = false;
        btn.textContent = 'Analyze sentiment';
    }
});
