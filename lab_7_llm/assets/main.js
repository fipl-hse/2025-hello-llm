const btn = document.getElementById('translate-button');
const input = document.getElementById('text-input');
const output = document.getElementById('text-output');

    btn.addEventListener('click', async function() {
        const inputText = input.value.trim();

        if (!inputText) {
            alert('Please enter text to translate');
            return;
        }

        btn.disabled = true;
        btn.textContent = 'Translating...';

        try {
            const response = await fetch('/infer', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    question: inputText
                })
            });

            const data = await response.json();
            output.value = data.infer;
            alert('Translation complete!');

        } catch (error) {
            output.value = '';
            alert('Translation failed');
        } finally {
            btn.disabled = false;
            btn.textContent = 'Translate to French';
        }
    });
});
