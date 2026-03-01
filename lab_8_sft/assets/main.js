document.getElementById('btn').onclick = async function() {
    const text = document.getElementById('input').value.trim();
    const output = document.getElementById('output');
    const loading = document.getElementById('loading');
    const useBaseModel = document.getElementById('useBaseModel').checked;

    if (!text) {
        output.textContent = 'Enter the text, please';
        return;
    }

    this.disabled = true;
    loading.style.display = 'block';
    output.textContent = '';

    try {
        const response = await fetch('/infer', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                question: text,
                is_base_model: useBaseModel
            })
        });

        const data = await response.json();
        output.textContent = JSON.stringify(data.infer, null, 2);
    } catch (error) {
        output.textContent = 'Error: ' + error.message;
    }

    this.disabled = false;
    loading.style.display = 'none';
};
