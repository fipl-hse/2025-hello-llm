const queryInput = document.getElementById('query-input');
const sendButton = document.getElementById('send-button');
const outputBox = document.getElementById('output');
const useBaseModelCheckbox = document.getElementById('use-base-model');

    
sendButton.addEventListener('click', handleQuery);
queryInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        handleQuery();
    }
});

function handleQuery() {
    const question = queryInput.value.trim();
    
    if (!question) {
        alert('Please enter a query first!');
        return;
    }

    // Show loading state
    sendButton.disabled = true;
    outputBox.textContent = 'Processing...';

    const useBaseModel = useBaseModelCheckbox.checked;
    
    // Send request to server
    const requestData = {
        question: question,
        use_base_model: useBaseModel
    };
    
    fetch('/infer', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData)
    })
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        // Display response
        outputBox.textContent = data.infer;
    })

    .finally(() => {
        // Reset UI
        queryInput.value = '';
        sendButton.disabled = false;
        sendButton.textContent = 'Send query';
        queryInput.focus();
    });
}
